// codec_graph.hpp - graph-side conv/snake building blocks shared by the codec
// (token2wav) stacks: minicpmo CFM+HiFT, qwen3omni c2w, qwen25omni BigVGAN.
//
// WHAT IS SHARED (moved verbatim, same ops / operand order / node order -
// the vocoder parity gates compare the euler trajectories byte-for-byte and
// vocoders are extremely sensitive to conv layout, see the view-nb trap note
// that came over with the minicpmo conv):
//   * the planar-layout helpers and zero pads (to_planar/pad_left/pad_sym);
//   * the im2col(F32) + one-mul_mat dense conv core and the depthwise core
//     (qwen3omni dw_causal and qwen25omni conv_dw_f32 share it exactly);
//   * the three snake flavors as three functions - they differ in REAL model
//     contract (HiFT divides by alpha, c2w/BigVGAN by exp(beta)+1e-9, with
//     exp on the host vs in-graph) and in ops (mul vs sqr nodes). Unifying
//     them would change op streams, so they stay distinct on purpose;
//   * the minicpmo per-tap dilation path with its (k, ic)-reordered kernel
//     cache and tap-dump debug statics (debug-only behavior, kept attached to
//     the conv that owns it).
//
// WHAT STAYS PER-MODEL (skeleton vs contract, per the common-lib plan):
// graph topology and block wiring, activation layout between blocks
// (minicpmo/qwen25omni planar [T, C] vs qwen3omni CT [C, T]), weight plumbing
// (wctx tensors vs per-build uploaded bytes vs dequantized F32 inputs),
// padding style (ggml_fill vs uploaded zeros), anti-aliased filter machinery,
// and the host DSP (codec_dsp.hpp is the shared part of that).
#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdlib>
#include <map>
#include <vector>

namespace vqa {
namespace codec {

// ---------------------------------------------------------------- layout

// planar helpers
inline ggml_tensor * to_planar(ggml_context * c, ggml_tensor * x) {   // [D, T]->[T, D]
    return ggml_cont(c, ggml_transpose(c, x));
}
inline ggml_tensor * to_dfast(ggml_context * c, ggml_tensor * x) {    // [T, D]->[D, T]
    return ggml_cont(c, ggml_transpose(c, x));
}
// left-pad a planar [T, C] tensor with t0 zero frames (causal convs)
inline ggml_tensor * pad_left(ggml_context * c, ggml_tensor * x, int t0) {
    ggml_tensor * z = ggml_fill(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, t0,
                                                      x->ne[1]), 0.0f);
    return ggml_concat(c, z, x, 0);
}
// symmetric zero pad of a planar [T, C] tensor (ggml_pad's kernel is not
// usable here - it corrupted rows in this graph)
inline ggml_tensor * pad_sym(ggml_context * c, ggml_tensor * x, int t) {
    ggml_tensor * z = ggml_fill(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, t,
                                                      x->ne[1]), 0.0f);
    return ggml_concat(c, ggml_concat(c, z, x, 0), z, 0);
}

// ------------------------------------------------------------- conv cores

// depthwise conv with an F32 im2col: in [T, C] planar, k [K, 1, C] F32 ->
// [OW, C] planar (the qwen3omni dw_causal / qwen25omni conv_dw_f32 core)
inline ggml_tensor * conv_dw_cols(ggml_context * c, ggml_tensor * k,
                                  ggml_tensor * in, int st, int pad, int dil) {
    ggml_tensor * nb4 = ggml_reshape_4d(c, in, in->ne[0], 1, in->ne[1], 1);
    ggml_tensor * im = ggml_im2col(c, k, nb4, st, 0, pad, 0, dil, 0, false,
                                   GGML_TYPE_F32);                    // [K, OW, C]
    ggml_tensor * r = ggml_mul_mat(c, im, k);                        // [OW, 1, C]
    return ggml_reshape_3d(c, r, r->ne[0], r->ne[2], 1);             // [OW, C]
}

// ------------------------------------------------------------ snake variants

// HiFT snake: x + sin(alpha*x)^2 / alpha  (alpha [1, C] view, alpha_logscale
// false) - the minicpmo flavor
inline ggml_tensor * snake_hift(ggml_context * c, ggml_tensor * x, ggml_tensor * alpha) {
    ggml_tensor * s = ggml_sin(c, ggml_mul(c, x, alpha));
    return ggml_add(c, x, ggml_div(c, ggml_mul(c, s, s), alpha));
}

// c2w snake: x + sin(e^a x)^2 / (e^b + 1e-9), per channel; alpha/beta arrive
// as host vectors and the exp/1/(exp+eps) tables are uploaded as graph inputs
// (the qwen3omni flavor). Upload(ne, nd, data, nbytes) -> ggml_tensor*.
template <typename Upload>
inline ggml_tensor * snake_exp(ggml_context * c, Upload && upload,
                               const std::vector<float> & alpha,
                               const std::vector<float> & beta, ggml_tensor * x) {
    std::vector<float> a(alpha.size()), ib(alpha.size());
    for (size_t i = 0; i < alpha.size(); i++) {
        a[i] = std::exp(alpha[i]);
        ib[i] = 1.0f / (std::exp(beta[i]) + 1e-9f);
    }
    const int64_t ne[1] = { (int64_t)alpha.size() };
    ggml_tensor * sg = ggml_sin(c, ggml_mul(c, x, upload(ne, 1, a.data(), a.size() * 4)));
    return ggml_add(c, x, ggml_mul(c, ggml_sqr(c, sg),
                                   upload(ne, 1, ib.data(), ib.size() * 4)));
}

// --------------------------------------------- uploaded-kernel causal convs
// (CT layout: ne0 = channels contiguous per time step; the qwen3omni c2w
// stack's kernels are uploaded per build as F32 graph inputs, k-fastest)

// dense causal conv: CT in -> CT out (stride 1, dilation d, groups=1)
template <typename Upload>
inline ggml_tensor * conv_causal(ggml_context * c, Upload && upload,
                                 const std::vector<float> & w,
                                 const std::vector<float> & b, bool has_bias,
                                 int k, int ci, int co, int dilation,
                                 ggml_tensor * x) {
    const int Ci = (int)x->ne[0];
    const int pad_left = (k - 1) * dilation;
    ggml_tensor * xp = x;
    if (pad_left > 0) {
        std::vector<float> z((size_t)Ci * pad_left, 0.0f);
        const int64_t ne[2] = { Ci, pad_left };
        xp = ggml_concat(c, upload(ne, 2, z.data(), z.size() * 4), x, 1);
    }
    // host bytes are [OC, IC, K] k-fastest == ggml [K, IC, OC]
    const int64_t wne[3] = { k, (int64_t)ci, (int64_t)co };
    ggml_tensor * wt = upload(wne, 3, w.data(), w.size() * 4);
    ggml_tensor * in = ggml_cont(c, ggml_transpose(c, xp));         // [T+pad, Ci]
    ggml_tensor * im = ggml_im2col(c, wt, in, 1, 0, 0, 0, dilation, 0, false,
                                   GGML_TYPE_F32);
    ggml_tensor * y = ggml_mul_mat(c,
        ggml_reshape_2d(c, im, im->ne[0], im->ne[2] * im->ne[1]),
        ggml_reshape_2d(c, wt, wt->ne[0] * wt->ne[1], wt->ne[2]));  // [T, OC]
    ggml_tensor * yc = to_planar(c, y);
    if (has_bias) {
        const int64_t bne[1] = { (int64_t)b.size() };
        yc = ggml_add(c, yc, upload(bne, 1, b.data(), b.size() * 4));
    }
    return yc;
}

// depthwise causal conv (groups=C): CT in -> CT out
template <typename Upload>
inline ggml_tensor * dw_causal(ggml_context * c, Upload && upload,
                               const std::vector<float> & w,
                               const std::vector<float> & b, bool has_bias,
                               int k, int co, int dilation,
                               ggml_tensor * x) {
    const int Ci = (int)x->ne[0];
    const int pad_left = (k - 1) * dilation;
    ggml_tensor * xp = x;
    if (pad_left > 0) {
        std::vector<float> z((size_t)Ci * pad_left, 0.0f);
        const int64_t ne[2] = { Ci, pad_left };
        xp = ggml_concat(c, upload(ne, 2, z.data(), z.size() * 4), x, 1);
    }
    const int64_t kne[3] = { k, 1, (int64_t)co };
    ggml_tensor * kt = upload(kne, 3, w.data(), w.size() * 4);
    ggml_tensor * in = ggml_cont(c, ggml_transpose(c, xp));          // [T+pad, C]
    ggml_tensor * y3 = conv_dw_cols(c, kt, in, 1, 0, dilation);      // [T, C]
    ggml_tensor * yc = to_planar(c, y3);
    if (has_bias) {
        const int64_t bne[1] = { (int64_t)b.size() };
        yc = ggml_add(c, yc, upload(bne, 1, b.data(), b.size() * 4));
    }
    return yc;
}

// transposed conv: CT in -> CT out, trim (k-s) at BOTH ends (HF TransConvNet)
template <typename Upload>
inline ggml_tensor * transconv(ggml_context * c, Upload && upload,
                               const std::vector<float> & w,
                               const std::vector<float> & b, bool has_bias,
                               int k, int ci, int oc, int stride,
                               ggml_tensor * x) {
    const int r = stride;
    const int raw = ((int)x->ne[1] - 1) * r + k;
    const int trim = k - r, to = raw - 2 * trim;
    const int64_t wne[3] = { k, oc, (int64_t)ci };                   // [K, OC, IC] bytes
    ggml_tensor * wt = upload(wne, 3, w.data(), w.size() * 4);
    ggml_tensor * in = ggml_cont(c, ggml_transpose(c, x));           // [T, IC]
    ggml_tensor * yt = ggml_conv_transpose_1d(c, wt, in, r, 0, 1);   // [raw, OC]
    ggml_tensor * yc = ggml_cont(c, ggml_view_2d(c, yt, to, oc, yt->nb[1],
                                                 (size_t)trim * yt->nb[0]));
    yc = to_planar(c, yc);
    if (has_bias) {
        const int64_t bne[1] = { (int64_t)b.size() };
        yc = ggml_add(c, yc, upload(bne, 1, b.data(), b.size() * 4));
    }
    return yc;
}

// ================================================================ HiFT conv
// The minicpmo planar-conv pipeline: conv1d via im2col + ONE mul_mat over the
// stored [IC*K, OC] kernel (rows (ic, k) ic-major, exactly the im2col column
// order - verified: [0, x[0,0], x[1,0], 0, x[0,1], x[1,1], ...] for k3 p1),
// with an explicit per-tap path for s==1 (this ggml's im2col mis-samples for
// dilation > 2) and a full-conv+decimate path for s>2. Moved verbatim from
// minicpmo_vocoder.cpp including the tap-dump debug statics (debug-only
// behavior, owned by the conv that reads it).

inline ggml_tensor * g_conv1_cols = nullptr;   // debug: last conv1 cols
inline ggml_tensor * g_conv1_y = nullptr;      // debug: raw mulmat out
inline ggml_tensor * g_conv1_pl = nullptr;     // debug: planar out
inline ggml_cgraph * gf_dbg = nullptr;         // debug: graph for tap sinks
// lazy (k, ic)-reordered copies of conv kernels for the per-tap conv path
inline std::map<const void *, ggml_tensor *> g_reord;
inline int ntap = 0;                           // debug: per-tap conv counter
inline ggml_backend_t g_voc_backend = nullptr;

inline ggml_tensor * voc_reordered_kernel(ggml_tensor * w, int IC, int K) {
    auto it = g_reord.find(w->data);
    if (it != g_reord.end()) return it->second;
    struct ggml_init_params ip = { 8ull * 1024 * 1024, NULL, true };
    ggml_context * rc = ggml_init(ip);
    ggml_tensor * dst = ggml_new_tensor_2d(rc, GGML_TYPE_F32,
                                           w->ne[0], w->ne[1]);
    ggml_backend_buffer_t bbuf = ggml_backend_alloc_ctx_tensors(rc, g_voc_backend);
    std::vector<float> in_((size_t)w->ne[0] * w->ne[1]);
    ggml_backend_tensor_get(w, in_.data(), 0, in_.size() * 4);
    std::vector<float> out_(in_.size());
    const size_t R = (size_t)w->ne[0];
    for (size_t oc = 0; oc < (size_t)w->ne[1]; oc++)
        for (int ic = 0; ic < IC; ic++)
            for (int k = 0; k < K; k++)
                out_[(size_t)(k * IC + ic) + oc * R] =
                    in_[(size_t)(ic * K + k) + oc * R];
    ggml_backend_tensor_set(dst, out_.data(), 0, out_.size() * 4);
    (void)bbuf;   // buffer owns the data; intentionally leaked with the map
    g_reord.emplace(w->data, dst);
    // idempotence: a kernel that is ALREADY (k, ic)-ordered (the load-time
    // reorder slots rb/srb c1[1]/c1[2]) must map to itself, or the second
    // reorder would swap it back to (ic, k) and desync the per-tap gather.
    g_reord.emplace(dst->data, dst);
    return dst;
}

// conv1d over planar [T, C] activations with bias, stride s / zero-pad p /
// dilation d -> planar [T2, OC]
inline ggml_tensor * conv1_planar(ggml_context * c, ggml_tensor * w, ggml_tensor * b,
                                  ggml_tensor * x, int s, int p, int d) {
    if (ggml_n_dims(w) == 3)   // 3-D stored kernel (cam_ll/tdnn): gguf
        w = ggml_reshape_2d(c, w, w->ne[0] * w->ne[1], w->ne[2]);  // [K*IC, OC]
    ggml_tensor * xp = (p > 0) ? pad_sym(c, x, p) : x;
    const int IC = (int)x->ne[1];
    const int OC = (int)w->ne[1];
    const int K = (int)(w->ne[0] / IC);
    const int T = (int)xp->ne[0];
    const int T2 = (T - d * (K - 1) - 1) / s + 1;
    ggml_tensor * fake = ggml_new_tensor_3d(c, GGML_TYPE_F32, K, IC, OC);
    ggml_tensor * y;
    if (s == 1) {
        // ggml_im2col mis-samples for dilation > 2: gather the taps
        // explicitly. cols rows are (k, ic) - k outer, ic inner - so the
        // kernel must be the (k, ic)-reordered copy built at load.
        // NOTE: materialize the transposed layout FIRST (dup over the
        // transpose tensor itself, which carries the real nb), then
        // row-slice the dense copy. A VIEW of a transposed tensor loses
        // nb[0] (ggml views reset it to type_size), so ops on it silently
        // read row-major - do not cont/view through such a view.
        ggml_tensor * xpv = ggml_cont(c, ggml_transpose(c, xp)); // [IC, T] dense
        // per-tap mul_mat accumulation (this ggml's concat along dim0 is
        // unreliable at conv scales - it scrambled blocks of correct inputs).
        // IMPORTANT: never cont a view with a nonzero byte offset (the CPU
        // dup scrambles it here). Slice xp as a contiguous-row view first
        // (row-copy dup path, exact), then transpose that dense copy
        // (to_planar pattern, exact); same for the kernel slices.
        const int conv_id = ++ntap;
        const bool tap_full = (IC == 256 && OC == 256 && K == 7
                               && getenv("MINICPMO_VOC_TAPDUMP")
                               && conv_id <= 8);
        ggml_tensor * acc = nullptr;
        for (int k = 0; k < K; k++) {
            ggml_tensor * x_kp = ggml_cont(c, ggml_view_2d(c, xp, T2, IC,
                                               xp->nb[1], (size_t)k * d * 4));
            ggml_tensor * x_k = ggml_cont(c, ggml_transpose(c, x_kp));
            ggml_tensor * w_r = voc_reordered_kernel(w, IC, K);
            ggml_tensor * w_k = ggml_cont(c, ggml_view_2d(c, w_r, IC, OC,
                                              (size_t)IC * K * 4,
                                              (size_t)k * IC * 4));
            ggml_tensor * zz = ggml_mul_mat(c, w_k, x_k);           // [OC, T2]
            if (tap_full) {
                char nm[32];
                snprintf(nm, sizeof(nm), "tap_z%d_%d", conv_id, k);
                ggml_tensor * tz = ggml_cont(c, zz);
                ggml_set_name(tz, nm);
                ggml_set_output(tz);
                ggml_build_forward_expand(gf_dbg, tz);
            }
            acc = acc ? ggml_add(c, acc, zz) : zz;
        }
        y = acc;
        if (tap_full) {
            auto dump_sink = [&](ggml_tensor * t, const char * nm) {
                ggml_tensor * tc = ggml_cont(c, t);
                ggml_set_name(tc, nm);
                ggml_set_output(tc);
                ggml_build_forward_expand(gf_dbg, tc);
            };
            dump_sink(x, "tap_x");
            dump_sink(xpv, "tap_xpv");
            dump_sink(acc, "tap_acc");
            dump_sink(y, "tap_y");
        }
    } else if (s > 2) {
        // ggml_im2col mis-samples for stride > 2; run the FULL stride-1
        // conv (verified exact) and decimate: conv stride s out[t] ==
        // full[s*t], a strided view of the [OC, Mf] output.
        const int Mf = T - K + 1;                       // stride-1 length
        ggml_tensor * cols1 = ggml_im2col(c, fake, xp, 1, 0, 0, 0, 1,
                                          0, false, GGML_TYPE_F32);
        ggml_tensor * yf = ggml_mul_mat(c, w,
            ggml_reshape_2d(c, cols1, IC * K, Mf));     // [OC, Mf]
        y = ggml_view_2d(c, yf, OC, T2, yf->nb[1] * s, 0);
    } else {
        ggml_tensor * cols = ggml_im2col(c, fake, xp, s, 0, 0, 0, d, 0, false,
                                         GGML_TYPE_F32);  // [IC*K, OW]
        g_conv1_cols = cols;
        y = ggml_mul_mat(c, w,
            ggml_reshape_2d(c, cols, IC * K, T2));
    }
    g_conv1_y = y;
    y = to_planar(c, y);                             // [T, OC]
    g_conv1_pl = y;
    if (b) {   // planar [T, C]: the bias must broadcast on channels ([1, C])
        y = ggml_add(c, y, ggml_reshape_2d(c, b, 1, b->ne[0]));
    }
    return y;
}

}  // namespace codec
}  // namespace vqa
