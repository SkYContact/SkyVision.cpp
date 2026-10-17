// vit_block.hpp - the SigLIP-style ViT block shared by both vision towers.
//
// moondream3.1's SigLIP and VideoChat3's ViT are the same animal: 27 layers,
// dim 1152, 16 heads (head_dim 72), LayerNorm(1e-5) with affine, fused qkv +
// bias, bidirectional attention over contiguous token groups (multi-crop /
// multi-clip, both unmasked FA), wo + bias, residual, and a GELU-tanh MLP.
// Only the attention wiring differs (2D rope on VideoChat3; grouping shape),
// so the block body lives here once and each backend supplies hooks.
//
// The block emits the exact op sequence both engines had inline; hooks cover
// dtype casting (moondream's MD_VISION_F16ACT) and layer-0 parity sinks
// (VC3_STOP_AFTER), so refactored graphs are numerically identical.
#pragma once

#include "ggml.h"

#include <functional>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <string.h>

namespace vqa {
namespace vit {

struct BlockWeights {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;
    ggml_tensor *qkv_w = nullptr, *qkv_b = nullptr;
    ggml_tensor *wo_w = nullptr, *wo_b = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *fc0_w = nullptr, *fc0_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
};

struct BlockHooks {
    // required: rope/grouping/flash-attn over the [3*H*hd, ...] qkv projection,
    // returning attention output laid out as (head, hd) rows per token — i.e.
    // ready for the wo matmul. Returns nullptr if a layer-0 probe inside it
    // stopped the graph (VideoChat3's rope sinks).
    ggml_tensor *(*attention)(void *ud, ggml_tensor *qkv) = nullptr;
    // optional activation-dtype cast (moondream MD_VISION_F16ACT); null = identity
    ggml_tensor *(*to_act)(void *ud, ggml_tensor *t) = nullptr;
    // optional cast to F32 before the LayerNorms (f16act experiment); null = identity
    ggml_tensor *(*to_f32)(void *ud, ggml_tensor *t) = nullptr;
    // optional MLP activation; null = ggml_gelu (tanh). VideoMAE's nn.GELU() is the
    // exact erf form, so it passes ggml_gelu_erf here instead of writing its own block.
    ggml_tensor *(*gelu)(void *ud, ggml_tensor *t) = nullptr;
    // optional layer-0 parity sink, stage "qkv" | "attn_out" | "out"; return
    // true after handling the truncation (build/run/free) to stop the block
    bool (*probe_layer0)(void *ud, const char *stage, ggml_tensor *t) = nullptr;
    // opt-in LN affine folding (VideoChat3's experimental load-time folds):
    // drop the post-norm mul / add from the graph
    bool drop_ln_mul = false;
    bool drop_ln_add = false;
    void *ud = nullptr;
};

// x = x + Wo·Attn(LN1(x)); x = x + FC1·gelu_tanh(FC0·LN2(x)).
// Returns the next residual stream, or nullptr when a layer-0 probe stopped.
inline ggml_tensor * block(ggml_context *ctx, ggml_tensor *x, const BlockWeights &w,
                           float ln_eps, const BlockHooks &h, int layer) {
    auto act = [&](ggml_tensor *t) { return h.to_act ? h.to_act(h.ud, t) : t; };
    auto f32 = [&](ggml_tensor *t) { return h.to_f32 ? h.to_f32(h.ud, t) : t; };

    ggml_tensor *l = ggml_norm(ctx, f32(x), ln_eps);
    if (!h.drop_ln_mul) l = ggml_mul(ctx, l, f32(w.ln1_w));
    if (!h.drop_ln_add) l = ggml_add(ctx, l, f32(w.ln1_b));
    l = act(l);
    ggml_tensor *qkv = ggml_mul_mat(ctx, w.qkv_w, l);
    qkv = act(qkv);
    qkv = ggml_add(ctx, qkv, w.qkv_b);
    if (layer == 0 && h.probe_layer0 && h.probe_layer0(h.ud, "qkv", qkv)) return nullptr;

    ggml_tensor *attn = h.attention(h.ud, qkv);
    if (!attn) return nullptr;
    attn = ggml_mul_mat(ctx, w.wo_w, act(attn));
    attn = act(attn);
    attn = ggml_add(ctx, attn, w.wo_b);
    x = ggml_add(ctx, x, act(attn));   // residual: raw stream (f16act keeps f16+f16)
    if (layer == 0 && h.probe_layer0 && h.probe_layer0(h.ud, "attn_out", x)) return nullptr;

    ggml_tensor *l2 = ggml_norm(ctx, f32(x), ln_eps);
    if (!h.drop_ln_mul) l2 = ggml_mul(ctx, l2, f32(w.ln2_w));
    if (!h.drop_ln_add) l2 = ggml_add(ctx, l2, f32(w.ln2_b));
    l2 = act(l2);
    ggml_tensor *f = ggml_mul_mat(ctx, w.fc0_w, l2);
    f = act(f);
    f = ggml_add(ctx, f, w.fc0_b);
    f = h.gelu ? h.gelu(h.ud, f) : ggml_gelu(ctx, f);
    f = ggml_mul_mat(ctx, w.fc1_w, f);
    f = act(f);
    f = ggml_add(ctx, f, w.fc1_b);
    x = ggml_add(ctx, x, act(f));
    if (layer == 0 && h.probe_layer0 && h.probe_layer0(h.ud, "out", x)) return nullptr;
    return x;
}

// ---------------------------------------------------------------------------
// Generic ViT attention hook env + callback.
//
// The three vision backends (moss_vl, qwen3_5/Marlin, videochat3) each had a
// hand-rolled copy of the same skeleton: strided per-head views into the fused
// qkv, rope (either ggml GGML_ROPE_TYPE_VISION or a manual in-graph pair
// rotation over host-built cos/sin tables), an optional layer-0 truncation
// sink after rope, unmasked flash-attn over the whole token stream or per
// contiguous clip, and a final reshape to (head, hd) rows for the wo matmul
// (optionally de-padding a zero-padded rope layout, or skipping the de-pad
// entirely when the wo weight was folded at load time).
//
// VitHookEnv covers every axis of variation; the callback below reproduces
// each backend's original op sequence bit-for-bit:
//   - t_rope != nullptr          -> manual pair rotation (moss_vl: pairs
//                                   (j, j+vhd_f/2) rotated by the [2, vhd_f/2,
//                                   N] cos/sin table); else ggml_rope_multi
//                                   with GGML_ROPE_TYPE_VISION, sections
//                                   {vhd/4, vhd/4}, n_dims vhd_f/2.
//   - rope_pad_rescale           -> rescale rope_theta by vhd_f/vhd so each
//                                   real pair keeps its exact frequency when
//                                   the converter zero-padded hd (72 -> 80).
//   - clips != nullptr && >1     -> one unmasked FA per clip (cn = steps*hw
//                                   tokens, cont + concat along the token
//                                   axis); else one unmasked FA over all N.
//   - wo_padded                  -> FA output feeds wo directly via a plain
//                                   reshape (load-time wo fold absorbed the
//                                   zero pad rows).
//   - vhd_f != vhd && !wo_padded -> de-pad view (first vhd of vhd_f rows per
//                                   head) then cont + reshape to n_embd; else
//                                   cont(fa) + reshape.
//   - stop_after                 -> optional layer-0 parity truncation right
//                                   after rope ("v_blk0_q_rope"); gated on
//                                   layer 0 unless rope_stop_all_layers.
//   - dbg_names                  -> layer-0 debug tensor names (non-numeric).
struct VitHookEnv {
    ggml_context *gctx = nullptr;
    ggml_cgraph *gf = nullptr;             // only used by the stop_after sinks
    int n_head = 0, n_embd = 0;
    int N = 0;                             // tokens in this graph
    int vhd = 0;                           // real head dim
    int vhd_f = 0;                         // padded head dim (== vhd if unpadded)
    int hw = 0;                            // tokens per clip (clip mode only)
    int n_ctx = 0;                         // rope freq_base ctx (ggml rope mode)
    int layer = 0;
    float rope_theta = 10000.0f;           // ggml rope mode
    bool rope_pad_rescale = false;
    bool rope_stop_all_layers = false;     // qwen3_5 checks every layer
    bool wo_padded = false;
    bool dbg_names = false;
    ggml_tensor *t_vpos = nullptr;         // ggml rope positions
    ggml_tensor *t_rope = nullptr;         // manual-mode cos/sin tables
    const std::vector<int> *clips = nullptr;  // per-clip temporal steps; null
                                              // or size 1 -> single FA over N
    const char *stop_after = nullptr;
    std::function<bool(const char *)> run;
    int state = 0;                         // 1 = truncated ok, 2 = sink failed
};

inline ggml_tensor * vit_attention_hook(void * ud_, ggml_tensor * qkv) {
    VitHookEnv & e = *(VitHookEnv *)ud_;
    ggml_context * c = e.gctx;
    const int H = e.n_head, N = e.N, vhd = e.vhd, vhd_f = e.vhd_f;

    // qkv packs q | k | v along ne0; each [vhd_f]-wide head slice works as-is
    // thanks to the converter's pad/permute - feed strided views straight into
    // rope/FA, no cont(), no runtime padding copies.
    const size_t off = (size_t)H * vhd_f * qkv->nb[0];
    auto head_view = [&](int block) {
        return ggml_view_3d(c, qkv, vhd_f, H, N,
                            (size_t)vhd_f * qkv->nb[0], qkv->nb[1], (size_t)block * off);
    };
    ggml_tensor * q = head_view(0);
    ggml_tensor * k = head_view(1);
    ggml_tensor * v = head_view(2);
    if (e.dbg_names && e.layer == 0) ggml_set_name(qkv, "v_blk0_qkv");

    if (e.t_rope) {
        // manual in-graph rotation: pair (j, j+vhd_f/2) by angle tables
        // broadcast per head; t_rope layout: cos plane [vhd_f/2, N] then sin
        const size_t rs = ggml_element_size(e.t_rope);
        ggml_tensor * cos_t = ggml_view_3d(c, e.t_rope, vhd_f / 2, 1, N, rs, (size_t)(vhd_f / 2) * rs, 0);
        ggml_tensor * sin_t = ggml_view_3d(c, e.t_rope, vhd_f / 2, 1, N, rs, (size_t)(vhd_f / 2) * rs,
                                           (size_t)N * (vhd_f / 2) * rs);
        auto rope_pair = [&](ggml_tensor * x) {
            ggml_tensor * re = ggml_view_3d(c, x, vhd_f / 2, H, N, x->nb[1], x->nb[2], 0);
            ggml_tensor * im = ggml_view_3d(c, x, vhd_f / 2, H, N, x->nb[1], x->nb[2],
                                            (size_t)(vhd_f / 2) * ggml_element_size(x));
            ggml_tensor * ore = ggml_sub(c, ggml_mul(c, re, cos_t), ggml_mul(c, im, sin_t));
            ggml_tensor * oim = ggml_add(c, ggml_mul(c, re, sin_t), ggml_mul(c, im, cos_t));
            return ggml_concat(c, ore, oim, 0);
        };
        q = rope_pair(q);
        k = rope_pair(k);
    } else {
        int sections[GGML_MROPE_SECTIONS] = { vhd / 4, vhd / 4, 0, 0 };
        const float base = e.rope_pad_rescale ? powf(e.rope_theta, (float)vhd_f / vhd) : e.rope_theta;
        q = ggml_rope_multi(c, q, e.t_vpos, nullptr, vhd_f / 2, sections, GGML_ROPE_TYPE_VISION,
                            e.n_ctx, base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_multi(c, k, e.t_vpos, nullptr, vhd_f / 2, sections, GGML_ROPE_TYPE_VISION,
                            e.n_ctx, base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    if (e.dbg_names && e.layer == 0) {
        ggml_set_name(q, "v_blk0_q_rope");
        ggml_set_name(k, "v_blk0_k_rope");
    }
    if (e.stop_after && (e.rope_stop_all_layers || e.layer == 0) &&
        !strcmp(e.stop_after, "v_blk0_q_rope")) {
        ggml_set_name(q, "v_blk0_q_rope");
        ggml_build_forward_expand(e.gf, q);
        if (!e.run("v_blk0_q_rope")) e.state = 2; else e.state = 1;
        return nullptr;
    }

    // unmasked FA: clip-local when clips split the stream (masked flash-attn
    // falls off the fast path; clip tokens are contiguous, so per-clip
    // unmasked FA is exact), else one call over the whole stream.
    auto fa_in = [&](ggml_tensor * t, int n_tok, size_t tok_off) {
        return ggml_view_3d(c, t, vhd_f, n_tok, H, t->nb[2], t->nb[1], tok_off * t->nb[2]);
    };
    ggml_tensor * fa = nullptr;
    if (!e.clips || e.clips->size() == 1) {
        fa = ggml_flash_attn_ext(c, fa_in(q, N, 0), fa_in(k, N, 0), fa_in(v, N, 0), nullptr,
                                 1.0f / sqrtf((float)vhd), 0.0f, 0.0f);
    } else {
        size_t coff = 0;
        for (int t : *e.clips) {
            const int cn = t * e.hw;
            ggml_tensor * fc = ggml_flash_attn_ext(c, fa_in(q, cn, coff), fa_in(k, cn, coff),
                                                   fa_in(v, cn, coff), nullptr,
                                                   1.0f / sqrtf((float)vhd), 0.0f, 0.0f);
            fc = ggml_cont(c, fc);
            fa = fa ? ggml_concat(c, fa, fc, 2) : fc;   // [hd, heads, tokens]: concat tokens
            coff += (size_t)cn;
        }
    }
    if (e.dbg_names && e.layer == 0) ggml_set_name(fa, "v_blk0_attn");
    if (e.wo_padded)
        return ggml_reshape_2d(c, fa, (int64_t)H * vhd_f, N);
    // de-pad vhd_f -> vhd rows per head, then (d, h) rows per token for wo
    ggml_tensor * attn_src = fa;
    if (vhd_f != vhd)
        attn_src = ggml_view_3d(c, fa, vhd, H, N,
                                (size_t)vhd_f * ggml_element_size(fa),
                                (size_t)vhd_f * H * ggml_element_size(fa), 0);
    ggml_tensor * out = ggml_reshape_2d(c, ggml_cont(c, attn_src), e.n_embd, N);
    if (e.dbg_names && e.layer == 0) ggml_set_name(out, "v_blk0_attnout");
    return out;
}

// optional layer-0 parity sink ("qkv" | "attn_out" | "out"); without
// stop_after it just stamps the v_blk0_* debug names and never truncates.
inline bool vit_probe_layer0_hook(void * ud_, const char * stage, ggml_tensor * t) {
    VitHookEnv & e = *(VitHookEnv *)ud_;
    char nm[48];
    snprintf(nm, sizeof(nm), "v_blk0_%s", stage);
    ggml_set_name(t, nm);
    if (e.stop_after && !strcmp(e.stop_after, nm)) {
        ggml_build_forward_expand(e.gf, t);
        if (!e.run(nm)) e.state = 2; else e.state = 1;
        return true;   // truncated: caller returns after freeing gctx
    }
    return false;
}

// patch-embedding bias -> position table fold: host-side F32, so the graph
// drops one full [dim, tokens] add. pos is column-major [dim, tokens] (or a
// flat concatenation of such blocks), bias broadcasts along the fast dim.
inline void fold_patch_bias(float *pos, size_t n_pos, int dim, const float *bias) {
    for (size_t k = 0; k < n_pos; k++) pos[k] += bias[k % dim];
}

}  // namespace vit
}  // namespace vqa
