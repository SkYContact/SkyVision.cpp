// minicpmo_vocoder.cpp - the MiniCPM-o 4.5 token2wav vocoder (CosyVoice2
// flow.pt + hift.pt + campplus.onnx), pure ggml graphs + host DSP glue.
//
// Truth: stepaudio2 Token2wav (site-packages, the official implementation;
// CosyVoice2 names are legacy). Offline path only (no stream cache):
//   campplus(fbank of the prompt wav) -> spk [192]           (CAM++, ggml)
//   flow.inference(codes, prompt tokens, prompt mel, spk)     (ggml)
//       token embedding -> UpsampleConformerEncoderV2 (6+4 rel-pos blocks,
//       nearest x2 + conv k5) -> encoder_proj -> mu [80, T2]
//       CFM: x = rand_noise buffer slice (the official module buffer, baked
//       in the GGUF); 10 euler steps with cosine t_span and CFG 0.7; the DiT
//       estimator runs twice per step (cond + zeroed-tracks uncond);
//       generated mel = feat[:, :, T2p:] -> hift mel [80, T_mel]
//   hift(mel)                                                (ggml + host)
//       ConvRNNF0Predictor (ggml) -> f0; nearest x480; SineGen2 source (host
//       DSP: modulo / linear-interp / cumsum phases plus the three torch RNG
//       draws) -> s; STFT(s) hann16/hop4 reflect (host); decode graph:
//       conv_pre, 3x [lrelu0.1, ConvTranspose1d (ggml conv_transpose_1d
//       p0=0 + view crop), source fusion, mean of 3 Snake resblocks],
//       lrelu0.01, conv_post -> magnitude/phase; ISTFT hann16/hop4 (host)
//       -> clamp 0.99 -> PCM 24 kHz.
//
// Layouts: flow-graph activations are [D, T] (mul_mat native); conv
// activations [T, C] planar (ggml conv_1d native); campplus conv2d data
// ne=[W, H, C, N]. Linear weights [in, out]; conv weights keep the torch
// layout (== ggml kernel order [K, IC, OC] / [KW, KH, IC, OC]);
// ConvTranspose weights [K, OC, IC]. F16 operands + F32 norms/biases.
//
// Gates: MINICPMO_VOC_CODES=<codes.txt> teacher-forces the codebook ids;
// MINICPMO_VOC_RNG_DIR=<dir> loads the official HiFT RNG draws
// (rand_ini.f32 / sine_noise.f32 / src_noise.f32) instead of the engine
// generator; MINICPMO_VOC_MEL_LOAD=<mel.f32> bypasses flow (vocoder-only
// parity); MINICPMO_VOC_DUMP=<prefix> dumps spk/mel/f0/pcm as raw f32;
// MINICPMO_VOC_SEED=<n> seeds the engine HiFT RNG (default 0).
#include "minicpmo.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "core/engine/gguf_util.hpp"
#include "core/engine/weight_load.hpp"
#include "core/util/mmap.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include <map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace minicpmo {

namespace {

constexpr int N_HARM = 9;
constexpr int ISTFT_N = 16, ISTFT_HOP = 4;
constexpr float SINE_AMP = 0.1f, NOISE_STD = 0.003f, VOICED_TH = 10.f;
constexpr float LRELU01 = 0.01f, LRELU1 = 0.1f;

// torch CPU MT19937 float uniform (at::uniform_real_distribution<float>)
struct Vocrng {
    std::mt19937 mt;
    explicit Vocrng(uint32_t seed) : mt(seed) {}
    uint32_t next() { return mt(); }
    float uniform() { return (float)(next() & 0xFFFFFFu) * (1.0f / 16777216.0f); }
    float normal() {                      // Box-Muller (engine production only)
        float u1 = uniform(), u2 = uniform();
        if (u1 < 1e-9f) u1 = 1e-9f;
        return std::sqrt(-2.f * std::log(u1)) * std::cos(2.f * (float)M_PI * u2);
    }
};

void host_f32(ggml_tensor * t, std::vector<float> & out) {
    const size_t n = (size_t)ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return;
    }
    std::vector<ggml_fp16_t> raw(n);
    ggml_backend_tensor_get(t, raw.data(), 0, n * sizeof(ggml_fp16_t));
    for (size_t i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(raw[i]);
}

// planar helpers
ggml_tensor * to_planar(ggml_context * c, ggml_tensor * x) {   // [D, T]->[T, D]
    return ggml_cont(c, ggml_transpose(c, x));
}
ggml_tensor * to_dfast(ggml_context * c, ggml_tensor * x) {    // [T, D]->[D, T]
    return ggml_cont(c, ggml_transpose(c, x));
}
// left-pad a planar [T, C] tensor with t0 zero frames (causal convs)
ggml_tensor * pad_left(ggml_context * c, ggml_tensor * x, int t0) {
    ggml_tensor * z = ggml_fill(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, t0,
                                                      x->ne[1]), 0.0f);
    return ggml_concat(c, z, x, 0);
}
// symmetric zero pad of a planar [T, C] tensor (ggml_pad's kernel is not
// usable here - it corrupted rows in this graph)
ggml_tensor * pad_sym(ggml_context * c, ggml_tensor * x, int t) {
    ggml_tensor * z = ggml_fill(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, t,
                                                      x->ne[1]), 0.0f);
    return ggml_concat(c, ggml_concat(c, z, x, 0), z, 0);
}
// conv1d via ggml_im2col + one mul_mat. The stored kernel is 2-D
// [IC*K, OC]: rows are (ic, k) ic-major, exactly matching the im2col column
// order (verified: [0, x[0,0], x[1,0], 0, x[0,1], x[1,1], ...] for k3 p1),
// so ONE mul_mat computes the conv. The kernel passed to ggml_im2col is a
// shape-only fake (im2col reads only its ne for the output geometry; its
// data comes from b).
static ggml_tensor * g_conv1_cols = nullptr;   // debug: last conv1 cols
static ggml_tensor * g_conv1_y = nullptr;      // debug: raw mulmat out
static ggml_tensor * g_conv1_pl = nullptr;     // debug: planar out
static ggml_cgraph * gf_dbg = nullptr;         // debug: graph for tap sinks
// lazy (k, ic)-reordered copies of conv kernels for the per-tap conv path
static std::map<const void *, ggml_tensor *> g_reord;
static int ntap = 0;                           // debug: per-tap conv counter
static ggml_backend_t g_voc_backend = nullptr;
static ggml_tensor * voc_reordered_kernel(ggml_tensor * w, int IC, int K) {
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
ggml_tensor * conv1(ggml_context * c, ggml_tensor * w, ggml_tensor * b,
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
ggml_tensor * snake(ggml_context * c, ggml_tensor * x, ggml_tensor * alpha) {
    // x + sin(alpha*x)^2 / alpha  (alpha [1, C] view, alpha_logscale false)
    ggml_tensor * s = ggml_sin(c, ggml_mul(c, x, alpha));
    return ggml_add(c, x, ggml_div(c, ggml_mul(c, s, s), alpha));
}

}  // namespace

// ---------------------------------------------------------------- impl
struct Vocoder::impl {
    ggml_gallocr_t allocr = nullptr;      // per-graph gallocrs: sharing one
    ggml_gallocr_t allocr_cf = nullptr;   // pool across the five graphs makes
    ggml_gallocr_t allocr_fe = nullptr;   // gallocr alias live buffers
    ggml_gallocr_t allocr_dit = nullptr;  // (qwen25omni hit the same bug)
    ggml_gallocr_t allocr_f0 = nullptr;
    ggml_gallocr_t allocr_dec = nullptr;
    ggml_backend_buffer_t dec_buf = nullptr;   // NO_GALLOCR decode alloc
    vqa::MMap mm;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_context * bn_ctx = nullptr;      // precomputed BN iv/bb tensors
    ggml_backend_buffer_t bn_buf = nullptr;
    vqa::AlignedBuf host_f32;             // promoted F32 conv kernels

    ggml_context * cf_ctx = nullptr;
    ggml_cgraph * cf_gf = nullptr;
    ggml_tensor * cf_in = nullptr, * cf_out = nullptr;

    ggml_context * fe_ctx = nullptr;
    ggml_cgraph * fe_gf = nullptr;
    ggml_tensor * fe_tok = nullptr, * fe_pos1 = nullptr, * fe_pos2 = nullptr;
    ggml_tensor * fe_mu = nullptr;
    int fe_T = -1;

    ggml_context * dit_ctx = nullptr;
    ggml_cgraph * dit_gf = nullptr;
    ggml_tensor * d_x = nullptr, * d_mu = nullptr, * d_spk = nullptr,
               * d_cond = nullptr, * d_t = nullptr, * d_out = nullptr;
    ggml_tensor * d_mask = nullptr;
    int dit_T = -1;

    ggml_context * f0_ctx = nullptr;
    ggml_cgraph * f0_gf = nullptr;
    ggml_tensor * f0_in = nullptr, * f0_out = nullptr;

    ggml_context * dec_ctx = nullptr;
    ggml_cgraph * dec_gf = nullptr;
    ggml_tensor * dec_mel = nullptr, * dec_sstft = nullptr;
    ggml_tensor * dec_mag = nullptr, * dec_ph = nullptr;
    int dec_Tmel = -1;

    std::vector<float> spk, f0, mel;
    int n_gen_mel = 0;
    std::vector<float> hann16;
    std::vector<float> gate_rand_ini, gate_sine_noise, gate_src_noise;
    bool gate_rng = false;
    Vocrng rng{0};

    bool load_ok = false;

    void free_graphs() {
        for (ggml_context * c : {cf_ctx, fe_ctx, dit_ctx, f0_ctx, dec_ctx})
            if (c) { ggml_free(c); }
        cf_ctx = fe_ctx = dit_ctx = f0_ctx = dec_ctx = nullptr;
    }
    ~impl() {
        free_graphs();
        if (wbuf) ggml_backend_buffer_free(wbuf);
        if (wctx) ggml_free(wctx);
        if (allocr) ggml_gallocr_free(allocr);
        if (dec_buf) ggml_backend_buffer_free(dec_buf);
        for (ggml_gallocr_t g : {allocr_cf, allocr_fe, allocr_dit,
                                 allocr_f0, allocr_dec})
            if (g) ggml_gallocr_free(g);
    }
};

// ---------------------------------------------------------------- load
bool Vocoder::load(ggml_backend_t backend_, const char * path) {
    backend = backend_;
    g_voc_backend = backend_;
    if (!p) p = new impl;
    p->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    p->allocr_cf = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    p->allocr_fe = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    p->allocr_dit = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    p->allocr_f0 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    p->allocr_dec = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!p->mm.load(path)) { fprintf(stderr, "vocoder: mmap failed: %s\n", path); return false; }
    fprintf(stderr, "vocoder file: %.2f GB\n", p->mm.size / 1e9);
    struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, true };
    p->wctx = ggml_init(ip);
    if (!p->wctx) return false;
    struct gguf_init_params gp = { true, &p->wctx };
    gguf_context * gu = gguf_init_from_file(path, gp);
    if (!gu) { fprintf(stderr, "vocoder: gguf init failed\n"); return false; }

    n_mel = vqa::gguf::u32(gu, "token2wav.n_mel", n_mel);
    codebook = vqa::gguf::u32(gu, "token2wav.codebook", codebook);
    flow_d = vqa::gguf::u32(gu, "token2wav.flow_d", flow_d);
    flow_heads = vqa::gguf::u32(gu, "token2wav.flow_heads", flow_heads);
    flow_hd = vqa::gguf::u32(gu, "token2wav.flow_head_dim", flow_hd);
    n_cfm = vqa::gguf::u32(gu, "token2wav.cfm_block_count", n_cfm);
    n_cfm_up = vqa::gguf::u32(gu, "token2wav.cfm_up_block_count", n_cfm_up);
    n_dit = vqa::gguf::u32(gu, "token2wav.dit_block_count", n_dit);
    dit_d = vqa::gguf::u32(gu, "token2wav.dit_hidden", dit_d);
    cfg_rate = vqa::gguf::f32(gu, "token2wav.cfg_rate", cfg_rate);
    n_timesteps = vqa::gguf::u32(gu, "token2wav.n_timesteps", n_timesteps);
    dit_eps = vqa::gguf::f32(gu, "token2wav.dit_eps", dit_eps);
    dit_qk_eps = vqa::gguf::f32(gu, "token2wav.dit_qk_eps", dit_qk_eps);
    conf_ln_eps = vqa::gguf::f32(gu, "token2wav.conformer_ln_eps", conf_ln_eps);
    embed_ln_eps = vqa::gguf::f32(gu, "token2wav.embed_ln_eps", embed_ln_eps);
    n_prompt_tokens = vqa::gguf::u32(gu, "token2wav.prompt.tokens_len", 0);
    n_prompt_mels = vqa::gguf::u32(gu, "token2wav.prompt.mels_len", 0);
    n_fbank = vqa::gguf::u32(gu, "token2wav.spk.fbank_len", 0);
    n_rand = vqa::gguf::u32(gu, "token2wav.rand_noise_len", 0);

    vqa::weights::bind_from_mmap(gu, p->wctx, p->mm);
    gguf_free(gu);
    fprintf(stderr, "vocoder: bind_from_mmap ok\n");
    // conv kernels ride ggml im2col/conv_transpose, which want plain F32
    // rows on the CPU path (minicpmo audio conv precedent); in this GGUF
    // every >=3-D F16 tensor is a conv kernel; the hifi up-conv kernels are
    // stored flattened 2-D [OC*K, IC] and get promoted + re-shaped below
    {
        size_t total = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(p->wctx); t;
             t = ggml_get_next_tensor(p->wctx, t))
            if (t->type == GGML_TYPE_F16)
                total += (size_t)ggml_nelements(t) * 4;
        p->host_f32.resize(total);
        size_t off = 0;
        int np = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(p->wctx); t;
             t = ggml_get_next_tensor(p->wctx, t)) {
            if (t->type != GGML_TYPE_F16)
                continue;
            const int64_t n = ggml_nelements(t);
            float * dst = (float *)(p->host_f32.data() + off);
            const ggml_fp16_t * src = (const ggml_fp16_t *)t->data;
            for (int64_t i = 0; i < n; i++) dst[i] = ggml_fp16_to_fp32(src[i]);
            t->data = dst;
            t->type = GGML_TYPE_F32;
            t->nb[0] = sizeof(float);
            for (int k = 1; k < 4; k++) t->nb[k] = t->nb[k - 1] * t->ne[k - 1];
            off += (size_t)n * 4;
            np++;
        }
        fprintf(stderr, "vocoder: promoted %d conv kernels (%.1f MB f32)\n",
                np, total / 1e6);
        // the up-conv kernels were flattened to 2-D [OC*K, IC] by the
        // converter; ggml_conv_transpose_1d wants torch [IC, OC, K] i.e.
        // ggml ne [K, OC, IC] - identical memory, pure view
        for (int i = 0; i < 3; i++) {
            char nm[64];
            snprintf(nm, sizeof(nm), "hifi.up.%d.weight", i);
            ggml_tensor * t = ggml_get_tensor(p->wctx, nm);
            snprintf(nm, sizeof(nm), "hifi.up.%d.bias", i);
            ggml_tensor * tb = ggml_get_tensor(p->wctx, nm);
            if (!t || !tb || t->ne[2] != 1 || t->ne[0] % tb->ne[0]) {
                fprintf(stderr, "vocoder: bad up-conv %d\n", i);
                return false;
            }
            const int64_t IC = t->ne[1];
            const int64_t K = t->ne[0] / tb->ne[0];
            t->ne[0] = K; t->ne[1] = tb->ne[0]; t->ne[2] = IC; t->ne[3] = 1;
            for (int k = 0; k < 4; k++)
                t->nb[k] = k ? t->nb[k - 1] * t->ne[k - 1] : sizeof(float);
        }
    }
    fprintf(stderr, "vocoder: promote ok (%.1f MB)\n",
            p->host_f32.size() / 1e6);
    vqa::weights::Bound bound = vqa::weights::bind_to_backend(
        p->wctx, backend_, p->mm, { { p->host_f32.data(), p->host_f32.size() } });
    if (!bound.ok) { fprintf(stderr, "vocoder: bind failed\n"); return false; }
    p->wbuf = bound.main_buf;
    ggml_backend_buffer_set_usage(p->wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(p->wctx, n);
        if (!t) fprintf(stderr, "vocoder: missing tensor %s\n", n);
        return t;
    };
    rand_noise = gt("t2w.rand_noise");
    prompt_tokens = gt("t2w.prompt.tokens");
    prompt_mels = gt("t2w.prompt.mels");
    spk_fbank = gt("t2w.spk.fbank");
    spk_emb_ref = gt("t2w.spk.emb_ref");
    input_emb = gt("cfm.input_emb.weight");
    spk_proj_w = gt("cfm.spk_proj.weight"); spk_proj_b = gt("cfm.spk_proj.bias");
    embed_w = gt("cfm.embed.weight"); embed_b = gt("cfm.embed.bias");
    embed_ln_w = gt("cfm.embed_ln.weight"); embed_ln_b = gt("cfm.embed_ln.bias");
    la_conv1_w = gt("cfm.lookahead.conv1.weight"); la_conv1_b = gt("cfm.lookahead.conv1.bias");
    la_conv2_w = gt("cfm.lookahead.conv2.weight"); la_conv2_b = gt("cfm.lookahead.conv2.bias");
    char b[96];
    auto gcl = [&](int i, ConfLayer & L, const char * pfx) {
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "cfm.%s%d.%s", pfx, i, suf); return gt(b); };
        L.ln_mha_w = g("ln_mha.weight"); L.ln_mha_b = g("ln_mha.bias");
        L.ln_ff_w = g("ln_ff.weight");   L.ln_ff_b = g("ln_ff.bias");
        L.wq = g("wq.weight"); L.bq = g("wq.bias");
        L.wk = g("wk.weight"); L.bk = g("wk.bias");
        L.wv = g("wv.weight"); L.bv = g("wv.bias");
        L.wo = g("wo.weight"); L.bo = g("wo.bias");
        L.wpos = g("wpos.weight");
        L.pos_u = g("pos_bias_u"); L.pos_v = g("pos_bias_v");
        L.ffn_gate_w = g("ffn_gate.weight"); L.ffn_gate_b = g("ffn_gate.bias");
        L.ffn_down_w = g("ffn_down.weight"); L.ffn_down_b = g("ffn_down.bias");
        return L.wq && L.wk && L.wv && L.wo && L.wpos && L.ffn_gate_w && L.ffn_down_w;
    };
    cl.resize(n_cfm);
    for (int i = 0; i < n_cfm; i++)
        if (!gcl(i, cl[i], "blk.")) return false;
    cul.resize(n_cfm_up);
    for (int i = 0; i < n_cfm_up; i++)
        if (!gcl(i, cul[i], "ublkc.")) return false;
    up_conv_w = gt("cfm.up.conv.weight"); up_conv_b = gt("cfm.up.conv.bias");
    up_embed_w = gt("cfm.up_embed.weight"); up_embed_b = gt("cfm.up_embed.bias");
    up_embed_ln_w = gt("cfm.up_embed_ln.weight"); up_embed_ln_b = gt("cfm.up_embed_ln.bias");
    after_norm_w = gt("cfm.after_norm.weight"); after_norm_b = gt("cfm.after_norm.bias");
    proj_w = gt("cfm.proj.weight"); proj_b = gt("cfm.proj.bias");

    t_mlp0_w = gt("dit.t_mlp0.weight"); t_mlp0_b = gt("dit.t_mlp0.bias");
    t_mlp1_w = gt("dit.t_mlp1.weight"); t_mlp1_b = gt("dit.t_mlp1.bias");
    in_proj_w = gt("dit.in_proj.weight"); in_proj_b = gt("dit.in_proj.bias");
    dl.resize(n_dit);
    for (int i = 0; i < n_dit; i++) {
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "dit.blk.%d.%s", i, suf); return gt(b); };
        DiTLayer & L = dl[i];
        L.wq = g("wq.weight"); L.bq = g("wq.bias");
        L.wk = g("wk.weight"); L.bk = g("wk.bias");
        L.wv = g("wv.weight"); L.bv = g("wv.bias");
        L.wo = g("wo.weight"); L.bo = g("wo.bias");
        L.qn_w = g("q_norm.weight"); L.qn_b = g("q_norm.bias");
        L.kn_w = g("k_norm.weight"); L.kn_b = g("k_norm.bias");
        L.mlp_gate_w = g("mlp_gate.weight"); L.mlp_gate_b = g("mlp_gate.bias");
        L.mlp_down_w = g("mlp_down.weight"); L.mlp_down_b = g("mlp_down.bias");
        L.conv1_w = g("conv1.weight"); L.conv1_b = g("conv1.bias");
        L.conv_ln_w = g("conv_ln.weight"); L.conv_ln_b = g("conv_ln.bias");
        L.conv2_w = g("conv2.weight"); L.conv2_b = g("conv2.bias");
        L.adaln_w = g("adaln.weight"); L.adaln_b = g("adaln.bias");
        if (!L.wq || !L.adaln_w || !L.conv1_w) return false;
    }
    fin_adaln_w = gt("dit.fin.adaln.weight"); fin_adaln_b = gt("dit.fin.adaln.bias");
    fin_out_w = gt("dit.fin.out.weight"); fin_out_b = gt("dit.fin.out.bias");

    for (int i = 0; i < 5; i++) {
        snprintf(b, sizeof(b), "hifi.f0.c%d.weight", i); f0_c_w[i] = gt(b);
        snprintf(b, sizeof(b), "hifi.f0.c%d.bias", i); f0_c_b[i] = gt(b);
    }
    f0_head_w = gt("hifi.f0.head.weight"); f0_head_b = gt("hifi.f0.head.bias");
    src_lin_w = gt("hifi.src_linear.weight"); src_lin_b = gt("hifi.src_linear.bias");
    conv_pre_w = gt("hifi.conv_pre.weight"); conv_pre_b = gt("hifi.conv_pre.bias");
    const int upk[3] = {16, 11, 7}, ups_[3] = {8, 5, 3}, upp[3] = {4, 3, 2};
    for (int i = 0; i < 3; i++) {
        snprintf(b, sizeof(b), "hifi.up.%d.weight", i); ups[i].w = gt(b);
        snprintf(b, sizeof(b), "hifi.up.%d.bias", i); ups[i].b = gt(b);
        ups[i].k = upk[i]; ups[i].s = ups_[i]; ups[i].p = upp[i];
    }
    const int ks[3] = {3, 7, 11}, sks[3] = {7, 7, 11};
    for (int r = 0; r < 9; r++) {
        ResBlock & R = rb[r];
        for (int j = 0; j < 3; j++) {
            R.k[j] = ks[r % 3];
            for (int c = 1; c <= 2; c++) {
                snprintf(b, sizeof(b), "hifi.rb.%d.c%d.%d.weight", r, c, j);
                (c == 1 ? R.c1[j].w : R.c2[j].w) = gt(b);
                snprintf(b, sizeof(b), "hifi.rb.%d.c%d.%d.bias", r, c, j);
                (c == 1 ? R.c1[j].b : R.c2[j].b) = gt(b);
            }
            snprintf(b, sizeof(b), "hifi.rb.%d.a1.%d", r, j); R.a1[j] = gt(b);
            snprintf(b, sizeof(b), "hifi.rb.%d.a2.%d", r, j); R.a2[j] = gt(b);
        }
    }
    for (int i = 0; i < 3; i++) {
        snprintf(b, sizeof(b), "hifi.sdown.%d.weight", i); sdown[i].w = gt(b);
        snprintf(b, sizeof(b), "hifi.sdown.%d.bias", i); sdown[i].b = gt(b);
    }
    for (int r = 0; r < 3; r++) {
        ResBlock & R = srb[r];
        for (int j = 0; j < 3; j++) {
            R.k[j] = sks[r];
            for (int c = 1; c <= 2; c++) {
                snprintf(b, sizeof(b), "hifi.srb.%d.c%d.%d.weight", r, c, j);
                (c == 1 ? R.c1[j].w : R.c2[j].w) = gt(b);
                snprintf(b, sizeof(b), "hifi.srb.%d.c%d.%d.bias", r, c, j);
                (c == 1 ? R.c1[j].b : R.c2[j].b) = gt(b);
            }
            snprintf(b, sizeof(b), "hifi.srb.%d.a1.%d", r, j); R.a1[j] = gt(b);
            snprintf(b, sizeof(b), "hifi.srb.%d.a2.%d", r, j); R.a2[j] = gt(b);
        }
    }
    conv_post_w = gt("hifi.conv_post.weight"); conv_post_b = gt("hifi.conv_post.bias");

    auto gc2 = [&](SpkConv2d & c2d, const char * n) {
        c2d.w = gt((std::string(n) + ".weight").c_str());
        c2d.b = gt((std::string(n) + ".bias").c_str());
        return c2d.w != nullptr;
    };
    if (!gc2(head_conv1, "spk.head.conv1") || !gc2(l10c1, "spk.layer1.0.conv1") ||
        !gc2(l10c2, "spk.layer1.0.conv2") || !gc2(l10sc, "spk.layer1.0.shortcut") ||
        !gc2(l11c1, "spk.layer1.1.conv1") || !gc2(l11c2, "spk.layer1.1.conv2") ||
        !gc2(l20c1, "spk.layer2.0.conv1") || !gc2(l20c2, "spk.layer2.0.conv2") ||
        !gc2(l20sc, "spk.layer2.0.shortcut") ||
        !gc2(l21c1, "spk.layer2.1.conv1") || !gc2(l21c2, "spk.layer2.1.conv2") ||
        !gc2(head_conv2, "spk.conv2") || !gc2(tdnn, "spk.tdnn"))
        return false;
    const int NB[3] = {12, 24, 16};
    for (int bi = 1; bi <= 3; bi++)
        for (int ti = 1; ti <= NB[bi - 1]; ti++) {
            Tdnnd & T = tdnnd[bi - 1][ti - 1];
            auto g = [&](const char * suf) {
                snprintf(b, sizeof(b), "spk.blk%d.tdnnd%d.%s", bi, ti, suf);
                return gt(b);
            };
            T.linear1_w = g("linear1.weight"); T.linear1_b = g("linear1.bias");
            T.cam_ll_w = g("cam_ll.weight");
            T.cam1_w = g("cam1.weight"); T.cam1_b = g("cam1.bias");
            T.cam2_w = g("cam2.weight"); T.cam2_b = g("cam2.bias");
        }
    for (int ti = 1; ti <= 3; ti++) {
        Transit & R = transit[ti - 1];
        snprintf(b, sizeof(b), "spk.transit%d.conv.weight", ti);
        R.conv.w = gt(b);
        snprintf(b, sizeof(b), "spk.transit%d.conv.bias", ti);
        R.conv.b = ggml_get_tensor(p->wctx, b);   // optional (baked in the BN)
    }
    {   // precompute the 55 entry BNs as x*iv + bb on the host
        struct ggml_init_params bp = { 8ull * 1024 * 1024, NULL, true };
        p->bn_ctx = ggml_init(bp);
        struct BnJob { ggml_tensor * w_, * b_, * m_, * v_, * * iv, * * bb; };
        std::vector<BnJob> jobs;
        for (int bi = 1; bi <= 3; bi++)
            for (int ti = 1; ti <= NB[bi - 1]; ti++) {
                Tdnnd & T = tdnnd[bi - 1][ti - 1];
                snprintf(b, sizeof(b), "spk.blk%d.tdnnd%d.bn.w", bi, ti);
                ggml_tensor * w_ = gt(b);
                snprintf(b, sizeof(b), "spk.blk%d.tdnnd%d.bn.b", bi, ti);
                ggml_tensor * b_ = gt(b);
                snprintf(b, sizeof(b), "spk.blk%d.tdnnd%d.bn.m", bi, ti);
                ggml_tensor * m_ = gt(b);
                snprintf(b, sizeof(b), "spk.blk%d.tdnnd%d.bn.v", bi, ti);
                ggml_tensor * v_ = gt(b);
                if (!w_) return false;
                jobs.push_back({ w_, b_, m_, v_, &T.bn_iv, &T.bn_bb });
            }
        for (int ti = 1; ti <= 3; ti++) {
            Transit & R = transit[ti - 1];
            snprintf(b, sizeof(b), "spk.transit%d.bn.w", ti);
            ggml_tensor * w_ = gt(b);
            snprintf(b, sizeof(b), "spk.transit%d.bn.b", ti);
            ggml_tensor * b_ = gt(b);
            snprintf(b, sizeof(b), "spk.transit%d.bn.m", ti);
            ggml_tensor * m_ = gt(b);
            snprintf(b, sizeof(b), "spk.transit%d.bn.v", ti);
            ggml_tensor * v_ = gt(b);
            if (!w_) return false;
            jobs.push_back({ w_, b_, m_, v_, &R.bn_iv, &R.bn_bb });
        }
        for (BnJob & j : jobs) {
            *j.iv = ggml_new_tensor_1d(p->bn_ctx, GGML_TYPE_F32, j.w_->ne[0]);
            *j.bb = ggml_new_tensor_1d(p->bn_ctx, GGML_TYPE_F32, j.w_->ne[0]);
        }
        // (k, ic)-ordered kernel copies for the per-tap conv1 fallback used
        // when ggml_im2col would mis-sample (dilation > 2). The strided
        // convs (sdown, s>2) keep the original (ic, k) order - their fallback
        // decimates a full stride-1 conv, which consumes the stored order.
        struct KiJob { ggml_tensor * src; ggml_tensor * * dst;
                       ggml_tensor * * wp; int K2; };
        std::vector<KiJob> kjobs;
        auto reorder_slot = [&](ggml_tensor ** wp, int K2) {
            kjobs.push_back({ *wp, wp, wp, K2 });
        };
        for (int r = 0; r < 9; r++) {
            reorder_slot(&rb[r].c1[1].w, rb[r].k[1]);
            reorder_slot(&rb[r].c1[2].w, rb[r].k[2]);
        }
        for (int r = 0; r < 3; r++) {
            reorder_slot(&srb[r].c1[1].w, srb[r].k[1]);
            reorder_slot(&srb[r].c1[2].w, srb[r].k[2]);
        }
        for (KiJob & j : kjobs)
            *j.dst = ggml_new_tensor_2d(p->bn_ctx, GGML_TYPE_F32,
                                        j.src->ne[0], j.src->ne[1]);

        ggml_backend_buffer_t bbuf =
            ggml_backend_alloc_ctx_tensors(p->bn_ctx, backend);
        p->bn_buf = bbuf;
        for (BnJob & j : jobs) {
            const int C = (int)j.w_->ne[0];
            std::vector<float> wv(C), bv(C), mv(C), vv(C), oiv(C), obb(C);
            ggml_backend_tensor_get(j.w_, wv.data(), 0, C * 4);
            ggml_backend_tensor_get(j.b_, bv.data(), 0, C * 4);
            ggml_backend_tensor_get(j.m_, mv.data(), 0, C * 4);
            ggml_backend_tensor_get(j.v_, vv.data(), 0, C * 4);
            for (int k = 0; k < C; k++) {
                oiv[k] = wv[k] / std::sqrt(vv[k] + 1e-5f);
                obb[k] = bv[k] - mv[k] * oiv[k];
            }
            ggml_backend_tensor_set(*j.iv, oiv.data(), 0, C * 4);
            ggml_backend_tensor_set(*j.bb, obb.data(), 0, C * 4);
        }
        for (KiJob & j : kjobs) {
            const int IC2 = (int)(j.src->ne[0] / j.K2), OC2 = (int)j.src->ne[1];
            const size_t R = (size_t)j.src->ne[0];   // rows = IC*K
            std::vector<float> in_(R * OC2);
            ggml_backend_tensor_get(j.src, in_.data(), 0, in_.size() * 4);
            std::vector<float> out_(in_.size());
            // ggml 2-D [IC*K, OC] flat layout: element (row, oc) at
            // row + oc*R. Rows are (ic, k) ic-major; the reordered copy has
            // rows (k, ic) so the per-tap gather (cols block k) matches.
            for (size_t oc = 0; oc < (size_t)OC2; oc++)
                for (int ic = 0; ic < IC2; ic++)
                    for (int k = 0; k < j.K2; k++)
                        out_[(size_t)(k * IC2 + ic) + oc * R] =
                            in_[(size_t)(ic * j.K2 + k) + oc * R];
            ggml_backend_tensor_set(*j.dst, out_.data(), 0, out_.size() * 4);
            *j.wp = *j.dst;   // repoint the conv to the reordered kernel
            // register the pre-reordered copy as ALREADY (k, ic)-ordered, or
            // voc_reordered_kernel would treat it as a stored (ic, k) kernel
            // and swap it back (double reorder -> per-tap gather desyncs)
            g_reord.emplace((*j.dst)->data, *j.dst);
        }
    }
    dense_w = gt("spk.dense.weight"); dense_b = gt("spk.dense.bias");

    p->hann16.resize(ISTFT_N);
    for (int k = 0; k < ISTFT_N; k++)
        p->hann16[k] = 0.5f - 0.5f * std::cos(2.f * (float)M_PI * k / ISTFT_N);

    if (getenv("MINICPMO_VOC_PROBE")) {   // minimal conv_1d sanity probe
        struct ggml_init_params pp = { 64ull * 1024 * 1024, NULL, true };
        ggml_context * pc = ggml_init(pp);
        ggml_cgraph * pgf = ggml_new_graph(pc);
        // isolated real-weight probe: f0 conv c0 on the gate mel, whisper-exact
        {
            ggml_tensor * wm = ggml_get_tensor(p->wctx, "hifi.f0.c0.weight");
            ggml_tensor * bm = ggml_get_tensor(p->wctx, "hifi.f0.c0.bias");
            ggml_tensor * x2 = ggml_new_tensor_2d(pc, GGML_TYPE_F32, 204, 80);
            ggml_backend_buffer_t qbuf = ggml_backend_alloc_ctx_tensors(pc, backend);
            {
                FILE * f3 = fopen("gate/gm_mel.bin", "rb");
                std::vector<float> xv((size_t)204 * 80);
                if (f3) { fread(xv.data(), 4, xv.size(), f3); fclose(f3); }
                ggml_backend_tensor_set(x2, xv.data(), 0, xv.size() * 4);
            }
            ggml_cgraph * qgf2 = ggml_new_graph(pc);
            // dump the exact im2col ggml builds internally (same params)
            ggml_tensor * ic = ggml_im2col(pc, wm, x2, 1, 0, 1, 0, 1, 0,
                                           false, GGML_TYPE_F32);
            ggml_build_forward_expand(qgf2, ic);
            FILE * f5 = fopen("gate/gm_im2col.f32", "wb");
            ggml_tensor * y2 = ggml_conv_1d(pc, wm, x2, 1, 1, 1);
            ggml_tensor * yb = ggml_add(pc, ggml_cont(pc, ggml_transpose(pc, y2)), bm);
            ggml_build_forward_expand(qgf2, yb);
            ggml_gallocr_t qal2 = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(qal2, qgf2)) return false;
            ggml_backend_graph_compute(backend, qgf2);
            {
                std::vector<float> iv((size_t)ggml_nelements(ic));
                ggml_backend_tensor_get(ic, iv.data(), 0, iv.size() * 4);
                fwrite(iv.data(), 4, iv.size(), f5); fclose(f5);
            }
            std::vector<float> yv((size_t)ggml_nelements(yb));
            ggml_backend_tensor_get(yb, yv.data(), 0, yv.size() * 4);
            FILE * f4 = fopen("gate/gm_conv0_out.f32", "wb");
            if (f4) { fwrite(yv.data(), 4, yv.size(), f4); fclose(f4); }
            fprintf(stderr, "vocoder probe: real c0 dumped (%lld)\n",
                    (long long)yv.size());
            ggml_backend_buffer_free(qbuf);
            ggml_free(pc);
            return true;   // probe mode stops here
        }
        struct Case { int k, s, p0; };
        const Case cases[] = { {3, 1, 1}, {3, 2, 1}, {5, 2, 2}, {5, 1, 2}, {7, 1, 3}, {11, 1, 5} };
        for (const Case & cs : cases) {
            char nm[64];
            snprintf(nm, sizeof(nm), "probe k%d s%d p%d", cs.k, cs.s, cs.p0);
            struct ggml_init_params ip2 = { 256ull * 1024 * 1024, NULL, true };
            ggml_context * qc = ggml_init(ip2);
            ggml_cgraph * qgf = ggml_new_graph(qc);
            ggml_tensor * x2 = ggml_new_tensor_2d(qc, GGML_TYPE_F32,
                                                  (int64_t)(204 + 2 * cs.p0), 80);
            ggml_tensor * w2 = ggml_new_tensor_3d(qc, GGML_TYPE_F32,
                                                  (int64_t)cs.k, 80, 64);
            ggml_backend_buffer_t qbuf = ggml_backend_alloc_ctx_tensors(qc, backend);
            {
                std::vector<float> xv((size_t)x2->ne[0] * 80);
                for (size_t i = 0; i < xv.size(); i++) xv[i] = std::sin(0.01f * (float)i) * 3.f;
                ggml_backend_tensor_set(x2, xv.data(), 0, xv.size() * 4);
                char fp[512];
                snprintf(fp, sizeof(fp), "%s_probe_x.f32",
                         getenv("MINICPMO_VOC_CP_DBG") ? getenv("MINICPMO_VOC_CP_DBG") : "gate/cp");
                FILE * f3 = fopen(fp, "wb");
                if (f3) { fwrite(xv.data(), 4, xv.size(), f3); fclose(f3); }
                std::vector<float> wv((size_t)cs.k * 80 * 64);
                for (size_t i = 0; i < wv.size(); i++) wv[i] = (float)(i % 97) * 0.013f - 0.6f;
                ggml_backend_tensor_set(w2, wv.data(), 0, wv.size() * 4);
            }
            ggml_tensor * y = ggml_conv_1d(qc, w2, x2, cs.s, cs.p0, 1);
            ggml_build_forward_expand(qgf, y);
            ggml_gallocr_t qal = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(qal, qgf)) {
                fprintf(stderr, "vocoder probe: alloc failed\n");
                return false;
            }
            ggml_backend_graph_compute(backend, qgf);
            std::vector<float> yv((size_t)ggml_nelements(y));
            ggml_backend_tensor_get(y, yv.data(), 0, yv.size() * 4);
            int nan = 0;
            for (float q : yv) if (std::isnan(q)) nan++;
            fprintf(stderr, "vocoder probe: %s out %lld nan=%d [%.4f..%.4f]\n",
                    nm, (long long)ggml_nelements(y), nan, yv[0], yv[yv.size() - 1]);
            {
                char fp[512];
                snprintf(fp, sizeof(fp), "%s_probe_out_%d_%d_%d.f32",
                         getenv("MINICPMO_VOC_CP_DBG") ? getenv("MINICPMO_VOC_CP_DBG") : "gate/cp",
                         cs.k, cs.s, cs.p0);
                FILE * f4 = fopen(fp, "wb");
                if (f4) { fwrite(yv.data(), 4, yv.size(), f4); fclose(f4); }
            }
            ggml_backend_buffer_free(qbuf);
            ggml_free(qc);
        }
    }

    if (const char * rd = getenv("MINICPMO_VOC_RNG_DIR")) {
        auto loadf = [&](const char * nm, std::vector<float> & out) {
            std::string fp = std::string(rd) + "/" + nm;
            FILE * f = fopen(fp.c_str(), "rb");
            if (!f) return false;
            fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
            out.resize(sz / 4);
            bool ok = fread(out.data(), 4, out.size(), f) == out.size();
            fclose(f);
            return ok;
        };
        p->gate_rng = loadf("rand_ini.f32", p->gate_rand_ini) &&
                      loadf("sine_noise.f32", p->gate_sine_noise) &&
                      loadf("src_noise.f32", p->gate_src_noise);
        fprintf(stderr, "vocoder: gate RNG %s\n", p->gate_rng ? "loaded" : "MISSING");
    }
    if (const char * s = getenv("MINICPMO_VOC_SEED")) p->rng = Vocrng((uint32_t)atoll(s));

    p->load_ok = true;
    fprintf(stderr, "vocoder: loaded (cfm %d+%d, dit %d x%d, campplus 12/24/16,"
            " prompt %d tok, rand %d)\n", n_cfm, n_cfm_up, n_dit, dit_d,
            n_prompt_tokens, n_rand);
    return true;
}

// ------------------------------------------------------ campplus forward
bool Vocoder::loaded() const { return p != nullptr && p->load_ok; }

bool Vocoder::run(const std::vector<int> & codes, std::vector<float> & pcm) {
    if (!p || !p->load_ok) return false;
    {
        ggml_tensor * w = ggml_get_tensor(p->wctx, "hifi.srb.0.c2.0.weight");
    }
    ggml_backend_t B = backend;
    if (getenv("MINICPMO_VOC_NTHREADS"))
        ggml_backend_cpu_set_n_threads(B, atoi(getenv("MINICPMO_VOC_NTHREADS")));
    {   // WORKAROUND: earlier graph build/compute stages overflow into this
        // object's heap block and smash ResBlock/layer weight pointer slots
        // (observed: srb[0].c2[0].w repointed into a gallocr pool). Rebind
        // every name-owned pointer at run() entry. Slots whose tensors were
        // intentionally replaced at load (the (k, ic)-reordered per-tap
        // kernels rb/srb c1[1]/c1[2], the reshaped up-conv views, and the
        // bn_ctx iv/bb tensors) are NOT rebound here.
        char b[96];
        auto g = [&](const char * n) { return ggml_get_tensor(p->wctx, n); };
        auto gl = [&](ConfLayer & L, int i, const char * pfx) {
            snprintf(b, sizeof(b), "cfm.%s%d.%s", pfx, i, "ln_mha.weight"); L.ln_mha_w = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.ln_mha.bias", pfx, i); L.ln_mha_b = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.ln_ff.weight", pfx, i); L.ln_ff_w = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.ln_ff.bias", pfx, i); L.ln_ff_b = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.wq.weight", pfx, i); L.wq = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.wq.bias", pfx, i); L.bq = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.wk.weight", pfx, i); L.wk = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.wk.bias", pfx, i); L.bk = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.wv.weight", pfx, i); L.wv = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.wv.bias", pfx, i); L.bv = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.wo.weight", pfx, i); L.wo = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.wo.bias", pfx, i); L.bo = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.wpos.weight", pfx, i); L.wpos = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.pos_bias_u", pfx, i); L.pos_u = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.pos_bias_v", pfx, i); L.pos_v = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.ffn_gate.weight", pfx, i); L.ffn_gate_w = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.ffn_gate.bias", pfx, i); L.ffn_gate_b = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.ffn_down.weight", pfx, i); L.ffn_down_w = g(b);
            snprintf(b, sizeof(b), "cfm.%s%d.ffn_down.bias", pfx, i); L.ffn_down_b = g(b);
        };
        for (int i = 0; i < n_cfm; i++) gl(cl[i], i, "blk.");
        for (int i = 0; i < n_cfm_up; i++) gl(cul[i], i, "ublkc.");
        for (int i = 0; i < n_dit; i++) {
            DiTLayer & L = dl[i];
            snprintf(b, sizeof(b), "dit.blk.%d.wq.weight", i); L.wq = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.wq.bias", i); L.bq = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.wk.weight", i); L.wk = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.wk.bias", i); L.bk = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.wv.weight", i); L.wv = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.wv.bias", i); L.bv = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.wo.weight", i); L.wo = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.wo.bias", i); L.bo = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.q_norm.weight", i); L.qn_w = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.q_norm.bias", i); L.qn_b = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.k_norm.weight", i); L.kn_w = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.k_norm.bias", i); L.kn_b = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.mlp_gate.weight", i); L.mlp_gate_w = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.mlp_gate.bias", i); L.mlp_gate_b = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.mlp_down.weight", i); L.mlp_down_w = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.mlp_down.bias", i); L.mlp_down_b = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.conv1.weight", i); L.conv1_w = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.conv1.bias", i); L.conv1_b = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.conv_ln.weight", i); L.conv_ln_w = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.conv_ln.bias", i); L.conv_ln_b = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.conv2.weight", i); L.conv2_w = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.conv2.bias", i); L.conv2_b = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.adaln.weight", i); L.adaln_w = g(b);
            snprintf(b, sizeof(b), "dit.blk.%d.adaln.bias", i); L.adaln_b = g(b);
        }
        for (int i = 0; i < 5; i++) {
            snprintf(b, sizeof(b), "hifi.f0.c%d.weight", i); f0_c_w[i] = g(b);
            snprintf(b, sizeof(b), "hifi.f0.c%d.bias", i); f0_c_b[i] = g(b);
        }
        conv_pre_w = g("hifi.conv_pre.weight"); conv_pre_b = g("hifi.conv_pre.bias");
        src_lin_w = g("hifi.src_linear.weight"); src_lin_b = g("hifi.src_linear.bias");
        for (int r = 0; r < 9; r++) {
            ResBlock & R = rb[r];
            snprintf(b, sizeof(b), "hifi.rb.%d.c1.0.weight", r); R.c1[0].w = g(b);
            snprintf(b, sizeof(b), "hifi.rb.%d.c1.0.bias", r); R.c1[0].b = g(b);
            snprintf(b, sizeof(b), "hifi.rb.%d.c2.0.weight", r); R.c2[0].w = g(b);
            snprintf(b, sizeof(b), "hifi.rb.%d.c2.0.bias", r); R.c2[0].b = g(b);
            snprintf(b, sizeof(b), "hifi.rb.%d.c1.1.bias", r); R.c1[1].b = g(b);
            snprintf(b, sizeof(b), "hifi.rb.%d.c2.1.weight", r); R.c2[1].w = g(b);
            snprintf(b, sizeof(b), "hifi.rb.%d.c2.1.bias", r); R.c2[1].b = g(b);
            snprintf(b, sizeof(b), "hifi.rb.%d.c1.2.bias", r); R.c1[2].b = g(b);
            snprintf(b, sizeof(b), "hifi.rb.%d.c2.2.weight", r); R.c2[2].w = g(b);
            snprintf(b, sizeof(b), "hifi.rb.%d.c2.2.bias", r); R.c2[2].b = g(b);
            for (int j = 0; j < 3; j++) {
                snprintf(b, sizeof(b), "hifi.rb.%d.a1.%d", r, j); R.a1[j] = g(b);
                snprintf(b, sizeof(b), "hifi.rb.%d.a2.%d", r, j); R.a2[j] = g(b);
            }
        }
        for (int i = 0; i < 3; i++) {
            snprintf(b, sizeof(b), "hifi.sdown.%d.weight", i); sdown[i].w = g(b);
            snprintf(b, sizeof(b), "hifi.sdown.%d.bias", i); sdown[i].b = g(b);
        }
        for (int r = 0; r < 3; r++) {
            ResBlock & R = srb[r];
            snprintf(b, sizeof(b), "hifi.srb.%d.c1.0.weight", r); R.c1[0].w = g(b);
            snprintf(b, sizeof(b), "hifi.srb.%d.c1.0.bias", r); R.c1[0].b = g(b);
            snprintf(b, sizeof(b), "hifi.srb.%d.c2.0.weight", r); R.c2[0].w = g(b);
            snprintf(b, sizeof(b), "hifi.srb.%d.c2.0.bias", r); R.c2[0].b = g(b);
            snprintf(b, sizeof(b), "hifi.srb.%d.c1.1.bias", r); R.c1[1].b = g(b);
            snprintf(b, sizeof(b), "hifi.srb.%d.c2.1.weight", r); R.c2[1].w = g(b);
            snprintf(b, sizeof(b), "hifi.srb.%d.c2.1.bias", r); R.c2[1].b = g(b);
            snprintf(b, sizeof(b), "hifi.srb.%d.c1.2.bias", r); R.c1[2].b = g(b);
            snprintf(b, sizeof(b), "hifi.srb.%d.c2.2.weight", r); R.c2[2].w = g(b);
            snprintf(b, sizeof(b), "hifi.srb.%d.c2.2.bias", r); R.c2[2].b = g(b);
            for (int j = 0; j < 3; j++) {
                snprintf(b, sizeof(b), "hifi.srb.%d.a1.%d", r, j); R.a1[j] = g(b);
                snprintf(b, sizeof(b), "hifi.srb.%d.a2.%d", r, j); R.a2[j] = g(b);
            }
        }
        conv_post_w = g("hifi.conv_post.weight"); conv_post_b = g("hifi.conv_post.bias");
        dense_w = g("spk.dense.weight"); dense_b = g("spk.dense.bias");
    }
    const int T2p = n_prompt_mels;
    const int n_gen = (int)codes.size();
    const int Ttok = n_prompt_tokens + n_gen;
    const int T2 = 2 * Ttok;
    const int Tmel = n_gen * 2;
    const int D = flow_d, NH = flow_heads, HD = flow_hd;
    pcm.clear();
    std::vector<float> flat;

    // ============================ campplus =================================
    {
        if (p->cf_ctx) { ggml_free(p->cf_ctx); p->cf_ctx = nullptr; }
        struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, true };
        p->cf_ctx = ggml_init(ip);
        ggml_context * c = p->cf_ctx;
        ggml_cgraph * gf = ggml_new_graph_custom(c, 16384, false);
        ggml_tensor * in = ggml_new_tensor_4d(c, GGML_TYPE_F32, n_fbank, 80, 1, 1);
        ggml_set_name(in, "cf_in");
        ggml_set_input(in);
        p->cf_in = in;

        auto b4 = [&](ggml_tensor * b) {              // conv2d bias [C] -> [1,1,C,1]
            return ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1);
        };
        auto c2 = [&](ggml_tensor * w, ggml_tensor * bias, ggml_tensor * x,
                      int sh) {                     // k3x3 p1, stride (1, sh)
            if (getenv("MINICPMO_VOC_DEBUG"))
                fprintf(stderr, "vocoder c2: w ne %lld/%lld/%lld/%lld x ne %lld/%lld/%lld/%lld sh %d\n",
                        (long long)w->ne[0], (long long)w->ne[1], (long long)w->ne[2], (long long)w->ne[3],
                        (long long)x->ne[0], (long long)x->ne[1], (long long)x->ne[2], (long long)x->ne[3], sh);
            ggml_tensor * y = ggml_conv_2d(c, w, x, 1, sh, 1, 1, 1, 1);
            y = ggml_add(c, y, b4(bias));
            return ggml_relu(c, y);
        };
        auto stage = [&](ggml_tensor * x, SpkConv2d & s1, SpkConv2d & s2,
                         SpkConv2d * sc, int sh) {
            ggml_tensor * h = c2(s1.w, s1.b, x, sh);
            ggml_tensor * cv2 = ggml_conv_2d(c, s2.w, h, 1, 1, 1, 1, 1, 1);
            cv2 = ggml_add(c, cv2, b4(s2.b));
            if (sc) {   // downsampling unit: shortcut replaces the identity
                ggml_tensor * y2 = ggml_conv_2d(c, sc->w, x, 1, sh, 0, 0, 1, 1);
                y2 = ggml_add(c, y2, b4(sc->b));
                h = ggml_add(c, cv2, y2);
            } else {    // plain unit: identity residual
                h = ggml_add(c, cv2, x);
            }
            return ggml_relu(c, h);
        };
        const bool gate = getenv("MINICPMO_VOC_CP_DBG") != nullptr;
        auto sink = [&](ggml_tensor * v, const char * nm) {
            ggml_tensor * s = ggml_dup(c, v);
            ggml_set_name(s, nm);
            ggml_set_output(s);
            ggml_build_forward_expand(gf, s);
        };
        ggml_tensor * x = c2(head_conv1.w, head_conv1.b, in, 1);
        if (gate) { sink(in, "cp_in"); sink(x, "h_stem"); }
        x = stage(x, l10c1, l10c2, &l10sc, 2);
        if (gate) sink(x, "h_l10");
        x = stage(x, l11c1, l11c2, nullptr, 1);
        if (gate) sink(x, "h_l11");
        x = stage(x, l20c1, l20c2, &l20sc, 2);
        if (gate) sink(x, "h_l20");
        x = stage(x, l21c1, l21c2, nullptr, 1);
        if (gate) sink(x, "h_l21");
        x = c2(head_conv2.w, head_conv2.b, x, 2);
        fprintf(stderr, "vocoder cp: head ok ne=%lldx%lldx%lld\n",
                (long long)x->ne[0], (long long)x->ne[1], (long long)x->ne[2]);
        // [W, 10, 32] -> fold channels c' = c*10 + h: pure view. Element
        // (w, h, c) sits at w + h*W + c*10*W = w + (c*10 + h)*W -> row
        // stride is W (= nb[1]).
        const int W2 = (int)x->ne[0];
        ggml_tensor * fold = ggml_cont(c, ggml_view_2d(c, x, W2, 320,
                                                       x->nb[1], 0));
        if (getenv("MINICPMO_VOC_CP_DBG")) {   // parity: head + fold sinks
            ggml_tensor * s0 = ggml_dup(c, x);
            ggml_set_name(s0, "cp_head");
            ggml_set_output(s0);   // gallocr must not recycle sink memory
            ggml_build_forward_expand(gf, s0);
            ggml_tensor * s1 = ggml_dup(c, fold);
            ggml_set_name(s1, "cp_fold");
            ggml_set_output(s1);
            ggml_build_forward_expand(gf, s1);
            p->cf_out = s1;                    // extra sinks resolved below
        }
        // tdnn conv k5 s2 p2 (planar [W2, 320]) - via the explicit
        // im2col + mul_mat conv1 helper (same kernel as every other conv).
        ggml_tensor * t = conv1(c, tdnn.w, tdnn.b, fold, 2, 2, 1);
        ggml_set_name(t, "cp_tdnn");
        if (getenv("MINICPMO_VOC_CP_DBG")) ggml_set_output(t);
        fprintf(stderr, "vocoder cp: tdnn ok ne=%lldx%lld\n",
                (long long)t->ne[0], (long long)t->ne[1]);
        const int Tt = (int)t->ne[0];
        // entry batch-norm (precomputed x*iv + bb) + relu on a planar
        // [T, C] tensor (channel axis ne[1])
        auto bn_relu = [&](ggml_tensor * x, ggml_tensor * iv, ggml_tensor * bb) {
            ggml_tensor * y = ggml_mul(c, x, ggml_reshape_2d(c, iv, 1,
                                                             iv->ne[0]));
            y = ggml_add(c, y, ggml_reshape_2d(c, bb, 1, bb->ne[0]));
            return ggml_relu(c, y);
        };
        ggml_tensor * cur = ggml_relu(c, t);   // concat base r1 (pre-BN!)
        int Ci = 128;
        const int NB[3] = {12, 24, 16};
        ggml_tensor * blk_out[3];
        for (int bi = 1; bi <= 3; bi++) {
            for (int ti = 1; ti <= NB[bi - 1]; ti++) {
                const Tdnnd & T = tdnnd[bi - 1][ti - 1];
                // entry BN (full running-stats normalize) + relu; the concat
                // base `cur` itself stays un-normalized for the next concat
                ggml_tensor * h = bn_relu(cur, T.bn_iv, T.bn_bb);
                if (gate && bi == 2 && ti == 1) sink(h, "c2_bn");
                if (gate && bi == 3 && ti == 1) sink(h, "c3_bn");
                if (gate && bi == 2 && ti == 2) sink(h, "c2b_bn");
                // linear1 1x1: planar -> d-fast -> mul_mat
                ggml_tensor * hd = ggml_mul_mat(c, T.linear1_w, to_dfast(c, h));
                hd = ggml_add(c, hd, T.linear1_b);
                hd = ggml_relu(c, hd);
                if (gate && bi == 2 && ti == 1) sink(hd, "c2_h");
                if (gate && bi == 3 && ti == 1) sink(hd, "c3_h");
                // cam local context conv k3 (on planar); blocks 2/3 use
                // dilation 2 with pads 2 (ONNX attrs), block1 is plain d1 p1
                const int ldil = (bi == 1) ? 1 : 2;
                ggml_tensor * hd_planar = to_planar(c, hd);
                ggml_tensor * ctx = conv1(c, T.cam_ll_w, nullptr, hd_planar,
                                          1, ldil, ldil);
                if (gate && bi == 1 && ti == 1) {
                    sink(hd, "b1_h");        // nonlinear2 relu out (d-fast)
                    sink(ctx, "b1_ctx");     // linear_local conv out (planar)
                }
                if (gate && bi == 2 && ti == 1) {
                    sink(ctx, "c2_ctx");
                    sink(hd_planar, "c2_hdp");
                }
                // stats: global mean + 100-frame window means
                ggml_tensor * gmean = ggml_scale(c, ggml_sum_rows(c, hd_planar),
                                                 1.0f / Tt);          // [1, 128]
                ggml_tensor * wmean = ggml_pool_1d(c, hd_planar,
                                                   GGML_OP_POOL_AVG, 100, 100, 0);
                wmean = ggml_interpolate(c, wmean, Tt, wmean->ne[1], 1, 1,
                                         GGML_SCALE_MODE_NEAREST);
                ggml_tensor * att_in = ggml_add(c, wmean, gmean);   // [Tt, 128]
                if (gate && bi == 1 && ti == 1) sink(att_in, "b1_att");
                if (gate && bi == 2 && ti == 1) sink(att_in, "c2_att");
                ggml_tensor * a = ggml_mul_mat(c, T.cam1_w, to_dfast(c, att_in));
                a = ggml_add(c, a, T.cam1_b);
                a = ggml_relu(c, a);
                a = ggml_mul_mat(c, T.cam2_w, a);
                a = ggml_add(c, a, T.cam2_b);
                a = ggml_sigmoid(c, a);                              // [32, Tt]
                if (gate && bi == 1 && ti == 1) sink(a, "b1_a");
                if (gate && bi == 2 && ti == 1) sink(a, "c2_a");
                ctx = ggml_mul(c, ctx, to_planar(c, a));
                if (gate && bi == 1 && ti == 1) sink(ctx, "b1_mul");
                if (gate && bi == 2 && ti == 1) sink(ctx, "c2_mul");
                cur = ggml_concat(c, cur, ctx, 1);                   // planar
                if (gate && bi == 2 && ti == 1) sink(cur, "c2_cat");
                Ci += 32;
            }
            // transit: BN(cat) -> relu -> conv 1x1; the next block's entry BN
            // normalizes this conv output directly (bi=3: relu instead)
            Transit & tr = transit[bi - 1];
            ggml_tensor * h = bn_relu(cur, tr.bn_iv, tr.bn_bb);
            ggml_tensor * cd = ggml_mul_mat(c, tr.conv.w, to_dfast(c, h));
            if (tr.conv.b) cd = ggml_add(c, cd, tr.conv.b);
            if (gate && bi == 1) sink(cd, "b1_tr");
            if (bi < 3) cur = to_planar(c, cd);
            else cur = ggml_relu(c, cd);                    // [512, Tt] d-fast
            {
                char bn[32];
                snprintf(bn, sizeof(bn), "cp_blk%d", bi);
                ggml_set_name(cur, bn);
                if (getenv("MINICPMO_VOC_CP_DBG")) ggml_set_output(cur);
            }
            blk_out[bi - 1] = cur;
        }
        // stats: mean + unbiased std over time (planar [Tt, 512]);
        // unbiased var = sum(dev^2) / (Tt - 1)
        ggml_tensor * hp = to_planar(c, cur);
        ggml_tensor * mean = ggml_scale(c, ggml_sum_rows(c, hp), 1.0f / Tt);
        ggml_tensor * dev = ggml_sub(c, hp, mean);
        ggml_tensor * var = ggml_scale(c, ggml_sum_rows(c, ggml_mul(c, dev, dev)),
                                       1.0f / (float)(Tt - 1));
        ggml_tensor * stats = ggml_concat(c, mean, ggml_sqrt(c, var), 1);
        if (gate) sink(stats, "cp_stats");
        ggml_tensor * emb = ggml_mul_mat(c, dense_w,
                                         to_dfast(c, stats));
        emb = ggml_add(c, emb, dense_b);
        if (gate) sink(emb, "cp_emb");
        ggml_set_name(emb, "cf_out");
        p->cf_out = emb;
        ggml_build_forward_expand(gf, emb);
        if (!ggml_gallocr_alloc_graph(p->allocr_cf, gf)) {
            fprintf(stderr, "vocoder: campplus alloc failed\n");
            return false;
        }
        {   // upload fbank as [W=T, H=80] planes
            std::vector<float> fb((size_t)n_fbank * 80);
            flat.resize((size_t)ggml_nelements(spk_fbank));
            ggml_backend_tensor_get(spk_fbank, flat.data(), 0,
                                    flat.size() * sizeof(float));
            for (int w = 0; w < n_fbank; w++)
                for (int m = 0; m < 80; m++)
                    fb[(size_t)m * n_fbank + w] = flat[(size_t)w * 80 + m];
            ggml_backend_tensor_set(in, fb.data(), 0, fb.size() * sizeof(float));
        }
        if (ggml_backend_graph_compute(B, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "vocoder: campplus compute failed\n");
            return false;
        }
        p->spk.resize((size_t)ggml_nelements(emb));
        ggml_backend_tensor_get(emb, p->spk.data(), 0,
                                p->spk.size() * sizeof(float));
        if (getenv("MINICPMO_VOC_CP_DBG")) {      // parity: stage sinks
            {   // weight sanity: read the stem kernel back from the backend
                std::vector<float> wv((size_t)9 * 32);
                ggml_backend_tensor_get(head_conv1.w, wv.data(), 0, wv.size() * 4);
                std::string wp = std::string(getenv("MINICPMO_VOC_CP_DBG"))
                                 + "_stemw.f32";
                FILE * fw = fopen(wp.c_str(), "wb");
                if (fw) { fwrite(wv.data(), 4, wv.size(), fw); fclose(fw); }
            }
            auto dumpn = [&](const char * nm, const char * suf) {
                ggml_tensor * t = ggml_graph_get_tensor(gf, nm);
                if (!t) return;
                std::vector<float> v((size_t)ggml_nelements(t));
                ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
                std::string fp = std::string(getenv("MINICPMO_VOC_CP_DBG"))
                                 + suf + ".f32";
                FILE * f = fopen(fp.c_str(), "wb");
                if (f) { fwrite(v.data(), 4, v.size(), f); fclose(f); }
                float mn = v[0], mx = v[0]; int nan = 0;
                for (float q : v) { if (std::isnan(q)) nan++; if (q < mn) mn = q; if (q > mx) mx = q; }
                fprintf(stderr, "vocoder cp sink %s nan=%d [%.4f..%.4f]\n",
                        suf, nan, mn, mx);
            };
            dumpn("cp_head", "_cp_head");
            dumpn("cp_fold", "_cp_fold");
            dumpn("cp_tdnn", "_cp_tdnn");
            dumpn("cp_in", "_cp_in");
            dumpn("h_stem", "_h_stem");
            dumpn("h_l10", "_h_l10");
            dumpn("h_l11", "_h_l11");
            dumpn("h_l20", "_h_l20");
            dumpn("h_l21", "_h_l21");
            dumpn("b1_h", "_b1_h");
            dumpn("b1_ctx", "_b1_ctx");
            dumpn("b1_att", "_b1_att");
            dumpn("b1_a", "_b1_a");
            dumpn("b1_mul", "_b1_mul");
            dumpn("c2_bn", "_c2_bn");
            dumpn("c2_h", "_c2_h");
            dumpn("c2_cat", "_c2_cat");
            dumpn("c2b_bn", "_c2b_bn");
            dumpn("c2_ctx", "_c2_ctx");
            dumpn("c2_hdp", "_c2_hdp");
            dumpn("c2_att", "_c2_att");
            dumpn("c2_a", "_c2_a");
            dumpn("c2_mul", "_c2_mul");
            dumpn("c3_bn", "_c3_bn");
            dumpn("c3_h", "_c3_h");
            dumpn("cp_stats", "_cp_stats");
            dumpn("cp_emb", "_cp_emb");
            dumpn("c3_ctx", "_c3_ctx");
            dumpn("b1_tr", "_b1_tr");
            dumpn("cp_blk1", "_cp_blk1");
            dumpn("cp_blk2", "_cp_blk2");
            dumpn("cp_blk3", "_cp_blk3");
        }
        if (getenv("MINICPMO_VOC_DUMP")) {
            std::string dp = std::string(getenv("MINICPMO_VOC_DUMP")) + "_spk.f32";
            FILE * f = fopen(dp.c_str(), "wb");
            if (f) { fwrite(p->spk.data(), 4, p->spk.size(), f); fclose(f); }
        }
fprintf(stderr, "vocoder stage: %s\n", "campplus");
        p->cf_gf = gf;
    }

    // ============================ flow encoder =============================
    if (p->fe_T != Ttok) {
        if (p->fe_ctx) { ggml_free(p->fe_ctx); p->fe_ctx = nullptr; }
        struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, true };
        p->fe_ctx = ggml_init(ip);
        ggml_context * c = p->fe_ctx;
        ggml_cgraph * gf = ggml_new_graph_custom(c, 32768, false);
        const int T = Ttok;
        ggml_tensor * tok = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
        ggml_set_input(tok); ggml_set_name(tok, "fe_tok");
        // espnet rel-pos tables, host-computed [512, 2T-1] / [512, 2T2-1]
        const int P1 = 2 * T - 1, P2 = 2 * T2 - 1;
        ggml_tensor * pos1 = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, P1);
        ggml_set_input(pos1); ggml_set_name(pos1, "fe_pos1");
        ggml_tensor * pos2 = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, P2);
        ggml_set_input(pos2); ggml_set_name(pos2, "fe_pos2");
        p->fe_tok = tok; p->fe_pos1 = pos1; p->fe_pos2 = pos2;


        const float xscale = std::sqrt((float)D);
        // rel-pos attention over the espnet pe table pos [512, P]
        auto conf_attn = [&](ggml_context * cc, ggml_tensor * l,
                             const ConfLayer & L, ggml_tensor * pos,
                             int Tq) -> ggml_tensor * {
            ggml_tensor * q = ggml_mul_mat(cc, L.wq, l);
            q = ggml_add(cc, q, L.bq);
            ggml_tensor * k = ggml_mul_mat(cc, L.wk, l);
            k = ggml_add(cc, k, L.bk);
            ggml_tensor * v = ggml_mul_mat(cc, L.wv, l);
            v = ggml_add(cc, v, L.bv);
            ggml_tensor * p_ = ggml_mul_mat(cc, L.wpos, pos);      // [512, P]
            if (getenv("MINICPMO_VOC_DEBUG"))
                fprintf(stderr, "vocoder fe: attn wq %lldx%lld l %lldx%lld pos %lldx%lld p %lldx%lld\n",
                        (long long)L.wq->ne[0], (long long)L.wq->ne[1],
                        (long long)l->ne[0], (long long)l->ne[1],
                        (long long)pos->ne[0], (long long)pos->ne[1],
                        (long long)p_->ne[0], (long long)p_->ne[1]);
            const int Pl = (int)p_->ne[1];
            ggml_tensor * qu = ggml_add(cc, q, L.pos_u);
            ggml_tensor * qv = ggml_add(cc, q, L.pos_v);
            // [64, Tq, NH] layouts (memory (64, NH, Tq))
            ggml_tensor * qu_r = ggml_permute(cc, ggml_reshape_3d(cc, qu, HD, NH, Tq), 0, 2, 1, 3);
            ggml_tensor * qv_r = ggml_permute(cc, ggml_reshape_3d(cc, qv, HD, NH, Tq), 0, 2, 1, 3);
            ggml_tensor * k_r = ggml_permute(cc, ggml_reshape_3d(cc, k, HD, NH, Tq), 0, 2, 1, 3);
            ggml_tensor * v_r = ggml_permute(cc, ggml_reshape_3d(cc, v, HD, NH, Tq), 0, 2, 1, 3);
            ggml_tensor * p_r = ggml_permute(cc, ggml_reshape_3d(cc, p_, HD, NH, Pl), 0, 2, 1, 3);   // [64, P, NH]
            ggml_tensor * ac = ggml_mul_mat(cc, k_r, qu_r);        // [Tq, Tq, NH]
            if (getenv("MINICPMO_VOC_DEBUG"))
                fprintf(stderr, "vocoder fe: ac ok\n");
            ggml_tensor * bd = ggml_mul_mat(cc, p_r, qv_r);        // [P, Tq, NH]
            if (getenv("MINICPMO_VOC_DEBUG"))
                fprintf(stderr, "vocoder fe: bd ok\n");
            // rel_shift: out[q, k] = bd[k - q + Tq - 1]; front-pad one key
            // slot and read the (P-1)-strided diagonal (espnet skew trick)
            ggml_tensor * zero = ggml_fill(cc, ggml_new_tensor_3d(
                cc, GGML_TYPE_F32, 1, Tq, NH), 0.0f);
            ggml_tensor * bdp = ggml_concat(cc, zero, bd, 0);      // [P+1, Tq, NH]
            ggml_tensor * bdf = ggml_reshape_2d(cc, bdp, (Pl + 1) * Tq, NH);
            ggml_tensor * bds = ggml_view_3d(cc, bdf, Tq, Tq, NH,
                                             (Pl - 1) * bdf->nb[0],
                                             (size_t)Pl * Tq * bdf->nb[0],
                                             (size_t)Tq * bdf->nb[0]);
            ggml_tensor * scores = ggml_scale(cc, ggml_add(cc, ac, bds),
                                              1.0f / std::sqrt((float)HD));
            ggml_tensor * probs = ggml_soft_max(cc, scores);
            // out[d, q] = sum_k probs[k, q] v[d, k]: A = v transposed [Tq, 64]
            ggml_tensor * v_t = ggml_cont(cc, ggml_transpose(cc, v_r));
            ggml_tensor * attn = ggml_mul_mat(cc, v_t, probs);     // [64, Tq, NH]
            attn = ggml_cont(cc, ggml_permute(cc, attn, 0, 2, 1, 3));
            attn = ggml_reshape_2d(cc, attn, D, Tq);
            return ggml_add(cc, ggml_mul_mat(cc, L.wo, attn), L.bo);
        };
        auto conf_layer = [&](ggml_context * cc, ggml_tensor * xdf,
                              const ConfLayer & L, ggml_tensor * pos,
                              int Tq) -> ggml_tensor * {
            ggml_tensor * l = ggml_norm(cc, xdf, conf_ln_eps);
            l = ggml_mul(cc, l, L.ln_mha_w);
            l = ggml_add(cc, l, L.ln_mha_b);
            xdf = ggml_add(cc, xdf, conf_attn(cc, l, L, pos, Tq));
            l = ggml_norm(cc, xdf, conf_ln_eps);
            l = ggml_mul(cc, l, L.ln_ff_w);
            l = ggml_add(cc, l, L.ln_ff_b);
            ggml_tensor * f = ggml_mul_mat(cc, L.ffn_gate_w, l);
            f = ggml_add(cc, f, L.ffn_gate_b);
            f = ggml_silu(cc, f);
            f = ggml_mul_mat(cc, L.ffn_down_w, f);
            f = ggml_add(cc, f, L.ffn_down_b);
            return ggml_add(cc, xdf, f);
        };

        ggml_tensor * x = ggml_get_rows(c, input_emb, tok);        // [512, T]
        x = ggml_mul_mat(c, embed_w, x);
        x = ggml_add(c, x, embed_b);
        x = ggml_norm(c, x, embed_ln_eps);
        x = ggml_mul(c, x, embed_ln_w);
        x = ggml_add(c, x, embed_ln_b);
        x = ggml_scale(c, x, xscale);
        // pre-lookahead (planar convs)
        ggml_tensor * xl = to_planar(c, x);
        xl = ggml_concat(c, xl, ggml_new_tensor_2d(c, GGML_TYPE_F32, 3, D), 0);
        xl = conv1(c, la_conv1_w, la_conv1_b, xl, 1, 0, 1);
        xl = ggml_leaky_relu(c, xl, LRELU01, false);
        xl = pad_left(c, xl, 2);
        xl = conv1(c, la_conv2_w, la_conv2_b, xl, 1, 0, 1);
        x = ggml_add(c, to_dfast(c, xl), x);                        // residual
        for (int i = 0; i < n_cfm; i++)
            x = conf_layer(c, x, cl[i], pos1, T);
        // upsample: nearest x2, left-pad 4, conv k5
        ggml_tensor * xu = to_planar(c, x);
        xu = ggml_interpolate(c, xu, T * 2, D, 1, 1, GGML_SCALE_MODE_NEAREST);
        xu = pad_left(c, xu, 4);
        xu = conv1(c, up_conv_w, up_conv_b, xu, 1, 0, 1);
        x = to_dfast(c, xu);
        x = ggml_mul_mat(c, up_embed_w, x);
        x = ggml_add(c, x, up_embed_b);
        x = ggml_norm(c, x, embed_ln_eps);
        x = ggml_mul(c, x, up_embed_ln_w);
        x = ggml_add(c, x, up_embed_ln_b);
        x = ggml_scale(c, x, xscale);
        for (int i = 0; i < n_cfm_up; i++)
            x = conf_layer(c, x, cul[i], pos2, T2);
        x = ggml_norm(c, x, embed_ln_eps);
        x = ggml_mul(c, x, after_norm_w);
        x = ggml_add(c, x, after_norm_b);
        ggml_tensor * mu = ggml_mul_mat(c, proj_w, x);
        mu = ggml_add(c, mu, proj_b);                               // [80, T2]
        ggml_set_name(mu, "fe_mu");
        p->fe_mu = mu;
        ggml_build_forward_expand(gf, mu);
        if (!ggml_gallocr_alloc_graph(p->allocr_fe, gf)) {
            fprintf(stderr, "vocoder: flow encoder alloc failed\n");
            return false;
        }
        // uploads: tokens = prompt ++ codes
        std::vector<int> toks((size_t)T);
        flat.resize((size_t)n_prompt_tokens);
        ggml_backend_tensor_get(prompt_tokens, flat.data(), 0,
                                flat.size() * sizeof(float));
        for (int i = 0; i < n_prompt_tokens; i++)
            toks[(size_t)i] = (int)flat[(size_t)i];
        for (int i = 0; i < n_gen; i++) toks[(size_t)(n_prompt_tokens + i)] = codes[(size_t)i];
        ggml_backend_tensor_set(tok, toks.data(), 0, sizeof(int) * T);
        // espnet pe tables
        auto build_pe = [&](std::vector<float> & out, int len) {
            const int P = 2 * len - 1;
            out.assign((size_t)D * P, 0.f);
            // pe_positive flipped + negative, d-fast [512, P]
            for (int i = 0; i < len; i++) {          // positive (flipped rows)
                int row = len - 1 - i;               // flip
                for (int d = 0; d < D; d++) {
                    float pos_i = (float)i;
                    float div = std::exp(-(float)std::log(10000.0) * (float)((d >> 1)) / (float)D);
                    float val = (d % 2 == 0) ? std::sin(pos_i * div) : std::cos(pos_i * div);
                    out[(size_t)d * P + row] = val;
                }
            }
            for (int i = 1; i < len; i++) {          // negative rows
                int row = len - 1 + i;
                for (int d = 0; d < D; d++) {
                    float pos_i = -(float)i;
                    float div = std::exp(-(float)std::log(10000.0) * (float)((d >> 1)) / (float)D);
                    float val = (d % 2 == 0) ? std::sin(pos_i * div) : std::cos(pos_i * div);
                    out[(size_t)d * P + row] = val;
                }
            }
        };
        static std::vector<float> pe1, pe2;
        if ((int)pe1.size() != D * P1) build_pe(pe1, T);
        if ((int)pe2.size() != D * P2) build_pe(pe2, T2);
        ggml_backend_tensor_set(pos1, pe1.data(), 0, pe1.size() * sizeof(float));
        ggml_backend_tensor_set(pos2, pe2.data(), 0, pe2.size() * sizeof(float));
        p->fe_gf = gf;
        p->fe_T = Ttok;
    }
    if (ggml_backend_graph_compute(B, p->fe_gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vocoder: flow encoder compute failed\n");
        return false;
    }
fprintf(stderr, "vocoder stage: %s\n", "flow enc");
    std::vector<float> mu((size_t)n_mel * T2);
    ggml_backend_tensor_get(p->fe_mu, mu.data(), 0, mu.size() * sizeof(float));

    // conds [80, T2]: prompt mels replicated to T2p then zeros
    std::vector<float> conds((size_t)n_mel * T2, 0.f);
    {
        flat.resize((size_t)n_prompt_mels * n_mel);
        ggml_backend_tensor_get(prompt_mels, flat.data(), 0, flat.size() * sizeof(float));
        for (int m = 0; m < n_mel; m++)
            for (int t = 0; t < T2p; t++)
                conds[(size_t)m * T2 + t] = flat[(size_t)t * n_mel + m];
    }
    // spk80 = affine(L2(spk))
    std::vector<float> spk80(n_mel);
    {
        std::vector<float> w, bw;
        host_f32(spk_proj_w, w);
        host_f32(spk_proj_b, bw);
        double sq = 0;
        for (float v : p->spk) sq += (double)v * v;
        float inv = (float)(1.0 / std::sqrt(std::max(sq, 1e-12)));
        for (int m = 0; m < n_mel; m++) {
            float acc = 0.f;
            for (int j = 0; j < 192; j++)
                acc += w[(size_t)m * 192 + j] * p->spk[(size_t)j] * inv;
            spk80[(size_t)m] = acc + bw[(size_t)m];
        }
    }

    // ============================ CFM (DiT x euler) ========================
    if (p->dit_T != T2) {
        if (p->dit_ctx) { ggml_free(p->dit_ctx); p->dit_ctx = nullptr; }
        struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, true };
        p->dit_ctx = ggml_init(ip);
        ggml_context * c = p->dit_ctx;
        ggml_cgraph * gf = ggml_new_graph_custom(c, 32768, false);
        auto in2 = [&](const char * n, int64_t d0, int64_t d1) {
            ggml_tensor * t = ggml_new_tensor_2d(c, GGML_TYPE_F32, d0, d1);
            ggml_set_input(t); ggml_set_name(t, n);
            return t;
        };
        p->d_x = in2("d_x", n_mel, T2);
        p->d_mu = in2("d_mu", n_mel, T2);
        p->d_spk = in2("d_spk", n_mel, T2);
        p->d_cond = in2("d_cond", n_mel, T2);
        p->d_t = in2("d_t", dit_d, 1);
        // all-visible F16 mask for flash attention
        p->d_mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, T2, T2);
        ggml_tensor * cat = p->d_x;
        cat = ggml_concat(c, cat, p->d_mu, 0);
        cat = ggml_concat(c, cat, p->d_spk, 0);
        cat = ggml_concat(c, cat, p->d_cond, 0);                   // [320, T2]
        ggml_tensor * x = ggml_mul_mat(c, in_proj_w, cat);
        x = ggml_add(c, x, in_proj_b);
        const int Tq = T2;
        for (int i = 0; i < n_dit; i++) {
            const DiTLayer & L = dl[i];
            ggml_tensor * mod = ggml_mul_mat(c, L.adaln_w, ggml_silu(c, p->d_t));
            mod = ggml_add(c, mod, L.adaln_b);                     // [4608, 1]
            const int64_t rb = mod->nb[1];
            ggml_tensor * sh_msa = ggml_view_2d(c, mod, dit_d, 1, rb, 0);
            ggml_tensor * sc_msa = ggml_view_2d(c, mod, dit_d, 1, rb, (size_t)dit_d * 4);
            ggml_tensor * ga_msa = ggml_view_2d(c, mod, dit_d, 1, rb, (size_t)(2 * dit_d) * 4);
            ggml_tensor * sh_mlp = ggml_view_2d(c, mod, dit_d, 1, rb, (size_t)(3 * dit_d) * 4);
            ggml_tensor * sc_mlp = ggml_view_2d(c, mod, dit_d, 1, rb, (size_t)(4 * dit_d) * 4);
            ggml_tensor * ga_mlp = ggml_view_2d(c, mod, dit_d, 1, rb, (size_t)(5 * dit_d) * 4);
            ggml_tensor * sh_cnv = ggml_view_2d(c, mod, dit_d, 1, rb, (size_t)(6 * dit_d) * 4);
            ggml_tensor * sc_cnv = ggml_view_2d(c, mod, dit_d, 1, rb, (size_t)(7 * dit_d) * 4);
            ggml_tensor * ga_cnv = ggml_view_2d(c, mod, dit_d, 1, rb, (size_t)(8 * dit_d) * 4);
            auto modulate = [&](ggml_tensor * xx, ggml_tensor * sh, ggml_tensor * sc) {
                ggml_tensor * n = ggml_norm(c, xx, dit_eps);
                return ggml_add(c, ggml_add(c, n, ggml_mul(c, n, sc)), sh);
            };
            ggml_tensor * h = modulate(x, sh_msa, sc_msa);
            // attention with qk-norm
            ggml_tensor * q = ggml_mul_mat(c, L.wq, h);
            q = ggml_add(c, q, L.bq);
            ggml_tensor * k = ggml_mul_mat(c, L.wk, h);
            k = ggml_add(c, k, L.bk);
            ggml_tensor * v = ggml_mul_mat(c, L.wv, h);
            v = ggml_add(c, v, L.bv);
            auto qkn = [&](ggml_tensor * t4, ggml_tensor * nw, ggml_tensor * nb) {
                ggml_tensor * r = ggml_norm(c, t4, dit_qk_eps);    // over HD
                r = ggml_mul(c, r, nw);
                return ggml_add(c, r, nb);
            };
            ggml_tensor * q3 = ggml_reshape_3d(c, q, HD, NH, Tq);
            ggml_tensor * k3 = ggml_reshape_3d(c, k, HD, NH, Tq);
            ggml_tensor * v3 = ggml_reshape_3d(c, v, HD, NH, Tq);
            q3 = qkn(q3, L.qn_w, L.qn_b);
            k3 = qkn(k3, L.kn_w, L.kn_b);
            ggml_tensor * qf = ggml_cont(c, ggml_permute(c, q3, 0, 2, 1, 3));
            ggml_tensor * kf = ggml_cont(c, ggml_permute(c, k3, 0, 2, 1, 3));
            ggml_tensor * vf = ggml_cont(c, ggml_permute(c, v3, 0, 2, 1, 3));
            ggml_tensor * fa = ggml_flash_attn_ext(c, qf, kf, vf, p->d_mask,
                                                   1.0f / std::sqrt((float)HD),
                                                   0.f, 0.f);
            ggml_tensor * attn = ggml_cont(c, fa);
            attn = ggml_reshape_2d(c, attn, dit_d, Tq);
            attn = ggml_add(c, ggml_mul_mat(c, L.wo, attn), L.bo);
            x = ggml_add(c, x, ggml_mul(c, attn, ga_msa));
            // causal conv branch
            h = modulate(x, sh_cnv, sc_cnv);
            ggml_tensor * hp = to_planar(c, h);
            hp = pad_left(c, hp, 2);
            hp = conv1(c, L.conv1_w, L.conv1_b, hp, 1, 0, 1);
            hp = to_dfast(c, hp);
            hp = ggml_norm(c, hp, dit_eps);
            hp = ggml_mul(c, hp, L.conv_ln_w);
            hp = ggml_add(c, hp, L.conv_ln_b);
            {   // Mish: x * tanh(softplus(x))
                ggml_tensor * sp = ggml_softplus(c, hp);
                hp = ggml_mul(c, hp, ggml_tanh(c, sp));
            }
            hp = pad_left(c, to_planar(c, hp), 2);
            hp = conv1(c, L.conv2_w, L.conv2_b, hp, 1, 0, 1);
            hp = to_dfast(c, hp);
            x = ggml_add(c, x, ggml_mul(c, hp, ga_cnv));
            // mlp (gelu tanh)
            h = modulate(x, sh_mlp, sc_mlp);
            ggml_tensor * ff = ggml_mul_mat(c, L.mlp_gate_w, h);
            ff = ggml_add(c, ff, L.mlp_gate_b);
            {
                ggml_tensor * g3 = ggml_mul(c, ggml_mul(c, ff, ff), ff);
                g3 = ggml_scale(c, g3, 0.044715f);
                g3 = ggml_add(c, g3, ff);
                g3 = ggml_scale(c, g3, 0.7978845608028654f);
                g3 = ggml_tanh(c, g3);
                ff = ggml_add(c, ggml_scale(c, ff, 0.5f),
                              ggml_scale(c, ggml_mul(c, g3, ff), 0.5f));
            }
            ff = ggml_mul_mat(c, L.mlp_down_w, ff);
            ff = ggml_add(c, ff, L.mlp_down_b);
            x = ggml_add(c, x, ggml_mul(c, ff, ga_mlp));
        }
        {   // final layer: shift, scale = chunk(2) - shift FIRST
            ggml_tensor * mod = ggml_mul_mat(c, fin_adaln_w, ggml_silu(c, p->d_t));
            mod = ggml_add(c, mod, fin_adaln_b);
            const int64_t rb = mod->nb[1];
            ggml_tensor * sh = ggml_view_2d(c, mod, dit_d, 1, rb, 0);
            ggml_tensor * sc = ggml_view_2d(c, mod, dit_d, 1, rb, (size_t)dit_d * 4);
            ggml_tensor * n = ggml_norm(c, x, dit_eps);
            n = ggml_add(c, ggml_add(c, n, ggml_mul(c, n, sc)), sh);
            x = ggml_mul_mat(c, fin_out_w, n);
            x = ggml_add(c, x, fin_out_b);
        }
        ggml_set_name(x, "d_out");
        p->d_out = x;
        ggml_build_forward_expand(gf, x);
        if (!ggml_gallocr_alloc_graph(p->allocr_dit, gf)) {
            fprintf(stderr, "vocoder: dit alloc failed\n");
            return false;
        }
        {   // all-visible mask once
            std::vector<ggml_fp16_t> m0((size_t)T2 * T2, ggml_fp32_to_fp16(0.f));
            ggml_backend_tensor_set(p->d_mask, m0.data(), 0, m0.size() * 2);
        }
        p->dit_gf = gf;
        p->dit_T = T2;
    }
    // euler loop
    {
        std::vector<float> x((size_t)n_mel * T2);
        {   // z = rand_noise[:, :, :T2]
            std::vector<float> buf((size_t)n_mel * n_rand);
            ggml_backend_tensor_get(rand_noise, buf.data(), 0, buf.size() * sizeof(float));
            for (int m = 0; m < n_mel; m++)
                for (int t = 0; t < T2; t++)
                    x[(size_t)m * T2 + t] = buf[(size_t)m * n_rand + t];
        }
        std::vector<float> dphi((size_t)n_mel * T2), dout((size_t)n_mel * T2);
        std::vector<float> zeros3((size_t)n_mel * T2, 0.f);
        auto eval_at = [&](const std::vector<float> & tracks_mu,
                           const std::vector<float> & tracks_spk,
                           const std::vector<float> & tracks_cond,
                           float tval, std::vector<float> & out) {
            // host time embedding: sinus(256) -> linear -> silu -> linear
            static std::vector<float> w0, b0, w1, b1;
            if (w0.empty()) {
                host_f32(t_mlp0_w, w0); host_f32(t_mlp0_b, b0);
                host_f32(t_mlp1_w, w1); host_f32(t_mlp1_b, b1);
            }
            float te[256];
            for (int i = 0; i < 128; i++) {
                float freq = std::exp(-std::log(10000.0f) * (float)i / 128.f);
                float arg = tval * 1000.f * freq;
                te[i] = std::cos(arg);
                te[128 + i] = std::sin(arg);
            }
            float h0[512];
            for (int o = 0; o < 512; o++) {
                float acc = b0[(size_t)o];
                const float * wr = &w0[(size_t)o * 256];
                for (int i = 0; i < 256; i++) acc += wr[i] * te[i];
                h0[o] = acc / (1.f + std::exp(-acc));
            }
            std::vector<float> te_in((size_t)dit_d);
            for (int o = 0; o < dit_d; o++) {
                float acc = b1[(size_t)o];
                const float * wr = &w1[(size_t)o * 512];
                for (int i = 0; i < 512; i++) acc += wr[i] * h0[i];
                te_in[(size_t)o] = acc;
            }
            ggml_backend_tensor_set(p->d_t, te_in.data(), 0, te_in.size() * 4);
            ggml_backend_tensor_set(p->d_x, x.data(), 0, x.size() * 4);
            ggml_backend_tensor_set(p->d_mu, tracks_mu.data(), 0, tracks_mu.size() * 4);
            ggml_backend_tensor_set(p->d_spk, tracks_spk.data(), 0, tracks_spk.size() * 4);
            ggml_backend_tensor_set(p->d_cond, tracks_cond.data(), 0, tracks_cond.size() * 4);
            if (const char * ed = getenv("MINICPMO_VOC_EST_DUMP")) {
                static int est_n = 0;
                if (est_n < 1) {   // first call only
                    char fp[512];
                    FILE * f;
                    f = fopen((std::string(ed) + "_x.f32").c_str(), "wb");
                    if (f) { fwrite(x.data(), 4, x.size(), f); fclose(f); }
                    f = fopen((std::string(ed) + "_mu.f32").c_str(), "wb");
                    if (f) { fwrite(tracks_mu.data(), 4, tracks_mu.size(), f); fclose(f); }
                    f = fopen((std::string(ed) + "_spk.f32").c_str(), "wb");
                    if (f) { fwrite(tracks_spk.data(), 4, tracks_spk.size(), f); fclose(f); }
                    f = fopen((std::string(ed) + "_cond.f32").c_str(), "wb");
                    if (f) { fwrite(tracks_cond.data(), 4, tracks_cond.size(), f); fclose(f); }
                    f = fopen((std::string(ed) + "_t.f32").c_str(), "wb");
                    if (f) { fwrite(te_in.data(), 4, te_in.size(), f); fclose(f); }
                    est_n++;
                }
            }
            if (ggml_backend_graph_compute(B, p->dit_gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "vocoder: dit compute failed\n");
                return false;
            }
            ggml_backend_tensor_get(p->d_out, out.data(), 0, out.size() * 4);
            return true;
        };
        // t_span = 1 - cos(linspace(0, 1, n+1) * pi/2)
        float t_span[11];
        for (int i = 0; i <= n_timesteps; i++) {
            float tv = (float)i / n_timesteps;
            t_span[i] = 1.f - std::cos(tv * 0.5f * (float)M_PI);
        }
        float t = t_span[0], dt = t_span[1] - t_span[0];
        const char * edump = getenv("MINICPMO_VOC_EULER_DUMP");
        for (int step = 1; step <= n_timesteps; step++) {
            if (!eval_at(mu, spk80, conds, t, dphi)) return false;
            if (!eval_at(zeros3, zeros3, zeros3, t, dout)) return false;
            if (edump) {                    // parity: per-step dphi cond/uncond
                char fp[512];
                snprintf(fp, sizeof(fp), "%s_dphi_c%d.f32", edump, step);
                FILE * f = fopen(fp, "wb");
                if (f) { fwrite(dphi.data(), 4, dphi.size(), f); fclose(f); }
                snprintf(fp, sizeof(fp), "%s_dphi_u%d.f32", edump, step);
                f = fopen(fp, "wb");
                if (f) { fwrite(dout.data(), 4, dout.size(), f); fclose(f); }
                snprintf(fp, sizeof(fp), "%s_x%d.f32", edump, step);
                f = fopen(fp, "wb");
                if (f) { fwrite(x.data(), 4, x.size(), f); fclose(f); }
            }
            for (size_t i = 0; i < x.size(); i++) {
                float d = (1.f + cfg_rate) * dphi[i] - cfg_rate * dout[i];
                x[i] += dt * d;
            }
            t += dt;
            if (step < n_timesteps) dt = t_span[step + 1] - t;
        }
        // generated mel = x[:, T2p:]
        p->n_gen_mel = Tmel;
        p->mel.assign((size_t)n_mel * Tmel, 0.f);
        for (int m = 0; m < n_mel; m++)
            for (int t2 = 0; t2 < Tmel; t2++)
                p->mel[(size_t)m * Tmel + t2] = x[(size_t)m * T2 + T2p + t2];
        if (const char * dp = getenv("MINICPMO_VOC_DUMP")) {
            std::string fp = std::string(dp) + "_mel.f32";
            FILE * f = fopen(fp.c_str(), "wb");
            if (f) { fwrite(p->mel.data(), 4, p->mel.size(), f); fclose(f); }
        }
    }
fprintf(stderr, "vocoder stage: %s\n", "cfm");
    if (const char * ml = getenv("MINICPMO_VOC_MEL_LOAD")) {   // vocoder-only gate
        FILE * f = fopen(ml, "rb");
        if (!f) { fprintf(stderr, "vocoder: cannot open %s\n", ml); return false; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        p->mel.resize(sz / 4);
        bool ok = fread(p->mel.data(), 4, p->mel.size(), f) == p->mel.size();
        fclose(f);
        if (!ok) return false;
        p->n_gen_mel = (int)(sz / 4 / n_mel);
        fprintf(stderr, "vocoder: mel override %d frames\n", p->n_gen_mel);
    }

    
    const int Tm = p->n_gen_mel;
    {
        if (p->f0_ctx) { ggml_free(p->f0_ctx); p->f0_ctx = nullptr; }
        struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, true };
        p->f0_ctx = ggml_init(ip);
        ggml_context * c = p->f0_ctx;
        ggml_cgraph * gf = ggml_new_graph_custom(c, 8192, false);
        ggml_tensor * in = ggml_new_tensor_2d(c, GGML_TYPE_F32, Tm, n_mel);
        ggml_set_input(in); ggml_set_name(in, "f0_in");
        p->f0_in = in;
        ggml_tensor * h = in;
        for (int i = 0; i < 5; i++) {
            h = conv1(c, f0_c_w[i], f0_c_b[i], h, 1, 1, 1);
            h = ggml_elu(c, h);   // NOTE: the module has a TRAILING ELU (idx 9)
            if (getenv("MINICPMO_VOC_CP_DBG")) {   // dup sinks: reuse-safe
                char nm[32];
                snprintf(nm, sizeof(nm), "f0_s%d", i);
                ggml_tensor * s = ggml_dup(c, h);
                ggml_set_name(s, nm);
                ggml_build_forward_expand(gf, s);
            }
        }
        h = ggml_mul_mat(c, f0_head_w, to_dfast(c, h));            // [1, Tm]
        h = ggml_add(c, h, f0_head_b);
        h = ggml_abs(c, h);
        ggml_set_name(h, "f0_out");
        p->f0_out = h;
        ggml_build_forward_expand(gf, h);
        if (!ggml_gallocr_alloc_graph(p->allocr_f0, gf)) {
            fprintf(stderr, "vocoder: f0 alloc failed\n");
            return false;
        }
        {   // upload mel: tensor element (t, m) at t + m*Tm -> the mel planes
            // [m][T] copy over verbatim (mp index = m*Tm + t)
            std::vector<float> mp((size_t)n_mel * Tm);
            for (int m = 0; m < n_mel; m++)
                for (int t = 0; t < Tm; t++)
                    mp[(size_t)m * Tm + t] = p->mel[(size_t)m * Tm + t];
            if (getenv("MINICPMO_VOC_DELTA")) {   // impulse response probe
                std::fill(mp.begin(), mp.end(), 0.f);
                mp[1] = 1.f;                      // delta at t=1, bin 0
            }
            ggml_backend_tensor_set(in, mp.data(), 0, mp.size() * 4);
        }
        fprintf(stderr, "vocoder: f0 graph nodes %d\n",
                ggml_graph_n_nodes(gf));
        if (getenv("MINICPMO_VOC_GRAPH_PRINT")) ggml_graph_print(gf);
        if (ggml_backend_graph_compute(B, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "vocoder: f0 compute failed\n");
            return false;
        }
        if (getenv("MINICPMO_VOC_CP_DBG")) {
            for (int ni = 0; ni < ggml_graph_n_nodes(gf); ni++) {
                ggml_tensor * t = ggml_graph_node(gf, ni);
                const char * nn = ggml_get_name(t);
                if (!t || (strncmp(nn, "f0_s", 4) != 0
                           && strncmp(nn, "d_", 2) != 0))
                    continue;
                std::vector<float> v((size_t)ggml_nelements(t));
                if (getenv("MINICPMO_VOC_DEBUG"))
                    fprintf(stderr, "vocoder dump %s ne %lldx%lldx%lld type %d nb %lld\n",
                            nn, (long long)t->ne[0], (long long)t->ne[1],
                            (long long)t->ne[2], (int)t->type,
                            (long long)ggml_nbytes(t));
                ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
                char fp[512];
                snprintf(fp, sizeof(fp), "%s_%s.f32",
                         getenv("MINICPMO_VOC_CP_DBG"), nn);
                FILE * f2 = fopen(fp, "wb");
                if (f2) { fwrite(v.data(), 4, v.size(), f2); fclose(f2); }
            }
        }
        p->f0.resize(Tm);
        ggml_backend_tensor_get(h, p->f0.data(), 0, (size_t)Tm * 4);
        p->f0_gf = gf;
    }

fprintf(stderr, "vocoder stage: %s\n", "f0");
    // ==================== NSF sine source (host DSP) =======================
    const int ups_factor = 120 * ISTFT_HOP;                        // 480
    const int Ts = Tm * ups_factor;
    std::vector<float> s(Ts);
    {
        // f0 nearest x480
        std::vector<float> f0u(Ts);
        for (int t = 0; t < Ts; t++) f0u[t] = p->f0[t / ups_factor];
        // rand_ini [9]
        float rand_ini[N_HARM];
        if (p->gate_rng) {
            // official SineGen2._f02sine draws rand_ini then forces
            // rand_ini[:, 0] = 0 (no initial phase noise on the fundamental);
            // the captured file holds the RAW torch.rand draw
            for (int h = 0; h < N_HARM; h++)
                rand_ini[h] = (h == 0) ? 0.f : p->gate_rand_ini[(size_t)h];
        } else {
            for (int h = 0; h < N_HARM; h++)
                rand_ini[h] = (h == 0) ? 0.f : p->rng.uniform();
        }
        // rad[t][h] = fmod(f0u[t]*(h+1)/sr, 1); first time frame += rand_ini
        static std::vector<float> rad;
        rad.assign((size_t)Ts * N_HARM, 0.f);
        for (int h = 0; h < N_HARM; h++)
            for (int t = 0; t < Ts; t++)
                rad[(size_t)t * N_HARM + h] =
                    std::fmod(f0u[t] * (h + 1) / 24000.f, 1.f);
        for (int h = 0; h < N_HARM; h++) rad[(size_t)h] += rand_ini[h];
        // linear interpolate 1/480 -> [204, 9]; cumsum; *480; interpolate x480
        auto lin_interp = [](const std::vector<float> & in, int nin,
                             std::vector<float> & out, int nout) {
            out.resize((size_t)nout);
            const double scale = (double)nin / nout;
            for (int i = 0; i < nout; i++) {
                double src = (i + 0.5) * scale - 0.5;
                if (src < 0) src = 0;
                if (src > nin - 1) src = nin - 1;
                const int i0 = (int)src;
                const int i1 = std::min(i0 + 1, nin - 1);
                const double fr = src - i0;
                out[(size_t)i] = (float)((1 - fr) * in[(size_t)i0] +
                                         fr * in[(size_t)i1]);
            }
        };
        std::vector<float> radd((size_t)Tm * N_HARM), ph((size_t)Tm * N_HARM),
            phu((size_t)Ts * N_HARM);
        for (int h = 0; h < N_HARM; h++) {
            std::vector<float> col(Ts), cold, phc, phcu;
            for (int t = 0; t < Ts; t++) col[(size_t)t] = rad[(size_t)t * N_HARM + h];
            lin_interp(col, Ts, cold, Tm);
            double acc = 0;
            for (int t = 0; t < Tm; t++) {
                acc += (double)cold[(size_t)t];
                phc.push_back((float)(acc * 2.0 * M_PI * ups_factor));
            }
            lin_interp(phc, Tm, phcu, Ts);
            for (int t = 0; t < Ts; t++)
                phu[(size_t)t * N_HARM + h] = phcu[(size_t)t];
        }
        // sine_wavs = sin(phu)*amp*uv + noise_amp*noise; merge = tanh(linear)
        std::vector<float> w9, b9;
        host_f32(src_lin_w, w9);
        host_f32(src_lin_b, b9);
        for (int t = 0; t < Ts; t++) {
            float merge = b9[0];
            for (int h = 0; h < N_HARM; h++) {
                const float uv = (f0u[t] > VOICED_TH) ? 1.f : 0.f;
                const float noise_amp = uv * NOISE_STD + (1.f - uv) * SINE_AMP / 3.f;
                const float noise = p->gate_rng
                    ? p->gate_sine_noise[(size_t)t * N_HARM + h]
                    : p->rng.normal();
                const float wv = std::sin(phu[(size_t)t * N_HARM + h]) * SINE_AMP * uv
                               + noise_amp * noise;
                merge += w9[(size_t)h] * wv;
            }
            s[(size_t)t] = std::tanh(merge);
        }
        if (const char * dp = getenv("MINICPMO_VOC_DUMP")) {
            std::string fp = std::string(dp) + "_s.f32";
            FILE * f = fopen(fp.c_str(), "wb");
            if (f) { fwrite(s.data(), 4, s.size(), f); fclose(f); }
        }
    }

fprintf(stderr, "vocoder stage: %s\n", "sine");
    // ==================== STFT(s) hann16 hop4 (host) =======================
    const int n_stft_frames = Ts / ISTFT_HOP + 1;                  // center=True
    std::vector<float> sstft((size_t)n_stft_frames * 18);
    {
        std::vector<float> sp((size_t)Ts + 2 * (ISTFT_N / 2));
        for (int i = 0; i < ISTFT_N / 2; i++)                       // reflect pad
            sp[(size_t)i] = s[(size_t)(ISTFT_N / 2 - i)];
        memcpy(sp.data() + ISTFT_N / 2, s.data(), Ts * 4);
        for (int i = 0; i < ISTFT_N / 2; i++)
            sp[(size_t)(ISTFT_N / 2 + Ts + i)] = s[(size_t)(Ts - 2 - i)];
        for (int f = 0; f < n_stft_frames; f++) {
            float re[ISTFT_N / 2 + 1] = {0}, im[ISTFT_N / 2 + 1] = {0};
            for (int kk = 0; kk < ISTFT_N; kk++) {
                float v = sp[(size_t)(f * ISTFT_HOP + kk)] * p->hann16[kk];
                for (int bin = 0; bin <= ISTFT_N / 2; bin++) {
                    float ang = 2.f * (float)M_PI * bin * kk / ISTFT_N;
                    re[bin] += v * std::cos(ang);
                    im[bin] -= v * std::sin(ang);
                }
            }
            for (int bin = 0; bin <= ISTFT_N / 2; bin++) {
                sstft[(size_t)f * 18 + bin] = re[bin];
                sstft[(size_t)f * 18 + 9 + bin] = im[bin];
            }
        }
    }

    // ============================ hift decode ==============================
    const int sd_str[3] = {15, 3, 1};
    {
        if (p->dec_ctx) { ggml_free(p->dec_ctx); p->dec_ctx = nullptr; }
        struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, true };
        p->dec_ctx = ggml_init(ip);
        ggml_context * c = p->dec_ctx;
        ggml_cgraph * gf = ggml_new_graph_custom(c, 32768, false);
        gf_dbg = gf;    // debug: per-tap sinks attach to this graph
        ntap = 0;       // debug: restart per-tap conv numbering at the decode
        ggml_tensor * mel_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, Tm, n_mel);
        ggml_set_input(mel_in); ggml_set_name(mel_in, "dec_mel");
        ggml_tensor * sstft_in = ggml_new_tensor_2d(c, GGML_TYPE_F32,
                                                    n_stft_frames, 18);
        ggml_set_input(sstft_in); ggml_set_name(sstft_in, "dec_sstft");
        p->dec_mel = mel_in; p->dec_sstft = sstft_in;
        ggml_tensor * x = conv1(c, conv_pre_w, conv_pre_b, mel_in, 1, 3, 1);
        const bool dbg = getenv("MINICPMO_VOC_CP_DBG") != nullptr;
        std::vector<std::pair<ggml_tensor *, const char *>> dec_sinks;
        auto dec_sink = [&](ggml_tensor * t, const char * nm) {
            if (!dbg) return;
            ggml_tensor * s2 = ggml_dup(c, t);
            ggml_set_name(s2, nm);
            ggml_set_output(s2);   // keep gallocr from recycling the dump
            ggml_build_forward_expand(gf, s2);
            dec_sinks.push_back({s2, nm});
        };
        dec_sink(mel_in, "d_melin");
        dec_sink(sstft_in, "d_sstft");
        dec_sink(x, "d_convpre");
        // source stft spec [T_stft, 18] reused by all three fusion branches
        auto resblock = [&](ggml_tensor * xin, const ResBlock & R) {
            for (int j = 0; j < 3; j++) {
                ggml_tensor * xt = snake(c, xin, ggml_reshape_2d(c, R.a1[j], 1, xin->ne[1]));
                // SAFE-PASS: a set_output dup is never recycled by gallocr;
                // convs read the dup copy (original buffer gets trashed).
                ggml_tensor * xt_safe = ggml_dup(c, xt);
                ggml_set_output(xt_safe);
                ggml_build_forward_expand(gf, xt_safe);
                xt = xt_safe;
                if (getenv("MINICPMO_VOC_SINK_J") && R.a1[0] == srb[0].a1[0]) {
                    char nm[32]; snprintf(nm, sizeof(nm), "d_srb0_j%din", j);
                    ggml_tensor * s = ggml_dup(c, xt);
                    ggml_set_name(s, nm);
                    ggml_set_output(s);
                    ggml_build_forward_expand(gf, s);
                    if (j == 0) {
                        ggml_tensor * sp2 = ggml_dup(c, conv1(c, R.c2[0].w,
                                R.c2[0].b, xt, 1, 3, 1));
                        ggml_set_name(sp2, "d_srb0_c20sp");
                        ggml_set_output(sp2);
                        ggml_build_forward_expand(gf, sp2);
                    }
                }
                xt = conv1(c, R.c1[j].w, R.c1[j].b, xt, 1, (R.k[j] * R.dil[j] - R.dil[j]) / 2, R.dil[j]);
                if (getenv("MINICPMO_VOC_SINK_J") && R.a1[0] == srb[0].a1[0]) {
                    char nm[32]; snprintf(nm, sizeof(nm), "d_srb0_j%dout", j);
                    ggml_tensor * s = ggml_dup(c, xt);
                    ggml_set_name(s, nm);
                    ggml_set_output(s);
                    ggml_build_forward_expand(gf, s);
                }
                xt = snake(c, xt, ggml_reshape_2d(c, R.a2[j], 1, xt->ne[1]));
                {
                    ggml_tensor * xt_safe = ggml_dup(c, xt);
                    ggml_set_output(xt_safe);
                    ggml_build_forward_expand(gf, xt_safe);
                    xt = xt_safe;
                }
                if (getenv("MINICPMO_VOC_SINK_J") && R.a1[0] == srb[0].a1[0]
                        && j == 2) {   // debug: srb0 j2 boundary dump
                    ggml_tensor * s = ggml_dup(c, xt);
                    ggml_set_name(s, "d_srb0_j2in");
                    ggml_set_output(s);
                    ggml_build_forward_expand(gf, s);
                }
                if (getenv("MINICPMO_VOC_SINK_J") && R.a1[0] == srb[0].a1[0]
                        && j == 0) {   // c2[j] input boundary (== torch a20)
                    ggml_tensor * s = ggml_dup(c, xt);
                    ggml_set_name(s, "d_srb0_a20");
                    ggml_set_output(s);
                    ggml_build_forward_expand(gf, s);
                }
                xt = conv1(c, R.c2[j].w, R.c2[j].b, xt, 1, (R.k[j] - 1) / 2, 1);
                if (getenv("MINICPMO_VOC_SINK_J") && R.a1[0] == srb[0].a1[0]
                        && j == 0) {
                    ggml_tensor * s = ggml_dup(c, xt);
                    ggml_set_name(s, "d_srb0_c20");
                    ggml_set_output(s);
                    ggml_build_forward_expand(gf, s);
                    ggml_tensor * s2 = ggml_dup(c, conv1(c, R.c1[0].w,
                            R.c1[0].b, xt, 1, 3, 1));
                    ggml_set_name(s2, "d_srb0_c20x");
                    ggml_set_output(s2);
                    ggml_build_forward_expand(gf, s2);
                }
                xin = ggml_add(c, xt, xin);
            }
            return xin;
        };
        for (int i = 0; i < 3; i++) {
            x = ggml_leaky_relu(c, x, LRELU1, false);
            fprintf(stderr, "vocoder dec: stage %d pre-convT x %lldx%lld\n", i,
                    (long long)x->ne[0], (long long)x->ne[1]);
            {   // conv transpose stride s, pad p via p0=0 call + view crop
                ggml_tensor * y = ggml_conv_transpose_1d(c, ups[i].w, x,
                                                         ups[i].s, 0, 1);
                const int64_t Lfull = y->ne[0];
                y = ggml_view_2d(c, y, Lfull - 2 * ups[i].p, y->ne[1],
                                 y->nb[1], (size_t)ups[i].p * 4);
                y = ggml_cont(c, y);            // convs want contiguous rows
                y = ggml_add(c, y, ggml_reshape_2d(c, ups[i].b, 1,
                                                   ups[i].b->ne[0]));
                x = y;
            }
            fprintf(stderr, "vocoder dec: stage %d post-convT x %lldx%lld\n", i,
                    (long long)x->ne[0], (long long)x->ne[1]);
            if (dbg && i == 0) dec_sink(x, "d_t0");
            if (i == 2) {
                if (dbg) fprintf(stderr, "vocoder dec: pad_reflect in\n");
                x = ggml_pad_reflect_1d(c, x, 1, 0);
                if (dbg) fprintf(stderr, "vocoder dec: pad_reflect out %lldx%lld\n",
                        (long long)x->ne[0], (long long)x->ne[1]);
            }
            // source fusion: Conv1d(18, ch, u*2, stride u, pad u//2)
            const int u = sd_str[i];
            ggml_tensor * si = (u > 1)
                ? conv1(c, sdown[i].w, sdown[i].b, sstft_in, u, u / 2, 1)
                : conv1(c, sdown[i].w, sdown[i].b, sstft_in, 1, 0, 1);
            if (dbg && i == 0) dec_sink(g_conv1_y, "d_yraw0");
            if (dbg && i == 0) dec_sink(si, "d_sdown0");
            si = resblock(si, srb[i]);
            if (dbg && i == 0) dec_sink(si, "d_si0");
            if (dbg) fprintf(stderr, "vocoder dec: stage %d si %lldx%lld x %lldx%lld\n",
                    i, (long long)si->ne[0], (long long)si->ne[1],
                    (long long)x->ne[0], (long long)x->ne[1]);
            x = ggml_add(c, x, si);
            // mean of the 3 resblocks (each is the FULL 3-j ResBlock:
            // snake -> conv d1 -> snake -> conv d3 -> snake -> conv d5 ->
            // snake -> conv d1, per official ResBlock.forward)
            ggml_tensor * acc = resblock(x, rb[i * 3 + 0]);
            for (int j = 1; j < 3; j++)
                acc = ggml_add(c, acc, resblock(x, rb[i * 3 + j]));
            x = ggml_scale(c, acc, 1.0f / 3.f);
            if (dbg) fprintf(stderr, "vocoder dec: stage %d resblocks done\n", i);
            if (dbg && i == 0) dec_sink(x, "d_up0pre");
            {
                char nm[32];
                snprintf(nm, sizeof(nm), "d_up%d", i);
                dec_sink(x, nm);
            }
        }
        x = ggml_leaky_relu(c, x, LRELU01, false);
        x = conv1(c, conv_post_w, conv_post_b, x, 1, 3, 1);        // [T_out, 18]
        {   // pin the conv_post output: the strided mag/ph views and the
            // debug dump all read this tensor; a set_output dup is never
            // recycled by gallocr (free_node skips OUTPUT tensors)
            ggml_tensor * xs = ggml_dup(c, x);
            ggml_set_output(xs);
            ggml_build_forward_expand(gf, xs);
            x = xs;
        }
        dec_sink(x, "d_post");
        // mag/phase views are strided (row stride 18) - make them contiguous
        // so the host ISTFT read matches (t, bin) dense layout. Official
        // decode: mag = clip(exp(x9), max=1e2); phase = sin(x9) (hifigan.py
        // decode(): "phase = torch.sin(x[:, 9:, :])").
        ggml_tensor * magc = ggml_cont(c, ggml_view_2d(c, x, x->ne[0], 9,
                                                       x->nb[1], 0));
        ggml_tensor * mag = magc;
        // x is planar [T, C=18]: the channel axis is the SLOW dim (stride
        // T*4 bytes), so channels 9..18 start at byte offset 9*T*4 - the
        // byte offset of channel 9, NOT 9*4 (which is "channel 0, t=9")
        ggml_tensor * ph = ggml_cont(c, ggml_view_2d(c, x, x->ne[0], 9,
                                                     x->nb[1],
                                                     (size_t)9 * x->ne[0] * 4));
        mag = ggml_clamp(c, ggml_exp(c, mag),
                         -std::numeric_limits<float>::infinity(), 1e2f);
        ph = ggml_sin(c, ph);
        ggml_set_name(mag, "dec_mag");
        ggml_set_name(ph, "dec_ph");
        p->dec_mag = mag; p->dec_ph = ph;
        ggml_build_forward_expand(gf, mag);
        ggml_build_forward_expand(gf, ph);
        if (getenv("MINICPMO_VOC_NO_GALLOCR")) {
            // WORKAROUND: this ggml's gallocr misassigns addresses for
            // graphs with chained views/dups (layout-dependent corruption).
            // Allocate every decode tensor in its own context buffer instead.
            if (p->dec_buf) ggml_backend_buffer_free(p->dec_buf);
            p->dec_buf = ggml_backend_alloc_ctx_tensors(p->dec_ctx, B);
            if (!p->dec_buf) { fprintf(stderr, "vocoder: decode alloc failed\n"); return false; }
        } else if (!ggml_gallocr_alloc_graph(p->allocr_dec, gf)) {
            fprintf(stderr, "vocoder: decode alloc failed\n");
            return false;
        }
        {   // upload mel as [W=T, H=80] planes (element (t, m) at t + m*Tm);
            // p->mel is m-major [m * Tm + t]
            std::vector<float> mp((size_t)n_mel * Tm);
            for (int m = 0; m < n_mel; m++)
                for (int t = 0; t < Tm; t++)
                    mp[(size_t)m * Tm + t] = p->mel[(size_t)m * Tm + t];
            ggml_backend_tensor_set(mel_in, mp.data(), 0,
                                    (size_t)n_mel * Tm * 4);
        }
        {   // (f, bin) f-major -> (bin, f) bin-major for ne [18, frames]
            std::vector<float> sb(sstft.size());
            if (const char * sl = getenv("MINICPMO_VOC_STFT_LOAD")) {
                // parity: official torch.stft spec, already bin-major
                FILE * f = fopen(sl, "rb");
                if (!f) { fprintf(stderr, "vocoder: cannot open %s\n", sl); return false; }
                fread(sb.data(), 4, sb.size(), f);
                fclose(f);
            } else {
                for (size_t f = 0; f < (size_t)n_stft_frames; f++)
                    for (int bin = 0; bin < 18; bin++)
                        sb[(size_t)bin * n_stft_frames + f] =
                            sstft[f * 18 + bin];
            }
            ggml_backend_tensor_set(sstft_in, sb.data(), 0, sb.size() * 4);
        }
        if (ggml_backend_graph_compute(B, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "vocoder: decode compute failed\n");
            return false;
        }
        const int T_out = (int)mag->ne[0];
        std::vector<float> magv((size_t)T_out * 9), phv((size_t)T_out * 9);
        ggml_backend_tensor_get(mag, magv.data(), 0, magv.size() * 4);
        ggml_backend_tensor_get(ph, phv.data(), 0, phv.size() * 4);
        p->dec_Tmel = Tm;
        fprintf(stderr, "vocoder dec: CP_DBG=%s sinks=%d nodes=%d\n",
                getenv("MINICPMO_VOC_CP_DBG") ? getenv("MINICPMO_VOC_CP_DBG") : "(unset)",
                (int)dec_sinks.size(), ggml_graph_n_nodes(gf));
        if (getenv("MINICPMO_VOC_CP_DBG")) {   // decode stage sinks
            int nprinted = 0;
            for (int ni = 0; ni < ggml_graph_n_nodes(gf); ni++) {
                ggml_tensor * t = ggml_graph_node(gf, ni);
                const char * nn = ggml_get_name(t);
                if (nprinted++ < 12)
                    fprintf(stderr, "vocoder dec node[%d] '%s'\n", ni, nn);
                if (!t || strncmp(nn, "d_", 2) != 0)
                    continue;
                fprintf(stderr, "vocoder dec: DUMPING %s\n", nn);
                std::vector<float> v((size_t)ggml_nelements(t));
                ggml_backend_tensor_get(t, v.data(), 0, v.size() * 4);
                char fp[512];
                snprintf(fp, sizeof(fp), "%s_%s.f32",
                         getenv("MINICPMO_VOC_CP_DBG"), nn);
                FILE * f2 = fopen(fp, "wb");
                if (f2) { fwrite(v.data(), 4, v.size(), f2); fclose(f2); }
            }
        }

        if (getenv("MINICPMO_VOC_CP_DBG")) {
            std::string mfp = std::string(getenv("MINICPMO_VOC_CP_DBG"))
                              + "_magv.f32";
            FILE * f3 = fopen(mfp.c_str(), "wb");
            fwrite(magv.data(), 4, magv.size(), f3); fclose(f3);
            f3 = fopen((std::string(getenv("MINICPMO_VOC_CP_DBG"))
                        + "_phv.f32").c_str(), "wb");
            fwrite(phv.data(), 4, phv.size(), f3); fclose(f3);
        }

        // ---- ISTFT (host): irfft + weighted overlap-add + window norm ----
        const int raw_len = (n_stft_frames - 1) * ISTFT_HOP + ISTFT_N;
        std::vector<float> acc((size_t)raw_len, 0.f), norm((size_t)raw_len, 0.f);
        const float inv_n = 1.f / ISTFT_N;
        for (int f = 0; f < n_stft_frames; f++) {
            for (int kk = 0; kk < ISTFT_N; kk++) {
                const size_t idx = (size_t)(f * ISTFT_HOP + kk);
                // irfft: Re(X0) + 2*sum_{b=1..7} Re(Xb e^{i ang}) + Re(X8)(-1)^k.
                // DC and Nyquist carry a network phase too - their cos factor
                // must be applied like every other bin (torch.istft does).
                float sum = magv[(size_t)0 * T_out + f]
                          * std::cos(phv[(size_t)0 * T_out + f]);
                for (int b = 1; b < ISTFT_N / 2; b++) {
                    const float m = magv[(size_t)b * T_out + f];
                    const float pp = phv[(size_t)b * T_out + f];
                    const float ang = 2.f * (float)M_PI * b * kk / ISTFT_N;
                    sum += 2.f * (m * std::cos(pp) * std::cos(ang)
                                  - m * std::sin(pp) * std::sin(ang));
                }
                sum += magv[(size_t)(ISTFT_N / 2) * T_out + f]
                     * std::cos(phv[(size_t)(ISTFT_N / 2) * T_out + f])
                     * ((kk % 2) ? -1.f : 1.f);
                acc[idx] += sum * inv_n * p->hann16[kk];
                norm[idx] += p->hann16[kk] * p->hann16[kk];
            }
        }
        // center=True: trim n_fft/2 from both ends of the raw timeline
        pcm.resize((size_t)Ts);
        for (int t = 0; t < Ts; t++) {
            const float v = acc[(size_t)(ISTFT_N / 2 + t)]
                          / std::max(norm[(size_t)(ISTFT_N / 2 + t)], 1e-12f);
            pcm[(size_t)t] = std::clamp(v, -0.99f, 0.99f);
        }
        if (const char * dp = getenv("MINICPMO_VOC_DUMP")) {
            char tn[32];
            for (int c2 = 1; c2 <= 8; c2++)
                for (int kk = 0; kk < 7; kk++) {
                    snprintf(tn, sizeof(tn), "tap_z%d_%d", c2, kk);
                    if (ggml_tensor * td = ggml_graph_get_tensor(gf, tn)) {
                        std::vector<float> dv((size_t)ggml_nelements(td));
                        ggml_backend_tensor_get(td, dv.data(), 0, dv.size() * 4);
                        std::string dfp = std::string("gate10/") + tn + ".f32";
                        FILE * df = fopen(dfp.c_str(), "wb");
                        if (df) { fwrite(dv.data(), 4, dv.size(), df); fclose(df); }
                    }
                }
            if (ggml_tensor * dd = ggml_graph_get_tensor(gf, "d_sdown0")) {
                std::vector<float> dv((size_t)ggml_nelements(dd));
                ggml_backend_tensor_get(dd, dv.data(), 0, dv.size() * 4);
                std::string dfp = std::string(dp) + "_sdown0.f32";
                FILE * df = fopen(dfp.c_str(), "wb");
                if (df) { fwrite(dv.data(), 4, dv.size(), df); fclose(df); }
            }
            std::string fp = std::string(dp) + "_f0.f32";
            FILE * f = fopen(fp.c_str(), "wb");
            if (f) { fwrite(p->f0.data(), 4, p->f0.size(), f); fclose(f); }
            fp = std::string(dp) + "_pcm.f32";
            f = fopen(fp.c_str(), "wb");
            if (f) { fwrite(pcm.data(), 4, pcm.size(), f); fclose(f); }
        }
        p->dec_gf = gf;
    }
    out_spk = p->spk;
    out_mel = p->mel;
    out_f0 = p->f0;
    return true;
}

}  // namespace minicpmo
