// embedgemma2_model.cpp - EmbeddingGemma-2 backend: Gemma-style bidirectional
// text encoder (projection-only PLE) + Gemma4 vision tower (SigLIP-like, axial
// 2D rope, 3x3 position pool) + Gemma4 audio tower (USM-style conformer with
// chunked local attention), fused by soft-token substitution.
//
// Shape of the thing (google/embeddinggemma-2, 740M):
//   text    24 layers, d=512, ff=2048, gelu (tanh) SwiGLU, RMSNorm 1e-6.
//           Layers 5/11/17/23 are full attention (head_dim 512, 1 kv head,
//           rope theta 1e6); the rest sliding (head_dim 256, 2 kv heads, theta
//           1e4, symmetric window radius 512). Attention is bidirectional
//           everywhere, q/k are head-RMS-normed (weight), v is weightlessly
//           RMS-normed, scale 1.0. PLE: inputs_embeds -> per_layer_model_proj
//           [12288, 512] * 1/sqrt(512) -> RMSNorm -> one gated block per layer
//           (gelu gate x ple slice -> proj -> RMSNorm -> +residual), then a
//           per-layer out_scale. Final norm -> projection 512 -> 768.
//   vision  16 layers d=768 ff=3072, patchify = one Linear(3*16*16, 768) on
//           2*(px/255 - 0.5) + learned (x, y) position table [2, 10240, 768].
//           Axial rope theta 100: 16 distinct freqs, layout H-H-W-W over the
//           64-dim head (q/k head-RMS-normed, v weightless, scale 1.0). No
//           tower-level final norm; output = 3x3 position-pool * sqrt(768),
//           then the 768 -> 512 soft-token projection.
//   audio   12 layers d=1024 ff=4096 (macaron FFN x2, residual weight 0.5),
//           mel 128 x 20ms/10ms (periodic Hann, semicausal left pad, HTK
//           filterbank, log(mel + 1e-3)) -> 2x (3x3 stride-2 conv + channel
//           LayerNorm + ReLU) -> Linear -> 1024. Attention: chunk 12, context
//           12 past + 0 future, content-dependent relative bias (13 sinusoid-
//           derived rows), softcap 50, per-dim softplus scale. LightDepthConv
//           k=5 causal GLU between. Every linear clips input/output to learned
//           bounds. Layer output = norm_out (no residual add). Final Linear
//           1024 -> 1536 -> (weightless RMSNorm) 1536 -> 512 soft tokens.
//
// Fusion: the prompt's <|image|>/<|video|>/<|audio|> markers expand into
// <|image> + N soft placeholders + <image|> (etc.); N comes from the tower
// output length. The text encoder runs over the whole sequence (no KV cache);
// the embedding is mean-pooled over all tokens and L2-normalized (MRL: slice
// + re-normalize for truncate_dim < 768). Reference behavior:
// transformers EmbeddingGemma2Processor / Gemma4*Model, verified against it.
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include "vqa.hpp"

#include "attention.hpp"
#include "backend.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "ggml_dump.hpp"
#include "gguf.h"
#include "gguf_util.hpp"
#include "load_util.hpp"
#include "resample.hpp"
#include "weight_load.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace eg2 {

struct TxtLayer {
    ggml_tensor *attn_norm = nullptr, *attn_post_norm = nullptr;
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    ggml_tensor *q_norm = nullptr, *k_norm = nullptr;
    ggml_tensor *ffn_norm = nullptr, *ffn_post_norm = nullptr;
    ggml_tensor *w_gate = nullptr, *w_up = nullptr, *w_down = nullptr;
    ggml_tensor *ple_gate = nullptr, *ple_proj = nullptr, *ple_post_norm = nullptr;
    ggml_tensor *out_scale = nullptr;
};

struct VisLayer {
    ggml_tensor *attn_norm = nullptr, *attn_post_norm = nullptr;
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    ggml_tensor *q_norm = nullptr, *k_norm = nullptr;
    ggml_tensor *ffn_norm = nullptr, *ffn_post_norm = nullptr;
    ggml_tensor *w_gate = nullptr, *w_up = nullptr, *w_down = nullptr;
};

// learned input/output clamp bounds of one ClippableLinear (host copies; the
// GGUF scalars may live in device memory after binding, so the graph is built
// from these plain floats)
struct AudClip {
    bool has_in = false, has_out = false;
    float in_min = 0, in_max = 0, out_min = 0, out_max = 0;
    ggml_tensor * apply(ggml_context * ctx, ggml_tensor * x, bool input_side) const {
        if (input_side) {
            if (has_in) x = ggml_clamp(ctx, x, in_min, in_max);
        } else {
            if (has_out) x = ggml_clamp(ctx, x, out_min, out_max);
        }
        return x;
    }
};

struct AudFF {   // macaron feed-forward block
    ggml_tensor *pre_norm = nullptr, *post_norm = nullptr;
    ggml_tensor *l1 = nullptr, *l2 = nullptr;
    AudClip c1, c2;
};

struct AudLayer {
    AudFF ff1, ff2;
    ggml_tensor *norm_pre_attn = nullptr, *norm_post_attn = nullptr, *norm_out = nullptr;
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *post = nullptr;
    AudClip cq, ck, cv, cp;
    ggml_tensor *rel_k = nullptr, *per_dim_scale = nullptr;
    ggml_tensor *lconv_pre_norm = nullptr, *lconv_conv_norm = nullptr;
    ggml_tensor *lconv_start = nullptr, *lconv_end = nullptr, *lconv_dw = nullptr;
    AudClip cs, ce;
};

// --------------------------------------------------------------- mel + fft

// in-place iterative radix-2 complex FFT (n = power of 2)
static void fft_radix2(float * re, float * im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / (float)len;
        float wr = cosf(ang), wi = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                int a = i + k, b = i + k + len / 2;
                float tr = re[b] * cr - im[b] * ci;
                float ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr; im[b] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
                float nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr; cr = nr;
            }
        }
    }
}

// HTK mel scale, matching transformers.audio_utils (2595*log10(1 + f/700))
static inline float hz_to_mel(float f) { return 2595.0f * log10f(1.0f + f / 700.0f); }
static inline float mel_to_hz(float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); }

// Gemma4AudioFeatureExtractor: waveform (16 kHz mono) -> log-mel [128, T] and
// the per-frame validity mask (a frame is valid when its window end lands on
// real audio - mirrors the padded raw-waveform mask the HF processor builds).
static bool compute_mel(const std::vector<float> & wav, int fft_n, int frame_len, int hop,
                        int n_bins, float floor_, float min_f, float max_f, int sr,
                        std::vector<float> & mel, std::vector<uint8_t> & valid,
                        int & T) {
    const int pad_left = frame_len / 2;                    // semicausal time padding
    const int unfold = frame_len + 1;
    const int total = pad_left + (int)wav.size();
    const int n_frames = (total - unfold) / hop + 1;
    if (n_frames <= 0) return false;
    T = n_frames;
    mel.assign((size_t)n_frames * n_bins, 0.0f);
    valid.assign((size_t)n_frames, 0);

    // periodic Hann over frame_len
    std::vector<float> win(frame_len);
    for (int n = 0; n < frame_len; n++) win[n] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * n / frame_len);

    // HTK triangular filterbank, no norm (transformers mel_filter_bank norm=None)
    std::vector<float> fb((size_t)n_bins * (fft_n / 2 + 1), 0.0f);
    {
        const float m_min = hz_to_mel(min_f), m_max = hz_to_mel(max_f);
        const int n_pts = n_bins + 2;
        std::vector<float> pts(n_pts);
        for (int i = 0; i < n_pts; i++) pts[i] = mel_to_hz(m_min + (m_max - m_min) * i / (n_pts - 1));
        std::vector<float> bins(fft_n / 2 + 1);
        for (int i = 0; i <= fft_n / 2; i++) bins[i] = (float)sr * 0.5f * i / (fft_n / 2);
        for (int b = 0; b <= fft_n / 2; b++) {
            for (int j = 0; j < n_bins; j++) {
                const float lo = pts[j], mid = pts[j + 1], hi = pts[j + 2];
                float w = 0.0f;
                if (bins[b] >= lo && bins[b] <= hi) {
                    if (bins[b] <= mid) w = lo == mid ? 1.0f : (bins[b] - lo) / (mid - lo);
                    else                w = mid == hi ? 1.0f : (hi - bins[b]) / (hi - mid);
                }
                fb[(size_t)j * (fft_n / 2 + 1) + b] = w;
            }
        }
    }

    std::vector<float> re(fft_n), im(fft_n);
    for (int f = 0; f < n_frames; f++) {
        std::fill(re.begin(), re.end(), 0.0f);
        std::fill(im.begin(), im.end(), 0.0f);
        const int base = f * hop;
        for (int n = 0; n < frame_len; n++) {          // unfold size 321, drop the last sample
            const int idx = base + n;
            float s = (idx >= pad_left && idx < total) ? wav[idx - pad_left] : 0.0f;
            re[n] = s * win[n];
        }
        fft_radix2(re.data(), im.data(), fft_n);
        for (int b = 0; b <= fft_n / 2; b++) {
            const float mag = sqrtf(re[b] * re[b] + im[b] * im[b]);
            for (int j = 0; j < n_bins; j++)
                mel[(size_t)f * n_bins + j] += mag * fb[(size_t)j * (fft_n / 2 + 1) + b];
        }
        // frame validity: padded-mask index i*hop + unfold - 1 must be real audio
        const int mask_idx = base + unfold - 1;
        valid[f] = (mask_idx >= pad_left && mask_idx < total) ? 1 : 0;
    }
    for (auto & v : mel) v = logf(v + floor_);
    return true;
}

// --------------------------------------------------------------- model

class EmbedGemma2 : public vqa::Model {
public:
    ~EmbedGemma2() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "embgemma2"; }
    bool embed(const vqa::MediaInput & media, const std::string & text,
               const vqa::EmbedParams & ep, std::vector<float> & out) override;
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp,
             std::string & out_text) override {
        // embedding-only backend: answer with the CSV of the pooled embedding
        std::vector<float> e;
        vqa::EmbedParams ep;
        ep.preset = "none";
        if (!embed(media, prompt, ep, e)) return false;
        char buf[32];
        for (size_t i = 0; i < e.size(); i++) {
            snprintf(buf, sizeof(buf), "%s%.6f", i ? ", " : "", (double)e[i]);
            out_text += buf;
        }
        return true;
    }

    int n_ctx = 8192;   // 32-frame video alone is ~3.9k tokens (encoder: masks are sized per actual T)
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

private:
    bool init_backend();
    bool load_weights(const char * path);

    // vision
    bool preprocess_image(const uint8_t * rgb, int w0, int h0, std::vector<float> & patches,
                          std::vector<int> & xs, std::vector<int> & ys, int & gw, int & gh,
                          int max_soft = 0);
    bool build_vision_graph(int P, int n_soft, int B, int pw, int ph);
    bool run_vision(const std::vector<float> & patches, const std::vector<int> & xs,
                    const std::vector<int> & ys, int P, int n_soft, int B, int pw, int ph,
                    std::vector<float> & soft);

    // audio
    bool build_audio_graph(int T, int T_soft);
    bool run_audio(const std::vector<float> & mel, const std::vector<uint8_t> & valid,
                   int T, int T_soft, std::vector<float> & soft);

    // text
    bool build_text_graph(int T, int S, bool have_media);
    bool run_text(const std::vector<int> & toks, const std::vector<int> & soft_pos,
                  const std::vector<float> & soft, std::vector<float> & embd);

    void dump(ggml_cgraph * gf, const char * nm) {
        if (!dump_dir.empty()) vqa::dump::write(gf, dump_dir.c_str(), nm);
    }

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    // text hyper
    int n_layer = 24, n_embd = 512, n_ff = 2048, n_vocab = 262144;
    int ple_dim = 512, out_dim = 768, window = 512;
    int n_head = 4, n_kv_head = 2, head_dim = 256;        // sliding layers
    int g_head_dim = 512, g_kv_head = 1;                  // full-attention layers
    float eps = 1e-6f, rope_swa = 1e4f, rope_full = 1e6f;
    bool is_full(int il) const { return (il % 6) == 5; }

    // vision hyper
    int v_layer = 16, v_embd = 768, v_head = 12, v_head_dim = 64, v_ff = 3072;
    int v_patch = 16, v_pool = 3, v_pos_table = 10240, v_max_soft = 280, v_resample = 3;
    int v_max_soft_video = 140;   // video processor uses half the image soft budget
    float v_rope = 100.0f, v_eps = 1e-6f;
    float v_fps = 1.0f;

    // audio hyper
    int a_layer = 12, a_embd = 1024, a_head = 8, a_ff = 4096, a_out_dim = 1536;
    int a_chunk = 12, a_ctx_left = 13, a_ctx_right = 0;
    float a_logit_cap = 50.0f, a_eps = 1e-6f, a_residual = 0.5f;
    int mel_bins = 128, mel_fft = 512, mel_frame = 320, mel_hop = 160, mel_sr = 16000;
    float mel_floor = 1e-3f, mel_max_f = 8000.0f, mel_min_f = 0.0f;

    // tokens
    int id_bos = 2, id_eos = 1, id_pad = 0;
    int id_boi = 255999, id_eoi = 258882, id_image = 258880, id_video = 258884;
    int id_boa = 256000, id_eoa = 258883, id_audio = 258881;

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    vqa::MMap mm;
    vqa::AlignedBuf host_f32;
    ggml_gallocr_t allocr = nullptr;

    // text weights
    ggml_tensor *t_embed = nullptr, *t_norm = nullptr, *t_out_proj = nullptr;
    ggml_tensor *t_ple_model_proj = nullptr, *t_ple_norm = nullptr;
    std::vector<TxtLayer> tl;

    // vision weights
    ggml_tensor *v_patch_proj = nullptr, *v_pos_table_x = nullptr, *v_pos_table_y = nullptr;
    int pos_row_stride = 10240;   // pos table row stride: y ids are offset by it
    std::vector<float> mm_proj_host;   // f32 snapshot of v.mm_proj for the host projection
    ggml_tensor *v_mm_proj = nullptr;
    std::vector<VisLayer> vl;

    // audio weights
    ggml_tensor *a_sub_in_proj = nullptr, *a_out_proj = nullptr, *a_out_bias = nullptr;
    ggml_tensor *a_mm_proj = nullptr, *a_rel_pos = nullptr;   // rel_pos built at load
    ggml_tensor *a_sub_l0_conv = nullptr, *a_sub_l0_norm = nullptr;
    ggml_tensor *a_sub_l1_conv = nullptr, *a_sub_l1_norm = nullptr;
    vqa::AlignedBuf rel_pos_storage;
    ggml_context * actx = nullptr;                            // owns a_rel_pos
    ggml_backend_buffer_t abuf = nullptr;
    std::vector<AudLayer> al;

    // vision graph (rebuilt when the patch count / batch changes)
    ggml_context * pctx = nullptr;          // soft-token projection mini graph
    ggml_cgraph * pgf = nullptr;
    ggml_gallocr_t pallocr = nullptr;
    ggml_context * vctx = nullptr;
    ggml_cgraph * vgf = nullptr;
    ggml_tensor *v_patches_in = nullptr, *v_xs_in = nullptr, *v_ys_in = nullptr;
    ggml_tensor *v_soft_out = nullptr;
    int vg_P = 0, vg_B = 0, vg_soft = 0, vg_pw = 0, vg_ph = 0;

    // audio graph
    ggml_context * actx2 = nullptr;
    ggml_cgraph * agf = nullptr;
    ggml_tensor *a_mel_in = nullptr, *a_soft_out = nullptr;
    std::vector<ggml_tensor *> a_diag;          // 13 [T, T] diagonal-select masks
    ggml_tensor *a_attn_mask_in = nullptr;      // [T, T] key validity (-inf masked)
    int ag_T = 0, ag_soft = 0;

    // text graph
    ggml_context * tctx = nullptr;
    ggml_cgraph * tgf = nullptr;
    int tg_T = 0, tg_S = 0, tg_pad = 0;
    ggml_tensor *t_toks_in = nullptr, *t_soft_pos_in = nullptr, *t_soft_in = nullptr;
    ggml_tensor *t_mask_swa = nullptr, *t_mask_full = nullptr;
    ggml_tensor *t_embd_out = nullptr;

    int vis_runs = 0;
};

EmbedGemma2::~EmbedGemma2() {
    if (pctx) ggml_free(pctx);
    if (pallocr) ggml_gallocr_free(pallocr);
    if (tctx) ggml_free(tctx);
    if (actx2) ggml_free(actx2);
    if (vctx) ggml_free(vctx);
    if (actx) ggml_free(actx);
    if (abuf) ggml_backend_buffer_free(abuf);
    if (allocr) ggml_gallocr_free(allocr);
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (backend) ggml_backend_free(backend);
}

bool EmbedGemma2::init_backend() {
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

bool EmbedGemma2::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer = vqa::gguf::u32(gu, "embgemma2.block_count", n_layer);
    n_embd = vqa::gguf::u32(gu, "embgemma2.embedding_length", n_embd);
    n_ff = vqa::gguf::u32(gu, "embgemma2.feed_forward_length", n_ff);
    n_head = vqa::gguf::u32(gu, "embgemma2.attention.head_count", n_head);
    n_kv_head = vqa::gguf::u32(gu, "embgemma2.attention.head_count_kv", n_kv_head);
    head_dim = vqa::gguf::u32(gu, "embgemma2.attention.key_length", head_dim);
    window = vqa::gguf::u32(gu, "embgemma2.attention.sliding_window", window);
    eps = vqa::gguf::f32(gu, "embgemma2.attention.layer_norm_rms_epsilon", eps);
    rope_full = vqa::gguf::f32(gu, "embgemma2.rope.freq_base", rope_full);
    rope_swa = vqa::gguf::f32(gu, "embgemma2.rope.freq_base_swa", rope_swa);
    ple_dim = vqa::gguf::u32(gu, "embgemma2.embedding_length_per_layer_input", ple_dim);
    out_dim = vqa::gguf::u32(gu, "embgemma2.embedding_out_length", out_dim);
    n_vocab = vqa::gguf::u32(gu, "embgemma2.vocab_size", n_vocab);

    v_layer = vqa::gguf::u32(gu, "embgemma2.vision.block_count", v_layer);
    v_embd = vqa::gguf::u32(gu, "embgemma2.vision.embedding_length", v_embd);
    v_ff = vqa::gguf::u32(gu, "embgemma2.vision.feed_forward_length", v_ff);
    v_head = vqa::gguf::u32(gu, "embgemma2.vision.attention.head_count", v_head);
    v_head_dim = vqa::gguf::u32(gu, "embgemma2.vision.attention.key_length", v_head_dim);
    v_patch = vqa::gguf::u32(gu, "embgemma2.vision.patch_size", v_patch);
    v_pool = vqa::gguf::u32(gu, "embgemma2.vision.pooling_kernel_size", v_pool);
    v_pos_table = vqa::gguf::u32(gu, "embgemma2.vision.position_table_size", v_pos_table);
    v_rope = vqa::gguf::f32(gu, "embgemma2.vision.rope.freq_base", v_rope);
    v_eps = vqa::gguf::f32(gu, "embgemma2.vision.layer_norm_rms_epsilon", v_eps);
    v_max_soft = vqa::gguf::u32(gu, "embgemma2.vision.max_soft_tokens", v_max_soft);
    v_max_soft_video = vqa::gguf::u32(gu, "embgemma2.vision.video_max_soft_tokens", v_max_soft_video);
    v_resample = vqa::gguf::u32(gu, "embgemma2.vision.resize_resample", v_resample);
    v_fps = vqa::gguf::f32(gu, "embgemma2.vision.video_fps", v_fps);

    a_layer = vqa::gguf::u32(gu, "embgemma2.audio.block_count", a_layer);
    a_embd = vqa::gguf::u32(gu, "embgemma2.audio.embedding_length", a_embd);
    a_ff = vqa::gguf::u32(gu, "embgemma2.audio.feed_forward_length", a_ff);
    a_head = vqa::gguf::u32(gu, "embgemma2.audio.attention.head_count", a_head);
    a_chunk = vqa::gguf::u32(gu, "embgemma2.audio.attention_chunk_size", a_chunk);
    a_ctx_left = vqa::gguf::u32(gu, "embgemma2.audio.attention_context_left", a_ctx_left);
    a_ctx_right = vqa::gguf::u32(gu, "embgemma2.audio.attention_context_right", a_ctx_right);
    a_logit_cap = vqa::gguf::f32(gu, "embgemma2.audio.attention_logit_cap", a_logit_cap);
    a_out_dim = vqa::gguf::u32(gu, "embgemma2.audio.out_proj_length", a_out_dim);
    a_eps = vqa::gguf::f32(gu, "embgemma2.audio.layer_norm_rms_epsilon", a_eps);
    a_residual = vqa::gguf::f32(gu, "embgemma2.audio.residual_weight", a_residual);

    mel_bins = vqa::gguf::u32(gu, "embgemma2.mel.bins", mel_bins);
    mel_fft = vqa::gguf::u32(gu, "embgemma2.mel.fft_length", mel_fft);
    mel_frame = vqa::gguf::u32(gu, "embgemma2.mel.frame_length", mel_frame);
    mel_hop = vqa::gguf::u32(gu, "embgemma2.mel.hop_length", mel_hop);
    mel_sr = vqa::gguf::u32(gu, "embgemma2.mel.sample_rate", mel_sr);
    mel_floor = vqa::gguf::f32(gu, "embgemma2.mel.floor", mel_floor);
    mel_max_f = vqa::gguf::f32(gu, "embgemma2.mel.max_frequency", mel_max_f);
    mel_min_f = vqa::gguf::f32(gu, "embgemma2.mel.min_frequency", mel_min_f);

    id_bos = vqa::gguf::u32(gu, "embgemma2.bos_token_id", id_bos);
    id_eos = vqa::gguf::u32(gu, "embgemma2.eos_token_id", id_eos);
    id_boi = vqa::gguf::u32(gu, "embgemma2.boi_token_id", id_boi);
    id_eoi = vqa::gguf::u32(gu, "embgemma2.eoi_token_id", id_eoi);
    id_image = vqa::gguf::u32(gu, "embgemma2.image_token_id", id_image);
    id_video = vqa::gguf::u32(gu, "embgemma2.video_token_id", id_video);
    id_boa = vqa::gguf::u32(gu, "embgemma2.boa_token_id", id_boa);
    id_eoa = vqa::gguf::u32(gu, "embgemma2.eoa_token_id", id_eoa);
    id_audio = vqa::gguf::u32(gu, "embgemma2.audio_token_id", id_audio);

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);

    vqa::weights::promote_1d_to_f32(wctx, host_f32);
    // NOTE: backend binding (which nulls host tensor pointers on GPU) is
    // deliberately deferred to the END of this function - below, mm_proj is
    // snapshotted to f32 and the audio clip scalars are read from host memory

    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) fprintf(stderr, "missing tensor: %s\n", n);
        return t;
    };
    t_embed = gt("t.embed.weight");
    t_norm = gt("t.norm.weight");
    t_out_proj = gt("t.out_proj.weight");
    t_ple_model_proj = gt("t.ple_model_proj.weight");
    t_ple_norm = gt("t.ple_norm.weight");
    tl.resize((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "t.blk.%d.%s", i, suf); return gt(b); };
        TxtLayer & L = tl[(size_t)i];
        L.attn_norm = g("attn_norm.weight");
        L.attn_post_norm = g("attn_post_norm.weight");
        L.wq = g("wq.weight"); L.wk = g("wk.weight"); L.wv = g("wv.weight"); L.wo = g("wo.weight");
        L.q_norm = g("q_norm.weight"); L.k_norm = g("k_norm.weight");
        L.ffn_norm = g("ffn_norm.weight"); L.ffn_post_norm = g("ffn_post_norm.weight");
        L.w_gate = g("w_gate.weight"); L.w_up = g("w_up.weight"); L.w_down = g("w_down.weight");
        L.ple_gate = g("ple_gate.weight"); L.ple_proj = g("ple_proj.weight");
        L.ple_post_norm = g("ple_post_norm.weight");
        L.out_scale = g("out_scale");
        if (!L.attn_norm || !L.wq || !L.wk || !L.wv || !L.wo || !L.q_norm || !L.k_norm ||
            !L.w_gate || !L.w_up || !L.w_down || !L.ple_gate || !L.ple_proj || !L.out_scale) {
            fprintf(stderr, "load: text layer %d has null tensors\n", i);
            return false;
        }
    }

    v_patch_proj = gt("v.patch_proj.weight");
    {
        // the [2, 10240, 768] table stays whole: the y lookup uses row ids
        // offset by 10240 (both halves are contiguous 768-wide row blocks)
        ggml_tensor * tbl = gt("v.pos_table.weight");
        if (!tbl) return false;
        v_pos_table_x = tbl;
        v_pos_table_y = tbl;
        pos_row_stride = (int)tbl->ne[1];
    }
    v_mm_proj = gt("v.mm_proj.weight");
    // host projection reads the weight bytes directly: snapshot as f32 before
    // GPU binding nulls the tensor data pointer
    {
        const int in_n = (int)v_mm_proj->ne[0], out_n = (int)v_mm_proj->ne[1];
        mm_proj_host.resize((size_t)in_n * out_n);
        vqa::gguf::tensor_to_f32(v_mm_proj, mm_proj_host.data());
    }
    vl.resize((size_t)v_layer);
    for (int i = 0; i < v_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "v.blk.%d.%s", i, suf); return gt(b); };
        VisLayer & L = vl[(size_t)i];
        L.attn_norm = g("attn_norm.weight");
        L.attn_post_norm = g("attn_post_norm.weight");
        L.wq = g("wq.weight"); L.wk = g("wk.weight"); L.wv = g("wv.weight"); L.wo = g("wo.weight");
        L.q_norm = g("q_norm.weight"); L.k_norm = g("k_norm.weight");
        L.ffn_norm = g("ffn_norm.weight"); L.ffn_post_norm = g("ffn_post_norm.weight");
        L.w_gate = g("w_gate.weight"); L.w_up = g("w_up.weight"); L.w_down = g("w_down.weight");
        if (!L.attn_norm || !L.wq || !L.wk || !L.wv || !L.wo || !L.q_norm || !L.k_norm ||
            !L.w_gate || !L.w_up || !L.w_down) {
            fprintf(stderr, "load: vision layer %d has null tensors\n", i);
            return false;
        }
    }

    a_sub_l0_conv = gt("a.sub.l0.conv.weight");
    a_sub_l0_norm = gt("a.sub.l0.norm.weight");
    a_sub_l1_conv = gt("a.sub.l1.conv.weight");
    a_sub_l1_norm = gt("a.sub.l1.norm.weight");
    a_sub_in_proj = gt("a.sub.in_proj.weight");
    a_out_proj = gt("a.out_proj.weight");
    a_out_bias = gt("a.out_proj.bias");
    a_mm_proj = gt("a.mm_proj.weight");
    al.resize((size_t)a_layer);
    // read one clamp scalar out of its (host) tensor data
    auto scalar_of = [&](ggml_tensor * t) -> float {
        return t ? ggml_get_f32_1d(t, 0) : 0.0f;
    };
    auto clip_of = [&](const char * base) {
        AudClip c;
        char s[160];
        ggml_tensor * t;
        snprintf(s, sizeof(s), "%s.input_min", base);   t = ggml_get_tensor(wctx, s);
        c.has_in = !!t; c.in_min = scalar_of(t);
        snprintf(s, sizeof(s), "%s.input_max", base);   t = ggml_get_tensor(wctx, s);
        c.in_max = scalar_of(t);
        snprintf(s, sizeof(s), "%s.output_min", base);  t = ggml_get_tensor(wctx, s);
        c.has_out = !!t; c.out_min = scalar_of(t);
        snprintf(s, sizeof(s), "%s.output_max", base);  t = ggml_get_tensor(wctx, s);
        c.out_max = scalar_of(t);
        return c;
    };
    for (int i = 0; i < a_layer; i++) {
        char b[128];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "a.blk.%d.%s", i, suf); return gt(b); };
        AudLayer & L = al[(size_t)i];
        snprintf(b, sizeof(b), "a.blk.%d", i);   // FF/clip names build on the base
        for (int ff = 0; ff < 2; ff++) {
            AudFF & F = ff == 0 ? L.ff1 : L.ff2;
            const char * fname = ff == 0 ? "ff1" : "ff2";
            char s[160];
            snprintf(s, sizeof(s), "%s.%s.pre_norm", b, fname);
            F.pre_norm = ggml_get_tensor(wctx, s);
            snprintf(s, sizeof(s), "%s.%s.post_norm", b, fname);
            F.post_norm = ggml_get_tensor(wctx, s);
            snprintf(s, sizeof(s), "%s.%s.l1.weight", b, fname);
            F.l1 = ggml_get_tensor(wctx, s);
            snprintf(s, sizeof(s), "%s.%s.l2.weight", b, fname);
            F.l2 = ggml_get_tensor(wctx, s);
            snprintf(s, sizeof(s), "%s.%s.l1", b, fname);
            F.c1 = clip_of(s);
            snprintf(s, sizeof(s), "%s.%s.l2", b, fname);
            F.c2 = clip_of(s);
            if (!F.pre_norm || !F.post_norm || !F.l1 || !F.l2) { fprintf(stderr, "load: FF %s null (pre=%p post=%p l1=%p l2=%p)\n", fname, (void*)F.pre_norm, (void*)F.post_norm, (void*)F.l1, (void*)F.l2); return false; }
        }
        L.norm_pre_attn = g("norm_pre_attn.weight");
        L.norm_post_attn = g("norm_post_attn.weight");
        L.norm_out = g("norm_out.weight");
        L.wq = g("attn.wq.weight"); L.wk = g("attn.wk.weight");
        L.wv = g("attn.wv.weight"); L.post = g("attn.post.weight");
        L.cq = clip_of((snprintf(b, sizeof(b), "a.blk.%d.attn.wq", i), b));
        L.ck = clip_of((snprintf(b, sizeof(b), "a.blk.%d.attn.wk", i), b));
        L.cv = clip_of((snprintf(b, sizeof(b), "a.blk.%d.attn.wv", i), b));
        L.cp = clip_of((snprintf(b, sizeof(b), "a.blk.%d.attn.post", i), b));
        L.rel_k = g("attn.rel_k.weight");
        L.per_dim_scale = g("attn.per_dim_scale.weight");
        L.lconv_pre_norm = g("lconv.pre_norm.weight");
        L.lconv_conv_norm = g("lconv.conv_norm.weight");
        L.lconv_start = g("lconv.start.weight");
        L.lconv_end = g("lconv.end.weight");
        L.lconv_dw = g("lconv.dw.weight");
        L.cs = clip_of((snprintf(b, sizeof(b), "a.blk.%d.lconv.start", i), b));
        L.ce = clip_of((snprintf(b, sizeof(b), "a.blk.%d.lconv.end", i), b));
        if (!L.norm_pre_attn || !L.wq || !L.post || !L.rel_k || !L.per_dim_scale ||
            !L.lconv_start || !L.lconv_end || !L.lconv_dw || !L.norm_out) {
            fprintf(stderr, "load: audio layer %d has null tensors\n", i);
            return false;
        }
    }

    // audio relative-position table [13, 1024]: rows are positions 12..0 of the
    // chunk context, [sin | cos] over 512 log-spaced timescales (constant, no
    // checkpoint weights) - built once and bound like a weight
    {
        const int ctx_size = a_chunk + (a_ctx_left - 1) + a_ctx_right;
        const int rows = ctx_size / 2 + 1;
        struct ggml_init_params ip = { 1ull * 1024 * 1024, NULL, true };
        actx = ggml_init(ip);
        if (!actx) { fprintf(stderr, "load: rel-pos ggml_init failed\n"); return false; }
        a_rel_pos = ggml_new_tensor_2d(actx, GGML_TYPE_F32, a_embd, rows);   // columns are position vectors
        rel_pos_storage.resize((size_t)rows * a_embd * sizeof(float));
        float * rp = (float *)rel_pos_storage.data();
        const int half = a_embd / 2;
        for (int r = 0; r < rows; r++) {
            const float pos = (float)(rows - 1 - r);
            for (int k = 0; k < half; k++) {
                const float its = pos * expf(-(float)k * logf(10000.0f) / (float)(half - 1));
                rp[(size_t)r * a_embd + k] = sinf(its);
                rp[(size_t)r * a_embd + half + k] = cosf(its);
            }
        }
        abuf = ggml_backend_alloc_ctx_tensors(actx, backend);
        if (!abuf) { fprintf(stderr, "load: rel-pos buffer alloc failed\n"); return false; }
        ggml_backend_tensor_set(a_rel_pos, rp, 0, ggml_nbytes(a_rel_pos));
        fprintf(stderr, "audio rel-pos table [%d, %d] ready\n", a_embd, rows);
    }

    ggml_tensor * finals[] = { t_embed, t_norm, t_out_proj, t_ple_model_proj,
                               v_patch_proj, v_mm_proj, a_sub_in_proj, a_out_proj,
                               a_mm_proj, a_rel_pos };
    for (size_t i = 0; i < sizeof(finals) / sizeof(finals[0]); i++)
        if (!finals[i]) fprintf(stderr, "load_weights: null core tensor #%zu\n", i);
    for (int i = 0; i < n_layer; i++)
        if (!tl[(size_t)i].attn_norm) fprintf(stderr, "load_weights: null text layer %d\n", i);
    for (int i = 0; i < v_layer; i++)
        if (!vl[(size_t)i].attn_norm) fprintf(stderr, "load_weights: null vis layer %d\n", i);
    for (int i = 0; i < a_layer; i++)
        if (!al[(size_t)i].wq) fprintf(stderr, "load_weights: null audio layer %d\n", i);
    const bool ok = t_embed && t_norm && t_out_proj && t_ple_model_proj && v_patch_proj &&
                    v_mm_proj && a_sub_in_proj && a_out_proj && a_mm_proj && a_rel_pos;
    if (ok) {
        vqa::weights::Bound bound = vqa::weights::bind_to_backend(
            wctx, backend, mm, { { host_f32.data(), host_f32.size() } });
        if (!bound.ok) return false;
        wbuf = bound.main_buf;
        if (!ggml_backend_is_cpu(backend)) mm.release();
    }
    return ok;
}

bool EmbedGemma2::load(const std::string & gguf_path, const std::string & tokenizer_path,
                       const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "EMBG2", n_threads, use_gpu, dump_dir);
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::GemmaSP)) return false;

    caps_.image = true;
    caps_.video = true;
    fprintf(stderr, "embgemma2: ready (%dL d=%d ple=%d out=%d | vision %dL d=%d hd=%d | "
                    "audio %dL d=%d heads=%d chunk=%d)\n",
            n_layer, n_embd, ple_dim, out_dim, v_layer, v_embd, v_head_dim,
            a_layer, a_embd, a_head, a_chunk);
    return true;
}

// ---------------------------------------------------------------- vision

// resize inside the patch budget (aspect preserved, sides multiple of
// pool*patch) and patchify to [768, P] with (x, y) patch coordinates; input
// values are 2*(x/255 - 0.5), channel-major within the patch
bool EmbedGemma2::preprocess_image(const uint8_t * rgb, int w0, int h0,
                                   std::vector<float> & patches, std::vector<int> & xs,
                                   std::vector<int> & ys, int & gw, int & gh,
                                   int max_soft) {
    const int P = v_patch, K = v_pool;
    const double total = (double)w0 * h0;
    const double budget = (double)(max_soft > 0 ? max_soft : v_max_soft) * K * K * P * P;
    const double factor = sqrt(budget / total);
    int side_mult = K * P;
    int tw = std::max(1, (int)(factor * w0)) / side_mult * side_mult;
    int th = std::max(1, (int)(factor * h0)) / side_mult * side_mult;
    if (tw == 0 && th == 0) { tw = side_mult; th = side_mult; }
    if (tw == 0) tw = side_mult;
    if (th == 0) th = side_mult;

    std::vector<float> resized;
    vqa::img::resize_antialias(rgb, w0, h0, tw, th, resized, v_resample);
    for (auto & v : resized) v = std::min(255.0f, std::max(0.0f, v));   // PIL clamps u8

    gw = tw / P;
    gh = th / P;
    const int P3 = 3 * P * P;
    patches.assign((size_t)P3 * gw * gh, 0.0f);
    xs.resize((size_t)gw * gh);
    ys.resize((size_t)gw * gh);
    int u = 0;
    for (int py = 0; py < gh; py++)
        for (int px = 0; px < gw; px++, u++) {
            float * row = patches.data() + (size_t)u * P3;
            int o = 0;
            // HF Gemma4 patches are channel-LAST within the patch: (dy, dx, c)
            for (int dy = 0; dy < P; dy++) {
                const float * s = resized.data() + ((size_t)(py * P + dy) * tw + px * P) * 3;
                for (int dx = 0; dx < P; dx++)
                    for (int c = 0; c < 3; c++)
                        row[o++] = (s[(size_t)dx * 3 + c] / 255.0f - 0.5f) * 2.0f;
            }
            xs[(size_t)u] = px;
            ys[(size_t)u] = py;
        }
    return true;
}

bool EmbedGemma2::build_vision_graph(int P, int n_soft, int B, int pw, int ph) {
    if (vctx) { ggml_free(vctx); vctx = nullptr; }
    const int E = v_embd, C3 = 3 * v_patch * v_patch;   // P is the PATCH COUNT here

    struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    vctx = ggml_init(ip);
    if (!vctx) return false;
    vgf = ggml_new_graph_custom(vctx, 32768, false);

    const int hd = v_head_dim;

    v_patches_in = ggml_new_tensor_3d(vctx, GGML_TYPE_F32, C3, P, B);
    ggml_set_name(v_patches_in, "v_patches");
    // coordinates repeated per frame (rope walks the P*B token axis); the
    // position-embedding get_rows reshape turns the same vector into [E, P, B]
    v_xs_in = ggml_new_tensor_1d(vctx, GGML_TYPE_I32, P * B);
    v_ys_in = ggml_new_tensor_1d(vctx, GGML_TYPE_I32, P * B);

    // patchify: x = proj(patches) + pos(x) + pos(y), scaling 2*(x-0.5) done host-side
    ggml_tensor * x = ggml_mul_mat(vctx, v_patch_proj, v_patches_in);   // [E, P, B]
    // the [E, 10240, 2] table is sliced into per-axis views here (the weight
    // context has no spare pool for views; this one does)
    ggml_tensor * tbl = v_pos_table_x;
    ggml_tensor * tbl_x = ggml_view_3d(vctx, tbl, tbl->ne[0], tbl->ne[1], 1,
                                       tbl->nb[1], tbl->nb[2], 0);
    ggml_tensor * tbl_y = ggml_view_3d(vctx, tbl, tbl->ne[0], tbl->ne[1], 1,
                                       tbl->nb[1], tbl->nb[2], tbl->nb[2]);
    ggml_tensor * pos_x = ggml_cast(vctx,
        ggml_reshape_3d(vctx, ggml_get_rows(vctx, tbl_x, v_xs_in), E, P, B), GGML_TYPE_F32);
    ggml_tensor * pos_y = ggml_cast(vctx,
        ggml_reshape_3d(vctx, ggml_get_rows(vctx, tbl_y, v_ys_in), E, P, B), GGML_TYPE_F32);
    x = ggml_add(vctx, x, ggml_add(vctx, pos_x, pos_y));
    ggml_set_name(x, "v_embed");

    // axial 2D rope, 2 parts of 32 dims: part 0 rotates with the x coordinate,
    // part 1 with y (the HF cos layout [fh, fh, fw, fw] with 32-dim rotate-half
    // parts reduces to a standard NEOX rope over 16 freqs, theta 100)
    auto rope_part = [&](ggml_tensor * q_in, int part, ggml_tensor * pos_ids) {
        ggml_tensor * qq = ggml_view_3d(vctx, q_in, hd / 2, q_in->ne[1], q_in->ne[2],
                                        q_in->nb[1], q_in->nb[2],
                                        (size_t)part * (hd / 2) * sizeof(float));
        return ggml_rope_ext(vctx, qq, pos_ids, nullptr, hd / 2, GGML_ROPE_TYPE_NEOX,
                             131072, v_rope, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    };
    auto rope_all = [&](ggml_tensor * q_in) {
        return ggml_concat(vctx, rope_part(q_in, 0, v_xs_in), rope_part(q_in, 1, v_ys_in), 0);
    };

    for (int il = 0; il < v_layer; il++) {
        VisLayer & L = vl[(size_t)il];
        ggml_tensor * h = ggml_rms_norm(vctx, x, v_eps);
        h = ggml_mul(vctx, h, L.attn_norm);

        ggml_tensor * q = ggml_mul_mat(vctx, L.wq, h);
        ggml_tensor * k = ggml_mul_mat(vctx, L.wk, h);
        ggml_tensor * v = ggml_mul_mat(vctx, L.wv, h);

        q = ggml_reshape_3d(vctx, q, hd, v_head, P * B);
        k = ggml_reshape_3d(vctx, k, hd, v_head, P * B);   // v_head == v_head_kv
        v = ggml_reshape_3d(vctx, v, hd, v_head, P * B);
        q = ggml_mul(vctx, ggml_rms_norm(vctx, q, v_eps), L.q_norm);
        k = ggml_mul(vctx, ggml_rms_norm(vctx, k, v_eps), L.k_norm);
        v = ggml_rms_norm(vctx, v, v_eps);

        q = rope_all(q);
        k = rope_all(k);

        // [hd, P*B, head] for flash attention - the caller runs one frame per
        // graph (B=1): HF attends per frame, and this fork's cont/permute on
        // 4D batch layouts miscompiles, so cross-frame batching is avoided
        ggml_tensor * qf = ggml_cont(vctx, ggml_permute(vctx, q, 0, 2, 1, 3));
        ggml_tensor * kf = ggml_cont(vctx, ggml_permute(vctx, k, 0, 2, 1, 3));
        ggml_tensor * vf = ggml_cont(vctx, ggml_permute(vctx, v, 0, 2, 1, 3));
        ggml_tensor * fa = ggml_flash_attn_ext(vctx, qf, kf, vf, nullptr, 1.0f, 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_3d(vctx, ggml_cont(vctx, fa), E, P, B);
        attn = ggml_mul_mat(vctx, L.wo, attn);
        // Gemma sandwich: post norms apply to the sublayer output before the add
        x = ggml_add(vctx, x, ggml_mul(vctx, ggml_rms_norm(vctx, attn, v_eps), L.attn_post_norm));

        h = ggml_rms_norm(vctx, x, v_eps);
        h = ggml_mul(vctx, h, L.ffn_norm);
        ggml_tensor * gate = ggml_gelu(vctx, ggml_mul_mat(vctx, L.w_gate, h));
        ggml_tensor * up = ggml_mul_mat(vctx, L.w_up, h);
        h = ggml_mul_mat(vctx, L.w_down, ggml_mul(vctx, gate, up));
        x = ggml_add(vctx, x, ggml_mul(vctx, ggml_rms_norm(vctx, h, v_eps), L.ffn_post_norm));
        if (il == 0) ggml_set_name(q, "v_blk0_q_rope");
        { char nm[24]; snprintf(nm, sizeof(nm), "v_blk%d_out", il); ggml_set_name(x, nm); }
    }
    ggml_set_name(x, "v_final");

    // the tower graph ends at per-patch features; pool / pre-norm / projection
    // run on the host (numpy-verified math - the in-graph view-chain pool is
    // miscompiled by this ggml's gallocr, see the parity notes)
    ggml_set_name(x, "v_feats");
    v_soft_out = x;
    ggml_build_forward_expand(vgf, x);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "v_patches", "v_embed", "v_blk0_out", "v_blk1_out", "v_blk2_out",
                                           "v_blk3_out", "v_blk7_out", "v_blk15_out",
                                           "v_final", "v_feats", nullptr };
        vqa::dump::mark_sinks(vctx, vgf, dn);
        vqa::dump::finalize_sinks(vctx, vgf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, vgf)) {
        fprintf(stderr, "vision graph allocation failed\n");
        return false;
    }
    vg_P = P; vg_B = B; vg_soft = n_soft; vg_pw = pw; vg_ph = ph;
    return true;
}

bool EmbedGemma2::run_vision(const std::vector<float> & patches, const std::vector<int> & xs,
                             const std::vector<int> & ys, int P, int n_soft, int B, int pw, int ph,
                             std::vector<float> & soft) {
    if (!vgf || vg_P != P || vg_B != B || vg_soft != n_soft || vg_pw != pw || vg_ph != ph) {
        if (!build_vision_graph(P, n_soft, B, pw, ph)) return false;
    }
    ggml_backend_tensor_set(v_patches_in, patches.data(), 0, sizeof(float) * patches.size());
    std::vector<int> xs_b, ys_b;   // frame-major: token index b*P + p
    xs_b.reserve((size_t)P * B);
    ys_b.reserve((size_t)P * B);
    for (int b = 0; b < B; b++) {
        xs_b.insert(xs_b.end(), xs.begin(), xs.end());
        ys_b.insert(ys_b.end(), ys.begin(), ys.end());
    }
    ggml_backend_tensor_set(v_xs_in, xs_b.data(), 0, sizeof(int) * xs_b.size());
    ggml_backend_tensor_set(v_ys_in, ys_b.data(), 0, sizeof(int) * ys_b.size());

    if (ggml_backend_graph_compute(backend, vgf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vision compute failed\n");
        return false;
    }
    if (!dump_dir.empty() && vis_runs == 0) {
        static const char * const dn[] = { "v_patches", "v_embed", "v_blk0_out", "v_blk1_out", "v_blk2_out",
                                           "v_blk3_out", "v_blk7_out", "v_blk15_out",
                                           "v_final", "v_feats", nullptr };
        for (int i = 0; dn[i]; i++) dump(vgf, dn[i]);
    }
    vis_runs++;

    // host: 3x3 position pool * sqrt(E) -> weightless RMSNorm -> Linear 768->512
    const int E = v_embd, K = v_pool, SD = (int)v_mm_proj->ne[1];
    std::vector<float> feats((size_t)P * B * E);
    ggml_backend_tensor_get(v_soft_out, feats.data(), 0, sizeof(float) * feats.size());
    const int ow = pw / K, oh = ph / K;
    std::vector<float> pooled((size_t)n_soft * B * E, 0.0f);
    const float scale = sqrtf((float)E) / (float)(K * K);
    for (int b = 0; b < B; b++)
        for (int iy = 0; iy < oh; iy++)
            for (int ix = 0; ix < ow; ix++) {
                float * dst = pooled.data() + ((size_t)(b * n_soft + iy * ow + ix)) * E;
                for (int dy = 0; dy < K; dy++)
                    for (int dx = 0; dx < K; dx++) {
                        const float * src = feats.data() +
                            (((size_t)b * P) + (size_t)((iy * K + dy) * pw + (ix * K + dx))) * E;
                        for (int e = 0; e < E; e++) dst[e] += src[e];
                    }
                for (int e = 0; e < E; e++) dst[e] *= scale;
            }
    const float * wp = mm_proj_host.data();
    soft.resize((size_t)SD * n_soft * B);
    for (size_t t = 0; t < (size_t)n_soft * B; t++) {
        const float * xrow = pooled.data() + t * E;
        double sq = 0;
        for (int e = 0; e < E; e++) sq += (double)xrow[e] * xrow[e];
        const float inv = sqrtf((float)E) / sqrtf((float)sq + v_eps);   // 1/RMS
        float * orow = soft.data() + t * SD;
        for (int o = 0; o < SD; o++) {
            const float * wrow = wp + (size_t)o * E;
            float acc = 0.0f;
            for (int e = 0; e < E; e++) acc += xrow[e] * inv * wrow[e];
            orow[o] = acc;
        }
    }
    if (getenv("EMBG2_DEBUG")) {
        FILE * f = fopen("/tmp/eg2_engine_soft.bin", "wb");
        if (f) { fwrite(soft.data(), sizeof(float), soft.size(), f); fclose(f); }
    }
    return true;
}

// ---------------------------------------------------------------- audio

bool EmbedGemma2::build_audio_graph(int T, int T_soft) {
    if (actx2) { ggml_free(actx2); actx2 = nullptr; }
    struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    actx2 = ggml_init(ip);
    if (!actx2) return false;
    agf = ggml_new_graph_custom(actx2, 32768, false);

    const int D = a_embd, hd = D / a_head;

    a_mel_in = ggml_new_tensor_2d(actx2, GGML_TYPE_F32, mel_bins, T);
    ggml_set_name(a_mel_in, "a_mel");

    // ---- subsample convs: two 3x3 stride-2 pad-1 convs over (freq, time).
    // ggml conv_2d keeps the HF kernel layout [C_out, C_in, 3, 3] and expects
    // the input as [W, H, C_in, N]; channel-wise LayerNorm needs C in ne0, so
    // each stage permutes to [C, W, H] for the norm and back for the next conv.
    const int W1 = (mel_bins + 2 - 3) / 2 + 1;
    const int H1 = (T + 2 - 3) / 2 + 1;
    ggml_tensor * c1 = ggml_conv_2d(actx2, a_sub_l0_conv, a_mel_in, 2, 2, 1, 1, 1, 1);
    if (getenv("EMBG2_DEBUG"))
        fprintf(stderr, "[dbg] conv out ne=[%lld %lld %lld %lld] kernel ne=[%lld %lld %lld %lld] mel ne=[%lld %lld]\n",
                (long long)c1->ne[0], (long long)c1->ne[1], (long long)c1->ne[2], (long long)c1->ne[3],
                (long long)a_sub_l0_conv->ne[0], (long long)a_sub_l0_conv->ne[1],
                (long long)a_sub_l0_conv->ne[2], (long long)a_sub_l0_conv->ne[3],
                (long long)a_mel_in->ne[0], (long long)a_mel_in->ne[1]);
    // [W1, H1, 128, 1] -> [128, W1, H1]
    c1 = ggml_cont(actx2, ggml_permute(actx2, c1, 1, 2, 0, 3));   // [W,H,C] -> [C,W,H]
    if (getenv("EMBG2_DEBUG")) fprintf(stderr, "[dbg] mark A conv1 pre-norm: c1 ne=[%lld %lld %lld] norm ne=[%lld %lld]\n",
            (long long)c1->ne[0], (long long)c1->ne[1], (long long)c1->ne[2],
            (long long)a_sub_l0_norm->ne[0], (long long)a_sub_l0_norm->ne[1]);
    {
        ggml_tensor * n1 = ggml_norm(actx2, c1, a_eps);
        if (getenv("EMBG2_DEBUG")) fprintf(stderr, "[dbg] norm(c1) ne=[%lld %lld %lld]\n",
                (long long)n1->ne[0], (long long)n1->ne[1], (long long)n1->ne[2]);
        c1 = ggml_relu(actx2, ggml_mul(actx2, n1, a_sub_l0_norm));
    }
    if (getenv("EMBG2_DEBUG")) fprintf(stderr, "[dbg] mark A2 conv1 post-norm\n");
    ggml_set_name(c1, "a_conv1");
    // back to [W1, H1, 128] for the second conv
    c1 = ggml_cont(actx2, ggml_permute(actx2, c1, 2, 0, 1, 3));   // [C,W,H] -> [W,H,C]
    const int C1 = (int)a_sub_l1_conv->ne[3];
    const int W2 = (W1 + 2 - 3) / 2 + 1;
    const int H2 = (H1 + 2 - 3) / 2 + 1;
    ggml_tensor * c2 = ggml_conv_2d(actx2, a_sub_l1_conv, c1, 2, 2, 1, 1, 1, 1);
    // [W2, H2, 32, 1] -> [32, W2, H2]
    c2 = ggml_cont(actx2, ggml_permute(actx2, c2, 1, 2, 0, 3));   // [W,H,C] -> [C,W,H]
    if (getenv("EMBG2_DEBUG")) fprintf(stderr, "[dbg] mark B conv2 pre-norm\n");
    c2 = ggml_relu(actx2, ggml_mul(actx2, ggml_norm(actx2, c2, a_eps), a_sub_l1_norm));
    ggml_set_name(c2, "a_c2");
    if (getenv("EMBG2_DEBUG")) fprintf(stderr, "[dbg] mark B2 conv2 post-norm\n");

    // flatten per time step -> Linear -> hidden [D, H2]. c2 is [C, W, H] in
    // ggml order (ne0 = C fastest), so the plain reshape yields the HF per-token
    // vector v[w*C + c] directly - no permute (that would flip to v[c*W + w]).
    c2 = ggml_reshape_2d(actx2, c2, C1 * W2, H2);
    ggml_tensor * h = ggml_mul_mat(actx2, a_sub_in_proj, c2);       // [D, H2]
    if (getenv("EMBG2_DEBUG")) fprintf(stderr, "[dbg] mark C in_proj done\n");
    ggml_set_name(h, "a_sub");

    // ---- conformer layers. The HF attention is chunked (12) with 12 past /
    // 0 future, expressed here as full T x T attention whose bias is
    // ac + bd + mask, bd[i][j] = q_i . rel_pos[clip(i - j, 0, 12)].
    // bd is built from D = q . rel_pos [T, 13] by placing column r on the
    // r-th diagonal (13 static diagonal-select masks).
    a_diag.assign(13, nullptr);
    for (int r = 0; r <= a_ctx_left - 1; r++) {
        ggml_tensor * m = ggml_new_tensor_2d(actx2, GGML_TYPE_F32, H2, H2);
        char nm[32];
        snprintf(nm, sizeof(nm), "a_diag_%d", r);
        ggml_set_name(m, nm);
        a_diag[(size_t)r] = m;
    }
    a_attn_mask_in = ggml_new_tensor_2d(actx2, GGML_TYPE_F32, H2, H2);
    ggml_set_name(a_attn_mask_in, "a_attn_mask");

    const float q_scale = powf((float)hd, -0.5f) / logf(2.0f);
    const float k_scale = logf(1.0f + (float)M_E) / logf(2.0f);

    // shared graph inputs (referenced by every layer, uploaded once in run_audio)
    ggml_tensor * zeros_pad = ggml_new_tensor_2d(actx2, GGML_TYPE_F32, 4, D);
    ggml_set_name(zeros_pad, "a_zeros");

    // one macaron feed-forward block (ClippableLinear = clamp in -> linear -> clamp out)
    auto macaron_ff = [&](const AudFF & F, ggml_tensor * x) {
        ggml_tensor * f = ggml_mul(actx2, ggml_rms_norm(actx2, x, a_eps), F.pre_norm);
        f = F.c1.apply(actx2, f, true);
        f = F.c1.apply(actx2, ggml_mul_mat(actx2, F.l1, f), false);
        f = ggml_silu(actx2, f);
        f = F.c2.apply(actx2, f, true);
        f = F.c2.apply(actx2, ggml_mul_mat(actx2, F.l2, f), false);
        f = ggml_mul(actx2, ggml_rms_norm(actx2, f, a_eps), F.post_norm);
        return ggml_scale(actx2, f, a_residual);
    };

    for (int il = 0; il < a_layer; il++) {
        AudLayer & L = al[(size_t)il];
        if (getenv("EMBG2_DEBUG")) fprintf(stderr, "[dbg] L%d ff1\n", il);
        h = ggml_add(actx2, h, macaron_ff(L.ff1, h));
        // HF layer: residual for the attention branch is the POST-ff1 state
        ggml_tensor * res = h;

        // attention (chunked local in HF: 12 past + 0 future, softcap 50)
        ggml_tensor * cur = ggml_mul(actx2, ggml_rms_norm(actx2, h, a_eps), L.norm_pre_attn);
        ggml_tensor * qc = L.cq.apply(actx2, cur, true);
        ggml_tensor * kc = L.ck.apply(actx2, cur, true);
        ggml_tensor * vc = L.cv.apply(actx2, cur, true);
        ggml_tensor * q = L.cq.apply(actx2, ggml_mul_mat(actx2, L.wq, qc), false);
        ggml_tensor * k = L.ck.apply(actx2, ggml_mul_mat(actx2, L.wk, kc), false);
        ggml_tensor * v = L.cv.apply(actx2, ggml_mul_mat(actx2, L.wv, vc), false);
        // q * q_scale * softplus(per_dim_scale), tiled over the 8 heads
        ggml_tensor * pds = ggml_repeat(actx2, ggml_reshape_2d(actx2, L.per_dim_scale, hd, 1), q);
        q = ggml_mul(actx2, ggml_scale(actx2, q, q_scale), ggml_softplus(actx2, pds));
        k = ggml_scale(actx2, k, k_scale);

        // per-head attention: [hd, T, H] tensors (head in ne2 = mul_mat batch,
        // keys in ne0 = softmax axis)
        auto split_heads = [&](ggml_tensor * x) {
            ggml_tensor * r = ggml_reshape_3d(actx2, x, hd, a_head, H2);   // [hd, H, T]
            return ggml_cont(actx2, ggml_permute(actx2, r, 0, 2, 1, 3));   // [hd, T, H]
        };
        ggml_tensor * qh = split_heads(q);
        ggml_tensor * kh = split_heads(k);
        ggml_tensor * vh = split_heads(v);
        if (getenv("EMBG2_DEBUG")) fprintf(stderr, "[dbg] qh ne=[%lld %lld %lld] kh ne=[%lld %lld %lld]\n",
                (long long)qh->ne[0], (long long)qh->ne[1], (long long)qh->ne[2],
                (long long)kh->ne[0], (long long)kh->ne[1], (long long)kh->ne[2]);
        // mul_mat(A,B) gives [A.ne1, B.ne1]: contract keys into ne0 so scores,
        // bd and the softmax axis all agree on [T_k, T_q, H]
        ggml_tensor * ac = ggml_mul_mat(actx2, kh, qh);                    // [T_k, T_q, H]
        // rel bias per head: relk = W_rel . table; D[r, i, h] = q_i^h . relk^h[r]
        ggml_tensor * rel_pos_p = ggml_mul_mat(actx2, L.rel_k, a_rel_pos); // [1024, 13]
        ggml_tensor * rkh = ggml_reshape_3d(actx2, rel_pos_p, hd, a_head, 13);
        rkh = ggml_cont(actx2, ggml_permute(actx2, rkh, 0, 2, 1, 3));      // [hd, 13, H]
        ggml_tensor * DrelT = ggml_mul_mat(actx2, rkh, qh);                // [13, T_q, H]
        ggml_tensor * mask3_shape = ggml_new_tensor_3d(actx2, GGML_TYPE_F32, H2, H2, a_head);
        ggml_tensor * mask3 = ggml_repeat(actx2, a_attn_mask_in, mask3_shape);  // [T_k, T_q, H]
        ggml_tensor * bd = nullptr;
        for (int r = 0; r <= a_ctx_left - 1; r++) {
            // DrelT is [13, T_q, H]: row r = DrelT[r, :, :] -> [1, T_q, H]
            ggml_tensor * row = ggml_view_3d(actx2, DrelT, 1, H2, a_head,
                                             DrelT->nb[1], DrelT->nb[2],
                                             (size_t)r * DrelT->nb[0]);
            ggml_tensor * row_t = ggml_repeat(actx2, row, mask3);          // [T_k, T_q, H]
            ggml_tensor * d3 = ggml_cont(actx2, ggml_repeat(actx2,
                ggml_reshape_3d(actx2, a_diag[(size_t)r], H2, H2, 1), mask3));  // [T_k, T_q, H]
            bd = bd ? ggml_add(actx2, bd, ggml_mul(actx2, row_t, d3)) : ggml_mul(actx2, row_t, d3);
        }
        if (il == 0) ggml_set_name(bd, "a_l0_bd");
        ggml_tensor * scores = ggml_add(actx2, ac, bd);
        scores = ggml_scale(actx2, scores, 1.0f / a_logit_cap);
        scores = ggml_tanh(actx2, scores);
        scores = ggml_scale(actx2, scores, a_logit_cap);
        scores = ggml_add(actx2, scores, mask3);
        ggml_tensor * prob = ggml_soft_max(actx2, scores);                 // softmax over keys
        if (il == 0) ggml_set_name(prob, "a_l0_prob");
        ggml_tensor * v3 = ggml_cont(actx2, ggml_permute(actx2,
            ggml_reshape_3d(actx2, v, hd, a_head, H2), 1, 2, 0, 3));       // [T, hd, H]
        if (getenv("EMBG2_DEBUG")) fprintf(stderr, "[dbg] prob ne=[%lld %lld %lld] v3 ne=[%lld %lld %lld]\n",
                (long long)prob->ne[0], (long long)prob->ne[1], (long long)prob->ne[2],
                (long long)v3->ne[0], (long long)v3->ne[1], (long long)v3->ne[2]);
        ggml_tensor * attn3 = ggml_mul_mat(actx2, prob, v3);               // [T_q, hd, H]
        // permute to [hd, H, T_q]: its contiguous layout (d + h*hd + t*D) IS
        // the [D, T] row-major matrix. cont_2d on a differently-ordered
        // permute is a linear reinterpretation (memcpy) and scrambles.
        ggml_tensor * attn = ggml_cont(actx2, ggml_permute(actx2, attn3, 2, 0, 1, 3));
        attn = ggml_reshape_2d(actx2, attn, D, H2);                        // [D, H2]
        if (il == 0) ggml_set_name(attn, "a_l0_merged");

        // attention output projection (HF self_attn.post, clipped linear)
        attn = L.cp.apply(actx2, attn, true);
        attn = L.cp.apply(actx2, ggml_mul_mat(actx2, L.post, attn), false);
        if (il == 0) ggml_set_name(attn, "a_l0_postout");

        attn = ggml_mul(actx2, ggml_rms_norm(actx2, attn, a_eps), L.norm_post_attn);
        h = ggml_add(actx2, attn, res);
        if (il == 0) ggml_set_name(h, "a_l0_attnres");

        if (getenv("EMBG2_DEBUG")) fprintf(stderr, "[dbg] L%d attn done\n", il);
        // LightDepthConv: GLU + causal depthwise conv k=5
        res = h;
        ggml_tensor * lc = ggml_mul(actx2, ggml_rms_norm(actx2, h, a_eps), L.lconv_pre_norm);
        lc = L.cs.apply(actx2, lc, true);
        lc = L.cs.apply(actx2, ggml_mul_mat(actx2, L.lconv_start, lc), false);
        {
            const int64_t half_d = lc->ne[0] / 2;
            // F.glu(dim=-1) splits contiguous halves of the channel axis
            // (ne0): channels [0, half) gate with sigmoid of [half, 2*half)
            ggml_tensor * a_part = ggml_view_2d(actx2, lc, half_d, lc->ne[1],
                                                lc->nb[1], 0);
            ggml_tensor * b_part = ggml_view_2d(actx2, lc, half_d, lc->ne[1],
                                                lc->nb[1], (size_t)half_d * lc->nb[0]);
            lc = ggml_mul(actx2, a_part, ggml_sigmoid(actx2, b_part));
        }
        // causal depthwise: out[c][t] = sum_k dw[c][k] * lc[c][t + k - 4]
        {
            ggml_tensor * lc_t = ggml_transpose(actx2, lc);      // [T, 1024]
            lc_t = ggml_concat(actx2, zeros_pad, lc_t, 0);       // [T+4, 1024]
            ggml_tensor * dw2 = ggml_view_2d(actx2, L.lconv_dw, 5, lc->ne[0],
                                             5 * ggml_type_size(L.lconv_dw->type), 0);
            ggml_tensor * acc = nullptr;
            for (int kk = 0; kk < 5; kk++) {
                ggml_tensor * shifted = ggml_view_2d(actx2, lc_t, H2, lc->ne[0],
                                                     lc_t->nb[1], (size_t)kk * lc_t->nb[0]);
                ggml_tensor * wv = ggml_view_2d(actx2, dw2, 1, lc->ne[0],
                                                dw2->nb[1], (size_t)kk * dw2->nb[0]);
                ggml_tensor * part = ggml_mul(actx2, shifted, wv);
                acc = acc ? ggml_add(actx2, acc, part) : part;
            }
            lc = ggml_cont(actx2, ggml_transpose(actx2, acc));   // back to [1024, T], contiguous
        }
        // HF clamps to +/- gradient_clipping (1e10, i.e. inert) here - NOT the
        // lconv_end input clip; applying that would distort the conv output
        lc = ggml_clamp(actx2, lc, -1e10f, 1e10f);
        lc = ggml_mul(actx2, ggml_rms_norm(actx2, lc, a_eps), L.lconv_conv_norm);
        lc = ggml_silu(actx2, lc);
        lc = L.ce.apply(actx2, lc, true);
        lc = L.ce.apply(actx2, ggml_mul_mat(actx2, L.lconv_end, lc), false);
        h = ggml_add(actx2, lc, res);
        if (il == 0) ggml_set_name(h, "a_l0_lconv");
        if (getenv("EMBG2_DEBUG")) fprintf(stderr, "[dbg] L%d lconv done\n", il);

        h = ggml_add(actx2, h, macaron_ff(L.ff2, h));

        // layer output: norm_out WITHOUT a residual add
        h = ggml_clamp(actx2, h, -1e9f, 1e9f);
        h = ggml_mul(actx2, ggml_rms_norm(actx2, h, a_eps), L.norm_out);
        if (il < 2) { char nm[24]; snprintf(nm, sizeof(nm), "a_blk%d_out", il); ggml_set_name(h, nm); }
    }

    // output projection (bias) -> weightless RMSNorm -> 1536 -> 512 soft tokens
    h = ggml_mul_mat(actx2, a_out_proj, h);
    h = ggml_add(actx2, h, a_out_bias);
    h = ggml_rms_norm(actx2, h, a_eps);
    ggml_tensor * soft = ggml_mul_mat(actx2, a_mm_proj, h);          // [512, H2]
    ggml_set_name(soft, "a_soft");
    a_soft_out = soft;
    ggml_build_forward_expand(agf, soft);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "a_mel", "a_conv1", "a_c2", "a_sub", "a_l0_bd", "a_l0_q", "a_l0_k", "a_l0_prob", "a_l0_ac", "a_l0_scores", "a_l0_qh", "a_l0_kh", "a_l0_attnres", "a_l0_ff1", "a_l0_v", "a_l0_merged", "a_l0_postout", "a_l0_glu", "a_l0_dwout", "a_l0_lstart", "a_l0_lend", "a_l0_lstart2",
                                           "a_l0_attn_raw", "a_l0_lconv", "a_blk0_out",
                                           "a_blk1_out", "a_soft", nullptr };
        vqa::dump::mark_sinks(actx2, agf, dn);
        vqa::dump::finalize_sinks(actx2, agf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, agf)) {
        fprintf(stderr, "audio graph allocation failed\n");
        return false;
    }
    ag_T = T; ag_soft = T_soft;
    return true;
}

bool EmbedGemma2::run_audio(const std::vector<float> & mel, const std::vector<uint8_t> & valid,
                            int T, int T_soft, std::vector<float> & soft) {
    if (!agf || ag_T != T || ag_soft != T_soft) {
        if (!build_audio_graph(T, T_soft)) return false;
    }
    ggml_backend_tensor_set(a_mel_in, mel.data(), 0, sizeof(float) * mel.size());

    // per-position validity after the two stride-2 subsamples: m2[i] = valid[4i]
    // (same arithmetic as the graph: H1 = (T-1)/2+1, H2 = (H1-1)/2+1)
    const int H1l = (T - 1) / 2 + 1;
    const int H2 = (H1l - 1) / 2 + 1;
    // tensors are [T_k, T_q]: element (j, i) at j + i*H2
    std::vector<float> mask((size_t)H2 * H2, -1e9f);
    for (int i = 0; i < H2; i++) {
        const bool qi = valid[(size_t)(i * 4 < (int)valid.size() ? i * 4 : valid.size() - 1)] != 0;
        for (int j = 0; j < H2; j++) {
            const bool kj = valid[(size_t)(j * 4 < (int)valid.size() ? j * 4 : valid.size() - 1)] != 0;
            if (!qi || (kj && i - j >= 0 && i - j < a_ctx_left - 1))   // HF: dist in [0, left-1)
                mask[(size_t)j + (size_t)i * H2] = 0.0f;
        }
    }
    ggml_backend_tensor_set(a_attn_mask_in, mask.data(), 0, sizeof(float) * mask.size());
    int il_dbg = 0;
    for (int r = 0; r <= a_ctx_left - 1; r++) {
        std::vector<float> m((size_t)H2 * H2, 0.0f);
        for (int i = 0; i < H2; i++)
            for (int j = 0; j < H2; j++)
                if (i - j == (a_ctx_left - 1) - r) m[(size_t)j + (size_t)i * H2] = 1.0f;
        ggml_backend_tensor_set(a_diag[(size_t)r], m.data(), 0, sizeof(float) * m.size());
        if (getenv("EMBG2_DEBUG") && il_dbg++ < 13)
            fprintf(stderr, "[dbg] diag r=%d sum=%.0f first=%d\n", r,
                    std::accumulate(m.begin(), m.end(), 0.0f), (int)m[0]);
    }

    if (ggml_backend_graph_compute(backend, agf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "audio compute failed\n");
        return false;
    }
    if (!dump_dir.empty()) {
        static const char * const dn[] = { "a_mel", "a_conv1", "a_c2", "a_sub", "a_l0_bd", "a_l0_q", "a_l0_k", "a_l0_prob", "a_l0_ac", "a_l0_scores", "a_l0_qh", "a_l0_kh", "a_l0_attnres", "a_l0_ff1", "a_l0_v", "a_l0_merged", "a_l0_postout", "a_l0_glu", "a_l0_dwout", "a_l0_lstart", "a_l0_lend", "a_l0_lstart2",
                                           "a_l0_attn_raw", "a_l0_lconv", "a_blk0_out",
                                           "a_blk1_out", "a_soft", "a_diag_0", nullptr };
        for (int i = 0; dn[i]; i++) dump(agf, dn[i]);
    }
    const int D = (int)a_soft_out->ne[0];
    soft.resize((size_t)D * T_soft);
    // gather only the valid positions, in order
    std::vector<float> raw((size_t)D * H2);
    ggml_backend_tensor_get(a_soft_out, raw.data(), 0, sizeof(float) * raw.size());
    int u = 0;
    for (int i = 0; i < H2 && u < T_soft; i++) {
        const int src = i * 4;
        const bool ok = (src < (int)valid.size() ? valid[(size_t)src] : 0) != 0;
        if (ok) memcpy(soft.data() + (size_t)u * D, raw.data() + (size_t)i * D, sizeof(float) * D);
        if (ok) u++;
    }
    if (getenv("EMBG2_DEBUG")) {
        FILE * f = fopen("/tmp/eg2_engine_asoft.bin", "wb");
        if (f) { fwrite(soft.data(), sizeof(float), soft.size(), f); fclose(f); }
    }
    return true;
}

// ---------------------------------------------------------------- text

bool EmbedGemma2::build_text_graph(int T, int S, bool have_media) {
    if (tctx) { ggml_free(tctx); tctx = nullptr; }
    struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    tctx = ggml_init(ip);
    if (!tctx) return false;
    tgf = ggml_new_graph_custom(tctx, 32768, false);

    // CUDA fattn needs the KV length to be a multiple of its KQ stride (256)
    // for head_dim 256/512 - pad the sequence the standard way and mask the
    // pad keys; CPU flash attention is generic and keeps the exact length
    const int Tq = ggml_backend_is_cpu(backend) ? T : (T + 255) / 256 * 256;
    tg_pad = Tq - T;

    t_toks_in = ggml_new_tensor_1d(tctx, GGML_TYPE_I32, Tq);
    ggml_set_name(t_toks_in, "toks");
    t_mask_swa = ggml_new_tensor_2d(tctx, GGML_TYPE_F16, Tq, Tq);
    ggml_set_name(t_mask_swa, "mask_swa");
    t_mask_full = ggml_new_tensor_2d(tctx, GGML_TYPE_F16, Tq, Tq);
    ggml_set_name(t_mask_full, "mask_full");
    if (S > 0) {
        t_soft_pos_in = ggml_new_tensor_1d(tctx, GGML_TYPE_I32, S);
        ggml_set_name(t_soft_pos_in, "soft_pos");
        t_soft_in = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_embd, S);
        ggml_set_name(t_soft_in, "soft_in");
    }

    ggml_tensor * x = ggml_get_rows(tctx, t_embed, t_toks_in);       // [512, T] f16
    x = ggml_cast(tctx, x, GGML_TYPE_F32);
    x = ggml_scale(tctx, x, sqrtf((float)n_embd));
    if (S > 0) x = ggml_set_rows(tctx, ggml_dup(tctx, x), t_soft_in, t_soft_pos_in);
    ggml_set_name(x, "t_emb");

    // projection-only PLE: [12288, T] -> [512, 24, T] normed -> [512, T, 24]
    ggml_tensor * pl = ggml_scale(tctx, ggml_mul_mat(tctx, t_ple_model_proj, x),
                                  1.0f / sqrtf((float)n_embd));
    pl = ggml_reshape_3d(tctx, pl, ple_dim, n_layer, Tq);
    pl = ggml_mul(tctx, ggml_rms_norm(tctx, pl, eps), t_ple_norm);
    pl = ggml_cont(tctx, ggml_permute(tctx, pl, 0, 2, 1, 3));        // [512, T, 24]
    ggml_set_name(pl, "ple");

    ggml_tensor * pos_in = ggml_new_tensor_1d(tctx, GGML_TYPE_I32, Tq);
    ggml_set_name(pos_in, "pos");
    for (int il = 0; il < n_layer; il++) {
        const bool full = is_full(il);
        const int hd = full ? g_head_dim : head_dim;
        const int nkv = full ? g_kv_head : n_kv_head;
        const int nh = n_head;
        TxtLayer & L = tl[(size_t)il];

        ggml_tensor * cur = ggml_mul(tctx, ggml_rms_norm(tctx, x, eps), L.attn_norm);
        ggml_tensor * q = ggml_mul_mat(tctx, L.wq, cur);
        ggml_tensor * k = ggml_mul_mat(tctx, L.wk, cur);
        ggml_tensor * v = ggml_mul_mat(tctx, L.wv, cur);

        q = ggml_reshape_3d(tctx, q, hd, nh, Tq);
        k = ggml_reshape_3d(tctx, k, hd, nkv, Tq);
        v = ggml_reshape_3d(tctx, v, hd, nkv, Tq);
        q = ggml_mul(tctx, ggml_rms_norm(tctx, q, eps), L.q_norm);
        k = ggml_mul(tctx, ggml_rms_norm(tctx, k, eps), L.k_norm);
        v = ggml_rms_norm(tctx, v, eps);

        const float theta = full ? rope_full : rope_swa;
        q = ggml_rope_ext(tctx, q, pos_in, nullptr, hd, GGML_ROPE_TYPE_NEOX, 262144,
                          theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(tctx, k, pos_in, nullptr, hd, GGML_ROPE_TYPE_NEOX, 262144,
                          theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        if (il == 0) ggml_set_name(q, "l0_qrope");
        ggml_tensor * qf = ggml_cont(tctx, ggml_permute(tctx, q, 0, 2, 1, 3));
        ggml_tensor * kf = ggml_cont(tctx, ggml_permute(tctx, k, 0, 2, 1, 3));
        ggml_tensor * vf = ggml_cont(tctx, ggml_permute(tctx, v, 0, 2, 1, 3));
        ggml_tensor * mask = full ? t_mask_full : t_mask_swa;
        ggml_tensor * fa = ggml_flash_attn_ext(tctx, qf, kf, vf, mask, 1.0f, 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(tctx, ggml_cont(tctx, fa), nh * hd, Tq);
        attn = ggml_mul_mat(tctx, L.wo, attn);
        if (il == 0) ggml_set_name(attn, "l0_attn_proj");

        // Gemma sandwich: post_attention_layernorm BEFORE the residual add
        cur = ggml_add(tctx, x, ggml_mul(tctx, ggml_rms_norm(tctx, attn, eps), L.attn_post_norm));
        if (il == 0) ggml_set_name(cur, "blk0_attn_out");

        ggml_tensor * f = ggml_mul(tctx, ggml_rms_norm(tctx, cur, eps), L.ffn_norm);
        ggml_tensor * gate = ggml_gelu(tctx, ggml_mul_mat(tctx, L.w_gate, f));
        ggml_tensor * up = ggml_mul_mat(tctx, L.w_up, f);
        f = ggml_mul_mat(tctx, L.w_down, ggml_mul(tctx, gate, up));
        cur = ggml_add(tctx, cur, ggml_mul(tctx, ggml_rms_norm(tctx, f, eps), L.ffn_post_norm));
        if (il == 0) ggml_set_name(f, "l0_ffn_out");

        // PLE block
        {
            ggml_tensor * slice = ggml_view_2d(tctx, pl, ple_dim, Tq, pl->nb[1],
                                               (size_t)il * pl->nb[2]);
            ggml_tensor * g = ggml_gelu(tctx, ggml_mul_mat(tctx, L.ple_gate, cur));
            g = ggml_mul(tctx, g, slice);
            g = ggml_mul_mat(tctx, L.ple_proj, g);
            g = ggml_mul(tctx, ggml_rms_norm(tctx, g, eps), L.ple_post_norm);
            cur = ggml_add(tctx, cur, g);
        }

        cur = ggml_mul(tctx, cur, L.out_scale);
        if (il == 0) ggml_set_name(cur, "blk0_out");
        x = cur;
    }

    x = ggml_mul(tctx, ggml_rms_norm(tctx, x, eps), t_norm);
    ggml_tensor * proj = ggml_mul_mat(tctx, t_out_proj, x);          // [768, T]
    ggml_set_name(proj, "t_proj");

    // mean pool over the REAL tokens only (rows [0, T); pad rows excluded)
    ggml_tensor * tpr = tg_pad > 0
        ? ggml_view_2d(tctx, proj, out_dim, T, proj->nb[1], 0)
        : proj;
    ggml_tensor * tp = ggml_cont(tctx, ggml_transpose(tctx, tpr));  // [T, 768] contiguous
    ggml_tensor * pooled = ggml_scale(tctx, ggml_reshape_1d(tctx, ggml_sum_rows(tctx, tp), out_dim),
                                      1.0f / (float)T);
    ggml_tensor * nrm = ggml_sqrt(tctx, ggml_sum(tctx, ggml_mul(tctx, pooled, pooled)));
    ggml_tensor * embd = ggml_div(tctx, pooled, nrm);
    ggml_set_name(embd, "embd");
    t_embd_out = embd;
    ggml_build_forward_expand(tgf, embd);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "t_emb", "ple", "l0_qrope", "l0_attn_proj",
                                           "blk0_attn_out", "l0_ffn_out", "blk0_out",
                                           "t_proj", "embd", nullptr };
        vqa::dump::mark_sinks(tctx, tgf, dn);
        vqa::dump::finalize_sinks(tctx, tgf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, tgf)) {
        fprintf(stderr, "text graph allocation failed\n");
        return false;
    }
    tg_T = T; tg_S = S;
    return true;
}

bool EmbedGemma2::run_text(const std::vector<int> & toks, const std::vector<int> & soft_pos,
                           const std::vector<float> & soft, std::vector<float> & embd) {
    const int T = (int)toks.size();
    const int S = (int)soft_pos.size();
    if (!tgf || tg_T != T || tg_S != S) {
        if (!build_text_graph(T, S, S > 0)) return false;
    }
    // pad slots (GPU only): any valid token id - their rows are excluded from
    // pooling and their keys are masked out of every real query's softmax
    std::vector<int> toks_q = toks;
    if (tg_pad > 0) toks_q.insert(toks_q.end(), tg_pad, toks.empty() ? 0 : toks.back());
    ggml_backend_tensor_set(t_toks_in, toks_q.data(), 0, sizeof(int) * toks_q.size());
    if (S > 0) {
        ggml_backend_tensor_set(t_soft_pos_in, soft_pos.data(), 0, sizeof(int) * soft_pos.size());
        ggml_backend_tensor_set(t_soft_in, soft.data(), 0, sizeof(float) * soft.size());
    }
    {
        const int Tq = T + tg_pad;
        // positions 0..Tq-1 for the rope, plus the two masks
        ggml_tensor * pos = ggml_get_tensor(tctx, "pos");
        std::vector<int> pv((size_t)Tq);
        for (int i = 0; i < Tq; i++) pv[(size_t)i] = i;
        ggml_backend_tensor_set(pos, pv.data(), 0, sizeof(int) * pv.size());

        const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t hide = ggml_fp32_to_fp16(-INFINITY);
        // full mask: real<->real open; pad keys hidden; pad queries see only
        // themselves (an all-hidden row would NaN the softmax)
        std::vector<ggml_fp16_t> full((size_t)Tq * Tq, hide);
        for (int i = 0; i < T; i++)
            for (int j = 0; j < T; j++) full[(size_t)i * Tq + j] = keep;
        for (int i = T; i < Tq; i++) full[(size_t)i * Tq + i] = keep;
        ggml_backend_tensor_set(t_mask_full, full.data(), 0, sizeof(ggml_fp16_t) * full.size());
        // sliding window mask, same pad policy
        std::vector<ggml_fp16_t> swa((size_t)Tq * Tq, hide);
        for (int i = 0; i < T; i++)
            for (int j = std::max(0, i - window); j <= std::min(T - 1, i + window); j++)
                swa[(size_t)i * Tq + j] = keep;
        for (int i = T; i < Tq; i++) swa[(size_t)i * Tq + i] = keep;
        ggml_backend_tensor_set(t_mask_swa, swa.data(), 0, sizeof(ggml_fp16_t) * swa.size());
    }
    if (ggml_backend_graph_compute(backend, tgf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "text compute failed\n");
        return false;
    }
    if (!dump_dir.empty()) {
        static const char * const dn[] = { "t_emb", "ple", "l0_qrope", "l0_attn_proj",
                                           "blk0_attn_out", "l0_ffn_out", "blk0_out",
                                           "t_proj", "embd", nullptr };
        for (int i = 0; dn[i]; i++) dump(tgf, dn[i]);
    }
    embd.resize((size_t)out_dim);
    ggml_backend_tensor_get(t_embd_out, embd.data(), 0, sizeof(float) * embd.size());
    return true;
}

// ---------------------------------------------------------------- embed

// the sentence-transformers prompt presets (config_sentence_transformers.json)
static const char * preset_prefix(const char * preset) {
    if (!preset || !strcmp(preset, "none") || !preset[0]) return "";
    if (!strcmp(preset, "query") || !strcmp(preset, "retrieval") ||
        !strcmp(preset, "search") || !strcmp(preset, "reranking")) return "task: search result | query: ";
    if (!strcmp(preset, "document") || !strcmp(preset, "doc")) return "title: none | text: ";
    if (!strcmp(preset, "sts") || !strcmp(preset, "similarity")) return "task: sentence similarity | query: ";
    if (!strcmp(preset, "classification")) return "task: classification | query: ";
    if (!strcmp(preset, "clustering")) return "task: clustering | query: ";
    if (!strcmp(preset, "code")) return "task: code retrieval | query: ";
    if (!strcmp(preset, "qa")) return "task: question answering | query: ";
    if (!strcmp(preset, "fact")) return "task: fact checking | query: ";
    return "";
}

bool EmbedGemma2::embed(const vqa::MediaInput & media, const std::string & text,
                        const vqa::EmbedParams & ep, std::vector<float> & out) {
    if (media.empty() && !media.has_audio() && text.empty()) return false;

    // ---- vision soft tokens (images / video frames share one tower batch)
    std::vector<float> vis_soft;
    int n_soft_per_frame = 0;
    int n_frames = 0;
    if (!media.empty()) {
        n_frames = (int)media.frames.size();
        std::vector<float> all_patches;
        std::vector<int> xs, ys;
        int gw = 0, gh = 0;
        for (int fi = 0; fi < n_frames; fi++) {
            std::vector<float> patches;
            std::vector<int> fx, fy;
            int fgw = 0, fgh = 0;
            const int soft_budget = (media.kind == vqa::MediaKind::Video) ? v_max_soft_video : 0;
            if (!preprocess_image(media.frames[(size_t)fi].rgb.data(), media.width,
                                  media.height, patches, fx, fy, fgw, fgh, soft_budget)) return false;
            if (fi == 0) { gw = fgw; gh = fgh; xs = fx; ys = fy; }
            else if (fgw != gw || fgh != gh) { xs.clear(); }
            all_patches.insert(all_patches.end(), patches.begin(), patches.end());
        }
        if (xs.empty()) return false;   // mixed frame sizes unsupported
        const int P = gw * gh;
        const int n_soft = (gw / v_pool) * (gh / v_pool);
        n_soft_per_frame = n_soft;
        // one frame per graph run: HF attention never crosses frames, and the
        // batched (B>1) vision graph is not layout-safe on this ggml fork
        for (int fi = 0; fi < n_frames; fi++) {
            std::vector<float> frame_patches(all_patches.begin() + (size_t)fi * P * 3 * v_patch * v_patch,
                                             all_patches.begin() + (size_t)(fi + 1) * P * 3 * v_patch * v_patch);
            std::vector<float> frame_soft;
            if (!run_vision(frame_patches, xs, ys, P, n_soft, 1, gw, gh, frame_soft)) return false;
            vis_soft.insert(vis_soft.end(), frame_soft.begin(), frame_soft.end());
        }
        fprintf(stderr, "vision: %dx%d -> %dx%d patches, %d soft tokens x %d frame(s)\n",
                media.width, media.height, gw, gh, n_soft, n_frames);
    }

    // ---- audio soft tokens
    std::vector<float> aud_soft;
    int aud_soft_n = 0;
    if (media.has_audio()) {
        std::vector<float> mel;
        std::vector<uint8_t> valid;
        int T = 0;
        if (!compute_mel(media.audio, mel_fft, mel_frame, mel_hop, mel_bins, mel_floor,
                         mel_min_f, mel_max_f, mel_sr, mel, valid, T)) {
            fprintf(stderr, "audio too short for mel extraction\n");
            return false;
        }
        const int H1 = (T - 1) / 2 + 1;
        const int H2 = (H1 - 1) / 2 + 1;
        int n_valid = 0;
        for (int i = 0; i < H2; i++) {
            const int src = i * 4;
            if (src < (int)valid.size() && valid[(size_t)src]) n_valid++;
        }
        aud_soft_n = std::min(n_valid, 280);
        if (aud_soft_n <= 0) { fprintf(stderr, "audio has no valid frames\n"); return false; }
        if (!run_audio(mel, valid, T, aud_soft_n, aud_soft)) return false;
        fprintf(stderr, "audio: %.1fs -> %d mel frames, %d soft tokens\n",
                (double)media.audio.size() / mel_sr, T, aud_soft_n);
        if (getenv("EMBG2_DEBUG")) {
            FILE * f = fopen("/tmp/eg2_engine_mel.bin", "wb");
            if (f) { fwrite(mel.data(), sizeof(float), mel.size(), f); fclose(f); }
        }
    }

    // ---- token sequence: BOS + prompt (with placeholder blocks) + EOS
    std::string prompt = preset_prefix(ep.preset) + text;
    {
        std::string block;
        if (n_soft_per_frame > 0) {
            for (int i = 0; i < n_soft_per_frame; i++) block += "<|image|>";
            block = "<|image>" + block + "<image|>";
        }
        std::string vblock;
        if (n_soft_per_frame > 0 && media.kind == vqa::MediaKind::Video) {
            vblock = block;
            std::string all;
            for (int fi = 0; fi < n_frames; fi++) all += vblock;
            block = all;
        }
        std::string ablock;
        if (aud_soft_n > 0) {
            for (int i = 0; i < aud_soft_n; i++) ablock += "<|audio|>";
            ablock = "<|audio>" + ablock + "<audio|>";
        }
        if (!block.empty() || !ablock.empty()) {
            // markers inside the prompt expand in place; otherwise the media
            // blocks are appended after the text (single-modality default)
            auto expand = [&](std::string & s, const std::string & marker,
                              const std::string & rep) {
                size_t p = s.find(marker);
                if (p != std::string::npos) s.replace(p, marker.size(), rep);
                else s += (s.empty() ? "" : " ") + rep;
            };
            // for video, `block` holds every frame's soft block concatenated
            // (vblock is just the first frame's)
            if (media.kind == vqa::MediaKind::Video && n_soft_per_frame > 0)
                expand(prompt, "<|video|>", block);
            else if (!block.empty()) expand(prompt, "<|image|>", block);
            if (!ablock.empty()) expand(prompt, "<|audio|>", ablock);
        }
    }
    std::vector<int> toks;
    toks.push_back(id_bos);
    {
        std::vector<int> ids = tok_.encode(prompt);
        toks.insert(toks.end(), ids.begin(), ids.end());
    }
    toks.push_back(id_eos);

    // placeholder positions must match the soft-token counts exactly
    std::vector<int> soft_pos;
    const int want_img = n_soft_per_frame > 0 ? n_soft_per_frame * n_frames : 0;
    const int want_aud = aud_soft_n;
    int got_img = 0, got_aud = 0;
    for (size_t i = 0; i < toks.size(); i++) {
        if (toks[i] == id_image) { soft_pos.push_back((int)i); got_img++; }
        else if (toks[i] == id_audio) { soft_pos.push_back((int)i); got_aud++; }
    }
    if (got_img != want_img || got_aud != want_aud) {
        fprintf(stderr, "placeholder mismatch: image %d/%d audio %d/%d\n",
                got_img, want_img, got_aud, want_aud);
        return false;
    }
    if ((int)toks.size() > n_ctx) {
        fprintf(stderr, "sequence %d exceeds n_ctx %d\n", (int)toks.size(), n_ctx);
        return false;
    }

    std::vector<float> soft(vis_soft);
    soft.insert(soft.end(), aud_soft.begin(), aud_soft.end());

    std::vector<float> embd;
    if (!run_text(toks, soft_pos, soft, embd)) return false;

    // Matryoshka: slice then re-normalize
    int dim = ep.truncate_dim > 0 && ep.truncate_dim <= out_dim ? ep.truncate_dim : out_dim;
    double sum = 0.0;
    for (int i = 0; i < dim; i++) sum += (double)embd[(size_t)i] * embd[(size_t)i];
    const float inv = (float)(1.0 / std::sqrt(std::max(sum, 1e-12)));
    out.resize((size_t)dim);
    for (int i = 0; i < dim; i++) out[(size_t)i] = embd[(size_t)i] * inv;
    return true;
}

}  // namespace eg2

namespace vqa {
Model * create_embedgemma2() { return new eg2::EmbedGemma2(); }
}  // namespace vqa
