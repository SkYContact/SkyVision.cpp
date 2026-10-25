// gemma4_model.cpp - Gemma 4 12B backend (Gemma4Unified, text-first port).
//
// Text trunk (google/gemma-4-12B, 48 decoder layers):
//   hidden 3840, 16 heads, gelu(tanh) SwiGLU 15360, RMSNorm eps 1e-6, tied
//   embeddings scaled by sqrt(3840) on the token path, final logit
//   softcapping 30 (scale -> tanh -> scale). Layer pattern = 5x
//   sliding_attention + 1x full_attention repeating (dense full-attn layers
//   at 5, 11, 17, ...):
//     * sliding layers: head_dim 256, 8 KV heads, rope theta 1e4 over all 256
//       dims, causal sliding window 1024;
//     * full-attn layers: head_dim 512, 1 global KV head, "proportional" rope
//       theta 1e6 with partial_rotary_factor 0.25. Proportional rope is
//       reproduced exactly like llama.cpp's converter (conversion/gemma.py,
//       Gemma4Model::generate_extra_tensors): rope runs over the FULL 512-dim
//       head with per-pair freq factors [1.0]*64 + [1e30]*192 - this fork's
//       rope divides the angle by the factor (ops.cpp rope_fill cache:
//       `theta/ff`), so 1e30 pins the trailing 384 dims unrotated.
//   Attention extras (per llama.cpp src/models/gemma4.cpp): attn scale 1.0,
//   per-head q/k RMSNorm (weighted) on the reshaped head, V weightless
//   RMS-normed, K==V weights shared (attention_k_eq_v: no v_proj - V is the
//   k_proj output), Gemma sandwich post-norms (attn_post_norm on the wo
//   output / ffn_post_norm on the down output, both before the residual add),
//   optional per-layer out_scale (layer_scalar), optional per-layer
//   embeddings (PLE, hidden_size_per_layer_input; 12B ships 0 = absent).
//
// Multimodal (Gemma4Unified, encoder-free - verified from the 12B checkpoint):
//   vision  no tower: host resize to <=280 soft tokens (patch 48, aspect
//           preserved, bicubic), pixels x/255 (do_normalize=false!), then
//           im2col-equivalent host patch extraction (HWC within the patch,
//           matching HF patch_dense columns) -> LayerNorm eps 1e-5 (patch_ln1)
//           -> patch_dense+bias -> LayerNorm (patch_ln2) -> + pos_table[x] +
//           pos_table[y] (get_rows into the [2,1120,3840] halves) ->
//           LayerNorm (pos_norm, 1e-5) -> RMSNorm (1e-6) -> mm_proj
//           [3840,3840]. No Gemma4ClippableLinear clamps ship in this
//           checkpoint (defaults absent -> skipped; a release that ships
//           .input_*/.output_* clamp scalars would need them applied here).
//   audio   no tower: raw 16 kHz mono waveform framed into ceil(n/640) frames
//           of 640 samples (zero-padded; 40 ms/token) -> RMSNorm 1e-6 over the
//           640 dim -> mm_proj [3840,640].
//   splice  <|image>(255999) + np embedding rows + <image|(258882) /
//           <|audio>(256000) + ntok rows + <audio|(258883); the embedding rows
//           REPLACE placeholder token rows (image 258880 / audio 258881) via
//           set_rows, unscaled (sqrt(3840) applies to token-id rows only).
//   mask    use_bidirectional_attention="vision" ==
//           LLAMA_NON_CAUSAL_TYPE_SWA_ONLY in llama.cpp: the whole media chunk
//           (markers included) is decoded with causal_attn=false, which makes
//           only the SWA layers non-causal (llama-kv-cache.cpp:1780 keeps the
//           dense layers causal). Exact per-(query,key) rule, reproduced row
//           by row here (see run_text): media rows in sliding layers attend
//           keys within window 1024 in BOTH directions but never past their
//           own chunk's closing marker (llama.cpp never has those keys in the
//           cache when the chunk is decoded); media rows in full-attn layers
//           stay causal; text rows are causal + window everywhere.
//
// Engine limitation: the KV cache keeps FULL-length rows for the sliding
// layers too (no SWA ring buffer / eviction in core/engine/attention.hpp);
// the window is enforced by the attention mask. Correct, but memory is
// n_ctx-sized for all 48 layers.
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include "core/vqa.hpp"

#include "core/engine/attention.hpp"
#include "core/engine/backend.hpp"
#include "core/engine/sampler.hpp"
#include "core/engine/weight_load.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/gguf_util.hpp"
#include "core/media/resample.hpp"
#include "core/runtime/load_util.hpp"
#include "core/util/mmap.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace gemma4 {

struct TxtLayer {
    ggml_tensor *attn_norm = nullptr, *attn_post_norm = nullptr;
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    ggml_tensor *q_norm = nullptr, *k_norm = nullptr;
    ggml_tensor *ffn_norm = nullptr, *ffn_post_norm = nullptr;
    ggml_tensor *w_gate = nullptr, *w_up = nullptr, *w_down = nullptr;
    ggml_tensor *ple_gate = nullptr, *ple_proj = nullptr, *ple_post_norm = nullptr;
    ggml_tensor *out_scale = nullptr;
    ggml_tensor *rope_freqs = nullptr;   // full-attn layers only
};

// per-layer-dimension KV cache: sliding layers are [8*256, n_ctx], full-attn
// layers [1*512, n_ctx] - vqa::KvCache assumes one uniform shape, so this
// backend rolls its own (same F16 layout and clear/free discipline)
struct KvCacheMixed {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;

    bool alloc(ggml_backend_t backend, const std::vector<int> & kv_dim, int n_ctx) {
        const int n_layer = (int)kv_dim.size();
        struct ggml_init_params ip = { (size_t)(n_layer * 2 + 8) * ggml_tensor_overhead(), NULL, true };
        ctx = ggml_init(ip);
        if (!ctx) return false;
        k.resize(n_layer);
        v.resize(n_layer);
        double bytes = 0;
        for (int i = 0; i < n_layer; i++) {
            k[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_dim[i], n_ctx);
            v[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_dim[i], n_ctx);
            bytes += 2.0 * (double)kv_dim[i] * n_ctx * sizeof(ggml_fp16_t);
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fprintf(stderr, "gemma4: KV cache allocation failed (%.2f GB)\n", bytes / 1e9);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        fprintf(stderr, "gemma4: kv cache %.2f GB (n_ctx %d, mixed per-layer dims)\n",
                bytes / 1e9, n_ctx);
        return true;
    }

    void clear() { if (buf) ggml_backend_buffer_clear(buf, 0); }

    void free() {
        if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
    }
};

struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *pos_in = nullptr;
    ggml_tensor *soft_pos_in = nullptr, *soft_in = nullptr;   // media splice
    ggml_tensor *mask_full = nullptr, *mask_swa = nullptr;
    ggml_tensor *logits = nullptr;
    // masks live in a private pre-allocated buffer OUTSIDE the gallocr pool:
    // this fork's gallocr frees-and-reuses input buffers after their last
    // consumer, which for F16 attention masks read across 48 layers proved
    // unsafe (post-compute readback shows the mask buffer clobbered; with
    // media the model went "blind"). A backend-owned buffer cannot be reused.
    ggml_context * mctx = nullptr;
    ggml_backend_buffer_t mbuf = nullptr;
};

// soft-token splice into one ask() call: embedding rows + their token
// positions, plus the per-token non-causal span bookkeeping (media rows and
// their chunk end, see the mask notes in the file header)
struct MediaSplice {
    struct Block { int bo, eo, placeholder, n_rows; };
    std::vector<Block> blocks;
    std::vector<int> soft_pos;                 // BATCH-RELATIVE token index of each row
    std::vector<float> soft;                   // n_embd-wide rows, concatenated
    std::vector<uint8_t> noncausal;            // per token: decoded non-causal
    std::vector<int> chunk_end;                // per token: closing-marker index
    bool empty() const { return blocks.empty(); }
};

class Gemma4 : public vqa::Model {
public:
    ~Gemma4() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "gemma4"; }

    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp,
             std::string & out_text) override;

    // Gemma4 official turn template (google-gemma-4-31B-it.jinja, thinking ON
    // as llama.cpp applies it): a system turn <|turn>system\n<|think|>\n
    // <turn|>\n opens the prompt, each turn is <|turn>user/model\n ...
    // <turn|>\n, and the generation prompt is bare <|turn>model\n - the model
    // emits its own thought channel <|channel>thought ... <channel|> before
    // the answer (stripped in generate()). All markers are encoded PIECEWISE
    // (see the use_added_tokens note in load()); never as the added ids.
    std::string format_history_turn(const std::string & user,
                                    const std::string & assistant) const override {
        return "<|turn>user\n" + user + "<turn|>\n"
             "<|turn>model\n" + assistant + "<turn|>\n";
    }

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool build_text_graph(TextGraph & g, int T, int S, const MediaSplice * mm);
    bool run_text(TextGraph & g, const std::vector<int> & toks,
                  std::vector<float> & logits_out, const MediaSplice * mm = nullptr);
    bool generate(const std::vector<int> & ids, const vqa::GenParams & gp,
                  std::string & out_text, const MediaSplice * mm = nullptr);

    // unified projections (encoder-free multimodal)
    bool preprocess_image(const uint8_t * rgb, int w0, int h0,
                          std::vector<float> & patches, std::vector<int> & px,
                          std::vector<int> & py, int & gw, int & gh,
                          int budget_override = -1);
    bool preprocess_video_hf(const uint8_t * rgb, int w0, int h0,
                             std::vector<float> & patches, std::vector<int> & px,
                             std::vector<int> & py, int & gw, int & gh, int budget);
    bool build_vision_graph(int np);
    bool run_vision(const std::vector<float> & patches, const std::vector<int> & px,
                    const std::vector<int> & py, int np, std::vector<float> & soft,
                    const std::vector<float> * valid = nullptr);
    bool build_audio_graph(int ntok);
    bool run_audio(const std::vector<float> & frames, int ntok,
                   std::vector<float> & soft);
    bool encode_media(const vqa::MediaInput & media, MediaSplice & mm);

    int n_layer_head(int il) const { (void)il; return n_head; }
    int hd_of(int il) const { return is_full(il) ? hd_full : hd_swa; }
    int nkv_of(int il) const { return is_full(il) ? n_kv_head_full : n_kv_head; }
    bool is_full(int il) const {
        if (il < (int)swa_pattern.size()) return swa_pattern[(size_t)il] == 0;
        return (il % 6) == 5;
    }

    // pre-load defaults so the CLI video gate lets frame dirs through; the
    // authoritative per-file caps are re-derived in load() (has_media_weights)
    vqa::Capabilities caps_ = []{ vqa::Capabilities c; c.image = c.video = c.audio = true; return c; }();
    vqa::Tokenizer tok_;

    // text hyper (defaults = gemma-4-12B config.json; GGUF keys override)
    int n_layer = 48, n_embd = 3840, n_ff = 15360, n_vocab = 262144;
    int n_head = 16, n_kv_head = 8;             // sliding layers
    int n_kv_head_full = 1;                     // full-attn layers (global KV head)
    int hd_swa = 256, hd_full = 512;
    int window = 1024;
    float eps = 1e-6f;
    float rope_theta_full = 1e6f, rope_theta_swa = 1e4f;
    int rope_dim_full = 512, rope_dim_swa = 256;
    float partial_rotary_full = 0.25f;
    float logit_softcapping = 30.0f;
    int ple_dim = 0;
    int max_pos = 262144;
    std::vector<int> swa_pattern;               // 1 = sliding_attention layer

    int id_bos = 2, id_eos = 1, id_eot = 106;   // <eos>, <turn|>
    // unified multimodal tokens (12B config.json)
    int id_boi = 255999, id_eoi = 258882, id_image = 258880;
    int id_boa = 256000, id_eoa = 258883, id_audio = 258881, id_video = 258884;

    // vision projection hyper (encoder-free)
    int v_patch = 48, v_max_soft = 280, v_min_soft = 70, v_pos_table = 1120;
    float v_eps = 1e-6f;
    // audio projection hyper
    int a_frame = 640, a_sr = 16000;
    bool has_media_weights = false;

    int n_ctx = 8192;
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    vqa::MMap mm;
    vqa::AlignedBuf host_f32;
    ggml_gallocr_t allocr = nullptr;

    ggml_tensor *t_embed = nullptr, *t_norm = nullptr;
    ggml_tensor *t_ple_table = nullptr, *t_ple_model_proj = nullptr, *t_ple_norm = nullptr;
    std::vector<TxtLayer> tl;

    // unified projection weights
    ggml_tensor *v_dense = nullptr, *v_dense_b = nullptr;
    ggml_tensor *v_ln1_w = nullptr, *v_ln1_b = nullptr;
    ggml_tensor *v_ln2_w = nullptr, *v_ln2_b = nullptr;
    ggml_tensor *v_posn_w = nullptr, *v_posn_b = nullptr;
    ggml_tensor *v_postbl = nullptr, *v_mm_proj = nullptr, *a_mm_proj = nullptr;

    // vision / audio graphs (rebuilt when the token count changes)
    ggml_context *vctx = nullptr;
    ggml_cgraph *vgf = nullptr;
    ggml_tensor *v_patches_in = nullptr, *v_px_in = nullptr, *v_py_in = nullptr;
    ggml_tensor *v_soft_out = nullptr;
    int vg_np = -1;
    ggml_context *actx = nullptr;
    ggml_cgraph *agf = nullptr;
    ggml_tensor *a_frames_in = nullptr, *a_soft_out = nullptr;
    ggml_tensor *v_valid_in = nullptr;
    int ag_nt = -1;
    ggml_tensor *rope_freqs_shared = nullptr;   // host fallback if no per-layer tensor
    ggml_context * rf_ctx = nullptr;            // owns rope_freqs_shared
    ggml_backend_buffer_t rf_buf = nullptr;

    KvCacheMixed kv;
    int n_past = 0;
    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

Gemma4::~Gemma4() {
    if (vctx) ggml_free(vctx);
    if (actx) ggml_free(actx);
    kv.free();
    if (rf_buf) ggml_backend_buffer_free(rf_buf);
    if (rf_ctx) ggml_free(rf_ctx);
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (allocr) ggml_gallocr_free(allocr);
    if (backend) ggml_backend_free(backend);
}

bool Gemma4::init_backend() {
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

// ---------------------------------------------------------------- load

bool Gemma4::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f GB\n", mm.size / 1e9);
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer     = vqa::gguf::u32(gu, "gemma4.block_count", n_layer);
    n_embd      = vqa::gguf::u32(gu, "gemma4.embedding_length", n_embd);
    n_ff        = vqa::gguf::u32(gu, "gemma4.feed_forward_length", n_ff);
    n_head      = vqa::gguf::u32(gu, "gemma4.attention.head_count", n_head);
    n_kv_head   = vqa::gguf::u32(gu, "gemma4.attention.head_count_kv", n_kv_head);
    n_kv_head_full = vqa::gguf::u32(gu, "gemma4.attention.head_count_kv_full", n_kv_head_full);
    hd_full     = vqa::gguf::u32(gu, "gemma4.attention.key_length", hd_full);
    hd_swa      = vqa::gguf::u32(gu, "gemma4.attention.key_length_swa", hd_swa);
    window      = vqa::gguf::u32(gu, "gemma4.attention.sliding_window", window);
    eps         = vqa::gguf::f32(gu, "gemma4.attention.layer_norm_rms_epsilon", eps);
    logit_softcapping = vqa::gguf::f32(gu, "gemma4.attention.final_logit_softcapping", logit_softcapping);
    rope_theta_full = vqa::gguf::f32(gu, "gemma4.rope.freq_base", rope_theta_full);
    rope_theta_swa  = vqa::gguf::f32(gu, "gemma4.rope.freq_base_swa", rope_theta_swa);
    rope_dim_full   = vqa::gguf::u32(gu, "gemma4.rope.dimension_count", hd_full);
    rope_dim_swa    = vqa::gguf::u32(gu, "gemma4.rope.dimension_count_swa", hd_swa);
    partial_rotary_full = vqa::gguf::f32(gu, "gemma4.rope.partial_rotary_factor_full", partial_rotary_full);
    ple_dim     = vqa::gguf::u32(gu, "gemma4.embedding_length_per_layer_input", ple_dim);
    n_vocab     = vqa::gguf::u32(gu, "gemma4.vocab_size", n_vocab);
    max_pos     = vqa::gguf::u32(gu, "gemma4.rope.context_length", max_pos);
    swa_pattern = vqa::gguf::arr_u32(gu, "gemma4.attention.sliding_window_pattern", {});
    id_bos = vqa::gguf::u32(gu, "gemma4.bos_token_id", id_bos);
    id_eos = vqa::gguf::u32(gu, "gemma4.eos_token_id", id_eos);
    id_eot = vqa::gguf::u32(gu, "gemma4.end_of_turn_token_id", id_eot);
    id_boi = vqa::gguf::u32(gu, "gemma4.boi_token_id", id_boi);
    id_eoi = vqa::gguf::u32(gu, "gemma4.eoi_token_id", id_eoi);
    id_image = vqa::gguf::u32(gu, "gemma4.image_token_id", id_image);
    id_boa = vqa::gguf::u32(gu, "gemma4.boa_token_id", id_boa);
    id_eoa = vqa::gguf::u32(gu, "gemma4.eoa_token_id", id_eoa);
    id_audio = vqa::gguf::u32(gu, "gemma4.audio_token_id", id_audio);
    id_video = vqa::gguf::u32(gu, "gemma4.video_token_id", id_video);
    v_patch = vqa::gguf::u32(gu, "gemma4.vision.patch_size", v_patch);
    v_max_soft = vqa::gguf::u32(gu, "gemma4.vision.max_soft_tokens", v_max_soft);
    v_min_soft = vqa::gguf::u32(gu, "gemma4.vision.min_soft_tokens", v_min_soft);
    v_pos_table = vqa::gguf::u32(gu, "gemma4.vision.pos_table_size", v_pos_table);
    v_eps = vqa::gguf::f32(gu, "gemma4.vision.rms_norm_eps", v_eps);
    a_frame = vqa::gguf::u32(gu, "gemma4.audio.frame_samples", a_frame);
    a_sr = vqa::gguf::u32(gu, "gemma4.audio.sample_rate", a_sr);

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);
    vqa::weights::promote_1d_to_f32(wctx, host_f32);

    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) fprintf(stderr, "missing tensor: %s\n", n);
        return t;
    };
    t_embed = gt("t.embed.weight");      // tied lm_head (tie_word_embeddings)
    t_norm  = gt("t.norm.weight");
    if (ple_dim > 0) {
        t_ple_table      = gt("t.ple_table.weight");
        t_ple_model_proj = gt("t.ple_model_proj.weight");
        t_ple_norm       = gt("t.ple_norm.weight");
    }
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
        if (!is_full(i)) L.rope_freqs = nullptr;
        else {
            L.rope_freqs = g("rope_freqs.weight");
            if (!L.rope_freqs) L.rope_freqs = ggml_get_tensor(wctx, "t.rope_freqs.weight");
        }
        // wv is optional (attention_k_eq_v); everything else is required
        if (!L.attn_norm || !L.wq || !L.wk || !L.wo || !L.q_norm || !L.k_norm ||
            !L.w_gate || !L.w_up || !L.w_down) {
            fprintf(stderr, "load: layer %d has null tensors\n", i);
            return false;
        }
        if (is_full(i) && !L.rope_freqs && !rope_freqs_shared) {
            // no GGUF rope_freqs: build the llama.cpp constant on the host
            // ([1.0]*n_rot_pairs + [1e30]*rest, see file header)
            const int pairs = hd_full / 2;
            const int rot_pairs = (int)(hd_full * partial_rotary_full) / 2;
            std::vector<float> ff((size_t)pairs, 1e30f);
            for (int j = 0; j < rot_pairs; j++) ff[(size_t)j] = 1.0f;
            struct ggml_init_params ip = { 1ull << 16, NULL, true };
            rf_ctx = ggml_init(ip);
            if (!rf_ctx) return false;
            rope_freqs_shared = ggml_new_tensor_1d(rf_ctx, GGML_TYPE_F32, pairs);
            rf_buf = ggml_backend_alloc_ctx_tensors(rf_ctx, backend);
            if (!rf_buf) return false;
            ggml_backend_tensor_set(rope_freqs_shared, ff.data(), 0, sizeof(float) * ff.size());
            L.rope_freqs = rope_freqs_shared;
            fprintf(stderr, "gemma4: built host rope_freqs (%d rotated pairs of %d)\n",
                    rot_pairs, pairs);
        }
        if (is_full(i) && !L.rope_freqs) L.rope_freqs = rope_freqs_shared;
    }

    if (!t_embed || !t_norm) return false;

    // unified projections (optional: a text-only GGUF keeps the text surface)
    {
        auto gv = [&](const char * n) { return ggml_get_tensor(wctx, n); };
        v_dense = gv("v.patch_dense.weight");   v_dense_b = gv("v.patch_dense.bias");
        v_ln1_w = gv("v.ln1.weight");           v_ln1_b = gv("v.ln1.bias");
        v_ln2_w = gv("v.ln2.weight");           v_ln2_b = gv("v.ln2.bias");
        v_posn_w = gv("v.pos_norm.weight");     v_posn_b = gv("v.pos_norm.bias");
        v_postbl = gv("v.pos_table.weight");
        v_mm_proj = gv("v.mm_proj.weight");
        a_mm_proj = gv("a.mm_proj.weight");
        has_media_weights = v_dense && v_ln1_w && v_ln2_w && v_posn_w &&
                            v_postbl && v_mm_proj && a_mm_proj;
        if (!has_media_weights)
            fprintf(stderr, "gemma4: no vision/audio projection tensors in GGUF "
                            "(text-only model file)\n");
    }
    if ((int)t_embed->ne[0] != n_embd) {
        fprintf(stderr, "gemma4: t.embed row length %lld != n_embd %d (tied head expected)\n",
                (long long)t_embed->ne[0], n_embd);
        return false;
    }
    if (ple_dim > 0 && (!t_ple_table || !t_ple_model_proj || !t_ple_norm)) return false;

    {
        vqa::weights::Bound bound = vqa::weights::bind_to_backend(
            wctx, backend, mm, { { host_f32.data(), host_f32.size() } });
        if (!bound.ok) return false;
        wbuf = bound.main_buf;
        if (!ggml_backend_is_cpu(backend)) mm.release();
    }
    return true;
}

bool Gemma4::load(const std::string & gguf_path, const std::string & tokenizer_path,
                  const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "GEMMA4", n_threads, use_gpu, dump_dir);
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::GemmaSP)) return false;
    // Official tokenization: special matching ON - the HF tokenizer resolves
    // <|turn>/<|think|>/<|channel>/media markers to their real added ids
    // (105/98/100/255999/...), verified bitwise against the official
    // Gemma4UnifiedProcessor (transformers 5.21.0.dev0). The old piecewise
    // path was a llama.cpp quirk, not the official format.

    {
        vqa::assign_token_ids(tok_, {
            { "<end_of_turn>", &id_eot },
        });
        id_eos = vqa::pick_token(tok_, "<eos>", id_eos);
        // TODO(gemma4-parity): confirm bos/eos/<end_of_turn> ids for the 262144
        // vocab from the checkpoint's generation_config.json + a real dump run
        // (defaults follow Gemma-2/3 conventions; the 12B ids are unverified).
    }

    // KV cache: per-layer (nkv * head_dim) mixed dims
    std::vector<int> kv_dim((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) kv_dim[(size_t)i] = nkv_of(i) * hd_of(i);
    if (!kv.alloc(backend, kv_dim, n_ctx)) return false;

    // encoder-free projections: image + audio caps when the GGUF ships them
    caps_ = {};
    if (has_media_weights) {
        caps_.image = true;
        caps_.video = true;   // per-frame image chunks (mtmd) / HF framing
        caps_.audio = true;   // raw 16 kHz waveform, 640 samples / soft token
        // video deferred: <|video|> id 258884 exists but the unified video
        // framing (per-frame soft budgets, ordering) is unverified - reject
        // with a clear message in ask() instead of guessing
    }
    int n_full = 0;
    for (int i = 0; i < n_layer; i++) n_full += is_full(i);
    fprintf(stderr, "gemma4: ready (%d layers: %d sliding hd=%d x%d kv + %d full hd=%d x1 kv, "
                    "d=%d ff=%d, window %d, softcap %.0f, ctx %d)\n",
            n_layer, n_layer - n_full, hd_swa, n_kv_head, n_full, hd_full,
            n_embd, n_ff, window, logit_softcapping, n_ctx);
    return true;
}

// ---------------------------------------------------------------- text

bool Gemma4::build_text_graph(TextGraph & g, int T, int S, const MediaSplice * mm) {
    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 32768, false);

    g.tok_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    g.soft_pos_in = g.soft_in = nullptr;   // stale pointers into the freed ctx
    // mark every graph input: without ggml_set_input this fork's gallocr may
    // hand the input tensor's buffer to an intermediate node MID-COMPUTE,
    // silently clobbering the uploaded masks (repro: mask_swa readback all
    // -inf after a "2 + 2 =" prefill -> the model echoed the prompt)
    ggml_set_input(g.tok_in);
    if (mm && !mm->empty()) {
        // soft-token splice: embedding rows REPLACE placeholder token rows
        g.soft_pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, (int64_t)mm->soft_pos.size());
        ggml_set_name(g.soft_pos_in, "soft_pos");
        g.soft_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd,
                                       (int64_t)mm->soft_pos.size());
        ggml_set_name(g.soft_in, "soft_in");
        ggml_set_input(g.soft_pos_in);
        ggml_set_input(g.soft_in);
    }
    // rope positions == absolute KV slots for plain text (gemma4 text rope is
    // 1D; the multimodal position layout arrives with the towers)
    g.pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.pos_in, "pos_in");
    // mask width matches the 64-padded KV view (FATTN_KQ_STRIDE): the CUDA
    // fattn kernels need mask->nb[1] % 16 == 0 AND width >= K->ne[1]; the
    // upload code fills the full width with -inf first, so pad columns are
    // inert.
    const int Sp = getenv("GEMMA4_NOPAD") ? S : GGML_PAD(S, 256);
    g.mask_full = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, Sp, T);
    ggml_set_name(g.mask_full, "mask_full");
    g.mask_swa = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, Sp, T);
    ggml_set_name(g.mask_swa, "mask_swa");
    ggml_set_input(g.mask_full);
    ggml_set_input(g.mask_swa);
    ggml_set_input(g.pos_in);

        // token embeddings, scaled by sqrt(n_embd) - Gemma normalizer on the
        // token path only (verified against unnormalized llama.cpp oracle
        // dumps: inp_scaled == E[ids] * sqrt(3840) exactly)
        ggml_tensor * x = ggml_get_rows(g.ctx, t_embed, g.tok_in);        // [D, T]
        x = ggml_cast(g.ctx, x, GGML_TYPE_F32);
        x = ggml_scale(g.ctx, x, sqrtf((float)n_embd));
    if (g.soft_in && !getenv("GEMMA4_PARITY_NO_SPLICE")) {
        // media rows overwrite the (scaled) placeholder embeddings with the
        // RAW projection output - the sqrt(n_embd) Gemma normalizer applies
        // to token-id rows only (llama.cpp build_inp_embd scale_rows)
        if (getenv("GEMMA4_DEBUG"))
            fprintf(stderr, "[dbg] set_rows: x [%lld,%lld] soft_in [%lld,%lld] pos [%lld]\n",
                    (long long)x->ne[0], (long long)x->ne[1],
                    (long long)g.soft_in->ne[0], (long long)g.soft_in->ne[1],
                    (long long)g.soft_pos_in->ne[0]);
        x = ggml_set_rows(g.ctx, ggml_dup(g.ctx, x), g.soft_in, g.soft_pos_in);
    }
    ggml_set_name(x, "emb");
    if (getenv("GEMMA4_PROBE_EMB")) {
        g.logits = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, x), n_embd, T);
        ggml_build_forward_expand(g.gf, g.logits);
        return true;
    }

    // per-layer embeddings (PLE) - only when the checkpoint ships a table.
    // 12B has hidden_size_per_layer_input = 0, so this path is dormant; it
    // mirrors llama.cpp gemma4 project_per_layer_inputs exactly.
    // TODO(gemma4-parity): no 12B checkpoint exercises this - verify against
    // llama.cpp on a PLE-bearing release (E2B/E4B) before relying on it.
    ggml_tensor * pl = nullptr;
    if (ple_dim > 0 && t_ple_table) {
        ggml_tensor * tab = ggml_get_rows(g.ctx, t_ple_table, g.tok_in);  // [ple*n_layer, T]
        tab = ggml_scale(g.ctx, ggml_cast(g.ctx, tab, GGML_TYPE_F32), sqrtf((float)ple_dim));
        ggml_tensor * proj = ggml_mul_mat(g.ctx, t_ple_model_proj, x);   // [ple*n_layer, T]
        proj = ggml_scale(g.ctx, proj, 1.0f / sqrtf((float)n_embd));
        proj = ggml_reshape_3d(g.ctx, proj, ple_dim, n_layer, T);
        proj = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, proj, eps), t_ple_norm);
        proj = ggml_add(g.ctx, proj, ggml_reshape_3d(g.ctx, tab, ple_dim, n_layer, T));
        proj = ggml_scale(g.ctx, proj, (float)M_SQRT1_2);
        pl = ggml_cont(g.ctx, ggml_permute(g.ctx, proj, 0, 2, 1, 3));   // [ple, T, n_layer]
        ggml_set_name(pl, "ple");
    }

    for (int il = 0; il < n_layer; il++) {
        const bool full = is_full(il);
        const int hd = full ? hd_full : hd_swa;
        const int nh = n_layer_head(il);
        const int nkv = nkv_of(il);
        TxtLayer & L = tl[(size_t)il];

        ggml_tensor * cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, x, eps), L.attn_norm);

        ggml_tensor * q = ggml_mul_mat(g.ctx, L.wq, cur);               // [nh*hd, T]
        ggml_tensor * k = ggml_mul_mat(g.ctx, L.wk, cur);               // [nkv*hd, T]
        // attention_k_eq_v: no v_proj - V is the k_proj output pre-norm
        ggml_tensor * v = L.wv ? ggml_mul_mat(g.ctx, L.wv, cur) : k;

        q = ggml_reshape_3d(g.ctx, q, hd, nh, T);
        k = ggml_reshape_3d(g.ctx, k, hd, nkv, T);
        v = ggml_reshape_3d(g.ctx, v, hd, nkv, T);
        q = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, q, eps), L.q_norm);
        k = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, k, eps), L.k_norm);
        v = ggml_rms_norm(g.ctx, v, eps);                               // weightless

        // rope: sliding = NEOX over hd dims, theta 1e4; full = NEOX over the
        // whole 512-dim head, theta 1e6, with the rope_freqs factors doing
        // the partial rotary 0.25 (see file header). NOTE: llama-model.cpp
        // reports LLAMA_ROPE_TYPE_NORM for GEMMA4, but empirically the gemma4
        // graph's rotations match NEOX half-split pairing exactly (verified
        // against oracle attention dumps, corr 1.0 vs 0.93 for NORM).
        ggml_tensor * ff = full ? L.rope_freqs : nullptr;
        const float theta = full ? rope_theta_full : rope_theta_swa;
        const int n_dims = full ? rope_dim_full : rope_dim_swa;
        q = ggml_rope_ext(g.ctx, q, g.pos_in, ff, n_dims, GGML_ROPE_TYPE_NEOX,
                          max_pos, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(g.ctx, k, g.pos_in, ff, n_dims, GGML_ROPE_TYPE_NEOX,
                          max_pos, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (il == 0) ggml_set_name(q, "t_l0_q_rope");
        if (il == 0 && getenv("GEMMA4_PROBE_ATTN") && L.wo) {
            // probe AFTER the full layer-0 block instead (wo/sandwich/ffn)
        }
        if (il == 0 && getenv("GEMMA4_PROBE_Q") && !getenv("GEMMA4_PROBE_ATTN")) {
            // debug: surface layer-0 roped Q as the graph result (trusted read)
            g.logits = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, q), nh * hd, T);
            ggml_build_forward_expand(g.gf, g.logits);
            return true;
        }

        // KV write (roped K, normed V) + attention over [0, S)
        ggml_tensor * k2 = ggml_reshape_2d(g.ctx, k, nkv * hd, T);
        ggml_tensor * v2 = ggml_reshape_2d(g.ctx, v, nkv * hd, T);
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.k[(size_t)il], k2, g.pos_in));
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.v[(size_t)il], v2, g.pos_in));

        // pad the KV view to a multiple of FATTN_KQ_STRIDE: the CUDA fattn
        // kernels for hd=512 require K->ne[1] % 64 == 0 (gqa_opt_applies).
        // Padding columns are masked -inf in both masks, so they contribute
        // nothing; beyond n_ctx there is nothing left to pad.
        const int Sp = std::min(n_ctx, getenv("GEMMA4_NOPAD") ? S : GGML_PAD(S, 256));
        ggml_tensor * kf = ggml_view_3d(g.ctx, kv.k[(size_t)il], hd, Sp, nkv,
                                        kv.k[(size_t)il]->nb[1], hd * sizeof(ggml_fp16_t), 0);
        ggml_tensor * vf = ggml_view_3d(g.ctx, kv.v[(size_t)il], hd, Sp, nkv,
                                        kv.v[(size_t)il]->nb[1], hd * sizeof(ggml_fp16_t), 0);
        ggml_tensor * qf = ggml_view_3d(g.ctx, q, hd, T, nh, q->nb[2], q->nb[1], 0);
        // gemma4: self.scaling = 1.0 (no pre-attention 1/sqrt(d) - the q/k
        // norms carry the scale, llama.cpp f_attention_scale = 1.0)
        ggml_tensor * attn = nullptr;
        if (!getenv("GEMMA4_NO_FA")) {
            ggml_tensor * fa = ggml_flash_attn_ext(g.ctx, qf, kf, vf,
                                                   full ? g.mask_full : g.mask_swa,
                                                   1.0f, 0.0f, 0.0f);
            if (il == 0) ggml_set_name(fa, "t_l0_fa");
            if (il == 0 && getenv("GEMMA4_PROBE_FA")) {
                g.logits = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, fa), nh * hd, T);
                ggml_build_forward_expand(g.gf, g.logits);
                return true;
            }
            attn = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, fa), nh * hd, T);
        } else {
            // manual path (llama.cpp non-FA branch): kq [S,T,nh] -> softmax
            // with F32 mask -> v^T kq -> [hd,T,nh] -> [nh*hd, T]
            ggml_tensor * kq = ggml_mul_mat(g.ctx, kf, qf);          // [S, T, nh]
            kq = ggml_soft_max_ext(g.ctx, kq, full ? g.mask_full : g.mask_swa,
                                   1.0f, 0.0f);
            ggml_tensor * vt = ggml_cont(g.ctx, ggml_transpose(g.ctx, vf));  // [S, hd, nh]
            ggml_tensor * kqv = ggml_mul_mat(g.ctx, vt, kq);         // [hd, T, nh]
            attn = ggml_cont_2d(g.ctx, ggml_permute(g.ctx, kqv, 0, 2, 1, 3),
                                nh * hd, T);
        }
        attn = ggml_mul_mat(g.ctx, L.wo, attn);
        if (il == 0) ggml_set_name(attn, "t_l0_attn");

        // Gemma sandwich: post-norm on the sublayer output, then residual add
        ggml_tensor * a = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, attn, eps), L.attn_post_norm);
        ggml_tensor * attn_out = ggml_add(g.ctx, x, a);
        if (il == 0) ggml_set_name(attn_out, "t_l0_attn_out");

        ggml_tensor * f = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, attn_out, eps), L.ffn_norm);
        ggml_tensor * gate = ggml_gelu(g.ctx, ggml_mul_mat(g.ctx, L.w_gate, f));  // tanh gelu
        ggml_tensor * up = ggml_mul_mat(g.ctx, L.w_up, f);
        f = ggml_mul_mat(g.ctx, L.w_down, ggml_mul(g.ctx, gate, up));
        cur = ggml_add(g.ctx, attn_out,
                       ggml_mul(g.ctx, ggml_rms_norm(g.ctx, f, eps), L.ffn_post_norm));
        if (il == 0) ggml_set_name(cur, "t_l0_out");

        // per-layer embedding block
        if (pl && L.ple_gate && L.ple_proj) {
            ggml_tensor * slice = ggml_view_2d(g.ctx, pl, ple_dim, T, pl->nb[1],
                                               (size_t)il * pl->nb[2]);
            ggml_tensor * pg = ggml_gelu(g.ctx, ggml_mul_mat(g.ctx, L.ple_gate, cur));
            pg = ggml_mul(g.ctx, pg, slice);
            pg = ggml_mul_mat(g.ctx, L.ple_proj, pg);
            pg = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, pg, eps), L.ple_post_norm);
            cur = ggml_add(g.ctx, cur, pg);
        }

        if (L.out_scale) cur = ggml_mul(g.ctx, cur, L.out_scale);
        x = cur;
        if (il == 0 && getenv("GEMMA4_PROBE_ATTN")) {
            g.logits = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, ggml_cast(g.ctx, x, GGML_TYPE_F32)), n_embd, T);
            ggml_build_forward_expand(g.gf, g.logits);
            return true;
        }
        if (getenv("GEMMA4_DEBUG")) { char nm[24]; snprintf(nm, sizeof(nm), "t_l%d_out", il); ggml_set_name(x, nm); }
    }

    x = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, x, eps), t_norm);
    ggml_set_name(x, "t_final_norm");

    ggml_tensor * last = ggml_view_2d(g.ctx, x, n_embd, 1, x->nb[1],
                                      (size_t)(T - 1) * x->nb[1]);
    ggml_tensor * logits = ggml_mul_mat(g.ctx, t_embed, last);         // tied head
    if (logit_softcapping > 0.0f) {
        logits = ggml_scale(g.ctx, logits, 1.0f / logit_softcapping);
        logits = ggml_tanh(g.ctx, logits);
        logits = ggml_scale(g.ctx, logits, logit_softcapping);
    }
    ggml_set_name(logits, "logits");
    g.logits = logits;
    ggml_build_forward_expand(g.gf, logits);

    if (!dump_dir.empty()) {
        static const char * const dn[] = {
            "emb", "ple", "t_l0_q_rope", "t_l0_attn", "t_l0_attn_out", "t_l0_out",
            "t_final_norm", "logits", nullptr,
        };
        vqa::dump::mark_sinks(g.ctx, g.gf, dn);
        if (getenv("GEMMA4_DEBUG")) {
            for (int i = 1; i < n_layer; i++) {
                char nm0[16], * nm = nm0;
                const char * one[2] = { nullptr, nullptr };
                snprintf(nm0, sizeof(nm0), "t_l%d_out", i);
                one[0] = nm0;
                vqa::dump::mark_sinks(g.ctx, g.gf, one);
            }
        }
        vqa::dump::finalize_sinks(g.ctx, g.gf, backend);
    }
    return true;
}

bool Gemma4::run_text(TextGraph & g, const std::vector<int> & toks,
                      std::vector<float> & logits_out, const MediaSplice * mm) {
    const int T = (int)toks.size();
    const int S = n_past + T;
    if (S > n_ctx) { fprintf(stderr, "context overflow (%d > %d)\n", S, n_ctx); return false; }

    if (!build_text_graph(g, T, S, mm)) return false;
    if (!ggml_gallocr_alloc_graph(allocr, g.gf)) {
        fprintf(stderr, "gemma4: text graph allocation failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }
    if (getenv("GEMMA4_PROBE_ATTN") || getenv("GEMMA4_PROBE_Q")) fprintf(stderr, "[dbg] set tok_in\n");
    ggml_backend_tensor_set(g.tok_in, toks.data(), 0, sizeof(int) * T);
    if (g.soft_in && mm) {
        ggml_backend_tensor_set(g.soft_pos_in, mm->soft_pos.data(), 0,
                                sizeof(int) * mm->soft_pos.size());
        ggml_backend_tensor_set(g.soft_in, mm->soft.data(), 0,
                                sizeof(float) * mm->soft.size());
    }
    std::vector<int> pos((size_t)T);
    for (int i = 0; i < T; i++) pos[(size_t)i] = n_past + i;
    if (getenv("GEMMA4_PROBE_ATTN") || getenv("GEMMA4_PROBE_Q")) fprintf(stderr, "[dbg] set pos_in\n");
    if (!getenv("GEMMA4_PROBE_EMB"))
        ggml_backend_tensor_set(g.pos_in, pos.data(), 0, sizeof(int) * T);
    // full-attn layers stay causal for EVERY row - llama.cpp keeps the dense
    // cache causal under LLAMA_NON_CAUSAL_TYPE_SWA_ONLY
    // (llama-kv-cache.cpp: causal_attn = swa_type == NONE)
    if (getenv("GEMMA4_PARITY_NOMASK")) {
        std::vector<ggml_fp16_t> z((size_t)g.mask_full->ne[0] * T, ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(g.mask_full, z.data(), 0, sizeof(ggml_fp16_t) * z.size());
        ggml_backend_tensor_set(g.mask_swa, z.data(), 0, sizeof(ggml_fp16_t) * z.size());
    } else
    if (!getenv("GEMMA4_PROBE_Q") && !getenv("GEMMA4_PROBE_ATTN") && !getenv("GEMMA4_PROBE_FA") && !getenv("GEMMA4_PROBE_EMB"))
        vqa::set_causal_mask(g.mask_full, n_past, T);
    // sliding layers: text rows = causal window (allowed s in
    // [qt-window+1, qt], matching llama.cpp is_masked_swa STANDARD:
    // masked iff p1 - p0 >= window). Media rows (decoded in their own
    // non-causal chunk in llama.cpp) additionally see keys AFTER them, but
    // only within the window and never past their chunk's closing marker
    // (those keys are not in the cache yet when llama.cpp decodes the chunk)
    {
        const int Sw = (int)g.mask_swa->ne[0];
        std::vector<ggml_fp16_t> mv((size_t)Sw * T, ggml_fp32_to_fp16(-INFINITY));
        for (int t = 0; t < T; t++) {
            const int qt = n_past + t;
            const int lo = std::max(0, qt - window + 1);
            int hi = std::min(Sw - 1, qt);
            if (mm && t < (int)mm->noncausal.size() && mm->noncausal[(size_t)t] &&
                !getenv("GEMMA4_PARITY_NO_BIDIR"))
                hi = std::min(Sw - 1, std::min(qt + window - 1,
                                               n_past + mm->chunk_end[(size_t)t]));
            for (int s = lo; s <= hi; s++)
                mv[(size_t)t * Sw + s] = ggml_fp32_to_fp16(0.0f);
        }
        if (!getenv("GEMMA4_PROBE_Q") && !getenv("GEMMA4_PROBE_FA") && !getenv("GEMMA4_PROBE_EMB"))
            ggml_backend_tensor_set(g.mask_swa, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());
        if (getenv("GEMMA4_DEBUG")) {
            const int r = T > 13 ? 13 : T - 1;   // row 13 only exists in prefill
            fprintf(stderr, "[dbg] mask_swa uploaded row%d: ", r);
            for (int s = 0; s < Sw; s++)
                fprintf(stderr, "%d", mv[(size_t)r * Sw + s] == ggml_fp32_to_fp16(0.0f) ? 1 : 0);
            fprintf(stderr, " | pre data=%p\n", (void *)g.mask_swa->data);
        }
    }
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "gemma4: text compute failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }
    if (!dump_dir.empty()) {
        static char dn[56][16] = { "emb", "t_l0_q_rope", "t_final_norm", "logits",
                                   "t_l0_attn", "t_l0_attn_out", "t_l0_out", "t_l0_fa" };
        static char * dnp[56];
        static bool dn_init = false;
        if (!dn_init) {
            dn_init = true;
            int u = 0;
            for (int i = 0; i < 8; i++) dnp[u++] = dn[i];
            if (getenv("GEMMA4_DEBUG")) {
                for (int i = 0; i < n_layer; i++) {
                    snprintf(dn[9 + i], 16, "t_l%d_out", i);
                    dnp[u++] = dn[9 + i];
                }
            }
            dnp[u] = nullptr;
        }
        static const char * const dn_dec[] = {
            "t_l0_q_rope", "t_final_norm", "logits", nullptr,
        };
        // decode-step graphs also name "emb" ([3840,1]) - keep the prefill's
        const char * const * dl = getenv("GEMMA4_DEBUG") ? (const char * const *)dnp
                                   : (mm ? (const char * const *)dnp : dn_dec);
        for (int i = 0; dl[i]; i++) vqa::dump::write(g.gf, dump_dir.c_str(), dl[i]);
    }
    if (getenv("GEMMA4_DEBUG")) {
        // read back the masks post-compute: if gallocr reused their buffers,
        // the contents differ from what was uploaded before compute
        const int Sw = (int)g.mask_swa->ne[0];
        std::vector<ggml_fp16_t> mv((size_t)Sw * g.mask_swa->ne[1]);
        ggml_backend_tensor_get(g.mask_swa, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());
        FILE * f = fopen("gemma4_dbg_mask_swa.bin", "wb");
        if (f) { fwrite(mv.data(), 2, mv.size(), f); fclose(f); }
        fprintf(stderr, "[dbg] mask_swa readback row%d: ", g.mask_swa->ne[1] > 13 ? 13 : (int)g.mask_swa->ne[1] - 1);
        {
            const int r = g.mask_swa->ne[1] > 13 ? 13 : (int)g.mask_swa->ne[1] - 1;
            for (int s = 0; s < Sw; s++)
                fprintf(stderr, "%d", ggml_fp16_to_fp32(mv[(size_t)r * Sw + s]) == 0.0f ? 1 : 0);
        }
        fprintf(stderr, " | post data=%p\n", (void *)g.mask_swa->data);
    }
    if (getenv("GEMMA4_DEBUG")) {
        // dump the layer-0 / layer-5 KV cache rows (graph outputs -> reliable)
        for (int il : { 0, 5 }) {
            const int rows = nkv_of(il) * hd_of(il);
            std::vector<ggml_fp16_t> kvh((size_t)rows * S);
            ggml_backend_tensor_get(kv.v[(size_t)il], kvh.data(), 0,
                                    sizeof(ggml_fp16_t) * kvh.size());
            std::vector<float> kvf(kvh.size());
            for (size_t i = 0; i < kvh.size(); i++) kvf[i] = ggml_fp16_to_fp32(kvh[i]);
            char nm[48];
            snprintf(nm, sizeof(nm), "gemma4_dbg_v_l%d.bin", il);
            FILE * f = fopen(nm, "wb");
            if (f) { fwrite(kvf.data(), 4, kvf.size(), f); fclose(f); }
            snprintf(nm, sizeof(nm), "gemma4_dbg_v_l%d.shape", il);
            f = fopen(nm, "w");
            if (f) { fprintf(f, "%d %d\n", rows, S); fclose(f); }
        }
    }
    if (getenv("GEMMA4_PROBE_Q") || getenv("GEMMA4_PROBE_ATTN") || getenv("GEMMA4_PROBE_FA") || getenv("GEMMA4_PROBE_EMB")) {
        const int64_t nel = ggml_nelements(g.logits);
        std::vector<float> pv((size_t)nel);
        ggml_backend_tensor_get(g.logits, pv.data(), 0, sizeof(float) * nel);
        FILE * pf = fopen(getenv("GEMMA4_PROBE_ATTN") ? "gemma4_probe_attn.bin" : (getenv("GEMMA4_PROBE_FA") ? "gemma4_probe_fa.bin" : (getenv("GEMMA4_PROBE_EMB") ? "gemma4_probe_emb.bin" : "gemma4_probe_q.bin")), "wb");
        if (pf) { fwrite(pv.data(), 4, pv.size(), pf); fclose(pf); }
        fprintf(stderr, "[probe_q] wrote %lld floats\n", (long long)nel);
    }
    if (!getenv("GEMMA4_PROBE_Q") && !getenv("GEMMA4_PROBE_ATTN") && !getenv("GEMMA4_PROBE_FA") && !getenv("GEMMA4_PROBE_EMB")) {
    logits_out.resize((size_t)n_vocab);
    ggml_backend_tensor_get(g.logits, logits_out.data(), 0, sizeof(float) * n_vocab);
    }
    n_past = S;
    ggml_free(g.ctx);
    g.ctx = nullptr;
    return true;
}

// ---------------------------------------------------------------- media

// resize inside the soft-token budget (aspect preserved, sides multiple of
// the patch size) and extract patches in HF HWC order (dy, dx, c) with pixel
// values x/255 (processor do_normalize=false - NOT the 2x-1 of gemma4v /
// embeddinggemma-2)
bool Gemma4::preprocess_image(const uint8_t * rgb, int w0, int h0,
                              std::vector<float> & patches, std::vector<int> & px,
                              std::vector<int> & py, int & gw, int & gh,
                              int budget_override) {
    const int P = v_patch;
    // sizing is an exact port of llama.cpp mtmd: clip_hparams
    // set_limit_image_tokens(70, 1120) on the 48 px patch (patch_area = 48*48)
    // + mtmd_image_preprocessor_dyn_size::get_target_size ->
    // img_tool::calc_size_preserved_ratio(align=48, min/max px, round=std::round).
    // (The HF processor's num_soft_tokens=280 cap is NOT what llama.cpp applies;
    // honoring 280 here produced a 16x16 grid for 400x400 inputs where the
    // oracle computes 9x9, breaking image parity.)
    const int tmin = budget_override > 0 ? budget_override : (getenv("GEMMA4_BUDGET") ? 0 : 70);
    const int tmax = budget_override > 0 ? budget_override : (getenv("GEMMA4_BUDGET") ? atoi(getenv("GEMMA4_BUDGET")) : 1120);
    const int min_px = tmin * P * P, max_px = tmax * P * P;
    auto round_by = [P](float x) { return (int)std::round(x / P) * P; };
    auto ceil_by  = [P](float x) { return (int)std::ceil(x / P) * P; };
    auto floor_by = [P](float x) { return (int)std::floor(x / P) * P; };
    int tw = std::max(P, round_by((float)w0));
    int th = std::max(P, round_by((float)h0));
    if ((long long)th * tw > max_px) {
        const double beta = std::sqrt((double)h0 * w0 / max_px);
        th = std::max(P, floor_by((float)(h0 / beta)));
        tw = std::max(P, floor_by((float)(w0 / beta)));
    } else if ((long long)th * tw < min_px) {
        const double beta = std::sqrt((double)min_px / ((double)h0 * w0));
        th = ceil_by((float)(h0 * beta));
        tw = ceil_by((float)(w0 * beta));
    }

    std::vector<float> resized;
    vqa::img::resize_antialias(rgb, w0, h0, tw, th, resized, 3);   // bicubic
    for (auto & v : resized) v = std::min(1.0f, std::max(0.0f, v / 255.0f));

    gw = tw / P;
    gh = th / P;
    const int P3 = 3 * P * P;
    if (gw * gh > v_pos_table) {
        fprintf(stderr, "gemma4: image needs %d patches > pos table %d\n", gw * gh, v_pos_table);
        return false;
    }
    patches.assign((size_t)P3 * gw * gh, 0.0f);
    px.resize((size_t)gw * gh);
    py.resize((size_t)gw * gh);
    int u = 0;
    for (int cy = 0; cy < gh; cy++)
        for (int cx = 0; cx < gw; cx++, u++) {
            float * row = patches.data() + (size_t)u * P3;
            int o = 0;
            for (int dy = 0; dy < P; dy++) {
                const float * s = resized.data() + ((size_t)(cy * P + dy) * tw + cx * P) * 3;
                for (int dx = 0; dx < P; dx++)
                    for (int c = 0; c < 3; c++)
                        row[o++] = s[(size_t)dx * 3 + c];
            }
            px[(size_t)u] = cx;
            py[(size_t)u] = cy;
        }
    return true;
}

// Gemma4UnifiedVisionEmbedder graph: patches [3P*P, np] (HWC columns, HF
// order - the weight is NOT permuted to CHW like llama.cpp's, which uses
// ggml im2col) -> LN(1e-5) -> dense+bias -> LN(1e-5) -> +pos[x]+pos[y] ->
// pos_norm LN(1e-5) -> RMSNorm(1e-6) -> mm_proj. No Gemma4ClippableLinear
// clamps exist in this checkpoint (a release shipping .input_/.output_*
// clamp scalars would need clamps around the two linears).
bool Gemma4::build_vision_graph(int np) {
    if (vctx) { ggml_free(vctx); vctx = nullptr; }
    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    vctx = ggml_init(ip);
    if (!vctx) return false;
    vgf = ggml_new_graph_custom(vctx, 4096, false);
    const int P3 = 3 * v_patch * v_patch;
    const float ln_eps = 1e-5f;   // pytorch LayerNorm default

    v_patches_in = ggml_new_tensor_2d(vctx, GGML_TYPE_F32, P3, np);
    ggml_set_name(v_patches_in, "v_patches");
    ggml_set_input(v_patches_in);
    v_px_in = ggml_new_tensor_1d(vctx, GGML_TYPE_I32, np);
    ggml_set_name(v_px_in, "v_px");
    ggml_set_input(v_px_in);
    v_py_in = ggml_new_tensor_1d(vctx, GGML_TYPE_I32, np);
    ggml_set_name(v_py_in, "v_py");
    ggml_set_input(v_py_in);
    // HF video pads short frames to max_soft_tokens with zero patches and
    // (-1,-1) positions: pos contribution is masked to zero there
    // (Gemma4UnifiedVisionEmbedder.forward: clamp(min=0) * valid)
    v_valid_in = ggml_new_tensor_1d(vctx, GGML_TYPE_F32, np);
    ggml_set_name(v_valid_in, "v_valid");
    ggml_set_input(v_valid_in);

    auto ln = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
        x = ggml_norm(vctx, x, ln_eps);
        x = ggml_mul(vctx, x, w);
        return ggml_add(vctx, x, b);
    };

    ggml_tensor * x = ln(v_patches_in, v_ln1_w, v_ln1_b);
    x = ggml_mul_mat(vctx, v_dense, x);
    x = ggml_add(vctx, x, v_dense_b);
    x = ln(x, v_ln2_w, v_ln2_b);
    ggml_set_name(x, "v_after_ln2");

    // pos table [2, 1120, 3840]: two contiguous row blocks (x then y)
    {
        const int64_t pos_size = v_postbl->ne[1];
        const size_t nb1 = ggml_row_size(v_postbl->type, v_postbl->ne[0]);
        ggml_tensor * tbl_x = ggml_view_2d(vctx, v_postbl, v_postbl->ne[0],
                                           pos_size, nb1, 0);
        ggml_tensor * tbl_y = ggml_view_2d(vctx, v_postbl, v_postbl->ne[0],
                                           pos_size, nb1, pos_size * nb1);
        ggml_tensor * pos = ggml_add(vctx, ggml_get_rows(vctx, tbl_x, v_px_in),
                                      ggml_get_rows(vctx, tbl_y, v_py_in));
        pos = ggml_mul(vctx, pos, ggml_reshape_2d(vctx, v_valid_in, 1, np));
        x = ggml_add(vctx, x, pos);
    }
    x = ln(x, v_posn_w, v_posn_b);
    x = ggml_rms_norm(vctx, x, v_eps);
    x = ggml_mul_mat(vctx, v_mm_proj, x);
    ggml_set_name(x, "v_soft");
    v_soft_out = x;
    ggml_build_forward_expand(vgf, x);
    if (!dump_dir.empty()) {
        static const char * const dn[] = { "v_patches", "v_after_ln2", "v_soft", nullptr };
        vqa::dump::mark_sinks(vctx, vgf, dn);
        vqa::dump::finalize_sinks(vctx, vgf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, vgf)) {
        fprintf(stderr, "gemma4: vision graph allocation failed\n");
        return false;
    }
    vg_np = np;
    return true;
}

bool Gemma4::run_vision(const std::vector<float> & patches, const std::vector<int> & px,
                        const std::vector<int> & py, int np, std::vector<float> & soft,
                        const std::vector<float> * valid) {
    if (!vgf || vg_np != np) {
        if (!build_vision_graph(np)) return false;
    }
    ggml_backend_tensor_set(v_patches_in, patches.data(), 0, sizeof(float) * patches.size());
    // HF video padding rows carry (-1,-1): clamp to 0 for the pos-table
    // gather (the row's pos contribution is killed by the valid mask)
    std::vector<int> pxc = px, pyc = py;
    for (auto & v : pxc) v = std::max(0, v);
    for (auto & v : pyc) v = std::max(0, v);
    ggml_backend_tensor_set(v_px_in, pxc.data(), 0, sizeof(int) * np);
    ggml_backend_tensor_set(v_py_in, pyc.data(), 0, sizeof(int) * np);
    if (valid) ggml_backend_tensor_set(v_valid_in, valid->data(), 0, sizeof(float) * np);
    else {
        std::vector<float> ones((size_t)np, 1.0f);
        ggml_backend_tensor_set(v_valid_in, ones.data(), 0, sizeof(float) * np);
    }
    if (ggml_backend_graph_compute(backend, vgf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "gemma4: vision compute failed\n");
        return false;
    }
    if (!dump_dir.empty()) {
        vqa::dump::write(vgf, dump_dir.c_str(), "v_after_ln2");
        vqa::dump::write(vgf, dump_dir.c_str(), "v_soft");
    }
    soft.resize((size_t)n_embd * np);
    ggml_backend_tensor_get(v_soft_out, soft.data(), 0, sizeof(float) * soft.size());
    if (getenv("GEMMA4_DEBUG")) {
        FILE * f = fopen("gemma4_dbg_patches.bin", "wb");
        if (f) { fwrite(patches.data(), 4, patches.size(), f); fclose(f); }
        f = fopen("gemma4_dbg_px.bin", "wb");
        if (f) { fwrite(px.data(), 4, px.size(), f); fwrite(py.data(), 4, py.size(), f); fclose(f); }
        f = fopen("gemma4_dbg_vsoft.bin", "wb");
        if (f) { fwrite(soft.data(), 4, soft.size(), f); fclose(f); }
    }
    return true;
}

// audio: raw 640-sample frames (zero-padded tail) -> RMSNorm(1e-6) over the
// 640 dim -> mm_proj [3840, 640]; input layout [640, ntok] = frame-major
bool Gemma4::build_audio_graph(int ntok) {
    if (actx) { ggml_free(actx); actx = nullptr; }
    struct ggml_init_params ip = { 16ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    actx = ggml_init(ip);
    if (!actx) return false;
    agf = ggml_new_graph_custom(actx, 1024, false);

    a_frames_in = ggml_new_tensor_2d(actx, GGML_TYPE_F32, a_frame, ntok);
    ggml_set_name(a_frames_in, "a_frames");
    ggml_set_input(a_frames_in);
    ggml_tensor * x = ggml_rms_norm(actx, a_frames_in, v_eps);
    x = ggml_mul_mat(actx, a_mm_proj, x);
    ggml_set_name(x, "a_soft");
    a_soft_out = x;
    ggml_build_forward_expand(agf, x);
    if (!ggml_gallocr_alloc_graph(allocr, agf)) {
        fprintf(stderr, "gemma4: audio graph allocation failed\n");
        return false;
    }
    ag_nt = ntok;
    return true;
}

bool Gemma4::run_audio(const std::vector<float> & frames, int ntok,
                       std::vector<float> & soft) {
    if (!agf || ag_nt != ntok) {
        if (!build_audio_graph(ntok)) return false;
    }
    ggml_backend_tensor_set(a_frames_in, frames.data(), 0, sizeof(float) * frames.size());
    if (ggml_backend_graph_compute(backend, agf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "gemma4: audio compute failed\n");
        return false;
    }
    soft.resize((size_t)n_embd * ntok);
    ggml_backend_tensor_get(a_soft_out, soft.data(), 0, sizeof(float) * soft.size());
    if (const char * dp = getenv("GEMMA4_DUMP_SOFT")) {
        FILE * f = fopen(dp, "wb");
        if (f) { fwrite(soft.data(), sizeof(float), soft.size(), f); fclose(f); }
    }
    return true;
}

// HF Gemma4UnifiedVideoProcessor sizing (get_aspect_ratio_preserving_size):
// largest floor-to-(pooling*patch) dims whose area fits max_soft*pooling^2
// teacher patches; no min-upscale (short frames pad instead).
bool Gemma4::preprocess_video_hf(const uint8_t * rgb, int w0, int h0,
                                 std::vector<float> & patches, std::vector<int> & px,
                                 std::vector<int> & py, int & gw, int & gh, int budget) {
    const int P = v_patch, K = 3;
    const double max_px_area = (double)budget * K * K * 16 * 16;   // teacher-patch area cap
    const double factor = std::sqrt(max_px_area / ((double)w0 * h0));
    const int side = K * 16;
    auto floor_by = [side](float x) { return (int)std::floor(x / side) * side; };
    int tw = floor_by((float)(factor * w0));
    int th = floor_by((float)(factor * h0));
    const int max_side = (budget / (K * K)) * side;
    if (th == 0 && tw == 0) {
        fprintf(stderr, "gemma4: hf-video resize to 0x0\n");
        return false;
    }
    if (th == 0) { th = side; tw = std::min((int)std::floor((double)w0 / h0) * side, max_side); }
    else if (tw == 0) { tw = side; th = std::min((int)std::floor((double)h0 / w0) * side, max_side); }
    std::vector<float> resized;
    vqa::img::resize_antialias(rgb, w0, h0, tw, th, resized, 3);   // bicubic
    // torchvision resizes uint8 tensors and ROUNDS BACK to uint8 before the
    // 1/255 rescale - mirror that quantization for bit-parity
    for (auto & v : resized) v = std::min(255.0f, std::max(0.0f, std::round(v))) / 255.0f;

    gw = tw / P;
    gh = th / P;
    const int P3 = 3 * P * P;
    patches.assign((size_t)P3 * gw * gh, 0.0f);
    px.assign((size_t)gw * gh, 0);
    py.assign((size_t)gw * gh, 0);
    int u = 0;
    for (int cy = 0; cy < gh; cy++)
        for (int cx = 0; cx < gw; cx++, u++) {
            float * row = patches.data() + (size_t)u * P3;
            int o = 0;
            for (int dy = 0; dy < P; dy++) {
                const float * s = resized.data() + ((size_t)(cy * P + dy) * tw + cx * P) * 3;
                for (int dx = 0; dx < P; dx++)
                    for (int c = 0; c < 3; c++)
                        row[o++] = s[(size_t)dx * 3 + c];
            }
            px[(size_t)u] = cx;
            py[(size_t)u] = cy;
        }
    return true;
}

// run the unified projections over the media and collect the splice blocks
bool Gemma4::encode_media(const vqa::MediaInput & media, MediaSplice & mm) {
    if (!media.empty() && media.kind == vqa::MediaKind::Video) {
        // HF official video framing, a port of Gemma4UnifiedVideoProcessor
        // (verified against transformers 5.21.0.dev0, see docs/dev-notes):
        // per frame - resize floor-to-48 under a 70*9 teacher-patch area cap
        // (bicubic antialias), 48 px patchify, then PAD to exactly 70 patches
        // (zero patch, (-1,-1) position, pos contribution masked in the graph).
        // Token layout (replace_video_token): "mm:ss <|image> <|video|>*n
        // <image|>" blocks joined by spaces, markers as REAL added ids.
        const int budget = 70;
        std::vector<float> allsoft, allvalid;
        for (size_t fi = 0; fi < media.frames.size(); fi++) {
            std::vector<float> patches, soft;
            std::vector<int> px, py;
            int gw, gh;
            if (!preprocess_video_hf(media.frames[fi].rgb.data(), media.width, media.height,
                                     patches, px, py, gw, gh, budget)) return false;
            const int n = gw * gh;   // HF pads to 70 only for batching; the
            // model strips (-1,-1) rows and scatters VALID embeddings onto the
            // n placeholder tokens - so we skip padding entirely
            if (const char * dp = getenv("GEMMA4_DUMP_PATCHES")) {
                FILE * fp = fopen(dp, "wb");
                if (fp) { fwrite(patches.data(), sizeof(float), patches.size(), fp); fclose(fp); }
            }
            if (!run_vision(patches, px, py, n, soft)) return false;
            if (getenv("GEMMA4_ZERO_SOFT")) std::fill(soft.begin(), soft.end(), 0.0f);
            mm.soft.insert(mm.soft.end(), soft.begin(), soft.end());
            fprintf(stderr, "gemma4: hf-video frame %zu/%zu %dx%d -> %dx%d patches (cap %d)\n",
                    fi + 1, media.frames.size(), media.width, media.height, gw, gh, budget);
            mm.blocks.push_back({ id_video, id_eoi, id_video, n });
            mm.soft_pos.resize(mm.soft_pos.size() + (size_t)n);
        }
    }
    if (!media.empty() && media.kind != vqa::MediaKind::Video) {
        std::vector<float> patches, soft;
        std::vector<int> px, py;
        int gw, gh;
        // official Gemma4UnifiedImageProcessor sizing: same floor-to-48
        // aspect algorithm as the video path, max_soft_tokens=280 (the old
        // mtmd 70..1120 round-based sizing was a llama.cpp deviation)
        if (!preprocess_video_hf(media.frames[0].rgb.data(), media.width, media.height,
                                 patches, px, py, gw, gh, v_max_soft)) return false;
        const int np = gw * gh;
        if (!run_vision(patches, px, py, np, soft)) return false;
        int bo = id_boi, eo = id_eoi, ph = id_image;
        if (getenv("GEMMA4_ABLATION_IMG_AS_AUDIO")) { bo = id_boa; eo = id_eoa; ph = id_audio; }
        if (getenv("GEMMA4_ZERO_SOFT")) std::fill(soft.begin(), soft.end(), 0.0f);
        mm.blocks.push_back({ bo, eo, ph, np });
        mm.soft.insert(mm.soft.end(), soft.begin(), soft.end());
        mm.soft_pos.resize(mm.soft_pos.size() + (size_t)np);
        fprintf(stderr, "gemma4: image %dx%d -> %dx%d patches (%d soft tokens)\n",
                media.width, media.height, gw, gh, np);
    }
    if (media.has_audio()) {
        // raw 16 kHz mono frames; the engine's load_audio already resamples
        if (media.sample_rate != a_sr) {
            fprintf(stderr, "gemma4: audio sample rate %d != %d unsupported\n",
                    media.sample_rate, a_sr);
            return false;
        }
        const size_t n = media.audio.size();
        const int ntok = (int)((n + a_frame - 1) / a_frame);
        std::vector<float> frames((size_t)a_frame * ntok, 0.0f);
        std::copy(media.audio.begin(), media.audio.begin() + (std::min(n, frames.size())),
                  frames.begin());
        std::vector<float> soft;
        if (!run_audio(frames, ntok, soft)) return false;
        if (getenv("GEMMA4_ZERO_SOFT")) std::fill(soft.begin(), soft.end(), 0.0f);
        mm.blocks.push_back({ id_boa, id_eoa, id_audio, ntok });
        mm.soft.insert(mm.soft.end(), soft.begin(), soft.end());
        mm.soft_pos.resize(mm.soft_pos.size() + (size_t)ntok);
        fprintf(stderr, "gemma4: audio %.2f s -> %d soft tokens (%d samples/token)\n",
                (float)n / a_sr, ntok, a_frame);
    }
    return true;
}

// ---------------------------------------------------------------- ask

bool Gemma4::generate(const std::vector<int> & ids, const vqa::GenParams & gp,
                      std::string & out_text, const MediaSplice * mm) {
    int max_new = gp.max_tokens;
    if (n_past + (int)ids.size() + max_new > n_ctx) {
        max_new = n_ctx - n_past - (int)ids.size();
        fprintf(stderr, "warning: prompt %d + max_new %d > n_ctx %d, clamping max_new to %d\n",
                n_past + (int)ids.size(), gp.max_tokens, n_ctx, max_new);
        if (max_new <= 0) return false;
    }
    n_past = 0;
    kv.clear();

    auto t0 = std::chrono::steady_clock::now();
    TextGraph g;
    std::vector<float> logits;
    if (getenv("GEMMA4_PREFILL1")) {
        // debug: prefill one token per graph (isolates T>1 batch bugs)
        for (size_t i = 0; i < ids.size(); i++) {
            std::vector<int> one = { ids[i] };
            MediaSplice mm1;
            if (mm && i < mm->noncausal.size() && mm->noncausal[i]) {
                mm1.noncausal = { 1 };
                mm1.chunk_end = { mm->chunk_end[i] - (int)i };
            }
            for (size_t r = 0; mm && r < mm->soft_pos.size(); r++) {
                if (mm->soft_pos[r] == (int)i) {
                    mm1.soft_pos.push_back(0);
                    mm1.soft.resize((size_t)n_embd);
                    memcpy(mm1.soft.data(), mm->soft.data() + r * n_embd,
                           sizeof(float) * n_embd);
                    mm1.blocks.push_back({ 0, 0, 0, 1 });
                    break;
                }
            }
            if (!run_text(g, one, logits, mm1.empty() ? nullptr : &mm1)) return false;
        }
    } else if (!run_text(g, ids, logits, mm)) return false;
    auto t1 = std::chrono::steady_clock::now();

    out_text.clear();
    // generation_config suppress_tokens: <audio|>/<image|> must never be
    // sampled (the model emits them pathologically when confused)
    for (int id : { 258883, 258882 }) {
        if (id < (int)logits.size()) logits[(size_t)id] = -INFINITY;
    }
    if (getenv("GEMMA4_FIRSTTOK")) {
        // top-5 prefill logits for oracle comparison
        std::vector<std::pair<float,int>> v;
        for (int i = 0; i < (int)logits.size(); i++) v.push_back({logits[(size_t)i], i});
        std::partial_sort(v.begin(), v.begin()+5, v.end(), std::greater<>());
        fprintf(stderr, "FIRSTTOK:");
        for (int k = 0; k < 5; k++) fprintf(stderr, " %d(%.3f)", v[(size_t)k].second, v[(size_t)k].first);
        fprintf(stderr, "\n");
    }
    if (getenv("GEMMA4_FIRSTTOK")) {
        // top-5 prefill logits for oracle comparison
        std::vector<std::pair<float,int>> v;
        for (int i = 0; i < (int)logits.size(); i++) v.push_back({logits[(size_t)i], i});
        std::partial_sort(v.begin(), v.begin()+5, v.end(), std::greater<>());
        fprintf(stderr, "FIRSTTOK:");
        for (int k = 0; k < 5; k++) fprintf(stderr, " %d(%.3f)", v[(size_t)k].second, v[(size_t)k].first);
        fprintf(stderr, "\n");
    }
    if (getenv("GEMMA4_FIRSTTOK")) {
        // top-5 prefill logits for oracle comparison
        std::vector<std::pair<float,int>> v;
        for (int i = 0; i < (int)logits.size(); i++) v.push_back({logits[(size_t)i], i});
        std::partial_sort(v.begin(), v.begin()+5, v.end(), std::greater<>());
        fprintf(stderr, "FIRSTTOK:");
        for (int k = 0; k < 5; k++) fprintf(stderr, " %d(%.3f)", v[(size_t)k].second, v[(size_t)k].first);
        fprintf(stderr, "\n");
    }
    int next = vqa::sample_logits(logits, gp, rng_state);
    int n_dec = 0;
    std::vector<int> step(1);
    for (int s = 0; s < max_new; s++) {
        if (next == id_eos || next == id_eot) break;
        if (getenv("GEMMA4_DEBUG")) fprintf(stderr, "[dbg] tok %d\n", next);
        std::string piece = tok_.decode({ next });
        out_text += piece;
        if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
        // turn end: the model usually emits the single <turn|> token (106),
        // but may also start the NEXT turn piecewise ("<|turn>") - stop on both
        if (out_text.find("<|im_end|>") != std::string::npos) break;
        if (auto p = out_text.find("<|turn"); p != std::string::npos) {
            out_text.resize(p);
            break;
        }
        if (auto p2 = out_text.find("<turn|>"); p2 != std::string::npos) {
            out_text.resize(p2);
            break;
        }
        n_dec++;
        if (s + 1 >= max_new) break;
        step[0] = next;
        if (!run_text(g, step, logits)) break;
        for (int id : { 258883, 258882 }) {   // suppress every step, not just prefill
            if (id < (int)logits.size()) logits[(size_t)id] = -INFINITY;
        }
        next = vqa::sample_logits(logits, gp, rng_state);
    }
    auto t2 = std::chrono::steady_clock::now();
    const double ms_pre = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_dec = std::chrono::duration<double, std::milli>(t2 - t1).count();
    fprintf(stderr, "[perf] prefill %d tok in %.0f ms (%.1f tok/s) | decode %d tok in %.0f ms (%.2f tok/s)\n",
            (int)ids.size(), ms_pre, ms_pre > 0 ? ids.size() * 1000.0 / ms_pre : 0.0,
            n_dec, ms_dec, ms_dec > 0 ? n_dec * 1000.0 / ms_dec : 0.0);
    // A model turn may open with a thought channel (<|channel>thought\n ...
    // <channel|>); strip it (and any stray channel markers) from the answer.
    if (getenv("GEMMA4_RAW_OUT")) {
        fprintf(stderr, "[raw] %s\n", out_text.c_str());
    } else {
        // trim at the ChatML turn end marker (and drop anything after it)
        if (auto eot = out_text.find("<|im_end|>"); eot != std::string::npos)
            out_text.resize(eot);
        auto pos = out_text.find("<|channel>thought");
        if (pos != std::string::npos) {
            auto end = out_text.find("<channel|>", pos);
            out_text = end == std::string::npos ? out_text.substr(0, pos)
                                                : out_text.substr(0, pos) + out_text.substr(end + 10);
        }
        for (const char * m : { "<|channel>", "<channel|>" }) {
            const size_t ml = strlen(m);
            for (auto p = out_text.find(m); p != std::string::npos; p = out_text.find(m))
                out_text.erase(p, ml);
        }
    }
    return true;
}

bool Gemma4::ask(const vqa::MediaInput & media, const std::string & prompt,
                 const std::string & /*mode*/, const vqa::GenParams & gp,
                 std::string & out_text) {
    if (!has_media_weights && (!media.empty() || media.has_audio())) {
        fprintf(stderr, "gemma4: this GGUF carries no vision/audio projection "
                        "tensors (text-only model file)\n");
        return false;
    }
    // video = per-frame image chunks (llama.cpp mtmd generic path), handled in
    // encode_media; no <|video|> framing upstream.
    // Gemma4 turn template: ChatML form (matches the llama.cpp oracle's
    // actual prompt ids, see format_history_turn above).
    MediaSplice mm;
    if (!encode_media(media, mm)) return false;

    std::vector<int> ids;
    if (!getenv("GEMMA4_PARITY_NOBOS")) ids.push_back(id_bos);
    if (const char * ri = getenv("GEMMA4_RAWIDS")) {
        // parity path: feed exact token ids (comma separated, BOS NOT added)
        std::vector<int> ids2;
        for (const char * p = ri; *p;) {
            char * e = nullptr;
            long v = strtol(p, &e, 10);
            if (e == p) break;
            ids2.push_back((int)v);
            p = (*e == ',') ? e + 1 : e;
        }
        if (getenv("GEMMA4_DUMP_IDS")) {
            fprintf(stderr, "prompt ids:");
            for (int id : ids2) fprintf(stderr, " %d", id);
            fprintf(stderr, "\n");
        }
        return generate(ids2, gp, out_text, nullptr);
    }
    if (getenv("GEMMA4_RAW")) {
        // parity path: BOS + raw prompt, no turn template (mirrors llama-cli
        // -p "..." --no-jinja)
        std::vector<int> enc = tok_.encode(prompt);
        ids.insert(ids.end(), enc.begin(), enc.end());
        if (getenv("GEMMA4_DUMP_IDS")) {
            fprintf(stderr, "prompt ids:");
            for (int id : ids) fprintf(stderr, " %d", id);
            fprintf(stderr, "\n");
        }
        return generate(ids, gp, out_text, nullptr);
    }
    // thinking-on layout, exactly as llama.cpp renders it (verified id dump):
    // empty system turn with <|think|>, then the user turn
    // "<turn|>" is a single merged vocab piece (id 106) in the oracle's
    // encoding; the piece encoder splits it into <,turn,|> - splice 106 in
    // manually
    {
        const char * sys = getenv("GEMMA4_SYS");
        std::string sblock = sys ? std::string(sys) : std::string();
        std::vector<int> enc = tok_.encode(
            "<|turn>system\n" + sblock + (sys ? "<turn|>\n" : "<|think|>\n<turn|>\n"));
        ids.insert(ids.end(), enc.begin(), enc.end());
    }
    {
        std::vector<int> enc = tok_.encode("<|turn>user\n");
        ids.insert(ids.end(), enc.begin(), enc.end());
    }
    int hf_frame = 0;
    const std::vector<double> * hf_timestamps = &media.timestamps;
    std::vector<const MediaSplice::Block *> audio_blocks;
    bool audio_first = false;
    int row = 0;
        // emit one media block; official layout (Gemma4UnifiedProcessor):
        // image/video blocks sit BEFORE the user text, audio blocks AFTER it
        auto emit_block = [&](const MediaSplice::Block & blk) {
            const int start = (int)ids.size();
            int bo_len = 1;
            if (blk.bo == id_video) {
                // HF per-frame layout (replace_video_token):
                //   "[ ]mm:ss " <|image>(real id) <|video|>*70 <image|>(real id)
                // markers are REAL added ids (HF parses template specials),
                // timestamps are ordinary pieces
                const int fi = hf_frame++;
                double ts = 0.0;
                if (hf_timestamps && fi < (int)hf_timestamps->size())
                    ts = (*hf_timestamps)[(size_t)fi];
                char tbuf[8];
                snprintf(tbuf, sizeof(tbuf), "%02d:%02d", (int)(ts / 60.0), ((int)ts) % 60);
                std::vector<int> t = tok_.encode(std::string(fi ? " " : "") + tbuf + " ");
                ids.insert(ids.end(), t.begin(), t.end());
                ids.push_back(id_boi);
                bo_len = (int)t.size() + 1;
            } else {
                // real added ids (<|image>=255999 / <|audio>=256000)
                ids.push_back(blk.bo);
                bo_len = 1;
            }
            for (int i = 0; i < blk.n_rows; i++) ids.push_back(blk.placeholder);
            if (blk.eo != -1) {
                ids.push_back(blk.eo);   // real added id (<image|>/<audio|>)
            }
            const int soft0 = start + bo_len;
            const int end = soft0 + blk.n_rows - 1;   // last soft row
            // bidirectional SWA applies to VISION chunks only: mtmd_decode_
            // use_non_causal() returns true for projector gemma4uv but falls
            // through to default (false) for the audio projector gemma4ua -
            // audio rows decode causally like text
            const bool bidir = blk.bo != id_boa;
            for (int r = start; r <= end; r++) {
                mm.noncausal[(size_t)r] = bidir ? 1 : 0;
                mm.chunk_end[(size_t)r] = end;
            }
            for (int r = 0; r < blk.n_rows; r++)
                mm.soft_pos[(size_t)(row++)] = soft0 + r;
        };
    if (!mm.empty()) {
        // media blocks sit between the turn opener and the user text: each
        // chunk = <|image|>/<|audio|> opener + soft placeholder rows +
        // closing marker; markers decode non-causally with their chunk
        // buffer must cover every media row incl. per-block marker tokens and the
        // trailing prompt: multi-block (video) overflows the old +8 margin
        mm.noncausal.resize((size_t)ids.size() + mm.soft_pos.size() +
                            16 * mm.blocks.size() + 64, 0);
        mm.chunk_end.resize(mm.noncausal.size(), 0);

        audio_first = getenv("GEMMA4_ABLATION_AUDIO_FIRST") != nullptr;
        for (const auto & blk : mm.blocks)
            if (blk.bo != id_boa || audio_first) emit_block(blk);
        if (!audio_first)
            for (const auto & blk : mm.blocks)
                if (blk.bo == id_boa) audio_blocks.push_back(&blk);
    }
    {
        // user text, then audio blocks (official: audio follows the text),
        // then close the turn and open a bare model turn
        std::vector<int> enc = tok_.encode(prompt);
        ids.insert(ids.end(), enc.begin(), enc.end());
        if (!audio_first)
            for (const auto * blk : audio_blocks) emit_block(*blk);
        enc = tok_.encode("<turn|>\n<|turn>model\n");
        ids.insert(ids.end(), enc.begin(), enc.end());
    }
    if (getenv("GEMMA4_DUMP_IDS")) {
        fprintf(stderr, "prompt ids:");
        for (int id : ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }
    return generate(ids, gp, out_text, mm.empty() ? nullptr : &mm);
}

}  // namespace gemma4

namespace vqa {
Model * create_gemma4() { return new gemma4::Gemma4(); }
}  // namespace vqa
