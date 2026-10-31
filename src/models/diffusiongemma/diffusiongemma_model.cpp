// diffusiongemma_model.cpp - DiffusionGemma 26B-A4B (block-diffusion) backend.
//
// google/diffusiongemma-26B-A4B-it is NOT an autoregressive decoder. It is one
// shared 30-layer sparse-MoE trunk driven by two different attention regimes
// over a 256-token "canvas" (HF: DiffusionGemmaForBlockDiffusion, transformers
// 5.8):
//
//   encoder  causal (sliding 1024 on SWA layers) prefill of every token the
//            model has already committed; writes the KV cache. Vision chunks
//            are decoded bidirectionally inside their own <|image> ... <image|>
//            span (config.use_bidirectional_attention == "vision", the Gemma 3
//            block mask of create_masks_for_generate).
//   decoder  reads that cache READ-ONLY and refines the canvas with
//            bidirectional attention over [cached prefix + canvas]. HF builds
//            the decoder mask with mask_function=bidirectional_mask_function and
//            forces the full mask even on sliding layers ("the DiT module
//            doesn't need a sliding mask and has to attend fully"), so this
//            backend uploads one all-visible mask for both layer types.
//
// One canvas costs up to `max_denoising_steps` (48) decoder passes; each pass
// proposes a token per canvas slot, the EntropyBoundSampler keeps the low
// entropy proposals and renoises the rest. When the canvas stops changing and
// the mean entropy drops below `confidence_threshold`, the argmax canvas is
// committed: it is appended to the sequence and re-encoded causally into the
// KV cache (which overwrites the canvas' temporary scratch slots), then the
// next canvas starts.
//
// Trunk (config.json text_config):
//   hidden 2816, 30 layers, 16 heads, gelu_pytorch_tanh,
//     sliding layers: head_dim 256 / 8 KV heads / theta 1e4 over all 256 dims
//                     / window 1024 / v_proj present
//     full layers:    head_dim 512 / 2 KV heads / theta 1e6 "proportional"
//                     partial_rotary_factor 0.25 / NO v_proj (attention_k_eq_v:
//                     V is the k_proj output, weightless-RMS-normed)
//   per-head q/k RMSNorm (weighted) + weightless v RMSNorm, attn scale 1.0,
//   Gemma sandwich post-norms, per-layer layer_scalar, tied embeddings scaled
//   by sqrt(2816) on the token path, final logit softcap 30.
//   FFN is a PARALLEL pair: a dense GeGLU MLP and the MoE branch both read the
//   post-attention residual, are post-normed separately (post_ffn_norm_1 /
//   post_ffn_norm_2), summed, post-normed again, and added to the residual.
//   MoE: 128 experts top-8, router = weightless RMSNorm -> * router_scale /
//   sqrt(D) -> proj -> softmax(fp32) -> top-k -> renormalize -> * per-expert
//   scale; experts are GeGLU (gate_up fused in the checkpoint, split here).
//   The encoder and decoder share every trunk weight; the ONLY per-side
//   weights in the checkpoint are the two layer_scalar vectors
//   (t.enc.{i}.out_scale / t.blk.{i}.out_scale).
//
// Vision (config.json vision_config, Gemma4VisionModel):
//   27 layers, hidden 1152, patch 16, 16 heads, head_dim 72, GeGLU 4304,
//   no cls token, learned 2-D position table (2 x 10240 x 1152, x and y summed),
//   axial 2-D rope over the 72-dim head (36 dims rotate on x, 36 on y, theta
//   100), bidirectional attention, q/k weighted RMSNorm + weightless v RMSNorm,
//   attn scale 1.0. Then: patches -> 3x3 spatial mean pool (pooling_kernel_size)
//   -> * sqrt(1152) -> (x - std_bias) * std_scale -> weightless RMSNorm ->
//   embed_vision projection [1152 -> 2816]. 280 soft tokens per image, spliced
//   over <|image> placeholders (258880) between <|image>(255999)/<image|>(258882).
//
// Performance note: one decoder pass emits logits of [262144, 256] floats
// (~268 MB) because every canvas slot needs the full vocabulary. This backend
// reads them back to the host for sampling - the reference implementation does
// the same on device. On CPU that dominates the runtime (tens of seconds per
// canvas on 26B), so CUDA is strongly recommended.
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include "core/vqa.hpp"

#include "core/engine/attention.hpp"
#include "core/engine/backend.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/gguf_util.hpp"
#include "core/engine/weight_load.hpp"
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
#include <filesystem>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace dgemma {

// ------------------------------------------------------------------ weights

// Which tensors stay in host memory under the two-stage layout. The vision tower
// (1.1 GB, compute bound) and its projection go on the card, the language trunk
// (28 GB, of which 24 GB are expert tables read every step) stays in host memory.
// Each stage is a SINGLE backend, so no activation crosses a segment boundary: the
// KV cache is written in place by ggml_set_rows, which a multi-backend graph
// cannot do (that is why the scheduler path was dropped).
static int g_dg_gpu_layers = 999;   // blocks >= this stay in host memory (DG_GPU_LAYERS)
static bool dg_host_pred(const char * nm) {
    if (nm[0] == 'v' && nm[1] == '.') return false;           // tower -> device
    if (!strncmp(nm, "t.blk.", 6))                            // per-block card budget
        return atoi(nm + 6) >= g_dg_gpu_layers;
    return g_dg_gpu_layers == 0;                              // embed/norm/sc follow it too
}

struct TxtLayer {
    ggml_tensor *attn_norm = nullptr, *attn_post_norm = nullptr;
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    ggml_tensor *q_norm = nullptr, *k_norm = nullptr;
    ggml_tensor *ffn_norm = nullptr, *ffn_post_norm = nullptr;
    ggml_tensor *ffn_norm2 = nullptr, *ffn_post_norm_1 = nullptr, *ffn_post_norm_2 = nullptr;
    ggml_tensor *w_gate = nullptr, *w_up = nullptr, *w_down = nullptr;
    ggml_tensor *router_proj = nullptr, *router_scale = nullptr, *router_pes = nullptr;
    ggml_tensor *gate_exps = nullptr, *up_exps = nullptr, *down_exps = nullptr;
    ggml_tensor *out_scale_dec = nullptr;   // decoder layer_scalar
    ggml_tensor *out_scale_enc = nullptr;   // encoder layer_scalar (same trunk, own scalar)
    ggml_tensor *rope_freqs = nullptr;      // full-attn layers (proportional rope)
};

struct VisLayer {
    ggml_tensor *attn_norm = nullptr, *attn_post_norm = nullptr;
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    ggml_tensor *q_norm = nullptr, *k_norm = nullptr;
    ggml_tensor *ffn_norm = nullptr, *ffn_post_norm = nullptr;
    ggml_tensor *w_gate = nullptr, *w_up = nullptr, *w_down = nullptr;
};

// per-layer-dimension KV cache (sliding 8*256, full 2*512); vqa::KvCache
// assumes one uniform shape, so this backend rolls its own - same F16 layout
// and clear/free discipline as the gemma4 backend.
struct KvCacheMixed {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;

    bool alloc(ggml_backend_t backend, const std::vector<int> & kv_dim, int n_ctx) {
        const int n_layer = (int)kv_dim.size();
        struct ggml_init_params ip = { (size_t)(n_layer * 2 + 8) * ggml_tensor_overhead(), NULL, true };
        ctx = ggml_init(ip);
        if (!ctx) return false;
        k.resize((size_t)n_layer);
        v.resize((size_t)n_layer);
        double bytes = 0;
        for (int i = 0; i < n_layer; i++) {
            k[(size_t)i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_dim[(size_t)i], n_ctx);
            v[(size_t)i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_dim[(size_t)i], n_ctx);
            bytes += 2.0 * (double)kv_dim[(size_t)i] * n_ctx * sizeof(ggml_fp16_t);
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fprintf(stderr, "diffusiongemma: KV cache allocation failed (%.2f GB)\n", bytes / 1e9);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        fprintf(stderr, "diffusiongemma: kv cache %.2f GB (n_ctx %d, mixed per-layer dims)\n",
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
    ggml_tensor *soft_pos_in = nullptr, *soft_in = nullptr;
    ggml_tensor *sc_idx = nullptr, *sc_w = nullptr;      // decoder: self-conditioning top-k
    ggml_tensor *mask_full = nullptr, *mask_swa = nullptr;
    ggml_tensor *logits = nullptr;                       // decoder only
    int T = 0, S = 0, Sp = 0;
};

struct VisGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *patches = nullptr, *px = nullptr, *py = nullptr;
    ggml_tensor *out = nullptr;
    int np = 0;
};

struct ProjGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *in = nullptr;
    ggml_tensor *out = nullptr;
    int nt = 0;
};

// sampler knobs (generation_config.json, overridable from the GGUF)
struct DiffParams {
    int max_steps = 48;
    float t_max = 0.8f, t_min = 0.4f;
    float entropy_bound = 0.1f;
    float confidence_thr = 0.005f;
    int stability_thr = 1;
};

// ------------------------------------------------------------------- class

class DiffusionGemma : public vqa::Model {
public:
    ~DiffusionGemma() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "diffusiongemma"; }

    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp,
             std::string & out_text) override;

    // Gemma 4 turn template (same family as the gemma4 backend): a system turn
    // opens the prompt, each turn is <|turn>user/model\n ... <turn|>\n, and the
    // generation prompt is a bare <|turn>model\n. The model emits an (often
    // empty) <|channel>thought ... <channel|> block before the answer, which
    // generate() strips.
    std::string format_history_turn(const std::string & user,
                                    const std::string & assistant) const override {
        return "<|turn>user\n" + user + "<turn|>\n"
               "<|turn>model\n" + assistant + "<turn|>\n";
    }
    std::string format_system(const std::string & sys) const override {
        if (sys.empty()) return std::string();
        return "<|turn>system\n" + sys + "\n<turn|>\n";
    }

private:
    bool init_backend();
    bool load_weights(const char * path);
    // encoder/decoder graphs run on the trunk backend; vision/proj graphs run on
    // `backend` with their own allocator (see run_vision / run_proj)
    //
    // Both placements are single-backend per graph, so every operand a graph
    // reads must be reachable from that one backend: host memory for the CPU, the
    // backend's own buffer type otherwise. A weight left in the other backend is a
    // foreign pointer that the kernels dereference blindly (a crash with no
    // message), so graphs are checked before they are computed.
    bool operands_ok(ggml_cgraph * gf, ggml_backend_t be, const char * stage);
    bool graph_run(ggml_cgraph * gf) {
        if (!operands_ok(gf, text_backend, "text")) return false;
        return ggml_backend_graph_compute(text_backend, gf) == GGML_STATUS_SUCCESS;
    }
    bool graph_run_dev(ggml_cgraph * gf) {
        if (!operands_ok(gf, backend, "vision/proj")) return false;
        return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }

    bool build_graph(TextGraph & g, bool decoder_mode, int T, int S,
                     int n_soft, int n_sc);
    bool run_encoder(const std::vector<int> & toks, const std::vector<int> & soft_pos,
                     const std::vector<float> & soft, const std::vector<uint8_t> * media_rows,
                     int n_media_start, int n_media_end);
    bool run_decoder(const std::vector<int> & canvas, const std::vector<int> & sc_idx,
                     const std::vector<float> & sc_w, int n_sc, std::vector<float> & logits);

    bool build_vision_graph(int np);
    bool run_vision(const std::vector<float> & patches, const std::vector<int> & px,
                    const std::vector<int> & py, int np, std::vector<float> & hidden);
    bool build_proj_graph(int nt);
    bool run_proj(const std::vector<float> & soft_in, int nt, std::vector<float> & soft_out);
    bool preprocess_image(const uint8_t * rgb, int w0, int h0, std::vector<float> & patches,
                         std::vector<int> & px, std::vector<int> & py, int & gw, int & gh);
    bool encode_image(const vqa::MediaInput & media, std::vector<float> & soft, int & n_soft);

    bool generate(const std::vector<int> & prompt_ids, const std::vector<int> & soft_pos,
                  const std::vector<float> & soft, const vqa::GenParams & gp,
                  std::string & out_text);
    bool sample_canvas(std::vector<float> & logits, std::vector<int> & canvas,
                       std::vector<int> & argmax_canvas, std::vector<int> & sc_idx,
                       std::vector<float> & sc_w, int n_sc, float temperature,
                       std::vector<float> & entropy, uint64_t & rng);

    int hd_of(int il) const { return is_full(il) ? hd_full : hd_swa; }
    int nkv_of(int il) const { return is_full(il) ? n_kv_head_full : n_kv_head; }
    bool is_full(int il) const {
        if (il < (int)swa_pattern.size()) return swa_pattern[(size_t)il] == 0;
        return (il % 6) == 5;
    }

    vqa::Capabilities caps_ = []{ vqa::Capabilities c; c.image = true; c.video = true; return c; }();
    vqa::Tokenizer tok_;

    // text hyper (defaults = diffusiongemma-26B-A4B-it config.json)
    int n_layer = 30, n_embd = 2816, n_ff = 2112, n_ff_exp = 704, n_vocab = 262144;
    int n_expert = 128, n_expert_used = 8;
    int n_head = 16, n_kv_head = 8, n_kv_head_full = 2;
    int hd_swa = 256, hd_full = 512;
    int window = 1024;
    float eps = 1e-6f;
    float rope_theta_full = 1e6f, rope_theta_swa = 1e4f;
    float partial_rotary_full = 0.25f;
    float logit_softcapping = 30.0f;
    int canvas_len = 256;
    int max_pos = 262144;
    std::vector<int> swa_pattern;   // 1 = sliding_attention

    int id_bos = 2, id_eos = 1, id_eot = 106;
    int id_channel = 100, id_channel_end = 101;   // <|channel> ... <channel|> (reasoning block)
    int id_boi = 255999, id_eoi = 258882, id_image = 258880;

    // vision tower hyper
    int v_layer = 27, v_embd = 1152, v_ff = 4304;
    int v_head = 16, v_kv = 16, v_hd = 72;
    int v_patch = 16, v_pool = 3, v_pos_table = 10240, v_soft = 280;
    float v_theta = 100.0f, v_eps = 1e-6f;
    bool has_vision = false;

    DiffParams diff;

    int n_ctx = 8192;
    bool use_gpu = true;
    int n_threads = 0;
    std::string dump_dir;

    ggml_backend_t backend = nullptr;       // card (vision tower) or the only backend
    ggml_backend_t backend_cpu = nullptr;   // trunk host backend in two-stage mode
    ggml_backend_t text_backend = nullptr;  // what the encoder/decoder graphs run on
    bool split_mode = false;                // tower on card, trunk in host mmap
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_buffer_t host_wbuf = nullptr;
    std::vector<ggml_backend_buffer_t> host_block_bufs;   // wraps of promoted f32 blocks
    vqa::MMap mm;
    vqa::AlignedBuf host_f32;
    ggml_gallocr_t allocr = nullptr;

    ggml_tensor *t_embed = nullptr, *t_norm = nullptr;
    ggml_tensor *t_sc_pre = nullptr, *t_sc_gate = nullptr;
    ggml_tensor *t_sc_up = nullptr, *t_sc_down = nullptr;
    std::vector<TxtLayer> tl;

    ggml_tensor *v_dense = nullptr, *v_pos_table_t = nullptr;
    ggml_tensor *v_std_bias = nullptr, *v_std_scale = nullptr, *v_mm_proj = nullptr;
    std::vector<VisLayer> vl;

    ggml_tensor *rope_freqs_shared = nullptr;
    ggml_context * rf_ctx = nullptr;
    ggml_backend_buffer_t rf_buf = nullptr;

    KvCacheMixed kv;
    int n_past = 0;

    // cached graphs
    ggml_context * gctx = nullptr;          // owns the current text graph
    ggml_cgraph * gf_enc = nullptr, *gf_dec = nullptr;
    int g_enc_T = -1, g_enc_S = -1, g_enc_nsoft = -1;
    int g_dec_S = -1, g_dec_Sp = -1;
    ggml_context * vctx = nullptr;
    ggml_cgraph * vgf = nullptr;
    ggml_backend_buffer_t vbuf = nullptr;   // flat allocation (DG_VISION_CTXALLOC)
    ggml_gallocr_t vallocr = nullptr;       // own allocator: vision must not be
                                            // overwritten by a later text graph
    int vg_np = -1;
    ggml_context * pctx = nullptr;
    ggml_backend_buffer_t pbuf = nullptr;   // the projection graph owns its buffer
    ggml_cgraph * pgf = nullptr;
    int pg_nt = -1;
};

DiffusionGemma::~DiffusionGemma() {
    for (ggml_backend_buffer_t b : host_block_bufs) if (b) ggml_backend_buffer_free(b);
    if (host_wbuf) ggml_backend_buffer_free(host_wbuf);
    if (backend_cpu) ggml_backend_free(backend_cpu);
    if (vbuf) ggml_backend_buffer_free(vbuf);
    if (vctx) ggml_free(vctx);
    if (vallocr) ggml_gallocr_free(vallocr);
    if (pbuf) ggml_backend_buffer_free(pbuf);
    if (pctx) ggml_free(pctx);
    kv.free();
    if (rf_buf) ggml_backend_buffer_free(rf_buf);
    if (rf_ctx) ggml_free(rf_ctx);
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (allocr) ggml_gallocr_free(allocr);
    if (backend) ggml_backend_free(backend);
}

// Every operand a graph reads must be reachable from the single backend that
// computes it. Splitting one graph across two backends is what the old scheduler
// did; without it there is no copy between them, so a weight parked in the other
// backend's buffer is a foreign pointer the kernels dereference blindly. The check
// is cheap (one walk per graph build) and turns that crash into a message.
bool DiffusionGemma::operands_ok(ggml_cgraph * gf, ggml_backend_t be, const char * stage) {
    if (!gf || !be) return false;
    const ggml_backend_buffer_type_t own = ggml_backend_get_default_buffer_type(be);
    int nbad = 0;
    const int n_nodes = ggml_graph_n_nodes(gf);
    for (int i = 0; i < n_nodes; i++) {
        const ggml_tensor * node = ggml_graph_node(gf, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            const ggml_tensor * src = node->src[s];
            if (!src) continue;
            if (!src->buffer) {
                if (nbad++ < 6)
                    fprintf(stderr, "diffusiongemma: [%s] operand '%s' of '%s' has no buffer\n",
                            stage, src->name, node->name);
                continue;
            }
            if (ggml_backend_buffer_is_host(src->buffer)) continue;
            if (ggml_backend_buffer_get_type(src->buffer) == own) continue;
            if (nbad++ < 6)
                fprintf(stderr, "diffusiongemma: [%s] operand '%s' of '%s' lives in %s, "
                                "which %s cannot read\n",
                        stage, src->name, node->name,
                        ggml_backend_buffer_name(src->buffer), ggml_backend_name(be));
        }
    }
    if (nbad > 0) {
        fprintf(stderr, "diffusiongemma: [%s] %d operand(s) are on the wrong backend; "
                        "the graph cannot run on %s\n", stage, nbad, ggml_backend_name(be));
        return false;
    }
    return true;
}

bool DiffusionGemma::init_backend() {
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;

    // Placement: one backend per graph, decided once here. A backend can only read
    // weights that live in its own buffers (plus host memory for the CPU), so each
    // graph is bound to exactly one of the two stages and never split mid-graph.
    //   default         -> vision tower on the card, language trunk in host mmap
    //                      (28.9 GB of weights on a 12 GB card; the trunk is the
    //                      speed-neutral part, the tower is what the card is for)
    //   --cpu           -> everything on the CPU (has_gpu == false, no split)
    //   DG_GPU=1        -> everything on the card (needs ~30 GB VRAM, full speed)
    //   DG_ALL_HOST=1   -> reference path: drop the card, everything on the CPU
    //   DG_TEXT_GPU=1   -> also put the trunk on the card (only useful with a big card)
    //
    // DG_ALL_HOST is resolved here rather than in the weight binder: "all weights in
    // host memory" and "vision graph on the card" are contradictory, and the only
    // coherent reading is the pure CPU path.
    if (getenv("DG_ALL_HOST") && !ggml_backend_is_cpu(backend)) {
        fprintf(stderr, "diffusiongemma: DG_ALL_HOST=1 - reference path, no device backend\n");
        ggml_backend_free(backend);
        use_gpu = false;
        backend = vqa::init_backend(false, n_threads);
        if (!backend) return false;
    }
    const bool has_gpu = !ggml_backend_is_cpu(backend);
    const bool two_stage = has_gpu && getenv("DG_GPU") == nullptr;
    if (two_stage) {
        backend_cpu = vqa::init_backend(false, n_threads);
        if (!backend_cpu) {
            fprintf(stderr, "diffusiongemma: no CPU backend for the language trunk\n");
            return false;
        }
        if (getenv("DG_CPU_REF")) ggml_backend_cpu_set_use_ref(backend_cpu, true);
        split_mode = true;
        const bool text_gpu = getenv("DG_TEXT_GPU") != nullptr;
        text_backend = text_gpu ? backend : backend_cpu;
        fprintf(stderr, "diffusiongemma: two-stage placement - vision tower on %s, "
                        "language trunk on %s (%.2f GB of trunk weights stay in host mmap)%s\n",
                ggml_backend_name(backend), ggml_backend_name(text_backend),
                28.9 - 1.1, text_gpu ? " [DG_TEXT_GPU]" : "");
        g_dg_gpu_layers = text_gpu ? 999 : 0;      // 0: every trunk block stays host
    } else {
        text_backend = backend;
    }
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(text_backend));
    return allocr != nullptr;
}

// ---------------------------------------------------------------- load

bool DiffusionGemma::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f GB\n", mm.size / 1e9);
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    const char * A = "diffusiongemma";
    auto key = [&](const char * suf) { static std::string s; s = std::string(A) + "." + suf; return s.c_str(); };

    n_layer = vqa::gguf::u32(gu, key("block_count"), n_layer);
    n_embd = vqa::gguf::u32(gu, key("embedding_length"), n_embd);
    n_ff = vqa::gguf::u32(gu, key("feed_forward_length"), n_ff);
    n_ff_exp = vqa::gguf::u32(gu, key("expert_feed_forward_length"), n_ff_exp);
    n_expert = vqa::gguf::u32(gu, key("expert_count"), n_expert);
    n_expert_used = vqa::gguf::u32(gu, key("expert_used_count"), n_expert_used);
    n_head = vqa::gguf::u32(gu, key("attention.head_count"), n_head);
    n_kv_head = vqa::gguf::u32(gu, key("attention.head_count_kv"), n_kv_head);
    n_kv_head_full = vqa::gguf::u32(gu, key("attention.head_count_kv_full"), n_kv_head_full);
    hd_full = vqa::gguf::u32(gu, key("attention.key_length"), hd_full);
    hd_swa = vqa::gguf::u32(gu, key("attention.key_length_swa"), hd_swa);
    window = vqa::gguf::u32(gu, key("attention.sliding_window"), window);
    eps = vqa::gguf::f32(gu, key("attention.layer_norm_rms_epsilon"), eps);
    logit_softcapping = vqa::gguf::f32(gu, key("attention.final_logit_softcapping"), logit_softcapping);
    rope_theta_full = vqa::gguf::f32(gu, key("rope.freq_base"), rope_theta_full);
    rope_theta_swa = vqa::gguf::f32(gu, key("rope.freq_base_swa"), rope_theta_swa);
    partial_rotary_full = vqa::gguf::f32(gu, key("rope.partial_rotary_factor_full"), partial_rotary_full);
    max_pos = vqa::gguf::u32(gu, key("rope.context_length"), max_pos);
    n_vocab = vqa::gguf::u32(gu, key("vocab_size"), n_vocab);
    canvas_len = vqa::gguf::u32(gu, key("canvas_length"), canvas_len);
    swa_pattern = vqa::gguf::arr_u32(gu, key("attention.sliding_window_pattern"), {});
    id_bos = vqa::gguf::u32(gu, key("bos_token_id"), id_bos);
    id_eos = vqa::gguf::u32(gu, key("eos_token_id"), id_eos);
    id_eot = vqa::gguf::u32(gu, key("end_of_turn_token_id"), id_eot);
    id_boi = vqa::gguf::u32(gu, key("boi_token_id"), id_boi);
    id_eoi = vqa::gguf::u32(gu, key("eoi_token_id"), id_eoi);
    id_image = vqa::gguf::u32(gu, key("image_token_id"), id_image);
    diff.max_steps = vqa::gguf::u32(gu, key("diffusion.max_denoising_steps"), diff.max_steps);
    diff.t_max = vqa::gguf::f32(gu, key("diffusion.t_max"), diff.t_max);
    diff.t_min = vqa::gguf::f32(gu, key("diffusion.t_min"), diff.t_min);
    diff.entropy_bound = vqa::gguf::f32(gu, key("diffusion.entropy_bound"), diff.entropy_bound);
    diff.confidence_thr = vqa::gguf::f32(gu, key("diffusion.confidence_threshold"), diff.confidence_thr);
    diff.stability_thr = vqa::gguf::u32(gu, key("diffusion.stability_threshold"), diff.stability_thr);

    v_layer = vqa::gguf::u32(gu, key("vision.block_count"), v_layer);
    v_embd = vqa::gguf::u32(gu, key("vision.embedding_length"), v_embd);
    v_ff = vqa::gguf::u32(gu, key("vision.feed_forward_length"), v_ff);
    v_head = vqa::gguf::u32(gu, key("vision.attention.head_count"), v_head);
    v_kv = vqa::gguf::u32(gu, key("vision.attention.head_count_kv"), v_kv);
    v_hd = vqa::gguf::u32(gu, key("vision.attention.key_length"), v_hd);
    v_theta = vqa::gguf::f32(gu, key("vision.rope.freq_base"), v_theta);
    v_eps = vqa::gguf::f32(gu, key("vision.attention.layer_norm_rms_epsilon"), v_eps);
    v_patch = vqa::gguf::u32(gu, key("vision.patch_size"), v_patch);
    v_pool = vqa::gguf::u32(gu, key("vision.pooling_kernel_size"), v_pool);
    v_pos_table = vqa::gguf::u32(gu, key("vision.position_embedding_size"), v_pos_table);
    v_soft = vqa::gguf::u32(gu, key("vision.max_soft_tokens"), v_soft);

    if (const char * e = getenv("DG_STEPS")) diff.max_steps = atoi(e);
    if (const char * e = getenv("DG_TMAX")) diff.t_max = (float)atof(e);
    if (const char * e = getenv("DG_TMIN")) diff.t_min = (float)atof(e);
    if (const char * e = getenv("DG_ENTROPY_BOUND")) diff.entropy_bound = (float)atof(e);
    if (const char * e = getenv("DG_CONF_THR")) diff.confidence_thr = (float)atof(e);

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);
    vqa::weights::promote_1d_to_f32(wctx, host_f32);

    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) fprintf(stderr, "missing tensor: %s\n", n);
        return t;
    };
    t_embed = gt("t.embed.weight");
    t_norm = gt("t.norm.weight");
    t_sc_pre = gt("t.sc.pre_norm.weight");
    t_sc_gate = gt("t.sc.gate.weight");
    t_sc_up = gt("t.sc.up.weight");
    t_sc_down = gt("t.sc.down.weight");
    // self_conditioning.post_norm is weightless (with_scale=False): no tensor
    if (!t_embed || !t_norm || !t_sc_pre || !t_sc_gate || !t_sc_up || !t_sc_down)
        return false;
    if ((int)t_embed->ne[0] != n_embd) {
        fprintf(stderr, "diffusiongemma: embed row length %lld != n_embd %d (tied head expected)\n",
                (long long)t_embed->ne[0], n_embd);
        return false;
    }

    tl.resize((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) {
        char b[128];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "t.blk.%d.%s", i, suf); return gt(b); };
        TxtLayer & L = tl[(size_t)i];
        L.attn_norm = g("attn_norm.weight");
        L.attn_post_norm = g("attn_post_norm.weight");
        L.wq = g("wq.weight"); L.wk = g("wk.weight"); L.wv = g("wv.weight"); L.wo = g("wo.weight");
        L.q_norm = g("q_norm.weight"); L.k_norm = g("k_norm.weight");
        L.ffn_norm = g("ffn_norm.weight"); L.ffn_post_norm = g("ffn_post_norm.weight");
        L.ffn_norm2 = g("ffn_norm2.weight");
        L.ffn_post_norm_1 = g("ffn_post_norm_1.weight");
        L.ffn_post_norm_2 = g("ffn_post_norm_2.weight");
        L.w_gate = g("w_gate.weight"); L.w_up = g("w_up.weight"); L.w_down = g("w_down.weight");
        L.router_proj = g("router_proj.weight");
        L.router_scale = g("router_scale.weight");
        L.router_pes = g("router_per_expert_scale.weight");
        L.gate_exps = g("gate_exps.weight");
        L.up_exps = g("up_exps.weight");
        L.down_exps = g("down_exps.weight");
        L.out_scale_dec = g("out_scale");
        snprintf(b, sizeof(b), "t.enc.%d.out_scale", i);
        L.out_scale_enc = gt(b);
        if (!L.attn_norm || !L.wq || !L.wk || !L.wo || !L.q_norm || !L.k_norm ||
            !L.w_gate || !L.w_up || !L.w_down || !L.ffn_norm || !L.ffn_post_norm ||
            !L.ffn_norm2 || !L.ffn_post_norm_1 || !L.ffn_post_norm_2 ||
            !L.router_proj || !L.router_scale || !L.router_pes ||
            !L.gate_exps || !L.up_exps || !L.down_exps) {
            fprintf(stderr, "load: layer %d has null tensors\n", i);
            return false;
        }
        if (is_full(i) && !L.wv) {
            // attention_k_eq_v: V is the k_proj output (no v_proj in full layers)
        } else if (!L.wv) {
            fprintf(stderr, "load: sliding layer %d is missing v_proj\n", i);
            return false;
        }
        if (is_full(i) && !L.rope_freqs && !rope_freqs_shared) {
            // proportional rope 0.25: rotate the first 128 of 512 head dims by
            // pinning the trailing pairs with 1e30 (this fork's rope divides the
            // angle by the factor, same trick the gemma4 backend uses)
            const int pairs = hd_full / 2;
            const int rot_pairs = (int)(hd_full * partial_rotary_full) / 2;
            std::vector<float> ff((size_t)pairs, 1e30f);
            for (int j = 0; j < rot_pairs; j++) ff[(size_t)j] = 1.0f;
            struct ggml_init_params ip = { 1ull << 16, NULL, true };
            rf_ctx = ggml_init(ip);
            if (!rf_ctx) return false;
            rope_freqs_shared = ggml_new_tensor_1d(rf_ctx, GGML_TYPE_F32, pairs);
            // the rope frequencies are read by the TEXT graph, so they must live
            // where that graph runs - in split placement `backend` is the card and
            // the trunk runs on the CPU, where a device pointer is an access
            // violation inside the first rope node
            rf_buf = ggml_backend_alloc_ctx_tensors(rf_ctx, text_backend);
            if (!rf_buf) return false;
            ggml_backend_tensor_set(rope_freqs_shared, ff.data(), 0, sizeof(float) * ff.size());
        }
        if (is_full(i)) L.rope_freqs = rope_freqs_shared;
    }

    // ---- vision tower (optional: a text-only GGUF keeps the text surface)
    v_dense = ggml_get_tensor(wctx, "v.patch_dense.weight");
    v_pos_table_t = ggml_get_tensor(wctx, "v.pos_table.weight");
    v_std_bias = ggml_get_tensor(wctx, "v.std_bias.weight");
    v_std_scale = ggml_get_tensor(wctx, "v.std_scale.weight");
    v_mm_proj = ggml_get_tensor(wctx, "v.mm_proj.weight");
    has_vision = v_dense && v_pos_table_t && v_std_bias && v_std_scale && v_mm_proj;
    if (has_vision) {
        vl.resize((size_t)v_layer);
        for (int i = 0; i < v_layer; i++) {
            char b[128];
            auto g = [&](const char * suf) { snprintf(b, sizeof(b), "v.blk.%d.%s", i, suf); return ggml_get_tensor(wctx, b); };
            VisLayer & L = vl[(size_t)i];
            L.attn_norm = g("attn_norm.weight");
            L.attn_post_norm = g("attn_post_norm.weight");
            L.wq = g("wq.weight"); L.wk = g("wk.weight"); L.wv = g("wv.weight"); L.wo = g("wo.weight");
            L.q_norm = g("q_norm.weight"); L.k_norm = g("k_norm.weight");
            L.ffn_norm = g("ffn_norm.weight"); L.ffn_post_norm = g("ffn_post_norm.weight");
            L.w_gate = g("w_gate.weight"); L.w_up = g("w_up.weight"); L.w_down = g("w_down.weight");
            if (!L.attn_norm || !L.wq || !L.wk || !L.wv || !L.wo || !L.q_norm || !L.k_norm ||
                !L.ffn_norm || !L.ffn_post_norm || !L.w_gate || !L.w_up || !L.w_down) {
                fprintf(stderr, "load: vision layer %d has null tensors\n", i);
                return false;
            }
        }
    } else {
        fprintf(stderr, "diffusiongemma: no vision tower tensors in GGUF (text-only model file)\n");
    }

    if (split_mode) {
        // trunk -> host mmap, tower + projection -> device; the mmap must stay
        // mapped because the trunk weights are read in place
        if (const char * gl = getenv("DG_GPU_LAYERS")) g_dg_gpu_layers = atoi(gl);
        vqa::weights::SplitBound sb = vqa::weights::split_bind_to_backends(
            wctx, backend, mm, dg_host_pred,
            { { host_f32.data(), host_f32.size() } });
        if (!sb.ok) {
            fprintf(stderr, "diffusiongemma: split weight placement failed\n");
            return false;
        }
        wbuf = sb.dev_buf;
        host_wbuf = sb.host_buf;
        host_block_bufs = sb.host_block_bufs;
        sb.dev_buf = sb.host_buf = nullptr;
        sb.host_block_bufs.clear();          // ownership moved to the model
        // tag the device buffer as weights; the host one stays a plain mmap wrap
        // (tagging it made the CPU graph fault on this fork)
        if (wbuf) ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    } else {
        vqa::weights::Bound bound = vqa::weights::bind_to_backend(
            wctx, backend, mm, { { host_f32.data(), host_f32.size() } });
        if (!bound.ok) return false;
        wbuf = bound.main_buf;
        if (!ggml_backend_is_cpu(backend)) mm.release();
    }

    // DG_DUMP_WEIGHTS=<dir> dumps the raw bytes of a few tensors straight from
    // the backend buffer, so a "the graph is wrong" report can be separated from
    // "the weights bound wrong" (compare with gguf.quants.dequantize on the
    // numpy side).
    if (const char * wdir = getenv("DG_DUMP_WEIGHTS")) {
        std::error_code ec;
        std::filesystem::create_directories(wdir, ec);
        auto dump_raw = [&](const char * nm) {
            ggml_tensor * t = ggml_get_tensor(wctx, nm);
            if (!t) return;
            std::vector<uint8_t> b((size_t)ggml_nbytes(t));
            ggml_backend_tensor_get(t, b.data(), 0, b.size());
            std::string base = std::string(wdir) + "/w_" + nm;
            for (auto & ch : base) if (ch == '.') ch = '_';
            if (FILE * f = fopen((base + ".bin").c_str(), "wb")) {
                fwrite(b.data(), 1, b.size(), f);
                fclose(f);
            }
            if (FILE * f = fopen((base + ".txt").c_str(), "w")) {
                fprintf(f, "%d %d %d %d type=%d\t%.0f\n", (int)t->ne[0], (int)t->ne[1],
                        (int)t->ne[2], (int)t->ne[3], (int)t->type, (double)ggml_nbytes(t));
                fclose(f);
            }
        };
        dump_raw("v.patch_dense.weight");
        dump_raw("v.pos_table.weight");
        dump_raw("v.blk.0.wq.weight");
    }
    if (getenv("DG_DEBUG")) {
        auto show = [](const char * n, ggml_tensor * t) {
            fprintf(stderr, "[w] %-18s data=%p buffer=%s type=%s nbytes=%zu\n", n,
                    t ? t->data : nullptr,
                    (t && t->buffer) ? ggml_backend_buffer_name(t->buffer) : "-",
                    t ? ggml_type_name(t->type) : "-", t ? (size_t)ggml_nbytes(t) : 0);
        };
        show("t.embed", t_embed);
        show("t.norm", t_norm);
        if (!tl.empty()) {
            show("t.blk.0.wq", tl[0].wq);
            show("t.blk.0.gate_exps", tl[0].gate_exps);
            show("t.blk.0.router_scale", tl[0].router_scale);
        }
        if (!vl.empty()) show("v.blk.0.wq", vl[0].wq);
        show("v.patch_dense", v_dense);
    }
    return true;
}

bool DiffusionGemma::load(const std::string & gguf_path, const std::string & tokenizer_path,
                          const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "DG", n_threads, use_gpu, dump_dir);
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::GemmaSP)) return false;
    // media markers keep the GGUF ids: the checkpoint's tokenizer names the
    // placeholder `<image>` (258880) while the brackets are `<|image>` (255999)
    // / `<image|>` (258882), and the chat template only emits the single
    // `<|image|>` marker which the processor expands. Resolving pieces here
    // would risk latching onto the placeholder id instead.
    vqa::assign_token_ids(tok_, {
        { "<end_of_turn>", &id_eot },
        { "<eos>", &id_eos },
        { "<|channel>", &id_channel },
        { "<channel|>", &id_channel_end },
    });

    std::vector<int> kv_dim((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) kv_dim[(size_t)i] = nkv_of(i) * hd_of(i);
    // Split placement keeps the KV cache on the host: the kv rows are written by
    // ggml_set_rows from whichever segment computes that layer's k/v, and a
    // segment cannot write into another backend's buffer (that is an illegal
    // access). Host-resident KV costs one activation copy per attention block -
    // a few MB at canvas-sized Sp - and keeps every write local.
    if (!kv.alloc(text_backend, kv_dim, n_ctx)) return false;

    caps_ = {};
    caps_.image = has_vision;
    caps_.video = has_vision;   // per-frame image chunks
    caps_.timestamps_prompt = false;

    int n_full = 0;
    for (int i = 0; i < n_layer; i++) n_full += is_full(i) ? 1 : 0;
    fprintf(stderr, "diffusiongemma: ready (%d layers: %d sliding hd=%d x%d kv + %d full hd=%d x%d kv, "
                    "d=%d dense_ff=%d moe=%dx%d(top %d), window %d, canvas %d, steps %d, ctx %d)\n",
            n_layer, n_layer - n_full, hd_swa, n_kv_head, n_full, hd_full, n_kv_head_full,
            n_embd, n_ff, n_expert, n_ff_exp, n_expert_used, window, canvas_len,
            diff.max_steps, n_ctx);
    return true;
}

// -------------------------------------------------------------- text graph

bool DiffusionGemma::build_graph(TextGraph & g, bool decoder_mode, int T, int S,
                                 int n_soft, int n_sc) {
    const bool with_soft = n_soft > 0;
    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 32768, false);
    g.T = T; g.S = S;
    const int Sp = std::min(n_ctx, GGML_PAD(S, 256));
    g.Sp = Sp;

    g.tok_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    ggml_set_input(g.tok_in);
    g.pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.pos_in, "pos_in");
    ggml_set_input(g.pos_in);

    g.soft_pos_in = g.soft_in = nullptr;
    if (with_soft) {
        g.soft_pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, (int64_t)n_soft);
        ggml_set_name(g.soft_pos_in, "soft_pos");
        ggml_set_input(g.soft_pos_in);
        g.soft_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, (int64_t)n_soft);
        ggml_set_name(g.soft_in, "soft_in");
        ggml_set_input(g.soft_in);
    }
    g.sc_idx = g.sc_w = nullptr;
    if (decoder_mode) {
        // 1-D ids: this fork's ggml_get_rows requires a->ne[2] == b->ne[1], so a
        // [K, T] index matrix against the 2-D embedding table is illegal. A flat
        // K*T list reshapes back into [D, K, T] with column index k + K*t.
        g.sc_idx = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, (int64_t)n_sc * T);
        ggml_set_name(g.sc_idx, "sc_idx");
        ggml_set_input(g.sc_idx);
        g.sc_w = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, n_sc, T);
        ggml_set_name(g.sc_w, "sc_w");
        ggml_set_input(g.sc_w);
    }

    // masks live in the gallocr pool here (unlike the gemma4 backend, whose
    // comment records a clobbering bug): both are graph inputs explicitly
    // marked with ggml_set_input, and every consumer reads them uniformly.
    g.mask_full = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, Sp, T);
    ggml_set_name(g.mask_full, "mask_full");
    ggml_set_input(g.mask_full);
    g.mask_swa = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, Sp, T);
    ggml_set_name(g.mask_swa, "mask_swa");
    ggml_set_input(g.mask_swa);

    // token embeddings scaled by sqrt(n_embd) (Gemma normalizer, token path only)
    ggml_tensor * x = ggml_get_rows(g.ctx, t_embed, g.tok_in);
    x = ggml_cast(g.ctx, x, GGML_TYPE_F32);
    x = ggml_scale(g.ctx, x, sqrtf((float)n_embd));
    if (g.soft_in) {
        // media rows REPLACE the placeholder rows with the raw projection
        // output (already in language space - no sqrt scaling)
        x = ggml_set_rows(g.ctx, ggml_dup(g.ctx, x), g.soft_in, g.soft_pos_in);
    }

    if (decoder_mode) {
        // self-conditioning: E[embed] under the previous step's softmax, taken
        // over the top-k entries the host kept (w is pre-normalized, idx holds
        // the vocabulary rows). Its MLP output is added to the canvas
        // embeddings, then a weightless RMSNorm closes the block.
        ggml_tensor * sce = ggml_get_rows(g.ctx, t_embed, g.sc_idx);      // [D, K*T]
        sce = ggml_reshape_3d(g.ctx, ggml_cast(g.ctx, sce, GGML_TYPE_F32), n_embd, n_sc, T);
        sce = ggml_mul(g.ctx, sce, g.sc_w);                              // [D, K, T]
        sce = ggml_cont(g.ctx, ggml_permute(g.ctx, sce, 1, 0, 2, 3));    // [K, D, T]
        sce = ggml_reshape_2d(g.ctx, ggml_sum_rows(g.ctx, sce), n_embd, T);
        sce = ggml_scale(g.ctx, sce, sqrtf((float)n_embd));
        ggml_tensor * z = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, sce, eps), t_sc_pre);
        ggml_tensor * gz = ggml_mul_mat(g.ctx, t_sc_gate, z);
        ggml_tensor * uz = ggml_mul_mat(g.ctx, t_sc_up, z);
        ggml_tensor * h = ggml_mul_mat(g.ctx, t_sc_down, ggml_geglu_split(g.ctx, gz, uz));
        x = ggml_rms_norm(g.ctx, ggml_add(g.ctx, x, h), eps);            // post_norm (no scale)
    }
    ggml_set_name(x, "emb");

    const float router_scale = 1.0f / sqrtf((float)n_embd);

    for (int il = 0; il < n_layer; il++) {
        const bool full = is_full(il);
        const int hd = full ? hd_full : hd_swa;
        const int nkv = nkv_of(il);
        TxtLayer & L = tl[(size_t)il];

        // ---- attention
        ggml_tensor * cur = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, x, eps), L.attn_norm);
        ggml_tensor * q = ggml_mul_mat(g.ctx, L.wq, cur);               // [nh*hd, T]
        ggml_tensor * k = ggml_mul_mat(g.ctx, L.wk, cur);               // [nkv*hd, T]
        ggml_tensor * vraw = L.wv ? ggml_mul_mat(g.ctx, L.wv, cur) : k; // k_eq_v before norm
        q = ggml_reshape_3d(g.ctx, q, hd, n_head, T);
        k = ggml_reshape_3d(g.ctx, k, hd, nkv, T);
        vraw = ggml_reshape_3d(g.ctx, vraw, hd, nkv, T);
        q = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, q, eps), L.q_norm);
        k = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, k, eps), L.k_norm);
        ggml_tensor * v = ggml_rms_norm(g.ctx, vraw, eps);              // weightless

        const float theta = full ? rope_theta_full : rope_theta_swa;
        const int n_dims = hd;   // full layers rotate all 512 dims, freq factors pin 384
        ggml_tensor * ff = full ? L.rope_freqs : nullptr;
        q = ggml_rope_ext(g.ctx, q, g.pos_in, ff, n_dims, GGML_ROPE_TYPE_NEOX,
                          max_pos, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(g.ctx, k, g.pos_in, ff, n_dims, GGML_ROPE_TYPE_NEOX,
                          max_pos, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (il == 0) ggml_set_name(q, "t_l0_q_rope");

        // KV write. The decoder writes the canvas' scratch rows at
        // [n_past, n_past+T); the next encoder pass overwrites exactly those
        // rows with the committed (causal, encoder-scalar) values.
        ggml_tensor * k2 = ggml_reshape_2d(g.ctx, k, nkv * hd, T);
        ggml_tensor * v2 = ggml_reshape_2d(g.ctx, v, nkv * hd, T);
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.k[(size_t)il], k2, g.pos_in));
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.v[(size_t)il], v2, g.pos_in));

        ggml_tensor * kf = ggml_view_3d(g.ctx, kv.k[(size_t)il], hd, Sp, nkv,
                                        kv.k[(size_t)il]->nb[1], hd * sizeof(ggml_fp16_t), 0);
        ggml_tensor * vf = ggml_view_3d(g.ctx, kv.v[(size_t)il], hd, Sp, nkv,
                                        kv.v[(size_t)il]->nb[1], hd * sizeof(ggml_fp16_t), 0);
        ggml_tensor * qf = ggml_view_3d(g.ctx, q, hd, T, n_head, q->nb[2], q->nb[1], 0);
        ggml_tensor * mask = full ? g.mask_full : g.mask_swa;
        // attn scale 1.0: the q/k RMSNorms carry the scale (HF scaling = 1.0)
        ggml_tensor * fa = ggml_flash_attn_ext(g.ctx, qf, kf, vf, mask, 1.0f, 0.0f, 0.0f);
        if (il == 0) ggml_set_name(fa, "t_l0_fa");
        ggml_tensor * attn = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, fa), n_head * hd, T);
        attn = ggml_mul_mat(g.ctx, L.wo, attn);
        if (il == 0) ggml_set_name(attn, "t_l0_attn");

        ggml_tensor * res = ggml_add(g.ctx, x, ggml_mul(g.ctx, ggml_rms_norm(g.ctx, attn, eps), L.attn_post_norm));
        if (il == 0) ggml_set_name(res, "t_l0_resid");

        // ---- parallel FFN: dense GeGLU MLP + MoE, each post-normed, then summed
        ggml_tensor * f = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, res, eps), L.ffn_norm);
        ggml_tensor * mgate = ggml_mul_mat(g.ctx, L.w_gate, f);
        ggml_tensor * mup = ggml_mul_mat(g.ctx, L.w_up, f);
        // fused GeGLU: one kernel instead of gelu + mul + a temp tensor
        ggml_tensor * mlp = ggml_mul_mat(g.ctx, L.w_down, ggml_geglu_split(g.ctx, mgate, mup));
        ggml_tensor * h1 = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, mlp, eps), L.ffn_post_norm_1);

        // router: weightless RMSNorm * scale / sqrt(D) -> proj -> softmax -> top-k
        ggml_tensor * rr = ggml_rms_norm(g.ctx, res, eps);
        rr = ggml_mul(g.ctx, rr, L.router_scale);
        rr = ggml_scale(g.ctx, rr, router_scale);
        ggml_tensor * rlogits = ggml_mul_mat(g.ctx, L.router_proj, rr);   // [E, T]
        ggml_tensor * probs = ggml_soft_max(g.ctx, rlogits);              // f32 softmax
        ggml_tensor * sel = ggml_top_k(g.ctx, probs, n_expert_used);      // [K, T] I32
        ggml_tensor * pw = ggml_get_rows(g.ctx,
                ggml_reshape_3d(g.ctx, probs, 1, n_expert, T), sel);      // [1, K, T]
        pw = ggml_reshape_2d(g.ctx, pw, n_expert_used, T);
        pw = ggml_div(g.ctx, pw, ggml_sum_rows(g.ctx, pw));               // renormalize over k
        // per-expert scale gather: flat ids, same reason as sc_idx above
        ggml_tensor * sel_flat = ggml_reshape_1d(g.ctx, sel, (int64_t)n_expert_used * T);
        ggml_tensor * pes = ggml_get_rows(g.ctx,
                ggml_reshape_2d(g.ctx, L.router_pes, 1, n_expert), sel_flat);
        pes = ggml_reshape_3d(g.ctx, pes, 1, n_expert_used, T);            // [1, K, T]
        pw = ggml_mul(g.ctx, pw, ggml_reshape_2d(g.ctx, pes, n_expert_used, T));
        ggml_tensor * w8 = ggml_reshape_3d(g.ctx, pw, 1, n_expert_used, T);

        ggml_tensor * route_in = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, res, eps), L.ffn_norm2);
        ggml_tensor * cur3 = ggml_reshape_3d(g.ctx, route_in, n_embd, 1, T);
        ggml_tensor * gu = ggml_mul_mat_id(g.ctx, L.gate_exps, cur3, sel);   // [Ie, K, T]
        ggml_tensor * eu = ggml_mul_mat_id(g.ctx, L.up_exps, cur3, sel);     // [Ie, K, T]
        ggml_tensor * act = ggml_geglu_split(g.ctx, gu, eu);   // fused GeGLU per expert
        ggml_tensor * ff3 = ggml_mul_mat_id(g.ctx, L.down_exps, act, sel);   // [D, K, T]
        ggml_tensor * wb = ggml_view_3d(g.ctx, w8, 1, n_expert_used, T, w8->nb[1], w8->nb[2], 0);
        ggml_tensor * prod = ggml_mul(g.ctx, ff3, wb);
        ggml_tensor * pk = ggml_cont(g.ctx, ggml_permute(g.ctx, prod, 1, 0, 2, 3));   // [K, D, T]
        ggml_tensor * moe = ggml_reshape_2d(g.ctx, ggml_sum_rows(g.ctx, pk), n_embd, T);
        ggml_tensor * h2 = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, moe, eps), L.ffn_post_norm_2);
        if (il == 0) ggml_set_name(h2, "t_l0_moe");

        ggml_tensor * hsum = ggml_add(g.ctx, h1, h2);
        ggml_tensor * hnorm = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, hsum, eps), L.ffn_post_norm);
        x = ggml_add(g.ctx, res, hnorm);
        ggml_tensor * oscale = decoder_mode ? L.out_scale_dec : L.out_scale_enc;
        if (oscale) x = ggml_mul(g.ctx, x, oscale);
        if (il == 0) ggml_set_name(x, "t_l0_out");
    }

    if (!decoder_mode) {
        // the encoder only fills the KV cache - no head, no logits
        g.logits = nullptr;
        return true;
    }

    x = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, x, eps), t_norm);
    ggml_set_name(x, "t_final_norm");
    // every canvas slot needs its own distribution: tied head over all T rows
    ggml_tensor * logits = ggml_mul_mat(g.ctx, t_embed, x);              // [vocab, T]
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
            "emb", "t_l0_q_rope", "t_l0_attn", "t_l0_resid", "t_l0_moe", "t_l0_out",
            "t_final_norm", "logits", nullptr,
        };
        vqa::dump::mark_sinks(g.ctx, g.gf, dn);
        vqa::dump::finalize_sinks(g.ctx, g.gf, backend);
    }
    return true;
}

// mask upload helpers -------------------------------------------------------

static void fill_causal_swa(ggml_tensor * mask, int n_past, int T, int window,
                            const std::vector<std::pair<int, int>> * bidir_spans) {
    const int Sw = (int)mask->ne[0];
    std::vector<ggml_fp16_t> mv((size_t)Sw * T, ggml_fp32_to_fp16(-INFINITY));
    for (int t = 0; t < T; t++) {
        const int qt = n_past + t;
        const int lo = std::max(0, qt - window + 1);
        const int hi = std::min(Sw - 1, qt);
        for (int s = lo; s <= hi; s++) mv[(size_t)t * Sw + s] = ggml_fp32_to_fp16(0.0f);
    }
    // media spans (absolute query indices [q0, q1)) decode bidirectionally:
    // every row inside the span sees the whole span, and nothing past it
    if (bidir_spans) {
        for (const auto & sp : *bidir_spans) {
            const int q0 = std::max(0, sp.first - n_past);
            const int q1 = std::min(T, sp.second - n_past);
            if (q0 >= q1) continue;
            for (int t = q0; t < q1; t++) {
                for (int s = std::max(0, sp.first); s < sp.second; s++) {
                    if (s < Sw) mv[(size_t)t * Sw + s] = ggml_fp32_to_fp16(0.0f);
                }
            }
        }
    }
    ggml_backend_tensor_set(mask, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());
}

static void fill_causal(ggml_tensor * mask, int n_past, int T,
                        const std::vector<std::pair<int, int>> * bidir_spans) {
    const int Sw = (int)mask->ne[0];
    std::vector<ggml_fp16_t> mv((size_t)Sw * T, ggml_fp32_to_fp16(-INFINITY));
    for (int t = 0; t < T; t++) {
        const int qt = n_past + t;
        const int hi = std::min(Sw - 1, qt);
        for (int s = 0; s <= hi; s++) mv[(size_t)t * Sw + s] = ggml_fp32_to_fp16(0.0f);
    }
    // full-attention layers carry the same vision block mask as the sliding
    // ones (HF create_masks_for_generate passes block_sequence_ids to every
    // layer type), so media rows see their whole span there too
    if (bidir_spans) {
        for (const auto & sp : *bidir_spans) {
            const int q0 = std::max(0, sp.first - n_past);
            const int q1 = std::min(T, sp.second - n_past);
            for (int t = q0; t < q1; t++)
                for (int s = std::max(0, sp.first); s < sp.second && s < Sw; s++)
                    mv[(size_t)t * Sw + s] = ggml_fp32_to_fp16(0.0f);
        }
    }
    ggml_backend_tensor_set(mask, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());
}

static void fill_full(ggml_tensor * mask, int n_past, int T) {
    // decoder: every canvas row attends to the whole cached prefix AND to the
    // whole canvas (HF forces the full bidirectional mask, sliding layers too)
    const int Sw = (int)mask->ne[0];
    const int visible = std::min(Sw, n_past + T);
    std::vector<ggml_fp16_t> mv((size_t)Sw * T, ggml_fp32_to_fp16(-INFINITY));
    for (int t = 0; t < T; t++)
        for (int s = 0; s < visible; s++) mv[(size_t)t * Sw + s] = ggml_fp32_to_fp16(0.0f);
    ggml_backend_tensor_set(mask, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());
}

// ------------------------------------------------------------- enc/dec runs

bool DiffusionGemma::run_encoder(const std::vector<int> & toks, const std::vector<int> & soft_pos,
                                 const std::vector<float> & soft,
                                 const std::vector<uint8_t> * media_rows,
                                 int media_lo, int media_hi) {
    const int T = (int)toks.size();
    const int S = n_past + T;
    if (S > n_ctx) { fprintf(stderr, "diffusiongemma: context overflow (%d > %d)\n", S, n_ctx); return false; }
    const bool with_soft = !soft.empty();
    const int n_soft = with_soft ? (int)(soft.size() / (size_t)n_embd) : 0;
    if (with_soft && (int)soft_pos.size() != n_soft) {
        fprintf(stderr, "diffusiongemma: soft position count %zu != %d embedded rows\n",
                soft_pos.size(), n_soft);
        return false;
    }

    TextGraph g;
    ggml_gallocr_t al = allocr;
    const bool cache_hit = (gf_enc && g_enc_T == T && g_enc_S == S && g_enc_nsoft == n_soft);
    if (cache_hit) {
        g.ctx = nullptr;
        g.gf = gf_enc;
        g.tok_in = ggml_graph_get_tensor(gf_enc, "tok_in");
        g.pos_in = ggml_graph_get_tensor(gf_enc, "pos_in");
        g.mask_full = ggml_graph_get_tensor(gf_enc, "mask_full");
        g.mask_swa = ggml_graph_get_tensor(gf_enc, "mask_swa");
        g.soft_pos_in = with_soft ? ggml_graph_get_tensor(gf_enc, "soft_pos") : nullptr;
        g.soft_in = with_soft ? ggml_graph_get_tensor(gf_enc, "soft_in") : nullptr;
        g.sc_idx = g.sc_w = g.logits = nullptr;
        g.T = T; g.S = S; g.Sp = std::min(n_ctx, GGML_PAD(S, 256));
    } else {
        if (gctx) { ggml_free(gctx); gctx = nullptr; gf_enc = nullptr; }
        struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, true };
        gctx = ggml_init(ip);
        if (!gctx) return false;
        g.ctx = gctx;
        if (getenv("DG_DEBUG"))
            fprintf(stderr, "[dg] encoder build T=%d S=%d split=%d backend=%s\n", T, S,
                    (int)split_mode, ggml_backend_name(text_backend));
        if (!build_graph(g, /*decoder*/ false, T, S, n_soft, 0)) return false;
        if (!ggml_gallocr_alloc_graph(al, g.gf)) {
            fprintf(stderr, "diffusiongemma: encoder graph allocation failed\n");
            return false;
        }
        gf_enc = g.gf;
        g_enc_T = T; g_enc_S = S; g_enc_nsoft = n_soft;
    }

    if (getenv("DG_DEBUG")) fprintf(stderr, "[dg] enc alloc done\n");
    ggml_backend_tensor_set(g.tok_in, toks.data(), 0, sizeof(int) * T);
    std::vector<int> pos((size_t)T);
    for (int i = 0; i < T; i++) pos[(size_t)i] = n_past + i;
    ggml_backend_tensor_set(g.pos_in, pos.data(), 0, sizeof(int) * T);
    if (with_soft) {
        ggml_backend_tensor_set(g.soft_pos_in, soft_pos.data(), 0, sizeof(int) * soft_pos.size());
        ggml_backend_tensor_set(g.soft_in, soft.data(), 0, sizeof(float) * soft.size());
    }

    // contiguous media runs, in ABSOLUTE token positions (the mask helpers
    // index columns by absolute KV slot and map back with `- n_past`)
    std::vector<std::pair<int, int>> spans;
    if (media_rows && media_hi > media_lo) {
        int start = -1;
        for (int i = media_lo; i <= media_hi; i++) {
            const bool in = (i < media_hi) && (*media_rows)[(size_t)i];
            if (in && start < 0) start = i;
            if (!in && start >= 0) { spans.push_back({ n_past + start, n_past + i }); start = -1; }
        }
    }
    fill_causal(g.mask_full, n_past, T, spans.empty() ? nullptr : &spans);
    fill_causal_swa(g.mask_swa, n_past, T, window, spans.empty() ? nullptr : &spans);

    if (getenv("DG_DEBUG")) fprintf(stderr, "[dg] enc inputs set, computing\n");
    if (!graph_run(g.gf)) {
        fprintf(stderr, "diffusiongemma: encoder compute failed\n");
        return false;
    }
    if (getenv("DG_DEBUG")) fprintf(stderr, "[dg] enc done S=%d\n", S);
    n_past = S;
    return true;
}

bool DiffusionGemma::run_decoder(const std::vector<int> & canvas, const std::vector<int> & sc_idx,
                                 const std::vector<float> & sc_w, int n_sc,
                                 std::vector<float> & logits) {
    const int T = (int)canvas.size();
    const int S = n_past + T;
    if (S > n_ctx) { fprintf(stderr, "diffusiongemma: context overflow (%d > %d)\n", S, n_ctx); return false; }
    const int Sp = std::min(n_ctx, GGML_PAD(S, 256));

    TextGraph g;
    const bool cache_hit = (gf_dec && g_dec_S == S && g_dec_Sp == Sp && n_sc > 0);
    if (cache_hit) {
        g.ctx = nullptr;
        g.gf = gf_dec;
        g.tok_in = ggml_graph_get_tensor(gf_dec, "tok_in");
        g.pos_in = ggml_graph_get_tensor(gf_dec, "pos_in");
        g.mask_full = ggml_graph_get_tensor(gf_dec, "mask_full");
        g.mask_swa = ggml_graph_get_tensor(gf_dec, "mask_swa");
        g.sc_idx = ggml_graph_get_tensor(gf_dec, "sc_idx");
        g.sc_w = ggml_graph_get_tensor(gf_dec, "sc_w");
        g.soft_in = g.soft_pos_in = nullptr;
        g.logits = ggml_graph_get_tensor(gf_dec, "logits");
        g.T = T; g.S = S; g.Sp = Sp;
    } else {
        // NOTE: gctx is shared with the encoder graph; rebuilding the decoder
        // graph drops the cached encoder graph too (they alternate, so this
        // only costs one extra build per canvas).
        if (gctx) { ggml_free(gctx); gctx = nullptr; gf_enc = nullptr; gf_dec = nullptr; }
        struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, true };
        gctx = ggml_init(ip);
        if (!gctx) return false;
        g.ctx = gctx;
        if (!build_graph(g, /*decoder*/ true, T, S, 0, n_sc)) return false;
        if (!ggml_gallocr_alloc_graph(allocr, g.gf)) {
            fprintf(stderr, "diffusiongemma: decoder graph allocation failed\n");
            return false;
        }
        gf_dec = g.gf;
        g_dec_S = S; g_dec_Sp = Sp;
        g_enc_T = g_enc_S = -1;
    }

    ggml_backend_tensor_set(g.tok_in, canvas.data(), 0, sizeof(int) * T);
    std::vector<int> pos((size_t)T);
    for (int i = 0; i < T; i++) pos[(size_t)i] = n_past + i;
    ggml_backend_tensor_set(g.pos_in, pos.data(), 0, sizeof(int) * T);
    ggml_backend_tensor_set(g.sc_idx, sc_idx.data(), 0, sizeof(int) * sc_idx.size());
    ggml_backend_tensor_set(g.sc_w, sc_w.data(), 0, sizeof(float) * sc_w.size());
    fill_full(g.mask_full, n_past, T);
    fill_full(g.mask_swa, n_past, T);

    if (!graph_run(g.gf)) {
        fprintf(stderr, "diffusiongemma: decoder compute failed\n");
        return false;
    }
    logits.assign((size_t)n_vocab * T, 0.0f);
    ggml_backend_tensor_get(g.logits, logits.data(), 0, sizeof(float) * logits.size());
    if (!dump_dir.empty()) {
        vqa::dump::write(g.gf, dump_dir.c_str(), "t_final_norm");
        vqa::dump::write(g.gf, dump_dir.c_str(), "logits");
    }
    return true;
}

// ------------------------------------------------------------------ vision

// The core/engine dump sink relies on gallocr for a tensor's lifetime, which
// does not hold when the graph is allocated by the tower's own allocator: the
// dup it appends in a side context is invisible to the lifetime analysis, so
// the source tensors get recycled before the dup runs and the dump reads back
// garbage (that is why the first two parity attempts showed corr ~0). Marking
// them as graph OUTPUTS instead makes gallocr keep them alive; they are read
// right after compute.
static const char * const kVisionDumps[] = {
    "v_patches", "v_patch_pre", "v_patch_out", "v_l0_q_raw", "v_l0_q_norm", "v_l0_q_rope",
    "v_l0_attnraw", "v_l0_attn", "v_l0_res", "v_l0_out", "v_final", nullptr,
};

static void dump_graph_tensor(const std::string & dir, ggml_cgraph * gf, const char * name,
                              ggml_backend_t backend) {
    if (dir.empty()) return;
    ggml_tensor * t = ggml_graph_get_tensor(gf, name);
    if (!t || t->type != GGML_TYPE_F32) return;
    std::vector<float> buf((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (FILE * f = fopen((dir + "/cpp_" + name + ".bin").c_str(), "wb")) {
        fwrite(buf.data(), sizeof(float), buf.size(), f);
        fclose(f);
    }
    if (FILE * f = fopen((dir + "/cpp_" + name + ".shape").c_str(), "w")) {
        fprintf(f, "%lld %lld %lld %lld\n", (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3]);
        fclose(f);
    }
}

// host-side dumps for parity: the graph-side dump sinks cannot be trusted for
// this tower (the patch input tensor is a graph input whose buffer gallocr
// recycles), so the values that actually enter / leave each stage are written
// straight from host memory.
static void dump_host_f32(const std::string & dir, const char * name,
                          const std::vector<float> & v, int ne0, int ne1) {
    if (dir.empty()) return;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);   // first write creates it
    std::string p = dir + "/host_" + name + ".bin";
    if (FILE * f = fopen(p.c_str(), "wb")) {
        fwrite(v.data(), sizeof(float), v.size(), f);
        fclose(f);
    }
    std::string s = dir + "/host_" + name + ".shape";
    if (FILE * f = fopen(s.c_str(), "w")) {
        fprintf(f, "%d %d 1 1\n", ne0, ne1);
        fclose(f);
    }
}

bool DiffusionGemma::preprocess_image(const uint8_t * rgb, int w0, int h0,
                                      std::vector<float> & patches, std::vector<int> & px,
                                      std::vector<int> & py, int & gw, int & gh) {
    // Gemma4ImageProcessor.get_aspect_ratio_preserving_size: fit the patch
    // budget (max_soft_tokens * pooling^2 patches) and make both sides
    // divisible by pooling * patch_size, bicubic resample, x/255, no normalize
    const int P = v_patch;
    const int max_patches = v_soft * v_pool * v_pool;
    const double total_px = (double)h0 * (double)w0;
    const double target_px = (double)max_patches * (double)(P * P);
    const double factor = std::sqrt(target_px / total_px);
    const int side = v_pool * P;
    int th = (int)std::floor(factor * h0 / side) * side;
    int tw = (int)std::floor(factor * w0 / side) * side;
    const int max_side = (max_patches / (v_pool * v_pool)) * side;
    if (th == 0 && tw == 0) {
        fprintf(stderr, "diffusiongemma: image resizes to 0x0 (needs at least %dx%d)\n", side, side);
        return false;
    }
    if (th == 0) {
        th = side;
        tw = std::min((int)std::floor((double)w0 / (double)h0) * side, max_side);
        if (tw < side) tw = side;
    } else if (tw == 0) {
        tw = side;
        th = std::min((int)std::floor((double)h0 / (double)w0) * side, max_side);
        if (th < side) th = side;
    }
    if ((long long)th * tw > (long long)target_px + (long long)side * side) {
        fprintf(stderr, "diffusiongemma: resize %dx%d exceeds the patch budget\n", th, tw);
        return false;
    }

    // DG_VISION_GRID="w h" forces the patch grid (debug: shrinks the tower's
    // batch to isolate batch-size dependent backend problems)
    if (const char * e = getenv("DG_VISION_GRID")) {
        int ggw = 0, ggh = 0;
        if (sscanf(e, "%d %d", &ggw, &ggh) == 2 && ggw > 0 && ggh > 0) {
            tw = ggw * side;
            th = ggh * side;
            fprintf(stderr, "diffusiongemma: DG_VISION_GRID forcing %dx%d patches\n", ggw, ggh);
        }
    }

    std::vector<float> resized;
    vqa::img::resize_antialias(rgb, w0, h0, tw, th, resized, 3);   // bicubic, PIL semantics
    // rescale to [0,1] (do_rescale 1/255, no mean/std) and map to [-1,1]:
    // Gemma4VisionPatchEmbedder starts with 2 * (pixel - 0.5)
    const float inv = 1.0f / 255.0f;
    for (auto & v : resized) v = 2.0f * (std::min(1.0f, std::max(0.0f, v * inv)) - 0.5f);

    gw = tw / P;
    gh = th / P;
    const int np = gw * gh;
    if (np > v_pos_table) {
        fprintf(stderr, "diffusiongemma: %d patches exceeds the position table %d\n", np, v_pos_table);
        return false;
    }
    // patch layout matches HF convert_image_to_patches: (dy, dx, c) columns
    const int P3 = 3 * P * P;
    patches.assign((size_t)P3 * np, 0.0f);
    px.resize((size_t)np);
    py.resize((size_t)np);
    int u = 0;
    for (int cy = 0; cy < gh; cy++)
        for (int cx = 0; cx < gw; cx++, u++) {
            float * row = patches.data() + (size_t)u * P3;
            int o = 0;
            for (int dy = 0; dy < P; dy++) {
                const float * s = resized.data() + ((size_t)(cy * P + dy) * tw + cx * P) * 3;
                for (int dx = 0; dx < P; dx++)
                    for (int c = 0; c < 3; c++) row[o++] = s[(size_t)dx * 3 + c];
            }
            px[(size_t)u] = cx;
            py[(size_t)u] = cy;
        }
    // DG_VISION_CONST=1 replicates patch 0 into every column: if the engine's
    // per-column results differ, the problem is the batched mat-mul layout, not
    // the weights (all 1152 outputs must be identical for identical inputs)
    if (getenv("DG_VISION_CONST")) {
        for (int i = 1; i < np; i++)
            std::copy(patches.begin(), patches.begin() + P3, patches.begin() + (size_t)i * P3);
        for (int i = 0; i < np; i++) { px[(size_t)i] = 0; py[(size_t)i] = 0; }
        fprintf(stderr, "diffusiongemma: DG_VISION_CONST replicated patch 0 into %d columns "
                        "(P3=%d, %dx%d grid)\n", np, P3, gw, gh);
        fprintf(stderr, "  p0=%.6f %.6f %.6f   p1=%.6f %.6f %.6f   p_all[last]=%.6f\n",
                patches[0], patches[1], patches[2],
                patches[(size_t)P3], patches[(size_t)P3 + 1], patches[(size_t)P3 + 2],
                patches[(size_t)P3 * (np - 1)]);
    }
    return true;
}

// axial 2-D rope: head_dim splits in half, the first half rotates on the patch's
// x coordinate and the second on its y (HF recomposes the frequency table as
// [h,h,w,w] and then splits the head into two 36-dim halves, each rotated with
// the NEOX half-split pairing)
static ggml_tensor * axial_rope(ggml_context * ctx, ggml_tensor * t,
                                ggml_tensor * pos_x, ggml_tensor * pos_y,
                                int hd, float theta, int max_pos) {
    const int half = hd / 2;
    ggml_tensor * a = ggml_view_3d(ctx, t, half, t->ne[1], t->ne[2], t->nb[1], t->nb[2], 0);
    ggml_tensor * b = ggml_view_3d(ctx, t, half, t->ne[1], t->ne[2], t->nb[1], t->nb[2],
                                   (size_t)half * t->nb[0]);
    ggml_tensor * ra = ggml_rope_ext(ctx, ggml_cont(ctx, a), pos_x, nullptr, half,
                                     GGML_ROPE_TYPE_NEOX, max_pos, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * rb = ggml_rope_ext(ctx, ggml_cont(ctx, b), pos_y, nullptr, half,
                                     GGML_ROPE_TYPE_NEOX, max_pos, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * out = ggml_concat(ctx, ra, rb, 0);
    return ggml_cont(ctx, out);
}

bool DiffusionGemma::build_vision_graph(int np) {
    if (vctx) { ggml_free(vctx); vctx = nullptr; }
    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    vctx = ggml_init(ip);
    if (!vctx) return false;
    vgf = ggml_new_graph_custom(vctx, 8192, false);
    ggml_context * c = vctx;

    const int P3 = 3 * v_patch * v_patch;
    VisGraph vg;
    vg.ctx = c;
    vg.gf = vgf;
    vg.np = np;
    // F16 on purpose: the tower weights are F16, and feeding an F32 activation
    // makes ggml convert it into a vec_dot buffer first. With 2000+ patch
    // columns that conversion path is what produced mismatched columns here, so
    // the input is handed over in the weights' own type instead.
    vg.patches = ggml_new_tensor_2d(c, GGML_TYPE_F16, P3, np);
    ggml_set_name(vg.patches, "v_patches");
    ggml_set_input(vg.patches);
    vg.px = ggml_new_tensor_1d(c, GGML_TYPE_I32, np);
    ggml_set_name(vg.px, "v_px");
    ggml_set_input(vg.px);
    vg.py = ggml_new_tensor_1d(c, GGML_TYPE_I32, np);
    ggml_set_name(vg.py, "v_py");
    ggml_set_input(vg.py);

    // patches arrive already mapped to [-1,1] by preprocess_image (HF does
    // pixel_values = 2 * (x - 0.5) inside Gemma4VisionPatchEmbedder)
    // DG_VISION_SKIP=mlp|attn|all|pos drops one piece of the tower so a
    // divergence can be attributed without trusting in-graph dumps (this graph
    // cannot use them: its gallocr recycles inputs)
    bool skip_mlp = false, skip_attn = false, skip_pos = false;
    if (const char * e = getenv("DG_VISION_SKIP")) {
        const bool all = strcmp(e, "all") == 0;
        skip_mlp = all || strcmp(e, "mlp") == 0;
        skip_attn = all || strcmp(e, "attn") == 0;
        skip_pos = all || strcmp(e, "pos") == 0;
    }

    ggml_tensor * x = ggml_mul_mat(c, v_dense, vg.patches);   // [1152, np]
    ggml_set_name(x, "v_patch_pre");
    {
        // learned 2-D position table [hidden, pos, 2]: slice the two halves
        // along ne2 and gather by the patch's x / y grid coordinate
        ggml_tensor * xt = ggml_view_2d(c, v_pos_table_t, v_embd, v_pos_table,
                                        v_pos_table_t->nb[1], 0);
        ggml_tensor * yt = ggml_view_2d(c, v_pos_table_t, v_embd, v_pos_table,
                                        v_pos_table_t->nb[1], v_pos_table_t->nb[2]);
        ggml_tensor * xe = ggml_get_rows(c, xt, vg.px);
        ggml_tensor * ye = ggml_get_rows(c, yt, vg.py);
        if (!skip_pos) x = ggml_add(c, x, ggml_add(c, xe, ye));
    }
    ggml_set_name(x, "v_patch_out");   // patch embedding INCLUDING positions

    // DG_VISION_LAYERS=N truncates the tower after N blocks: the host-side
    // `host_v_final` dump then holds block N-1's output, which is what
    // `ref_v_l0_out` / a truncated oracle run exposes - the cheapest way to
    // bisect a divergence inside 27 blocks.
    int v_layers_run = v_layer;
    if (const char * e = getenv("DG_VISION_LAYERS")) {
        const int n = atoi(e);
        if (n > 0 && n < v_layers_run) v_layers_run = n;
    }
    for (int il = 0; il < v_layers_run; il++) {
        VisLayer & L = vl[(size_t)il];
        ggml_tensor * cur = ggml_mul(c, ggml_rms_norm(c, x, v_eps), L.attn_norm);
        ggml_tensor * q = ggml_mul_mat(c, L.wq, cur);
        ggml_tensor * k = ggml_mul_mat(c, L.wk, cur);
        ggml_tensor * v = ggml_mul_mat(c, L.wv, cur);
        if (il == 0) ggml_set_name(q, "v_l0_q_raw");
        q = ggml_reshape_3d(c, q, v_hd, v_head, np);
        k = ggml_reshape_3d(c, k, v_hd, v_kv, np);
        v = ggml_reshape_3d(c, v, v_hd, v_kv, np);
        q = ggml_mul(c, ggml_rms_norm(c, q, v_eps), L.q_norm);
        if (il == 0) ggml_set_name(q, "v_l0_q_norm");
        k = ggml_mul(c, ggml_rms_norm(c, k, v_eps), L.k_norm);
        v = ggml_rms_norm(c, v, v_eps);
        if (!getenv("DG_VISION_NOROPE")) {   // debug: isolate the axial RoPE
            q = axial_rope(c, q, vg.px, vg.py, v_hd, v_theta, v_pos_table);
            k = axial_rope(c, k, vg.px, vg.py, v_hd, v_theta, v_pos_table);
        }
        if (il == 0) ggml_set_name(q, "v_l0_q_rope");
        // flash_attn_ext wants [head_dim, n_tokens, n_heads] while the tensors are
        // [head_dim, n_heads, n_tokens], so the two strides swap with the dims
        ggml_tensor * qf = ggml_view_3d(c, q, v_hd, np, v_head, q->nb[2], q->nb[1], 0);
        ggml_tensor * kf = ggml_view_3d(c, k, v_hd, np, v_kv, k->nb[2], k->nb[1], 0);
        ggml_tensor * vf = ggml_view_3d(c, v, v_hd, np, v_kv, v->nb[2], v->nb[1], 0);
        // bidirectional over all patches, no padding (batch of one image)
        ggml_tensor * fa = ggml_flash_attn_ext(c, qf, kf, vf, nullptr, 1.0f, 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(c, ggml_cont(c, fa), v_head * v_hd, np);
        if (il == 0) ggml_set_name(attn, "v_l0_attnraw");
        attn = ggml_mul_mat(c, L.wo, attn);
        if (il == 0) ggml_set_name(attn, "v_l0_attn");
        ggml_tensor * res = skip_attn
            ? x
            : ggml_add(c, x, ggml_mul(c, ggml_rms_norm(c, attn, v_eps), L.attn_post_norm));
        if (il == 0) ggml_set_name(res, "v_l0_res");
        ggml_tensor * f = ggml_mul(c, ggml_rms_norm(c, res, v_eps), L.ffn_norm);
        ggml_tensor * fg = ggml_mul_mat(c, L.w_gate, f);
        ggml_tensor * fu = ggml_mul_mat(c, L.w_up, f);
        ggml_tensor * fd = ggml_mul_mat(c, L.w_down, ggml_geglu_split(c, fg, fu));
        x = skip_mlp
            ? res
            : ggml_add(c, res, ggml_mul(c, ggml_rms_norm(c, fd, v_eps), L.ffn_post_norm));
        if (il == 0) ggml_set_name(x, "v_l0_out");
    }
    ggml_set_name(x, "v_final");
    vg.out = x;
    // the graph needs an explicit output expansion: without it nothing is in the
    // graph and ggml_graph_get_tensor("v_final") comes back null
    ggml_build_forward_expand(vgf, x);

    // mark the parity dumps as graph outputs BEFORE allocating: gallocr reads
    // those flags when it plans buffer reuse
    if (!dump_dir.empty()) {
        for (int i = 0; kVisionDumps[i]; i++)
            if (ggml_tensor * t = ggml_graph_get_tensor(vgf, kVisionDumps[i])) ggml_set_output(t);
    }

    // DG_VISION_CTXALLOC allocates the whole tower ctx at once instead of going
    // through the reuse-planning gallocr. The tower's tensors are short-lived
    // per image, so a flat allocation costs little and rules the allocator out
    // when results disagree with a hand-computed reference.
    if (getenv("DG_VISION_CTXALLOC")) {
        if (vbuf) { ggml_backend_buffer_free(vbuf); vbuf = nullptr; }
        vbuf = ggml_backend_alloc_ctx_tensors(vctx, backend);
        if (!vbuf) {
            fprintf(stderr, "diffusiongemma: vision ctx allocation failed\n");
            return false;
        }
        fprintf(stderr, "diffusiongemma: vision graph allocated flat (%.1f MB)\n",
                (double)ggml_backend_buffer_get_size(vbuf) / 1e6);
    } else {
        if (!vallocr) vallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!vallocr || !ggml_gallocr_alloc_graph(vallocr, vgf)) {
            fprintf(stderr, "diffusiongemma: vision graph allocation failed\n");
            return false;
        }
    }
    vg_np = np;
    return true;
}

bool DiffusionGemma::run_vision(const std::vector<float> & patches, const std::vector<int> & px,
                                const std::vector<int> & py, int np, std::vector<float> & hidden) {
    if (!has_vision) return false;
    if (vg_np != np) {
        if (!build_vision_graph(np)) return false;
    }
    ggml_tensor * tp = ggml_graph_get_tensor(vgf, "v_patches");
    ggml_tensor * tpx = ggml_graph_get_tensor(vgf, "v_px");
    ggml_tensor * tpy = ggml_graph_get_tensor(vgf, "v_py");
    ggml_tensor * tout = ggml_graph_get_tensor(vgf, "v_final");
    // the position inputs drop out of the graph when the debug switches skip
    // them, so guard every set (they are graph inputs, not required here)
    std::vector<ggml_fp16_t> patches16;
    if (tp && tp->type == GGML_TYPE_F16) {
        patches16.resize(patches.size());
        for (size_t i = 0; i < patches.size(); i++) patches16[i] = ggml_fp32_to_fp16(patches[i]);
        ggml_backend_tensor_set(tp, patches16.data(), 0, sizeof(ggml_fp16_t) * patches16.size());
    } else if (tp) {
        ggml_backend_tensor_set(tp, patches.data(), 0, sizeof(float) * patches.size());
    }
    if (tpx) ggml_backend_tensor_set(tpx, px.data(), 0, sizeof(int) * px.size());
    if (tpy) ggml_backend_tensor_set(tpy, py.data(), 0, sizeof(int) * py.size());
    if (!tout) {
        fprintf(stderr, "diffusiongemma: vision graph has no v_final output\n");
        return false;
    }
    auto echo_in = [&](const char * when) {
        if (!getenv("DG_VISION_ECHO") || !tp) return;
        const size_t n = (size_t)ggml_nelements(tp);
        std::vector<ggml_fp16_t> back(n);
        ggml_backend_tensor_get(tp, back.data(), 0, n * sizeof(ggml_fp16_t));
        const size_t cols = (size_t)np;
        const size_t p3 = (size_t)tp->ne[0];
        fprintf(stderr, "[echo %s] p0=%.4f p1=%.4f pLast=%.4f  (cols=%zu, P3=%zu)\n", when,
                (double)ggml_fp16_to_fp32(back[0]),
                (double)ggml_fp16_to_fp32(back[p3]),
                (double)ggml_fp16_to_fp32(back[p3 * (cols - 1)]), cols, p3);
    };
    echo_in("after-upload");

    if (!graph_run_dev(vgf)) {
        fprintf(stderr, "diffusiongemma: vision compute failed\n");
        return false;
    }
    echo_in("after-compute");
    hidden.assign((size_t)v_embd * np, 0.0f);
    ggml_backend_tensor_get(tout, hidden.data(), 0, sizeof(float) * hidden.size());
    if (!dump_dir.empty()) {
        // write the REAL dims of the output tensor (not an assumption): a silent
        // transpose here is exactly what makes a correct result look wrong
        dump_host_f32(dump_dir, "v_final", hidden, (int)tout->ne[0], (int)tout->ne[1]);
        if (getenv("DG_VISION_ECHO"))
            fprintf(stderr, "[vision] v_final ne = %lld x %lld x %lld (np=%d)\n",
                    (long long)tout->ne[0], (long long)tout->ne[1], (long long)tout->ne[2], np);
        for (int i = 0; kVisionDumps[i]; i++)
            if (strcmp(kVisionDumps[i], "v_final")) dump_graph_tensor(dump_dir, vgf, kVisionDumps[i], backend);
        // patch grid coordinates for the oracle (the graph carries them as I32,
        // which the F32 dump writer cannot represent)
        std::string pf = dump_dir + "/v_pxpy.txt";
        if (FILE * f = fopen(pf.c_str(), "w")) {
            fprintf(f, "%d\n", np);
            for (int i = 0; i < np; i++) fprintf(f, "%d %d\n", px[(size_t)i], py[(size_t)i]);
            fclose(f);
        }
    }
    return true;
}

bool DiffusionGemma::build_proj_graph(int nt) {
    if (pctx) {
        if (pbuf) { ggml_backend_buffer_free(pbuf); pbuf = nullptr; }
        ggml_free(pctx); pctx = nullptr;
    }
    struct ggml_init_params ip = { 8ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    pctx = ggml_init(ip);
    if (!pctx) return false;
    pgf = ggml_new_graph_custom(pctx, 128, false);
    ggml_tensor * in = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, v_embd, nt);
    ggml_set_name(in, "p_in");
    ggml_tensor * y = ggml_rms_norm(pctx, in, v_eps);           // weightless pre-projection norm
    y = ggml_mul_mat(pctx, v_mm_proj, y);                      // [2816, nt]
    ggml_set_name(y, "v_soft");
    ggml_build_forward_expand(pgf, y);
    if (!dump_dir.empty()) {
        static const char * const pn[] = { "p_in", "v_soft", nullptr };
        vqa::dump::mark_sinks(pctx, pgf, pn);
        vqa::dump::finalize_sinks(pctx, pgf, backend);
    }
    // dedicated buffer (not the shared gallocr): the projection runs between
    // encoder passes and must never alias the text graph's activations
    pbuf = ggml_backend_alloc_ctx_tensors(pctx, backend);
    if (!pbuf) {
        fprintf(stderr, "diffusiongemma: projection buffer allocation failed\n");
        return false;
    }
    pg_nt = nt;
    return true;
}

bool DiffusionGemma::run_proj(const std::vector<float> & soft_in, int nt, std::vector<float> & soft_out) {
    if (pg_nt != nt) {
        if (!build_proj_graph(nt)) return false;
    }
    ggml_tensor * in = ggml_graph_get_tensor(pgf, "p_in");
    ggml_tensor * out = ggml_graph_get_tensor(pgf, "v_soft");
    ggml_backend_tensor_set(in, soft_in.data(), 0, sizeof(float) * soft_in.size());
    if (!graph_run_dev(pgf)) {
        fprintf(stderr, "diffusiongemma: projection compute failed\n");
        return false;
    }
    soft_out.assign((size_t)n_embd * nt, 0.0f);
    ggml_backend_tensor_get(out, soft_out.data(), 0, sizeof(float) * soft_out.size());
    if (!dump_dir.empty()) vqa::dump::write(pgf, dump_dir.c_str(), "v_soft");
    return true;
}

// image -> v_soft rows of language-space embeddings (pooler + standardize on
// the host: both are pure per-channel reductions, and doing them here keeps the
// graph free of gather-heavy one-hot pooling)
bool DiffusionGemma::encode_image(const vqa::MediaInput & media, std::vector<float> & soft, int & n_soft) {
    if (!has_vision || media.frames.empty()) return false;
    const vqa::Frame & fr = media.frames[0];
    std::vector<float> patches;
    std::vector<int> px, py;
    int gw = 0, gh = 0;
    if (!preprocess_image(fr.rgb.data(), media.width, media.height, patches, px, py, gw, gh)) return false;
    const int np = gw * gh;
    dump_host_f32(dump_dir, "v_patches", patches, 3 * v_patch * v_patch, np);
    std::vector<float> hidden;
    if (!run_vision(patches, px, py, np, hidden)) return false;

    // 3x3 spatial mean pool over the patch grid (pooling_kernel_size), then
    // * sqrt(hidden) and (x - std_bias) * std_scale - all in F32 like HF
    const int k = v_pool;
    const int ogw = gw / k, ogh = gh / k;
    n_soft = ogw * ogh;
    if (n_soft <= 0) return false;
    const float root = std::sqrt((float)v_embd);
    std::vector<float> pooled((size_t)v_embd * n_soft, 0.0f);
    for (int oy = 0; oy < ogh; oy++)
        for (int ox = 0; ox < ogw; ox++) {
            const int slot = oy * ogw + ox;
            float * dst = pooled.data() + (size_t)slot * v_embd;
            for (int dy = 0; dy < k; dy++)
                for (int dx = 0; dx < k; dx++) {
                    const int src_patch = (oy * k + dy) * gw + (ox * k + dx);
                    const float * src = hidden.data() + (size_t)src_patch * v_embd;
                    for (int c = 0; c < v_embd; c++) dst[c] += src[c];
                }
            const float inv = 1.0f / (float)(k * k);
            for (int c = 0; c < v_embd; c++) dst[c] *= inv * root;
        }
    // standardize with the checkpoint's per-channel bias/scale
    {
        std::vector<float> bias((size_t)v_embd), scale((size_t)v_embd);
        ggml_backend_tensor_get(v_std_bias, bias.data(), 0, sizeof(float) * bias.size());
        ggml_backend_tensor_get(v_std_scale, scale.data(), 0, sizeof(float) * scale.size());
        for (int t = 0; t < n_soft; t++) {
            float * row = pooled.data() + (size_t)t * v_embd;
            for (int c = 0; c < v_embd; c++) row[c] = (row[c] - bias[(size_t)c]) * scale[(size_t)c];
        }
    }
    if (!run_proj(pooled, n_soft, soft)) return false;
    dump_host_f32(dump_dir, "v_pooled", pooled, v_embd, n_soft);
    dump_host_f32(dump_dir, "v_soft", soft, n_embd, n_soft);
    return true;
}

// ------------------------------------------------------------------ sampler

// per-column softmax + multinomial proposal + entropy, single image (T columns
// of `n_vocab` logits). Parallel over columns: each column is independent.
bool DiffusionGemma::sample_canvas(std::vector<float> & logits, std::vector<int> & canvas,
                                   std::vector<int> & argmax_canvas, std::vector<int> & sc_idx,
                                   std::vector<float> & sc_w, int n_sc, float temperature,
                                   std::vector<float> & entropy, uint64_t & rng) {
    const int T = (int)canvas.size();
    const int V = n_vocab;
    if ((int)logits.size() < (size_t)V * T) return false;
    entropy.assign((size_t)T, 0.0f);
    argmax_canvas.assign((size_t)T, 0);
    std::vector<int> proposal((size_t)T, 0);

    int n_thr = (int)std::thread::hardware_concurrency();
    if (n_thr < 1) n_thr = 1;
    if (n_thr > T) n_thr = T;
    std::vector<std::thread> pool;
    std::vector<uint64_t> seeds((size_t)T);
    for (int t = 0; t < T; t++) {
        seeds[(size_t)t] = rng + 0x9E3779B97F4A7C15ull * (uint64_t)(t + 1);
    }
    auto work = [&](int t0, int t1) {
        for (int t = t0; t < t1; t++) {
            float * lg = logits.data() + (size_t)t * V;
            float mx = -INFINITY;
            for (int v = 0; v < V; v++) {
                const float s = lg[v] / temperature;
                lg[v] = s;
                if (s > mx) { mx = s; argmax_canvas[(size_t)t] = v; }
            }
            double sum = 0.0;
            for (int v = 0; v < V; v++) sum += std::exp((double)(lg[v] - mx));
            const double lse = (double)mx + std::log(sum);
            double H = 0.0;
            for (int v = 0; v < V; v++) {
                const double p = std::exp((double)lg[v] - lse);
                lg[v] = (float)p;                     // reuse the buffer as probabilities
                if (p > 1e-12) H -= p * std::log(p);
            }
            entropy[(size_t)t] = (float)H;
            // multinomial proposal (xorshift64* on a per-column seed)
            uint64_t s = seeds[(size_t)t];
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            seeds[(size_t)t] = s;
            const double u = (double)((s * 2685821657736338717ull) >> 11) * (1.0 / 9007199254740992.0);
            double acc = 0.0;
            int pick = V - 1;
            for (int v = 0; v < V; v++) {
                acc += (double)lg[v];
                if (acc >= u) { pick = v; break; }
            }
            proposal[(size_t)t] = pick;
        }
    };
    if (n_thr == 1) {
        work(0, T);
    } else {
        const int chunk = (T + n_thr - 1) / n_thr;
        for (int i = 0; i < n_thr; i++) {
            const int t0 = i * chunk;
            const int t1 = std::min(T, t0 + chunk);
            if (t0 >= t1) break;
            pool.emplace_back(work, t0, t1);
        }
        for (auto & th : pool) th.join();
    }

    // entropy-bound acceptance: keep a prefix of the sorted-by-entropy slots
    // whose joint mutual-information bound (cumsum - max) stays under the bound
    std::vector<int> order((size_t)T);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return entropy[(size_t)a] < entropy[(size_t)b]; });
    std::vector<uint8_t> accept((size_t)T, 0);
    double cum = 0.0;
    for (int i = 0; i < T; i++) {
        const int t = order[(size_t)i];
        cum += entropy[(size_t)t];
        if (cum - entropy[(size_t)t] <= diff.entropy_bound) accept[(size_t)t] = 1;
    }
    for (int t = 0; t < T; t++) {
        if (accept[(size_t)t]) canvas[(size_t)t] = proposal[(size_t)t];
        else {
            uint64_t s = seeds[(size_t)t] ^ 0x2545F4914F6CDD1Dull * (uint64_t)(t + 1);
            s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
            seeds[(size_t)t] = s;
            canvas[(size_t)t] = (int)((s * 2685821657736338717ull) >> 33) % V;
        }
    }
    rng = seeds[(size_t)(T - 1)] ^ 0xD1B54A32D192ED03ull;

    if (getenv("DG_DEBUG") && T > 0) {
        // what does the model actually predict for the FIRST canvas slot?
        const float * pr = logits.data();
        std::vector<int> top(5, -1);
        for (int v = 0; v < V; v++) {
            for (int s = 0; s < 5; s++) {
                if (top[(size_t)s] < 0 || pr[v] > pr[top[(size_t)s]]) {
                    for (int m = 4; m > s; m--) top[(size_t)m] = top[(size_t)m - 1];
                    top[(size_t)s] = v;
                    break;
                }
            }
        }
        fprintf(stderr, "[dg] slot0 top5:");
        for (int s = 0; s < 5; s++) {
            const std::string piece = tok_.decode(std::vector<int>{ top[(size_t)s] });
            fprintf(stderr, " %d[%s](%.3f)", top[(size_t)s], piece.c_str(), pr[top[(size_t)s]]);
        }
        fprintf(stderr, "  entropy=%.3f\n", entropy.empty() ? -1.0f : entropy[0]);
    }

    // self-conditioning payload: top-k probabilities + their token ids, taken
    // from THIS step's processed logits (HF feeds the full softmax back; the
    // top-k truncation is the only deviation, see the file header)
    if (n_sc > 0) {
        int K = n_sc;
        if (K > V) K = V;
        sc_idx.assign((size_t)K * T, 0);
        sc_w.assign((size_t)K * T, 0.0f);
        std::vector<int> idx((size_t)V);
        for (int t = 0; t < T; t++) {
            const float * pr = logits.data() + (size_t)t * V;
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                              [&](int a, int b) { return pr[a] > pr[b]; });
            // renormalize over the kept mass so the conditioning stays a
            // proper expectation (HF's full softmax sums to 1)
            double mass = 0.0;
            for (int k = 0; k < K; k++) mass += (double)pr[idx[(size_t)k]];
            const double inv = mass > 0.0 ? 1.0 / mass : 0.0;
            for (int k = 0; k < K; k++) {
                // layout matches ggml: [K, T] / [1, K, T] index with ne1 = k
                const size_t at = (size_t)t * K + (size_t)k;
                sc_idx[at] = idx[(size_t)k];
                sc_w[at] = (float)((double)pr[idx[(size_t)k]] * inv);
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------- generate

bool DiffusionGemma::generate(const std::vector<int> & prompt_ids, const std::vector<int> & soft_pos,
                              const std::vector<float> & soft, const vqa::GenParams & gp,
                              std::string & out_text) {
    out_text.clear();
    n_past = 0;
    kv.clear();
    const auto t_gen0 = std::chrono::steady_clock::now();
    int n_steps = 0;                 // denoising passes (the real cost unit here)
    // prefill the prompt causally (vision rows bidirectional inside their span)
    {
        std::vector<uint8_t> media_rows;
        if (!soft.empty() && id_image >= 0) {
            media_rows.assign(prompt_ids.size(), 0);
            int first = -1, last = -1;
            for (size_t i = 0; i < prompt_ids.size(); i++) {
                if (prompt_ids[i] == id_image) { media_rows[i] = 1; if (first < 0) first = (int)i; last = (int)i; }
            }
            // the <|image> / <image|> markers bracket the span
            if (first > 0 && prompt_ids[(size_t)(first - 1)] == id_boi) first--;
            if (last + 1 < (int)prompt_ids.size() && prompt_ids[(size_t)(last + 1)] == id_eoi) last++;
            if (!run_encoder(prompt_ids, soft_pos, soft, &media_rows, first, last + 1)) return false;
        } else {
            if (!run_encoder(prompt_ids, soft_pos, soft, nullptr, 0, 0)) return false;
        }
    }
    const auto t_pre = std::chrono::steady_clock::now();

    const int T = canvas_len;
    // self-conditioning truncation: HF feeds the FULL softmax back as the
    // expected embedding, which costs a 262144-wide gather per canvas slot per
    // step. The top-k renormalized expectation is the shipped approximation
    // (DG_SC_TOPK widens it; k = vocab recovers the exact HF behaviour).
    int n_sc = 256;
    if (const char * e = getenv("DG_SC_TOPK")) { n_sc = atoi(e); if (n_sc < 1) n_sc = 1; }
    if (n_sc > n_vocab) n_sc = n_vocab;
    std::vector<int> canvas((size_t)T), argmax_canvas((size_t)T);
    std::vector<int> sc_idx, sc_w_next_idx;
    std::vector<float> sc_w, entropy;
    uint64_t rng = 0x9E3779B97F4A7C15ull ^ (uint64_t)(gp.seed >= 0 ? gp.seed : 12345);
    // canvas starts from a uniform random draw (HF: sampler.initialize_canvas)
    for (int t = 0; t < T; t++) {
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
        canvas[(size_t)t] = (int)((rng * 2685821657736338717ull) >> 33) % n_vocab;
    }
    sc_idx.assign((size_t)n_sc * T, 0);
    sc_w.assign((size_t)n_sc * T, 0.0f);

    std::vector<int> all = prompt_ids;
    int n_canvas = 0;
    const int max_canvas = std::max(1, (gp.max_tokens + T - 1) / T);

    for (int cv = 0; cv < max_canvas; cv++) {
        std::vector<float> logits;
        std::vector<int> prev_argmax;
        int stable = 0;
        bool stopped = false;
        // the first denoising step of every canvas runs with a zero
        // conditioning signal (HF passes self_conditioning_logits=None there)
        std::vector<float> last_sc_w((size_t)n_sc * T, 0.0f);
        std::vector<int> last_sc_idx((size_t)n_sc * T, 0);
        for (int step = diff.max_steps; step >= 1 && !stopped; step--) {
            if (!run_decoder(canvas, last_sc_idx, last_sc_w, n_sc, logits)) return false;
            n_steps++;
            const float temperature = diff.t_min + (diff.t_max - diff.t_min) * ((float)step / (float)diff.max_steps);
            std::vector<int> next_idx;
            std::vector<float> next_w;
            if (!sample_canvas(logits, canvas, argmax_canvas, next_idx, next_w, n_sc,
                               temperature, entropy, rng)) return false;
            // stability: the argmax canvas must repeat `stability_thr` steps
            if (prev_argmax == argmax_canvas) stable++;
            else stable = 0;
            prev_argmax = argmax_canvas;
            // confidence: mean entropy of the processed logits
            double mean_h = 0.0;
            for (float h : entropy) mean_h += h;
            mean_h /= (double)entropy.size();
            if (getenv("DG_DEBUG"))
                fprintf(stderr, "[dg] canvas %d step %2d  T=%.2f  meanH=%.4f  stable=%d  "
                                "tok(top1 prob)=%.3f\n",
                        cv, step, temperature, mean_h, stable, 1.0f);
            if (stable >= diff.stability_thr && mean_h < diff.confidence_thr) {
                if (getenv("DG_DEBUG"))
                    fprintf(stderr, "[dg] canvas %d stop at step %d (mean H %.5f)\n", cv, step, mean_h);
                stopped = true;
            }
            last_sc_idx.swap(next_idx);
            last_sc_w.swap(next_w);
            if (gp.on_stage) gp.on_stage("denoise", diff.max_steps - step + 1, diff.max_steps, gp.on_stage_ud);
        }

        // commit the argmax canvas: append, stop at EOS, then re-encode
        int n_new = 0;
        bool hit_eos = false;
        for (int t = 0; t < T; t++) {
            const int tok = argmax_canvas[(size_t)t];
            if (tok == id_eos || tok == id_eot) { hit_eos = true; break; }
            all.push_back(tok);
            n_new++;
        }
        if (n_new > 0) {
            std::vector<int> chunk(all.end() - n_new, all.end());
            // the committed tokens go back through the encoder: this overwrites
            // the decoder's scratch KV rows at [n_past, n_past + n_new)
            std::vector<int> none;
            std::vector<float> nof;
            if (!run_encoder(chunk, none, nof, nullptr, 0, 0)) return false;
        }
        n_canvas++;
        if (hit_eos) break;
        if ((int)all.size() - (int)prompt_ids.size() >= gp.max_tokens) break;
        if (n_past + T > n_ctx) {
            fprintf(stderr, "diffusiongemma: context window exhausted (%d + %d > %d)\n", n_past, T, n_ctx);
            break;
        }
    }

    // detokenize (streaming) and strip the thought channel. The channel is
    // delimited by TOKENS (<|channel> thought ... <channel|>), and the special
    // tokens decode to "", so string matching cannot find the opener - filter ids.
    const size_t start = prompt_ids.size();
    std::vector<int> gen;
    {
        bool in_thought = false;
        for (size_t i = start; i < all.size(); i++) {
            const int id = all[i];
            if (id == id_channel) { in_thought = true; continue; }
            if (id == id_channel_end) { in_thought = false; continue; }
            if (!in_thought) gen.push_back(id);
        }
    }
    // Decoding grows the id prefix instead of single tokens so multi-byte UTF-8
    // pieces never split.
    std::string emitted;
    for (size_t k = 0; k < gen.size(); k += 8) {
        const size_t e = std::min(gen.size(), k + 8);
        std::string sofar = tok_.decode(std::vector<int>(gen.begin(), gen.begin() + (long)e));
        if (gp.on_token && sofar.size() > emitted.size())
            gp.on_token(sofar.c_str() + emitted.size(), gp.on_token_ud);
        emitted.swap(sofar);
    }
    out_text = emitted;
    while (!out_text.empty() && (out_text.front() == '\n' || out_text.front() == ' ')) out_text.erase(out_text.begin());
    // Cost unit is the denoising pass, not the token: one canvas of canvas_len
    // tokens costs up to max_steps decoder passes. tok/s is reported for
    // comparison with the other backends, steps/s and s/canvas for tuning.
    {
        const auto t_end = std::chrono::steady_clock::now();
        const double ms_pre = std::chrono::duration<double, std::milli>(t_pre - t_gen0).count();
        const double ms_dec = std::chrono::duration<double, std::milli>(t_end - t_pre).count();
        const size_t n_tok = all.size() - start;
        fprintf(stderr, "[perf] prefill %d tok in %.0f ms | decode %zu tok in %.0f ms "
                        "(%.2f tok/s, %.2f s/canvas, %d canvases, %d steps, %.1f ms/step)\n",
                (int)prompt_ids.size(), ms_pre, n_tok, ms_dec,
                ms_dec > 0 ? n_tok * 1000.0 / ms_dec : 0.0,
                n_canvas > 0 ? ms_dec / 1000.0 / n_canvas : 0.0,
                n_canvas, n_steps, n_steps > 0 ? ms_dec / n_steps : 0.0);
    }
    return true;
}

bool DiffusionGemma::ask(const vqa::MediaInput & media, const std::string & prompt,
                         const std::string & mode, const vqa::GenParams & gp,
                         std::string & out_text) {
    (void)mode;
    std::vector<float> soft;
    int n_soft = 0;
    if (!media.empty()) {
        if (!has_vision) {
            fprintf(stderr, "diffusiongemma: this GGUF has no vision tower (text-only)\n");
            return false;
        }
        if (media.kind == vqa::MediaKind::Video && media.frames.size() > 1) {
            fprintf(stderr, "diffusiongemma: video is decoded as a single image chunk "
                            "(the checkpoint has one image path); using frame 0\n");
        }
        // tower + pooling + projection; reported separately because it is the only
        // part that runs on the card in the default placement
        const auto t_vis0 = std::chrono::steady_clock::now();
        if (!encode_image(media, soft, n_soft)) return false;
        const double ms_vis = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_vis0).count();
        fprintf(stderr, "[perf] vision %d soft tokens in %.0f ms on %s\n", n_soft, ms_vis,
                ggml_backend_name(backend));
        // DG_VISION_ONLY=1 stops right after the tower: used to bisect the
        // vision path without paying for the 26B text pass
        if (getenv("DG_VISION_ONLY")) {
            fprintf(stderr, "diffusiongemma: DG_VISION_ONLY set, stopping after the vision tower\n");
            return false;
        }
    }

    // chat template: optional system turn (thinking is off by default for the
    // -it checkpoint), then the user turn, then the generation prompt
    std::vector<int> ids;
    auto push = [&](const std::string & s) {
        for (int t : tok_.encode(s)) ids.push_back(t);
    };
    if (!gp.history.empty()) {
        for (int t : tok_.encode(gp.history)) ids.push_back(t);
    } else {
        push("<|turn>system\n");
        push("<turn|>\n");
    }
    push("<|turn>user\n");
    std::vector<int> soft_pos;
    if (n_soft > 0) {
        const int at = id_boi >= 0 ? id_boi : id_image;
        ids.push_back(at);
        for (int i = 0; i < n_soft; i++) {
            soft_pos.push_back((int)ids.size());
            ids.push_back(id_image);
        }
        ids.push_back(id_eoi);
    }
    for (int t : tok_.encode(prompt)) ids.push_back(t);
    push("<turn|>\n");
    push("<|turn>model\n");

    if (n_soft > 0 && (int)soft.size() != n_soft * n_embd) {
        fprintf(stderr, "diffusiongemma: vision projection size mismatch\n");
        return false;
    }
    return generate(ids, soft_pos, soft, gp, out_text);
}

}  // namespace dgemma

// factory entry point (declared in core/runtime/backend_factories.hpp)
namespace vqa {
Model * create_diffusiongemma() { return new dgemma::DiffusionGemma(); }
}  // namespace vqa
