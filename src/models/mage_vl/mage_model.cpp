// mage_model.cpp - Microsoft Mage-VL-4B backend (Qwen3-4B text decoder + Mage-ViT Codec-ViT).
//
// Conventions (same as the other backends):
//   * GGUF linear weights are stored as ne = [in, out]; consumption is
//     ggml_mul_mat(W, x[in, T]) -> [out, T].
//   * the text decoder is fed a host-assembled F32 embedding matrix [n_embd, T]
//     (token rows read back from the possibly quantized token_embd, vision rows
//     spliced in); text stack = plain Qwen3 (1D RoPE NEOX, qk-norm, SwiGLU).
//   * text attention uses a per-layer F16 KV cache [n_kv_head*head_dim, n_ctx]
//     with flash attention and an explicit F16 causal mask; single-token decode
//     runs a static graph eligible for CUDA graph capture.
//   * the vision tower (Mage-ViT, 24 blocks) runs in frame groups: patches are
//     laid out in 2x2 block order, attention is block-diagonal per frame
//     (frames backend: one chunk per frame; codec windows: frame_window=4
//     frames per chunk). RoPE is per-dim (4:6:6 T:H:W split) - built in-graph
//     from host cos/sin tables, no weight permutation needed.
//   * the merger is a straight reshape: block layout puts each 2x2 merge group
//     in 4 consecutive rows, so [N,1024] -> LN -> [N/4,4096] -> MLP -> [N/4,2560].
#include "vqa.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#   include "ggml-cuda.h"   // ggml_backend_cuda_init
#endif
#include "tokenizer.hpp"
#include "ggml_dump.hpp"

#include <map>
#include "stage_buf.hpp"
#include "mage_preprocess.hpp"
#include "mage_gate.hpp"
#include "ggml_dump.hpp"

#include <map>
#include "attention.hpp"
#include "sampler.hpp"
#include "vqa_util.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace mage_vl {

// ---------------------------------------------------------------- structures
struct VisLayer {
    ggml_tensor *norm0_w = nullptr, *norm0_b = nullptr;
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *wqkv_w = nullptr, *wqkv_b = nullptr;
    ggml_tensor *wo_w = nullptr, *wo_b = nullptr;
    ggml_tensor *fc0_w = nullptr, *fc0_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
};

struct TxtLayer {
    ggml_tensor *norm = nullptr;
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    ggml_tensor *qn = nullptr, *kn = nullptr;
    ggml_tensor *ffn_norm = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
};

struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *pos_in = nullptr, *mask_in = nullptr;
    ggml_tensor *img_in = nullptr, *img_pos_in = nullptr;
    ggml_tensor *logits = nullptr;
};

class MageVL : public vqa::Model {
public:
    ~MageVL() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "mage-vl-4b"; }
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp, std::string & out_text) override;
    // rt streaming: frames-only turns go through the gate (System 1 decides
    // whether the event is response-worthy); questions bypass it
    bool ask_session(const vqa::MediaInput & media, const std::string & prompt,
                     const vqa::GenParams & gp, std::string & out_text) override {
        if (!prompt.empty() || !gate_ready_) return ask(media, prompt, "", gp, out_text);
        std::vector<float> scores;
        if (!gate_scores(media, scores) || scores.empty()) return ask(media, "", "", gp, out_text);
        const float tau = [] { const char * e = getenv("MAGE_GATE_TAU"); return e ? atof(e) : 0.5f; }();
        if (scores.back() < tau) {
            fprintf(stderr, "gate: silence (p=%.3f < %.2f)\n", (double)scores.back(), (double)tau);
            out_text.clear();
            return true;   // silent turn: nothing for the stream to emit
        }
        fprintf(stderr, "gate: response (p=%.3f >= %.2f)\n", (double)scores.back(), (double)tau);
        return ask(media, "Please describe the video content in detail based on the provided information.",
                   "", gp, out_text);
    }

    // tunables, set before load()
    int n_ctx = 8192;
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool alloc_kv();
    // frames per vision graph (VRAM knob): attention is per-frame anyway, so
    // packing only amortizes graph launch overhead
    int vision_pack = 8;
    bool encode_vision(const vqa::MediaInput & media, std::vector<float> & feats, int * n_tokens,
                       std::vector<float> * frame_secs = nullptr, int token_budget = 0);
    bool build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all = false);
    bool run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<float> * img_feats,
                  const std::vector<int> * img_pos, std::vector<float> & logits_out);
    std::string render_prompt(int n_img_tokens, const std::string & user_text, bool video,
                              const std::vector<float> * frame_secs = nullptr) const;
    int sample(const std::vector<float> & logits, const vqa::GenParams & gp);
    float randf();
    void dump(const ggml_cgraph * gf, const char * name);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    int n_layer = 36, n_embd = 2560, n_head = 32, n_kv_head = 8, head_dim = 128;
    int n_ff = 9728, n_vocab = 151936;
    float rms_eps = 1e-6f, rope_theta = 5e6f;
    mage::VisConfig vc;

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;         // weights: mmap-backed (CPU) or uploaded
    ggml_backend_buffer_t wbuf_plain = nullptr;   // host f32 block holding the 1D tensors
    vqa::AlignedBuf host_f32_storage;
    vqa::MMap mm;

    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr, *out_head = nullptr;
    std::vector<TxtLayer> tl;
    std::vector<VisLayer> vl;
    ggml_tensor *v_patch_w = nullptr;
    ggml_tensor *v_pre_w = nullptr, *v_pre_b = nullptr;
    ggml_tensor *mm_lnq_w = nullptr, *mm_lnq_b = nullptr;
    ggml_tensor *mm_fc1_w = nullptr, *mm_fc1_b = nullptr;
    ggml_tensor *mm_fc2_w = nullptr, *mm_fc2_b = nullptr;

    vqa::KvCache kv;
    ggml_gallocr_t allocr = nullptr;
    vqa::StageBuffer tower_buf;

    // static decode graph (T=1, fixed KV window per block): CUDA graph capture
    ggml_context * dec_ctx = nullptr;
    ggml_cgraph * dec_gf = nullptr;
    int dec_blk = 0;
    ggml_tensor * dec_tok = nullptr, * dec_pos = nullptr, * dec_mask = nullptr;
    int n_past = 0;

    int id_image_pad = 151655, id_video_pad = 151656;
    int id_vision_start = 151652, id_vision_end = 151653;
    int id_im_start = 151644, id_im_end = 151645, id_eos = 151645, id_endoftext = 151643;

    void (*stage_cb)(const char *, int, int, void *) = nullptr;
    void * stage_ud = nullptr;

    // ---- StreamMind gate (System 1, proactive streaming) ----
    bool load_gate(const char * path);
    bool build_gate_graph();
    void gate_cleanup();
    // perception tokens for one segment: per-frame mean-pooled [T, 2560]
    bool gate_scores(const vqa::MediaInput & media, std::vector<float> & scores);
    std::string gate_path;                 // discovered *-gate.gguf
    bool gate_ready_ = false;

    // gate weights (host F32 for the mamba/nets) + cls graph state
    mage::MambaWeights gm_;
    mage::ClsWeights cls_;                  // host fallback (MAGE_GATE_HOST=1)
    std::vector<float> gm_norm_w_, gm_norm_b_, gm_pre_w_, gm_pre_b_, gm_post_w_, gm_post_b_;
    ggml_backend_t gate_backend_ = nullptr;     // cls graph runs here (CPU or CUDA)
    ggml_gallocr_t gate_allocr_ = nullptr;
    ggml_context * gate_wctx_ = nullptr;        // cls tensors (mmap-backed)
    ggml_backend_buffer_t gate_wbuf_ = nullptr;
    vqa::MMap gate_mm_;
    ggml_context * gate_gctx_ = nullptr;
    ggml_cgraph * gate_gf_ = nullptr;
    ggml_tensor * gate_tok_in_ = nullptr;       // [2560, 1] perception token
    ggml_tensor * gate_logits_ = nullptr;       // [2, 1]
    vqa::AlignedBuf gate_f32_;                  // 1D cls norms as host F32
    std::vector<float> gate_emb0_;              // cls embed row 0 (target token)
    std::vector<float> tmp_gate_f32_;
    bool gate_graph_built_ = false;

    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

// ---------------------------------------------------------------- helpers
static float gguf_f32(struct gguf_context * g, const char * key, float def) {
    int64_t id = gguf_find_key(g, key);
    return id >= 0 ? gguf_get_val_f32(g, id) : def;
}
static int gguf_u32(struct gguf_context * g, const char * key, int def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return (int)gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (int)gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:  return (int)gguf_get_val_i64(g, id);
        default: return def;
    }
}

static void tensor_to_f32(ggml_tensor * t, float * dst) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        memcpy(dst, t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n; i++) dst[i] = ggml_fp16_to_fp32(s[i]);
    } else {
        const struct ggml_type_traits * tt = ggml_get_type_traits(t->type);
        tt->to_float(t->data, dst, n);
    }
}

// Mage-ViT rotary: per-dim frequencies (head dim 2p rotates by angle_a,
// dim 2p+1 by angle_b - NOT one shared angle), interleaved rotation exactly
// like the reference rotate_half. cos/sin tables come in as [pairs, N]
// (host-computed from the patch positions) and broadcast over the head axis
// via ggml's elementwise broadcasting: a [1, pairs, 1, N] table against a
// [1, pairs, H, N] re/im view.
static ggml_tensor * rope_mage(ggml_context * ctx, struct ggml_cgraph * gf, ggml_tensor * x,
                               ggml_tensor * cos_a, ggml_tensor * sin_a,
                               ggml_tensor * cos_b, ggml_tensor * sin_b) {
    const int64_t hd = x->ne[0], H = x->ne[1], N = x->ne[2];
    const int64_t pairs = hd / 2;
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    ggml_tensor * x4 = ggml_reshape_4d(ctx, x, 2, pairs, H, N);
    ggml_tensor * re = ggml_view_4d(ctx, x4, 1, pairs, H, N,
                                    x4->nb[1], x4->nb[2], x4->nb[3], 0);
    ggml_tensor * im = ggml_view_4d(ctx, x4, 1, pairs, H, N,
                                    x4->nb[1], x4->nb[2], x4->nb[3], x4->nb[0]);
    ggml_tensor * ca = ggml_reshape_4d(ctx, cos_a, 1, pairs, 1, N);
    ggml_tensor * sa = ggml_reshape_4d(ctx, sin_a, 1, pairs, 1, N);
    ggml_tensor * cb = ggml_reshape_4d(ctx, cos_b, 1, pairs, 1, N);
    ggml_tensor * sb = ggml_reshape_4d(ctx, sin_b, 1, pairs, 1, N);
    // out[2p]   = re*cos_a - im*sin_a
    // out[2p+1] = im*cos_b + re*sin_b
    ggml_tensor * ore = ggml_sub(ctx, ggml_mul(ctx, re, ca), ggml_mul(ctx, im, sa));
    ggml_tensor * oim = ggml_add(ctx, ggml_mul(ctx, im, cb), ggml_mul(ctx, re, sb));
    // concat gives [2, pairs, H, N] (all re-dims then all im-dims); interleave
    // to reference dim order (2p, 2p+1) with a (pairs, 2) swap before reshape
    if (getenv("MAGE_ROPE_OFF")) return x;   // ablation
    ggml_tensor * cat = ggml_concat(ctx, ore, oim, 0);
    cat = ggml_cont(ctx, ggml_permute(ctx, cat, 1, 0, 2, 3));   // [pairs, 2, H, N] -> interleaved dims
    return ggml_reshape_3d(ctx, cat, hd, H, N);
}

float MageVL::randf() {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (float)(((rng_state * 0x2545F4914F6CDD1Dull) >> 11) * (1.0 / 9007199254740992.0));
}

void MageVL::gate_cleanup() {
    if (gate_gctx_) ggml_free(gate_gctx_);
    if (gate_allocr_) ggml_gallocr_free(gate_allocr_);
    if (gate_wbuf_) ggml_backend_buffer_free(gate_wbuf_);
    if (gate_backend_) ggml_backend_free(gate_backend_);
    gate_gctx_ = nullptr; gate_allocr_ = nullptr; gate_wbuf_ = nullptr; gate_backend_ = nullptr;
}

MageVL::~MageVL() {
    gate_cleanup();
    if (allocr) ggml_gallocr_free(allocr);
    if (dec_ctx) ggml_free(dec_ctx);
    kv.free();
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wbuf_plain) ggml_backend_buffer_free(wbuf_plain);
    if (wctx) ggml_free(wctx);
    if (backend) ggml_backend_free(backend);
}

bool MageVL::init_backend() {
    if (use_gpu) {
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
#endif
    }
    if (backend) {
        fprintf(stderr, "backend: %s\n", ggml_backend_name(backend));
    } else {
        backend = ggml_backend_cpu_init();
        if (!backend) { fprintf(stderr, "no backend available\n"); return false; }
        if (n_threads <= 0) n_threads = (int)std::max(1u, std::thread::hardware_concurrency() / 2);
        ggml_backend_cpu_set_n_threads(backend, n_threads);
        fprintf(stderr, "backend: %s (%d threads)\n", ggml_backend_name(backend), n_threads);
    }
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

// ---------------------------------------------------------------- load
bool MageVL::load(const std::string & gguf_path, const std::string & tokenizer_path,
                  const vqa::LoadOptions & opt) {
    n_ctx = opt.n_ctx > 0 ? opt.n_ctx : n_ctx;
    n_threads = opt.n_threads;
    use_gpu = opt.use_gpu;
    dump_dir = opt.dump_dir;
    if (const char * d = getenv("MAGE_DUMP_DIR")) dump_dir = d;   // dev override
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::Qwen2Regex)) return false;
    if (!alloc_kv()) return false;

    auto pick = [&](const char * s, int def) {
        int v = tok_.id_of(s);
        return v >= 0 ? v : def;
    };
    id_image_pad = pick("<|image_pad|>", id_image_pad);
    id_video_pad = pick("<|video_pad|>", id_video_pad);
    id_vision_start = pick("<|vision_start|>", id_vision_start);
    id_vision_end = pick("<|vision_end|>", id_vision_end);
    id_im_start = pick("<|im_start|>", id_im_start);
    id_im_end = pick("<|im_end|>", id_im_end);
    id_eos = id_im_end;

    caps_ = {};
    caps_.image = true;
    caps_.video = true;
    caps_.timestamps_prompt = true;

    if (const char * e = getenv("MAGE_PACK")) vision_pack = std::max(1, atoi(e));
    // System 1: auto-discover the StreamMind gate next to the gguf (or $MAGE_GATE)
    if (const char * e = getenv("MAGE_GATE")) gate_path = e;
    if (gate_path.empty()) {
        std::string dir = gguf_path;
        const size_t slash = dir.find_last_of("/\\");
        if (slash != std::string::npos) dir = dir.substr(0, slash); else dir = ".";
        for (const std::string & cand : { dir + "/mage-gate-f16.gguf", dir + "/mage-gate-q8_0.gguf" })
            if (FILE * f = fopen(cand.c_str(), "rb")) { fclose(f); gate_path = cand; break; }
    }
    if (!gate_path.empty()) { fprintf(stderr, "mage_gate: discovering %s\n", gate_path.c_str()); load_gate(gate_path.c_str()); }
    fprintf(stderr, "mage_vl: ready (%d text layers, %d vision layers, ctx %d%s)\n",
            n_layer, vc.n_layer, n_ctx, gate_ready_ ? ", StreamMind gate loaded" : "");
    return true;
}

// ---------------------------------------------------------------- gate
bool MageVL::load_gate(const char * path) {
    vqa::MMap mm;
    if (!mm.load(path)) { fprintf(stderr, "mage_gate: mmap failed: %s\n", path); return false; }
    ggml_context * wctx = nullptr;
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "mage_gate: gguf init failed\n"); return false; }
    const size_t data_off = gguf_get_data_offset(gu);
    const int64_t nt = gguf_get_n_tensors(gu);
    for (int64_t i = 0; i < nt; i++) {
        if (ggml_tensor * t = ggml_get_tensor(wctx, gguf_get_tensor_name(gu, i)))
            t->data = (char *)mm.ptr + data_off + gguf_get_tensor_offset(gu, i);
    }
    gguf_free(gu);

    gm_.load(nullptr, wctx);       // mamba parts -> host F32
    cls_.load(wctx);               // cls classifier -> host F32 (fallback path)
    auto f32n = [&](const char * n, std::vector<float> & out) {
        if (ggml_tensor * t = ggml_get_tensor(wctx, n)) mage::tensor_to_f32_host(t, out);
    };
    f32n("gm.norm_fn.weight", gm_norm_w_);
    f32n("gm.norm_fn.bias", gm_norm_b_);
    f32n("gm.pre.fc3.weight", gm_pre_w_);
    f32n("gm.pre.fc3.bias", gm_pre_b_);
    f32n("gm.post.fc3.weight", gm_post_w_);
    f32n("gm.post.fc3.bias", gm_post_b_);
    if (ggml_tensor * t = ggml_get_tensor(wctx, "cls.token_embd.weight")) {
        std::vector<float> emb;
        mage::tensor_to_f32_host(t, emb);
        gate_emb0_.assign(emb.begin(), emb.begin() + n_embd);
    }

    // pick the cls backend (CPU by default; CUDA with MAGE_GATE_GPU=1)
#ifdef GGML_USE_CUDA
    if (getenv("MAGE_GATE_GPU")) gate_backend_ = ggml_backend_cuda_init(0);
#endif
    if (!gate_backend_) gate_backend_ = ggml_backend_cpu_init();
    if (!gate_backend_) return false;
    if (n_threads > 0) ggml_backend_cpu_set_n_threads(gate_backend_, std::min(n_threads, 8));
    gate_allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gate_backend_));

    // 1D tensors (norms, biases) become F32 BEFORE allocation: the cls graph
    // multiplies activations by them and ggml's binary ops reject f32 x f16.
    // Single pass: convert into a staging pool (indices, so growth is safe),
    // then materialize the pool and upload everything from it.
    std::map<ggml_tensor *, size_t> f32_idx;
    std::vector<std::pair<ggml_tensor *, const void *>> ups;
    std::vector<float> f32pool;
    for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
        ups.push_back({t, t->data});
        if (!(t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1) || t->type == GGML_TYPE_F32) continue;
        mage::tensor_to_f32_host(t, tmp_gate_f32_);
        f32_idx[t] = f32pool.size();
        f32pool.insert(f32pool.end(), tmp_gate_f32_.begin(), tmp_gate_f32_.end());
        const int64_t n = ggml_nelements(t);
        t->type = GGML_TYPE_F32;          // sized as F32 at alloc time
        t->nb[0] = sizeof(float);
        t->nb[1] = (size_t)n * sizeof(float);
        t->nb[2] = t->nb[1];
        t->nb[3] = t->nb[2];
    }
    gate_f32_.resize(f32pool.size() * sizeof(float));
    memcpy(gate_f32_.data(), f32pool.data(), gate_f32_.size());
    for (auto & u : ups) u.first->data = nullptr;   // alloc skips non-null data
    gate_wbuf_ = ggml_backend_alloc_ctx_tensors(wctx, gate_backend_);
    if (!gate_wbuf_) { fprintf(stderr, "mage_gate: weight alloc failed\n"); return false; }
    for (auto & u : ups) {
        auto it = f32_idx.find(u.first);
        if (it != f32_idx.end())
            ggml_backend_tensor_set(u.first, gate_f32_.data() + it->second, 0, ggml_nbytes(u.first));
        else
            ggml_backend_tensor_set(u.first, u.second, 0, ggml_nbytes(u.first));
    }
    gate_wctx_ = wctx;
    gate_ready_ = true;
    fprintf(stderr, "mage_gate: StreamMind gate loaded from %s (System 1 active)\n", path);
    return true;
}

void MageVL::dump(const ggml_cgraph * gf, const char * name) {
    if (dump_dir.empty()) return;
    ggml_tensor * t = vqa::dump::lookup((ggml_cgraph *)gf, name);   // registry first (sinks)
    if (!t || !t->data) return;
    std::string path = dump_dir + "/cpp_" + name + ".bin";
    std::vector<float> tmp((size_t)ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, tmp.data(), 0, sizeof(float) * tmp.size());
    } else {
        std::vector<ggml_fp16_t> h(ggml_nelements(t));
        ggml_backend_tensor_get(t, h.data(), 0, sizeof(ggml_fp16_t) * h.size());
        for (size_t i = 0; i < h.size(); i++) tmp[i] = ggml_fp16_to_fp32(h[i]);
    }
    FILE * f = fopen(path.c_str(), "wb");
    if (f) { fwrite(tmp.data(), sizeof(float), tmp.size(), f); fclose(f); }
    FILE * m = fopen((dump_dir + "/cpp_" + name + ".shape").c_str(), "w");
    if (m) {
        fprintf(m, "%lld %lld %lld %lld\n", (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3]);
        fclose(m);
    }
}

bool MageVL::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f GB\n", mm.size / 1e9);

    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer   = gguf_u32(gu, "mage_vl.block_count", n_layer);
    n_embd    = gguf_u32(gu, "mage_vl.embedding_length", n_embd);
    n_head    = gguf_u32(gu, "mage_vl.attention.head_count", n_head);
    n_kv_head = gguf_u32(gu, "mage_vl.attention.head_count_kv", n_kv_head);
    head_dim  = gguf_u32(gu, "mage_vl.attention.key_length", head_dim);
    n_ff      = gguf_u32(gu, "mage_vl.feed_forward_length", n_ff);
    n_vocab   = gguf_u32(gu, "mage_vl.vocab_size", n_vocab);
    rms_eps   = gguf_f32(gu, "mage_vl.attention.layer_norm_rms_epsilon", rms_eps);
    rope_theta = gguf_f32(gu, "mage_vl.rope.freq_base", rope_theta);

    vc.n_layer  = gguf_u32(gu, "mage_vl.vision.block_count", vc.n_layer);
    vc.n_embd   = gguf_u32(gu, "mage_vl.vision.embedding_length", vc.n_embd);
    vc.n_head   = gguf_u32(gu, "mage_vl.vision.attention.head_count", vc.n_head);
    vc.n_ff     = gguf_u32(gu, "mage_vl.vision.feed_forward_length", vc.n_ff);
    vc.patch    = gguf_u32(gu, "mage_vl.vision.patch_size", vc.patch);
    vc.merge    = gguf_u32(gu, "mage_vl.vision.merge_size", vc.merge);
    vc.window   = gguf_u32(gu, "mage_vl.vision.frame_window", vc.window);
    vc.eps      = gguf_f32(gu, "mage_vl.vision.layer_norm_epsilon", vc.eps);
    vc.rope_theta = gguf_f32(gu, "mage_vl.vision.rope_theta", vc.rope_theta);
    vc.freq_t   = gguf_u32(gu, "mage_vl.vision.freq_dims_t", vc.freq_t);
    vc.freq_h   = gguf_u32(gu, "mage_vl.vision.freq_dims_h", vc.freq_h);
    vc.freq_w   = gguf_u32(gu, "mage_vl.vision.freq_dims_w", vc.freq_w);
    for (int i = 0; i < 3; i++) {
        char kbuf[64];
        snprintf(kbuf, sizeof(kbuf), "mage_vl.vision.image_mean.%d", i);
        vc.mean[i] = gguf_f32(gu, kbuf, vc.mean[i]);
        snprintf(kbuf, sizeof(kbuf), "mage_vl.vision.image_std.%d", i);
        vc.stdv[i] = gguf_f32(gu, kbuf, vc.stdv[i]);
    }

    const size_t data_off = gguf_get_data_offset(gu);
    const int64_t nt = gguf_get_n_tensors(gu);
    for (int64_t i = 0; i < nt; i++) {
        const char * name = gguf_get_tensor_name(gu, i);
        ggml_tensor * t = ggml_get_tensor(wctx, name);
        if (!t) continue;
        t->data = (char *)mm.ptr + data_off + gguf_get_tensor_offset(gu, i);
    }
    gguf_free(gu);

    // 1D tensors (norms, biases) -> host F32: ggml elementwise ops need matching types
    vqa::AlignedBuf host_f32;
    {
        size_t total = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            if (t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1) continue;
            if (t->type == GGML_TYPE_F32) continue;
            total += (size_t)ggml_nelements(t) * sizeof(float);
        }
        host_f32.resize(total);
        size_t off = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            if (t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1) continue;
            if (t->type == GGML_TYPE_F32) continue;
            const int64_t n = ggml_nelements(t);
            float * dst = (float *)(host_f32.data() + off);
            tensor_to_f32(t, dst);
            t->data = dst;
            t->type = GGML_TYPE_F32;
            t->nb[0] = sizeof(float);
            t->nb[1] = (size_t)n * sizeof(float);
            t->nb[2] = t->nb[1];
            t->nb[3] = t->nb[2];
            off += (size_t)n * sizeof(float);
        }
        fprintf(stderr, "f32-converted 1D tensors: %.1f MB\n", total / 1e6);
    }

    {
        std::vector<std::pair<ggml_tensor *, const void *>> uploads;
        size_t total = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            uploads.push_back({t, t->data});
            total += ggml_nbytes(t);
        }

        if (ggml_backend_is_cpu(backend)) {
            wbuf = ggml_backend_cpu_buffer_from_ptr(mm.ptr, mm.size);
            const char * f32_begin = host_f32.empty() ? nullptr : host_f32.data();
            const char * f32_end = f32_begin ? f32_begin + host_f32.size() : nullptr;
            if (f32_begin) wbuf_plain = ggml_backend_cpu_buffer_from_ptr(host_f32.data(), host_f32.size());
            for (auto & u : uploads) {
                ggml_tensor * t = u.first;
                const char * q = (const char *)t->data;
                t->buffer = (f32_begin && q >= f32_begin && q < f32_end) ? wbuf_plain : wbuf;
            }
            fprintf(stderr, "weights: %.2f GB zero-copy from mmap\n", total / 1e9);
        } else {
            for (auto & u : uploads) u.first->data = nullptr;
            wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
            if (!wbuf) { fprintf(stderr, "weight allocation failed (need %.2f GB)\n", total / 1e9); return false; }
            for (auto & u : uploads) ggml_backend_tensor_set(u.first, u.second, 0, ggml_nbytes(u.first));
            fprintf(stderr, "uploaded %.2f GB of weights\n", total / 1e9);
        }
    }
    if (ggml_backend_is_cpu(backend)) {
        host_f32_storage = std::move(host_f32);
    } else {
        host_f32.clear();
        host_f32.shrink_to_fit();
        mm.release();
    }

    auto gt = [&](const char * name) {
        ggml_tensor * t = ggml_get_tensor(wctx, name);
        if (!t) fprintf(stderr, "missing tensor: %s\n", name);
        return t;
    };
    tok_embd = gt("token_embd.weight");
    out_norm = gt("output_norm.weight");
    out_head = gt("output.weight");
    v_patch_w = gt("v.patch_embd.weight");
    v_pre_w = gt("v.pre_norm.weight");
    v_pre_b = gt("v.pre_norm.bias");
    mm_lnq_w = gt("mm.ln_q.weight");
    mm_lnq_b = gt("mm.ln_q.bias");
    mm_fc1_w = gt("mm.fc1.weight");
    mm_fc1_b = gt("mm.fc1.bias");
    mm_fc2_w = gt("mm.fc2.weight");
    mm_fc2_b = gt("mm.fc2.bias");

    tl.resize(n_layer);
    for (int i = 0; i < n_layer; i++) {
        char buf[128];
        auto g = [&](const char * suffix) {
            snprintf(buf, sizeof(buf), "blk.%d.%s", i, suffix);
            return gt(buf);
        };
        TxtLayer & L = tl[i];
        L.norm = g("attn_norm.weight");
        L.q = g("attn_q.weight");
        L.k = g("attn_k.weight");
        L.v = g("attn_v.weight");
        L.o = g("attn_output.weight");
        L.qn = g("attn_q_norm.weight");
        L.kn = g("attn_k_norm.weight");
        L.ffn_norm = g("ffn_norm.weight");
        L.gate = g("ffn_gate.weight");
        L.up = g("ffn_up.weight");
        L.down = g("ffn_down.weight");
    }

    vl.resize(vc.n_layer);
    for (int i = 0; i < vc.n_layer; i++) {
        char buf[128];
        auto g = [&](const char * suffix) {
            snprintf(buf, sizeof(buf), "v.blk.%d.%s", i, suffix);
            return gt(buf);
        };
        VisLayer & L = vl[i];
        L.norm0_w = g("norm0.weight");
        L.norm0_b = g("norm0.bias");
        L.norm1_w = g("norm1.weight");
        L.norm1_b = g("norm1.bias");
        L.wqkv_w = g("wqkv.weight");
        L.wqkv_b = g("wqkv.bias");
        L.wo_w = g("wo.weight");
        L.wo_b = g("wo.bias");
        L.fc0_w = g("mlp.fc0.weight");
        L.fc0_b = g("mlp.fc0.bias");
        L.fc1_w = g("mlp.fc1.weight");
        L.fc1_b = g("mlp.fc1.bias");
    }
    return tok_embd != nullptr && out_norm != nullptr && out_head != nullptr && v_patch_w != nullptr;
}

bool MageVL::alloc_kv() {
    return kv.alloc(backend, n_layer, n_kv_head, head_dim, n_ctx);
}


// ---------------------------------------------------------------- vision
bool MageVL::encode_vision(const vqa::MediaInput & media, std::vector<float> & feats, int * n_tokens,
                           std::vector<float> * frame_secs, int token_budget) {
    *n_tokens = 0;
    if (media.empty()) return true;
    const int T = (int)media.frames.size();
    const int src_h = media.height, src_w = media.width;
    const int factor = vc.patch * vc.merge;

    // Qwen2VL processor budgets: images max 4.0 MP; the frames-video path
    // shares the same image processor budget
    long long max_pixels = 4000000LL;
    if (token_budget > 0) {
        const long long min_total = 4LL * T;                    // min grid 2x2 merged
        if (token_budget < min_total) {
            fprintf(stderr, "image/video cannot fit in n_ctx: needs >= %d merged vision tokens "
                            "(%d frame%s), budget %d -- raise --n-ctx or reduce --max-tokens\n",
                    (int)min_total, T, T == 1 ? "" : "s", token_budget);
            return false;
        }
        const long long per_grid = token_budget / T;
        if (per_grid * (long long)factor * factor < max_pixels) {
            max_pixels = per_grid * factor * factor;
            fprintf(stderr, "vision budget %d tokens (%d frame%s): max_pixels clamped to %.2f MP\n",
                    token_budget, T, T == 1 ? "" : "s", max_pixels / 1e6);
        }
    }
    // per-frame resolution cap for videos (attention is quadratic per frame)
    if (media.kind == vqa::MediaKind::Video) {
        double vmp = 0.75;
        if (const char * e = getenv("MAGE_VIDEO_MP")) vmp = atof(e);
        if (vmp > 0 && (double)max_pixels > vmp * 1e6) {
            max_pixels = (long long)(vmp * 1e6);
            fprintf(stderr, "video frame cap: %.2f MP/frame (MAGE_VIDEO_MP to override)\n", vmp);
        }
    }
    int H = 0, W = 0;
    if (!vc3::smart_resize(src_h, src_w, factor, 3136, (int)std::min(max_pixels, (long long)INT_MAX), &H, &W)) {
        fprintf(stderr, "image too small (%dx%d, factor %d)\n", src_w, src_h, factor);
        return false;
    }
    // VRAM guard: patch-level activations scale with (H/patch)*(W/patch)*pack
    {
        const long long n_patches = (long long)(H / vc.patch) * (W / vc.patch) * std::min(T, vision_pack);
        const long long n_max = 48000;
        if (n_patches > n_max) {
            const double scale = (double)n_max / (double)n_patches;
            max_pixels = std::min(max_pixels, (long long)(max_pixels * scale));
            int H2, W2;
            if (!vc3::smart_resize(src_h, src_w, factor, 3136, (int)std::min(max_pixels, (long long)INT_MAX), &H2, &W2)) {
                fprintf(stderr, "image too small (%dx%d, factor %d)\n", src_w, src_h, factor);
                return false;
            }
            fprintf(stderr, "vision VRAM guard: %lld patches > %lld, shrinking %dx%d -> %dx%d each\n",
                    n_patches, n_max, W, H, W2, H2);
            H = H2; W = W2;
        }
    }
    if (media.kind == vqa::MediaKind::Video)
        fprintf(stderr, "video %d frame%s %dx%d -> %dx%d each\n", T, T == 1 ? "" : "s", src_w, src_h, W, H);
    else
        fprintf(stderr, "image %dx%d -> %dx%d\n", src_w, src_h, W, H);

    // preprocess every frame: resize + normalize + block-layout patchify
    std::vector<float> rgb, patches;
    int gh = 0, gw = 0;
    for (int ti = 0; ti < T; ti++) {
        vc3::resize_bicubic_antialias(media.frames[ti].rgb.data(), src_w, src_h, W, H, rgb);
        mage::patchify_block(rgb.data(), H, W, vc.patch, vc.mean, vc.stdv, patches, &gh, &gw);
    }
    const int hw = gh * gw;
    const int N = hw * T;

    // prompt timestamps: seconds of each sampled frame (processor: frame_idx/fps)
    if (frame_secs) {
        frame_secs->clear();
        const double def_fps = media.fps > 0 ? media.fps : 1.0;
        for (int ti = 0; ti < T; ti++) {
            const double t = ti < (int)media.timestamps.size() ? media.timestamps[ti] : ti / def_fps;
            frame_secs->push_back((float)t);
        }
    }

    // patch positions (t = dense frame index) + per-dim rope tables
    std::vector<int> pos;
    mage::build_patch_positions(T, gh, gw, vc.merge, pos);
    std::vector<float> cos_a, sin_a, cos_b, sin_b;
    mage::rope_tables(vc, pos.data(), N, cos_a, sin_a, cos_b, sin_b);

    const int vhd = vc.n_embd / vc.n_head;
    const int C3 = 3 * vc.patch * vc.patch;

    // frames-backend attention chunks: one chunk per frame (the processor emits
    // each frame as a separate [1,H,W] sample). codec windows later: window=4.
    std::vector<int> graph_off;
    for (int off = 0; off < T; off += vision_pack) graph_off.push_back(off);

    if (!tower_buf.ensure(backend, vc.n_embd, (int64_t)hw * (int64_t)T)) return false;
    if (stage_cb) stage_cb("vision", 0, T, stage_ud);
    const char * stop_after = getenv("MAGE_STOP_AFTER");
    const int vl_max = [&] { const char * e = getenv("MAGE_VL"); return e ? std::min(vc.n_layer, atoi(e)) : vc.n_layer; }();

    int frame0 = 0;
    for (size_t c = 0; c < graph_off.size(); c++) {
        const int nf = std::min(vision_pack, T - graph_off[c]);
        const int Nc = hw * nf;
        struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
        ggml_context * gctx = ggml_init(ip);
        if (!gctx) return false;
        struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);

        ggml_tensor * t_patches = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, C3, Nc);
        ggml_set_name(t_patches, "patches");
        // per-dim rotary tables for this frame range: contiguous [frame0*hw,
        // +Nc) row slice of the full-stream tables ([pairs, N], stream-ordered)
        const int pairs_tbl = vhd / 2;
        ggml_tensor * t_ca = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, pairs_tbl, Nc);
        ggml_tensor * t_sa = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, pairs_tbl, Nc);
        ggml_tensor * t_cb = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, pairs_tbl, Nc);
        ggml_tensor * t_sb = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, pairs_tbl, Nc);
        if (c == 0 && getenv("MAGE_DUMP_ROPE")) {
            ggml_set_name(t_ca, "v_rope_ca"); ggml_set_name(t_sa, "v_rope_sa");
        }
        const float * ca_c = cos_a.data() + (size_t)frame0 * hw * pairs_tbl;
        const float * sa_c = sin_a.data() + (size_t)frame0 * hw * pairs_tbl;
        const float * cb_c = cos_b.data() + (size_t)frame0 * hw * pairs_tbl;
        const float * sb_c = sin_b.data() + (size_t)frame0 * hw * pairs_tbl;
        const float * patches_c = patches.data() + (size_t)frame0 * hw * C3;

        ggml_tensor * x = ggml_mul_mat(gctx, v_patch_w, t_patches);
        x = ggml_norm(gctx, x, vc.eps);                 // layernorm_pre
        x = ggml_add(gctx, ggml_mul(gctx, x, v_pre_w), v_pre_b);
        ggml_set_name(x, "v_patch_out");

        auto run_chunk = [&](const char * out_name, std::vector<float> * dest) -> bool {
            if (!dump_dir.empty()) {
                static const char * const dump_names[] = {
                    "patches", "v_patch_out", "v_blk0_qkv", "v_blk0_q_rope",
                    "v_blk0_attn_out", "v_blk0_out", "v_final",
                    "v_rope_ca", "v_rope_sa", nullptr,
                };
                vqa::dump::mark_sinks(gctx, gf, dump_names);
            }
            if (!ggml_gallocr_alloc_graph(allocr, gf)) {
                fprintf(stderr, "vision graph allocation failed\n");
                return false;
            }
            if (t_patches->buffer) ggml_backend_tensor_set(t_patches, patches_c, 0, sizeof(float) * (size_t)Nc * C3);
            if (t_ca->buffer) ggml_backend_tensor_set(t_ca, ca_c, 0, sizeof(float) * (size_t)Nc * pairs_tbl);
            if (t_sa->buffer) ggml_backend_tensor_set(t_sa, sa_c, 0, sizeof(float) * (size_t)Nc * pairs_tbl);
            if (t_cb->buffer) ggml_backend_tensor_set(t_cb, cb_c, 0, sizeof(float) * (size_t)Nc * pairs_tbl);
            if (t_sb->buffer) ggml_backend_tensor_set(t_sb, sb_c, 0, sizeof(float) * (size_t)Nc * pairs_tbl);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "vision compute failed\n");
                return false;
            }
            if (c == 0) {
                if (getenv("MAGE_DUMP_ROPE")) { dump(gf, "v_rope_ca"); dump(gf, "v_rope_sa"); }
                dump(gf, "patches");
                dump(gf, "v_patch_out");
                dump(gf, "v_blk0_qkv");
                dump(gf, "v_blk0_q_rope");
                dump(gf, "v_blk0_attn_out");
                dump(gf, "v_blk0_out");
            }
            dump(gf, out_name);
            if (dest) {
                ggml_tensor * out = ggml_graph_get_tensor(gf, out_name);
                dest->resize((size_t)out->ne[1] * out->ne[0]);
                ggml_backend_tensor_get(out, dest->data(), 0, sizeof(float) * dest->size());
            }
            return true;
        };

        // parity truncation only applies to the first graph (parity runs are images)
        const char * stop_c = (c == 0) ? stop_after : nullptr;
        if (stop_c && !strcmp(stop_c, "v_patch_out")) {
            ggml_build_forward_expand(gf, x);
            if (!run_chunk("v_patch_out", &feats)) { ggml_free(gctx); return false; }
            *n_tokens = (int)(feats.size() / vc.n_embd);
            ggml_free(gctx);
            return true;
        }

        for (int il = 0; il < vl_max; il++) {
            VisLayer & L = vl[il];
            ggml_tensor * l = ggml_norm(gctx, x, vc.eps);
            l = ggml_add(gctx, ggml_mul(gctx, l, L.norm0_w), L.norm0_b);
            ggml_tensor * qkv = ggml_mul_mat(gctx, L.wqkv_w, l);
            qkv = ggml_add(gctx, qkv, L.wqkv_b);
            if (il == 0) ggml_set_name(qkv, "v_blk0_qkv");
            if (stop_c && il == 0 && !strcmp(stop_c, "v_blk0_qkv")) {
                ggml_build_forward_expand(gf, qkv);
                if (!run_chunk("v_blk0_qkv", &feats)) { ggml_free(gctx); return false; }
                ggml_free(gctx);
                return true;
            }

            // qkv packs q|k|v blocks of n_head heads along ne0
            const size_t hbytes = (size_t)vhd * qkv->nb[0];
            auto qkv_head_view = [&](int block) {
                return ggml_view_3d(gctx, qkv, vhd, vc.n_head, Nc,
                                    hbytes, qkv->nb[1], (size_t)block * vc.n_head * hbytes);
            };
            ggml_tensor * q = qkv_head_view(0);
            ggml_tensor * k = qkv_head_view(1);
            ggml_tensor * v = qkv_head_view(2);
            q = rope_mage(gctx, gf, q, t_ca, t_sa, t_cb, t_sb);
            k = rope_mage(gctx, gf, k, t_ca, t_sa, t_cb, t_sb);
            if (il == 0) {
                ggml_set_name(q, "v_blk0_q_rope");
                if (stop_c && !strcmp(stop_c, "v_blk0_q_rope")) {
                    ggml_build_forward_expand(gf, q);
                    if (!run_chunk("v_blk0_q_rope", &feats)) { ggml_free(gctx); return false; }
                    ggml_free(gctx);
                    return true;
                }
            }

            // per-frame unmasked flash attention (each frame is its own chunk,
            // contiguous), concat along the token axis - same trick as vc3
            const float scale = 1.0f / std::sqrt((float)vhd);
            ggml_tensor * fa = nullptr;
            for (int f = 0; f < nf; f++) {
                const size_t off = (size_t)f * hw;
                auto cin = [&](ggml_tensor * t) {
                    return ggml_view_3d(gctx, t, vhd, hw, vc.n_head, t->nb[2], t->nb[1], off * t->nb[2]);
                };
                ggml_tensor * fc = ggml_flash_attn_ext(gctx, cin(q), cin(k), cin(v), nullptr, scale, 0.0f, 0.0f);
                fc = ggml_cont(gctx, fc);
                fa = fa ? ggml_concat(gctx, fa, fc, 2) : fc;
            }
            ggml_tensor * attn = ggml_reshape_2d(gctx, fa, vc.n_embd, Nc);
            if (il == 0) ggml_set_name(attn, "v_blk0_attn_out");
            attn = ggml_mul_mat(gctx, L.wo_w, attn);
            attn = ggml_add(gctx, attn, L.wo_b);
            x = ggml_add(gctx, x, attn);
            if (il == 0) {
                ggml_set_name(x, "v_blk0_out");
                if (stop_c && !strcmp(stop_c, "v_blk0_out")) {
                    ggml_build_forward_expand(gf, x);
                    if (!run_chunk("v_blk0_out", &feats)) { ggml_free(gctx); return false; }
                    ggml_free(gctx);
                    return true;
                }
            }

            ggml_tensor * l2 = ggml_norm(gctx, x, vc.eps);
            l2 = ggml_add(gctx, ggml_mul(gctx, l2, L.norm1_w), L.norm1_b);
            ggml_tensor * f = ggml_mul_mat(gctx, L.fc0_w, l2);
            f = ggml_add(gctx, f, L.fc0_b);
            f = ggml_gelu_erf(gctx, f);                    // config hidden_act = gelu (erf)
            f = ggml_mul_mat(gctx, L.fc1_w, f);
            f = ggml_add(gctx, f, L.fc1_b);
            x = ggml_add(gctx, x, f);
        }

        ggml_set_name(x, "v_final");
        // write this group's tower output into the persistent device buffer as
        // a trailing sink (computed by the same run_chunk pass - no host trip)
        ggml_tensor * out_sink = tower_buf.sink(gctx, x, (int64_t)frame0 * hw);
        ggml_build_forward_expand(gf, out_sink);
        if (!run_chunk("v_final", nullptr)) { ggml_free(gctx); return false; }
        ggml_free(gctx);
        frame0 += nf;
        if (stage_cb) stage_cb("vision", frame0, T, stage_ud);
    }

    // ---- merger + projector over the concatenated tower output ----
    {
        struct ggml_init_params ip2 = { 32ull * 1024 * 1024, NULL, /*no_alloc*/ true };
        ggml_context * mctx = ggml_init(ip2);
        if (!mctx) return false;
        struct ggml_cgraph * mgf = ggml_new_graph_custom(mctx, 8192, false);
        ggml_tensor * x = tower_buf.view(mctx, N, 0);          // device-side [1024, N]

        // block layout puts every 2x2 merge group in 4 consecutive rows, so the
        // reference merger (LN -> view(-1, 4*D) -> MLP) is a plain reshape here
        x = ggml_norm(mctx, x, vc.eps);
        x = ggml_add(mctx, ggml_mul(mctx, x, mm_lnq_w), mm_lnq_b);
        x = ggml_reshape_2d(mctx, x, (int64_t)vc.n_embd * vc.merge * vc.merge,
                            (int64_t)N / (vc.merge * vc.merge));
        ggml_set_name(x, "v_merge");
        ggml_tensor * h = ggml_mul_mat(mctx, mm_fc1_w, x);
        h = ggml_add(mctx, h, mm_fc1_b);
        h = ggml_gelu_erf(mctx, h);
        h = ggml_mul_mat(mctx, mm_fc2_w, h);
        h = ggml_add(mctx, h, mm_fc2_b);
        ggml_set_name(h, "v_proj_out");
        ggml_build_forward_expand(mgf, h);
        if (!dump_dir.empty()) {
            static const char * const dump_names[] = { "v_merge", "v_proj_out", nullptr };
            vqa::dump::mark_sinks(mctx, mgf, dump_names);
        }
        if (!ggml_gallocr_alloc_graph(allocr, mgf)) {
            fprintf(stderr, "merger graph allocation failed\n");
            ggml_free(mctx);
            return false;
        }
        if (ggml_backend_graph_compute(backend, mgf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "merger compute failed\n");
            ggml_free(mctx);
            return false;
        }
        dump(mgf, "v_merge");
        dump(mgf, "v_proj_out");
        feats.resize((size_t)h->ne[1] * h->ne[0]);
        ggml_backend_tensor_get(h, feats.data(), 0, sizeof(float) * feats.size());
        *n_tokens = (int)h->ne[1];
        fprintf(stderr, "vision: %d patches (%dx%d x %d frame%s) -> %d tokens\n",
                N, gh, gw, T, T == 1 ? "" : "s", *n_tokens);
        ggml_free(mctx);
    }
    return true;
}


// gate cls graph: inputs = one perception token [2560,1]; the graph embeds
// target id 0, runs the 2-token pair through the 4-layer vocab-2 Qwen3 and
// returns the pos-0 logits [2,1]. F16 K/V for flash attention (required).
bool MageVL::build_gate_graph() {
    if (gate_graph_built_) return true;
    struct ggml_init_params ip = { 32ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    gate_gctx_ = ggml_init(ip);
    if (!gate_gctx_) return false;
    gate_gf_ = ggml_new_graph_custom(gate_gctx_, 4096, false);

    auto gt = [&](const char * n) { return ggml_get_tensor(gate_wctx_, n); };
    ggml_tensor * tok_embd = gt("cls.token_embd.weight");
    ggml_tensor * out_w = gt("cls.output.weight");
    ggml_tensor * out_norm = gt("cls.output_norm.weight");

    const int T = 2, D = n_embd, HD = head_dim, NH = n_head, NKV = n_kv_head;
    gate_tok_in_ = ggml_new_tensor_2d(gate_gctx_, GGML_TYPE_F32, D, T);
    ggml_set_name(gate_tok_in_, "gate_pair_in");
    // keep the pair input alive for dumps (it is a graph leaf)
    if (!dump_dir.empty()) {
        static const char * const pn[] = { "gate_pair_in", nullptr };
        vqa::dump::mark_sinks(gate_gctx_, gate_gf_, pn);
    }

    ggml_tensor * x = gate_tok_in_;   // host assembles the full [D, 2] pair

    ggml_tensor * mask = ggml_new_tensor_2d(gate_gctx_, GGML_TYPE_F16, T, T);
    ggml_set_name(mask, "gate_mask");
    ggml_tensor * pos = ggml_new_tensor_1d(gate_gctx_, GGML_TYPE_I32, T);
    ggml_set_name(pos, "gate_pos");

    for (int il = 0; il < 4; il++) {
        char buf[96];
        auto g = [&](const char * sfx) {
            snprintf(buf, sizeof(buf), "cls.blk.%d.%s", il, sfx);
            return gt(buf);
        };
        ggml_tensor * l = ggml_rms_norm(gate_gctx_, x, rms_eps);
        l = ggml_mul(gate_gctx_, l, g("attn_norm.weight"));
        if (il == 0) ggml_set_name(l, "gate_l0_l");
        ggml_tensor * q = ggml_mul_mat(gate_gctx_, g("attn_q.weight"), l);
        ggml_tensor * k = ggml_mul_mat(gate_gctx_, g("attn_k.weight"), l);
        ggml_tensor * v = ggml_mul_mat(gate_gctx_, g("attn_v.weight"), l);
        ggml_tensor * q3 = ggml_reshape_3d(gate_gctx_, q, HD, NH, T);
        q3 = ggml_rms_norm(gate_gctx_, q3, rms_eps);
        q3 = ggml_mul(gate_gctx_, q3, g("attn_q_norm.weight"));
        ggml_tensor * k3 = ggml_reshape_3d(gate_gctx_, k, HD, NKV, T);
        k3 = ggml_rms_norm(gate_gctx_, k3, rms_eps);
        k3 = ggml_mul(gate_gctx_, k3, g("attn_k_norm.weight"));
        if (il == 0) {
            ggml_set_name(q3, "gate_l0_qraw");
            ggml_set_name(q3, "gate_l0_q");
            ggml_set_name(k3, "gate_l0_k");
            ggml_set_name(v, "gate_l0_v");
        }
        q3 = ggml_rope_ext(gate_gctx_, q3, pos, nullptr, HD, GGML_ROPE_TYPE_NEOX, 8192,
                           1000000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_ext(gate_gctx_, k3, pos, nullptr, HD, GGML_ROPE_TYPE_NEOX, 8192,
                           1000000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * qf = ggml_view_3d(gate_gctx_, q3, HD, T, NH, q3->nb[2], q3->nb[1], 0);
        ggml_tensor * kf = ggml_cast(gate_gctx_, ggml_cont(gate_gctx_,
            ggml_view_3d(gate_gctx_, k3, HD, T, NKV, k3->nb[2], k3->nb[1], 0)), GGML_TYPE_F16);
        ggml_tensor * vf = ggml_cast(gate_gctx_, ggml_cont(gate_gctx_,
            ggml_view_3d(gate_gctx_, v, HD, T, NKV, HD * sizeof(float), v->nb[1], 0)), GGML_TYPE_F16);
        ggml_tensor * fa = ggml_flash_attn_ext(gate_gctx_, qf, kf, vf, mask,
                                               1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        if (il == 0) ggml_set_name(fa, "gate_l0_fa");
        ggml_tensor * attn = ggml_reshape_2d(gate_gctx_, ggml_cont(gate_gctx_, fa), NH * HD, T);
        attn = ggml_mul_mat(gate_gctx_, g("attn_output.weight"), attn);
        x = ggml_add(gate_gctx_, x, attn);
        if (il == 0) ggml_set_name(x, "gate_l0_attn_out");
        ggml_tensor * m = ggml_rms_norm(gate_gctx_, x, rms_eps);
        m = ggml_mul(gate_gctx_, m, g("ffn_norm.weight"));
        ggml_tensor * gate = ggml_mul_mat(gate_gctx_, g("ffn_gate.weight"), m);
        ggml_tensor * up = ggml_mul_mat(gate_gctx_, g("ffn_up.weight"), m);
        m = ggml_swiglu_split(gate_gctx_, gate, up);
        if (il == 0) ggml_set_name(m, "gate_l0_swiglu");
        m = ggml_mul_mat(gate_gctx_, g("ffn_down.weight"), m);
        x = ggml_add(gate_gctx_, x, m);
        if (il == 0) ggml_set_name(x, "gate_l0_out");
    }
    x = ggml_rms_norm(gate_gctx_, x, rms_eps);
    x = ggml_mul(gate_gctx_, x, out_norm);
    ggml_tensor * first = ggml_view_2d(gate_gctx_, x, D, 1, x->nb[1], 0);
    gate_logits_ = ggml_mul_mat(gate_gctx_, out_w, first);      // [2, 1]
    ggml_set_name(gate_logits_, "gate_logits");
    ggml_build_forward_expand(gate_gf_, gate_logits_);
    if (!dump_dir.empty()) {
        static const char * const gn[] = { "gate_x", "gate_l0_l",
                                           "gate_l0_q", "gate_l0_k", "gate_l0_v",
                                           "gate_l0_fa", "gate_l0_attn_out",
                                           "gate_l0_swiglu", "gate_l0_out", nullptr };
        vqa::dump::mark_sinks(gate_gctx_, gate_gf_, gn);
    }
    if (!ggml_gallocr_alloc_graph(gate_allocr_, gate_gf_)) {
        fprintf(stderr, "mage_gate: cls graph alloc failed\n");
        return false;
    }
    gate_graph_built_ = true;
    return true;
}

// Gate scoring over a whole segment: per-frame mean-pooled perception tokens
// run through pre_net -> mamba (recurrent) -> post_net, the last frame's token
// goes through the cls pair. Returns per-frame p(speak).
bool MageVL::gate_scores(const vqa::MediaInput & media, std::vector<float> & scores) {
    scores.clear();
    if (!gate_ready_) return false;
    std::vector<float> feats;
    int n_tok = 0;
    if (!encode_vision(media, feats, &n_tok, nullptr, 0)) return false;
    const int D = n_embd;
    const int T = n_tok ? std::max(1, n_tok / std::max(1, (int)media.frames.size())) : 0;
    if (!T) return false;
    const int nF = n_tok / T;

    mage::GateStream stream;
    stream.reset(&gm_);
    std::vector<float> scratch(stream.scratch_size());
    FILE * gate_dbg = nullptr;
    if (const char * e = getenv("MAGE_GATE_DUMP")) gate_dbg = fopen(e, "wb");
    std::vector<float> pooled((size_t)D);
    std::vector<float> tok(D);
    int32_t ids1 = 0;
    std::vector<ggml_fp16_t> mask_v((size_t)4);
    mask_v[0] = ggml_fp32_to_fp16(0.0f);
    mask_v[1] = ggml_fp32_to_fp16(-INFINITY);
    mask_v[2] = ggml_fp32_to_fp16(0.0f);
    mask_v[3] = ggml_fp32_to_fp16(0.0f);
    int32_t pos_v[2] = { 0, 1 };

    for (int f = 0; f < nF; f++) {
        std::fill(pooled.begin(), pooled.end(), 0.0f);
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D; d++)
                pooled[d] += feats[(size_t)(f * T + t) * D + d];
        for (int d = 0; d < D; d++) pooled[d] /= (float)T;
        if (gate_dbg) fwrite(pooled.data(), sizeof(float), D, gate_dbg);
        // pre_net: fc3 + leaky_relu
        std::vector<float> tok(D);
        for (int d = 0; d < D; d++) {
            float v = gm_pre_b_[d];
            const float * w = gm_pre_w_.data() + (size_t)d * D;
            for (int c = 0; c < D; c++) v += w[c] * pooled[c];
            tok[d] = v > 0 ? v : 0.01f * v;
        }
        // mamba block (residual semantics per mamba_ssm Block), then norm_fn
        std::vector<float> mixed(D);
        stream.step(tok.data(), mixed.data(), scratch);
        for (int d = 0; d < D; d++) tok[d] = mixed[d] + 2.0f * tok[d];
        {
            float mu = 0, var = 0;
            for (int d = 0; d < D; d++) mu += tok[d];
            mu /= D;
            for (int d = 0; d < D; d++) { float e = tok[d] - mu; var += e * e; }
            var /= D;
            float is = 1.0f / std::sqrt(var + 1e-5f);
            for (int d = 0; d < D; d++) tok[d] = (tok[d] - mu) * is * gm_norm_w_[d] + gm_norm_b_[d];
        }
        if (gate_dbg) fwrite(tok.data(), sizeof(float), D, gate_dbg);
        // post_net: fc3(leaky_relu(x)) - leaky on the INPUT (official PostNet)
        std::vector<float> post(D);
        for (int d = 0; d < D; d++) {
            float v = gm_post_b_[d];
            const float * w = gm_post_w_.data() + (size_t)d * D;
            for (int c = 0; c < D; c++) {
                const float xc = tok[c];
                v += w[c] * (xc > 0 ? xc : 0.01f * xc);
            }
            post[d] = v;
        }
        fprintf(stderr, "[gv] f%d post done tok[0..3]=%.4f %.4f %.4f %.4f post[0]=%.4f\n", f,
                tok[0], tok[1], tok[2], tok[3], post[0]);
        if (gate_dbg) fwrite(post.data(), sizeof(float), D, gate_dbg);
        float lg[2];
        // host loop is the default (numerically verified vs the reference);
        // the ggml-graph variant is behind MAGE_GATE_GRAPH=1 (still NaN)
        if (!getenv("MAGE_GATE_GRAPH")) {
            cls_.forward(post.data(), lg, scratch);
        } else {
            if (!build_gate_graph()) return false;
            // host assembles the [D, 2] pair: col0 = perception token, col1 = embed(0)
            std::vector<float> pairv((size_t)D * 2);
            memcpy(pairv.data(), post.data(), sizeof(float) * D);
            memcpy(pairv.data() + D, gate_emb0_.data(), sizeof(float) * D);
            ggml_backend_tensor_set(gate_tok_in_, pairv.data(), 0, sizeof(float) * D * 2);
            {
                std::vector<ggml_fp16_t> mask_v(4);
                mask_v[0] = ggml_fp32_to_fp16(0.0f);
                mask_v[1] = ggml_fp32_to_fp16(-INFINITY);
                mask_v[2] = ggml_fp32_to_fp16(0.0f);
                mask_v[3] = ggml_fp32_to_fp16(0.0f);
                ggml_backend_tensor_set(ggml_graph_get_tensor(gate_gf_, "gate_mask"), mask_v.data(), 0, sizeof(ggml_fp16_t) * 4);
                int32_t pos_v[2] = { 0, 1 };
                ggml_backend_tensor_set(ggml_graph_get_tensor(gate_gf_, "gate_pos"), pos_v, 0, sizeof(pos_v));
            }
            if (ggml_backend_graph_compute(gate_backend_, gate_gf_) != GGML_STATUS_SUCCESS) return false;
            if (!dump_dir.empty()) {
                dump(gate_gf_, "gate_x"); dump(gate_gf_, "gate_l0_l"); dump(gate_gf_, "gate_l0_q");
                dump(gate_gf_, "gate_l0_k"); dump(gate_gf_, "gate_l0_v"); dump(gate_gf_, "gate_l0_fa");
                dump(gate_gf_, "gate_l0_attn_out"); dump(gate_gf_, "gate_l0_swiglu");
                dump(gate_gf_, "gate_l0_out"); dump(gate_gf_, "gate_logits");
            }
            ggml_backend_tensor_get(gate_logits_, lg, 0, sizeof(lg));
        }
        float m = std::max(lg[0], lg[1]);
        float e0 = std::exp(lg[0] - m), e1 = std::exp(lg[1] - m);
        scores.push_back(e1 / (e0 + e1));
    }
    if (gate_dbg) { fclose(gate_dbg); fprintf(stderr, "gate: perception tokens dumped\n"); }
    return true;
}

// ---------------------------------------------------------------- text
bool MageVL::build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all) {
    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 16384, false);

    const int D = n_embd, HD = head_dim, NH = n_head, NKV = n_kv_head;

    g.tok_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    g.pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.pos_in, "pos_in");
    g.mask_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, S, T);
    ggml_set_name(g.mask_in, "mask_in");

    ggml_tensor * x = ggml_get_rows(g.ctx, tok_embd, g.tok_in);      // [D, T]
    if (n_img > 0) {
        g.img_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, n_img);
        ggml_set_name(g.img_in, "img_in");
        g.img_pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_img);
        ggml_set_name(g.img_pos_in, "img_pos_in");
        x = ggml_set_rows(g.ctx, ggml_dup(g.ctx, x), g.img_in, g.img_pos_in);
    }
    ggml_set_name(x, "emb");

    int max_layers = n_layer;
    if (const char * e = getenv("MAGE_MAX_LAYERS")) max_layers = std::min(max_layers, atoi(e));
    for (int il = 0; il < max_layers; il++) {
        TxtLayer & L = tl[il];
        ggml_tensor * l = ggml_rms_norm(g.ctx, x, rms_eps);
        l = ggml_mul(g.ctx, l, L.norm);
        if (il == 0) ggml_set_name(l, "t_l0_norm");

        ggml_tensor * q = ggml_mul_mat(g.ctx, L.q, l);
        ggml_tensor * k = ggml_mul_mat(g.ctx, L.k, l);
        ggml_tensor * v = ggml_mul_mat(g.ctx, L.v, l);
        if (il == 0) { ggml_set_name(q, "t_l0_q"); ggml_set_name(k, "t_l0_k"); }

        ggml_tensor * q3 = ggml_reshape_3d(g.ctx, q, HD, NH, T);
        q3 = ggml_rms_norm(g.ctx, q3, rms_eps);
        q3 = ggml_mul(g.ctx, q3, L.qn);
        ggml_tensor * k3 = ggml_reshape_3d(g.ctx, k, HD, NKV, T);
        k3 = ggml_rms_norm(g.ctx, k3, rms_eps);
        k3 = ggml_mul(g.ctx, k3, L.kn);
        if (il == 0) { ggml_set_name(q3, "t_l0_q_norm"); ggml_set_name(k3, "t_l0_k_norm"); }

        q3 = ggml_rope_ext(g.ctx, q3, g.pos_in, nullptr, HD, GGML_ROPE_TYPE_NEOX, n_ctx,
                           rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_ext(g.ctx, k3, g.pos_in, nullptr, HD, GGML_ROPE_TYPE_NEOX, n_ctx,
                           rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (il == 0) { ggml_set_name(q3, "t_l0_q_rope"); ggml_set_name(k3, "t_l0_k_rope"); }

        ggml_tensor * k2 = ggml_reshape_2d(g.ctx, k3, NKV * HD, T);
        ggml_tensor * v2 = ggml_reshape_2d(g.ctx, v, NKV * HD, T);
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.k[il], k2, g.pos_in));
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.v[il], v2, g.pos_in));

        ggml_tensor * qf = ggml_view_3d(g.ctx, q3, HD, T, NH, q3->nb[2], q3->nb[1], 0);
        ggml_tensor * kf = ggml_view_3d(g.ctx, kv.k[il], HD, S, NKV,
                                        kv.k[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * vf = ggml_view_3d(g.ctx, kv.v[il], HD, S, NKV,
                                        kv.v[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * fa = ggml_flash_attn_ext(g.ctx, qf, kf, vf, g.mask_in,
                                               1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, fa), NH * HD, T);
        if (il == 0) ggml_set_name(attn, "t_l0_attn");
        attn = ggml_mul_mat(g.ctx, L.o, attn);
        x = ggml_add(g.ctx, x, attn);
        if (il == 0) ggml_set_name(x, "t_l0_resid");

        ggml_tensor * m = ggml_rms_norm(g.ctx, x, rms_eps);
        m = ggml_mul(g.ctx, m, L.ffn_norm);
        ggml_tensor * gate = ggml_mul_mat(g.ctx, L.gate, m);
        ggml_tensor * up = ggml_mul_mat(g.ctx, L.up, m);
        m = ggml_swiglu_split(g.ctx, gate, up);
        if (il == 0) ggml_set_name(m, "t_l0_swiglu");
        m = ggml_mul_mat(g.ctx, L.down, m);
        x = ggml_add(g.ctx, x, m);
        if (il == 0) ggml_set_name(x, "t_l0_out");
    }

    x = ggml_rms_norm(g.ctx, x, rms_eps);
    x = ggml_mul(g.ctx, x, out_norm);
    ggml_set_name(x, "t_final_norm");

    if (logits_all) {
        g.logits = ggml_mul_mat(g.ctx, out_head, x);
    } else {
        ggml_tensor * last = ggml_view_2d(g.ctx, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
        g.logits = ggml_mul_mat(g.ctx, out_head, last);
    }
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);
    if (!dump_dir.empty()) {
        static const char * const dump_names[] = {
            "emb", "t_l0_norm", "t_l0_q", "t_l0_q_norm", "t_l0_q_rope", "t_l0_attn",
            "t_l0_out", "t_final_norm", nullptr,
        };
        vqa::dump::mark_sinks(g.ctx, g.gf, dump_names);
    }
    return true;
}

bool MageVL::run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<float> * img_feats,
                      const std::vector<int> * img_pos, std::vector<float> & logits_out) {
    const int T = (int)toks.size();
    const int S = n_past + T;
    if (S > n_ctx) { fprintf(stderr, "context overflow (%d > %d)\n", S, n_ctx); return false; }
    const int n_img = (img_feats && img_pos) ? (int)(img_feats->size() / n_embd) : 0;

    // single-token decode: a static graph with a block-rounded KV window.
    // The identical gf object across steps lets ggml capture a CUDA graph,
    // cutting per-step kernel-launch overhead (the moondream/vc3 pattern).
    ggml_tensor * logits_t = nullptr;
    const bool use_static = (T == 1) && n_img == 0 && !ggml_backend_is_cpu(backend);
    if (use_static) {
        const int blk = std::min(vqa::block_for(S), n_ctx);
        if (!dec_gf || dec_blk != blk) {
            if (dec_ctx) ggml_free(dec_ctx);
            struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, true };
            dec_ctx = ggml_init(ip);
            TextGraph dg;
            dg.ctx = dec_ctx;
            if (!build_text_graph(dg, 1, blk, 0)) { ggml_free(dec_ctx); dec_ctx = nullptr; dec_gf = nullptr; return false; }
            dec_gf = dg.gf;
            if (!ggml_gallocr_alloc_graph(allocr, dec_gf)) {
                fprintf(stderr, "static decode graph alloc failed\n");
                ggml_free(dec_ctx); dec_ctx = nullptr; dec_gf = nullptr;
                return false;
            }
            dec_blk = blk;
            dec_tok = ggml_graph_get_tensor(dec_gf, "tok_in");
            dec_pos = ggml_graph_get_tensor(dec_gf, "pos_in");
            dec_mask = ggml_graph_get_tensor(dec_gf, "mask_in");
        }
        ggml_backend_tensor_set(dec_tok, toks.data(), 0, sizeof(int));
        ggml_backend_tensor_set(dec_pos, &n_past, 0, sizeof(int));
        vqa::set_causal_mask(dec_mask, n_past, 1);
        if (ggml_backend_graph_compute(backend, dec_gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "text compute failed\n");
            return false;
        }
        logits_out.resize(n_vocab);
        ggml_backend_tensor_get(ggml_graph_get_tensor(dec_gf, "logits"), logits_out.data(), 0, sizeof(float) * n_vocab);
        n_past = S;
        return true;
    }

    if (!build_text_graph(g, T, S, n_img)) return false;
    if (!ggml_gallocr_alloc_graph(allocr, g.gf)) {
        fprintf(stderr, "text graph allocation failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }

    ggml_backend_tensor_set(g.tok_in, toks.data(), 0, sizeof(int) * T);
    if (n_img > 0) {
        ggml_backend_tensor_set(g.img_in, img_feats->data(), 0, sizeof(float) * img_feats->size());
        ggml_backend_tensor_set(g.img_pos_in, img_pos->data(), 0, sizeof(int) * n_img);
    }
    {
        std::vector<int> posv(T);
        for (int i = 0; i < T; i++) posv[i] = n_past + i;
        ggml_backend_tensor_set(g.pos_in, posv.data(), 0, sizeof(int) * T);
    }
    vqa::set_causal_mask(g.mask_in, n_past, T);

    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "text compute failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }
    dump(g.gf, "emb");
    dump(g.gf, "t_l0_norm");
    dump(g.gf, "t_l0_q_rope");
    dump(g.gf, "t_l0_out");
    dump(g.gf, "t_final_norm");
    dump(g.gf, "logits");

    logits_out.resize(n_vocab);
    ggml_backend_tensor_get(g.logits, logits_out.data(), 0, sizeof(float) * n_vocab);
    n_past = S;
    ggml_free(g.ctx);
    g.ctx = nullptr;
    return true;
}

std::string MageVL::render_prompt(int n_img_tokens, const std::string & user_text, bool video,
                                  const std::vector<float> * frame_secs) const {
    // the checkpoint's chat template: default system prompt + user turn; the
    // processor rewrites a video placeholder into per-frame timestamped blocks
    // (" <X.X seconds>" tags, image_pad each), so both paths emit image_pad
    std::string s = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
    if (n_img_tokens > 0 && video && frame_secs && !frame_secs->empty()) {
        char ts[32];
        for (float sec : *frame_secs) {
            snprintf(ts, sizeof(ts), "<%.1f seconds>", (double)sec);
            s += ts;
            s += "<|vision_start|>";
            for (int i = 0; i < n_img_tokens; i++) s += "<|image_pad|>";
            s += "<|vision_end|>";
        }
    } else if (n_img_tokens > 0) {
        s += "<|vision_start|>";
        for (int i = 0; i < n_img_tokens; i++) s += "<|image_pad|>";
        s += "<|vision_end|>";
    }
    s += user_text;
    s += "<|im_end|>\n<|im_start|>assistant\n";
    return s;
}

int MageVL::sample(const std::vector<float> & logits, const vqa::GenParams & gp) {
    return vqa::sample_logits(logits, gp, rng_state);
}

bool MageVL::ask(const vqa::MediaInput & media, const std::string & prompt,
                 const std::string & mode, const vqa::GenParams & gp, std::string & out_text) {
    // ---- System 1 modes ----------------------------------------------------
    // "gate[:tau]"      score only: per-frame p(speak) report, no generation
    // "gated[:tau]"     generate only when the last frame crosses tau
    if (mode.rfind("gate", 0) == 0) {
        if (!gate_ready_) { out_text = "(StreamMind gate not loaded: drop mage-gate-f16.gguf next to the model)"; return true; }
        const float tau = mode.size() > 4 && mode[4] == ':' ? atof(mode.c_str() + 5) : 0.5f;
        std::vector<float> scores;
        if (!gate_scores(media, scores)) { out_text = "(gate scoring failed)"; return false; }
        const double fps = media.fps > 0 ? media.fps : 2.0;
        char line[96];
        out_text.clear();
        for (size_t i = 0; i < scores.size(); i++) {
            snprintf(line, sizeof(line), "[t=%.1fs] p(speak)=%.3f %s\n",
                     (double)i / fps, (double)scores[i], scores[i] >= tau ? "RESPONSE" : "silence");
            out_text += line;
            if (gp.on_token) gp.on_token(line, gp.on_token_ud);
        }
        return true;
    }
    if (mode.rfind("gated", 0) == 0 && gate_ready_) {
        const float tau = mode.size() > 5 && mode[5] == ':' ? atof(mode.c_str() + 6) : 0.5f;
        std::vector<float> scores;
        if (!gate_scores(media, scores)) return false;
        if (!scores.empty() && scores.back() < tau) {
            out_text = "(silence)";
            if (gp.on_token) gp.on_token(out_text.c_str(), gp.on_token_ud);
            fprintf(stderr, "gate: silence (p=%.3f < %.2f)\n", (double)scores.back(), (double)tau);
            return true;
        }
        if (!scores.empty())
            fprintf(stderr, "gate: response (p=%.3f >= %.2f)\n", (double)scores.back(), (double)tau);
        // fall through to normal generation
    }

    std::vector<float> feats;
    int n_img = 0;
    stage_cb = gp.on_stage;
    stage_ud = gp.on_stage_ud;
    const auto tv0 = std::chrono::steady_clock::now();
    std::vector<float> frame_secs;
    const bool video_kind = media.kind == vqa::MediaKind::Video;
    // spend at most n_ctx - max_tokens - text on vision tokens; measure the text
    // overhead with a 1-placeholder prompt so an oversized image is clamped or
    // rejected before the multi-second vision encode
    {
        const int text_overhead = (int)tok_.encode(render_prompt(1, prompt, video_kind, nullptr)).size();
        const int n_frames = (int)media.frames.size();
        const int budget = n_ctx - gp.max_tokens - text_overhead - 16 - 12 * n_frames;
        if (budget <= 0) {
            fprintf(stderr, "n_ctx %d too small: prompt %d tokens + max_new %d leaves nothing for the image\n",
                    n_ctx, text_overhead, gp.max_tokens);
            return false;
        }
        // media cache: the same picture/video re-encoded on every follow-up
        // question wasted seconds - key on the frame bytes and reuse the feats
        static struct {
            uint64_t key = 0;
            std::vector<float> f;
            int n = 0;
            std::vector<float> secs;
        } vcache;
        uint64_t key = 1469598103934665603ull;
        for (const auto & fr : media.frames) {
            key ^= fr.rgb.size(); key *= 1099511628211ull;
            for (size_t k = 0; k < fr.rgb.size(); k++) { key ^= fr.rgb[k]; key *= 1099511628211ull; }
        }
        key ^= (uint64_t)media.width << 32 ^ (uint64_t)media.height;
        if (vcache.key == key && vcache.n > 0) {
            feats = vcache.f; n_img = vcache.n; frame_secs = vcache.secs;
            fprintf(stderr, "vision: cache hit (%d tokens)\n", n_img);
        } else if (encode_vision(media, feats, &n_img, &frame_secs, budget)) {
            vcache.key = key; vcache.f = feats; vcache.n = n_img; vcache.secs = frame_secs;
        } else {
            return false;
        }
    }
    fprintf(stderr, "[perf] vision encode: %.0f ms (%d tokens)\n",
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tv0).count(), n_img);

    const int n_per_frame = video_kind && !frame_secs.empty() ? n_img / (int)frame_secs.size() : n_img;
    const std::string text = render_prompt(n_per_frame, prompt, video_kind, &frame_secs);
    const std::vector<int> ids = tok_.encode(text);
    std::vector<int> img_pos;
    for (int i = 0; i < (int)ids.size(); i++) if (ids[i] == id_image_pad) img_pos.push_back(i);
    if ((int)img_pos.size() != n_img) {
        fprintf(stderr, "placeholder mismatch: %d image_pad tokens vs %d vision tokens\n",
                (int)img_pos.size(), n_img);
        if (n_img > 0) return false;
    }
    fprintf(stderr, "prompt: %d tokens (%d image), max_new %d\n", (int)ids.size(), n_img, gp.max_tokens);
    int max_new = gp.max_tokens;
    if ((int)ids.size() + max_new > n_ctx) {
        max_new = n_ctx - (int)ids.size();
        fprintf(stderr, "warning: prompt %d + max_new %d > n_ctx %d, clamping max_new to %d\n",
                (int)ids.size(), gp.max_tokens, n_ctx, max_new);
        if (max_new <= 0) return false;
    }

    n_past = 0;
    kv.clear();
    auto t0 = std::chrono::steady_clock::now();
    TextGraph g;
    std::vector<float> logits;
    if (!run_text(g, ids, n_img > 0 ? &feats : nullptr, n_img > 0 ? &img_pos : nullptr, logits))
        return false;
    auto t1 = std::chrono::steady_clock::now();

    out_text.clear();
    int next = sample(logits, gp);
    int n_dec = 0;
    for (int step = 0; step < max_new; step++) {
        if (next == id_eos || next == id_endoftext) break;
        const std::string piece = tok_.decode({next});
        out_text += piece;
        if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
        n_dec++;
        if (step + 1 >= max_new) break;
        if (!run_text(g, {next}, nullptr, nullptr, logits)) break;
        next = sample(logits, gp);
    }
    auto t2 = std::chrono::steady_clock::now();
    const double ms_prefill = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_decode = std::chrono::duration<double, std::milli>(t2 - t1).count();
    fprintf(stderr, "[perf] prefill %d tok in %.0f ms (%.1f tok/s) | decode %d tok in %.0f ms (%.2f tok/s)\n",
            (int)ids.size(), ms_prefill, ms_prefill > 0 ? ids.size() * 1000.0 / ms_prefill : 0.0,
            n_dec, ms_decode, ms_decode > 0 ? n_dec * 1000.0 / ms_decode : 0.0);
    return true;
}

}  // namespace mage_vl

namespace vqa {
Model * create_mage_vl() { return new mage_vl::MageVL(); }
}  // namespace vqa
