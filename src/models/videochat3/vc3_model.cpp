// vc3_model.cpp - VideoChat3-4B backend (Qwen3 text decoder + VideoChat3 ViT).
//
// Conventions:
//   * GGUF linear weights are stored as ne = [in, out]; consumption is
//     ggml_mul_mat(W, x[in, T]) -> [out, T].
//   * the text decoder is fed a host-assembled F32 embedding matrix [n_embd, T]
//     (token rows read back from the possibly quantized token_embd, image rows
//     spliced in), so a single graph shape covers text and image prompts.
//   * text attention uses a per-layer F16 KV cache [n_kv_head*head_dim, n_ctx]
//     with flash attention and an explicit F16 causal mask.
//   * the vision tower runs as its own graph per clip: patch embed, 27 blocks
//     with 2D rope and full (non causal) attention, final LN, merge, projector.
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
#include "vc3_preprocess.hpp"
#include "attention.hpp"
#include "sampler.hpp"
#include "vqa_util.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace vc3 {

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

class VideoChat3 : public vqa::Model {
public:
    ~VideoChat3() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "videochat3-4b"; }
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp, std::string & out_text) override;

    // tunables, set before load()
    int n_ctx = 8192;
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool alloc_kv();
    // clip_marks: per temporal clip (timestamp, token count) for prompt building
    bool encode_vision(const vqa::MediaInput & media, std::vector<float> & feats, int * n_tokens,
                       std::vector<std::pair<float,int>> * clip_marks = nullptr);
    bool build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all = false);
    bool run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<float> * img_feats,
                  const std::vector<int> * img_pos, std::vector<float> & logits_out);
    std::string render_prompt(int n_img_tokens, const std::string & user_text, bool video,
                               const std::vector<std::pair<float,int>> * clip_marks = nullptr) const;
    int sample(const std::vector<float> & logits, const vqa::GenParams & gp);
    float randf();
    void dump(const ggml_cgraph * gf, const char * name);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    int n_layer = 36, n_embd = 2560, n_head = 32, n_kv_head = 8, head_dim = 128;
    int n_ff = 9728, n_vocab = 151936;
    float rms_eps = 1e-6f, rope_theta = 5e6f;
    VisConfig vc;

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;         // weights: mmap-backed (CPU) or uploaded
    ggml_backend_buffer_t wbuf_plain = nullptr;   // host f32 block holding the 1D tensors
    std::vector<char> host_f32_storage;           // keeps that block alive on the CPU path
    vqa::MMap mm;

    std::vector<float> pos_emb_host, pos_time_host;

    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr;
    std::vector<TxtLayer> tl;
    std::vector<VisLayer> vl;
    ggml_tensor *v_patch_w = nullptr, *v_patch_b = nullptr;
    ggml_tensor *v_final_w = nullptr, *v_final_b = nullptr;
    ggml_tensor *mm_pre_w = nullptr, *mm_pre_b = nullptr;
    ggml_tensor *mm_l1_w = nullptr, *mm_l1_b = nullptr;
    ggml_tensor *mm_l2_w = nullptr, *mm_l2_b = nullptr;

    vqa::KvCache kv;
    // batch decode (vc3_batch.inc)
    int bat_n = 0;
    ggml_context * bat_ctx = nullptr;
    ggml_cgraph * bat_gf = nullptr;
    ggml_tensor * bat_tok = nullptr, * bat_pos = nullptr, * bat_mask = nullptr;
    int step_batch(const std::vector<int> & toks, const std::vector<int> & posv,
                   const std::vector<int> & lo, const std::vector<int> & hi, float * logits_out);
    int run_batch(const std::string & path, int n_gen);
    // static decode graph (T=1, fixed KV window per block): same gf object across
    // steps -> eligible for ggml CUDA graph capture (see moondream Engine::step)
    ggml_context * dec_ctx = nullptr;
    ggml_cgraph * dec_gf = nullptr;
    int dec_blk = 0;
    ggml_tensor * dec_tok = nullptr, * dec_pos = nullptr, * dec_mask = nullptr;
    ggml_gallocr_t allocr = nullptr;
    int n_past = 0;

    int id_image_pad = 151655, id_video_pad = 151656;
    int id_vision_start = 151652, id_vision_end = 151653;
    int id_im_start = 151644, id_im_end = 151645, id_eos = 151645, id_endoftext = 151643;

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

// per-pair complex rotation on an interleaved head vector: pair p = (v[2p], v[2p+1]).
// cos_t/sin_t are [1, pairs, H, N] - pre-expanded on the host to re/im's exact shape,
// so every elementwise op is same-shape. ore/oim land in a preallocated interleaved
// tensor via two cpy ops - a dim0 concat here explodes into thousands of tiny kernel
// launches under WDDM.
// per-pair complex rotation on an interleaved head vector: pair p = (v[2p], v[2p+1]).
// cos_t/sin_t are [1, pairs, H*N] - pre-expanded on the host to re/im's exact shape.
// Everything is kept 3d: the CUDA concat launcher loops over ne[3], so a 4d layout
// with ne3=N explodes into thousands of tiny kernel launches under WDDM.
static ggml_tensor * rope_2d(ggml_context * ctx, struct ggml_cgraph * gf, ggml_tensor * x,
                             ggml_tensor * cos_t, ggml_tensor * sin_t) {
    const int64_t hd = x->ne[0], H = x->ne[1], N = x->ne[2];
    const int64_t pairs = hd / 2, M = H * N;
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, 2, pairs, M);
    ggml_tensor * re = ggml_cont(ctx, ggml_view_3d(ctx, x3, 1, pairs, M, x3->nb[1], x3->nb[2], 0));
    ggml_tensor * im = ggml_cont(ctx, ggml_view_3d(ctx, x3, 1, pairs, M, x3->nb[1], x3->nb[2], x3->nb[0]));
    ggml_tensor * c = ggml_reshape_3d(ctx, cos_t, 1, pairs, M);
    ggml_tensor * sn = ggml_reshape_3d(ctx, sin_t, 1, pairs, M);
    ggml_tensor * ore = ggml_sub(ctx, ggml_mul(ctx, re, c), ggml_mul(ctx, im, sn));
    ggml_tensor * oim = ggml_add(ctx, ggml_mul(ctx, re, sn), ggml_mul(ctx, im, c));
    return ggml_reshape_3d(ctx, ggml_concat(ctx, ore, oim, 0), hd, H, N);
}

float VideoChat3::randf() {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (float)(((rng_state * 0x2545F4914F6CDD1Dull) >> 11) * (1.0 / 9007199254740992.0));
}

VideoChat3::~VideoChat3() {
    if (allocr) ggml_gallocr_free(allocr);
    if (bat_ctx) ggml_free(bat_ctx);
    if (dec_ctx) ggml_free(dec_ctx);
    kv.free();
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wbuf_plain) ggml_backend_buffer_free(wbuf_plain);
    if (wctx) ggml_free(wctx);
    if (backend) ggml_backend_free(backend);
}

bool VideoChat3::init_backend() {
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
bool VideoChat3::load(const std::string & gguf_path, const std::string & tokenizer_path,
                      const vqa::LoadOptions & opt) {
    n_ctx = opt.n_ctx > 0 ? opt.n_ctx : n_ctx;
    n_threads = opt.n_threads;
    use_gpu = opt.use_gpu;
    dump_dir = opt.dump_dir;
    if (const char * d = getenv("VC3_DUMP_DIR")) dump_dir = d;   // dev override
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
    caps_.proactive_states = true;
    caps_.timestamps_prompt = true;
    fprintf(stderr, "videochat3: ready (%d text layers, %d vision layers, ctx %d)\n",
            n_layer, vc.n_layer, n_ctx);
    return true;
}

bool VideoChat3::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f GB\n", mm.size / 1e9);

    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer   = gguf_u32(gu, "videochat3.block_count", n_layer);
    n_embd    = gguf_u32(gu, "videochat3.embedding_length", n_embd);
    n_head    = gguf_u32(gu, "videochat3.attention.head_count", n_head);
    n_kv_head = gguf_u32(gu, "videochat3.attention.head_count_kv", n_kv_head);
    head_dim  = gguf_u32(gu, "videochat3.attention.key_length", head_dim);
    n_ff      = gguf_u32(gu, "videochat3.feed_forward_length", n_ff);
    n_vocab   = gguf_u32(gu, "videochat3.vocab_size", n_vocab);
    rms_eps   = gguf_f32(gu, "videochat3.attention.layer_norm_rms_epsilon", rms_eps);
    rope_theta = gguf_f32(gu, "videochat3.rope.freq_base", rope_theta);

    vc.n_layer = gguf_u32(gu, "videochat3.vision.block_count", vc.n_layer);
    vc.n_embd  = gguf_u32(gu, "videochat3.vision.embedding_length", vc.n_embd);
    vc.n_head  = gguf_u32(gu, "videochat3.vision.attention.head_count", vc.n_head);
    vc.n_ff    = gguf_u32(gu, "videochat3.vision.feed_forward_length", vc.n_ff);
    vc.patch   = gguf_u32(gu, "videochat3.vision.patch_size", vc.patch);
    vc.merge   = gguf_u32(gu, "videochat3.vision.merge_size", vc.merge);
    vc.temporal_merge = gguf_u32(gu, "videochat3.vision.temporal_merge_size", vc.temporal_merge);
    vc.pos_h   = gguf_u32(gu, "videochat3.vision.pos_emb_height", vc.pos_h);
    vc.pos_w   = gguf_u32(gu, "videochat3.vision.pos_emb_width", vc.pos_w);
    vc.layernorm_eps = gguf_f32(gu, "videochat3.vision.layer_norm_epsilon", vc.layernorm_eps);
    vc.rope_theta = gguf_f32(gu, "videochat3.vision.rope_theta", vc.rope_theta);
    if (gguf_u32(gu, "videochat3.vision.rope_layout", 0) != 2) {
        fprintf(stderr, "this build expects a rope_layout=2 GGUF (wqkv rows permuted + zero-padded for the fused vision rope); re-run convert/convert_videochat3.py\n");
        return false;
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
    std::vector<char> host_f32;
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
        ggml_tensor * pe = ggml_get_tensor(wctx, "v.pos_emb.weight");
        ggml_tensor * tw = ggml_get_tensor(wctx, "v.pos_emb.time_weight");
        if (!pe || !tw) { fprintf(stderr, "missing positional embedding tensors\n"); return false; }
        pos_emb_host.resize(ggml_nelements(pe));
        pos_time_host.resize(ggml_nelements(tw));
        tensor_to_f32(pe, pos_emb_host.data());
        tensor_to_f32(tw, pos_time_host.data());
    }

    {
        std::vector<std::pair<ggml_tensor *, const void *>> uploads;
        size_t total = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            uploads.push_back({t, t->data});
            total += ggml_nbytes(t);
        }

        if (ggml_backend_is_cpu(backend)) {
            // zero-copy (same trick as the moondream engine): the weights keep
            // pointing into the mmap, the f32-converted 1D tensors into their host
            // block. Avoids a second multi-GB commit on Windows, where the mmap
            // already counts against the commit limit - load is faster and the
            // resident footprint is roughly halved.
            wbuf = ggml_backend_cpu_buffer_from_ptr(mm.ptr, mm.size);
            const char * f32_begin = host_f32.empty() ? nullptr : host_f32.data();
            const char * f32_end = f32_begin ? f32_begin + host_f32.size() : nullptr;
            if (f32_begin) wbuf_plain = ggml_backend_cpu_buffer_from_ptr(host_f32.data(), host_f32.size());
            for (auto & u : uploads) {
                ggml_tensor * t = u.first;
                const char * p = (const char *)t->data;
                t->buffer = (f32_begin && p >= f32_begin && p < f32_end) ? wbuf_plain : wbuf;
            }
            fprintf(stderr, "weights: %.2f GB zero-copy from mmap (%s)\n", total / 1e9,
                    f32_begin ? "1D tensors in host f32 block" : "all in mmap");
        } else {
            for (auto & u : uploads) u.first->data = nullptr;
            wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
            if (!wbuf) { fprintf(stderr, "weight allocation failed (need %.2f GB)\n", total / 1e9); return false; }
            for (auto & u : uploads) ggml_backend_tensor_set(u.first, u.second, 0, ggml_nbytes(u.first));
            fprintf(stderr, "uploaded %.2f GB of weights\n", total / 1e9);
        }
    }
    if (ggml_backend_is_cpu(backend)) {
        host_f32_storage = std::move(host_f32);   // the 1D tensors point into it
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
    v_patch_w = gt("v.patch_embd.weight");
    v_patch_b = gt("v.patch_embd.bias");
    v_final_w = gt("v.final_norm.weight");
    v_final_b = gt("v.final_norm.bias");
    mm_pre_w = gt("mm.pre_norm.weight");
    mm_pre_b = gt("mm.pre_norm.bias");
    mm_l1_w = gt("mm.linear_1.weight");
    mm_l1_b = gt("mm.linear_1.bias");
    mm_l2_w = gt("mm.linear_2.weight");
    mm_l2_b = gt("mm.linear_2.bias");

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
    return tok_embd != nullptr && out_norm != nullptr && v_patch_w != nullptr;
}

bool VideoChat3::alloc_kv() {
    return kv.alloc(backend, n_layer, n_kv_head, head_dim, n_ctx);
}

void VideoChat3::dump(const ggml_cgraph * gf, const char * name) {
    if (dump_dir.empty()) return;
    ggml_tensor * t = ggml_graph_get_tensor((ggml_cgraph *)gf, name);
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

// ---------------------------------------------------------------- vision
bool VideoChat3::encode_vision(const vqa::MediaInput & media, std::vector<float> & feats, int * n_tokens,
                               std::vector<std::pair<float,int>> * clip_marks) {
    *n_tokens = 0;
    if (media.empty()) return true;
    const int T = (int)media.frames.size();
    const int src_h = media.height, src_w = media.width;
    const int factor = vc.patch * vc.merge;

    int H = 0, W = 0;
    // qwen_vl_utils defaults used by the official demo: min 56*56, max 16384*28*28
    if (!smart_resize(src_h, src_w, factor, 3136, 16384 * 28 * 28, &H, &W)) {
        fprintf(stderr, "image too small (%dx%d, factor %d)\n", src_w, src_h, factor);
        return false;
    }
    if (media.kind == vqa::MediaKind::Video)
        fprintf(stderr, "video %d frame%s %dx%d -> %dx%d each\n", T, T == 1 ? "" : "s", src_w, src_h, W, H);
    else
        fprintf(stderr, "image %dx%d -> %dx%d\n", src_w, src_h, W, H);

    // clips of temporal_merge frames each (modeling_videochat3.py split_grid_thws_clip_by_clip):
    // attention is block-diagonal per clip, the temporal pos index restarts per clip and
    // the merger mean-pools the time axis of each clip
    std::vector<int> clip_t;
    for (int off = 0; off < T; off += vc.temporal_merge)
        clip_t.push_back(std::min(vc.temporal_merge, T - off));
    // per-clip prompt timestamps: mean of the first/last frame time in the clip
    // (VideoChat3Processor._calculate_timestamps semantics)
    std::vector<std::pair<float,int>> marks;
    if (media.kind == vqa::MediaKind::Video) {
        const double def_fps = media.fps > 0 ? media.fps : 1.0;
        int off = 0;
        for (int t : clip_t) {
            const double t0 = off < (int)media.timestamps.size() ? media.timestamps[off] : off / def_fps;
            const double t1 = (off + t - 1) < (int)media.timestamps.size() ? media.timestamps[off + t - 1] : (off + t - 1) / def_fps;
            marks.push_back({ (float)((t0 + t1) / 2.0), 0 });   // token count filled after patchify
            off += t;
        }
    }

    std::vector<float> rgb, patches, pos, pos_clip;
    int gh = 0, gw = 0;
    for (int ti = 0; ti < T; ti++) {
        resize_bicubic_antialias(media.frames[ti].rgb.data(), src_w, src_h, W, H, rgb);
        std::vector<float> p;
        patchify(rgb.data(), H, W, vc, p, &gh, &gw);
        patches.insert(patches.end(), p.begin(), p.end());
    }
    const int hw = gh * gw;
    const int N = hw * T;
    for (int t : clip_t) {
        pos_embed_for_grid(pos_emb_host.data(), pos_time_host.data(), vc, t, gh, gw, pos_clip);
        pos.insert(pos.end(), pos_clip.begin(), pos_clip.end());
    }

    const int vhd = vc.n_embd / vc.n_head;
    const bool pad80 = vhd == 72;
    // 2D positions for ggml's fused GGML_ROPE_TYPE_VISION kernel: the converter
    // permutes the wqkv q/k rows into its (x-pairs | y-pairs, re|im) layout, so
    // rope is one fused kernel instead of the former ~9-node rotation subgraph
    std::vector<int> vpos((size_t)N * 4, 0);   // 4 position columns, vision uses 2
    for (int s = 0; s < N; s++) {
        const int g = s % hw;   // frame-local (h, w): clips restart the 2D grid
        vpos[s] = g % gw;              // section 0 coordinate = x
        vpos[(size_t)N + s] = g / gw;          // section 1 coordinate = y
    }

    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) return false;
    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);

    ggml_tensor * t_patches = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 3 * vc.patch * vc.patch, N);
    ggml_set_name(t_patches, "patches");
    ggml_tensor * t_pos = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, vc.n_embd, N);
    ggml_set_name(t_pos, "vpos");
    ggml_tensor * t_vpos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, (int64_t)N * 4);
    ggml_set_name(t_vpos, "vxy");
    ggml_tensor * t_zmask = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, N, N);
    ggml_set_name(t_zmask, "vzmask");

    ggml_tensor * x = ggml_mul_mat(gctx, v_patch_w, t_patches);
    x = ggml_add(gctx, x, v_patch_b);
    x = ggml_add(gctx, x, t_pos);
    ggml_set_name(x, "v_patch_out");

    // parity helper: truncate the graph at a named node (the sink is then never
    // buffer-reused, so its dump is valid) -- scripts/parity_vc3_video.py
    const char * stop_after = getenv("VC3_STOP_AFTER");
    auto run = [&](const char * out_name) -> bool {
        const auto ra = std::chrono::steady_clock::now();
        if (!ggml_gallocr_alloc_graph(allocr, gf)) {
            fprintf(stderr, "vision graph allocation failed\n");
            return false;
        }
        static const bool vprof = getenv("VC3_PROF") != nullptr;
        static double v_alloc = 0, v_input = 0, v_comp = 0; static int v_runs = 0;
        const auto rb = std::chrono::steady_clock::now();
        // truncated graphs may leave unused inputs unallocated (buffer == NULL)
        if (t_patches->buffer) ggml_backend_tensor_set(t_patches, patches.data(), 0, sizeof(float) * patches.size());
        if (t_pos->buffer)     ggml_backend_tensor_set(t_pos, pos.data(), 0, sizeof(float) * pos.size());
        if (t_vpos->buffer) ggml_backend_tensor_set(t_vpos, vpos.data(), 0, sizeof(int) * vpos.size());
        fprintf(stderr, "[run] inputs set\n");
        if (t_zmask->buffer) {
            // block-diagonal attention: tokens attend within their clip only
            // (cu_seqlens mask in modeling_videochat3.py). Single clip = all zeros.
            std::vector<ggml_fp16_t> mask((size_t)N * N, ggml_fp32_to_fp16(0.0f));
            if (clip_t.size() > 1) {
                std::vector<int> tok_clip(N);
                {
                    int s = 0;
                    for (size_t c = 0; c < clip_t.size(); c++)
                        for (int i = 0; i < clip_t[c] * hw; i++) tok_clip[s++] = (int)c;
                }
                const ggml_fp16_t neg = ggml_fp32_to_fp16(-INFINITY);
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        if (tok_clip[i] != tok_clip[j]) mask[(size_t)i * N + j] = neg;
            }
            ggml_backend_tensor_set(t_zmask, mask.data(), 0, sizeof(ggml_fp16_t) * mask.size());
        }
        const auto rc = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "vision compute failed\n");
            return false;
        }
        const auto rd = std::chrono::steady_clock::now();
        if (vprof) {
            v_runs++;
            v_alloc += std::chrono::duration<double,std::milli>(rb-ra).count();
            v_input += std::chrono::duration<double,std::milli>(rc-rb).count();
            v_comp  += std::chrono::duration<double,std::milli>(rd-rc).count();
            fprintf(stderr, "[vprof] run=%d alloc %.1fms input %.1fms compute %.1fms\n",
                    v_runs, v_alloc/v_runs, v_input/v_runs, v_comp/v_runs);
        }
        dump(gf, "patches");
        dump(gf, "vpos");
        dump(gf, "vcos");
        dump(gf, "vsin");
        dump(gf, "v_patch_out");
        dump(gf, "v_blk0_qkv");
        dump(gf, "v_blk0_q_rope");
        dump(gf, "v_blk0_k_rope");
        dump(gf, "v_blk0_attn_out");
        dump(gf, "v_blk0_out");
        dump(gf, "v_final");
        dump(gf, "v_merge");
        dump(gf, "v_proj_out");
        dump(gf, out_name);   // sink dump is always buffer-safe
        ggml_tensor * out = ggml_graph_get_tensor(gf, out_name);
        const int n = (int)out->ne[1];
        const int d = (int)out->ne[0];   // vision dim (!= n_embd before the projector)
        feats.resize((size_t)n * d);
        ggml_backend_tensor_get(out, feats.data(), 0, sizeof(float) * feats.size());
        *n_tokens = n;
    {
        const int per_clip_tok = (gh / vc.merge) * (gw / vc.merge);
        for (auto & m : marks) m.second = per_clip_tok;
        if (clip_marks) *clip_marks = marks;
    }
        fprintf(stderr, "vision: %d patches (%dx%d) -> %d tokens%s\n", N, gh, gw, n,
                stop_after ? " (truncated for parity)" : "");
        return true;
    };
    if (stop_after && !strcmp(stop_after, "v_patch_out")) {
        ggml_build_forward_expand(gf, x);
        if (!run("v_patch_out")) { ggml_free(gctx); return false; }
        ggml_free(gctx);
        return true;
    }

    int vl_max = vc.n_layer;
    { const char * e = getenv("VC3_VL"); if (e) vl_max = std::min(vl_max, atoi(e)); }
    for (int il = 0; il < vl_max; il++) {
        VisLayer & L = vl[il];
        ggml_tensor * l = ggml_norm(gctx, x, vc.layernorm_eps);
        l = ggml_mul(gctx, l, L.norm0_w);
        l = ggml_add(gctx, l, L.norm0_b);
        ggml_tensor * qkv = ggml_mul_mat(gctx, L.wqkv_w, l);
        qkv = ggml_add(gctx, qkv, L.wqkv_b);

        // qkv packs the three padded projections along ne0; the converter pads and
        // permutes the rows so each [80]-wide head slice works as-is - feed strided
        // views straight into rope/FA, no cont(), no runtime padding copies
        const int vhd_f = pad80 ? 80 : vhd;
        const size_t off = (size_t)vc.n_head * vhd_f * qkv->nb[0];
        auto qkv_head_view = [&](int block) {
            return ggml_view_3d(gctx, qkv, vhd_f, vc.n_head, N,
                                (size_t)vhd_f * qkv->nb[0], qkv->nb[1], (size_t)block * off);
        };
        ggml_tensor * q = qkv_head_view(0);
        ggml_tensor * k = qkv_head_view(1);
        ggml_tensor * v = qkv_head_view(2);
        if (il == 0) ggml_set_name(qkv, "v_blk0_qkv");
        if (il == 0 && stop_after && !strcmp(stop_after, "v_blk0_qkv")) {
            ggml_build_forward_expand(gf, qkv);
            if (!run("v_blk0_qkv")) { ggml_free(gctx); return false; }
            ggml_free(gctx);
            return true;
        }

        // rope: the converter's 72->80 pad makes n_dims 40; rescale freq_base so
        // each real pair keeps its exact frequency (theta**(-p/18) == (theta**(20/18))**(-2p/40))
        int sections[GGML_MROPE_SECTIONS] = { vhd / 4, vhd / 4, 0, 0 };
        const float rope_base = pad80 ? powf(vc.rope_theta, (float)vhd_f / vhd) : vc.rope_theta;
        q = ggml_rope_multi(gctx, q, t_vpos, nullptr, vhd_f / 2, sections, GGML_ROPE_TYPE_VISION,
                            n_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_multi(gctx, k, t_vpos, nullptr, vhd_f / 2, sections, GGML_ROPE_TYPE_VISION,
                            n_ctx, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (il == 0) {
            ggml_set_name(q, "v_blk0_q_rope"); ggml_set_name(k, "v_blk0_k_rope");
            if (stop_after && !strcmp(stop_after, "v_blk0_q_rope")) {
                ggml_build_forward_expand(gf, q);
                if (!run("v_blk0_q_rope")) { ggml_free(gctx); return false; }
                ggml_free(gctx);
                return true;
            }
        }

        // attention runs clip-locally with NO mask: a masked flash-attn falls off
        // the fast path, and clip tokens are contiguous, so per-clip unmasked FA
        // (same trick as SkyVision.cpp's multi-crop SigLIP) is both exact and fast.
        auto fa_in = [&](ggml_tensor * t, int n_tok, size_t tok_off) {
            return ggml_view_3d(gctx, t, vhd_f, n_tok, vc.n_head, t->nb[2], t->nb[1], tok_off * t->nb[2]);
        };
        ggml_tensor * fa = nullptr;
        if (clip_t.size() == 1) {
            ggml_tensor * qf = fa_in(q, N, 0);
            ggml_tensor * kf = fa_in(k, N, 0);
            ggml_tensor * vf = fa_in(v, N, 0);
            fa = ggml_flash_attn_ext(gctx, qf, kf, vf, nullptr, 1.0f / sqrtf((float)vhd), 0.0f, 0.0f);
        } else {
            size_t coff = 0;
            for (int t : clip_t) {
                const int cn = t * hw;
                ggml_tensor * qf = fa_in(q, cn, coff);
                ggml_tensor * kf = fa_in(k, cn, coff);
                ggml_tensor * vf = fa_in(v, cn, coff);
                ggml_tensor * fc = ggml_flash_attn_ext(gctx, qf, kf, vf, nullptr, 1.0f / sqrtf((float)vhd), 0.0f, 0.0f);
                fc = ggml_cont(gctx, fc);
                fa = fa ? ggml_concat(gctx, fa, fc, 2) : fc;   // fa out is [hd, heads, tokens]: concat tokens (dim2)
                coff += (size_t)cn;
            }
        }
        // fa is [vhd_f, n_head, N]; drop the padding cols and lay out rows as tokens
        ggml_tensor * attn_src = fa;
        if (pad80)
            attn_src = ggml_view_3d(gctx, fa, vhd, vc.n_head, N,
                                    (size_t)vhd_f * sizeof(float), (size_t)vhd_f * vc.n_head * sizeof(float), 0);
        ggml_tensor * attn =
                ggml_reshape_2d(gctx, ggml_cont(gctx, attn_src), vc.n_embd, N);
        attn = ggml_mul_mat(gctx, L.wo_w, attn);
        attn = ggml_add(gctx, attn, L.wo_b);
        x = ggml_add(gctx, x, attn);
        if (il == 0) {
            ggml_set_name(x, "v_blk0_attn_out");
            if (stop_after && !strcmp(stop_after, "v_blk0_attn_out")) {
                ggml_build_forward_expand(gf, x);
                if (!run("v_blk0_attn_out")) { ggml_free(gctx); return false; }
                ggml_free(gctx);
                return true;
            }
        }

        ggml_tensor * m = ggml_norm(gctx, x, vc.layernorm_eps);
        m = ggml_mul(gctx, m, L.norm1_w);
        m = ggml_add(gctx, m, L.norm1_b);
        m = ggml_mul_mat(gctx, L.fc0_w, m);
        m = ggml_add(gctx, m, L.fc0_b);
        m = ggml_gelu(gctx, m);                       // gelu_pytorch_tanh
        m = ggml_mul_mat(gctx, L.fc1_w, m);
        m = ggml_add(gctx, m, L.fc1_b);
        x = ggml_add(gctx, x, m);
        if (il == 0) {
            ggml_set_name(x, "v_blk0_out");
            if (stop_after && !strcmp(stop_after, "v_blk0_out")) {
                ggml_build_forward_expand(gf, x);
                if (!run("v_blk0_out")) { ggml_free(gctx); return false; }
                ggml_free(gctx);
                return true;
            }
        }
    }
    x = ggml_norm(gctx, x, vc.layernorm_eps);
    x = ggml_mul(gctx, x, v_final_w);
    x = ggml_add(gctx, x, v_final_b);
    ggml_set_name(x, "v_final");
    if (stop_after && !strcmp(stop_after, "v_final")) {
        ggml_build_forward_expand(gf, x);
        if (!run("v_final")) { ggml_free(gctx); return false; }
        ggml_free(gctx);
        return true;
    }

    // merge: the reference reads the sequence as (t, nh, kh, nw, kw) and mean-pools
    // the time axis of each clip, then folds (kh,kw) into the feature dim
    // (patch_merger in modeling_videochat3.py); the processor emits (nh, nw, kh, kw)
    const int nh = gh / vc.merge, nw = gw / vc.merge;
    const int kh = vc.merge, kw = vc.merge;
    {
        // temporal mean-pool each clip: sum its frames' [D, hw] slices, scale by 1/t
        ggml_tensor * pooled = nullptr;
        size_t off = 0;
        for (int t : clip_t) {
            ggml_tensor * acc = ggml_view_2d(gctx, x, vc.n_embd, hw, x->nb[1], off);
            for (int ti = 1; ti < t; ti++)
                acc = ggml_add(gctx, acc,
                        ggml_view_2d(gctx, x, vc.n_embd, hw, x->nb[1], off + (size_t)ti * hw * x->nb[1]));
            acc = ggml_scale(gctx, acc, 1.0f / t);
            pooled = pooled ? ggml_concat(gctx, pooled, acc, 1) : acc;
            off += (size_t)t * hw * x->nb[1];
        }
        x = pooled;
        ggml_tensor * x4 = ggml_reshape_4d(gctx, x, (int64_t)vc.n_embd * kw, nw, kh, (int64_t)nh * clip_t.size());
        x4 = ggml_cont(gctx, ggml_permute(gctx, x4, 0, 2, 1, 3));
        ggml_tensor * x3 = ggml_reshape_3d(gctx, x4, vc.n_embd, kw * kh, (int64_t)nh * nw * clip_t.size());
        ggml_set_name(x3, "v_merge");
        if (stop_after && !strcmp(stop_after, "v_merge")) {
            ggml_build_forward_expand(gf, x3);
            if (!run("v_merge")) { ggml_free(gctx); return false; }
            ggml_free(gctx);
            return true;
        }
        ggml_tensor * pn = ggml_norm(gctx, x3, 1e-5f);
        pn = ggml_mul(gctx, pn, mm_pre_w);
        pn = ggml_add(gctx, pn, mm_pre_b);
        ggml_tensor * x2 = ggml_reshape_2d(gctx, pn, (int64_t)vc.n_embd * kw * kh, (int64_t)nh * nw * clip_t.size());
        ggml_tensor * h = ggml_mul_mat(gctx, mm_l1_w, x2);
        h = ggml_add(gctx, h, mm_l1_b);
        h = ggml_gelu_erf(gctx, h);                   // nn.GELU() == exact erf
        h = ggml_mul_mat(gctx, mm_l2_w, h);
        h = ggml_add(gctx, h, mm_l2_b);
        ggml_set_name(h, "v_proj_out");
        ggml_build_forward_expand(gf, h);
    }

    if (!run("v_proj_out")) { ggml_free(gctx); return false; }
    ggml_free(gctx);
    return true;
}

// ---------------------------------------------------------------- text
bool VideoChat3::build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all) {
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

    // token embedding stays on the device (works with a quantized token_embd and
    // avoids one host round-trip per prompt token); image rows are scattered into
    // it exactly like llama.cpp does for multimodal batches
    ggml_tensor * x = ggml_get_rows(g.ctx, tok_embd, g.tok_in);      // [D, T]
    if (n_img > 0) {
        g.img_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, n_img);
        ggml_set_name(g.img_in, "img_in");
        g.img_pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_img);
        ggml_set_name(g.img_pos_in, "img_pos_in");
        // ggml_set_rows writes into its destination, so it needs its own copy
        x = ggml_set_rows(g.ctx, ggml_dup(g.ctx, x), g.img_in, g.img_pos_in);
    }
    ggml_set_name(x, "emb");

    int max_layers = n_layer;
    if (const char * e = getenv("VC3_MAX_LAYERS")) max_layers = std::min(max_layers, atoi(e));
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

        // q3 is [HD, NH, T]; flash attention wants [HD, n_q, n_head], so swap the
        // two trailing strides. The KV cache is [NKV*HD, n_ctx]: one position is
        // NKV*HD halves and one head inside it is HD halves.
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
        static const bool no_fusion = getenv("VC3_NO_FUSION") != nullptr;
        if (no_fusion) {
            m = ggml_mul(g.ctx, ggml_silu(g.ctx, gate), up);          // silu(gate) * up
        } else {
            m = ggml_swiglu_split(g.ctx, gate, up);                   // fused, no intermediate
        }
        if (il == 0) ggml_set_name(m, "t_l0_swiglu");
        m = ggml_mul_mat(g.ctx, L.down, m);
        x = ggml_add(g.ctx, x, m);
        if (il == 0) ggml_set_name(x, "t_l0_out");
    }

    x = ggml_rms_norm(g.ctx, x, rms_eps);
    x = ggml_mul(g.ctx, x, out_norm);
    ggml_set_name(x, "t_final_norm");

    if (logits_all) {
        g.logits = ggml_mul_mat(g.ctx, tok_embd, x);   // [vocab, T] for batch decode
    } else {
        ggml_tensor * last = ggml_view_2d(g.ctx, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
        g.logits = ggml_mul_mat(g.ctx, tok_embd, last);
    }
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);
    return true;
}

bool VideoChat3::run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<float> * img_feats,
                          const std::vector<int> * img_pos, std::vector<float> & logits_out) {
    const int T = (int)toks.size();
    const int S = n_past + T;
    if (S > n_ctx) { fprintf(stderr, "context overflow (%d > %d)\n", S, n_ctx); return false; }
    const int n_img = (img_feats && img_pos) ? (int)(img_feats->size() / n_embd) : 0;
    static const bool prof = getenv("VC3_PROF") != nullptr;
    static double tb = 0, ta = 0, tc = 0; static int ns = 0;
    auto tick = [](){ return std::chrono::steady_clock::now(); };
    auto tp0 = tick();

    // single-token decode: a static graph with a block-rounded KV window.
    // The identical gf object across steps lets ggml capture a CUDA graph,
    // cutting per-step kernel-launch overhead (the moondream Engine pattern).
    ggml_tensor * logits_t = nullptr;
    const bool use_static = (T == 1) && n_img == 0 && !ggml_backend_is_cpu(backend);
    if (use_static) {
        const int blk = std::min(vqa::block_for(S), n_ctx);
        if (!dec_gf || dec_blk != blk) {
            if (dec_ctx) ggml_free(dec_ctx);
            struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, true };
            dec_ctx = ggml_init(ip);
            TextGraph dg;
            // build directly into the persistent context
            const int D = n_embd;
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
        auto tp2 = tick();
        if (ggml_backend_graph_compute(backend, dec_gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "text compute failed\n");
            return false;
        }
        if (prof) {
            ns++;
            tb += std::chrono::duration<double,std::milli>(tp2-tp0).count();
            tc += std::chrono::duration<double,std::milli>(tick()-tp2).count();
            if (ns % 50 == 0)
                fprintf(stderr, "[prof] steps=%d build+input %.2fms compute %.2fms (avg, static)\n", ns, tb/ns, tc/ns);
        }
        logits_out.resize(n_vocab);
        ggml_backend_tensor_get(ggml_graph_get_tensor(dec_gf, "logits"), logits_out.data(), 0, sizeof(float) * n_vocab);
        n_past = S;
        return true;
    }

    if (!build_text_graph(g, T, S, n_img)) return false;
    auto tp1 = tick();
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
    auto tp2 = tick();

    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "text compute failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }
    auto tp3 = tick();
    if (prof) {
        ns++;
        tb += std::chrono::duration<double,std::milli>(tp1-tp0).count();
        ta += std::chrono::duration<double,std::milli>(tp2-tp1).count();
        tc += std::chrono::duration<double,std::milli>(tp3-tp2).count();
        if (ns % 50 == 0)
            fprintf(stderr, "[prof] steps=%d build %.2fms alloc+input %.2fms compute %.2fms (avg)\n",
                    ns, tb/ns, ta/ns, tc/ns);
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

std::string VideoChat3::render_prompt(int n_img_tokens, const std::string & user_text, bool video,
                                       const std::vector<std::pair<float,int>> * clip_marks) const {
    std::string s = "<|im_start|>user\n";
    if (n_img_tokens > 0) {
        const char * pad = video ? "<|video_pad|>" : "<|image_pad|>";
        if (video && clip_marks && !clip_marks->empty()) {
            // official processor format: one timestamped segment per temporal clip
            // ("<t seconds>" = mean of the clip's first/last frame times)
            char ts[32];
            for (const auto & m : *clip_marks) {
                snprintf(ts, sizeof(ts), "<%.1f seconds>", m.first);
                s += ts;
                s += "<|vision_start|>";
                for (int i = 0; i < m.second; i++) s += pad;
                s += "<|vision_end|>";
            }
        } else {
            s += "<|vision_start|>";
            for (int i = 0; i < n_img_tokens; i++) s += pad;
            s += "<|vision_end|>";
        }
    }
    s += user_text;
    s += "<|im_end|>\n<|im_start|>assistant\n";
    return s;
}

int VideoChat3::sample(const std::vector<float> & logits, const vqa::GenParams & gp) {
    return vqa::sample_logits(logits, gp, rng_state);
}

bool VideoChat3::ask(const vqa::MediaInput & media, const std::string & prompt,
                     const std::string & /*mode*/, const vqa::GenParams & gp, std::string & out_text) {
    std::vector<float> feats;
    int n_img = 0;
    const auto tv0 = std::chrono::steady_clock::now();
    std::vector<std::pair<float,int>> clip_marks;
    if (!encode_vision(media, feats, &n_img, &clip_marks)) return false;
    fprintf(stderr, "[perf] vision encode: %.0f ms (%d tokens)\n",
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tv0).count(), n_img);

    const bool video = media.kind == vqa::MediaKind::Video;
    const std::string text = render_prompt(n_img, prompt, video, &clip_marks);
    const std::vector<int> ids = tok_.encode(text);
    const int pad_id = video ? id_video_pad : id_image_pad;
    std::vector<int> img_pos;
    for (int i = 0; i < (int)ids.size(); i++) if (ids[i] == pad_id) img_pos.push_back(i);
    if ((int)img_pos.size() != n_img) {
        fprintf(stderr, "placeholder mismatch: %d image_pad tokens vs %d vision tokens\n",
                (int)img_pos.size(), n_img);
        if (n_img > 0) return false;
    }
    fprintf(stderr, "prompt: %d tokens (%d image), max_new %d\n", (int)ids.size(), n_img, gp.max_tokens);

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
    for (int step = 0; step < gp.max_tokens; step++) {
        if (next == id_eos || next == id_endoftext) break;
        const std::string piece = tok_.decode({next});
        out_text += piece;
        if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
        n_dec++;
        if (step + 1 >= gp.max_tokens) break;
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

#include "vc3_batch.inc"

}  // namespace vc3

namespace vqa {
Model * create_videochat3() { return new vc3::VideoChat3(); }
}  // namespace vqa
