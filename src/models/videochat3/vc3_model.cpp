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
    bool encode_vision(const vqa::MediaInput & media, std::vector<float> & feats, int * n_tokens);
    bool build_text_graph(TextGraph & g, int T, int S, int n_img);
    bool run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<float> * img_feats,
                  const std::vector<int> * img_pos, std::vector<float> & logits_out);
    std::string render_prompt(int n_img_tokens, const std::string & user_text) const;
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

    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_context * kv_ctx = nullptr;
    std::vector<ggml_tensor *> kv_k, kv_v;
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

// per-pair complex rotation on an interleaved head vector: pair p = (v[2p], v[2p+1])
static ggml_tensor * rope_2d(ggml_context * ctx, ggml_tensor * x,
                             ggml_tensor * cos_t, ggml_tensor * sin_t) {
    const int64_t hd = x->ne[0], H = x->ne[1], N = x->ne[2];
    const int64_t pairs = hd / 2;
    ggml_tensor * x4 = ggml_reshape_4d(ctx, x, 2, pairs, H, N);
    ggml_tensor * re = ggml_view_4d(ctx, x4, 1, pairs, H, N, x4->nb[1], x4->nb[2], x4->nb[3], 0);
    ggml_tensor * im = ggml_view_4d(ctx, x4, 1, pairs, H, N, x4->nb[1], x4->nb[2], x4->nb[3], x4->nb[0]);
    ggml_tensor * c = ggml_reshape_4d(ctx, cos_t, 1, pairs, 1, N);
    ggml_tensor * s = ggml_reshape_4d(ctx, sin_t, 1, pairs, 1, N);
    ggml_tensor * ore = ggml_sub(ctx, ggml_mul(ctx, re, c), ggml_mul(ctx, im, s));
    ggml_tensor * oim = ggml_add(ctx, ggml_mul(ctx, re, s), ggml_mul(ctx, im, c));
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
    if (kv_buf) ggml_backend_buffer_free(kv_buf);
    if (kv_ctx) ggml_free(kv_ctx);
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
    const size_t kv_bytes = (size_t)n_ctx * (size_t)n_kv_head * head_dim * sizeof(ggml_fp16_t);
    struct ggml_init_params ip = { (size_t)(n_layer * 2 + 8) * ggml_tensor_overhead(), NULL, true };
    kv_ctx = ggml_init(ip);
    if (!kv_ctx) return false;
    kv_k.resize(n_layer);
    kv_v.resize(n_layer);
    for (int i = 0; i < n_layer; i++) {
        kv_k[i] = ggml_new_tensor_2d(kv_ctx, GGML_TYPE_F16, n_kv_head * head_dim, n_ctx);
        kv_v[i] = ggml_new_tensor_2d(kv_ctx, GGML_TYPE_F16, n_kv_head * head_dim, n_ctx);
    }
    kv_buf = ggml_backend_alloc_ctx_tensors(kv_ctx, backend);
    if (!kv_buf) {
        fprintf(stderr, "KV cache allocation failed (%.2f GB)\n", (double)n_layer * 2 * kv_bytes / 1e9);
        return false;
    }
    ggml_backend_buffer_clear(kv_buf, 0);
    fprintf(stderr, "kv cache: %.2f GB (n_ctx %d)\n", (double)n_layer * 2 * kv_bytes / 1e9, n_ctx);
    return true;
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
bool VideoChat3::encode_vision(const vqa::MediaInput & media, std::vector<float> & feats, int * n_tokens) {
    *n_tokens = 0;
    if (media.empty()) return true;
    if (media.kind == vqa::MediaKind::Video) {
        fprintf(stderr, "video input is not implemented yet\n");
        return false;
    }
    const vqa::Frame & fr = media.frames[0];
    const int src_h = media.height, src_w = media.width;
    const int factor = vc.patch * vc.merge;

    int H = 0, W = 0;
    // qwen_vl_utils defaults used by the official demo: min 56*56, max 16384*28*28
    if (!smart_resize(src_h, src_w, factor, 3136, 16384 * 28 * 28, &H, &W)) {
        fprintf(stderr, "image too small (%dx%d, factor %d)\n", src_w, src_h, factor);
        return false;
    }
    fprintf(stderr, "image %dx%d -> %dx%d\n", src_w, src_h, W, H);

    std::vector<float> rgb;
    resize_bicubic_antialias(fr.rgb.data(), src_w, src_h, W, H, rgb);

    std::vector<float> patches;
    int gh = 0, gw = 0;
    patchify(rgb.data(), H, W, vc, patches, &gh, &gw);
    const int N = gh * gw;

    std::vector<float> pos;
    pos_embed_for_grid(pos_emb_host.data(), pos_time_host.data(), vc, 1, gh, gw, pos);

    const int vhd = vc.n_embd / vc.n_head;
    std::vector<float> rcos, rsin;
    rope_2d_tables(vhd, 1, gh, gw, vc.rope_theta, rcos, rsin);

    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) return false;
    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);

    ggml_tensor * t_patches = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 3 * vc.patch * vc.patch, N);
    ggml_set_name(t_patches, "patches");
    ggml_tensor * t_pos = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, vc.n_embd, N);
    ggml_set_name(t_pos, "vpos");
    ggml_tensor * t_cos = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, vhd / 2, N);
    ggml_set_name(t_cos, "vcos");
    ggml_tensor * t_sin = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, vhd / 2, N);
    ggml_set_name(t_sin, "vsin");
    ggml_tensor * t_zmask = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, N, N);
    ggml_set_name(t_zmask, "vzmask");

    ggml_tensor * x = ggml_mul_mat(gctx, v_patch_w, t_patches);
    x = ggml_add(gctx, x, v_patch_b);
    x = ggml_add(gctx, x, t_pos);
    ggml_set_name(x, "v_patch_out");

    for (int il = 0; il < vc.n_layer; il++) {
        VisLayer & L = vl[il];
        ggml_tensor * l = ggml_norm(gctx, x, vc.layernorm_eps);
        l = ggml_mul(gctx, l, L.norm0_w);
        l = ggml_add(gctx, l, L.norm0_b);
        ggml_tensor * qkv = ggml_mul_mat(gctx, L.wqkv_w, l);
        qkv = ggml_add(gctx, qkv, L.wqkv_b);

        const size_t off = (size_t)vc.n_embd * qkv->nb[0];
        // the three projections are packed along ne0 of qkv, so each slice is a
        // strided view: make it contiguous before reshaping into heads
        ggml_tensor * q = ggml_reshape_3d(gctx,
                ggml_cont(gctx, ggml_view_2d(gctx, qkv, vc.n_embd, N, qkv->nb[1], 0)), vhd, vc.n_head, N);
        ggml_tensor * k = ggml_reshape_3d(gctx,
                ggml_cont(gctx, ggml_view_2d(gctx, qkv, vc.n_embd, N, qkv->nb[1], off)), vhd, vc.n_head, N);
        ggml_tensor * v = ggml_reshape_3d(gctx,
                ggml_cont(gctx, ggml_view_2d(gctx, qkv, vc.n_embd, N, qkv->nb[1], 2 * off)), vhd, vc.n_head, N);
        if (il == 0) ggml_set_name(qkv, "v_blk0_qkv");

        q = rope_2d(gctx, q, t_cos, t_sin);
        k = rope_2d(gctx, k, t_cos, t_sin);
        if (il == 0) { ggml_set_name(q, "v_blk0_q_rope"); ggml_set_name(k, "v_blk0_k_rope"); }

        ggml_tensor * qf = ggml_view_3d(gctx, q, vhd, N, vc.n_head, q->nb[2], q->nb[1], 0);
        ggml_tensor * kf = ggml_view_3d(gctx, k, vhd, N, vc.n_head, k->nb[2], k->nb[1], 0);
        ggml_tensor * vf = ggml_view_3d(gctx, v, vhd, N, vc.n_head, v->nb[2], v->nb[1], 0);
        ggml_tensor * fa = ggml_flash_attn_ext(gctx, qf, kf, vf, t_zmask,
                                               1.0f / sqrtf((float)vhd), 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(gctx, ggml_cont(gctx, fa), vc.n_embd, N);
        attn = ggml_mul_mat(gctx, L.wo_w, attn);
        attn = ggml_add(gctx, attn, L.wo_b);
        x = ggml_add(gctx, x, attn);
        if (il == 0) ggml_set_name(x, "v_blk0_attn_out");

        ggml_tensor * m = ggml_norm(gctx, x, vc.layernorm_eps);
        m = ggml_mul(gctx, m, L.norm1_w);
        m = ggml_add(gctx, m, L.norm1_b);
        m = ggml_mul_mat(gctx, L.fc0_w, m);
        m = ggml_add(gctx, m, L.fc0_b);
        m = ggml_gelu(gctx, m);                       // gelu_pytorch_tanh
        m = ggml_mul_mat(gctx, L.fc1_w, m);
        m = ggml_add(gctx, m, L.fc1_b);
        x = ggml_add(gctx, x, m);
        if (il == 0) ggml_set_name(x, "v_blk0_out");
    }
    x = ggml_norm(gctx, x, vc.layernorm_eps);
    x = ggml_mul(gctx, x, v_final_w);
    x = ggml_add(gctx, x, v_final_b);
    ggml_set_name(x, "v_final");

    // merge: the reference reads the sequence as (nh, kh, nw, kw) and folds (kh,kw)
    // into the feature dim; the processor emits (nh, nw, kh, kw) - reproduced verbatim
    const int nh = gh / vc.merge, nw = gw / vc.merge;
    const int kh = vc.merge, kw = vc.merge;
    {
        ggml_tensor * x4 = ggml_reshape_4d(gctx, x, (int64_t)vc.n_embd * kw, nw, kh, nh);
        x4 = ggml_cont(gctx, ggml_permute(gctx, x4, 0, 2, 1, 3));
        ggml_tensor * x3 = ggml_reshape_3d(gctx, x4, vc.n_embd, kw * kh, nh * nw);
        ggml_set_name(x3, "v_merge");
        ggml_tensor * pn = ggml_norm(gctx, x3, 1e-5f);
        pn = ggml_mul(gctx, pn, mm_pre_w);
        pn = ggml_add(gctx, pn, mm_pre_b);
        ggml_tensor * x2 = ggml_reshape_2d(gctx, pn, vc.n_embd * kw * kh, nh * nw);
        ggml_tensor * h = ggml_mul_mat(gctx, mm_l1_w, x2);
        h = ggml_add(gctx, h, mm_l1_b);
        h = ggml_gelu_erf(gctx, h);                   // nn.GELU() == exact erf
        h = ggml_mul_mat(gctx, mm_l2_w, h);
        h = ggml_add(gctx, h, mm_l2_b);
        ggml_set_name(h, "v_proj_out");
        ggml_build_forward_expand(gf, h);
    }

    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "vision graph allocation failed\n");
        ggml_free(gctx);
        return false;
    }
    ggml_backend_tensor_set(t_patches, patches.data(), 0, sizeof(float) * patches.size());
    ggml_backend_tensor_set(t_pos, pos.data(), 0, sizeof(float) * pos.size());
    ggml_backend_tensor_set(t_cos, rcos.data(), 0, sizeof(float) * rcos.size());
    ggml_backend_tensor_set(t_sin, rsin.data(), 0, sizeof(float) * rsin.size());
    {
        std::vector<ggml_fp16_t> zeros((size_t)N * N, ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(t_zmask, zeros.data(), 0, sizeof(ggml_fp16_t) * zeros.size());
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vision compute failed\n");
        ggml_free(gctx);
        return false;
    }

    dump(gf, "patches");
    dump(gf, "vpos");
    dump(gf, "vcos");
    dump(gf, "vsin");
    dump(gf, "v_patch_out");
    dump(gf, "v_blk0_out");
    dump(gf, "v_final");
    dump(gf, "v_merge");
    dump(gf, "v_proj_out");

    ggml_tensor * out = ggml_graph_get_tensor(gf, "v_proj_out");
    const int n = (int)out->ne[1];
    feats.resize((size_t)n * n_embd);
    ggml_backend_tensor_get(out, feats.data(), 0, sizeof(float) * feats.size());
    *n_tokens = n;
    fprintf(stderr, "vision: %d patches (%dx%d) -> %d tokens\n", N, gh, gw, n);
    ggml_free(gctx);
    return true;
}

// ---------------------------------------------------------------- text
bool VideoChat3::build_text_graph(TextGraph & g, int T, int S, int n_img) {
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
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv_k[il], k2, g.pos_in));
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv_v[il], v2, g.pos_in));

        // q3 is [HD, NH, T]; flash attention wants [HD, n_q, n_head], so swap the
        // two trailing strides. The KV cache is [NKV*HD, n_ctx]: one position is
        // NKV*HD halves and one head inside it is HD halves.
        ggml_tensor * qf = ggml_view_3d(g.ctx, q3, HD, T, NH, q3->nb[2], q3->nb[1], 0);
        ggml_tensor * kf = ggml_view_3d(g.ctx, kv_k[il], HD, S, NKV,
                                        kv_k[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * vf = ggml_view_3d(g.ctx, kv_v[il], HD, S, NKV,
                                        kv_v[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
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

    ggml_tensor * last = ggml_view_2d(g.ctx, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
    g.logits = ggml_mul_mat(g.ctx, tok_embd, last);
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
    {
        std::vector<ggml_fp16_t> mv((size_t)S * T, ggml_fp32_to_fp16(-INFINITY));
        for (int t = 0; t < T; t++)
            for (int s = 0; s < S; s++)
                if (s <= n_past + t) mv[(size_t)t * S + s] = ggml_fp32_to_fp16(0.0f);
        ggml_backend_tensor_set(g.mask_in, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());
    }

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

std::string VideoChat3::render_prompt(int n_img_tokens, const std::string & user_text) const {
    std::string s = "<|im_start|>user\n";
    if (n_img_tokens > 0) {
        s += "<|vision_start|>";
        for (int i = 0; i < n_img_tokens; i++) s += "<|image_pad|>";
        s += "<|vision_end|>";
    }
    s += user_text;
    s += "<|im_end|>\n<|im_start|>assistant\n";
    return s;
}

int VideoChat3::sample(const std::vector<float> & logits, const vqa::GenParams & gp) {
    if (gp.temperature <= 0.0f && gp.top_k <= 0) {
        int best = 0;
        float bv = -INFINITY;
        for (int i = 0; i < (int)logits.size(); i++)
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
    std::vector<std::pair<float, int>> cand(logits.size());
    for (int i = 0; i < (int)logits.size(); i++) cand[i] = {logits[i], i};
    std::sort(cand.begin(), cand.end(),
              [](const std::pair<float, int> & a, const std::pair<float, int> & b) { return a.first > b.first; });
    int keep = (int)cand.size();
    if (gp.top_k > 0) keep = std::min(keep, gp.top_k);
    const float inv_t = 1.0f / std::max(gp.temperature, 1e-4f);
    std::vector<float> p(keep);
    const float maxl = cand[0].first;
    float sum = 0.0f;
    for (int i = 0; i < keep; i++) { p[i] = expf((cand[i].first - maxl) * inv_t); sum += p[i]; }
    if (sum <= 0.0f) return cand[0].second;
    for (int i = 0; i < keep; i++) p[i] /= sum;
    if (gp.top_p < 1.0f) {
        float acc = 0.0f;
        int n = 0;
        while (n < keep && acc < gp.top_p) { acc += p[n]; n++; }
        keep = std::max(1, n);
    }
    const float r = randf();
    float acc = 0.0f;
    for (int i = 0; i < keep; i++) {
        acc += p[i];
        if (r <= acc) return cand[i].second;
    }
    return cand[keep - 1].second;
}

bool VideoChat3::ask(const vqa::MediaInput & media, const std::string & prompt,
                     const std::string & /*mode*/, const vqa::GenParams & gp, std::string & out_text) {
    std::vector<float> feats;
    int n_img = 0;
    if (!encode_vision(media, feats, &n_img)) return false;

    const std::string text = render_prompt(n_img, prompt);
    const std::vector<int> ids = tok_.encode(text);
    std::vector<int> img_pos;
    for (int i = 0; i < (int)ids.size(); i++) if (ids[i] == id_image_pad) img_pos.push_back(i);
    if ((int)img_pos.size() != n_img) {
        fprintf(stderr, "placeholder mismatch: %d image_pad tokens vs %d vision tokens\n",
                (int)img_pos.size(), n_img);
        if (n_img > 0) return false;
    }
    fprintf(stderr, "prompt: %d tokens (%d image), max_new %d\n", (int)ids.size(), n_img, gp.max_tokens);

    n_past = 0;
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

}  // namespace vc3

namespace vqa {
Model * create_videochat3() { return new vc3::VideoChat3(); }
}  // namespace vqa
