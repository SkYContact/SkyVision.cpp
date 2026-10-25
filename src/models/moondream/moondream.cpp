// SkyVision.cpp — standalone ggml inference engine for moondream3.1-9B-A2B
// Milestone 1: text-only, CPU, greedy decode, single stream, batch=1 steps.
//
// Model math (from moondream/moondream3-preview official reference text.py/layers.py/rope.py):
//   x = embd(tok)
//   per layer (parallel residual, LayerNorm with bias):
//     l   = LN(x)
//     qkv = Wqkv·l + b                    (6144 = 32h·64 q + 2·32h·64 kv, no GQA)
//     tau: tq = tanh(tau_wq·gelu(qkv))    ([32, T] per-head scalar)
//          tv = tanh(tau_wv·gelu(qkv))
//          ps  = 1 + (sigmoid(alpha·log(pos+1)) - 0.5)
//          q *= tq + ps ;  v *= tv + ps
//     rope: half-rotation on first 32 of 64 dims, theta 1.5e6 (NeoX style)
//     attn: causal softmax, scale 1/sqrt(64); out = Wout·attn + b
//     mlp:  dense 0..3 : fc1(2048->8192)+b, gelu, fc2+b
//           moe  4..23 : router(2048->64)+b, top8 by logit, softmax over the top8,
//                        fused fc1 [E, 2I, H]: h = rows[0:I], g = rows[I:2I],
//                        act = gelu(h)·(g+1), out = fc2·act, weighted sum over k
//   x = x + attn_out + mlp_out
//   logits = lm_head(LN(x)) + b

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#   include "ggml-cuda.h"
#endif
#include "ggml-alloc.h"

#include "core/engine/attention.hpp"
#include "core/engine/sampler.hpp"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <utility>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>

// stb_image is declared here and implemented once in src/core/media_io.cpp
#include "stb_image.h"
#include "moondream_vision.hpp"
#include "moondream_preprocess.hpp"
#include "moondream_region.hpp"

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   define NOMINMAX
#   include <windows.h>
#   include <shellapi.h>
#else
#   include <sys/mman.h>
#   include <fcntl.h>
#   include <unistd.h>
#endif

// ---------------------------------------------------------------- shared core
// The mmap, the byte-level BPE tokenizer and the utf8/json helpers live in
// src/core so that every backend shares one implementation. vqa::Tokenizer
// defaults to the plain GPT-2 mode this vocabulary expects (no Qwen-style
// regex pre-tokenizer, no <|...|> specials).
#include "core/text/tokenizer.hpp"
#include "core/util/mmap.hpp"
#include "core/text/utf8.hpp"
#include "core/engine/backend.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/weight_load.hpp"
using vqa::MMap;
using vqa::Tokenizer;
using vqa::init_byte_table;


// ---------------------------------------------------------------- model
struct Layer {
    struct ggml_tensor * attn_norm_w = nullptr, * attn_norm_b = nullptr;
    struct ggml_tensor * qkv_w = nullptr, * qkv_b = nullptr;
    struct ggml_tensor * attn_out_w = nullptr, * attn_out_b = nullptr;
    struct ggml_tensor * tau_wq = nullptr, * tau_wv = nullptr, * tau_alpha = nullptr;
    struct ggml_tensor * ffn_up_w = nullptr, * ffn_up_b = nullptr;
    struct ggml_tensor * ffn_down_w = nullptr, * ffn_down_b = nullptr;
    struct ggml_tensor * router_w = nullptr, * router_b = nullptr;
    struct ggml_tensor * up_exps = nullptr, * down_exps = nullptr;
};

struct Model {
    int n_layer = 24, n_head = 32, n_kv_head = 32, n_embd = 2048, n_vocab = 51200;
    int n_head_dim = 64, n_rot = 32, n_experts = 64, n_experts_used = 8;
    int inter = 1024, ff = 8192, moe_start = 4;
    float rope_theta = 1500000.0f;

    struct ggml_context * wctx = nullptr;
    struct ggml_backend_buffer * wbuf = nullptr;
    struct ggml_context * cctx = nullptr;
    void * f32_cpu = nullptr;
    MMap mm;

    struct ggml_tensor * tok_embd = nullptr;
    struct ggml_tensor * out_norm_w = nullptr, * out_norm_b = nullptr;
    struct ggml_tensor * out_w = nullptr, * out_b = nullptr;
    std::vector<Layer> layers;
    std::vector<float> tau_alpha_host;   // host copy of all tau_alpha for per-step ps precompute

    bool load(const char * path, ggml_backend_t backend);
};

bool Model::load(const char * path, ggml_backend_t backend) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f GB\n", mm.size / 1e9);

    struct gguf_context * gu = nullptr;
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    auto get_u32 = [&](const char * k, int & v) {
        int64_t id = gguf_find_key(gu, k);
        if (id >= 0) v = (int)gguf_get_val_u32(gu, id);
    };
    get_u32("moondream3.block_count", n_layer);
    get_u32("moondream3.embedding_length", n_embd);
    get_u32("moondream3.attention.head_count", n_head);
    get_u32("moondream3.attention.head_count_kv", n_kv_head);
    get_u32("moondream3.vocab_size", n_vocab);
    get_u32("moondream3.expert_count", n_experts);
    get_u32("moondream3.expert_used_count", n_experts_used);
    get_u32("moondream3.moe_start_layer", moe_start);
    {
        int64_t id = gguf_find_key(gu, "moondream3.rope.freq_base");
        if (id >= 0) rope_theta = gguf_get_val_f32(gu, id);
    }
    n_head_dim = n_embd / n_head;
    n_rot = n_head_dim / 2;

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);

    // small F16 tensors (norms, biases) -> CPU-side F32 copies (mixed-type binary ops unsupported)
    struct ggml_backend_buffer * f32buf = nullptr;
    void * f32data = nullptr;
    size_t f32_total = 0;
    {
        struct ggml_init_params ip2 = { ggml_tensor_overhead() * 4096, NULL, true };
        cctx = ggml_init(ip2);
        size_t total = 0;
        for (auto t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            if (t->type == GGML_TYPE_F32 || ggml_nbytes(t) > (8u << 20)) continue;
            total += ggml_nelements(t) * sizeof(float);
        }
        f32data = malloc(total);
        f32_total = total;
        size_t off = 0;
        for (auto t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            if (t->type == GGML_TYPE_F32 || ggml_nbytes(t) > (8u << 20)) continue;
            int64_t n = ggml_nelements(t);
            const uint16_t * src = (const uint16_t *)t->data;
            float * dst = (float *)((char *)f32data + off);
            for (int64_t j = 0; j < n; j++) dst[j] = ggml_fp16_to_fp32(src[j]);
            t->data = dst;
            t->type = GGML_TYPE_F32;
            t->nb[0] = sizeof(float);
            t->nb[1] = t->ne[0] * sizeof(float);
            t->nb[2] = t->nb[1] * t->ne[1];
            t->nb[3] = t->nb[2] * t->ne[2];
            off += n * sizeof(float);
        }
        f32_cpu = f32data;   // kept for CPU zero-copy path; freed after upload on GPU path
        // host-side alpha snapshot for the per-step tau ps precompute
        tau_alpha_host.resize((size_t)n_layer * n_head);
        {
            int i = 0;
            for (auto t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
                if (strncmp(t->name, "blk.", 4) == 0 && strstr(t->name, ".tau_alpha.") != nullptr) {
                    memcpy(&tau_alpha_host[(size_t)i * n_head], t->data, sizeof(float) * n_head);
                    i++;
                }
            }
        }
        fprintf(stderr, "f32-copied small tensors: %.1f MB\n", total / 1e6);
    }

    if (ggml_backend_is_cpu(backend)) {
        // zero-copy: keep pointing weights into the mmap / f32 host memory (avoids a second
        // 10GB commit on Windows, where mmap already counts against the commit limit)
        wbuf = ggml_backend_cpu_buffer_from_ptr(mm.ptr, mm.size);
        f32buf = f32_total ? ggml_backend_cpu_buffer_from_ptr(f32data, f32_total) : nullptr;
        for (auto t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t))
            t->buffer = (t->data >= f32data && t->data < (char *)f32data + f32_total) ? f32buf : wbuf;
    } else {
        // record host-side sources, then allocate every weight on the backend and upload
        std::vector<std::pair<struct ggml_tensor *, const void *>> uploads;
        size_t total = 0;
        for (auto t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            uploads.push_back({t, t->data});
            total += ggml_nbytes(t);
        }
        for (auto & [t, src] : uploads) t->data = nullptr;   // backend alloc only assigns data==NULL tensors
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
        if (!wbuf) {
            fprintf(stderr, "weight alloc on backend failed (need %.2f GB, errno %d)\n", total / 1e9, errno);
            return false;
        }
        for (auto & [t, src] : uploads)
            ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
        fprintf(stderr, "uploaded %.2f GB weights to backend\n", total / 1e9);
        // host-side copies no longer needed; on Windows the mmap eats commit quota
        mm.release();
        free(f32_cpu); f32_cpu = nullptr;
    }

    layers.resize(n_layer);
    tok_embd   = ggml_get_tensor(wctx, "token_embd.weight");
    out_norm_w = ggml_get_tensor(wctx, "output_norm.weight");
    out_norm_b = ggml_get_tensor(wctx, "output_norm.bias");
    out_w      = ggml_get_tensor(wctx, "output.weight");
    out_b      = ggml_get_tensor(wctx, "output.bias");
    if (!tok_embd || !out_w) { fprintf(stderr, "missing core tensors\n"); return false; }

    std::vector<char> buf(256);
    auto gt = [&](const char * fmt, int i, const char * suffix) {
        snprintf(buf.data(), buf.size(), fmt, i, suffix);
        return ggml_get_tensor(wctx, buf.data());
    };
    for (int i = 0; i < n_layer; i++) {
        Layer & L = layers[i];
        L.attn_norm_w = gt("blk.%d.attn_norm.%s", i, "weight");
        L.attn_norm_b = gt("blk.%d.attn_norm.%s", i, "bias");
        L.qkv_w = gt("blk.%d.attn_qkv.%s", i, "weight");
        L.qkv_b = gt("blk.%d.attn_qkv.%s", i, "bias");
        L.attn_out_w = gt("blk.%d.attn_out.%s", i, "weight");
        L.attn_out_b = gt("blk.%d.attn_out.%s", i, "bias");
        L.tau_wq = gt("blk.%d.tau_wq.%s", i, "weight");
        L.tau_wv = gt("blk.%d.tau_wv.%s", i, "weight");
        L.tau_alpha = gt("blk.%d.tau_alpha.%s", i, "weight");
        if (i < moe_start) {
            L.ffn_up_w = gt("blk.%d.ffn_up.%s", i, "weight");
            L.ffn_up_b = gt("blk.%d.ffn_up.%s", i, "bias");
            L.ffn_down_w = gt("blk.%d.ffn_down.%s", i, "weight");
            L.ffn_down_b = gt("blk.%d.ffn_down.%s", i, "bias");
        } else {
            L.router_w = gt("blk.%d.ffn_gate_inp.%s", i, "weight");
            L.router_b = gt("blk.%d.ffn_gate_inp.%s", i, "bias");
            L.up_exps = gt("blk.%d.ffn_up_exps.%s", i, "weight");
            L.down_exps = gt("blk.%d.ffn_down_exps.%s", i, "weight");
        }
    }
    fprintf(stderr, "model loaded: %d layers, %d embd, %d vocab, experts %d/%d from layer %d\n",
            n_layer, n_embd, n_vocab, n_experts, n_experts_used, moe_start);
    return true;
}

// ---------------------------------------------------------------- engine
struct Engine {
    Model model;
    Tokenizer tok;
    int n_past = 0;
    std::vector<int *> leaked_ids;   // per-step host id buffers (freed never; few KB)
    int n_ctx = 1024;
    vqa::KvCache kv;
    ggml_backend_t backend = nullptr;
    struct ggml_gallocr * gallocr = nullptr;
    // static decode graph: shape-invariant (KV views cover full n_ctx, mask hides unused
    // slots), so the same gctx/gf are reused every token — required for ggml CUDA graph
    // capture (graph key = nodes[0] pointer, must stay stable across steps)
    struct ggml_context * dec_ctx = nullptr;
    struct ggml_cgraph * dec_gf = nullptr;
    struct ggml_tensor * dec_tok_in = nullptr, * dec_pos_in = nullptr, * dec_mask_in = nullptr;
    // static batch decode graph (T = bat_n fixed, logits for all columns) — see
    // src/moondream_batch.inc
    struct ggml_context * bat_ctx = nullptr;
    struct ggml_cgraph * bat_gf = nullptr;
    struct ggml_tensor * bat_tok_in = nullptr, * bat_pos_in = nullptr, * bat_mask_in = nullptr;
    int bat_n = 0;
    // start a fresh conversation: clears the KV cache and position counter (the
    // static decode/batch graphs keep working — only their input values depend on it)
    void reset() {
        n_past = 0;
        kv.clear();
    }
    int step_batch(const std::vector<int> & toks, const std::vector<int> & posv,
                   const std::vector<int> & lo, const std::vector<int> & hi, float * logits_out);

    ~Engine() {
        if (dec_ctx) ggml_free(dec_ctx);
        if (bat_ctx) ggml_free(bat_ctx);
        kv.free();
        if (gallocr) ggml_gallocr_free(gallocr);
        if (backend) ggml_backend_free(backend);
    }

    bool init(const char * gguf_path, const char * tok_json, int n_ctx);
    struct ggml_cgraph * build_graph(struct ggml_context * gctx, const int * toks, int n_toks,
                                     int * * host_bufs, bool fixed_ctx = false,
                                     bool emb_input = false, bool logits_all = false);   // host mem for inputs
    int step(const std::vector<int> & toks, float * last_logits_out,
             const float * emb_prefix = nullptr, int n_prefix = 0,
             float * hidden_out = nullptr, bool emb_bidir = false);
};

bool Engine::init(const char * gguf_path, const char * tok_json, int n_ctx_) {
#ifdef GGML_USE_CUDA
    if (!getenv("MD_CPU")) {
        backend = ggml_backend_cuda_init(0);
        if (backend) {
            size_t free_mm = 0, total_mm = 0;
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_dev_memory(dev, &free_mm, &total_mm);
            fprintf(stderr, "backend: CUDA (%s), free %.2f GB / %.2f GB\n",
                    ggml_backend_name(backend), free_mm / 1e9, total_mm / 1e9);
        }
    }
#endif
    if (!backend) {
        backend = ggml_backend_cpu_init();
        int nth = vqa::default_n_threads();                                // physical cores; SMT hurts GEMV
        if (const char * e = getenv("MD_THREADS")) nth = atoi(e);          // ggml CPU default is only 4
        ggml_backend_cpu_set_n_threads(backend, nth);
        fprintf(stderr, "backend: %s (%d threads)\n", ggml_backend_name(backend), nth);
    }
    if (!model.load(gguf_path, backend)) return false;
    if (!tok.load(tok_json)) return false;
    n_ctx = n_ctx_;

    if (!kv.alloc(backend, model.n_layer, model.n_kv_head, model.n_head_dim, n_ctx)) return false;
    // unused cache slots must be zero: flash-attn reads the full fixed-size window
    // and NaN garbage would poison softmax even under -inf mask (kv.alloc clears)

    // Pre-grow the gallocr pool to the largest graph shape we will ever run
    // (an embedding prefill of the full context). ggml_gallocr FREEs the whole
    // pool whenever a new graph needs a larger chunk — any previously bound
    // graph (e.g. the static decode graph) is left with dangling tensor->data
    // pointers and the next tensor_set crashes with cudaMemcpy invalid-argument.
    // Sizing the pool once up front means no later graph can trigger a realloc.
    {
        struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, /*no_alloc*/ true };
        struct ggml_context * gctx = ggml_init(ip);
        int * hb[2] = { nullptr, nullptr };
        struct ggml_cgraph * gf = build_graph(gctx, nullptr, n_ctx, hb,
                                              /*fixed_ctx*/ false, /*emb_input*/ true);
        if (!gallocr) gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(gallocr, gf))
            fprintf(stderr, "warning: gallocr pre-grow alloc failed\n");
        ggml_free(gctx);
        if (getenv("MD_DUMP")) vqa::dump::dump_registry_clear();   // this graph is thrown away
    }
    return true;
}

#define MMUL(ctx, a, b) ([&]() {     if ((a)->ne[0] != (b)->ne[0] || (a)->ne[2] != (b)->ne[2] || (a)->ne[3] != (b)->ne[3])         fprintf(stderr, "BAD mul_mat: a ne[%lld,%lld,%lld,%lld] t%d  b ne[%lld,%lld,%lld,%lld] t%d\n",             (long long)(a)->ne[0], (long long)(a)->ne[1], (long long)(a)->ne[2], (long long)(a)->ne[3], (int)(a)->type,             (long long)(b)->ne[0], (long long)(b)->ne[1], (long long)(b)->ne[2], (long long)(b)->ne[3], (int)(b)->type);     return ggml_mul_mat((ctx), (a), (b)); }())

struct ggml_cgraph * Engine::build_graph(struct ggml_context * gctx, const int * toks, int n_toks,
                                         int * * host_bufs, bool fixed_ctx, bool emb_input, bool logits_all) {
    static const bool verbose = getenv("MD_TRACE") != nullptr;
    Model & M = model;
    const int D = M.n_embd, H = M.n_head, HD = M.n_head_dim, KV = M.n_kv_head * M.n_head_dim;
    const int T = n_toks, P = n_past;

    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);

    // graph inputs: data is bound in step() via ggml_backend_tensor_set after gallocr alloc
    struct ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_set_name(pos, "pos_in");
    (void)toks;

    struct ggml_tensor * inp = nullptr;
    if (emb_input) {
        // vision prefix: raw embeddings [D, T] fed directly (bos + 729 image tokens)
        inp = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, D, T);
        ggml_set_name(inp, "emb_in");   // note: single name; later lookups rely on it
    } else {
        struct ggml_tensor * tok_in = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
        ggml_set_name(tok_in, "tok_in");
        inp = ggml_get_rows(gctx, M.tok_embd, tok_in);
        ggml_set_name(inp, "d_h_emb");
    }

    // one causal mask shared by ALL layers (built once; step() fills it via tensor_set —
    // per-layer copies would leave 23 of them reading uninitialized memory)
    const int S_att = fixed_ctx ? n_ctx : P + T;
    struct ggml_tensor * shared_mask = nullptr;
    if (!getenv("MD_NO_FA") && (fixed_ctx || T > 1)) {
        shared_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, S_att, T);
        ggml_set_name(shared_mask, "mask_in");
    }

    // shared constants (one graph node reused by every MoE layer)
    struct ggml_tensor * one_fill = ggml_fill(gctx, ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, 1, 1), 1.0f);

    int il_max = M.n_layer, il_min = 0;
    { const char * e = getenv("MD_MAX_LAYER"); if (e) il_max = atoi(e); }
    { const char * e = getenv("MD_MIN_LAYER"); if (e) il_min = atoi(e); }
    for (int il = il_min; il < il_max; il++) {
        Layer & L = M.layers[il];

        struct ggml_tensor * l = ggml_norm(gctx, inp, 1e-5f);
        l = ggml_mul(gctx, l, L.attn_norm_w);
        l = ggml_add(gctx, l, L.attn_norm_b);
        if (il == 0) ggml_set_name(l, "d_l0_ln");

        if (verbose) fprintf(stderr, "[b] L%d norm ok\n", il);
        struct ggml_tensor * qkv = MMUL(gctx, L.qkv_w, l);
        qkv = ggml_add(gctx, qkv, L.qkv_b);                       // [6144, T]
        if (il == 0) ggml_set_name(qkv, "d_l0_qkv");

        if (verbose) fprintf(stderr, "[b] L%d qkv ok\n", il);
        struct ggml_tensor * g_act = ggml_gelu_erf(gctx, qkv);   // ref text.py uses exact erf gelu
        struct ggml_tensor * tok_q = ggml_tanh(gctx, MMUL(gctx, L.tau_wq, g_act)); // [H, T]
        struct ggml_tensor * tok_v = ggml_tanh(gctx, MMUL(gctx, L.tau_wv, g_act));

        // ps = sigmoid(alpha*log(pos+1)) + 0.5 depends only on weights + positions: precomputed
        // on the host in step() and injected as a graph input (saves ~12 tiny nodes per layer)
        struct ggml_tensor * ps2 = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, H, T);
        ggml_format_name(ps2, "tau_ps_%d", il);

        // per-token layout: [q (32x64) | k (32x64) | v (32x64)]; cont() because rope needs formal contiguity
        struct ggml_tensor * q = ggml_cont(gctx, ggml_view_3d(gctx, qkv, HD, H, T,
                HD * sizeof(float), qkv->nb[1], 0));
        struct ggml_tensor * k = ggml_cont(gctx, ggml_view_3d(gctx, qkv, HD, M.n_kv_head, T,
                HD * sizeof(float), qkv->nb[1], H * HD * sizeof(float)));
        struct ggml_tensor * v = ggml_cont(gctx, ggml_view_3d(gctx, qkv, HD, M.n_kv_head, T,
                HD * sizeof(float), qkv->nb[1], (H + M.n_kv_head) * HD * sizeof(float)));

        if (verbose) fprintf(stderr, "[b] L%d tau ok\n", il);
        // tau attention scaling (q,v *= tanh(wq.gelu(qkv)) + ps) - official
        // moondream 3.1 architecture, non-optional
        struct ggml_tensor * sq = nullptr, * sv = nullptr;
        sq = ggml_reshape_3d(gctx, ggml_add(gctx, tok_q, ps2), 1, H, T);
        q = ggml_mul(gctx, q, sq);
        sv = ggml_reshape_3d(gctx, ggml_add(gctx, tok_v, ps2), 1, H, T);
        v = ggml_mul(gctx, v, sv);
        if (il == 0) { ggml_set_name(q, "d_l0_q_tau"); }

        q = ggml_rope_ext(gctx, q, pos, nullptr, M.n_rot, GGML_ROPE_TYPE_NEOX, n_ctx,
                          M.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(gctx, k, pos, nullptr, M.n_rot, GGML_ROPE_TYPE_NEOX, n_ctx,
                          M.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (il == 0) { ggml_set_name(q, "d_l0_q_rope"); ggml_set_name(k, "d_l0_k_rope"); }

        if (verbose) fprintf(stderr, "[b] L%d rope ok\n", il);
        // flatten heads: [HD*Hkv, T]
        struct ggml_tensor * k2 = ggml_reshape_2d(gctx, k, KV, T);
        struct ggml_tensor * v2 = ggml_reshape_2d(gctx, v, KV, T);
        if (il == 0) ggml_set_name(v2, "d_l0_v2");

        // write into KV cache rows P..P+T via scatter
        ggml_build_forward_expand(gf, ggml_set_rows(gctx, kv.k[il], k2, pos));
        ggml_build_forward_expand(gf, ggml_set_rows(gctx, kv.v[il], v2, pos));

        const int S = fixed_ctx ? n_ctx : P + T;   // static decode graph: fixed-size KV views
        static const bool no_fa = getenv("MD_NO_FA") != nullptr;
        struct ggml_tensor * attn = nullptr;
        if (!no_fa) {
            // fused flash attention: q [HD,T,H], k/v [HD,S,H] views straight into tensors
            // (views are row-contiguous, no cont() copies needed)
            struct ggml_tensor * qf = ggml_view_3d(gctx, q, HD, T, H, q->nb[2], q->nb[1], 0);
            struct ggml_tensor * kf = ggml_view_3d(gctx, kv.k[il], HD, S, H,
                    KV * sizeof(ggml_fp16_t), HD * sizeof(ggml_fp16_t), 0);
            struct ggml_tensor * vf = ggml_view_3d(gctx, kv.v[il], HD, S, H,
                    KV * sizeof(ggml_fp16_t), HD * sizeof(ggml_fp16_t), 0);
            // causal mask: shared tensor (may be NULL for single-shot decode without fixed ctx)
            struct ggml_tensor * mask = shared_mask;
            struct ggml_tensor * fa = ggml_flash_attn_ext(gctx, qf, kf, vf, mask,
                    1.0f / sqrtf((float)HD), 0.0f, 0.0f);          // result ne = [HD, H, T]
            if (il == 0) ggml_set_name(fa, "d_l0_attn_out3");
            attn = ggml_reshape_2d(gctx, fa, D, T);                // contiguous [HD,H,T] == [D,T]
        } else {
        // per-head attention with explicit 2D matmuls; results concatenated along D (h-major)
        for (int h = 0; h < H; h++) {
            struct ggml_tensor * qh  = ggml_view_2d(gctx, q, HD, T, q->nb[2], h * q->nb[1]);
            struct ggml_tensor * kh  = ggml_view_2d(gctx, kv.k[il], HD, S, KV * sizeof(ggml_fp16_t), h * HD * sizeof(ggml_fp16_t));
            struct ggml_tensor * wth = ggml_cont(gctx, ggml_transpose(gctx,
                    ggml_view_2d(gctx, kv.v[il], HD, S, KV * sizeof(ggml_fp16_t), h * HD * sizeof(ggml_fp16_t))));  // [S, HD]
            struct ggml_tensor * kq = MMUL(gctx, kh, qh);              // [S, T]
            kq = ggml_scale(gctx, kq, 1.0f / sqrtf((float)HD));
            if (P > 0 || T > 1) {
                kq = ggml_diag_mask_inf(gctx, kq, P);
            }
            if (il == 0 && h == 0) ggml_set_name(kq, "d_l0_scores");
            kq = ggml_soft_max(gctx, kq);
            if (il == 0 && h == 0) ggml_set_name(kq, "d_l0_attnw");
            struct ggml_tensor * oh = MMUL(gctx, wth, kq);             // [HD, T]
            attn = attn ? ggml_concat(gctx, attn, oh, 0) : oh;
        }
        }
        if (il == 0) ggml_set_name(attn, "d_l0_attn_out");
        attn = MMUL(gctx, L.attn_out_w, attn);
        attn = ggml_add(gctx, attn, L.attn_out_b);
        if (il == 0) ggml_set_name(attn, "d_l0_attn_proj");

        struct ggml_tensor * ffn = nullptr;
        if (verbose) fprintf(stderr, "[b] L%d attn ok\n", il);
        if (il < M.moe_start) {
            ffn = MMUL(gctx, L.ffn_up_w, l);
            ffn = ggml_add(gctx, ffn, L.ffn_up_b);
            ffn = ggml_gelu(gctx, ffn);
            ffn = MMUL(gctx, L.ffn_down_w, ffn);
            ffn = ggml_add(gctx, ffn, L.ffn_down_b);
            if (il == 0) ggml_set_name(ffn, "d_l0_mlp");
        } else {
            struct ggml_tensor * rlogits = ggml_add(gctx, MMUL(gctx, L.router_w, l), L.router_b); // [E, T]
            struct ggml_tensor * sel = ggml_top_k(gctx, rlogits, M.n_experts_used);   // [K, T] I32
            struct ggml_tensor * sel_logits = ggml_get_rows(gctx,
                    ggml_reshape_3d(gctx, rlogits, 1, M.n_experts, T), sel);           // [1, K, T]
            sel_logits = ggml_reshape_2d(gctx, sel_logits, M.n_experts_used, T);
            struct ggml_tensor * w = ggml_soft_max(gctx, sel_logits);                  // [K, T]
            w = ggml_reshape_3d(gctx, w, 1, M.n_experts_used, T);

            struct ggml_tensor * cur = ggml_reshape_3d(gctx, l, D, 1, T);
            struct ggml_tensor * gu = ggml_mul_mat_id(gctx, L.up_exps, cur, sel);      // [2I, K, T]
            struct ggml_tensor * h = ggml_view_3d(gctx, gu, M.inter, M.n_experts_used, T, gu->nb[1], gu->nb[2], 0);
            struct ggml_tensor * g = ggml_view_3d(gctx, gu, M.inter, M.n_experts_used, T, gu->nb[1], gu->nb[2], M.inter * gu->nb[0]);
            // h is a row-strided view of gu (first I of 2I rows per expert slice); the CUDA unary
            // kernel only accepts contiguous src0, so materialize it like the rope path above
            cur = ggml_gelu_erf(gctx, ggml_cont(gctx, h));   // ref layers.py MoE GeGLU uses F.gelu (erf)
            g = ggml_add(gctx, g, one_fill);           // g + 1
            cur = ggml_mul(gctx, cur, g);
            struct ggml_tensor * ff3 = ggml_mul_mat_id(gctx, L.down_exps, cur, sel);   // [D, K, T]
            // weighted K-sum: broadcast-mul router weights, permute K to ne0, one sum_rows
            // (2 compute kernels instead of 15 mul/add + 16 view nodes)
            struct ggml_tensor * wb = ggml_view_3d(gctx, w, 1, M.n_experts_used, T, w->nb[1], w->nb[2], 0);
            struct ggml_tensor * prod = ggml_mul(gctx, ff3, wb);                       // [D, K, T]
            struct ggml_tensor * pk = ggml_cont(gctx, ggml_permute(gctx, prod, 1, 0, 2, 3));  // [K, D, T]
            ffn = ggml_reshape_2d(gctx, ggml_sum_rows(gctx, pk), D, T);                // [1,D,T] -> [D,T]
        }

        if (verbose) fprintf(stderr, "[b] L%d ffn ok\n", il);
        inp = ggml_add(gctx, ggml_add(gctx, inp, attn), ffn);
        if (il == 0) ggml_set_name(inp, "d_h_L0");
    }

    struct ggml_tensor * x = inp;
    if (T > 1 && !logits_all) {   // batch graphs need logits for every column
        x = ggml_view_2d(gctx, x, D, 1, x->nb[1], (T - 1) * x->nb[1]);
    }
    {   // pre-LM-head residual stream, consumed by the region head (detect/point).
        // the view must be explicitly added to the graph — an unconsumed tensor
        // never enters it and graph lookups would silently return NULL
        struct ggml_tensor * hid = ggml_view_2d(gctx, x, D, 1, x->nb[1], 0);
        ggml_set_name(hid, "hidden");
        ggml_build_forward_expand(gf, hid);
    }
    x = ggml_norm(gctx, x, 1e-5f);
    x = ggml_mul(gctx, x, M.out_norm_w);
    x = ggml_add(gctx, x, M.out_norm_b);
    struct ggml_tensor * logits = MMUL(gctx, M.out_w, x);
    logits = ggml_add(gctx, logits, M.out_b);
    ggml_set_name(logits, "logits");

    ggml_build_forward_expand(gf, logits);
    if (getenv("MD_DUMP")) {
        static const char * const dump_names[] = {
            "d_h_emb", "d_l0_ln", "d_l0_qkv", "d_l0_q_tau", "d_l0_q_rope", "d_l0_k_rope",
            "d_l0_v2", "d_l0_scores", "d_l0_attnw", "d_l0_attn_out3", "d_l0_attn_out",
            "d_l0_attn_proj", "d_l0_mlp", "d_h_L0", "hidden", nullptr,
        };
        vqa::dump::mark_sinks(gctx, gf, dump_names);
    }
    // attach the keep nodes at the true tail (no-op when nothing is registered)
    vqa::dump::finalize_sinks(gctx, gf, backend);
    return gf;
}

int Engine::step(const std::vector<int> & toks, float * last_logits_out,
                 const float * emb_prefix, int n_prefix, float * hidden_out, bool emb_bidir) {
    static const bool prof = getenv("MD_PROF") != nullptr;
    static double t_build = 0, t_alloc = 0, t_input = 0, t_compute = 0;
    static int n_steps = 0;
    auto tick = []() { return std::chrono::steady_clock::now(); };
    n_steps++;
    const bool emb_mode = emb_prefix != nullptr;
    const int n_seq = emb_mode ? n_prefix : (int)toks.size();

    // static (shape-invariant) graph only pays off on CUDA: it enables CUDA graph capture;
    // on CPU the full-window flash attention costs more than the saved graph rebuild
    const bool use_static = (n_seq == 1) && !emb_mode && !ggml_backend_is_cpu(backend);
    struct ggml_context * gctx = nullptr;
    struct ggml_cgraph * gf = nullptr;

    auto t0 = tick();
    if (use_static) {
        if (!dec_ctx) {   // build once: fixed n_ctx KV views + always-present mask
            if (getenv("MD_DUMP")) vqa::dump::dump_registry_clear();
            struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, /*no_alloc*/ true };
            dec_ctx = ggml_init(ip);
            int * hb[2] = { nullptr, nullptr };
            dec_gf = build_graph(dec_ctx, toks.data(), 1, hb, /*fixed_ctx*/ true);
            if (!gallocr) gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(gallocr, dec_gf)) { fprintf(stderr, "decode graph alloc failed\n"); return -1; }
            dec_tok_in = ggml_graph_get_tensor(dec_gf, "tok_in");
            dec_pos_in = ggml_graph_get_tensor(dec_gf, "pos_in");
            dec_mask_in = ggml_graph_get_tensor(dec_gf, "mask_in");
        }
        gctx = dec_ctx; gf = dec_gf;
    } else {
        const size_t ctx_size = 512ull * 1024 * 1024;   // graph metadata only (no_alloc + gallocr)
        if (getenv("MD_DUMP")) vqa::dump::dump_registry_clear();   // per-call graph
        struct ggml_init_params ip = { ctx_size, NULL, /*no_alloc*/ true };
        gctx = ggml_init(ip);
        if (!gctx) return -1;
        int * host_bufs[2] = { nullptr, nullptr };
        gf = build_graph(gctx, toks.data(), n_seq, host_bufs, /*fixed_ctx*/ false, /*emb_input*/ emb_mode);
        if (!gallocr) gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(gallocr, gf)) { fprintf(stderr, "graph alloc failed\n"); ggml_free(gctx); return -1; }
    }
    auto t1 = tick();

    // bind inputs
    {
        struct ggml_tensor * tok_in = use_static ? dec_tok_in : ggml_graph_get_tensor(gf, "tok_in");
        struct ggml_tensor * pos_in = use_static ? dec_pos_in : ggml_graph_get_tensor(gf, "pos_in");
        if (emb_mode) {
            // vision prefix embeddings [D, T]; memory layout d-fastest matches row-major [T][D]
            struct ggml_tensor * emb_in = ggml_graph_get_tensor(gf, "emb_in");
            ggml_backend_tensor_set(emb_in, emb_prefix, 0, sizeof(float) * (size_t)model.n_embd * n_seq);
        } else {
            ggml_backend_tensor_set(tok_in, toks.data(), 0, sizeof(int) * toks.size());
        }
        int posv0 = n_past;
        ggml_backend_tensor_set(pos_in, &posv0, 0, sizeof(int));
        if (!use_static && n_seq > 1) {
            std::vector<int> posv(n_seq);
            for (int t = 0; t < n_seq; t++) posv[t] = n_past + t;
            ggml_backend_tensor_set(pos_in, posv.data(), 0, sizeof(int) * n_seq);
        }
        struct ggml_tensor * mask = use_static ? dec_mask_in : ggml_graph_get_tensor(gf, "mask_in");
        if (mask) {   // causal F16 [S, T]: allowed when key s <= past + query t (fixed graph: also -inf on unused tail)
            const int T = n_seq;
            const int S = (int)mask->ne[0];
            std::vector<ggml_fp16_t> mv(S * T);
            for (int t = 0; t < T; t++)
                for (int s = 0; s < S; s++)
                    // the image prefix (bos + 729 crops) attends bidirectionally, like the
                    // official attn_mask[..., :730, :730] = 1; region embeddings decode causally
                    mv[t * S + s] = ggml_fp32_to_fp16((s <= n_past + t || (emb_mode && emb_bidir)) ? 0.0f : -INFINITY);
            ggml_backend_tensor_set(mask, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());
        }
        // precomputed tau position factors: ps[h] = sigmoid(alpha_il[h]*log(pos+1)) + 0.5
        {
            const int T = n_seq, H = model.n_head;
            std::vector<float> psv(H * T);
            char nm[32];
            for (int il = 0; il < model.n_layer; il++) {
                snprintf(nm, sizeof(nm), "tau_ps_%d", il);
                struct ggml_tensor * t = use_static ? ggml_graph_get_tensor(dec_gf, nm)
                                                    : ggml_graph_get_tensor(gf, nm);
                if (!t) continue;
                for (int k = 0; k < T; k++) {
                    const float p = n_past + k + 1;
                    for (int h = 0; h < H; h++)
                        psv[k * H + h] = 1.0f / (1.0f + expf(-model.tau_alpha_host[il * H + h] * logf(p))) + 0.5f;
                }
                ggml_backend_tensor_set(t, psv.data(), 0, sizeof(float) * psv.size());
            }
        }
    }
    auto t2 = tick();

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        if (!use_static) ggml_free(gctx);
        return -1;
    }
    auto t3 = tick();
    if (prof) {
        t_build  += std::chrono::duration<double>(t1 - t0).count();
        t_input  += std::chrono::duration<double>(t2 - t1).count();
        t_compute+= std::chrono::duration<double>(t3 - t2).count();
        if (n_steps % 50 == 0)
            fprintf(stderr, "[prof] steps=%d setup+input=%.2fms compute=%.2fms\n",
                    n_steps, (t_build + t_input)/n_steps*1e3, t_compute/n_steps*1e3);
    }

    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (hidden_out) {
        struct ggml_tensor * hid = ggml_graph_get_tensor(gf, "hidden");
        if (hid) ggml_backend_tensor_get(hid, hidden_out, 0, sizeof(float) * model.n_embd);
    }
    if (const char * ddir = getenv("MD_DUMP")) {
        const char * names[] = { "d_h_emb", "d_l0_ln", "d_l0_qkv", "d_l0_q_tau", "d_l0_q_rope", "d_l0_k_rope",
                                 "d_l0_v2", "d_l0_scores", "d_l0_attnw", "d_l0_attn_out3", "d_l0_attn_out", "d_l0_attn_proj", "d_l0_mlp", "d_h_L0" };
        std::vector<uint8_t> tmp;
        for (const char * nm : names) {
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, nm);
            if (!t || !t->data) { fprintf(stderr, "dump: %s missing\n", nm); continue; }
            tmp.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, tmp.data(), 0, ggml_nbytes(t));
            char path[512];
            snprintf(path, sizeof(path), "%s\\cpp_%s.bin", ddir, nm);
            FILE * df = fopen(path, "wb");
            int64_t n0 = t->ne[0], n1 = t->ne[1], n2 = t->ne[2];
            for (int64_t i2 = 0; i2 < n2; i2++)
                for (int64_t i1 = 0; i1 < n1; i1++)
                    fwrite(tmp.data() + i2 * t->nb[2] + i1 * t->nb[1], sizeof(float), n0, df);
            fclose(df);
        }
    }
    int argmax = 0;
    if (logits && logits->data) {
        std::vector<float> lv(model.n_vocab);
        ggml_backend_tensor_get(logits, lv.data(), 0, sizeof(float) * model.n_vocab);
        // ids 1..9 are reserved control tokens and never sampled
        argmax = vqa::greedy_argmax(lv.data(), model.n_vocab, 1, 9);
        if (last_logits_out) memcpy(last_logits_out, lv.data(), sizeof(float) * model.n_vocab);
    }
    n_past += n_seq;
    if (!use_static) ggml_free(gctx);   // static decode graph stays alive for CUDA graph capture
    return argmax;
}

#include "moondream_batch.inc"   // after the Engine definition; adds step_batch + run_batch
#include "moondream_skills.inc"  // detect/point/gaze + image KV cache (shared)

#include "moondream_vqa_adapter.inc"   // vqa::Model adapter (unified front end)
