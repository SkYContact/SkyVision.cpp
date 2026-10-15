// attention.hpp - shared KV-cache and attention-mask plumbing.
// Every backend allocates the same F16 [n_kv_head*head_dim, n_ctx] K/V pair per
// layer and builds the same causal / windowed F16 masks; keep one copy here.
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace vqa {

// F16 KV cache for a dense transformer decoder: n_layer pairs of
// [n_kv_head*head_dim, n_ctx] tensors in their own context + backend buffer.
struct KvCache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;
    int n_layer = 0, n_ctx = 0;

    bool alloc(ggml_backend_t backend, int n_layer_, int n_kv_head, int head_dim, int n_ctx_) {
        n_layer = n_layer_;
        n_ctx = n_ctx_;
        const size_t kv_bytes = (size_t)n_ctx * n_kv_head * head_dim * sizeof(ggml_fp16_t);
        struct ggml_init_params ip = { (size_t)(n_layer * 2 + 8) * ggml_tensor_overhead(), NULL, true };
        ctx = ggml_init(ip);
        if (!ctx) return false;
        k.resize(n_layer);
        v.resize(n_layer);
        for (int i = 0; i < n_layer; i++) {
            k[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_head * head_dim, n_ctx);
            v[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_head * head_dim, n_ctx);
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fprintf(stderr, "KV cache allocation failed (%.2f GB)\n", (double)n_layer * 2 * kv_bytes / 1e9);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        fprintf(stderr, "kv cache: %.2f GB (n_ctx %d)\n", (double)n_layer * 2 * kv_bytes / 1e9, n_ctx);
        return true;
    }

    void clear() {
        if (buf) ggml_backend_buffer_clear(buf, 0);
    }

    void free() {
        if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
    }
};

// causal mask for a [S, T] F16 graph input: query t may attend key s <= base + t.
// dst points at the graph input tensor (S = dst->ne[0]).
inline void set_causal_mask(ggml_tensor * dst, int base, int T) {
    const int S = (int)dst->ne[0];
    std::vector<ggml_fp16_t> mv((size_t)S * T, ggml_fp32_to_fp16(-INFINITY));
    for (int t = 0; t < T; t++)
        for (int s = 0; s <= std::min(S - 1, base + t); s++)
            mv[(size_t)t * S + s] = ggml_fp32_to_fp16(0.0f);
    ggml_backend_tensor_set(dst, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());
}

// per-column visibility windows for batch decode: column c attends keys in
// [lo[c], hi[c]) plus the keys it is about to write at pos[c] (its own query).
inline void set_window_mask(ggml_tensor * dst, const std::vector<int> & lo,
                            const std::vector<int> & hi, const std::vector<int> & pos) {
    const int S = (int)dst->ne[0];
    const int T = (int)pos.size();
    std::vector<ggml_fp16_t> mv((size_t)S * T, ggml_fp32_to_fp16(-INFINITY));
    for (int t = 0; t < T; t++) {
        const int own = pos[t];   // the key this query writes (always visible to itself)
        for (int s = lo[t]; s < hi[t]; s++)
            mv[(size_t)t * S + s] = ggml_fp32_to_fp16(0.0f);
        if (own >= 0 && own < S) mv[(size_t)t * S + own] = ggml_fp32_to_fp16(0.0f);
    }
    ggml_backend_tensor_set(dst, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());
}

// static decode graphs want fixed-size KV windows: round the context up to a
// block so the graph object stays identical across steps (required for ggml CUDA
// graph capture) while attention still only streams the used part of the block.
inline int block_for(int s, int block = 512) {
    const int b = std::max(block, 1);
    return ((std::max(s, 1) + b - 1) / b) * b;
}

}  // namespace vqa
