// decode_graph.hpp - one reusable decode graph per backend, shared by every
// decoder that used to rebuild its graph for every generated token.
//
// The old shape (copied into half a dozen backends) was:
//
//   for each token:
//     TextGraph d;                                   // fresh 32-256 MB context
//     build_text_graph(d, 1, n_past + 1, ...);       // N nodes
//     gallocr_alloc_graph(allocr, d.gf);             // buffer assignment
//     set_causal_mask(d.mask_in, n_past, 1);
//     run_text(d, ...);
//     ggml_free(d.ctx);                              // throw it all away
//
// Two costs, both measured: the build + alloc is ~0.5 ms of a ~2.5 ms CUDA step,
// and - less obvious - it defeats ggml-cuda's CUDA graph cache, which keys on
// cgraph->nodes[0]. A context per token means the node addresses change every
// step, so a capture is never replayed. With one reused graph the addresses and
// the shapes are stable and ggml-cuda can capture on the first step and replay
// afterwards.
//
// Fix: keep the graph, bucket the KV window. Query rows beyond n_past are masked
// -inf by set_causal_mask, so widening the window is numerically exact - it only
// makes flash attention read keys that cannot contribute. That is a bad trade
// where attention is the bottleneck and a good one where kernel launch is, so
// the bucket width follows the backend (see decode_bucket).
#pragma once

#include "ggml.h"

#include <algorithm>
#include <cstdlib>

namespace vqa {

// KV window for a decode step: s = n_past + 1 rounded up to the backend's bucket,
// clamped to n_ctx. CPU: 128 (attention-bound, keep the dead-key tail short),
// GPU: 1024 (launch-bound, fewer rebuilds and a stable shape for graph capture).
// VQA_DECODE_BUCKET=N overrides the width for A/B testing (N=1 restores the exact
// per-token window, i.e. the pre-bucketing behaviour).
inline int decode_bucket(int s, int n_ctx, bool cpu) {
    static const int env = [] {
        const char * e = getenv("VQA_DECODE_BUCKET");
        return e ? atoi(e) : 0;
    }();
    const int b = env > 0 ? env : (cpu ? 128 : 1024);
    int S = ((std::max(s, 1) + b - 1) / b) * b;
    if (n_ctx > 0 && S > n_ctx) S = n_ctx;
    return S;
}

// Owns one decode graph and rebuilds it only when the shape key changes.
//   Graph must be an aggregate with a `ctx` member (every backend's TextGraph is).
//   prepare(key, build): build(g) returns false on failure.
template <class Graph>
struct GraphSlot {
    Graph g{};
    int key = -1;

    template <class Build>
    bool prepare(int want, Build && build) {
        if (g.ctx && key == want) return true;
        reset();
        if (!build(g)) {
            reset();
            return false;
        }
        key = want;
        return true;
    }

    void reset() {
        if (g.ctx) ggml_free(g.ctx);
        g = Graph{};
        key = -1;
    }
};

}  // namespace vqa
