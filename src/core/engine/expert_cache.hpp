// expert_cache.hpp - expert-granularity device residency cache for routed MoE
// models (Strata-style tier 2: experts live in host memory, a bounded set of
// them is mirrored into device slots and computed on the GPU).
//
// CONTRACT (frozen; integration in keye_vl depends on these exact semantics):
//
//   One "expert" = T tensors (keye: up/gate/down, T=3), all sharing the same
//   per-layer layout. Every (layer, expert) pair maps to a fixed host address
//   (slice of the mmap'd expert weight tensors).
//
//   ExpertCache::init(dev, spec, budget_bytes)
//       - allocates as many device slots as fit in budget_bytes (a slot holds
//         one expert's T tensors); device context/buffer owned by the cache
//       - spec describes per-layer tensor layouts and host addresses
//   acquire(layer, expert)
//       - HIT: touches LRU, returns the slot's T device tensors
//       - MISS: picks a victim slot (LRU, never the tensors returned by any
//         acquire in the CURRENT step - see begin_step), synchronously copies
//         host->device (ggml_backend_tensor_set), returns the new tensors
//       - returned pointers are valid until the next eviction of that slot
//   begin_step() at the start of every decode step (pins the step's acquires
//   so a later layer's miss cannot evict an earlier layer's slot mid-step);
//   the pin list is cleared by begin_step of the next step.
//   stats() returns {hits, misses, evictions, resident} for logging.
//
// v1 is synchronous (no prefetch thread, no async copies); the weak
// one-layer-ahead prefetch is a later addition behind the same API.
#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace vqa {

struct ExpertTensorDesc {
    int64_t ne[4] = { 0, 0, 0, 0 };   // ggml shape of ONE expert's tensor
    ggml_type type = GGML_TYPE_F32;
    size_t host_stride = 0;           // bytes between consecutive experts
    // host base pointer for layer L is spec.layer_base[L][tensor_idx]
};

struct ExpertCacheSpec {
    int n_layers = 0;
    int n_experts = 0;
    int tensors_per_expert = 0;
    std::vector<std::vector<const uint8_t *>> layer_base; // [layer][t] host ptrs
    std::vector<ExpertTensorDesc> desc;                   // [t] layout
};

struct ExpertCacheStats {
    uint64_t hits = 0, misses = 0, evictions = 0, swap_bytes = 0;
    int resident = 0, n_slots = 0;
};

class ExpertCache {
public:
    ~ExpertCache();

    // see the header contract. budget_bytes is a hard cap on device memory
    // used for slots (excludes per-tensor overhead).
    bool init(ggml_backend_t dev, const ExpertCacheSpec & spec, size_t budget_bytes);

    // call once at the start of each decode step
    void begin_step();

    // returns tensors_per_expert device tensors (or empty on error)
    std::vector<ggml_tensor *> acquire(int layer, int expert);

    ExpertCacheStats stats() const;

    void free_cache();

    // ------------------------------------------------------------------ v2
    // async prefetch (additive contract; v1 semantics unchanged):
    //   prefetch(pairs) - for each (layer, expert) not already resident and not
    //     already in flight: a background thread copies host->pinned staging,
    //     then enqueues pinned->device cudaMemcpyAsync on the cache's private
    //     stream into a chosen victim slot (LRU, skipping pinned/in-flight).
    //     Already-resident or in-flight entries are no-ops.
    //   acquire(layer, expert) - if the entry is in flight, blocks on its
    //     event before returning the tensors (same return contract as v1).
    //   in_flight() - count, for stats.
    // enable_async() must be called after init to start the worker thread;
    // without it prefetch() is a no-op and behavior is exactly v1.
    bool enable_async();
    void prefetch(const std::vector<std::pair<int, int>> & layer_experts);
    int in_flight() const;

    // ------------------------------------------------------------------ v3
    // slot-table layout (additive contract): after init(), call
    // enable_table_layout() BEFORE any acquire. Storage is then ONE
    // contiguous 3D tensor per tensor index - [ne0, ne1, n_slots] - and every
    // acquire() returns a VIEW into it (same byte contents as before).
    // Purpose: a decode path can batch the whole selected-expert set through
    // ggml_mul_mat_id(table, x, ids) instead of per-expert ggml_mul_mat
    // (24 GEMV launches -> 3), which also unlocks the CUDA backend's
    // fused mul_mat_id+mul_mat_id+GLU pass.
    //   table(t)         - the 3D slot table for tensor index t (nullptr if
    //                      table layout is off)
    //   slot_index(view) - the view's ne[2] position in its table (the id to
    //                      pass to mul_mat_id); -1 if not a slot view
    // NOTE: eviction REUSES slots, so ids are only valid between acquire()
    // and the next acquire() on the same cache; read them right after
    // acquiring, before building the graph that consumes them.
// v3 decision: table layout is chosen in init() from env EC_TABLE (default
// ON, i.e. EC_TABLE unset or !=0); enable_table_layout() before init forces
// it on regardless of EC_TABLE. The header stays otherwise frozen.
bool enable_table_layout();
    ggml_tensor * table(int t);
    int slot_index(const ggml_tensor * slot_view) const;

private:
    struct Slot {
        std::vector<ggml_tensor *> tensors;   // device tensors_per_expert
        int layer = -1, expert = -1;          // -1 = free
        uint64_t last_use = 0;                // LRU stamp
    };
    ggml_backend_t dev_ = nullptr;
    ggml_context * ctx_ = nullptr;            // owns the slot tensors
    ggml_backend_buffer_t buf_ = nullptr;
    ExpertCacheSpec spec_;
    std::vector<Slot> slots_;
    uint64_t clock_ = 0;
    ExpertCacheStats stats_;
    std::vector<int> pinned_;                 // slot indices acquired this step

    struct Async;                             // internal: thread + stream + staging (cpp-side)
    Async * async_ = nullptr;
};

}  // namespace vqa
