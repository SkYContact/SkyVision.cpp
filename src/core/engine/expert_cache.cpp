// expert_cache.cpp - v1 synchronous implementation of ExpertCache
// (see expert_cache.hpp for the frozen contract).
#include "core/engine/expert_cache.hpp"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace vqa {

ExpertCache::~ExpertCache() {
    free_cache();
}

namespace {

bool is_pinned(const std::vector<int> & pinned, int slot) {
    return std::find(pinned.begin(), pinned.end(), slot) != pinned.end();
}

// ---- v3 slot-table state (cpp-side registry; the frozen header carries no
// extra data members). One entry per live ExpertCache: `force_table` is set
// when enable_table_layout() is called before init(), `tables` holds the 3D
// slot tables [ne0, ne1, n_slots] once a table-layout init succeeded. Slot
// tensors are then ggml_view_3d slices of the tables (slots_[s].tensors[t],
// created once at init - view->data is valid because the views are created
// AFTER ggml_backend_alloc_ctx_tensors has assigned table->data).
struct EcV3 {
    bool force_table = false;
    std::vector<ggml_tensor *> tables;   // empty = table layout not in use
};
std::unordered_map<const ExpertCache *, EcV3> g_v3;

EcV3 & v3of(ExpertCache * self) { return g_v3[self]; }

}  // namespace

bool ExpertCache::init(ggml_backend_t dev, const ExpertCacheSpec & spec, size_t budget_bytes) {
    free_cache();

    if (!dev || spec.n_layers <= 0 || spec.n_experts <= 0 ||
        spec.tensors_per_expert <= 0 ||
        (int)spec.desc.size() != spec.tensors_per_expert ||
        (int)spec.layer_base.size() != spec.n_layers) {
        fprintf(stderr, "ExpertCache::init: invalid spec\n");
        return false;
    }

    // slot_bytes = sum over the expert's tensors of ne0*..*nek * type size
    // (computed without a ggml context; ggml_nbytes is later used to verify
    // per-tensor sizes when copying).
    size_t slot_bytes = 0;
    for (int t = 0; t < spec.tensors_per_expert; ++t) {
        const auto & d = spec.desc[t];
        int nd = 0;
        while (nd < GGML_MAX_DIMS && d.ne[nd] > 0) ++nd;
        if (nd == 0) {
            fprintf(stderr, "ExpertCache::init: tensor %d has zero shape\n", t);
            return false;
        }
        size_t count = 1;
        for (int i = 0; i < nd; ++i) count *= (size_t)d.ne[i];
        if (d.type < 0 || d.type >= GGML_TYPE_COUNT) {
            fprintf(stderr, "ExpertCache::init: bad ggml_type %d\n", (int)d.type);
            return false;
        }
        slot_bytes += count * ggml_type_size(d.type);
    }

    if (slot_bytes == 0 || budget_bytes < slot_bytes) {
        fprintf(stderr, "ExpertCache::init: budget too small (slot=%zu budget=%zu)\n",
                slot_bytes, budget_bytes);
        return false;
    }
    const int n_slots = (int)(budget_bytes / slot_bytes);
    if (getenv("KEYE_EC_CACHE_DBG"))
        fprintf(stderr, "ExpertCache::init: debug slot_bytes=%zu n_slots=%d budget=%zu\n",
                slot_bytes, n_slots, budget_bytes);
    if (n_slots <= 0) {
        fprintf(stderr, "ExpertCache::init: no slots fit\n");
        return false;
    }

    // ---- v3: table-layout decision (env EC_TABLE, default ON) ------------
    // Tables are allocated in the cache's own ctx + device buffer below; the
    // per-slot tensors become views of them. enable_table_layout() called
    // BEFORE init() forces the layout on regardless of EC_TABLE.
    EcV3 & v3 = v3of(this);
    const char * ec_table_env = getenv("EC_TABLE");
    const bool want_table = v3.force_table ||
                            (!ec_table_env || atoi(ec_table_env) != 0);
    v3.tables.clear();

    // metadata-only context (no_alloc: tensor data lives in the device buffer
    // allocated below), but the context still needs a pool for tensor/object
    // metadata. Use the public ggml_tensor_overhead() per tensor (the earlier
    // flat 256B/tensor estimate was too small for the object header + tensor
    // struct + name and GGML_ASSERTed in ggml_new_object). Table layout needs
    // T table tensors + n_slots*T view tensors; classic layout n_slots*T.
    const size_t meta_bytes =
        (size_t)(n_slots + (want_table ? 1 : 0)) * spec.tensors_per_expert *
        ggml_tensor_overhead() + 1024;
    struct ggml_init_params ip = { meta_bytes, nullptr, /*no_alloc*/ true };
    ctx_ = ggml_init(ip);
    if (!ctx_) {
        fprintf(stderr, "ExpertCache::init: ggml_init failed\n");
        return false;
    }

    if (want_table) {
        v3.tables.resize(spec.tensors_per_expert, nullptr);
        for (int t = 0; t < spec.tensors_per_expert; ++t) {
            const auto & d = spec.desc[t];
            int nd = 0;
            while (nd < GGML_MAX_DIMS && d.ne[nd] > 0) ++nd;
            int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
            for (int i = 0; i < nd; ++i) ne[i] = d.ne[i];
            const int64_t ne3[3] = { ne[0], ne[1], (int64_t)n_slots };
            v3.tables[t] = ggml_new_tensor(ctx_, d.type, 3, ne3);
            if (!v3.tables[t]) {
                fprintf(stderr, "ExpertCache::init: table alloc failed\n");
                free_cache();
                return false;
            }
        }
        buf_ = ggml_backend_alloc_ctx_tensors(ctx_, dev);
        if (!buf_) {
            fprintf(stderr, "ExpertCache::init: device alloc failed (%d slots x %zu bytes)\n",
                    n_slots, slot_bytes);
            free_cache();
            return false;
        }
        // views AFTER allocation: ggml_view_3d resolves view->data from the
        // (now non-NULL) table->data, so every swap-in path (backend_tensor_
        // set on the view, cudaMemcpyAsync(view->data, ...)) lands at
        // table data + slot offset directly.
        slots_.assign(n_slots, Slot());
        for (int s = 0; s < n_slots; ++s) {
            auto & slot = slots_[s];
            slot.tensors.resize(spec.tensors_per_expert);
            for (int t = 0; t < spec.tensors_per_expert; ++t) {
                ggml_tensor * tb = v3.tables[t];
                slot.tensors[t] = ggml_view_3d(ctx_, tb, tb->ne[0], tb->ne[1], 1,
                                               tb->nb[1], tb->nb[2],
                                               (size_t)s * tb->nb[2]);
                if (!slot.tensors[t]) {
                    fprintf(stderr, "ExpertCache::init: slot view alloc failed\n");
                    free_cache();
                    return false;
                }
            }
        }
        dev_ = dev;
        spec_ = spec;
        clock_ = 0;
        stats_ = ExpertCacheStats{};
        stats_.n_slots = n_slots;
        pinned_.clear();
        fprintf(stderr, "ExpertCache: %d slots x %zu bytes (%.2f MiB) on %s"
                " [table layout: %d tables x %d slices]\n",
                n_slots, slot_bytes, (double)(n_slots * slot_bytes) / (1024.0 * 1024.0),
                ggml_backend_name(dev), spec.tensors_per_expert, n_slots);
        return true;
    }

    slots_.assign(n_slots, Slot());
    for (int s = 0; s < n_slots; ++s) {
        auto & slot = slots_[s];
        slot.tensors.resize(spec.tensors_per_expert);
        for (int t = 0; t < spec.tensors_per_expert; ++t) {
            const auto & d = spec.desc[t];
            int nd = 0;
            while (nd < GGML_MAX_DIMS && d.ne[nd] > 0) ++nd;
            int64_t ne[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
            for (int i = 0; i < nd; ++i) ne[i] = d.ne[i];
            slot.tensors[t] = ggml_new_tensor(ctx_, d.type, nd, ne);
            if (!slot.tensors[t]) {
                fprintf(stderr, "ExpertCache::init: tensor alloc failed\n");
                free_cache();
                return false;
            }
        }
    }

    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, dev);
    if (!buf_) {
        fprintf(stderr, "ExpertCache::init: device alloc failed (%d slots x %zu bytes)\n",
                n_slots, slot_bytes);
        free_cache();
        return false;
    }

    dev_ = dev;
    spec_ = spec;
    clock_ = 0;
    stats_ = ExpertCacheStats{};
    stats_.n_slots = n_slots;
    pinned_.clear();
    fprintf(stderr, "ExpertCache: %d slots x %zu bytes (%.2f MiB) on %s\n",
            n_slots, slot_bytes, (double)(n_slots * slot_bytes) / (1024.0 * 1024.0),
            ggml_backend_name(dev));
    return true;
}

void ExpertCache::begin_step() {
    pinned_.clear();
}

// v2.1: async prefetch. One worker thread + one private CUDA stream (created
// LOW PRIORITY - see below) + a pinned staging pool (cudaHostAlloc, grows on
// demand, entries recycled only after their completion event has been polled
// done) + a small cudaEvent free-list + a staging THREAD POOL that parallelizes
// the pageable->pinned memcpy (single-threaded measured only ~4.3 GB/s on this
// machine; mmap cold pages make the serial copy the swap-wait bottleneck).
//
// Staging-pool ownership with block-parallel copies (design note): we keep ONE
// staging entry per (job, tensor), exactly as v2. The pool threads copy
// DISJOINT fixed-size chunks (EC_CHUNK bytes) into that single entry, so no
// two threads ever write the same staging byte; the entry stays busy for the
// whole job until the copy event has been recorded, exactly like v2. This is
// the simplest scheme that needs no cross-thread exclusion and cannot clobber
// a neighbouring in-flight tensor's staging. The worker issues the tensor's
// cudaMemcpyAsync only AFTER all of its chunks report done (remaining==0
// barrier below), preserving the "device copy sees fully staged bytes"
// invariant.
//
// Stream priority: the private stream is created with priority -1 (lower than
// the default-priority compute stream used by graph_compute/GEMV). Under PCIe
// bus / SM contention the decode kernels win, the prefetch copies yield; no
// cross-stream sync semantics change (acquire still cudaEventSynchronize's on
// the entry's event before returning the slot tensors, so data-readiness for
// the compute stream is exactly the v2 contract). The stream keeps the
// cudaStreamNonBlocking flag so it never serializes against the default
// stream. Priority is only a scheduling hint - on some drivers/devices it may
// be a no-op, which is safe (falls back to v2 contention behavior).
// ---------------------------------------------------------------------------

#ifdef GGML_USE_CUDA

struct ExpertCache::Async {
    struct Job { int layer, expert; };
    struct InFlight {                       // event==nullptr: reserved, copies issued
        int layer, expert, slot;            // but the record hasn't been made yet
        cudaEvent_t event;
    };
    struct Staging { uint8_t * ptr; size_t size; cudaEvent_t done_event; bool busy; };

    ExpertCache & ec;
    cudaStream_t stream = nullptr;
    std::mutex mtx;                         // guards queue/inflight/slots_
    std::condition_variable cv;             // worker wakeup + "event now set"
    std::deque<Job> queue;
    std::vector<InFlight> inflight;
    std::vector<Staging> staging;
    std::vector<cudaEvent_t> free_events;
    bool stop = false;
    std::thread worker;

    // ---- staging thread pool (v2.1) --------------------------------------
    // Chunk queue consumed by stage_threads[]. The worker bumps `remaining`
    // once per chunk BEFORE pushing it (both under smtx), so `remaining`
    // reaching 0 is only possible after every chunk has been handed out and
    // finished - that is the all-chunks-done barrier stage_copy() waits on.
    struct Chunk { const uint8_t * src; uint8_t * dst; size_t len; };
    static constexpr size_t EC_CHUNK = size_t(4) << 20;  // 4 MiB slices
    int n_stage_threads = 1;
    std::vector<std::thread> stage_threads;
    std::mutex smtx;
    std::condition_variable scv_push;       // pool: new chunks available
    std::condition_variable scv_done;       // worker: remaining hit zero
    std::deque<Chunk> chunkq;
    int remaining = 0;                      // outstanding chunk count
    bool pool_stop = false;

    explicit Async(ExpertCache & e) : ec(e) {
        // low-priority copy stream: yields to the default-priority compute
        // stream under contention; NonBlocking kept from v2
        cudaError_t serr = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1);
        if (serr != cudaSuccess)
            serr = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (serr == cudaSuccess) {
            start_stage_pool();
            fprintf(stderr, "ExpertCache::Async: %d staging thread(s)%s, "
                    "copy stream priority -1\n",
                    n_stage_threads,
                    n_stage_threads > 1 ? " (EC_STAGE_THREADS)" : "");
            worker = std::thread([this] { loop(); });
        }
    }

    void start_stage_pool() {
        int n = 0;
        if (const char * s = getenv("EC_STAGE_THREADS")) n = atoi(s);
        if (n <= 0) {
            // default: min(4, hw_threads/2)
            unsigned hw = std::thread::hardware_concurrency();
            n = hw ? (int)(hw / 2) : 1;
            if (n > 4) n = 4;
            if (n < 1) n = 1;
        }
        if (n > 16) n = 16;
        n_stage_threads = n;
        for (int i = 1; i < n; ++i)   // thread 0 is the worker itself
            stage_threads.emplace_back([this] { stage_loop(); });
    }

    // pool worker: pop a 4 MiB chunk, memcpy, count it down
    void stage_loop() {
        for (;;) {
            Chunk c{};
            {
                std::unique_lock<std::mutex> lk(smtx);
                scv_push.wait(lk, [this] { return pool_stop || !chunkq.empty(); });
                if (pool_stop && chunkq.empty()) return;
                c = chunkq.front();
                chunkq.pop_front();
            }
            memcpy(c.dst, c.src, c.len);
            {
                std::lock_guard<std::mutex> lk(smtx);
                if (--remaining == 0) scv_done.notify_all();
            }
        }
    }

    // pageable -> pinned copy for ONE tensor; sliced into EC_CHUNK chunks that
    // the pool copies in parallel (chunks are disjoint, see the design note at
    // the top of this section). Returns only after every chunk finished, so the
    // caller may immediately cudaMemcpyAsync from the staging entry.
    void stage_copy(const uint8_t * src, uint8_t * dst, size_t n) {
        if (n_stage_threads <= 1 || n <= EC_CHUNK) {
            memcpy(dst, src, n);            // EC_STAGE_THREADS=1: v2 baseline
            return;
        }
        const size_t nch = (n + EC_CHUNK - 1) / EC_CHUNK;
        {
            std::lock_guard<std::mutex> lk(smtx);
            remaining += (int)nch;
            for (size_t c = 0; c < nch; ++c) {
                const size_t off = c * EC_CHUNK;
                chunkq.push_back(Chunk{src + off, dst + off,
                                       std::min(EC_CHUNK, n - off)});
            }
        }
        scv_push.notify_all();
        std::unique_lock<std::mutex> lk(smtx);
        scv_done.wait(lk, [this] { return remaining == 0; });
    }

    ~Async() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop = true;
        }
        cv.notify_all();
        if (worker.joinable()) worker.join();
        // the worker is the only task producer: after joining it the pool is
        // (or will immediately go) idle; drain + join before freeing staging
        {
            std::lock_guard<std::mutex> lk(smtx);
            pool_stop = true;
        }
        scv_push.notify_all();
        for (auto & t : stage_threads)
            if (t.joinable()) t.join();
        if (stream) {
            cudaStreamSynchronize(stream);  // let in-flight copies finish
            cudaStreamDestroy(stream);
        }
        for (const auto & f : inflight)
            if (f.event) cudaEventDestroy(f.event);
        for (cudaEvent_t e : free_events) cudaEventDestroy(e);
        for (auto & s : staging)
            if (s.ptr) cudaFreeHost(s.ptr);
    }

    // ---- decode-thread entry points --------------------------------------

    void enqueue(const std::vector<std::pair<int, int>> & pairs) {
        std::lock_guard<std::mutex> lk(mtx);
        for (const auto & p : pairs) {
            if (p.first < 0 || p.first >= ec.spec_.n_layers ||
                p.second < 0 || p.second >= ec.spec_.n_experts)
                continue;
            bool dup = false;
            for (const auto & s : ec.slots_)
                if (s.layer == p.first && s.expert == p.second) { dup = true; break; }
            if (!dup)
                for (const auto & f : inflight)
                    if (f.layer == p.first && f.expert == p.second) { dup = true; break; }
            if (!dup)
                for (const auto & j : queue)
                    if (j.layer == p.first && j.expert == p.second) { dup = true; break; }
            if (dup) continue;
            queue.push_back(Job{p.first, p.second});
        }
        cv.notify_one();
    }

    int count() {
        std::lock_guard<std::mutex> lk(mtx);
        return (int)(queue.size() + inflight.size());
    }

    // if (layer,expert) is in flight (or still queued): wait for its event,
    // take it over the v1 hit path (hits++ / pin / LRU touch) and return the
    // slot's tensors. Returns {} if the entry is neither queued nor in flight.
    std::vector<ggml_tensor *> wait_and_take(int layer, int expert) {
        cudaEvent_t ev = nullptr;
        int slot = -1;
        {
            std::unique_lock<std::mutex> lk(mtx);
            for (;;) {
                int idx = -1;
                for (size_t i = 0; i < inflight.size(); ++i)
                    if (inflight[i].layer == layer && inflight[i].expert == expert) {
                        idx = (int)i; break;
                    }
                if (idx >= 0 && inflight[idx].event) {
                    ev = inflight[idx].event;
                    slot = inflight[idx].slot;
                    break;
                }
                if (idx < 0) {
                    // not in flight: still queued? else fall back to v1
                    bool queued = false;
                    for (const auto & j : queue)
                        if (j.layer == layer && j.expert == expert) { queued = true; break; }
                    if (!queued) return {};
                }
                // copies still being issued (or the job still queued): wait
                // for the worker to progress; the timeout covers dropped
                // entries whose notify we might have missed.
                cv.wait_for(lk, std::chrono::milliseconds(20));
            }
        }
        cudaEventSynchronize(ev);
        {
            std::lock_guard<std::mutex> lk(mtx);
            for (size_t i = 0; i < inflight.size(); ++i)
                if (inflight[i].layer == layer && inflight[i].expert == expert &&
                    inflight[i].slot == slot) {
                    inflight.erase(inflight.begin() + i); break;
                }
            free_events.push_back(ev);
            ++ec.clock_;
            ec.slots_[slot].last_use = ec.clock_;
            ++ec.stats_.hits;
            if (!is_pinned(ec.pinned_, slot)) ec.pinned_.push_back(slot);
            return ec.slots_[slot].tensors;
        }
    }

    bool slot_busy_locked(int s) const {
        for (const auto & f : inflight)
            if (f.slot == s) return true;
        return false;
    }

    // ---- worker thread ----------------------------------------------------

    void loop() {
        for (;;) {
            Job j{};
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [this] { return stop || !queue.empty(); });
                if (stop && queue.empty()) return;
                j = queue.front();
                queue.pop_front();
            }
            run(j);
        }
    }

    // staging pool: worker-only. Reuse an entry whose last completion event
    // has polled done; otherwise grow the pool with a new cudaHostAlloc.
    // Entries are marked busy for the whole job: the pinned buffer may not be
    // rewritten until the copy's event has been recorded (let alone completed).
    int get_staging(size_t size) {
        for (size_t i = 0; i < staging.size(); ++i) {
            auto & s = staging[i];
            if (!s.busy && s.size >= size &&
                (!s.done_event || cudaEventQuery(s.done_event) == cudaSuccess))
                { s.busy = true; return (int)i; }
        }
        void * p = nullptr;
        if (cudaHostAlloc(&p, size, cudaHostAllocDefault) != cudaSuccess)
            return -1;
        staging.push_back(Staging{(uint8_t *)p, size, nullptr, true});
        return (int)staging.size() - 1;
    }

    cudaEvent_t get_event() {
        if (!free_events.empty()) {
            cudaEvent_t e = free_events.back();
            free_events.pop_back();
            return e;
        }
        cudaEvent_t e = nullptr;
        cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        return e;
    }

    void run(const Job & j) {
        int victim = -1;
        bool evicted = false;
        {
            std::lock_guard<std::mutex> lk(mtx);
            // it may have been swapped in synchronously meanwhile
            for (const auto & s : ec.slots_)
                if (s.layer == j.layer && s.expert == j.expert) {
                    cv.notify_all();      // a waiter on this entry can hit now
                    return;
                }
            for (int s = 0; s < (int)ec.slots_.size(); ++s)
                if (ec.slots_[s].layer < 0) { victim = s; break; }
            if (victim < 0) {
                uint64_t best = UINT64_MAX;
                for (int s = 0; s < (int)ec.slots_.size(); ++s) {
                    if (is_pinned(ec.pinned_, s)) continue;
                    if (slot_busy_locked(s)) continue;
                    if (ec.slots_[s].last_use < best) {
                        best = ec.slots_[s].last_use;
                        victim = s;
                    }
                }
                if (victim < 0) {         // nowhere to land: give up entry
                    cv.notify_all();      // waiter falls back to the v1 path
                    return;
                }
                evicted = true;
            }
            inflight.push_back(InFlight{j.layer, j.expert, victim, nullptr});
        }

        // copies run WITHOUT the lock (slot is reserved in inflight above)
        const int T = ec.spec_.tensors_per_expert;
        std::vector<int> st_idx(T, -1);
        size_t total = 0;
        cudaError_t err = cudaSuccess;
        for (int t = 0; t < T && err == cudaSuccess; ++t) {
            ggml_tensor * ts = ec.slots_[victim].tensors[t];
            const size_t nb = ggml_nbytes(ts);
            const uint8_t * host = ec.spec_.layer_base[j.layer][t] +
                                   (size_t)j.expert * ec.spec_.desc[t].host_stride;
            const int si = get_staging(nb);
            if (si < 0) { err = cudaErrorMemoryAllocation; break; }
            st_idx[t] = si;
            // pageable(mmap) -> pinned: block-parallel via the staging pool;
            // stage_copy returns only when every chunk of THIS tensor is done,
            // so the async device copy below is guaranteed fully staged bytes
            stage_copy(host, staging[si].ptr, nb);
            err = cudaMemcpyAsync(ts->data, staging[si].ptr, nb,
                                  cudaMemcpyHostToDevice, stream);
            total += nb;
        }
        if (err != cudaSuccess) {
            for (int t = 0; t < T; ++t)
                if (st_idx[t] >= 0) staging[st_idx[t]].busy = false;
            std::lock_guard<std::mutex> lk(mtx);
            for (size_t i = 0; i < inflight.size(); ++i)
                if (inflight[i].slot == victim && !inflight[i].event) {
                    inflight.erase(inflight.begin() + i); break;
                }
            cv.notify_all();
            fprintf(stderr, "ExpertCache::Async: copy failed (%s), entry dropped\n",
                    cudaGetErrorString(err));
            return;
        }
        cudaEvent_t ev = get_event();
        cudaEventRecord(ev, stream);
        for (int t = 0; t < T; ++t)
            if (st_idx[t] >= 0) {
                staging[st_idx[t]].done_event = ev;
                staging[st_idx[t]].busy = false;
            }

        {
            std::lock_guard<std::mutex> lk(mtx);
            for (auto & f : inflight)
                if (f.layer == j.layer && f.expert == j.expert &&
                    f.slot == victim && !f.event) {
                    f.event = ev; break;
                }
            ec.slots_[victim].layer = j.layer;
            ec.slots_[victim].expert = j.expert;
            ++ec.clock_;
            ec.slots_[victim].last_use = ec.clock_;
            if (evicted) ++ec.stats_.evictions;
            ec.stats_.swap_bytes += total;
            cv.notify_all();             // wake acquire() waiters on this entry
        }
    }
};

bool ExpertCache::enable_async() {
    if (!ctx_) return false;
    if (async_) return true;
    Async * a = new Async(*this);
    if (!a->stream) {                    // stream/thread startup failed -> stay v1
        delete a;
        return false;
    }
    async_ = a;
    return true;
}

void ExpertCache::prefetch(const std::vector<std::pair<int, int>> & layer_experts) {
    if (async_) async_->enqueue(layer_experts);
}

int ExpertCache::in_flight() const {
    return async_ ? async_->count() : 0;
}

#else  // !GGML_USE_CUDA: v1 only

struct ExpertCache::Async {};

bool ExpertCache::enable_async() { return false; }

void ExpertCache::prefetch(const std::vector<std::pair<int, int>> & layer_experts) {
    (void) layer_experts;
}

int ExpertCache::in_flight() const { return 0; }

#endif  // GGML_USE_CUDA

std::vector<ggml_tensor *> ExpertCache::acquire(int layer, int expert) {
    if (!ctx_ || layer < 0 || layer >= spec_.n_layers ||
        expert < 0 || expert >= spec_.n_experts) {
        fprintf(stderr, "ExpertCache::acquire: invalid args (layer=%d expert=%d)\n",
                layer, expert);
        return {};
    }

    // 0. async in-flight? block on its event, then return via the hit path
    //    (the swap-in bytes were already counted at enqueue time).
#ifdef GGML_USE_CUDA
    if (async_) {
        auto ts = async_->wait_and_take(layer, expert);
        if (!ts.empty()) return ts;
    }
#endif

    // Only the decode thread calls acquire/begin_step; the async worker is the
    // only other writer of slots_. Lock the slot table for the scans so the
    // worker's victim/eviction writes cannot interleave.
    std::unique_lock<std::mutex> lk;
#ifdef GGML_USE_CUDA
    if (async_) lk = std::unique_lock<std::mutex>(async_->mtx);
#endif

    // 1. HIT?
    for (int s = 0; s < (int)slots_.size(); ++s) {
        auto & slot = slots_[s];
        if (slot.layer == layer && slot.expert == expert) {
            ++clock_;
            slot.last_use = clock_;
            ++stats_.hits;
            if (!is_pinned(pinned_, s)) pinned_.push_back(s);
            return slot.tensors;
        }
    }

    // 2. MISS: pick victim (free slot first, else LRU non-pinned and not
    //    being filled by an async copy).
    ++stats_.misses;
    int victim = -1;
    for (int s = 0; s < (int)slots_.size(); ++s) {
        if (slots_[s].layer < 0) { victim = s; break; }
    }
    if (victim < 0) {
        uint64_t best = UINT64_MAX;
        for (int s = 0; s < (int)slots_.size(); ++s) {
            if (is_pinned(pinned_, s)) continue;
#ifdef GGML_USE_CUDA
            if (async_ && async_->slot_busy_locked(s)) continue;
#endif
            if (slots_[s].last_use < best) {
                best = slots_[s].last_use;
                victim = s;
            }
        }
        if (victim < 0) {
            // All slots are pinned by acquires of the CURRENT step: cannot
            // evict without invalidating tensors the caller still holds.
            fprintf(stderr,
                    "ExpertCache::acquire: all %d slots pinned this step; "
                    "cannot bring in (layer=%d expert=%d) - cache too small\n",
                    (int)slots_.size(), layer, expert);
            return {};
        }
    }
    const bool evicted = slots_[victim].layer >= 0;
    if (evicted) ++stats_.evictions;
    // Claim the slot (and pin it) BEFORE releasing the lock so the async
    // worker cannot pick it as a victim while the synchronous copies run.
    slots_[victim].layer = layer;
    slots_[victim].expert = expert;
    if (!is_pinned(pinned_, victim)) pinned_.push_back(victim);
    if (lk.owns_lock()) lk.unlock();

    // 3. Swap in: host -> device, tensor by tensor.
    auto & slot = slots_[victim];
    const char * cdbg = getenv("KEYE_EC_CACHE_DBG");
    for (int t = 0; t < spec_.tensors_per_expert; ++t) {
        const size_t nb = ggml_nbytes(slot.tensors[t]);
        const uint8_t * host = spec_.layer_base[layer][t] +
                               (size_t)expert * spec_.desc[t].host_stride;
        ggml_backend_tensor_set(slot.tensors[t], host, 0, nb);
        stats_.swap_bytes += nb;
        if (cdbg && (layer == 6 || layer == 0)) {
            uint8_t back[32];
            ggml_backend_tensor_get(slot.tensors[t], back, 0, sizeof(back));
            const bool ok = memcmp(back, host, sizeof(back)) == 0;
            fprintf(stderr, "cachedbg: acquire L%d e%d t%d slot %d nb %zu host %p"
                    " dev %p -> %s\n", layer, expert, t, victim, nb, host,
                    slot.tensors[t]->data, ok ? "OK" : "MISMATCH");
        }
    }
    ++clock_;
    slot.last_use = clock_;
    return slot.tensors;
}

ExpertCacheStats ExpertCache::stats() const {
    // refresh the derived fields; counters are updated by acquire()
    ExpertCacheStats st = stats_;
    st.n_slots = (int)slots_.size();
    st.resident = 0;
    for (const auto & slot : slots_) {
        if (slot.layer >= 0) ++st.resident;
    }
    return st;
}

// ------------------------------------------------------------------ v3
bool ExpertCache::enable_table_layout() {
    if (!ctx_) {
        // pre-init call: force the layout on for the next init()
        v3of(this).force_table = true;
        return true;
    }
    const auto it = g_v3.find(this);
    return it != g_v3.end() && !it->second.tables.empty();
}

ggml_tensor * ExpertCache::table(int t) {
    const auto it = g_v3.find(this);
    if (it == g_v3.end() || t < 0 || t >= (int)it->second.tables.size())
        return nullptr;
    return it->second.tables[t];
}

int ExpertCache::slot_index(const ggml_tensor * slot_view) const {
    if (!slot_view || !slot_view->view_src) return -1;
    const auto it = g_v3.find(this);
    if (it == g_v3.end()) return -1;
    for (const ggml_tensor * tb : it->second.tables) {
        if (slot_view->view_src != tb) continue;
        const size_t nb2 = tb->nb[2];
        if (nb2 == 0 || slot_view->view_offs % nb2 != 0) return -1;
        const int64_t idx = (int64_t)(slot_view->view_offs / nb2);
        if (idx < 0 || idx >= (int64_t)slots_.size()) return -1;
        return (int)idx;
    }
    return -1;
}

void ExpertCache::free_cache() {
    // stop the async worker (syncs its stream) before the buffers vanish
    if (async_) {
        delete async_;
        async_ = nullptr;
    }
    // v3: table views live in ctx_ (freed below); drop the table pointers so
    // table()/slot_index() report "off" after free, but KEEP force_table -
    // init() calls free_cache() first and a pre-init enable_table_layout()
    // must survive into the new init
    const auto it = g_v3.find(this);
    if (it != g_v3.end()) it->second.tables.clear();
    // free order: device buffer -> metadata context. The ggml_backend_t itself
    // is owned by the caller and is NOT freed here.
    if (buf_) {
        ggml_backend_buffer_free(buf_);
        buf_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    slots_.clear();
    pinned_.clear();
    dev_ = nullptr;
    stats_ = ExpertCacheStats{};
    clock_ = 0;
}


}  // namespace vqa

// ---------------------------------------------------------------------------
// standalone test: cl /nologo /O2 /EHsc /DEXPERT_CACHE_TEST_MAIN ...
// ---------------------------------------------------------------------------
#ifdef EXPERT_CACHE_TEST_MAIN

#include "ggml-cuda.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (cond) { printf("[PASS] %s\n", msg); }                          \
        else { printf("[FAIL] %s (line %d)\n", msg, __LINE__); ++g_fail; } \
    } while (0)

using namespace vqa;

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);   // keep output if the test crashes
    const int NL = 2, NE = 8, NT = 2;
    const int64_t NE0 = 64, NE1 = 32;
    const size_t tensor_bytes = (size_t)(NE0 * NE1) * sizeof(float); // 8192
    const size_t slot_bytes = tensor_bytes * NT;                    // 16384

    // fake mmap'd host weights: [layer][tensor][expert][bytes]
    std::vector<std::vector<std::vector<uint8_t>>> host(
        NL, std::vector<std::vector<uint8_t>>(
                NT, std::vector<uint8_t>(NE * tensor_bytes)));
    for (int l = 0; l < NL; ++l) {
        for (int t = 0; t < NT; ++t) {
            for (int e = 0; e < NE; ++e) {
                uint8_t * p = host[l][t].data() + (size_t)e * tensor_bytes;
                for (size_t i = 0; i < tensor_bytes; ++i) {
                    // pattern unique per (l,t,e,i), all bytes nonzero-ish
                    p[i] = (uint8_t)(i * 7 + e * 13 + t * 29 + l * 71 + 1);
                }
            }
        }
    }

    ExpertCacheSpec spec;
    spec.n_layers = NL;
    spec.n_experts = NE;
    spec.tensors_per_expert = NT;
    spec.layer_base.resize(NL);
    for (int l = 0; l < NL; ++l) {
        spec.layer_base[l].resize(NT);
        for (int t = 0; t < NT; ++t) {
            spec.layer_base[l][t] = host[l][t].data();
        }
    }
    spec.desc.resize(NT);
    for (int t = 0; t < NT; ++t) {
        spec.desc[t].ne[0] = NE0;
        spec.desc[t].ne[1] = NE1;
        spec.desc[t].type = GGML_TYPE_F32;
        spec.desc[t].host_stride = tensor_bytes;
    }

    ggml_backend_t dev = ggml_backend_cuda_init(0);
    if (!dev) {
        fprintf(stderr, "FATAL: no CUDA backend\n");
        return 1;
    }
    fprintf(stderr, "backend: %s\n", ggml_backend_name(dev));

    ExpertCache cache;
    const size_t budget = 4 * slot_bytes; // room for exactly 4 slots
    CHECK(cache.init(dev, spec, budget), "init succeeds");
    CHECK(cache.stats().n_slots == 4, "n_slots == 4");

    // --- test 1: correctness of the host->device copy -------------------
    {
        auto ts = cache.acquire(0, 3);
        CHECK(ts.size() == (size_t)NT, "acquire(0,3) returns 2 tensors");
        bool eq = true;
        std::vector<uint8_t> back(tensor_bytes);
        for (int t = 0; t < NT; ++t) {
            ggml_backend_tensor_get(ts[t], back.data(), 0, tensor_bytes);
            const uint8_t * src = host[0][t].data() + (size_t)3 * tensor_bytes;
            if (memcmp(back.data(), src, tensor_bytes) != 0) eq = false;
        }
        CHECK(eq, "test1: device bytes == host source (memcmp all equal)");
    }

    // --- test 2: hit path -----------------------------------------------
    {
        auto st0 = cache.stats();
        auto ts = cache.acquire(0, 3);
        auto st1 = cache.stats();
        CHECK(ts.size() == (size_t)NT, "second acquire(0,3) returns tensors");
        CHECK(st1.hits == st0.hits + 1 && st1.misses == st0.misses,
              "test2: hits incremented, misses unchanged");
        // spot-check data is still right on the hit path
        std::vector<uint8_t> back(tensor_bytes);
        ggml_backend_tensor_get(ts[1], back.data(), 0, tensor_bytes);
        CHECK(memcmp(back.data(), host[0][1].data() + (size_t)3 * tensor_bytes,
                     tensor_bytes) == 0,
              "test2: hit-path data still matches");
    }

    // --- test 3: LRU eviction across steps ------------------------------
    {
        cache.begin_step();          // pins (0,3) from above? no: clears pins
        cache.acquire(0, 0);         // slot1 (0,3 already in slot0)
        cache.begin_step();          // nothing pinned now
        cache.acquire(0, 1);         // slot2
        cache.acquire(0, 2);         // slot3
        cache.acquire(1, 0);         // no free slot -> LRU non-pinned.
                                     // LRU is (0,3) (oldest stamp) -> evicted
        auto st = cache.stats();
        CHECK(st.evictions > 0, "test3: evictions > 0");
        CHECK(st.resident == 4, "test3: all 4 slots resident");
        // (0,3) was evicted; re-acquire must miss
        uint64_t m0 = st.misses;
        auto ts = cache.acquire(0, 3);
        CHECK(ts.size() == (size_t)NT, "test3: re-acquire(0,3) succeeds");
        CHECK(cache.stats().misses == m0 + 1, "test3: re-acquire(0,3) is a miss");
        std::vector<uint8_t> back(tensor_bytes);
        ggml_backend_tensor_get(ts[0], back.data(), 0, tensor_bytes);
        CHECK(memcmp(back.data(), host[0][0].data() + (size_t)3 * tensor_bytes,
                     tensor_bytes) == 0,
              "test3: swapped-back-in expert bytes correct");
    }

    // --- test 4: same-step pinning --------------------------------------
    {
        cache.begin_step();
        bool ok = true;
        ok = ok && cache.acquire(1, 1).size() == (size_t)NT;
        ok = ok && cache.acquire(1, 2).size() == (size_t)NT;
        ok = ok && cache.acquire(1, 3).size() == (size_t)NT;
        ok = ok && cache.acquire(1, 4).size() == (size_t)NT;
        CHECK(ok, "test4: filled 4 slots with same-step acquires");
        auto st0 = cache.stats();
        auto ts = cache.acquire(1, 5);   // all 4 pinned -> must fail
        CHECK(ts.empty(), "test4: 5th same-step acquire returns empty (all pinned)");
        auto st1 = cache.stats();
        CHECK(st1.evictions == st0.evictions, "test4: no eviction of pinned slot");
        // still alive, and a new step unblocks the cache
        cache.begin_step();
        auto ts2 = cache.acquire(1, 5);
        CHECK(ts2.size() == (size_t)NT, "test4: after begin_step, acquire works");
    }

    // --- v2: async prefetch ------------------------------------------------
    CHECK(cache.enable_async(), "async: enable_async() returns true");

    // test 5: prefetch a non-resident expert -> in flight -> acquire hits
    {
        cache.begin_step();
        uint64_t h0 = cache.stats().hits, m0 = cache.stats().misses;
        cache.prefetch({{0, 4}});
        CHECK(cache.in_flight() > 0, "test5: prefetch -> in_flight() > 0");
        auto ts = cache.acquire(0, 4);
        CHECK(ts.size() == (size_t)NT, "test5: acquire(0,4) after prefetch");
        bool eq = true;
        std::vector<uint8_t> back(tensor_bytes);
        for (int t = 0; t < NT; ++t) {
            ggml_backend_tensor_get(ts[t], back.data(), 0, tensor_bytes);
            if (memcmp(back.data(), host[0][t].data() + (size_t)4 * tensor_bytes,
                       tensor_bytes) != 0) eq = false;
        }
        CHECK(eq, "test5: prefetched bytes memcmp-correct");
        CHECK(cache.in_flight() == 0, "test5: in_flight() back to 0");
        auto st1 = cache.stats();
        CHECK(st1.hits == h0 + 1 && st1.misses == m0,
              "test5: prefetched acquire counted as hit (not miss)");
    }

    // test 6: resident / duplicate prefetch is a no-op
    {
        cache.begin_step();
        // (0,4) and (1,3) are resident at this point
        cache.prefetch({{0, 4}, {1, 3}, {1, 3}});
        CHECK(cache.in_flight() == 0, "test6: resident prefetch -> in_flight stays 0");
        cache.prefetch({{0, 5}, {0, 5}});          // same expert twice in one call
        CHECK(cache.in_flight() == 1, "test6: duplicate prefetch enqueues exactly 1");
        auto ts = cache.acquire(0, 5);
        CHECK(ts.size() == (size_t)NT && cache.in_flight() == 0,
              "test6: dup entry acquired, in_flight 0");
    }

    // test 7: staging recycle under burst load (pool starts empty and must
    // grow; entries may only be recycled after their event polled done)
    {
        const int NE2 = 16;
        std::vector<std::vector<std::vector<uint8_t>>> host2(
            NL, std::vector<std::vector<uint8_t>>(
                    NT, std::vector<uint8_t>(NE2 * tensor_bytes)));
        for (int l = 0; l < NL; ++l)
            for (int t = 0; t < NT; ++t)
                for (int e = 0; e < NE2; ++e) {
                    uint8_t * p = host2[l][t].data() + (size_t)e * tensor_bytes;
                    for (size_t i = 0; i < tensor_bytes; ++i)
                        p[i] = (uint8_t)(i * 11 + e * 17 + t * 23 + l * 5 + 3);
                }
        ExpertCacheSpec spec2;
        spec2.n_layers = NL;
        spec2.n_experts = NE2;
        spec2.tensors_per_expert = NT;
        spec2.layer_base.resize(NL);
        for (int l = 0; l < NL; ++l) {
            spec2.layer_base[l].resize(NT);
            for (int t = 0; t < NT; ++t)
                spec2.layer_base[l][t] = host2[l][t].data();
        }
        spec2.desc.resize(NT);
        for (int t = 0; t < NT; ++t) {
            spec2.desc[t].ne[0] = NE0;
            spec2.desc[t].ne[1] = NE1;
            spec2.desc[t].type = GGML_TYPE_F32;
            spec2.desc[t].host_stride = tensor_bytes;
        }
        ExpertCache c2;
        CHECK(c2.init(dev, spec2, 8 * slot_bytes), "test7: 8-slot cache init");
        CHECK(c2.enable_async(), "test7: async enabled");
        std::vector<std::pair<int, int>> all;
        for (int e = 0; e < NE2; ++e) all.push_back({e % 2, e});
        c2.begin_step();
        c2.prefetch(all);
        CHECK(c2.in_flight() > 0, "test7: burst of 16 enqueued");
        bool ok = true;
        std::vector<uint8_t> back2(tensor_bytes);
        for (int e = 0; e < NE2 && ok; ++e) {
            c2.begin_step();
            auto ts = c2.acquire(e % 2, e);
            if (ts.size() != (size_t)NT) { ok = false; break; }
            for (int t = 0; t < NT; ++t) {
                ggml_backend_tensor_get(ts[t], back2.data(), 0, tensor_bytes);
                if (memcmp(back2.data(),
                           host2[e % 2][t].data() + (size_t)e * tensor_bytes,
                           tensor_bytes) != 0) ok = false;
            }
        }
        CHECK(ok, "test7: all 16 experts byte-correct (no staging clobber/leak)");
        c2.free_cache();
    }

    // test 8: perf smoke - 8 experts x ~1.29MB, prefetched vs sync swap-in
    {
        const int NT3 = 1, NP = 8;
        const int64_t BE0 = 576, BE1 = 576;                 // 576*576*4 = 1,327,104 B
        const size_t tb = (size_t)(BE0 * BE1) * sizeof(float);
        std::vector<std::vector<std::vector<uint8_t>>> host3(
            1, std::vector<std::vector<uint8_t>>(
                   NT3, std::vector<uint8_t>(2 * NP * tb)));
        for (int t = 0; t < NT3; ++t)
            for (int e = 0; e < 2 * NP; ++e)
                for (size_t i = 0; i < tb; ++i)
                    host3[0][t].data()[(size_t)e * tb + i] = (uint8_t)(i + e * 3);

        ExpertCacheSpec spec3;
        spec3.n_layers = 1;
        spec3.n_experts = 2 * NP;
        spec3.tensors_per_expert = NT3;
        spec3.layer_base.resize(1);
        spec3.layer_base[0].resize(NT3, host3[0][0].data());
        spec3.desc.resize(NT3);
        spec3.desc[0].ne[0] = BE0;
        spec3.desc[0].ne[1] = BE1;
        spec3.desc[0].type = GGML_TYPE_F32;
        spec3.desc[0].host_stride = tb;

        ExpertCache ca, cs;
        CHECK(ca.init(dev, spec3, NP * tb), "test8: async cache init");
        CHECK(cs.init(dev, spec3, NP * tb), "test8: sync cache init");
        CHECK(ca.enable_async(), "test8: async enabled");

        std::vector<std::pair<int, int>> first8;
        for (int e = 0; e < NP; ++e) first8.push_back({0, e});
        ca.begin_step();
        ca.prefetch(first8);
        // give the worker a head start; acquire() syncs on the events anyway
        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        auto t0 = std::chrono::steady_clock::now();
        for (int e = 0; e < NP; ++e) ca.acquire(0, e);
        auto t1 = std::chrono::steady_clock::now();

        cs.begin_step();
        auto t2 = std::chrono::steady_clock::now();
        for (int e = 0; e < NP; ++e) cs.acquire(0, e);
        auto t3 = std::chrono::steady_clock::now();

        const double ms_async =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double ms_sync =
            std::chrono::duration<double, std::milli>(t3 - t2).count();
        printf("perf smoke: acquire 8 prefetched experts (8 x %.2f MB) = %.3f ms"
               " | sync swap-in of 8 = %.3f ms\n",
               tb / (1024.0 * 1024.0), ms_async, ms_sync);

        // data check on the prefetched path
        std::vector<uint8_t> back3(tb);
        bool ok = true;
        for (int e = 0; e < NP; ++e) {
            auto ts = ca.acquire(0, e);
            ggml_backend_tensor_get(ts[0], back3.data(), 0, tb);
            if (memcmp(back3.data(), host3[0][0].data() + (size_t)e * tb, tb) != 0)
                ok = false;
        }
        CHECK(ok, "test8: perf-path data still correct");
        ca.free_cache();
        cs.free_cache();
    }

    // --- test 9: staging bandwidth - 16 experts x 2 tensors x 8.19 MB
    // (262 MB total) prefetched through the async path; prints effective
    // GB/s. Run once with EC_STAGE_THREADS=1 (single-threaded control) and
    // once with the default/multithread pool to compare.
    {
        const int NL9 = 1, NT9 = 2, NE9 = 32, PF9 = 16;
        const int64_t B0 = 1280, B1 = 1600;             // 1280*1600*4 = 8.19 MB
        const size_t tb9 = (size_t)(B0 * B1) * sizeof(float);
        std::vector<std::vector<std::vector<uint8_t>>> host9(
            NL9, std::vector<std::vector<uint8_t>>(
                     NT9, std::vector<uint8_t>(NE9 * tb9)));
        for (int t = 0; t < NT9; ++t)
            for (int e = 0; e < NE9; ++e)
                for (size_t i = 0; i < tb9; ++i)
                    host9[0][t].data()[(size_t)e * tb9 + i] =
                        (uint8_t)(i * 5 + e * 7 + t * 3 + 1);

        ExpertCacheSpec spec9;
        spec9.n_layers = NL9;
        spec9.n_experts = NE9;
        spec9.tensors_per_expert = NT9;
        spec9.layer_base.resize(NL9);
        for (int l = 0; l < NL9; ++l) {
            spec9.layer_base[l].resize(NT9);
            for (int t = 0; t < NT9; ++t)
                spec9.layer_base[l][t] = host9[0][t].data();
        }
        spec9.desc.resize(NT9);
        for (int t = 0; t < NT9; ++t) {
            spec9.desc[t].ne[0] = B0;
            spec9.desc[t].ne[1] = B1;
            spec9.desc[t].type = GGML_TYPE_F32;
            spec9.desc[t].host_stride = tb9;
        }

        ExpertCache c9;
        CHECK(c9.init(dev, spec9, PF9 * tb9 * NT9), "test9: bandwidth cache init");
        CHECK(c9.enable_async(), "test9: async enabled");

        std::vector<std::pair<int, int>> pf;
        for (int e = 0; e < PF9; ++e) pf.push_back({0, e});
        c9.begin_step();
        const auto b0 = std::chrono::steady_clock::now();
        c9.prefetch(pf);
        bool ok9 = true;
        std::vector<uint8_t> back9(tb9);
        for (int e = 0; e < PF9 && ok9; ++e) {
            auto ts = c9.acquire(0, e);
            if (ts.size() != (size_t)NT9) { ok9 = false; break; }
            for (int t = 0; t < NT9; ++t) {
                ggml_backend_tensor_get(ts[t], back9.data(), 0, tb9);
                if (memcmp(back9.data(),
                           host9[0][t].data() + (size_t)e * tb9, tb9) != 0)
                    ok9 = false;
            }
        }
        const auto b1 = std::chrono::steady_clock::now();
        const double ms9 =
            std::chrono::duration<double, std::milli>(b1 - b0).count();
        const double mb9 = (double)PF9 * NT9 * tb9 / (1024.0 * 1024.0);
        const char * env9 = getenv("EC_STAGE_THREADS");
        printf("bandwidth: %d experts x %d tensors x %.2f MB = %.1f MB prefetched"
               " in %.2f ms -> %.2f GB/s (EC_STAGE_THREADS=%s)\n",
               PF9, NT9, tb9 / (1024.0 * 1024.0), mb9, ms9,
               mb9 / 1024.0 / (ms9 / 1e3), env9 ? env9 : "default");
        CHECK(ok9, "test9: bandwidth-path data still correct");
        c9.free_cache();
    }

    // --- v3: slot-table layout --------------------------------------------
    // (tests 10-12; note every cache in this binary now inits the table layout
    // by default - EC_TABLE=1 - so tests 1-9 above ALREADY exercise it; run
    // the binary with EC_TABLE=0 to re-verify the classic layout)

    // test 10: acquire() returns table views; bytes, slot_index semantics
    {
        ExpertCache c10;
        // force the layout on BEFORE init so the test also passes with
        // EC_TABLE=0 (the classic layout is covered by tests 1-9 then)
        CHECK(c10.enable_table_layout(), "test10: pre-init enable_table_layout()");
        CHECK(c10.init(dev, spec, 4 * slot_bytes), "test10: table cache init");
        CHECK(c10.enable_table_layout(), "test10: enable_table_layout() true after init");
        ggml_tensor * t0 = c10.table(0), * t1 = c10.table(1);
        CHECK(t0 && t1, "test10: table(0)/table(1) exposed");
        CHECK(t0->ne[0] == NE0 && t0->ne[1] == NE1 && t0->ne[2] == 4,
              "test10: table shape [ne0,ne1,n_slots]");
        auto ts = c10.acquire(0, 5);
        CHECK(ts.size() == (size_t)NT, "test10: acquire returns 2 views");
        bool eq = true;
        std::vector<uint8_t> back(tensor_bytes);
        for (int t = 0; t < NT; ++t) {
            ggml_backend_tensor_get(ts[t], back.data(), 0, tensor_bytes);
            const uint8_t * src = host[0][t].data() + (size_t)5 * tensor_bytes;
            if (memcmp(back.data(), src, tensor_bytes) != 0) eq = false;
        }
        CHECK(eq, "test10: table-view device bytes == host source (memcmp)");
        const int idx = c10.slot_index(ts[0]);
        CHECK(idx >= 0 && idx < 4, "test10: slot_index(view) in [0, n_slots)");
        CHECK(c10.slot_index(ts[1]) == idx,
              "test10: both tensors of one slot share the index");
        CHECK(c10.slot_index(t0) == -1, "test10: slot_index(table) == -1");
        CHECK(c10.slot_index(nullptr) == -1, "test10: slot_index(nullptr) == -1");
        // the view's data must be the table slice (swap-in target check)
        CHECK(ts[0]->view_src == t0 &&
              ts[0]->data == (char *) t0->data + (size_t) idx * t0->nb[2],
              "test10: view points at table data + slot offset");
        c10.free_cache();
        CHECK(c10.table(0) == nullptr, "test10: table() null after free_cache");
    }

    // test 11: eviction reuses a table slot; new data lands in the same slice
    {
        ExpertCache c11;
        c11.enable_table_layout();   // force on (EC_TABLE-independent)
        CHECK(c11.init(dev, spec, 4 * slot_bytes), "test11: 4-slot table cache init");
        c11.begin_step();
        bool fill_ok = true;
        for (int e = 0; e < 4; ++e)
            if (c11.acquire(0, e).size() != (size_t)NT) fill_ok = false;
        CHECK(fill_ok, "test11: filled 4 table slots");
        c11.begin_step();
        auto ts4 = c11.acquire(0, 4);                    // evicts LRU (0,0)
        CHECK(ts4.size() == (size_t)NT, "test11: acquire(0,4) succeeds");
        CHECK(c11.slot_index(ts4[0]) >= 0, "test11: evicted-into view is a table slice");
        auto ts0 = c11.acquire(0, 0);                    // evicts (0,1) now
        CHECK(ts0.size() == (size_t)NT, "test11: re-acquire(0,0) succeeds");
        std::vector<uint8_t> back(tensor_bytes);
        ggml_backend_tensor_get(ts0[1], back.data(), 0, tensor_bytes);
        CHECK(memcmp(back.data(), host[0][1].data() + (size_t)0 * tensor_bytes,
                     tensor_bytes) == 0,
              "test11: evicted slot slice now holds the NEW expert's bytes");
        CHECK(c11.slot_index(ts0[0]) >= 0, "test11: slot_index valid after reuse");
        // the earlier view (ts4) must still be intact (its slot is pinned)
        ggml_backend_tensor_get(ts4[0], back.data(), 0, tensor_bytes);
        CHECK(memcmp(back.data(), host[0][0].data() + (size_t)4 * tensor_bytes,
                     tensor_bytes) == 0,
              "test11: pinned slot's slice not clobbered by the new swap-in");
        c11.free_cache();
    }

    // test 12: mul_mat_id over the cache's F32 table vs hand computation
    // table [D=2, I=3, S=4] (ne0 = input dim, ne1 = out rows); x=[1,2];
    // decode experts 1 and 3; out[i,k] = sum_j W_k[j*D + i] * x[j]
    {
        const int64_t I = 3, Dd = 2, S12 = 4;
        const size_t tb12 = (size_t)(I * Dd) * sizeof(float);
        std::vector<uint8_t> host12((size_t)8 * tb12);   // contiguous [expert]
        for (int e = 0; e < 8; ++e)
            for (size_t i = 0; i < tb12; ++i)
                host12[(size_t)e * tb12 + i] = (uint8_t)(i * 3 + e * 7 + 1);
        ExpertCacheSpec s12;
        s12.n_layers = 1; s12.n_experts = 8; s12.tensors_per_expert = 1;
        s12.layer_base.assign(1, { host12.data() });
        s12.desc.resize(1);
        s12.desc[0].ne[0] = Dd; s12.desc[0].ne[1] = I;
        s12.desc[0].type = GGML_TYPE_F32;
        s12.desc[0].host_stride = tb12;
        ExpertCache c12;
        c12.enable_table_layout();   // force on (EC_TABLE-independent)
        CHECK(c12.init(dev, s12, (size_t)S12 * tb12), "test12: F32 table cache init");
        ggml_tensor * tab = c12.table(0);
        CHECK(tab && tab->type == GGML_TYPE_F32 && tab->ne[2] == S12,
              "test12: F32 slot table exposed");
        // decode experts 1 and 3 into slots (free slots first: 0 then 1)
        c12.begin_step();
        auto va = c12.acquire(0, 1);
        auto vb = c12.acquire(0, 3);
        const int id_a = c12.slot_index(va[0]), id_b = c12.slot_index(vb[0]);
        CHECK(id_a >= 0 && id_b >= 0 && id_a != id_b, "test12: two distinct slot ids");

        ggml_context * cctx = ggml_init({ 1ull << 20, NULL, true });
        ggml_cgraph * gf12 = ggml_new_graph_custom(cctx, 128, false);
        ggml_tensor * xin = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, Dd, 1);
        ggml_tensor * idin = ggml_new_tensor_2d(cctx, GGML_TYPE_I32, 2, 1);
        ggml_tensor * x3 = ggml_reshape_3d(cctx, xin, Dd, 1, 1);
        ggml_tensor * out = ggml_mul_mat_id(cctx, tab, x3, idin);   // [I, 2, 1]
        ggml_build_forward_expand(gf12, out);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dev));
        CHECK(ga && ggml_gallocr_alloc_graph(ga, gf12), "test12: graph alloc");
        const float xv12[2] = { 1.0f, 2.0f };
        const int32_t ids12[2] = { id_a, id_b };
        ggml_backend_tensor_set(xin, xv12, 0, sizeof(xv12));
        ggml_backend_tensor_set(idin, ids12, 0, sizeof(ids12));
        CHECK(ggml_backend_graph_compute(dev, gf12) == GGML_STATUS_SUCCESS,
              "test12: compute");
        // host copies of the two experts' bytes (d0 = I fastest)
        const float * e1 = (const float *) (host12.data() + (size_t)1 * tb12);
        const float * e3 = (const float *) (host12.data() + (size_t)3 * tb12);
        std::vector<float> ov((size_t)(I * 2));
        ggml_backend_tensor_get(out, ov.data(), 0, sizeof(float) * ov.size());
        bool ok12 = true;
        for (int k = 0; k < 2 && ok12; ++k) {
            const float * ex = k == 0 ? e1 : e3;
            for (int i = 0; i < I && ok12; ++i) {
                // as[i,j,e] lives at j + i*Dd + e*stride (d0 = input dim)
                const float want = ex[(size_t)i * Dd + 0] * 1.0f
                                 + ex[(size_t)i * Dd + 1] * 2.0f;
                if (fabsf(ov[(size_t)k * I + i] - want) > 1e-6f) ok12 = false;
            }
        }
        CHECK(ok12, "test12: mul_mat_id(table, x, ids) == hand computation");
        ggml_gallocr_free(ga);
        ggml_free(cctx);
        c12.free_cache();
    }

    auto st = cache.stats();
    printf("final stats: hits=%llu misses=%llu evictions=%llu swap_bytes=%llu resident=%d n_slots=%d\n",
           (unsigned long long)st.hits, (unsigned long long)st.misses,
           (unsigned long long)st.evictions, (unsigned long long)st.swap_bytes,
           st.resident, st.n_slots);

    cache.free_cache();
    ggml_backend_free(dev);

    if (g_fail == 0) printf("ALL TESTS PASSED\n");
    else printf("%d TEST(S) FAILED\n", g_fail);
    return g_fail ? 1 : 0;
}

#endif  // EXPERT_CACHE_TEST_MAIN
