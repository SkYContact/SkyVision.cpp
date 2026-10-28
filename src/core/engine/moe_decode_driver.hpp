// moe_decode_driver.hpp - the shared orchestration layer for ExpertCache
// decode paths (extracted verbatim from the keye-vl2 and qwen3omni
// integrations, which were two hand copies of the same skeleton).
//
// The model-specific half stays in each backend: graph A's op sequence (each
// model's own T=1 layer math, built into a fixed vqa::StepGraph via
// Hooks::build_a) and the per-step input feeding beyond tok/hid (Hooks::feed_a
// - positions, mask, extra leaves; the two integrations differ in rope
// position layout and mask dtype). Everything else - the KV-window block
// rebuild, per-layer acquire/pin, one-layer-ahead prefetch, graph B (v3 slot
// tables) and graph C (final norm + head), timers - lives here once.
//
// CONTRACT (frozen; keye_vl and qwen3omni consume these exact semantics):
//   AIO          - the leaves/graph-A outputs every model must expose
//   Hooks        - geometry, weights, cache set, and the model callbacks
//   State        - driver-owned per-model decode state (graphs, timers)
//   init_grouped - layout grouping + VRAM budgeting + ExpertCache init
//   build_graphs - the n_layer fixed graph-A StepGraphs
//   decode_step  - one token step: graph A per layer -> readback -> acquire
//                  -> graph B -> graph C, with the optional hooks below
//
// Third-consumer extensions (nemotronomni hybrid decoder; all default to the
// exact keye_vl/qwen3omni semantics, i.e. nullptr/3/false keeps this driver
// bit-for-bit on the previous behavior):
//   Hooks::moe_of          - driver layer -> MoE-slot index for non-uniform
//                            stacks (nemotronomni interleaves Mamba2 /
//                            attention / MoE layers); -1 marks a layer with no
//                            routed experts: its graph A carries the WHOLE
//                            layer and decode_step skips prefetch/acquire/
//                            graph B for it (Hooks::attn_only must report true
//                            there so hid passes through)
//   Hooks::tensors_per_expert - 2 for non-gated experts (up/down only)
//   Hooks::build_b         - replaces the fixed gated-SILU graph B for MoE
//                            blocks the moe::build_expert_ffn[_table]_res
//                            shapes cannot express (nemotronomni: relu^2
//                            non-gated experts + dense shared expert, with the
//                            standard path's exact reduction order)
//   InitSpec::tensors_per_expert / InitSpec::budget_is_hard - cache side of
//                            the above; budget_is_hard keeps the slot count
//                            strictly budget-derived (the default raises it to
//                            layers*K*2, which a 12 GB card cannot host next
//                            to a 30B F16 model's dense weights)
//
// Hook call points inside decode_step (all optional, nullptr = skip):
//   attn_only(il)          - true: skip the MoE for this layer (isolation
//                            debugging; hid = the residual row)
//   trace(il, phase)       - phase 0 post-readback, 1 acquired, 2 ffn built,
//                            3 allocated, 4 inputs set
//   post_a(il, ...)        - after graph-A readback (trajectory capture)
//   post_b(il, hid)        - after graph-B hidden readback (traj h24, debug)
//   layer_check(il, hid_io)- xchk-style cross-check; may overwrite hid with a
//                            reference value (keye KEYE_EC_REF isolation)
#pragma once

#include "expert_cache.hpp"
#include "moe_compute.hpp"
#include "attention.hpp"
#include "step_graph.hpp"


#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace vqa {
namespace moe_decode {

// ---------------------------------------------------------------- graph-A IO
struct AIO {
    ggml_tensor * tok_in = nullptr;   // I32 [1] (first layer only)
    ggml_tensor * hid_in = nullptr;   // F32 [D] (layers > 0)
    ggml_tensor * m_o = nullptr;      // ffn_norm output row -> graph B input
    ggml_tensor * x_o = nullptr;      // residual row at the layer's entry
    ggml_tensor * sel_o = nullptr;    // I32 [K] top-k expert ids
    ggml_tensor * w8_o = nullptr;     // F32 [K] renormalized softmax weights
    void * model_io = nullptr;        // model-specific leaves (mask/pos/...)
};

struct Hooks {
    int n_layer = 0, n_embd = 0, n_expert_used = 0, n_vocab = 0;
    float rms_eps = 1e-6f;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_tensor * out_norm = nullptr;    // graph C weights
    ggml_tensor * out_w = nullptr;
    std::vector<vqa::ExpertCache> * caches = nullptr;
    const std::vector<int> * group = nullptr;   // layer -> layout group
    bool async_ok = false;
    const char * log_tag = "ec";         // "keye ec" / "qomni ec"
    const char * pf_env = nullptr;       // "KEYE_EC_PF" / "QOMNI_EC_PF" (0 = off)
    const char * spec_env = nullptr;     // ..._EC_SPEC (0 = no speculation)
    // per-Hooks one-time env caches (set on the first decode_step)
    mutable const char * pf_off_cache = nullptr;
    mutable int spec_cache = 1;
    mutable bool env_cached = false;

    bool (*build_a)(void * model, int il, vqa::StepGraph & sg, AIO & io);
    bool (*feed_a)(void * model, int il, AIO & io, vqa::StepGraph & sg,
                   const void * step_ctx, int cur_tok, int n_past,
                   const float * hid);
    void (*post_a)(void * model, int il, const float * xmid, const float * m,
                   const int * sel, const float * w8) = nullptr;
    void (*post_b)(void * model, int il, const float * hid) = nullptr;
    bool (*attn_only)(void * model, int il) = nullptr;
    void (*trace)(void * model, int il, int phase) = nullptr;
    bool (*layer_check)(void * model, int il, float * hid_io) = nullptr;
    // ---- third-consumer extensions (nullptr / 0 = legacy semantics) --------
    // driver layer -> MoE-slot index (caches/group space); nullptr = il maps
    // 1:1 (uniform-MoE stacks). -1: the layer has no routed experts - its
    // graph A computes the entire layer, decode_step skips the MoE half and
    // attn_only(il) must return true for it (hid = the residual row).
    const int * moe_of = nullptr;
    int tensors_per_expert = 3;        // slot tensors per selected expert
    // model-specific graph B: receives the per-layer scratch ctx + graph, the
    // normed row (moe_in), the residual row (xmid_in), the acquired slot
    // tensors ([K][tensors_per_expert], spec order) and the [K, T] F32 weight
    // input; returns the final tensor to read back into hid (nullptr = fail).
    // When set, the driver's default build (gate/up/down vectors + slot-table
    // ids) is skipped entirely - the model owns the MoE math, including any
    // cache table()/slot_index() use (slot ids stay valid: no further acquire
    // happens before this call). Input leaves the hook creates must NOT be
    // written here (no buffer yet before the alloc); stash them and upload in
    // feed_b.
    ggml_tensor * (*build_b)(void * model, int il, ggml_context * cb,
                             ggml_cgraph * fb, ggml_tensor * moe_in,
                             ggml_tensor * xmid_in,
                             const std::vector<ggml_tensor *> & slots,
                             ggml_tensor * w_in) = nullptr;
    // optional: called right after the graph-B allocation, before the compute
    // - the point where graph-input bytes can be uploaded (the driver feeds
    // its own moe_in/xmid_in/w_in the same way; build_b stashes any input
    // leaves it created on the model side keyed by il).
    void (*feed_b)(void * model, int il) = nullptr;
};

struct State {
    std::vector<vqa::StepGraph> ga;
    std::vector<AIO> ga_io;
    std::vector<std::vector<int>> prev_sel;
    int S = 0;
    bool ga_ready = false;
    double ms_a = 0, ms_b = 0, ms_swap = 0, ms_rb = 0, ms_fin = 0;
    uint64_t steps = 0;
};

inline double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t).count();
}

// ------------------------------------------------------------------ ec_init
// Groups layers by their (type, shape) expert-tensor signature and builds one
// ExpertCache per group. `tensor(l, t)` returns layer l's expert tensor
// (t: 0 up, 1 gate, 2 down). `cpu_budget` > 0 selects host-resident validation
// mode (whole model already in the host mmap); otherwise this is the GPU
// split placement and the budget comes from free VRAM minus `vram_reserve`.
// Returns false only on a hard init error; `enabled` is set false (and a
// reason logged) when the mode cannot run, leaving the caller to fall back.
struct InitSpec {
    ggml_backend_t backend = nullptr;
    int n_layer = 0, n_expert = 0, n_expert_used = 0;
    const ggml_tensor * (*tensor)(int layer, int t);   // 0 up / 1 gate / 2 down
    size_t cpu_budget = 0;        // > 0: host-resident validation mode
    size_t vram_reserve = (size_t)2.0e9;
    bool stage_threads_16 = false;   // default EC_STAGE_THREADS=16 (qwen3omni)
    const char * log_tag = "ec";
    bool & enabled;                // caller's flag; cleared when unusable
    std::vector<vqa::ExpertCache> & caches;
    std::vector<int> & group;      // layer -> group
    int & n_groups;                // out
    bool & async_ok;               // out
    // ---- third-consumer extensions (defaults = legacy behavior) ------------
    int tensors_per_expert = 3;    // expert = 2 tensors for non-gated FFNs
    bool budget_is_hard = false;   // slots strictly from the budget: the legacy
                                   // path raises n_want to layers*K*2 (276
                                   // x 20 MB slots will not fit a 12 GB card
                                   // beside a 30B F16 dense trunk)
};

inline bool init_grouped(InitSpec s) {
    const bool cpu_mode = s.cpu_budget > 0;
    size_t budget = 0;
    if (!cpu_mode) {
#ifdef GGML_USE_CUDA
        size_t free_vram = 0, total_vram = 0;
        ggml_backend_cuda_get_device_memory(0, &free_vram, &total_vram);
        budget = free_vram > s.vram_reserve ? free_vram - s.vram_reserve : 0;
        if (budget == 0) {
            fprintf(stderr, "%s: < %.1f GB free VRAM -> disabled\n",
                    s.log_tag, s.vram_reserve / 1e9);
            s.enabled = false;
            return true;
        }
        fprintf(stderr, "%s: free VRAM %.2f GB, slot budget %.2f GB\n",
                s.log_tag, free_vram / 1e9, budget / 1e9);
#else
        (void)budget;
        fprintf(stderr, "%s: not a CUDA build -> disabled\n", s.log_tag);
        s.enabled = false;
        return true;
#endif
    } else {
        budget = s.cpu_budget;
        fprintf(stderr, "%s: CPU validation mode (host-resident slots,"
                " budget %.1f GB)\n", s.log_tag, budget / 1e9);
    }
    if (s.stage_threads_16 && !getenv("EC_STAGE_THREADS")) {
#ifdef _WIN32
        _putenv_s("EC_STAGE_THREADS", "16");
#else
        setenv("EC_STAGE_THREADS", "16", 0);
#endif
    }
    // group layers by their expert-tensor (type, shape) signature
    const int NTE = s.tensors_per_expert > 0 ? s.tensors_per_expert : 3;
    std::vector<int> grp((size_t)s.n_layer, 0);
    int n_grp = 0;
    for (int l = 0; l < s.n_layer; l++) {
        std::vector<const ggml_tensor *> ex(NTE);
        for (int t = 0; t < NTE; t++) ex[(size_t)t] = s.tensor(l, t);
        int g = -1;
        for (int l2 = 0; l2 < l; l2++) {
            bool same = true;
            for (int t = 0; t < NTE; t++) {
                const ggml_tensor * ey = s.tensor(l2, t);
                if (ex[(size_t)t]->type != ey->type
                    || ex[(size_t)t]->ne[0] != ey->ne[0]
                    || ex[(size_t)t]->ne[1] != ey->ne[1]) { same = false; break; }
            }
            if (same) { g = grp[(size_t)l2]; break; }
        }
        if (g < 0) g = n_grp++;
        grp[(size_t)l] = g;
    }
    s.group = grp;
    s.n_groups = n_grp;
    s.caches.assign((size_t)n_grp, vqa::ExpertCache());
    int total_slots = 0;
    for (int g = 0; g < n_grp; g++) {
        int first = -1, layers = 0;
        for (int l = 0; l < s.n_layer; l++)
            if (grp[(size_t)l] == g) { ++layers; if (first < 0) first = l; }
        vqa::ExpertCacheSpec spec;
        spec.n_layers = s.n_layer;
        spec.n_experts = s.n_expert;
        spec.tensors_per_expert = NTE;
        spec.layer_base.assign((size_t)s.n_layer,
                               std::vector<const uint8_t *>(NTE, nullptr));
        spec.desc.resize((size_t)NTE);
        size_t slot_bytes = 0, est_slot = 0;
        for (int t = 0; t < NTE; t++) {
            const ggml_tensor * ex0 = s.tensor(first, t);
            spec.desc[(size_t)t].ne[0] = ex0->ne[0];
            spec.desc[(size_t)t].ne[1] = ex0->ne[1];
            spec.desc[(size_t)t].ne[2] = 1;
            spec.desc[(size_t)t].ne[3] = 1;
            spec.desc[(size_t)t].type = ex0->type;
            spec.desc[(size_t)t].host_stride = ggml_nbytes((ggml_tensor *)ex0) / ex0->ne[2];
            slot_bytes += spec.desc[(size_t)t].host_stride;
            // the cache's internal slot estimate (elements x type size)
            // overcounts block-quants; rescaled via n_want below
            est_slot += (size_t)ex0->ne[0] * ex0->ne[1]
                      * ggml_type_size(ex0->type);
        }
        for (int l = 0; l < s.n_layer; l++) {
            if (grp[(size_t)l] != g) continue;
            for (int t = 0; t < NTE; t++)
                spec.layer_base[(size_t)l][(size_t)t] =
                    (const uint8_t *) s.tensor(l, t)->data;
        }
        const size_t gbudget = budget * (size_t)layers / (size_t)s.n_layer;
        if (gbudget < (size_t)8 * slot_bytes) {
            fprintf(stderr, "%s: group %d budget %.2f GB < 8 slots x %.1f MB"
                    " -> disabled\n", s.log_tag, g, gbudget / 1e9, slot_bytes / 1e6);
            s.enabled = false;
            return true;
        }
        int n_want = (int)(gbudget / slot_bytes);
        if (n_want > layers * s.n_expert) n_want = layers * s.n_expert;
        const int n_min = layers * s.n_expert_used * 2;   // a step pins 2x K
        if (!s.budget_is_hard && n_want < n_min)
            n_want = std::min(layers * s.n_expert, n_min);
        if (!s.caches[(size_t)g].init(s.backend, spec, (size_t)n_want * est_slot)) {
            fprintf(stderr, "%s: ExpertCache init failed (group %d) -> falling back\n",
                    s.log_tag, g);
            s.enabled = false;
            return true;
        }
        total_slots += s.caches[(size_t)g].stats().n_slots;
        fprintf(stderr, "%s: group %d (%d layers, slot %.2f MB) ready\n",
                s.log_tag, g, layers, slot_bytes / 1e6);
    }
    fprintf(stderr, "%s: ready, %d slots total\n", s.log_tag, total_slots);
    if (total_slots < s.n_layer * s.n_expert_used) {
        fprintf(stderr, "%s: WARNING %d slots < %d experts per decode"
                " step, expect swapping\n", s.log_tag, total_slots,
                s.n_layer * s.n_expert_used);
    }
    s.async_ok = true;
    for (auto & c : s.caches)
        if (!c.enable_async()) s.async_ok = false;
    fprintf(stderr, "%s: async prefetch %s\n", s.log_tag,
            s.async_ok ? "enabled" : "unavailable -> synchronous (v1)");
    return true;
}

// ------------------------------------------------------------- graph builds
inline bool build_graphs(Hooks & h, State & st, void * model) {
    st.ga.assign((size_t)h.n_layer, vqa::StepGraph());
    st.ga_io.assign((size_t)h.n_layer, AIO());
    st.prev_sel.assign((size_t)h.n_layer, std::vector<int>());
    for (int il = 0; il < h.n_layer; il++) {
        // private gallocr per holder (the pool must never be re-reserved by
        // the per-step graph B)
        if (!h.build_a(model, il, st.ga[(size_t)il], st.ga_io[(size_t)il])) {
            fprintf(stderr, "%s: graph A build failed (layer %d)\n", h.log_tag, il);
            return false;
        }
    }
    st.ga_ready = true;
    fprintf(stderr, "%s: %d fixed layer-A step graphs ready (S=%d)\n",
            h.log_tag, h.n_layer, st.S);
    return true;
}

// ---------------------------------------------------------------- decode step
// One autoregressive step. `step_ctx` is passed through to feed_a (qwen3omni:
// the 4-plane mpos row; keye: nullptr - positions derive from n_past).
inline bool decode_step(Hooks & h, State & st, void * model, const void * step_ctx,
                        int cur_tok, int n_past, std::vector<float> & logits) {
    const int D = h.n_embd;
    const int T = 1;
    const int K = h.n_expert_used;

    // KV window block for the EC step. MUST match the model standard decode
    // bucket (decode_graph.hpp decode_bucket: 128 CPU / 1024 GPU): the CPU
    // flash_attn_ext kernel splits the KV range into per-thread chunks whose
    // online-softmax partials are recombined with expf rescaling, so a padded
    // window of a DIFFERENT width is mathematically equivalent but not
    // bit-equal - a 1-ulp attention difference flips router top-k near-ties
    // and the whole reply diverges (qwen3omni xchk drifted at step 16 with
    // the old 512 block; bit-exact with 128). MOE_EC_BLOCK overrides for A/B.
    int ec_block = ggml_backend_is_cpu(h.backend) ? 128 : 1024;
    if (const char * e = getenv("MOE_EC_BLOCK")) ec_block = atoi(e);
    const int S_need = vqa::block_for(n_past + 1, ec_block);
    if (!st.ga_ready || S_need != st.S) {
        if (st.ga_ready) {
            fprintf(stderr, "%s: KV window crossed a %d block (%d -> %d),"
                    " rebuilding %d layer-A graphs\n", h.log_tag, ec_block, st.S, S_need,
                    h.n_layer);
            st.ga_ready = false;
        }
        st.S = S_need;
        if (!build_graphs(h, st, model)) return false;
    }
    const int S = st.S;

    for (auto & c : *h.caches) c.begin_step();
    std::vector<float> hid((size_t)D);
    std::vector<float> m_host((size_t)D), xmid_host((size_t)D);
    int sel_host[16];
    float w8_host[16];

    const auto t0 = std::chrono::steady_clock::now();
    for (int il = 0; il < h.n_layer; il++) {
        vqa::StepGraph & sg = st.ga[(size_t)il];
        AIO & io = st.ga_io[(size_t)il];
        // non-uniform stacks: moe_of maps the driver layer to its MoE slot
        // (caches/group space); -1 = the layer carries no routed experts and
        // graph A already computed the whole layer (incl. Mamba2 state writes)
        const int moe_idx = h.moe_of ? h.moe_of[il] : il;
        // skip evaluated BEFORE prefetch so a skipped layer's garbage router
        // readback never warms the cache (legacy attn_only debug isolation
        // moves with it - results unchanged, only cache warmth differs)
        const bool skip_moe = moe_idx < 0
                           || (h.attn_only && h.attn_only(model, il));

        // ---------------- graph A: fixed StepGraph, re-fed ----------------
        const auto ta0 = std::chrono::steady_clock::now();
        if (!h.feed_a(model, il, io, sg, step_ctx, cur_tok, n_past, hid.data()))
            return false;
        if (il == 0) {
            if (!sg.set_input(io.tok_in, &cur_tok, sizeof(int))) return false;
        } else {
            if (!sg.set_input(io.hid_in, hid.data(), sizeof(float) * D)) return false;
        }
        if (!sg.run()) {
            fprintf(stderr, "%s: graph A compute failed (layer %d)\n", h.log_tag, il);
            return false;
        }
        st.ms_a += ms_since(ta0);
        const auto trb0 = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(io.m_o, m_host.data(), 0, sizeof(float) * D);
        ggml_backend_tensor_get(io.x_o, xmid_host.data(), 0, sizeof(float) * D);
        ggml_backend_tensor_get(io.sel_o, sel_host, 0, sizeof(int) * K);
        ggml_backend_tensor_get(io.w8_o, w8_host, 0, sizeof(float) * K);
        st.ms_rb += ms_since(trb0);
        if (h.trace) h.trace(model, il, 0);
        if (h.post_a) h.post_a(model, il, xmid_host.data(), m_host.data(),
                               sel_host, w8_host);

        // prefetch: this layer's fresh selections (the worker pipelines the
        // copies while acquire/graph-B-build run) plus ONE-LAYER-AHEAD
        // speculation (hot experts repeat across steps). No-op without
        // enable_async. pf_env=0 disables every call, spec_env=0 the
        // speculation half.
        if (!h.env_cached) {   // one-time lookup (per-Hooks, not global)
            h.pf_off_cache = h.pf_env ? getenv(h.pf_env) : nullptr;
            h.spec_cache = h.spec_env && getenv(h.spec_env)
                         ? atoi(getenv(h.spec_env)) : 1;
            h.env_cached = true;
        }
        if (!skip_moe && h.async_ok && h.pf_off_cache == nullptr) {
            const int spec = h.spec_cache;
            std::vector<std::pair<int, int>> pf;
            for (int ke = 0; ke < K; ke++) pf.push_back({ moe_idx, sel_host[ke] });
            const int next_moe = h.moe_of && il + 1 < h.n_layer
                               ? h.moe_of[il + 1] : il + 1;
            const bool has_next = spec && next_moe >= 0
                                && !st.prev_sel[(size_t)il + 1].empty();
            const bool next_same_grp = has_next
                && (*h.group)[(size_t)next_moe] == (*h.group)[(size_t)moe_idx];
            if (next_same_grp)
                for (int e : st.prev_sel[(size_t)il + 1]) pf.push_back({ next_moe, e });
            (*h.caches)[(size_t)(*h.group)[(size_t)moe_idx]].prefetch(pf);
            // next-layer pairs may belong to the other layout group's cache
            if (has_next && !next_same_grp) {
                std::vector<std::pair<int, int>> pf2;
                for (int e : st.prev_sel[(size_t)il + 1]) pf2.push_back({ next_moe, e });
                (*h.caches)[(size_t)(*h.group)[(size_t)next_moe]].prefetch(pf2);
            }
        }

        if (skip_moe) {
            hid = xmid_host;   // whole layer already in graph A / isolation
            continue;
        }

        // ---------------- graph B: the K slot experts ----------------
        const auto tsw0 = std::chrono::steady_clock::now();
        const int NTE = h.tensors_per_expert > 0 ? h.tensors_per_expert : 3;
        std::vector<ggml_tensor *> slots((size_t)NTE * (size_t)K);
        std::vector<ggml_tensor *> gate(K), up(K), down(K);
        vqa::ExpertCache & ec_c = (*h.caches)[(size_t)(*h.group)[(size_t)moe_idx]];
        bool ec_tbl = false;
        if (!h.build_b) {
            // legacy path owns the slot-table pairing (3 gate/up/down tensors)
            ec_tbl = ec_c.table(0) && ec_c.table(1) && ec_c.table(2);
        }
        int ids_gu_host[16] = {}, ids_dn_host[16] = {};
        for (int ke = 0; ke < K; ke++) {
            std::vector<ggml_tensor *> sl = ec_c.acquire(moe_idx, sel_host[ke]);
            if ((int)sl.size() != NTE) {
                fprintf(stderr, "%s: acquire failed (layer %d expert %d)\n",
                        h.log_tag, il, sel_host[ke]);
                return false;
            }
            for (int t = 0; t < NTE; t++)
                slots[(size_t)ke * NTE + (size_t)t] = sl[(size_t)t];
            if (!h.build_b) {   // spec order: 0 up / 1 gate / 2 down
                gate[(size_t)ke] = sl[1];
                up[(size_t)ke]   = sl[0];
                down[(size_t)ke] = sl[2];
                ids_gu_host[ke] = ec_c.slot_index(sl[1]);   // gate/up ids
                ids_dn_host[ke] = ec_c.slot_index(sl[2]);   // down ids
                if (ids_gu_host[ke] < 0 || ids_dn_host[ke] < 0) ec_tbl = false;
            }
        }
        st.ms_swap += ms_since(tsw0);
        st.prev_sel[(size_t)il].assign(sel_host, sel_host + K);
        if (h.trace) h.trace(model, il, 1);

        ggml_context * cb = ggml_init({ 16ull << 20, NULL, true });
        if (!cb) return false;
        ggml_cgraph * fb = ggml_new_graph_custom(cb, 8192, false);
        ggml_tensor * moe_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, D, T);
        ggml_tensor * xmid_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, D, T);
        ggml_tensor * w_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, K, T);

        ggml_tensor * hout;
        ggml_tensor * ids_gu_in = nullptr, * ids_dn_in = nullptr;
        if (h.build_b) {
            // model-specific MoE math (non-gated / shared-expert shapes);
            // the ids readback contract: slot ids are read inside the hook,
            // before any further acquire touches the cache
            hout = h.build_b(model, il, cb, fb, moe_in, xmid_in, slots, w_in);
            if (!hout) {
                fprintf(stderr, "%s: build_b failed (layer %d)\n", h.log_tag, il);
                ggml_free(cb);
                return false;
            }
        } else {
            if (ec_tbl) {
                ids_gu_in = ggml_new_tensor_2d(cb, GGML_TYPE_I32, K, T);
                ids_dn_in = ggml_new_tensor_2d(cb, GGML_TYPE_I32, K, T);
                hout = vqa::moe::build_expert_ffn_table_res(cb, fb, moe_in, xmid_in,
                        ec_c.table(1), ec_c.table(0), ec_c.table(2),
                        ids_gu_in, ids_dn_in, w_in, GGML_UNARY_OP_SILU);
            } else {
                hout = vqa::moe::build_expert_ffn_res(cb, fb, moe_in, xmid_in,
                                                      gate, up, down, w_in,
                                                      GGML_UNARY_OP_SILU);
            }
            if (!hout) {
                fprintf(stderr, "%s: build_expert_ffn failed (layer %d)\n", h.log_tag, il);
                ggml_free(cb);
                return false;
            }
        }
        if (h.trace) h.trace(model, il, 2);
        const auto tb0 = std::chrono::steady_clock::now();
        if (!ggml_gallocr_alloc_graph(h.allocr, fb)) {
            fprintf(stderr, "%s: graph B alloc failed (layer %d)\n", h.log_tag, il);
            ggml_free(cb);
            return false;
        }
        if (h.trace) h.trace(model, il, 3);
        ggml_backend_tensor_set(moe_in, m_host.data(), 0, sizeof(float) * D);
        ggml_backend_tensor_set(xmid_in, xmid_host.data(), 0, sizeof(float) * D);
        ggml_backend_tensor_set(w_in, w8_host, 0, sizeof(float) * K);
        if (ids_gu_in) {   // slot ids are valid only until the next acquire
            ggml_backend_tensor_set(ids_gu_in, ids_gu_host, 0, sizeof(int) * K);
            ggml_backend_tensor_set(ids_dn_in, ids_dn_host, 0, sizeof(int) * K);
        }
        if (h.feed_b) h.feed_b(model, il);
        if (h.trace) h.trace(model, il, 4);
        if (ggml_backend_graph_compute(h.backend, fb) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: graph B compute failed (layer %d)\n", h.log_tag, il);
            ggml_free(cb);
            return false;
        }
        st.ms_b += ms_since(tb0);
        const auto trb1 = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(hout, hid.data(), 0, sizeof(float) * D);
        st.ms_rb += ms_since(trb1);
        ggml_free(cb);
        if (h.layer_check && !h.layer_check(model, il, hid.data())) {
            // hook may have overwritten hid with a reference value
        }
        if (h.post_b) h.post_b(model, il, hid.data());
    }

    // ---------------- final norm + output head ----------------
    {
        ggml_context * cc = ggml_init({ 8ull << 20, NULL, true });
        if (!cc) return false;
        ggml_cgraph * fc = ggml_new_graph_custom(cc, 1024, false);
        ggml_tensor * h_in = ggml_new_tensor_2d(cc, GGML_TYPE_F32, D, T);
        ggml_tensor * fn = ggml_rms_norm(cc, h_in, h.rms_eps);
        fn = ggml_mul(cc, fn, h.out_norm);
        ggml_tensor * lg = ggml_mul_mat(cc, h.out_w, fn);
        ggml_build_forward_expand(fc, lg);
        if (!ggml_gallocr_alloc_graph(h.allocr, fc)) { ggml_free(cc); return false; }
        ggml_backend_tensor_set(h_in, hid.data(), 0, sizeof(float) * D);
        if (ggml_backend_graph_compute(h.backend, fc) != GGML_STATUS_SUCCESS) {
            ggml_free(cc);
            return false;
        }
        logits.resize((size_t)h.n_vocab);
        ggml_backend_tensor_get(lg, logits.data(), 0, sizeof(float) * logits.size());
        ggml_free(cc);
    }
    st.ms_fin += ms_since(t0);
    st.steps++;
    return true;
}

}  // namespace moe_decode
}  // namespace vqa
