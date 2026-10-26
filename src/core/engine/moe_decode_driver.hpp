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
    std::vector<int> & group;      // out: layer -> group
    int & n_groups;                // out
    bool & async_ok;               // out
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
    std::vector<int> grp((size_t)s.n_layer, 0);
    int n_grp = 0;
    for (int l = 0; l < s.n_layer; l++) {
        const ggml_tensor * ex[3] = { s.tensor(l, 0), s.tensor(l, 1), s.tensor(l, 2) };
        int g = -1;
        for (int l2 = 0; l2 < l; l2++) {
            const ggml_tensor * ey[3] = { s.tensor(l2, 0), s.tensor(l2, 1), s.tensor(l2, 2) };
            bool same = true;
            for (int t = 0; t < 3; t++)
                if (ex[t]->type != ey[t]->type || ex[t]->ne[0] != ey[t]->ne[0]
                    || ex[t]->ne[1] != ey[t]->ne[1]) { same = false; break; }
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
        const ggml_tensor * ex0[3] = { s.tensor(first, 0), s.tensor(first, 1), s.tensor(first, 2) };
        vqa::ExpertCacheSpec spec;
        spec.n_layers = s.n_layer;
        spec.n_experts = s.n_expert;
        spec.tensors_per_expert = 3;
        spec.layer_base.assign((size_t)s.n_layer, std::vector<const uint8_t *>(3, nullptr));
        spec.desc.resize(3);
        size_t slot_bytes = 0, est_slot = 0;
        for (int t = 0; t < 3; t++) {
            spec.desc[(size_t)t].ne[0] = ex0[t]->ne[0];
            spec.desc[(size_t)t].ne[1] = ex0[t]->ne[1];
            spec.desc[(size_t)t].ne[2] = 1;
            spec.desc[(size_t)t].ne[3] = 1;
            spec.desc[(size_t)t].type = ex0[t]->type;
            spec.desc[(size_t)t].host_stride = ggml_nbytes((ggml_tensor *)ex0[t]) / ex0[t]->ne[2];
            slot_bytes += spec.desc[(size_t)t].host_stride;
            // the cache's internal slot estimate (elements x type size)
            // overcounts block-quants; rescaled via n_want below
            est_slot += (size_t)ex0[t]->ne[0] * ex0[t]->ne[1]
                      * ggml_type_size(ex0[t]->type);
        }
        for (int l = 0; l < s.n_layer; l++) {
            if (grp[(size_t)l] != g) continue;
            for (int t = 0; t < 3; t++)
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
        if (n_want < n_min) n_want = std::min(layers * s.n_expert, n_min);
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

    // KV window rounded to a 512 block so the graph shape only changes every
    // 512 positions (mask hides the padding); the per-layer graphs rebuild
    // once per block, not per step
    const int S_need = vqa::block_for(n_past + 1, 512);
    if (!st.ga_ready || S_need != st.S) {
        if (st.ga_ready) {
            fprintf(stderr, "%s: KV window crossed a 512 block (%d -> %d),"
                    " rebuilding %d layer-A graphs\n", h.log_tag, st.S, S_need,
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
        if (h.async_ok && (h.pf_env == nullptr || getenv(h.pf_env) == nullptr)) {
            const int spec = h.spec_env && getenv(h.spec_env)
                           ? atoi(getenv(h.spec_env)) : 1;
            std::vector<std::pair<int, int>> pf;
            for (int ke = 0; ke < K; ke++) pf.push_back({ il, sel_host[ke] });
            const bool has_next = spec && il + 1 < h.n_layer
                                && !st.prev_sel[(size_t)il + 1].empty();
            const bool next_same_grp = has_next
                && (*h.group)[(size_t)il + 1] == (*h.group)[(size_t)il];
            if (next_same_grp)
                for (int e : st.prev_sel[(size_t)il + 1]) pf.push_back({ il + 1, e });
            (*h.caches)[(size_t)(*h.group)[(size_t)il]].prefetch(pf);
            // next-layer pairs may belong to the other layout group's cache
            if (has_next && !next_same_grp) {
                std::vector<std::pair<int, int>> pf2;
                for (int e : st.prev_sel[(size_t)il + 1]) pf2.push_back({ il + 1, e });
                (*h.caches)[(size_t)(*h.group)[(size_t)il + 1]].prefetch(pf2);
            }
        }

        if (h.attn_only && h.attn_only(model, il)) {
            hid = xmid_host;   // isolation: skip the MoE, pass the residual
            continue;
        }

        // ---------------- graph B: the K slot experts ----------------
        const auto tsw0 = std::chrono::steady_clock::now();
        std::vector<ggml_tensor *> gate(K), up(K), down(K);
        vqa::ExpertCache & ec_c = (*h.caches)[(size_t)(*h.group)[(size_t)il]];
        // v3: with the table layout the K experts go through 3 mul_mat_id
        // launches over the slot tables, and gate/up + GLU hit the CUDA
        // fused pass
        const bool ec_tbl_avail = ec_c.table(0) && ec_c.table(1) && ec_c.table(2);
        int ids_gu_host[16] = {}, ids_dn_host[16] = {};
        bool ec_tbl = ec_tbl_avail;
        for (int ke = 0; ke < K; ke++) {
            std::vector<ggml_tensor *> sl = ec_c.acquire(il, sel_host[ke]);
            if (sl.size() != 3) {
                fprintf(stderr, "%s: acquire failed (layer %d expert %d)\n",
                        h.log_tag, il, sel_host[ke]);
                return false;
            }
            gate[(size_t)ke] = sl[1];
            up[(size_t)ke]   = sl[0];
            down[(size_t)ke] = sl[2];
            ids_gu_host[ke] = ec_c.slot_index(sl[1]);   // gate/up ids
            ids_dn_host[ke] = ec_c.slot_index(sl[2]);   // down ids
            if (ids_gu_host[ke] < 0 || ids_dn_host[ke] < 0) ec_tbl = false;
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

        ggml_tensor * ids_gu_in = nullptr, * ids_dn_in = nullptr;
        ggml_tensor * hout;
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
        if (ec_tbl) {   // slot ids are valid only until the next acquire
            ggml_backend_tensor_set(ids_gu_in, ids_gu_host, 0, sizeof(int) * K);
            ggml_backend_tensor_set(ids_dn_in, ids_dn_host, 0, sizeof(int) * K);
        }
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
