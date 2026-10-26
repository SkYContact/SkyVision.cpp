// op_profile.hpp - per-op-segment timing for a ggml compute graph.
//
// Lifted from keye_vl_model.cpp's KEYE_PROFILE machinery (its lesson is baked
// in: splitting at EVERY node measures only the scheduler's per-split dispatch
// overhead - 4400 nodes/step made every op look like ~20 ms of nothing - so we
// split only at the ops that plausibly dominate a decode step, and attribute
// the segment between two splits to the closing op).
//
// Usage: when your env flag is set, route the graph through
// vqa::profile_graph() instead of ggml_backend_graph_compute(); it prints a
// per-op breakdown when QOP_PROFILE_SUMMARY=<n steps> steps have accumulated.
#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace vqa {
namespace op_profile {

struct State {
    std::map<std::string, double> ms_by_op;
    double ms_all = 0.0;
    std::chrono::steady_clock::time_point last;
    bool started = false;
};

inline bool is_split_op(ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_GLU:
        case GGML_OP_ROPE:
        case GGML_OP_IM2COL:
        case GGML_OP_NORM:
            return true;
        default:
            return false;
    }
}

inline bool cb(ggml_tensor * t, bool ask, void * ud) {
    State * st = (State *)ud;
    if (ask) {
        if (!st->started) { st->last = std::chrono::steady_clock::now(); st->started = true; }
        return is_split_op(t);
    }
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - st->last).count();
    st->last = std::chrono::steady_clock::now();
    st->ms_all += ms;
    st->ms_by_op[ggml_op_name(t->op)] += ms;
    return true;
}

// runs `gf` under the profiler and (every `every` calls, or on the last one)
// prints the accumulated per-op table
inline bool profile_graph(ggml_backend_t backend, ggml_cgraph * gf, State & st,
                          const char * label, int step, int every) {
    ggml_backend_sched_t sched =
        ggml_backend_sched_new(&backend, nullptr, 1, (size_t)ggml_graph_n_nodes(gf),
                               /*parallel*/ false, /*op_offload*/ false);
    if (!sched) return false;
    ggml_backend_sched_set_eval_callback(sched, cb, &st);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        ggml_backend_sched_free(sched);
        return false;
    }
    const enum ggml_status ec = ggml_backend_sched_graph_compute(sched, gf);
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_free(sched);
    if (ec != GGML_STATUS_SUCCESS) return false;
    if ((step + 1) % every == 0) {
        fprintf(stderr, "[%s profile] step %d, %.1f ms/step:\n", label, step + 1,
                st.ms_all / (step + 1));
        double rest = st.ms_all;
        for (const auto & e : st.ms_by_op) {
            fprintf(stderr, "  %-18s %8.2f ms (%5.1f%%)\n", e.first.c_str(),
                    e.second / (step + 1), 100.0 * e.second / st.ms_all);
            rest -= e.second;
        }
        fprintf(stderr, "  %-18s %8.2f ms (%5.1f%%)\n", "(other/elementwise)",
                rest / (step + 1), 100.0 * rest / st.ms_all);
        st.ms_by_op.clear();
        st.ms_all = 0.0;
    }
    return true;
}

}  // namespace op_profile
}  // namespace vqa
