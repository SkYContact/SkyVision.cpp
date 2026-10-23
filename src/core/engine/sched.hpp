// sched.hpp - two-tier graph compute for split-placed (Strata-style) models.
//
// Companion to weights::split_bind_to_backends: the dense weights live in a
// device buffer, the MoE expert tensors in a host mmap buffer, both tagged
// GGML_BACKEND_BUFFER_USAGE_WEIGHTS. ggml_backend_sched then assigns every
// node to the backend its weights live on (attention/rope/FA/norm on the GPU,
// the expert mul_mat_ids on the CPU) and inserts the activation copies at
// the split points. op_offload is disabled on purpose - the whole point is
// that the experts STAY on the host.
//
// Usage per compute step (graphs are rebuilt per step, the llama.cpp decode
// pattern): build the graph in a no_alloc ctx as usual, then call compute();
// sched reset + alloc_graph + compute runs it. Graph inputs are assigned to
// the device backend (backend 0).
#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>

namespace vqa {

struct SplitSched {
    ggml_backend_t dev = nullptr;
    ggml_backend_t host = nullptr;
    ggml_backend_sched_t sched = nullptr;

    bool init(ggml_backend_t dev_, ggml_backend_t host_, size_t graph_nodes = 16384,
              bool pipeline = true) {
        dev = dev_;
        host = host_;
        ggml_backend_t backends[2] = { dev, host };
        // pipeline=true: the CPU expert segments of layer N overlap the GPU
        // attention segments of layer N+1 (events-based, needs backend events)
        sched = ggml_backend_sched_new(backends, nullptr, 2, graph_nodes,
                                       /*parallel*/ pipeline, /*op_offload*/ false);
        if (!sched) fprintf(stderr, "split sched init failed\n");
        return sched != nullptr;
    }

    // step 1 of 2: allocate the graph. Unassigned weight tensors drive the
    // per-node backend choice (see header note); graph input leaves default
    // to the device (backend 0). tensor_set on graph inputs is only valid
    // after this.
    bool alloc(struct ggml_cgraph * gf) {
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            const int n = ggml_graph_n_nodes(gf);
            for (int i = 0; i < n; i++) {
                struct ggml_tensor * node = ggml_graph_node(gf, i);
                if (node && ggml_backend_sched_get_tensor_backend(sched, node) == nullptr)
                    fprintf(stderr, "split sched: node %d (%s %s) has no backend\n",
                            i, ggml_op_desc(node), node->name);
            }
            fprintf(stderr, "split sched alloc failed\n");
            return false;
        }
        return true;
    }

    // step 2 of 2: execute the allocated graph
    bool run(struct ggml_cgraph * gf) {
        return ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    }

    // convenience one-shot for graphs without inputs
    bool compute(struct ggml_cgraph * gf) {
        return alloc(gf) && run(gf);
    }

    void free_sched() {
        if (sched) { ggml_backend_sched_free(sched); sched = nullptr; }
    }
};

}  // namespace vqa
