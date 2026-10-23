// step_graph.hpp - build-once/run-many holder for fixed-shape decode graphs.
//
// The dynamic decode paths build many small per-layer graphs each token; the
// expensive part is not the compute but the graph build + allocation. When a
// graph's shapes are identical every step (decode with a block-aligned KV
// window, like attention.hpp's block_for), the whole graph - including its
// input leaves - can be built once and re-fed:
//
//   StepGraph sg;
//   sg.build(backend, [](ggml_context * c, void * ud) -> ggml_tensor * {
//       // create input leaves (op NONE) + compute; return the output;
//       // stash input tensor pointers somewhere reachable from ud
//   }, ud);
//   sg.set_input(name_or_ptr, data, bytes);   // uploads into the leaf
//   sg.run();                                  // graph_compute
//   ggml_backend_tensor_get(sg.out(), ...);    // read results
//
// CONTRACT:
//   - build() allocates via the caller's gallocr-free path: it creates its
//     own ggml_gallocr on the backend the FIRST time and reuses it; graphs
//     must keep identical shapes/nodes across run() calls (set_input only
//     uploads bytes, never resizes).
//   - input leaves are found by NAME (set_input takes the tensor pointer
//     directly - the caller kept it from the builder).
//   - NOT thread-safe; one holder per graph; graphs are small.
#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>

namespace vqa {

class StepGraph {
public:
    ~StepGraph();

    // builder returns the graph's sink tensor (nullptr = fail); it must also
    // create every input leaf in the same context. ud is passed through.
    // allocr: optional SHARED gallocr (recommended: one per backend, passed
    // to every layer's StepGraph - same-shape graphs reuse its buffers);
    // nullptr = this holder allocates and owns a private one.
    bool build(ggml_backend_t backend, ggml_tensor * (*builder)(ggml_context *, void *), void * ud,
               ggml_gallocr_t shared_allocr = nullptr);

    // upload into a leaf created by the builder (caller holds the pointer)
    bool set_input(ggml_tensor * leaf, const void * data, size_t bytes);

    bool run();

    ggml_tensor * out() const { return out_; }
    ggml_context * ctx() const { return ctx_; }

    void free_graph();

private:
    ggml_context * ctx_ = nullptr;
    ggml_cgraph * gf_ = nullptr;
    ggml_gallocr_t allocr_ = nullptr;    // shared (not owned) or private (owned)
    bool owns_allocr_ = false;
    ggml_backend_t backend_ = nullptr;
    ggml_tensor * out_ = nullptr;
};

}  // namespace vqa
