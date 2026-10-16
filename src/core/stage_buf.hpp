// stage_buf.hpp - a persistent on-device staging tensor shared ACROSS graphs.
//
// The KV-cache pattern generalized: a [dim, max_rows] F32 tensor on the
// compute backend that producer graphs write into slice-by-slice (a trailing
// ggml_cpy sink, exactly like the recurrent-state write-back) and consumer
// graphs read back as views. The data never touches the host, which matters
// for video towers executed chunk-by-chunk (per temporal step / per clip):
// the old pattern pulled every chunk to the host and fed it back into the
// merger graph (2 transfers + one copy per chunk, both CPU and CUDA pay).
//
// gallocr treats views whose ->data already points into this buffer like any
// other pre-existing input (the KV cache relies on the same property), so no
// per-graph allocation happens here after alloc().
#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>

namespace vqa {

struct StageBuffer {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * t = nullptr;      // [dim, max_rows] F32
    int64_t dim = 0, max_rows = 0;

    // Grows on demand (reallocates when a longer video needs more rows);
    // returns false on allocation failure.
    bool ensure(ggml_backend_t backend, int64_t dim_, int64_t rows) {
        if (t && dim == dim_ && rows <= max_rows) return true;
        free();
        dim = dim_;
        max_rows = rows;
        struct ggml_init_params ip = { 8 * ggml_tensor_overhead(), NULL, true };
        ctx = ggml_init(ip);
        if (!ctx) return false;
        t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, max_rows);
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { fprintf(stderr, "stage buffer allocation failed (%.1f MB)\n",
                            (double)dim * max_rows * 4 / 1e6); free(); return false; }
        ggml_backend_buffer_clear(buf, 0);
        return true;
    }

    // View of `rows` rows starting at `row0`, created inside a graph context.
    ggml_tensor * view(ggml_context * gctx, int64_t rows, int64_t row0) const {
        return ggml_view_2d(gctx, t, dim, rows, t->nb[1], (size_t)row0 * t->nb[1]);
    }

    // Producer side: append `ggml_cpy(x, view(...))` as a trailing sink.
    ggml_tensor * sink(ggml_context * gctx, ggml_tensor * x, int64_t row0) const {
        return ggml_cpy(gctx, x, view(gctx, x->ne[1], row0));
    }

    void free() {
        if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        t = nullptr;
    }

    ~StageBuffer() { free(); }
};

}  // namespace vqa
