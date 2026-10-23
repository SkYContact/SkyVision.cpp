// weight_load.hpp - the "load tail" every backend had copy-pasted:
//   1) point each GGUF tensor's data at its bytes inside the mmap,
//   2) rewrite the 1D tensors (norms / biases / dt vectors) into host F32,
//   3) hand the tensors to the backend: CPU keeps zero-copy pointers into the
//      mmap (and into the host F32 blocks), GPU allocates one device buffer and
//      uploads.
//
// Step 3 is the fiddly one: a tensor whose data no longer points into the mmap
// (because step 2 or a load-time weight fold rewrote it) must get the buffer that
// owns those host bytes, otherwise ggml_backend_tensor_get/set on it is out of
// bounds. That range test is exactly what was duplicated in three backends.
#pragma once

#include "ggml-backend.h"
#include "ggml-cpu.h"   // ggml_backend_cpu_buffer_from_ptr / ggml_backend_is_cpu
#include "ggml.h"
#include "gguf.h"
#include "core/engine/gguf_util.hpp"
#include "core/util/aligned_buf.hpp"
#include "core/util/mmap.hpp"

#include <cstdio>
#include <utility>
#include <vector>

namespace vqa {
namespace weights {

// (1) no_alloc context + mmap: bind every named tensor to its file offset.
inline void bind_from_mmap(struct gguf_context * gu, ggml_context * wctx, const MMap & mm) {
    const size_t data_off = gguf_get_data_offset(gu);
    const int64_t nt = gguf_get_n_tensors(gu);
    for (int64_t i = 0; i < nt; i++) {
        const char * name = gguf_get_tensor_name(gu, i);
        ggml_tensor * t = ggml_get_tensor(wctx, name);
        if (!t) continue;
        t->data = (char *)mm.ptr + data_off + gguf_get_tensor_offset(gu, i);
    }
}

// (2) every 1D tensor that is not already F32 is expanded into `storage`, and the
// tensor is rewritten in place (type/nb included) so kernels see plain f32 rows.
// Quantized 1D tensors otherwise read as garbage in norm/bias positions.
// `extra_names` promotes additional named tensors of any shape (null-
// terminated list, may be null): backends whose graphs use a multi-dim
// weight in elementwise ops need those as plain F32 too (mage_vl's mamba
// conv kernel). Strides are recomputed per dim, which is correct for any
// promoted shape.
inline void promote_1d_to_f32(ggml_context * wctx, AlignedBuf & storage,
                              const char * const * extra_names = nullptr) {
    auto wants = [&](ggml_tensor * t) {
        if (t->type == GGML_TYPE_F32) return false;
        if (t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1) return true;
        if (!extra_names) return false;
        for (const char * const * n = extra_names; *n; ++n)
            if (!strcmp(t->name, *n)) return true;
        return false;
    };
    size_t total = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
        if (!wants(t)) continue;
        total += (size_t)ggml_nelements(t) * sizeof(float);
    }
    if (total == 0) return;
    storage.resize(total);
    size_t off = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
        if (!wants(t)) continue;
        const int64_t n = ggml_nelements(t);
        float * dst = (float *)(storage.data() + off);
        gguf::tensor_to_f32(t, dst);
        t->data = dst;
        t->type = GGML_TYPE_F32;
        t->nb[0] = sizeof(float);
        for (int i = 1; i < 4; i++) t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
        off += (size_t)n * sizeof(float);
    }
    fprintf(stderr, "f32-converted 1D tensors: %.1f MB\n", storage.size() / 1e6);
}

// host bytes that some tensors now point into (mmap aside)
struct HostBlock {
    const void * data = nullptr;
    size_t size = 0;
};

struct Bound {
    ggml_backend_buffer_t main_buf = nullptr;              // mmap zero-copy or device buffer
    std::vector<ggml_backend_buffer_t> host_bufs;          // CPU only, one per host block
    bool ok = false;
};

// (3) assign buffers (CPU) or allocate + upload (GPU). Buffers created here are
// owned by the caller: free main_buf and every entry of host_bufs.
inline Bound bind_to_backend(ggml_context * wctx, ggml_backend_t backend, const MMap & mm,
                             const std::vector<HostBlock> & hosts = {}) {
    Bound out;
    std::vector<std::pair<ggml_tensor *, const void *>> uploads;
    size_t total = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
        uploads.push_back({t, t->data});
        total += ggml_nbytes(t);
    }

    if (ggml_backend_is_cpu(backend)) {
        out.main_buf = ggml_backend_cpu_buffer_from_ptr(mm.ptr, mm.size);
        if (!out.main_buf) { fprintf(stderr, "weight buffer (mmap) failed\n"); return out; }
        for (const HostBlock & h : hosts) {
            if (!h.data || h.size == 0) continue;
            ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr((void *)h.data, h.size);
            if (!b) { fprintf(stderr, "weight buffer (host block) failed\n"); return out; }
            out.host_bufs.push_back(b);
        }
        for (auto & u : uploads) {
            const char * p = (const char *)u.first->data;
            u.first->buffer = out.main_buf;
            for (size_t i = 0; i < hosts.size(); i++) {
                const HostBlock & h = hosts[i];
                if (!h.data || h.size == 0) continue;
                const char * b = (const char *)h.data;
                if (p >= b && p < b + h.size) {
                    // index maps through the non-empty blocks only
                    size_t k = 0;
                    for (size_t j = 0; j < i; j++)
                        if (hosts[j].data && hosts[j].size) k++;
                    u.first->buffer = out.host_bufs[k];
                    break;
                }
            }
        }
        fprintf(stderr, "weights: %.2f GB zero-copy from mmap (%zu host f32 block%s)\n",
                total / 1e9, out.host_bufs.size(), out.host_bufs.size() == 1 ? "" : "s");
    } else {
        for (auto & u : uploads) u.first->data = nullptr;
        out.main_buf = ggml_backend_alloc_ctx_tensors(wctx, backend);
        if (!out.main_buf) {
            fprintf(stderr, "weight allocation failed (need %.2f GB)\n", total / 1e9);
            return out;
        }
        for (auto & u : uploads)
            ggml_backend_tensor_set(u.first, u.second, 0, ggml_nbytes(u.first));
        fprintf(stderr, "uploaded %.2f GB of weights\n", total / 1e9);
    }
    out.ok = true;
    return out;
}

// Host f32 copy of a weight tensor. Needed when a graph builder reads weight
// values on the host (mel filterbanks, mm_proj snapshots) but the tensor now
// lives in a device buffer after bind_to_backend - its ->data is a device
// pointer that must not be dereferenced.
inline void snapshot_to_f32_host(ggml_tensor * t, std::vector<float> & out) {
    out.resize((size_t)ggml_nelements(t));
    gguf::tensor_to_f32(t, out.data());
}

// ---------------------------------------------------------------------------
// (5) split bind for MoE models (Strata-style two-tier placement).
//
// In a MoE the routed expert FFN weights are the bulk of the checkpoint but
// each token touches only n_expert_used of them (8/128 for Keye, 8/64 for
// moondream), so they can stay in host memory (zero-copy mmap, computed by
// the CPU backend) while everything else - attention, norms, embeddings, the
// vision tower - is uploaded to the GPU. This lets a 19 GB 30B-A3B checkpoint
// run on a 12 GB card: GPU keeps ~3 GB of dense weights + KV + compute, the
// experts stream from RAM at ~1 GB per decoded token of traffic.
//
// The node assignment itself is done by ggml_backend_sched with op_offload:
// mul_mats whose src0 lives in a host buffer run on the CPU backend, the rest
// on the device, with automatic activation copies at the split points.

// default predicate: the engine's expert weight naming conventions
// (llama.cpp "ffn_*_exps", moondream's "moe_*_w"). VQA_GPULAYERS=N moves the
// first N layers' experts to the device as well (layer-granular expert
// caching: fewer host↔device split points, faster decode at the cost of VRAM).
inline bool moe_expert_pred(const char * name) {
    if (strstr(name, "_exps") == nullptr && strstr(name, "moe_up_w") == nullptr &&
        strstr(name, "moe_down_w") == nullptr) return false;
    if (const char * gl = getenv("VQA_GPULAYERS")) {
        int n = atoi(gl);
        int layer = -1;
        if (const char * b = strstr(name, "blk.")) layer = atoi(b + 4);
        if (const char * b = strstr(name, ".blk.")) layer = atoi(b + 5);
        if (layer >= 0 && layer < n) return false;   // this layer's experts go on the device
    }
    return true;
}

struct SplitBound {
    ggml_backend_buffer_t dev_buf = nullptr;    // device weights (owned, free it)
    ggml_backend_buffer_t host_buf = nullptr;   // mmap wrap for host tensors (from_ptr)
    std::vector<ggml_tensor *> host_tensors;    // tensors kept in host memory
    bool ok = false;
};

// Split placement for one no_alloc weight context: predicate(name) == true
// keeps the tensor in host memory (mmap), everything else is allocated on
// `dev` and uploaded. `hosts` carries promoted host f32 blocks the same way
// bind_to_backend does (CPU side only).
inline SplitBound split_bind_to_backends(ggml_context * wctx, ggml_backend_t dev,
                                         const MMap & mm,
                                         bool (*keep_on_host)(const char *) = moe_expert_pred,
                                         const std::vector<HostBlock> & hosts = {}) {
    SplitBound out;
    std::vector<std::pair<ggml_tensor *, const void *>> uploads;   // device uploads
    std::vector<ggml_tensor *> dev_tensors;
    size_t dev_total = 0, host_total = 0;
    const char * keep_env = getenv("VQA_SPLIT_KEEP");              // force everything host
    const bool all_host = keep_env && !strcmp(keep_env, "all");
    for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
        if ((keep_on_host && keep_on_host(t->name)) || all_host) {
            t->buffer = nullptr;
            out.host_tensors.push_back(t);
            host_total += ggml_nbytes(t);
        } else {
            uploads.push_back({t, t->data});
            dev_tensors.push_back(t);
            dev_total += ggml_nbytes(t);
        }
    }
    if (dev_tensors.empty()) { out.ok = true; return out; }

    // host side: wrap the mmap so the CPU backend can compute on it in place
    out.host_buf = ggml_backend_cpu_buffer_from_ptr(mm.ptr, mm.size);
    if (!out.host_buf) { fprintf(stderr, "split: host buffer failed\n"); return out; }
    for (ggml_tensor * t : out.host_tensors) {
        t->buffer = out.host_buf;
        // promoted f32 blocks live in their own host allocations
        const char * p = (const char *) t->data;
        for (const HostBlock & h : hosts) {
            if (!h.data || h.size == 0) continue;
            if (p >= (const char *) h.data && p < (const char *) h.data + h.size) {
                // best effort: wrap the block too (rare path - experts are
                // quantized 2D/3D, never promoted)
                ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr((void *) h.data, h.size);
                if (b) t->buffer = b;
                break;
            }
        }
    }

    // device side: allocate just the dev subset and upload
    for (auto & u : uploads) u.first->data = nullptr;
    out.dev_buf = ggml_backend_buft_alloc_buffer_n(ggml_backend_get_default_buffer_type(dev),
                                                   dev_tensors.data(), (int) dev_tensors.size());
    if (!out.dev_buf) {
        fprintf(stderr, "split: device allocation failed (need %.2f GB)\n", dev_total / 1e9);
        return out;
    }
    for (auto & u : uploads)
        ggml_backend_tensor_set(u.first, u.second, 0, ggml_nbytes(u.first));
    fprintf(stderr, "split weights: %.2f GB on %s, %.2f GB experts in host mmap (%zu tensors)\n",
            dev_total / 1e9, ggml_backend_name(dev), host_total / 1e9, out.host_tensors.size());
    out.ok = true;
    return out;
}

}  // namespace weights
}  // namespace vqa
