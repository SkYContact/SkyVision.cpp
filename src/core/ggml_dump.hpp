// ggml_dump.hpp - trustworthy parity dumps for named graph tensors.
//
// gallocr reuses the buffer of any tensor that is not a direct graph node, so
// dumping a named intermediate AFTER compute can read another node's data (the
// "only sinks are trustworthy" lesson, hit independently by all three
// backends). Two-phase protocol:
//
//   1. BEFORE gallocr_alloc_graph: dump::mark_sinks(ctx, gf, names) attaches a
//      trailing identity-scale consumer to every named tensor, extending its
//      buffer lifetime past the whole compute. Note ggml_build_forward_expand
//      alone is a NO-OP for tensors already reachable in the graph - the fresh
//      consumer node is what does the work.
//   2. AFTER compute: dump::write(gf, dir, name) fetches the tensor and writes
//      dir/cpp_<name>.bin (+ .shape with the ggml ne[] layout).
//
// Marked tensors are registered by POINTER (dump_registry): graph name lookup
// can be ambiguous when a name collides across differently-shaped graphs, so
// dump()/write() read through the registry first.
//
// Cost when unused: mark_sinks is guarded by the caller's dump-enabled flag;
// the extra nodes only exist while a dump is being taken. Call
// dump_registry_clear() before rebuilding a graph to drop stale entries.
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstring>

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace vqa {
namespace dump {

inline std::vector<std::pair<std::string, ggml_tensor *>> & dump_registry() {
    static std::vector<std::pair<std::string, ggml_tensor *>> reg;
    return reg;
}

// sink dup tensors live in private contexts whose buffers come from
// ggml_backend_alloc_ctx_tensors (NOT the gallocr pool). A dup output inside
// the pool is a graph leaf nothing consumes, so gallocr frees its memory the
// moment the dup executes and later nodes overwrite it -> the file written
// after compute contains another node's data (repro'd: dup(xkv_knorm) shared
// an address with dup(v_patch_out)). Pre-allocated tensors are skipped by
// gallocr (treated like weights), so the copied value survives the graph.
inline std::vector<std::pair<ggml_context *, ggml_backend_buffer_t>> & sink_pool() {
    static std::vector<std::pair<ggml_context *, ggml_backend_buffer_t>> v;
    return v;
}

inline void dump_registry_clear() {
    dump_registry().clear();
    for (auto & p : sink_pool()) {
        if (p.second) ggml_backend_buffer_free(p.second);
        if (p.first) ggml_free(p.first);
    }
    sink_pool().clear();
}

// sink names are REGISTERED as the graph is built and only attached at the
// very end (finalize_sinks). Attaching a keep node mid-build lets gallocr
// hand the tensor's buffer to later nodes (the keep is then mid-graph, not
// last-use), so the dumped value is silently overwritten - repro'd in
// convert/sink_repro.cpp (mid-build mark -> 99.99% corrupted values).
inline std::vector<std::string> & pending_sinks() {
    static std::vector<std::string> v;
    return v;
}

inline void mark_sinks(ggml_context * ctx, ggml_cgraph * gf, const char * const * names) {
    (void)ctx; (void)gf;
    if (!names) return;
    for (const char * const * n = names; *n; ++n) pending_sinks().push_back(*n);
}

// call ONCE per graph, after the build is complete and BEFORE
// ggml_gallocr_alloc_graph: resolves pending names and appends the keep
// nodes at the true tail of the graph.
inline void finalize_sinks(ggml_context * ctx, ggml_cgraph * gf, ggml_backend_t backend) {
    if (!ctx || !gf || pending_sinks().empty()) { pending_sinks().clear(); return; }
    ggml_context * sctx = ggml_init({ 128ull << 20, nullptr, true });
    size_t ndup = 0;
    for (const std::string & want : pending_sinks()) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (!t->name[0] || strcmp(t->name, want.c_str())) continue;
            // ggml_scale(t) keeps are NOT reliably honored by this fork's
            // gallocr (repro: values overwritten mid-graph). Force a real
            // value copy into a fresh tensor - its buffer is owned by the
            // dump alone.
            ggml_tensor * dup = ggml_dup(sctx, t);
            ggml_set_name(dup, "dump_sink");
            ggml_build_forward_expand(gf, dup);
            dump_registry().push_back({want, dup});
            ndup++;
            break;
        }
    }
    pending_sinks().clear();
    if (ndup) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(sctx, backend);
        if (!buf) { fprintf(stderr, "dump: sink buffer alloc failed\n"); }
        sink_pool().push_back({ sctx, buf });
    } else {
        ggml_free(sctx);
    }
}

// compat overload (standalone CPU probes)
inline void finalize_sinks(ggml_context * ctx, ggml_cgraph * gf) {
    static ggml_backend_t sink_cpu = ggml_backend_cpu_init();
    finalize_sinks(ctx, gf, sink_cpu);
}

// registry-first tensor lookup (falls back to graph name lookup)
inline ggml_tensor * lookup(ggml_cgraph * gf, const char * name) {
    for (auto it = dump_registry().rbegin(); it != dump_registry().rend(); ++it)
        if (it->first == name) return it->second;
    return ggml_graph_get_tensor(gf, name);
}

inline void write_tensor(ggml_tensor * t, const char * dir, const char * name) {
    if (!t || !t->data) return;
    std::string path = std::string(dir) + "/cpp_" + name + ".bin";
    std::vector<float> tmp;
    const uint8_t * src;
    if (t->type == GGML_TYPE_F32) {
        tmp.resize((size_t)ggml_nelements(t));
        src = (const uint8_t *)tmp.data();
        ggml_backend_tensor_get(t, tmp.data(), 0, sizeof(float) * tmp.size());
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h((size_t)ggml_nelements(t));
        ggml_backend_tensor_get(t, h.data(), 0, sizeof(ggml_fp16_t) * h.size());
        tmp.resize(h.size());
        src = (const uint8_t *)h.data();
        for (size_t i = 0; i < h.size(); i++) tmp[i] = ggml_fp16_to_fp32(h[i]);
    } else {
        const struct ggml_type_traits * tt = ggml_get_type_traits(t->type);
        tmp.resize((size_t)ggml_nelements(t));
        src = (const uint8_t *)tmp.data();
        tt->to_float(t->data, tmp.data(), ggml_nelements(t));
    }
    FILE * f = fopen(path.c_str(), "wb");
    if (f) { fwrite(src, sizeof(float), tmp.size(), f); fclose(f); }
    FILE * m = fopen((std::string(dir) + "/cpp_" + name + ".shape").c_str(), "w");
    if (m) {
        fprintf(m, "%lld %lld %lld %lld\n", (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3]);
        fclose(m);
    }
}

inline void write(ggml_cgraph * gf, const char * dir, const char * name) {
    write_tensor(lookup(gf, name), dir, name);
}

}  // namespace dump
}  // namespace vqa
