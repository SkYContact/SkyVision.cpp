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
// Cost when unused: mark_sinks is guarded by the caller's dump-enabled flag;
// the extra nodes only exist while a dump is being taken.
#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace vqa {
namespace dump {

inline void mark_sinks(ggml_context * ctx, ggml_cgraph * gf, const char * const * names) {
    if (!ctx || !gf || !names) return;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (!t->name[0]) continue;
        for (const char * const * n = names; *n; ++n) {
            if (!strcmp(t->name, *n)) {
                ggml_tensor * keep = ggml_scale(ctx, t, 1.0f);
                ggml_set_name(keep, "dump_sink");
                ggml_build_forward_expand(gf, keep);
                break;
            }
        }
    }
}

inline void write(ggml_cgraph * gf, const char * dir, const char * name) {
    ggml_tensor * t = ggml_graph_get_tensor(gf, name);
    if (!t || !t->data) return;
    std::string path = std::string(dir) + "/cpp_" + name + ".bin";
    std::vector<float> tmp;
    std::vector<uint8_t> raw;
    const uint8_t * src;
    if (t->type == GGML_TYPE_F32) {
        tmp.resize((size_t)ggml_nelements(t));
        src = (const uint8_t *)tmp.data();
        ggml_backend_tensor_get(t, tmp.data(), 0, sizeof(float) * tmp.size());
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h((size_t)ggml_nelements(t));
        ggml_backend_tensor_get(t, h.data(), 0, sizeof(ggml_fp16_t) * h.size());
        tmp.resize(h.size());
        for (size_t i = 0; i < h.size(); i++) tmp[i] = ggml_fp16_to_fp32(h[i]);
        src = (const uint8_t *)tmp.data();
    } else {
        const ggml_type_traits * tt = ggml_get_type_traits(t->type);
        tmp.resize((size_t)ggml_nelements(t));
        tt->to_float(t->data, tmp.data(), ggml_nelements(t));
        src = (const uint8_t *)tmp.data();
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

}  // namespace dump
}  // namespace vqa
