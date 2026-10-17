// gguf_util.hpp - GGUF metadata readers + small loading helpers shared by every
// backend. Header-only (no global state): these used to be copy-pasted verbatim
// into all five backends, which is exactly the kind of drift this file exists to
// stop - a key that is BOOL in one converter and UINT32 in another must not make
// two backends interpret the same model differently.
#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cstring>
#include <string>
#include <vector>

namespace vqa {
namespace gguf {

// Scalar float metadata; `def` when the key is absent (or of another type).
inline float f32(struct gguf_context * g, const char * key, float def) {
    const int64_t id = gguf_find_key(g, key);
    return id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_FLOAT32 ? gguf_get_val_f32(g, id) : def;
}

// Integer metadata, tolerant of every scalar type the converters emit -
// including BOOL, because python writes `use_mean_pooling: true` as a bool and
// gguf-py keeps it that way.
inline int u32(struct gguf_context * g, const char * key, int def) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return (int)gguf_get_val_u32(g, id);
        case GGUF_TYPE_BOOL:   return gguf_get_val_bool(g, id) ? 1 : 0;
        case GGUF_TYPE_INT32:  return (int)gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (int)gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:  return (int)gguf_get_val_i64(g, id);
        default: return def;
    }
}

inline bool boolean(struct gguf_context * g, const char * key, bool def) {
    return u32(g, key, def ? 1 : 0) != 0;
}

inline std::string str(struct gguf_context * g, const char * key, const std::string & def = {}) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(g, id);
}

// u32 array metadata (token ids, per-layer lists); `def` when absent/empty.
inline std::vector<int> arr_u32(struct gguf_context * g, const char * key,
                                const std::vector<int> & def = {}) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY) return def;
    const int64_t n = gguf_get_arr_n(g, id);
    if (n <= 0) return def;
    const int32_t * arr = (const int32_t *)gguf_get_arr_data(g, id);
    return std::vector<int>(arr, arr + n);
}

// Arbitrary GGML tensor -> host float array (F32/F16/quantized).
inline void tensor_to_f32(ggml_tensor * t, float * dst) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        memcpy(dst, t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n; i++) dst[i] = ggml_fp16_to_fp32(s[i]);
    } else {
        const struct ggml_type_traits * tt = ggml_get_type_traits(t->type);
        tt->to_float(t->data, dst, n);
    }
}

}  // namespace gguf
}  // namespace vqa
