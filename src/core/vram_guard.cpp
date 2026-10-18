// vram_guard.cpp - the VRAM head-room probe taken before a model load.
//
// A load that cannot fit used to die deep inside a ggml buffer allocation
// (on Windows, where CUDA reserves commit charge, often as the mystical
// error 1455) - possibly taking an otherwise healthy server down with it.
// The guard measures free device memory through the ggml backend device API
// (no direct CUDA linkage) and predicts the requirement from the GGUF's own
// size and architecture metadata. Coarse by design; the escape hatch is
// SKYVISION_FORCE_LOAD=1. Same convention as backend.cpp for the ifdef: a
// CPU-only build has no GPU device and admits everything at zero cost.
// Lives in its own translation unit so tools/vram_probe.cpp can link the
// exact same logic without pulling the whole engine in.
#include "model_registry.hpp"

#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#else
  #include <sys/stat.h>
#endif

namespace vqa {
namespace {

uint64_t side_file_size(const std::string & path) {
    if (path.empty()) return 0;
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA fa;
    if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &fa)) return 0;
    return ((uint64_t)fa.nFileSizeHigh << 32) | (uint64_t)fa.nFileSizeLow;
#else
    struct stat st {};
    if (stat(path.c_str(), &st) != 0) return 0;
    return (uint64_t)st.st_size;
#endif
}

// unsigned metadata reader for the shape counts. Converters here disagree on
// the spelling: most write u32, some i32, the occasional one f32 - accept any
// of them when the value is a sane non-negative integer.
bool gguf_u32(gguf_context * g, const char * key, uint32_t & out) {
    const int k = gguf_find_key(g, key);
    if (k < 0) return false;
    switch (gguf_get_kv_type(g, k)) {
        case GGUF_TYPE_UINT32:
            out = gguf_get_val_u32(g, k);
            return true;
        case GGUF_TYPE_INT32: {
            const int32_t v = gguf_get_val_i32(g, k);
            if (v < 0) return false;
            out = (uint32_t)v;
            return true;
        }
        case GGUF_TYPE_FLOAT32: {
            const float v = gguf_get_val_f32(g, k);
            if (v < 0.0f || v != (float)(uint32_t)v) return false;
            out = (uint32_t)v;
            return true;
        }
        default:
            return false;
    }
}

// coarse KV-cache bytes: 2 (K+V) x n_layer x n_ctx x (k_dim + v_dim) x 2 bytes
// (f16 cache). Dims come from "<arch>.*" metadata; when the file lacks them
// the estimate degrades to 0 and the guard falls back to weights-only - that
// alone still catches the catastrophic case (a second 8GB model next to a
// resident one). Best effort, never fatal: any read failure just lowers the
// estimate. Key prefixes are probed in order: the registry-normalised label
// (marlin vs qwen3_5), the file's own general.architecture, and the literal
// "{arch}" placeholder (the moondream3 converter writes it unsubstituted).
uint64_t kv_bytes_of(const std::string & path, const std::string & arch, int n_ctx) {
    struct gguf_init_params p = { true, nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), p);
    if (!g) return 0;
    std::string raw = arch;
    const int ka = gguf_find_key(g, "general.architecture");
    if (ka >= 0) raw = gguf_get_val_str(g, ka);
    const std::string prefixes[3] = { arch, raw, "{arch}" };

    uint32_t layers = 0, embd = 0, heads = 0, kv_heads = 0, kdim = 0, vdim = 0;
    auto read_u32 = [&](std::initializer_list<const char *> suffixes, uint32_t & out) {
        for (const auto & pfx : prefixes)
            for (const char * sfx : suffixes)
                if (gguf_u32(g, (pfx + sfx).c_str(), out)) return true;
        return false;
    };
    bool ok = read_u32({ ".block_count" }, layers);
    ok = read_u32({ ".embedding_length" }, embd) && ok;
    ok = read_u32({ ".attention.head_count" }, heads) && ok;
    if (!ok || layers == 0 || embd == 0 || heads == 0) { gguf_free(g); return 0; }
    if (!read_u32({ ".attention.head_count_kv" }, kv_heads)) kv_heads = heads;   // plain MHA
    if (!read_u32({ ".attention.key_length" }, kdim)) kdim = embd / heads;       // head_dim
    if (!read_u32({ ".attention.value_length" }, vdim)) vdim = kdim;
    gguf_free(g);
    if (n_ctx <= 0) n_ctx = 8192;   // LoadOptions' default window
    return 2ull * layers * (uint64_t)n_ctx * (kdim + vdim) * 2ull;
}

}  // namespace

// public (declared in model_registry.hpp): tools/vram_probe.cpp prints it even
// in a CPU-only build, where vram_check itself skips without calling it
uint64_t vram_kv_bytes(const std::string & path, const std::string & arch, int n_ctx) {
    return kv_bytes_of(path, arch, n_ctx);
}

VramCheck vram_check(const ModelEntry & e, const LoadOptions & opt) {
    VramCheck r;
    r.weights_b = e.size + side_file_size(e.vision_path);
    r.need_b = r.weights_b + r.weights_b / 10;   // x1.1: activations + runtime
#ifndef GGML_USE_CUDA
    (void)opt;
    return r;   // CPU-only build: weights land in system RAM, nothing to guard
#else
    if (!opt.use_gpu) return r;                  // --cpu: system RAM, ditto
    if (getenv("SKYVISION_FORCE_LOAD")) return r;   // documented escape hatch
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) return r;                          // no GPU backend registered
    ggml_backend_dev_memory(dev, &r.free_b, &r.total_b);
    if (r.total_b == 0) return r;                // device would not say: admit
    r.have_gpu = true;
    r.device = ggml_backend_dev_description(dev);
    r.kv_b = vram_kv_bytes(e.path, e.arch, opt.n_ctx);
    r.need_b += r.kv_b;
    vram_admits(r, r.err);
    return r;
#endif
}

bool vram_admits(const VramCheck & r, std::string & err) {
    if (!r.have_gpu || r.need_b <= r.free_b) return true;
    char need[64], free_s[64], total[64];
    snprintf(need, sizeof(need), "%.1f GiB", (double)r.need_b / 1073741824.0);
    snprintf(free_s, sizeof(free_s), "%.1f GiB", (double)r.free_b / 1073741824.0);
    snprintf(total, sizeof(total), "%.1f GiB", (double)r.total_b / 1073741824.0);
    err = "refusing to load: estimated " + std::string(need) + " needed (" +
          std::to_string(r.weights_b) + " B weights x1.1 + " + std::to_string(r.kv_b) +
          " B KV) but " + r.device + " has only " + free_s + " free of " + total +
          ". Free VRAM first (unload other models / close other processes), pick a "
          "lower quantisation, restart the server with --cpu, or set "
          "SKYVISION_FORCE_LOAD=1 to try anyway.";
    return false;
}

}  // namespace vqa
