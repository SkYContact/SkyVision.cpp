// seg_test.hpp - shared harness for the per-architecture seg-test drivers
// (tools/*_seg_test.cpp): the seeded LCG, the weight workspace, the graph
// runner, and the cpp_<name>.bin/.shape dump convention, in one place.
//
// Pure code motion out of tools/minicpmo_seg_test.cpp and
// tools/nemotronomni_seg_test.cpp (which carried character-identical copies;
// the qwen3omni trio still has its own). The parity gates compare the drivers'
// dump sets BYTE-FOR-BYTE, so nothing here may be "cleaned up": the LCG
// constants, the draw order, the tensor creation order, and the dump naming
// are all load-bearing.
//
// A driver .cpp keeps only its scenario definitions and the main() wiring:
//
//   #include "core/test/seg_test.hpp"
//   static void scenario_x(ggml_backend_t, ggml_gallocr_t, const char * dir);
//   int main(int argc, char ** argv) {
//       g_rng = Rng(0x...u);            // per-model seed, BEFORE any scenario
//       ... backend/allocr bring-up (see the existing drivers) ...
//       scenario_x(backend, allocr, dir);
//   }
//
// Environment knobs (debug only; default runs are unaffected):
//   SEG_NTHREADS=1  single-thread CPU graph compute (drivers' main())
//   SEG_NOSINK=1    skip the sink dups - run in the no-dump gallocr layout the
//                   production graph runs in (NaN hunt: sink lifetime
//                   extension masks a reuse bug there)
//   FA_KV16=1       (minicpmo driver) flash-attn probe uses the production
//                   F16 KV-cache dtype
#pragma once

#include "core/engine/ggml_dump.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#ifdef _WIN32
#include <crtdbg.h>
#endif
// no modal assert dialogs under the test runners (Windows CRT debug reports
// go to the debugger output instead)
inline const bool _no_assert_dialog = []{
#ifdef _WIN32
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_DEBUG);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
    return true;
}();

// ------------------------------------------------------------- seeded RNG
struct Rng {
    uint32_t s = 1u;
    explicit Rng(uint32_t seed) : s(seed ? seed : 1u) {}
    float u() {   // uniform [-1, 1]
        s += 0x6D2B79F5u;
        uint32_t t = s;
        t = (t ^ (t >> 15)) * (t | 1u);
        t ^= t + (t ^ (t >> 7)) * (t | 61u);
        const uint32_t m = (t ^ (t >> 14)) & 0x7fffffffu;
        return ((float)(int)m - 1073741824.0f) / 1073741824.0f;
    }
};

// driver-global RNG; each driver seeds its own model's constant at the top of
// main(), before any scenario runs (the numpy references re-draw the same
// stream, so the seed + draw order are part of the parity contract)
inline Rng g_rng{1u};

inline std::vector<float> rnd(size_t n, float scale) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) v[i] = g_rng.u() * scale;
    return v;
}

// ------------------------------------------------------- weight workspace
// tensors are created first, allocated in one backend buffer, then filled -
// the same shape the engine's no_alloc wctx + bind_to_backend takes, so the
// graphs see weights exactly like production (buffer-backed, gallocr-skipped).
struct WW {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    void init(ggml_backend_t be, size_t n_tensors) {
        backend = be;
        struct ggml_init_params ip = { (n_tensors + 8) * ggml_tensor_overhead(),
                                       NULL, /*no_alloc*/ true };
        ctx = ggml_init(ip);
    }
    ggml_tensor * make(const char * name, int64_t ne0, int64_t ne1 = 1,
                       int64_t ne2 = 1, ggml_type type = GGML_TYPE_F32) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        ggml_set_name(t, name);
        return t;
    }
    void alloc() {
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { fprintf(stderr, "weight buffer alloc failed\n"); exit(1); }
        ggml_backend_buffer_clear(buf, 0);   // fresh states / caches
    }
    void fill(ggml_tensor * t, const std::vector<float> & v) {
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, v.data(), 0, sizeof(float) * v.size());
        } else {
            std::vector<ggml_fp16_t> h(v.size());
            for (size_t i = 0; i < v.size(); i++) h[i] = ggml_fp32_to_fp16(v[i]);
            ggml_backend_tensor_set(t, h.data(), 0, sizeof(ggml_fp16_t) * h.size());
        }
    }
    ~WW() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

inline void dt(ggml_tensor * t, const char * dir, const char * name) {
    vqa::dump::write_tensor(t, dir, name);
}

// host vector -> f32 bin + .shape sidecar (ne order, padded to 4 entries)
inline void dt_f(const std::vector<float> & v, std::vector<int64_t> ne,
                 const char * dir, const char * name) {
    std::filesystem::create_directories(dir);
    FILE * f = fopen((std::string(dir) + "/cpp_" + name + ".bin").c_str(), "wb");
    if (f) { fwrite(v.data(), sizeof(float), v.size(), f); fclose(f); }
    f = fopen((std::string(dir) + "/cpp_" + name + ".shape").c_str(), "w");
    if (f) {
        for (size_t i = 0; i < 4; i++)
            fprintf(f, "%lld%s", i < ne.size() ? (long long)ne[i] : 1, i < 3 ? " " : "\n");
        fclose(f);
    }
}

// I32 tensors (top-k ids): read raw ints, write as f32 bins under the same
// naming convention so the numpy reader stays uniform
inline void dt_i32(ggml_tensor * t, const char * dir, const char * name) {
    const size_t n = (size_t)ggml_nelements(t);
    std::vector<int32_t> iv(n);
    ggml_backend_tensor_get(t, iv.data(), 0, sizeof(int32_t) * n);
    std::vector<float> fv(iv.begin(), iv.end());
    std::filesystem::create_directories(dir);
    FILE * f = fopen((std::string(dir) + "/cpp_" + name + ".bin").c_str(), "wb");
    if (f) { fwrite(fv.data(), sizeof(float), fv.size(), f); fclose(f); }
    f = fopen((std::string(dir) + "/cpp_" + name + ".shape").c_str(), "w");
    if (f) {
        fprintf(f, "%lld %lld %lld %lld\n", (long long)t->ne[0],
                (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
        fclose(f);
    }
}

// ------------------------------------------------------------ graph runner
struct Run {
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;
    const char * dir = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    std::vector<const char *> sinks;

    void begin() {
        struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, true };
        ctx = ggml_init(ip);
        gf = ggml_new_graph_custom(ctx, 8192, false);
    }
    bool alloc() {   // after the build: attach sinks, then assign buffers
        // SEG_NOSINK=1: skip the sink dups - the no-dump gallocr layout the
        // production graph runs in (NaN hunt: sink lifetime extension masks
        // a reuse bug there)
        if (!sinks.empty() && !getenv("SEG_NOSINK")) {
            // mark_sinks wants a null-terminated name list (engine callers
            // pass static arrays ending in nullptr)
            std::vector<const char *> ns(sinks.begin(), sinks.end());
            ns.push_back(nullptr);
            vqa::dump::mark_sinks(ctx, gf, ns.data());
            vqa::dump::finalize_sinks(ctx, gf, backend);
        }
        if (!ggml_gallocr_alloc_graph(allocr, gf)) {
            fprintf(stderr, "graph alloc failed\n");
            return false;
        }
        return true;
    }
    bool compute() {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            return false;
        }
        return true;
    }
    // names whose dup is created (value kept alive) but written manually
    // after compute (ggml has no to_float for I32 - write_tensor would call
    // a null handler)
    std::vector<const char *> raw;

    void dump_all() const {
        if (!dir) return;
        for (const char * n : sinks) {
            bool is_raw = false;
            for (const char * r2 : raw) is_raw = is_raw || !strcmp(r2, n);
            if (!is_raw) vqa::dump::write(gf, dir, n);
        }
    }
    void end() {
        vqa::dump::dump_registry_clear();
        ggml_free(ctx);
        ctx = nullptr;
        gf = nullptr;
    }
};
