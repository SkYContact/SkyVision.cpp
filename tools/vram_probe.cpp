// vram_probe.cpp - standalone probe for the registry's VRAM guard (roadmap §6.3-E).
//
//   vram-probe [model.gguf]
//
// 1. prints every device the ggml backend registry exposes (name/type/memory),
// 2. for the optional GGUF argument prints exactly what vram_check() predicts
//    for it (weights x1.1 + coarse KV vs free VRAM) and the guard's verdict,
// 3. unit-tests the vram_admits() threshold decision with constructed
//    insufficient-memory scenarios (no GPU required).
// Built as its own target from src/core/runtime/vram_guard.cpp - not part of vqa-cli.
#include "core/runtime/model_registry.hpp"

#include "ggml-backend.h"

#include <cstdio>
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

using vqa::LoadOptions;
using vqa::ModelEntry;
using vqa::VramCheck;

static int g_fail = 0;

static void expect(bool ok, const char * what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) g_fail = 1;
}

static uint64_t gib(double v) { return (uint64_t)(v * 1073741824.0); }

int main(int argc, char ** argv) {
    // ---- 1. device enumeration (what the guard looks at) ----
    printf("== ggml backend devices ==\n");
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        size_t free_b = 0, total_b = 0;
        ggml_backend_dev_memory(d, &free_b, &total_b);
        const char * type = "?";
        switch (ggml_backend_dev_type(d)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU: type = "cpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_GPU: type = "gpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU: type = "igpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: type = "accel"; break;
            default: break;
        }
        printf("  [%zu] %-14s %-30s free %.1f / total %.1f GiB\n", i, type,
               ggml_backend_dev_description(d), (double)free_b / 1073741824.0,
               (double)total_b / 1073741824.0);
    }
    if (!n) printf("  (none)\n");

    // ---- 2. live estimate for a real GGUF ----
    if (argc > 1) {
        printf("\n== vram_check(%s) ==\n", argv[1]);
        ModelEntry e;
        e.id = "probe";
        e.path = argv[1];
        // e.arch stays empty: the real registry normalises it, but kv_bytes_of
        // falls back to the file's own general.architecture, so the estimate is
        // identical without linking vqa_registry.cpp (it pulls every backend in)
#ifdef _WIN32
        {   // ftell is 32-bit here: >2GB GGUFs overflow it (all the big ones do)
            WIN32_FILE_ATTRIBUTE_DATA fa;
            if (GetFileAttributesExA(argv[1], GetFileExInfoStandard, &fa))
                e.size = ((uint64_t)fa.nFileSizeHigh << 32) | fa.nFileSizeLow;
        }
#else
        {
            struct stat st;
            if (stat(argv[1], &st) == 0) e.size = (uint64_t)st.st_size;
        }
#endif
        const VramCheck r = vram_check(e, LoadOptions());
        // in a CPU-only build vram_check skips at zero cost (kv_b stays 0), so
        // the estimator is also called directly to show what a CUDA build adds
        const uint64_t kv = r.kv_b ? r.kv_b : vqa::vram_kv_bytes(e.path, e.arch, 0);
        const uint64_t need = r.weights_b + r.weights_b / 10 + kv;
        std::string err;
        const bool admits = vram_admits(r, err);
        printf("  device      : %s\n", r.have_gpu ? r.device.c_str() : "(none visible: guard skips)");
        printf("  free/total  : %.1f / %.1f GiB\n", (double)r.free_b / 1073741824.0,
               (double)r.total_b / 1073741824.0);
        printf("  weights     : %.2f GiB (gguf + vision sidecar)\n", (double)r.weights_b / 1073741824.0);
        printf("  kv est      : %.2f GiB (coarse, f16 cache at n_ctx)\n", (double)kv / 1073741824.0);
        printf("  need (x1.1) : %.2f GiB\n", (double)need / 1073741824.0);
        printf("  verdict     : %s\n", admits ? "ADMIT" : "REFUSE");
        if (!admits) printf("  advice      : %s\n", err.c_str());
    }

    // ---- 3. threshold decision unit tests (constructed scenarios) ----
    printf("\n== vram_admits unit tests ==\n");
    {
        VramCheck r;   // the OOM the guard exists for: 8.4G moss next to a resident 7.3G omni
        r.have_gpu = true; r.device = "fake 12G card";
        r.free_b = gib(2.0); r.total_b = gib(12.0);
        r.weights_b = gib(8.4); r.kv_b = gib(1.5); r.need_b = r.weights_b + r.weights_b / 10 + r.kv_b;
        std::string err;
        expect(!vram_admits(r, err), "insufficient: need 10.7 GiB vs 2.0 GiB free -> refuse");
        expect(!err.empty() && err.find("--cpu") != std::string::npos &&
               err.find("SKYVISION_FORCE_LOAD") != std::string::npos,
               "insufficient: error text carries actionable advice (--cpu / FORCE_LOAD)");
    }
    {
        VramCheck r;   // a 515M smolvlm on an empty card
        r.have_gpu = true; r.device = "fake 12G card";
        r.free_b = gib(11.5); r.total_b = gib(12.0);
        r.weights_b = gib(0.5); r.kv_b = gib(0.2); r.need_b = r.weights_b + r.weights_b / 10 + r.kv_b;
        std::string err;
        expect(vram_admits(r, err), "sufficient: need 0.75 GiB vs 11.5 GiB free -> admit");
        expect(err.empty(), "sufficient: no error text");
    }
    {
        VramCheck r;   // exact fit counts as fitting (<=)
        r.have_gpu = true; r.device = "fake";
        r.free_b = gib(1.0); r.total_b = gib(12.0); r.need_b = gib(1.0);
        std::string err;
        expect(vram_admits(r, err), "boundary: need == free -> admit");
    }
    {
        VramCheck r;   // CPU-only build / --cpu / no GPU: guard is a no-op
        r.have_gpu = false; r.need_b = gib(64.0); r.free_b = 0;
        std::string err;
        expect(vram_admits(r, err), "no gpu: guard admits regardless (zero-overhead skip)");
    }
    {
        VramCheck r;   // device present but would not report memory
        r.have_gpu = false; r.free_b = 0; r.total_b = 0; r.need_b = gib(64.0);
        std::string err;
        expect(vram_admits(r, err), "silent device: admit (no numbers to judge)");
    }

    printf("\n%s\n", g_fail ? "RESULT: FAIL" : "RESULT: ALL PASS");
    return g_fail;
}
