// backend.cpp - see backend.hpp.
#include "core/engine/backend.hpp"

#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#   include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cstdio>
#include <thread>
#include <vector>

#ifdef _WIN32
#   ifndef NOMINMAX
#       define NOMINMAX   // windows.h min/max macros break std::max below
#   endif
#   include <windows.h>
#else
#   include <cstring>
#   include <set>
#   include <utility>
#endif

namespace vqa {

namespace {

int logical_threads() {
    const unsigned hc = std::thread::hardware_concurrency();
    return (int)std::max(1u, hc);
}

#if defined(_WIN32)
int physical_cores() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return 0;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
                                          (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &len))
        return 0;
    int cores = 0;
    for (DWORD off = 0; off + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= len; ) {
        auto * e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf.data() + off);
        if (e->Relationship == RelationProcessorCore) cores++;
        if (e->Size == 0) break;
        off += e->Size;
    }
    return cores;
}
#elif defined(__linux__)
int physical_cores() {
    FILE * f = fopen("/proc/cpuinfo", "rb");
    if (!f) return 0;
    std::set<std::pair<int, int>> cores;
    char line[256];
    int phys = -1, core = -1;
    while (fgets(line, sizeof(line), f)) {
        int v = -1;
        if (sscanf(line, "physical id : %d", &v) == 1) phys = v;
        else if (sscanf(line, "core id : %d", &v) == 1) core = v;
        else if (line[0] == '\n' || line[0] == '\r') {   // end of one processor block
            if (phys >= 0 && core >= 0) cores.insert({phys, core});
            phys = core = -1;
        }
    }
    if (phys >= 0 && core >= 0) cores.insert({phys, core});
    fclose(f);
    return (int)cores.size();
}
#else
int physical_cores() { return 0; }   // unknown platform: keep the logical/2 heuristic
#endif

}  // namespace

int default_n_threads() {
    int cores = physical_cores();
    if (cores <= 0) cores = (int)std::max(1, logical_threads() / 2);
    return cores;
}

ggml_backend_t init_backend(bool use_gpu, int & n_threads) {
    ggml_backend_t backend = nullptr;
    if (use_gpu) {
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
#endif
    }
    if (backend) {
        fprintf(stderr, "backend: %s\n", ggml_backend_name(backend));
        return backend;
    }
    backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "no backend available\n");
        return nullptr;
    }
    if (n_threads <= 0) n_threads = default_n_threads();
    // VQA_POLL=<0-100> (polling level): worker threads spin-wait for work
    // instead of sleeping. Without it every op pays a wake-the-threads sync
    // (~0.5-1 ms on Windows), which on decode-sized ops costs ~10x - measured:
    // a 2048x4096 Q4_K GEMV runs 1.0 ms -> 0.087 ms with poll=100. The trade
    // is idle CPU burn while the model sits between steps, so it stays opt-in.
    static ggml_threadpool_t poll_pool = nullptr;   // backend keeps the pointer
    const char * poll_env = getenv("VQA_POLL");
    if (poll_env && atoi(poll_env) > 0) {
        struct ggml_threadpool_params tp = ggml_threadpool_params_default(n_threads);
        tp.poll = (uint32_t) atoi(poll_env);
        poll_pool = ggml_threadpool_new(&tp);
        if (poll_pool) {
            ggml_backend_cpu_set_threadpool(backend, poll_pool);
            fprintf(stderr, "backend: %s (%d threads, poll=%s)\n",
                    ggml_backend_name(backend), n_threads, poll_env);
            return backend;
        }
    }
    ggml_backend_cpu_set_n_threads(backend, n_threads);
    fprintf(stderr, "backend: %s (%d threads)\n", ggml_backend_name(backend), n_threads);
    return backend;
}

}  // namespace vqa
