// backend.hpp - the ggml backend bring-up every backend used to copy-paste:
// CUDA first (when the build has it and the caller allows it), CPU otherwise,
// with the worker-count default in one place.
//
// The default matters: ggml's CPU kernels on this model family are GEMV/small-GEMM
// bound, where SMT threads *hurt*. The old copy-pasted default was
// hardware_concurrency()/2, which under-counts a hybrid CPU badly (a 20-core
// i7-14700K reports 28 logical -> 14 threads instead of 20: measured 979 ms vs
// 823 ms per VideoMAE clip, i.e. 16% wasted). default_n_threads() counts physical
// cores instead.
#pragma once

#include "ggml-backend.h"

namespace vqa {

// Physical core count (Windows: relation-processor-core; Linux: unique
// (package, core) pairs in /proc/cpuinfo); falls back to logical/2.
int default_n_threads();

// Returns a CUDA backend when use_gpu and the build has GGML_USE_CUDA (silently
// falls back to CPU), else a CPU backend. `n_threads` is in/out: <= 0 picks
// default_n_threads(), otherwise the caller's value wins. Logs one line, like
// every backend did inline before.
ggml_backend_t init_backend(bool use_gpu, int & n_threads);

}  // namespace vqa
