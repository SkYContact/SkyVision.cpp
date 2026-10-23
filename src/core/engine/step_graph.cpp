// step_graph.cpp - implements vqa::StepGraph (see step_graph.hpp).
//
// Build-once/run-many holder. The context is created with no_alloc=true; the
// gallocr (private, or caller-shared) assigns device buffers once at build().
// run() only re-executes ggml_backend_graph_compute - the allocation plan and
// every tensor->data pointer stay valid as long as the SAME gallocr is not
// re-reserved by a differently-shaped graph. That means:
//   - a PRIVATE gallocr (default) is never touched again -> graph state is
//     fully stable across runs (inputs keep their bytes between runs, though
//     re-uploading them is cheap and recommended);
//   - a SHARED gallocr is only safe if every graph ever allocated on it has
//     an identical allocation plan (same shapes/node counts) - the allocator
//     then reuses the same pool and re-runs are stable after re-setting the
//     inputs.

#include "step_graph.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdio>

namespace vqa {

// The cgraph under construction, so a build() callback can attach side-effect
// nodes (e.g. ggml_set_rows KV writes) as extra sinks via
// ggml_build_forward_expand(step_graph_builder_gf(), node). Null outside a
// build() callback. (Not in the frozen header; declare it where needed.)
static thread_local ggml_cgraph * tl_builder_gf = nullptr;
ggml_cgraph * step_graph_builder_gf() { return tl_builder_gf; }

StepGraph::~StepGraph() { free_graph(); }

bool StepGraph::build(ggml_backend_t backend,
                      ggml_tensor * (*builder)(ggml_context *, void *), void * ud,
                      ggml_gallocr_t shared_allocr) {
    free_graph();
    if (!backend || !builder) return false;
    backend_ = backend;

    struct ggml_init_params ip = { 64ull << 20, NULL, /*no_alloc*/ true };
    ctx_ = ggml_init(ip);
    if (!ctx_) return false;
    gf_ = ggml_new_graph_custom(ctx_, 8192, false);
    if (!gf_) { free_graph(); return false; }

    tl_builder_gf = gf_;
    ggml_tensor * out = builder(ctx_, ud);
    tl_builder_gf = nullptr;
    if (!out) { free_graph(); return false; }
    out_ = out;
    ggml_build_forward_expand(gf_, out_);

    if (shared_allocr) {
        allocr_ = shared_allocr;
        owns_allocr_ = false;
    } else {
        allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        owns_allocr_ = true;
    }
    if (!allocr_ || !ggml_gallocr_alloc_graph(allocr_, gf_)) {
        fprintf(stderr, "StepGraph: graph allocation failed\n");
        if (owns_allocr_ && allocr_) ggml_gallocr_free(allocr_);
        allocr_ = nullptr;
        owns_allocr_ = false;
        free_graph();           // frees ctx_ only; shared allocr untouched
        return false;
    }
    return true;
}

bool StepGraph::set_input(ggml_tensor * leaf, const void * data, size_t bytes) {
    if (!ctx_ || !leaf || !data || bytes == 0) return false;
    ggml_backend_tensor_set(leaf, data, 0, bytes);   // void return in this ggml
    return true;
}

bool StepGraph::run() {
    if (!gf_ || !backend_) return false;
    return ggml_backend_graph_compute(backend_, gf_) == GGML_STATUS_SUCCESS;
}

void StepGraph::free_graph() {
    if (ctx_) { ggml_free(ctx_); ctx_ = nullptr; }
    gf_ = nullptr;
    out_ = nullptr;
    backend_ = nullptr;
    if (owns_allocr_ && allocr_) ggml_gallocr_free(allocr_);
    allocr_ = nullptr;
    owns_allocr_ = false;
}

}  // namespace vqa

#ifdef STEP_GRAPH_TEST_MAIN
// Self-contained test: tiny graph out = a + a*b on the CPU backend.
//   1. build / run / read back with hand-computed values
//   2. re-run with fresh inputs (same allocation reused)
//   3. two holders sharing one gallocr, run interleaved
#include <cmath>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s (line %d)\n", msg, __LINE__); g_fail++; } } while (0)

struct TestUd {
    ggml_tensor * a = nullptr, * b = nullptr;   // input leaves
    int tag = 0;                                 // which op combo to build
    bool saw_gf = false;                         // builder_gf visible?
};

static ggml_tensor * test_builder(ggml_context * c, void * ud) {
    TestUd * u = (TestUd *) ud;
    u->saw_gf = vqa::step_graph_builder_gf() != nullptr;
    u->a = ggml_new_tensor_1d(c, GGML_TYPE_F32, 2);
    u->b = ggml_new_tensor_1d(c, GGML_TYPE_F32, 2);
    ggml_tensor * t = ggml_mul(c, u->a, u->b);
    if (u->tag == 0) return ggml_add(c, u->a, t);       // a + a*b
    return ggml_sub(c, ggml_add(c, u->a, u->b), t);     // a + b - a*b
}

static bool run_case(vqa::StepGraph & sg, TestUd & u,
                     const float * a, const float * b, const float * exp) {    if (!sg.set_input(u.a, a, sizeof(float) * 2)) return false;
    if (!sg.set_input(u.b, b, sizeof(float) * 2)) return false;
    if (!sg.run()) return false;
    float o[2];
    ggml_backend_tensor_get(sg.out(), o, 0, sizeof(o));
    for (int i = 0; i < 2; i++)
        if (fabsf(o[i] - exp[i]) > 1e-6f) {
            fprintf(stderr, "FAIL value [%d] got %.7f expected %.7f\n", i, o[i], exp[i]);
            g_fail++;
            return true;
        }
    return true;
}

int main(void) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(cpu, "cpu backend");
    if (!cpu) return 1;

    {   // 1+2: private allocr, build/run/rerun
        vqa::StepGraph sg;
        TestUd u;
        float a[2], b[2], e[2];
        CHECK(sg.build(cpu, test_builder, &u), "build");
        CHECK(u.saw_gf, "builder sees the graph under construction");
        a[0] = 1; a[1] = 2;  b[0] = 3; b[1] = -1;  e[0] = 4; e[1] = 0;
        CHECK(run_case(sg, u, a, b, e), "run1");
        a[0] = 0.5f; a[1] = 4;  b[0] = 2; b[1] = 0.25f;  e[0] = 1.5f; e[1] = 5;
        CHECK(run_case(sg, u, a, b, e), "rerun");
        sg.free_graph();
        CHECK(sg.out() == nullptr, "freed");
    }
    {   // 3: two holders sharing one gallocr (identical allocation plan)
        ggml_gallocr_t shared = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(cpu));
        CHECK(shared, "shared allocr");
        vqa::StepGraph s1, s2;
        TestUd u1, u2;
        u1.tag = 0; u2.tag = 0;   // sharing requires identical allocation plans
        CHECK(s1.build(cpu, test_builder, &u1, shared), "build s1");
        CHECK(s2.build(cpu, test_builder, &u2, shared), "build s2");
        // both compute a + a*b; interleaved runs must stay independent
        float a[2], b[2], e[2];
        a[0] = 1; a[1] = 2;  b[0] = 3; b[1] = -1;  e[0] = 4; e[1] = 0;
        CHECK(run_case(s1, u1, a, b, e), "s1 run");
        CHECK(run_case(s2, u2, a, b, e), "s2 run");
        a[0] = 2; a[1] = 2;  b[0] = 2; b[1] = 2;  e[0] = 6; e[1] = 6;
        CHECK(run_case(s1, u1, a, b, e), "s1 rerun (interleaved)");
        CHECK(run_case(s2, u2, a, b, e), "s2 rerun (interleaved)");
        s1.free_graph();
        s2.free_graph();
        ggml_gallocr_free(shared);
    }
    ggml_backend_free(cpu);
    if (g_fail) { fprintf(stderr, "step_graph test: %d FAILURES\n", g_fail); return 1; }
    fprintf(stderr, "step_graph test: all OK\n");
    return 0;
}
#endif
