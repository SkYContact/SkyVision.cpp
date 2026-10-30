// moe_cpu_bench.cpp - standalone three-layer attribution bench for the
// nemotronomni CPU decode wall (opt campaign P1, phase 1).
//
// No model file needed: quantizes random weights with the engine's own
// ggml_quantize_chunk into the model's REAL (type, shape) pairs
// (nemotronomni q6k_mixed: routed experts q5_0 [2688x1856x128], shared expert
// q5_0 [2688x3712], mamba q5_0/q6_K, lm_head q8_0) and measures
//
//   [1] machine peak - warm 4 GiB memcpy bandwidth (read+write)
//   [2] kernel layer - the per-step GEMV op list, heap-resident (warm):
//       effective GB/s per op vs the memcpy peak; step totals
//   [3] graph layer  - the EC driver's per-step FRESH graph B (23x/step,
//       ggml_init+build+alloc+free per layer) vs a PERSISTENT build-once
//       graph with the same ops (the phase-2 candidate), outputs bitwise
//       compared; plus graph C (norm + lm_head) and build-only microcosts
//
// Bitwise gate: fresh-vs-persistent outputs must be byte-identical on the
// same inputs (same op sequence, same reduction order; only allocation
// addresses differ).
//
// Build against the repo's prebuilt CPU libs (no repo source rebuilt):
//   cl /O2 /MD /utf-8 /I ggml/include tools/moe_cpu_bench.cpp ^
//      /link build/ggml/src/Release/ggml-base.lib ^
//           build/ggml/src/Release/ggml-cpu.lib advapi32.lib
//
// Usage: moe_cpu_bench [threads]   (default 16)

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t).count();
}

double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// ---------------------------------------------------------------- section 1
void bench_memcpy() {
    const size_t NB = (size_t)4 << 30;   // 4 GiB >> LLC, warm heap
    char * a = (char *)malloc(NB);
    char * b = (char *)malloc(NB);
    if (!a || !b) { fprintf(stderr, "oom\n"); exit(1); }
    memset(a, 1, NB); memset(b, 2, NB);
    std::vector<double> ms;
    for (int it = 0; it < 6; it++) {
        const auto t0 = std::chrono::steady_clock::now();
        memcpy(b, a, NB);
        ms.push_back(ms_since(t0));
    }
    const double m = med(ms);
    fprintf(stderr, "[1] memcpy 4 GiB warm (1 thread): %.0f ms -> %.1f GB/s"
            " (rd+wr)\n", m, 2.0 * NB / m / 1e6);
    free(a); free(b);
}

// ---------------------------------------------------------------- section 2
struct Shape {
    const char * name;
    int64_t k;        // weight ne0 = row length (input dim)
    int64_t n;        // weight ne1 = output dim
    int64_t e;        // weight ne2 (128 for the expert tables, else 1)
    int rhs;          // b ne11 columns (6 for the down-proj batched GEMV)
    ggml_type type;
    int reps;         // occurrences per decode step
};

// weights kept hot in a process-lifetime backend buffer; a fresh graph per
// shape; timed op = graph_compute (the mul_mat[_id] dominates by far)
void bench_kernels(ggml_backend_t backend) {
    const Shape shapes[] = {
        { "routed_up",   2688, 1856, 128, 1, GGML_TYPE_Q5_0, 23 },
        { "routed_down", 1856, 2688, 128, 6, GGML_TYPE_Q5_0, 23 },
        { "shexp_up",    2688, 3712, 1,   1, GGML_TYPE_Q5_0, 23 },
        { "shexp_down",  3712, 2688, 1,   1, GGML_TYPE_Q5_0, 23 },
        { "ssm_in",      2688, 10304, 1,  1, GGML_TYPE_Q5_0, 23 },
        { "ssm_out",     4096, 2688, 1,   1, GGML_TYPE_Q6_K, 23 },
        { "attn_q",      2688, 4096, 1,   1, GGML_TYPE_Q5_0, 6 },
        { "attn_k",      2688, 256, 1,    1, GGML_TYPE_Q5_0, 6 },
        { "attn_v",      2688, 256, 1,    1, GGML_TYPE_Q5_0, 6 },
        { "attn_o",      4096, 2688, 1,   1, GGML_TYPE_Q6_K, 6 },
        { "lm_head",     2688, 131072, 1, 1, GGML_TYPE_Q8_0, 1 },
    };
    ggml_gallocr_t allocr = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));

    fprintf(stderr, "[2] %-12s %-5s %12s %10s %9s %8s %9s\n",
            "op", "wtype", "shape", "MBread", "ms/op", "GB/s", "ms/step");
    double step_bytes = 0, step_ms = 0;
    // dispatch floor first: a 2-node graph on a 4-element tensor
    {
        ggml_context * cf = ggml_init({ 1ull << 20, NULL, true });
        ggml_tensor * a = ggml_new_tensor_1d(cf, GGML_TYPE_F32, 4);
        ggml_tensor * s = ggml_sum(cf, a);
        ggml_cgraph * gf = ggml_new_graph_custom(cf, 8, false);
        ggml_build_forward_expand(gf, s);
        ggml_gallocr_t fa = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(fa, gf);
        std::vector<double> fm;
        for (int it = 0; it < 200; it++) {
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(backend, gf);
            fm.push_back(ms_since(t0));
        }
        fprintf(stderr, "[2] %-12s %-5s %12s %10s %9.3f %8s %9s\n",
                "floor", "-", "graph_compute", "-", med(fm), "-",
                "dispatch+sync");
        ggml_gallocr_free(fa);
        ggml_free(cf);
    }
    for (const Shape & sh : shapes) {
        ggml_init_params ip{ 64ull << 20, NULL, true };
        ggml_context * cw = ggml_init(ip);
        const int64_t ne[3] = { sh.k, sh.n, sh.e };
        ggml_tensor * w = ggml_new_tensor(cw, sh.type, sh.e > 1 ? 3 : 2, ne);
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(
            cw, ggml_backend_get_default_buffer_type(backend));
        {   // quantize random f32 into the real layout
            ggml_context * cq = ggml_init({ 64ull << 20, NULL, true });
            const int64_t fne[2] = { sh.k, sh.n * sh.e };
            ggml_tensor * f = ggml_new_tensor(cq, GGML_TYPE_F32, 2, fne);
            float * fd = (float *)malloc(ggml_nbytes(f));
            for (size_t i = 0; i < ggml_nelements(f); i++)
                fd[i] = (float)((rand() % 20001 - 10000) / 10000.0 * 0.05);
            ggml_quantize_chunk(sh.type, fd, w->data, 0, sh.n * sh.e, sh.k,
                                nullptr);
            free(fd);
            ggml_free(cq);
        }
        ggml_context * ctx = ggml_init({ 64ull << 20, NULL, true });
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.k, sh.rhs);
        ggml_tensor * r;
        ggml_tensor * ids = nullptr;
        if (sh.e > 1) {
            ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 6, 1);
            r = ggml_mul_mat_id(ctx, w, x, ids);            // [N, 6, 1]
        } else {
            r = ggml_mul_mat(ctx, w, x);                    // [N, rhs]
        }
        ggml_tensor * sink = ggml_sum(ctx, r);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(gf, sink);
        if (!ggml_gallocr_alloc_graph(allocr, gf)) { fprintf(stderr, "alloc\n"); exit(1); }
        { std::vector<float> z(sh.k * sh.rhs, 0.01f);
          ggml_backend_tensor_set(x, z.data(), 0, z.size() * sizeof(float)); }
        if (ids) {
            int32_t idv[6];
            for (int i = 0; i < 6; i++) idv[i] = (int32_t)((i * 17) % (int)sh.e);
            ggml_backend_tensor_set(ids, idv, 0, sizeof(idv));
        }

        std::vector<double> ms;
        for (int it = 0; it < 10; it++) {
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(backend, gf);
            ms.push_back(ms_since(t0));
        }
        const double m = med(ms);
        // honest read bytes: mul_mat_id reads only the 6 selected experts
        const double bytes = (double)ggml_nbytes(w) * (sh.e > 1 ? 6.0 / sh.e : 1.0);
        step_bytes += bytes * sh.reps;
        step_ms += m * sh.reps;
        fprintf(stderr, "[2] %-12s %-5s %5lld x %-6lld %10.2f %9.3f %8.1f %9.2f\n",
                sh.name, ggml_type_name(sh.type), (long long)sh.n,
                (long long)sh.k, bytes / 1e6, m, bytes / m / 1e6,
                m * sh.reps);
        ggml_free(ctx);
        ggml_backend_buffer_free(bw);
        ggml_free(cw);
    }
    fprintf(stderr, "[2] STEP TOTAL (kernels, warm heap): %.2f GB in %.2f ms"
            " -> %.1f GB/s effective\n", step_bytes / 1e9, step_ms,
            step_bytes / step_ms / 1e6);

    // [2b] per-call vs per-node overhead: N independent attn_k-shaped GEMVs
    // in ONE graph. fixed per-call cost -> total ~ flat; per-node cost ->
    // total ~ N x single-op time.
    {
        const int NS[] = { 1, 4, 16 };
        for (int ni = 0; ni < 3; ni++) {
            const int NOPS = NS[ni];
            ggml_init_params ip{ 64ull << 20, NULL, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_0, 2688, 256);
            ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2688);
            ggml_tensor * acc = nullptr;
            for (int i = 0; i < NOPS; i++) {
                ggml_tensor * r = ggml_mul_mat(ctx, w, x);
                acc = acc ? ggml_add(ctx, acc, r) : r;
            }
            ggml_tensor * sink = ggml_sum(ctx, acc);
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, 64, false);
            ggml_build_forward_expand(gf, sink);
            ggml_gallocr_t fa = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(fa, gf)) { fprintf(stderr, "a\n"); exit(1); }
            std::vector<float> z(2688, 0.01f);
            ggml_backend_tensor_set(x, z.data(), 0, z.size() * sizeof(float));
            std::vector<double> ms;
            for (int it = 0; it < 30; it++) {
                const auto t0 = std::chrono::steady_clock::now();
                ggml_backend_graph_compute(backend, gf);
                ms.push_back(ms_since(t0));
            }
            fprintf(stderr, "[2b] %2d x attn_k GEMV in one graph: %.3f ms"
                    " (%.3f ms/op)\n", NOPS, med(ms), med(ms) / NOPS);
            ggml_gallocr_free(fa);
            ggml_free(ctx);
        }
    }
    ggml_gallocr_free(allocr);
}

// ---------------------------------------------------------------- section 3
// exact nemotronomni graph-B replica (table branch of ec_build_b, VERBATIM
// op sequence) + graph C, fresh-per-step vs persistent build-once.
struct BWeight {
    ggml_tensor * up_t = nullptr, * down_t = nullptr;   // slot tables
    ggml_tensor * shu = nullptr, * shd = nullptr;       // shared expert
    ggml_tensor * out_norm = nullptr, * out_w = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

static void build_weights(ggml_backend_t backend, BWeight & wt,
                          int D, int I, int SHI, int NE, int V) {
    ggml_init_params ip{ 64ull << 20, NULL, true };
    ggml_context * cw = ggml_init(ip);
    const int64_t upn[3] = { D, I, NE }, dnn[3] = { I, D, NE };
    const int64_t sun[2] = { D, SHI }, sdn[2] = { SHI, D };
    const int64_t vn[2] = { D, V };
    wt.up_t = ggml_new_tensor(cw, GGML_TYPE_Q5_0, 3, upn);
    wt.down_t = ggml_new_tensor(cw, GGML_TYPE_Q5_0, 3, dnn);
    wt.shu = ggml_new_tensor(cw, GGML_TYPE_Q5_0, 2, sun);
    wt.shd = ggml_new_tensor(cw, GGML_TYPE_Q5_0, 2, sdn);
    wt.out_norm = ggml_new_tensor_1d(cw, GGML_TYPE_F32, D);
    wt.out_w = ggml_new_tensor(cw, GGML_TYPE_Q8_0, 2, vn);
    fprintf(stderr, "    bw: alloc ctx tensors...\n");
    wt.buf = ggml_backend_alloc_ctx_tensors_from_buft(
        cw, ggml_backend_get_default_buffer_type(backend));
    fprintf(stderr, "    bw: buffer %p, filling zero...\n", (void *)wt.buf);
    for (ggml_tensor * t : std::vector<ggml_tensor *>{ wt.up_t, wt.down_t,
         wt.shu, wt.shd, wt.out_norm, wt.out_w }) {
        std::vector<uint8_t> z(ggml_nbytes(t), 0);
        ggml_backend_tensor_set(t, z.data(), 0, ggml_nbytes(t));
    }
    fprintf(stderr, "    bw: done (ctx intentionally leaked: tensor metadata"
            " lives in it)\n");
    // NOTE: do NOT ggml_free(cw) - the ggml_tensor structs themselves live in
    // this no_alloc ctx; only their DATA is in wt.buf. Freeing it turns every
    // later t->ne read into use-after-free (observed: ne printed as 0x0x0x0).
}

// VERBATIM nemotronomni_expert_decode.inc ec_build_b (table branch):
// up -> relu^2 -> down through 2 mul_mat_id + dense shared expert + residual
static ggml_tensor * build_b_ops(ggml_context * cb, ggml_cgraph * fb,
                                 ggml_tensor * moe_in, ggml_tensor * xmid_in,
                                 ggml_tensor * w_in, ggml_tensor * ids_in,
                                 const BWeight & wt, int D, int K, int T) {
    ggml_tensor * m3 = ggml_reshape_3d(cb, moe_in, D, 1, T);
    ggml_tensor * gu = ggml_mul_mat_id(cb, wt.up_t, m3, ids_in);
    ggml_tensor * act = ggml_sqr(cb, ggml_relu(cb, gu));
    ggml_tensor * ff3 = ggml_mul_mat_id(cb, wt.down_t, act, ids_in);
    ggml_tensor * wb = ggml_reshape_3d(cb, w_in, 1, K, T);
    ggml_tensor * prod = ggml_mul(cb, ff3, wb);
    ggml_tensor * pk = ggml_cont(cb, ggml_permute(cb, prod, 1, 0, 2, 3));
    ggml_tensor * routed = ggml_reshape_2d(cb, ggml_sum_rows(cb, pk), D, T);
    ggml_tensor * sh = ggml_mul_mat(cb, wt.shu, moe_in);
    sh = ggml_sqr(cb, ggml_relu(cb, sh));
    sh = ggml_mul_mat(cb, wt.shd, sh);
    ggml_tensor * out = ggml_add(cb, xmid_in, ggml_add(cb, routed, sh));
    ggml_build_forward_expand(fb, out);
    return out;
}

struct BLeaves {
    ggml_tensor * moe_in, * xmid_in, * w_in, * ids_in;
};

static void fresh_graph_b(ggml_backend_t backend, ggml_gallocr_t allocr,
                          const BWeight & wt,
                          const float * hm, const float * hx, const float * hw,
                          const int32_t * hi,
                          float * hout, int D, int K,
                          double * t_build, double * t_alloc,
                          double * t_upload, double * t_compute,
                          double * t_read, double * t_free) {
    auto t00 = std::chrono::steady_clock::now();
    ggml_context * cb = ggml_init({ 16ull << 20, NULL, true });
    ggml_cgraph * fb = ggml_new_graph_custom(cb, 8192, false);
    // driver-owned input leaves (moe_decode_driver.hpp:487-489)
    BLeaves lv;
    lv.moe_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, D, 1);
    lv.xmid_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, D, 1);
    lv.w_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, K, 1);
    lv.ids_in = ggml_new_tensor_2d(cb, GGML_TYPE_I32, K, 1);
    ggml_tensor * hout_t = build_b_ops(cb, fb, lv.moe_in, lv.xmid_in, lv.w_in,
                                       lv.ids_in, wt, D, K, 1);
    *t_build += ms_since(t00);
    auto t01 = std::chrono::steady_clock::now();
    if (!ggml_gallocr_alloc_graph(allocr, fb)) { fprintf(stderr, "alloc\n"); exit(1); }
    *t_alloc += ms_since(t01);
    auto t02 = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(lv.moe_in, hm, 0, sizeof(float) * D);
    ggml_backend_tensor_set(lv.xmid_in, hx, 0, sizeof(float) * D);
    ggml_backend_tensor_set(lv.w_in, hw, 0, sizeof(float) * K);
    ggml_backend_tensor_set(lv.ids_in, hi, 0, sizeof(int) * K);
    *t_upload += ms_since(t02);
    auto t03 = std::chrono::steady_clock::now();
    ggml_backend_graph_compute(backend, fb);
    *t_compute += ms_since(t03);
    auto t04 = std::chrono::steady_clock::now();
    ggml_backend_tensor_get(hout_t, hout, 0, sizeof(float) * D);
    *t_read += ms_since(t04);
    ggml_free(cb);
    auto t05 = std::chrono::steady_clock::now();
    *t_free += ms_since(t05);
}

// persistent: build once, re-feed (phase-2 candidate pattern)
struct PersGraphB {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    BLeaves lv{};
    ggml_tensor * out = nullptr;
};

static bool persist_graph_b_build(ggml_backend_t backend, PersGraphB & pg,
                                  const BWeight & wt, int D, int K) {
    ggml_init_params ip{ 64ull << 20, NULL, true };
    pg.ctx = ggml_init(ip);
    pg.gf = ggml_new_graph_custom(pg.ctx, 8192, false);
    pg.lv.moe_in = ggml_new_tensor_2d(pg.ctx, GGML_TYPE_F32, D, 1);
    pg.lv.xmid_in = ggml_new_tensor_2d(pg.ctx, GGML_TYPE_F32, D, 1);
    pg.lv.w_in = ggml_new_tensor_2d(pg.ctx, GGML_TYPE_F32, K, 1);
    pg.lv.ids_in = ggml_new_tensor_2d(pg.ctx, GGML_TYPE_I32, K, 1);
    pg.out = build_b_ops(pg.ctx, pg.gf, pg.lv.moe_in, pg.lv.xmid_in,
                         pg.lv.w_in, pg.lv.ids_in, wt, D, K, 1);
    pg.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!pg.out || !ggml_gallocr_alloc_graph(pg.allocr, pg.gf)) return false;
    return true;
}

static void persist_graph_b_run(ggml_backend_t backend, PersGraphB & pg,
                                const float * hm, const float * hx,
                                const float * hw, const int32_t * hi,
                                float * hout, int D, int K,
                                double * t_upload, double * t_compute,
                                double * t_read) {
    auto t02 = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(pg.lv.moe_in, hm, 0, sizeof(float) * D);
    ggml_backend_tensor_set(pg.lv.xmid_in, hx, 0, sizeof(float) * D);
    ggml_backend_tensor_set(pg.lv.w_in, hw, 0, sizeof(float) * K);
    ggml_backend_tensor_set(pg.lv.ids_in, hi, 0, sizeof(int) * K);
    *t_upload += ms_since(t02);
    auto t03 = std::chrono::steady_clock::now();
    ggml_backend_graph_compute(backend, pg.gf);
    *t_compute += ms_since(t03);
    auto t04 = std::chrono::steady_clock::now();
    ggml_backend_tensor_get(pg.out, hout, 0, sizeof(float) * D);
    *t_read += ms_since(t04);
}

void bench_graphs(ggml_backend_t backend) {
    const int D = 2688, I = 1856, SHI = 3712, K = 6, NE = 128, V = 131072;
    BWeight wt;
    build_weights(backend, wt, D, I, SHI, NE, V);

    // host-side per-step inputs
    std::vector<float> hm(D), hx(D), hw(K);
    std::vector<int32_t> hi(K);
    for (int i = 0; i < D; i++) { hm[i] = 0.01f * (i % 17 - 8); hx[i] = 0.01f * (i % 13 - 6); }
    for (int i = 0; i < K; i++) { hw[i] = 1.0f / K; hi[i] = (int32_t)(i * 23 % NE); }
    BLeaves in{};   // unused in host-pointer form below
    std::vector<float> of(D), op(D);

    // -- fresh mode, 60 iters (the EC driver's per-step pattern) --
    double fb_ = 0, fa = 0, fu = 0, fc = 0, fr = 0, ff = 0;
    ggml_gallocr_t allocr = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    const int NF = 60;
    for (int it = 0; it < NF; it++)
        fresh_graph_b(backend, allocr, wt, hm.data(), hx.data(), hw.data(),
                      hi.data(), of.data(), D, K, &fb_, &fa, &fu, &fc, &fr, &ff);
    ggml_gallocr_free(allocr);
    fprintf(stderr, "[3] graph B FRESH  x23/step: build %.3f alloc %.3f"
            " upload %.3f compute %.3f read %.3f free %.3f ms/call"
            " -> x23: build %.1f alloc %.1f comp %.1f ms/step\n",
            fb_ / NF, fa / NF, fu / NF, fc / NF, fr / NF, ff / NF,
            fb_ / NF * 23, fa / NF * 23, fc / NF * 23);

    // -- persistent mode, 200 iters --
    PersGraphB pg;
    if (!persist_graph_b_build(backend, pg, wt, D, K)) { fprintf(stderr, "pgb\n"); exit(1); }
    double pu = 0, pc = 0, pr = 0;
    const int NP = 200;
    for (int it = 0; it < NP; it++)
        persist_graph_b_run(backend, pg, hm.data(), hx.data(), hw.data(),
                            hi.data(), op.data(), D, K, &pu, &pc, &pr);
    fprintf(stderr, "[3] graph B PERSIST x23/step: upload %.3f compute %.3f"
            " read %.3f ms/call -> x23 comp %.1f ms/step\n",
            pu / NP, pc / NP, pr / NP, pc / NP * 23);

    if (memcmp(of.data(), op.data(), sizeof(float) * D) != 0) {
        int nd = 0;
        for (int i = 0; i < D; i++) if (of[i] != op[i]) nd++;
        fprintf(stderr, "[3] BITWISE MISMATCH fresh vs persistent: %d/%d\n",
                nd, D);
    } else {
        fprintf(stderr, "[3] BITWISE OK fresh == persistent (%d floats)\n", D);
    }
    const double build_all = (fb_ + fa + ff) / NF * 23;
    fprintf(stderr, "[3] graph B build+alloc+free per step (23x): %.1f ms"
            " (persistent saves this entirely)\n", build_all);

    // ---- graph C replica: rms_norm + mul + lm_head mul_mat ----
    {
        ggml_gallocr_t ac = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        std::vector<float> lg(V), hid(D);
        for (int i = 0; i < D; i++) hid[i] = 0.01f * (i % 11 - 5);
        double cb_ = 0, cc = 0, cr = 0;
        const int NC = 20;
        for (int it = 0; it < NC; it++) {
            auto t00 = std::chrono::steady_clock::now();
            ggml_context * ccx = ggml_init({ 8ull << 20, NULL, true });
            ggml_cgraph * fc = ggml_new_graph_custom(ccx, 1024, false);
            ggml_tensor * h_in = ggml_new_tensor_2d(ccx, GGML_TYPE_F32, D, 1);
            ggml_tensor * fn = ggml_rms_norm(ccx, h_in, 1e-6f);
            fn = ggml_mul(ccx, fn, wt.out_norm);
            ggml_tensor * lgo = ggml_mul_mat(ccx, wt.out_w, fn);
            ggml_build_forward_expand(fc, lgo);
            if (!ggml_gallocr_alloc_graph(ac, fc)) { fprintf(stderr, "c-alloc\n"); exit(1); }
            cb_ += ms_since(t00);
            ggml_backend_tensor_set(h_in, hid.data(), 0, sizeof(float) * D);
            auto t01 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(backend, fc);
            cc += ms_since(t01);
            auto t02 = std::chrono::steady_clock::now();
            ggml_backend_tensor_get(lgo, lg.data(), 0, sizeof(float) * V);
            cr += ms_since(t02);
            ggml_free(ccx);
        }
        fprintf(stderr, "[3] graph C FRESH x1/step: build %.3f compute %.3f"
                " read %.3f ms\n", cb_ / NC, cc / NC, cr / NC);
        ggml_gallocr_free(ac);
    }

    // ---- build-only microcost: init+new_graph+build_b ops+free, no alloc ----
    {
        const int NB2 = 200;
        auto t00 = std::chrono::steady_clock::now();
        for (int it = 0; it < NB2; it++) {
            ggml_context * cb = ggml_init({ 16ull << 20, NULL, true });
            ggml_cgraph * fb = ggml_new_graph_custom(cb, 8192, false);
            BLeaves lv;
            lv.moe_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, D, 1);
            lv.xmid_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, D, 1);
            lv.w_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, K, 1);
            lv.ids_in = ggml_new_tensor_2d(cb, GGML_TYPE_I32, K, 1);
            build_b_ops(cb, fb, lv.moe_in, lv.xmid_in, lv.w_in, lv.ids_in,
                        wt, D, K, 1);
            ggml_free(cb);
        }
        const double m = ms_since(t00) / NB2;
        fprintf(stderr, "[3] build-only per graph B (init+nodes+free):"
                " %.3f ms -> x23 = %.1f ms/step\n", m, m * 23);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const int threads = argc > 1 ? atoi(argv[1]) : 16;
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "no cpu backend\n"); return 1; }
    ggml_backend_cpu_set_n_threads(backend, threads);
    fprintf(stderr, "== moe_cpu_bench: threads %d, nemotronomni q6k_mixed"
            " real shapes ==\n", threads);
    const bool graphs_only = argc > 2 && strcmp(argv[2], "graphs") == 0;
    if (!graphs_only) {
        bench_memcpy();
        bench_kernels(backend);
    }
    bench_graphs(backend);
    ggml_backend_free(backend);
    fprintf(stderr, "== done ==\n");
    return 0;
}
