// minimal repro: does mark_sinks actually keep a tensor's buffer intact
// through a graph that reuses scratch aggressively?
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml_dump.hpp"
#include <cstdio>
#include <vector>
#include <cstring>

int main(int argc, char ** argv) {
    const int M = 24576, V = 884, K = 4096, KV = 1024;
    ggml_backend_t be = ggml_backend_cpu_init();
    // weight [K, M] F32
    ggml_context * wctx = ggml_init({ 64ull << 20, nullptr, true });
    ggml_tensor * W = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, M);
    std::vector<float> wdata((size_t)K * M);
    for (size_t i = 0; i < wdata.size(); i++) wdata[i] = (float)((i * 7919) % 1000) / 1000.f - 0.5f;

    struct ggml_init_params ip = { 128ull << 20, nullptr, true };
    ggml_context * gctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 4096, false);
    ggml_tensor * t_x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, V);
    ggml_tensor * a = ggml_mul_mat(gctx, W, t_x);            // [M, V]
    ggml_set_name(a, "sink_a");
    // cont(view) chain exactly like append_cross_kv
    ggml_tensor * b = ggml_cont(gctx, ggml_view_2d(gctx, a, KV, V, a->nb[1], 0));
    ggml_tensor * b3 = ggml_reshape_3d(gctx, b, 128, 8, V);
    ggml_set_name(b3, "sink_b");
    b3 = ggml_rms_norm(gctx, b3, 1e-6f);
    // plain NEOX rope (positions from a small I32 tensor)
    ggml_tensor * t_pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, V);
    int secs[GGML_MROPE_SECTIONS] = {0, 0, 0, 0};
    b3 = ggml_rope_multi(gctx, b3, t_pos, nullptr, 128, secs, GGML_ROPE_TYPE_NEOX,
                         0, 10000.f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * b2 = ggml_reshape_2d(gctx, b3, KV, V);
    // 12-layer loop writing into a persistent cache + a trailing main output,
    // mirroring the engine's cross-KV graph structure
    
    ggml_context * cctx = ggml_init({ 8ull << 20, nullptr, true });
    std::vector<ggml_tensor *> ckl, cvl;
    for (int i = 0; i < 12; i++) {
        ckl.push_back(ggml_new_tensor_2d(cctx, GGML_TYPE_F16, KV, 1024));
        cvl.push_back(ggml_new_tensor_2d(cctx, GGML_TYPE_F16, KV, 1024));
    }
    ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, be);
    ggml_tensor * t_rows = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, V);
    for (int li = 0; li < 12; li++) {
        ggml_tensor * kk = ggml_cont(gctx, ggml_view_2d(gctx, a, KV, V, a->nb[1],
                                                        (size_t)(li * 2) * KV * sizeof(float)));
        ggml_tensor * k3 = ggml_reshape_3d(gctx, kk, 128, 8, V);
        if (li == 0) ggml_set_name(k3, "sink_b");
        k3 = ggml_rms_norm(gctx, k3, 1e-6f);
        k3 = ggml_rope_multi(gctx, k3, t_pos, nullptr, 128, secs, GGML_ROPE_TYPE_NEOX,
                             0, 10000.f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * k2 = ggml_reshape_2d(gctx, k3, KV, V);
        if (li == 0) ggml_set_name(k2, "sink_k2");
        ggml_build_forward_expand(gf, ggml_set_rows(gctx, ckl[li], k2, t_rows));
        ggml_tensor * v2 = ggml_cont(gctx, ggml_view_2d(gctx, a, KV, V, a->nb[1],
                                                        (size_t)(li * 2 + 1) * KV * sizeof(float)));
        ggml_build_forward_expand(gf, ggml_set_rows(gctx, cvl[li], v2, t_rows));
        if (li == 0) {   // engine marks sinks mid-build, then keeps building
            static const char * const names[] = { "sink_a", "sink_b", "sink_k2", nullptr };
            vqa::dump::mark_sinks(gctx, gf, names);
        }
    }
    ggml_build_forward_expand(gf, ggml_scale(gctx, a, 1.0f));   // trailing main output
    vqa::dump::finalize_sinks(gctx, gf);   // NEW infra: attach keeps at the tail
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(al, gf);
    std::vector<float> x((size_t)K * V);
    for (size_t i = 0; i < x.size(); i++) x[i] = (float)((i * 104729) % 997) / 997.f - 0.5f;
    ggml_backend_tensor_set(W, wdata.data(), 0, sizeof(float) * wdata.size());
    ggml_backend_tensor_set(t_x, x.data(), 0, sizeof(float) * x.size());
    {
        std::vector<int> pos(V), rows(V);
        for (int i = 0; i < V; i++) { pos[i] = i; rows[i] = i; }
        ggml_backend_tensor_set(t_pos, pos.data(), 0, sizeof(int) * V);
        ggml_backend_tensor_set(t_rows, rows.data(), 0, sizeof(int) * V);
    }
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { printf("compute failed\n"); return 1; }

    std::vector<float> va((size_t)M * V), vb((size_t)KV * V);
    ggml_backend_tensor_get(vqa::dump::lookup(gf, "sink_a"), va.data(), 0, sizeof(float) * va.size());
    ggml_backend_tensor_get(vqa::dump::lookup(gf, "sink_b"), vb.data(), 0, sizeof(float) * vb.size());
    // sink_b should equal sink_a's first KV rows, token-major layout
    double maxd = 0; long bad = 0;
    for (int t = 0; t < V; t++)
        for (int d = 0; d < KV; d++) {
            float ref = va[(size_t)t * M + d];
            float got = vb[(size_t)t * KV + d];
            double dd = fabs((double)ref - got);
            if (dd > 1e-4) bad++;
            if (dd > maxd) maxd = dd;
        }
    printf("sink_b vs sink_a slice: max %.5f  bad %ld / %d\n", maxd, bad, V * KV);
    printf("VERDICT: %s\n", bad == 0 ? "SINK-OK" : "SINK-BROKEN");
    return 0;
}
