// qwen3omni_pw_test.cpp - root-cause the "ggml graph path computes ConvNeXt
// pw wrong" mystery: feed the dumped LN activations (cnx_ln.f32, [1024,80])
// through a ggml-graph pw1+GELU+pw2 and compare with the host loops and the
// dumped reference (cnx_pw.f32).
#include "qwen3omni_talker.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <vector>

#ifdef _WIN32
#include <crtdbg.h>
#endif
static const bool _no_assert_dialog = []{
#ifdef _WIN32
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_DEBUG);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
#endif
    return true;
}();

static double cmp(const char * name, const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0, md = 0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
        double d = fabs((double)a[i] - b[i]); if (d > md) md = d;
    }
    printf("  %-24s corr %.6f maxdiff %.4g\n", name, dot / (sqrt(na) * sqrt(nb) + 1e-30), md);
    return dot / (sqrt(na) * sqrt(nb) + 1e-30);
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : "E:/SkyVision-weights/qwen3-omni/qwen3omni-talker-q8_0.gguf";
    const char * dir = argc > 2 ? argv[2] : "D:/moondream.cpp/parity/qwen3omni";
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    qwen3omni::Talker t;
    if (!t.load(path, backend, allocr)) return 1;

    const int C = 1024, T = 80;
    std::vector<float> ln((size_t)C * T), ref((size_t)C * T);
    { FILE * f = fopen((std::string(dir) + "/cnx_ln.f32").c_str(), "rb");
      if (!f || fread(ln.data(), 4, ln.size(), f) != ln.size()) { printf("cnx_ln read fail\n"); return 1; } fclose(f); }
    { FILE * f = fopen((std::string(dir) + "/cnx_pw.f32").c_str(), "rb");
      if (!f || fread(ref.data(), 4, ref.size(), f) != ref.size()) { printf("cnx_pw read fail\n"); return 1; } fclose(f); }

    ggml_tensor * w1 = t.up_cnx[0].pw1_w, * b1 = t.up_cnx[0].pw1_b;
    ggml_tensor * w2 = t.up_cnx[0].pw2_w, * b2 = t.up_cnx[0].pw2_b;
    printf("pw1 %s [%lld,%lld] pw2 %s [%lld,%lld]\n", ggml_type_name(w1->type),
           (long long)w1->ne[0], (long long)w1->ne[1], ggml_type_name(w2->type),
           (long long)w2->ne[0], (long long)w2->ne[1]);
    const int C4 = (int)w1->ne[1];

    // ---- graph path: x[C,T] -> mul_mat(w1) -> +b -> gelu -> mul_mat(w2) -> +b
    std::vector<float> out((size_t)C * T);
    {
        ggml_context * c = ggml_init({ 16 << 20, nullptr, true });
        ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, C, T);
        ggml_tensor * h = ggml_mul_mat(c, w1, x);
        if (b1) h = ggml_add(c, h, b1);
        h = ggml_gelu(c, h);
        ggml_tensor * y = ggml_mul_mat(c, w2, h);
        if (b2) y = ggml_add(c, y, b2);
        ggml_cgraph * gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, y);
        if (!ggml_gallocr_alloc_graph(allocr, gf)) return 1;
        // host activations are channel-major [ch][T]; mul_mat wants token-major
        std::vector<float> lnT((size_t)C * T);
        for (int ch = 0; ch < C; ch++) for (int tt = 0; tt < T; tt++)
            lnT[(size_t)tt * C + ch] = ln[(size_t)ch * T + tt];
        ggml_backend_tensor_set(x, lnT.data(), 0, sizeof(float) * lnT.size());
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return 1;
        std::vector<float> outT((size_t)C * T);
        ggml_backend_tensor_get(y, outT.data(), 0, sizeof(float) * outT.size());
        for (int ch = 0; ch < C; ch++) for (int tt = 0; tt < T; tt++)
            out[(size_t)ch * T + tt] = outT[(size_t)tt * C + ch];
        ggml_free(c);
    }
    cmp("graph vs dumped ref", out, ref);

    // ---- host reference (same loops as convnext())
    std::vector<float> w1c, b1c, w2c, b2c;
    {
        auto rd = [&](ggml_tensor * w, std::vector<float> & dst) {
            ggml_context * c2 = ggml_init({ 8 << 20, nullptr, true });
            int nr = (int)w->ne[1];
            std::vector<int> ids(nr); for (int q = 0; q < nr; q++) ids[q] = q;
            ggml_tensor * idt = ggml_new_tensor_1d(c2, GGML_TYPE_I32, nr);
            ggml_tensor * src = w->type == GGML_TYPE_F32 ? w : ggml_cast(c2, w, GGML_TYPE_F32);
            ggml_tensor * rr = ggml_get_rows(c2, src, idt);
            ggml_cgraph * g2 = ggml_new_graph(c2);
            ggml_build_forward_expand(g2, rr);
            ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            ggml_gallocr_alloc_graph(al, g2);
            ggml_backend_tensor_set(idt, ids.data(), 0, sizeof(int) * nr);
            ggml_backend_graph_compute(backend, g2);
            dst.resize(ggml_nelements(w));
            ggml_backend_tensor_get(rr, dst.data(), 0, dst.size() * 4);
            ggml_gallocr_free(al); ggml_free(c2);
        };
        rd(w1, w1c); rd(w2, w2c);
        for (ggml_tensor * b : { b1, b2 }) if (b) {
            std::vector<float> v(ggml_nelements(b));
            if (b->type == GGML_TYPE_F16) {
                std::vector<uint16_t> h2(v.size());
                ggml_backend_tensor_get(b, h2.data(), 0, h2.size() * 2);
                for (size_t i = 0; i < v.size(); i++) v[i] = ggml_fp16_to_fp32(h2[i]);
            } else ggml_backend_tensor_get(b, v.data(), 0, v.size() * 4);
            (b == b1 ? b1c : b2c) = v;
        }
        std::vector<float> mid((size_t)C4 * T);
        for (int o = 0; o < C4; o++) {
            const float * wr = &w1c[(size_t)o * C];
            for (int tt = 0; tt < T; tt++) {
                float acc = b1c[(size_t)o];
                for (int ch = 0; ch < C; ch++) acc += wr[ch] * ln[(size_t)ch * T + tt];
                mid[(size_t)o * T + tt] = acc;
            }
        }
        for (size_t i = 0; i < mid.size(); i++) {
            float v = mid[i];
            mid[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
        }
        std::vector<float> host((size_t)C * T);
        for (int o = 0; o < C; o++) {
            const float * wr = &w2c[(size_t)o * C4];
            for (int tt = 0; tt < T; tt++) {
                float acc = b2c[(size_t)o];
                for (int ch = 0; ch < C4; ch++) acc += wr[ch] * mid[(size_t)ch * T + tt];
                host[(size_t)o * T + tt] = acc;
            }
        }
        cmp("host vs dumped ref", host, ref);
        cmp("graph vs host", out, host);
    }
    return 0;
}
