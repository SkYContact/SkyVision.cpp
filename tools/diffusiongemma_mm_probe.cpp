// diffusiongemma-mm-probe - minimal reproducer for the batched mat-mul problem
// seen while checking the DiffusionGemma vision tower against its HF oracle.
//
// Feed a weight matrix [ne0 = in, ne1 = out] and an input whose 64 columns are
// ALL IDENTICAL: every output column must therefore be identical too. The probe
// runs the exact same graph on the CPU backend and (when built with CUDA) on the
// CUDA backend, so "engine bug" and "my graph is wrong" can be told apart.
//
//   build-cuda\diffusiongemma-mm-probe.exe
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#   include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const int NCOL = 64;

static int probe(ggml_backend_t be, const char * tag, int ne0, int ne1,
                 ggml_type wtype, ggml_type xtype) {
    ggml_init_params ip = { 256ull * 1024 * 1024, nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return 1;

    ggml_tensor * W = ggml_new_tensor_2d(ctx, wtype, ne0, ne1);
    ggml_set_name(W, "W");
    ggml_tensor * X = ggml_new_tensor_2d(ctx, xtype, ne0, NCOL);
    ggml_set_name(X, "X");
    // the engine marks both as graph inputs; mirror that (gallocr treats
    // ggml_set_input tensors differently from plain leaves)
    ggml_set_input(W);
    ggml_set_input(X);
    ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
    ggml_set_name(Y, "Y");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!al || !ggml_gallocr_alloc_graph(al, gf)) {
        printf("[%-22s] graph alloc failed\n", tag);
        return 1;
    }

    // deterministic values, all input columns identical
    std::vector<float> wf((size_t)ne0 * ne1);
    for (size_t i = 0; i < wf.size(); i++) wf[i] = 0.001f * ((int)(i % 97) - 48);
    std::vector<float> xf((size_t)ne0);
    for (int d = 0; d < ne0; d++) xf[d] = 0.01f * ((d % 31) - 15);

    auto upload = [&](ggml_tensor * t, const std::vector<float> & src, int rows) {
        if (t->type == GGML_TYPE_F32) {
            std::vector<float> buf((size_t)rows * src.size());
            for (int r = 0; r < rows; r++)
                memcpy(buf.data() + (size_t)r * src.size(), src.data(), src.size() * sizeof(float));
            ggml_backend_tensor_set(t, buf.data(), 0, buf.size() * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> buf((size_t)rows * src.size());
            for (int r = 0; r < rows; r++)
                for (size_t d = 0; d < src.size(); d++)
                    buf[(size_t)r * src.size() + d] = ggml_fp32_to_fp16(src[d]);
            ggml_backend_tensor_set(t, buf.data(), 0, buf.size() * sizeof(ggml_fp16_t));
        } else {
            return false;
        }
        return true;
    };
    // W is fully filled already (one "row" of upload); X has identical columns
    if (!upload(W, wf, 1) || !upload(X, xf, NCOL)) {
        printf("[%-22s] unsupported type pair\n", tag);
        return 1;
    }

    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        printf("[%-22s] compute failed\n", tag);
        return 1;
    }
    std::vector<float> y((size_t)ne1 * NCOL, 0.0f);
    ggml_backend_tensor_get(Y, y.data(), 0, y.size() * sizeof(float));

    double maxd = 0.0;
    int worst = 0;
    for (int c = 1; c < NCOL; c++)
        for (int o = 0; o < ne1; o++) {
            const double d = std::fabs((double)y[(size_t)c * ne1 + o] - (double)y[o]);
            if (d > maxd) { maxd = d; worst = c; }
        }
    const bool ok = maxd < 1e-3;
    printf("[%-22s] ne0=%-5d ne1=%-5d w=%-4s x=%-4s  max col-vs-col0 diff=%.3e "
           "(worst col %d)  %s\n",
           tag, ne0, ne1, ggml_type_name(wtype), ggml_type_name(xtype), maxd, worst,
           ok ? "OK" : "MISMATCH");
    if (!ok)
        printf("      col0[0..3]=%.5f %.5f %.5f %.5f   col%d[0..3]=%.5f %.5f %.5f %.5f\n",
               y[0], y[1], y[2], y[3], worst,
               y[(size_t)worst * ne1], y[(size_t)worst * ne1 + 1],
               y[(size_t)worst * ne1 + 2], y[(size_t)worst * ne1 + 3]);

    ggml_gallocr_free(al);
    ggml_free(ctx);
    return ok ? 0 : 2;
}

// Real-data mode: run the tower's first mul_mat with the actual weights from the
// GGUF and the actual patch columns the engine dumped, so the engine's numbers
// can be compared against the same op computed here.
static int probe_real(ggml_backend_t be, const char * gguf_path, const char * patches_path) {
    ggml_context * gctx = nullptr;
    gguf_init_params gp = { /*no_alloc*/ false, /*ctx*/ &gctx };
    gguf_context * gu = gguf_init_from_file(gguf_path, gp);
    if (!gu || !gctx) { printf("cannot open %s\n", gguf_path); return 1; }
    ggml_tensor * W = ggml_get_tensor(gctx, "v.patch_dense.weight");
    if (!W) { printf("v.patch_dense.weight not found\n"); return 1; }
    const int64_t ne0 = W->ne[0];      // 768
    const int64_t ne1 = W->ne[1];      // 1152

    std::vector<float> wf((size_t)ggml_nelements(W));
    ggml_get_type_traits(W->type)->to_float(W->data, wf.data(), ggml_nelements(W));

    FILE * fp = fopen(patches_path, "rb");
    if (!fp) { printf("cannot open %s\n", patches_path); return 1; }
    std::vector<float> pf((size_t)ne0 * NCOL);
    if (fread(pf.data(), sizeof(float), pf.size(), fp) != pf.size()) {
        printf("patch file too small\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);
    printf("real data: W type=%s ne=[%lld,%lld], patch[0][0..3]=%.5f %.5f %.5f %.5f\n",
           ggml_type_name(W->type), (long long)ne0, (long long)ne1,
           pf[0], pf[1], pf[2], pf[3]);

    ggml_init_params ip = { 256ull * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ne0, ne1);
    ggml_set_name(W2, "W2");
    ggml_set_input(W2);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, NCOL);
    ggml_set_name(X, "X");
    ggml_set_input(X);
    ggml_tensor * Y = ggml_mul_mat(ctx, W2, X);
    ggml_set_name(Y, "Y");
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(al, gf)) { printf("alloc failed\n"); return 1; }

    std::vector<ggml_fp16_t> w16(wf.size());
    for (size_t i = 0; i < wf.size(); i++) w16[i] = ggml_fp32_to_fp16(wf[i]);
    ggml_backend_tensor_set(W2, w16.data(), 0, w16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(X, pf.data(), 0, pf.size() * sizeof(float));
    ggml_backend_graph_compute(be, gf);
    std::vector<float> y((size_t)ne1 * NCOL, 0.0f);
    ggml_backend_tensor_get(Y, y.data(), 0, y.size() * sizeof(float));

    double maxd = 0.0;
    for (int c = 1; c < NCOL; c++)
        for (int o = 0; o < ne1; o++)
            maxd = std::max(maxd, (double)std::fabs(y[(size_t)c * ne1 + o] - (double)y[o]));
    printf("real data: col0[0..3]=%.5f %.5f %.5f %.5f  max col diff=%.3e\n",
           y[0], y[1], y[2], y[3], maxd);
    ggml_gallocr_free(al);
    ggml_free(ctx);
    ggml_free(gctx);
    gguf_free(gu);
    return 0;
}

int main(int argc, char ** argv) {
    struct ggml_backend_t_be { ggml_backend_t be; const char * tag; };
    std::vector<ggml_backend_t_be> backends;
    backends.push_back({ ggml_backend_cpu_init(), "cpu" });
#ifdef GGML_USE_CUDA
    if (ggml_backend_t cuda = ggml_backend_cuda_init(0)) backends.push_back({ cuda, "cuda:0" });
    else printf("(cuda backend unavailable)\n");
#else
    printf("(built without CUDA)\n");
#endif

    int bad = 0;
    for (auto & b : backends) {
        printf("--- %s ---\n", b.tag);
        if (!strcmp(b.tag, "cpu")) {
            // the engine sets a thread count; 1 does not reproduce the issue but
            // 16 may, and `use_ref` selects the scalar reference kernels
            ggml_backend_cpu_set_n_threads(b.be, 16);
            if (getenv("PROBE_REF")) ggml_backend_cpu_set_use_ref(b.be, true);
        }
        // the shapes the DiffusionGemma vision tower uses, plus the text width
        bad += probe(b.be, (std::string(b.tag) + " vision 768").c_str(), 768, 1152,
                     GGML_TYPE_F16, GGML_TYPE_F32) == 2;
        bad += probe(b.be, (std::string(b.tag) + " vision 768 f16in").c_str(), 768, 1152,
                     GGML_TYPE_F16, GGML_TYPE_F16) == 2;
        bad += probe(b.be, (std::string(b.tag) + " text 2816").c_str(), 2816, 1152,
                     GGML_TYPE_F16, GGML_TYPE_F32) == 2;
        bad += probe(b.be, (std::string(b.tag) + " vision 1152").c_str(), 1152, 1152,
                     GGML_TYPE_F16, GGML_TYPE_F32) == 2;
        ggml_backend_free(b.be);
    }
    if (argc >= 3) {
        printf("\n=== real checkpoint data ===\n");
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(cpu, 16);
        probe_real(cpu, argv[1], argv[2]);
#ifdef GGML_USE_CUDA
        if (ggml_backend_t cuda = ggml_backend_cuda_init(0)) {
            probe_real(cuda, argv[1], argv[2]);
            ggml_backend_free(cuda);
        }
#endif
        ggml_backend_free(cpu);
    }

    printf("\n%s: %d failing configuration(s)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
