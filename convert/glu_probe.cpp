// minimal repro: q6_K gate/up mul_mat + geglu_split vs gelu+mul, CUDA, timed
#include "ggml.h"
#include "ggml-cuda.h"
#include <chrono>
#include <cstdio>
#include <vector>
#include <cstring>

int main() {
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) { printf("no cuda\n"); return 1; }
    const int D = 3840, FF = 15360, T = 26;
    ggml_context * wctx = ggml_init({(size_t)(128ull<<20), NULL, true});
    ggml_tensor * wg = ggml_new_tensor_2d(wctx, GGML_TYPE_Q6_K, D, FF);
    ggml_tensor * wu = ggml_new_tensor_2d(wctx, GGML_TYPE_Q6_K, D, FF);
    ggml_backend_buffer_t wb = ggml_backend_alloc_ctx_tensors(wctx, cuda);
    { std::vector<uint8_t> rnd(ggml_nbytes(wg)); for (auto&b:rnd) b=rand(); ggml_backend_tensor_set(wg, rnd.data(),0,rnd.size()); ggml_backend_tensor_set(wu, rnd.data(),0,rnd.size()); }

    for (int mode = 0; mode < 2; mode++) {
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            ggml_context * c = ggml_init({(size_t)(16ull<<20), NULL, true});
            ggml_cgraph * gf = ggml_new_graph(c);
            ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
            ggml_set_input(x); ggml_set_name(x, "x");
            ggml_tensor * g = ggml_mul_mat(c, wg, x);
            ggml_tensor * u = ggml_mul_mat(c, wu, x);
            ggml_tensor * act = mode ? (ggml_tensor*)ggml_geglu_split(c, g, u)
                                     : (ggml_tensor*)ggml_mul(c, ggml_gelu(c, g), u);
            ggml_tensor * out = ggml_mul_mat(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, FF, D), act);
            ggml_set_name(out, "out"); ggml_set_output(out);
            ggml_build_forward_expand(gf, out);
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cuda));
            ggml_gallocr_alloc_graph(ga, gf);
            std::vector<float> xin((size_t)D*T, 0.5f);
            ggml_backend_tensor_set(x, xin.data(), 0, sizeof(float)*xin.size());
            auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(cuda, gf);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
            if (ms < best) best = ms;
            ggml_gallocr_free(ga); ggml_free(c);
        }
        printf("%s: %.2f ms\n", mode ? "geglu_split" : "gelu+mul ", best);
    }
    return 0;
}
