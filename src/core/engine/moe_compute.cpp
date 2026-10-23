// moe_compute.cpp - implements vqa::moe::build_expert_ffn (see moe_compute.hpp).
//
// The per-expert op sequence is copied verbatim from the verified keye decode
// path (keye_expert_decode.inc v1 graph B / the prefill mul_mat_id segment):
//   per expert k:  gu  = mul_mat(gate_k, x)          // [I, T]
//                  upr = mul_mat(up_k,   x)          // [I, T]
//                  a   = mul(unary(gu, act), upr)    // [I, T]
//                  f   = mul_mat(down_k, a)          // [D, T]
//   weighted sum:  term_k = mul(f, view(w, row k))   // [D, T] * [1, T] bcast
//   out = x + Σ term_k
// The v1 path used ggml_scale(f, w_host[k]); mul by a [1, T] view of the w
// input is the same single F32 multiply per element, so the math is unchanged
// while w becomes a per-step graph input (required for fixed-shape graphs).

#include "moe_compute.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace vqa {
namespace moe {

// General form (internal; declared where needed, NOT in the frozen header):
// the routed-FFN input x and the residual stream may be different tensors
// (keye: x = ffn_norm output, res = pre-FFN hidden). build_expert_ffn is the
// frozen-header wrapper with res = x.
ggml_tensor * build_expert_ffn_res(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * x,
                                   ggml_tensor * res,
                                   const std::vector<ggml_tensor *> & gate,
                                   const std::vector<ggml_tensor *> & up,
                                   const std::vector<ggml_tensor *> & down,
                                   ggml_tensor * w,
                                   ggml_unary_op act) {
    const size_t n_used = gate.size();
    if (!ctx || !gf || !x || !res || !w || n_used == 0) return nullptr;
    if (up.size() != n_used || down.size() != n_used) return nullptr;
    if (w->type != GGML_TYPE_F32) return nullptr;
    if (w->ne[0] != (int64_t)n_used) return nullptr;      // w [n_used, T]
    if (w->ne[1] != x->ne[1] || w->ne[2] != 1 || w->ne[3] != 1) return nullptr;
    if (res->ne[0] != x->ne[0] || res->ne[1] != x->ne[1]) return nullptr;
    for (size_t k = 0; k < n_used; k++) {
        if (!gate[k] || !up[k] || !down[k]) return nullptr;
        // ggml_can_mul_mat(w, x): w rows must match x rows (no public helper)
        if (gate[k]->ne[0] != x->ne[0] || up[k]->ne[0] != x->ne[0]) return nullptr;
        // down_k must accept the activation rows produced by gate/up_k and
        // produce rows matching x (for the residual add)
        if (down[k]->ne[0] != gate[k]->ne[1] || down[k]->ne[0] != up[k]->ne[1]) return nullptr;
        if (down[k]->ne[1] != x->ne[0]) return nullptr;
    }

    ggml_tensor * acc = nullptr;
    for (size_t k = 0; k < n_used; k++) {
        ggml_tensor * gu  = ggml_mul_mat(ctx, gate[k], x);
        ggml_tensor * upr = ggml_mul_mat(ctx, up[k],  x);
    // silu path uses ggml_swiglu_split (GGML_OP_GLU): mathematically
    // identical to mul(silu(a), b) but the CUDA backend only fuses
    // mul_mat+GLU / mul_mat_id+GLU subgraphs when the GLU op is used.
    ggml_tensor * a   = act == GGML_UNARY_OP_SILU
        ? ggml_swiglu_split(ctx, gu, upr)
        : ggml_mul(ctx, ggml_unary(ctx, gu, act), upr);
        ggml_tensor * f   = ggml_mul_mat(ctx, down[k], a);
        // expert k's weights for each token: elements (k, t), t = 0..T-1 -
        // a [1, T] view at offset k*nb[0] with row stride nb[1]
        ggml_tensor * wk  = ggml_view_2d(ctx, w, 1, w->ne[1],
                                         w->nb[1], (size_t)k * w->nb[0]);
        ggml_tensor * term = ggml_mul(ctx, f, wk);
        acc = acc ? ggml_add(ctx, acc, term) : term;
    }
    ggml_tensor * out = ggml_add(ctx, res, acc);
    ggml_build_forward_expand(gf, out);
    return out;
}

ggml_tensor * build_expert_ffn(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * x,
                               const std::vector<ggml_tensor *> & gate,
                               const std::vector<ggml_tensor *> & up,
                               const std::vector<ggml_tensor *> & down,
                               ggml_tensor * w,
                               ggml_unary_op act) {
    return build_expert_ffn_res(ctx, gf, x, x, gate, up, down, w, act);
}

// v3 slot-table form (internal general form; the frozen-header wrapper with
// res = x is build_expert_ffn_table - keye's residual is the pre-FFN hidden,
// not the ffn_norm output, mirroring build_expert_ffn_res above).
ggml_tensor * build_expert_ffn_table_res(ggml_context * ctx, ggml_cgraph * gf,
                                         ggml_tensor * x, ggml_tensor * res,
                                         ggml_tensor * gate_table,
                                         ggml_tensor * up_table,
                                         ggml_tensor * down_table,
                                         ggml_tensor * gate_up_ids,
                                         ggml_tensor * down_ids,
                                         ggml_tensor * w,
                                         ggml_unary_op act) {
    if (!ctx || !gf || !x || !res || !w) return nullptr;
    if (!gate_table || !up_table || !down_table || !gate_up_ids || !down_ids) return nullptr;
    if (w->type != GGML_TYPE_F32) return nullptr;
    const int64_t K = w->ne[0];                       // n_used
    const int64_t T = x->ne[1];
    if (K <= 0 || w->ne[1] != T || w->ne[2] != 1 || w->ne[3] != 1) return nullptr;
    if (res->ne[0] != x->ne[0] || res->ne[1] != T) return nullptr;
    if (gate_up_ids->type != GGML_TYPE_I32 || down_ids->type != GGML_TYPE_I32) return nullptr;
    if (gate_up_ids->ne[0] != K || down_ids->ne[0] != K) return nullptr;
    if (gate_up_ids->ne[1] != T || down_ids->ne[1] != T) return nullptr;
    // gate/up tables must be interchangeable ([I, D, n_slots], same shape and
    // stride, so the CUDA mul_mat_id+GLU fusion matcher accepts the pair)
    if (gate_table->type != up_table->type ||
        gate_table->ne[0] != up_table->ne[0] || gate_table->ne[1] != up_table->ne[1] ||
        gate_table->nb[1] != up_table->nb[1]) return nullptr;
    if (gate_table->ne[0] != x->ne[0]) return nullptr;         // input dim D
    if (up_table->ne[0]   != x->ne[0]) return nullptr;
    if (down_table->ne[0] != gate_table->ne[1]) return nullptr; // act rows = I
    if (down_table->ne[1] != x->ne[0]) return nullptr;          // out rows = D

    // gate/up mul_mat_ids must share src[1] (x3) AND src[2] (gate_up_ids);
    // build gate first so glu->src[0] is the gate output (fusion contract).
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, x->ne[0], 1, T);
    ggml_tensor * gu  = ggml_mul_mat_id(ctx, gate_table, x3, gate_up_ids);  // [I, K, T]
    ggml_tensor * upr = ggml_mul_mat_id(ctx, up_table,   x3, gate_up_ids);  // [I, K, T]
    ggml_tensor * a   = act == GGML_UNARY_OP_SILU
        ? ggml_swiglu_split(ctx, gu, upr)
        : ggml_mul(ctx, ggml_unary(ctx, gu, act), upr);                     // [I, K, T]
    ggml_tensor * f   = ggml_mul_mat_id(ctx, down_table, a, down_ids);      // [D, K, T]
    ggml_tensor * w3  = ggml_reshape_3d(ctx, w, 1, K, T);
    ggml_tensor * prod = ggml_mul(ctx, f, w3);                              // [D, K, T]
    ggml_tensor * pk = ggml_cont(ctx, ggml_permute(ctx, prod, 1, 0, 2, 3)); // [K, D, T]
    ggml_tensor * moe = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, pk), x->ne[0], T);
    ggml_tensor * out = ggml_add(ctx, res, moe);
    ggml_build_forward_expand(gf, out);
    return out;
}

ggml_tensor * build_expert_ffn_table(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * x,
                                     ggml_tensor * gate_table, ggml_tensor * up_table,
                                     ggml_tensor * down_table,
                                     ggml_tensor * gate_up_ids, ggml_tensor * down_ids,
                                     ggml_tensor * w,
                                     ggml_unary_op act) {
    return build_expert_ffn_table_res(ctx, gf, x, x, gate_table, up_table, down_table,
                                      gate_up_ids, down_ids, w, act);
}

}  // namespace moe
}  // namespace vqa

#ifdef MOE_COMPUTE_TEST_MAIN
// Self-contained numeric test (F32, hand-computed expectations).
//
// D=2, I=3, K=2, T=1. silu(z) = z / (1 + exp(-z)):
//   silu(2) = 1.7615942, silu(1) = 0.7310586, silu(0) = 0
// Expert 0: gu=[2,1,0], up=[1,2,3] -> act=[1.7615942, 1.4621172, 0]
//           down -> f=[1.7615942, 1.4621172], w0=0.5
// Expert 1: gu=[1,2,0], up=[2,1,0] -> act=[1.4621172, 1.7615942, 0]
//           down -> f=[1.7615942, 1.4621172], w1=1.0
// out = x + 0.5*f0 + 1.0*f1 = [1+0.8807971+1.7615942, 2+0.7310586+1.4621172]
//     = [3.6423913, 4.1931758]
// Also checks a nullptr return on mismatched w rows.
#include <cstdlib>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s (line %d)\n", msg, __LINE__); g_fail++; } } while (0)

int main(void) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    CHECK(cpu, "cpu backend");
    if (!cpu) return 1;

    struct ggml_init_params ip = { 16ull << 20, NULL, /*no_alloc*/ false };
    ggml_context * c = ggml_init(ip);
    CHECK(c, "ctx");

    const float gate0[6] = { 0,1, 1,0, 0,0 };   // [I=3 rows, D=2]
    const float up0_[6]  = { 1,0, 0,1, 1,1 };
    const float down0[6] = { 1,0,0, 0,1,0 };    // [D=2 rows, I=3]
    const float gate1[6] = { 1,0, 0,1, 0,0 };
    const float up1_[6]  = { 0,1, 1,0, 0,0 };
    const float down1[6] = { 0,1,0, 1,0,0 };
    const float xv[2]    = { 1.0f, 2.0f };
    const float wv[2]    = { 0.5f, 1.0f };

    auto mk2 = [&](const float * d, int r0, int r1) {
        ggml_tensor * t = ggml_new_tensor_2d(c, GGML_TYPE_F32, r0, r1);
        memcpy(t->data, d, sizeof(float) * r0 * r1);
        return t;
    };
    std::vector<ggml_tensor *> gate = { mk2(gate0, 2, 3), mk2(gate1, 2, 3) };
    std::vector<ggml_tensor *> up   = { mk2(up0_,  2, 3), mk2(up1_,  2, 3) };
    std::vector<ggml_tensor *> down = { mk2(down0, 3, 2), mk2(down1, 3, 2) };

    ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2, 1);
    memcpy(x->data, xv, sizeof(xv));
    ggml_tensor * w = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2, 1);
    memcpy(w->data, wv, sizeof(wv));

    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_tensor * out = vqa::moe::build_expert_ffn(c, gf, x, gate, up, down, w);
    CHECK(out, "build");
    if (out) {
        CHECK(ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS, "compute");
        const float * o = (const float *) out->data;
        const float exp0[2] = { 3.6423913f, 4.1931758f };
        CHECK(fabsf(o[0] - exp0[0]) < 1e-5f && fabsf(o[1] - exp0[1]) < 1e-5f,
              "values");
        fprintf(stderr, "moe out = [%.7f, %.7f] expected [%.7f, %.7f]\n",
                o[0], o[1], exp0[0], exp0[1]);
    }
    // bad-args: w with the wrong row count must return nullptr
    ggml_tensor * wbad = ggml_new_tensor_2d(c, GGML_TYPE_F32, 3, 1);
    ggml_cgraph * gf2 = ggml_new_graph(c);
    CHECK(vqa::moe::build_expert_ffn(c, gf2, x, gate, up, down, wbad) == nullptr,
          "bad w rows -> nullptr");

    ggml_free(c);
    ggml_backend_free(cpu);
    if (g_fail) { fprintf(stderr, "moe_compute test: %d FAILURES\n", g_fail); return 1; }
    fprintf(stderr, "moe_compute test: all OK\n");
    return 0;
}
#endif
