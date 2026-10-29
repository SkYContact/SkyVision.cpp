// minicpmo_llm.cpp - the shared Qwen3/Llama branch builders (attention + FFN)
// used by the text trunk, the tk. TTS sub-tower and the synthetic parity
// driver. The block bodies follow llm_block.hpp's verbatim op order (VC3
// lineage): rms -> qkv (+ per-head q/k RMSNorm for Qwen3) -> NEOX rope ->
// F16 KV set_rows -> causal flash attention -> o -> residual side of the
// caller; ffn = rms -> swiglu -> down.
#include "minicpmo.hpp"

#include "ggml.h"

#include <cmath>

namespace minicpmo {

ggml_tensor * llm_attn_branch(ggml_context * ctx, ggml_cgraph * gf,
                              const LlmAttnW & w, ggml_tensor * k_cache,
                              ggml_tensor * v_cache, ggml_tensor * l,
                              ggml_tensor * mask, ggml_tensor * pos,
                              int n_head, int n_kv_head, int head_dim,
                              float rope_theta, int n_ctx, float rms_eps,
                              const char * dbg) {
    const int64_t D = l->ne[0];
    const int64_t T = l->ne[1];
    // the mask row count is the KV window contract (FA reads mask[ic] for ic
    // up to k->ne[1] with no bounds check - nemotronomni attn_branch lesson)
    const int64_t S = mask ? mask->ne[0] : k_cache->ne[1];
    const int64_t HD = head_dim, NH = n_head, NKV = n_kv_head;
    GGML_ASSERT(w.q->ne[0] == D && w.q->ne[1] == NH * HD);
    GGML_ASSERT(w.k->ne[0] == D && w.k->ne[1] == NKV * HD);
    GGML_ASSERT(w.v->ne[0] == D && w.v->ne[1] == NKV * HD);
    GGML_ASSERT(w.o->ne[0] == NH * HD && w.o->ne[1] == D);
    GGML_ASSERT(k_cache->ne[0] == NKV * HD && v_cache->ne[0] == NKV * HD);

    ggml_tensor * q = ggml_mul_mat(ctx, w.q, l);
    ggml_tensor * k = ggml_mul_mat(ctx, w.k, l);
    ggml_tensor * v = ggml_mul_mat(ctx, w.v, l);
    ggml_tensor * q3 = ggml_reshape_3d(ctx, q, HD, NH, T);
    ggml_tensor * k3 = ggml_reshape_3d(ctx, k, HD, NKV, T);
    if (w.qn) {
        q3 = ggml_rms_norm(ctx, q3, rms_eps);
        q3 = ggml_mul(ctx, q3, w.qn);
    }
    if (w.kn) {
        k3 = ggml_rms_norm(ctx, k3, rms_eps);
        k3 = ggml_mul(ctx, k3, w.kn);
    }
    if (dbg) { ggml_set_name(q3, (std::string(dbg) + "_q_norm").c_str()); }
    if (pos) {
        q3 = ggml_rope_ext(ctx, q3, pos, nullptr, (int)HD, GGML_ROPE_TYPE_NEOX,
                           n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_ext(ctx, k3, pos, nullptr, (int)HD, GGML_ROPE_TYPE_NEOX,
                           n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    if (dbg) { ggml_set_name(q3, (std::string(dbg) + "_q_rope").c_str()); }

    ggml_tensor * k2 = ggml_reshape_2d(ctx, k3, NKV * HD, T);
    ggml_tensor * v2 = ggml_reshape_2d(ctx, v, NKV * HD, T);
    if (k2->type != k_cache->type) k2 = ggml_cast(ctx, k2, k_cache->type);
    if (v2->type != v_cache->type) v2 = ggml_cast(ctx, v2, v_cache->type);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache, k2, pos));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache, v2, pos));

    ggml_tensor * kf = ggml_view_3d(ctx, k_cache, HD, S, NKV,
                                    k_cache->nb[1], HD * sizeof(ggml_fp16_t), 0);
    ggml_tensor * vf = ggml_view_3d(ctx, v_cache, HD, S, NKV,
                                    v_cache->nb[1], HD * sizeof(ggml_fp16_t), 0);
    ggml_tensor * qf = ggml_view_3d(ctx, q3, HD, T, NH, q3->nb[2], q3->nb[1], 0);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, qf, kf, vf, mask,
                                           1.0f / std::sqrt((float)HD),
                                           0.0f, 0.0f);
    ggml_tensor * attn = ggml_reshape_2d(ctx, ggml_cont(ctx, fa), NH * HD, T);
    if (dbg) ggml_set_name(attn, (std::string(dbg) + "_attn").c_str());
    return ggml_mul_mat(ctx, w.o, attn);
}

ggml_tensor * llm_ffn_branch(ggml_context * ctx, const LlmFfnW & w,
                             ggml_tensor * l, float rms_eps, const char * dbg) {
    GGML_ASSERT(w.gate && w.up && w.down);
    ggml_tensor * gate = ggml_mul_mat(ctx, w.gate, l);
    ggml_tensor * up = ggml_mul_mat(ctx, w.up, l);
    ggml_tensor * m = ggml_swiglu_split(ctx, gate, up);   // silu(gate) * up
    m = ggml_mul_mat(ctx, w.down, m);
    if (dbg) ggml_set_name(m, (std::string(dbg) + "_swiglu").c_str());
    return m;
}

}  // namespace minicpmo
