// whisper_tower.hpp - the post-LN Whisper encoder block shared by the audio
// towers (qwen25omni, qwen3omni, minicpmo). Companion to
// core/media/audio_feats.hpp, which already owns the shared mel frontend.
//
// The block body was moved VERBATIM out of the three backends (qwen25omni
// audio tower loop, minicpmo AudLayer loop, qwen3omni encoder loop): same
// ops, same operand order, same node creation order - the parity gates
// (a_l0_* dump sinks, tower cos gates) compare byte-for-byte, so nothing was
// "cleaned up" in transit. What the three towers really share is exactly this
// per-layer body; everything else stays the caller's:
//   * mel chunking, conv frontend, position tables (per-chunk restart vs
//     0..T2-1), attention masks (block-diag / window / 1s-cumulative) and
//     pooling/projection heads differ per model and are built by the caller;
//   * the F16 [T, T] mask is caller-owned and passed in;
//   * k_proj carries no bias in whisper (qwen25omni, minicpmo) - leave
//     LayerW::bk null and the add node simply does not exist; qwen3omni's
//     k_proj has one (pass it). No other shape is expressed here.
//
// Ownership split follows llm_block.hpp / vit_block.hpp: the BLOCK owns the
// op sequence above, the CALLER owns weights, masks, geometry and debug
// naming (BlockHooks::dbg_name, the llm_block precedent - observe-only, it
// stamps names and never replaces tensors; the a_l0_* sinks are looked up by
// name downstream, so the names are part of the contract).
#pragma once

#include "ggml.h"

#include <math.h>

namespace vqa {
namespace whisper {

// one encoder layer's weights; any field may be null where the model lacks
// it (bk null = no k bias add node - the whisper contract)
struct LayerW {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;    // self_attn_layer_norm
    ggml_tensor *wq = nullptr, *bq = nullptr;
    ggml_tensor *wk = nullptr, *bk = nullptr;          // bk null = whisper
    ggml_tensor *wv = nullptr, *bv = nullptr;
    ggml_tensor *wo = nullptr, *bo = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;    // final_layer_norm
    ggml_tensor *ff1_w = nullptr, *ff1_b = nullptr;    // fc1 / w_gate
    ggml_tensor *ff2_w = nullptr, *ff2_b = nullptr;    // fc2 / w_down
};

// observe-only debug naming (llm_block::BlockHooks::dbg_name precedent):
// seg is "q", "k", "attn", "resid" (post attention residual) or "out"
// (block output). Return the name to stamp or nullptr to leave unnamed.
struct BlockHooks {
    const char *(*dbg_name)(void *ud, int il, const char *seg) = nullptr;
    void *ud = nullptr;
};

// x [D, T] residual stream -> next residual stream.
//   x = x + Wo·FA(LN1(x)) ; x = x + W2·gelu_erf(W1·LN2(x))
// FA is flash_attn_ext over the caller's F16 mask with scale 1/sqrt(D/NH),
// k_proj bias only when L.bk is set. Node order below is the common
// qwen25omni/minicpmo/qwen3omni layer body - keep it stable.
inline ggml_tensor * block(ggml_context * ctx, ggml_tensor * x, int il,
                           const LayerW & L, ggml_tensor * mask,
                           int D, int n_head, int T, float ln_eps,
                           const BlockHooks & h) {
    const int HD = D / n_head;
    const float scale = 1.0f / sqrtf((float)HD);

    ggml_tensor * hh = ggml_norm(ctx, x, ln_eps);   // LayerNorm, carries bias
    hh = ggml_add(ctx, ggml_mul(ctx, hh, L.ln1_w), L.ln1_b);

    ggml_tensor * q = ggml_mul_mat(ctx, L.wq, hh);
    q = ggml_add(ctx, q, L.bq);
    ggml_tensor * k = ggml_mul_mat(ctx, L.wk, hh);
    if (L.bk) k = ggml_add(ctx, k, L.bk);           // qwen3omni only
    ggml_tensor * v = ggml_mul_mat(ctx, L.wv, hh);
    v = ggml_add(ctx, v, L.bv);
    if (const char * nm = h.dbg_name ? h.dbg_name(h.ud, il, "q") : nullptr)
        ggml_set_name(q, nm);
    if (const char * nm = h.dbg_name ? h.dbg_name(h.ud, il, "k") : nullptr)
        ggml_set_name(k, nm);

    ggml_tensor * q3 = ggml_reshape_3d(ctx, q, HD, n_head, T);
    ggml_tensor * k3 = ggml_reshape_3d(ctx, k, HD, n_head, T);
    ggml_tensor * v3 = ggml_reshape_3d(ctx, v, HD, n_head, T);
    ggml_tensor * qf = ggml_cont(ctx, ggml_permute(ctx, q3, 0, 2, 1, 3));
    ggml_tensor * kf = ggml_cont(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3));
    ggml_tensor * vf = ggml_cont(ctx, ggml_permute(ctx, v3, 0, 2, 1, 3));
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, qf, kf, vf, mask, scale, 0.0f, 0.0f);
    ggml_tensor * attn = ggml_reshape_2d(ctx, ggml_cont(ctx, fa), D, T);
    if (const char * nm = h.dbg_name ? h.dbg_name(h.ud, il, "attn") : nullptr)
        ggml_set_name(attn, nm);
    attn = ggml_mul_mat(ctx, L.wo, attn);
    attn = ggml_add(ctx, attn, L.bo);
    x = ggml_add(ctx, x, attn);
    if (const char * nm = h.dbg_name ? h.dbg_name(h.ud, il, "resid") : nullptr)
        ggml_set_name(x, nm);

    hh = ggml_norm(ctx, x, ln_eps);
    hh = ggml_add(ctx, ggml_mul(ctx, hh, L.ln2_w), L.ln2_b);
    ggml_tensor * g = ggml_mul_mat(ctx, L.ff1_w, hh);
    g = ggml_add(ctx, g, L.ff1_b);
    g = ggml_gelu_erf(ctx, g);
    g = ggml_mul_mat(ctx, L.ff2_w, g);
    g = ggml_add(ctx, g, L.ff2_b);
    x = ggml_add(ctx, x, g);
    if (const char * nm = h.dbg_name ? h.dbg_name(h.ud, il, "out") : nullptr)
        ggml_set_name(x, nm);
    return x;
}

}  // namespace whisper
}  // namespace vqa
