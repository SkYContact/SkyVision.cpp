// vit_block.hpp - the SigLIP-style ViT block shared by both vision towers.
//
// moondream3.1's SigLIP and VideoChat3's ViT are the same animal: 27 layers,
// dim 1152, 16 heads (head_dim 72), LayerNorm(1e-5) with affine, fused qkv +
// bias, bidirectional attention over contiguous token groups (multi-crop /
// multi-clip, both unmasked FA), wo + bias, residual, and a GELU-tanh MLP.
// Only the attention wiring differs (2D rope on VideoChat3; grouping shape),
// so the block body lives here once and each backend supplies hooks.
//
// The block emits the exact op sequence both engines had inline; hooks cover
// dtype casting (moondream's MD_VISION_F16ACT) and layer-0 parity sinks
// (VC3_STOP_AFTER), so refactored graphs are numerically identical.
#pragma once

#include "ggml.h"

namespace vqa {
namespace vit {

struct BlockWeights {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;
    ggml_tensor *qkv_w = nullptr, *qkv_b = nullptr;
    ggml_tensor *wo_w = nullptr, *wo_b = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *fc0_w = nullptr, *fc0_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
};

struct BlockHooks {
    // required: rope/grouping/flash-attn over the [3*H*hd, ...] qkv projection,
    // returning attention output laid out as (head, hd) rows per token — i.e.
    // ready for the wo matmul. Returns nullptr if a layer-0 probe inside it
    // stopped the graph (VideoChat3's rope sinks).
    ggml_tensor *(*attention)(void *ud, ggml_tensor *qkv) = nullptr;
    // optional activation-dtype cast (moondream MD_VISION_F16ACT); null = identity
    ggml_tensor *(*to_act)(void *ud, ggml_tensor *t) = nullptr;
    // optional cast to F32 before the LayerNorms (f16act experiment); null = identity
    ggml_tensor *(*to_f32)(void *ud, ggml_tensor *t) = nullptr;
    // optional MLP activation; null = ggml_gelu (tanh). VideoMAE's nn.GELU() is the
    // exact erf form, so it passes ggml_gelu_erf here instead of writing its own block.
    ggml_tensor *(*gelu)(void *ud, ggml_tensor *t) = nullptr;
    // optional layer-0 parity sink, stage "qkv" | "attn_out" | "out"; return
    // true after handling the truncation (build/run/free) to stop the block
    bool (*probe_layer0)(void *ud, const char *stage, ggml_tensor *t) = nullptr;
    // opt-in LN affine folding (VideoChat3's experimental load-time folds):
    // drop the post-norm mul / add from the graph
    bool drop_ln_mul = false;
    bool drop_ln_add = false;
    void *ud = nullptr;
};

// x = x + Wo·Attn(LN1(x)); x = x + FC1·gelu_tanh(FC0·LN2(x)).
// Returns the next residual stream, or nullptr when a layer-0 probe stopped.
inline ggml_tensor * block(ggml_context *ctx, ggml_tensor *x, const BlockWeights &w,
                           float ln_eps, const BlockHooks &h, int layer) {
    auto act = [&](ggml_tensor *t) { return h.to_act ? h.to_act(h.ud, t) : t; };
    auto f32 = [&](ggml_tensor *t) { return h.to_f32 ? h.to_f32(h.ud, t) : t; };

    ggml_tensor *l = ggml_norm(ctx, f32(x), ln_eps);
    if (!h.drop_ln_mul) l = ggml_mul(ctx, l, f32(w.ln1_w));
    if (!h.drop_ln_add) l = ggml_add(ctx, l, f32(w.ln1_b));
    l = act(l);
    ggml_tensor *qkv = ggml_mul_mat(ctx, w.qkv_w, l);
    qkv = act(qkv);
    qkv = ggml_add(ctx, qkv, w.qkv_b);
    if (layer == 0 && h.probe_layer0 && h.probe_layer0(h.ud, "qkv", qkv)) return nullptr;

    ggml_tensor *attn = h.attention(h.ud, qkv);
    if (!attn) return nullptr;
    attn = ggml_mul_mat(ctx, w.wo_w, act(attn));
    attn = act(attn);
    attn = ggml_add(ctx, attn, w.wo_b);
    x = ggml_add(ctx, x, act(attn));   // residual: raw stream (f16act keeps f16+f16)
    if (layer == 0 && h.probe_layer0 && h.probe_layer0(h.ud, "attn_out", x)) return nullptr;

    ggml_tensor *l2 = ggml_norm(ctx, f32(x), ln_eps);
    if (!h.drop_ln_mul) l2 = ggml_mul(ctx, l2, f32(w.ln2_w));
    if (!h.drop_ln_add) l2 = ggml_add(ctx, l2, f32(w.ln2_b));
    l2 = act(l2);
    ggml_tensor *f = ggml_mul_mat(ctx, w.fc0_w, l2);
    f = act(f);
    f = ggml_add(ctx, f, w.fc0_b);
    f = h.gelu ? h.gelu(h.ud, f) : ggml_gelu(ctx, f);
    f = ggml_mul_mat(ctx, w.fc1_w, f);
    f = act(f);
    f = ggml_add(ctx, f, w.fc1_b);
    x = ggml_add(ctx, x, act(f));
    if (layer == 0 && h.probe_layer0 && h.probe_layer0(h.ud, "out", x)) return nullptr;
    return x;
}

// patch-embedding bias -> position table fold: host-side F32, so the graph
// drops one full [dim, tokens] add. pos is column-major [dim, tokens] (or a
// flat concatenation of such blocks), bias broadcasts along the fast dim.
inline void fold_patch_bias(float *pos, size_t n_pos, int dim, const float *bias) {
    for (size_t k = 0; k < n_pos; k++) pos[k] += bias[k % dim];
}

}  // namespace vit
}  // namespace vqa
