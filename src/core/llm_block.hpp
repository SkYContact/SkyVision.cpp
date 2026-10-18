// llm_block.hpp - the Qwen-style causal decoder layer shared by the LLM towers.
//
// Companion to vit_block.hpp (see that file for the interface style): the 13
// hand-rolled per-layer decoder loops across the backends are being collapsed
// into this one block() graph builder. VideoChat3 (Qwen2/Qwen3 family:
// RMSNorm -> WQ/WK/WV -> optional per-head q/k RMSNorm -> NEOX rope ->
// flash_attn_ext over an F16 KV cache -> wo -> residual -> RMSNorm -> SwiGLU)
// is the first tenant; its layer body was moved here VERBATIM - same ops,
// same operand order, same node creation order (adjacent-node reordering can
// shift f16 results, so nothing was "cleaned up" in transit).
//
// Ownership split (same philosophy as vit_block):
//   * the BLOCK owns the op sequence above;
//   * the CALLER owns everything backend-specific: the KV cache tensors
//     (allocation, dtype, per-layer storage - passed in per layer), the causal
//     / batch masks (created once per graph, shared by all layers), position
//     ids, rope hyperparams, and the weight tensors (null fields are legal and
//     mean "skip that op": qn/kn null -> no per-head norm, i.e. plain Qwen2).
//   * BACKEND-ONLY behavior (dump-sink names, env-driven fusion toggles,
//     parity probes) hangs off BlockHooks, never off #defines in here.
//
// ---------------------------------------------------------------- migration
// roadmap - hooks each remaining backend needs before it can move in:
//   * moss_vl (cross-attention layers interleaved with self-attn layers):
//       - LayerWeights needs cross_attn_* fields (or a second LayerWeights
//         plus a LayerHooks::attention hook that overrides the FA call, since
//         cross-attn reads a DIFFERENT cache keyed by the encoder, not pos);
//       - a BlockHooks::skip_attn-like gate is NOT enough: cross layers also
//         omit rope and use encoder positions, so the rope/FA segment should
//         become overridable via hooks.attention (like vit_block) before
//         moss_vl migrates; kv write-back must be suppressible (no set_rows
//         into a self-attn cache).
//   * qwen3_5 (delta-net linear-attention hybrid layers): its recurrent layers
//         share norm/wo/SwiGLU but replace rope+FA+cache entirely with a
//         chunked delta-rule scan; needs the same hooks.attention override
//         (plus hooks.attention_in to feed it the post-norm stream rather
//         than per-head q/k/v, or a weights.fused_qkv_null fast path), and a
//         BlockHooks::after_layer for its per-layer recurrent state plumbing.
//   * smolvlm (no rope at all): zero new machinery - rope_pos == nullptr
//         should mean "skip rope" (today VideoChat3 always ropes); then it is
//         just LayerWeights with qn/kn null. Also the first chance to let
//         mask == nullptr mean causal-by-construction if its backend relies
//         on that (VideoChat3 always passes one).
#pragma once

#include "ggml.h"

#include <array>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

namespace vqa {
namespace llm {

// one decoder layer's weights; any field may be null (missing/shared weights
// are reported by the caller's own bind/load loop, not here)
struct LayerWeights {
    ggml_tensor *norm = nullptr;                       // attn_norm
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    ggml_tensor *qn = nullptr, *kn = nullptr;          // per-head q/k norm (Qwen3); null = skip
    ggml_tensor *ffn_norm = nullptr;
    ggml_tensor *gate = nullptr, *up = nullptr, *down = nullptr;
};

// per-graph (not per-layer) build parameters; the cache tensors are the
// CALLER's - this file never allocates or owns KV state
struct LayerCtx {
    ggml_cgraph *gf = nullptr;         // for set_rows write-back expansion
    ggml_tensor *pos = nullptr;        // position ids [T]
    ggml_tensor *mask = nullptr;       // single-sequence F16 causal mask [S, T]
    ggml_tensor *k_cache = nullptr;    // this layer's cache [n_kv_head*head_dim, n_ctx]
    ggml_tensor *v_cache = nullptr;
    int T = 0, S = 0;                  // tokens this step, KV span to attend over
    int n_head = 0, n_kv_head = 0, head_dim = 0;
    int n_ctx = 0;                     // rope freq_base context
    float rms_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    // batch path: per-group (col_start, col_count, kv_base, window) plus the
    // pre-created per-group masks (shared across layers; null -> single path)
    const std::vector<std::array<int, 4>> *bat_groups = nullptr;
    const std::vector<ggml_tensor *> *bat_masks = nullptr;
};

struct BlockHooks {
    // dump/probe anchors; all optional, all may inspect and name tensors but
    // must not replace them (return values keep the stream intact)
    void (*after_rope)(void *ud, int il, ggml_tensor *q, ggml_tensor *k) = nullptr;
    void (*after_attn)(void *ud, int il, ggml_tensor *attn) = nullptr;   // pre-wo
    void (*after_attn_proj)(void *ud, int il, ggml_tensor *x) = nullptr; // post-residual
    void (*after_layer)(void *ud, int il, ggml_tensor *x) = nullptr;     // block output
    // false = separate silu()*up nodes (parity with VC3_NO_FUSION=1 graphs)
    bool swiglu_split = true;
    // stamp the layer-0 t_l0_* debug names (parity sinks look them up by name)
    bool dbg_layer0 = true;
    void *ud = nullptr;
};

// x = x + Wo·FA(Rope(q/k-norm(WQ·LN(x)), WK·LN(x), WV·LN(x)))
// x = x + Wdown·SwiGLU(Wgate·LN2(x), Wup·LN2(x))
// Returns the next residual stream. The op sequence below is VideoChat3's
// build_text_graph layer body moved verbatim; keep node order stable.
inline ggml_tensor * block(ggml_context *ctx, ggml_tensor *x, int il,
                           const LayerWeights &L, const LayerCtx &c, const BlockHooks &h) {
    const int HD = c.head_dim, NH = c.n_head, NKV = c.n_kv_head, T = c.T, S = c.S;
    const bool dbg = h.dbg_layer0 && il == 0;

    ggml_tensor * l = ggml_rms_norm(ctx, x, c.rms_eps);
    l = ggml_mul(ctx, l, L.norm);
    if (dbg) ggml_set_name(l, "t_l0_norm");

    ggml_tensor * q = ggml_mul_mat(ctx, L.q, l);
    ggml_tensor * k = ggml_mul_mat(ctx, L.k, l);
    ggml_tensor * v = ggml_mul_mat(ctx, L.v, l);
    if (dbg) { ggml_set_name(q, "t_l0_q"); ggml_set_name(k, "t_l0_k"); }

    ggml_tensor * q3 = ggml_reshape_3d(ctx, q, HD, NH, T);
    q3 = ggml_rms_norm(ctx, q3, c.rms_eps);
    q3 = ggml_mul(ctx, q3, L.qn);
    ggml_tensor * k3 = ggml_reshape_3d(ctx, k, HD, NKV, T);
    k3 = ggml_rms_norm(ctx, k3, c.rms_eps);
    k3 = ggml_mul(ctx, k3, L.kn);
    if (dbg) { ggml_set_name(q3, "t_l0_q_norm"); ggml_set_name(k3, "t_l0_k_norm"); }

    q3 = ggml_rope_ext(ctx, q3, c.pos, nullptr, HD, GGML_ROPE_TYPE_NEOX, c.n_ctx,
                       c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k3 = ggml_rope_ext(ctx, k3, c.pos, nullptr, HD, GGML_ROPE_TYPE_NEOX, c.n_ctx,
                       c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    if (h.after_rope) h.after_rope(h.ud, il, q3, k3);
    if (dbg) { ggml_set_name(q3, "t_l0_q_rope"); ggml_set_name(k3, "t_l0_k_rope"); }

    ggml_tensor * k2 = ggml_reshape_2d(ctx, k3, NKV * HD, T);
    ggml_tensor * v2 = ggml_reshape_2d(ctx, v, NKV * HD, T);
    ggml_build_forward_expand(c.gf, ggml_set_rows(ctx, c.k_cache, k2, c.pos));
    ggml_build_forward_expand(c.gf, ggml_set_rows(ctx, c.v_cache, v2, c.pos));

    // q3 is [HD, NH, T]; flash attention wants [HD, n_q, n_head], so swap the
    // two trailing strides. The KV cache is [NKV*HD, n_ctx]: one position is
    // NKV*HD halves and one head inside it is HD halves.
    ggml_tensor * fa = nullptr;
    if (c.bat_groups) {
        // batch layout: each group is one sequence with its own KV window
        // view at its own base - attention work scales with the window, not
        // with n_ctx (the old single masked FA swept all 8192 slots for
        // every column, 7/8 of it masked-off garbage)
        for (size_t gi = 0; gi < c.bat_groups->size(); gi++) {
            const auto & gr = (*c.bat_groups)[gi];
            const int cs = gr[0], cc = gr[1], cbase = gr[2];
            // FA's CUDA kernels want the KV/mask width 32-aligned; pad the
            // window and hide the tail slots with the mask
            int cwin = (gr[3] + 31) & ~31;
            if (cbase + cwin > c.n_ctx) cwin = c.n_ctx - cbase;
            ggml_tensor * qs = ggml_view_3d(ctx, q3, HD, NH, cc, q3->nb[1], q3->nb[2],
                                            (size_t)cs * q3->nb[2]);
            ggml_tensor * qfs = ggml_view_3d(ctx, qs, HD, cc, NH, qs->nb[2], qs->nb[1], 0);
            ggml_tensor * kfs = ggml_view_3d(ctx, c.k_cache, HD, cwin, NKV,
                                             c.k_cache->nb[1], HD * sizeof(ggml_fp16_t),
                                             (size_t)cbase * c.k_cache->nb[1]);
            ggml_tensor * vfs = ggml_view_3d(ctx, c.v_cache, HD, cwin, NKV,
                                             c.v_cache->nb[1], HD * sizeof(ggml_fp16_t),
                                             (size_t)cbase * c.v_cache->nb[1]);
            // NOTE: look the mask up by NAME only at input-set time; inside
            // the build the tensors are not graph members yet, so use the
            // pre-created pointer directly
            ggml_tensor * ms = (*c.bat_masks)[gi];
            ggml_tensor * fs = ggml_flash_attn_ext(ctx, qfs, kfs, vfs, ms,
                                                   1.0f / sqrtf((float)HD), 0.0f, 0.0f);
            fa = fa ? ggml_concat(ctx, fa, fs, 1) : fs;
        }
    } else {
        ggml_tensor * qf = ggml_view_3d(ctx, q3, HD, T, NH, q3->nb[2], q3->nb[1], 0);
        ggml_tensor * kf = ggml_view_3d(ctx, c.k_cache, HD, S, NKV,
                                        c.k_cache->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * vf = ggml_view_3d(ctx, c.v_cache, HD, S, NKV,
                                        c.v_cache->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * fa2 = ggml_flash_attn_ext(ctx, qf, kf, vf, c.mask,
                                                1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        fa = fa2;
    }
    ggml_tensor * attn = ggml_reshape_2d(ctx, ggml_cont(ctx, fa), NH * HD, T);
    if (h.after_attn) h.after_attn(h.ud, il, attn);
    if (dbg) ggml_set_name(attn, "t_l0_attn");
    attn = ggml_mul_mat(ctx, L.o, attn);
    x = ggml_add(ctx, x, attn);
    if (h.after_attn_proj) h.after_attn_proj(h.ud, il, x);
    if (dbg) ggml_set_name(x, "t_l0_resid");

    ggml_tensor * m = ggml_rms_norm(ctx, x, c.rms_eps);
    m = ggml_mul(ctx, m, L.ffn_norm);
    ggml_tensor * gate = ggml_mul_mat(ctx, L.gate, m);
    ggml_tensor * up = ggml_mul_mat(ctx, L.up, m);
    if (h.swiglu_split) {
        m = ggml_swiglu_split(ctx, gate, up);               // fused, no intermediate
    } else {
        m = ggml_mul(ctx, ggml_silu(ctx, gate), up);        // silu(gate) * up
    }
    if (dbg) ggml_set_name(m, "t_l0_swiglu");
    m = ggml_mul_mat(ctx, L.down, m);
    x = ggml_add(ctx, x, m);
    if (h.after_layer) h.after_layer(h.ud, il, x);
    if (dbg) ggml_set_name(x, "t_l0_out");
    return x;
}

}  // namespace llm
}  // namespace vqa
