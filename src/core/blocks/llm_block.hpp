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
//   * moss_vl: DONE - cross-attention layers (and its self layers, whose IMROPE
//         rope/KV-slot split the single LayerCtx::pos cannot express) go through
//         hooks.attention + hooks.ffn_out; LayerWeights grew the cross fields
//         (cq/cqn/co + tanh gates ga/gm + the xattn flag). ck/cv/ckn stay
//         backend-side: the cross KV is projected once at load/media time into
//         the caller's own caches, never written from the text graph.
//   * qwen3_5: DONE - the 18 gated-delta linear-attention layers go through
//         hooks.attention (the hook receives the post-norm stream, exactly the
//         moss precedent; the "attention_in" split and after_layer plumbing
//         this note once asked for proved unnecessary). The recurrent states
//         (conv/ssm, 3 rollback slots) stay the caller's and are captured by
//         the hook closure; the 6 full-attention layers remain handwritten
//         in the backend (interleaved q|gate projection + IMROPE + the
//         per-token verify FA are shapes the block has no opinion on).
//         Attention-only layers (videojev trim: no post-norm/MLP at all) are
//         served by the !L.gate skip in block().
//   * smolvlm: DONE - it needed qn/kn null (no per-head norms) and that is all.
//         The "no rope" note here was wrong: SmolLM2 ropes (NEOX, theta 1e5), so
//         it passes pos like VideoChat3. mask==nullptr still always means an
//         explicit mask; no tenant relies on causal-by-construction yet.
//   * qwen25omni: DONE - the thinker needed three opt-ins, all default-off so
//         the earlier tenants' node streams are untouched: (1) LayerWeights
//         bq/bk/bv (Qwen2-style attention biases, added straight after each
//         q/k/v mul_mat); (2) LayerCtx::mrope_sec - rope goes through
//         ggml_rope_multi (MROPE, sections 16/24/24) with the 4T [t,h,w]
//         position input; (3) LayerCtx::rows - the thinker's KV slot index
//         (slot_in) is NOT its rope position (image blocks advance the MRoPE
//         index by max(grid_h, grid_w)), so set_rows takes rows when set. Its
//         activations are F32 against an F16 cache, so block() casts the
//         write-back to the cache dtype when they differ (the F16 tenants
//         never see the cast node). The FA query head-swap keeps block()'s
//         view_3d trick in place of the old cont(permute) copy - the view
//         addresses the very same bytes per FA row, and the omni parity set
//         (text/image/audio logits + greedy streams) stayed byte-identical.
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
    ggml_tensor *bq = nullptr, *bk = nullptr, *bv = nullptr;   // attention biases
    // (Qwen2-style); null = skip, the add node simply does not exist
    ggml_tensor *qn = nullptr, *kn = nullptr;          // per-head q/k norm (Qwen3); null = skip
    // cross-attention layer (moss_vl): the hook path projects cq itself and
    // adds tanh(ga) * co(...); cqn is the cross query per-head norm. null on
    // self-attention layers (flagged by xattn so the caller can route caches).
    ggml_tensor *cq = nullptr, *cqn = nullptr, *co = nullptr;
    ggml_tensor *ga = nullptr, *gm = nullptr;          // cross tanh gates [1]
    bool xattn = false;                                // true = cross-attention layer
    ggml_tensor *ffn_norm = nullptr;
    ggml_tensor *gate = nullptr, *up = nullptr, *down = nullptr;
};

// per-graph (not per-layer) build parameters; the cache tensors are the
// CALLER's - this file never allocates or owns KV state
struct LayerCtx {
    ggml_cgraph *gf = nullptr;         // for set_rows write-back expansion
    ggml_tensor *pos = nullptr;        // position ids [T] (MROPE tenants: the
                                       // [4T] t/h/w stack rope_multi wants);
                                       // null = no rope
    ggml_tensor *rows = nullptr;       // KV write-back row indices; null = pos.
                                       // Needed when the slot index is not the
                                       // rope position (qwen25omni: image rows
                                       // advance the MRoPE index by max(gh,gw))
    ggml_tensor *mask = nullptr;       // single-sequence F16 causal mask [S, T]
    ggml_tensor *k_cache = nullptr;    // this layer's cache [n_kv_head*head_dim, n_ctx]
    ggml_tensor *v_cache = nullptr;
    int T = 0, S = 0;                  // tokens this step, KV span to attend over
    int n_head = 0, n_kv_head = 0, head_dim = 0;
    int n_ctx = 0;                     // rope freq_base context
    float rms_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    const int *mrope_sec = nullptr;    // 3 MROPE sections; non-null switches the
                                       // rope to ggml_rope_multi (Qwen-VL/omni)
    // Load-time normalization folding (weight_load.hpp §4b LnFoldPlan: the
    // scale/bias of every RMSNorm was folded into the following projection
    // weights, W' = W diag(gamma), b' = b + W beta), so the graph must NOT
    // apply L.norm / L.ffn_norm again. Set from the GGUF metadata
    // "<arch>.ln_affine_folded" or from a load-time fold.
    bool norm_folded = false;
    // batch path: per-group (col_start, col_count, kv_base, window) plus the
    // pre-created per-group masks (shared across layers; null -> single path)
    const std::vector<std::array<int, 4>> *bat_groups = nullptr;
    const std::vector<ggml_tensor *> *bat_masks = nullptr;
};

struct BlockHooks {
    // ---- observe hooks (existing tenants): may inspect and name tensors but
    // must NOT replace them - return values are ignored, the stream is intact.
    void (*after_rope)(void *ud, int il, ggml_tensor *q, ggml_tensor *k) = nullptr;
    void (*after_attn)(void *ud, int il, ggml_tensor *attn) = nullptr;   // pre-wo
    void (*after_attn_proj)(void *ud, int il, ggml_tensor *x) = nullptr; // post-residual
    void (*after_layer)(void *ud, int il, ggml_tensor *x) = nullptr;     // block output
    // ---- override hooks (moss_vl): take over a whole segment of the block.
    // attention: given the post attention-norm stream l = rms_norm(x)*L.norm
    // [D, T], build the ENTIRE attention branch - projections, rope (any kind),
    // KV write-back (into whatever caches the backend owns; use c.gf for
    // ggml_build_forward_expand), flash attention and the output projection
    // incl. gates - and return its [D, T] output; the block only does
    // x = x + return_value. Return nullptr to skip the attention residual
    // entirely (moss cross layer with no vision: the reference zeroes the
    // branch, which is equivalent to not adding it).
    ggml_tensor *(*attention)(void *ud, int il, ggml_tensor *l, const LayerCtx &c) = nullptr;
    // ffn_out: wrap the MLP branch output m = down*swiglu(...) AFTER the down
    // projection, BEFORE the residual add (moss: *colmask *tanh(gm)). Return
    // the replacement, or nullptr to skip this layer's MLP residual.
    ggml_tensor *(*ffn_out)(void *ud, int il, ggml_tensor *m) = nullptr;
    // debug naming for the shared segments: return the name to stamp or
    // nullptr to leave unnamed. seg is "norm" (post attention-norm stream),
    // "resid" (post attention residual add, hook path only) or "out" (block
    // output, nullptr when ffn_out skipped the add). Only consulted when
    // non-null; when null, the legacy dbg_layer0 naming below applies.
    const char *(*dbg_name)(void *ud, int il, const char *seg) = nullptr;
    // false = separate silu()*up nodes (parity with VC3_NO_FUSION=1 graphs)
    bool swiglu_split = true;
    // stamp the layer-0 t_l0_* debug names (parity sinks look them up by name);
    // ignored when dbg_name is set (moss_vl names t_l0_*/t_x2_* through it)
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
    const bool dbg = !h.dbg_name && h.dbg_layer0 && il == 0;

    ggml_tensor * l = ggml_rms_norm(ctx, x, c.rms_eps);
    if (!c.norm_folded) l = ggml_mul(ctx, l, L.norm);
    if (const char * nm = h.dbg_name ? h.dbg_name(h.ud, il, "norm") : (dbg ? "t_l0_norm" : nullptr))
        ggml_set_name(l, nm);

    if (h.attention) {
        // override path: the hook owns the whole attention branch (see the
        // BlockHooks::attention note); the block only adds its output
        if (ggml_tensor * a = h.attention(h.ud, il, l, c)) {
            x = ggml_add(ctx, x, a);
            if (const char * nm = h.dbg_name ? h.dbg_name(h.ud, il, "resid") : nullptr)
                ggml_set_name(x, nm);
        }
    } else {
    ggml_tensor * q = ggml_mul_mat(ctx, L.q, l);
    if (L.bq) q = ggml_add(ctx, q, L.bq);
    ggml_tensor * k = ggml_mul_mat(ctx, L.k, l);
    if (L.bk) k = ggml_add(ctx, k, L.bk);
    ggml_tensor * v = ggml_mul_mat(ctx, L.v, l);
    if (L.bv) v = ggml_add(ctx, v, L.bv);
    if (dbg) { ggml_set_name(q, "t_l0_q"); ggml_set_name(k, "t_l0_k"); }

    ggml_tensor * q3 = ggml_reshape_3d(ctx, q, HD, NH, T);
    // per-head q/k RMSNorm is Qwen3-only; null (plain Qwen2 / Llama-style) skips
    // the whole reshape->norm->mul segment
    if (L.qn) {
        q3 = ggml_rms_norm(ctx, q3, c.rms_eps);
        q3 = ggml_mul(ctx, q3, L.qn);
    }
    ggml_tensor * k3 = ggml_reshape_3d(ctx, k, HD, NKV, T);
    if (L.kn) {
        k3 = ggml_rms_norm(ctx, k3, c.rms_eps);
        k3 = ggml_mul(ctx, k3, L.kn);
    }
    if (dbg && L.qn && L.kn) { ggml_set_name(q3, "t_l0_q_norm"); ggml_set_name(k3, "t_l0_k_norm"); }

    // pos null = no rope at all; NOTE pos also indexes the KV write-back below
    // (unless c.rows overrides it), so a rope-less tenant must still pass row
    // indices (or the cache fields must stay null with an out-of-block
    // write-back). mrope_sec switches the NEOX rope_ext for the multi-section
    // rope (same hyperparams; pos is then the [4T] t/h/w stack).
    if (c.pos) {
        if (c.mrope_sec) {
            int sec[4] = { c.mrope_sec[0], c.mrope_sec[1], c.mrope_sec[2], 0 };
            q3 = ggml_rope_multi(ctx, q3, c.pos, nullptr, HD, sec, GGML_ROPE_TYPE_MROPE,
                                 c.n_ctx, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k3 = ggml_rope_multi(ctx, k3, c.pos, nullptr, HD, sec, GGML_ROPE_TYPE_MROPE,
                                 c.n_ctx, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        } else {
            q3 = ggml_rope_ext(ctx, q3, c.pos, nullptr, HD, GGML_ROPE_TYPE_NEOX, c.n_ctx,
                               c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k3 = ggml_rope_ext(ctx, k3, c.pos, nullptr, HD, GGML_ROPE_TYPE_NEOX, c.n_ctx,
                               c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }
    }
    if (h.after_rope) h.after_rope(h.ud, il, q3, k3);
    if (dbg) { ggml_set_name(q3, "t_l0_q_rope"); ggml_set_name(k3, "t_l0_k_rope"); }

    // the cache may be a narrower dtype than the activation stream (qwen25omni:
    // F32 stream, F16 cache) - set_rows does not convert, so cast first; F16
    // tenants pass both sides F16 and never see these nodes
    ggml_tensor * k2 = ggml_reshape_2d(ctx, k3, NKV * HD, T);
    if (c.k_cache && k2->type != c.k_cache->type) k2 = ggml_cast(ctx, k2, c.k_cache->type);
    ggml_tensor * v2 = ggml_reshape_2d(ctx, v, NKV * HD, T);
    if (c.v_cache && v2->type != c.v_cache->type) v2 = ggml_cast(ctx, v2, c.v_cache->type);
    ggml_tensor * rows = c.rows ? c.rows : c.pos;
    ggml_build_forward_expand(c.gf, ggml_set_rows(ctx, c.k_cache, k2, rows));
    ggml_build_forward_expand(c.gf, ggml_set_rows(ctx, c.v_cache, v2, rows));

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
    }   // ---- end of the non-hook attention segment (kept verbatim)

    // attention-only layer (qwen3_5's VideoJev trim): no post-norm/MLP at all
    // - the block ends after the attention residual. Existing tenants always
    // pass both, so the guard never fires on their node streams.
    if (!L.ffn_norm || !L.gate) return x;
    ggml_tensor * m = ggml_rms_norm(ctx, x, c.rms_eps);
    if (!c.norm_folded) m = ggml_mul(ctx, m, L.ffn_norm);
    ggml_tensor * gate = ggml_mul_mat(ctx, L.gate, m);
    ggml_tensor * up = ggml_mul_mat(ctx, L.up, m);
    if (h.swiglu_split) {
        m = ggml_swiglu_split(ctx, gate, up);               // fused, no intermediate
    } else {
        m = ggml_mul(ctx, ggml_silu(ctx, gate), up);        // silu(gate) * up
    }
    if (dbg) ggml_set_name(m, "t_l0_swiglu");
    m = ggml_mul_mat(ctx, L.down, m);
    if (h.ffn_out) {
        ggml_tensor * m2 = h.ffn_out(h.ud, il, m);
        if (!m2) return x;    // this layer's MLP residual is skipped entirely
        m = m2;
    }
    x = ggml_add(ctx, x, m);
    if (h.after_layer) h.after_layer(h.ud, il, x);
    if (const char * nm = h.dbg_name ? h.dbg_name(h.ud, il, "out") : (dbg ? "t_l0_out" : nullptr))
        ggml_set_name(x, nm);
    return x;
}

}  // namespace llm
}  // namespace vqa
