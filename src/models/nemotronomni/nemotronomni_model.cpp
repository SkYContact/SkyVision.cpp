// nemotronomni_model.cpp - NVIDIA Nemotron-3-Nano-Omni-30B-A3B backend
// (text-chain graph segment, wave 3a: Mamba2 + NoPE GQA + MoE).
//
// Architecture truth (transcribed from ref/modeling_nemotron_h.py +
// ref/config.json; D:\S2TT\nemotron3omni-port\arch-spec.md is the normative
// table). All numbers below are the real-checkpoint values the GGUF mapping
// targets; the unit tests run scaled-down shapes with the same structure.
//
//   52 hybrid layers, hybrid_override_pattern
//   "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
//   = 23x Mamba2 (M) + 6x full attention (*, idx 5,12,19,26,33,42)
//   + 23x MoE (E). Every layer: x = x + mixer(rms_norm(x)), eps 1e-5,
//   NO post-norm (modeling_nemotron_h.py:1033-1052).
//
//   attention (NemotronHAttention :926-980):
//     GQA 32 q / 2 kv, head_dim 128, scale 128^-0.5, no biases, no sliding
//     window, no q/k norm. RoPE: the HF forward applies NO rope at all
//     (eager_attention_forward is a bare softmax attention, :900-922) -
//     NoPE. The rope.dimension_count=84 found in community GGUF headers is
//     converter metadata that llama.cpp's nemotron-h graph builder never
//     consumes. NEMOTRONOMNI_ROPE=partial84 compiles the alternative path
//     (NEOX over the first 84 of 128 head dims, theta 1e4) for the wave-D4
//     A/B ruling.
//
//   Mamba2 (NemotronHMamba2Mixer :241-702):
//     in_proj [10304, 2688] one block, split order [z 4096 | xBC 6144 | dt 64]
//     (torch_forward :505-508); z and dt never pass through the conv. xBC:
//     depthwise causal conv1d k=4 + bias (added outside ggml_ssm_conv), then
//     silu; split -> x 4096 | B 1024 | C 1024. heads 64 x head_dim 64,
//     n_groups 8 (B/C repeat_interleave: 8 heads share a group), d_state 128.
//     A = -exp(A_log) as a per-head scalar [1, 64] F32.
//     dt = softplus(dt_raw + dt_bias) then clamp(min 1e-3) (:601-602; the
//     upper clamp is commented out upstream). ggml_ssm_scan's Mamba-2 branch
//     applies softplus INTERNALLY and has no clamp, so the graph feeds it
//     inverse_softplus(clamp(softplus(dt+bias), 1e-3)) - bit-faithful to HF
//     where llama.cpp's mamba2 simply drops the clamp (NEMOTRONOMNI_DT_CLAMP=0
//     restores the upstream behavior for A/B).
//     y = state.C + x*D (D per head) -> y * silu(z) -> grouped RMSNorm
//     (8 groups x 512; Zamba2RMSNormGated multiplies the GATE BEFORE the
//     group norm, :679 + transformers modeling_zamba2.Zamba2RMSNormGated)
//     -> out_proj [2688, 4096].
//
//   MoE (NemotronHMoE :740-852):
//     router logits in fp32 (gate weights stored F32). score = sigmoid;
//     selection score = sigmoid + e_score_correction_bias (F32 buffer,
//     SELECTION ONLY - the routing weights are the unbiased sigmoid).
//     top-6 of 128 (n_group=1/topk_group=1: the group filter is a no-op).
//     weights = selected sigmoids / (sum + 1e-20) x routed_scaling 2.5.
//     Experts: NON-gated up[1856] -> relu^2 -> down[2688] (no gate branch).
//     Shared expert x1 (inter 3712), same relu^2 MLP on the same normed
//     input, added to the routed sum (NOT scaled by 2.5), one residual add.
//
//   embeddings [131072, 2688]; untied lm_head; final RMSNorm.
//
// GGUF tensor contract (wave 3b converter; llama.cpp nemotron_h_moe naming;
// 2D weights follow the ggml mul_mat orientation: ne[0] = IN, ne[1] = OUT,
// i.e. HF shapes reversed):
//   token_embd.weight [D, vocab]  output_norm.weight [D]  lm_head.weight [D, vocab]
//   blk.N.attn_norm.weight [D]                          (every layer's pre-norm)
//   M: blk.N.ssm_in.weight [D, d_in_proj]  blk.N.ssm_conv1d.weight [4, conv_dim]
//      blk.N.ssm_conv1d.bias [conv_dim]    blk.N.ssm_dt.bias [n_head]
//      blk.N.ssm_a [1, n_head] F32         (converter-computed -exp(A_log);
//                                           ssm_a_log fallback applies -exp here)
//      blk.N.ssm_d [1, n_head]             blk.N.ssm_norm.weight [d_inner/n_group, n_group]
//      blk.N.ssm_out.weight [d_inner, D]
//   *: blk.N.attn_q.weight [D, n_head*head_dim]   attn_k/attn_v [D, n_kv*head_dim]
//      blk.N.attn_o.weight [n_head*head_dim, D]
//   E: blk.N.ffn_gate_inp.weight [D, n_exp] F32
//      blk.N.ffn_exp_probs_b [n_exp] F32
//      blk.N.ffn_up_exps.weight [D, I, E]  blk.N.ffn_down_exps.weight [I, D, E]
//      blk.N.ffn_up_shexp.weight [D, Ish]  blk.N.ffn_down_shexp.weight [Ish, D]
// metadata: nemotronomni.block_count / embedding_length /
//   attention.head_count / attention.head_count_kv / attention.key_length /
//   attention.layer_norm_rms_epsilon / vocab_size / expert_count /
//   expert_used_count / expert_feed_forward_length /
//   expert_shared_feed_forward_length / ssm.conv_kernel / ssm.inner_size /
//   ssm.state_size / ssm.group_count / ssm.head_count / ssm.head_dim /
//   hybrid_override_pattern (raw string)
//
// Wave-3a scope: the text graph plus the three branch builders below - they
// are the unit-parity surface, driven by tools/nemotronomni_seg_test.cpp
// with synthetic weights and compared against numpy references. Wave 3b wired
// the real GGUF (metadata-conformant loader, checkpoint-side tokenizer,
// registry row, ChatML/raw ask()): the text end-to-end chain matches the
// official-stack oracle on prefill logits and greedy generation. Vision/audio
// towers and EC expert residency stay with wave 4.
#include "core/vqa.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#   include "ggml-cuda.h"
#endif
#include "core/text/tokenizer.hpp"
#include "core/engine/attention.hpp"
#include "core/engine/decode_graph.hpp"
#include "core/engine/sampler.hpp"
#include "core/util/aligned_buf.hpp"
#include "core/util/mmap.hpp"
#include "core/engine/backend.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/gguf_util.hpp"
#include "core/runtime/load_util.hpp"
#include "core/engine/weight_load.hpp"
#include "nemotronomni_text.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace nemo {

static const char * const DEFAULT_PATTERN =
    "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";

// ------------------------------------------------------------------ branches
// Each builder takes the post-norm stream and returns its [D, T] branch
// output; the caller adds the residual. `dbg` (null or e.g. "m0") prefixes
// the parity dump names.

// Mamba2 mixer. ggml_ssm_scan applies softplus(dt) internally, so with
// f_one non-null dt is pre-wrapped through
//   inverse_softplus(clamp(softplus(dt+bias), dt_min))
// to reproduce modeling_nemotron_h.py:601-602 exactly (the kernel then
// re-applies its own softplus). f_one = [1] F32 constant 1.0; pass null to
// drop the clamp (llama.cpp mamba2 semantics).
ggml_tensor * mamba2_branch(ggml_context * ctx, ggml_cgraph * gf,
                            const Mamba2W & w, Mamba2State & st,
                            ggml_tensor * l, ggml_tensor * ids,
                            float rms_eps, ggml_tensor * f_one, float dt_min,
                            const char * dbg) {
    const int64_t D = l->ne[0];
    const int64_t T = l->ne[1];
    const int64_t nh = w.a->ne[1];
    const int64_t d_inner = w.out_proj->ne[0];   // ggml ne: [in, out]
    const int64_t hd = d_inner / nh;             // head_dim, NOT the d vector
    const int64_t n_group = w.norm->ne[1];
    const int64_t d_state = st.ssm->ne[0];
    const int64_t conv_dim = w.conv1d->ne[1];
    const int64_t d_conv = w.conv1d->ne[0];
    GGML_ASSERT(d_inner % nh == 0);
    GGML_ASSERT(w.in_proj->ne[0] == D);
    GGML_ASSERT(w.in_proj->ne[1] == d_inner + conv_dim + nh);
    GGML_ASSERT(w.out_proj->ne[1] == D);
    GGML_ASSERT(conv_dim == d_inner + 2 * n_group * d_state);
    GGML_ASSERT(st.ssm->ne[1] == hd && st.ssm->ne[2] == nh);
    GGML_ASSERT((int64_t)st.conv->ne[0] == d_conv - 1 && st.conv->ne[1] == conv_dim);
    GGML_ASSERT(w.norm->ne[0] * n_group == d_inner);

    // in_proj -> [z | xBC | dt] row views over the [d_in_proj, T] result
    ggml_tensor * zxBCdt = ggml_mul_mat(ctx, w.in_proj, l);
    const size_t es = ggml_element_size(zxBCdt);
    ggml_tensor * z = ggml_view_3d(ctx, zxBCdt, hd, nh, T,
                                   hd * es, zxBCdt->nb[1], 0);
    ggml_tensor * xBC = ggml_view_2d(ctx, zxBCdt, conv_dim, T, zxBCdt->nb[1],
                                     d_inner * es);
    ggml_tensor * dt_raw = ggml_view_2d(ctx, zxBCdt, nh, T, zxBCdt->nb[1],
                                        (d_inner + conv_dim) * es);
    if (dbg) ggml_set_name(zxBCdt, (std::string(dbg) + "_inproj").c_str());

    // causal conv over xBC, rolling window (k-1 raw xBC rows) in front
    ggml_tensor * conv_st = ggml_reshape_3d(ctx, st.conv, d_conv - 1, conv_dim, 1);
    ggml_tensor * conv_in = ggml_concat(ctx, conv_st, ggml_transpose(ctx, xBC), 0);
    ggml_tensor * conv_last = ggml_view_3d(ctx, conv_in, d_conv - 1, conv_dim, 1,
                                           conv_in->nb[1], conv_in->nb[2],
                                           (size_t)(conv_in->ne[0] - (d_conv - 1))
                                               * conv_in->nb[0]);
    // state writes are serialized into the main chain with +0 edges: the CUDA
    // multi-stream scheduler may otherwise overlap the copy with later nodes
    // that reuse the source buffer (qwen3_5 tie_state, load bearing)
    ggml_tensor * state_dep = nullptr;
    auto tie_state = [&](ggml_tensor * cpy_out) {
        ggml_build_forward_expand(gf, cpy_out);
        ggml_tensor * s = ggml_scale(ctx, ggml_sum(ctx, cpy_out), 0.0f);
        state_dep = state_dep ? ggml_add(ctx, state_dep, s) : s;
    };
    tie_state(ggml_cpy(ctx, conv_last, conv_st));

    ggml_tensor * xBCc = ggml_ssm_conv(ctx, conv_in, w.conv1d);   // [conv_dim, T, 1]
    if (w.conv1d_b) xBCc = ggml_add(ctx, xBCc, w.conv1d_b);
    xBCc = ggml_silu(ctx, xBCc);
    if (dbg) ggml_set_name(xBCc, (std::string(dbg) + "_xbc").c_str());

    // dt: +bias -> softplus -> clamp(dt_min) -> inverse softplus
    ggml_tensor * dt = ggml_add(ctx, ggml_cont(ctx, dt_raw), w.dt_bias);
    if (f_one) {
        ggml_tensor * sp = ggml_softplus(ctx, dt);
        // the inverse round trip log(exp(sp)-1) overflows F32 once sp >= ~88
        // (real weights DO reach that) and returns exactly sp above sp ~ 17
        // (the -1 vanishes in F32 rounding), so clamp the upper side at 80:
        // semantically a no-op for every in-range value, and inf-proof
        sp = ggml_clamp(ctx, sp, dt_min, 80.0f);
        dt = ggml_log(ctx, ggml_sub(ctx, ggml_exp(ctx, sp), f_one));
    }
    dt = ggml_reshape_3d(ctx, dt, nh, T, 1);
    if (dbg) ggml_set_name(dt, (std::string(dbg) + "_dt").c_str());

    // ssm inputs: x/B/C strided views of the conv output (x | B | C)
    ggml_tensor * x = ggml_view_4d(ctx, xBCc, hd, nh, T, 1,
                                   hd * es, xBCc->nb[1], xBCc->nb[2], 0);
    ggml_tensor * B = ggml_view_4d(ctx, xBCc, d_state, n_group, T, 1,
                                   d_state * es, xBCc->nb[1], xBCc->nb[2],
                                   d_inner * es);
    ggml_tensor * C = ggml_view_4d(ctx, xBCc, d_state, n_group, T, 1,
                                   d_state * es, xBCc->nb[1], xBCc->nb[2],
                                   (d_inner + n_group * d_state) * es);
    ggml_tensor * s0 = ggml_reshape_4d(ctx, st.ssm, d_state, hd, nh, 1);
    ggml_tensor * res = ggml_ssm_scan(ctx, s0, x, dt, w.a, B, C, ids, /*K=*/1);
    ggml_set_name(res, "ssm_res");
    ggml_tensor * y = ggml_view_4d(ctx, res, hd, nh, T, 1,
                                   hd * es, nh * hd * es, nh * hd * T * es, 0);
    ggml_tensor * new_state =
        ggml_view_3d(ctx, res, d_state, hd, nh, d_state * es, d_state * hd * es,
                     ggml_nelements(x) * es);
    tie_state(ggml_cpy(ctx, new_state, st.ssm));

    // y + D*x (D broadcasts per head), then the z gate BEFORE the group norm
    ggml_tensor * dview = ggml_reshape_3d(ctx, w.d, 1, nh, 1);
    y = ggml_add(ctx, y, ggml_mul(ctx, x, dview));
    if (dbg) ggml_set_name(y, (std::string(dbg) + "_yd").c_str());
    y = ggml_swiglu_split(ctx, ggml_cont(ctx, z), y);

    // grouped RMSNorm (n_group x d_inner/n_group): after the reshape every
    // row of d_inner/n_group is one group
    y = ggml_reshape_4d(ctx, y, d_inner / n_group, n_group, T, 1);
    y = ggml_rms_norm(ctx, y, rms_eps);
    y = ggml_mul(ctx, y, w.norm);
    y = ggml_reshape_2d(ctx, y, d_inner, T);
    if (dbg) ggml_set_name(y, (std::string(dbg) + "_yn").c_str());

    ggml_tensor * out = ggml_mul_mat(ctx, w.out_proj, y);
    if (state_dep) out = ggml_add(ctx, out, state_dep);
    if (dbg) ggml_set_name(out, (std::string(dbg) + "_out").c_str());
    return out;
}

// Full attention: GQA, no q/k norm, causal mask, F16 KV cache, NoPE by
// default. pos_rope non-null switches on the partial-rope A/B path (NEOX
// over the first `rope_dims` of head_dim, llama.cpp GGUF metadata semantics).
ggml_tensor * attn_branch(ggml_context * ctx, ggml_cgraph * gf, const AttnW & w,
                          ggml_tensor * k_cache, ggml_tensor * v_cache,
                          ggml_tensor * l, ggml_tensor * mask,
                          ggml_tensor * pos_slot, ggml_tensor * pos_rope,
                          int rope_dims, float rope_theta, int n_ctx,
                          int n_head, int n_kv_head, int head_dim,
                          const char * dbg) {
    const int64_t D = l->ne[0];
    const int64_t T = l->ne[1];
    // KV window must match the mask width (FA reads mask[ic] for ic up to
    // k->ne[1] with no bounds check): viewing the whole n_ctx cache while the
    // causal mask only covers [0, n_past+T) made every attention read past the
    // mask buffer - benign zeros in most layouts, NaN garbage in the full
    // 52-layer T=1 decode layout. The mask row count is the window contract.
    const int64_t S = mask ? mask->ne[0] : k_cache->ne[1];
    const int64_t HD = head_dim, NH = n_head, NKV = n_kv_head;
    GGML_ASSERT(w.q->ne[0] == D && w.q->ne[1] == NH * HD);
    GGML_ASSERT(w.k->ne[0] == D && w.k->ne[1] == NKV * HD);
    GGML_ASSERT(w.v->ne[0] == D && w.v->ne[1] == NKV * HD);
    GGML_ASSERT(w.o->ne[0] == NH * HD && w.o->ne[1] == D);
    GGML_ASSERT(k_cache->ne[0] == NKV * HD && v_cache->ne[0] == NKV * HD);

    ggml_tensor * q = ggml_mul_mat(ctx, w.q, l);                  // [NH*HD, T]
    ggml_tensor * k = ggml_mul_mat(ctx, w.k, l);
    ggml_tensor * v = ggml_mul_mat(ctx, w.v, l);
    ggml_tensor * q3 = ggml_reshape_3d(ctx, q, HD, NH, T);
    ggml_tensor * k3 = ggml_reshape_3d(ctx, k, HD, NKV, T);
    if (dbg) { ggml_set_name(q, (std::string(dbg) + "_q").c_str()); }

    // NoPE by default: NO rope call anywhere on this path
    if (pos_rope) {
        q3 = ggml_rope_ext(ctx, q3, pos_rope, nullptr, rope_dims, GGML_ROPE_TYPE_NEOX,
                           n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_ext(ctx, k3, pos_rope, nullptr, rope_dims, GGML_ROPE_TYPE_NEOX,
                           n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    if (dbg) {
        ggml_set_name(q3, (std::string(dbg) + "_q_pre_fa").c_str());
        ggml_set_name(k3, (std::string(dbg) + "_k_pre_fa").c_str());
    }

    ggml_tensor * k2 = ggml_reshape_2d(ctx, k3, NKV * HD, T);
    ggml_tensor * v2 = ggml_reshape_2d(ctx, v, NKV * HD, T);
    if (k2->type != k_cache->type) k2 = ggml_cast(ctx, k2, k_cache->type);
    if (v2->type != v_cache->type) v2 = ggml_cast(ctx, v2, v_cache->type);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache, k2, pos_slot));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache, v2, pos_slot));

    ggml_tensor * kf = ggml_view_3d(ctx, k_cache, HD, S, NKV,
                                    k_cache->nb[1], HD * sizeof(ggml_fp16_t), 0);
    ggml_tensor * vf = ggml_view_3d(ctx, v_cache, HD, S, NKV,
                                    v_cache->nb[1], HD * sizeof(ggml_fp16_t), 0);
    ggml_tensor * qf = ggml_view_3d(ctx, q3, HD, T, NH, q3->nb[2], q3->nb[1], 0);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, qf, kf, vf, mask,
                                           1.0f / sqrtf((float)HD), 0.0f, 0.0f);
    ggml_tensor * attn = ggml_reshape_2d(ctx, ggml_cont(ctx, fa), NH * HD, T);
    if (dbg) ggml_set_name(attn, (std::string(dbg) + "_attn").c_str());

    ggml_tensor * out = ggml_mul_mat(ctx, w.o, attn);
    if (dbg) ggml_set_name(out, (std::string(dbg) + "_out").c_str());
    return out;
}

// MoE FFN: sigmoid routing in fp32, correction bias only in the selection,
// top-k, unbiased-sigmoid weights normalized then scaled, relu^2 non-gated
// experts, shared expert added on the same normed input.
ggml_tensor * moe_branch(ggml_context * ctx, const MoEW & w, ggml_tensor * m,
                         int n_used, float routed_scaling, const char * dbg) {
    const int64_t D = m->ne[0];
    const int64_t T = m->ne[1];
    const int64_t E = w.up_exps->ne[2];
    GGML_ASSERT(w.up_exps->ne[0] == D);
    GGML_ASSERT(w.down_exps->ne[0] == w.up_exps->ne[1]);
    GGML_ASSERT(w.down_exps->ne[2] == E);
    GGML_ASSERT(w.router->ne[0] == D && w.router->ne[1] == E);



    ggml_tensor * rlogits = ggml_mul_mat(ctx, w.router, m);            // [E, T] fp32
    ggml_tensor * sig = ggml_sigmoid(ctx, rlogits);
    // selection only; the weights stay the UNBIASED sigmoid
    ggml_tensor * sel_score = w.probs_b ? ggml_add(ctx, sig, w.probs_b) : sig;
    ggml_tensor * sel = ggml_top_k(ctx, sel_score, n_used);             // [K, T] I32
    ggml_tensor * wsel = ggml_get_rows(ctx, ggml_reshape_3d(ctx, sig, 1, E, T), sel);
    wsel = ggml_reshape_2d(ctx, wsel, n_used, T);                       // [K, T]
    ggml_tensor * wsum = ggml_sum_rows(ctx, wsel);                      // [1, T]
    if (w.f_eps) wsum = ggml_add(ctx, wsum, w.f_eps);
    ggml_tensor * w8 = ggml_div(ctx, wsel, wsum);
    w8 = ggml_scale(ctx, w8, routed_scaling);
    if (dbg) {
        ggml_set_name(rlogits, (std::string(dbg) + "_rlogits").c_str());
        ggml_set_name(sel, (std::string(dbg) + "_sel").c_str());
        ggml_set_name(w8, (std::string(dbg) + "_w8").c_str());
    }

    ggml_tensor * m3 = ggml_reshape_3d(ctx, m, D, 1, T);
    ggml_tensor * gu = ggml_mul_mat_id(ctx, w.up_exps, m3, sel);        // [I, K, T]
    ggml_tensor * act = ggml_sqr(ctx, ggml_relu(ctx, gu));              // relu^2, non-gated
    ggml_tensor * ff3 = ggml_mul_mat_id(ctx, w.down_exps, act, sel);    // [D, K, T]
    ggml_tensor * wb = ggml_reshape_3d(ctx, w8, 1, n_used, T);
    ggml_tensor * prod = ggml_mul(ctx, ff3, wb);                        // [D, K, T]
    ggml_tensor * pk = ggml_cont(ctx, ggml_permute(ctx, prod, 1, 0, 2, 3)); // [K, D, T]
    ggml_tensor * routed = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, pk), D, T);

    // shared expert (relu^2 MLP on the same normed input), added unscaled
    ggml_tensor * sh = ggml_mul_mat(ctx, w.shexp_up, m);
    sh = ggml_sqr(ctx, ggml_relu(ctx, sh));
    sh = ggml_mul_mat(ctx, w.shexp_down, sh);

    ggml_tensor * out = ggml_add(ctx, routed, sh);
    if (dbg) {
        ggml_set_name(routed, (std::string(dbg) + "_routed").c_str());
        ggml_set_name(sh, (std::string(dbg) + "_shexp").c_str());
        ggml_set_name(out, (std::string(dbg) + "_out").c_str());
    }
    return out;
}

// ------------------------------------------------------------------ model
struct TxtLayer {
    char kind = 'M';                   // 'M' mamba2, '*' attention, 'E' moe
    ggml_tensor *norm = nullptr;       // attn_norm (the layer pre-norm)
    Mamba2W m;
    AttnW a;
    MoEW e;
};

struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *pos_in = nullptr, *mask_in = nullptr;
    ggml_tensor *ssm_ids = nullptr;
    ggml_tensor *logits = nullptr;
};

class NemotronOmni : public vqa::Model {
public:
    NemotronOmni() { caps_.image = true; caps_.video = true; caps_.audio = true; }
    ~NemotronOmni() override;
    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "nemotron-3-nano-omni"; }
    // wave 3b: text-only ask(); media injection lands with the towers
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp,
             std::string & out_text) override;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool alloc_state();
    bool build_text_graph(TextGraph & g, int T, int S, bool logits_all = false);
    bool run_text(TextGraph & g, const std::vector<int> & toks,
                  const std::vector<int> & pos, std::vector<float> & logits_out);
    bool generate(const std::vector<int> & ids, const vqa::GenParams & gp,
                  std::vector<int> & out_ids, std::string & out_text);
    int sample(const std::vector<float> & logits, const vqa::GenParams & gp);
    void dump(ggml_cgraph * gf, const char * name);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    // real-checkpoint dims (arch-spec §2); the GGUF metadata overrides
    int n_layer = 52, n_embd = 2688, n_head = 32, n_kv_head = 2, head_dim = 128;
    int n_vocab = 131072;
    int n_expert = 128, n_expert_used = 6, n_ff_exp = 1856, n_ff_shexp = 3712;
    // Mamba2: 64 heads x 64, n_groups 8, d_state 128, conv 4
    int ssm_n_head = 64, ssm_head_dim = 64, ssm_n_group = 8, ssm_d_state = 128,
        ssm_d_conv = 4;
    float rms_eps = 1e-5f;
    float routed_scaling = 2.5f;
    std::string pattern = DEFAULT_PATTERN;
    // rope ruling (port-plan §5): off = NoPE (default); partial84 = the
    // llama.cpp-GGUF-metadata alternative, kept compilable for the D4 A/B
    bool rope_partial84 = false;
    int rope_dims = 84;
    float rope_theta = 1e4f;
    // HF dt clamp(min=1e-3); off restores llama.cpp semantics.
    // WAVE-3B FINDING, re-verified post mask-window fix: default OFF. The
    // wrap is stable now (the wave-3b "layout flip" was the attention
    // mask/KV-window mismatch, not the wrap's extra nodes), but clamp-ON
    // sits ~10x further from the official-stack oracle logits (mean_abs
    // ~1.3 vs ~0.11 on the prompt A gen steps), so the checkpoint is best
    // served without the clamp. With the wrap off the full chain matches a
    // float64 reference to F16 noise (K=1 max_abs 0.0026, all argmax equal).
    bool dt_clamp = false;
    float dt_min = 1e-3f;
    int eos_id = 11;                  // metadata eos_token_id (</s> = 2 is the
                                      // generation_config secondary stop)
    bool logits_all_env = false;      // NEMOTRONOMNI_LOGITS_ALL: prefill fetches
                                      // the whole [vocab, T] logits block

    int n_ctx = 8192;
    bool use_gpu = false;
    bool split_mode = false;      // EC wave: `reuse = decode_step && !split_mode`
    int n_threads = 0;
    std::string dump_dir;
    std::string model_dir_hint;

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    vqa::AlignedBuf host_f32_storage;    // promoted 1D (+ ssm_norm/ssm_a) tensors
    vqa::AlignedBuf a_storage;           // computed A = -exp(A_log) blocks
    vqa::MMap mm;

    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr, *out_w = nullptr;
    std::vector<TxtLayer> tl;
    int n_mamba = 0, n_attn = 0;

    vqa::KvCache kv;                     // the 6 attention layers only
    // persistent Mamba2 states: one conv+ssm pair per M layer (v1 single
    // rollback slot; the qwen3_5 3-slot machinery returns with MTP)
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    std::vector<Mamba2State> st;
    ggml_tensor *f_one = nullptr;        // [1] F32 = 1.0 (dt clamp round trip)
    ggml_tensor *f_eps = nullptr;        // [1] F32 = 1e-20 (router denom)

    ggml_gallocr_t allocr = nullptr;
    vqa::GraphSlot<TextGraph> dec_slot;  // bucketed persistent decode graph
    int n_past = 0;
    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

NemotronOmni::~NemotronOmni() {
    dec_slot.reset();
    kv.free();
    if (sbuf) ggml_backend_buffer_free(sbuf);
    if (sctx) ggml_free(sctx);
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (allocr) ggml_gallocr_free(allocr);
    if (backend) ggml_backend_free(backend);
}

bool NemotronOmni::init_backend() {
    if (getenv("NEMOTRONOMNI_GPU")) use_gpu = true;
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

bool NemotronOmni::load(const std::string & gguf_path, const std::string & tokenizer_path,
                        const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "NEMO", n_threads, use_gpu, dump_dir);
    rope_partial84 = getenv("NEMOTRONOMNI_ROPE") &&
                     strcmp(getenv("NEMOTRONOMNI_ROPE"), "partial84") == 0;
    if (getenv("NEMOTRONOMNI_DT_CLAMP")) dt_clamp = atoi(getenv("NEMOTRONOMNI_DT_CLAMP")) != 0;
    logits_all_env = getenv("NEMOTRONOMNI_LOGITS_ALL") != nullptr;
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    // tokenizer: the checkpoint ships the Mistral/Pixtral byte-BPE family
    // (131072 vocab). Wave-3b pinning (R10): the engine Qwen2Regex pre + the
    // shared byte-level BPE merge loop reproduce the oracle input_ids bit for
    // bit on both parity prompts (gate 1); the Mistral split regex differs
    // from the Qwen2 one only around contractions/mixed-case runs, which the
    // HF tokenizers build behind this checkpoint never exercises here.
    if (!tokenizer_path.empty() && !tok_.load(tokenizer_path.c_str(),
                                              vqa::TokenizerMode::Qwen2Regex))
        return false;
    if (tok_.size() > 0) {
        const int64_t diff = tok_.size() > n_vocab ? tok_.size() - (int64_t)n_vocab
                                                   : (int64_t)n_vocab - tok_.size();
        if (diff * 5 > n_vocab) {
            fprintf(stderr, "nemotronomni: tokenizer vocab %d vs model vocab %d"
                            " - wrong tokenizer.json\n", tok_.size(), n_vocab);
            return false;
        }
    }
    if (!kv.alloc(backend, n_attn, n_kv_head, head_dim, n_ctx)) return false;
    if (!alloc_state()) return false;
    fprintf(stderr, "nemotronomni: %d layers d=%d (%d mamba / %d attn / %d moe),"
                    " experts %d/%d+%d x%.2f, ssm %dx%d g%d s%d%s%s eos=%d\n",
            n_layer, n_embd, n_mamba, n_attn, n_layer - n_mamba - n_attn,
            n_expert_used, n_expert, n_ff_shexp, routed_scaling,
            ssm_n_head, ssm_head_dim, ssm_n_group, ssm_d_state,
            rope_partial84 ? " rope=partial84(A/B)" : " rope=off(NoPE)",
            dt_clamp ? "" : " dt_clamp=off", eos_id);
    return true;
}

bool NemotronOmni::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f GB\n", mm.size / 1e9);
    struct ggml_init_params wip = { 512ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    wctx = ggml_init(wip);
    if (!wctx) return false;
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer       = vqa::gguf::u32(gu, "nemotronomni.block_count", n_layer);
    n_embd        = vqa::gguf::u32(gu, "nemotronomni.embedding_length", n_embd);
    n_head        = vqa::gguf::u32(gu, "nemotronomni.attention.head_count", n_head);
    head_dim      = vqa::gguf::u32(gu, "nemotronomni.attention.key_length", head_dim);
    n_vocab       = vqa::gguf::u32(gu, "nemotronomni.vocab_size", n_vocab);
    rms_eps       = vqa::gguf::f32(gu, "nemotronomni.attention.layer_norm_rms_epsilon", rms_eps);
    n_expert      = vqa::gguf::u32(gu, "nemotronomni.expert_count", n_expert);
    n_expert_used = vqa::gguf::u32(gu, "nemotronomni.expert_used_count", n_expert_used);
    n_ff_exp      = vqa::gguf::u32(gu, "nemotronomni.expert_feed_forward_length", n_ff_exp);
    n_ff_shexp    = vqa::gguf::u32(gu, "nemotronomni.expert_shared_feed_forward_length", n_ff_shexp);
    routed_scaling = vqa::gguf::f32(gu, "nemotronomni.expert_scale", routed_scaling);
    eos_id        = vqa::gguf::u32(gu, "nemotronomni.eos_token_id", eos_id);
    ssm_d_conv    = vqa::gguf::u32(gu, "nemotronomni.ssm.conv_kernel", ssm_d_conv);
    // this converter writes ssm.time_step_rank (= head count) and ssm.inner_size
    // instead of llama.cpp's ssm.head_count/head_dim pair
    ssm_n_head    = vqa::gguf::u32(gu, "nemotronomni.ssm.time_step_rank", ssm_n_head);
    ssm_n_group   = vqa::gguf::u32(gu, "nemotronomni.ssm.group_count", ssm_n_group);
    ssm_d_state   = vqa::gguf::u32(gu, "nemotronomni.ssm.state_size", ssm_d_state);
    {
        const int inner = vqa::gguf::u32(gu, "nemotronomni.ssm.inner_size",
                                         ssm_n_head * ssm_head_dim);
        if (inner != ssm_n_head * ssm_head_dim) {
            fprintf(stderr, "nemotronomni: ssm.inner_size %d != head_count %d *"
                            " head_dim %d\n", inner, ssm_n_head, ssm_head_dim);
            return false;
        }
    }
    {
        const std::string pat = vqa::gguf::str(gu, "nemotronomni.hybrid_override_pattern");
        if (!pat.empty()) {
            if ((int)pat.size() == n_layer) pattern = pat;
            else fprintf(stderr, "nemotronomni: pattern '%s' (%zu) != block_count %d"
                                " - keeping default\n", pat.c_str(), pat.size(), n_layer);
        }
    }
    if ((int)pattern.size() != n_layer) {
        fprintf(stderr, "nemotronomni: pattern '%s' (%zu) != block_count %d\n",
                pattern.c_str(), pattern.size(), n_layer);
        return false;
    }
    // head_count_kv is a PER-LAYER array in this family's GGUF ('*' layers
    // carry the width, hybrid layers 0 - converter contract 10.7-1)
    std::vector<int> kv_arr = vqa::gguf::arr_u32(gu, "nemotronomni.attention.head_count_kv");
    if ((int)kv_arr.size() == n_layer) {
        int seen = -1;
        bool ok = true;
        for (int i = 0; i < n_layer && ok; i++) {
            if (pattern[i] == '*') {
                if (seen < 0) seen = kv_arr[(size_t)i];
                else ok = kv_arr[(size_t)i] == seen;
            } else {
                ok = kv_arr[(size_t)i] == 0;
            }
        }
        if (ok && seen > 0) n_kv_head = seen;
        else fprintf(stderr, "nemotronomni: head_count_kv array inconsistent with"
                            " the hybrid pattern - keeping n_kv_head %d\n", n_kv_head);
    } else {
        n_kv_head = vqa::gguf::u32(gu, "nemotronomni.attention.head_count_kv", n_kv_head);
    }
    n_mamba = n_attn = 0;
    for (char c : pattern) {
        if (c == 'M') n_mamba++;
        else if (c == '*') n_attn++;
        else if (c != 'E') {
            fprintf(stderr, "nemotronomni: bad pattern char '%c'\n", c);
            return false;
        }
    }

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);

    // 2D tensors the graphs use inside elementwise ops need plain F32 rows
    // (promote_1d_to_f32's extra-names path): the grouped ssm_norm [I/G, G],
    // the per-head A [1, n_head] (both already F32 in this converter's GGUF -
    // the promote is a no-op there) and the F16 conv kernel ggml_ssm_conv
    // requires as F32 rows (ops.cpp asserts src1->nb[0] == sizeof(float);
    // mage_vl's mamba conv kernel is the same precedent)
    std::vector<std::string> extra;
    std::vector<const char *> extra_names;
    for (int i = 0; i < n_layer; i++) {
        const std::string pre = "blk." + std::to_string(i) + ".";
        if (ggml_get_tensor(wctx, (pre + "ssm_norm.weight").c_str()))
            extra.push_back(pre + "ssm_norm.weight");
        if (ggml_get_tensor(wctx, (pre + "ssm_conv1d.weight").c_str()))
            extra.push_back(pre + "ssm_conv1d.weight");
        const bool has_a = ggml_get_tensor(wctx, (pre + "ssm_a").c_str()) != nullptr;
        const std::string aname = pre + (has_a ? "ssm_a" : "ssm_a_log");
        if (ggml_get_tensor(wctx, aname.c_str())) extra.push_back(aname);
    }
    for (const std::string & s : extra) extra_names.push_back(s.c_str());
    extra_names.push_back(nullptr);
    vqa::weights::promote_1d_to_f32(wctx, host_f32_storage, extra_names.data());

    // A = -exp(A_log): computed once on the host, written over the ssm_a_log
    // tensor in place (shape rewritten [n_head] -> [1, n_head]) - the
    // convert_mage_gate "A_log -> A F32" precedent, load-time edition
    for (int i = 0; i < n_layer; i++) {
        char b[96];
        snprintf(b, sizeof(b), "blk.%d.ssm_a", i);
        if (ggml_get_tensor(wctx, b)) continue;
        snprintf(b, sizeof(b), "blk.%d.ssm_a_log", i);
        ggml_tensor * t = ggml_get_tensor(wctx, b);
        if (!t) continue;
        const int64_t nh = t->ne[0];
        std::vector<float> logv((size_t)nh), av((size_t)nh);
        vqa::gguf::tensor_to_f32(t, logv.data());
        for (int64_t h = 0; h < nh; h++) av[(size_t)h] = -std::exp(logv[(size_t)h]);
        // grow the shared block, then repoint the tensor into it
        const size_t off = a_storage.size();
        a_storage.resize(off + (size_t)nh * sizeof(float));
        memcpy(a_storage.data() + off, av.data(), (size_t)nh * sizeof(float));
        t->data = a_storage.data() + off;
        t->type = GGML_TYPE_F32;
        t->ne[0] = 1; t->ne[1] = nh;
        t->nb[0] = sizeof(float); t->nb[1] = sizeof(float);
        snprintf(b, sizeof(b), "blk.%d.ssm_a", i);
        ggml_set_name(t, b);
    }

    // CPU: zero-copy from the mmap. GPU/split placement is the EC wave's
    // problem (weight_load.hpp §5); for now a non-CPU backend takes the
    // plain upload path.
    vqa::weights::Bound bound = vqa::weights::bind_to_backend(
        wctx, backend, mm, { { host_f32_storage.data(), host_f32_storage.size() },
                             { a_storage.data(), a_storage.size() } });
    if (!bound.ok) return false;
    wbuf = bound.main_buf;
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (!ggml_backend_is_cpu(backend)) mm.release();

    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) fprintf(stderr, "missing tensor: %s\n", n);
        return t;
    };
    tok_embd = gt("token_embd.weight");
    out_norm = gt("output_norm.weight");
    out_w    = gt("lm_head.weight");
    if (!tok_embd || !out_norm || !out_w) return false;
    if ((int)tok_embd->ne[0] != n_embd || (int)out_w->ne[0] != n_embd) {
        fprintf(stderr, "nemotronomni: embedding/lm_head width %lld/%lld != n_embd %d\n",
                (long long)tok_embd->ne[0], (long long)out_w->ne[0], n_embd);
        return false;
    }

    tl.resize((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "blk.%d.%s", i, suf); return gt(b); };
        TxtLayer & L = tl[(size_t)i];
        L.kind = pattern[i];
        L.norm = g("attn_norm.weight");
        if (!L.norm) return false;
        if (L.kind == 'M') {
            L.m.in_proj  = g("ssm_in.weight");
            L.m.conv1d   = g("ssm_conv1d.weight");
            L.m.conv1d_b = g("ssm_conv1d.bias");
            L.m.dt_bias  = g("ssm_dt.bias");
            L.m.a        = ggml_get_tensor(wctx, ("blk." + std::to_string(i) + ".ssm_a").c_str());
            L.m.d        = g("ssm_d");
            L.m.norm     = g("ssm_norm.weight");
            L.m.out_proj = g("ssm_out.weight");
            if (!L.m.in_proj || !L.m.conv1d || !L.m.dt_bias || !L.m.a || !L.m.d ||
                !L.m.norm || !L.m.out_proj)
                return false;
            if ((int)L.m.norm->ne[1] != ssm_n_group) {
                fprintf(stderr, "nemotronomni: blk.%d ssm_norm is [%lld,%lld],"
                                " expected [*, %d]\n", i,
                        (long long)L.m.norm->ne[0], (long long)L.m.norm->ne[1],
                        ssm_n_group);
                return false;
            }
        } else if (L.kind == '*') {
            L.a.q = g("attn_q.weight");
            L.a.k = g("attn_k.weight");
            L.a.v = g("attn_v.weight");
            // this converter names the output projection attn_o.weight
            // (contract 10.2), not llama.cpp's attn_output.weight
            L.a.o = g("attn_o.weight");
            if (!L.a.q || !L.a.k || !L.a.v || !L.a.o) return false;
            if (L.a.q->ne[0] != n_embd || L.a.q->ne[1] != n_head * head_dim ||
                L.a.k->ne[0] != n_embd || L.a.k->ne[1] != n_kv_head * head_dim ||
                L.a.o->ne[0] != n_head * head_dim || L.a.o->ne[1] != n_embd) {
                fprintf(stderr, "nemotronomni: blk.%d attention shapes"
                                " q[%lld,%lld] k[%lld,%lld] o[%lld,%lld]\n", i,
                        (long long)L.a.q->ne[0], (long long)L.a.q->ne[1],
                        (long long)L.a.k->ne[0], (long long)L.a.k->ne[1],
                        (long long)L.a.o->ne[0], (long long)L.a.o->ne[1]);
                return false;
            }
        } else {
            L.e.router     = g("ffn_gate_inp.weight");
            L.e.probs_b    = g("ffn_exp_probs_b");
            L.e.up_exps    = g("ffn_up_exps.weight");
            L.e.down_exps  = g("ffn_down_exps.weight");
            L.e.shexp_up   = g("ffn_up_shexp.weight");
            L.e.shexp_down = g("ffn_down_shexp.weight");
            if (!L.e.router || !L.e.up_exps || !L.e.down_exps ||
                !L.e.shexp_up || !L.e.shexp_down)
                return false;
            if ((int)L.e.up_exps->ne[2] != n_expert ||
                (int)L.e.up_exps->ne[1] != n_ff_exp ||
                (int)L.e.shexp_up->ne[1] != n_ff_shexp) {
                fprintf(stderr, "nemotronomni: blk.%d expert shapes up[%lld,%lld,%lld]"
                                " shexp_up[%lld,%lld]\n", i,
                        (long long)L.e.up_exps->ne[0], (long long)L.e.up_exps->ne[1],
                        (long long)L.e.up_exps->ne[2],
                        (long long)L.e.shexp_up->ne[0], (long long)L.e.shexp_up->ne[1]);
                return false;
            }
        }
    }
    return true;
}

bool NemotronOmni::alloc_state() {
    const int conv_dim = ssm_n_head * ssm_head_dim + 2 * ssm_n_group * ssm_d_state;
    // 2 scalar constants + 2 tensors per mamba layer
    struct ggml_init_params ip = { (size_t)(2 * n_mamba + 8) * ggml_tensor_overhead(),
                                   NULL, true };
    sctx = ggml_init(ip);
    if (!sctx) return false;
    st.resize((size_t)n_mamba);
    for (int i = 0; i < n_mamba; i++) {
        st[(size_t)i].conv = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, ssm_d_conv - 1, conv_dim);
        st[(size_t)i].ssm = ggml_new_tensor_3d(sctx, GGML_TYPE_F32,
                                               ssm_d_state, ssm_head_dim, ssm_n_head);
    }
    f_one = ggml_new_tensor_1d(sctx, GGML_TYPE_F32, 1);
    f_eps = ggml_new_tensor_1d(sctx, GGML_TYPE_F32, 1);
    sbuf = ggml_backend_alloc_ctx_tensors(sctx, backend);
    if (!sbuf) { fprintf(stderr, "nemotronomni: ssm state alloc failed\n"); return false; }
    ggml_backend_buffer_clear(sbuf, 0);
    const float one = 1.0f, eps = 1e-20f;
    ggml_backend_tensor_set(f_one, &one, 0, sizeof(float));
    ggml_backend_tensor_set(f_eps, &eps, 0, sizeof(float));
    fprintf(stderr, "nemotronomni states: %.1f MB (%d mamba layers)\n",
            (double)(n_mamba * ((ssm_d_conv - 1) * conv_dim +
                                ssm_d_state * ssm_head_dim * ssm_n_head) * 4) / 1e6,
            n_mamba);
    return true;
}

bool NemotronOmni::build_text_graph(TextGraph & g, int T, int S, bool logits_all) {
    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 32768, false);
    ggml_context * c = g.ctx;
    const int D = n_embd;

    g.tok_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    ggml_set_input(g.tok_in);
    // absolute positions: KV slot index (and rope positions on the A/B path)
    g.pos_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_name(g.pos_in, "pos_in");
    ggml_set_input(g.pos_in);
    g.mask_in = ggml_new_tensor_2d(c, GGML_TYPE_F16, S, T);
    ggml_set_name(g.mask_in, "mask_in");
    ggml_set_input(g.mask_in);
    g.ssm_ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    ggml_set_name(g.ssm_ids, "ssm_ids");
    ggml_set_input(g.ssm_ids);

    ggml_tensor * x = ggml_get_rows(c, tok_embd, g.tok_in);           // [D, T]
    ggml_set_name(x, "emb");
    ggml_tensor * pos_slot = g.pos_in;   // KV slot == absolute position
    ggml_tensor * pos_rope = rope_partial84 ? g.pos_in : nullptr;   // NoPE when null

    int si = 0, kv_idx = 0;
    int n_m = 0, n_a = 0, n_e = 0;
    int max_layers = n_layer;
    if (const char * e = getenv("NEMOTRONOMNI_MAX_LAYERS"))
        max_layers = std::min(max_layers, atoi(e));

    for (int il = 0; il < max_layers; il++) {
        TxtLayer & L = tl[(size_t)il];
        ggml_tensor * l = ggml_rms_norm(c, x, rms_eps);
        l = ggml_mul(c, l, L.norm);
        if (L.kind == 'M') {
            const char * dbg = (n_m == 0 && !dump_dir.empty()) ? "m0" : nullptr;
            if (dbg) ggml_set_name(l, "m0_in");
            ggml_tensor * mo = mamba2_branch(c, g.gf, L.m, st[(size_t)si], l,
                                             g.ssm_ids, rms_eps,
                                             dt_clamp ? f_one : nullptr, dt_min, dbg);
            x = ggml_add(c, x, mo);
            si++; n_m++;
        } else if (L.kind == '*') {
            const char * dbg = (n_a == 0 && !dump_dir.empty()) ? "a0" : nullptr;
            if (dbg) ggml_set_name(l, "a0_in");
            ggml_tensor * ao = attn_branch(c, g.gf, L.a, kv.k[(size_t)kv_idx],
                                           kv.v[(size_t)kv_idx], l, g.mask_in,
                                           pos_slot, pos_rope, rope_dims, rope_theta,
                                           n_ctx, n_head, n_kv_head, head_dim, dbg);
            x = ggml_add(c, x, ao);
            kv_idx++; n_a++;
        } else {
            const char * dbg = (n_e == 0 && !dump_dir.empty()) ? "e0" : nullptr;
            if (dbg) ggml_set_name(l, "e0_in");
            L.e.f_eps = f_eps;
            ggml_tensor * mo = moe_branch(c, L.e, l, n_expert_used,
                                          routed_scaling, dbg);
            x = ggml_add(c, x, mo);
            n_e++;
        }
        if (il == 0) ggml_set_name(x, "t_l0_out");
    }

    x = ggml_rms_norm(c, x, rms_eps);
    x = ggml_mul(c, x, out_norm);
    ggml_set_name(x, "t_final_norm");

    if (logits_all) {
        g.logits = ggml_mul_mat(c, out_w, x);                     // [vocab, T]
    } else {
        ggml_tensor * last = ggml_view_2d(c, x, D, 1, x->nb[1],
                                          (size_t)(T - 1) * x->nb[1]);
        g.logits = ggml_mul_mat(c, out_w, last);
    }
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);

    if (!dump_dir.empty()) {
        static const char * const dn[] = {
            "emb", "m0_in", "m0_inproj", "m0_dt", "m0_xbc", "m0_yd", "m0_yn", "m0_out",
            "a0_in", "a0_q", "a0_q_pre_fa", "a0_k_pre_fa", "a0_attn", "a0_out",
            "e0_in", "e0_rlogits", "e0_sel", "e0_w8", "e0_routed", "e0_shexp", "e0_out",
            "t_final_norm", nullptr,
        };
        vqa::dump::mark_sinks(c, g.gf, dn);
        vqa::dump::finalize_sinks(c, g.gf, backend);
    }
    return true;
}

bool NemotronOmni::run_text(TextGraph & g, const std::vector<int> & toks,
                            const std::vector<int> & pos, std::vector<float> & logits_out) {
    const int T = (int)toks.size();
    const int S = n_past + T;
    if (S > n_ctx) { fprintf(stderr, "context overflow (%d > %d)\n", S, n_ctx); return false; }
    if ((int)pos.size() != T) { fprintf(stderr, "pos/tok length mismatch\n"); return false; }

    // decode (T == 1) runs on one persistent bucketed graph per KV window
    // (decode_graph.hpp); the !split_mode gate keeps the slot single-backend
    // (the EC wave re-audits this with the Strata scheduler)
    const bool decode_step = (T == 1);
    // decode reuses one bucketed persistent graph per KV window
    // (decode_graph.hpp GraphSlot - the qwen3_5/qwen3omni architecture).
    // Wave-3b had this behind NEMOTRONOMNI_BUCKET because the attention
    // KV-window/mask-width mismatch (see attn_branch) poisoned every
    // non-fresh layout with NaN; with the window fixed the slot graph is
    // bit-identical to per-step rebuilds, so it is the default again.
    // NEMOTRONOMNI_BUCKET=0 forces the fresh-graph debug path.
    bool reuse_graph = decode_step && !split_mode;
    if (const char * b = getenv("NEMOTRONOMNI_BUCKET"))
        reuse_graph = reuse_graph && atoi(b) != 0;
    const int window_S = reuse_graph ? vqa::decode_bucket(S, n_ctx,
                                   ggml_backend_is_cpu(backend)) : S;
    TextGraph * dg = &g;
    // NEMOTRONOMNI_LOGITS_ALL=1: prefill returns the whole [vocab, T] block
    // (oracle parity: per-position logits comparison). Decode steps never do.
    const bool want_all = logits_all_env && T > 1;
    if (reuse_graph) {
        if (!dec_slot.prepare(window_S, [&](TextGraph & d) {
                return build_text_graph(d, 1, window_S);
            }))
            return false;
        dg = &dec_slot.g;
    } else if (!build_text_graph(g, T, S, want_all)) {
        return false;
    }
    // split_mode never engages this wave; the EC wave swaps the alloc in for
    // its SplitSched exactly like qwen3omni (sched.hpp)
    if (!ggml_gallocr_alloc_graph(allocr, dg->gf)) {
        fprintf(stderr, "nemotronomni: text graph allocation failed\n");
        if (reuse_graph) dec_slot.reset();
        else { ggml_free(g.ctx); g.ctx = nullptr; }
        return false;
    }
    // truncated graphs (NEMOTRONOMNI_MAX_LAYERS with no '*' layer) never put
    // pos/mask inputs into the graph - only set what the gallocr allocated
    if (dg->tok_in->buffer) ggml_backend_tensor_set(dg->tok_in, toks.data(), 0, sizeof(int) * T);
    if (dg->pos_in->buffer) ggml_backend_tensor_set(dg->pos_in, pos.data(), 0, sizeof(int) * T);
    const int32_t ids0 = 0;
    if (dg->ssm_ids->buffer) ggml_backend_tensor_set(dg->ssm_ids, &ids0, 0, sizeof(int32_t));
    if (dg->mask_in->buffer) vqa::set_causal_mask(dg->mask_in, n_past, T);
    if (ggml_backend_graph_compute(backend, dg->gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "nemotronomni: text compute failed\n");
        return false;
    }
    ggml_tensor * lg = ggml_graph_get_tensor(dg->gf, "logits");
    logits_out.resize((size_t)n_vocab * (want_all ? (size_t)T : 1));
    ggml_backend_tensor_get(lg, logits_out.data(), 0,
                            sizeof(float) * logits_out.size());
    if (want_all && getenv("NEMOTRONOMNI_PREFILL_DUMP")) {
        const char * pd = getenv("NEMOTRONOMNI_PREFILL_DUMP");
        FILE * f = fopen(pd, "wb");
        if (f) {
            fwrite(logits_out.data(), sizeof(float), logits_out.size(), f);
            fclose(f);
            fprintf(stderr, "nemotronomni: prefill logits [%d,%d] -> %s\n",
                    n_vocab, T, pd);
        } else {
            fprintf(stderr, "nemotronomni: cannot write %s\n", pd);
        }
    }
    if (T > 1 && getenv("NEMOTRONOMNI_STATE_DUMP")) {
        // mamba states live in their own buffer (never gallocr-managed), so
        // this readback is trustworthy unlike mid-graph sink dumps
        FILE * f = fopen(getenv("NEMOTRONOMNI_STATE_DUMP"), "wb");
        if (f) {
            for (int i = 0; i < n_mamba; i++) {
                std::vector<float> v((size_t)ssm_d_state * ssm_head_dim * ssm_n_head);
                ggml_backend_tensor_get(st[(size_t)i].ssm, v.data(), 0,
                                        sizeof(float) * v.size());
                fwrite(v.data(), sizeof(float), v.size(), f);
            }
            fclose(f);
            fprintf(stderr, "nemotronomni: %d mamba states -> %s\n", n_mamba,
                    getenv("NEMOTRONOMNI_STATE_DUMP"));
        }
    }
    if (!dump_dir.empty()) {
        dump(dg->gf, "emb"); dump(dg->gf, "logits"); dump(dg->gf, "t_final_norm");
        if (T > 1) {
            dump(dg->gf, "m0_in"); dump(dg->gf, "m0_out");
            dump(dg->gf, "m0_inproj"); dump(dg->gf, "m0_dt"); dump(dg->gf, "m0_xbc");
            dump(dg->gf, "m0_yd"); dump(dg->gf, "m0_yn");
            dump(dg->gf, "a0_attn"); dump(dg->gf, "a0_out");
            dump(dg->gf, "e0_in"); dump(dg->gf, "e0_out");
        }
    }
    n_past = S;
    if (!reuse_graph) { ggml_free(g.ctx); g.ctx = nullptr; }   // the decode slot persists
    return true;
}

void NemotronOmni::dump(ggml_cgraph * gf, const char * name) {
    if (dump_dir.empty()) return;
    vqa::dump::write(gf, dump_dir.c_str(), name);
}

int NemotronOmni::sample(const std::vector<float> & logits, const vqa::GenParams & gp) {
    return vqa::sample_logits(logits, gp, rng_state);
}

bool NemotronOmni::generate(const std::vector<int> & ids, const vqa::GenParams & gp,
                            std::vector<int> & out_ids, std::string & out_text) {
    out_ids.clear();
    out_text.clear();
    if ((int)ids.size() >= n_ctx) {
        fprintf(stderr, "nemotronomni: prompt %zu tok fills n_ctx %d\n", ids.size(), n_ctx);
        return false;
    }
    int max_new = gp.max_tokens;
    if ((int)ids.size() + max_new > n_ctx) {
        max_new = n_ctx - (int)ids.size();
        fprintf(stderr, "nemotronomni: clamping max_new to %d\n", max_new);
    }
    // fresh conversation: zero the KV cache and every recurrent state
    kv.clear();
    if (sbuf) ggml_backend_buffer_clear(sbuf, 0);
    n_past = 0;

    TextGraph g;
    std::vector<float> logits;
    std::vector<int> pos(ids.size());
    for (size_t i = 0; i < ids.size(); i++) pos[i] = (int)i;
    if (!run_text(g, ids, pos, logits)) return false;
    // with NEMOTRONOMNI_LOGITS_ALL the block is [vocab, T]: the distribution
    // of the next token is the LAST position's row
    auto last_row = [&]() {
        return std::vector<float>(logits.end() - (size_t)n_vocab, logits.end());
    };
    int cur = sample(last_row(), gp);
    // NEMOTRONOMNI_STEP_DUMP=<prefix>: per-step logits that produced each
    // token, <prefix>_step<k>.f32 (oracle gen-step parity)
    const char * step_dump = getenv("NEMOTRONOMNI_STEP_DUMP");
    for (int step = 0; step < max_new; step++) {
        // dump first: the eos step's logits are parity material too (the
        // oracle gen dump includes the step that produced <|im_end|>)
        if (step_dump) {
            char pth[512];
            snprintf(pth, sizeof(pth), "%s_step%d.f32", step_dump, step);
            FILE * f = fopen(pth, "wb");
            if (f) {
                fwrite(logits.data() + (logits.size() - (size_t)n_vocab),
                       sizeof(float), (size_t)n_vocab, f);
                fclose(f);
            }
        }
        if (cur == eos_id || cur == 2) break;   // <|im_end|> / </s>
        out_ids.push_back(cur);
        if (tok_.size() > 0) {
            const std::string piece = tok_.decode({ cur });
            out_text += piece;
            if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
        }
        if (step + 1 >= max_new) break;
        const int p = n_past;
        if (!run_text(g, { cur }, { p }, logits)) return false;
        cur = sample(logits, gp);
    }
    return true;
}

bool NemotronOmni::ask(const vqa::MediaInput & media, const std::string & prompt,
                       const std::string & mode, const vqa::GenParams & gp,
                       std::string & out_text) {
    if (!media.empty() || media.has_audio()) {
        fprintf(stderr, "nemotronomni: vision/audio towers land with wave 4;"
                        " text-only for now\n");
        return false;
    }
    // mode "raw": the prompt is already the full continuation text (oracle
    // parity path - no chat template, no BOS). Otherwise ChatML with the
    // checkpoint's default template: empty system preamble + thinking on
    // (chat_template.jinja add_generation_prompt/enable_thinking defaults).
    std::string text;
    if (mode == "raw") {
        text = gp.history + prompt;
    } else {
        text = gp.history + "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n" +
               prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n";
    }
    const std::vector<int> ids = tok_.encode(text);
    if (getenv("NEMOTRONOMNI_DUMP_IDS")) {
        fprintf(stderr, "nemotronomni prompt ids (%zu):", ids.size());
        for (int id : ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }
    std::vector<int> out_ids;
    if (!generate(ids, gp, out_ids, out_text)) return false;
    if (gp.on_stage) gp.on_stage("done", (int)out_ids.size(), (int)out_ids.size(), gp.on_stage_ud);
    return true;
}

}  // namespace nemo

namespace vqa {
Model * create_nemotronomni() { return new nemo::NemotronOmni(); }
}  // namespace vqa
