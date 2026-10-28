// nemotronomni_text.hpp - the NemotronH text-chain branch surface.
//
// One header so three consumers share one contract (no duplicate struct
// drift): the model graph (nemotronomni_model.cpp), the component parity
// driver (tools/nemotronomni_seg_test.cpp, synthetic weights) and the wave-3b
// EC decode graph. Each branch builder takes the post-norm stream and returns
// its [D, T] branch output; the caller adds the residual.
//
// Numerics contract is documented at the top of nemotronomni_model.cpp
// (HF modeling_nemotron_h.py is the reference; the dt clamp note and the
// NoPE decision live there).
#pragma once

#include "ggml.h"

namespace nemo {

// One branch's weight tensors - exactly what a GGUF-mapped (or synthetic)
// weight context provides. 2D weights follow the ggml mul_mat orientation:
// ne[0] = IN, ne[1] = OUT (HF shapes reversed, converter contract).
struct Mamba2W {
    ggml_tensor *in_proj = nullptr;    // [D, d_in_proj]  (out rows: z | xBC | dt)
    ggml_tensor *conv1d = nullptr;     // [d_conv, conv_dim]
    ggml_tensor *conv1d_b = nullptr;   // [conv_dim] (null = no bias)
    ggml_tensor *dt_bias = nullptr;    // [n_head]
    ggml_tensor *a = nullptr;          // [1, n_head] F32, = -exp(A_log)
    ggml_tensor *d = nullptr;          // [n_head]
    ggml_tensor *norm = nullptr;       // [d_inner/n_group, n_group]
    ggml_tensor *out_proj = nullptr;   // [d_inner, D]
};

struct AttnW {
    ggml_tensor *q = nullptr;          // [D, n_head*head_dim]
    ggml_tensor *k = nullptr;          // [D, n_kv_head*head_dim]
    ggml_tensor *v = nullptr;          // [D, n_kv_head*head_dim]
    ggml_tensor *o = nullptr;          // [n_head*head_dim, D]
};

struct MoEW {
    ggml_tensor *router = nullptr;     // [D, n_exp] F32
    ggml_tensor *probs_b = nullptr;    // [n_exp] F32 (null = no bias)
    ggml_tensor *up_exps = nullptr;    // [D, I, n_exp]
    ggml_tensor *down_exps = nullptr;  // [I, D, n_exp]
    ggml_tensor *shexp_up = nullptr;   // [D, Ish]
    ggml_tensor *shexp_down = nullptr; // [Ish, D]
    ggml_tensor *f_eps = nullptr;      // [1] F32 holding 1e-20 (null = skip)
};

// Persistent per-layer recurrent states (the CALLER's asset, qwen3_5
// precedent): conv window [d_conv-1, conv_dim] and SSM state
// [d_state, head_dim, n_head], both F32. The graphs cpy the rolled window /
// new state back into these on every call (+0-tied into the main chain).
struct Mamba2State {
    ggml_tensor *conv = nullptr;       // [d_conv-1, conv_dim]
    ggml_tensor *ssm = nullptr;        // [d_state, head_dim, n_head]
};

// Mamba2 mixer. ggml_ssm_scan applies softplus(dt) internally, so with
// f_one non-null dt is pre-wrapped through
//   inverse_softplus(clamp(softplus(dt+bias), dt_min))
// to reproduce modeling_nemotron_h.py:601-602 exactly (the kernel then
// re-applies its own softplus). f_one = [1] F32 constant 1.0; pass null to
// drop the clamp (llama.cpp mamba2 semantics). ids = [1] I32 graph input
// holding the state slot index (0 for the single-sequence path).
ggml_tensor * mamba2_branch(ggml_context * ctx, ggml_cgraph * gf,
                            const Mamba2W & w, Mamba2State & st,
                            ggml_tensor * l, ggml_tensor * ids,
                            float rms_eps, ggml_tensor * f_one, float dt_min,
                            const char * dbg);

// Full attention: GQA, no q/k norm, causal mask, F16 KV cache, NoPE by
// default. pos_rope non-null switches on the partial-rope A/B path (NEOX
// over the first rope_dims of head_dim, llama.cpp GGUF-metadata semantics).
ggml_tensor * attn_branch(ggml_context * ctx, ggml_cgraph * gf, const AttnW & w,
                          ggml_tensor * k_cache, ggml_tensor * v_cache,
                          ggml_tensor * l, ggml_tensor * mask,
                          ggml_tensor * pos_slot, ggml_tensor * pos_rope,
                          int rope_dims, float rope_theta, int n_ctx,
                          int n_head, int n_kv_head, int head_dim,
                          const char * dbg);

// MoE FFN: sigmoid routing in fp32, correction bias only in the selection,
// top-n_used, unbiased-sigmoid weights normalized then scaled by
// routed_scaling (2.5), relu^2 non-gated experts, shared expert added on the
// same normed input (unscaled).
ggml_tensor * moe_branch(ggml_context * ctx, const MoEW & w, ggml_tensor * m,
                         int n_used, float routed_scaling, const char * dbg);

}  // namespace nemo
