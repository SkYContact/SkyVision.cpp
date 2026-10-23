// moe_compute.hpp - the generic routed-expert FFN graph segment, shared by
// every MoE backend's dynamic decode path.
//
// CONTRACT:
//   out = x + Σ_k w[k] · down_k · act(gate_k·x ⊙ up_k·x)
//
//   - x is the post ffn_norm hidden [n_embd, T] (decode: T=1); the residual
//     add is INCLUDED (returns x + moe_out) so callers splice it directly
//     into the layer stream.
//   - gate/up/down are per-selected-expert tensors (typically ExpertCache
//     slots on the device; nothing here knows where they came from).
//   - w is a graph input tensor [n_used, T] (F32) holding the router's
//     softmax weights - the caller uploads it per step.
//   - act: GGML_UNARY_OP_SILU (keye/qwen-moe) or GELU variants (moondream
//     uses erf-gelu); passed through to ggml_unary.
//   - n_used = number of selected experts; gate/up/down are indexed the same
//     way w's rows are (slot order must match weight order).
//
// v1 note: per-expert ggml_mul_mat over [n_ff, n_embd] slots. For decode
// (T=1) these are GEMV shapes; no fused kernel is attempted (the engine's
// CUDA GEMV path already handles them efficiently).
#pragma once

#include "ggml.h"

#include <vector>

namespace vqa {
namespace moe {

// returns x + Σ w_k · down_k · act(gate_k·x ⊙ up_k·x); nullptr on bad args
ggml_tensor * build_expert_ffn(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * x,
                               const std::vector<ggml_tensor *> & gate,
                               const std::vector<ggml_tensor *> & up,
                               const std::vector<ggml_tensor *> & down,
                               ggml_tensor * w,
                               ggml_unary_op act = GGML_UNARY_OP_SILU);

// ------------------------------------------------------------------ v3
// slot-table form (additive contract, ExpertCache v3 table layout):
//   gate_table/up_table/down_table are the cache's 3D slot tables
//   [ne0, ne1, n_slots] (see ExpertCache::table), gate_up_ids / down_ids are
//   I32 graph inputs [n_used, T] holding the slot_index() ids of the acquired
//   experts (gate/up MUST share one ids tensor so the CUDA backend's fused
//   mul_mat_id+mul_mat_id+GLU pass can trigger; down may use a different ids
//   tensor when its table lives in a different cache). Math is identical to
//   build_expert_ffn; the whole expert set runs through 3 ggml_mul_mat_id
//   launches instead of n_used*3 GEMVs. Returns x + Σ ... or nullptr.
ggml_tensor * build_expert_ffn_table(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * x,
                                     ggml_tensor * gate_table, ggml_tensor * up_table,
                                     ggml_tensor * down_table,
                                     ggml_tensor * gate_up_ids, ggml_tensor * down_ids,
                                     ggml_tensor * w,
                                     ggml_unary_op act = GGML_UNARY_OP_SILU);

}  // namespace moe
}  // namespace vqa
