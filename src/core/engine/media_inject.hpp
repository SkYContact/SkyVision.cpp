// media_inject.hpp - the graph-side media injection block shared by the omni
// backends: create the mm_in/mm_pos_in graph inputs and scatter the vision /
// audio tower embeddings onto their placeholder rows of the embedded prompt.
//
// Moved verbatim out of nemotronomni_model.cpp and minicpmo_model.cpp (the
// two character-identical copies; qwen3omni keeps its own img_in-named
// variant because its "emb" naming point differs). Caller-owns, like
// attention.hpp / llm_block.hpp: the caller keeps the returned input pointers
// and uploads the feats / row indices after alloc. Two contracts are
// load-bearing here:
//   * n_mm == 0 must not create ANY node - the decode graphs are reused
//     bit-identically across steps and the bucketed decode path depends on
//     the prefill-only injection never appearing there;
//   * the tensor names ("mm_in" / "mm_pos_in"), the dup-then-set_rows op
//     pair and the node order are looked up by name downstream and are part
//     of the byte-compared graph contract.
#pragma once

#include "ggml.h"

namespace vqa {
namespace engine {

// x [D, T] embedded prompt -> same stream with the n_mm media rows scattered
// on (dup first: set_rows needs a fresh buffer). Returns the input tensors
// through mm_in / mm_pos_in (untouched when n_mm <= 0).
inline ggml_tensor * inject_media_rows(ggml_context * c, ggml_tensor * x,
                                       ggml_tensor ** mm_in,
                                       ggml_tensor ** mm_pos_in,
                                       int D, int n_mm) {
    if (n_mm <= 0) return x;
    // media rows (vision/audio tower embeddings) scattered onto their
    // placeholder positions before the trunk; decode (n_mm == 0) never
    // builds these, keeping the bucketed decode graph bit-identical
    *mm_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n_mm);
    ggml_set_name(*mm_in, "mm_in");
    ggml_set_input(*mm_in);
    *mm_pos_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_mm);
    ggml_set_name(*mm_pos_in, "mm_pos_in");
    ggml_set_input(*mm_pos_in);
    return ggml_set_rows(c, ggml_dup(c, x), *mm_in, *mm_pos_in);
}

}  // namespace engine
}  // namespace vqa
