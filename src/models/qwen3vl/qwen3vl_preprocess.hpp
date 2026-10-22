// qwen3vl_preprocess.hpp - Qwen3-VL (OneStreamer-4B) vision preprocessing, pure C++.
//
// Mirrors transformers 4.57 Qwen3VLVisionModel + the Qwen3VL processor:
//   smart_resize (factor = patch*merge = 32; ~448x448 per frame for video)
//   -> bicubic antialias resize (shared with the VC3 / qwen3_5 backends)
//   -> normalize (x/127.5 - 1, i.e. mean = std = 0.5) and 3D patchify: the conv
//      kernel (C=3, t=2, 16, 16) folded into a matmul, patch vector (c, t, y, x)
//   -> learned 48x48 absolute position table bilinearly interpolated to the patch
//      grid with ALIGN_CORNERS semantics (core/media/resample.hpp)
//   -> the patch stream is emitted in BLOCK-MAJOR order (mh, mw, ir, ic) so that
//      the 2x2 merger fold is a plain reshape over the token stream.
#pragma once

#include <cstdint>
#include <vector>

namespace qwen3vl {

struct VisConfig {
    int patch = 16;
    int temporal_patch = 2;
    int merge = 2;
    int n_embd = 1024;
    int n_head = 16;
    int n_layer = 24;
    int n_ff = 4096;
    int pos_side = 48;      // sqrt(num_position_embeddings = 2304)
    float layernorm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    int out_hidden = 2560;  // == the text width (DeepStack mergers share it)
    // tower blocks whose output feeds a DeepStack merger into text layers 0..n-1
    std::vector<int> deepstack;
};

// 3D patchify of `tframes` (= temporal_patch) RGB float frames [h*w*3 each,
// already resized] into `patches` [3*t*p*p, gh*gw] column-major per token, plus
// the interpolated pos rows [n_embd, gh*gw]. Appended in BLOCK-MAJOR order;
// `coords` receives the per-token full-resolution (h, w) rope coordinates.
void patchify3d(const float * const * tframes, int h, int w, const VisConfig & vc,
                const float * pos_table /* [pos_side*pos_side, n_embd] */,
                std::vector<float> & patches, std::vector<float> & pos,
                std::vector<int> & coords, int * grid_h, int * grid_w);

}  // namespace qwen3vl
