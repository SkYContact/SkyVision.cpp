// qwen3_5_preprocess.hpp - Qwen3_5-2B (Qwen3.5-VL) preprocessing, pure C++.
//
// Mirrors transformers 5.7 Qwen3_5VisionModel + the qwen-vl-utils processor:
//   smart_resize (factor = patch*merge = 32; video ~448x448 per frame)
//   -> bicubic antialias resize (shared with the VC3 backend)
//   -> normalize (x/127.5 - 1) and 3D patchify: conv kernel (C=3, t=2, 16, 16)
//      folded into a matmul, patch vector laid out (c, t, y, x)
//   -> learned 48x48 absolute position table bilinearly interpolated to the
//      patch grid with ALIGN_CORNERS semantics (linspace + floor/ceil clip)
//   -> the patch stream is emitted in BLOCK-MAJOR order (mh, mw, ir, ic):
//      the reference assigns pos_embed rows and rope (h, w) coordinates in
//      2x2-merge-block order while conv output is row-major, and the merger
//      groups 2x2 neighbours; emitting block-major at the host makes every
//      downstream stage (pos add, per-frame attention, rope coords, merger
//      reshape) plain row-major over the stream. Verified equivalent to the
//      reference layout in docs/dev-notes.md "Qwen3_5-2B".
#pragma once

#include <cstdint>
#include <vector>

namespace qwen3_5 {

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
    int out_hidden = 2048;
};

// 3D patchify of `tframes` (= temporal_patch) RGB float frames [h*w*3 each,
// already resized) into `patches` [3*t*p*p, gh*gw] column-major per token, and
// the learned pos table [n_embd, gh*gw]. Output is appended in BLOCK-MAJOR
// order; `coords` receives the per-token full-resolution (h, w) rope coords.
void patchify3d(const float * const * tframes, int h, int w, const VisConfig & vc,
                const float * pos_table /* [pos_side*pos_side, n_embd] */,
                std::vector<float> & patches, std::vector<float> & pos,
                std::vector<int> & coords, int * grid_h, int * grid_w);

}  // namespace qwen3_5
