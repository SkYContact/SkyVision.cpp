// moss_vl_preprocess.hpp - MOSS-VL vision preprocessing, pure C++ (no ggml).
//
// Mirrors MossVLImageProcessorFast + MossVLVisionModel input side:
//   smart_resize (factor = patch*merge = 32, min_pixels = 4096)
//   -> bicubic antialias resize (shared with the VC3 backend)
//   -> normalize (x/127.5 - 1) and patchify with temporal_patch_size = 1:
//      patch vector laid out (c, y, x), tokens in BLOCK-MAJOR order (mh, mw, ir, ic)
//   -> learned 48x48 absolute position table bilinearly interpolated to the
//      patch grid (torch.linspace endpoints, floor/ceil clip)
//   -> per-token (row, col) patch-grid coordinates for the 2D vision rope.
#pragma once

#include <cstdint>
#include <vector>

namespace moss_vl {

struct VisConfig {
    int patch = 16;
    int merge = 2;
    int n_embd = 1152;
    int n_head = 16;
    int n_layer = 27;
    int n_ff = 4304;
    int pos_side = 48;      // sqrt(num_position_embeddings = 2304)
    int out_hidden = 4096;
    float layernorm_eps = 1e-6f;
    float rope_theta = 10000.0f;
};

// Patchify one resized RGB frame: `rgb` interleaved float [h*w*3] in [0,255].
// Appends patches [3*p*p, gh*gw] (block-major columns), interpolated pos rows
// [n_embd, gh*gw] and (row, col) coords [gh*gw * 2] in the same block-major order.
void patchify_frame(const float * rgb, int h, int w, const VisConfig & vc,
                    const float * pos_table /* [pos_side*pos_side, n_embd] */,
                    std::vector<float> & patches, std::vector<float> & pos,
                    std::vector<int> & coords, int * grid_h, int * grid_w);

}  // namespace moss_vl
