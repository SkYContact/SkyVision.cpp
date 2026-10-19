// videomae_preprocess.hpp - VideoMAE clip preprocessing, pure C++ (no ggml).
//
// Mirrors transformers' VideoMAEImageProcessor + the sampling the model card's
// example does by hand:
//   uniform frame sampling to num_frames (linspace indices, last frame padded)
//   -> shortest edge resized to image_size (PIL bicubic, antialiased)
//   -> center crop image_size x image_size
//   -> rescale 1/255 then normalize with the checkpoint's mean/std
//   -> pack tubelets: token order (t', h', w'), element order (c, dt, dh, dw)
//      so one token row lines up with the Conv3d weight of shape
//      [dim, C, tubelet, patch, patch] flattened to [dim, C*tubelet*patch*patch].
#pragma once

#include "core/vqa.hpp"

#include <cstdint>
#include <vector>

namespace videomae {

struct PreConfig {
    int image_size = 224;        // center crop size
    int resize_shortest = 0;     // shortest edge before the crop (0 = image_size;
                                 // ViViT resizes to 256 and then crops 224)
    int resample = 3;            // PILImageResampling: 2 = bilinear, 3 = bicubic
    int patch = 16;
    int tubelet = 2;
    int num_frames = 16;
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
};

// `want` frame indices out of `n`: linspace(0, n-1, want), truncated like
// numpy's astype(int); when n < want the last index repeats (the model needs a
// full clip and HF's own example errors out instead - padding is friendlier).
void sample_frame_indices(int n, int want, std::vector<int> & idx);

// One RGB8 frame -> planar CHW float32 of `image_size`^2 normalized pixels.
void preprocess_frame(const uint8_t * rgb, int w, int h, const PreConfig & pc, float * dst);

// Whole media item -> patch rows [C*tubelet*patch*patch, n_tokens].
// False when the media has no frames.
bool pack_clip(const vqa::MediaInput & media, const PreConfig & pc,
               std::vector<float> & patches, int * n_tokens, int * grid_t, int * grid_h,
               int * grid_w);

}  // namespace videomae
