// moondream_preprocess.hpp - multi-crop Lanczos image preprocessing for
// moondream3.1 (official moondream/torch/image_crops.py overlap_crop_image +
// vision.py prepare_crops), pure C++, no ggml dependency.
//
// Pipeline (defaults = official VisionConfig):
//   - global crop: whole image Lanczos-resized to 378x378 (crop index 0)
//   - local crops: image resized to (th*266+112, tw*266+112), then cut into
//     th*tw 378x378 tiles with 4-patch (56 px) overlap margins, row-major
//   - output: n_crops = th*tw + 1 tensors of [3,378,378] float32, CHW,
//     normalized ((v/255)-0.5)/0.5 - exactly the VisionModel::encode_crops
//     input format.
//
// The Lanczos resampler bit-matches PIL (Pillow 10, the version the parity
// reference environment uses) Resample.c: filter support 3.0, per-output
// normalized weights, fixpoint 1<<22 coefficients, uint8 intermediate between
// the horizontal and vertical passes, horizontal pass always first.
#pragma once

#include <cstdint>
#include <vector>

namespace mdprep {

// official select_tiling (image_crops.py). Returns tile counts (th, tw).
void select_tiling(int height, int width, int crop_window, int max_crops,
                   int * th, int * tw);

// PIL-equivalent Lanczos resize lives in core/resample now:
// vqa::img::resize_lanczos_coeffs (see src/core/resample.hpp).

// Full official prepare_crops. rgb: interleaved RGB8 [h*w*3].
// Returns n_crops*3*crop*crop float32 (CHW, normalized); *th/*tw get the
// local-crop tiling, *n_crops_out the crop count (== vector size / crop area).
std::vector<float> prepare_crops(const uint8_t * rgb, int w, int h,
                                 int * th, int * tw, int * n_crops_out,
                                 int max_crops = 12, int overlap_margin = 4,
                                 int crop_size = 378, int patch_size = 14);

} // namespace mdprep
