// resample.hpp - image resampling / position-table kernels shared by the vision
// front ends (VideoChat3, Qwen3.5-VL, MOSS-VL, VideoMAE).
//
// These used to live in models/videochat3/vc3_preprocess.cpp, and three other
// model directories reached across with #include "../videochat3/..." - i.e. a
// shared library living inside one backend. Same code, same numbers, just in the
// shared layer where a backend-specific resampler does not belong.
#pragma once

#include <cstdint>
#include <vector>

namespace vqa {
namespace img {

// Qwen2VL smart_resize (round to `factor`, then clamp into [min_pixels, max_pixels]
// by scaling both sides). Returns false when the image is smaller than `factor`.
bool smart_resize(int height, int width, int factor, int min_pixels, int max_pixels,
                  int * out_h, int * out_w);

// Separable antialias bicubic resize (PIL / torchvision semantics, a = -0.5) of an
// interleaved RGB8 (HWC) buffer. `out` is interleaved RGB float [dh*dw*3] in [0,255].
void resize_bicubic_antialias(const uint8_t * src, int sw, int sh, int dw, int dh,
                              std::vector<float>& out);

// torchvision-uint8 antialias bicubic (HF fast image processors): int16
// fixed-point weights and per-pass uint8 rounding - bit-exact vs torch
void resize_tv_bicubic_aa(const uint8_t * src, int sw, int sh, int dw, int dh,
                          std::vector<float>& out);

// Same, with a selectable kernel. `pil_resample` is HF's `resample` value (a
// PILImageResampling enum): 1 = LANCZOS, 2 = BILINEAR, 3 = BICUBIC.
// round_u8: torchvision resizes uint8 input to a uint8 output (float pipeline,
// one rounding at the end); videomae/vivit preprocessors need it for exactness.
void resize_antialias(const uint8_t * src, int sw, int sh, int dw, int dh,
                      std::vector<float>& out, int pil_resample = 3, bool round_u8 = false);

// Bilinear interpolation of a learned (side, side, dim) position table onto a
// (gh, gw) grid: torch.linspace(0, side-1, g) endpoints, floor/ceil clipping.
// Writes (gh*gw, dim) row-major. Qwen2-VL / MOSS-VL semantic (VideoChat3 uses the
// bicubic/torch variant instead, which stays in vc3_preprocess.cpp).
void interp_pos_grid(const float * table, int side, int dim, int gh, int gw,
                     std::vector<float>& out);

}  // namespace img
}  // namespace vqa
