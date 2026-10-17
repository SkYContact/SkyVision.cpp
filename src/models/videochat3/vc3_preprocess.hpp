// vc3_preprocess.hpp - VideoChat3 image/video preprocessing, pure C++ (no ggml).
//
// Pipeline (mirrors Qwen2VLImageProcessor / transformers fast processor):
//   smart_resize (factor = patch*merge = 28, min_pixels = 56*56, max_pixels = 16M)
//   -> bicubic resize with antialias (torchvision/PIL semantics, a = -0.5)
//   -> rescale 1/255, normalize (x-0.5)/0.5  == x/127.5 - 1
//   -> patchify into (3*14*14, N) in *merge grouped* order:
//        idx = ((bi * (gw/2) + bj) * kh + ma) * kw + mb
//   -> positional embedding table: bicubic interpolation of the learned (64,64,D)
//      grid, torch F.interpolate semantics (A = -0.75, align_corners = false)
#pragma once

#include "resample.hpp"

#include <cstdint>
#include <vector>

namespace vc3 {

// shared image kernels now live in core/resample.hpp; keep the vc3 names so the
// call sites in this backend (and the pos-embedding path below) read unchanged
using vqa::img::resize_bicubic_antialias;
using vqa::img::smart_resize;

struct VisConfig {
    int patch = 14;
    int merge = 2;
    int temporal_merge = 4;
    int pos_h = 64;
    int pos_w = 64;
    int n_embd = 1152;
    int n_head = 16;
    int n_layer = 27;
    int n_ff = 4304;
    float layernorm_eps = 1e-5f;
    float rope_theta = 10000.0f;
};

// smart_resize / resize_bicubic_antialias: see core/resample.hpp (aliased above).

// Normalize + patchify. `rgb` is interleaved RGB float [h*w*3] in [0,255].
// Produces `patches` of shape (3*patch*patch, N) laid out for ggml ne = [C*P*P, N],
// and sets grid_h/grid_w.
void patchify(const float * rgb, int h, int w, const VisConfig & vc,
              std::vector<float> & patches, int * grid_h, int * grid_w);

// Position embedding for one (t,h,w) clip: interpolates the learned (pos_h,pos_w,D)
// grid to (h,w) and adds the temporal embedding rows when t > 1.
// Writes (h*w*t, D) row-major.
void pos_embed_for_grid(const float * weight,          // (pos_h*pos_w*D)
                        const float * time_weight,     // (max_t*D)
                        const VisConfig & vc,
                        int t, int h, int w,
                        std::vector<float> & out);

// 2D rotary tables for one clip. Produces cos/sin each of shape (n_pairs=hd/2, n_pos)
// where n_pos = t*h*w, row-major over positions. Layout of a head vector is
// interleaved pairs: (v[0],v[1]), (v[2],v[3]), ... ; even pairs rotate with the x
// coordinate, odd pairs with the y coordinate, both using freqs 1/theta^(4i/hd).
void rope_2d_tables(int hd, int t, int h, int w, float theta,
                    std::vector<float> & cos_out, std::vector<float> & sin_out);

}  // namespace vc3
