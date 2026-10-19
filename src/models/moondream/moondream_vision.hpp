// moondream_vision.hpp — SigLIP vision encoder + projection for moondream3.1-9B-A2B
// (milestone 6). Self-contained: load moondream3.1-vision.gguf, build ggml graphs,
// run forward. Region head not included.
//
// Math (from moondream/torch/vision.py):
//   patches  = [B,3,378,378] -> [B,729,588] row-major patch tokens
//   x        = patch_emb(patches) + pos_emb                     [729,1152] per crop
//   per block (27):
//     x = x + attn(LN1(x))   qkv 1152->3456, 16 heads x head_dim 72,
//                           bidirectional softmax 1/sqrt(72), proj 1152->1152
//     x = x + mlp(LN2(x))    fc1 1152->4304, gelu(tanh approx), fc2 4304->1152
//   x = post_ln(x)                                               (encoder output)
//   projection: stitch local-crop features (margin 4 patches), adaptive-avg-pool
//               the [C,H,W] map to 27x27, concat with global crop features ->
//               [729,2304], fc1 2304->8192, gelu, fc2 8192->2048   (image embedding)
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

#include "core/util/mmap.hpp"

struct VisionModel {
    // config (defaults = official VisionConfig; overridden by GGUF metadata)
    int enc_dim = 1152, patch = 14, n_layers = 27, ff = 4304, n_heads = 16;
    int proj_out = 2048, crop = 378, in_ch = 3, max_crops = 12, overlap = 4, proj_inner = 8192;
    int head_dim = 72, grid = 27, patch_dim = 588;

    // weight storage
    struct ggml_context * wctx = nullptr;
    struct ggml_backend_buffer * wbuf = nullptr;
    void * f32_cpu = nullptr;
    vqa::MMap mm;                 // shared mmap helper (src/core/vqa_util)

    bool load(const char * gguf_path, ggml_backend_t backend);

    // crops: n_crops tensors of [3, crop, crop] float32, already normalized
    // ((v/255)-0.5)/0.5, CHW layout. Returns [n_crops, grid*grid, enc_dim]
    // (post_ln applied only when max_layer == n_layers; otherwise the residual
    // stream after block max_layer-1, i.e. raw hidden). If per_layer_dump is
    // given, appends one [grid*grid*enc_dim] vector per (layer, crop).
    std::vector<float> encode_crops(const float * crops, int n_crops, int max_layer,
                                    ggml_backend_t backend,
                                    std::vector<std::vector<float>> * per_layer_dump = nullptr);

    // enc: encoder output [n_crops, grid*grid, enc_dim] (as returned by
    // encode_crops with max_layer == n_layers). tiling = (th, tw) local crop grid.
    // Returns the final image embedding [grid*grid, proj_out].
    std::vector<float> project(const float * enc, int n_crops, int th, int tw,
                               ggml_backend_t backend);

    // Convenience: simplified single-crop path for raw uint8 RGB input.
    // Bilinear resize to crop x crop -> global crop only (no multi-crop tiling;
    // exact Lanczos multi-crop preprocessing currently lives in Python parity land).
    // Returns [grid*grid, proj_out] using the global crop as "reconstructed".
    std::vector<float> encode_image(const uint8_t * rgb, int w, int h,
                                    ggml_backend_t backend);

    ~VisionModel();
};

// helper: pick backend like the text engine (CUDA unless MD_CPU is set)
ggml_backend_t vision_pick_backend();
