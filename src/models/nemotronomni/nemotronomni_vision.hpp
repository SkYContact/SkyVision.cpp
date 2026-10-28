// nemotronomni_vision.hpp - C-RADIOv4-H vision tower for the Nemotron-3-Omni
// backend (ViT-H/16, 32 blocks, d1280, dynamic single tile).
//
// Architecture truth (ref/C-RADIOv4-H + ref/image_processing.py + ref/modeling.py;
// oracle parity: nemotron3omni-port/oracle/dump_repo_vision):
//   preprocess  target patch grid (wp, hp) from the dynamic single-tile budget
//               (1024 <= wp*hp <= 13312, Python round() banker's semantics),
//               torch F.interpolate(bicubic, align_corners=False, antialias=True)
//               on the uint8->fp32 tensor, then (x/255 - mean)/std with CLIP stats.
//   patch gen   Im2Patches (c, yy, xx) vectors -> Linear[1280, 768] (no bias);
//               pos_embed [16384, 1280] interpolated bilinear
//               align_corners=False to a (max(hp,wp))^2 square then window-crop
//               (repo cpe-mode eval branch); cls+register [10, 1280] prepended.
//   blocks      timm VisionTransformer: LayerNorm(1e-6) with affine, fused
//               qkv + bias, 16 heads x 80, scale 80^-0.5, NO rope, wo + bias,
//               GELU(erf) MLP 1280->5120->1280. NO final norm (model_norm
//               false), no patch LayerNorm.
//   head        drop the 10 prefix tokens, pixel_shuffle v2 (two permutes),
//               mlp1 = RMSNorm(5120, 1e-5, fp32) -> Linear[20480, 5120] ->
//               relu^2 -> Linear[2688, 20480] -> [wp*hp/4, 2688].
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <string>
#include <vector>

namespace nemo {

struct RadioVLayer {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;
    ggml_tensor *wqkv = nullptr, *bqkv = nullptr;   // [1280, 3840] / [3840]
    ggml_tensor *wo = nullptr, *bo = nullptr;       // [1280, 1280] / [1280]
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *fc0_w = nullptr, *fc0_b = nullptr; // [1280, 5120] / [5120]
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr; // [5120, 1280] / [1280]
};

struct RadioTower {
    // real-checkpoint dims; the loader overrides from GGUF metadata
    int n_layer = 32, n_embd = 1280, n_head = 16, n_ff = 5120, patch = 16;
    int n_cls = 10;                    // 4 cls + 6 registers, all dropped
    float ln_eps = 1e-6f;
    float img_mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };
    float img_std[3] = { 0.26862954f, 0.26130258f, 0.27577711f };
    int proj_hidden = 20480;           // mlp1 middle width
    int llm_dim = 2688;
    int min_patches = 1024, max_patches = 13312, max_model_len = 16384;

    // weights (bound ggml tensors; 2D follow the mul_mat orientation ne[0]=IN)
    ggml_tensor *patch_w = nullptr;    // v.patch_embd.weight [768, 1280]
    ggml_tensor *cls_tok = nullptr;    // v.cls_token.weight  [1280, 10] F32
    ggml_tensor *pos_embd = nullptr;   // v.pos_embd.weight  [1280, 16384] F32
    std::vector<RadioVLayer> vl;
    ggml_tensor *mm_ln = nullptr;      // mm.ln        [5120] F32
    ggml_tensor *mm_fc0 = nullptr;     // mm.fc0.weight [5120, 20480]
    ggml_tensor *mm_fc1 = nullptr;     // mm.fc1.weight [20480, 2688]

    // graph (rebuilt whenever the token count changes)
    ggml_context * vctx = nullptr;
    ggml_cgraph * vgf = nullptr;
    ggml_gallocr_t allocr = nullptr;   // borrowed from the model
    ggml_backend_t backend = nullptr;  // borrowed from the model
    ggml_tensor *v_in = nullptr, *v_pos_in = nullptr, *v_feats = nullptr;
    int v_n_tok = -1;
    std::string dump_dir;              // parity dump root ("" = off)

    // last preprocess result (injection bookkeeping + parity)
    int grid_wp = 0, grid_hp = 0, n_tok = 0;

    ~RadioTower() { free_graph(); }
    void free_graph();

    // Dynamic single-tile budget: normalized pixels (3, th, tw) CHW + grid.
    // `pixels` is the exact tower input (parity gate 1).
    bool preprocess(const uint8_t * rgb, int w0, int h0, std::vector<float> & pixels);

    bool build_graph(int n_patch, int wp, int hp);

    // runs the graph; embeds = [n_tok, 2688] row-major
    bool encode(ggml_backend_t backend, const std::vector<float> & pixels,
                std::vector<float> & embeds, const char * dump_dir);
};

}  // namespace nemo
