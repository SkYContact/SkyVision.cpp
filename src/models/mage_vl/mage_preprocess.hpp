// mage_preprocess.hpp - Mage-VL image/video preprocessing, pure C++ (no ggml).
//
// Pipeline (mirrors the checkpoint's Qwen2VLImageProcessor slow path +
// MageVLVideoProcessor / processing_mage_vl.py):
//   smart_resize (factor = patch*merge = 32, min_pixels = 3136, max_pixels)
//   -> bicubic resize with antialias (PIL BICUBIC semantics, resample=3)
//   -> rescale 1/255, normalize (x - mean) / std   (CLIP stats)
//   -> patchify into (3*patch*patch, N) in *block layout* order
//        stream order = [t, hm, wm, a, b]   (a,b = the 2x2 merge group)
//      consecutive 4 rows are exactly one merger group - the Mage merger is a
//      plain reshape of the whole stream (no pooling, no permute).
//   -> patch positions (t,h,w) per patch, block layout, used for the per-dim
//      rotary tables (freq split 4:6:6 over the 32 half-dims, each dim its own
//      frequency - NOT the standard one-frequency-per-pair RoPE).
#pragma once

#include "core/media/resample.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace mage {

struct VisConfig {
    int patch = 16;
    int merge = 2;
    int window = 4;          // frame_windows_size: attention chunk (frames per block)
    int n_embd = 1024;
    int n_head = 16;
    int n_layer = 24;
    int n_ff = 4096;
    int freq_t = 8, freq_h = 12, freq_w = 12;   // per-dim freq split of one half
    float eps = 1e-6f;
    float rope_theta = 10000.0f;
    float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
};

// Normalize (CLIP stats) + patchify one frame into block layout.
// `rgb` is interleaved RGB float [h*w*3] in [0,255] (already resized).
// Output columns appended to `patches` ([C*P*P, N_total], ggml ne order).
inline void patchify_block(const float * rgb, int h, int w, int patch,
                           const float mean[3], const float stdv[3],
                           std::vector<float> & patches, int * grid_h, int * grid_w) {
    const int gh = h / patch, gw = w / patch, cpp = 3 * patch * patch;
    *grid_h = gh;
    *grid_w = gw;
    const size_t base = patches.size();
    patches.resize(base + (size_t)gh * gw * cpp);
    // block layout: for each merge group (hm, wm), the 2x2 inner patches (a, b)
    for (int hm = 0; hm < gh / 2; hm++) {
        for (int wm = 0; wm < gw / 2; wm++) {
            for (int a = 0; a < 2; a++) {
                for (int b = 0; b < 2; b++) {
                    const int r0 = (hm * 2 + a) * patch, c0 = (wm * 2 + b) * patch;
                    float * col = patches.data() + base +
                        ((size_t)(hm * (gw / 2) + wm) * 4 + a * 2 + b) * cpp;
                    for (int c = 0; c < 3; c++) {
                        const float m = mean[c], s = stdv[c];
                        for (int y = 0; y < patch; y++) {
                            const float * src = rgb + ((size_t)(r0 + y) * w + c0) * 3 + c;
                            float * dst = col + (size_t)c * patch * patch + y * patch;
                            for (int x = 0; x < patch; x++)
                                dst[x] = (src[x * 3] / 255.0f - m) / s;
                        }
                    }
                }
            }
        }
    }
}

// Patch positions (t,h,w) for T frames of one grid (h,w), block layout.
// t_coords run 0..T-1 (frames backend); codec windows pass real frame numbers.
// Output: pos[s*3 + {0,1,2}] = (t,h,w) for stream row s.
inline void build_patch_positions(int t, int h, int w, int merge,
                                  std::vector<int> & pos) {
    const int gh = h, gw = w;   // patch grid dims
    // row-major (t, gh, gw) then block permute [t, gh/merge, gw/merge, merge, merge]
    pos.resize((size_t)t * gh * gw * 3);
    size_t out = 0;
    for (int ti = 0; ti < t; ti++)
        for (int hm = 0; hm < gh / merge; hm++)
            for (int wm = 0; wm < gw / merge; wm++)
                for (int a = 0; a < merge; a++)
                    for (int b = 0; b < merge; b++) {
                        pos[out * 3 + 0] = ti;
                        pos[out * 3 + 1] = hm * merge + a;
                        pos[out * 3 + 2] = wm * merge + b;
                        out++;
                    }
}

// Per-dim rotary tables for a stream of n patches with positions `pos` [n,3].
// inv_freq[d] (d in one half, 32 dims): t split | h split | w split, each
// theta^(-i / split). The reference doubles the half table (cat[f, f] -> 64
// dims) and rotates interleaved pairs, so pair j (head dims 2j, 2j+1) uses
// inv_freq[(2j) % half] and inv_freq[(2j+1) % half] - the two dims of a pair
// rotate by DIFFERENT angles.
// Tables are shaped [pairs, n] (row-major over (n, pairs)) to broadcast
// against the re/im views inside the graph.
inline void rope_tables(const VisConfig & vc, const int * pos, int n,
                        std::vector<float> & cos_a, std::vector<float> & sin_a,
                        std::vector<float> & cos_b, std::vector<float> & sin_b) {
    const int hd = vc.n_embd / vc.n_head;
    const int half = hd / 2, pairs = hd / 2;
    float inv_freq[64];
    for (int d = 0; d < half; d++) {
        int split, i;
        if (d < vc.freq_t)               { split = vc.freq_t; i = d; }
        else if (d < vc.freq_t + vc.freq_h) { split = vc.freq_h; i = d - vc.freq_t; }
        else                              { split = vc.freq_w; i = d - vc.freq_t - vc.freq_h; }
        inv_freq[d] = std::pow(vc.rope_theta, -(float)i / (float)split);
    }
    cos_a.resize((size_t)n * pairs); sin_a.resize((size_t)n * pairs);
    cos_b.resize((size_t)n * pairs); sin_b.resize((size_t)n * pairs);
    for (int s = 0; s < n; s++) {
        const float th[3] = { (float)pos[s * 3 + 0], (float)pos[s * 3 + 1], (float)pos[s * 3 + 2] };
        auto angle = [&](int d) {
            float p;
            if (d < vc.freq_t) p = th[0];
            else if (d < vc.freq_t + vc.freq_h) p = th[1];
            else p = th[2];
            return p * inv_freq[d];
        };
        for (int j = 0; j < pairs; j++) {
            const int de = (2 * j) % half, dof = (2 * j + 1) % half;
            cos_a[(size_t)s * pairs + j] = std::cos(angle(de));
            sin_a[(size_t)s * pairs + j] = std::sin(angle(de));
            cos_b[(size_t)s * pairs + j] = std::cos(angle(dof));
            sin_b[(size_t)s * pairs + j] = std::sin(angle(dof));
        }
    }
}

}  // namespace mage
