// videomae_preprocess.cpp - see videomae_preprocess.hpp.
#include "videomae_preprocess.hpp"

#include "resample.hpp"   // vqa::img::resize_bicubic_antialias (PIL semantics)

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace videomae {

void sample_frame_indices(int n, int want, std::vector<int> & idx) {
    idx.clear();
    idx.reserve((size_t)want);
    if (n <= 0 || want <= 0) return;
    for (int i = 0; i < want; i++) {
        // numpy.linspace(0, n-1, want).astype(int): round to nearest on the
        // half-open grid, then truncate - exact for n == want (identity)
        const int j = want > 1 ? (int)((long long)i * (n - 1) / (want - 1)) : 0;
        idx.push_back(std::min(j, n - 1));
    }
}

void preprocess_frame(const uint8_t * rgb, int w, int h, const PreConfig & pc, float * dst) {
    const int S = pc.image_size;
    const int R = pc.resize_shortest > 0 ? pc.resize_shortest : S;
    // shortest edge -> R, the other side scaled by the aspect ratio (HF uses an
    // int() cast there, so truncation rather than rounding)
    int rw = w, rh = h;
    if (w <= h) {
        rw = R;
        rh = std::max(R, (int)((double)R * h / (double)w));
    } else {
        rh = R;
        rw = std::max(R, (int)((double)R * w / (double)h));
    }

    std::vector<float> resized;
    // interleaved RGB, [0,255]; HF's `resample` is a PILImageResampling value
    vqa::img::resize_antialias(rgb, w, h, rw, rh, resized,
                               /*bilinear=*/pc.resample == 2, /*round_u8=*/true);

    const int oy = (rh - S) / 2, ox = (rw - S) / 2;
    for (int y = 0; y < S; y++) {
        const float * row = resized.data() + ((size_t)(y + oy) * rw + ox) * 3;
        for (int x = 0; x < S; x++) {
            for (int c = 0; c < 3; c++) {
                const float v = row[(size_t)x * 3 + c] / 255.0f;
                dst[(size_t)c * S * S + (size_t)y * S + x] = (v - pc.mean[c]) / pc.std[c];
            }
        }
    }
}

bool pack_clip(const vqa::MediaInput & media, const PreConfig & pc,
               std::vector<float> & patches, int * n_tokens, int * grid_t, int * grid_h,
               int * grid_w) {
    *n_tokens = 0;
    *grid_t = *grid_h = *grid_w = 0;
    if (media.empty()) return false;

    const int S = pc.image_size, P = pc.patch, TU = pc.tubelet, T = pc.num_frames;
    const int gt = T / TU, gh = S / P, gw = S / P;
    std::vector<int> idx;
    sample_frame_indices((int)media.frames.size(), T, idx);

    // planar pixels, one contiguous CHW block per sampled frame: planes[t][c][y][x]
    std::vector<float> planes((size_t)3 * T * S * S);
    for (int t = 0; t < T; t++)
        preprocess_frame(media.frames[(size_t)idx[(size_t)t]].rgb.data(), media.width,
                         media.height, pc, planes.data() + (size_t)t * 3 * S * S);

    const int C3 = 3 * TU * P * P;
    *grid_t = gt;
    *grid_h = gh;
    *grid_w = gw;
    *n_tokens = gt * gh * gw;
    patches.assign((size_t)C3 * (*n_tokens), 0.0f);

    int u = 0;
    for (int tp = 0; tp < gt; tp++)
        for (int yy = 0; yy < gh; yy++)
            for (int xx = 0; xx < gw; xx++) {
                float * row = patches.data() + (size_t)u * C3;
                int v = 0;
                for (int c = 0; c < 3; c++)
                    for (int dt = 0; dt < TU; dt++) {
                        const int t = tp * TU + dt;
                        const float * base = planes.data() + ((size_t)t * 3 + c) * S * S;
                        for (int dh = 0; dh < P; dh++) {
                            const float * src = base + (size_t)(yy * P + dh) * S + xx * P;
                            for (int dw = 0; dw < P; dw++) row[v++] = src[dw];
                        }
                    }
                u++;
            }
    return true;
}

}  // namespace videomae
