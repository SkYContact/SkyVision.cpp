#include "vc3_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vc3 {

// smart_resize + resize_bicubic_antialias live in core/resample.cpp now.

// torch F.interpolate(mode="bicubic", align_corners=False) kernel (a = -0.75)
static inline double cubic_torch(double x, double A) {
    x = std::fabs(x);
    if (x <= 1.0) return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    if (x < 2.0) return ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    return 0.0;
}

// ---------------------------------------------------------------- patchify
void patchify(const float * rgb, int h, int w, const VisConfig & vc,
              std::vector<float> & patches, int * grid_h, int * grid_w) {
    const int P = vc.patch, M = vc.merge;
    const int gh = h / P, gw = w / P;
    const int nh = gh / M, nw = gw / M;
    const int C = 3 * P * P;
    const int N = gh * gw;
    *grid_h = gh;
    *grid_w = gw;
    patches.assign((size_t)C * N, 0.0f);

    const float inv = 1.0f / 127.5f;   // (x/255 - 0.5)/0.5
    // tokens in merge-block order (nh, nw, kh, kw) — the official processor's
    // layout (verified by exact-match against Qwen2VLImageProcessorFast output).
    // The merger's reshape chain in encode_vision assumes this order; an earlier
    // switch to row-major (gh, gw) fixed pos/rope alignment but silently made the
    // 2x2 pooling mix patches from different grid rows — scene-level content
    // survived, fine detail (small text) was destroyed.
    for (int H2 = 0; H2 < nh; H2++) {
        for (int W2 = 0; W2 < nw; W2++) {
            for (int ph = 0; ph < M; ph++) {
                for (int pw = 0; pw < M; pw++) {
                    const int hi = H2 * M + ph, wi = W2 * M + pw;
                    const size_t n = (size_t)((H2 * nw + W2) * M + ph) * M + pw;
                    float * dst = &patches[n * C];
                    for (int c = 0; c < 3; c++) {
                        for (int p = 0; p < P; p++) {
                            const float * row = rgb + ((size_t)(hi * P + p) * w + wi * P) * 3 + c;
                            float * drow = dst + (size_t)(c * P + p) * P;
                            for (int q = 0; q < P; q++) {
                                drow[q] = row[(size_t)q * 3] * inv - 1.0f;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- pos embedding
// bicubic interpolation of (sh,sw,C) -> (dh,dw,C), torch F.interpolate semantics
static void interpolate_bicubic_torch(const float * src, int sh, int sw, int dh, int dw, int C,
                                      std::vector<float> & out) {
    static const double A = -0.75;
    out.assign((size_t)dh * dw * C, 0.0f);
    if (sh == dh && sw == dw) { memcpy(out.data(), src, sizeof(float) * (size_t)sh * sw * C); return; }

    const double rh = (double)sh / (double)dh;
    const double rw = (double)sw / (double)dw;

    // precompute tap indices + weights per output row / column
    struct Taps { int i0[4]; double w[4]; };
    auto make_taps = [](int in, int out, double ratio) {
        std::vector<Taps> ts(out);
        for (int o = 0; o < out; o++) {
            double real = ((double)o + 0.5) * ratio - 0.5;
            int i0 = (int)std::floor(real);
            double lam = real - i0;
            Taps & t = ts[o];
            for (int k = 0; k < 4; k++) {
                int idx = i0 - 1 + k;
                int c = std::min(std::max(idx, 0), in - 1);     // clamped read
                t.i0[k] = c;
                t.w[k] = cubic_torch((double)idx - real, A);
            }
        }
        return ts;
    };
    std::vector<Taps> th = make_taps(sh, dh, rh);
    std::vector<Taps> tw = make_taps(sw, dw, rw);

    std::vector<float> tmp((size_t)dh * sw * C);
    for (int y = 0; y < dh; y++) {
        const Taps & t = th[y];
        for (int x = 0; x < sw; x++) {
            for (int c = 0; c < C; c++) {
                double acc = 0;
                for (int k = 0; k < 4; k++)
                    acc += t.w[k] * (double)src[((size_t)t.i0[k] * sw + x) * C + c];
                tmp[((size_t)y * sw + x) * C + c] = (float)acc;
            }
        }
    }
    for (int y = 0; y < dh; y++) {
        for (int x = 0; x < dw; x++) {
            const Taps & t = tw[x];
            for (int c = 0; c < C; c++) {
                double acc = 0;
                for (int k = 0; k < 4; k++)
                    acc += t.w[k] * (double)tmp[((size_t)y * sw + t.i0[k]) * C + c];
                out[((size_t)y * dw + x) * C + c] = (float)acc;
            }
        }
    }
}

void pos_embed_for_grid(const float * weight, const float * time_weight, const VisConfig & vc,
                        int t, int h, int w, std::vector<float> & out) {
    const int D = vc.n_embd;
    std::vector<float> pos2d;
    if (h == vc.pos_h && w == vc.pos_w) {
        pos2d.assign(weight, weight + (size_t)h * w * D);
    } else {
        interpolate_bicubic_torch(weight, vc.pos_h, vc.pos_w, h, w, D, pos2d);
    }
    const size_t per_t = (size_t)h * w * D;
    out.assign(per_t * t, 0.0f);
    // positions are assigned SEQUENTIALLY to the token stream: the official
    // patch_embed consumes the interpolated (h, w) table row by row and the
    // official rope does the same (freqs_cis[:h,:w].reshape(-1)), regardless of
    // each token's true geometric position — the model was trained with the
    // processor's merge-block token order paired with sequential positions
    for (int ti = 0; ti < t; ti++) {
        const float * tv = time_weight + (size_t)std::min(ti, vc.temporal_merge - 1) * D;
        float * dst = out.data() + (size_t)ti * per_t;
        if (t == 1) {
            memcpy(dst, pos2d.data(), sizeof(float) * per_t);
        } else {
            for (size_t i = 0; i < per_t; i++) dst[i] = pos2d[i] + tv[i % D];
        }
    }
}

// ---------------------------------------------------------------- 2D rope
void rope_2d_tables(int hd, int t, int h, int w, float theta,
                    std::vector<float> & cos_out, std::vector<float> & sin_out) {
    const int pairs = hd / 2;          // complex values per head
    const int half = hd / 4;           // x pairs and y pairs
    const int npos = t * h * w;
    cos_out.assign((size_t)pairs * npos, 1.0f);
    sin_out.assign((size_t)pairs * npos, 0.0f);

    std::vector<double> freq(half);
    for (int i = 0; i < half; i++) freq[i] = 1.0 / std::pow((double)theta, (double)(4 * i) / (double)hd);

    for (int s = 0; s < npos; s++) {
        const int g = s % (h * w);      // row-major (h,w) position inside one frame
        const int ys = g / w, xs = g % w;
        for (int i = 0; i < half; i++) {
            const double ax = (double)xs * freq[i];
            const double ay = (double)ys * freq[i];
            // pair 2i (x) and pair 2i+1 (y)
            cos_out[(size_t)(2 * i) * npos + s] = (float)std::cos(ax);
            sin_out[(size_t)(2 * i) * npos + s] = (float)std::sin(ax);
            cos_out[(size_t)(2 * i + 1) * npos + s] = (float)std::cos(ay);
            sin_out[(size_t)(2 * i + 1) * npos + s] = (float)std::sin(ay);
        }
    }
}

}  // namespace vc3
