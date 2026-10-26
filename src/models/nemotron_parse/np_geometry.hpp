// np_geometry.hpp - NaViT geometry for the Nemotron-3.5-Parse vision front end.
//
// Two host-side pieces the graph cannot express, both taken from the official
// processor / `hf_prefix_geometry.py` and verified against torch:
//
//   * the aspect-preserving resize + NaViT grid pad (width to a multiple of 64
//     because the neck merges 4 patch columns, height to a multiple of 16)
//   * the content-normalized 2D position field: the learned 128x128 RADIO table
//     is bilinearly (antialiased) resampled to the page's CONTENT patch grid,
//     then sampled per patch of the PADDED grid at the patch centre expressed in
//     content coordinates. Padding columns therefore land on the border value,
//     which is what the model was trained with.
//
// The antialias weights reproduce torch's
// `F.interpolate(mode="bilinear", antialias=True, align_corners=False)`:
//
//   scale  = in / out
//   filt   = max(scale, 1)
//   center = scale * (i + 0.5) - 0.5                  (pixel-index coordinates)
//   i range: floor(center - filt + 0.5) .. ceil(center + filt - 0.5)
//   w(j)   = max(0, 1 - |j - center| / filt)          (triangle, support 1)
//   taps outside [0, in) are DROPPED (not clamped) and the row is renormalised
//
// measured max |diff| vs torch over 128->13 / 128->26 / 128->61 / 60->13 is
// ~2e-6 (float32), i.e. the resampler is not the parity bottleneck.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace nemotron_parse {

// Padded patch grid + the content rectangle inside it, in pixels.
struct NavitGrid {
    int rows = 0, cols = 0;                  // padded grid, in patches
    int content_rows = 0, content_cols = 0;  // content grid, in patches
    int content_w = 0, content_h = 0;        // content size, in pixels
    int padded_w = 0, padded_h = 0;
};

inline int round_up(int v, int m) { return ((std::max(v, 1) + m - 1) / m) * m; }

// The processor's `_resize_with_aspect_ratio`: only ever shrinks, and never
// stretches past the (height, width) canvas.
inline void aspect_size(int w0, int h0, int max_h, int max_w, int * out_w, int * out_h) {
    const double ar = (double)w0 / (double)std::max(h0, 1);
    int h = h0, w = w0;
    if (h > max_h) {
        h = max_h;
        w = (int)(h * ar);
    }
    if (w > max_w) {
        w = max_w;
        h = (int)(w / ar);
    }
    *out_w = std::max(w, 1);
    *out_h = std::max(h, 1);
}

inline NavitGrid make_grid(int content_w, int content_h, int patch = 16) {
    NavitGrid g;
    g.content_w = content_w;
    g.content_h = content_h;
    g.padded_w = round_up(content_w, 64);
    g.padded_h = round_up(content_h, 16);
    g.rows = g.padded_h / patch;
    g.cols = g.padded_w / patch;
    g.content_rows = (content_h + patch - 1) / patch;
    g.content_cols = (content_w + patch - 1) / patch;
    return g;
}

// Sparse separable resample weights: out i reads in[off[i] .. off[i] + n[i])
// with in[off[i] + j] scaled by w[woff[i] + j].
struct AxisWeights {
    std::vector<int> off, n, woff;
    std::vector<float> w;   // concatenated

    void build(int n_in, int n_out) {
        off.assign((size_t)n_out, 0);
        n.assign((size_t)n_out, 0);
        woff.assign((size_t)n_out, 0);
        w.clear();
        const double scale = (double)n_in / (double)std::max(n_out, 1);
        const double filt = std::max(scale, 1.0);
        for (int i = 0; i < n_out; i++) {
            const double center = scale * (i + 0.5) - 0.5;
            const int lo = (int)std::floor(center - filt + 0.5);
            const int hi = (int)std::ceil(center + filt - 0.5);
            const size_t base = w.size();
            double acc = 0.0;
            int first = -1, last = -1;
            for (int j = lo; j <= hi; j++) {
                if (j < 0 || j >= n_in) continue;          // dropped, not clamped
                const double t = 1.0 - std::fabs(((double)j - center) / filt);
                if (t <= 0.0) continue;
                if (first < 0) first = j;
                last = j;
                w.push_back((float)t);
                acc += t;
            }
            if (first < 0) {                                // degenerate: nearest tap
                const int j = std::min(std::max((int)std::lround(center), 0), n_in - 1);
                first = last = j;
                w.push_back(1.0f);
                acc = 1.0;
            }
            const float inv = (float)(1.0 / acc);
            for (size_t k = base; k < w.size(); k++) w[k] *= inv;
            off[(size_t)i] = first;
            n[(size_t)i] = last - first + 1;
            woff[(size_t)i] = (int)base;
        }
    }
};

// [C, sh, sw] float planes -> [C, dh, dw], separable, torch-AA semantics.
inline void interp_bilinear_aa(const float * src, int C, int sh, int sw,
                               float * dst, int dh, int dw) {
    AxisWeights wh, ww;
    wh.build(sh, dh);
    ww.build(sw, dw);
    std::vector<float> tmp((size_t)C * dh * sw);
    // vertical pass: [C, sh, sw] -> [C, dh, sw]
    for (int c = 0; c < C; c++) {
        const float * s = src + (size_t)c * sh * sw;
        float * t = tmp.data() + (size_t)c * dh * sw;
        for (int y = 0; y < dh; y++) {
            const int o = wh.off[(size_t)y], n = wh.n[(size_t)y];
            const float * wgt = wh.w.data() + wh.woff[(size_t)y];
            float * row = t + (size_t)y * sw;
            std::fill(row, row + sw, 0.0f);
            for (int j = 0; j < n; j++) {
                const float wv = wgt[j];
                const float * sr = s + (size_t)(o + j) * sw;
                for (int x = 0; x < sw; x++) row[x] += wv * sr[x];
            }
        }
    }
    // horizontal pass: [C, dh, sw] -> [C, dh, dw]
    for (int c = 0; c < C; c++) {
        const float * s = tmp.data() + (size_t)c * dh * sw;
        float * d = dst + (size_t)c * dh * dw;
        for (int y = 0; y < dh; y++) {
            const float * row = s + (size_t)y * sw;
            float * out = d + (size_t)y * dw;
            for (int x = 0; x < dw; x++) {
                const int o = ww.off[(size_t)x], n = ww.n[(size_t)x];
                const float * wgt = ww.w.data() + ww.woff[(size_t)x];
                float acc = 0.0f;
                for (int j = 0; j < n; j++) acc += wgt[j] * row[o + j];
                out[x] = acc;
            }
        }
    }
}

// grid_sample(mode="bilinear", padding_mode="border", align_corners=False) of a
// single 2D plane: reads one value at the normalised coordinate (gx, gy) in
// [-1, 1].
inline float sample_bilinear_border(const float * plane, int h, int w, float gx, float gy) {
    const float fx = std::min(std::max(((gx + 1.0f) * 0.5f) * (float)w - 0.5f, 0.0f), (float)(w - 1));
    const float fy = std::min(std::max(((gy + 1.0f) * 0.5f) * (float)h - 0.5f, 0.0f), (float)(h - 1));
    const int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
    const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
    const float tx = fx - (float)x0, ty = fy - (float)y0;
    const float a = plane[(size_t)y0 * w + x0], b = plane[(size_t)y0 * w + x1];
    const float c = plane[(size_t)y1 * w + x0], d = plane[(size_t)y1 * w + x1];
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

// Content-normalised position field for one page:
//   pos_embed [dim, side*side] (row-major 128x128), grid from make_grid()
//   out       [dim, rows*cols]  column-major per token (out[c + dim*idx])
inline void build_positions(const float * pos_embed, int dim, int side,
                            const NavitGrid & g, int patch, std::vector<float> & out) {
    const int cr = g.content_rows, cc = g.content_cols;
    std::vector<float> table((size_t)dim * cr * cc);
    if (cr == side && cc == side) {
        std::copy(pos_embed, pos_embed + (size_t)dim * side * side, table.begin());
    } else {
        interp_bilinear_aa(pos_embed, dim, side, side, table.data(), cr, cc);
    }

    const int n_tok = g.rows * g.cols;
    out.assign((size_t)dim * n_tok, 0.0f);

    // per-patch centres, in content-normalised [0,1]
    std::vector<float> ny((size_t)g.rows), nx((size_t)g.cols);
    for (int r = 0; r < g.rows; r++) {
        const float start = (float)(r * patch);
        const float extent = std::min(std::max((float)g.content_h - start, 0.0f), (float)patch);
        ny[(size_t)r] = std::min(start + 0.5f * extent, (float)g.content_h) / (float)g.content_h;
    }
    for (int c = 0; c < g.cols; c++) {
        const float start = (float)(c * patch);
        const float extent = std::min(std::max((float)g.content_w - start, 0.0f), (float)patch);
        nx[(size_t)c] = std::min(start + 0.5f * extent, (float)g.content_w) / (float)g.content_w;
    }

    std::vector<float> plane((size_t)cr * cc);
    for (int c = 0; c < dim; c++) {
        const float * src_c = table.data() + (size_t)c * cr * cc;
        std::copy(src_c, src_c + (size_t)cr * cc, plane.begin());
        for (int r = 0; r < g.rows; r++) {
            const float gy = 2.0f * ny[(size_t)r] - 1.0f;
            for (int col = 0; col < g.cols; col++) {
                const float gx = 2.0f * nx[(size_t)col] - 1.0f;
                out[(size_t)c + (size_t)dim * ((size_t)r * g.cols + col)] =
                    sample_bilinear_border(plane.data(), cr, cc, gx, gy);
            }
        }
    }
}

// 16x16 patches, row-major over the grid, element order (channel, dy, dx) - the
// same order the official processor's unfold/permute produces. Pixels outside
// the content rectangle are the white pad the processor adds.
inline void pack_patches(const uint8_t * rgb, int w, int h, const NavitGrid & g,
                         const float mean[3], const float std_[3], int patch,
                         std::vector<float> & out) {
    const int n_tok = g.rows * g.cols;
    out.assign((size_t)n_tok * 3 * patch * patch, 0.0f);
    const float pad_value = 255.0f;
    for (int r = 0; r < g.rows; r++)
        for (int c = 0; c < g.cols; c++) {
            float * row = out.data() + (size_t)(r * g.cols + c) * 3 * patch * patch;
            int o = 0;
            for (int ch = 0; ch < 3; ch++)
                for (int dy = 0; dy < patch; dy++) {
                    const int py = r * patch + dy;
                    for (int dx = 0; dx < patch; dx++) {
                        const int px = c * patch + dx;
                        const float v = (py < h && px < w) ? (float)rgb[((size_t)py * w + px) * 3 + ch]
                                                          : pad_value;
                        row[o++] = (v / 255.0f - mean[ch]) / std_[ch];
                    }
                }
        }
}

}  // namespace nemotron_parse
