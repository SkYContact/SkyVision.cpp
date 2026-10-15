// moondream_preprocess.cpp - see .hpp. Pure C++ port of the official
// moondream/torch/image_crops.py + PIL Resample.c (bit-exact target).
#include "moondream_preprocess.hpp"

#include <algorithm>
#include <cmath>

static const double MD_PI = 3.14159265358979323846;
#include <cstring>
#include <vector>

namespace mdprep {

// ---------------------------------------------------------------- tiling
void select_tiling(int height, int width, int crop_window, int max_crops,
                   int * th_out, int * tw_out) {
    int th = 1, tw = 1;
    if (height > crop_window || width > crop_window) {
        int min_h = (height + crop_window - 1) / crop_window;
        int min_w = (width + crop_window - 1) / crop_window;
        if (min_h * min_w > max_crops) {
            double ratio = std::sqrt((double)max_crops / (min_h * (double)min_w));
            th = std::max(1, (int)std::floor(min_h * ratio));
            tw = std::max(1, (int)std::floor(min_w * ratio));
        } else {
            th = (int)std::floor(std::sqrt(max_crops * (double)height / width));
            tw = (int)std::floor(std::sqrt(max_crops * (double)width / height));
            th = std::max(th, min_h);
            tw = std::max(tw, min_w);
            if (th * tw > max_crops) {
                if (tw > th) tw = max_crops / th;
                else         th = max_crops / tw;
            }
        }
    }
    *th_out = std::max(1, th);
    *tw_out = std::max(1, tw);
}

// ---------------------------------------------------------------- Lanczos
// PIL Resample.c replication for 8-bit channels.

static inline double sinc(double x) {
    if (x == 0.0) return 1.0;
    x *= MD_PI;
    return std::sin(x) / x;
}

static inline double lanczos_filter(double x) {  // support = 3.0
    if (x < 0.0) x = -x;
    if (x < 3.0) return sinc(x) * sinc(x / 3.0);
    return 0.0;
}

static const int PRECISION_BITS = 32 - 8 - 2;  // 22

struct Coeffs {
    int ksize = 0;
    std::vector<int> bounds;        // [out*2]: xmin, count
    std::vector<int32_t> kk;        // fixpoint 1<<22, PIL rounding
};

// box = (0, in) full axis; PIL in0/in1 defaults.
static void precompute_coeffs(int in_size, int out_size, Coeffs & c) {
    const double support_filter = 3.0;
    double scale = (double)in_size / out_size;   // PIL: (in1-in0)/out
    double filterscale = scale < 1.0 ? 1.0 : scale;
    double support = support_filter * filterscale;
    int ksize = (int)std::ceil(support) * 2 + 1;

    c.ksize = ksize;
    c.bounds.assign(out_size * 2, 0);
    c.kk.assign((size_t)out_size * ksize, 0);

    double inv_fs = 1.0 / filterscale;
    for (int xx = 0; xx < out_size; xx++) {
        double center = (xx + 0.5) * scale;
        int xmin = (int)(center - support + 0.5);
        int xmax = (int)(center + support + 0.5);
        if (xmin < 0) xmin = 0;
        if (xmax > in_size) xmax = in_size;
        int count = xmax - xmin;
        std::vector<double> w(count);
        double ww = 0.0;
        for (int x = 0; x < count; x++) {
            w[x] = lanczos_filter((x + xmin - center + 0.5) * inv_fs);
            ww += w[x];
        }
        int32_t * k = &c.kk[(size_t)xx * ksize];
        if (ww != 0.0) {
            for (int x = 0; x < count; x++) {
                double v = w[x] / ww;
                // PIL normalize_coeffs_8bpc rounding
                k[x] = (int32_t)(v < 0 ? (-0.5 + v * (1 << PRECISION_BITS))
                                       : ( 0.5 + v * (1 << PRECISION_BITS)));
            }
        }
        c.bounds[xx * 2 + 0] = xmin;
        c.bounds[xx * 2 + 1] = count;
    }
}

static inline uint8_t clip8(int64_t v) {  // PIL clip8 lookup equivalent
    v >>= PRECISION_BITS;
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

void lanczos_resize(const uint8_t * src, int sw, int sh, int sc,
                    int dw, int dh, uint8_t * dst) {
    if (sw == dw && sh == dh) {
        std::memcpy(dst, src, (size_t)sw * sh * sc);
        return;
    }
    bool need_h = dw != sw, need_v = dh != sh;
    // Pillow 10.4 (the version the parity reference uses) always runs the
    // horizontal pass first; the vertical-first heuristic is a newer-Pillow
    // addition. Keep the exact order the reference resampler uses.
    (void)0;

    Coeffs ch, cv;
    if (need_v) precompute_coeffs(sh, dh, cv);
    if (need_h) precompute_coeffs(sw, dw, ch);

    {
        // pass 1: horizontal -> temp [dw, sh] (uint8 per channel, like PIL)
        if (!need_h) {
            // vertical only: src -> dst
            for (int yy = 0; yy < dh; yy++) {
                int ymin = cv.bounds[yy * 2], cnt = cv.bounds[yy * 2 + 1];
                const int32_t * k = &cv.kk[(size_t)yy * cv.ksize];
                uint8_t * out = dst + (size_t)yy * dw * sc;
                for (int xx = 0; xx < dw; xx++) {
                    for (int ci = 0; ci < sc; ci++) {
                        int64_t ss = 1 << (PRECISION_BITS - 1);
                        for (int y = 0; y < cnt; y++)
                            ss += (int64_t)src[(size_t)(y + ymin) * sw * sc +
                                               (size_t)xx * sc + ci] * k[y];
                        out[(size_t)xx * sc + ci] = clip8(ss);
                    }
                }
            }
            return;
        }
        std::vector<uint8_t> tmp((size_t)dw * sh * sc);
        for (int y = 0; y < sh; y++) {
            const uint8_t * in = src + (size_t)y * sw * sc;
            uint8_t * out = tmp.data() + (size_t)y * dw * sc;
            for (int xx = 0; xx < dw; xx++) {
                int xmin = ch.bounds[xx * 2], cnt = ch.bounds[xx * 2 + 1];
                const int32_t * k = &ch.kk[(size_t)xx * ch.ksize];
                for (int ci = 0; ci < sc; ci++) {
                    // PIL accumulates per channel into int with round bias
                    int64_t ss = 1 << (PRECISION_BITS - 1);
                    for (int x = 0; x < cnt; x++)
                        ss += (int64_t)in[(size_t)(x + xmin) * sc + ci] * k[x];
                    out[(size_t)xx * sc + ci] = clip8(ss);
                }
            }
        }
        if (!need_v) { std::memcpy(dst, tmp.data(), tmp.size()); return; }
        // pass 2: vertical tmp -> dst
        for (int yy = 0; yy < dh; yy++) {
            int ymin = cv.bounds[yy * 2], cnt = cv.bounds[yy * 2 + 1];
            const int32_t * k = &cv.kk[(size_t)yy * cv.ksize];
            uint8_t * out = dst + (size_t)yy * dw * sc;
            for (int xx = 0; xx < dw; xx++) {
                for (int ci = 0; ci < sc; ci++) {
                    int64_t ss = 1 << (PRECISION_BITS - 1);
                    for (int y = 0; y < cnt; y++)
                        ss += (int64_t)tmp[(size_t)(y + ymin) * dw * sc +
                                           (size_t)xx * sc + ci] * k[y];
                    out[(size_t)xx * sc + ci] = clip8(ss);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- prepare_crops
std::vector<float> prepare_crops(const uint8_t * rgb, int w, int h,
                                 int * th_out, int * tw_out, int * n_crops_out,
                                 int max_crops, int overlap_margin,
                                 int crop_size, int patch_size) {
    const int margin = patch_size * overlap_margin;          // 56
    const int total_margin = margin * 2;                     // 112
    const int crop_patches = crop_size / patch_size;         // 27
    const int crop_window_patches = crop_patches - 2 * overlap_margin; // 19
    const int crop_window = crop_window_patches * patch_size;          // 266

    int th, tw;
    select_tiling(h - total_margin, w - total_margin, crop_window, max_crops,
                  &th, &tw);
    if (th_out) *th_out = th;
    if (tw_out) *tw_out = tw;

    const int n_crops = th * tw + 1;
    const int tgt_h = th * crop_window + total_margin;
    const int tgt_w = tw * crop_window + total_margin;

    // resized full image (for local crops) + global crop
    std::vector<uint8_t> tiled((size_t)tgt_h * tgt_w * 3);
    lanczos_resize(rgb, w, h, 3, tgt_w, tgt_h, tiled.data());
    std::vector<uint8_t> global((size_t)crop_size * crop_size * 3);
    lanczos_resize(rgb, w, h, 3, crop_size, crop_size, global.data());

    std::vector<float> out((size_t)n_crops * 3 * crop_size * crop_size);
    auto store_crop = [&](int idx, const uint8_t * src, int stride_px) {
        const size_t stride = (size_t)stride_px * 3;
        float * o = out.data() + (size_t)idx * 3 * crop_size * crop_size;
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < crop_size; y++)
                for (int x = 0; x < crop_size; x++) {
                    float v = (float)src[(size_t)y * stride + (size_t)x * 3 + c];
                    o[((size_t)c * crop_size + y) * crop_size + x] =
                        ((v / 255.0f) - 0.5f) / 0.5f;
                }
    };
    store_crop(0, global.data(), crop_size);

    for (int i = 0; i < th; i++) {
        for (int j = 0; j < tw; j++) {
            const int y0 = i * crop_window, x0 = j * crop_window;
            const int y_end = std::min(y0 + crop_size, tgt_h);
            const int x_end = std::min(x0 + crop_size, tgt_w);
            // zero-padded crop (crops buffer is pre-zeroed by vector{})
            const int idx = 1 + i * tw + j;
            float * o = out.data() + (size_t)idx * 3 * crop_size * crop_size;
            for (int c = 0; c < 3; c++)
                for (int y = y0; y < y_end; y++)
                    for (int x = x0; x < x_end; x++) {
                        float v = (float)tiled[(size_t)y * tgt_w * 3 +
                                               (size_t)x * 3 + c];
                        o[((size_t)c * crop_size + (y - y0)) * crop_size +
                          (x - x0)] = ((v / 255.0f) - 0.5f) / 0.5f;
                    }
        }
    }
    if (n_crops_out) *n_crops_out = n_crops;
    return out;
}

} // namespace mdprep

#ifdef PREPROCESS_TEST_MAIN
// CLI parity harness (driven by scripts/test_preprocess.py):
//   moondream-preprocess-test --img img.bin --w W --h H --out out.bin
// img.bin: raw RGB8. out.bin: int32 n_crops, th, tw, then n*3*378*378 float32.
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char ** argv) {
    std::string img, outp;
    int W = 0, H = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--img")) img = argv[++i];
        else if (!strcmp(argv[i], "--out")) outp = argv[++i];
        else if (!strcmp(argv[i], "--w")) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--h")) H = atoi(argv[++i]);
    }
    if (img.empty() || outp.empty() || W <= 0 || H <= 0) {
        fprintf(stderr, "usage: %s --img file --w W --h H --out file\n", argv[0]);
        return 1;
    }
    std::vector<uint8_t> rgb((size_t)W * H * 3);
    FILE * f = fopen(img.c_str(), "rb");
    if (!f || fread(rgb.data(), 1, rgb.size(), f) != rgb.size()) {
        fprintf(stderr, "read img failed\n"); return 1;
    }
    fclose(f);

    int th, tw, n;
    std::vector<float> crops = mdprep::prepare_crops(rgb.data(), W, H,
                                                     &th, &tw, &n);
    fprintf(stderr, "preprocess: %dx%d -> tiling %dx%d, %d crops\n",
            W, H, th, tw, n);

    f = fopen(outp.c_str(), "wb");
    const int32_t hdr[3] = {n, th, tw};
    fwrite(hdr, 4, 3, f);
    fwrite(crops.data(), 4, crops.size(), f);
    fclose(f);
    return 0;
}
#endif
