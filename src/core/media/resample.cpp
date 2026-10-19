// resample.cpp - see resample.hpp.
#include "core/media/resample.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vqa {
namespace img {

// ---------------------------------------------------------------- smart_resize
static int round_by_factor(int n, int f) { return (int)nearbyint((double)n / (double)f) * f; }
static int ceil_by_factor(double n, int f) { return (int)std::ceil(n / f) * f; }
static int floor_by_factor(double n, int f) { return (int)std::floor(n / f) * f; }

bool smart_resize(int height, int width, int factor, int min_pixels, int max_pixels,
                  int * out_h, int * out_w) {
    if (height < factor || width < factor) return false;
    int h_bar = std::max(factor, round_by_factor(height, factor));
    int w_bar = std::max(factor, round_by_factor(width, factor));
    if ((long long)h_bar * w_bar > (long long)max_pixels) {
        double beta = std::sqrt((double)height * (double)width / (double)max_pixels);
        h_bar = std::max(factor, floor_by_factor((double)height / beta, factor));
        w_bar = std::max(factor, floor_by_factor((double)width / beta, factor));
    } else if ((long long)h_bar * w_bar < (long long)min_pixels) {
        double beta = std::sqrt((double)min_pixels / ((double)height * (double)width));
        h_bar = ceil_by_factor((double)height * beta, factor);
        w_bar = ceil_by_factor((double)width * beta, factor);
    }
    *out_h = h_bar;
    *out_w = w_bar;
    return true;
}

// ---------------------------------------------------------------- resamplers
// antialias bicubic kernel used by PIL / torchvision (a = -0.5)
static inline double bicubic_aa(double x) {
    x = std::fabs(x);
    if (x <= 1.0) return (1.5 * x - 2.5) * x * x + 1.0;
    if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    return 0.0;
}

// PIL's BILINEAR kernel (triangle, support 1.0)
static inline double bilinear_aa(double x) {
    x = std::fabs(x);
    return x < 1.0 ? 1.0 - x : 0.0;
}

// PIL's LANCZOS kernel (a = 3): sinc(x) * sinc(x / 3)
static inline double lanczos_aa(double x) {
    const double ax = std::fabs(x);
    if (ax < 1e-12) return 1.0;
    if (ax >= 3.0) return 0.0;
    constexpr double kPi = 3.14159265358979323846;   // M_PI is not defined on MSVC
    const double px = kPi * ax;
    return (std::sin(px) / px) * (std::sin(px / 3.0) / (px / 3.0));
}

struct Filter {
    int start = 0;              // first source index
    std::vector<double> w;      // weights
};

// builds the antialias filter bank for one axis (PIL semantics: the support is
// widened by the downscale factor, which is what makes it antialiased)
static void build_aa_filters(int in, int out, std::vector<Filter>& fs, int pil_resample = 3) {
    const double scale = (double)in / (double)out;
    const double filterscale = std::max(1.0, scale);
    const double support = (pil_resample == 1 ? 3.0 : pil_resample == 2 ? 1.0 : 2.0) * filterscale;
    fs.resize(out);
    for (int o = 0; o < out; o++) {
        const double center = ((double)o + 0.5) * scale;
        int xmin = (int)std::floor(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = (int)std::ceil(center + support + 0.5);
        if (xmax > in) xmax = in;
        Filter & f = fs[o];
        f.start = xmin;
        f.w.resize(xmax - xmin);
        double sum = 0.0;
        for (int x = xmin; x < xmax; x++) {
            double t = ((double)x - center + 0.5) / filterscale;
            double w = pil_resample == 1 ? lanczos_aa(t)
                       : pil_resample == 2 ? bilinear_aa(t) : bicubic_aa(t);
            f.w[x - xmin] = w;
            sum += w;
        }
        if (sum != 0.0) for (double & w : f.w) w /= sum;
    }
}

void resize_antialias(const uint8_t * src, int sw, int sh, int dw, int dh,
                      std::vector<float>& out, int pil_resample, bool round_u8) {
    std::vector<Filter> fh, fv;
    build_aa_filters(sw, dw, fh, pil_resample);
    build_aa_filters(sh, dh, fv, pil_resample);

    // horizontal pass: (sw,sh) -> (dw,sh), float
    std::vector<float> tmp((size_t)dw * sh * 3);
    for (int y = 0; y < sh; y++) {
        const uint8_t * row = src + (size_t)y * sw * 3;
        for (int x = 0; x < dw; x++) {
            const Filter & f = fh[x];
            double acc[3] = {0, 0, 0};
            for (size_t k = 0; k < f.w.size(); k++) {
                const uint8_t * px = row + (size_t)(f.start + (int)k) * 3;
                acc[0] += f.w[k] * (double)px[0];
                acc[1] += f.w[k] * (double)px[1];
                acc[2] += f.w[k] * (double)px[2];
            }
            float * dst = &tmp[((size_t)y * dw + x) * 3];
            dst[0] = (float)acc[0];
            dst[1] = (float)acc[1];
            dst[2] = (float)acc[2];
        }
    }
    // vertical pass: (dw,sh) -> (dw,dh)
    out.assign((size_t)dw * dh * 3, 0.0f);
    for (int y = 0; y < dh; y++) {
        const Filter & f = fv[y];
        for (int x = 0; x < dw; x++) {
            double acc[3] = {0, 0, 0};
            for (size_t k = 0; k < f.w.size(); k++) {
                const float * px = &tmp[((size_t)(f.start + (int)k) * dw + x) * 3];
                acc[0] += f.w[k] * (double)px[0];
                acc[1] += f.w[k] * (double)px[1];
                acc[2] += f.w[k] * (double)px[2];
            }
            float * dst = &out[((size_t)y * dw + x) * 3];
            if (round_u8) {
                // torchvision resizes uint8 input to a uint8 output: the whole
                // pipeline stays float and rounds ONCE, at the end. Skipping
                // this leaves a ~1-2 grey level residual against the processor.
                for (int c = 0; c < 3; c++) {
                    int v = (int)(acc[c] + 0.5);
                    dst[c] = (float)(v < 0 ? 0 : v > 255 ? 255 : v);
                }
            } else {
                dst[0] = (float)acc[0];
                dst[1] = (float)acc[1];
                dst[2] = (float)acc[2];
            }
        }
    }
}

void resize_bicubic_antialias(const uint8_t * src, int sw, int sh, int dw, int dh,
                              std::vector<float>& out) {
    resize_antialias(src, sw, sh, dw, dh, out, /*pil_resample=*/3);
}

// ---------------------------------------------------------------- torch-style
// torchvision's uint8 antialias bicubic (what the HF Qwen2-VL/Omni fast image
// processor calls): float Keys-a=-0.5 weights normalized to 1, then quantized
// to int16 fixed point; each separable pass accumulates in integers, adds a
// half-step bias and shifts back, rounding to uint8 between passes. Bit-exact
// replication matters because the vision tower chaotically amplifies even
// sub-grey-level preprocessing differences.
static void build_tv_weights(int in, int out, std::vector<Filter>& fs,
                             int * precision) {
    const double scale = (double)in / (double)out;
    const double support = scale >= 1.0 ? 2.0 * scale : 2.0;
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    const int max_interp = (int)std::ceil(support) * 2 + 1;
    fs.resize(out);
    double wt_max = 0.0;
    for (int o = 0; o < out; o++) {
        const double center = scale * ((double)o + 0.5);
        int xmin = (int)(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xend = (int)(center + support + 0.5);
        if (xend > in) xend = in;
        int xsize = xend - xmin;
        if (xsize > max_interp) xsize = max_interp;
        if (xsize < 0) xsize = 0;
        Filter & f = fs[o];
        f.start = xmin;
        f.w.resize(xsize);
        double sum = 0.0;
        for (int j = 0; j < xsize; j++) {
            const double w = bicubic_aa((j + xmin - center + 0.5) * invscale);
            f.w[j] = w;
            sum += w;
        }
        if (sum != 0.0)
            for (double & w : f.w) w /= sum;
        for (double w : f.w) wt_max = std::max(wt_max, std::fabs(w));
    }
    // aten grows the shift while the scaled max weight fits int16; the AVX
    // uint8 kernels (pillow-simd heritage) run a fixed high precision - sweep
    // on-silicon matched p=17 (taps*255 must still fit the int accumulators:
    // 2^17 * 255 * taps << 2^31)
    int p = 0;
    for (; p < 22; p++) {
        const int next_value = (int)(0.5 + wt_max * (double)(1 << (p + 1)));
        if (next_value >= (1 << 15)) break;
    }
    *precision = p;
}

void resize_tv_bicubic_aa(const uint8_t * src, int sw, int sh, int dw, int dh,
                          std::vector<float>& out) {
    // torch's fast image processors feed a channels-last uint8 tensor, which
    // dispatches to the pillow-simd-style AVX kernel: int16 fixed-point
    // weights, integer accumulation, uint8 rounding after EACH separable
    // pass. p=14 matches on-silicon (absmax 1/255, 99.6% of pixels exact).
    auto build = [](int in, int out_n, std::vector<Filter>& fs) {
        const double scale = (double)in / (double)out_n;
        const double support = scale >= 1.0 ? 2.0 * scale : 2.0;
        const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
        const int max_interp = (int)std::ceil(support) * 2 + 1;
        fs.resize(out_n);
        for (int o = 0; o < out_n; o++) {
            const double center = scale * ((double)o + 0.5);
            int xmin = (int)(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            int xend = (int)(center + support + 0.5);
            if (xend > in) xend = in;
            int xsize = xend - xmin;
            if (xsize > max_interp) xsize = max_interp;
            if (xsize < 0) xsize = 0;
            Filter & f = fs[o];
            f.start = xmin;
            f.w.resize(xsize);
            double sum = 0.0;
            for (int j = 0; j < xsize; j++) {
                const double w = bicubic_aa((j + xmin - center + 0.5) * invscale);
                f.w[j] = w;
                sum += w;
            }
            if (sum != 0.0)
                for (double & w : f.w) w /= sum;
        }
    };
    std::vector<Filter> fh, fv;
    build(sw, dw, fh);
    build(sh, dh, fv);
    const int P = 14;
    auto qw = [P](const Filter & f) {
        std::vector<int> q(f.w.size());
        for (size_t k = 0; k < f.w.size(); k++) {
            const double v = f.w[k] * (double)(1 << P);
            q[k] = f.w[k] >= 0 ? (int)(0.5 + v) : (int)(-0.5 + v);
        }
        return q;
    };
    // horizontal pass: uint8 -> uint8
    std::vector<uint8_t> tmp((size_t)dw * sh * 3);
    for (int y = 0; y < sh; y++) {
        const uint8_t * row = src + (size_t)y * sw * 3;
        for (int x = 0; x < dw; x++) {
            const std::vector<int> w = qw(fh[x]);
            int acc[3] = {1 << (P - 1), 1 << (P - 1), 1 << (P - 1)};
            for (size_t k = 0; k < w.size(); k++) {
                const uint8_t * px = row + (size_t)(fh[x].start + (int)k) * 3;
                for (int c = 0; c < 3; c++) acc[c] += (int)px[c] * w[k];
            }
            uint8_t * dst = &tmp[((size_t)y * dw + x) * 3];
            for (int c = 0; c < 3; c++)
                dst[c] = (uint8_t)std::clamp(acc[c] >> P, 0, 255);
        }
    }
    // vertical pass: uint8 -> uint8
    out.assign((size_t)dw * dh * 3, 0.0f);
    for (int y = 0; y < dh; y++) {
        const std::vector<int> w = qw(fv[y]);
        for (int x = 0; x < dw; x++) {
            int acc[3] = {1 << (P - 1), 1 << (P - 1), 1 << (P - 1)};
            for (size_t k = 0; k < w.size(); k++) {
                const uint8_t * px = &tmp[((size_t)(fv[y].start + (int)k) * dw + x) * 3];
                for (int c = 0; c < 3; c++) acc[c] += (int)px[c] * w[k];
            }
            float * dst = &out[((size_t)y * dw + x) * 3];
            for (int c = 0; c < 3; c++)
                dst[c] = (float)std::clamp(acc[c] >> P, 0, 255);
        }
    }
}

// ---------------------------------------------------------------- PIL Lanczos
// PIL Resample.c replication for 8-bit channels (was
// models/moondream/moondream_preprocess.cpp's lanczos_resize - bit-exact
// target Pillow 10.4).

static const double MD_PI = 3.14159265358979323846;

static inline double md_sinc(double x) {
    if (x == 0.0) return 1.0;
    x *= MD_PI;
    return std::sin(x) / x;
}

static inline double md_lanczos_filter(double x) {  // support = 3.0
    if (x < 0.0) x = -x;
    if (x < 3.0) return md_sinc(x) * md_sinc(x / 3.0);
    return 0.0;
}

// PIL Resample.c cubic_filter (cubic convolution a = -0.5, support = 2.0)
static inline double md_cubic_filter(double x) {
    if (x < 0.0) x = -x;
    if (x < 1.0) return 1.5 * x * x * x - 2.5 * x * x + 1.0;
    if (x < 2.0) return -0.5 * x * x * x + 2.5 * x * x - 4.0 * x + 2.0;
    return 0.0;
}

// PIL Resample.c triangle_filter (BILINEAR, support = 1.0)
static inline double md_triangle_filter(double x) {
    if (x < 0.0) x = -x;
    return x < 1.0 ? 1.0 - x : 0.0;
}

static const int MD_PRECISION_BITS = 32 - 8 - 2;  // 22

struct MdCoeffs {
    int ksize = 0;
    std::vector<int> bounds;        // [out*2]: xmin, count
    std::vector<int32_t> kk;        // fixpoint 1<<22, PIL rounding
};

// box = (0, in) full axis; PIL in0/in1 defaults.
static void md_precompute_coeffs(int in_size, int out_size,
                                 double support_filter, double (*filter)(double),
                                 MdCoeffs & c) {
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
            w[x] = filter((x + xmin - center + 0.5) * inv_fs);
            ww += w[x];
        }
        int32_t * k = &c.kk[(size_t)xx * ksize];
        if (ww != 0.0) {
            for (int x = 0; x < count; x++) {
                double v = w[x] / ww;
                // PIL normalize_coeffs_8bpc rounding
                k[x] = (int32_t)(v < 0 ? (-0.5 + v * (1 << MD_PRECISION_BITS))
                                       : ( 0.5 + v * (1 << MD_PRECISION_BITS)));
            }
        }
        c.bounds[xx * 2 + 0] = xmin;
        c.bounds[xx * 2 + 1] = count;
    }
}

static inline uint8_t md_clip8(int64_t v) {  // PIL clip8 lookup equivalent
    v >>= MD_PRECISION_BITS;
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

// the shared Resample.c 8bpc pipeline: horizontal pass -> uint8 temp, then
// vertical pass (Pillow 10.4 runs the horizontal pass first; the vertical-first
// heuristic is a newer-Pillow addition)
static void pil_coeffs_resize(const uint8_t * src, int sw, int sh, int sc,
                              int dw, int dh, uint8_t * dst,
                              const MdCoeffs & ch, const MdCoeffs & cv,
                              bool need_h, bool need_v) {
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
                        int64_t ss = 1 << (MD_PRECISION_BITS - 1);
                        for (int y = 0; y < cnt; y++)
                            ss += (int64_t)src[(size_t)(y + ymin) * sw * sc +
                                               (size_t)xx * sc + ci] * k[y];
                        out[(size_t)xx * sc + ci] = md_clip8(ss);
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
                    int64_t ss = 1 << (MD_PRECISION_BITS - 1);
                    for (int x = 0; x < cnt; x++)
                        ss += (int64_t)in[(size_t)(x + xmin) * sc + ci] * k[x];
                    out[(size_t)xx * sc + ci] = md_clip8(ss);
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
                    int64_t ss = 1 << (MD_PRECISION_BITS - 1);
                    for (int y = 0; y < cnt; y++)
                        ss += (int64_t)tmp[(size_t)(y + ymin) * dw * sc +
                                           (size_t)xx * sc + ci] * k[y];
                    out[(size_t)xx * sc + ci] = md_clip8(ss);
                }
            }
        }
    }
}

void resize_lanczos_coeffs(const uint8_t * src, int sw, int sh, int sc,
                           int dw, int dh, uint8_t * dst) {
    if (sw == dw && sh == dh) {
        std::memcpy(dst, src, (size_t)sw * sh * sc);
        return;
    }
    bool need_h = dw != sw, need_v = dh != sh;
    MdCoeffs ch, cv;
    if (need_v) md_precompute_coeffs(sh, dh, 3.0, md_lanczos_filter, cv);
    if (need_h) md_precompute_coeffs(sw, dw, 3.0, md_lanczos_filter, ch);
    pil_coeffs_resize(src, sw, sh, sc, dw, dh, dst, ch, cv, need_h, need_v);
}

void resize_bicubic_coeffs(const uint8_t * src, int sw, int sh, int sc,
                           int dw, int dh, uint8_t * dst) {
    if (sw == dw && sh == dh) {
        std::memcpy(dst, src, (size_t)sw * sh * sc);
        return;
    }
    bool need_h = dw != sw, need_v = dh != sh;
    MdCoeffs ch, cv;
    if (need_v) md_precompute_coeffs(sh, dh, 2.0, md_cubic_filter, cv);
    if (need_h) md_precompute_coeffs(sw, dw, 2.0, md_cubic_filter, ch);
    pil_coeffs_resize(src, sw, sh, sc, dw, dh, dst, ch, cv, need_h, need_v);
}

void resize_bilinear_coeffs(const uint8_t * src, int sw, int sh, int sc,
                            int dw, int dh, uint8_t * dst) {
    if (sw == dw && sh == dh) {
        std::memcpy(dst, src, (size_t)sw * sh * sc);
        return;
    }
    bool need_h = dw != sw, need_v = dh != sh;
    MdCoeffs ch, cv;
    if (need_v) md_precompute_coeffs(sh, dh, 1.0, md_triangle_filter, cv);
    if (need_h) md_precompute_coeffs(sw, dw, 1.0, md_triangle_filter, ch);
    pil_coeffs_resize(src, sw, sh, sc, dw, dh, dst, ch, cv, need_h, need_v);
}

// ---------------------------------------------------------------- torch bicubic
// bicubic interpolation of (sh,sw,C) -> (dh,dw,C), torch F.interpolate
// semantics (was models/videochat3/vc3_preprocess.cpp's
// interpolate_bicubic_torch). Keys kernel a = -0.75, align_corners = false.
static inline double vc3_cubic_torch(double x, double A) {
    x = std::fabs(x);
    if (x <= 1.0) return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    if (x < 2.0) return ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    return 0.0;
}

void resize_bicubic_torch(const float * src, int sh, int sw, int dh, int dw, int C,
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
                t.w[k] = vc3_cubic_torch((double)idx - real, A);
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

// ---------------------------------------------------------------- pos table
void interp_pos_grid(const float * table, int side, int dim, int gh, int gw,
                     std::vector<float>& out) {
    out.assign((size_t)gh * gw * dim, 0.0f);
    std::vector<float> ys(gh), xs(gw);
    for (int i = 0; i < gh; i++) ys[i] = gh > 1 ? (float)i * (side - 1) / (gh - 1) : 0.0f;
    for (int j = 0; j < gw; j++) xs[j] = gw > 1 ? (float)j * (side - 1) / (gw - 1) : 0.0f;
    const int smax = side - 1;
    for (int i = 0; i < gh; i++) {
        const int y0 = (int)ys[i];
        const int y1 = y0 + 1 <= smax ? y0 + 1 : smax;
        const float dy = ys[i] - y0;
        for (int j = 0; j < gw; j++) {
            const int x0 = (int)xs[j];
            const int x1 = x0 + 1 <= smax ? x0 + 1 : smax;
            const float dx = xs[j] - x0;
            const float * r00 = table + (size_t)(y0 * side + x0) * dim;
            const float * r01 = table + (size_t)(y0 * side + x1) * dim;
            const float * r10 = table + (size_t)(y1 * side + x0) * dim;
            const float * r11 = table + (size_t)(y1 * side + x1) * dim;
            float * o = out.data() + ((size_t)i * gw + j) * dim;
            for (int d = 0; d < dim; d++) {
                const float a = r00[d] * (1 - dx) + r01[d] * dx;
                const float b = r10[d] * (1 - dx) + r11[d] * dx;
                o[d] = a * (1 - dy) + b * dy;
            }
        }
    }
}

}  // namespace img
}  // namespace vqa
