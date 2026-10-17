// resample.cpp - see resample.hpp.
#include "resample.hpp"

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

struct Filter {
    int start = 0;              // first source index
    std::vector<double> w;      // weights
};

// builds the antialias filter bank for one axis (PIL semantics: the support is
// widened by the downscale factor, which is what makes it antialiased)
static void build_aa_filters(int in, int out, std::vector<Filter>& fs, bool bilinear = false) {
    const double scale = (double)in / (double)out;
    const double filterscale = std::max(1.0, scale);
    const double support = (bilinear ? 1.0 : 2.0) * filterscale;
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
            double w = bilinear ? bilinear_aa(t) : bicubic_aa(t);
            f.w[x - xmin] = w;
            sum += w;
        }
        if (sum != 0.0) for (double & w : f.w) w /= sum;
    }
}

void resize_antialias(const uint8_t * src, int sw, int sh, int dw, int dh,
                      std::vector<float>& out, bool bilinear) {
    std::vector<Filter> fh, fv;
    build_aa_filters(sw, dw, fh, bilinear);
    build_aa_filters(sh, dh, fv, bilinear);

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
            dst[0] = (float)acc[0];
            dst[1] = (float)acc[1];
            dst[2] = (float)acc[2];
        }
    }
}

void resize_bicubic_antialias(const uint8_t * src, int sw, int sh, int dw, int dh,
                              std::vector<float>& out) {
    resize_antialias(src, sw, sh, dw, dh, out, /*bilinear*/ false);
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
