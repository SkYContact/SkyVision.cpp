// marlin_preprocess.cpp - see marlin_preprocess.hpp.
#include "marlin_preprocess.hpp"

#include <cmath>

namespace marlin {

void patchify3d(const float * const * tframes, int h, int w, const VisConfig & vc,
                const float * pos_table, std::vector<float> & patches,
                std::vector<float> & pos, std::vector<int> & coords,
                int * grid_h, int * grid_w) {
    const int p = vc.patch, tp = vc.temporal_patch, C = 3, m = vc.merge;
    const int gh = h / p, gw = w / p;
    *grid_h = gh;
    *grid_w = gw;
    const int dim = vc.n_embd, side = vc.pos_side;
    const int C3 = C * tp * p * p;

    // normalized frames: x/127.5 - 1
    auto px = [&](int t, int y, int x, int c) {
        return tframes[t][((size_t)y * w + x) * C + c] / 127.5f - 1.0f;
    };

    // interpolated pos rows cache for the (gh, gw) grid
    std::vector<float> pos_rows((size_t)gh * gw * dim);
    {
        std::vector<float> ys(gh), xs(gw);
        for (int i = 0; i < gh; i++) ys[i] = gh > 1 ? (float)i * (side - 1) / (gh - 1) : 0.0f;
        for (int j = 0; j < gw; j++) xs[j] = gw > 1 ? (float)j * (side - 1) / (gw - 1) : 0.0f;
        for (int i = 0; i < gh; i++) {
            const int y0 = (int)ys[i], x00max = side - 1;
            const int y1 = y0 + 1 <= x00max ? y0 + 1 : x00max;
            const float dy = ys[i] - y0;
            for (int j = 0; j < gw; j++) {
                const int x0 = (int)xs[j];
                const int x1 = x0 + 1 <= x00max ? x0 + 1 : x00max;
                const float dx = xs[j] - x0;
                const float * r00 = pos_table + (size_t)(y0 * side + x0) * dim;
                const float * r01 = pos_table + (size_t)(y0 * side + x1) * dim;
                const float * r10 = pos_table + (size_t)(y1 * side + x0) * dim;
                const float * r11 = pos_table + (size_t)(y1 * side + x1) * dim;
                float * o = pos_rows.data() + ((size_t)i * gw + j) * dim;
                for (int d = 0; d < dim; d++) {
                    const float a = r00[d] * (1 - dx) + r01[d] * dx;
                    const float b = r10[d] * (1 - dx) + r11[d] * dx;
                    o[d] = a * (1 - dy) + b * dy;
                }
            }
        }
    }

    patches.reserve(patches.size() + (size_t)gh * gw * C3);
    pos.reserve(pos.size() + (size_t)gh * gw * dim);
    coords.reserve(coords.size() + (size_t)gh * gw);
    std::vector<float> vec(C3);
    for (int bh = 0; bh < gh / m; bh++)
        for (int bw = 0; bw < gw / m; bw++)
            for (int ih = 0; ih < m; ih++)
                for (int iw = 0; iw < m; iw++) {
                    const int gy = bh * m + ih, gx = bw * m + iw;
                    // patch vector: (c, t, y, x)
                    int v = 0;
                    for (int c = 0; c < C; c++)
                        for (int t = 0; t < tp; t++)
                            for (int y = 0; y < p; y++)
                                for (int x = 0; x < p; x++)
                                    vec[v++] = px(t, gy * p + y, gx * p + x, c);
                    patches.insert(patches.end(), vec.begin(), vec.end());
                    const float * o = pos_rows.data() + ((size_t)gy * gw + gx) * dim;
                    pos.insert(pos.end(), o, o + dim);
                    coords.push_back(gy);
                    coords.push_back(gx);
                }
}

}  // namespace marlin
