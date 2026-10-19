// qwen3_5_preprocess.cpp - see qwen3_5_preprocess.hpp.
#include "qwen3_5_preprocess.hpp"

#include "core/media/resample.hpp"

#include <cmath>

namespace qwen3_5 {

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

    // interpolated pos rows for the (gh, gw) grid: core/resample.hpp (shared with MOSS-VL)
    std::vector<float> pos_rows;
    vqa::img::interp_pos_grid(pos_table, side, dim, gh, gw, pos_rows);

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

}  // namespace qwen3_5
