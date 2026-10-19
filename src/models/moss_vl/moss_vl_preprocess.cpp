// moss_vl_preprocess.cpp - see moss_vl_preprocess.hpp.
#include "moss_vl_preprocess.hpp"

#include "core/media/resample.hpp"

#include <cmath>

namespace moss_vl {

void patchify_frame(const float * rgb, int h, int w, const VisConfig & vc,
                    const float * pos_table, std::vector<float> & patches,
                    std::vector<float> & pos, std::vector<int> & coords,
                    int * grid_h, int * grid_w) {
    const int p = vc.patch, m = vc.merge, dim = vc.n_embd, side = vc.pos_side;
    const int gh = h / p, gw = w / p;
    *grid_h = gh;
    *grid_w = gw;
    const int C3 = 3 * p * p;

    // normalized pixels: x/127.5 - 1
    auto px = [&](int y, int x, int c) {
        return rgb[((size_t)y * w + x) * 3 + c] / 127.5f - 1.0f;
    };

    // bilinear interp of the learned (side,side,dim) table to (gh,gw):
    // torch.linspace(0, side-1, g) endpoints, floor/ceil clipped
    // (core/resample.hpp, shared with Qwen3.5-VL)
    std::vector<float> pos_rows;
    vqa::img::interp_pos_grid(pos_table, side, dim, gh, gw, pos_rows);

    // tokens in block-major order (mh, mw, ir, ic); patch vector (c, y, x)
    patches.reserve(patches.size() + (size_t)gh * gw * C3);
    pos.reserve(pos.size() + (size_t)gh * gw * dim);
    coords.reserve(coords.size() + (size_t)gh * gw * 2);
    std::vector<float> vec(C3);
    for (int bh = 0; bh < gh / m; bh++)
        for (int bw = 0; bw < gw / m; bw++)
            for (int ih = 0; ih < m; ih++)
                for (int iw = 0; iw < m; iw++) {
                    const int gy = bh * m + ih, gx = bw * m + iw;
                    int v = 0;
                    for (int c = 0; c < 3; c++)
                        for (int y = 0; y < p; y++)
                            for (int x = 0; x < p; x++)
                                vec[v++] = px(gy * p + y, gx * p + x, c);
                    patches.insert(patches.end(), vec.begin(), vec.end());
                    const float * o = pos_rows.data() + ((size_t)gy * gw + gx) * dim;
                    pos.insert(pos.end(), o, o + dim);
                    coords.push_back(gy);
                    coords.push_back(gx);
                }
}

}  // namespace moss_vl
