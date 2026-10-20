// minicpm_preprocess.hpp - MiniCPM-V 4.7 (openbmb/MiniCPM-V-4.7-*) host-side prep.
//
// The 4.7 family ("1B" dense, "35B-A3B" MoE) is NOT a new architecture on the
// text side: its `text_config.model_type` is literally `qwen3_5_text` (24 layers,
// 3 gated-delta linear layers + 1 full-attention layer per group of 4, IMROPE
// 11/11/10 over 64 of 256 head dims, interleaved q|gate projection). The text
// trunk therefore runs on the existing qwen3_5 backend unchanged - this backend
// only adds the parts MiniCPM-V does differently:
//
//   * SigLIP vision tower (27 layers, d=1152, 16 heads, patch 14, GELU-tanh,
//     NO rope: a learned 70x70 position table addressed by *nearest* bucket ids,
//     so the whole tower is the shared vqa::vit::block with env.no_rope=true)
//   * a window-attention merger inserted after vision layer 6 ("16x" mode): it
//     pools each 2x2 patch window into one token by (a) attention restricted to
//     the window, then (b) LN->Linear->GELU->Linear over the 4*1152 flattened
//     window, plus the window mean as residual
//   * a projection MLP merging 2x2 again (4608 -> 4608 -> d_text) = 16x total
//   * "4x" mode: the window merger is skipped, only the projection merges
//   * slice_mode: a high-resolution image is sent as a thumbnail plus up to
//     max_slice_nums detail crops (official MiniCPMV4_6ImageProcessor layout)
//   * "canvas M-RoPE": image tokens do not get a 1-D run number, they are laid
//     out on a 2-D canvas; <image>/</image>/<slice>/</slice>/\n markers are
//     pinned to canvas corners and the text counter resumes past the canvas
//
// Everything here is host-side (no ggml): pixel prep, position-table ids, the
// canvas position ids and the prompt/marker layout. Mirrors transformers
// models/minicpmv4_7/modeling_minicpmv4_7.py + models/minicpmv4_6/
// image_processing_minicpmv4_6.py + processing_minicpmv4_7.py.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/media/resample.hpp"   // PIL-exact bicubic (resize_bicubic_coeffs)

namespace minicpm {

struct VisConfig {
    int patch = 14;
    int n_embd = 1152;
    int n_head = 16;
    int n_layer = 27;
    int n_ff = 4304;
    int pos_side = 70;        // sqrt(num_position_embeddings) of the learned 2-D table
    int insert_layer = 6;     // window merger goes AFTER this encoder layer
    int window = 2;           // vit merger window (2x2)
    int merge = 2;            // MLP merger kernel (2x2) -> 16x total with the window
    float ln_eps = 1e-6f;
    int scale_resolution = 448;   // thumbnail / per-slice long-side target
    int max_slice_nums = 9;       // detail crops per image (slice_mode)
    // visual token downsampling: 16 = vit window merger + 2x2 projection (default),
    // 4 = projection only (the config's "4x" mode, 4x more visual tokens)
    int downsample = 16;
};

// The shipped preprocessor_config.json overrides the class default (ImageNet) with
// image_mean = image_std = 0.5, i.e. the plain x/127.5 - 1 used by the Qwen-VL
// family. Verified against the oracle: a 235 gray lands on 0.843, not 1.906.
static const float kMean[3] = { 0.5f, 0.5f, 0.5f };
static const float kStd[3]  = { 0.5f, 0.5f, 0.5f };

inline int ensure_divide(int x, int step) {
    return std::max((int)std::lround((double)x / step) * step, step);
}

// find_best_resize(): shrink to ~scale_resolution on the long side (aspect kept),
// then round both sides to a multiple of patch*merge*window (= 56), which is what
// makes the two 2x2 merges exact. `allow_upscale` mirrors the processor: a
// thumbnail without slices, and the per-slice refine size, are scaled up too.
inline void best_size(int h, int w, const VisConfig & vc, int * out_h, int * out_w,
                      bool allow_upscale = false) {
    const int step = vc.patch * vc.merge * vc.window;    // 56
    const double sr = (double)vc.scale_resolution;
    int th = h, tw = w;
    if ((double)h * (double)w > sr * sr || allow_upscale) {
        const double ar = (double)w / (double)h;
        th = (int)(sr / sqrt(ar));
        tw = (int)(th * ar);
    }
    *out_h = ensure_divide(th, step);
    *out_w = ensure_divide(tw, step);
}

// get_sliced_grid(): rows x cols of detail crops, 0 x 0 when the image already
// fits in one thumbnail. Picks the factorisation closest to the image aspect.
inline void slice_grid(int h, int w, const VisConfig & vc, int * rows, int * cols) {
    *rows = *cols = 0;
    const double log_ratio = log((double)w / (double)h);
    const double ratio = (double)h * (double)w /
                         ((double)vc.scale_resolution * (double)vc.scale_resolution);
    const int multiple = std::min((int)ceil(ratio), vc.max_slice_nums);
    if (multiple <= 1) return;
    int best_rows = 1, best_cols = 1;
    double min_error = 1e30;
    for (int d = -1; d <= 1; d++) {
        const int num_slices = multiple + d;
        if (num_slices == 1 || num_slices > vc.max_slice_nums) continue;
        for (int nr = 1; nr <= num_slices; nr++) {
            if (num_slices % nr) continue;
            const int nc = num_slices / nr;
            const double err = fabs(log_ratio - log((double)nc / (double)nr));
            if (err < min_error || (err == min_error && nr > best_rows)) {
                best_rows = nr; best_cols = nc; min_error = err;
            }
        }
    }
    *rows = best_rows;
    *cols = best_cols;
}

// get_vision_nearest_position_ids(): bucket i/h into a 1/pos_side grid with
// torch.bucketize(boundaries=1/pos_side.., right=True) semantics. right=True means
// boundaries[i-1] <= x < boundaries[i], so the bucket is the number of boundaries
// that are <= x, i.e. floor(pos_side*i/h) -- verified against the oracle: at native
// resolution (h == pos_side) the mapping must be the identity, which ceil breaks.
// Flat table row = bucket_h * pos_side + bucket_w.
inline void nearest_pos_ids(int gh, int gw, int pos_side, std::vector<int> & ids) {
    ids.resize((size_t)gh * gw);
    const int H = pos_side;
    auto bucket = [&](int i, int n) { return (H * i) / n; };   // floor
    for (int r = 0; r < gh; r++) {
        const int bh = bucket(r, gh);
        for (int c = 0; c < gw; c++) ids[(size_t)r * gw + c] = bh * H + bucket(c, gw);
    }
}

// NaViT patch extraction: every patch becomes a column of [3*patch*patch, N] with
// values ordered (channel, ky, kx) - the same flattening a [oc, c, ky, kx] conv
// weight gets in the converter, so patch_embedding is one ggml_mul_mat.
// `rgb` is interleaved RGB8, w x h, tightly packed. Pixels are rescaled and
// normalized here (mean/std 0.5).
inline void patchify(const uint8_t * rgb, int w, int h, const VisConfig & vc,
                     std::vector<float> & patches, int * gh, int * gw) {
    const int p = vc.patch;
    *gh = h / p;
    *gw = w / p;
    const int N = (*gh) * (*gw), C3 = 3 * p * p;
    patches.assign((size_t)N * C3, 0.0f);
    for (int r = 0; r < *gh; r++) {
        for (int c = 0; c < *gw; c++) {
            float * col = patches.data() + (size_t)(r * (*gw) + c) * C3;
            for (int ch = 0; ch < 3; ch++) {
                for (int ky = 0; ky < p; ky++) {
                    const int y = r * p + ky;
                    for (int kx = 0; kx < p; kx++) {
                        const int x = c * p + kx;
                        const float v = rgb[(size_t)(y * w + x) * 3 + ch] / 255.0f;
                        col[((size_t)ch * p + ky) * p + kx] = (v - kMean[ch]) / kStd[ch];
                    }
                }
            }
        }
    }
}

// One vision-encoder input: the thumbnail or one detail crop.
struct Crop {
    int gh = 0, gw = 0;              // patch grid
    std::vector<float> patches;      // [3*p*p, gh*gw]
};

// One image (or one video frame): thumbnail + slices. `crops[0]` is the
// thumbnail, the rest are the detail crops in row-major order.
struct ImagePlan {
    std::vector<Crop> crops;
    int rows = 0, cols = 0;          // slice grid (0 = no slices)
    int tokens() const { return (int)crops.size(); }
};

// Fill `plan` from an RGB8 image (`rgb` is w*h*3, tightly packed). `allow_slices`
// is false for video frames (the video processor path keeps one crop per frame).
// torchvision-bit-exact resize into an RGB8 buffer (the kernel rounds to uint8 per
// pass internally; it hands back floats in [0,255] so we round once more here).
inline void resize_rgb8(const uint8_t * src, int w, int h, int dw, int dh, uint8_t * dst) {
    std::vector<float> tmp;
    vqa::img::resize_tv_bicubic_aa(src, w, h, dw, dh, tmp);
    for (size_t i = 0; i < tmp.size(); i++) {
        const float v = tmp[i] + 0.5f;
        dst[i] = (uint8_t)(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
    }
}

inline bool prepare_image(const uint8_t * src, int w, int h, const VisConfig & vc,
                          bool allow_slices, ImagePlan & plan) {
    plan = ImagePlan{};
    int rows = 0, cols = 0;
    if (allow_slices) slice_grid(h, w, vc, &rows, &cols);
    plan.rows = rows;
    plan.cols = cols;

    // ---- thumbnail: upscaled only when there are no detail crops
    int th = 0, tw = 0;
    best_size(h, w, vc, &th, &tw, rows == 0);
    {
        Crop c;
        std::vector<uint8_t> u8((size_t)tw * th * 3);
        if (th == h && tw == w) memcpy(u8.data(), src, u8.size());
        // HF's fast image processors resize through torchvision (uint8 in/out,
        // antialias bicubic): resize_tv_bicubic_aa is bit-exact vs that path,
        // while the PIL variant here would differ by 1-2 gray levels per channel
        else resize_rgb8(src, w, h, tw, th, u8.data());
        patchify(u8.data(), tw, th, vc, c.patches, &c.gh, &c.gw);
        plan.crops.push_back(std::move(c));
    }

    if (rows == 0 || cols == 0) return true;

    // ---- detail crops: resize the whole image to grid * cell, then cut cells.
    // cell = find_best_resize(ceil(w/cols), ceil(h/rows), allow_upscale=True)
    const int rw = (w + cols - 1) / cols, rh = (h + rows - 1) / rows;
    int cell_h = 0, cell_w = 0;
    best_size(rh, rw, vc, &cell_h, &cell_w, true);
    const int refine_h = cell_h * rows, refine_w = cell_w * cols;
    std::vector<uint8_t> refine((size_t)refine_w * refine_h * 3);
    resize_rgb8(src, w, h, refine_w, refine_h, refine.data());
    std::vector<uint8_t> cell((size_t)cell_w * cell_h * 3);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const uint8_t * base = refine.data() +
                                   ((size_t)r * cell_h * refine_w + (size_t)c * cell_w) * 3;
            for (int y = 0; y < cell_h; y++)
                memcpy(cell.data() + (size_t)y * cell_w * 3, base + (size_t)y * refine_w * 3,
                       (size_t)cell_w * 3);
            Crop crop;
            patchify(cell.data(), cell_w, cell_h, vc, crop.patches, &crop.gh, &crop.gw);
            plan.crops.push_back(std::move(crop));
        }
    }
    return true;
}

// Window gather order for the vit merger: windows tile the patch grid row-major and
// hold their tokens row-major, i.e. exactly get_vision_window_index() with
// spatial_merge_size=1 / window_size=2 / patch_size=1 (grids are multiples of 4 so
// the reference's -100 padding never fires).
inline void window_index(int gh, int gw, int win, std::vector<int> & idx) {
    const int mh = gh / win, mw = gw / win;
    idx.resize((size_t)gh * gw);
    size_t o = 0;
    for (int wr = 0; wr < mh; wr++)
        for (int wc = 0; wc < mw; wc++)
            for (int ky = 0; ky < win; ky++)
                for (int kx = 0; kx < win; kx++)
                    idx[o++] = (wr * win + ky) * gw + (wc * win + kx);
}

// Gather order for the MLP merger's 2x2 regroup on an ALREADY downsampled grid
// (gh x gw = window count): column j of the merged tensor must read
// (ky, kx, dim) with dim fastest, so dim0 = 4*d when four [d, n] gathers are
// concatenated along dim0 in (ky, kx) order.
inline void merge_index(int gh, int gw, int merge, std::vector<int> & idx) {
    const int mh = gh / merge, mw = gw / merge;
    idx.resize((size_t)mh * mw * merge * merge);
    size_t o = 0;
    for (int ky = 0; ky < merge; ky++)
        for (int kx = 0; kx < merge; kx++)
            for (int r = 0; r < mh; r++)
                for (int c = 0; c < mw; c++)
                    idx[o++] = (r * merge + ky) * gw + (c * merge + kx);
}

// ---------------------------------------------------------------------------
// Canvas M-RoPE geometry, per visual block (one image, or one video frame)
// ---------------------------------------------------------------------------
struct BlockGeom {
    int thumb_h = 0, thumb_w = 0;    // LLM token grid of the thumbnail
    int slice_h = 0, slice_w = 0;    // LLM token grid of one detail crop
    int rows = 0, cols = 0;          // detail crop grid (0 = none)
};

// round(linspace(0, span-1, n)) with torch's round-half-to-even
inline std::vector<int> linspace_round(int n, int span) {
    std::vector<int> v((size_t)n, 0);
    if (n <= 0) return v;
    if (n == 1) { v[0] = 0; return v; }
    for (int i = 0; i < n; i++)
        v[i] = (int)std::nearbyint((double)i * (double)(span - 1) / (double)(n - 1));
    return v;
}

// Canvas M-RoPE position ids for a whole prompt ([4][T]: t, h, w, kv slot).
//
// Per visual block at text counter `origin` (single image = one block; video =
// one block per frame, laid out back to back):
//   <image>   t=origin  h=origin-1            w=origin-1
//   thumb     t=origin  h=origin+lin(0,canvas_h-1) w=origin+lin(0,canvas_w-1)
//   </image>  t=origin  h=origin+canvas_h     w=origin+canvas_w
//   <slice>k  t=origin  h=origin+h_off        w=origin+w_off
//   slice pad t=origin  h=origin+h_off+row    w=origin+w_off+col
//   </slice>k t=origin  h=origin+h_off+sh-1   w=origin+w_off+sw-1
//   "\n"      t=origin  h=origin+row_end-1    w=origin+canvas_w  (one per row break)
// then the text counter resumes at origin + max(canvas_h, canvas_w) + 1.
inline bool canvas_mpos(const std::vector<int> & ids, const std::vector<BlockGeom> & geoms,
                        int id_start, int id_end, int id_slice_start, int id_slice_end,
                        int id_pad_a, int id_pad_b, std::vector<int> & mpos) {
    const int T = (int)ids.size();
    std::vector<int> ta((size_t)T, 0), ha((size_t)T, 0), wa((size_t)T, 0);
    auto is_pad = [&](int id) { return id == id_pad_a || (id_pad_b && id == id_pad_b); };
    int cur = 0;
    size_t gi = 0;
    for (int i = 0; i < T;) {
        if (ids[i] != id_start) {
            ta[i] = ha[i] = wa[i] = cur;
            cur++;
            i++;
            continue;
        }
        if (gi >= geoms.size()) {
            fprintf(stderr, "minicpm: more visual blocks in the prompt than planned (%zu)\n", gi);
            return false;
        }
        const BlockGeom g = geoms[gi++];
        const int origin = cur;
        // thumbnail
        int j = i + 1;
        while (j < T && is_pad(ids[j])) j++;
        const int n_thumb = j - i - 1;
        if (n_thumb != g.thumb_h * g.thumb_w) {
            fprintf(stderr, "minicpm: thumbnail has %d pads, grid %dx%d\n",
                    n_thumb, g.thumb_h, g.thumb_w);
            return false;
        }
        const int canvas_h = g.rows ? g.rows * g.slice_h : g.thumb_h;
        const int canvas_w = g.cols ? g.cols * g.slice_w : g.thumb_w;
        ta[i] = origin; ha[i] = origin - 1; wa[i] = origin - 1;
        {
            const std::vector<int> hs = linspace_round(g.thumb_h, canvas_h);
            const std::vector<int> ws = linspace_round(g.thumb_w, canvas_w);
            for (int r = 0; r < g.thumb_h; r++)
                for (int c = 0; c < g.thumb_w; c++) {
                    const int k = i + 1 + r * g.thumb_w + c;
                    ta[k] = origin; ha[k] = origin + hs[r]; wa[k] = origin + ws[c];
                }
        }
        if (j < T && ids[j] == id_end) {
            ta[j] = origin; ha[j] = origin + canvas_h; wa[j] = origin + canvas_w;
            j++;
        } else {
            fprintf(stderr, "minicpm: missing </image> marker after the thumbnail\n");
            return false;
        }
        // detail crops, row-major, "\n" between rows
        for (int r = 0; r < g.rows; r++) {
            for (int c = 0; c < g.cols; c++) {
                if (j >= T || ids[j] != id_slice_start) {
                    fprintf(stderr, "minicpm: missing <slice> marker at crop %d/%d\n",
                            r * g.cols + c, g.rows * g.cols);
                    return false;
                }
                const int h_off = r * g.slice_h, w_off = c * g.slice_w;
                ta[j] = origin; ha[j] = origin + h_off; wa[j] = origin + w_off;
                j++;
                int k = j;
                while (k < T && is_pad(ids[k])) k++;
                if (k - j != g.slice_h * g.slice_w) {
                    fprintf(stderr, "minicpm: slice has %d pads, grid %dx%d\n",
                            k - j, g.slice_h, g.slice_w);
                    return false;
                }
                for (int rr = 0; rr < g.slice_h; rr++)
                    for (int cc = 0; cc < g.slice_w; cc++) {
                        const int t = j + rr * g.slice_w + cc;
                        ta[t] = origin;
                        ha[t] = origin + h_off + rr;
                        wa[t] = origin + w_off + cc;
                    }
                j = k;
                if (j < T && ids[j] == id_slice_end) {
                    ta[j] = origin;
                    ha[j] = origin + h_off + g.slice_h - 1;
                    wa[j] = origin + w_off + g.slice_w - 1;
                    j++;
                } else {
                    fprintf(stderr, "minicpm: missing </slice> marker\n");
                    return false;
                }
            }
            if (r + 1 < g.rows) {
                // the "\n" that ends a row of slices sits past the right edge
                if (j >= T || ids[j] == id_slice_start) {
                    fprintf(stderr, "minicpm: missing newline between slice rows\n");
                    return false;
                }
                ta[j] = origin;
                ha[j] = origin + (r + 1) * g.slice_h - 1;
                wa[j] = origin + g.cols * g.slice_w;
                j++;
            }
        }
        cur = origin + (canvas_h > canvas_w ? canvas_h : canvas_w) + 1;
        i = j;
        // The reference's frame span reaches exactly one token past the last visual
        // marker (the template's "\n" between the block and the question): that
        // token keeps the canvas origin on all three axes and does NOT advance the
        // text counter (verified against the oracle: index 465 of a sliced prompt).
        if (i < T && ids[i] != id_start) {
            ta[i] = origin; ha[i] = origin; wa[i] = origin;
            i++;
        }
    }
    mpos.clear();
    mpos.reserve((size_t)4 * T);
    mpos.insert(mpos.end(), ta.begin(), ta.end());
    mpos.insert(mpos.end(), ha.begin(), ha.end());
    mpos.insert(mpos.end(), wa.begin(), wa.end());
    for (int i = 0; i < T; i++) mpos.push_back(i);     // absolute KV slots
    return true;
}

// One visual block as the processor renders it: <image>thumbnail</image> followed
// by rows of <slice>crops</slice> ("\n" between rows, none inside a row).
inline void append_image_block(std::string & s, const int * crop_tokens, int n_crops,
                               int rows, int cols, const std::string & pad,
                               const std::string & start, const std::string & end,
                               const std::string & slice_start, const std::string & slice_end) {
    s += start;
    for (int i = 0; i < crop_tokens[0]; i++) s += pad;
    s += end;
    if (!rows || !cols) return;
    for (int r = 0; r < rows; r++) {
        if (r) s += "\n";
        for (int c = 0; c < cols; c++) {
            const int k = 1 + r * cols + c;
            s += slice_start;
            for (int i = 0; i < (k < n_crops ? crop_tokens[k] : 0); i++) s += pad;
            s += slice_end;
        }
    }
}

// ChatML prelude/epilogue of chat_template.jinja (non-thinking variant) plus the
// template's "\n" between the visual block and the question.
inline std::string wrap_prompt(const std::string & visual, const std::string & user_text) {
    std::string s = "<|im_start|>user\n";
    s += visual;
    if (!visual.empty()) s += "\n";
    s += user_text;
    s += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return s;
}

}  // namespace minicpm
