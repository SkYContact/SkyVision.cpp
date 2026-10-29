// minicpmo_vision.cpp - MiniCPM-o 4.5 vision front end: MiniCPM-V LPU 9-slice
// image tiling (host, PIL-exact bicubic), the SigLIP-SO400M NaFiT tower
// (27L x 1152, bucketized 70x70 position ids, per-tile graph) and the 64-query
// perceiver resampler. Numerics contract in minicpmo.hpp.
#include "minicpmo.hpp"

#include "core/engine/ggml_dump.hpp"
#include "core/media/resample.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace minicpmo {

// ------------------------------------------------------------------ host

static int ensure_divide(int length, int n) {
    return std::max((int)llround((double)length / n) * n, n);
}

void best_resize(int width, int height, int scale_resolution, int patch_size,
                 bool allow_upscale, int * out_w, int * out_h) {
    if ((double)width * height > (double)scale_resolution * scale_resolution || allow_upscale) {
        const double r = (double)width / height;
        height = (int)(scale_resolution / std::sqrt(r));
        width = (int)(height * r);
    }
    *out_w = ensure_divide(width, patch_size);
    *out_h = ensure_divide(height, patch_size);
}

bool slice_grid(int width, int height, int max_slice_nums, int scale_resolution,
                int * out_cols, int * out_rows) {
    const double log_ratio = std::log((double)width / height);
    const double ratio = (double)width * height /
                         ((double)scale_resolution * scale_resolution);
    const int multiple = std::min((int)std::ceil(ratio), max_slice_nums);
    if (multiple <= 1) return false;
    bool have = false;
    int best_cols = 1, best_rows = 1;
    double min_err = 1e300;
    for (int cand : { multiple - 1, multiple, multiple + 1 }) {
        if (cand == 1 || cand > max_slice_nums) continue;
        for (int m = 1; m <= cand; m++) {
            if (cand % m != 0) continue;
            const int cols = m, rows = cand / m;
            const double err = std::fabs(log_ratio -
                                         std::log((double)cols / rows));
            if (!have || err < min_err) {   // strict: first candidate wins ties
                have = true;
                min_err = err;
                best_cols = cols;
                best_rows = rows;
            }
        }
    }
    if (!have) return false;
    *out_cols = best_cols;
    *out_rows = best_rows;
    return true;
}

void refine_size(int width, int height, int grid_cols, int grid_rows,
                 int scale_resolution, int patch_size, int * out_w, int * out_h) {
    const int rw = ensure_divide(width, grid_cols);
    const int rh = ensure_divide(height, grid_rows);
    int bw, bh;
    best_resize(rw / grid_cols, rh / grid_rows, scale_resolution, patch_size,
                /*allow_upscale*/ true, &bw, &bh);
    *out_w = bw * grid_cols;
    *out_h = bh * grid_rows;
}

void vision_pos_ids(int h_patches, int w_patches, int grid_side,
                    std::vector<int> & ids) {
    // fractional coord i/nb bucketized against boundaries k/grid_side.
    // torch.bucketize(..., right=True) is LEFT-closed (verified against the
    // official stack: x on a boundary lands in the LATER bucket), so
    // idx = floor(side*i/nb) - plain integer division.
    ids.resize((size_t)h_patches * w_patches);
    for (int i = 0; i < h_patches; i++) {
        const int bh = grid_side * i / h_patches;
        for (int j = 0; j < w_patches; j++) {
            const int bw = grid_side * j / w_patches;
            ids[(size_t)i * w_patches + j] = bh * grid_side + bw;
        }
    }
}

void sincos_pos_2d(int h, int w, int dim, std::vector<float> & out) {
    const int half = dim / 2;              // per-grid half (2048 of 4096)
    const int q = half / 2;                // sin/cos sub-half (1024)
    out.assign((size_t)h * w * dim, 0.0f);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float * row = out.data() + ((size_t)i * w + j) * dim;
            const float cx = (float)j, cy = (float)i;
            for (int k = 0; k < q; k++) {
                const double om = std::pow(10000.0, -(double)k / q);
                // first half encodes the COLUMN grid (meshgrid w-first),
                // second half the ROW grid; each = [sin | cos]
                row[k] = (float)std::sin((double)cx * om);
                row[q + k] = (float)std::cos((double)cx * om);
                row[half + k] = (float)std::sin((double)cy * om);
                row[half + q + k] = (float)std::cos((double)cy * om);
            }
        }
    }
}

void set_chunk_mask(ggml_tensor * dst, int chunk) {
    const int S = (int)dst->ne[0];
    std::vector<ggml_fp16_t> mv((size_t)S * S, ggml_fp32_to_fp16(-INFINITY));
    for (int q = 0; q < S; q++) {
        const int hi = std::min((q / chunk + 1) * chunk, S);
        for (int k = 0; k < hi; k++)
            mv[(size_t)q * S + k] = ggml_fp32_to_fp16(0.0f);
    }
    ggml_backend_tensor_set(dst, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());
}

// ------------------------------------------------------------------ preprocess

bool VisionTower::preprocess(const uint8_t * rgb, int width, int height,
                             std::vector<TilePre> & out) {
    out.clear();
    const int P = patch, S = 448;
    int cols = 0, rows = 0;
    const bool sliced = slice_grid(width, height, 9, S, &cols, &rows);

    auto push_tile = [&](const uint8_t * px, int tw_, int th_) {
        TilePre t;
        t.h_p = th_ / P;
        t.w_p = tw_ / P;
        const int n = t.h_p * t.w_p;
        t.patches.resize((size_t)588 * n);
        // patch vectors (c, y, x) C-order per token, tokens row-major: the
        // exact layout torch's reshape_by_patch + get_vision_embedding
        // roundtrip reconstructs for the folded [1152, 588] patch conv
        for (int p = 0; p < n; p++) {
            const int pi = p / t.w_p, pj = p % t.w_p;
            float * dst = t.patches.data() + (size_t)p * 588;
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < P; y++)
                    for (int x = 0; x < P; x++) {
                        const size_t si = (((size_t)(pi * P + y)) * tw_ +
                                           (size_t)pj * P + x) * 3 + c;
                        dst[(size_t)(c * P + y) * P + x] =
                            ((float)px[si] / 255.0f - img_mean[c]) / img_std[c];
                    }
        }
        vision_pos_ids(t.h_p, t.w_p, grid_side, t.pos_ids);
        sincos_pos_2d(t.h_p, t.w_p, rs_embed, t.sincos);
        out.push_back(std::move(t));
    };

    // tile 0: thumbnail (upscale allowed only when no slicing)
    int tw, th;
    best_resize(width, height, S, P, /*allow_upscale*/ !sliced, &tw, &th);
    std::vector<uint8_t> thumb((size_t)tw * th * 3);
    vqa::img::resize_bicubic_coeffs(rgb, width, height, 3, tw, th, thumb.data());
    push_tile(thumb.data(), tw, th);

    if (sliced) {
        int rw, rh;
        refine_size(width, height, cols, rows, S, P, &rw, &rh);
        std::vector<uint8_t> refine((size_t)rw * rh * 3);
        vqa::img::resize_bicubic_coeffs(rgb, width, height, 3, rw, rh, refine.data());
        const int sw = rw / cols, sh = rh / rows;
        std::vector<uint8_t> cell((size_t)sw * sh * 3);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                for (int y = 0; y < sh; y++)
                    memcpy(&cell[(size_t)y * sw * 3],
                           &refine[(((size_t)(r * sh + y)) * rw + (size_t)c * sw) * 3],
                           (size_t)sw * 3);
                push_tile(cell.data(), sw, sh);
            }
    }
    return !out.empty();
}

// ------------------------------------------------------------------ graph

// LayerNorm with affine (bias tensor may be null on synthetic paths).
static ggml_tensor * layer_norm(ggml_context * c, ggml_tensor * x,
                                ggml_tensor * w, ggml_tensor * b, float eps) {
    ggml_tensor * h = ggml_norm(c, x, eps);
    h = ggml_mul(c, h, w);
    if (b) h = ggml_add(c, h, b);
    return h;
}

ggml_tensor * resampler_branch(ggml_context * c, const ResamplerW & w,
                               ggml_tensor * x, ggml_tensor * pos,
                               int n_head, const char * dbg) {
    const int64_t D = w.query->ne[0];      // 4096
    const int64_t Q = w.query->ne[1];      // 64
    const int64_t n = x->ne[1];
    const int64_t dh = D / n_head;
    const float eps = 1e-6f;

    // keys/values: kv_proj -> ln_kv; keys += 2D sincos
    ggml_tensor * kv = ggml_mul_mat(c, w.kv_proj, x);            // [D, n]
    kv = layer_norm(c, kv, w.ln_kv_w, w.ln_kv_b, eps);
    ggml_tensor * keys = ggml_add(c, kv, pos);                   // [D, n]
    if (dbg) { ggml_set_name(kv, "rs_kv"); ggml_set_name(keys, "rs_keys"); }

    // queries: ln_q(learned query) - [D, Q]
    ggml_tensor * q = layer_norm(c, w.query, w.ln_q_w, w.ln_q_b, eps);

    // fused in_proj: q|k|v row blocks 0:4096:8192 along the OUT dim (ne[1]).
    // A weight block of D out-rows is D * nb[1] bytes; the F32 bias blocks
    // are D contiguous elements each.
    const size_t rb = (size_t)D * w.in_proj->nb[1];
    const size_t bb = (size_t)D * ggml_element_size(w.in_proj_b);
    ggml_tensor * wq = ggml_view_2d(c, w.in_proj, D, D, w.in_proj->nb[1], 0);
    ggml_tensor * wk = ggml_view_2d(c, w.in_proj, D, D, w.in_proj->nb[1], rb);
    ggml_tensor * wv = ggml_view_2d(c, w.in_proj, D, D, w.in_proj->nb[1], 2 * rb);
    ggml_tensor * bq = ggml_view_2d(c, w.in_proj_b, D, 1, bb, 0);
    ggml_tensor * bk = ggml_view_2d(c, w.in_proj_b, D, 1, bb, bb);
    ggml_tensor * bv = ggml_view_2d(c, w.in_proj_b, D, 1, bb, 2 * bb);
    // cont() the strided block views: the CPU mul_mat's F16 vec_dot path
    // mis-reads rows of a strided src0 at large offsets (repro: k/v blocks
    // of the 4096x12288 in_proj at +33MB came out scrambled while the +0
    // q block was exact)
    ggml_tensor * wqc = ggml_cont(c, wq);
    ggml_tensor * wkc = ggml_cont(c, wk);
    ggml_tensor * wvc = ggml_cont(c, wv);
    ggml_tensor * iq = ggml_add(c, ggml_mul_mat(c, wqc, q), bq);          // [D, Q]
    ggml_tensor * ik = ggml_add(c, ggml_mul_mat(c, wkc, keys), bk);
    ggml_tensor * iv = ggml_add(c, ggml_mul_mat(c, wvc, kv), bv);
    if (dbg) {
        ggml_set_name(iq, "rs_iq");
        ggml_set_name(ik, "rs_ik");
        ggml_set_name(iv, "rs_iv");
    }

    // cross attention: [dh, heads, T] FA layout, unmasked
    ggml_tensor * q3 = ggml_reshape_3d(c, iq, dh, n_head, Q);
    ggml_tensor * k3 = ggml_reshape_3d(c, ik, dh, n_head, n);
    ggml_tensor * v3 = ggml_reshape_3d(c, iv, dh, n_head, n);
    ggml_tensor * qf = ggml_cont(c, ggml_permute(c, q3, 0, 2, 1, 3));
    ggml_tensor * kf = ggml_cont(c, ggml_permute(c, k3, 0, 2, 1, 3));
    ggml_tensor * vf = ggml_cont(c, ggml_permute(c, v3, 0, 2, 1, 3));
    ggml_tensor * fa = ggml_flash_attn_ext(c, qf, kf, vf, nullptr,
                                           1.0f / std::sqrt((float)dh),
                                           0.0f, 0.0f);
    if (dbg) ggml_set_name(fa, "rs_fa");
    ggml_tensor * attn = ggml_reshape_2d(c, ggml_cont(c, fa), D, Q);
    if (dbg) ggml_set_name(attn, "rs_attn");
    if (dbg && !strcmp(dbg, "rr")) ggml_set_name(kv, "rr_kv");
    if (dbg && !strcmp(dbg, "rr")) ggml_set_name(keys, "rr_keys");
    if (dbg && !strcmp(dbg, "rr")) ggml_set_name(iq, "rr_iq");
    if (dbg && !strcmp(dbg, "rr")) ggml_set_name(fa, "rr_fa");
    if (dbg && !strcmp(dbg, "rr")) ggml_set_name(attn, "rr_attn");

    ggml_tensor * out = ggml_mul_mat(c, w.out_proj, attn);
    out = ggml_add(c, out, w.out_proj_b);
    if (dbg) ggml_set_name(out, "rr_o1");
    out = layer_norm(c, out, w.ln_post_w, w.ln_post_b, eps);
    if (dbg) ggml_set_name(out, "rr_lout");
    // RIGHT multiply x @ proj. NOTE: intentionally done on the HOST by the
    // caller (this graph ends at ln_post): the in-graph mul_mat(proj, x)
    // mis-computes here (CPU F16 GEMM path, second [4096,4096] F16 A in one
    // graph at N=64 - stages verified exact, output scrambled; repro'd in
    // tools/minicpmo_seg_test.cpp scenario_rs_real)
    if (dbg) ggml_set_name(out, "v_rs_out");
    return out;
}

// SigLIP tower over one tile + resampler. Returns token rows [64, 4096].
static ggml_tensor * siglip_tile(ggml_context * c, VisionTower & vt,
                                 int n, bool dbg0) {
    const int D = vt.n_embd, NH = vt.n_head, HD = D / NH;

    ggml_tensor * pix = ggml_new_tensor_2d(c, GGML_TYPE_F32, 588, n);
    ggml_set_name(pix, "v_pix_in");
    ggml_set_input(pix);
    ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    ggml_set_name(ids, "v_pos_ids");
    ggml_set_input(ids);
    ggml_tensor * pos = ggml_new_tensor_2d(c, GGML_TYPE_F32, vt.rs_embed, n);
    ggml_set_name(pos, "v_rs_pos");
    ggml_set_input(pos);

    ggml_tensor * x = ggml_mul_mat(c, vt.patch_w, pix);
    if (dbg0) ggml_set_name(x, "v_mulmat");
    x = ggml_add(c, x, vt.patch_b);
    if (dbg0) ggml_set_name(x, "v_conv");
    x = ggml_add(c, x, ggml_get_rows(c, vt.pos_embd, ids));
    if (dbg0) ggml_set_name(x, "v_embed");

    for (int il = 0; il < (int)vt.vl.size(); il++) {
        const SiglipLayer & L = vt.vl[(size_t)il];
        ggml_tensor * h = layer_norm(c, x, L.ln1_w, L.ln1_b, vt.ln_eps);
        ggml_tensor * q = ggml_mul_mat(c, L.wq, h);
        q = ggml_add(c, q, L.bq);
        ggml_tensor * k = ggml_mul_mat(c, L.wk, h);
        k = ggml_add(c, k, L.bk);
        ggml_tensor * v = ggml_mul_mat(c, L.wv, h);
        v = ggml_add(c, v, L.bv);
        // head_dim 72 is not a supported FA width - pad every head to 80
        // with zeros (q/k pad dims contribute nothing to the dot product,
        // v pad dims are cut off after attention; the moondream SigLIP
        // precedent). qkv stays [HD, NH, n] - the pad is per head.
        const int NHp = 80;
        ggml_tensor * zpad = ggml_new_tensor_3d(c, GGML_TYPE_F32, 8, NH, n);
        ggml_set_name(zpad, "v_zpad");
        auto pad80 = [&](ggml_tensor * t) {
            return ggml_cont(c, ggml_concat(c, t, zpad, 0));
        };
        ggml_tensor * q3 = pad80(ggml_reshape_3d(c, q, HD, NH, n));
        ggml_tensor * k3 = pad80(ggml_reshape_3d(c, k, HD, NH, n));
        ggml_tensor * v3 = pad80(ggml_reshape_3d(c, v, HD, NH, n));
        // official stack forces eager attention; one unmasked FA over the
        // whole tile is the same softmax (single tile per graph: no pad rows)
        ggml_tensor * qf = ggml_cont(c, ggml_permute(c, q3, 0, 2, 1, 3));
        ggml_tensor * kf = ggml_cont(c, ggml_permute(c, k3, 0, 2, 1, 3));
        ggml_tensor * vf = ggml_cont(c, ggml_permute(c, v3, 0, 2, 1, 3));
        ggml_tensor * fa = ggml_flash_attn_ext(c, qf, kf, vf, nullptr,
                                               1.0f / std::sqrt((float)HD),
                                               0.0f, 0.0f);
        // de-pad: first 72 dims of each head. FA output ne = [80, n, NH]
        // (token axis ne1, head axis ne2)
        ggml_tensor * fa72 = ggml_view_3d(c, fa, HD, n, NH,
                                          NHp * ggml_element_size(fa),
                                          NHp * n * ggml_element_size(fa), 0);
        ggml_tensor * attn = ggml_reshape_2d(c, ggml_cont(c, fa72), D, n);
        attn = ggml_mul_mat(c, L.wo, attn);
        attn = ggml_add(c, attn, L.bo);
        x = ggml_add(c, x, attn);

        h = layer_norm(c, x, L.ln2_w, L.ln2_b, vt.ln_eps);
        ggml_tensor * f = ggml_mul_mat(c, L.fc1_w, h);
        f = ggml_add(c, f, L.fc1_b);
        f = ggml_gelu(c, f);               // gelu_pytorch_tanh
        f = ggml_mul_mat(c, L.fc2_w, f);
        f = ggml_add(c, f, L.fc2_b);
        x = ggml_add(c, x, f);
        if (dbg0 && il == 0) ggml_set_name(x, "v_blk0_out");
    }
    x = layer_norm(c, x, vt.post_ln_w, vt.post_ln_b, vt.ln_eps);
    if (dbg0) ggml_set_name(x, "v_siglip_out");

    if (dbg0) ggml_set_name(x, "v_siglip_out");
    return x;
}

// resampler-only graph (fed by the direct-read siglip output of the
// previous graph - every stage of the tower is then a direct read)
static ggml_tensor * resampler_tile(ggml_context * c, VisionTower & vt,
                                    int n, bool dbg0) {
    ggml_tensor * sig = ggml_new_tensor_2d(c, GGML_TYPE_F32, vt.n_embd, n);
    ggml_set_name(sig, "v_siglip_in");
    ggml_set_input(sig);
    ggml_tensor * pos = ggml_new_tensor_2d(c, GGML_TYPE_F32, vt.rs_embed, n);
    ggml_set_name(pos, "v_rs_pos");
    ggml_set_input(pos);
    ggml_tensor * r = resampler_branch(c, vt.rs, sig, pos, vt.rs_heads,
                                       dbg0 ? "rs" : nullptr);
    if (dbg0) ggml_set_name(r, "v_rs_out");
    // ln_out rows (tokens x 4096); the proj right-multiply runs on the host
    return r;
}

bool VisionTower::encode(std::vector<TilePre> & tiles, std::vector<float> & feats) {
    if (tiles.empty() || vl.empty()) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    feats.clear();
    feats.reserve(tiles.size() * (size_t)query_num * rs_embed);
    const bool want_dump = dump_dir && dump_dir[0];

    for (size_t ti = 0; ti < tiles.size(); ti++) {
        const VisionTower::TilePre & t = tiles[ti];
        const int n = t.h_p * t.w_p;
        if (want_dump) vqa::dump::dump_registry_clear();
        ggml_context * c = nullptr;
        ggml_cgraph * gf = nullptr;
        {
            struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, true };
            c = ggml_init(ip);
            if (!c) return false;
            gf = ggml_new_graph_custom(c, 32768, false);
        }
        // ---- graph 1: SigLIP tower -> [1152, n] (direct read) -------------
        ggml_tensor * out = siglip_tile(c, *this, n, false);
        ggml_build_forward_expand(gf, out);
        if (want_dump && ti == 0) {
            static const char * const dn[] = {
                "v_pix_in", "v_embed", "v_blk0_out", "v_siglip_out", nullptr,
            };
            vqa::dump::mark_sinks(c, gf, dn);
            vqa::dump::finalize_sinks(c, gf, backend);
        }
        if (!ggml_gallocr_alloc_graph(allocr, gf)) {
            fprintf(stderr, "minicpmo vision: graph alloc failed\n");
            ggml_free(c);
            return false;
        }
        {   // the FA head-dim pad MUST be zeroed (gallocr memory is garbage)
            ggml_tensor * zp = ggml_graph_get_tensor(gf, "v_zpad");
            const size_t zb = (size_t)zp->ne[0] * zp->ne[1] * zp->ne[2]
                              * ggml_element_size(zp);
            std::vector<uint8_t> z(zb, 0);
            ggml_backend_tensor_set(zp, z.data(), 0, zb);
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "v_pix_in"),
                                t.patches.data(), 0,
                                sizeof(float) * t.patches.size());
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "v_pos_ids"),
                                t.pos_ids.data(), 0,
                                sizeof(int) * t.pos_ids.size());
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "minicpmo vision: compute failed\n");
            ggml_free(c);
            return false;
        }
        if (want_dump && ti == 0) {
            vqa::dump::write(gf, dump_dir, "v_pix_in");
            fprintf(stderr,
                    "vision ids grid_side=%d h=%d w=%d: %d %d %d %d ..."
                    " %d %d %d %d\n",
                    grid_side, t.h_p, t.w_p, t.pos_ids[0], t.pos_ids[1],
                    t.pos_ids[2], t.pos_ids[3], t.pos_ids[n - 4],
                    t.pos_ids[n - 3], t.pos_ids[n - 2], t.pos_ids[n - 1]);
            vqa::dump::write(gf, dump_dir, "v_embed");
            vqa::dump::write(gf, dump_dir, "v_blk0_out");
            vqa::dump::write(gf, dump_dir, "v_siglip_out");
        }
        std::vector<float> sig((size_t)n_embd * n);
        ggml_backend_tensor_get(out, sig.data(), 0,
                                sizeof(float) * sig.size());

        // ---- graph 2: resampler -> ln_out rows [4096, 64] (direct read) ---
        ggml_free(c);
        {
            struct ggml_init_params ip2 = { 64ull * 1024 * 1024, NULL, true };
            c = ggml_init(ip2);
            if (!c) return false;
            gf = ggml_new_graph_custom(c, 8192, false);
        }
        out = resampler_tile(c, *this, n, want_dump && ti == 0);
        ggml_build_forward_expand(gf, out);
        if (!ggml_gallocr_alloc_graph(allocr, gf)) {
            fprintf(stderr, "minicpmo vision: resampler alloc failed\n");
            ggml_free(c);
            return false;
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "v_siglip_in"),
                                sig.data(), 0, sizeof(float) * sig.size());
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "v_rs_pos"),
                                t.sincos.data(), 0,
                                sizeof(float) * t.sincos.size());
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "minicpmo vision: resampler compute failed\n");
            ggml_free(c);
            return false;
        }
        if (want_dump && ti == 0) {
            vqa::dump::write(gf, dump_dir, "v_rs_pos");
            vqa::dump::write(gf, dump_dir, "v_rs_out");
        }
        const int64_t rows = out->ne[1];
        size_t off = feats.size();
        feats.resize(off + (size_t)rows * rs_embed);
        ggml_backend_tensor_get(out, feats.data() + off, 0,
                                sizeof(float) * (size_t)rows * rs_embed);
        if (rs_host_proj.empty()) {
            // f32 host copy of the proj weight (F16 in the GGUF)
            const size_t n = (size_t)rs_embed * rs_embed;
            std::vector<ggml_fp16_t> h(n);
            ggml_backend_tensor_get(rs.proj, h.data(), 0, sizeof(ggml_fp16_t) * n);
            rs_host_proj.resize(n);
            for (size_t i = 0; i < n; i++) rs_host_proj[i] = ggml_fp16_to_fp32(h[i]);
        }
        // x @ proj: feats rows are tokens [rows, 4096]; proj [in, out] flat
        std::vector<float> src(feats.begin() + off, feats.end());
        for (int64_t t = 0; t < rows; t++) {
            const float * xv = &src[(size_t)t * rs_embed];
            float * ov = &feats[off + (size_t)t * rs_embed];
            for (int64_t j = 0; j < rs_embed; j++) ov[j] = 0.0f;
            // x @ proj: proj flat is [out][in] (o-major), so accumulate
            // column j over k with the row pointer walking in
            for (int64_t k = 0; k < rs_embed; k++) {
                const float xv_k = xv[k];
                if (xv_k == 0.0f) continue;
                const float * pr = &rs_host_proj[(size_t)k * rs_embed];
                for (int64_t j = 0; j < rs_embed; j++) ov[j] += xv_k * pr[j];
            }
        }
        if (want_dump && ti == 0) {
            // parity surface: the tile's final (host-proj'd) resampler rows
            FILE * fp = fopen(std::string(std::string(dump_dir) +
                                          "/host_feats.bin").c_str(), "wb");
            if (fp) {
                fwrite(feats.data(), 4, feats.size(), fp);
                fclose(fp);
            }
        }
        ggml_free(c);
        n_tok = (int)(feats.size() / rs_embed);
    }
    return true;
}

}  // namespace minicpmo
