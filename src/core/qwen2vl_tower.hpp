// qwen2vl_tower.hpp - the Qwen2.5-VL vision tower shared by the qwen2vl and
// qwen25omni backends.
//
// The tower is identical down to the op sequence in both engines: Conv3d patch
// embed folded into one matmul (temporal taps duplicated by the processor),
// window-major reorder of the 2x2 merged-token groups, block-diagonal window
// attention with full-attention every n_wa_pattern-th layer, 4-section 2D
// M-RoPE (h, w, h, w), RMSNorm + fused qkv with biases + SwiGLU MLP, and the
// merger (per-token RMSNorm over v_embd *before* the 2x2 fold, then GELU-erf
// MLP). The code below is lifted verbatim from the two model backends; every
// float constant, op order, mask construction and rope position fill is
// unchanged. Differences between the two call sites are fields:
//   * cfg (layers/dims/eps/theta/CLIP mean+std) - filled from each model's gguf
//   * hooks.max_layers_env - "Q2V_MAX_VLAYERS" vs "QOMNI_MAX_VLAYERS"
//   * the optional debug probes (omni's QOMNI_VB0/VPATCH/VLAYER dumps,
//     qwen2vl's dump_dir sinks) ride in through nullable hook callbacks so the
//     refactored graphs stay numerically identical.
#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <math.h>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace vqa {
namespace qwen2vl_tower {

struct Layer {
    ggml_tensor *ln1 = nullptr, *ln2 = nullptr;
    ggml_tensor *wqkv = nullptr, *bqkv = nullptr;
    ggml_tensor *wo = nullptr, *bo = nullptr;
    ggml_tensor *wg = nullptr, *bg = nullptr;
    ggml_tensor *wu = nullptr, *bu = nullptr;
    ggml_tensor *wd = nullptr, *bd = nullptr;
};

struct Weights {
    ggml_tensor *patch_w = nullptr;
    ggml_tensor *merger_ln = nullptr, *mm1_w = nullptr, *mm1_b = nullptr;
    ggml_tensor *mm2_w = nullptr, *mm2_b = nullptr;
    std::vector<Layer> layers;
};

struct Config {
    int n_layer = 32, n_embd = 1280, n_head = 16, n_hd = 80;
    int n_patch = 14, n_temporal = 2, n_merge = 2, n_out = 2048;
    int n_window = 112, n_wa_pattern = 8;
    float eps = 1e-6f, rope_theta = 10000.0f;
    float img_mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };   // CLIP stats
    float img_std[3]  = { 0.26862954f, 0.26130258f, 0.27577711f };
};

struct Hooks {
    // env var name that truncates the tower for bisection (""/null = full run)
    const char *max_layers_env = nullptr;
    void *ud = nullptr;
    // called right after the patch-embed matmul (omni: QOMNI_VB0_DUMP probe +
    // QOMNI_VPATCH_DUMP dup sink); x is the [v_embd, N] patch output
    void (*after_patch)(void *ud, ggml_context *ctx, ggml_cgraph *gf, int v_embd, int N,
                        ggml_tensor *x) = nullptr;
    // called just before the layer loop (omni: QOMNI_VLAYER_DUMP buffers)
    void (*before_layers)(void *ud, ggml_context *ctx, ggml_cgraph *gf, int v_layer,
                          int v_embd, int N) = nullptr;
    // called after layer-0's attention residual (omni: vb0_attn.bin probe);
    // x is the post-attention residual stream
    void (*after_layer0_attn)(void *ud, ggml_context *ctx, ggml_cgraph *gf, int v_embd,
                              int N, ggml_tensor *x) = nullptr;
    // called at the end of every layer body (omni: per-layer hidden copies)
    void (*after_layer)(void *ud, ggml_context *ctx, ggml_cgraph *gf, int il, int v_embd,
                        int N, ggml_tensor *x) = nullptr;
    // called after the graph is built, before gallocr allocation; return false
    // to abort (qwen2vl: dump_dir mark_sinks + finalize_sinks)
    bool (*finalize_build)(void *ud, ggml_context *ctx, ggml_cgraph *gf,
                           ggml_backend_t backend) = nullptr;
    // called right after compute, before reading v_feats (per-site debug dumps)
    void (*after_run)(void *ud, ggml_cgraph *gf) = nullptr;
};

struct State {
    ggml_context *ctx = nullptr;
    ggml_cgraph *gf = nullptr;
    ggml_tensor *in = nullptr, *pos = nullptr, *mask = nullptr;
    ggml_tensor *idx_in = nullptr, *idx_out = nullptr;
    ggml_tensor *feats = nullptr;
    int n_patches_x = 0, n_patches_y = 0;   // current graph grid

    void free() {
        if (ctx) ggml_free(ctx);
        ctx = nullptr;
    }
};

struct Tower {
    Config cfg;
    Weights w;
    State st;
    Hooks hooks;

    void free() { st.free(); }

    // ------------------------------------------------------------------
    // build_vision_graph: one graph per (gh, gw); leaves the merged features
    // [n_out, nm] tensor in st.feats.
    bool build(int gh, int gw, ggml_gallocr_t allocr, ggml_backend_t backend) {
        if (st.ctx) { ggml_free(st.ctx); st.ctx = nullptr; }
        const int v_layer = cfg.n_layer, v_embd = cfg.n_embd, v_head = cfg.n_head;
        const int v_hd = cfg.n_hd, v_patch = cfg.n_patch, v_temporal = cfg.n_temporal;
        const int v_merge = cfg.n_merge, v_window = cfg.n_window;
        const int n_wa_pattern = cfg.n_wa_pattern;
        const float v_eps = cfg.eps, v_rope_theta = cfg.rope_theta;
        const int N = gh * gw;
        const int nm = N / (v_merge * v_merge);
        const int C3 = 3 * v_temporal * v_patch * v_patch;

        struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, /*no_alloc*/ true };
        st.ctx = ggml_init(ip);
        if (!st.ctx) return false;
        ggml_context * vctx = st.ctx;
        st.gf = ggml_new_graph_custom(vctx, 65536, false);
        ggml_cgraph * vgf = st.gf;

        st.in = ggml_new_tensor_2d(vctx, GGML_TYPE_F32, C3, N);
        ggml_set_name(st.in, "v_patches");
        st.pos = ggml_new_tensor_1d(vctx, GGML_TYPE_I32, 4 * N);
        ggml_set_name(st.pos, "v_pos");
        const bool wa = n_wa_pattern > 0;
        if (wa) {
            st.mask = ggml_new_tensor_2d(vctx, GGML_TYPE_F16, N, N);
            ggml_set_name(st.mask, "v_mask");
            st.idx_in = ggml_new_tensor_1d(vctx, GGML_TYPE_I32, nm);
            ggml_set_name(st.idx_in, "v_idx_in");
            st.idx_out = ggml_new_tensor_1d(vctx, GGML_TYPE_I32, nm);
            ggml_set_name(st.idx_out, "v_idx_out");
        }

        // window-major reorder, applied to the 2x2 groups (4 patch rows at a time)
        ggml_tensor * x = st.in;
        if (wa) {
            x = ggml_reshape_2d(vctx, x, C3 * 4, nm);
            x = ggml_get_rows(vctx, x, st.idx_in);
            x = ggml_reshape_2d(vctx, x, C3, N);
        }
        x = ggml_mul_mat(vctx, w.patch_w, x);               // [v_embd, N]
        ggml_set_name(x, "v_patch_out");
        if (hooks.after_patch)
            hooks.after_patch(hooks.ud, vctx, vgf, v_embd, N, x);

        // 4D M-RoPE sections: head_dim/4 each, positions (h, w, h, w)
        int vsec[4] = { v_hd / 4, v_hd / 4, v_hd / 4, v_hd / 4 };

        int v_max = v_layer;   // MAX_VLAYERS=N truncates the tower for bisecting
        if (hooks.max_layers_env && hooks.max_layers_env[0]) {
            if (const char * e = getenv(hooks.max_layers_env))
                v_max = std::min(v_layer, atoi(e));
        }
        if (hooks.before_layers)
            hooks.before_layers(hooks.ud, vctx, vgf, v_layer, v_embd, N);
        for (int il = 0; il < v_max; il++) {
            const Layer & L = w.layers[(size_t)il];
            const bool full_attn = !wa || ((il + 1) % n_wa_pattern == 0);

            ggml_tensor * h = ggml_rms_norm(vctx, x, v_eps);
            h = ggml_mul(vctx, h, L.ln1);
            ggml_tensor * qkv = ggml_mul_mat(vctx, L.wqkv, h);   // [3*v_embd, N]
            qkv = ggml_add(vctx, qkv, L.bqkv);
            if (il == 0) ggml_set_name(qkv, "v_blk0_qkv");

            const size_t row = qkv->nb[1];
            ggml_tensor * q = ggml_view_2d(vctx, qkv, v_embd, N, row, 0);
            ggml_tensor * k = ggml_view_2d(vctx, qkv, v_embd, N, row,
                                           (size_t)v_embd * sizeof(float));
            ggml_tensor * v = ggml_view_2d(vctx, qkv, v_embd, N, row,
                                           (size_t)2 * v_embd * sizeof(float));
            ggml_tensor * q3 = ggml_reshape_3d(vctx, ggml_cont(vctx, q), v_hd, v_head, N);
            ggml_tensor * k3 = ggml_reshape_3d(vctx, ggml_cont(vctx, k), v_hd, v_head, N);
            ggml_tensor * v3 = ggml_reshape_3d(vctx, ggml_cont(vctx, v), v_hd, v_head, N);

            q3 = ggml_rope_multi(vctx, q3, st.pos, nullptr, v_hd / 2, vsec,
                                 GGML_ROPE_TYPE_VISION, 32768, v_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k3 = ggml_rope_multi(vctx, k3, st.pos, nullptr, v_hd / 2, vsec,
                                 GGML_ROPE_TYPE_VISION, 32768, v_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            if (il == 0) ggml_set_name(q3, "v_blk0_q_rope");

            ggml_tensor * qf = ggml_cont(vctx, ggml_permute(vctx, q3, 0, 2, 1, 3));   // [hd, N, head]
            ggml_tensor * kf = ggml_cont(vctx, ggml_permute(vctx, k3, 0, 2, 1, 3));
            ggml_tensor * vf = ggml_cont(vctx, ggml_permute(vctx, v3, 0, 2, 1, 3));
            ggml_tensor * mask = full_attn ? nullptr : st.mask;
            ggml_tensor * fa = ggml_flash_attn_ext(vctx, qf, kf, vf, mask,
                                                   1.0f / sqrtf((float)v_hd), 0.0f, 0.0f);
            // FA returns [hd, heads, tokens]: flatten straight to [v_embd, N]
            ggml_tensor * attn = ggml_reshape_2d(vctx, ggml_cont(vctx, fa), v_embd, N);
            attn = ggml_mul_mat(vctx, L.wo, attn);
            attn = ggml_add(vctx, attn, L.bo);
            x = ggml_add(vctx, x, attn);
            if (il == 0) ggml_set_name(x, "v_blk0_attn_out");
            if (il == 0 && hooks.after_layer0_attn)
                hooks.after_layer0_attn(hooks.ud, vctx, vgf, v_embd, N, x);

            h = ggml_rms_norm(vctx, x, v_eps);
            h = ggml_mul(vctx, h, L.ln2);
            ggml_tensor * g = ggml_mul_mat(vctx, L.wg, h);
            g = ggml_add(vctx, g, L.bg);
            ggml_tensor * u = ggml_mul_mat(vctx, L.wu, h);
            u = ggml_add(vctx, u, L.bu);
            h = ggml_mul(vctx, ggml_silu(vctx, g), u);
            h = ggml_mul_mat(vctx, L.wd, h);
            h = ggml_add(vctx, h, L.bd);
            x = ggml_add(vctx, x, h);
            if (il == 0) ggml_set_name(x, "v_blk0_out");
            if (il == 15) ggml_set_name(x, "v_blk15_out");
            if (hooks.after_layer)
                hooks.after_layer(hooks.ud, vctx, vgf, il, v_embd, N, x);
        }
        ggml_set_name(x, "v_final");

        // merger: HF normalizes per token over context_dim (1280) and only then folds
        // the 2x2 group into one 4*1280 row - doing the RMSNorm after the reshape
        // normalizes over 5120 and destroys the features
        ggml_tensor * xn = ggml_rms_norm(vctx, x, v_eps);
        xn = ggml_mul(vctx, xn, w.merger_ln);
        ggml_tensor * m = ggml_reshape_2d(vctx, xn, v_embd * v_merge * v_merge, nm);
        ggml_set_name(m, "v_merge");
        m = ggml_mul_mat(vctx, w.mm1_w, m);
        m = ggml_add(vctx, m, w.mm1_b);
        m = ggml_gelu_erf(vctx, m);
        m = ggml_mul_mat(vctx, w.mm2_w, m);
        m = ggml_add(vctx, m, w.mm2_b);
        ggml_set_name(m, "v_proj_out");
        if (wa) m = ggml_get_rows(vctx, m, st.idx_out);   // window-major -> raster
        ggml_set_name(m, "v_feats");
        st.feats = m;
        ggml_build_forward_expand(vgf, m);

        if (hooks.finalize_build &&
            !hooks.finalize_build(hooks.ud, vctx, vgf, backend))
            return false;
        if (!ggml_gallocr_alloc_graph(allocr, vgf)) {
            fprintf(stderr, "vision graph allocation failed\n");
            return false;
        }
        st.n_patches_x = gw;
        st.n_patches_y = gh;
        return true;
    }

    // ------------------------------------------------------------------
    // run_vision: uploads patches, fills the window-major rope positions and
    // the block-diagonal mask, computes, reads back the merged features.
    bool run(ggml_backend_t backend, const std::vector<float> & patches,
             std::vector<float> & feats) {
        const int v_patch = cfg.n_patch, v_merge = cfg.n_merge;
        const int v_window = cfg.n_window, n_wa_pattern = cfg.n_wa_pattern;
        const int N = st.n_patches_x * st.n_patches_y;
        const int nm = N / (v_merge * v_merge);

        ggml_backend_tensor_set(st.in, patches.data(), 0, sizeof(float) * patches.size());

        // positions: 4 arrays (h, w, h, w) in *window-major* patch order
        {
            std::vector<int> pos((size_t)4 * N, 0);
            std::vector<int> idx((size_t)nm), inv((size_t)nm);
            const int gw = st.n_patches_x, gh = st.n_patches_y;
            const int mw = gw / v_merge, mh = gh / v_merge;
            const int wgrid = v_window / v_patch / v_merge;   // 4 merged tokens per window side
            int dst = 0;
            if (n_wa_pattern > 0) {
                for (int y = 0; y < mh; y += wgrid) {
                    for (int x = 0; x < mw; x += wgrid) {
                        const int wh = std::min(wgrid, mh - y), ww = std::min(wgrid, mw - x);
                        for (int dy = 0; dy < wh; dy++)
                            for (int dx = 0; dx < ww; dx++) {
                                const int src = (y + dy) * mw + (x + dx);
                                idx[(size_t)src] = dst;
                                inv[(size_t)dst] = src;
                                dst++;
                            }
                    }
                }
            } else {
                for (int i = 0; i < nm; i++) { idx[(size_t)i] = i; inv[(size_t)i] = i; }
            }
            // patch j of merge group g (raster) sits at window-major slot idx[g]*4 + (j%4)
            for (int g = 0; g < nm; g++) {
                const int r = g / mw, c = g % mw;
                for (int j = 0; j < v_merge * v_merge; j++) {
                    const int dy = j / v_merge, dx = j % v_merge;
                    const int p = idx[(size_t)g] * v_merge * v_merge + j;
                    const int py = r * v_merge + dy, px = c * v_merge + dx;
                    pos[(size_t)p] = py;
                    pos[(size_t)N + p] = px;
                    pos[(size_t)2 * N + p] = py;
                    pos[(size_t)3 * N + p] = px;
                }
            }
            ggml_backend_tensor_set(st.pos, pos.data(), 0, sizeof(int) * pos.size());
            if (n_wa_pattern > 0) {
                ggml_backend_tensor_set(st.idx_in, inv.data(), 0, sizeof(int) * inv.size());
                ggml_backend_tensor_set(st.idx_out, idx.data(), 0, sizeof(int) * idx.size());
                // block-diagonal mask over the reordered patch tokens
                std::vector<ggml_fp16_t> mask((size_t)N * N, ggml_fp32_to_fp16(-INFINITY));
                for (int y = 0; y < mh; y += wgrid) {
                    for (int x = 0; x < mw; x += wgrid) {
                        const int wh = std::min(wgrid, mh - y), ww = std::min(wgrid, mw - x);
                        const int got = wh * ww;   // merged tokens in this window
                        int lo = -1;
                        for (int dy = 0; dy < wh && lo < 0; dy++)
                            for (int dx = 0; dx < ww; dx++)
                                if (idx[(size_t)((y + dy) * mw + (x + dx))] >= 0) {
                                    lo = idx[(size_t)((y + dy) * mw + (x + dx))];
                                    break;
                                }
                        if (lo < 0) continue;
                        const int p0 = lo * v_merge * v_merge;
                        const int p1 = p0 + got * v_merge * v_merge;
                        for (int r = p0; r < p1; r++)
                            for (int s = p0; s < p1; s++)
                                mask[(size_t)r * N + s] = ggml_fp32_to_fp16(0.0f);
                    }
                }
                ggml_backend_tensor_set(st.mask, mask.data(), 0, sizeof(ggml_fp16_t) * mask.size());
            }
        }

        if (ggml_backend_graph_compute(backend, st.gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "vision compute failed\n");
            return false;
        }
        if (hooks.after_run) hooks.after_run(hooks.ud, st.gf);
        const int n = (int)st.feats->ne[1], d = (int)st.feats->ne[0];
        feats.resize((size_t)n * d);
        ggml_backend_tensor_get(st.feats, feats.data(), 0, sizeof(float) * feats.size());
        (void)nm;
        return true;
    }

    // ------------------------------------------------------------------
    // patchify: 2x2 group-order patch extraction + CLIP mean/std, identical in
    // both backends. `resized` is [rh*rw*3] float RGB. Fills `patches` with
    // [C3, gh*gw] rows (temporal tap duplicated, like the official processor).
    void patchify(const float * resized, int rw, int rh, int gh, int gw,
                  std::vector<float> & patches) const {
        const int v_patch = cfg.n_patch, v_temporal = cfg.n_temporal, v_merge = cfg.n_merge;
        const int P = v_patch;
        const int N = gh * gw;
        const int C3 = 3 * v_temporal * P * P;
        patches.assign((size_t)C3 * N, 0.0f);
        const float * v_mean = cfg.img_mean;
        const float * v_std = cfg.img_std;
        {
            // the official processor emits patches in 2x2 group order:
            //   (t, gh/2, gw/2, merge_h, merge_w) x (c, dt, dh, dw)
            // (see Qwen2VLImageProcessor._preprocess: reshape + transpose(0,3,6,4,7,2,1,5,8))
            const int mh = gh / v_merge, mw = gw / v_merge;
            int u = 0;
            for (int gy = 0; gy < mh; gy++)
                for (int gx = 0; gx < mw; gx++)
                    for (int dy = 0; dy < v_merge; dy++)
                        for (int dx = 0; dx < v_merge; dx++) {
                            const int py = gy * v_merge + dy, px = gx * v_merge + dx;
                            float * row = patches.data() + (size_t)u * C3;
                            int o = 0;
                            for (int c = 0; c < 3; c++)
                                for (int dt = 0; dt < v_temporal; dt++)   // same pixels twice
                                    for (int dh = 0; dh < P; dh++) {
                                        const float * src = resized +
                                            ((size_t)(py * P + dh) * rw + px * P) * 3 + c;
                                        for (int dw = 0; dw < P; dw++)
                                            row[o++] = (src[(size_t)dw * 3] / 255.0f - v_mean[c]) /
                                                       v_std[c];
                                    }
                            u++;
                        }
        }
    }
};

}  // namespace qwen2vl_tower
}  // namespace vqa
