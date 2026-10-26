// qwen3vl_tower.hpp - the Qwen3-VL ViT tower shared by the qwen3vl and
// qwen3omni backends.
//
// Lifted verbatim from qwen3vl_model.cpp's encode_vision (graph side): patch
// matmul + interpolated pos add, 27-ish ViT blocks through the shared
// vqa::vit::block (fused qkv, 2D VISION rope via t_vpos, per-head flash
// attention, GELU MLP), DeepStack taps (own postshuffle-norm merger each,
// staged for the text graph), and the final pre-shuffle-norm merger. Every
// float constant, op order, mask and rope position fill is unchanged from the
// qwen3vl backend - only the field access became parameters.
//
// Ownership split:
//   * THIS FILE owns the tower GRAPH + compute loop over frames;
//   * the CALLER owns everything model-specific: the weight tensors (any GGUF
//     naming), the host preprocessing (resize + patchify3d + pos interpolation
//     stay in each backend's preprocess header), the staging buffers
//     (StageBuffer for the tower output and each DeepStack tap), the text
//     width (DeepStack output width), and the env prefix (QWEN3VL_/QOMNI_).
#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "core/engine/backend.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/stage_buf.hpp"
#include "core/blocks/vit_block.hpp"

namespace vqa {
namespace qwen3vl_tower {

struct Config {
    int n_layer = 27, n_embd = 1152, n_head = 16, n_ff = 4304;
    int patch = 16, temporal_patch = 2, merge = 2;
    float layernorm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    std::vector<int> deepstack;      // block indices that feed a DeepStack merger
};

struct Layer {
    ggml_tensor *norm0_w = nullptr, *norm0_b = nullptr;
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *wqkv_w = nullptr, *wqkv_b = nullptr;
    ggml_tensor *wo_w = nullptr, *wo_b = nullptr;
    ggml_tensor *fc0_w = nullptr, *fc0_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
};

struct Merger {
    ggml_tensor *norm_w = nullptr, *norm_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
};

struct Weights {
    ggml_tensor *patch_w = nullptr;                  // [n_embd, C3] (conv folded)
    std::vector<Layer> layers;
    Merger main;                                     // pre-shuffle-norm merger
    std::vector<Merger> ds;                          // DeepStack mergers (one per tap)
};

// one frame's host-side inputs, produced by the caller's preprocess
struct FrameInput {
    const float * patches;                           // [C3, hw]
    const float * pos;                               // [n_embd, hw] (pos+bias folded)
    const int * coords;                              // [hw, 2] (h, w) rope coords
};

struct RunCtx {
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;
    const Config * cfg = nullptr;
    const Weights * w = nullptr;
    int text_width = 2048;           // DeepStack merger output width
    StageBuffer * tower_buf = nullptr;
    StageBuffer * ds_buf = nullptr;  // [n_taps] (may be null when no deepstack)
    const char * env_prefix = "QWEN3VL";             // <prefix>_STOP_AFTER / _VL
    std::string dump_dir;
    // write "v_final" / "v_proj_out" class dumps into this dir when non-empty
};

// H, W: resized frame size; gh, gw: patch grid. T frames share the grid.
// Returns false on failure; on success feats holds [text-width merged output
// (main merger: n_embd*4 -> out 2048-ish), n_tokens = mh*mw*T] and the tap
// outputs sit in ds_buf for the text graph.
inline bool run(RunCtx & rc, const FrameInput * frames, int T, int H, int W, int gh, int gw,
         std::vector<float> & feats, int * n_tokens) {
    const Config & vc = *rc.cfg;
    const int hw = gh * gw;
    const int mh = gh / vc.merge, mw = gw / vc.merge;
    const int n_ds = (int)vc.deepstack.size();
    const int vhd = vc.n_embd / vc.n_head;
    const int C3 = 3 * vc.temporal_patch * vc.patch * vc.patch;

    char env_stop[64], env_vl[64];
    snprintf(env_stop, sizeof(env_stop), "%s_STOP_AFTER", rc.env_prefix);
    snprintf(env_vl, sizeof(env_vl), "%s_VL", rc.env_prefix);
    const char * stop_after = getenv(env_stop);

    if (!rc.tower_buf->ensure(rc.backend, vc.n_embd, (int64_t)hw * T)) return false;
    if (n_ds > 0 && rc.ds_buf)
        for (int k = 0; k < n_ds; k++)
            if (!rc.ds_buf[k].ensure(rc.backend, rc.text_width, (int64_t)mh * mw * T)) return false;

    for (int s = 0; s < T; s++) {
        const int Nc = hw;
        if (!rc.dump_dir.empty()) vqa::dump::dump_registry_clear();
        struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
        ggml_context * gctx = ggml_init(ip);
        if (!gctx) return false;
        struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);

        ggml_tensor * t_patches = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, C3, Nc);
        ggml_tensor * t_pos = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, vc.n_embd, Nc);
        ggml_set_name(t_pos, "v_pos_in");
        ggml_tensor * t_vpos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, (int64_t)Nc * 4);
        ggml_set_name(t_patches, "patches");
        std::vector<int> vpos((size_t)Nc * 4, 0);
        for (int i = 0; i < Nc; i++) {
            vpos[i] = frames[s].coords[(size_t)i * 2];                    // h (section 0)
            vpos[(size_t)Nc + i] = frames[s].coords[(size_t)i * 2 + 1];   // w (section 1)
        }

        ggml_tensor * x = ggml_mul_mat(gctx, rc.w->patch_w, t_patches);
        // note: qwen3vl's converter folds the patch bias into the pos table on
        // the host; if a caller keeps a separate bias it must pre-fold it
        x = ggml_add(gctx, x, t_pos);
        ggml_set_name(x, "v_patch_out");

        bool sinks_ready = false;
        auto run_chunk = [&](const char * out_name, std::vector<float> * dest) -> bool {
            if (!rc.dump_dir.empty() && !sinks_ready) {
                static const char * const dump_names[] = {
                    "patches", "v_pos_in", "v_patch_out", "v_blk0_attn_out", "v_blk0_out",
                    "v_final", nullptr,
                };
                vqa::dump::mark_sinks(gctx, gf, dump_names);
                vqa::dump::finalize_sinks(gctx, gf, rc.backend);
                sinks_ready = true;
            }
            if (!ggml_gallocr_alloc_graph(rc.allocr, gf)) {
                fprintf(stderr, "vision graph allocation failed\n");
                return false;
            }
            if (t_patches->buffer)
                ggml_backend_tensor_set(t_patches, frames[s].patches, 0,
                                        sizeof(float) * (size_t)Nc * C3);
            if (t_pos->buffer)
                ggml_backend_tensor_set(t_pos, frames[s].pos, 0,
                                        sizeof(float) * (size_t)Nc * vc.n_embd);
            if (t_vpos->buffer)
                ggml_backend_tensor_set(t_vpos, vpos.data(), 0, sizeof(int) * vpos.size());
            if (ggml_backend_graph_compute(rc.backend, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "vision compute failed\n");
                return false;
            }
            if (s == 0 && !rc.dump_dir.empty()) {
                vqa::dump::write(gf, rc.dump_dir.c_str(), "patches");
                vqa::dump::write(gf, rc.dump_dir.c_str(), "v_pos_in");
                vqa::dump::write(gf, rc.dump_dir.c_str(), "v_patch_out");
                vqa::dump::write(gf, rc.dump_dir.c_str(), "v_blk0_attn_out");
                vqa::dump::write(gf, rc.dump_dir.c_str(), "v_blk0_out");
            }
            if (!rc.dump_dir.empty() && out_name)
                vqa::dump::write(gf, rc.dump_dir.c_str(), out_name);
            if (dest) {
                ggml_tensor * out = ggml_graph_get_tensor(gf, out_name);
                dest->resize((size_t)out->ne[1] * out->ne[0]);
                ggml_backend_tensor_get(out, dest->data(), 0, sizeof(float) * dest->size());
            }
            return true;
        };

        const char * stop_s = (s == 0) ? stop_after : nullptr;
        if (stop_s && !strcmp(stop_s, "v_patch_out")) {
            ggml_build_forward_expand(gf, x);
            if (!run_chunk("v_patch_out", &feats)) { ggml_free(gctx); return false; }
            *n_tokens = (int)(feats.size() / vc.n_embd);
            ggml_free(gctx);
            return true;
        }

        int vl_max = vc.n_layer;
        { const char * e = getenv(env_vl); if (e) vl_max = std::min(vl_max, atoi(e)); }
        vqa::vit::VitHookEnv env;
        env.gctx = gctx; env.gf = gf; env.t_vpos = t_vpos;
        env.n_head = vc.n_head; env.n_embd = vc.n_embd;
        env.N = Nc; env.vhd = vhd; env.vhd_f = vhd;
        env.n_ctx = 32768; env.rope_theta = vc.rope_theta;
        env.rope_stop_all_layers = true;
        env.stop_after = stop_s; env.state = 0;
        env.run = [&run_chunk](const char * name) -> bool {
            std::vector<float> sink; return run_chunk(name, &sink);
        };
        for (int il = 0; il < vl_max; il++) {
            const Layer & L = rc.w->layers[il];
            vqa::vit::BlockWeights bw{L.norm0_w, L.norm0_b, L.wqkv_w, L.wqkv_b, L.wo_w, L.wo_b,
                                      L.norm1_w, L.norm1_b, L.fc0_w, L.fc0_b, L.fc1_w, L.fc1_b};
            vqa::vit::BlockHooks bh;
            bh.attention = vqa::vit::vit_attention_hook;
            bh.probe_layer0 = vqa::vit::vit_probe_layer0_hook;
            bh.ud = &env;
            x = vqa::vit::block(gctx, x, bw, vc.layernorm_eps, bh, il);
            if (!x) { ggml_free(gctx); return env.state == 1; }

            // DeepStack tap: this block's output goes through its own merger
            // (LayerNorm over the POST-fold width 4*hidden) and is staged for
            // the text graph, which adds it after decoder layer k
            if (n_ds > 0 && rc.ds_buf) {
                for (int k = 0; k < n_ds; k++) {
                    if (vc.deepstack[k] != il) continue;
                    const Merger & M = rc.w->ds[k];
                    ggml_tensor * dn = ggml_reshape_2d(gctx, x,
                        (int64_t)vc.n_embd * vc.merge * vc.merge,
                        (int64_t)Nc / (vc.merge * vc.merge));
                    dn = ggml_norm(gctx, dn, vc.layernorm_eps);
                    dn = ggml_mul(gctx, dn, M.norm_w);
                    dn = ggml_add(gctx, dn, M.norm_b);
                    dn = ggml_mul_mat(gctx, M.fc1_w, dn);
                    dn = ggml_add(gctx, dn, M.fc1_b);
                    dn = ggml_gelu_erf(gctx, dn);
                    dn = ggml_mul_mat(gctx, M.fc2_w, dn);
                    dn = ggml_add(gctx, dn, M.fc2_b);
                    ggml_build_forward_expand(gf, rc.ds_buf[k].sink(gctx, dn, (int64_t)s * mh * mw));
                }
            }
        }
        ggml_set_name(x, "v_final");
        ggml_tensor * out_sink = rc.tower_buf->sink(gctx, x, (int64_t)s * hw);
        ggml_build_forward_expand(gf, out_sink);
        if (stop_s && !strcmp(stop_s, "v_final")) {
            if (!run_chunk("v_final", &feats)) { ggml_free(gctx); return false; }
            *n_tokens = (int)(feats.size() / vc.n_embd);
            ggml_free(gctx);
            return true;
        }
        if (!run_chunk("v_final", nullptr)) { ggml_free(gctx); return false; }
        ggml_free(gctx);
    }

    // ---- merger: LayerNorm(hidden) per patch -> 2x2 fold -> fc1 -> GELU -> fc2
    const int N = hw * T;
    {
        if (!rc.dump_dir.empty()) vqa::dump::dump_registry_clear();
        struct ggml_init_params ip2 = { 32ull * 1024 * 1024, NULL, true };
        ggml_context * mctx = ggml_init(ip2);
        if (!mctx) return false;
        struct ggml_cgraph * mgf = ggml_new_graph_custom(mctx, 8192, false);
        ggml_tensor * x = rc.tower_buf->view(mctx, N, 0);

        bool sinks_ready = false;
        auto run_merger = [&](const char * out_name) -> bool {
            if (!rc.dump_dir.empty() && !sinks_ready) {
                static const char * const dump_names[] = { "v_merge", "v_proj_out", nullptr };
                vqa::dump::mark_sinks(mctx, mgf, dump_names);
                vqa::dump::finalize_sinks(mctx, mgf, rc.backend);
                sinks_ready = true;
            }
            if (!ggml_gallocr_alloc_graph(rc.allocr, mgf)) {
                fprintf(stderr, "merger graph allocation failed\n");
                return false;
            }
            if (ggml_backend_graph_compute(rc.backend, mgf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "merger compute failed\n");
                return false;
            }
            if (!rc.dump_dir.empty()) {
                vqa::dump::write(mgf, rc.dump_dir.c_str(), "v_merge");
                vqa::dump::write(mgf, rc.dump_dir.c_str(), "v_proj_out");
            }
            ggml_tensor * out = ggml_graph_get_tensor(mgf, out_name);
            const int n = (int)out->ne[1], d = (int)out->ne[0];
            feats.resize((size_t)n * d);
            ggml_backend_tensor_get(out, feats.data(), 0, sizeof(float) * feats.size());
            *n_tokens = n;
            return true;
        };

        ggml_tensor * xn = ggml_norm(mctx, x, vc.layernorm_eps);
        xn = ggml_mul(mctx, xn, rc.w->main.norm_w);
        xn = ggml_add(mctx, xn, rc.w->main.norm_b);
        ggml_tensor * x4 = ggml_reshape_2d(mctx, xn, (int64_t)vc.n_embd * vc.merge * vc.merge,
                                           (int64_t)mh * mw * T);
        ggml_set_name(x4, "v_merge");
        if (stop_after && !strcmp(stop_after, "v_merge")) {
            ggml_build_forward_expand(mgf, x4);
            if (!run_merger("v_merge")) { ggml_free(mctx); return false; }
            ggml_free(mctx);
            *n_tokens = mh * mw * T;
            return true;
        }
        ggml_tensor * h = ggml_mul_mat(mctx, rc.w->main.fc1_w, x4);
        h = ggml_add(mctx, h, rc.w->main.fc1_b);
        h = ggml_gelu_erf(mctx, h);
        h = ggml_mul_mat(mctx, rc.w->main.fc2_w, h);
        h = ggml_add(mctx, h, rc.w->main.fc2_b);
        ggml_set_name(h, "v_proj_out");
        ggml_build_forward_expand(mgf, h);
        if (!run_merger("v_proj_out")) { ggml_free(mctx); return false; }
        ggml_free(mctx);
    }
    (void)H; (void)W;
    return true;
}

}  // namespace qwen3vl_tower

// mrope position ids for the text prompt (get_rope_index): text runs share
// (p, p, p); every vision block walks its merged grid as (t = cur, h, w) and
// the running counter then advances by max(mh, mw). The fourth plane carries
// the absolute KV slot. Shared by every backend that feeds this tower's
// merged tokens into an interleaved-MRoPE Qwen3 text trunk.
inline void build_mpos_qwen3vl(const std::vector<int> & ids, const std::vector<int> & img_pos,
                               int n_img, int mh, int mw, std::vector<int> & mpos) {
    mpos.clear();
    mpos.reserve((size_t)4 * ids.size());
    std::vector<int> axis[3];
    const int per = mh * mw;
    const int n_blk = (per > 0 && n_img > 0) ? n_img / per : 0;
    if (n_img > 0 && !img_pos.empty()) {
        const size_t v0 = (size_t)img_pos[0];
        int cur = 0;
        for (size_t i = 0; i < v0; i++) {
            axis[0].push_back(cur); axis[1].push_back(cur); axis[2].push_back(cur);
            cur++;
        }
        for (int b = 0; b < n_blk; b++) {
            for (int k = 0; k < per; k++) {
                axis[0].push_back(cur);
                axis[1].push_back(cur + k / mw);
                axis[2].push_back(cur + k % mw);
            }
            cur += std::max(mh, mw);
        }
        for (size_t i = v0 + (size_t)n_img; i < ids.size(); i++) {
            axis[0].push_back(cur); axis[1].push_back(cur); axis[2].push_back(cur);
            cur++;
        }
    } else {
        for (size_t i = 0; i < ids.size(); i++) {
            axis[0].push_back((int)i); axis[1].push_back((int)i); axis[2].push_back((int)i);
        }
    }
    mpos.insert(mpos.end(), axis[0].begin(), axis[0].end());
    mpos.insert(mpos.end(), axis[1].begin(), axis[1].end());
    mpos.insert(mpos.end(), axis[2].begin(), axis[2].end());
    for (int i = 0; i < (int)ids.size(); i++) mpos.push_back(i);    // absolute KV slots
}

}  // namespace vqa
