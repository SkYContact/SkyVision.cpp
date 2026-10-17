// qwen2vl_model.cpp - Qwen2.5-VL backend (Qwen2 text trunk + Qwen2.5 ViT tower).
//
// Graph layout follows llama.cpp's proven implementation:
//   * the patch embedding is Conv3d(3, n_embd, temporal=2, 14, 14) with the two
//     temporal taps summed - folded into one matmul here (a still image repeats
//     the same pixels, so the fold is exact);
//   * the vision tower is RMSNorm (qwen2.5 uses RMS, qwen2 does LayerNorm) +
//     fused qkv with 2D M-RoPE (4 sections of head_dim/4, positions h,w,h,w) +
//     SwiGLU MLP, all with biases;
//   * window attention: patch tokens are reordered so each 112px window (4x4
//     merged tokens = 8x8 patches) is contiguous, then a block-diagonal mask is
//     applied; every 8th layer (fullatt_block_indexes) runs global instead;
//   * the merger reshapes each 2x2 group into one 4*n_embd row (the reorder
//     already put them adjacent), so no extra permute is needed, then RMS norm +
//     2-layer MLP with GELU; a final gather undoes the window reorder;
//   * the text trunk is plain Qwen2: RMSNorm, GQA, qkv bias, SwiGLU, and
//     section MRoPE (16/24/24 of head_dim/2) over 3 position axes.
#include "vqa.hpp"

#include "attention.hpp"
#include "backend.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "ggml_dump.hpp"
#include "gguf.h"
#include "gguf_util.hpp"
#include "load_util.hpp"
#include "resample.hpp"
#include "sampler.hpp"
#include "weight_load.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace qwen2vl {

struct TxtLayer {
    ggml_tensor *norm = nullptr;
    ggml_tensor *wq = nullptr, *bq = nullptr;
    ggml_tensor *wk = nullptr, *bk = nullptr;
    ggml_tensor *wv = nullptr, *bv = nullptr;
    ggml_tensor *wo = nullptr;
    ggml_tensor *ffn_norm = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
};

struct VisLayer {
    ggml_tensor *ln1 = nullptr, *ln2 = nullptr;
    ggml_tensor *wqkv = nullptr, *bqkv = nullptr;
    ggml_tensor *wo = nullptr, *bo = nullptr;
    ggml_tensor *wg = nullptr, *bg = nullptr;
    ggml_tensor *wu = nullptr, *bu = nullptr;
    ggml_tensor *wd = nullptr, *bd = nullptr;
};

// one text graph: prefill (T tokens) or one decode step (T=1)
struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *pos_in = nullptr, *slot_in = nullptr, *mask_in = nullptr;
    ggml_tensor *img_in = nullptr, *img_pos_in = nullptr;
    ggml_tensor *logits = nullptr;
};

class Qwen25VL : public vqa::Model {
public:
    ~Qwen25VL() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "qwen2.5-vl"; }
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp, std::string & out_text) override;

    // tunables (set before load)
    int n_ctx = 4096;
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool alloc_kv();

    // vision: one graph per (gh, gw); returns merged features [n_embd, nm] in
    // raster order plus the grid
    bool encode_vision(const vqa::MediaInput & media, std::vector<float> & feats, int * n_tokens,
                       int grid[3]);
    bool build_vision_graph(int gh, int gw);
    bool run_vision(const std::vector<float> & patches, std::vector<float> & feats);

    bool build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all);
    bool run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<int> & pos4,
                  const std::vector<float> * img_feats, const std::vector<int> * img_pos,
                  std::vector<float> & logits, int slot_base = 0);

    void dump(const ggml_cgraph * gf, const char * name);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    // text
    int n_layer = 36, n_embd = 2048, n_head = 16, n_kv_head = 2, head_dim = 128;
    int head_dim_rope = 128, n_ff = 11008, n_vocab = 151936;
    float rms_eps = 1e-6f, rope_theta = 1e6f;
    int mrope_section[3] = {16, 24, 24};

    // vision
    int v_layer = 32, v_embd = 1280, v_head = 16, v_hd = 80, v_ff = 3420;
    int v_patch = 14, v_temporal = 2, v_merge = 2, v_out = 2048;
    int v_window = 112, n_wa_pattern = 8;
    float v_eps = 1e-6f, v_rope_theta = 10000.0f;
    float v_mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };   // CLIP stats (processor config)
    float v_std[3] = { 0.26862954f, 0.26130258f, 0.27577711f };
    int n_patches_x = 0, n_patches_y = 0;      // current vision graph grid

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    vqa::MMap mm;
    vqa::AlignedBuf host_f32;
    ggml_gallocr_t allocr = nullptr;

    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr;
    std::vector<TxtLayer> tl;

    ggml_tensor *v_patch_w = nullptr;
    ggml_tensor *v_merger_ln = nullptr, *v_mm1_w = nullptr, *v_mm1_b = nullptr;
    ggml_tensor *v_mm2_w = nullptr, *v_mm2_b = nullptr;
    std::vector<VisLayer> vl;

    // vision graph (rebuilt per image)
    ggml_context * vctx = nullptr;
    ggml_cgraph * vgf = nullptr;
    ggml_tensor *v_in = nullptr, *v_pos = nullptr, *v_mask = nullptr;
    ggml_tensor *v_idx_in = nullptr, *v_idx_out = nullptr;
    ggml_tensor *v_feats = nullptr;

    vqa::KvCache kv;
    int n_past = 0;
    int next_pos = 0;      // MRoPE index for the next decoded token

    int id_image_pad = 151655, id_video_pad = 151656;
    int id_vision_start = 151652, id_vision_end = 151653;
    int id_im_start = 151644, id_im_end = 151645, id_eos = 151645;

    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

// ---------------------------------------------------------------- lifecycle

Qwen25VL::~Qwen25VL() {
    kv.free();
    if (vctx) ggml_free(vctx);
    if (allocr) ggml_gallocr_free(allocr);
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (backend) ggml_backend_free(backend);
}

bool Qwen25VL::init_backend() {
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

bool Qwen25VL::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer   = vqa::gguf::u32(gu, "qwen2vl.block_count", n_layer);
    n_embd    = vqa::gguf::u32(gu, "qwen2vl.embedding_length", n_embd);
    n_head    = vqa::gguf::u32(gu, "qwen2vl.attention.head_count", n_head);
    n_kv_head = vqa::gguf::u32(gu, "qwen2vl.attention.head_count_kv", n_kv_head);
    head_dim  = vqa::gguf::u32(gu, "qwen2vl.attention.key_length", head_dim);
    n_ff      = vqa::gguf::u32(gu, "qwen2vl.feed_forward_length", n_ff);
    rms_eps   = vqa::gguf::f32(gu, "qwen2vl.attention.layer_norm_rms_epsilon", rms_eps);
    rope_theta = vqa::gguf::f32(gu, "qwen2vl.rope.freq_base", rope_theta);
    {
        const std::vector<int> sec = vqa::gguf::arr_u32(gu, "qwen2vl.rope.mrope_section",
                                                       {16, 24, 24});
        for (int i = 0; i < 3 && i < (int)sec.size(); i++) mrope_section[i] = sec[(size_t)i];
    }
    id_image_pad   = vqa::gguf::u32(gu, "qwen2vl.image_token_id", id_image_pad);
    id_vision_start = vqa::gguf::u32(gu, "qwen2vl.vision_start_id", id_vision_start);
    id_vision_end  = vqa::gguf::u32(gu, "qwen2vl.vision_end_id", id_vision_end);
    id_video_pad   = vqa::gguf::u32(gu, "qwen2vl.video_token_id", id_video_pad);

    v_layer   = vqa::gguf::u32(gu, "qwen2vl.vision.block_count", v_layer);
    v_embd    = vqa::gguf::u32(gu, "qwen2vl.vision.embedding_length", v_embd);
    v_ff      = vqa::gguf::u32(gu, "qwen2vl.vision.feed_forward_length", v_ff);
    v_head    = vqa::gguf::u32(gu, "qwen2vl.vision.attention.head_count", v_head);
    v_patch   = vqa::gguf::u32(gu, "qwen2vl.vision.patch_size", v_patch);
    v_temporal = vqa::gguf::u32(gu, "qwen2vl.vision.temporal_patch_size", v_temporal);
    v_merge   = vqa::gguf::u32(gu, "qwen2vl.vision.spatial_merge_size", v_merge);
    v_out     = vqa::gguf::u32(gu, "qwen2vl.vision.out_hidden_size", v_out);
    v_window  = vqa::gguf::u32(gu, "qwen2vl.vision.window_size", v_window);
    v_eps     = vqa::gguf::f32(gu, "qwen2vl.vision.layer_norm_epsilon", v_eps);
    {
        int64_t id = gguf_find_key(gu, "qwen2vl.vision.image_mean");
        if (id >= 0 && gguf_get_arr_n(gu, id) == 3) {
            const float * a = (const float *)gguf_get_arr_data(gu, id);
            for (int c = 0; c < 3; c++) v_mean[c] = a[c];
        }
        id = gguf_find_key(gu, "qwen2vl.vision.image_std");
        if (id >= 0 && gguf_get_arr_n(gu, id) == 3) {
            const float * a = (const float *)gguf_get_arr_data(gu, id);
            for (int c = 0; c < 3; c++) v_std[c] = a[c];
        }
    }
    {
        const std::vector<int> fb = vqa::gguf::arr_u32(gu, "qwen2vl.vision.fullatt_blocks", {7});
        n_wa_pattern = (fb.empty() ? 8 : fb[0] + 1);
    }
    v_hd = v_embd / (v_head > 0 ? v_head : 1);

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);

    vqa::weights::promote_1d_to_f32(wctx, host_f32);
    {
        vqa::weights::Bound bound = vqa::weights::bind_to_backend(
            wctx, backend, mm, { { host_f32.data(), host_f32.size() } });
        if (!bound.ok) return false;
        wbuf = bound.main_buf;
        if (!ggml_backend_is_cpu(backend)) mm.release();
    }

    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) fprintf(stderr, "missing tensor: %s\n", n);
        return t;
    };
    tok_embd = gt("t.embed.weight");
    out_norm = gt("t.norm.weight");
    tl.resize((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "t.blk.%d.%s", i, suf); return gt(b); };
        TxtLayer & L = tl[(size_t)i];
        L.norm = g("attn_norm.weight");
        L.wq = g("wq.weight"); L.bq = g("bq");
        L.wk = g("wk.weight"); L.bk = g("bk");
        L.wv = g("wv.weight"); L.bv = g("bv");
        L.wo = g("wo.weight");
        L.ffn_norm = g("ffn_norm.weight");
        L.gate = g("w_gate.weight"); L.up = g("w_up.weight"); L.down = g("w_down.weight");
        if (!L.norm || !L.wq || !L.wk || !L.wv || !L.wo || !L.ffn_norm || !L.gate || !L.up || !L.down)
            return false;
    }
    v_patch_w = gt("v.patch_embed.weight");
    v_merger_ln = gt("v.merger.ln.weight");
    v_mm1_w = gt("v.merger.fc1.weight"); v_mm1_b = gt("v.merger.fc1.bias");
    v_mm2_w = gt("v.merger.fc2.weight"); v_mm2_b = gt("v.merger.fc2.bias");
    vl.resize((size_t)v_layer);
    for (int i = 0; i < v_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "v.blk.%d.%s", i, suf); return gt(b); };
        VisLayer & L = vl[(size_t)i];
        L.ln1 = g("norm1.weight"); L.ln2 = g("norm2.weight");
        L.wqkv = g("wqkv.weight"); L.bqkv = g("wqkv.bias");
        L.wo = g("wo.weight"); L.bo = g("wo.bias");
        L.wg = g("w_gate.weight"); L.bg = g("w_gate.bias");
        L.wu = g("w_up.weight"); L.bu = g("w_up.bias");
        L.wd = g("w_down.weight"); L.bd = g("w_down.bias");
        if (!L.ln1 || !L.ln2 || !L.wqkv || !L.wo || !L.wg || !L.wu || !L.wd) return false;
    }
    fprintf(stderr, "qwen2.5-vl: %d text layers, %d vision layers, %d patches/clip max\n",
            n_layer, v_layer, 0);
    return tok_embd && out_norm;
}

bool Qwen25VL::alloc_kv() {
    if (!kv.alloc(backend, n_layer, n_kv_head, head_dim, n_ctx)) return false;
    return true;
}

bool Qwen25VL::load(const std::string & gguf_path, const std::string & tokenizer_path,
                    const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "QWEN2VL", n_threads, use_gpu, dump_dir);
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::Qwen2Regex)) return false;
    if (!alloc_kv()) return false;

    vqa::assign_token_ids(tok_, {
        { "<|image_pad|>",    &id_image_pad },
        { "<|video_pad|>",    &id_video_pad },
        { "<|vision_start|>", &id_vision_start },
        { "<|vision_end|>",   &id_vision_end },
        { "<|im_start|>",     &id_im_start },
        { "<|im_end|>",       &id_im_end },
    });
    id_eos = id_im_end;
    caps_.image = true;
    caps_.video = false;
    fprintf(stderr, "qwen2.5-vl: ready (%dL d=%d nh=%d nkv=%d hd=%d vocab=%d)\n",
            n_layer, n_embd, n_head, n_kv_head, head_dim, n_vocab);
    return true;
}

// ---------------------------------------------------------------- vision

bool Qwen25VL::build_vision_graph(int gh, int gw) {
    if (vctx) { ggml_free(vctx); vctx = nullptr; }
    const int N = gh * gw;
    const int nm = N / (v_merge * v_merge);
    const int C3 = 3 * v_temporal * v_patch * v_patch;

    struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    vctx = ggml_init(ip);
    if (!vctx) return false;
    vgf = ggml_new_graph_custom(vctx, 65536, false);

    v_in = ggml_new_tensor_2d(vctx, GGML_TYPE_F32, C3, N);
    ggml_set_name(v_in, "v_patches");
    v_pos = ggml_new_tensor_1d(vctx, GGML_TYPE_I32, 4 * N);
    ggml_set_name(v_pos, "v_pos");
    const bool wa = n_wa_pattern > 0;
    if (wa) {
        v_mask = ggml_new_tensor_2d(vctx, GGML_TYPE_F16, N, N);
        ggml_set_name(v_mask, "v_mask");
        v_idx_in = ggml_new_tensor_1d(vctx, GGML_TYPE_I32, nm);
        ggml_set_name(v_idx_in, "v_idx_in");
        v_idx_out = ggml_new_tensor_1d(vctx, GGML_TYPE_I32, nm);
        ggml_set_name(v_idx_out, "v_idx_out");
    }

    // window-major reorder, applied to the 2x2 groups (4 patch rows at a time)
    ggml_tensor * x = v_in;
    if (wa) {
        x = ggml_reshape_2d(vctx, x, C3 * 4, nm);
        x = ggml_get_rows(vctx, x, v_idx_in);
        x = ggml_reshape_2d(vctx, x, C3, N);
    }
    x = ggml_mul_mat(vctx, v_patch_w, x);               // [v_embd, N]
    ggml_set_name(x, "v_patch_out");

    // 4D M-RoPE sections: head_dim/4 each, positions (h, w, h, w)
    int vsec[4] = { v_hd / 4, v_hd / 4, v_hd / 4, v_hd / 4 };

    int v_max = v_layer;   // Q2V_MAX_VLAYERS=N truncates the tower for bisecting
    if (const char * e = getenv("Q2V_MAX_VLAYERS")) v_max = std::min(v_layer, atoi(e));
    for (int il = 0; il < v_max; il++) {
        VisLayer & L = vl[(size_t)il];
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

        q3 = ggml_rope_multi(vctx, q3, v_pos, nullptr, v_hd / 2, vsec,
                             GGML_ROPE_TYPE_VISION, 32768, v_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_multi(vctx, k3, v_pos, nullptr, v_hd / 2, vsec,
                             GGML_ROPE_TYPE_VISION, 32768, v_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (il == 0) ggml_set_name(q3, "v_blk0_q_rope");

        ggml_tensor * qf = ggml_cont(vctx, ggml_permute(vctx, q3, 0, 2, 1, 3));   // [hd, N, head]
        ggml_tensor * kf = ggml_cont(vctx, ggml_permute(vctx, k3, 0, 2, 1, 3));
        ggml_tensor * vf = ggml_cont(vctx, ggml_permute(vctx, v3, 0, 2, 1, 3));
        ggml_tensor * mask = full_attn ? nullptr : v_mask;
        ggml_tensor * fa = ggml_flash_attn_ext(vctx, qf, kf, vf, mask,
                                               1.0f / sqrtf((float)v_hd), 0.0f, 0.0f);
        // FA returns [hd, heads, tokens]: flatten straight to [v_embd, N]
        ggml_tensor * attn = ggml_reshape_2d(vctx, ggml_cont(vctx, fa), v_embd, N);
        attn = ggml_mul_mat(vctx, L.wo, attn);
        attn = ggml_add(vctx, attn, L.bo);
        x = ggml_add(vctx, x, attn);
        if (il == 0) ggml_set_name(x, "v_blk0_attn_out");

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
    }
    ggml_set_name(x, "v_final");

    // merger: HF normalizes per token over context_dim (1280) and only then folds
    // the 2x2 group into one 4*1280 row - doing the RMSNorm after the reshape
    // normalizes over 5120 and destroys the features
    ggml_tensor * xn = ggml_rms_norm(vctx, x, v_eps);
    xn = ggml_mul(vctx, xn, v_merger_ln);
    ggml_tensor * m = ggml_reshape_2d(vctx, xn, v_embd * v_merge * v_merge, nm);
    ggml_set_name(m, "v_merge");
    m = ggml_mul_mat(vctx, v_mm1_w, m);
    m = ggml_add(vctx, m, v_mm1_b);
    m = ggml_gelu_erf(vctx, m);
    m = ggml_mul_mat(vctx, v_mm2_w, m);
    m = ggml_add(vctx, m, v_mm2_b);
    ggml_set_name(m, "v_proj_out");
    if (wa) m = ggml_get_rows(vctx, m, v_idx_out);   // window-major -> raster
    ggml_set_name(m, "v_feats");
    v_feats = m;
    ggml_build_forward_expand(vgf, m);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "v_patch_out", "v_blk0_qkv", "v_blk0_attn_out",
                                           "v_blk0_out", "v_merge", "v_proj_out", nullptr };
        vqa::dump::mark_sinks(vctx, vgf, dn);
        vqa::dump::finalize_sinks(vctx, vgf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, vgf)) {
        fprintf(stderr, "vision graph allocation failed\n");
        return false;
    }
    n_patches_x = gw;
    n_patches_y = gh;
    return true;
}

bool Qwen25VL::run_vision(const std::vector<float> & patches, std::vector<float> & feats) {
    const int N = n_patches_x * n_patches_y;
    const int nm = N / (v_merge * v_merge);

    ggml_backend_tensor_set(v_in, patches.data(), 0, sizeof(float) * patches.size());

    // positions: 4 arrays (h, w, h, w) in *window-major* patch order
    {
        std::vector<int> pos((size_t)4 * N, 0);
        std::vector<int> idx((size_t)nm), inv((size_t)nm);
        const int gw = n_patches_x, gh = n_patches_y;
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
        ggml_backend_tensor_set(v_pos, pos.data(), 0, sizeof(int) * pos.size());
        if (n_wa_pattern > 0) {
            ggml_backend_tensor_set(v_idx_in, inv.data(), 0, sizeof(int) * inv.size());
            ggml_backend_tensor_set(v_idx_out, idx.data(), 0, sizeof(int) * idx.size());
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
            ggml_backend_tensor_set(v_mask, mask.data(), 0, sizeof(ggml_fp16_t) * mask.size());
        }
    }

    if (ggml_backend_graph_compute(backend, vgf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vision compute failed\n");
        return false;
    }
    if (!dump_dir.empty()) {
        static const char * const dn[] = { "v_patches", "v_patch_out", "v_blk0_qkv",
                                           "v_blk0_attn_out", "v_blk0_out", "v_blk15_out", "v_final", "v_merge",
                                           "v_proj_out", "v_feats", nullptr };
        for (int i = 0; dn[i]; i++) dump(vgf, dn[i]);
    }
    const int n = (int)v_feats->ne[1], d = (int)v_feats->ne[0];
    feats.resize((size_t)n * d);
    ggml_backend_tensor_get(v_feats, feats.data(), 0, sizeof(float) * feats.size());
    (void)nm;
    return true;
}

bool Qwen25VL::encode_vision(const vqa::MediaInput & media, std::vector<float> & feats,
                             int * n_tokens, int grid[3]) {
    if (media.empty()) return false;
    // smart_resize: multiples of patch*merge, total pixels within the official bounds
    const int factor = v_patch * v_merge;
    const int min_px = 4 * factor * factor;               // 4*28*28
    const int max_px = 16384 * factor * factor / 4;       // 4096 merged tokens cap
    int rh = 0, rw = 0;
    if (!vqa::img::smart_resize(media.height, media.width, factor, min_px, max_px, &rh, &rw)) {
        fprintf(stderr, "image too small for the vision tower\n");
        return false;
    }
    const int gh = rh / v_patch, gw = rw / v_patch;

    // patches (temporal tap duplicated, like the official processor)
    std::vector<float> resized;
    vqa::img::resize_antialias(media.frames[0].rgb.data(), media.width, media.height, rw, rh,
                               resized, /*bilinear*/ false);
    if (getenv("Q2V_DUMP_CACHE")) {   // raw resized RGB, for preprocessing debugging
        FILE * f = fopen((dump_dir + "/dbg_resized.bin").c_str(), "wb");
        if (f) { fwrite(resized.data(), sizeof(float), resized.size(), f); fclose(f); }
    }
    const int N = gh * gw, P = v_patch;
    const int C3 = 3 * v_temporal * P * P;
    std::vector<float> patches((size_t)C3 * N, 0.0f);
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
                                    const float * src = resized.data() +
                                        ((size_t)(py * P + dh) * rw + px * P) * 3 + c;
                                    for (int dw = 0; dw < P; dw++)
                                        row[o++] = (src[(size_t)dw * 3] / 255.0f - v_mean[c]) /
                                                   v_std[c];
                                }
                        u++;
                    }
    }

    // Q2V_FEATS=<file.npy.raw> + <file.shape>: load features from disk instead of
    // running the tower (parity debugging: isolates tower errors from the text side)
    if (const char * fp = getenv("Q2V_FEATS")) {
        const std::string shape_path = std::string(fp) + ".shape";
        FILE * sf = fopen(shape_path.c_str(), "rb");
        FILE * ff = fopen(fp, "rb");
        if (!sf || !ff) { fprintf(stderr, "Q2V_FEATS: cannot open %s\n", fp); return false; }
        long rows = 0, cols = 0;
        if (fscanf(sf, "%ld %ld", &rows, &cols) != 2) { fclose(sf); fclose(ff); return false; }
        fclose(sf);
        feats.resize((size_t)rows * cols);
        size_t got = fread(feats.data(), sizeof(float), feats.size(), ff);
        fclose(ff);
        if (got != feats.size()) { fprintf(stderr, "Q2V_FEATS: short read\n"); return false; }
        *n_tokens = (int)rows;
        grid[0] = 1;
        grid[1] = gh / v_merge;
        grid[2] = gw / v_merge;
        fprintf(stderr, "vision: %ld tokens from %s (tower skipped)\n", rows, fp);
        return true;
    }
    if (!build_vision_graph(gh, gw)) return false;
    if (!run_vision(patches, feats)) return false;
    *n_tokens = (int)feats.size() / v_out;
    grid[0] = 1;
    grid[1] = gh / v_merge;
    grid[2] = gw / v_merge;
    fprintf(stderr, "vision: %dx%d -> %d patches -> %d tokens\n", rw, rh, N, *n_tokens);
    return true;
}

// ---------------------------------------------------------------- text

bool Qwen25VL::build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all) {
    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
    struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 32768, false);

    const int D = n_embd, HD = head_dim, NH = n_head, NKV = n_kv_head;

    g.tok_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    // MRoPE positions: [t(T), h(T), w(T), unused(T)] — ggml reads 4 arrays
    g.pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, 4 * T);
    ggml_set_name(g.pos_in, "pos_in");
    // KV cache slots are absolute sequence positions; the MRoPE t axis is NOT one
    // (every image token shares t = st), so they must be separate inputs
    g.slot_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.slot_in, "slot_in");
    g.mask_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, S, T);
    ggml_set_name(g.mask_in, "mask_in");

    ggml_tensor * x = ggml_get_rows(g.ctx, tok_embd, g.tok_in);   // [D, T]
    if (n_img > 0) {
        g.img_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, n_img);
        ggml_set_name(g.img_in, "img_in");
        g.img_pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_img);
        ggml_set_name(g.img_pos_in, "img_pos_in");
        // ggml_set_rows writes into its destination, so it needs its own copy
        x = ggml_set_rows(g.ctx, ggml_dup(g.ctx, x), g.img_in, g.img_pos_in);
    }
    ggml_set_name(x, "t_emb");
    ggml_tensor * img = nullptr;

    int sec[4] = { mrope_section[0], mrope_section[1], mrope_section[2], 0 };
    for (int il = 0; il < n_layer; il++) {
        TxtLayer & L = tl[(size_t)il];
        ggml_tensor * l = ggml_rms_norm(g.ctx, x, rms_eps);
        l = ggml_mul(g.ctx, l, L.norm);
        if (il == 0) ggml_set_name(l, "t_l0_norm");

        ggml_tensor * q = ggml_mul_mat(g.ctx, L.wq, l);
        q = ggml_add(g.ctx, q, L.bq);
        ggml_tensor * k = ggml_mul_mat(g.ctx, L.wk, l);
        k = ggml_add(g.ctx, k, L.bk);
        ggml_tensor * v = ggml_mul_mat(g.ctx, L.wv, l);
        v = ggml_add(g.ctx, v, L.bv);
        if (il == 0) { ggml_set_name(q, "t_l0_q"); ggml_set_name(k, "t_l0_k"); }

        ggml_tensor * q3 = ggml_reshape_3d(g.ctx, q, HD, NH, T);
        ggml_tensor * k3 = ggml_reshape_3d(g.ctx, k, HD, NKV, T);
        q3 = ggml_rope_multi(g.ctx, q3, g.pos_in, nullptr, HD, sec, GGML_ROPE_TYPE_MROPE,
                             n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_multi(g.ctx, k3, g.pos_in, nullptr, HD, sec, GGML_ROPE_TYPE_MROPE,
                             n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (il == 0) {
            ggml_set_name(q3, "t_l0_q_rope");
            ggml_set_name(k3, "t_l0_k_rope");
            ggml_set_name(v, "t_l0_v");
        }

        // the cache is F16, so cast explicitly: ggml_set_rows copies element-wise
        // and a mixed dtype silently writes garbage
        ggml_tensor * k2 = ggml_cast(g.ctx, ggml_reshape_2d(g.ctx, k3, NKV * HD, T),
                                     GGML_TYPE_F16);
        ggml_tensor * v2 = ggml_cast(g.ctx, ggml_reshape_2d(g.ctx, v, NKV * HD, T),
                                     GGML_TYPE_F16);
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.k[il], k2, g.slot_in));
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.v[il], v2, g.slot_in));

        ggml_tensor * qf = ggml_cont(g.ctx, ggml_permute(g.ctx, q3, 0, 2, 1, 3));   // [HD, T, NH]
        ggml_tensor * kf = ggml_view_3d(g.ctx, kv.k[il], HD, S, NKV,
                                        kv.k[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * vf = ggml_view_3d(g.ctx, kv.v[il], HD, S, NKV,
                                        kv.v[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * fa = ggml_flash_attn_ext(g.ctx, qf, kf, vf, g.mask_in,
                                               1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        // FA already returns [hd, heads, tokens] - no permute, just flatten per token
        ggml_tensor * attn = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, fa), NH * HD, T);
        if (il == 0) ggml_set_name(attn, "t_l0_attn");
        attn = ggml_mul_mat(g.ctx, L.wo, attn);
        x = ggml_add(g.ctx, x, attn);
        if (il == 0) ggml_set_name(x, "t_l0_resid");

        ggml_tensor * m = ggml_rms_norm(g.ctx, x, rms_eps);
        m = ggml_mul(g.ctx, m, L.ffn_norm);
        ggml_tensor * gate = ggml_mul_mat(g.ctx, L.gate, m);
        ggml_tensor * up = ggml_mul_mat(g.ctx, L.up, m);
        m = ggml_swiglu_split(g.ctx, gate, up);
        m = ggml_mul_mat(g.ctx, L.down, m);
        x = ggml_add(g.ctx, x, m);
        if (il == 0) ggml_set_name(x, "t_l0_out");
    }
    (void)img;

    x = ggml_rms_norm(g.ctx, x, rms_eps);
    x = ggml_mul(g.ctx, x, out_norm);
    ggml_set_name(x, "t_final_norm");

    if (logits_all) {
        g.logits = ggml_mul_mat(g.ctx, tok_embd, x);
    } else {
        ggml_tensor * last = ggml_view_2d(g.ctx, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
        g.logits = ggml_mul_mat(g.ctx, tok_embd, last);
    }
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "t_l0_norm", "t_l0_q_rope", "t_l0_k_rope", "t_l0_v",
                                           "t_l0_attn", "t_l0_out", "t_final_norm", nullptr };
        vqa::dump::mark_sinks(g.ctx, g.gf, dn);
        vqa::dump::finalize_sinks(g.ctx, g.gf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, g.gf)) {
        fprintf(stderr, "text graph allocation failed\n");
        return false;
    }
    return true;
}

bool Qwen25VL::run_text(TextGraph & g, const std::vector<int> & toks,
                        const std::vector<int> & pos4, const std::vector<float> * img_feats,
                        const std::vector<int> * img_pos, std::vector<float> & logits,
                        int slot_base) {
    const int T = (int)toks.size();
    ggml_backend_tensor_set(g.tok_in, toks.data(), 0, sizeof(int) * toks.size());
    ggml_backend_tensor_set(g.pos_in, pos4.data(), 0, sizeof(int) * pos4.size());
    {
        std::vector<int> slots((size_t)T);
        for (int i = 0; i < T; i++) slots[(size_t)i] = slot_base + i;
        ggml_backend_tensor_set(g.slot_in, slots.data(), 0, sizeof(int) * slots.size());
    }
    if (g.img_in && img_feats && img_pos) {
        ggml_backend_tensor_set(g.img_in, img_feats->data(), 0,
                                sizeof(float) * img_feats->size());
        ggml_backend_tensor_set(g.img_pos_in, img_pos->data(), 0, sizeof(int) * img_pos->size());
    }
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "text compute failed\n");
        return false;
    }
    if (!dump_dir.empty()) {
        static const char * const dn[] = { "t_emb", "t_l0_norm", "t_l0_q_rope", "t_l0_k_rope", "t_l0_v", "t_l0_attn", "t_l0_out",
                                           "t_final_norm", nullptr };
        for (int i = 0; dn[i]; i++) dump(g.gf, dn[i]);
        dump(g.gf, "logits");
        if (getenv("Q2V_DUMP_CACHE")) {   // raw layer-0 K cache (F16), first T slots
            const size_t row = sizeof(ggml_fp16_t) * (size_t)n_kv_head * head_dim;
            std::vector<ggml_fp16_t> c((size_t)T * n_kv_head * head_dim);
            for (int t = 0; t < T; t++)
                ggml_backend_tensor_get(kv.k[0], c.data() + (size_t)t * n_kv_head * head_dim,
                                        (size_t)t * kv.k[0]->nb[1], row);
            FILE * f = fopen((dump_dir + "/cache_k0.bin").c_str(), "wb");
            if (f) { fwrite(c.data(), sizeof(ggml_fp16_t), c.size(), f); fclose(f); }
        }
    }
    logits.resize((size_t)n_vocab);
    ggml_backend_tensor_get(g.logits, logits.data(), 0, sizeof(float) * logits.size());
    return true;
}

void Qwen25VL::dump(const ggml_cgraph * gf, const char * name) {
    if (dump_dir.empty()) return;
    vqa::dump::write((ggml_cgraph *)gf, dump_dir.c_str(), name);
}

// ---------------------------------------------------------------- ask

bool Qwen25VL::ask(const vqa::MediaInput & media, const std::string & prompt,
                   const std::string & /*mode*/, const vqa::GenParams & gp,
                   std::string & out_text) {
    out_text.clear();
    std::vector<float> feats;
    int n_img = 0;
    int grid[3] = { 0, 0, 0 };
    if (!media.empty()) {
        if (gp.on_stage) gp.on_stage("vision", 0, 1, gp.on_stage_ud);
        if (!encode_vision(media, feats, &n_img, grid)) return false;
        if (gp.on_stage) gp.on_stage("vision", 1, 1, gp.on_stage_ud);
    }

    std::string text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
    if (n_img > 0) {
        text += "<|vision_start|>";
        for (int i = 0; i < n_img; i++) text += "<|image_pad|>";
        text += "<|vision_end|>";
    }
    text += prompt;
    text += "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<int> toks = tok_.encode(text);
    if ((int)toks.size() + gp.max_tokens + 8 > n_ctx) {
        fprintf(stderr, "prompt %zu + %d tokens exceed n_ctx %d\n", toks.size(), gp.max_tokens, n_ctx);
        return false;
    }
    const int T = (int)toks.size();

    // section MRoPE positions (official get_rope_index): text keeps the running
    // index, image tokens take (0, h, w) offset by the position the image starts
    // at; the index advances by max(grid_h, grid_w) once the image is consumed
    std::vector<int> pos((size_t)4 * T, 0);
    std::vector<int> img_pos;
    img_pos.reserve((size_t)n_img);
    {
        const int gh = grid[0] > 0 ? grid[1] : 0, gw = grid[0] > 0 ? grid[2] : 0;
        int st = 0, seen = 0;
        for (int i = 0; i < T; i++) {
            if (n_img > 0 && toks[(size_t)i] == id_image_pad && seen < n_img) {
                const int k = seen % (gh * gw);
                pos[(size_t)i] = st;                          // t (grid_t == 1 -> 0)
                pos[(size_t)T + i] = st + k / gw;              // h
                pos[(size_t)2 * T + i] = st + k % gw;          // w
                img_pos.push_back(i);
                seen++;
                if (seen % (gh * gw) == 0) st += std::max(gh, gw);
            } else {
                pos[(size_t)i] = pos[(size_t)T + i] = pos[(size_t)2 * T + i] = st;
                st++;
            }
        }
    }

    TextGraph g;
    if (!build_text_graph(g, T, T, n_img, false)) { ggml_free(g.ctx); return false; }
    vqa::set_causal_mask(g.mask_in, 0, T);
    std::vector<float> logits;
    if (!run_text(g, toks, pos, n_img > 0 ? &feats : nullptr, n_img > 0 ? &img_pos : nullptr,
                  logits)) { ggml_free(g.ctx); return false; }
    ggml_free(g.ctx);
    n_past = T;
    // the next decode position is max(rope position)+1, not T: image tokens advance
    // the MRoPE index by max(grid_h, grid_w)
    next_pos = 0;
    for (int i = 0; i < T; i++)
        next_pos = std::max({ next_pos, pos[(size_t)i] + 1, pos[(size_t)T + i] + 1,
                              pos[(size_t)2 * T + i] + 1 });
    if (getenv("Q2V_DEBUG")) {
        fprintf(stderr, "tokens(%d):", T);
        for (int t : toks) fprintf(stderr, " %d", t);
        std::vector<float> c = logits;
        fprintf(stderr, "\nfirst-token top5:");
        for (int k = 0; k < 5; k++) {
            int best = 0;
            for (int i = 0; i < n_vocab; i++) if (c[(size_t)i] > c[(size_t)best]) best = i;
            fprintf(stderr, " %d(%.3f)", best, c[(size_t)best]);
            c[(size_t)best] = -INFINITY;
        }
        fprintf(stderr, "\n");
    }

    std::vector<int> out;
    int cur = vqa::sample_logits(logits, gp, rng_state);
    auto emit = [&](int t) {
        std::string piece = tok_.decode({ t });
        out.push_back(t);
        if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
    };
    if (cur != id_eos) emit(cur);

    for (int i = 1; i < gp.max_tokens && cur != id_eos; i++) {
        const int S = n_past + 1;
        TextGraph d;
        if (!build_text_graph(d, 1, S, 0, false)) { ggml_free(d.ctx); return false; }
        vqa::set_causal_mask(d.mask_in, n_past, 1);
        std::vector<int> p1((size_t)4, next_pos);
        std::vector<float> lg;
        if (!run_text(d, { cur }, p1, nullptr, nullptr, lg, n_past)) { ggml_free(d.ctx); return false; }
        ggml_free(d.ctx);
        n_past++;
        next_pos++;
        cur = vqa::sample_logits(lg, gp, rng_state);
        if (cur == id_eos) break;
        emit(cur);
    }
    out_text = tok_.decode(out);
    return true;
}

}  // namespace qwen2vl

namespace vqa {
Model * create_qwen2vl() { return new qwen2vl::Qwen25VL(); }
}  // namespace vqa
