// smolvlm_model.cpp - SmolVLM (Idefics3) backend: SigLIP tower + pixel-shuffle
// connector + SmolLM2 text trunk (Llama-style, no biases).
//
// Shape of the thing (256M checkpoint):
//   vision  SigLIP-base: Conv2d(3,768,16,16) + bias, learned 32x32 positions, 12
//           layers of LayerNorm(+bias) -> attention(qkv/out bias, no rope) -> MLP
//           (fc1/fc2 + bias, gelu tanh), then a post LayerNorm.
//   connector  4x4 pixel shuffle -> one Linear 12288 -> 576 (no bias), so each
//           512x512 tile becomes 64 text tokens.
//   text    SmolLM2-135M: 30 layers, d=576, 9 heads / 3 kv, head_dim 64,
//           RMSNorm(1e-5), SwiGLU, rope theta 1e5, untied lm_head.
//
// Idefics3 preprocessing (what the official processor does, verified against it):
//   resize so the longest edge is 2048, then resize AGAIN so both sides are
//   multiples of the 512 tile (aspect not preserved, like HF), cut the refined
//   image into 512x512 tiles row-major, and append one global 512x512 view of the
//   whole refined image.  The prompt expands <image> into
//     <fake><row_R_col_C><image>*64        (per tile, row-major, "\n" per row)
//     <fake><global-img><image>*64<fake>   (the global view, last)
//   so the features must come out in the same order: tiles first, global last.
#include "core/vqa.hpp"

#include "core/engine/attention.hpp"
#include "core/engine/backend.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "core/engine/ggml_dump.hpp"
#include "gguf.h"
#include "core/engine/gguf_util.hpp"
#include "core/runtime/load_util.hpp"
#include "core/blocks/llm_block.hpp"
#include "core/media/resample.hpp"
#include "core/engine/sampler.hpp"
#include "core/engine/weight_load.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace smolvlm {

struct VisLayer {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr, *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *wq = nullptr, *bq = nullptr, *wk = nullptr, *bk = nullptr;
    ggml_tensor *wv = nullptr, *bv = nullptr, *wo = nullptr, *bo = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr, *fc2_w = nullptr, *fc2_b = nullptr;
};

struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *pos_in = nullptr, *mask_in = nullptr;
    ggml_tensor *img_in = nullptr, *img_pos_in = nullptr;
    ggml_tensor *logits = nullptr;
};

class SmolVLM : public vqa::Model {
public:
    ~SmolVLM() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "smolvlm"; }
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp,
             std::string & out_text) override;

    int n_ctx = 4096;
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool alloc_kv();

    bool preprocess(const vqa::MediaInput & media, std::vector<std::vector<float>> & tiles,
                    int * gx, int * gy);
    bool build_vision_graph(int n_batch);
    bool run_vision(const std::vector<std::vector<float>> & batch, std::vector<float> & feats);
    bool encode_image(const vqa::MediaInput & media, std::vector<float> & feats, int * n_tiles,
                      int * gx, int * gy);

    bool build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_last);
    bool run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<int> & pos,
                  const std::vector<float> * img_feats, const std::vector<int> * img_pos,
                  std::vector<float> & logits, int slot_base = 0);

    void dump(const ggml_cgraph * gf, const char * nm);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    // text
    int n_layer = 30, n_embd = 576, n_head = 9, n_kv_head = 3, head_dim = 64;
    int n_ff = 1536, n_vocab = 49280;
    float rms_eps = 1e-5f, rope_theta = 1e5f;

    // vision
    int v_layer = 12, v_embd = 768, v_head = 12, v_ff = 3072;
    int v_patch = 16, v_image = 512, v_merge = 4, v_tile = 512, v_longest = 2048;
    int v_resample = 1, image_seq_len = 64;
    float v_eps = 1e-6f;
    float v_mean[3] = { 0.5f, 0.5f, 0.5f }, v_std[3] = { 0.5f, 0.5f, 0.5f };

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    vqa::MMap mm;
    vqa::AlignedBuf host_f32;
    ggml_gallocr_t allocr = nullptr;

    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr, *lm_head = nullptr;
    std::vector<vqa::llm::LayerWeights> tl;   // core/llm_block.hpp

    ggml_tensor *v_patch_w = nullptr, *v_patch_b = nullptr, *v_pos = nullptr;
    ggml_tensor *v_post_w = nullptr, *v_post_b = nullptr, *v_proj = nullptr;
    std::vector<VisLayer> vl;

    // vision graph (fixed shape: one 512x512 tile)
    ggml_context * vctx = nullptr;
    ggml_cgraph * vgf = nullptr;
    ggml_tensor *v_in = nullptr, *v_feats = nullptr;
    int v_batch = 0;      // crops currently baked into the vision graph

    vqa::KvCache kv;
    int n_past = 0;
    int vis_runs = 0;   // parity dumps only cover the first tile

    int id_im_start = 1, id_im_end = 2, id_tile_tag = 49152, id_row_col = 49153;
    int id_fake = 49189, id_image = 49190, id_eou = 49279;
    int id_eos = 2;

    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

// ---------------------------------------------------------------- lifecycle

SmolVLM::~SmolVLM() {
    kv.free();
    if (vctx) ggml_free(vctx);
    if (allocr) ggml_gallocr_free(allocr);
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (backend) ggml_backend_free(backend);
}

bool SmolVLM::init_backend() {
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

bool SmolVLM::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer   = vqa::gguf::u32(gu, "smolvlm.block_count", n_layer);
    n_embd    = vqa::gguf::u32(gu, "smolvlm.embedding_length", n_embd);
    n_head    = vqa::gguf::u32(gu, "smolvlm.attention.head_count", n_head);
    n_kv_head = vqa::gguf::u32(gu, "smolvlm.attention.head_count_kv", n_kv_head);
    head_dim  = vqa::gguf::u32(gu, "smolvlm.attention.key_length", head_dim);
    n_ff      = vqa::gguf::u32(gu, "smolvlm.feed_forward_length", n_ff);
    rms_eps   = vqa::gguf::f32(gu, "smolvlm.attention.layer_norm_rms_epsilon", rms_eps);
    rope_theta = vqa::gguf::f32(gu, "smolvlm.rope.freq_base", rope_theta);

    v_layer   = vqa::gguf::u32(gu, "smolvlm.vision.block_count", v_layer);
    v_embd    = vqa::gguf::u32(gu, "smolvlm.vision.embedding_length", v_embd);
    v_head    = vqa::gguf::u32(gu, "smolvlm.vision.attention.head_count", v_head);
    v_ff      = vqa::gguf::u32(gu, "smolvlm.vision.feed_forward_length", v_ff);
    v_patch   = vqa::gguf::u32(gu, "smolvlm.vision.patch_size", v_patch);
    v_image   = vqa::gguf::u32(gu, "smolvlm.vision.image_size", v_image);
    v_tile    = vqa::gguf::u32(gu, "smolvlm.vision.tile_size", v_tile);
    v_longest = vqa::gguf::u32(gu, "smolvlm.vision.longest_edge", v_longest);
    v_merge   = vqa::gguf::u32(gu, "smolvlm.vision.merge_size", v_merge);
    v_resample = vqa::gguf::u32(gu, "smolvlm.vision.resize_resample", v_resample);
    image_seq_len = vqa::gguf::u32(gu, "smolvlm.vision.image_seq_len", image_seq_len);
    v_eps     = vqa::gguf::f32(gu, "smolvlm.vision.layer_norm_epsilon", v_eps);
    {
        int64_t id = gguf_find_key(gu, "smolvlm.vision.image_mean");
        if (id >= 0 && gguf_get_arr_n(gu, id) == 3) {
            const float * a = (const float *)gguf_get_arr_data(gu, id);
            for (int c = 0; c < 3; c++) v_mean[c] = a[c];
        }
        id = gguf_find_key(gu, "smolvlm.vision.image_std");
        if (id >= 0 && gguf_get_arr_n(gu, id) == 3) {
            const float * a = (const float *)gguf_get_arr_data(gu, id);
            for (int c = 0; c < 3; c++) v_std[c] = a[c];
        }
    }
    id_im_start = vqa::gguf::u32(gu, "smolvlm.im_start_token_id", id_im_start);
    id_tile_tag = vqa::gguf::u32(gu, "smolvlm.tile_token_id", id_tile_tag);
    id_row_col  = vqa::gguf::u32(gu, "smolvlm.row_col_token_id", id_row_col);
    id_fake     = vqa::gguf::u32(gu, "smolvlm.fake_token_id", id_fake);
    id_image    = vqa::gguf::u32(gu, "smolvlm.image_token_id", id_image);
    id_eou      = vqa::gguf::u32(gu, "smolvlm.eou_token_id", id_eou);

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
    lm_head = gt("t.lm_head.weight");
    tl.resize((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "t.blk.%d.%s", i, suf); return gt(b); };
        vqa::llm::LayerWeights & L = tl[(size_t)i];   // qn/kn stay null: no per-head norms
        L.norm = g("attn_norm.weight");
        L.q = g("wq.weight"); L.k = g("wk.weight"); L.v = g("wv.weight"); L.o = g("wo.weight");
        L.ffn_norm = g("ffn_norm.weight");
        L.gate = g("w_gate.weight"); L.up = g("w_up.weight"); L.down = g("w_down.weight");
        if (!L.norm || !L.q || !L.k || !L.v || !L.o || !L.gate || !L.up || !L.down) return false;
    }

    v_patch_w = gt("v.patch_embd.weight");
    v_patch_b = gt("v.patch_embd.bias");
    v_pos = gt("v.pos_embd.weight");
    v_post_w = gt("v.post_ln.weight");
    v_post_b = gt("v.post_ln.bias");
    v_proj = gt("v.mm_proj.weight");
    vl.resize((size_t)v_layer);
    for (int i = 0; i < v_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "v.blk.%d.%s", i, suf); return gt(b); };
        VisLayer & L = vl[(size_t)i];
        L.ln1_w = g("ln1.weight"); L.ln1_b = g("ln1.bias");
        L.ln2_w = g("ln2.weight"); L.ln2_b = g("ln2.bias");
        L.wq = g("wq.weight"); L.bq = g("bq");
        L.wk = g("wk.weight"); L.bk = g("bk");
        L.wv = g("wv.weight"); L.bv = g("bv");
        L.wo = g("wo.weight"); L.bo = g("bo");
        L.fc1_w = g("fc1.weight"); L.fc1_b = g("fc1.bias");
        L.fc2_w = g("fc2.weight"); L.fc2_b = g("fc2.bias");
        if (!L.ln1_w || !L.ln2_w || !L.wq || !L.wk || !L.wv || !L.wo || !L.fc1_w || !L.fc2_w) return false;
    }
    return tok_embd && out_norm && lm_head && v_patch_w && v_pos && v_proj;
}

bool SmolVLM::alloc_kv() { return kv.alloc(backend, n_layer, n_kv_head, head_dim, n_ctx); }

bool SmolVLM::load(const std::string & gguf_path, const std::string & tokenizer_path,
                   const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "SMOLVLM", n_threads, use_gpu, dump_dir);
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::Gpt2Regex)) return false;
    if (!alloc_kv()) return false;

    caps_.image = true;
    caps_.video = false;
    fprintf(stderr, "smolvlm: ready (%dL d=%d nh=%d/%d hd=%d | vision %dL d=%d patch=%d merge=%d "
                    "-> %d tokens/tile)\n", n_layer, n_embd, n_head, n_kv_head, head_dim,
            v_layer, v_embd, v_patch, v_merge, image_seq_len);
    return true;
}

// ---------------------------------------------------------------- vision

// The whole crop set goes through ONE graph: the batch dimension is the crop, so
// a 13-crop image is a single compute instead of 13 sequential ones (the matmuls
// and the attention then have B times the work per launch, which is what the CPU
// backend wants; the FLOPs themselves are unchanged).
bool SmolVLM::build_vision_graph(int n_batch) {
    if (vctx) { ggml_free(vctx); vctx = nullptr; }
    const int B = std::max(1, n_batch);
    const int P = v_patch;
    const int n_patch = (v_tile / P) * (v_tile / P);     // 32x32 = 1024
    const int C3 = 3 * P * P;
    const int side = v_tile / P;
    const int nm = side / v_merge;
    const int E = v_embd;

    struct ggml_init_params ip = { 1024ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    vctx = ggml_init(ip);
    if (!vctx) return false;
    vgf = ggml_new_graph_custom(vctx, 32768, false);

    v_in = ggml_new_tensor_3d(vctx, GGML_TYPE_F32, C3, n_patch, B);
    ggml_set_name(v_in, "v_patches");

    ggml_tensor * x = ggml_mul_mat(vctx, v_patch_w, v_in);   // [E, n_patch, B]
    x = ggml_add(vctx, x, v_patch_b);                        // [E] broadcasts
    x = ggml_add(vctx, x, v_pos);                            // [E, n_patch] broadcasts
    ggml_set_name(x, "v_patch_out");

    const int hd = E / v_head;
    for (int il = 0; il < v_layer; il++) {
        VisLayer & L = vl[(size_t)il];
        ggml_tensor * h = ggml_norm(vctx, x, v_eps);
        h = ggml_mul(vctx, h, L.ln1_w);
        h = ggml_add(vctx, h, L.ln1_b);

        ggml_tensor * q = ggml_add(vctx, ggml_mul_mat(vctx, L.wq, h), L.bq);
        ggml_tensor * k = ggml_add(vctx, ggml_mul_mat(vctx, L.wk, h), L.bk);
        ggml_tensor * v = ggml_add(vctx, ggml_mul_mat(vctx, L.wv, h), L.bv);
        if (il == 0) { ggml_set_name(q, "v_blk0_q"); ggml_set_name(k, "v_blk0_k"); }

        // [hd, n_patch, head, B] - the batch is the trailing dim for flash attention
        ggml_tensor * qf = ggml_cont(vctx, ggml_permute(vctx,
            ggml_reshape_4d(vctx, q, hd, v_head, n_patch, B), 0, 2, 1, 3));
        ggml_tensor * kf = ggml_cont(vctx, ggml_permute(vctx,
            ggml_reshape_4d(vctx, k, hd, v_head, n_patch, B), 0, 2, 1, 3));
        ggml_tensor * vf = ggml_cont(vctx, ggml_permute(vctx,
            ggml_reshape_4d(vctx, v, hd, v_head, n_patch, B), 0, 2, 1, 3));
        ggml_tensor * fa = ggml_flash_attn_ext(vctx, qf, kf, vf, nullptr,
                                               1.0f / sqrtf((float)hd), 0.0f, 0.0f);
        // FA returns [hd, head, n_patch, B]: (head, hd) is already token-major
        ggml_tensor * attn = ggml_reshape_3d(vctx, ggml_cont(vctx, fa), E, n_patch, B);
        attn = ggml_add(vctx, ggml_mul_mat(vctx, L.wo, attn), L.bo);
        x = ggml_add(vctx, x, attn);
        if (il == 0) ggml_set_name(x, "v_blk0_attn_out");

        h = ggml_norm(vctx, x, v_eps);
        h = ggml_mul(vctx, h, L.ln2_w);
        h = ggml_add(vctx, h, L.ln2_b);
        h = ggml_add(vctx, ggml_mul_mat(vctx, L.fc1_w, h), L.fc1_b);
        h = ggml_gelu(vctx, h);                       // gelu_pytorch_tanh
        h = ggml_add(vctx, ggml_mul_mat(vctx, L.fc2_w, h), L.fc2_b);
        x = ggml_add(vctx, x, h);
        if (il == 0) ggml_set_name(x, "v_blk0_out");
    }

    x = ggml_norm(vctx, x, v_eps);
    x = ggml_mul(vctx, x, v_post_w);
    x = ggml_add(vctx, x, v_post_b);
    ggml_set_name(x, "v_final");

    // ---- 4x4 pixel shuffle (llama.cpp's op sequence, batch kept last)
    ggml_tensor * c = ggml_reshape_4d(vctx, x, E, side, side, B);
    c = ggml_reshape_4d(vctx, c, E * v_merge, nm, side, B);
    c = ggml_permute(vctx, c, 0, 2, 1, 3);
    c = ggml_cont_4d(vctx, c, E * v_merge * v_merge, nm, nm, B);
    c = ggml_permute(vctx, c, 0, 2, 1, 3);
    c = ggml_reshape_3d(vctx, ggml_cont(vctx, c), E * v_merge * v_merge, nm * nm, B);
    ggml_set_name(c, "v_shuffle");

    ggml_tensor * m = ggml_mul_mat(vctx, v_proj, c);               // [576, 64, B]
    ggml_set_name(m, "v_feats");
    v_feats = m;
    ggml_build_forward_expand(vgf, m);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "v_patches", "v_patch_out", "v_blk0_attn_out",
                                           "v_blk0_out", "v_final", "v_shuffle", "v_feats",
                                           nullptr };
        vqa::dump::mark_sinks(vctx, vgf, dn);
        vqa::dump::finalize_sinks(vctx, vgf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, vgf)) {
        fprintf(stderr, "vision graph allocation failed\n");
        return false;
    }
    v_batch = B;
    return true;
}

bool SmolVLM::run_vision(const std::vector<std::vector<float>> & batch,
                         std::vector<float> & feats) {
    std::vector<float> in;
    for (const auto & c : batch) in.insert(in.end(), c.begin(), c.end());
    ggml_backend_tensor_set(v_in, in.data(), 0, sizeof(float) * in.size());
    if (ggml_backend_graph_compute(backend, vgf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vision compute failed\n");
        return false;
    }
    // parity dumps only make sense for a single crop (the reference exports tile 0)
    if (!dump_dir.empty() && vis_runs == 0) {
        static const char * const dn[] = { "v_patches", "v_patch_out", "v_blk0_attn_out",
                                           "v_blk0_out", "v_final", "v_shuffle", "v_feats",
                                           nullptr };
        for (int i = 0; dn[i]; i++) dump(vgf, dn[i]);
    }
    vis_runs++;
    const int d = (int)v_feats->ne[0];
    const int per = (int)v_feats->ne[1];
    const int n = (int)v_feats->ne[2];
    feats.resize((size_t)n * per * d);
    ggml_backend_tensor_get(v_feats, feats.data(), 0, sizeof(float) * feats.size());
    return true;
}

bool SmolVLM::preprocess(const vqa::MediaInput & media, std::vector<std::vector<float>> & tiles,
                         int * gx, int * gy) {
    if (media.empty()) return false;
    const int w0 = media.width, h0 = media.height;
    const uint8_t * rgb = media.frames[0].rgb.data();

    // step 1: longest edge (aspect preserved). HF's processor always resizes to
    // 2048 - that is the official default (an old 1024 cap silently deviated;
    // it is available via SMOLVLM_MAX_EDGE for constrained devices only).
    int longest = v_longest;
    if (const char * e = getenv("SMOLVLM_MAX_EDGE")) {
        const int cap = atoi(e);
        if (cap > 0) longest = cap;
    }
    const double scale = (double)longest / (double)std::max(w0, h0);
    int rw = std::max(1, (int)std::lround(w0 * scale));
    int rh = std::max(1, (int)std::lround(h0 * scale));
    std::vector<float> resized;
    vqa::img::resize_antialias(rgb, w0, h0, rw, rh, resized, v_resample);
    if (getenv("SMOLVLM_DUMP_RGB")) {
        FILE * f = fopen((dump_dir + "/dbg_resize1.bin").c_str(), "wb");
        if (f) { fwrite(resized.data(), sizeof(float), resized.size(), f); fclose(f); }
    }

    // step 2: round both sides UP to a multiple of the tile, resizing again
    *gx = (rw + v_tile - 1) / v_tile;
    *gy = (rh + v_tile - 1) / v_tile;
    const int fw = *gx * v_tile, fh = *gy * v_tile;
    std::vector<float> refined;
    {
        // the engine's resize takes u8 input; round the intermediate back to u8
        // (the official processor keeps float here, so this is a known small delta)
        std::vector<uint8_t> tmp((size_t)rw * rh * 3);
        for (size_t i = 0; i < tmp.size(); i++) {
            float v = resized[i];
            tmp[i] = (uint8_t)std::min(255.0f, std::max(0.0f, std::nearbyintf(v)));
        }
        vqa::img::resize_antialias(tmp.data(), rw, rh, fw, fh, refined, v_resample);
    }

    auto norm_to = [&](const std::vector<float> & src, int sw, int sh, int ox, int oy,
                       std::vector<float> & out) {
        const int P = v_patch;
        const int np = (v_tile / P) * (v_tile / P);
        const int C3 = 3 * P * P;
        out.assign((size_t)C3 * np, 0.0f);
        int u = 0;
        for (int py = 0; py < v_tile / P; py++)
            for (int px = 0; px < v_tile / P; px++) {
                float * row = out.data() + (size_t)u * C3;
                int o = 0;
                for (int c = 0; c < 3; c++)
                    for (int dh = 0; dh < P; dh++) {
                        const float * s = src.data() +
                            ((size_t)(oy + py * P + dh) * sw + (ox + px * P)) * 3 + c;
                        for (int dw = 0; dw < P; dw++)
                            row[o++] = (s[(size_t)dw * 3] / 255.0f - v_mean[c]) / v_std[c];
                    }
                u++;
            }
    };

    // tiles first (row-major), then the global view of the whole refined image
    tiles.clear();
    for (int ty = 0; ty < *gy; ty++)
        for (int tx = 0; tx < *gx; tx++) {
            std::vector<float> t;
            norm_to(refined, fw, fh, tx * v_tile, ty * v_tile, t);
            tiles.push_back(std::move(t));
        }
    {
        // the global view is the refined image resized to tile x tile
        std::vector<uint8_t> tmp((size_t)fw * fh * 3);
        for (size_t i = 0; i < tmp.size(); i++) {
            float v = refined[i];
            tmp[i] = (uint8_t)std::min(255.0f, std::max(0.0f, std::nearbyintf(v)));
        }
        std::vector<float> global;
        vqa::img::resize_antialias(tmp.data(), fw, fh, v_tile, v_tile, global, v_resample);
        std::vector<float> t;
        norm_to(global, v_tile, v_tile, 0, 0, t);
        tiles.push_back(std::move(t));
    }
    return true;
}

bool SmolVLM::encode_image(const vqa::MediaInput & media, std::vector<float> & feats,
                           int * n_tiles, int * gx, int * gy) {
    std::vector<std::vector<float>> tiles;
    if (!preprocess(media, tiles, gx, gy)) return false;
    // one compute for the whole crop set; parity dumps need a single crop because
    // the reference only exports tile 0
    const int B = dump_dir.empty() ? (int)tiles.size() : 1;
    feats.clear();
    for (int off = 0; off < (int)tiles.size(); off += B) {
        const int n = std::min(B, (int)tiles.size() - off);
        std::vector<std::vector<float>> chunk(tiles.begin() + off, tiles.begin() + off + n);
        if (!vgf || v_batch != n) {
            if (!build_vision_graph(n)) return false;
        }
        std::vector<float> f;
        if (!run_vision(chunk, f)) return false;
        feats.insert(feats.end(), f.begin(), f.end());
    }
    *n_tiles = (int)(tiles.size() - 1);     // the global view is not a tile
    fprintf(stderr, "vision: %dx%d -> %dx%d tiles + global, %zu images, %d tokens\n",
            media.width, media.height, *gx, *gy, tiles.size(), (int)(feats.size() / n_embd));
    return true;
}

// ---------------------------------------------------------------- text

bool SmolVLM::build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_last) {
    struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 32768, false);

    const int D = n_embd, HD = head_dim, NH = n_head, NKV = n_kv_head;

    g.tok_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    g.pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.pos_in, "pos_in");
    g.mask_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, S, T);
    ggml_set_name(g.mask_in, "mask_in");

    ggml_tensor * x = ggml_get_rows(g.ctx, tok_embd, g.tok_in);
    if (n_img > 0) {
        g.img_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, n_img);
        ggml_set_name(g.img_in, "img_in");
        g.img_pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_img);
        ggml_set_name(g.img_pos_in, "img_pos_in");
        x = ggml_set_rows(g.ctx, ggml_dup(g.ctx, x), g.img_in, g.img_pos_in);
    }
    ggml_set_name(x, "t_emb");

    // ---- decoder layer loop: shared vqa::llm::block (core/llm_block.hpp) ----
    // SmolLM2 trunk moved verbatim: RMSNorm -> WQ/WK/WV -> NEOX rope (theta 1e5,
    // the llm_block roadmap's "no rope" note was wrong) -> KV write-back ->
    // flash_attn_ext over the F16 cache -> wo -> residual -> RMSNorm -> fused
    // SwiGLU. No per-head q/k norms (qn/kn null) and no batch path (bat_groups
    // null); the KV row indices are pos_in, whose values are exactly the slot
    // numbers the old slot_in input carried (i on prefill, slot_base on decode).
    vqa::llm::LayerCtx lc;
    lc.gf = g.gf;
    lc.pos = g.pos_in;
    lc.mask = g.mask_in;
    lc.T = T; lc.S = S;
    lc.n_head = NH; lc.n_kv_head = NKV; lc.head_dim = HD;
    lc.n_ctx = n_ctx;
    lc.rms_eps = rms_eps;
    lc.rope_theta = rope_theta;
    vqa::llm::BlockHooks lh;   // dbg_layer0 keeps the t_l0_* probe names stamped
    for (int il = 0; il < n_layer; il++) {
        lc.k_cache = kv.k[il];
        lc.v_cache = kv.v[il];
        x = vqa::llm::block(g.ctx, x, il, tl[il], lc, lh);
    }

    x = ggml_rms_norm(g.ctx, x, rms_eps);
    x = ggml_mul(g.ctx, x, out_norm);
    ggml_set_name(x, "t_final_norm");

    ggml_tensor * last = logits_last
        ? ggml_view_2d(g.ctx, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]) : x;
    g.logits = ggml_mul_mat(g.ctx, lm_head, last);
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "t_emb", "t_l0_norm", "t_l0_q_rope", "t_l0_attn",
                                           "t_l0_out", "t_final_norm", nullptr };
        vqa::dump::mark_sinks(g.ctx, g.gf, dn);
        vqa::dump::finalize_sinks(g.ctx, g.gf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, g.gf)) {
        fprintf(stderr, "text graph allocation failed\n");
        return false;
    }
    return true;
}

bool SmolVLM::run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<int> & pos,
                       const std::vector<float> * img_feats, const std::vector<int> * img_pos,
                       std::vector<float> & logits, int slot_base) {
    const int T = (int)toks.size();
    ggml_backend_tensor_set(g.tok_in, toks.data(), 0, sizeof(int) * toks.size());
    ggml_backend_tensor_set(g.pos_in, pos.data(), 0, sizeof(int) * pos.size());
    if (g.img_in && img_feats && img_pos) {
        ggml_backend_tensor_set(g.img_in, img_feats->data(), 0, sizeof(float) * img_feats->size());
        ggml_backend_tensor_set(g.img_pos_in, img_pos->data(), 0, sizeof(int) * img_pos->size());
    }
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "text compute failed\n");
        return false;
    }
    // only the prefill graph is comparable with the reference (later decode steps
    // would overwrite the files)
    if (!dump_dir.empty() && slot_base == 0 && T > 1) {
        static const char * const dn[] = { "t_emb", "t_l0_norm", "t_l0_q_rope", "t_l0_attn",
                                           "t_l0_out", "t_final_norm", "logits", nullptr };
        for (int i = 0; dn[i]; i++) dump(g.gf, dn[i]);
    }
    logits.resize((size_t)n_vocab);
    ggml_backend_tensor_get(g.logits, logits.data(), 0, sizeof(float) * logits.size());
    return true;
}

void SmolVLM::dump(const ggml_cgraph * gf, const char * nm) {
    if (dump_dir.empty()) return;
    vqa::dump::write((ggml_cgraph *)gf, dump_dir.c_str(), nm);
}

// ---------------------------------------------------------------- ask

bool SmolVLM::ask(const vqa::MediaInput & media, const std::string & prompt,
                  const std::string & /*mode*/, const vqa::GenParams & gp,
                  std::string & out_text) {
    out_text.clear();
    std::vector<float> feats;
    int n_tiles = 0, gx = 0, gy = 0;
    if (!media.empty()) {
        if (gp.on_stage) gp.on_stage("vision", 0, 1, gp.on_stage_ud);
        const auto tv0 = std::chrono::steady_clock::now();
        if (!encode_image(media, feats, &n_tiles, &gx, &gy)) return false;
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tv0).count();
        fprintf(stderr, "[perf] smolvlm vision: %.0f ms for %d crops (%.0f ms/crop)\n",
                ms, n_tiles + 1, ms / (double)(n_tiles + 1));
        if (gp.on_stage) gp.on_stage("vision", 1, 1, gp.on_stage_ud);
    }

    // Idefics3 chat template + the official <image> expansion
    std::string text = "<|im_start|>User:";
    if (n_tiles > 0) {
        for (int r = 0; r < gy; r++) {
            for (int c = 0; c < gx; c++) {
                text += "<fake_token_around_image>";
                text += "<row_" + std::to_string(r + 1) + "_col_" + std::to_string(c + 1) + ">";
                for (int i = 0; i < image_seq_len; i++) text += "<image>";
            }
            text += "\n";
        }
        text += "\n<fake_token_around_image><global-img>";
        for (int i = 0; i < image_seq_len; i++) text += "<image>";
        text += "<fake_token_around_image>";
    }
    text += prompt;
    text += "<end_of_utterance>\nAssistant:";

    std::vector<int> toks = tok_.encode(text);
    const int T = (int)toks.size();
    if (T + gp.max_tokens + 8 > n_ctx) {
        fprintf(stderr, "prompt %d + %d tokens exceed n_ctx %d\n", T, gp.max_tokens, n_ctx);
        return false;
    }
    std::vector<int> pos((size_t)T);
    for (int i = 0; i < T; i++) pos[(size_t)i] = i;

    std::vector<int> img_pos;
    if (!feats.empty()) {
        const int want = (int)(feats.size() / n_embd);
        for (int i = 0; i < T && (int)img_pos.size() < want; i++)
            if (toks[(size_t)i] == id_image) img_pos.push_back(i);
        if ((int)img_pos.size() != want) {
            fprintf(stderr, "image token mismatch: %zu positions for %d features\n",
                    img_pos.size(), want);
            return false;
        }
    }

    TextGraph g;
    const auto tp0 = std::chrono::steady_clock::now();
    if (!build_text_graph(g, T, T, (int)img_pos.size(), true)) { ggml_free(g.ctx); return false; }
    vqa::set_causal_mask(g.mask_in, 0, T);
    std::vector<float> logits;
    if (!run_text(g, toks, pos, img_pos.empty() ? nullptr : &feats,
                  img_pos.empty() ? nullptr : &img_pos, logits)) {
        ggml_free(g.ctx);
        return false;
    }
    ggml_free(g.ctx);
    n_past = T;
    fprintf(stderr, "[perf] smolvlm prefill: %.0f ms for %d tokens\n",
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tp0).count(), T);

    if (getenv("SMOLVLM_DEBUG")) {
        fprintf(stderr, "tokens(%d) of which %zu image:", T, img_pos.size());
        for (int i = 0; i < std::min(T, 24); i++) fprintf(stderr, " %d", toks[(size_t)i]);
        std::vector<float> c = logits;
        fprintf(stderr, "\nfirst-token top5:");
        for (int k = 0; k < 5; k++) {
            int best = 0;
            for (int i = 0; i < n_vocab; i++) if (c[(size_t)i] > c[(size_t)best]) best = i;
            fprintf(stderr, " %d(%.2f)", best, c[(size_t)best]);
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
        if (!build_text_graph(d, 1, S, 0, true)) { ggml_free(d.ctx); return false; }
        vqa::set_causal_mask(d.mask_in, n_past, 1);
        std::vector<int> p1(1, n_past);
        std::vector<float> lg;
        if (!run_text(d, { cur }, p1, nullptr, nullptr, lg, n_past)) {
            ggml_free(d.ctx);
            return false;
        }
        ggml_free(d.ctx);
        n_past++;
        cur = vqa::sample_logits(lg, gp, rng_state);
        if (cur == id_eos) break;
        emit(cur);
    }
    out_text = tok_.decode(out);
    return true;
}

}  // namespace smolvlm

namespace vqa {
Model * create_smolvlm() { return new smolvlm::SmolVLM(); }
}  // namespace vqa
