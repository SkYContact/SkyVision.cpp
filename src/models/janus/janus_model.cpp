// janus_model.cpp - DeepSeek Janus-Pro (understanding path) backend: timm
// SigLIP tower + 2-layer MLP aligner + plain llama trunk (no biases, no qk
// norms, rope theta 1e4, untied 102400-vocab lm_head).
//
// Shape of the thing (Janus-Pro-1B checkpoint):
//   vision  SigLIP-large-patch16-384 (timm flavour): Conv2d(3,1024,16,16) +
//           bias, learned 24x24 positions (no cls token), 24 pre-norm blocks of
//           LayerNorm(+bias) -> FUSED qkv (+bias, exact-erf GELU MLP) -> wo
//           (+bias), then a final LayerNorm. 576 tokens per image.
//   aligner Linear(1024->2048) -> GELU(erf) -> Linear(2048->2048); the output
//           rows replace the 576 "<image_placeholder>" embedding rows.
//   text    llama 1.5B: 24 layers, d=2048, 16:16 heads, head_dim 128, ff 5632,
//           RMSNorm(1e-5), SwiGLU, theta 1e4.
//   The generation path (VQ tokenizer / gen_aligner / gen_head) is not part of
//   this engine: this is the understanding (image VQA) side only.
//
// Preprocessing (official VLMImageProcessor): bicubic resize with antialias so
// the longest edge is 384 (aspect preserved), expand2square pad with the
// (127,127,127) background derived from image_mean, /255, (x-0.5)/0.5.
//
// Chat template (official deepseek sft format):
//   <BOS><|User|>: {question}\n<image_placeholder>x576\n\n<|Assistant|>:
// (the trailing sep2 / EOS 100001 closes the assistant turn while decoding).
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
#include "core/blocks/vit_block.hpp"
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

namespace janus {

struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *pos_in = nullptr, *mask_in = nullptr;
    ggml_tensor *img_in = nullptr, *img_pos_in = nullptr;
    ggml_tensor *logits = nullptr;
};

class Janus : public vqa::Model {
public:
    ~Janus() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "janus"; }
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

    bool preprocess(const vqa::MediaInput & media, std::vector<float> & out);
    bool build_vision_graph();
    bool run_vision(const std::vector<float> & patches, std::vector<float> & feats);
    bool encode_image(const vqa::MediaInput & media, std::vector<float> & feats);

    bool build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_last);
    bool run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<int> & pos,
                  const std::vector<float> * img_feats, const std::vector<int> * img_pos,
                  std::vector<float> & logits, int slot_base = 0);

    void dump(const ggml_cgraph * gf, const char * nm);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    // text (llama trunk)
    int n_layer = 24, n_embd = 2048, n_head = 16, n_kv_head = 16, head_dim = 128;
    int n_ff = 5632, n_vocab = 102400;
    float rms_eps = 1e-5f, rope_theta = 1e4f;

    // vision (timm SigLIP-large-384)
    int v_layer = 24, v_embd = 1024, v_head = 16, v_ff = 4096;
    int v_patch = 16, v_image = 384;
    int image_seq_len = 576;
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
    ggml_tensor *v_post_w = nullptr, *v_post_b = nullptr;
    ggml_tensor *v_mm0_w = nullptr, *v_mm0_b = nullptr;
    ggml_tensor *v_mm2_w = nullptr, *v_mm2_b = nullptr;
    std::vector<vqa::vit::BlockWeights> vl;   // core/blocks/vit_block.hpp

    // vision graph (fixed shape: one 384x384 image)
    ggml_context * vctx = nullptr;
    ggml_cgraph * vgf = nullptr;
    ggml_tensor *v_in = nullptr, *v_feats = nullptr;
    int vis_runs = 0;    // parity dumps only cover the first run

    vqa::KvCache kv;
    int n_past = 0;

    int id_image = 100581;
    int id_eos = 100001;

    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

// ---------------------------------------------------------------- lifecycle

Janus::~Janus() {
    kv.free();
    if (vctx) ggml_free(vctx);
    if (allocr) ggml_gallocr_free(allocr);
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (backend) ggml_backend_free(backend);
}

bool Janus::init_backend() {
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

bool Janus::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer   = vqa::gguf::u32(gu, "janus.block_count", n_layer);
    n_embd    = vqa::gguf::u32(gu, "janus.embedding_length", n_embd);
    n_head    = vqa::gguf::u32(gu, "janus.attention.head_count", n_head);
    n_kv_head = vqa::gguf::u32(gu, "janus.attention.head_count_kv", n_kv_head);
    head_dim  = vqa::gguf::u32(gu, "janus.attention.key_length", head_dim);
    n_ff      = vqa::gguf::u32(gu, "janus.feed_forward_length", n_ff);
    n_vocab   = vqa::gguf::u32(gu, "janus.vocab_size", n_vocab);
    rms_eps   = vqa::gguf::f32(gu, "janus.attention.layer_norm_rms_epsilon", rms_eps);
    rope_theta = vqa::gguf::f32(gu, "janus.rope.freq_base", rope_theta);

    v_layer   = vqa::gguf::u32(gu, "janus.vision.block_count", v_layer);
    v_embd    = vqa::gguf::u32(gu, "janus.vision.embedding_length", v_embd);
    v_head    = vqa::gguf::u32(gu, "janus.vision.attention.head_count", v_head);
    v_ff      = vqa::gguf::u32(gu, "janus.vision.feed_forward_length", v_ff);
    v_patch   = vqa::gguf::u32(gu, "janus.vision.patch_size", v_patch);
    v_image   = vqa::gguf::u32(gu, "janus.vision.image_size", v_image);
    image_seq_len = vqa::gguf::u32(gu, "janus.vision.image_seq_len", image_seq_len);
    v_eps     = vqa::gguf::f32(gu, "janus.vision.layer_norm_epsilon", v_eps);
    {
        int64_t id = gguf_find_key(gu, "janus.vision.image_mean");
        if (id >= 0 && gguf_get_arr_n(gu, id) == 3) {
            const float * a = (const float *)gguf_get_arr_data(gu, id);
            for (int c = 0; c < 3; c++) v_mean[c] = a[c];
        }
        id = gguf_find_key(gu, "janus.vision.image_std");
        if (id >= 0 && gguf_get_arr_n(gu, id) == 3) {
            const float * a = (const float *)gguf_get_arr_data(gu, id);
            for (int c = 0; c < 3; c++) v_std[c] = a[c];
        }
    }
    id_image = vqa::gguf::u32(gu, "janus.image_token_id", id_image);
    id_eos   = vqa::gguf::u32(gu, "janus.eos_token_id", id_eos);

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
        vqa::llm::LayerWeights & L = tl[(size_t)i];   // plain llama: qn/kn null
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
    v_mm0_w = gt("v.mm_proj0.weight"); v_mm0_b = gt("v.mm_proj0.bias");
    v_mm2_w = gt("v.mm_proj2.weight"); v_mm2_b = gt("v.mm_proj2.bias");
    vl.resize((size_t)v_layer);
    for (int i = 0; i < v_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "v.blk.%d.%s", i, suf); return gt(b); };
        vqa::vit::BlockWeights & L = vl[(size_t)i];
        L.ln1_w = g("ln1.weight"); L.ln1_b = g("ln1.bias");
        L.ln2_w = g("ln2.weight"); L.ln2_b = g("ln2.bias");
        L.qkv_w = g("qkv.weight"); L.qkv_b = g("qkv.bias");
        L.wo_w = g("wo.weight"); L.wo_b = g("wo.bias");
        L.fc0_w = g("fc1.weight"); L.fc0_b = g("fc1.bias");
        L.fc1_w = g("fc2.weight"); L.fc1_b = g("fc2.bias");
        if (!L.ln1_w || !L.ln2_w || !L.qkv_w || !L.wo_w || !L.fc0_w || !L.fc1_w) return false;
    }
    return tok_embd && out_norm && lm_head && v_patch_w && v_pos && v_mm0_w && v_mm2_w;
}

bool Janus::alloc_kv() { return kv.alloc(backend, n_layer, n_kv_head, head_dim, n_ctx); }

bool Janus::load(const std::string & gguf_path, const std::string & tokenizer_path,
                 const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "JANUS", n_threads, use_gpu, dump_dir);
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::DeepSeekBpe)) return false;
    if (!alloc_kv()) return false;

    caps_.image = true;
    caps_.video = false;
    fprintf(stderr, "janus: ready (%dL d=%d nh=%d/%d hd=%d | vision %dL d=%d patch=%d "
                    "-> %d tokens)\n", n_layer, n_embd, n_head, n_kv_head, head_dim,
            v_layer, v_embd, v_patch, image_seq_len);
    return true;
}

// ---------------------------------------------------------------- vision

// one fixed-shape graph: 24x24 patches of one square image. The tower is the
// shared vqa::vit::block (fused qkv + biases is exactly the timm SigLIP block)
// with two Janus specifics supplied through the hooks: bidirectional flash
// attention over all 576 tokens, and the exact-erf GELU the timm MLP uses.
bool Janus::build_vision_graph() {
    if (vctx) { ggml_free(vctx); vctx = nullptr; }
    const int P = v_patch;
    const int side = v_image / P;               // 24
    const int n_patch = side * side;            // 576
    const int C3 = 3 * P * P;
    const int E = v_embd, H = v_head, hd = E / H;

    struct ggml_init_params ip = { 1024ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    vctx = ggml_init(ip);
    if (!vctx) return false;
    vgf = ggml_new_graph_custom(vctx, 32768, false);

    v_in = ggml_new_tensor_2d(vctx, GGML_TYPE_F32, C3, n_patch);
    ggml_set_name(v_in, "v_patches");

    ggml_tensor * x = ggml_mul_mat(vctx, v_patch_w, v_in);   // [E, n_patch]
    x = ggml_add(vctx, x, v_patch_b);                        // [E] broadcasts
    x = ggml_add(vctx, x, v_pos);                            // [E, n_patch]
    ggml_set_name(x, "v_patch_out");

    struct VisCtx { ggml_context * ctx; int H, hd, N; };
    VisCtx vc = { vctx, H, hd, n_patch };
    vqa::vit::BlockHooks vh;
    vh.ud = &vc;
    // heads of the [3*n_embd, N] fused qkv; block b (q/k/v) occupies rows
    // [b*n_embd, b*n_embd + n_embd). FA returns [hd, heads, tokens] laid out as
    // (head, hd) rows per token, so a plain reshape feeds wo (siglip precedent)
    vh.attention = [](void * ud, ggml_tensor * qkv) -> ggml_tensor * {
        VisCtx * c = (VisCtx *)ud;
        ggml_context * ctx = c->ctx;
        auto head = [&](int block) {
            return ggml_view_3d(ctx, qkv, c->hd, c->N, c->H, qkv->nb[1],
                                (size_t)c->hd * qkv->nb[0],
                                (size_t)block * c->H * c->hd * qkv->nb[0]);
        };
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, head(0), head(1), head(2), nullptr,
                                               1.0f / sqrtf((float)c->hd), 0.0f, 0.0f);
        return ggml_reshape_2d(ctx, fa, c->H * c->hd, c->N);
    };
    vh.gelu = [](void * ud, ggml_tensor * t) -> ggml_tensor * {
        return ggml_gelu_erf(((VisCtx *)ud)->ctx, t);   // timm nn.GELU() is the erf form
    };
    for (int il = 0; il < v_layer; il++) {
        x = vqa::vit::block(vctx, x, vl[(size_t)il], v_eps, vh, il);
        if (!x) return false;
    }

    // final LayerNorm, then the 2-layer aligner: Linear -> GELU(erf) -> Linear
    x = ggml_norm(vctx, x, v_eps);
    x = ggml_mul(vctx, x, v_post_w);
    x = ggml_add(vctx, x, v_post_b);
    ggml_set_name(x, "v_final");

    x = ggml_add(vctx, ggml_mul_mat(vctx, v_mm0_w, x), v_mm0_b);
    x = ggml_gelu_erf(vctx, x);
    x = ggml_add(vctx, ggml_mul_mat(vctx, v_mm2_w, x), v_mm2_b);
    ggml_set_name(x, "v_feats");
    v_feats = x;
    ggml_build_forward_expand(vgf, x);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "v_patches", "v_patch_out", "v_final",
                                           "v_feats", nullptr };
        vqa::dump::mark_sinks(vctx, vgf, dn);
        vqa::dump::finalize_sinks(vctx, vgf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, vgf)) {
        fprintf(stderr, "vision graph allocation failed\n");
        return false;
    }
    return true;
}

bool Janus::run_vision(const std::vector<float> & patches, std::vector<float> & feats) {
    ggml_backend_tensor_set(v_in, patches.data(), 0, sizeof(float) * patches.size());
    if (ggml_backend_graph_compute(backend, vgf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vision compute failed\n");
        return false;
    }
    if (!dump_dir.empty() && vis_runs == 0) {
        static const char * const dn[] = { "v_patches", "v_patch_out", "v_final", "v_feats",
                                           nullptr };
        for (int i = 0; dn[i]; i++) dump(vgf, dn[i]);
    }
    vis_runs++;
    feats.resize((size_t)n_embd * image_seq_len);
    ggml_backend_tensor_get(v_feats, feats.data(), 0, sizeof(float) * feats.size());
    return true;
}

// official VLMImageProcessor: bicubic+antialias resize (aspect preserved, longest
// edge -> image_size), expand2square pad with (mean*255), /255, (x-mean)/std
bool Janus::preprocess(const vqa::MediaInput & media, std::vector<float> & out) {
    if (media.empty()) return false;
    const int w0 = media.width, h0 = media.height;
    const uint8_t * rgb = media.frames[0].rgb.data();

    const int maxs = std::max(w0, h0);
    const int rw = std::max(1, (int)std::lround((double)h0 / maxs * v_image));
    const int rh = std::max(1, (int)std::lround((double)w0 / maxs * v_image));
    std::vector<float> resized;                     // [rh, rw, 3], 0-255
    vqa::img::resize_antialias(rgb, w0, h0, rw, rh, resized, /*resample bicubic*/ 3);

    // expand2square: pad with mean*255 (= the official background_color)
    const int S = v_image;
    const uint8_t bg[3] = {
        (uint8_t)(int)(v_mean[0] * 255.0f + 0.5f),
        (uint8_t)(int)(v_mean[1] * 255.0f + 0.5f),
        (uint8_t)(int)(v_mean[2] * 255.0f + 0.5f),
    };
    // torchvision resize gives (H=rw? no: size=[new_h, new_w]); we resized to
    // rw x rh with resize_antialias(w, h, rw, rh) meaning width rw, height rh,
    // matching size=[rh_h, rw_w] in the processor (height first there).
    std::vector<float> square((size_t)S * S * 3);
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            float * dst = &square[((size_t)y * S + x) * 3];
            const int sy = y - (S - rh) / 2;       // paste centred, like expand2square
            const int sx = x - (S - rw) / 2;
            if (sy >= 0 && sy < rh && sx >= 0 && sx < rw) {
                const float * src = &resized[((size_t)sy * rw + sx) * 3];
                for (int c = 0; c < 3; c++)
                    dst[c] = (src[c] / 255.0f - v_mean[c]) / v_std[c];
            } else {
                for (int c = 0; c < 3; c++)
                    dst[c] = ((float)bg[c] / 255.0f - v_mean[c]) / v_std[c];
            }
        }
    }
    if (getenv("JANUS_DUMP_PIXELS")) {
        FILE * f = fopen((dump_dir + "/dbg_pixels.bin").c_str(), "wb");
        if (f) { fwrite(square.data(), sizeof(float), square.size(), f); fclose(f); }
    }

    // unfold to patches: tokens ordered (py, px), elements (c, dh, dw)
    const int P = v_patch, side = S / P;
    const int C3 = 3 * P * P;
    out.assign((size_t)C3 * side * side, 0.0f);
    int u = 0;
    for (int py = 0; py < side; py++)
        for (int px = 0; px < side; px++) {
            float * row = out.data() + (size_t)u * C3;
            int o = 0;
            for (int c = 0; c < 3; c++)
                for (int dh = 0; dh < P; dh++) {
                    const float * s = &square[(((size_t)(py * P + dh)) * S + (px * P)) * 3 + c];
                    for (int dw = 0; dw < P; dw++)
                        row[o++] = s[(size_t)dw * 3];
                }
            u++;
        }
    return true;
}

bool Janus::encode_image(const vqa::MediaInput & media, std::vector<float> & feats) {
    std::vector<float> patches;
    if (!preprocess(media, patches)) return false;
    if (!vgf && !build_vision_graph()) return false;
    if (!run_vision(patches, feats)) return false;
    fprintf(stderr, "vision: %dx%d -> %dx%d, %d tokens\n", media.width, media.height,
            v_image, v_image, image_seq_len);
    return true;
}

// ---------------------------------------------------------------- text

bool Janus::build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_last) {
    struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 32768, false);

    g.tok_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    g.pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.pos_in, "pos_in");
    g.mask_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, S, T);
    ggml_set_name(g.mask_in, "mask_in");

    ggml_tensor * x = ggml_get_rows(g.ctx, tok_embd, g.tok_in);
    if (n_img > 0) {
        g.img_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_embd, n_img);
        ggml_set_name(g.img_in, "img_in");
        g.img_pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_img);
        ggml_set_name(g.img_pos_in, "img_pos_in");
        x = ggml_set_rows(g.ctx, ggml_dup(g.ctx, x), g.img_in, g.img_pos_in);
    }
    ggml_set_name(x, "t_emb");

    vqa::llm::LayerCtx lc;
    lc.gf = g.gf;
    lc.pos = g.pos_in;
    lc.mask = g.mask_in;
    lc.T = T; lc.S = S;
    lc.n_head = n_head; lc.n_kv_head = n_kv_head; lc.head_dim = head_dim;
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
        ? ggml_view_2d(g.ctx, x, n_embd, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]) : x;
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

bool Janus::run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<int> & pos,
                     const std::vector<float> * img_feats, const std::vector<int> * img_pos,
                     std::vector<float> & logits, int slot_base) {
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
    if (!dump_dir.empty() && slot_base == 0 && toks.size() > 1) {
        static const char * const dn[] = { "t_emb", "t_l0_norm", "t_l0_q_rope", "t_l0_attn",
                                           "t_l0_out", "t_final_norm", "logits", nullptr };
        for (int i = 0; dn[i]; i++) dump(g.gf, dn[i]);
    }
    logits.resize((size_t)n_vocab);
    ggml_backend_tensor_get(g.logits, logits.data(), 0, sizeof(float) * logits.size());
    return true;
}

void Janus::dump(const ggml_cgraph * gf, const char * nm) {
    if (dump_dir.empty()) return;
    vqa::dump::write((ggml_cgraph *)gf, dump_dir.c_str(), nm);
}

// ---------------------------------------------------------------- ask

bool Janus::ask(const vqa::MediaInput & media, const std::string & prompt,
                const std::string & /*mode*/, const vqa::GenParams & gp,
                std::string & out_text) {
    out_text.clear();
    std::vector<float> feats;
    if (!media.empty()) {
        if (gp.on_stage) gp.on_stage("vision", 0, 1, gp.on_stage_ud);
        const auto tv0 = std::chrono::steady_clock::now();
        if (!encode_image(media, feats)) return false;
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tv0).count();
        fprintf(stderr, "[perf] janus vision: %.0f ms\n", ms);
        if (gp.on_stage) gp.on_stage("vision", 1, 1, gp.on_stage_ud);
    }

    // official deepseek sft format: role ": " + content, sep "\n\n", assistant
    // turn opens with "<|Assistant|>:" and closes with the EOS token
    std::string text = "<|User|>: " + prompt;
    if (!feats.empty()) {
        text += "\n";
        for (int i = 0; i < image_seq_len; i++) text += "<image_placeholder>";
    }
    text += "\n\n<|Assistant|>:";

    std::vector<int> toks = tok_.encode(text);
    // the official tokenizer call prepends BOS (add_special_tokens=True)
    if (toks.empty() || toks[0] != 100000) toks.insert(toks.begin(), 100000);
    const int T = (int)toks.size();
    if (T + gp.max_tokens + 8 > n_ctx) {
        fprintf(stderr, "prompt %d + %d tokens exceed n_ctx %d\n", T, gp.max_tokens, n_ctx);
        return false;
    }
    std::vector<int> pos((size_t)T);
    for (int i = 0; i < T; i++) pos[(size_t)i] = i;

    std::vector<int> img_pos;
    if (!feats.empty()) {
        for (int i = 0; i < T && (int)img_pos.size() < image_seq_len; i++)
            if (toks[(size_t)i] == id_image) img_pos.push_back(i);
        if ((int)img_pos.size() != image_seq_len) {
            fprintf(stderr, "image token mismatch: %zu positions for %d features\n",
                    img_pos.size(), image_seq_len);
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
    fprintf(stderr, "[perf] janus prefill: %.0f ms for %d tokens\n",
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - tp0).count(), T);

    if (getenv("JANUS_DEBUG")) {
        fprintf(stderr, "tokens(%d) of which %zu image:", T, img_pos.size());
        for (int i = 0; i < std::min(T, 16); i++) fprintf(stderr, " %d", toks[(size_t)i]);
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

}  // namespace janus

namespace vqa {
Model * create_janus() { return new janus::Janus(); }
}  // namespace vqa
