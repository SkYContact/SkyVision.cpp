// nemotron_parse_model.cpp - NVIDIA Nemotron-3.5-Parse backend: a C-RADIOv4-SO400M
// vision tower + conv neck feeding a 10 layer mBART-style decoder that emits the
// document markup (text + layout classes + normalized boxes, all as tokens).
//
// Shape of the thing (0.63B checkpoint, everything F32 in the release):
//   vision  ViT: Linear(768 -> 1152) over 16x16 patches (the processor already
//           applied CLIP normalization), plus a learned 128x128 position field
//           resampled to the page's content grid (np_geometry.hpp), 10 learned
//           class tokens concatenated at the front, then 27 timm Blocks:
//           LayerNorm(1e-6, +bias) -> fused qkv(+bias, no rope, full attention)
//           -> wo -> residual -> LayerNorm -> MLP 1152->4304->1152 with GELU(erf).
//           NOTE: there is no final norm - the block stack output goes straight
//           into the neck, and class tokens 0/1 also become the page summary.
//   neck     Conv1d(1152 -> 1024, k=1)+bias -> LN -> reshape to the patch grid ->
//           Conv2d(1024, 1024, k=(1,4), stride=(1,4)) (width merge x4, folded to a
//           matmul by the converter) -> LN; summary = concat(cls0, cls1) [2304] ->
//           Linear(2304 -> 1024) -> LN -> appended as the LAST visual token.
//   text    mBART prefix decoder: 10 layers, d=1024, 16 q / 8 kv heads (head_dim
//           64), pre-LN: x += attn(LN1(x)); x += fc2(gelu(fc1(LN2(x)))).
//           **no rope and no positional embedding at all** - order comes from the
//           causal mask only, and `pos` exists purely to index the KV write-back.
//           Embeddings are scaled by sqrt(1024); one LayerNorm runs over the
//           concatenated [visual prefix | text] stream before layer 0; a final
//           LayerNorm feeds a tied lm_head (vocabulary 72512).
//
// The visual features are the KV prefix: text tokens attend to all of them, they
// only attend to themselves. Multi-page inputs are handled by the caller
// (one ask per page) - the reference packs pages into one batch with a mask.
#include "core/vqa.hpp"

#include "core/blocks/vit_block.hpp"
#include "core/engine/attention.hpp"
#include "core/engine/backend.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/gguf_util.hpp"
#include "core/engine/sampler.hpp"
#include "core/engine/weight_load.hpp"
#include "core/media/resample.hpp"
#include "core/runtime/load_util.hpp"
#include "np_geometry.hpp"
#include "np_markup.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace nemotron_parse {

// The only prompt this checkpoint answers: request boxes, classes, markdown and
// text-in-picture (the official `DEFAULT_PROMPT` in run_transformers_inference.py).
// It is entirely made of control tokens, so it tokenizes to 6 ids.
static const char * kPrompt =
    "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>";

// FNV-1a over the prompt and the page pixels: the repeat-page cache key.
static uint64_t hash_page(const vqa::MediaInput & m, const std::string & prompt) {
    if (m.empty()) return 0;
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void * p, size_t n) {
        const uint8_t * b = (const uint8_t *)p;
        for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    };
    mix(prompt.data(), prompt.size());
    const int32_t wh[2] = { m.width, m.height };
    mix(wh, sizeof(wh));
    for (const auto & f : m.frames) mix(f.rgb.data(), f.rgb.size());
    return h ? h : 1;
}

struct VisLayer {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr, *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *wqkv = nullptr, *bqkv = nullptr;
    ggml_tensor *wo = nullptr, *bo = nullptr;
    ggml_tensor *fc0_w = nullptr, *fc0_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
};

struct DecLayer {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;
    // q|k|v fused into one projection ([n_qkv, n_embd]): one GEMV per layer
    // instead of three, then sliced with views
    ggml_tensor *wqkv = nullptr, *bqkv = nullptr;
    ggml_tensor *wo = nullptr, *bo = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
};

struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *pos_in = nullptr, *mask_in = nullptr;
    ggml_tensor *vis_in = nullptr;
    ggml_tensor *logits = nullptr;
};

class NemotronParse : public vqa::Model {
public:
    ~NemotronParse() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "nemotron_parse"; }
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp,
             std::string & out_text) override;

    // "json" / "blocks" | "markdown" / "md" | "text" / "plain" render the parsed
    // answer; anything else (default) returns the raw control-token stream.
    static std::string render_answer(const std::string & raw, const std::string & mode,
                                     int img_w, int img_h);

    int n_ctx = 8192;
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool alloc_kv();

    bool preprocess(const vqa::MediaInput & media, std::vector<float> & patches,
                    std::vector<float> & positions, NavitGrid & grid);
    bool build_vision_graph(int n_patch, int rows, int cols);
    bool encode_image(const vqa::MediaInput & media, std::vector<float> & feats, int * n_vis);

    bool build_text_graph(TextGraph & g, int T_tok, int n_vis, int S, bool logits_last);
    bool run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<int> & pos,
                  const float * vis, int n_vis, std::vector<float> & logits);

    ggml_tensor * dec_layer(TextGraph & g, ggml_tensor * x, const DecLayer & L, int il,
                            int T, int S);
    void dump(const ggml_cgraph * gf, const char * nm);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    // decoder
    int n_layer = 10, n_embd = 1024, n_head = 16, n_kv_head = 8, head_dim = 64;
    int n_ff = 4096, n_vocab = 72512;
    float ln_eps = 1e-5f, embed_scale = 32.0f;
    int id_eos = 2, id_pad = 1, id_bos = 0;
    float rep_penalty = 1.1f;   // generation_config.json

    // vision tower
    int v_layer = 27, v_embd = 1152, v_head = 16, v_ff = 4304;
    int v_patch = 16, v_pos_side = 128, v_cls = 10, v_summary = 2;
    float v_eps = 1e-6f;
    int neck_dim = 1024, neck_merge = 4;

    // preprocessing canvas (the processor's longest-edge box)
    int canvas_h = 2048, canvas_w = 1664;
    float v_mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };
    float v_std[3] = { 0.26862954f, 0.26130258f, 0.27577711f };

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    vqa::MMap mm;
    vqa::AlignedBuf host_f32;
    vqa::AlignedBuf host_fold;                 // folded normalization weights
    vqa::weights::LnFoldPlan fold;
    bool ln_folded = false;                    // norms' affine already in the weights
    ggml_gallocr_t allocr = nullptr;

    // decoder weights
    ggml_tensor *tok_embd = nullptr, *emb_norm_w = nullptr, *emb_norm_b = nullptr;
    ggml_tensor *out_norm_w = nullptr, *out_norm_b = nullptr;
    std::vector<DecLayer> dl;

    // vision weights
    ggml_tensor *v_patch_w = nullptr, *v_cls_tok = nullptr;
    std::vector<VisLayer> vl;
    ggml_tensor *n_conv1_w = nullptr, *n_conv1_b = nullptr;
    ggml_tensor *n_ln1_w = nullptr, *n_ln1_b = nullptr;
    ggml_tensor *n_conv2_w = nullptr;
    ggml_tensor *n_ln2_w = nullptr, *n_ln2_b = nullptr;
    ggml_tensor *n_sum_w = nullptr, *n_sum_b = nullptr;
    ggml_tensor *n_ln3_w = nullptr, *n_ln3_b = nullptr;
    std::vector<float> pos_table;   // host copy of the 128x128 position field

    // vision graph (rebuilt whenever the patch grid changes)
    ggml_context * vctx = nullptr;
    ggml_cgraph * vgf = nullptr;
    ggml_tensor *v_in = nullptr, *v_pos_in = nullptr, *v_feats = nullptr;
    int v_n_patch = 0;

    vqa::KvCache kv;
    int n_past = 0;
    double ms_compute = 0.0;   // accumulated graph_compute time (build vs compute split)

    // Build-once decode graph. The KV window is bucketed so the graph keeps a
    // CONSTANT shape across steps: that kills the per-token build+alloc cost and
    // is load-bearing for ggml-cuda's CUDA graph cache (keyed on cgraph->nodes[0],
    // so a fresh ggml_context per step would never replay).
    TextGraph dec_g;
    int dec_g_bucket = -1;
    // Repeat-page cache. The vision tower is the expensive half (17 s on CPU), so
    // features are kept for the last two pages; the first-step logits and the
    // prefix length are only valid for the page that currently OWNS the KV cache.
    struct PageCache {
        uint64_t hash = 0;
        std::vector<float> feats;
        int n_vis = 0;
        std::vector<float> first_logits;   // valid only while kv_owner matches
        int prefix = 0;
        bool kv_owner = false;
    };
    std::vector<PageCache> pages;         // at most 2

    uint64_t rng_state = 0x243F6A8885A308D3ull;
};

// ---------------------------------------------------------------- lifecycle

NemotronParse::~NemotronParse() {
    kv.free();
    if (dec_g.ctx) { ggml_free(dec_g.ctx); dec_g.ctx = nullptr; }
    if (vctx) ggml_free(vctx);
    if (allocr) ggml_gallocr_free(allocr);
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (backend) ggml_backend_free(backend);
}

bool NemotronParse::init_backend() {
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

bool NemotronParse::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    auto u32 = [&](const char * k, int def) { return vqa::gguf::u32(gu, k, def); };
    auto f32v = [&](const char * k, float def) { return vqa::gguf::f32(gu, k, def); };

    n_layer    = u32("nemotron_parse.block_count", n_layer);
    n_embd     = u32("nemotron_parse.embedding_length", n_embd);
    n_head     = u32("nemotron_parse.attention.head_count", n_head);
    n_kv_head  = u32("nemotron_parse.attention.head_count_kv", n_kv_head);
    head_dim   = u32("nemotron_parse.attention.key_length", head_dim);
    n_ff       = u32("nemotron_parse.feed_forward_length", n_ff);
    n_vocab    = u32("nemotron_parse.vocab_size", n_vocab);
    ln_eps     = f32v("nemotron_parse.attention.layer_norm_epsilon", ln_eps);
    embed_scale = f32v("nemotron_parse.embedding_scale", embed_scale);
    id_eos     = u32("nemotron_parse.eos_token_id", id_eos);
    id_pad     = u32("nemotron_parse.pad_token_id", id_pad);
    id_bos     = u32("nemotron_parse.bos_token_id", id_bos);

    v_layer    = u32("nemotron_parse.vision.block_count", v_layer);
    v_embd     = u32("nemotron_parse.vision.embedding_length", v_embd);
    v_head     = u32("nemotron_parse.vision.attention.head_count", v_head);
    v_ff       = u32("nemotron_parse.vision.feed_forward_length", v_ff);
    v_patch    = u32("nemotron_parse.vision.patch_size", v_patch);
    v_eps      = f32v("nemotron_parse.vision.layer_norm_epsilon", v_eps);
    v_pos_side = u32("nemotron_parse.vision.pos_source", v_pos_side);
    v_cls      = u32("nemotron_parse.vision.class_tokens", v_cls);
    v_summary  = u32("nemotron_parse.vision.summary_tokens", v_summary);
    canvas_h   = u32("nemotron_parse.vision.image_height", canvas_h);
    canvas_w   = u32("nemotron_parse.vision.image_width", canvas_w);
    neck_dim   = u32("nemotron_parse.vision.neck_length", neck_dim);
    neck_merge = u32("nemotron_parse.vision.neck_merge", neck_merge);
    {
        int64_t id = gguf_find_key(gu, "nemotron_parse.vision.image_mean");
        if (id >= 0 && gguf_get_arr_n(gu, id) == 3) {
            const float * a = (const float *)gguf_get_arr_data(gu, id);
            for (int c = 0; c < 3; c++) v_mean[c] = a[c];
        }
        id = gguf_find_key(gu, "nemotron_parse.vision.image_std");
        if (id >= 0 && gguf_get_arr_n(gu, id) == 3) {
            const float * a = (const float *)gguf_get_arr_data(gu, id);
            for (int c = 0; c < 3; c++) v_std[c] = a[c];
        }
    }

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);

    vqa::weights::promote_1d_to_f32(wctx, host_f32);
    {
        // Fold every normalization's affine into the matmul that consumes it (see
        // weight_load.hpp): norm(x)*gamma+beta followed by W y + b becomes
        // (W diag gamma) norm(x) + (b + W beta), so the graphs skip the mul/add.
        // Device backends by default, CPU on request (the weights stop being
        // zero-copy mmap pages there).
        if (vqa::weights::ln_fold_enabled(ggml_backend_is_cpu(backend))) {
            auto g = [&](const char * n) { return ggml_get_tensor(wctx, n); };
            for (int i = 0; i < n_layer; i++) {
                char b[64];
                auto lay = [&](const char * suf) {
                    snprintf(b, sizeof(b), "t.blk.%d.%s", i, suf);
                    return g(b);
                };
                fold.add(lay("wqkv.weight"), lay("ln1.weight"), lay("ln1.bias"), lay("bqkv"));
                fold.add(lay("fc1.weight"), lay("ln2.weight"), lay("ln2.bias"), lay("fc1.bias"));
            }
            for (int i = 0; i < v_layer; i++) {
                char b[64];
                auto lay = [&](const char * suf) {
                    snprintf(b, sizeof(b), "v.blk.%d.%s", i, suf);
                    return g(b);
                };
                fold.add(lay("wqkv.weight"), lay("ln1.weight"), lay("ln1.bias"), lay("bqkv"));
                fold.add(lay("fc0.weight"), lay("ln2.weight"), lay("ln2.bias"), lay("fc0.bias"));
            }
            fold.apply(host_fold);
            ln_folded = !fold.empty();
        }
        vqa::weights::Bound bound = vqa::weights::bind_to_backend(
            wctx, backend, mm,
            { { host_f32.data(), host_f32.size() }, { host_fold.data(), host_fold.size() } });
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
    emb_norm_w = gt("t.emb_norm.weight");
    emb_norm_b = gt("t.emb_norm.bias");
    out_norm_w = gt("t.norm.weight");
    out_norm_b = gt("t.norm.bias");
    dl.resize((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "t.blk.%d.%s", i, suf); return gt(b); };
        DecLayer & L = dl[(size_t)i];
        L.ln1_w = g("ln1.weight"); L.ln1_b = g("ln1.bias");
        L.wqkv = g("wqkv.weight"); L.bqkv = g("bqkv");
        L.wo = g("wo.weight"); L.bo = g("bo");
        L.ln2_w = g("ln2.weight"); L.ln2_b = g("ln2.bias");
        L.fc1_w = g("fc1.weight"); L.fc1_b = g("fc1.bias");
        L.fc2_w = g("fc2.weight"); L.fc2_b = g("fc2.bias");
        if (!L.ln1_w || !L.wqkv || !L.wo || !L.fc1_w || !L.fc2_w) return false;
    }

    v_patch_w = gt("v.patch_embd.weight");
    v_cls_tok = gt("v.cls_token.weight");
    vl.resize((size_t)v_layer);
    for (int i = 0; i < v_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "v.blk.%d.%s", i, suf); return gt(b); };
        VisLayer & L = vl[(size_t)i];
        L.ln1_w = g("ln1.weight"); L.ln1_b = g("ln1.bias");
        L.ln2_w = g("ln2.weight"); L.ln2_b = g("ln2.bias");
        L.wqkv = g("wqkv.weight"); L.bqkv = g("bqkv");
        L.wo = g("wo.weight"); L.bo = g("bo");
        L.fc0_w = g("fc0.weight"); L.fc0_b = g("fc0.bias");
        L.fc1_w = g("fc1.weight"); L.fc1_b = g("fc1.bias");
        if (!L.ln1_w || !L.wqkv || !L.wo || !L.fc0_w || !L.fc1_w) return false;
    }
    n_conv1_w = gt("v.neck.conv1.weight");
    n_conv1_b = gt("v.neck.conv1.bias");
    n_ln1_w = gt("v.neck.ln1.weight");
    n_ln1_b = gt("v.neck.ln1.bias");
    n_conv2_w = gt("v.neck.conv2.weight");
    n_ln2_w = gt("v.neck.ln2.weight");
    n_ln2_b = gt("v.neck.ln2.bias");
    n_sum_w = gt("v.neck.sum_proj.weight");
    n_sum_b = gt("v.neck.sum_proj.bias");
    n_ln3_w = gt("v.neck.ln3.weight");
    n_ln3_b = gt("v.neck.ln3.bias");
    if (!tok_embd || !emb_norm_w || !out_norm_w || !v_patch_w || !v_cls_tok ||
        !n_conv1_w || !n_conv2_w || !n_sum_w || !n_ln3_w) return false;

    // the position field is interpolated on the host, so keep an f32 copy.
    // GGUF stores it token-major ([1152, 16384]: 1152 values per patch slot) while
    // np_geometry works on [dim][side*side] planes, so transpose once here.
    ggml_tensor * pos = gt("v.pos_embd.weight");
    if (!pos) return false;
    {
        std::vector<float> raw;
        vqa::weights::snapshot_to_f32_host(pos, raw);
        const size_t side2 = (size_t)v_pos_side * v_pos_side;
        const size_t dim = (size_t)v_embd;
        if (raw.size() != side2 * dim) {
            fprintf(stderr, "position field has %zu values, expected %zu\n", raw.size(), side2 * dim);
            return false;
        }
        pos_table.resize(raw.size());
        for (size_t idx = 0; idx < side2; idx++)
            for (size_t c = 0; c < dim; c++)
                pos_table[c * side2 + idx] = raw[idx * dim + c];
    }
    return true;
}

bool NemotronParse::alloc_kv() { return kv.alloc(backend, n_layer, n_kv_head, head_dim, n_ctx); }

bool NemotronParse::load(const std::string & gguf_path, const std::string & tokenizer_path,
                         const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "NEMOTRON_PARSE", n_threads, use_gpu, dump_dir);
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::Gpt2Regex)) return false;
    // this model's answer IS control tokens (<x_0.5>, <class_Text>, ...), so the
    // decoder must not drop added tokens the way a chat backend does
    tok_.decode_added = true;
    if (!alloc_kv()) return false;

    caps_.image = true;
    caps_.video = false;
    caps_.detect = false;
    fprintf(stderr, "nemotron_parse: ready (%dL d=%d nh=%d/%d hd=%d | vision %dL d=%d "
                    "patch=%d cls=%d | neck %d merge=%d | canvas %dx%d)\n",
            n_layer, n_embd, n_head, n_kv_head, head_dim, v_layer, v_embd, v_patch,
            v_cls, neck_dim, neck_merge, canvas_w, canvas_h);
    return true;
}

// ---------------------------------------------------------------- vision

bool NemotronParse::build_vision_graph(int n_patch, int rows, int cols) {
    if (vctx) { ggml_free(vctx); vctx = nullptr; }
    vqa::dump::dump_registry_clear();
    const int E = v_embd, P = v_patch, N = v_cls + n_patch;

    // no_alloc: this context only holds node metadata (~700 tensors), so 32 MB is
    // plenty - ggml_init mallocs the arena eagerly
    struct ggml_init_params ip = { 32ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    vctx = ggml_init(ip);
    if (!vctx) return false;
    vgf = ggml_new_graph_custom(vctx, 32768, false);

    v_in = ggml_new_tensor_2d(vctx, GGML_TYPE_F32, 3 * P * P, n_patch);
    ggml_set_name(v_in, "v_patches");
    v_pos_in = ggml_new_tensor_2d(vctx, GGML_TYPE_F32, E, n_patch);
    ggml_set_name(v_pos_in, "v_positions");

    ggml_tensor * x = ggml_mul_mat(vctx, v_patch_w, v_in);
    x = ggml_add(vctx, x, v_pos_in);            // learned field, resampled on the host
    ggml_set_name(x, "v_patch_out");
    // class tokens first, exactly like `torch.cat((cls, patches), dim=1)`
    x = ggml_concat(vctx, v_cls_tok, x, 1);

    // ---- shared ViT block: no rope, no mask, GELU(erf), LayerNorm with bias
    vqa::vit::VitHookEnv venv;
    venv.gctx = vctx;
    venv.gf = vgf;
    venv.n_head = v_head;
    venv.n_embd = E;
    venv.N = N;
    venv.vhd = E / v_head;
    venv.vhd_f = venv.vhd;
    // head_dim has no padding here, so the FA output can feed wo through a plain
    // reshape - without this the hook takes the de-pad path and copies it
    venv.wo_padded = true;
    venv.no_rope = true;
    venv.dbg_names = !dump_dir.empty();
    vqa::vit::BlockHooks vh;
    vh.drop_ln_mul = ln_folded;   // gamma already folded into wqkv / fc0
    vh.drop_ln_add = ln_folded;
    vh.attention = vqa::vit::vit_attention_hook;
    vh.gelu = [](void * ud, ggml_tensor * t) -> ggml_tensor * {
        // timm's nn.GELU() (approximate="none") is the erf form
        return ggml_gelu_erf(((vqa::vit::VitHookEnv *)ud)->gctx, t);
    };
    vh.probe_layer0 = vqa::vit::vit_probe_layer0_hook;
    vh.ud = &venv;
    for (int il = 0; il < v_layer; il++) {
        vqa::vit::BlockWeights w;
        const VisLayer & L = vl[(size_t)il];
        w.ln1_w = L.ln1_w; w.ln1_b = L.ln1_b;
        w.qkv_w = L.wqkv; w.qkv_b = L.bqkv;
        w.wo_w = L.wo; w.wo_b = L.bo;
        w.ln2_w = L.ln2_w; w.ln2_b = L.ln2_b;
        w.fc0_w = L.fc0_w; w.fc0_b = L.fc0_b;
        w.fc1_w = L.fc1_w; w.fc1_b = L.fc1_b;
        venv.layer = il;
        x = vqa::vit::block(vctx, x, w, v_eps, vh, il);
        if (!x) { fprintf(stderr, "vision block %d failed\n", il); return false; }
    }
    ggml_set_name(x, "v_final");

    auto layer_norm = [&](ggml_tensor * t, ggml_tensor * w, ggml_tensor * b) {
        t = ggml_norm(vctx, t, v_eps);
        t = ggml_mul(vctx, t, w);
        return ggml_add(vctx, t, b);
    };

    // ---- summary token: class tokens 0 and 1, concatenated along the feature
    // axis. They are adjacent in memory (token stride == nb[1]), so one view
    // expresses the concat and no copy is needed.
    ggml_tensor * sum_in = ggml_view_2d(vctx, x, (int64_t)E * v_summary, 1, x->nb[1], 0);
    ggml_tensor * sum = ggml_add(vctx, ggml_mul_mat(vctx, n_sum_w, sum_in), n_sum_b);
    sum = layer_norm(sum, n_ln3_w, n_ln3_b);

    // ---- patch features -> neck
    ggml_tensor * fx = ggml_view_2d(vctx, x, E, n_patch, x->nb[1], (size_t)v_cls * x->nb[1]);
    ggml_tensor * f = ggml_add(vctx, ggml_mul_mat(vctx, n_conv1_w, fx), n_conv1_b);
    f = layer_norm(f, n_ln1_w, n_ln1_b);

    // width merge x4: 4 adjacent patch columns become one token (the converter
    // laid the conv2 weight out as k*neck_dim + in_channel to match this reshape)
    const int n2 = n_patch / neck_merge;
    if (n_patch % neck_merge) { fprintf(stderr, "patch grid not divisible by merge\n"); return false; }
    f = ggml_reshape_2d(vctx, f, (int64_t)neck_dim * neck_merge, n2);
    f = ggml_mul_mat(vctx, n_conv2_w, f);
    f = layer_norm(f, n_ln2_w, n_ln2_b);
    ggml_set_name(f, "v_neck");

    v_feats = ggml_concat(vctx, f, sum, 1);   // [neck_dim, rows*cols/4 + 1]
    ggml_set_name(v_feats, "v_feats");
    ggml_build_forward_expand(vgf, v_feats);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "v_patches", "v_positions", "v_patch_out",
                                           "v_blk0_out", "v_final", "v_neck", "v_feats",
                                           nullptr };
        vqa::dump::mark_sinks(vctx, vgf, dn);
        vqa::dump::finalize_sinks(vctx, vgf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, vgf)) {
        fprintf(stderr, "vision graph allocation failed\n");
        return false;
    }
    if (getenv("NEMOTRON_PARSE_PROFILE")) {
        std::map<std::string, int> hist;
        for (int i = 0; i < ggml_graph_n_nodes(vgf); i++)
            hist[ggml_op_name(ggml_graph_node(vgf, i)->op)]++;
        fprintf(stderr, "[graph] vision n_patch=%d: %d nodes\n", n_patch,
                ggml_graph_n_nodes(vgf));
        for (const auto & e : hist)
            fprintf(stderr, "[graph]   %-16s %d\n", e.first.c_str(), e.second);
    }
    v_n_patch = n_patch;
    (void)rows; (void)cols;
    return true;
}

bool NemotronParse::preprocess(const vqa::MediaInput & media, std::vector<float> & patches,
                               std::vector<float> & positions, NavitGrid & grid) {
    if (media.empty() || media.frames.empty()) return false;
    const int w0 = media.width, h0 = media.height;
    const uint8_t * rgb = media.frames[0].rgb.data();

    // the processor resizes into a (canvas_h, canvas_w) box, only ever shrinking;
    // NEMOTRON_PARSE_MAX_EDGE scales that box down for CPU runs (the tower is
    // O(n_patch^2) per layer, so this is the one knob that matters)
    int max_h = canvas_h, max_w = canvas_w;
    if (const char * e = getenv("NEMOTRON_PARSE_MAX_EDGE")) {
        const int cap = atoi(e);
        if (cap > 0) {
            const double s = (double)cap / (double)canvas_h;
            max_h = std::max(16, (int)std::lround(canvas_h * s));
            max_w = std::max(64, (int)std::lround(canvas_w * s));
        }
    }
    int cw = 0, ch = 0;
    aspect_size(w0, h0, max_h, max_w, &cw, &ch);

    std::vector<uint8_t> resized;
    if (cw == w0 && ch == h0) {
        resized.assign(rgb, rgb + (size_t)w0 * h0 * 3);
    } else {
        resized.assign((size_t)cw * ch * 3, 0);
        vqa::img::resize_bilinear_coeffs(rgb, w0, h0, 3, cw, ch, resized.data());
    }

    grid = make_grid(cw, ch, v_patch);
    pack_patches(resized.data(), cw, ch, grid, v_mean, v_std, v_patch, patches);

    build_positions(pos_table.data(), v_embd, v_pos_side, grid, v_patch, positions);
    fprintf(stderr, "vision: %dx%d -> %dx%d content, grid %dx%d patches (%dx%d tokens/row "
                    "after merge) = %d visual tokens\n",
            w0, h0, cw, ch, grid.rows, grid.cols, grid.rows, grid.cols / neck_merge,
            grid.rows * grid.cols / neck_merge + 1);
    return true;
}

bool NemotronParse::encode_image(const vqa::MediaInput & media, std::vector<float> & feats,
                                 int * n_vis) {
    std::vector<float> patches, positions;
    NavitGrid grid;
    const auto t0 = std::chrono::steady_clock::now();
    if (!preprocess(media, patches, positions, grid)) return false;
    const int n_patch = grid.rows * grid.cols;
    if (!vgf || v_n_patch != n_patch) {
        if (!build_vision_graph(n_patch, grid.rows, grid.cols)) return false;
    }
    ggml_backend_tensor_set(v_in, patches.data(), 0, sizeof(float) * patches.size());
    ggml_backend_tensor_set(v_pos_in, positions.data(), 0, sizeof(float) * positions.size());
    if (ggml_backend_graph_compute(backend, vgf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vision compute failed\n");
        return false;
    }
    if (!dump_dir.empty()) {
        static const char * const dn[] = { "v_patches", "v_positions", "v_patch_out",
                                           "v_blk0_out", "v_final", "v_neck", "v_feats",
                                           nullptr };
        for (int i = 0; dn[i]; i++) dump(vgf, dn[i]);
    }
    const int d = (int)v_feats->ne[0];
    const int n = (int)v_feats->ne[1];
    feats.resize((size_t)d * n);
    ggml_backend_tensor_get(v_feats, feats.data(), 0, sizeof(float) * feats.size());
    *n_vis = n;
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[perf] nemotron_parse vision: %.0f ms for %d tokens\n", ms, n);
    return true;
}

// ---------------------------------------------------------------- text

// one pre-LN mBART decoder layer; no rope, `pos_in` only indexes the KV write
ggml_tensor * NemotronParse::dec_layer(TextGraph & g, ggml_tensor * x, const DecLayer & L,
                                       int il, int T, int S) {
    ggml_context * c = g.ctx;
    const int HD = head_dim, NH = n_head, NKV = n_kv_head;

    ggml_tensor * l = ggml_norm(c, x, ln_eps);
    if (!ln_folded) {   // the fold pushed gamma into wqkv and W*beta into bqkv
        l = ggml_mul(c, l, L.ln1_w);
        l = ggml_add(c, l, L.ln1_b);
    }

    // one projection for q|k|v: [NH*HD + 2*NKV*HD, T], then slice
    const int qd = NH * HD, kvd = NKV * HD;
    ggml_tensor * qkv = ggml_add(c, ggml_mul_mat(c, L.wqkv, l), L.bqkv);
    ggml_tensor * qs = ggml_view_2d(c, qkv, qd, T, qkv->nb[1], 0);
    ggml_tensor * ks = ggml_view_2d(c, qkv, kvd, T, qkv->nb[1], (size_t)qd * qkv->nb[0]);
    ggml_tensor * vs = ggml_view_2d(c, qkv, kvd, T, qkv->nb[1],
                                    (size_t)(qd + kvd) * qkv->nb[0]);
    // set_rows wants row-contiguous sources (and FA wants a contiguous q), so the
    // slices are materialised - three cheap copies against three extra GEMVs
    ggml_tensor * q = ggml_cont(c, qs);
    ggml_tensor * k2 = ggml_cont(c, ks);
    ggml_tensor * v2 = ggml_cont(c, vs);
    ggml_build_forward_expand(g.gf, ggml_set_rows(c, kv.k[(size_t)il], k2, g.pos_in));
    ggml_build_forward_expand(g.gf, ggml_set_rows(c, kv.v[(size_t)il], v2, g.pos_in));

    ggml_tensor * q3 = ggml_reshape_3d(c, q, HD, NH, T);
    ggml_tensor * qf = ggml_view_3d(c, q3, HD, T, NH, q3->nb[2], q3->nb[1], 0);
    ggml_tensor * kf = ggml_view_3d(c, kv.k[(size_t)il], HD, S, NKV,
                                    kv.k[(size_t)il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
    ggml_tensor * vf = ggml_view_3d(c, kv.v[(size_t)il], HD, S, NKV,
                                    kv.v[(size_t)il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
    ggml_tensor * fa = ggml_flash_attn_ext(c, qf, kf, vf, g.mask_in,
                                           1.0f / sqrtf((float)HD), 0.0f, 0.0f);
    // the FA output is already contiguous ([hd, h, T] -> [hd*h, T]), so no copy
    ggml_tensor * attn = ggml_reshape_2d(c, fa, NH * HD, T);
    attn = ggml_add(c, ggml_mul_mat(c, L.wo, attn), L.bo);
    x = ggml_add(c, x, attn);
    if (il == 0) ggml_set_name(x, "t_l0_resid");

    ggml_tensor * m = ggml_norm(c, x, ln_eps);
    if (!ln_folded) {
        m = ggml_mul(c, m, L.ln2_w);
        m = ggml_add(c, m, L.ln2_b);
    }
    m = ggml_add(c, ggml_mul_mat(c, L.fc1_w, m), L.fc1_b);
    m = ggml_gelu_erf(c, m);
    m = ggml_add(c, ggml_mul_mat(c, L.fc2_w, m), L.fc2_b);
    x = ggml_add(c, x, m);
    if (il == 0) ggml_set_name(x, "t_l0_out");
    return x;
}

bool NemotronParse::build_text_graph(TextGraph & g, int T_tok, int n_vis, int S,
                                     bool logits_last) {
    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();   // one graph per decode step
    struct ggml_init_params ip = { 32ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 32768, false);

    const int D = n_embd;
    const int T = n_vis + T_tok;

    g.tok_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T_tok);
    ggml_set_name(g.tok_in, "tok_in");
    g.pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.pos_in, "pos_in");
    g.mask_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, S, T);
    ggml_set_name(g.mask_in, "mask_in");

    ggml_tensor * x = ggml_get_rows(g.ctx, tok_embd, g.tok_in);   // [D, T_tok]
    x = ggml_scale(g.ctx, x, embed_scale);
    if (n_vis > 0) {
        g.vis_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, n_vis);
        ggml_set_name(g.vis_in, "vis_in");
        x = ggml_concat(g.ctx, g.vis_in, x, 1);                   // [D, n_vis + T_tok]
    }
    x = ggml_norm(g.ctx, x, ln_eps);
    x = ggml_mul(g.ctx, x, emb_norm_w);
    x = ggml_add(g.ctx, x, emb_norm_b);
    ggml_set_name(x, "t_emb");

    for (int il = 0; il < n_layer; il++) x = dec_layer(g, x, dl[(size_t)il], il, T, S);

    x = ggml_norm(g.ctx, x, ln_eps);
    x = ggml_mul(g.ctx, x, out_norm_w);
    x = ggml_add(g.ctx, x, out_norm_b);
    ggml_set_name(x, "t_final_norm");

    ggml_tensor * last = logits_last
        ? ggml_view_2d(g.ctx, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]) : x;
    g.logits = ggml_mul_mat(g.ctx, tok_embd, last);   // tied lm_head
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "t_emb", "t_l0_resid", "t_l0_out",
                                           "t_final_norm", "logits", nullptr };
        vqa::dump::mark_sinks(g.ctx, g.gf, dn);
        vqa::dump::finalize_sinks(g.ctx, g.gf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, g.gf)) {
        fprintf(stderr, "text graph allocation failed (S=%d T=%d)\n", S, T);
        return false;
    }
    if (getenv("NEMOTRON_PARSE_PROFILE")) {
        static bool dumped = false;
        if (!dumped) {   // op histogram of the decode shape, once per process
            dumped = true;
            std::map<std::string, int> hist;
            for (int i = 0; i < ggml_graph_n_nodes(g.gf); i++)
                hist[ggml_op_name(ggml_graph_node(g.gf, i)->op)]++;
            fprintf(stderr, "[graph] text T=%d: %d nodes\n", T, ggml_graph_n_nodes(g.gf));
            for (const auto & e : hist)
                fprintf(stderr, "[graph]   %-16s %d\n", e.first.c_str(), e.second);
        }
    }
    return true;
}

bool NemotronParse::run_text(TextGraph & g, const std::vector<int> & toks,
                             const std::vector<int> & pos, const float * vis, int n_vis,
                             std::vector<float> & logits) {
    ggml_backend_tensor_set(g.tok_in, toks.data(), 0, sizeof(int) * toks.size());
    ggml_backend_tensor_set(g.pos_in, pos.data(), 0, sizeof(int) * pos.size());
    if (g.vis_in && vis && n_vis > 0)
        ggml_backend_tensor_set(g.vis_in, vis, 0, sizeof(float) * (size_t)n_embd * n_vis);
    const auto t0 = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "text compute failed\n");
        return false;
    }
    ms_compute += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    logits.resize((size_t)n_vocab);
    ggml_backend_tensor_get(g.logits, logits.data(), 0, sizeof(float) * logits.size());
    return true;
}

void NemotronParse::dump(const ggml_cgraph * gf, const char * nm) {
    if (dump_dir.empty()) return;
    vqa::dump::write((ggml_cgraph *)gf, dump_dir.c_str(), nm);
}

// ---------------------------------------------------------------- ask

// HF RepetitionPenaltyLogitsProcessor over the whole input_ids (prompt + output).
// `seen` must hold each id ONCE (HF uses input_ids.unique()): applying it per
// occurrence double-penalises repeated ids - and this model repeats coordinate
// tokens constantly, so the bias is what flips a greedy choice.
static void apply_repetition_penalty(std::vector<float> & logits,
                                     const std::vector<int> & seen, float penalty) {
    if (penalty <= 0.0f || penalty == 1.0f) return;
    for (int id : seen) {
        if (id < 0 || id >= (int)logits.size()) continue;
        float & s = logits[(size_t)id];
        s = s < 0.0f ? s * penalty : s / penalty;
    }
}

std::string NemotronParse::render_answer(const std::string & raw, const std::string & mode,
                                         int img_w, int img_h) {
    if (mode.empty() || mode == "raw") return raw;
    if (mode == "markdown" || mode == "md") {
        std::vector<Block> blocks;
        parse_blocks(raw, blocks);
        return to_markdown(blocks);
    }
    if (mode == "json" || mode == "blocks") {
        std::vector<Block> blocks;
        parse_blocks(raw, blocks);
        return to_json(blocks, img_w, img_h, raw);
    }
    if (mode == "text" || mode == "plain") {
        std::vector<Block> blocks;
        parse_blocks(raw, blocks);
        return to_plain(blocks);
    }
    return raw;   // unknown mode: keep the model's own markup
}

bool NemotronParse::ask(const vqa::MediaInput & media, const std::string & prompt,
                        const std::string & mode, const vqa::GenParams & gp,
                        std::string & out_text) {
    out_text.clear();
    n_past = 0;
    if (media.empty()) {
        fprintf(stderr, "nemotron_parse: an image is required (document page)\n");
        return false;
    }

    // The model has exactly one task: the control-token prompt. A caller-supplied
    // prompt that spells those control tokens out is passed through verbatim, so
    // the CLI can ask for a different output combination manually.
    const std::string text = (prompt.find("<predict") != std::string::npos) ? prompt : kPrompt;
    std::vector<int> toks = tok_.encode(text);
    if (toks.empty()) {
        fprintf(stderr, "nemotron_parse: empty prompt\n");
        return false;
    }

    // Same page asked twice (the MCP/Web UI pattern): the vision tower and the
    // prefix pass depend only on the image and the fixed prompt, so reuse both -
    // including the first-step logits, since the KV cache is already filled.
    const uint64_t h = hash_page(media, text);
    PageCache * cached = nullptr;
    for (auto & e : pages) if (e.hash == h && !e.feats.empty()) cached = &e;
    const bool kv_hit = cached && cached->kv_owner && !cached->first_logits.empty();
    std::vector<float> feats;
    int n_vis = 0;
    std::vector<float> logits;
    std::vector<int> pos;
    int T = 0;
    if (kv_hit) {
        n_vis = cached->n_vis;
        T = cached->prefix;
        logits = cached->first_logits;
        feats = cached->feats;
        fprintf(stderr, "nemotron_parse: page unchanged - reusing %d visual tokens and the "
                        "prefix KV (%d tokens)\n", n_vis, T);
    } else if (cached) {
        feats = cached->feats;             // tower skipped, prefix must be re-run
        n_vis = cached->n_vis;
        fprintf(stderr, "nemotron_parse: cached page - skipping the vision tower (%d tokens)\n",
                n_vis);
    } else {
        if (gp.on_stage) gp.on_stage("vision", 0, 1, gp.on_stage_ud);
        if (!encode_image(media, feats, &n_vis)) return false;
        if (gp.on_stage) gp.on_stage("vision", 1, 1, gp.on_stage_ud);
    }
    const bool hit = kv_hit;

    const int T_tok = (int)toks.size();
    if (!hit) T = n_vis + T_tok;
    if (T > n_ctx) {
        fprintf(stderr, "prefix %d tokens exceed n_ctx %d (raise --n-ctx)\n", T, n_ctx);
        return false;
    }
    int max_new = gp.max_tokens > 0 ? gp.max_tokens : 512;
    if (T + max_new > n_ctx) {
        const int fit = std::max(1, n_ctx - T);
        if (fit < max_new) {
            fprintf(stderr, "nemotron_parse: clamping --max-tokens %d -> %d to fit n_ctx %d\n",
                    max_new, fit, n_ctx);
            max_new = fit;
        }
    }

    if (!hit) {
        pos.resize((size_t)T);
        for (int i = 0; i < T; i++) pos[(size_t)i] = i;
        TextGraph g;
        const auto tp0 = std::chrono::steady_clock::now();
        if (!build_text_graph(g, T_tok, n_vis, T, true)) { ggml_free(g.ctx); return false; }
        vqa::set_causal_mask(g.mask_in, 0, T);
        if (!run_text(g, toks, pos, feats.data(), n_vis, logits)) { ggml_free(g.ctx); return false; }
        if (!dump_dir.empty()) dump(g.gf, "logits");
        ggml_free(g.ctx);
        fprintf(stderr, "[perf] nemotron_parse prefill: %.0f ms for %d tokens (%d visual)\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count(),
                T, n_vis);
    }
    // this page now owns the KV cache (so it may reuse its first-step logits);
    // everybody else keeps the features and loses the logits
    for (auto & e : pages) e.kv_owner = false;
    if (!cached) {
        if (pages.size() >= 2) pages.erase(pages.begin());
        pages.emplace_back();
        cached = &pages.back();
        cached->hash = h;
    }
    if (cached->feats.empty()) cached->feats = feats;
    cached->n_vis = n_vis;
    cached->prefix = T;
    cached->first_logits = logits;
    cached->kv_owner = true;
    n_past = T;

    std::vector<int> out;
    std::vector<uint8_t> seen_flag((size_t)n_vocab, 0);
    std::vector<int> seen;                       // unique ids, like HF's input_ids.unique()
    auto mark = [&](int t) {
        if (t < 0 || t >= n_vocab || seen_flag[(size_t)t]) return;
        seen_flag[(size_t)t] = 1;
        seen.push_back(t);
    };
    for (int t : toks) mark(t);
    auto emit = [&](int t) {
        std::string piece = tok_.decode({ t });
        out.push_back(t);
        if (gp.stream && gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
    };
    apply_repetition_penalty(logits, seen, rep_penalty);
    int cur = vqa::sample_logits(logits, gp, rng_state);
    int generated = 0;
    const auto td0 = std::chrono::steady_clock::now();
    const double compute_before = ms_compute;
    while (cur != id_eos && generated < max_new) {
        emit(cur);
        mark(cur);
        generated++;
        // Bucket the KV window so the decode graph keeps a CONSTANT shape: the
        // build+alloc cost disappears and ggml-cuda's CUDA graph cache can replay
        // (it keys on cgraph->nodes[0], so a fresh context per token never hits).
        // Unused rows are masked -inf by set_causal_mask, so widening S is exact.
        // The bucket is backend-dependent on purpose: a wider window makes flash
        // attention attend to dead keys, which on CPU costs more than the rebuild
        // it saves (CPU attention is the bottleneck, GPU launch overhead is).
        const int bucket = ggml_backend_is_cpu(backend) ? 128 : 1024;
        int S = ((n_past + 1 + bucket - 1) / bucket) * bucket;
        if (S > n_ctx) S = n_ctx;
        if (dec_g_bucket != S) {
            if (dec_g.ctx) ggml_free(dec_g.ctx);
            dec_g = TextGraph{};
            dec_g_bucket = -1;
            if (!build_text_graph(dec_g, 1, 0, S, true)) {
                if (dec_g.ctx) ggml_free(dec_g.ctx);
                dec_g.ctx = nullptr;
                return false;
            }
            dec_g_bucket = S;
        }
        vqa::set_causal_mask(dec_g.mask_in, n_past, 1);
        std::vector<int> p1(1, n_past);
        std::vector<int> tok1(1, cur);
        std::vector<float> lg;
        if (!run_text(dec_g, tok1, p1, nullptr, 0, lg)) return false;
        if (!dump_dir.empty()) dump(dec_g.gf, "logits");   // last step wins: parity hook
        n_past++;
        apply_repetition_penalty(lg, seen, rep_penalty);
        cur = vqa::sample_logits(lg, gp, rng_state);
    }
    if (generated >= max_new && cur != id_eos)
        fprintf(stderr, "nemotron_parse: hit the %d token cap (raise --max-tokens for long "
                        "pages; a full page is often 2-8k tokens)\n", max_new);
    if (generated > 0) {
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - td0).count();
        fprintf(stderr, "[perf] nemotron_parse decode: %.1f ms/token (%.1f tok/s, %d tokens)\n",
                ms / generated, generated * 1000.0 / ms, generated);
        if (getenv("NEMOTRON_PARSE_PROFILE")) {
            const double comp = (ms_compute - compute_before) / generated;
            fprintf(stderr, "[perf] nemotron_parse split: compute %.2f ms/token, "
                            "graph build + alloc %.2f ms/token\n",
                    comp, ms / generated - comp);
        }
    }

    const std::string raw = tok_.decode(out);
    out_text = render_answer(raw, mode, media.width, media.height);
    return true;
}

}  // namespace nemotron_parse

namespace vqa {
Model * create_nemotron_parse() { return new nemotron_parse::NemotronParse(); }
}  // namespace vqa
