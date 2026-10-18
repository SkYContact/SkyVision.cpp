// siglip_model.cpp - SigLIP image tower backend (round 1: pooled embedding).
//
// One self-contained GGUF (arch `siglip`), converted from any transformers
// `siglip` vision checkpoint (convert/convert_siglip.py). Pure vision tower -
// no tokenizer, no KV cache, no decoding: one 224x224 image produces the 768-d
// pooled embedding that `SiglipModel.get_image_features` returns, or (without
// the head) the post_layernorm token features.
//
// Forward, matching transformers.models.siglip exactly:
//   patch_embed: Conv2d(C, dim, k=patch, stride=patch, valid) folded into a
//     matmul over tokens ordered (h', w') with elements (c, dh, dw)
//   + learned position table (row-major grid)
//   12 pre-norm blocks: LN -> full (non-causal) attention with q|k|v biases ->
//     LN -> MLP with gelu_pytorch_tanh (ggml_gelu IS the tanh form)
//   post_layernorm, then the attention-pooling head:
//     torch.nn.MultiheadAttention(1 probe token <- 196 image tokens, 12 heads)
//     -> residual -> LN -> MLP -> residual, take the probe token. In this
//     release lineage get_image_features IS this pooler output (no projection
//     linear - siglip.has_projection is false in the GGUF).
//   The head rides in the GGUF from round 1 and is built whenever
//   siglip.has_head is set (SIGLIP_NO_HEAD=1 falls back to the post-LN tokens
//   for stage-by-stage parity work).
//
// The whole image fits in one graph (196 tokens), so it is built ONCE in
// load() and replayed per image - same shape the CUDA graph capture path
// expects, like VideoMAE.
//
// Preprocessing, matching SiglipImageProcessor (the slow PIL processor this
// checkpoint ships): square bicubic resize to image_size (PIL antialiased,
// uint8 between the passes) -> rescale 1/255 -> normalize (x-0.5)/0.5.
//
// The block body is core/vit_block.hpp (pre-LN + biased attention + gelu_tanh
// MLP - exactly the SigLIP block); the pooling head is written out inline
// because torch's MultiheadAttention split (probe as the only query) has no
// equivalent in the shared block.
#include "vqa.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#   include "ggml-cuda.h"
#endif

#include "backend.hpp"
#include "ggml_dump.hpp"
#include "gguf_util.hpp"
#include "load_util.hpp"
#include "resample.hpp"
#include "vit_block.hpp"
#include "weight_load.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace siglip {

using vqa::vit::BlockWeights;

class Siglip : public vqa::Model {
public:
    Siglip() = default;
    ~Siglip() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return arch_name.c_str(); }
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp,
             std::string & out_text) override;
    bool embed(const vqa::MediaInput & media, const std::string & text,
               const vqa::EmbedParams & ep, std::vector<float> & out) override;

    // hyper-parameters (metadata of the converted checkpoint)
    std::string arch_name = "siglip";
    int n_layer = 12, n_embd = 768, n_head = 12, n_ff = 3072;
    int patch = 16, image_size = 224, num_channels = 3;
    int resample = 3;                 // PIL: 2 bilinear, 3 bicubic
    int n_patches = 0;                // (image_size/patch)^2
    int hidden_act = 2;               // 2 = gelu(tanh) == gelu_pytorch_tanh
    float ln_eps = 1e-6f;
    bool has_head = true;
    bool has_classifier = false;      // SiglipForImageClassification fine-tunes
    int n_labels = 0;
    std::vector<std::string> labels;
    float mean[3] = {0.5f, 0.5f, 0.5f}, std[3] = {0.5f, 0.5f, 0.5f};

    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool build_graph();
    // One RGB8 image -> pooled (n_embd). When the checkpoint has no head the
    // output is the mean of the post-LN token features.
    // One RGB8 image -> pooled (n_embd). When the checkpoint has no head the
    // output is the mean of the post-LN token features. `probs` is filled when
    // the checkpoint carries the classification head.
    bool encode(const vqa::MediaInput & media, std::vector<float> & pooled,
                std::vector<float> * probs = nullptr);
    // SiglipImageProcessor: resize -> normalize -> pack patch rows. Writes the
    // CHW normalized pixels and the packed [C*ph*pw, n_patches] matrix.
    bool preprocess(const uint8_t * rgb, int w, int h, std::vector<float> & pixels,
                    std::vector<float> & patches);
    void dump(const char * name);

    vqa::Capabilities caps_;
    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    vqa::MMap mm;

    // static graph, kept alive across calls
    ggml_context * gctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * in_patches = nullptr;
    ggml_tensor * t_pooled = nullptr;
    ggml_tensor * t_probs = nullptr;

    ggml_tensor * patch_w = nullptr, * patch_b = nullptr, * pos_emb = nullptr;
    ggml_tensor * norm_w = nullptr, * norm_b = nullptr;
    ggml_tensor * h_probe = nullptr, * h_wqkv = nullptr, * h_bqkv = nullptr;
    ggml_tensor * h_wo = nullptr, * h_bwo = nullptr;
    ggml_tensor * h_ln_w = nullptr, * h_ln_b = nullptr;
    ggml_tensor * h_fc1_w = nullptr, * h_fc1_b = nullptr;
    ggml_tensor * h_fc2_w = nullptr, * h_fc2_b = nullptr;
    ggml_tensor * cls_w = nullptr, * cls_b = nullptr;
    std::vector<BlockWeights> layers;
};

Siglip::~Siglip() {
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (allocr) ggml_gallocr_free(allocr);
    if (gctx) ggml_free(gctx);   // static graph (its tensors live in allocr)
    if (backend) ggml_backend_free(backend);
}

bool Siglip::init_backend() {
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

// ---------------------------------------------------------------- load
bool Siglip::load(const std::string & gguf_path, const std::string & tokenizer_path,
                  const vqa::LoadOptions & opt) {
    (void)tokenizer_path;   // a pure image tower needs no tokenizer
    vqa::apply_load_options(opt, "SIGLIP", n_threads, use_gpu, dump_dir);
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!build_graph()) return false;

    caps_ = {};
    caps_.image = true;
    fprintf(stderr, "%s: ready (%d layers, %d patch tokens, %s%s)\n", arch_name.c_str(),
            n_layer, n_patches, has_head ? "attention-pool head" : "no head",
            has_classifier ? " + classifier" : "");
    if (has_classifier) {
        fprintf(stderr, "%s: %d labels:", arch_name.c_str(), n_labels);
        for (int i = 0; i < std::min(n_labels, 8); i++) fprintf(stderr, " %s", labels[(size_t)i].c_str());
        fprintf(stderr, "%s\n", n_labels > 8 ? " ..." : "");
    }
    return true;
}

bool Siglip::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f MB\n", mm.size / 1e6);

    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    arch_name = vqa::architecture_of(path);
    if (arch_name.empty()) arch_name = "siglip";
    const std::string pre = arch_name + ".";
    auto k = [&](const char * s) { return pre + s; };

    n_layer      = vqa::gguf::u32(gu, k("block_count").c_str(), n_layer);
    n_embd       = vqa::gguf::u32(gu, k("embedding_length").c_str(), n_embd);
    n_head       = vqa::gguf::u32(gu, k("attention.head_count").c_str(), n_head);
    n_ff         = vqa::gguf::u32(gu, k("feed_forward_length").c_str(), n_ff);
    patch        = vqa::gguf::u32(gu, k("patch_size").c_str(), patch);
    image_size   = vqa::gguf::u32(gu, k("image_size").c_str(), image_size);
    num_channels = vqa::gguf::u32(gu, k("num_channels").c_str(), num_channels);
    resample     = vqa::gguf::u32(gu, k("resize_resample").c_str(), 3);
    ln_eps       = vqa::gguf::f32(gu, k("layer_norm_epsilon").c_str(), ln_eps);
    has_head     = vqa::gguf::boolean(gu, k("has_head").c_str(), has_head);
    has_classifier = vqa::gguf::boolean(gu, k("has_classifier").c_str(), false);
    n_labels     = vqa::gguf::u32(gu, k("num_labels").c_str(), 0);
    labels.clear();
    {
        int64_t id = gguf_find_key(gu, k("labels").c_str());
        if (id >= 0) {
            const int64_t n = gguf_get_arr_n(gu, id);
            for (int64_t i = 0; i < n; i++) labels.push_back(gguf_get_arr_str(gu, id, (int)i));
        }
    }
    if (has_classifier && (n_labels <= 0 || (int)labels.size() != n_labels)) {
        fprintf(stderr, "classifier metadata inconsistent (num_labels %d, %zu labels)\n",
                n_labels, labels.size());
        gguf_free(gu);
        return false;
    }
    {
        // transformers names: gelu = exact erf, gelu_tanh/gelu_pytorch_tanh = the
        // tanh form (SigLIP ships gelu_pytorch_tanh; the engine's default)
        const std::string act = vqa::gguf::str(gu, k("hidden_act").c_str(), "gelu_pytorch_tanh");
        if (act == "gelu_tanh" || act == "gelu_pytorch_tanh") hidden_act = 2;
        else if (act == "gelu" || act == "gelu_python" || act == "gelu_accurate") hidden_act = 0;
        else {
            hidden_act = 2;
            fprintf(stderr, "note: hidden_act '%s' is not implemented, using gelu(tanh)\n",
                    act.c_str());
        }
    }
    if (n_embd % n_head) {
        fprintf(stderr, "embedding_length %d is not divisible by head_count %d\n",
                n_embd, n_head);
        gguf_free(gu);
        return false;
    }
    {
        int64_t id = gguf_find_key(gu, k("image_mean").c_str());
        if (id >= 0 && gguf_get_arr_n(gu, id) == 3) {
            const float * a = (const float *)gguf_get_arr_data(gu, id);
            for (int c = 0; c < 3; c++) mean[c] = a[c];
        }
        id = gguf_find_key(gu, k("image_std").c_str());
        if (id >= 0 && gguf_get_arr_n(gu, id) == 3) {
            const float * a = (const float *)gguf_get_arr_data(gu, id);
            for (int c = 0; c < 3; c++) std[c] = a[c];
        }
    }
    if (image_size % patch) {
        fprintf(stderr, "image_size %d is not divisible by patch %d\n", image_size, patch);
        gguf_free(gu);
        return false;
    }
    n_patches = (image_size / patch) * (image_size / patch);

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);

    {
        vqa::weights::Bound bound = vqa::weights::bind_to_backend(wctx, backend, mm);
        if (!bound.ok) return false;
        wbuf = bound.main_buf;
        if (!ggml_backend_is_cpu(backend)) mm.release();
    }

    auto gt = [&](const char * name) {
        ggml_tensor * t = ggml_get_tensor(wctx, name);
        if (!t) fprintf(stderr, "missing tensor: %s\n", name);
        return t;
    };
    patch_w = gt("v.patch_embd.weight");
    patch_b = gt("v.patch_embd.bias");
    pos_emb = gt("v.pos_emb.weight");
    norm_w = gt("v.norm.weight");
    norm_b = gt("v.norm.bias");
    if (has_head) {
        h_probe = gt("v.head.probe");
        h_wqkv = gt("v.head.wqkv.weight");
        h_bqkv = gt("v.head.wqkv.bias");
        h_wo = gt("v.head.wo.weight");
        h_bwo = gt("v.head.wo.bias");
        h_ln_w = gt("v.head.ln.weight");
        h_ln_b = gt("v.head.ln.bias");
        h_fc1_w = gt("v.head.fc1.weight");
        h_fc1_b = gt("v.head.fc1.bias");
        h_fc2_w = gt("v.head.fc2.weight");
        h_fc2_b = gt("v.head.fc2.bias");
        if (!h_probe || !h_wqkv || !h_wo || !h_ln_w || !h_fc2_w) {
            fprintf(stderr, "siglip.has_head but head tensors are missing\n");
            return false;
        }
    }
    if (has_classifier) {
        cls_w = gt("v.cls.weight");
        cls_b = gt("v.cls.bias");
        if (!cls_w || !cls_b) {
            fprintf(stderr, "siglip.has_classifier but v.cls.* tensors are missing\n");
            return false;
        }
    }
    layers.resize(n_layer);
    for (int i = 0; i < n_layer; i++) {
        char buf[128];
        auto g = [&](const char * suffix) {
            snprintf(buf, sizeof(buf), "v.blk.%d.%s", i, suffix);
            return gt(buf);
        };
        BlockWeights & L = layers[(size_t)i];
        L.ln1_w = g("ln1.weight");   L.ln1_b = g("ln1.bias");
        L.qkv_w = g("wqkv.weight");  L.qkv_b = g("wqkv.bias");
        L.wo_w  = g("wo.weight");    L.wo_b  = g("wo.bias");
        L.ln2_w = g("ln2.weight");   L.ln2_b = g("ln2.bias");
        L.fc0_w = g("fc1.weight");   L.fc0_b = g("fc1.bias");
        L.fc1_w = g("fc2.weight");   L.fc1_b = g("fc2.bias");
    }
    return patch_w && patch_b && pos_emb && norm_w && norm_b;
}

// ---------------------------------------------------------------- graph
namespace {

// state shared by the vit::block hooks (one build per process)
struct BuildCtx {
    ggml_context * ctx = nullptr;
    int n_embd = 0, n_head = 0, hd = 0, N = 0;
    int act = 2;   // 0 gelu(erf) | 2 gelu(tanh)
    bool names = false;
};

// heads of the [3*n_embd, N] qkv projection; block b occupies rows
// [b*n_embd, b*n_embd + n_embd). Flash attention returns [hd, heads, tokens]
// already laid out as (head, hd) rows per token, so a plain reshape feeds wo.
ggml_tensor * attn_hook(void * ud, ggml_tensor * qkv) {
    BuildCtx * b = (BuildCtx *)ud;
    auto head = [&](int block) {
        return ggml_view_3d(b->ctx, qkv, b->hd, b->N, b->n_head, qkv->nb[1],
                            (size_t)b->hd * qkv->nb[0], (size_t)block * b->n_embd * qkv->nb[0]);
    };
    ggml_tensor * fa = ggml_flash_attn_ext(b->ctx, head(0), head(1), head(2), nullptr,
                                           1.0f / sqrtf((float)b->hd), 0.0f, 0.0f);
    return ggml_reshape_2d(b->ctx, fa, b->n_embd, b->N);
}

ggml_tensor * gelu_hook(void * ud, ggml_tensor * t) {
    BuildCtx * b = (BuildCtx *)ud;
    return b->act == 0 ? ggml_gelu_erf(b->ctx, t) : ggml_gelu(b->ctx, t);
}

// layer-0 parity sinks: name only (the graph is permanent, nothing to truncate)
bool probe_layer0(void * ud, const char * stage, ggml_tensor * t) {
    BuildCtx * b = (BuildCtx *)ud;
    if (b->names) {
        char name[64];
        snprintf(name, sizeof(name), "v_blk0_%s", stage);   // qkv | attn_out | out
        ggml_set_name(t, name);
    }
    return false;
}

}  // namespace

bool Siglip::build_graph() {
    if (n_patches <= 0 || !patch_w || !patch_b || !pos_emb) return false;
    const int Np = n_patches;
    const int C3 = (int)patch_w->ne[0];
    const int hd = n_embd / n_head;
    if ((int)patch_w->ne[1] != n_embd || (int)pos_emb->ne[0] != n_embd ||
        (int)pos_emb->ne[1] < Np) {
        fprintf(stderr, "weight geometry mismatch: patch_embd [%lld, %lld], pos [%lld, %lld], "
                        "expected [%d, %d] / [%d, >=%d]\n",
                (long long)patch_w->ne[0], (long long)patch_w->ne[1],
                (long long)pos_emb->ne[0], (long long)pos_emb->ne[1], n_embd, C3, n_embd, Np);
        return false;
    }

    struct ggml_init_params ip = { 128ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    auto tb0 = std::chrono::steady_clock::now();
    gctx = ggml_init(ip);
    if (!gctx) return false;
    gf = ggml_new_graph_custom(gctx, 65536, false);

    BuildCtx b;
    b.ctx = gctx;
    b.n_embd = n_embd;
    b.n_head = n_head;
    b.hd = hd;
    b.N = Np;
    b.act = hidden_act;
    b.names = !dump_dir.empty();

    in_patches = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, C3, Np);
    ggml_set_name(in_patches, "patches");

    ggml_tensor * x = ggml_mul_mat(gctx, patch_w, in_patches);
    x = ggml_add(gctx, x, patch_b);
    x = ggml_add(gctx, x, ggml_view_2d(gctx, pos_emb, n_embd, Np, pos_emb->nb[1], 0));
    ggml_set_name(x, "v_patch_out");

    vqa::vit::BlockHooks hooks;
    hooks.attention = attn_hook;
    hooks.gelu = gelu_hook;
    hooks.probe_layer0 = probe_layer0;
    hooks.ud = &b;

    int max_layers = n_layer;   // SIGLIP_MAX_LAYERS=N: layer-by-layer parity debugging
    if (const char * e = getenv("SIGLIP_MAX_LAYERS")) max_layers = std::min(n_layer, atoi(e));
    for (int il = 0; il < max_layers; il++)
        x = vqa::vit::block(gctx, x, layers[(size_t)il], ln_eps, hooks, il);

    x = ggml_norm(gctx, x, ln_eps);
    x = ggml_mul(gctx, x, norm_w);
    x = ggml_add(gctx, x, norm_b);
    ggml_set_name(x, "v_final");

    ggml_tensor * out = x;
    if (has_head && !getenv("SIGLIP_NO_HEAD")) {
        // torch.nn.MultiheadAttention with the probe as the ONLY query:
        // q = Wq·probe (1 token), k/v = Wk,Wv·image tokens (N), unmasked scaled
        // dot product, out_proj. in_proj packs the q|k|v output rows, so each
        // projection is a strided [E, E] view of the fused F16 weight.
        const size_t es = ggml_element_size(h_wqkv);
        auto wrows = [&](int block) {
            return ggml_view_2d(gctx, h_wqkv, n_embd, n_embd, h_wqkv->nb[1],
                                (size_t)block * n_embd * h_wqkv->nb[1]);
        };
        auto brows = [&](int block) {
            return ggml_view_1d(gctx, h_bqkv, n_embd, (size_t)block * n_embd * sizeof(float));
        };
        ggml_tensor * probe = ggml_reshape_2d(gctx, h_probe, n_embd, 1);
        ggml_tensor * q = ggml_add(gctx, ggml_mul_mat(gctx, wrows(0), probe), brows(0));
        ggml_tensor * k = ggml_add(gctx, ggml_mul_mat(gctx, wrows(1), x), brows(1));
        ggml_tensor * v = ggml_add(gctx, ggml_mul_mat(gctx, wrows(2), x), brows(2));
        // [hd, tokens, head] views: token stride E (nb[1]), head stride hd
        auto heads = [&](ggml_tensor * t, int64_t tok) {
            return ggml_view_3d(gctx, t, hd, tok, n_head, t->nb[1],
                                (size_t)hd * ggml_element_size(t), 0);
        };
        ggml_tensor * fa = ggml_flash_attn_ext(gctx, ggml_reshape_3d(gctx, q, hd, 1, n_head),
                                               heads(k, Np), heads(v, Np), nullptr,
                                               1.0f / sqrtf((float)hd), 0.0f, 0.0f);
        ggml_tensor * pooled = ggml_reshape_2d(gctx, fa, n_embd, 1);   // (head, hd) rows
        pooled = ggml_mul_mat(gctx, h_wo, pooled);
        pooled = ggml_add(gctx, pooled, h_bwo);
        ggml_set_name(pooled, "v_head_attn");

        // residual -> LN -> MLP -> residual (SiglipMultiheadAttentionPoolingHead)
        ggml_tensor * res = pooled;
        pooled = ggml_norm(gctx, pooled, ln_eps);
        pooled = ggml_mul(gctx, pooled, h_ln_w);
        pooled = ggml_add(gctx, pooled, h_ln_b);
        ggml_set_name(pooled, "v_head_ln");
        pooled = ggml_mul_mat(gctx, h_fc1_w, pooled);
        pooled = ggml_add(gctx, pooled, h_fc1_b);   // linear() = Wx + b BEFORE the act
        ggml_set_name(pooled, "v_head_z");
        pooled = gelu_hook(&b, pooled);
        ggml_set_name(pooled, "v_head_fc1");
        pooled = ggml_mul_mat(gctx, h_fc2_w, pooled);
        pooled = ggml_add(gctx, pooled, h_fc2_b);
        ggml_set_name(pooled, "v_head_mlp");
        pooled = ggml_add(gctx, res, pooled);                    // single token: take [0]
        ggml_set_name(pooled, "v_pooled");
        out = pooled;
    } else {
        // no head: mean-pool the post-LN tokens so embed() still yields one vector
        ggml_tensor * xt = ggml_cont(gctx, ggml_permute(gctx, x, 1, 0, 2, 3));
        ggml_tensor * s = ggml_reshape_2d(gctx, ggml_sum_rows(gctx, xt), n_embd, 1);
        out = ggml_scale(gctx, s, 1.0f / (float)Np);
        ggml_set_name(out, "v_pooled");
    }
    ggml_tensor * probs_t = nullptr;
    ggml_tensor * pooled_out = out;   // the embedding node (v_pooled)
    if (cls_w && cls_b) {
        // SiglipForImageClassification: classifier(mean(last_hidden_state)) -
        // the patch-token MEAN of the post-LN stream, NOT the pooling head's
        // output (the head weights ride along unused in these fine-tunes).
        ggml_tensor * xt = ggml_cont(gctx, ggml_permute(gctx, x, 1, 0, 2, 3));
        ggml_tensor * meanpooled = ggml_scale(gctx,
            ggml_reshape_2d(gctx, ggml_sum_rows(gctx, xt), n_embd, 1), 1.0f / (float)Np);
        ggml_set_name(meanpooled, "v_cls_pool");
        ggml_tensor * logits = ggml_mul_mat(gctx, cls_w, meanpooled);
        logits = ggml_add(gctx, logits, cls_b);
        ggml_set_name(logits, "v_logits");
        probs_t = ggml_soft_max(gctx, logits);
        ggml_set_name(probs_t, "v_probs");
        out = probs_t;
    }
    ggml_build_forward_expand(gf, out);
    if (probs_t) ggml_build_forward_expand(gf, pooled_out);
    // embed() reads the head-pooled tensor too; it is NOT an ancestor of probs
    // (the classifier consumes the token mean), so expand it explicitly or its
    // name lookup misses and its buffer is never computed.

    if (b.names) {
        static const char * const dump_names[] = {
            "patches", "v_patch_out", "v_blk0_qkv", "v_blk0_attn_out", "v_blk0_out",
            "v_final", "v_head_attn", "v_head_ln", "v_head_z", "v_head_fc1", "v_head_mlp",
            "v_pooled", "v_cls_pool", "v_logits", "v_probs", nullptr,
        };
        vqa::dump::mark_sinks(gctx, gf, dump_names);
    }
    vqa::dump::finalize_sinks(gctx, gf, backend);

    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "graph allocation failed\n");
        return false;
    }
    fprintf(stderr, "[perf] %s: graph build+alloc %.0f ms\n", arch_name.c_str(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb0).count());
    t_pooled = ggml_graph_get_tensor(gf, "v_pooled");
    t_probs = probs_t ? ggml_graph_get_tensor(gf, "v_probs") : nullptr;
    if (!t_pooled || (probs_t && !t_probs)) { fprintf(stderr, "graph output missing\n"); return false; }
    return true;
}

// ---------------------------------------------------------------- preprocess
bool Siglip::preprocess(const uint8_t * rgb, int w, int h, std::vector<float> & pixels,
                        std::vector<float> & patches) {
    const int S = image_size;
    // square resize straight to (S, S) - SiglipImageProcessor has no crop step.
    // resample 3 (SigLIP releases) / 2 (several fine-tunes) go through the PIL
    // Resample.c BICUBIC/BILINEAR replications (8bpc fixpoint, bit-exact vs the
    // slow HF processor); anything else falls back to the generic antialias path.
    std::vector<uint8_t> resized8((size_t)S * S * 3);
    if (resample == 3) {
        vqa::img::resize_bicubic_coeffs(rgb, w, h, 3, S, S, resized8.data());
    } else if (resample == 2) {
        vqa::img::resize_bilinear_coeffs(rgb, w, h, 3, S, S, resized8.data());
    } else {
        std::vector<float> resized;
        vqa::img::resize_antialias(rgb, w, h, S, S, resized,
                                   /*pil_resample=*/resample, /*round_u8=*/true);
        for (size_t i = 0; i < resized8.size(); i++)
            resized8[i] = (uint8_t)std::min(255.0f, std::max(0.0f, std::nearbyintf(resized[i])));
    }
    pixels.resize((size_t)num_channels * S * S);
    for (int y = 0; y < S; y++) {
        const uint8_t * row = resized8.data() + (size_t)y * S * 3;
        for (int x = 0; x < S; x++) {
            for (int c = 0; c < num_channels; c++) {
                const float v = row[(size_t)x * 3 + c] / 255.0f;
                pixels[(size_t)c * S * S + (size_t)y * S + x] = (v - mean[c]) / std[c];
            }
        }
    }
    // pack tokens (h', w') row-major, elements (c, dh, dw) - matches the Conv2d
    // weight flattened to [dim, C*ph*pw]
    const int G = S / patch;
    const int C3 = num_channels * patch * patch;
    patches.assign((size_t)C3 * G * G, 0.0f);
    for (int c = 0; c < num_channels; c++)
        for (int y = 0; y < S; y++)
            for (int x = 0; x < S; x++) {
                int t = (y / patch) * G + (x / patch);
                int f = c * patch * patch + (y % patch) * patch + (x % patch);
                patches[(size_t)t * C3 + f] = pixels[(size_t)c * S * S + (size_t)y * S + x];
            }
    return true;
}

// ---------------------------------------------------------------- forward
bool Siglip::encode(const vqa::MediaInput & media, std::vector<float> & pooled,
                    std::vector<float> * probs) {
    pooled.clear();
    if (probs) probs->clear();
    if (!gf) { fprintf(stderr, "%s: graph not built\n", arch_name.c_str()); return false; }
    if (media.empty()) { fprintf(stderr, "%s: no image\n", arch_name.c_str()); return false; }

    std::vector<float> pixels, patches;
    if (!preprocess(media.frames[0].rgb.data(), media.width, media.height,
                    pixels, patches)) return false;
    if ((int)patches.size() != (int)in_patches->ne[0] * (int)in_patches->ne[1]) {
        fprintf(stderr, "patch count mismatch: %zu vs graph %lld\n", patches.size(),
                (long long)(in_patches->ne[0] * in_patches->ne[1]));
        return false;
    }

    auto t0 = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(in_patches, patches.data(), 0, sizeof(float) * patches.size());
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return false;
    }
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "[perf] %s: image %d tokens x %d layers in %.0f ms\n", arch_name.c_str(),
            n_patches, n_layer,
            std::chrono::duration<double, std::milli>(t1 - t0).count());

    pooled.resize((size_t)n_embd);
    ggml_backend_tensor_get(t_pooled, pooled.data(), 0, sizeof(float) * (size_t)n_embd);
    if (probs && t_probs) {
        probs->resize((size_t)n_labels);
        ggml_backend_tensor_get(t_probs, probs->data(), 0, sizeof(float) * (size_t)n_labels);
    }

    if (!dump_dir.empty()) {
        // engine-side pixels for the preprocessing diff (oracle: ref_pixel_values)
        {
            std::filesystem::create_directories(dump_dir);
            FILE * f = fopen((dump_dir + "/cpp_pixels.bin").c_str(), "wb");
            if (f) {
                fwrite(pixels.data(), sizeof(float), pixels.size(), f);
                fclose(f);
            }
            FILE * m = fopen((dump_dir + "/cpp_pixels.shape").c_str(), "w");
            if (m) {
                fprintf(m, "%d %d %d 1\n", num_channels, image_size, image_size);
                fclose(m);
            }
        }
        dump("patches");
        dump("v_patch_out");
        dump("v_final");
        dump("v_head_attn");
        dump("v_head_ln");
        dump("v_head_z");
        dump("v_head_fc1");
        dump("v_head_mlp");
        dump("v_pooled");
        if (t_probs) {
            dump("v_logits");
            dump("v_probs");
        }
    }
    return true;
}

void Siglip::dump(const char * name) {
    if (dump_dir.empty()) return;
    vqa::dump::write(gf, dump_dir.c_str(), name);
}

// ---------------------------------------------------------------- ask / embed
bool Siglip::ask(const vqa::MediaInput & media, const std::string & prompt,
                 const std::string & mode, const vqa::GenParams & gp,
                 std::string & out_text) {
    (void)prompt;   // no text side in round 1
    (void)gp;
    out_text.clear();
    if (media.empty()) {
        fprintf(stderr, "%s: no image (pass --image)\n", arch_name.c_str());
        return false;
    }
    if (gp.on_stage) gp.on_stage("vision", 0, 1, gp.on_stage_ud);
    std::vector<float> pooled, probs;
    if (!encode(media, pooled, &probs)) return false;
    if (gp.on_stage) gp.on_stage("vision", 1, 1, gp.on_stage_ud);

    char buf[512];
    if (mode == "embed") {
        // the pooled embedding (pre-logits), retrieval-compatible with --embed
        out_text = "[";
        for (size_t i = 0; i < pooled.size(); i++) {
            snprintf(buf, sizeof(buf), "%s%.6f", i ? ", " : "", (double)pooled[i]);
            out_text += buf;
        }
        out_text += "]";
        return true;
    }
    if (!probs.empty()) {
        // videomae-style top-k; mode may carry the k as a number (default 5)
        int k = 5;
        if (!mode.empty() && mode.find_first_not_of("0123456789") == std::string::npos)
            k = std::max(1, atoi(mode.c_str()));
        std::vector<int> order(probs.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
        const int kk = std::min<int>(k, (int)probs.size());
        std::partial_sort(order.begin(), order.begin() + kk, order.end(),
                          [&](int a, int b) { return probs[(size_t)a] > probs[(size_t)b]; });
        for (int i = 0; i < kk; i++) {
            const int id = order[(size_t)i];
            const char * label = id < (int)labels.size() ? labels[(size_t)id].c_str() : "?";
            snprintf(buf, sizeof(buf), "%.6f  %s\n", (double)probs[(size_t)id], label);
            out_text += buf;
        }
        return true;
    }
    double l2 = 0.0, mean_v = 0.0;
    for (float v : pooled) { l2 += (double)v * v; mean_v += v; }
    l2 = sqrt(l2);
    mean_v /= (double)std::max<size_t>(1, pooled.size());
    snprintf(buf, sizeof(buf),
             "%d-d pooled embedding, l2 %.4f, mean %.5f\n"
             "(first values: %.5f %.5f %.5f %.5f)\n",
             n_embd, l2, mean_v, (double)pooled[0], (double)pooled[1], (double)pooled[2],
             (double)pooled[3]);
    out_text = buf;
    return true;
}

bool Siglip::embed(const vqa::MediaInput & media, const std::string & text,
                   const vqa::EmbedParams & ep, std::vector<float> & out) {
    (void)text;    // image-only in round 1 (text tower is round 2)
    if (media.empty()) {
        fprintf(stderr, "%s: embedding needs an image (--image)\n", arch_name.c_str());
        return false;
    }
    if (ep.truncate_dim > 0 && ep.truncate_dim != n_embd)
        fprintf(stderr, "note: truncate_dim %d ignored, embedding is %d-d\n",
                ep.truncate_dim, n_embd);
    std::vector<float> pooled;
    if (!encode(media, pooled)) return false;
    // embed() contract: one L2-normalized vector (HF-side retrieval compares
    // normalized features too, so the raw scale never matters)
    double sum = 0.0;
    for (float v : pooled) sum += (double)v * v;
    const float inv = (float)(1.0 / std::sqrt(std::max(sum, 1e-12)));
    out.resize(pooled.size());
    for (size_t i = 0; i < pooled.size(); i++) out[i] = pooled[i] * inv;
    return true;
}

}  // namespace siglip

namespace vqa {
Model * create_siglip() { return new siglip::Siglip(); }
}  // namespace vqa
