// vjepa2_model.cpp - V-JEPA 2 video encoder backend (Meta FAIR ViT-*).
//
// One self-contained GGUF (arch `vjepa2`), converted from any HF `vjepa2`
// checkpoint whose encoder matches transformers' VJEPA2Model
// (facebook/vjepa2-vitl-fpc64-256 and the ViT-g/predictor siblings). Like the
// videomae backend it is a pure video tower: no tokenizer, no KV cache, no
// decoding - one forward pass over (T/tubelet)*(H/patch)*(W/patch) tokens
// produces the mean-pooled embedding of the post-LayerNorm stream.
//
// Forward, matching transformers.models.vjepa2 exactly:
//   patch_embed: Conv3d(C, dim, k=(tubelet, patch, patch)) folded into a matmul
//     over tokens ordered (t', h', w') with elements (c, dt, dh, dw)
//   NO position table: V-JEPA 2 replaces it with 3-D decomposed RoPE applied to
//     q and k inside every layer (see the RoPE notes below)
//   NO cls token, NO register tokens
//   24 pre-norm blocks: LN -> full (non-causal) attention, q/k/v each with bias
//     and projection bias -> LN -> MLP with GELU(erf)
//   encoder.layernorm, then mean over the token axis (get_vision_features
//     pooled by hand - this checkpoint has no attentive pooler / classifier)
//
// 3-D RoPE (the part that is easy to get wrong):
//   head_dim 64 is split into temporal | height | width | untouched-tail where
//   each rotated segment is seg = 2*((head_dim//3)//2) = 20 dims (60 rotated, 4
//   left as-is). The token's (t', h', w') grid index is the position id of each
//   of the three passes.
//   `rotate_queries_or_keys` is NOT a plain RoPE: its
//   `emb_cos = freq.cos().repeat(1, 1, 1, 2)` TILES the frequency table instead
//   of interleaving it, so an adjacent dimension pair (2u, 2u+1) rotates its even
//   member with frequency 2u and its odd member with 2u+1 (upstream keeps that on
//   purpose: "fixing the bug would break compatibility with the pretrained
//   model", vjepa2 PR #15). The converter therefore permutes each rotated
//   segment of the q/k rows to [even dims | odd dims] and the graph applies the
//   two halves against host-built cos/sin tables (rope_3d below).
//
// The clip shape never changes, so the graph is built ONCE in load() and then
// replayed (stable ggml_cgraph => CUDA graph capture works, CPU skips the
// per-clip build + allocator pass). Only the patch tensor is refilled.
//
// The block body is core/vit_block.hpp, shared with the moondream / VideoChat3 /
// VideoMAE towers; the attention hook only swaps in per-head 3-D RoPE before the
// unmasked flash-attn, and the MLP hook passes VideoMAE's exact erf GELU.
#include "core/vqa.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#   include "ggml-cuda.h"
#endif

#include "core/engine/backend.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/gguf_util.hpp"
#include "core/runtime/load_util.hpp"
#include "core/engine/weight_load.hpp"
#include "videomae_preprocess.hpp"   // pack_clip: Conv3d tubelet patchify (shared)
#include "core/blocks/vit_block.hpp"
#include "core/util/mmap.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace vjepa2 {

using vqa::vit::BlockWeights;

class VJEPA2 : public vqa::Model {
public:
    VJEPA2() = default;
    ~VJEPA2() override;

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
    std::string arch_name = "vjepa2";
    int n_layer = 24, n_embd = 1024, n_head = 16, n_ff = 4096;
    int patch = 16, tubelet = 2, image_size = 256, num_frames = 64;
    int resize_shortest = 0, resample = 2;
    int grid_t = 0, grid_h = 0, grid_w = 0, n_patches = 0;
    // 3-D RoPE geometry: three rotated segments of `rope_seg` dims each, plus a
    // `rope_tail` remainder that is left unrotated (head_dim - 3*rope_seg)
    int rope_seg = 0, rope_tail = 0;
    int hidden_act = 0;   // 0 = gelu (erf, nn.GELU) | 1 = gelu_quick | 2 = gelu (tanh)
    float rope_theta = 10000.0f, ln_eps = 1e-6f;
    bool use_rope = true, has_final_norm = true, mean_pool = true;
    // heads carried by this GGUF (the encoder is always present)
    bool has_pooler = false, has_classifier = false, has_predictor = false;
    int n_pooler = 0, n_labels = 0;
    std::vector<std::string> labels;
    int pred_layer = 12, pred_dim = 384, pred_head = 12, pred_ff = 1536, pred_n_mask = 10;
    int pred_seg = 0, pred_tail = 0;
    float mean[3] = {0.485f, 0.456f, 0.406f}, std[3] = {0.229f, 0.224f, 0.225f};

    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool build_graph();
    ggml_tensor * build_pooler(ggml_tensor * x);
    bool encode(const vqa::MediaInput & media, std::vector<float> & pooled,
                std::vector<float> & probs, int * n_tokens);
    bool predict(std::vector<int> ctx_idx, std::vector<int> tgt_idx, int mask_index,
                 std::vector<float> & latents, const std::vector<float> * features);
    bool predict_cmd(const std::string & spec, const std::vector<float> & pooled,
                     std::string & out_text);
    void dump(const char * name);

    vqa::Capabilities caps_;
    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    vqa::MMap mm;

    ggml_context * gctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * in_patches = nullptr;
    ggml_tensor * t_pooled = nullptr;
    ggml_tensor * t_final = nullptr;
    ggml_tensor * t_probs = nullptr;

    ggml_tensor * patch_w = nullptr, * patch_b = nullptr;
    ggml_tensor * norm_w = nullptr, * norm_b = nullptr;
    std::vector<BlockWeights> layers;

    // attentive pooler (fine-tune checkpoints): N self-attention layers, then a
    // single learned query token that cross-attends to the token stream. The
    // cross attention has no output projection (upstream calls this out).
    ggml_tensor * pool_query = nullptr;
    std::vector<BlockWeights> pool_layers;
    struct CrossWeights {
        ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;
        ggml_tensor *q_w = nullptr, *q_b = nullptr;
        ggml_tensor *kv_w = nullptr, *kv_b = nullptr;
        ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;
        ggml_tensor *fc0_w = nullptr, *fc0_b = nullptr;
        ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    } pool_cross;
    ggml_tensor * cls_w = nullptr, * cls_b = nullptr;

    // masked-latent predictor (the world-model half)
    ggml_tensor * p_emb_w = nullptr, * p_emb_b = nullptr, * p_mask = nullptr;
    ggml_tensor * p_norm_w = nullptr, * p_norm_b = nullptr;
    ggml_tensor * p_proj_w = nullptr, * p_proj_b = nullptr;
    std::vector<BlockWeights> pred_layers;

    // predictor graph, rebuilt only when the context/target split changes
    struct PredictorGraph {
        ggml_context * ctx = nullptr;
        ggml_cgraph * gf = nullptr;
        ggml_gallocr_t allocr = nullptr;
        ggml_tensor *in_feat = nullptr, *in_ctx = nullptr, *in_order = nullptr;
        ggml_tensor *in_tgt = nullptr, *in_mask = nullptr;
        ggml_tensor *t_rope_e = nullptr, *t_rope_o = nullptr;
        ggml_tensor * out = nullptr;
        int n_ctx = 0, n_tgt = 0;
        void reset();
    } pg;
};

VJEPA2::~VJEPA2() {
    pg.reset();
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (allocr) ggml_gallocr_free(allocr);
    if (gctx) ggml_free(gctx);
    if (backend) ggml_backend_free(backend);
}

bool VJEPA2::init_backend() {
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

// ---------------------------------------------------------------- load
bool VJEPA2::load(const std::string & gguf_path, const std::string & tokenizer_path,
                  const vqa::LoadOptions & opt) {
    (void)tokenizer_path;   // a pure video tower needs no tokenizer
    vqa::apply_load_options(opt, "VJEPA2", n_threads, use_gpu, dump_dir);
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!build_graph()) return false;

    caps_ = {};
    caps_.image = true;    // a still image becomes a clip of identical frames
    caps_.video = true;
    std::string heads;
    heads += has_pooler ? "attentive pooler" : "mean-pooled embedding";
    if (has_classifier) heads += ", " + std::to_string(n_labels) + "-way classifier";
    if (has_predictor)
        heads += ", predictor " + std::to_string(pred_layer) + "x" + std::to_string(pred_dim);
    fprintf(stderr, "%s: ready (%d layers, %d patch tokens = %dD/%d x %dx%d, "
                    "3D rope %d+%d+%d+%d, %s)\n",
            arch_name.c_str(), n_layer, n_patches, grid_t, num_frames, grid_h, grid_w,
            rope_seg, rope_seg, rope_seg, rope_tail, heads.c_str());
    return true;
}

bool VJEPA2::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f MB\n", mm.size / 1e6);

    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    arch_name = vqa::architecture_of(path);
    if (arch_name.empty()) arch_name = "vjepa2";
    const std::string pre = arch_name + ".";
    auto k = [&](const char * s) { return pre + s; };

    n_layer      = vqa::gguf::u32(gu, k("block_count").c_str(), n_layer);
    n_embd       = vqa::gguf::u32(gu, k("embedding_length").c_str(), n_embd);
    n_head       = vqa::gguf::u32(gu, k("attention.head_count").c_str(), n_head);
    n_ff         = vqa::gguf::u32(gu, k("feed_forward_length").c_str(), n_ff);
    patch        = vqa::gguf::u32(gu, k("patch_size").c_str(), patch);
    tubelet      = vqa::gguf::u32(gu, k("tubelet_size").c_str(), tubelet);
    image_size   = vqa::gguf::u32(gu, k("image_size").c_str(), image_size);
    num_frames   = vqa::gguf::u32(gu, k("num_frames").c_str(), num_frames);
    resize_shortest = vqa::gguf::u32(gu, k("resize_shortest").c_str(), 0);
    resample     = vqa::gguf::u32(gu, k("resize_resample").c_str(), 2);
    ln_eps       = vqa::gguf::f32(gu, k("layer_norm_epsilon").c_str(), ln_eps);
    rope_theta   = vqa::gguf::f32(gu, k("rope_theta").c_str(), rope_theta);
    {
        // the rope math below assumes q/k rows were written as
        // [even dims | odd dims] per rotated segment (convert_vjepa2.py)
        const std::string layout = vqa::gguf::str(gu, k("rope_layout").c_str(), "even-odd");
        if (layout != "even-odd") {
            fprintf(stderr, "unsupported vjepa2.rope_layout '%s' (expected 'even-odd'); "
                            "re-convert with convert/convert_vjepa2.py\n", layout.c_str());
            gguf_free(gu);
            return false;
        }
    }
    use_rope     = vqa::gguf::boolean(gu, k("use_rope").c_str(), true);
    has_final_norm = vqa::gguf::boolean(gu, k("has_final_norm").c_str(), true);
    mean_pool    = vqa::gguf::boolean(gu, k("mean_pool").c_str(), true);
    has_pooler   = vqa::gguf::boolean(gu, k("has_pooler").c_str(), false);
    has_classifier = vqa::gguf::boolean(gu, k("has_classifier").c_str(), false);
    has_predictor  = vqa::gguf::boolean(gu, k("has_predictor").c_str(), false);
    n_pooler     = vqa::gguf::u32(gu, k("pooler.layers").c_str(), 0);
    n_labels     = vqa::gguf::u32(gu, k("num_labels").c_str(), 0);
    {
        const std::string pp = pre + "predictor.";
        pred_layer   = (int)vqa::gguf::u32(gu, (pp + "block_count").c_str(), pred_layer);
        pred_dim     = (int)vqa::gguf::u32(gu, (pp + "embedding_length").c_str(), pred_dim);
        pred_head    = (int)vqa::gguf::u32(gu, (pp + "attention.head_count").c_str(), pred_head);
        pred_ff      = (int)vqa::gguf::u32(gu, (pp + "feed_forward_length").c_str(), pred_ff);
        pred_n_mask  = (int)vqa::gguf::u32(gu, (pp + "num_mask_tokens").c_str(), pred_n_mask);
    }
    labels.clear();
    if (has_classifier) {
        const int64_t id = gguf_find_key(gu, k("labels").c_str());
        const int64_t n = id >= 0 ? gguf_get_arr_n(gu, id) : 0;
        for (int64_t i = 0; i < n; i++) labels.push_back(gguf_get_arr_str(gu, id, i));
        if (n_labels <= 0 || (int)labels.size() != n_labels) {
            fprintf(stderr, "classifier metadata inconsistent (num_labels %d, %zu labels)\n",
                    n_labels, labels.size());
            gguf_free(gu);
            return false;
        }
    }
    {
        const std::string act = vqa::gguf::str(gu, k("hidden_act").c_str(), "gelu");
        if (act == "gelu" || act == "gelu_python" || act == "gelu_accurate") hidden_act = 0;
        else if (act == "gelu_fast" || act == "gelu_new" || act == "gelu_tanh" ||
                 act == "gelu_pytorch_tanh") hidden_act = 2;
        else if (act == "gelu_quick" || act == "quick_gelu") hidden_act = 1;
        else {
            hidden_act = 0;
            fprintf(stderr, "note: hidden_act '%s' is not implemented, using gelu(erf)\n",
                    act.c_str());
        }
    }
    if (n_embd % n_head) {
        fprintf(stderr, "embedding_length %d is not divisible by head_count %d\n", n_embd, n_head);
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
    if (num_frames % tubelet || image_size % patch) {
        fprintf(stderr, "geometry %dx%d frames %d / patch %d tubelet %d is not divisible\n",
                image_size, image_size, num_frames, patch, tubelet);
        gguf_free(gu);
        return false;
    }
    grid_t = num_frames / tubelet;
    grid_h = image_size / patch;
    grid_w = image_size / patch;
    n_patches = grid_t * grid_h * grid_w;

    const int hd = n_embd / n_head;
    rope_seg = 2 * ((hd / 3) / 2);
    rope_tail = hd - 3 * rope_seg;
    if (use_rope && rope_seg <= 0) {
        fprintf(stderr, "head_dim %d is too small for the 3-D rope split\n", hd);
        gguf_free(gu);
        return false;
    }

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
    if (has_final_norm) {
        norm_w = gt("v.norm.weight");
        norm_b = gt("v.norm.bias");
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

    // --------------- heads: attentive pooler, classifier, masked predictor
    auto bind_blk = [&](std::vector<BlockWeights> & dst, int n, const char * prefix) {
        dst.resize((size_t)n);
        for (int i = 0; i < n; i++) {
            char buf[128];
            auto g = [&](const char * suffix) {
                snprintf(buf, sizeof(buf), "%s.%d.%s", prefix, i, suffix);
                return gt(buf);
            };
            BlockWeights & L = dst[(size_t)i];
            L.ln1_w = g("ln1.weight");   L.ln1_b = g("ln1.bias");
            L.qkv_w = g("wqkv.weight");  L.qkv_b = g("wqkv.bias");
            L.wo_w  = g("wo.weight");    L.wo_b  = g("wo.bias");
            L.ln2_w = g("ln2.weight");   L.ln2_b = g("ln2.bias");
            L.fc0_w = g("fc1.weight");   L.fc0_b = g("fc1.bias");
            L.fc1_w = g("fc2.weight");   L.fc1_b = g("fc2.bias");
        }
    };

    if (has_pooler) {
        pool_query = gt("pool.query");
        bind_blk(pool_layers, n_pooler, "pool.self");
        pool_cross.ln1_w = gt("pool.cross.ln1.weight");
        pool_cross.ln1_b = gt("pool.cross.ln1.bias");
        pool_cross.q_w = gt("pool.cross.wq.weight");
        pool_cross.q_b = gt("pool.cross.wq.bias");
        pool_cross.kv_w = gt("pool.cross.wkv.weight");
        pool_cross.kv_b = gt("pool.cross.wkv.bias");
        pool_cross.ln2_w = gt("pool.cross.ln2.weight");
        pool_cross.ln2_b = gt("pool.cross.ln2.bias");
        pool_cross.fc0_w = gt("pool.cross.fc1.weight");
        pool_cross.fc0_b = gt("pool.cross.fc1.bias");
        pool_cross.fc1_w = gt("pool.cross.fc2.weight");
        pool_cross.fc1_b = gt("pool.cross.fc2.bias");
        if (!pool_query || !pool_cross.q_w || !pool_cross.kv_w) return false;
    }
    if (has_classifier) {
        cls_w = gt("cls.weight");
        cls_b = gt("cls.bias");
        if (!cls_w || !cls_b) return false;
    }
    if (has_predictor) {
        p_emb_w = gt("p.emb.weight");
        p_emb_b = gt("p.emb.bias");
        p_mask = gt("p.mask_tokens");
        p_norm_w = gt("p.norm.weight");
        p_norm_b = gt("p.norm.bias");
        p_proj_w = gt("p.proj.weight");
        p_proj_b = gt("p.proj.bias");
        bind_blk(pred_layers, pred_layer, "p.blk");
        if (pred_dim % pred_head) {
            fprintf(stderr, "predictor: %d dims over %d heads does not divide\n",
                    pred_dim, pred_head);
            return false;
        }
        const int phd = pred_dim / pred_head;
        pred_seg = 2 * ((phd / 3) / 2);
        pred_tail = phd - 3 * pred_seg;
        if (!p_emb_w || !p_mask || !p_proj_w || !p_norm_w || pred_seg <= 0) {
            fprintf(stderr, "predictor weights missing or the rope split failed\n");
            return false;
        }
    }
    return patch_w && patch_b && (!has_final_norm || (norm_w && norm_b));
}

void VJEPA2::dump(const char * name) {
    if (dump_dir.empty()) return;
    vqa::dump::write(gf, dump_dir.c_str(), name);
}

// ---------------------------------------------------------------- graph
namespace {

// The block body, the flash-attn wiring and the pair-rotation rope all come from
// core/vit_block.hpp (the same code path moondream / VideoChat3 / VideoMAE /
// moss_vl use); this backend only supplies the rope angle tables.
struct Hooks {
    vqa::vit::VitHookEnv env;
    int act = 0;        // 0 gelu(erf) | 1 gelu_quick | 2 gelu(tanh)
    bool names = false;
};

ggml_tensor * gelu_hook(void * ud, ggml_tensor * t) {
    Hooks * h = (Hooks *)ud;
    switch (h->act) {
        case 1:  return ggml_gelu_quick(h->env.gctx, t);   // x*sigmoid(1.702x)
        case 2:  return ggml_gelu(h->env.gctx, t);         // tanh approximation
        default: return ggml_gelu_erf(h->env.gctx, t);     // nn.GELU() == exact erf
    }
}

ggml_tensor * attention_hook(void * ud, ggml_tensor * qkv) {
    return vqa::vit::vit_attention_hook(&((Hooks *)ud)->env, qkv);
}

bool probe_layer0(void * ud, const char * stage, ggml_tensor * t) {
    Hooks * h = (Hooks *)ud;
    if (h->names) {
        char name[64];
        snprintf(name, sizeof(name), "v_blk0_%s", stage);   // qkv | attn_out | out
        ggml_set_name(t, name);
    }
    return false;
}

// --------------------------------------------------------------- rope tables
// V-JEPA 2's rotation, faithful to facebookresearch/vjepa2
// `rotate_queries_or_keys` - including the frequency duplication upstream keeps
// on purpose ("fixing the bug would break compatibility with the pretrained
// model", vjepa2 PR #15):
//
//   emb_cos = freq.cos().repeat(1, 1, 1, 2)   <- TILES [c0..c9, c0..c9],
//                                                it does not interleave
//   y       = interleave(-x[odd], x[even])
//   out     = x * emb_cos + y * emb_sin
//
// so an adjacent dim pair (2u, 2u+1) rotates member 0 with frequency 2u and
// member 1 with 2u+1. The converter writes the q/k rows as [member 0 of every
// pair | member 1 of every pair], which makes the generic pair rotation in
// vit_block.hpp apply directly - only the second member needs its own angle
// table (t_rope_odd).
struct RopeCol {
    int axis;    // 0 = t' | 1 = h' | 2 = w' | -1 = unrotated (identity)
    int freq0;   // frequency index of the first pair member
    int freq1;   // ... and of the second member (NOT always freq0 + 1: with the
                 // tiled table it is (2u + 1) % (seg/2), which wraps whenever
                 // seg/2 is odd - the 384-d predictor has seg 10 -> seg/2 == 5)
};

// column k rotates pair u = k % (seg/2) of rotated segment a = k / (seg/2):
// member 0 uses table index (2u) % (seg/2), member 1 (2u + 1) % (seg/2)
std::vector<RopeCol> rope_columns(int half, int seg) {
    std::vector<RopeCol> cols;
    const int per_seg = seg / 2;
    const int pairs = 3 * per_seg;
    cols.reserve((size_t)half);
    for (int k = 0; k < half; k++) {
        if (k >= pairs || per_seg <= 0) {
            cols.push_back({ -1, -1, -1 });
            continue;
        }
        const int u = k % per_seg;
        cols.push_back({ k / per_seg, (2 * u) % per_seg, (2 * u + 1) % per_seg });
    }
    return cols;
}

ggml_tensor * new_rope_table(ggml_context * c, int half, int n_tok) {
    // flat [2, half, tokens]: cos plane then sin plane (vit_block.hpp layout,
    // same shape moss_vl uses)
    return ggml_new_tensor_1d(c, GGML_TYPE_F32, (int64_t)2 * half * n_tok);
}

// `half` is the table width (head_dim/2: one column per dimension pair) while
// `omega_den` is the exponent scale of the frequency table (seg/2, the rotated
// segment length halved) - they only coincide when the whole head rotates.
void fill_rope_tables(ggml_tensor * te, ggml_tensor * to, int half, int n_tok,
                      int omega_den, const std::vector<RopeCol> & cols,
                      const int32_t * pt, const int32_t * ph, const int32_t * pw,
                      float theta) {
    const size_t plane = (size_t)half * (size_t)n_tok;
    std::vector<float> ec(plane), es(plane), oc(plane), os(plane);
    const float denom = (float)omega_den;
    for (int t = 0; t < n_tok; t++) {
        for (int k = 0; k < half; k++) {
            const RopeCol & col = cols[(size_t)k];
            float ae = 0.0f, ao = 0.0f;   // angle 0 -> identity (tail dims)
            if (col.axis >= 0) {
                const int32_t p = col.axis == 0 ? pt[t] : (col.axis == 1 ? ph[t] : pw[t]);
                ae = (float)p / powf(theta, (float)col.freq0 / denom);
                ao = (float)p / powf(theta, (float)col.freq1 / denom);
            }
            const size_t i = (size_t)k + (size_t)half * (size_t)t;
            ec[i] = cosf(ae);
            es[i] = sinf(ae);
            oc[i] = cosf(ao);
            os[i] = sinf(ao);
        }
    }
    const size_t nb = plane * sizeof(float);
    ggml_backend_tensor_set(te, ec.data(), 0, nb);
    ggml_backend_tensor_set(te, es.data(), nb, nb);
    ggml_backend_tensor_set(to, oc.data(), 0, nb);
    ggml_backend_tensor_set(to, os.data(), nb, nb);
}

}  // namespace

bool VJEPA2::build_graph() {
    if (n_patches <= 0 || !patch_w || !patch_b) return false;
    const int N = n_patches;
    const int hd = n_embd / n_head;
    const int rhalf = hd / 2;   // rope table width: one column per head-dim pair
    const int C3 = (int)patch_w->ne[0];
    if ((int)patch_w->ne[1] != n_embd) {
        fprintf(stderr, "weight geometry mismatch: patch_embd [%lld, %lld], expected [%d, %d]\n",
                (long long)patch_w->ne[0], (long long)patch_w->ne[1], C3, n_embd);
        return false;
    }

    struct ggml_init_params ip = { 128ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    auto tb0 = std::chrono::steady_clock::now();
    gctx = ggml_init(ip);
    if (!gctx) return false;
    gf = ggml_new_graph_custom(gctx, 65536, false);

    Hooks hk;
    hk.act = hidden_act;
    hk.names = !dump_dir.empty();
    vqa::vit::VitHookEnv & env = hk.env;
    env.gctx = gctx;
    env.gf = gf;
    env.n_head = n_head;
    env.n_embd = n_embd;
    env.N = N;
    env.vhd = hd;
    env.vhd_f = hd;
    env.dbg_names = hk.names;
    env.t_rope = new_rope_table(gctx, rhalf, N);
    env.t_rope_odd = new_rope_table(gctx, rhalf, N);

    in_patches = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, C3, N);
    ggml_set_name(in_patches, "patches");

    ggml_tensor * x = ggml_mul_mat(gctx, patch_w, in_patches);
    x = ggml_add(gctx, x, patch_b);
    ggml_set_name(x, "v_patch_out");

    vqa::vit::BlockHooks hooks;
    hooks.attention = attention_hook;
    hooks.gelu = gelu_hook;
    hooks.probe_layer0 = probe_layer0;
    hooks.ud = &hk;

    int max_layers = n_layer;   // VJEPA2_MAX_LAYERS=N: layer-by-layer parity debugging
    if (const char * e = getenv("VJEPA2_MAX_LAYERS")) max_layers = std::min(n_layer, atoi(e));
    for (int il = 0; il < max_layers; il++) {
        env.layer = il;
        x = vqa::vit::block(gctx, x, layers[(size_t)il], ln_eps, hooks, il);
    }

    if (has_final_norm) {   // encoder.layernorm
        x = ggml_norm(gctx, x, ln_eps);
        x = ggml_mul(gctx, x, norm_w);
        x = ggml_add(gctx, x, norm_b);
    }
    ggml_set_name(x, "v_final");

    // fine-tune checkpoints pool with the attentive pooler; the pretraining
    // releases use the plain mean (official get_vision_features does the same)
    ggml_tensor * pool = has_pooler ? build_pooler(x) : nullptr;
    if (!pool) {
        if (has_pooler) {
            fprintf(stderr, "pooler build failed\n");
            return false;
        }
        if (mean_pool) {
            // ggml_sum_rows reduces dim0, so transpose the token axis first
            ggml_tensor * xt = ggml_cont(gctx, ggml_permute(gctx, x, 1, 0, 2, 3));
            ggml_tensor * s = ggml_reshape_2d(gctx, ggml_sum_rows(gctx, xt), n_embd, 1);
            pool = ggml_scale(gctx, s, 1.0f / (float)N);
        } else {
            pool = ggml_view_2d(gctx, x, n_embd, 1, x->nb[1], 0);   // token 0
        }
    }
    ggml_set_name(pool, "v_pooled");

    ggml_tensor * probs_t = nullptr;
    if (cls_w && cls_b) {
        ggml_tensor * logits = ggml_mul_mat(gctx, cls_w, pool);
        logits = ggml_add(gctx, logits, cls_b);
        ggml_set_name(logits, "v_logits");
        probs_t = ggml_soft_max(gctx, logits);
        ggml_set_name(probs_t, "v_probs");
    }
    ggml_build_forward_expand(gf, probs_t ? probs_t : pool);

    if (hk.names) {
        static const char * const dump_names[] = {
            "patches", "v_patch_out", "v_blk0_qkv", "v_blk0_q_rope", "v_blk0_k_rope",
            "v_blk0_attn", "v_blk0_attn_out", "v_blk0_out", "v_final", "v_pooled",
            "v_logits", "v_probs", nullptr,
        };
        vqa::dump::mark_sinks(gctx, gf, dump_names);
    }
    vqa::dump::finalize_sinks(gctx, gf, backend);

    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "graph allocation failed\n");
        return false;
    }

    // the rope tables only depend on the fixed clip geometry: fill them once
    {
        std::vector<int32_t> pt((size_t)N), ph((size_t)N), pw((size_t)N);
        const int per_frame = grid_h * grid_w;
        for (int t = 0; t < N; t++) {
            pt[(size_t)t] = t / per_frame;
            ph[(size_t)t] = (t % per_frame) / grid_w;
            pw[(size_t)t] = t % grid_w;
        }
        fill_rope_tables(env.t_rope, env.t_rope_odd, rhalf, N, rope_seg / 2,
                         rope_columns(rhalf, rope_seg),
                         pt.data(), ph.data(), pw.data(), rope_theta);
    }

    fprintf(stderr, "[perf] %s: graph build+alloc %.0f ms\n", arch_name.c_str(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb0).count());
    t_pooled = ggml_graph_get_tensor(gf, "v_pooled");
    t_final = ggml_graph_get_tensor(gf, "v_final");
    t_probs = probs_t ? ggml_graph_get_tensor(gf, "v_probs") : nullptr;
    if (!t_pooled || (probs_t && !t_probs)) {
        fprintf(stderr, "graph output missing\n");
        return false;
    }
    return true;
}

// Attentive pooler: `n_pooler` plain pre-norm self-attention blocks (no rope,
// no mask) over the token stream, then one learned query token cross-attends to
// that stream. The cross attention has q_proj + fused k/v but NO output
// projection, and the query itself is not normalized - only the stream is.
ggml_tensor * VJEPA2::build_pooler(ggml_tensor * x) {
    const int N = n_patches;
    const int hd = n_embd / n_head;
    const size_t es = ggml_element_size(x);

    Hooks hk;
    hk.act = hidden_act;
    hk.names = false;
    hk.env.gctx = gctx;
    hk.env.gf = gf;
    hk.env.n_head = n_head;
    hk.env.n_embd = n_embd;
    hk.env.N = N;
    hk.env.vhd = hd;
    hk.env.vhd_f = hd;
    hk.env.no_rope = true;          // the pooler carries no position information
    vqa::vit::BlockHooks hooks;
    hooks.attention = attention_hook;
    hooks.gelu = gelu_hook;
    hooks.ud = &hk;

    for (int il = 0; il < n_pooler; il++) {
        hk.env.layer = il;
        x = vqa::vit::block(gctx, x, pool_layers[(size_t)il], ln_eps, hooks, il);
    }

    // --- cross attention: query tokens x (normalized) stream
    ggml_tensor * h = ggml_norm(gctx, x, ln_eps);
    h = ggml_add(gctx, ggml_mul(gctx, h, pool_cross.ln1_w), pool_cross.ln1_b);
    ggml_tensor * kv = ggml_mul_mat(gctx, pool_cross.kv_w, h);
    kv = ggml_add(gctx, kv, pool_cross.kv_b);                       // [2*n_embd, N]
    ggml_tensor * qv = ggml_reshape_2d(gctx, pool_query, n_embd, 1);   // one query token
    ggml_tensor * q = ggml_mul_mat(gctx, pool_cross.q_w, qv);
    q = ggml_add(gctx, q, pool_cross.q_b);                           // [n_embd, 1]

    // flash-attn wants [head_dim, tokens, heads] on all three operands
    const size_t qkv_tok = kv->nb[1];
    ggml_tensor * qa = ggml_view_3d(gctx, q, hd, 1, n_head, q->nb[1], (size_t)hd * es, 0);
    ggml_tensor * ka = ggml_view_3d(gctx, kv, hd, N, n_head, qkv_tok, (size_t)hd * es, 0);
    ggml_tensor * va = ggml_view_3d(gctx, kv, hd, N, n_head, qkv_tok, (size_t)hd * es,
                                    (size_t)n_embd * es);
    ggml_tensor * fa = ggml_flash_attn_ext(gctx, qa, ka, va, nullptr,
                                           1.0f / sqrtf((float)hd), 0.0f, 0.0f);
    ggml_tensor * attn = ggml_reshape_2d(gctx, ggml_cont(gctx, fa), n_embd, 1);

    ggml_tensor * o = ggml_add(gctx, qv, attn);   // residual on the query token
    ggml_tensor * l2 = ggml_norm(gctx, o, ln_eps);
    l2 = ggml_add(gctx, ggml_mul(gctx, l2, pool_cross.ln2_w), pool_cross.ln2_b);
    ggml_tensor * f = ggml_mul_mat(gctx, pool_cross.fc0_w, l2);
    f = ggml_add(gctx, f, pool_cross.fc0_b);
    f = gelu_hook(&hk, f);
    f = ggml_mul_mat(gctx, pool_cross.fc1_w, f);
    f = ggml_add(gctx, f, pool_cross.fc1_b);
    o = ggml_add(gctx, o, f);
    ggml_set_name(o, "v_pooled_raw");
    return o;
}

// ---------------------------------------------------------------- forward
void VJEPA2::PredictorGraph::reset() {
    if (gf) { ggml_free(ctx); gf = nullptr; ctx = nullptr; }
    if (allocr) { ggml_gallocr_free(allocr); allocr = nullptr; }
    out = in_feat = in_ctx = in_order = in_tgt = in_mask = nullptr;
    t_rope_e = t_rope_o = nullptr;
    n_ctx = n_tgt = 0;
}

bool VJEPA2::encode(const vqa::MediaInput & media, std::vector<float> & pooled,
                    std::vector<float> & probs, int * n_tokens) {
    pooled.clear();
    probs.clear();
    *n_tokens = 0;
    if (!gf) { fprintf(stderr, "%s: graph not built\n", arch_name.c_str()); return false; }

    videomae::PreConfig pc;
    pc.image_size = image_size;
    pc.resize_shortest = resize_shortest;
    pc.resample = resample;
    pc.patch = patch;
    pc.tubelet = tubelet;
    pc.num_frames = num_frames;
    for (int c = 0; c < 3; c++) { pc.mean[c] = mean[c]; pc.std[c] = std[c]; }

    std::vector<float> patches;
    int gt = 0, gh = 0, gw = 0;
    if (!videomae::pack_clip(media, pc, patches, n_tokens, &gt, &gh, &gw)) return false;
    if (*n_tokens != n_patches) {
        fprintf(stderr, "clip produced %d tokens, model expects %d\n", *n_tokens, n_patches);
        return false;
    }

    auto t0 = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(in_patches, patches.data(), 0, sizeof(float) * patches.size());
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        return false;
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[perf] %s: clip %d tokens x %d layers in %.0f ms\n", arch_name.c_str(),
            n_patches, n_layer, ms);

    pooled.resize((size_t)n_embd);
    ggml_backend_tensor_get(t_pooled, pooled.data(), 0, sizeof(float) * (size_t)n_embd);
    if (t_probs) {
        probs.resize((size_t)n_labels);
        ggml_backend_tensor_get(t_probs, probs.data(), 0, sizeof(float) * (size_t)n_labels);
    }

    dump("patches");
    dump("v_patch_out");
    dump("v_blk0_qkv");
    dump("v_blk0_q_rope");
    dump("v_blk0_k_rope");
    dump("v_blk0_attn");
    dump("v_blk0_out");
    dump("v_final");
    dump("v_pooled");
    if (t_probs) {
        dump("v_logits");
        dump("v_probs");
    }
    return true;
}

// Masked-latent prediction (the world-model half). `ctx_idx` are the token
// indices of the encoder stream the predictor gets to see, `tgt_idx` the ones it
// must predict; the predictor embeds the context, appends learned mask tokens at
// the target positions, sorts the whole sequence by position, runs its own
// narrow stack (384-d, own rope tables), then projects the target tokens back to
// the 1024-d encoder space.
//
// `features` may be null, in which case the caller already ran encode() and the
// v_final stream is still resident in the main graph.
bool VJEPA2::predict(std::vector<int> ctx_idx, std::vector<int> tgt_idx, int mask_index,
                     std::vector<float> & latents, const std::vector<float> * features) {
    const int N = n_patches;
    const int n_ctx = (int)ctx_idx.size();
    const int n_tgt = (int)tgt_idx.size();
    latents.clear();
    if (!has_predictor) {
        fprintf(stderr, "%s: this GGUF has no predictor (re-convert without "
                        "--no-predictor)\n", arch_name.c_str());
        return false;
    }
    if (n_ctx <= 0 || n_tgt <= 0) {
        fprintf(stderr, "predict: the context and target index sets must be non-empty\n");
        return false;
    }

    auto check = [&](std::vector<int> & v, const char * what) {
        for (int & i : v) {
            if (i < 0 || i >= N) {
                fprintf(stderr, "predict: %s index %d is outside 0..%d\n", what, i, N - 1);
                return false;
            }
        }
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return true;
    };
    if (!check(ctx_idx, "context") || !check(tgt_idx, "target")) return false;
    for (int i : ctx_idx)
        if (std::binary_search(tgt_idx.begin(), tgt_idx.end(), i)) {
            fprintf(stderr, "predict: token %d is in both the context and the target set\n", i);
            return false;
        }

    // order: context tokens first, then mask tokens, sorted by position (this is
    // `sort_tokens` in transformers; positions are the token indices themselves)
    const int n_tot = n_ctx + n_tgt;
    if (pg.gf && (pg.n_ctx != n_ctx || pg.n_tgt != n_tgt)) pg.reset();

    auto tb0 = std::chrono::steady_clock::now();
    if (!pg.gf) {
        // ---- build the predictor graph once per (n_ctx, n_tgt) split
        struct ggml_init_params ip = { 32ull * 1024 * 1024, NULL, /*no_alloc*/ true };
        pg.ctx = ggml_init(ip);
        if (!pg.ctx) return false;
        pg.gf = ggml_new_graph_custom(pg.ctx, 32768, false);
        ggml_context * c = pg.ctx;
        const int phd = pred_dim / pred_head;
        const int phalf = phd / 2;   // table width, one column per dim pair

        pg.in_feat = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_embd, N);
        pg.in_ctx = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_ctx);
        pg.in_order = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tot);
        pg.in_tgt = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_tgt);
        pg.in_mask = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
        pg.t_rope_e = new_rope_table(c, phalf, n_tot);
        pg.t_rope_o = new_rope_table(c, phalf, n_tot);
        ggml_set_name(pg.in_feat, "p_in_feat");

        Hooks hk;
        hk.act = hidden_act;
        hk.names = false;
        vqa::vit::VitHookEnv & env = hk.env;
        env.gctx = c;
        env.n_head = pred_head;
        env.n_embd = pred_dim;
        env.N = n_tot;
        env.vhd = phd;
        env.vhd_f = phd;
        env.t_rope = pg.t_rope_e;
        env.t_rope_odd = pg.t_rope_o;
        vqa::vit::BlockHooks hooks;
        hooks.attention = attention_hook;
        hooks.gelu = gelu_hook;
        hooks.ud = &hk;

        ggml_tensor * ctx_feat = ggml_get_rows(c, pg.in_feat, pg.in_ctx);   // [n_embd, n_ctx]
        ggml_tensor * y = ggml_mul_mat(c, p_emb_w, ctx_feat);
        y = ggml_add(c, y, p_emb_b);                                       // [pred_dim, n_ctx]
        ggml_tensor * mt = ggml_get_rows(c, p_mask, pg.in_mask);           // [pred_dim, 1]
        ggml_tensor * mt_rep = ggml_new_tensor_2d(c, GGML_TYPE_F32, pred_dim, n_tgt);
        mt = ggml_repeat(c, mt, mt_rep);
        y = ggml_concat(c, y, mt, 1);                                      // [pred_dim, n_tot]
        y = ggml_get_rows(c, y, pg.in_order);                              // sorted by position
        for (int il = 0; il < pred_layer; il++) {
            env.layer = il;
            y = vqa::vit::block(c, y, pred_layers[(size_t)il], ln_eps, hooks, il);
        }
        y = ggml_norm(c, y, ln_eps);
        y = ggml_add(c, ggml_mul(c, y, p_norm_w), p_norm_b);
        y = ggml_get_rows(c, y, pg.in_tgt);                                // [pred_dim, n_tgt]
        y = ggml_mul_mat(c, p_proj_w, y);
        y = ggml_add(c, y, p_proj_b);                                      // [n_embd, n_tgt]
        pg.out = y;
        ggml_set_name(y, "p_latents");
        ggml_build_forward_expand(pg.gf, y);
        pg.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!pg.allocr || !ggml_gallocr_alloc_graph(pg.allocr, pg.gf)) {
            fprintf(stderr, "predictor graph allocation failed\n");
            pg.reset();
            return false;
        }
        pg.n_ctx = n_ctx;
        pg.n_tgt = n_tgt;
        fprintf(stderr, "[perf] %s: predictor graph (%d ctx + %d tgt tokens) %.0f ms\n",
                arch_name.c_str(), n_ctx, n_tgt,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tb0).count());
    }

    // ---- per-call inputs: features, gather orders, rope tables
    std::vector<float> feat;
    if (features) {
        feat = *features;
    } else {
        if (!t_final) return false;
        feat.resize((size_t)n_embd * (size_t)N);
        ggml_backend_tensor_get(t_final, feat.data(), 0,
                                sizeof(float) * (size_t)n_embd * (size_t)N);
    }
    if ((int)feat.size() != n_embd * N) {
        fprintf(stderr, "predict: feature buffer has %zu values, expected %d\n",
                feat.size(), n_embd * N);
        return false;
    }

    // pos[i] of every output slot, and the sort permutation over the
    // [context | mask] sequence
    std::vector<int32_t> pos((size_t)n_tot), order((size_t)n_tot);
    for (int i = 0; i < n_ctx; i++) pos[(size_t)i] = ctx_idx[(size_t)i];
    for (int i = 0; i < n_tgt; i++) pos[(size_t)(n_ctx + i)] = tgt_idx[(size_t)i];
    for (int i = 0; i < n_tot; i++) order[(size_t)i] = i;
    std::sort(order.begin(), order.end(),
              [&](int32_t a, int32_t b) { return pos[(size_t)a] < pos[(size_t)b]; });

    // after the stack the target tokens sit at their sorted slots: gather them
    // back in target order (unsort + "take the last n_tgt" in one step)
    std::vector<int32_t> tgt_slots((size_t)n_tgt);
    {
        std::vector<int32_t> slot_of_pos((size_t)N, -1);
        for (int j = 0; j < n_tot; j++) slot_of_pos[(size_t)pos[(size_t)order[(size_t)j]]] = j;
        for (int i = 0; i < n_tgt; i++) tgt_slots[(size_t)i] = slot_of_pos[(size_t)tgt_idx[(size_t)i]];
    }

    ggml_backend_tensor_set(pg.in_feat, feat.data(), 0, sizeof(float) * feat.size());
    ggml_backend_tensor_set(pg.in_ctx, ctx_idx.data(), 0, sizeof(int32_t) * (size_t)n_ctx);
    ggml_backend_tensor_set(pg.in_order, order.data(), 0, sizeof(int32_t) * (size_t)n_tot);
    ggml_backend_tensor_set(pg.in_tgt, tgt_slots.data(), 0, sizeof(int32_t) * (size_t)n_tgt);
    int32_t mi = mask_index % std::max(1, pred_n_mask);
    ggml_backend_tensor_set(pg.in_mask, &mi, 0, sizeof(int32_t));

    // rope tables for the sorted positions (per-axis grid decomposition)
    {
        const int per_frame = grid_h * grid_w;
        std::vector<int32_t> pt((size_t)n_tot), ph((size_t)n_tot), pw((size_t)n_tot);
        for (int j = 0; j < n_tot; j++) {
            const int32_t p = pos[(size_t)order[(size_t)j]];
            pt[(size_t)j] = p / per_frame;
            ph[(size_t)j] = (p % per_frame) / grid_w;
            pw[(size_t)j] = p % grid_w;
        }
        const int phalf = pred_dim / pred_head / 2;
        fill_rope_tables(pg.t_rope_e, pg.t_rope_o, phalf, n_tot, pred_seg / 2,
                         rope_columns(phalf, pred_seg),
                         pt.data(), ph.data(), pw.data(), rope_theta);
    }

    if (ggml_backend_graph_compute(backend, pg.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "predictor compute failed\n");
        return false;
    }
    latents.resize((size_t)n_embd * (size_t)n_tgt);
    ggml_backend_tensor_get(pg.out, latents.data(), 0, sizeof(float) * latents.size());
    return true;
}

// ---------------------------------------------------------------- ask / embed
bool VJEPA2::ask(const vqa::MediaInput & media, const std::string & prompt,
                 const std::string & mode, const vqa::GenParams & gp,
                 std::string & out_text) {
    (void)prompt;   // the prompt is meaningless for a video encoder
    (void)gp;
    out_text.clear();
    if (media.empty()) {
        fprintf(stderr, "%s: no frames (pass --video/--frames, or --image)\n",
                arch_name.c_str());
        return false;
    }
    if ((int)media.frames.size() != num_frames)
        fprintf(stderr, "%s: %zu frame%s in, using %d uniform samples\n", arch_name.c_str(),
                media.frames.size(), media.frames.size() == 1 ? "" : "s", num_frames);

    if (gp.on_stage) gp.on_stage("vision", 0, 1, gp.on_stage_ud);
    std::vector<float> pooled, probs;
    int n_tokens = 0;
    if (!encode(media, pooled, probs, &n_tokens)) return false;
    if (gp.on_stage) gp.on_stage("vision", 1, 1, gp.on_stage_ud);

    char buf[512];
    if (!dump_dir.empty()) {
        std::string path = dump_dir + "/vjepa2_pooled.f32";
        FILE * f = fopen(path.c_str(), "wb");
        if (f) { fwrite(pooled.data(), sizeof(float), pooled.size(), f); fclose(f); }
    }

    // masked-latent prediction: mode "predict", the spec rides in the prompt:
    //   ctx=0:4096,tgt=4096:8192[,mask=0]      (also accepts "0-4095" / "0,5,9")
    if (mode == "predict") return predict_cmd(prompt, pooled, out_text);

    if (!probs.empty()) {
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

    if (mode == "embed") {
        out_text = "[";
        for (size_t i = 0; i < pooled.size(); i++) {
            snprintf(buf, sizeof(buf), "%s%.6f", i ? ", " : "", (double)pooled[i]);
            out_text += buf;
        }
        out_text += "]";
        return true;
    }
    double l2 = 0.0, mean_v = 0.0;
    for (float v : pooled) { l2 += (double)v * v; mean_v += v; }
    l2 = sqrt(l2);
    mean_v /= (double)std::max<size_t>(1, pooled.size());
    snprintf(buf, sizeof(buf),
             "V-JEPA 2 encoder: %d-d %s video embedding (l2 %.4f, mean %.5f)\n"
             "(first values: %.5f %.5f %.5f %.5f)\n"
             "no classifier in this checkpoint: use --mode embed for the full vector%s\n",
             n_embd, has_pooler ? "attentive-pooled" : "mean-pooled", l2, mean_v,
             (double)pooled[0], (double)pooled[1], (double)pooled[2], (double)pooled[3],
             has_predictor ? ", --mode predict for masked-latent prediction" : "");
    out_text = buf;
    return true;
}

// `--mode predict` driver. The spec rides in the prompt so the Model interface
// (and every front end) stays unchanged:
//   ctx=0:4096,tgt=4096:8192            half-open token ranges
//   ctx=0-3999,tgt=4000-4095            inclusive ranges
//   ctx=0:2000/6000:7000,tgt=4000:5000/mask=2
// mask selects which learned mask token is used (default 1, matching
// transformers, whose predictor forwards the default and never the argument).
bool VJEPA2::predict_cmd(const std::string & spec, const std::vector<float> & pooled,
                         std::string & out_text) {
    (void)pooled;
    std::vector<int> ctx_idx, tgt_idx;
    // transformers' VJEPA2Predictor.forward never forwards its mask_index to the
    // embeddings, so it always uses index 1 (which is all zeros in the released
    // checkpoints). Default to the same token for parity; `mask=0` selects the
    // one trained mask token instead.
    int mask_index = 1;
    bool have_ctx = false, have_tgt = false;

    std::string cur;
    auto flush = [&]() {
        if (cur.empty()) return;
        const size_t eq = cur.find('=');
        if (eq == std::string::npos) { cur.clear(); return; }
        const std::string key = cur.substr(0, eq), val = cur.substr(eq + 1);
        cur.clear();
        std::vector<int> * dst = nullptr;
        if (key == "ctx" || key == "context") { dst = &ctx_idx; have_ctx = true; }
        else if (key == "tgt" || key == "target") { dst = &tgt_idx; have_tgt = true; }
        else if (key == "mask") { mask_index = atoi(val.c_str()); return; }
        else {
            fprintf(stderr, "predict: unknown key '%s' (ctx / tgt / mask)\n", key.c_str());
            return;
        }
        // items: "a:b" half-open, "a-b" inclusive, "n" single
        size_t p = 0;
        while (p <= val.size()) {
            const size_t q = std::min(val.size(), val.find('/', p));
            const std::string item = val.substr(p, q - p);
            p = q + 1;
            if (item.empty()) break;
            const size_t sep = item.find_first_of(":-");
            int a = 0, b = 0;
            if (sep == std::string::npos) {
                a = atoi(item.c_str());
                b = a + 1;
            } else {
                a = atoi(item.substr(0, sep).c_str());
                b = atoi(item.substr(sep + 1).c_str());
                if (item[sep] == '-') b += 1;   // inclusive upper bound
            }
            for (int i = a; i < b; i++) dst->push_back(i);
        }
    };
    for (char ch : spec) {
        if (ch == ',' || ch == ';' || ch == ' ') flush();
        else cur.push_back(ch);
    }
    flush();
    if (!have_ctx || !have_tgt) {
        fprintf(stderr, "predict: needs both ctx=<ranges> and tgt=<ranges>, e.g. "
                        "\"ctx=0:4096,tgt=4096:8192\"\n");
        return false;
    }

    std::vector<float> latents;
    const auto t0 = std::chrono::steady_clock::now();
    if (!predict(std::move(ctx_idx), std::move(tgt_idx), mask_index, latents, nullptr))
        return false;
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    const int n_tgt = pred_dim > 0 ? (int)(latents.size() / (size_t)n_embd) : 0;

    double l2 = 0.0;
    for (float v : latents) l2 += (double)v * v;
    l2 = sqrt(l2);
    fprintf(stderr, "[perf] %s: predictor %d target tokens x %d layers in %.0f ms\n",
            arch_name.c_str(), n_tgt, pred_layer, ms);

    std::string out_path;
    if (!dump_dir.empty()) {
        out_path = dump_dir + "/p_latents.f32";
        if (FILE * f = fopen(out_path.c_str(), "wb")) {
            fwrite(latents.data(), sizeof(float), latents.size(), f);
            fclose(f);
            if (FILE * sf = fopen((out_path + ".shape").c_str(), "w")) {
                fprintf(sf, "%d %d\n", n_embd, n_tgt);
                fclose(sf);
            }
        }
    }

    char buf[512];
    out_text = "{\"dim\": " + std::to_string(n_embd) +
               ", \"tokens\": " + std::to_string(n_tgt) +
               ", \"layers\": " + std::to_string(pred_layer) +
               ", \"l2\": ";
    snprintf(buf, sizeof(buf), "%.4f, \"first\": [", l2);
    out_text += buf;
    for (int i = 0; i < std::min<int>(16, (int)latents.size()); i++) {
        snprintf(buf, sizeof(buf), "%s%.6f", i ? ", " : "", (double)latents[(size_t)i]);
        out_text += buf;
    }
    out_text += "]";
    if (!out_path.empty()) out_text += ", \"out\": \"" + out_path + "\"";
    out_text += "}\n";
    return true;
}

bool VJEPA2::embed(const vqa::MediaInput & media, const std::string & text,
                   const vqa::EmbedParams & ep, std::vector<float> & out) {
    (void)text;   // pure video tower: text is not part of the input space
    (void)ep;
    out.clear();
    if (media.empty()) return false;
    std::vector<float> pooled, probs;
    int n_tokens = 0;
    if (!encode(media, pooled, probs, &n_tokens)) return false;
    double l2 = 0.0;
    for (float v : pooled) l2 += (double)v * v;
    const double inv = l2 > 0.0 ? 1.0 / sqrt(l2) : 0.0;
    out.resize(pooled.size());
    for (size_t i = 0; i < pooled.size(); i++) out[i] = (float)(pooled[i] * inv);
    return true;
}

}  // namespace vjepa2

namespace vqa {
Model * create_vjepa2() { return new vjepa2::VJEPA2(); }
}  // namespace vqa
