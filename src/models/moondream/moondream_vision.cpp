// moondream_vision.cpp — SigLIP vision encoder + projection (see .hpp for the math).
//
// ggml notes / pitfalls honored here (same as the text engine):
//  - ggml_mul_mat(a, b): result ne = [a.ne1, b.ne1, ...]; weights stored [in, out]
//    as ne=[in,out], activations [dim, tokens].
//  - CPU binary ops (add/mul) forbid mixed F32/F16: small F16 weights (norms,
//    biases, pos_emb) are converted to F32 copies at load time.
//  - avoid permute() semantics questions entirely: pos_emb is pre-transposed in
//    the GGUF; q/k/v are extracted with explicit-stride view_3d + cont().
//  - view strides are passed in bytes; for a [D,T] row-major F32 tensor
//    nb[0]=4, nb[1]=D*4.

#include "moondream_vision.hpp"
#include "vit_block.hpp"
#include <thread>
#include <atomic>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#   include "ggml-cuda.h"
#endif
#include "ggml-alloc.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   define NOMINMAX
#   include <windows.h>
#else
#   include <sys/mman.h>
#   include <fcntl.h>
#   include <unistd.h>
#endif

ggml_backend_t vision_pick_backend() {
#ifdef GGML_USE_CUDA
    if (!getenv("MD_CPU")) {
        ggml_backend_t b = ggml_backend_cuda_init(0);
        if (b) {
            fprintf(stderr, "vision backend: CUDA (%s)\n", ggml_backend_name(b));
            return b;
        }
    }
#endif
    ggml_backend_t b = ggml_backend_cpu_init();
    if (const char * nt = getenv("MD_THREADS"))
        ggml_backend_cpu_set_n_threads(b, atoi(nt));
    fprintf(stderr, "vision backend: %s\n", ggml_backend_name(b));
    return b;
}

// ---------------------------------------------------------------- weights
struct VisionLayer {
    struct ggml_tensor * ln1_w = nullptr, * ln1_b = nullptr;
    struct ggml_tensor * qkv_w = nullptr, * qkv_b = nullptr;
    struct ggml_tensor * attn_out_w = nullptr, * attn_out_b = nullptr;
    struct ggml_tensor * ln2_w = nullptr, * ln2_b = nullptr;
    struct ggml_tensor * ffn_up_w = nullptr, * ffn_up_b = nullptr;
    struct ggml_tensor * ffn_down_w = nullptr, * ffn_down_b = nullptr;
};

struct VisionWeights {
    struct ggml_tensor * patch_emb_w = nullptr, * patch_emb_b = nullptr;
    struct ggml_tensor * pos_emb = nullptr;              // [enc_dim, grid*grid] F32
    struct ggml_tensor * post_ln_w = nullptr, * post_ln_b = nullptr;
    struct ggml_tensor * proj_fc1_w = nullptr, * proj_fc1_b = nullptr;
    struct ggml_tensor * proj_fc2_w = nullptr, * proj_fc2_b = nullptr;
    std::vector<VisionLayer> layers;
    vqa::MMap mm;
    struct ggml_context * wctx = nullptr;
    void * f32_cpu = nullptr;
};

static VisionWeights g_w;   // single model instance is enough for this milestone

VisionModel::~VisionModel() = default;

bool VisionModel::load(const char * path, ggml_backend_t backend) {
    VisionWeights & W = g_w;
    if (!W.mm.load(path)) { fprintf(stderr, "vision mmap failed: %s\n", path); return false; }
    fprintf(stderr, "vision model file: %.2f GB\n", W.mm.size / 1e9);

    struct gguf_context * gu = nullptr;
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &W.wctx };
    gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "vision gguf init failed\n"); return false; }

    auto get_u32 = [&](const char * k, int & v) {
        int64_t id = gguf_find_key(gu, k);
        if (id >= 0) v = (int)gguf_get_val_u32(gu, id);
    };
    const char * v = "moondream3.vision.";
    get_u32("vision.block_count", n_layers);
    get_u32(std::string(std::string(v) + "enc_dim").c_str(), enc_dim);
    get_u32(std::string(std::string(v) + "patch_size").c_str(), patch);
    get_u32(std::string(std::string(v) + "enc_n_layers").c_str(), n_layers);
    get_u32(std::string(std::string(v) + "enc_ff_dim").c_str(), ff);
    get_u32(std::string(std::string(v) + "enc_n_heads").c_str(), n_heads);
    get_u32(std::string(std::string(v) + "proj_out_dim").c_str(), proj_out);
    get_u32(std::string(std::string(v) + "crop_size").c_str(), crop);
    get_u32(std::string(std::string(v) + "max_crops").c_str(), max_crops);
    get_u32(std::string(std::string(v) + "overlap_margin").c_str(), overlap);
    get_u32(std::string(std::string(v) + "proj_inner_dim").c_str(), proj_inner);
    head_dim = enc_dim / n_heads;
    grid = crop / patch;
    patch_dim = in_ch * patch * patch;

    const size_t data_off = gguf_get_data_offset(gu);
    const int64_t nt = gguf_get_n_tensors(gu);
    for (int64_t i = 0; i < nt; ++i) {
        const char * name = gguf_get_tensor_name(gu, i);
        struct ggml_tensor * t = ggml_get_tensor(W.wctx, name);
        if (!t) continue;
        t->data = (char *)W.mm.ptr + data_off + gguf_get_tensor_offset(gu, i);
    }
    gguf_free(gu);

    // small F16 tensors used in binary ops (norms, biases, pos_emb) -> CPU F32
    // copies: CPU add/mul forbid F32(x) x F16(w). Weights used only in mul_mat
    // stay F16 (mixed mul_mat is fine and battle-tested).
    struct ggml_backend_buffer * f32buf = nullptr;
    {
        size_t total = 0;
        for (auto t = ggml_get_first_tensor(W.wctx); t; t = ggml_get_next_tensor(W.wctx, t)) {
            if (t->type != GGML_TYPE_F16) continue;
            bool is_bias_like = (t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1) || t == ggml_get_tensor(W.wctx, "pos_emb.weight");
            if (!is_bias_like) continue;
            total += ggml_nelements(t) * sizeof(float);
        }
        void * f32data = malloc(total ? total : 1);
        size_t off = 0;
        for (auto t = ggml_get_first_tensor(W.wctx); t; t = ggml_get_next_tensor(W.wctx, t)) {
            if (t->type != GGML_TYPE_F16) continue;
            bool is_bias_like = (t->ne[1] == 1 && t->ne[2] == 1 && t->ne[3] == 1) || t == ggml_get_tensor(W.wctx, "pos_emb.weight");
            if (!is_bias_like) continue;
            int64_t n = ggml_nelements(t);
            const uint16_t * src = (const uint16_t *)t->data;
            float * dst = (float *)((char *)f32data + off);
            for (int64_t j = 0; j < n; j++) dst[j] = ggml_fp16_to_fp32(src[j]);
            t->data = dst;
            t->type = GGML_TYPE_F32;
            t->nb[0] = sizeof(float);
            t->nb[1] = t->ne[0] * sizeof(float);
            t->nb[2] = t->nb[1] * t->ne[1];
            t->nb[3] = t->nb[2] * t->ne[2];
            off += n * sizeof(float);
        }
        W.f32_cpu = f32data;

        // patch-embedding bias folded into the position table: pos is F32 here,
        // the fold is a host-side add, and the graph drops one [D, T] activation
        // add per crop batch (exact — same trick as the VideoChat3 tower)
        {
            ggml_tensor * pos = ggml_get_tensor(W.wctx, "pos_emb.weight");
            ggml_tensor * pb  = ggml_get_tensor(W.wctx, "patch_emb.bias");
            if (pos && pb && pos->type == GGML_TYPE_F32 && pb->type == GGML_TYPE_F32 &&
                pb->ne[0] == pos->ne[0] && pb->ne[1] == 1) {
                vqa::vit::fold_patch_bias((float *)pos->data, (size_t)ggml_nelements(pos),
                                          (int)pos->ne[0], (const float *)pb->data);
            }
        }

        if (ggml_backend_is_cpu(backend)) {
            struct ggml_backend_buffer * wbuf = ggml_backend_cpu_buffer_from_ptr(W.mm.ptr, W.mm.size);
            f32buf = total ? ggml_backend_cpu_buffer_from_ptr(f32data, total) : nullptr;
            for (auto t = ggml_get_first_tensor(W.wctx); t; t = ggml_get_next_tensor(W.wctx, t))
                t->buffer = (t->data >= f32data && t->data < (char *)f32data + total) ? f32buf : wbuf;
        } else {
            std::vector<std::pair<struct ggml_tensor *, const void *>> uploads;
            size_t up_total = 0;
            for (auto t = ggml_get_first_tensor(W.wctx); t; t = ggml_get_next_tensor(W.wctx, t)) {
                uploads.push_back({t, t->data});
                up_total += ggml_nbytes(t);
            }
            for (auto & [t, src] : uploads) t->data = nullptr;
            struct ggml_backend_buffer * wbuf = ggml_backend_alloc_ctx_tensors(W.wctx, backend);
            if (!wbuf) { fprintf(stderr, "vision weight alloc failed (%.2f GB)\n", up_total / 1e9); return false; }
            for (auto & [t, src] : uploads)
                ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
            fprintf(stderr, "uploaded %.2f GB vision weights to backend\n", up_total / 1e9);
            W.mm.release();
            free(W.f32_cpu); W.f32_cpu = nullptr;
        }
    }

    W.patch_emb_w = ggml_get_tensor(W.wctx, "patch_emb.weight");
    W.patch_emb_b = ggml_get_tensor(W.wctx, "patch_emb.bias");
    W.pos_emb     = ggml_get_tensor(W.wctx, "pos_emb.weight");
    W.post_ln_w   = ggml_get_tensor(W.wctx, "post_ln.weight");
    W.post_ln_b   = ggml_get_tensor(W.wctx, "post_ln.bias");
    W.proj_fc1_w  = ggml_get_tensor(W.wctx, "proj_mlp.fc1.weight");
    W.proj_fc1_b  = ggml_get_tensor(W.wctx, "proj_mlp.fc1.bias");
    W.proj_fc2_w  = ggml_get_tensor(W.wctx, "proj_mlp.fc2.weight");
    W.proj_fc2_b  = ggml_get_tensor(W.wctx, "proj_mlp.fc2.bias");
    if (!W.patch_emb_w || !W.pos_emb || !W.proj_fc1_w) { fprintf(stderr, "missing vision tensors\n"); return false; }
    if (getenv("MD_TRACE")) {
        float b0 = ((const float *)W.patch_emb_b->data)[0];
        float p0 = ((const float *)W.pos_emb->data)[0], p1 = ((const float *)W.pos_emb->data)[1];
        fprintf(stderr, "loaded patch_emb_b[0]=%.5f pos[0]=%.5f pos[1]=%.5f (types %d/%d)\n",
                b0, p0, p1, (int)W.patch_emb_b->type, (int)W.pos_emb->type);
    }

    W.layers.resize(n_layers);
    std::vector<char> buf(256);
    auto gt = [&](const char * fmt, int i, const char * suffix) {
        snprintf(buf.data(), buf.size(), fmt, i, suffix);
        return ggml_get_tensor(W.wctx, buf.data());
    };
    for (int i = 0; i < n_layers; i++) {
        VisionLayer & L = W.layers[i];
        L.ln1_w = gt("blk.%d.ln1.%s", i, "weight");
        L.ln1_b = gt("blk.%d.ln1.%s", i, "bias");
        L.qkv_w = gt("blk.%d.attn_qkv.%s", i, "weight");
        L.qkv_b = gt("blk.%d.attn_qkv.%s", i, "bias");
        L.attn_out_w = gt("blk.%d.attn_out.%s", i, "weight");
        L.attn_out_b = gt("blk.%d.attn_out.%s", i, "bias");
        L.ln2_w = gt("blk.%d.ln2.%s", i, "weight");
        L.ln2_b = gt("blk.%d.ln2.%s", i, "bias");
        L.ffn_up_w = gt("blk.%d.ffn_up.%s", i, "weight");
        L.ffn_up_b = gt("blk.%d.ffn_up.%s", i, "bias");
        L.ffn_down_w = gt("blk.%d.ffn_down.%s", i, "weight");
        L.ffn_down_b = gt("blk.%d.ffn_down.%s", i, "bias");
    }
    fprintf(stderr, "vision model loaded: %d layers, dim %d, heads %d (hd %d), ff %d, proj %d->%d\n",
            n_layers, enc_dim, n_heads, head_dim, ff, enc_dim * 2, proj_out);
    return true;
}

// ---- shared ViT block plumbing (vqa::vit::block) --------------------------
// The SigLIP block body lives in src/core/vit_block.hpp; each encoder path
// supplies only its attention wiring (single-crop FA / batched crops FA) and,
// for the batched path, the MD_VISION_F16ACT dtype casts.
namespace {
struct MdCropEnv {
    ggml_context *gctx;
    int D, H, HD, T, use_fa;
};

static ggml_tensor * md_crop_attention(void * ud_, ggml_tensor * qkv) {
    MdCropEnv & e = *(MdCropEnv *)ud_;
    ggml_context * gctx = e.gctx;
    const int D = e.D, H = e.H, HD = e.HD, T = e.T;
    // bidirectional attention; FA over [HD, T, H] views by default
    // (MD_VISION_FA=0 falls back to batched-3D matmuls; layout follows
    // llama.cpp clip.cpp build_attn)
    if (e.use_fa) {
        struct ggml_tensor * q3 = ggml_view_3d(gctx, qkv, HD, T, H, qkv->nb[1], HD * sizeof(float), 0);
        struct ggml_tensor * k3 = ggml_view_3d(gctx, qkv, HD, T, H, qkv->nb[1], HD * sizeof(float), (int64_t)H * HD * sizeof(float));
        struct ggml_tensor * v3 = ggml_view_3d(gctx, qkv, HD, T, H, qkv->nb[1], HD * sizeof(float), (int64_t)2 * H * HD * sizeof(float));
        return ggml_reshape_2d(gctx,
                ggml_flash_attn_ext(gctx, q3, k3, v3, nullptr, 1.0f / sqrtf((float)HD), 0.0f, 0.0f), D, T);
    }
    struct ggml_tensor * q3 = ggml_view_3d(gctx, qkv, HD, T, H, qkv->nb[1], HD * sizeof(float), 0);
    struct ggml_tensor * k3 = ggml_view_3d(gctx, qkv, HD, T, H, qkv->nb[1], HD * sizeof(float), (int64_t)H * HD * sizeof(float));
    struct ggml_tensor * v3 = ggml_view_3d(gctx, qkv, HD, T, H, qkv->nb[1], HD * sizeof(float), (int64_t)2 * H * HD * sizeof(float));
    // scores [T, T, H]: mul_mat res ne = [a.ne1, b.ne1, ne2]; softmax over ne0 = keys
    struct ggml_tensor * s = ggml_soft_max_ext(gctx, ggml_mul_mat(gctx, k3, q3),
            nullptr, 1.0f / sqrtf((float)HD), 0.0f);                    // [T, T, H]
    // out = v^T [T,HD,H] · w [T,T,H] -> [HD, T, H]; then [HD,H,T] -> [D, T]
    struct ggml_tensor * vt3 = ggml_cont(gctx, ggml_permute(gctx, v3, 1, 0, 2, 3));  // [T, HD, H]
    return ggml_cont_2d(gctx,
            ggml_permute(gctx, ggml_mul_mat(gctx, vt3, s), 0, 2, 1, 3), // [HD,T,H] -> [HD,H,T]
            D, T);
}

struct MdBatchEnv {
    ggml_context *gctx;
    int D, H, HD, T, C;
    ggml_type act_type = GGML_TYPE_F32;
};

static ggml_tensor * md_batch_attention(void * ud_, ggml_tensor * qkv) {
    MdBatchEnv & e = *(MdBatchEnv *)ud_;
    ggml_context * gctx = e.gctx;
    const int H = e.H, HD = e.HD, T = e.T, C = e.C;
    // q/k/v [HD, T, H, C]: within a token, heads live along ne0 (stride HD),
    // q/k/v blocks are offsets D apart; crops stride a whole qkv slab (nb[2])
    const size_t ts = ggml_type_size(qkv->type);
    auto cast_f32 = [&](ggml_tensor * t) {
        return qkv->type == GGML_TYPE_F32 ? t : ggml_cast(gctx, t, GGML_TYPE_F32);
    };
    struct ggml_tensor * q4 = ggml_view_4d(gctx, qkv, HD, T, H, C, qkv->nb[1], HD * ts, qkv->nb[2], 0);
    struct ggml_tensor * k4 = ggml_view_4d(gctx, qkv, HD, T, H, C, qkv->nb[1], HD * ts, qkv->nb[2], (int64_t)H * HD * ts);
    struct ggml_tensor * v4 = ggml_view_4d(gctx, qkv, HD, T, H, C, qkv->nb[1], HD * ts, qkv->nb[2], (int64_t)2 * H * HD * ts);
    // FA wants q in F32 (k/v may stay F16 — same conditional as the old inline code)
    struct ggml_tensor * fa = ggml_flash_attn_ext(gctx, cast_f32(q4), k4, v4,
            nullptr, 1.0f / sqrtf((float)HD), 0.0f, 0.0f);   // [HD, T, H, C]
    // same head-merge layout the single-crop path relies on after reshape_2d
    return ggml_reshape_3d(gctx, fa, e.D, T, C);
}

static ggml_tensor * md_to_act(void * ud_, ggml_tensor * t) {
    MdBatchEnv & e = *(MdBatchEnv *)ud_;
    return t->type == e.act_type ? t : ggml_cast(e.gctx, t, e.act_type);
}
static ggml_tensor * md_to_f32(void * ud_, ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(((MdBatchEnv *)ud_)->gctx, t, GGML_TYPE_F32);
}
}  // namespace

// ---------------------------------------------------------------- encoder graph
// runs ONE crop (grid*grid tokens) through blocks [0, max_layer); returns
// [grid*grid, enc_dim] token-major F32. Layer outputs optionally dumped.
static bool run_encoder_crop(VisionModel & M, ggml_backend_t backend, struct ggml_gallocr * gallocr,
                             const float * crop_chw, float * out_token_major,
                             std::vector<std::vector<float>> * per_layer_dump, int crop_idx,
                             int max_layer) {
    VisionWeights & W = g_w;
    const int D = M.enc_dim, H = M.n_heads, HD = M.head_dim;
    const int T = M.grid * M.grid;

    struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    struct ggml_context * gctx = ggml_init(ip);
    if (!gctx) return false;
    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);

    // patch extraction done on host. ggml tensors are COLUMN-major (ne[0] fastest):
    // the [588, 729] input must be laid out f + t*588, NOT row-major.
    // token t = (py/patch)*grid + (px/patch); feature f = c*patch*patch + (py%patch)*patch + (px%patch)
    std::vector<float> patches((size_t)M.patch_dim * T);
    const int CS = M.crop;
    for (int c = 0; c < M.in_ch; c++)
        for (int y = 0; y < CS; y++)
            for (int x = 0; x < CS; x++) {
                int t = (y / M.patch) * M.grid + (x / M.patch);
                int f = c * M.patch * M.patch + (y % M.patch) * M.patch + (x % M.patch);
                patches[(size_t)t * M.patch_dim + f] = crop_chw[(size_t)c * CS * CS + (size_t)y * CS + x];
            }

    struct ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, M.patch_dim, T);
    ggml_set_name(x, "patches_in");

    x = ggml_mul_mat(gctx, W.patch_emb_w, x);          // [D, T]
    // patch_emb_b is folded into pos_emb at load time (exact host F32 add)
    x = ggml_add(gctx, x, W.pos_emb);                  // pos_emb stored as ne=[D, T]
    ggml_set_name(x, "h_emb");

    // flash attention by default on both backends (earlier "CPU fattn breaks on
    // HD=72" was a misdiagnosis: the real bug was swapped view strides);
    // MD_VISION_FA=0 falls back to batched-3D matmuls
    static int use_fa = -1;
    if (use_fa < 0) {
        const char * e = getenv("MD_VISION_FA");
        use_fa = e ? atoi(e) : 1;
    }
    MdCropEnv env{gctx, D, H, HD, T, use_fa};
    for (int il = 0; il < max_layer; il++) {
        VisionLayer & L = W.layers[il];
        vqa::vit::BlockWeights bw{L.ln1_w, L.ln1_b, L.qkv_w, L.qkv_b, L.attn_out_w, L.attn_out_b,
                                  L.ln2_w, L.ln2_b, L.ffn_up_w, L.ffn_up_b, L.ffn_down_w, L.ffn_down_b};
        vqa::vit::BlockHooks bh;
        bh.attention = md_crop_attention;
        bh.ud = &env;
        x = vqa::vit::block(gctx, x, bw, 1e-5f, bh, il);
        if (!x) { ggml_free(gctx); return false; }

        char nm[32]; snprintf(nm, sizeof(nm), "h_L%d", il);
        ggml_set_name(x, nm);
    }

    if (max_layer == M.n_layers) {
        x = ggml_norm(gctx, x, 1e-5f);
        x = ggml_mul(gctx, x, W.post_ln_w);
        x = ggml_add(gctx, x, W.post_ln_b);
        ggml_set_name(x, "post_ln");
    }
    ggml_build_forward_expand(gf, x);

    if (!ggml_gallocr_alloc_graph(gallocr, gf)) { fprintf(stderr, "vision graph alloc failed\n"); ggml_free(gctx); return false; }

    struct ggml_tensor * pin = ggml_graph_get_tensor(gf, "patches_in");
    ggml_backend_tensor_set(pin, patches.data(), 0, sizeof(float) * patches.size());

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vision graph compute failed\n");
        ggml_free(gctx);
        return false;
    }

    // fetch [D, T] -> token-major [T, D]
    auto fetch = [&](struct ggml_tensor * t, float * dst) {
        std::vector<float> tmp(ggml_nelements(t));
        ggml_backend_tensor_get(t, tmp.data(), 0, ggml_nbytes(t));
        int64_t n0 = t->ne[0], n1 = t->ne[1];
        for (int64_t i1 = 0; i1 < n1; i1++)
            for (int64_t i0 = 0; i0 < n0; i0++)
                dst[i1 * n0 + i0] = tmp[i1 * n0 + i0];   // [D,T] row-major == column-major [T,D]
    };
    fetch(x, out_token_major);

    if (per_layer_dump) {
        for (int il = 0; il < max_layer; il++) {
            char nm[32]; snprintf(nm, sizeof(nm), "h_L%d", il);
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, nm);
            if (!t) continue;
            std::vector<float> d((size_t)T * D);
            fetch(t, d.data());
            per_layer_dump->push_back(std::move(d));
        }
        if (max_layer == M.n_layers) {
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, "post_ln");
            if (t) { std::vector<float> d((size_t)T * D); fetch(t, d.data()); per_layer_dump->push_back(std::move(d)); }
        }
    }
    (void)crop_idx;
    ggml_free(gctx);
    return true;
}

// batched variant: all C crops in ONE graph. Crops are independent, so the crop
// index rides as batch dim ne[2] (weights broadcast), and flash-attn batches
// heads and crops together in ne[2]/ne[3]. This amortizes kernel launches and
// keeps the GPU busy across the whole SigLIP forward instead of C sequential
// small graphs.
static bool run_encoder_crops_batched(VisionModel & M, ggml_backend_t backend, struct ggml_gallocr * gallocr,
                                      const float * crops, int n_crops, float * out_token_major,
                                      int max_layer) {
    VisionWeights & W = g_w;
    const int D = M.enc_dim, H = M.n_heads, HD = M.head_dim;
    const int T = M.grid * M.grid;

    // patch extraction on host for every crop -> [patch_dim, T, C] column-major.
    // ~4M scattered writes for 10 crops — parallel over crops to hide it.
    std::vector<float> patches((size_t)M.patch_dim * T * n_crops);
    const int CS = M.crop;
    {
        int nthreads = std::min(n_crops, (int)std::thread::hardware_concurrency());
        if (nthreads < 1) nthreads = 1;
        std::vector<std::thread> pool;
        std::atomic<int> next_crop{0};
        for (int th = 0; th < nthreads; th++) {
            pool.emplace_back([&]() {
                int c;
                while ((c = next_crop.fetch_add(1)) < n_crops) {
                    const float * crop_chw = crops + (size_t)c * M.in_ch * CS * CS;
                    float * pdst = patches.data() + (size_t)c * M.patch_dim * T;
                    for (int ch = 0; ch < M.in_ch; ch++)
                        for (int y = 0; y < CS; y++)
                            for (int x = 0; x < CS; x++) {
                                int t = (y / M.patch) * M.grid + (x / M.patch);
                                int f = ch * M.patch * M.patch + (y % M.patch) * M.patch + (x % M.patch);
                                pdst[(size_t)t * M.patch_dim + f] = crop_chw[(size_t)ch * CS * CS + (size_t)y * CS + x];
                            }
                }
            });
        }
        for (auto & t : pool) t.join();
    }

    struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    struct ggml_context * gctx = ggml_init(ip);
    if (!gctx) return false;
    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);

    // fp16 activations: MEASURED SLOWER than f32 here (490 vs 429ms/10 crops) —
    // ggml launches every cast as its own kernel, and cuBLAS already converts
    // f32 activations in-kernel, so fused-framework wins don't transfer.
    // Kept behind MD_VISION_F16ACT=1 for experimentation.
    static int f16act = -1;
    if (f16act < 0) {
        const char * e = getenv("MD_VISION_F16ACT");
        f16act = e ? atoi(e) : 0;
    }
    const ggml_type AT = f16act ? GGML_TYPE_F16 : GGML_TYPE_F32;
    auto to_act = [&](struct ggml_tensor * t) {   // -> activation dtype
        return t->type == AT ? t : ggml_cast(gctx, t, AT);
    };
    auto ln_f32 = [&](struct ggml_tensor * t) {   // layer norm in fp32
        struct ggml_tensor * f = t->type == GGML_TYPE_F32 ? t : ggml_cast(gctx, t, GGML_TYPE_F32);
        return f;
    };

    struct ggml_tensor * x = ggml_new_tensor_3d(gctx, AT, M.patch_dim, T, n_crops);
    ggml_set_name(x, "patches_in");

    x = ggml_mul_mat(gctx, W.patch_emb_w, x);          // [D, T, C] (f32 out)
    x = to_act(x);
    // patch_emb_b is folded into pos_emb at load time (exact host F32 add)
    x = ggml_add(gctx, x, W.pos_emb);                  // [D, T, 1] broadcasts over crops
    ggml_set_name(x, "h_emb");

    MdBatchEnv benv{gctx, D, H, HD, T, n_crops, AT};
    for (int il = 0; il < max_layer; il++) {
        VisionLayer & L = W.layers[il];
        vqa::vit::BlockWeights bw{L.ln1_w, L.ln1_b, L.qkv_w, L.qkv_b, L.attn_out_w, L.attn_out_b,
                                  L.ln2_w, L.ln2_b, L.ffn_up_w, L.ffn_up_b, L.ffn_down_w, L.ffn_down_b};
        vqa::vit::BlockHooks bh;
        bh.attention = md_batch_attention;
        bh.to_act = AT == GGML_TYPE_F32 ? nullptr : md_to_act;
        bh.to_f32 = AT == GGML_TYPE_F32 ? nullptr : md_to_f32;
        bh.ud = &benv;
        x = vqa::vit::block(gctx, x, bw, 1e-5f, bh, il);
        if (!x) { ggml_free(gctx); return false; }

        char nm[32]; snprintf(nm, sizeof(nm), "h_L%d", il);
        ggml_set_name(x, nm);
    }

    // final norm chain outputs F32 (norm is f32), so the fetch below is unchanged
    x = ggml_norm(gctx, ln_f32(x), 1e-5f);
    x = ggml_mul(gctx, x, ggml_cast(gctx, W.post_ln_w, GGML_TYPE_F32));
    x = ggml_add(gctx, x, ggml_cast(gctx, W.post_ln_b, GGML_TYPE_F32));
    ggml_set_name(x, "post_ln");
    ggml_build_forward_expand(gf, x);

    static bool dbg = getenv("MD_VISION_T") != nullptr;
    auto t_now = [&]() { return std::chrono::steady_clock::now(); };
    auto t0 = t_now();
    if (!ggml_gallocr_alloc_graph(gallocr, gf)) { fprintf(stderr, "vision graph alloc failed\n"); ggml_free(gctx); return false; }
    auto t1 = t_now();

    struct ggml_tensor * pin = ggml_graph_get_tensor(gf, "patches_in");
    if (AT == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> p16(patches.size());
        for (size_t i = 0; i < patches.size(); i++) p16[i] = ggml_fp32_to_fp16(patches[i]);
        ggml_backend_tensor_set(pin, p16.data(), 0, ggml_nbytes(pin));
    } else {
        ggml_backend_tensor_set(pin, patches.data(), 0, ggml_nbytes(pin));
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vision graph compute failed\n");
        ggml_free(gctx);
        return false;
    }
    auto t2 = t_now();

    // x is [D, T, C]; fetch whole buffer. The "transpose" index math is an
    // identity: out[(c*T+t)*D+d] == tmp[(c*T+t)*D+d] — column-major [D,T] per
    // crop IS token-major [T,D] in memory, so one memcpy replaces the loop.
    ggml_backend_tensor_get(x, out_token_major, 0, ggml_nbytes(x));
    if (dbg)
        fprintf(stderr, "  [chunk %dc] build+alloc %.0fms | compute %.0fms | fetch %.0fms\n", n_crops,
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count(),
                std::chrono::duration<double, std::milli>(t_now() - t2).count());
    ggml_free(gctx);
    return true;
}

std::vector<float> VisionModel::encode_crops(const float * crops, int n_crops, int max_layer,
                                             ggml_backend_t backend,
                                             std::vector<std::vector<float>> * per_layer_dump) {
    // profiling knob: cap layer count to isolate per-layer cost (output is then
    // garbage for real use — timing experiments only)
    if (const char * e = getenv("MD_VISION_LAYERS")) {
        int n = atoi(e);
        if (n > 0 && n < max_layer) max_layer = n;
    }
    const size_t crop_elems = (size_t)in_ch * crop * crop;
    const size_t T = (size_t)grid * grid;
    std::vector<float> out(n_crops * T * enc_dim);
    static int use_batch = -1;
    if (use_batch < 0) {
        const char * e = getenv("MD_VISION_BATCH");
        use_batch = e ? atoi(e) : 1;
    }
    if (use_batch && !per_layer_dump) {
        // all crops share weights and are independent -> batch them as ne[2].
        // Chunk size is bounded by FREE device memory, NOT by allocation failure:
        // under Windows WDDM an oversized cudaMalloc does not fail — the driver
        // silently demotes resident weight pages to shared system memory (PCIe
        // reads), permanently slowing decode afterwards. So we query free VRAM
        // up front and keep a safety margin. ~400MB/crop is a conservative
        // activation estimate (measured ~1.2GB pool peak for 4 crops + fragmentation).
        static int chunk_env = -1;
        if (chunk_env < 0) {
            const char * e = getenv("MD_VISION_CHUNK");
            chunk_env = e ? atoi(e) : 0;   // 0 = auto-size from free VRAM
        }
        int chunk = chunk_env;
        if (chunk < 1) {
            chunk = n_crops;
#if defined(GGML_USE_CUDA)
            if (ggml_backend_is_cuda(backend)) {
                size_t dev = 0, free_b = 0, total_b = 0;
                ggml_backend_cuda_get_device_memory(0, &free_b, &total_b);
                const size_t margin = 600ull << 20, per_crop = 400ull << 20;
                chunk = (int)std::min<size_t>(n_crops, free_b > margin ? (free_b - margin) / per_crop : 0);
                if (chunk < 1) {
                    fprintf(stderr, "vision: only %.2f GB free, falling back to chunk=1 "
                            "(risk of WDDM paging)\n", free_b / 1073741824.0);
                    chunk = 1;
                }
            }
#endif
            fprintf(stderr, "vision: free-VRAM-aware chunk = %d\n", chunk);
        }
        struct ggml_gallocr * ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        bool ok = true;
        for (int off = 0; off < n_crops && ok; off += chunk) {
            int n = std::min(chunk, n_crops - off);
            if (!run_encoder_crops_batched(*this, backend, ga, crops + (size_t)off * crop_elems,
                                           n, out.data() + (size_t)off * T * enc_dim, max_layer)) {
                ok = false;
            }
        }
        ggml_gallocr_free(ga);
        return ok ? out : std::vector<float>();
    }
    struct ggml_gallocr * gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    for (int c = 0; c < n_crops; c++) {
        if (!run_encoder_crop(*this, backend, gallocr, crops + c * crop_elems,
                              out.data() + c * T * enc_dim, per_layer_dump, c, max_layer)) {
            ggml_gallocr_free(gallocr);
            return {};
        }
    }
    ggml_gallocr_free(gallocr);
    return out;
}

// ---------------------------------------------------------------- projection
// adaptive average pooling (PyTorch semantics), 2D, channel-major [C, H, W]
static void adaptive_avg_pool_2d(const std::vector<float> & src, int C, int Hin, int Win,
                                 std::vector<float> & dst, int Hout, int Wout) {
    for (int c = 0; c < C; c++) {
        for (int o = 0; o < Hout; o++) {
            int rs = (int)std::floor((double)o * Hin / Hout);
            int re = (int)std::ceil((double)(o + 1) * Hin / Hout);
            if (re <= rs) re = rs + 1;
            for (int p = 0; p < Wout; p++) {
                int cs = (int)std::floor((double)p * Win / Wout);
                int ce = (int)std::ceil((double)(p + 1) * Win / Wout);
                if (ce <= cs) ce = cs + 1;
                float acc = 0;
                for (int r = rs; r < re; r++)
                    for (int w = cs; w < ce; w++)
                        acc += src[((size_t)c * Hin + r) * Win + w];
                dst[((size_t)c * Hout + o) * Wout + p] = acc / ((re - rs) * (ce - cs));
            }
        }
    }
}

std::vector<float> VisionModel::project(const float * enc, int n_crops, int th, int tw,
                                        ggml_backend_t backend) {
    VisionWeights & W = g_w;
    const int G = grid, D = enc_dim, M = overlap;
    const size_t T = (size_t)G * G;

    if (n_crops < 1) return {};
    // 1) stitch local crops (index 1..n-1) in patch space, margin = overlap patches
    const int FH = (G - 2 * M) * th + 2 * M;   // reconstructed feature map size
    const int FW = (G - 2 * M) * tw + 2 * M;
    std::vector<float> recon((size_t)D * FH * FW, 0.0f);   // [D, FH, FW] channel-major
    for (int i = 1; i < n_crops; i++) {
        int tile_y = (i - 1) / tw, tile_x = (i - 1) % tw;
        int xs = (tile_x == 0) ? 0 : M, xe = (tile_x == tw - 1) ? G : G - M;
        int ys = (tile_y == 0) ? 0 : M, ye = (tile_y == th - 1) ? G : G - M;
        int ox = tile_x * (G - 2 * M), oy = tile_y * (G - 2 * M);
        const float * crop = enc + (size_t)i * T * D;      // [G, G, D] token-major
        for (int py = ys; py < ye; py++)
            for (int px = xs; px < xe; px++)
                for (int c = 0; c < D; c++)
                    recon[((size_t)c * FH + (oy + py)) * FW + (ox + px)] =
                        crop[((size_t)py * G + px) * D + c];
    }

    // 2) adaptive avg pool [D, FH, FW] -> [D, G, G]
    std::vector<float> pooled((size_t)D * T);
    adaptive_avg_pool_2d(recon, D, FH, FW, pooled, G, G);
    if (getenv("MD_TRACE")) {
}

    // 3) concat [global | pooled] -> [2D, T] activation (ggml column-major: c fastest)
    const float * global_toks = enc;               // [T, D] token-major (crop 0)
    std::vector<float> in((size_t)2 * D * T);
    for (size_t t = 0; t < T; t++) {
        for (int c = 0; c < D; c++) in[(size_t)c * 1 + (size_t)t * 2 * D] = global_toks[t * D + c];
        for (int c = 0; c < D; c++) in[(size_t)(D + c) + (size_t)t * 2 * D] = pooled[(size_t)c * T + t];
    }

    struct ggml_init_params ip = { 128ull * 1024 * 1024, NULL, true };
    struct ggml_context * gctx = ggml_init(ip);
    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 1024, false);
    struct ggml_tensor * x = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 2 * D, (int64_t)T);
    ggml_set_name(x, "proj_in");
    x = ggml_mul_mat(gctx, W.proj_fc1_w, x);
    x = ggml_add(gctx, x, W.proj_fc1_b);
    x = ggml_gelu(gctx, x);
    x = ggml_mul_mat(gctx, W.proj_fc2_w, x);
    x = ggml_add(gctx, x, W.proj_fc2_b);
    ggml_set_name(x, "proj_out_t");
    ggml_build_forward_expand(gf, x);

    struct ggml_gallocr * gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(gallocr, gf)) { ggml_gallocr_free(gallocr); ggml_free(gctx); return {}; }
    struct ggml_tensor * pin = ggml_graph_get_tensor(gf, "proj_in");
    ggml_backend_tensor_set(pin, in.data(), 0, sizeof(float) * in.size());
    bool ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    std::vector<float> out(T * proj_out);
    if (ok) {
        std::vector<float> tmp(T * proj_out);
        ggml_backend_tensor_get(x, tmp.data(), 0, sizeof(float) * tmp.size());
        for (size_t t = 0; t < T; t++)
            for (int c = 0; c < proj_out; c++)
                out[t * proj_out + c] = tmp[(size_t)c + t * proj_out];   // ggml [P,T] is c-fastest
    }
    ggml_gallocr_free(gallocr);
    ggml_free(gctx);
    return ok ? out : std::vector<float>{};
}

// ---------------------------------------------------------------- convenience API
std::vector<float> VisionModel::encode_image(const uint8_t * rgb, int w, int h,
                                             ggml_backend_t backend) {
    // simplified preprocessing: single global crop, bilinear resize (NOT the
    // official Lanczos multi-crop pipeline; use encode_crops/project with
    // Python-prepared crops for bit-parity work)
    const int CS = crop, CH = in_ch;
    std::vector<float> crop((size_t)CH * CS * CS);
    for (int c = 0; c < CH; c++)
        for (int y = 0; y < CS; y++) {
            double sy = (y + 0.5) * h / CS - 0.5;
            int y0 = (int)std::floor(sy); int y1 = std::min(y0 + 1, h - 1); y0 = std::max(y0, 0);
            double fy = sy - std::floor(sy);
            for (int x = 0; x < CS; x++) {
                double sx = (x + 0.5) * w / CS - 0.5;
                int x0 = (int)std::floor(sx); int x1 = std::min(x0 + 1, w - 1); x0 = std::max(x0, 0);
                double fx = sx - std::floor(sx);
                double v = (1 - fy) * ((1 - fx) * rgb[(size_t)y0 * w * CH + (size_t)x0 * CH + c]
                                        + fx * rgb[(size_t)y0 * w * CH + (size_t)x1 * CH + c])
                         + fy * ((1 - fx) * rgb[(size_t)y1 * w * CH + (size_t)x0 * CH + c]
                                        + fx * rgb[(size_t)y1 * w * CH + (size_t)x1 * CH + c]);
                crop[(size_t)c * CS * CS + (size_t)y * CS + x] = (float)((v / 255.0 - 0.5) / 0.5);
            }
        }
    auto enc = encode_crops(crop.data(), 1, n_layers, backend);
    // single crop: local branch degenerates to zeros in the official pipeline for
    // tiling (1,1) with no extra crops; use the global crop as pooled features
    // (reconstruct of one tile with full margins == the crop itself)
    std::vector<float> emb = project(enc.data(), 1, 1, 1, backend);
    return emb;
}

// ---------------------------------------------------------------- test main
#ifdef VISION_TEST_MAIN
#include <string>

int main(int argc, char ** argv) {
    const char * gguf_path = nullptr, * crops_path = nullptr, * out_path = nullptr, * dump_dir = nullptr;
    int th = 0, tw = 0, max_layer = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) gguf_path = argv[++i];
        else if (!strcmp(argv[i], "--crops") && i + 1 < argc) crops_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-dir") && i + 1 < argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--tiling") && i + 2 < argc) { th = atoi(argv[++i]); tw = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--max-layer") && i + 1 < argc) max_layer = atoi(argv[++i]);
    }
    if (!gguf_path || !crops_path || !out_path) {
        fprintf(stderr, "usage: moondream-vision-test -m vision.gguf --crops crops.bin --out out.bin"
                        " [--tiling H W] [--max-layer N] [--dump-dir dir]\n"
                        "crops.bin: int32 n_crops, then n*3*378*378 float32 (CHW, normalized)\n");
        return 1;
    }

    FILE * f = fopen(crops_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", crops_path); return 1; }
    int n_crops = 0;
    if (fread(&n_crops, 4, 1, f) != 1) { fclose(f); return 1; }
    std::vector<float> crops((size_t)n_crops * 3 * 378 * 378);
    if (fread(crops.data(), 4, crops.size(), f) != crops.size()) { fprintf(stderr, "crops file truncated\n"); fclose(f); return 1; }
    fclose(f);
    fprintf(stderr, "crops: %d\n", n_crops);

    ggml_backend_t backend = vision_pick_backend();
    VisionModel vm;
    if (!vm.load(gguf_path, backend)) return 1;

    if (max_layer < 0) max_layer = vm.n_layers;

    std::vector<float> result;
    std::vector<std::vector<float>> layer_dump;
    std::vector<float> enc = vm.encode_crops(crops.data(), n_crops, max_layer, backend,
                                             dump_dir ? &layer_dump : nullptr);

    if (th > 0 && tw > 0 && max_layer == vm.n_layers) {
        result = vm.project(enc.data(), n_crops, th, tw, backend);
        fprintf(stderr, "final embedding: [729, %d]\n", vm.proj_out);
    } else {
        result = enc;
        fprintf(stderr, "encoder output: [%d, 729, %d] (max_layer=%d)\n", n_crops, vm.enc_dim, max_layer);
    }

    FILE * o = fopen(out_path, "wb");
    fwrite(result.data(), 4, result.size(), o);
    fclose(o);
    fprintf(stderr, "wrote %s (%zu floats)\n", out_path, result.size());

    if (dump_dir && !layer_dump.empty()) {
        const int T = vm.grid * vm.grid;
        const size_t per_crop = layer_dump.size() / n_crops;   // flat list is crop-major
        for (int c = 0; c < n_crops; c++) {
            for (size_t j = 0; j < per_crop; j++) {
                bool is_post = (max_layer == vm.n_layers && j == per_crop - 1);
                char path[512];
                if (is_post) snprintf(path, sizeof(path), "%s\\cpp_post_ln_c%d.bin", dump_dir, c);
                else snprintf(path, sizeof(path), "%s\\cpp_h_L%zu_c%d.bin", dump_dir, j, c);
                FILE * d = fopen(path, "wb");
                if (!d) { fprintf(stderr, "cannot write %s\n", path); continue; }
                fwrite(layer_dump[(size_t)c * per_crop + j].data(), 4, (size_t)T * vm.enc_dim, d);
                fclose(d);
            }
        }
        fprintf(stderr, "dumped %zu layer outputs x %d crops to %s\n", per_crop, n_crops, dump_dir);
    }
    return 0;
}
#endif
