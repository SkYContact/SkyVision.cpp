// moss_vl_model.cpp - MOSS-VL-Realtime backend (Qwen3 text decoder with
// interleaved MRoPE + SigLIP vision tower + Flamingo-style cross attention).
//
// Conventions (shared with the other backends):
//   * GGUF linear weights are stored as ne = [in, out]; ggml_mul_mat(W, x).
//   * the vision tower runs as ONE graph per media: patch embed, 27 blocks
//     with fused 2D vision rope (wqkv rows permuted + zero-padded 72->80 at
//     conversion, rope_layout=2, sections {18,18}, base' = theta^(80/72)),
//     deepstack taps at blocks 8/16/24, merger (4x LayerNorm -> concat ->
//     2x2 merge view -> fc1 -> gelu_erf -> fc2).
//   * every frame contributes (merged_tokens + 1) cross-attention KV rows:
//     a learnable separator column follows each frame's tokens. The 12 cross
//     layers project those rows once into per-layer cross KV caches (K gets
//     k_norm + MRoPE with the frame's vision positions, V is raw).
//   * text layers 2,6,...,46 have NO self-attention - they cross-attend from
//     the text stream (query, text MRoPE) into the cross KV behind tanh
//     scalars, and text rows with no visible frames skip both the cross
//     attention and the MLP entirely (the reference zeroes their outputs).
//   * text MRoPE: ggml_rope_multi(GGML_ROPE_TYPE_IMROPE, sections 24/20/20),
//     4th position axis = T axis (covers the two tail frequency pairs).
#include "vqa.hpp"
#include "vit_block.hpp"

#include <functional>
#include <array>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#   include "ggml-cuda.h"
#endif
#include "tokenizer.hpp"
#include "backend.hpp"
#include "ggml_dump.hpp"
#include "gguf_util.hpp"
#include "attention.hpp"
#include "sampler.hpp"
#include "vqa_util.hpp"
#include "resample.hpp"   // vqa::img::smart_resize / resize_bicubic_antialias
#include "moss_vl_preprocess.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <thread>
#include <vector>

namespace moss_vl {

// ---------------------------------------------------------------- structures
struct VisLayer {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *qkv_w = nullptr, *qkv_b = nullptr;
    ggml_tensor *wo_w = nullptr, *wo_b = nullptr;
    ggml_tensor *fc0_w = nullptr, *fc0_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
};

struct TxtLayer {
    bool xattn = false;
    ggml_tensor *norm = nullptr;                        // input_layernorm
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    ggml_tensor *qn = nullptr, *kn = nullptr;
    ggml_tensor *cq = nullptr, *ck = nullptr, *cv = nullptr, *co = nullptr;
    ggml_tensor *cqn = nullptr, *ckn = nullptr;
    ggml_tensor *ga = nullptr, *gm = nullptr;           // cross tanh gates [1]
    ggml_tensor *ffn_norm = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
};

// cross-attention visibility segment: text rows [start, start+len) see the
// first v_end rows of every cross KV cache
struct XSeg { int start = 0, len = 0, v_end = 0; };

struct FrameMeta {
    int eh = 0, ew = 0;        // merged grid per frame
    int tpf = 0;               // merged tokens per frame (eh*ew)
    int cs_start = 0;          // first cross-KV row of this frame (incl. separator)
    float t0 = 0;              // MRoPE T value of this frame's grid
};

struct PromptPlan {
    std::vector<int> ids;
    std::vector<int> mpos;          // [5*T]: t, h, w, e(=t), kv-slot
    std::vector<float> colmask;     // [T]: 1.0 for rows with >=1 visible frame
    std::vector<XSeg> segments;
    std::vector<FrameMeta> frames;  // video: one per frame; image: one entry
    int n_vtot = 0;                 // total cross-KV rows (tokens + separators)
    int rope_delta = 0;             // decode: position = slot + rope_delta
};

struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *mpos_in = nullptr, *mask_in = nullptr, *colmask_in = nullptr;
    ggml_tensor *logits = nullptr;
};

// shared state for the vision attention hook
struct VitEnv {
    ggml_context * gctx = nullptr;
    VisConfig * vc = nullptr;
    ggml_tensor * t_rope = nullptr;   // [2, vhd_f/2, N] F32: cos plane then sin plane
    int N = 0, n_ctx = 0;
    int vhd = 72, vhd_f = 80;
    int layer = 0;
    float rope_base = 10000.0f;
};

// qkv rows: q (16x80) | k (16x80) | v (16x80), converter-permuted into the
// rope_layout=2 head layout ([x-re 18 | y-re 18 | pad4 | x-im 18 | y-im 18 |
// pad4]); strided head views feed rope/FA with no runtime padding copies.
// The rotation is a plain in-graph pair rotation (pairs (j, j+40) with
// host-built angle tables) - the fused ggml VISION rope miscomputes pairs
// >= sections[0] on the CPU backend at this size (convert/rope_probe2.cpp).
// Full (non-causal) attention over the whole media: every frame's patches
// attend each other (the reference's cu_seqlens semantics).
static ggml_tensor * mvit_attention(void * ud_, ggml_tensor * qkv) {
    VitEnv & e = *(VitEnv *)ud_;
    ggml_context * c = e.gctx;
    const int vhd = e.vhd, vhd_f = e.vhd_f;
    const size_t off = (size_t)e.vc->n_head * vhd_f * ggml_element_size(qkv);
    auto head_view = [&](int block) {
        return ggml_view_3d(c, qkv, vhd_f, e.vc->n_head, e.N,
                            (size_t)vhd_f * ggml_element_size(qkv), qkv->nb[1],
                            (size_t)block * off);
    };
    ggml_tensor * q = head_view(0);
    ggml_tensor * k = head_view(1);
    ggml_tensor * v = head_view(2);
    if (e.layer == 0) ggml_set_name(qkv, "v_blk0_qkv");

    // in-graph rotation: pair (j, j+40) by angle tables broadcast per head;
    // t_rope layout: cos plane [40, N] then sin plane, 40 floats per token
    const size_t es = ggml_element_size(e.t_rope);
    ggml_tensor * cos_t = ggml_view_3d(c, e.t_rope, vhd_f / 2, 1, e.N, es, (size_t)(vhd_f / 2) * es, 0);
    ggml_tensor * sin_t = ggml_view_3d(c, e.t_rope, vhd_f / 2, 1, e.N, es, (size_t)(vhd_f / 2) * es,
                                       (size_t)e.N * (vhd_f / 2) * es);
    auto rope_pair = [&](ggml_tensor * x) {
        ggml_tensor * re = ggml_view_3d(c, x, vhd_f / 2, e.vc->n_head, e.N,
                                        x->nb[1], x->nb[2], 0);
        ggml_tensor * im = ggml_view_3d(c, x, vhd_f / 2, e.vc->n_head, e.N,
                                        x->nb[1], x->nb[2],
                                        (size_t)(vhd_f / 2) * ggml_element_size(x));
        ggml_tensor * ore = ggml_sub(c, ggml_mul(c, re, cos_t), ggml_mul(c, im, sin_t));
        ggml_tensor * oim = ggml_add(c, ggml_mul(c, re, sin_t), ggml_mul(c, im, cos_t));
        return ggml_concat(c, ore, oim, 0);
    };
    q = rope_pair(q);
    k = rope_pair(k);

    auto fa_in = [&](ggml_tensor * t) {
        return ggml_view_3d(c, t, vhd_f, e.N, e.vc->n_head, t->nb[2], t->nb[1], 0);
    };
    ggml_tensor * fa = ggml_flash_attn_ext(c, fa_in(q), fa_in(k), fa_in(v), nullptr,
                                           1.0f / sqrtf((float)vhd), 0.0f, 0.0f);
    if (e.layer == 0) {
        ggml_set_name(q, "v_blk0_q_rope");
        ggml_set_name(fa, "v_blk0_attn");
    }
    // de-pad 80 -> 72 rows per head, then (d, h) rows per token for wo
    ggml_tensor * attn_src = ggml_view_3d(c, fa, vhd, e.vc->n_head, e.N,
                                          (size_t)vhd_f * ggml_element_size(fa),
                                          (size_t)vhd_f * e.vc->n_head * ggml_element_size(fa), 0);
    ggml_tensor * attnout = ggml_reshape_2d(c, ggml_cont(c, attn_src), e.vc->n_embd, e.N);
    if (e.layer == 0) ggml_set_name(attnout, "v_blk0_attnout");
    return attnout;
}

class MossVL : public vqa::Model {
public:
    ~MossVL() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "moss-vl-realtime"; }
    bool ask_session(const vqa::MediaInput & media, const std::string & prompt,
                     const vqa::GenParams & gp, std::string & out_text) override;
    void session_reset() override { sess_active = false; }
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp, std::string & out_text) override;

    // tunables, set before load()
    int n_ctx = 8192;
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool alloc_kv();
    bool prepare_media(const vqa::MediaInput & media, int token_budget,
                       std::vector<float> & patches, std::vector<float> & pos,
                       std::vector<int> & vpos, int * gh, int * gw);
    bool build_prompt(const vqa::MediaInput & media, const std::string & user_text,
                      const std::vector<int> & grids, PromptPlan & plan,
                      const std::string & history = "");
    bool plan_from_text(const std::string & text, const std::vector<int> & grids,
                        PromptPlan & plan, int pos_base = 0, int slot_base = 0, int v_base = 0);
    bool encode_vision(const std::vector<float> & patches, const std::vector<float> & pos,
                       const std::vector<int> & vpos, int N, std::vector<float> & merged,
                       const PromptPlan * plan = nullptr);   // plan != null: also project cross-KV in-graph
    bool project_cross_kv(const std::vector<float> & merged, const PromptPlan & plan,
                          int row_base = 0);
    // appends the fused cross-KV projection chain onto a graph; cs is the
    // [out_hidden, V] cross-state input (host tensor or in-graph concat)
    bool append_cross_kv(ggml_context * gctx, ggml_cgraph * gf, ggml_tensor * cs,
                         int V, const PromptPlan & plan,
                         ggml_tensor ** t_mpos_out, std::vector<int> & mpos_out,
                         const char * tag = "");   // see the naming note in the impl
    bool build_text_graph(TextGraph & g, const PromptPlan & plan, int T, int S, int blk);
    bool run_text(TextGraph & g, const PromptPlan & plan, const std::vector<int> & toks,
                  int base_slot, std::vector<float> & logits_out);
    int sample(const std::vector<float> & logits, const vqa::GenParams & gp);
    void dump(const ggml_cgraph * gf, const char * name);
    void dump_host(const char * name, const void * data, size_t bytes,
                   std::initializer_list<int64_t> ne);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    int n_layer = 48, n_embd = 4096, n_head = 32, n_kv_head = 8, head_dim = 128;
    int n_ff = 12288, n_vocab = 151936;
    float rms_eps = 1e-6f, rope_theta = 5e6f;
    int mrope_t = 24, mrope_h = 20, mrope_w = 20;
    std::vector<int> xlayers;                 // cross-attention layer indices
    VisConfig vc;
    std::vector<int> ds_idx;                  // deepstack block indexes

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_backend_buffer_t wbuf_plain = nullptr;
    vqa::AlignedBuf host_f32_storage;
    vqa::MMap mm;

    std::vector<float> pos_emb_host;          // learned (48*48, 1152) table

    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr, *out_w = nullptr;
    ggml_tensor *v_patch_w = nullptr, *v_patch_b = nullptr;
    ggml_tensor *sep_w = nullptr;
    // fused cross-KV projection: [n_embd, nx*2*kv_dim] rows = [k(1024)|v(1024)] per cross layer
    ggml_context *xkv_fused_ctx = nullptr;
    ggml_backend_buffer_t xkv_fused_buf = nullptr;
    ggml_tensor *xkv_fused = nullptr;
    ggml_tensor *mm_fc1_w = nullptr, *mm_fc1_b = nullptr, *mm_fc2_w = nullptr, *mm_fc2_b = nullptr;
    ggml_tensor *mm_ln_w[4] = {}, *mm_ln_b[4] = {};
    std::vector<TxtLayer> tl;
    std::vector<VisLayer> vl;

    vqa::KvCache kv;                          // self-attn KV (self layers only)
    struct CrossKv {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        std::vector<ggml_tensor *> k, v;      // per cross layer [n_kv_head*head_dim, xctx]
        void clear() { if (buf) ggml_backend_buffer_clear(buf, 0); }
        void release() {
            if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
            if (ctx) { ggml_free(ctx); ctx = nullptr; }
            k.clear(); v.clear();
        }
    } xkv;
    int xctx = 16384;

    ggml_gallocr_t allocr = nullptr;
    int n_past = 0;

    // static single-token decode graph (same-object across steps: CUDA-graph eligible)
    ggml_context * dec_ctx = nullptr;
    ggml_cgraph * dec_gf = nullptr;
    int dec_blk = 0, dec_vtot = -1;
    ggml_tensor * dec_tok = nullptr, * dec_mpos = nullptr, * dec_mask = nullptr, * dec_colmask = nullptr;

    int id_image_pad = 151655, id_vision_start = 151652, id_vision_end = 151653;
    int id_im_start = 151644, id_im_end = 151645, id_eos = 151645, id_endoftext = 151643;
    int id_silence = -1;
    // live realtime session state (ask_session): text KV keeps growing via
    // n_past/rope_delta, cross-KV rows via sess_vrows
    bool sess_active = false;
    int sess_vrows = 0;
    PromptPlan sess_plan;
    int id_time_start = 151669, id_time_end = 151670;

    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

// ---------------------------------------------------------------- helpers


MossVL::~MossVL() {
    if (allocr) ggml_gallocr_free(allocr);
    if (dec_ctx) ggml_free(dec_ctx);
    kv.free();
    xkv.release();
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wbuf_plain) ggml_backend_buffer_free(wbuf_plain);
    if (wctx) ggml_free(wctx);
    if (backend) ggml_backend_free(backend);
}

bool MossVL::init_backend() {
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

// ---------------------------------------------------------------- load
bool MossVL::load(const std::string & gguf_path, const std::string & tokenizer_path,
                  const vqa::LoadOptions & opt) {
    n_ctx = opt.n_ctx > 0 ? opt.n_ctx : n_ctx;
    n_threads = opt.n_threads;
    use_gpu = opt.use_gpu;
    dump_dir = opt.dump_dir;
    if (const char * d = getenv("MOSS_DUMP_DIR")) dump_dir = d;
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::Qwen2Regex)) return false;
    if (!alloc_kv()) return false;

    auto pick = [&](const char * s, int def) {
        int v = tok_.id_of(s);
        return v >= 0 ? v : def;
    };
    id_image_pad = pick("<|image_pad|>", id_image_pad);
    id_vision_start = pick("<|vision_start|>", id_vision_start);
    id_vision_end = pick("<|vision_end|>", id_vision_end);
    id_im_start = pick("<|im_start|>", id_im_start);
    id_im_end = pick("<|im_end|>", id_im_end);
    id_silence = tok_.id_of("<|silence|>");
    id_time_start = pick("<|time_start|>", id_time_start);
    id_time_end = pick("<|time_end|>", id_time_end);
    id_eos = id_im_end;

    caps_ = {};
    caps_.image = true;
    caps_.video = true;
    caps_.timestamps_prompt = true;
    fprintf(stderr, "moss_vl: ready (%d layers, %d cross-attn, %d vision layers, ctx %d)\n",
            n_layer, (int)xlayers.size(), vc.n_layer, n_ctx);
    return true;
}

bool MossVL::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f GB\n", mm.size / 1e9);

    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer   = vqa::gguf::u32(gu, "moss_vl.block_count", n_layer);
    n_embd    = vqa::gguf::u32(gu, "moss_vl.embedding_length", n_embd);
    n_head    = vqa::gguf::u32(gu, "moss_vl.attention.head_count", n_head);
    n_kv_head = vqa::gguf::u32(gu, "moss_vl.attention.head_count_kv", n_kv_head);
    head_dim  = vqa::gguf::u32(gu, "moss_vl.attention.key_length", head_dim);
    n_ff      = vqa::gguf::u32(gu, "moss_vl.feed_forward_length", n_ff);
    n_vocab   = vqa::gguf::u32(gu, "moss_vl.vocab_size", n_vocab);
    rms_eps   = vqa::gguf::f32(gu, "moss_vl.attention.layer_norm_rms_epsilon", rms_eps);
    rope_theta = vqa::gguf::f32(gu, "moss_vl.rope.freq_base", rope_theta);
    mrope_t   = vqa::gguf::u32(gu, "moss_vl.rope.mrope_section_t", mrope_t);
    mrope_h   = vqa::gguf::u32(gu, "moss_vl.rope.mrope_section_h", mrope_h);
    mrope_w   = vqa::gguf::u32(gu, "moss_vl.rope.mrope_section_w", mrope_w);
    {
        std::vector<int> def;
        for (int i = 2; i < n_layer; i += 4) def.push_back(i);
        xlayers = vqa::gguf::arr_u32(gu, "moss_vl.cross_layer_indexes", def);
        std::sort(xlayers.begin(), xlayers.end());
    }

    vc.n_layer = vqa::gguf::u32(gu, "moss_vl.vision.block_count", vc.n_layer);
    vc.n_embd  = vqa::gguf::u32(gu, "moss_vl.vision.embedding_length", vc.n_embd);
    vc.n_head  = vqa::gguf::u32(gu, "moss_vl.vision.attention.head_count", vc.n_head);
    vc.n_ff    = vqa::gguf::u32(gu, "moss_vl.vision.feed_forward_length", vc.n_ff);
    vc.patch   = vqa::gguf::u32(gu, "moss_vl.vision.patch_size", vc.patch);
    vc.merge   = vqa::gguf::u32(gu, "moss_vl.vision.merge_size", vc.merge);
    {
        int npe = vqa::gguf::u32(gu, "moss_vl.vision.num_position_embeddings", 2304);
        vc.pos_side = npe > 0 ? (int)std::lround(std::sqrt((double)npe)) : 48;
    }
    vc.out_hidden = vqa::gguf::u32(gu, "moss_vl.vision.out_embedding_length", vc.out_hidden);
    vc.layernorm_eps = vqa::gguf::f32(gu, "moss_vl.vision.layer_norm_epsilon", vc.layernorm_eps);
    vc.rope_theta = vqa::gguf::f32(gu, "moss_vl.vision.rope_freq_base", vc.rope_theta);
    ds_idx = vqa::gguf::arr_u32(gu, "moss_vl.vision.deepstack_indexes", {8, 16, 24});
    if (vqa::gguf::u32(gu, "moss_vl.vision.rope_layout", 0) != 2) {
        fprintf(stderr, "this build expects a rope_layout=2 GGUF (wqkv rows permuted + zero-padded for the fused vision rope); re-run convert/convert_moss_vl.py\n");
        return false;
    }

    const size_t data_off = gguf_get_data_offset(gu);
    const int64_t nt = gguf_get_n_tensors(gu);
    for (int64_t i = 0; i < nt; i++) {
        const char * name = gguf_get_tensor_name(gu, i);
        ggml_tensor * t = ggml_get_tensor(wctx, name);
        if (!t) continue;
        t->data = (char *)mm.ptr + data_off + gguf_get_tensor_offset(gu, i);
    }
    gguf_free(gu);

    // conv3d weight (temporal kernel = 1) as a plain matmul: [p*p*3, n_embd].
    // Stored as ggml ne [p, p, 3, n_embd] (HF (out, c, ph, pw) row-major);
    // flattening to ne [p*p*3, n_embd] keeps the exact same bytes.
    {
        ggml_tensor * t = ggml_get_tensor(wctx, "v.patch_embd.weight");
        if (t && t->ne[3] == vc.n_embd) {
            const int64_t in = t->ne[0] * t->ne[1] * t->ne[2];
            t->ne[0] = in; t->ne[1] = vc.n_embd; t->ne[2] = 1; t->ne[3] = 1;
            const size_t row = (size_t)in * ggml_element_size(t);
            t->nb[0] = ggml_element_size(t);
            t->nb[1] = row; t->nb[2] = row; t->nb[3] = row;
        }
    }

    // 1D tensors (norms, biases, gates, separator) -> host F32
    vqa::AlignedBuf host_f32;
    {
        size_t total = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            if (t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1) continue;
            if (t->type == GGML_TYPE_F32) continue;
            total += (size_t)ggml_nelements(t) * sizeof(float);
        }
        host_f32.resize(total);
        size_t off = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            if (t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1) continue;
            if (t->type == GGML_TYPE_F32) continue;
            const int64_t n = ggml_nelements(t);
            float * dst = (float *)(host_f32.data() + off);
            vqa::gguf::tensor_to_f32(t, dst);
            t->data = dst;
            t->type = GGML_TYPE_F32;
            t->nb[0] = sizeof(float);
            t->nb[1] = (size_t)n * sizeof(float);
            t->nb[2] = t->nb[1];
            t->nb[3] = t->nb[2];
            off += (size_t)n * sizeof(float);
        }
        fprintf(stderr, "f32-converted 1D tensors: %.1f MB\n", total / 1e6);
    }

    {
        ggml_tensor * pe = ggml_get_tensor(wctx, "v.pos_emb.weight");
        if (!pe) { fprintf(stderr, "missing v.pos_emb.weight\n"); return false; }
        pos_emb_host.resize(ggml_nelements(pe));
        vqa::gguf::tensor_to_f32(pe, pos_emb_host.data());
    }

    {
        std::vector<std::pair<ggml_tensor *, const void *>> uploads;
        size_t total = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            uploads.push_back({t, t->data});
            total += ggml_nbytes(t);
        }
        if (ggml_backend_is_cpu(backend)) {
            // zero-copy: weights keep pointing into the mmap, the f32 1D
            // tensors into their host block (halves the resident footprint)
            wbuf = ggml_backend_cpu_buffer_from_ptr(mm.ptr, mm.size);
            const char * f32_begin = host_f32.empty() ? nullptr : host_f32.data();
            const char * f32_end = f32_begin ? f32_begin + host_f32.size() : nullptr;
            if (f32_begin) wbuf_plain = ggml_backend_cpu_buffer_from_ptr(host_f32.data(), host_f32.size());
            for (auto & u : uploads) {
                ggml_tensor * t = u.first;
                const char * q = (const char *)t->data;
                t->buffer = (f32_begin && q >= f32_begin && q < f32_end) ? wbuf_plain : wbuf;
            }
            fprintf(stderr, "weights: %.2f GB zero-copy from mmap\n", total / 1e9);
        } else {
            for (auto & u : uploads) u.first->data = nullptr;
            wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
            if (!wbuf) { fprintf(stderr, "weight allocation failed (need %.2f GB)\n", total / 1e9); return false; }
            for (auto & u : uploads) ggml_backend_tensor_set(u.first, u.second, 0, ggml_nbytes(u.first));
            fprintf(stderr, "uploaded %.2f GB of weights\n", total / 1e9);
        }
    }
    if (ggml_backend_is_cpu(backend)) {
        host_f32_storage = std::move(host_f32);
    } else {
        host_f32.clear();
        host_f32.shrink_to_fit();
        mm.release();
    }

    auto gt = [&](const char * name) {
        ggml_tensor * t = ggml_get_tensor(wctx, name);
        if (!t) fprintf(stderr, "missing tensor: %s\n", name);
        return t;
    };
    tok_embd = gt("token_embd.weight");
    out_norm = gt("output_norm.weight");
    out_w = gt("output.weight");
    v_patch_w = gt("v.patch_embd.weight");
    v_patch_b = gt("v.patch_embd.bias");
    sep_w = gt("mm.separator.weight");
    mm_fc1_w = gt("mm.fc1.weight");
    mm_fc1_b = gt("mm.fc1.bias");
    mm_fc2_w = gt("mm.fc2.weight");
    mm_fc2_b = gt("mm.fc2.bias");
    for (int i = 0; i < 4; i++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "mm.ln%d.weight", i);
        mm_ln_w[i] = gt(buf);
        snprintf(buf, sizeof(buf), "mm.ln%d.bias", i);
        mm_ln_b[i] = gt(buf);
    }
    if (!tok_embd || !out_norm || !out_w || !v_patch_w || !sep_w ||
        !mm_fc1_w || !mm_fc2_w || !mm_ln_w[0]) return false;

    tl.resize(n_layer);
    for (int i = 0; i < n_layer; i++) {
        TxtLayer & L = tl[i];
        L.xattn = std::binary_search(xlayers.begin(), xlayers.end(), i);
        char buf[128];
        auto g = [&](const char * suffix) {
            snprintf(buf, sizeof(buf), "blk.%d.%s", i, suffix);
            return gt(buf);
        };
        L.norm = g("attn_norm.weight");
        L.ffn_norm = g("ffn_norm.weight");
        L.gate = g("ffn_gate.weight");
        L.up = g("ffn_up.weight");
        L.down = g("ffn_down.weight");
        if (L.xattn) {
            L.cq = g("cross_attn_q.weight");
            L.ck = g("cross_attn_k.weight");
            L.cv = g("cross_attn_v.weight");
            L.co = g("cross_attn_output.weight");
            L.cqn = g("cross_attn_q_norm.weight");
            L.ckn = g("cross_attn_k_norm.weight");
            L.ga = g("cross_gate_attn.weight");
            L.gm = g("cross_gate_mlp.weight");
        } else {
            L.q = g("attn_q.weight");
            L.k = g("attn_k.weight");
            L.v = g("attn_v.weight");
            L.o = g("attn_output.weight");
            L.qn = g("attn_q_norm.weight");
            L.kn = g("attn_k_norm.weight");
        }
    }
    {   // fuse the per-layer cross k/v projections into one weight so the
        // cross-KV graph runs a single big GEMM (and quantizes cs once)
        const int nx = (int)xlayers.size();
        if (nx > 0) {
            struct ggml_init_params ip = { 1ull << 20, nullptr, /*no_alloc*/ true };
            xkv_fused_ctx = ggml_init(ip);
            xkv_fused = ggml_new_tensor_2d(xkv_fused_ctx, GGML_TYPE_Q6_K, n_embd,
                                           (int64_t)nx * 2 * n_kv_head * head_dim);
            xkv_fused_buf = ggml_backend_alloc_ctx_tensors(xkv_fused_ctx, backend);
            if (!xkv_fused_buf) { fprintf(stderr, "fused cross-kv alloc failed\n"); return false; }
            const size_t row_bytes = (size_t)(n_embd / 256) * ggml_type_size(GGML_TYPE_Q6_K); // 16 blocks/row
            std::vector<uint8_t> stage(row_bytes * n_kv_head * head_dim);
            int li = 0;
            for (int il : xlayers) {
                const TxtLayer & L = tl[il];
                const size_t base = (size_t)li * 2 * n_kv_head * head_dim * row_bytes;
                for (int half = 0; half < 2; half++) {
                    ggml_tensor * src = half ? L.cv : L.ck;
                    const size_t bytes = row_bytes * n_kv_head * head_dim;
                    ggml_backend_tensor_get(src, stage.data(), 0, bytes);
                    ggml_backend_tensor_set(xkv_fused, stage.data(),
                                            base + (size_t)half * n_kv_head * head_dim * row_bytes, bytes);
                }
                li++;
            }
            fprintf(stderr, "fused cross-kv projection: %d layers x [k|v] (%.1f MB)\n",
                    nx, ggml_nbytes(xkv_fused) / 1e6);
        }
    }

    vl.resize(vc.n_layer);
    for (int i = 0; i < vc.n_layer; i++) {
        char buf[128];
        auto g = [&](const char * suffix) {
            snprintf(buf, sizeof(buf), "v.blk.%d.%s", i, suffix);
            return gt(buf);
        };
        VisLayer & L = vl[i];
        L.ln1_w = g("ln1.weight");
        L.ln1_b = g("ln1.bias");
        L.ln2_w = g("ln2.weight");
        L.ln2_b = g("ln2.bias");
        L.qkv_w = g("wqkv.weight");
        L.qkv_b = g("wqkv.bias");
        L.wo_w = g("wo.weight");
        L.wo_b = g("wo.bias");
        L.fc0_w = g("ffn_up.weight");
        L.fc0_b = g("ffn_up.bias");
        L.fc1_w = g("ffn_down.weight");
        L.fc1_b = g("ffn_down.bias");
    }
    return true;
}

bool MossVL::alloc_kv() {
    const int n_self = n_layer - (int)xlayers.size();
    if (!kv.alloc(backend, n_self, n_kv_head, head_dim, n_ctx)) return false;
    xctx = std::min(n_ctx, 32768);
    if (const char * e = getenv("MOSS_XCTX")) xctx = std::max(1024, atoi(e));
    xkv.release();
    const size_t bytes = (size_t)xlayers.size() * 2 * xctx * n_kv_head * head_dim * sizeof(ggml_fp16_t);
    struct ggml_init_params ip = { ((size_t)xlayers.size() * 2 + 8) * ggml_tensor_overhead(), NULL, true };
    xkv.ctx = ggml_init(ip);
    if (!xkv.ctx) return false;
    xkv.k.resize(xlayers.size());
    xkv.v.resize(xlayers.size());
    for (size_t i = 0; i < xlayers.size(); i++) {
        xkv.k[i] = ggml_new_tensor_2d(xkv.ctx, GGML_TYPE_F16, n_kv_head * head_dim, xctx);
        xkv.v[i] = ggml_new_tensor_2d(xkv.ctx, GGML_TYPE_F16, n_kv_head * head_dim, xctx);
    }
    xkv.buf = ggml_backend_alloc_ctx_tensors(xkv.ctx, backend);
    if (!xkv.buf) { fprintf(stderr, "cross KV allocation failed (%.2f GB)\n", bytes / 1e9); return false; }
    ggml_backend_buffer_clear(xkv.buf, 0);
    fprintf(stderr, "cross kv cache: %.2f GB (%d rows)\n", bytes / 1e9, xctx);
    return true;
}

// ---------------------------------------------------------------- media prep
bool MossVL::prepare_media(const vqa::MediaInput & media, int token_budget,
                           std::vector<float> & patches, std::vector<float> & pos,
                           std::vector<int> & vpos, int * gh, int * gw) {
    if (media.empty()) return true;
    const int T = (int)media.frames.size();
    const int src_h = media.height, src_w = media.width;
    const int factor = vc.patch * vc.merge;

    long long max_pixels = 12845056LL;   // images: 16384*28*28 (vc3 default)
    if (media.kind == vqa::MediaKind::Video) {
        double vmp = 0.75;
        if (const char * e = getenv("MOSS_VIDEO_MP")) vmp = atof(e);
        if (vmp > 0) max_pixels = (long long)(vmp * 1e6);
    }
    if (token_budget > 0) {
        const long long per_frame = std::max((long long)token_budget / T - 1, 4LL);   // -1 separator
        max_pixels = std::min(max_pixels, per_frame * factor * factor);
    }

    int H = 0, W = 0;
    if (!vqa::img::smart_resize(src_h, src_w, factor, 4096, (int)std::min(max_pixels, (long long)INT_MAX), &H, &W)) {
        fprintf(stderr, "image too small (%dx%d, factor %d)\n", src_w, src_h, factor);
        return false;
    }
    // VRAM guard: the tower runs at patch level; cap ~48k patch rows
    {
        const long long n_patches = (long long)(H / vc.patch) * (W / vc.patch) * T;
        const long long n_max = 48000;
        if (n_patches > n_max) {
            const double scale = (double)n_max / (double)n_patches;
            max_pixels = (long long)(max_pixels * scale);
            int H2, W2;
            if (!vqa::img::smart_resize(src_h, src_w, factor, 4096, (int)std::min(max_pixels, (long long)INT_MAX), &H2, &W2)) {
                fprintf(stderr, "image too small after guard\n");
                return false;
            }
            fprintf(stderr, "vision VRAM guard: %lld patches > %lld, shrinking %dx%d -> %dx%d each\n",
                    n_patches, n_max, W, H, W2, H2);
            H = H2; W = W2;
        }
    }
    if (media.kind == vqa::MediaKind::Video)
        fprintf(stderr, "video %d frame%s %dx%d -> %dx%d each\n", T, T == 1 ? "" : "s", src_w, src_h, W, H);
    else
        fprintf(stderr, "image %dx%d -> %dx%d\n", src_w, src_h, W, H);

    *gh = H / vc.patch;
    *gw = W / vc.patch;
    std::vector<float> rgb;
    std::vector<int> coords;
    for (int ti = 0; ti < T; ti++) {
        vqa::img::resize_bicubic_antialias(media.frames[ti].rgb.data(), src_w, src_h, W, H, rgb);
        patchify_frame(rgb.data(), H, W, vc, pos_emb_host.data(), patches, pos, coords, gh, gw);
    }
    // fused vision rope positions, ggml mrope PLANAR layout: [row(N), col(N),
    // w(N), e(N)] - slot0 = row, slot1 = col, frame-local (sections {18,18}
    // map slot0 -> row freqs, slot1 -> col freqs for rope_layout=2)
    const int hw = (*gh) * (*gw);
    vpos.assign((size_t)T * hw * 4, 0);
    int * rp = vpos.data(), * cp = vpos.data() + (size_t)T * hw;
    for (int t = 0; t < T; t++)
        for (int s = 0; s < hw; s++) {
            rp[(size_t)t * hw + s] = coords[(size_t)(t * hw + s) * 2 + 0];
            cp[(size_t)t * hw + s] = coords[(size_t)(t * hw + s) * 2 + 1];
        }
    return true;
}

// ---------------------------------------------------------------- prompt
bool MossVL::build_prompt(const vqa::MediaInput & media, const std::string & user_text,
                          const std::vector<int> & grids, PromptPlan & plan,
                          const std::string & history) {
    const int F = (int)grids.size() / 2;
    const bool video = media.kind == vqa::MediaKind::Video;
    std::string text = history;
    text += "<|im_start|>user\n";
    if (F > 0) {
        if (video) {
            text += "<|vision_start|>";
            for (int f = 0; f < F; f++) {
                char ts[64];
                const double t = f < (int)media.timestamps.size() ? media.timestamps[f]
                                 : (media.fps > 0 ? f / media.fps : (double)f);
                snprintf(ts, sizeof(ts), "<|time_start|>%.1f seconds<|time_end|>", t);
                text += ts;
                text += "<|image_pad|>";
            }
            text += "<|vision_end|>";
        } else {
            text += "<|image_pad|>";
        }
    }
    text += user_text;
    text += "<|im_end|>\n<|im_start|>assistant\n";
    return plan_from_text(text, grids, plan);
}

// turn `text` (already formatted, image_pad placeholders for F frames given by
// `grids`) into a PromptPlan. The *_base offsets support incremental session
// turns appended onto live KV caches:
//   pos_base  - mrope position of the segment's first regular token
//   slot_base - self-attn KV slot of the segment's first token
//   v_base    - cross-KV rows already used by earlier turns
bool MossVL::plan_from_text(const std::string & text, const std::vector<int> & grids,
                            PromptPlan & plan, int pos_base, int slot_base, int v_base) {
    const int F = (int)grids.size() / 2;
    plan.ids = tok_.encode(text);
    std::vector<int> pads;
    for (int i = 0; i < (int)plan.ids.size(); i++)
        if (plan.ids[i] == id_image_pad) pads.push_back(i);
    if ((int)pads.size() != F) {
        fprintf(stderr, "placeholder mismatch: %d image_pad tokens vs %d frame(s)\n",
                (int)pads.size(), F);
        return false;
    }

    const int T = (int)plan.ids.size();
    plan.frames.resize(F);
    plan.colmask.assign(T, 0.0f);
    plan.n_vtot = v_base;
    int cs = 0;   // cs_start stays LOCAL to this segment (cross-KV mpos index)
    for (int k = 0; k < F; k++) {
        FrameMeta & fm = plan.frames[k];
        fm.eh = grids[k * 2 + 0];
        fm.ew = grids[k * 2 + 1];
        fm.tpf = fm.eh * fm.ew;
        fm.cs_start = cs;
        cs += fm.tpf + 1;
        plan.n_vtot += fm.tpf + 1;
    }

    // base position = count of regular tokens before i; image_pad rows sit one
    // before the post-shift value (t_val + max_hw)
    std::vector<int> base(T);
    {
        int c = 0;
        for (int i = 0; i < T; i++) { base[i] = c; if (plan.ids[i] != id_image_pad) c++; }
    }
    {
        int shift = 0;
        for (int k = 0; k < F; k++) {
            plan.frames[k].t0 = (float)(pos_base + base[pads[k]] + shift);
            shift += std::max(plan.frames[k].eh, plan.frames[k].ew) + 1;
        }
    }
    plan.mpos.assign((size_t)T * 5, 0);
    int last_pos = 0;
    for (int i = 0; i < T; i++) {
        int shift = 0;
        for (int k = 0; k < F; k++)
            if (pads[k] <= i) shift += std::max(plan.frames[k].eh, plan.frames[k].ew) + 1;
        int pos = pos_base + base[i] + shift;
        if (plan.ids[i] == id_image_pad) pos -= 1;
        // ggml mrope PLANAR layout: [t(T), h(T), w(T), e(T), slot(T)]
        for (int a = 0; a < 4; a++) plan.mpos[(size_t)a * T + i] = pos;
        plan.mpos[(size_t)4 * T + i] = slot_base + i;   // self-attn KV slot
        if (pos > last_pos) last_pos = pos;
    }
    // delta maps a follow-up decode slot to its mrope position: the token right
    // after this segment sits at slot slot_base+T with position last_pos+1
    plan.rope_delta = last_pos + 1 - (slot_base + T);
    // rows from the first image_pad onward see at least one frame (earlier
    // turns' frames count when appending)
    if (v_base > 0 && F == 0) {
        std::fill(plan.colmask.begin(), plan.colmask.end(), 1.0f);
    } else {
        for (int i = 0, k = 0; i < T; i++) {
            while (k < F && pads[k] == i) k++;
            plan.colmask[i] = (k > 0 || v_base > 0) ? 1.0f : 0.0f;
        }
    }
    // prefill segments: rows before pad_0 see what earlier turns had; rows
    // [pad_k, pad_{k+1}) see v_base + k+1 new frames
    plan.segments.clear();
    if (F > 0) {
        if (pads[0] > 0) plan.segments.push_back({0, pads[0], v_base});
        for (int k = 0; k < F; k++) {
            const int end = k + 1 < F ? pads[k + 1] : T;
            int v_end = v_base;
            for (int j = 0; j <= k; j++) v_end += plan.frames[j].tpf + 1;
            plan.segments.push_back({pads[k], end - pads[k], v_end});
        }
    } else if (v_base > 0) {
        plan.segments.push_back({0, T, v_base});
    }
    return true;
}

// ---------------------------------------------------------------- vision
// tower + deepstack + merger in ONE graph; output lands in host floats (also
// used as the per-media cache entry)
bool MossVL::encode_vision(const std::vector<float> & patches, const std::vector<float> & pos,
                           const std::vector<int> & vpos, int N, std::vector<float> & merged,
                           const PromptPlan * plan) {
    if (N == 0) return true;
    VitEnv env;
    env.vc = &vc;
    env.N = N;
    env.n_ctx = n_ctx;
    env.vhd = vc.n_embd / vc.n_head;                // 72
    env.vhd_f = env.vhd + 8;                        // zero-padded rows (rope_layout=2)
    // converter-permuted rows make n_dims 40; rescale the base so every real
    // pair keeps its exact frequency (theta^(-p/18))
    env.rope_base = powf(vc.rope_theta, (float)env.vhd_f / (float)env.vhd);

    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) return false;
    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);

    ggml_tensor * t_patches = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 3 * vc.patch * vc.patch, N);
    ggml_tensor * t_pos = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, vc.n_embd, N);
    // per-pair rotation angles: pair j<18 -> row freq, 18..35 -> col freq, 36..39 -> 0
    ggml_tensor * t_rope = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, (int64_t)N * 2 * (env.vhd_f / 2));
    env.gctx = gctx;
    env.t_rope = t_rope;

    ggml_tensor * x = ggml_mul_mat(gctx, v_patch_w, t_patches);
    ggml_set_name(x, "v_mm_out");
    x = ggml_add(gctx, x, v_patch_b);
    x = ggml_add(gctx, x, t_pos);
    ggml_set_name(x, "v_patch_out");

    std::vector<ggml_tensor *> ds_features;
    const bool ds_dump = !dump_dir.empty();  // v_ds dumps below
    for (int il = 0; il < vc.n_layer; il++) {
        VisLayer & L = vl[il];
        vqa::vit::BlockWeights bw{L.ln1_w, L.ln1_b, L.qkv_w, L.qkv_b, L.wo_w, L.wo_b,
                                  L.ln2_w, L.ln2_b, L.fc0_w, L.fc0_b, L.fc1_w, L.fc1_b};
        vqa::vit::BlockHooks bh;
        bh.ud = &env;
        bh.attention = mvit_attention;
        env.layer = il;
        if (il == 0 && !dump_dir.empty()) {
            bh.probe_layer0 = [](void * ud_, const char * stage, ggml_tensor * t) {
                char nm[48];
                snprintf(nm, sizeof(nm), "v_blk0_%s", stage);
                ggml_set_name(t, nm);
                return false;
            };
        }
        x = vqa::vit::block(gctx, x, bw, vc.layernorm_eps, bh, il);
        if (!x) { ggml_free(gctx); return false; }
        if (il == 0) ggml_set_name(x, "v_blk0_out");
        if (std::binary_search(ds_idx.begin(), ds_idx.end(), il) && !dump_dir.empty()) {
            char dsn[24];
            snprintf(dsn, sizeof(dsn), "v_ds%d", il);
            ggml_set_name(x, dsn);
        }
        if (std::binary_search(ds_idx.begin(), ds_idx.end(), il))
            ds_features.push_back(x);
    }
    ggml_set_name(x, "v_final");

    // merger: 4x independent LayerNorm -> concat -> 2x2 merge VIEW -> MLP.
    // token stream is (frame, mh, mw, ir, ic); 4 consecutive tokens form one
    // merge block, so the reference's view(-1, 18432) is a plain reshape.
    ggml_tensor * cat = nullptr;
    const int n_feat = 1 + (int)ds_features.size();
    for (int i = 0; i < n_feat; i++) {
        ggml_tensor * src = (i == 0) ? x : ds_features[i - 1];
        ggml_tensor * f = ggml_norm(gctx, src, 1e-6f);
        f = ggml_mul(gctx, f, mm_ln_w[i]);
        f = ggml_add(gctx, f, mm_ln_b[i]);
        cat = cat ? ggml_concat(gctx, cat, f, 0) : f;
    }
    ggml_tensor * m2 = ggml_reshape_2d(gctx, cat, (int64_t)vc.n_embd * 4 * n_feat, N / 4);
    ggml_tensor * h = ggml_mul_mat(gctx, mm_fc1_w, m2);
    h = ggml_add(gctx, h, mm_fc1_b);
    h = ggml_gelu_erf(gctx, h);                    // merger uses nn.GELU() == exact erf
    ggml_tensor * out = ggml_mul_mat(gctx, mm_fc2_w, h);
    out = ggml_add(gctx, out, mm_fc2_b);
    ggml_set_name(out, "v_merged");

    // v_merged is the graph's terminal output: its scratch buffer is never
    // recycled mid-compute (that recycling is exactly what corrupted the old
    // chained-in projection's readback), so the post-compute readback below
    // is trustworthy. The cross-KV projection always runs as the separate
    // project_cross_kv() graph - identical structure for cache hits and
    // misses, so both paths produce identical K/V by construction.
    ggml_build_forward_expand(gf, out);
    if (!dump_dir.empty()) {
        vqa::dump::dump_registry_clear();
        static const char * const dump_names[] = {
            "v_mm_out", "v_patch_out", "v_blk0_qkv", "v_blk0_q_rope", "v_blk0_attn",
            "v_blk0_attnout", "v_blk0_attn_out", "v_blk0_out", "v_ds8", "v_ds16", "v_ds24",
            "v_final", "v_merged", nullptr,
        };
        vqa::dump::mark_sinks(gctx, gf, dump_names);
        dump_host("patches", patches.data(), sizeof(float) * patches.size(),
                  {3 * vc.patch * vc.patch, N});
        dump_host("vpos", vpos.data(), sizeof(int) * vpos.size(), {N * 4});
        dump_host("pos_in", pos.data(), sizeof(float) * pos.size(), {vc.n_embd, N});
    }
    vqa::dump::finalize_sinks(gctx, gf, backend);
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "vision graph allocation failed\n");
        ggml_free(gctx);
        return false;
    }
    ggml_backend_tensor_set(t_patches, patches.data(), 0, sizeof(float) * patches.size());
    ggml_backend_tensor_set(t_pos, pos.data(), 0, sizeof(float) * pos.size());
    {
        const int pairs = env.vhd_f / 2, half = env.vhd / 4;   // 40 pairs, 18 per section
        std::vector<float> rope((size_t)N * 2 * pairs, 0.0f);
        float * cp = rope.data();
        float * sp = rope.data() + (size_t)N * pairs;
        const int * vr = vpos.data(), * vc_ = vpos.data() + N;
        for (int n = 0; n < N; n++) {
            const int row = vr[n], col = vc_[n];
            for (int j = 0; j < pairs; j++) {
                const float th = j < half ? row * powf(vc.rope_theta, -(float)j / half)
                                          : (j < 2 * half ? col * powf(vc.rope_theta, -(float)(j - half) / half)
                                                          : 0.0f);
                cp[(size_t)n * pairs + j] = cosf(th);
                sp[(size_t)n * pairs + j] = sinf(th);
            }
        }
        ggml_backend_tensor_set(t_rope, rope.data(), 0, sizeof(float) * rope.size());
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vision compute failed\n");
        ggml_free(gctx);
        return false;
    }
    dump(gf, "v_patch_out");
    dump(gf, "v_mm_out");
    dump(gf, "v_blk0_qkv");
    dump(gf, "v_blk0_q_rope");
    dump(gf, "v_blk0_attnout");
    dump(gf, "v_blk0_attn_out");
    dump(gf, "v_blk0_out");
    if (ds_dump) {
        dump(gf, "v_ds8");
        dump(gf, "v_ds16");
        dump(gf, "v_ds24");
    }
    dump(gf, "v_final");
    dump(gf, "v_merged");
    {   // read back the merger output: the per-media cache entry AND the
        // standalone re-projection input
        ggml_tensor * got = ggml_graph_get_tensor(gf, "v_merged");
        merged.resize((size_t)got->ne[0] * got->ne[1]);
        ggml_backend_tensor_get(got, merged.data(), 0, sizeof(float) * merged.size());
    }
    if (plan && plan->n_vtot > 0 && !dump_dir.empty()) {   // parity input for the numpy recompute
        const int V = plan->n_vtot;
        std::vector<float> sep(vc.out_hidden);
        ggml_backend_tensor_get(sep_w, sep.data(), 0, sizeof(float) * sep.size());
        std::vector<float> cs_host;
        cs_host.reserve((size_t)V * vc.out_hidden);
        size_t off = 0;
        for (const FrameMeta & fm : plan->frames) {
            cs_host.insert(cs_host.end(), merged.begin() + off * vc.out_hidden,
                           merged.begin() + (off + fm.tpf) * vc.out_hidden);
            cs_host.insert(cs_host.end(), sep.begin(), sep.end());
            off += fm.tpf;
        }
        dump_host("xkv_cs", cs_host.data(), sizeof(float) * cs_host.size(), {vc.out_hidden, V});
    }
    fprintf(stderr, "vision: %d patches -> %d merged tokens\n", N, (int)(merged.size() / vc.out_hidden));
    ggml_free(gctx);
    return true;
}

// appends the cross-KV projection chain onto an existing graph: input cs is
// [out_hidden, V] (host tensor or an in-graph frame/separator concat); output
// t_mpos_out/mpos_out receive the position tensor + host data to set post-alloc
bool MossVL::append_cross_kv(ggml_context * gctx, ggml_cgraph * gf, ggml_tensor * cs,
                             int V, const PromptPlan & plan,
                             ggml_tensor ** t_mpos_out, std::vector<int> & mpos_out,
                             const char * tag) {
    if (V > xctx) {
        fprintf(stderr, "vision tokens %d exceed cross-KV capacity %d (raise MOSS_XCTX)\n", V, xctx);
        return false;
    }
    ggml_tensor * t_rows = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, V);
    // vision MRoPE positions for the cross KV: per frame T = t0 (const),
    // H = t0 + y, W = t0 + x over the merged grid; separator (s,s,s)
    ggml_tensor * t_mpos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, (int64_t)V * 4);
    // ggml mrope PLANAR layout: [t(V), h(V), w(V), e(V)]
    std::vector<int> mpos((size_t)V * 4, 0);
    for (const FrameMeta & fm : plan.frames) {
        const int t0 = (int)fm.t0, mh = std::max(fm.eh, fm.ew);
        // merged tokens are ROW-major over the merged grid: y = m/ew, x = m%ew
        // (verified against official compute_vision_position_ids H/W planes)
        for (int m = 0; m < fm.tpf; m++) {
            const size_t i = (size_t)fm.cs_start + m;
            mpos[i] = t0; mpos[V + i] = t0 + m / fm.ew;
            mpos[2 * (size_t)V + i] = t0 + m % fm.ew; mpos[3 * (size_t)V + i] = t0;
        }
        const size_t i = (size_t)fm.cs_start + fm.tpf;
        mpos[i] = mpos[V + i] = mpos[2 * (size_t)V + i] = mpos[3 * (size_t)V + i] = t0 + mh;
    }
    int sections[GGML_MROPE_SECTIONS] = { mrope_t, mrope_h, mrope_w, 0 };
    // one fused GEMM for all cross layers' k/v projections: [nx*2048, V]
    //
    // The names below are LOAD-BEARING: ggml-cuda keys CUDA-graph reuse on the
    // FIRST NODE POINTER (which the ggml ctx pool happily recycles between
    // graphs) and then a memcmp of node properties. Unnamed/tagged alike, this
    // chain embedded in the vision graph and the standalone re-projection graph
    // (cache-hit path) are property-identical -> the OTHER graph's captured
    // instance gets wrongly replayed -> garbage KV on CUDA. The per-caller tag
    // keeps the properties distinct so each graph gets its own capture.
    // Do not remove the names or share one tag between callers.
    ggml_set_name(cs, tag[0] ? "xkv_cs_g_host" : "xkv_cs_g");
    ggml_tensor * kva = ggml_mul_mat(gctx, xkv_fused, cs);
    ggml_set_name(kva, tag[0] ? "xkv_kva_host" : "xkv_kva");
    const int64_t kv_dim = (int64_t)n_kv_head * head_dim;
    int li = 0;
    for (int il : xlayers) {
        TxtLayer & L = tl[il];
        // rows [li*2*kv_dim, +kv_dim) = k, next kv_dim rows = v (per-token
        // columns are contiguous, so slice rows then make contiguous)
        ggml_tensor * k = ggml_cont(gctx, ggml_view_2d(gctx, kva, kv_dim, V,
                                                       kva->nb[1], (size_t)(li * 2) * kv_dim * sizeof(float)));
        ggml_tensor * k3 = ggml_reshape_3d(gctx, k, head_dim, n_kv_head, V);
        if (li == 0) ggml_set_name(k3, tag[0] ? "xkv_kcont_host" : "xkv_kcont");
        k3 = ggml_rms_norm(gctx, k3, rms_eps);
        k3 = ggml_mul(gctx, k3, L.ckn);
        if (li == 0) ggml_set_name(k3, tag[0] ? "xkv_knorm_host" : "xkv_knorm");
        k3 = ggml_rope_multi(gctx, k3, t_mpos, nullptr, head_dim, sections,
                             GGML_ROPE_TYPE_IMROPE, n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * k2 = ggml_reshape_2d(gctx, k3, n_kv_head * head_dim, V);
        if (li == 0) ggml_set_name(k2, tag[0] ? "xkv_k0_host" : "xkv_k0");
        ggml_build_forward_expand(gf, ggml_set_rows(gctx, xkv.k[li], k2, t_rows));
        ggml_tensor * v2 = ggml_cont(gctx, ggml_view_2d(gctx, kva, kv_dim, V,
                                                        kva->nb[1], (size_t)(li * 2 + 1) * kv_dim * sizeof(float)));
        ggml_build_forward_expand(gf, ggml_set_rows(gctx, xkv.v[li], v2, t_rows));
        li++;
    }
    if (!dump_dir.empty()) {
        static const char * const dump_names[] = {
            "xkv_k0", "xkv_k0_host", "xkv_kcont", "xkv_kcont_host",
            "xkv_knorm", "xkv_knorm_host", "xkv_kva", "xkv_kva_host",
            "xkv_cs_g", "xkv_cs_g_host", nullptr,
        };
        vqa::dump::mark_sinks(gctx, gf, dump_names);
    }
    // hand the input tensors + host position data back for post-alloc filling
    ggml_set_name(t_rows, "xkv_t_rows");
    *t_mpos_out = t_mpos;
    mpos_out = mpos;
    return true;
}

bool MossVL::project_cross_kv(const std::vector<float> & merged, const PromptPlan & plan,
                              int row_base) {
    const double t_ck = ggml_time_ms();
    if (plan.n_vtot == 0) return true;
    const int M = (int)(merged.size() / vc.out_hidden);
    const int V = plan.n_vtot - row_base;   // rows of THIS batch (plan.n_vtot is cumulative)
    if (V <= 0) return true;
    if (row_base + V > xctx) {
        fprintf(stderr, "vision tokens %d exceed cross-KV capacity %d (raise MOSS_XCTX)\n", V, xctx);
        return false;
    }
    struct ggml_init_params ip = { 32ull * 1024 * 1024, NULL, true };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) return false;
    struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8192, false);

    // device-side cs, structurally IDENTICAL to the in-vision-graph projection:
    // a leaf F32 input feeding the fused GEMM computes wrong values on CUDA,
    // while the concat-chain form is verified correct, so cache hits reuse
    // exactly that structure (leaf = the merged features, concat per frame)
    ggml_tensor * t_m = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, vc.out_hidden, M);
    ggml_tensor * cs = nullptr;
    {
        ggml_tensor * sep_col = ggml_reshape_2d(gctx, sep_w, vc.out_hidden, 1);
        size_t off = 0;
        for (const FrameMeta & fm : plan.frames) {
            ggml_tensor * col = ggml_view_2d(gctx, t_m, vc.out_hidden, fm.tpf, t_m->nb[1],
                                             off * t_m->nb[1]);
            cs = cs ? ggml_concat(gctx, cs, col, 1) : col;
            cs = ggml_concat(gctx, cs, sep_col, 1);
            off += fm.tpf;
        }
    }
    ggml_tensor * t_mpos = nullptr;
    std::vector<int> mpos;
    if (!append_cross_kv(gctx, gf, cs, V, plan, &t_mpos, mpos, "host")) {
        ggml_free(gctx);
        return false;
    }
    std::vector<int> rows(V);
    for (int i = 0; i < V; i++) rows[i] = row_base + i;
    ggml_tensor * t_rows = ggml_graph_get_tensor(gf, "xkv_t_rows");   // the one set_rows reads

    vqa::dump::finalize_sinks(gctx, gf, backend);
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "cross kv graph allocation failed\n");
        ggml_free(gctx);
        return false;
    }
    ggml_backend_tensor_set(t_m, merged.data(), 0, sizeof(float) * merged.size());
    ggml_backend_tensor_set(t_mpos, mpos.data(), 0, sizeof(int) * mpos.size());
    ggml_backend_tensor_set(t_rows, rows.data(), 0, sizeof(int) * V);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cross kv compute failed\n");
        ggml_free(gctx);
        return false;
    }
    fprintf(stderr, "cross kv: %d rows projected into %d layers\n", V, (int)xlayers.size());
    if (getenv("MOSS_TIMING"))
        fprintf(stderr, "[timing] cross-kv compute %.0f ms\n", ggml_time_ms() - t_ck);
    dump(gf, "xkv_cs_g_host");
    dump(gf, "xkv_kva_host");
    dump(gf, "xkv_kcont_host");
    dump(gf, "xkv_knorm_host");
    dump(gf, "xkv_k0_host");
    if (!dump_dir.empty()) {   // parity inputs for the xkv numpy recompute
        std::vector<int> vmpos((size_t)V * 4, 0);
        for (const FrameMeta & fm : plan.frames) {
            const int t0 = (int)fm.t0, mh = std::max(fm.eh, fm.ew);
            for (int m = 0; m < fm.tpf; m++) {
                const size_t i = (size_t)fm.cs_start + m;
                vmpos[i] = t0; vmpos[V + i] = t0 + m / fm.ew;
                vmpos[2 * V + i] = t0 + m % fm.ew; vmpos[3 * V + i] = t0;
            }
            const size_t i = (size_t)fm.cs_start + fm.tpf;
            vmpos[i] = vmpos[V + i] = vmpos[2 * V + i] = vmpos[3 * V + i] = t0 + mh;
        }
        dump_host("xkv_mpos", vmpos.data(), sizeof(int) * vmpos.size(), {4 * V});
    }
    ggml_free(gctx);
    return true;
}

// ---------------------------------------------------------------- text
bool MossVL::build_text_graph(TextGraph & g, const PromptPlan & plan, int T, int S, int blk) {
    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 32768, false);

    const int D = n_embd, HD = head_dim, NH = n_head, NKV = n_kv_head;
    const bool has_vision = plan.n_vtot > 0;
    std::vector<XSeg> segs_own;
    if (has_vision) {
        if (T == 1) segs_own.push_back({0, 1, plan.n_vtot});   // decode: full visibility
        else segs_own = plan.segments;                          // prefill: prefix per frame
    }
    const std::vector<XSeg> & segs = segs_own;

    g.tok_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    g.mpos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, (int64_t)T * 5);
    ggml_set_name(g.mpos_in, "mpos_in");
    g.mask_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, S, T);
    ggml_set_name(g.mask_in, "mask_in");
    g.colmask_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 1, T);
    ggml_set_name(g.colmask_in, "colmask_in");
    // rope reads the first 4 axes; KV writes use the 5th (sequential slots)
    ggml_tensor * pos_rope = ggml_view_1d(g.ctx, g.mpos_in, (int64_t)T * 4, 0);
    ggml_tensor * pos_slot = ggml_view_1d(g.ctx, g.mpos_in, T, (size_t)T * 4 * sizeof(int));

    ggml_tensor * x = ggml_get_rows(g.ctx, tok_embd, g.tok_in);      // [D, T]
    ggml_set_name(x, "emb");

    int sections[GGML_MROPE_SECTIONS] = { mrope_t, mrope_h, mrope_w, 0 };
    const float scale = 1.0f / sqrtf((float)HD);

    int kv_idx = 0, xli = 0;
    for (int il = 0; il < n_layer; il++) {
        const char * pfx = (il == 0) ? "t_l0" : (il == 2) ? "t_x2" : nullptr;
        TxtLayer & L = tl[il];
        ggml_tensor * l = ggml_rms_norm(g.ctx, x, rms_eps);
        l = ggml_mul(g.ctx, l, L.norm);
        if (pfx) ggml_set_name(l, (std::string(pfx) + "_norm").c_str());

        if (!L.xattn) {
            // ---- self-attention decoder layer
            ggml_tensor * q = ggml_mul_mat(g.ctx, L.q, l);
            ggml_tensor * k = ggml_mul_mat(g.ctx, L.k, l);
            ggml_tensor * v = ggml_mul_mat(g.ctx, L.v, l);
            if (pfx) { ggml_set_name(q, (std::string(pfx) + "_q").c_str()); }

            ggml_tensor * q3 = ggml_reshape_3d(g.ctx, q, HD, NH, T);
            q3 = ggml_rms_norm(g.ctx, q3, rms_eps);
            q3 = ggml_mul(g.ctx, q3, L.qn);
            ggml_tensor * k3 = ggml_reshape_3d(g.ctx, k, HD, NKV, T);
            k3 = ggml_rms_norm(g.ctx, k3, rms_eps);
            k3 = ggml_mul(g.ctx, k3, L.kn);
            if (pfx) ggml_set_name(q3, (std::string(pfx) + "_q_norm").c_str());

            q3 = ggml_rope_multi(g.ctx, q3, pos_rope, nullptr, HD, sections,
                                 GGML_ROPE_TYPE_IMROPE, n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k3 = ggml_rope_multi(g.ctx, k3, pos_rope, nullptr, HD, sections,
                                 GGML_ROPE_TYPE_IMROPE, n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            if (pfx) ggml_set_name(q3, (std::string(pfx) + "_q_rope").c_str());

            ggml_tensor * k2 = ggml_reshape_2d(g.ctx, k3, NKV * HD, T);
            ggml_tensor * v2 = ggml_reshape_2d(g.ctx, v, NKV * HD, T);
            ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.k[kv_idx], k2, pos_slot));
            ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.v[kv_idx], v2, pos_slot));

            const int Swin = blk;
            ggml_tensor * qf = ggml_view_3d(g.ctx, q3, HD, T, NH, q3->nb[2], q3->nb[1], 0);
            ggml_tensor * kf = ggml_view_3d(g.ctx, kv.k[kv_idx], HD, Swin, NKV,
                                            kv.k[kv_idx]->nb[1], HD * sizeof(ggml_fp16_t), 0);
            ggml_tensor * vf = ggml_view_3d(g.ctx, kv.v[kv_idx], HD, Swin, NKV,
                                            kv.v[kv_idx]->nb[1], HD * sizeof(ggml_fp16_t), 0);
            ggml_tensor * fa = ggml_flash_attn_ext(g.ctx, qf, kf, vf, g.mask_in, scale, 0.0f, 0.0f);
            ggml_tensor * attn = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, fa), NH * HD, T);
            if (pfx) ggml_set_name(attn, (std::string(pfx) + "_attn").c_str());
            attn = ggml_mul_mat(g.ctx, L.o, attn);
            x = ggml_add(g.ctx, x, attn);
            if (pfx) ggml_set_name(x, (std::string(pfx) + "_resid").c_str());
            kv_idx++;
        } else if (has_vision) {
            // ---- cross-attention decoder layer (no self attention).
            // queries roam the text stream with TEXT MRoPE; keys/values live
            // in the cross KV (vision MRoPE applied once at projection time)
            ggml_tensor * q = ggml_mul_mat(g.ctx, L.cq, l);
            ggml_tensor * q3 = ggml_reshape_3d(g.ctx, q, HD, NH, T);
            q3 = ggml_rms_norm(g.ctx, q3, rms_eps);
            q3 = ggml_mul(g.ctx, q3, L.cqn);
            q3 = ggml_rope_multi(g.ctx, q3, pos_rope, nullptr, HD, sections,
                                 GGML_ROPE_TYPE_IMROPE, n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            ggml_tensor * attn_all = nullptr;
            for (const XSeg & sg : segs) {
                ggml_tensor * chunk = nullptr;
                if (sg.v_end == 0) {
                    // rows with no visible frames: the reference zeroes their
                    // attention output; skip the FA and emit exact zeros
                    chunk = ggml_scale(g.ctx, ggml_view_2d(g.ctx, x, D, sg.len, x->nb[1],
                                                           (size_t)sg.start * x->nb[1]), 0.0f);
                } else {
                    ggml_tensor * qf = ggml_view_3d(g.ctx, q3, HD, sg.len, NH,
                                                    q3->nb[2], q3->nb[1],
                                                    (size_t)sg.start * q3->nb[2]);
                    ggml_tensor * kf = ggml_view_3d(g.ctx, xkv.k[xli], HD, sg.v_end, NKV,
                                                    xkv.k[xli]->nb[1], HD * sizeof(ggml_fp16_t), 0);
                    ggml_tensor * vf = ggml_view_3d(g.ctx, xkv.v[xli], HD, sg.v_end, NKV,
                                                    xkv.v[xli]->nb[1], HD * sizeof(ggml_fp16_t), 0);
                    ggml_tensor * fa = ggml_flash_attn_ext(g.ctx, qf, kf, vf, nullptr, scale, 0.0f, 0.0f);
                    chunk = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, fa), NH * HD, sg.len);
                }
                attn_all = attn_all ? ggml_concat(g.ctx, attn_all, chunk, 1) : chunk;
            }
            attn_all = ggml_mul_mat(g.ctx, L.co, attn_all);
            x = ggml_add(g.ctx, x, ggml_mul(g.ctx, attn_all, ggml_tanh(g.ctx, L.ga)));
            if (pfx) ggml_set_name(x, (std::string(pfx) + "_xattn").c_str());
        }
        // if (!has_vision && L.xattn): passthrough - the reference zeroes both
        // the attention and the MLP output for rows without vision

        // MLP (all layers)
        ggml_tensor * m = ggml_rms_norm(g.ctx, x, rms_eps);
        m = ggml_mul(g.ctx, m, L.ffn_norm);
        ggml_tensor * gate = ggml_mul_mat(g.ctx, L.gate, m);
        ggml_tensor * up = ggml_mul_mat(g.ctx, L.up, m);
        m = ggml_swiglu_split(g.ctx, gate, up);
        m = ggml_mul_mat(g.ctx, L.down, m);
        if (L.xattn && has_vision) {
            m = ggml_mul(g.ctx, m, g.colmask_in);   // zero no-vision rows (reference mask)
            m = ggml_mul(g.ctx, m, ggml_tanh(g.ctx, L.gm));  // reference: residual + tanh(mlp_gate) * mlp
        }
        if (!(L.xattn && !has_vision)) {
            x = ggml_add(g.ctx, x, m);
            if (pfx) ggml_set_name(x, (std::string(pfx) + "_out").c_str());
        }
        if (L.xattn) xli++;
    }

    x = ggml_rms_norm(g.ctx, x, rms_eps);
    x = ggml_mul(g.ctx, x, out_norm);
    ggml_set_name(x, "t_final_norm");
    ggml_tensor * last = ggml_view_2d(g.ctx, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
    g.logits = ggml_mul_mat(g.ctx, out_w, last);
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);
    if (!dump_dir.empty()) {
        vqa::dump::dump_registry_clear();
        static const char * const dump_names[] = {
            "emb", "t_l0_norm", "t_l0_q", "t_l0_q_norm", "t_l0_q_rope", "t_l0_attn",
            "t_l0_out", "t_x2_norm", "t_x2_q_rope", "t_x2_xattn", "t_x2_out",
            "t_final_norm", nullptr,
        };
        vqa::dump::mark_sinks(g.ctx, g.gf, dump_names);
    }
    return true;
}

bool MossVL::run_text(TextGraph & g, const PromptPlan & plan, const std::vector<int> & toks,
                      int base_slot, std::vector<float> & logits_out) {
    const int T = (int)toks.size();
    const int S = n_past + T;
    if (S > n_ctx) { fprintf(stderr, "context overflow (%d > %d)\n", S, n_ctx); return false; }

    // single-token decode: static graph with a block-rounded KV window
    ggml_tensor * logits_t = nullptr;
    const bool use_static = (T == 1) && !getenv("MOSS_NO_STATIC");
    if (use_static) {
        const int blk = std::min(vqa::block_for(S), n_ctx);
        if (!dec_gf || dec_blk != blk || dec_vtot != plan.n_vtot) {
            if (dec_ctx) ggml_free(dec_ctx);
            struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, true };
            dec_ctx = ggml_init(ip);
            TextGraph dg;
            dg.ctx = dec_ctx;
            if (!build_text_graph(dg, plan, 1, blk, blk)) {
                ggml_free(dec_ctx); dec_ctx = nullptr; dec_gf = nullptr;
                return false;
            }
            dec_gf = dg.gf;
            vqa::dump::finalize_sinks(dec_ctx, dec_gf, backend);
            if (!ggml_gallocr_alloc_graph(allocr, dec_gf)) {
                fprintf(stderr, "static decode graph alloc failed\n");
                ggml_free(dec_ctx); dec_ctx = nullptr; dec_gf = nullptr;
                return false;
            }
            dec_blk = blk;
            dec_vtot = plan.n_vtot;
            dec_tok = ggml_graph_get_tensor(dec_gf, "tok_in");
            dec_mpos = ggml_graph_get_tensor(dec_gf, "mpos_in");
            dec_mask = ggml_graph_get_tensor(dec_gf, "mask_in");
            dec_colmask = ggml_graph_get_tensor(dec_gf, "colmask_in");
        }
        ggml_backend_tensor_set(dec_tok, toks.data(), 0, sizeof(int));
        int mp[5];
        const int pos = base_slot + plan.rope_delta;
        for (int a = 0; a < 4; a++) mp[a] = pos;
        mp[4] = base_slot;
        ggml_backend_tensor_set(dec_mpos, mp, 0, sizeof(mp));
        vqa::set_causal_mask(dec_mask, n_past, 1);
        if (dec_colmask && plan.n_vtot > 0) {   // colmask only enters the graph with vision
            const float one = 1.0f;
            ggml_backend_tensor_set(dec_colmask, &one, 0, sizeof(float));
        }
        if (ggml_backend_graph_compute(backend, dec_gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "text compute failed\n");
            return false;
        }
        logits_out.resize(n_vocab);
        ggml_backend_tensor_get(ggml_graph_get_tensor(dec_gf, "logits"), logits_out.data(),
                                0, sizeof(float) * n_vocab);
        n_past = S;
        return true;
    }

    if (!build_text_graph(g, plan, T, S, S)) return false;
    vqa::dump::finalize_sinks(g.ctx, g.gf, backend);
    if (!ggml_gallocr_alloc_graph(allocr, g.gf)) {
        fprintf(stderr, "text graph allocation failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }
    ggml_backend_tensor_set(g.tok_in, toks.data(), 0, sizeof(int) * T);
    {
        std::vector<int> mpos((size_t)T * 5);
        if (T == (int)plan.ids.size()) {
            mpos = plan.mpos;
        } else {
            // continuation step (CPU decode), planar [t,h,w,e,slot]
            for (int i = 0; i < T; i++) {
                const int p = base_slot + plan.rope_delta + i;
                for (int a = 0; a < 4; a++) mpos[(size_t)a * T + i] = p;
                mpos[(size_t)4 * T + i] = base_slot + i;
            }
        }
        ggml_backend_tensor_set(g.mpos_in, mpos.data(), 0, sizeof(int) * mpos.size());
        if (T == (int)plan.ids.size()) {   // parity dumps: prefill only
            dump_host("tok_in", toks.data(), sizeof(int) * T, {T});
            dump_host("mpos_in", mpos.data(), sizeof(int) * mpos.size(), {(int64_t)T * 5});
        }
    }
    vqa::set_causal_mask(g.mask_in, n_past, T);
    {
        std::vector<float> cm(T, 1.0f);
        if (T == (int)plan.colmask.size()) cm = plan.colmask;
        if (plan.n_vtot > 0)   // colmask only joins the graph when vision is present
            ggml_backend_tensor_set(g.colmask_in, cm.data(), 0, sizeof(float) * T);
        if (T == (int)plan.ids.size()) dump_host("colmask_in", cm.data(), sizeof(float) * T, {T});
    }
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "text compute failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }
    if (T == (int)plan.ids.size()) {   // parity dumps: prefill only
        dump(g.gf, "emb");
        dump(g.gf, "t_l0_norm");
        dump(g.gf, "t_l0_q");
        dump(g.gf, "t_l0_q_norm");
        dump(g.gf, "t_l0_q_rope");
        dump(g.gf, "t_l0_attn");
        dump(g.gf, "t_l0_out");
        dump(g.gf, "t_x2_norm");
        dump(g.gf, "t_x2_q_rope");
        dump(g.gf, "t_x2_xattn");
        dump(g.gf, "t_x2_out");
        dump(g.gf, "t_final_norm");
        dump(g.gf, "logits");
    }
    logits_out.resize(n_vocab);
    ggml_backend_tensor_get(g.logits, logits_out.data(), 0, sizeof(float) * n_vocab);
    n_past = S;
    ggml_free(g.ctx);
    g.ctx = nullptr;
    return true;
}

int MossVL::sample(const std::vector<float> & logits, const vqa::GenParams & gp) {
    return vqa::sample_logits(logits, gp, rng_state);
}

// raw host-side inputs for parity (no graph tensor round trip)
void MossVL::dump_host(const char * name, const void * data, size_t bytes,
                       std::initializer_list<int64_t> ne) {
    if (dump_dir.empty()) return;
    std::string path = dump_dir + "/cpp_" + name + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    if (f) { fwrite(data, 1, bytes, f); fclose(f); }
    FILE * m = fopen((dump_dir + "/cpp_" + name + ".shape").c_str(), "w");
    if (m) {
        for (size_t i = 0; i < ne.size(); i++)
            fprintf(m, "%lld%c", (long long)ne.begin()[i], i + 1 < ne.size() ? ' ' : '\n');
        fclose(m);
    }
}

void MossVL::dump(const ggml_cgraph * gf, const char * name) {
    if (dump_dir.empty()) return;
    vqa::dump::write((ggml_cgraph *)gf, dump_dir.c_str(), name);   // core/ggml_dump.hpp
}

bool MossVL::ask(const vqa::MediaInput & media, const std::string & prompt,
                 const std::string & /*mode*/, const vqa::GenParams & gp, std::string & out_text) {
    if (gp.on_stage) gp.on_stage("vision", 0, 1, gp.on_stage_ud);
    const auto tv0 = std::chrono::steady_clock::now();

    // per-media cross-state cache: the tower/merger output only depends on the
    // frame bytes; the cross-KV projection (k_norm + vision MRoPE) re-runs per
    // ask since the frame positions t0 depend on the prompt
    static struct {
        uint64_t key = 0;
        std::vector<float> merged;
    } vcache;

    // 1. measure the text overhead with a 1-frame probe, then size the vision
    //    budget (like vc3: clamp or reject BEFORE the multi-second vision encode)
    const int T = media.empty() ? 0 : (int)media.frames.size();
    int text_overhead = 0;
    {
        vqa::MediaInput probe = media;
        probe.frames.resize(1);
        PromptPlan pp;
        const int g4[2] = {4, 4};
        if (!build_prompt(probe, prompt, std::vector<int>(g4, g4 + 2), pp)) return false;
        text_overhead = (int)pp.ids.size() + 12 * std::max(T - 1, 0);
    }
    const int budget = n_ctx - gp.max_tokens - text_overhead - 16;
    if (budget <= 0) {
        fprintf(stderr, "n_ctx %d too small: prompt ~%d tokens + max_new %d\n",
                n_ctx, text_overhead, gp.max_tokens);
        return false;
    }

    std::vector<float> patches, pos;
    std::vector<int> vpos;
    int gh = 0, gw = 0;
    if (!prepare_media(media, budget, patches, pos, vpos, &gh, &gw)) return false;
    std::vector<int> grids;
    if (T > 0) {
        const int eh = gh / vc.merge, ew = gw / vc.merge;
        for (int f = 0; f < T; f++) { grids.push_back(eh); grids.push_back(ew); }
    }

    PromptPlan plan;
    if (!build_prompt(media, prompt, grids, plan, gp.history)) return false;
    if (plan.n_vtot + gp.max_tokens + (int)plan.ids.size() > n_ctx) {
        fprintf(stderr, "n_ctx %d too small: prompt %d + vision %d + max_new %d\n",
                n_ctx, (int)plan.ids.size(), plan.n_vtot, gp.max_tokens);
        return false;
    }

    // 2. vision encode (cache hits skip the tower, never the KV projection)
    n_past = 0;
    kv.clear();
    xkv.clear();
    sess_active = false;   // a fresh ask() invalidates any live session
    if (T > 0) {
        uint64_t key = 1469598103934665603ull;
        for (const auto & fr : media.frames) {
            key ^= fr.rgb.size(); key *= 1099511628211ull;
            for (size_t k = 0; k < fr.rgb.size(); k++) { key ^= fr.rgb[k]; key *= 1099511628211ull; }
        }
        key ^= (uint64_t)media.width << 32 ^ (uint64_t)media.height;
        // Both paths share ONE projection graph now: encode_vision produces
        // only the merger output (terminal node, trustworthy readback), and
        // the cross-KV projection always runs in project_cross_kv() - so a
        // cache hit and a fresh encode produce identical K/V by construction.
        // (History: the projection used to be chained into the vision graph
        // on cache misses; gallocr recycled v_merged's scratch buffer
        // mid-compute, so the cache-entry readback stored stale garbage and
        // every CUDA cache hit re-projected it -> hallucinated output. The
        // tensor-name tags in append_cross_kv fixed a separate CUDA-graph
        // replay bug and remain in place.)
        if (vcache.key == key && !vcache.merged.empty()) {
            fprintf(stderr, "vision: cache hit (%d merged tokens)\n",
                    (int)(vcache.merged.size() / vc.out_hidden));
        } else {
            const int N = (int)(patches.size() / (3 * vc.patch * vc.patch));
            if (!encode_vision(patches, pos, vpos, N, vcache.merged, &plan)) return false;
            vcache.key = key;
        }
        if (!project_cross_kv(vcache.merged, plan)) return false;
    }
    if (gp.on_stage) gp.on_stage("vision", 1, 1, gp.on_stage_ud);
    fprintf(stderr, "[perf] vision: %.0f ms (%d cross rows)\n",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv0).count(),
            plan.n_vtot);

    // 3. decode
    TextGraph g;
    std::vector<float> logits;
    const auto t_pf0 = std::chrono::steady_clock::now();
    if (!run_text(g, plan, plan.ids, 0, logits)) return false;
    const auto t1 = std::chrono::steady_clock::now();

    out_text.clear();
    int next = sample(logits, gp);
    int n_dec = 0;
    const int max_new = gp.max_tokens;
    for (int step = 0; step < max_new; step++) {
        if (next == id_eos || next == id_endoftext) break;
        // specials decode to the empty string; a stream of them is turn
        // machinery (silence / markers), not content — stop instead of
        // burning max_tokens on invisible output
        if (next >= 151643 && next != id_im_start && tok_.decode({next}).empty()) {
            fprintf(stderr, "moss_vl: stopping on empty special %d\n", next);
            break;
        }
        const std::string piece = tok_.decode({next});
        out_text += piece;
        if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
        n_dec++;
        if (step + 1 >= max_new) break;
        if (!run_text(g, plan, {next}, n_past, logits)) break;
        next = sample(logits, gp);
    }
    fprintf(stderr, "[perf] prefill %d tok in %.0f ms | decode %d tok in %.0f ms (%.2f tok/s)\n",
            (int)plan.ids.size(),
            std::chrono::duration<double, std::milli>(t1 - t_pf0).count(), n_dec,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count(),
            n_dec > 0 ? n_dec * 1000.0 / std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t1).count() : 0.0);
    sess_plan = plan;      // snapshot for a session started on top of this ask
    sess_vrows = plan.n_vtot;
    return true;
}

// realtime session turn: appends (frames | user prompt | both) onto the live
// KV caches instead of rebuilding the whole context. Mirrors the official
// real_time_generate splice format:
//   prompt turn: <|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n
//                (+ <|silence|> SFT-alignment token when frames follow)
//   frames:      <|vision_start|><|time_start|>t<|time_end|><|image_pad|>...<|vision_end|>
bool MossVL::ask_session(const vqa::MediaInput & media, const std::string & prompt,
                         const vqa::GenParams & gp, std::string & out_text) {
    if (!sess_active) {
        // turn 1: full prefill via the offline path, then keep the caches alive
        if (!ask(media, prompt, "", gp, out_text)) return false;
        sess_active = true;
        return true;
    }
    const auto t0 = std::chrono::steady_clock::now();

    // 1. vision for the new frames (host path: incremental rows into the cache)
    const int T_new = (int)media.frames.size();
    std::vector<float> patches, pos;
    std::vector<int> vpos;
    int gh = 0, gw = 0;
    if (T_new > 0) {
        const int budget = n_ctx - n_past - gp.max_tokens - 64;
        if (!prepare_media(media, budget, patches, pos, vpos, &gh, &gw)) return false;
    }
    std::vector<int> grids;
    if (T_new > 0) {
        const int eh = gh / vc.merge, ew = gw / vc.merge;
        for (int f = 0; f < T_new; f++) { grids.push_back(eh); grids.push_back(ew); }
    }

    // 2. build the splice text exactly like the official realtime loop
    std::string text;
    if (!prompt.empty()) {
        text += "<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        if (T_new > 0 && getenv("MOSS_RT_SILENCE")) text += "<|silence|>";   // SFT alignment (optional)
    }
    if (T_new > 0) {
        text += "<|vision_start|>";
        for (int f = 0; f < T_new; f++) {
            char ts[64];
            const double t = f < (int)media.timestamps.size() ? media.timestamps[f]
                             : (media.fps > 0 ? f / media.fps : (double)f);
            snprintf(ts, sizeof(ts), "<|time_start|>%.1f seconds<|time_end|>", t);
            text += ts;
            text += "<|image_pad|>";
        }
        text += "<|vision_end|>";
    }

    // 3. plan the segment against the live caches
    PromptPlan seg;
    const int pos_base = n_past + sess_plan.rope_delta;   // mrope pos of slot n_past
    if (!plan_from_text(text, grids, seg, pos_base, n_past, sess_vrows)) return false;
    if (seg.n_vtot + gp.max_tokens + n_past + (int)seg.ids.size() > n_ctx) {
        fprintf(stderr, "session context overflow (%d used)\n", n_past + (int)seg.ids.size());
        return false;
    }

    // 4. project the new frames' cross-KV rows (appended after the old ones)
    if (T_new > 0) {
        static struct {
            uint64_t key = 0;
            std::vector<float> merged;
        } scache;
        uint64_t key = 1469598103934665603ull;
        for (const auto & fr : media.frames) {
            key ^= fr.rgb.size(); key *= 1099511628211ull;
            for (size_t k = 0; k < fr.rgb.size(); k++) { key ^= fr.rgb[k]; key *= 1099511628211ull; }
        }
        key ^= (uint64_t)media.width << 32 ^ (uint64_t)media.height;
        if (scache.key == key && !scache.merged.empty() && ggml_backend_is_cpu(backend)) {
            fprintf(stderr, "session vision: cache hit\n");
        } else {
            const int N = (int)(patches.size() / (3 * vc.patch * vc.patch));
            if (!encode_vision(patches, pos, vpos, N, scache.merged)) return false;
            scache.key = key;
        }
        if (!project_cross_kv(scache.merged, seg, sess_vrows)) return false;
    }

    // 5. prefill the segment onto the live text KV, then decode
    TextGraph g;
    std::vector<float> logits;
    if (!run_text(g, seg, seg.ids, 0, logits)) return false;
    const auto t1 = std::chrono::steady_clock::now();

    out_text.clear();
    int next = sample(logits, gp);
    int n_dec = 0;
    const int max_new = gp.max_tokens;
    for (int step = 0; step < max_new; step++) {
        if (next == id_eos || next == id_endoftext) break;
        // specials decode to the empty string; a flood of them means the turn
        // is over in spirit (silence/turn markers etc.) — stop instead of
        // burning max_tokens on invisible output. The official realtime loop
        // feeds EVERY generated token (incl. <|silence|>) back into the
        // context, so the closing token must enter the KV before we break —
        // otherwise follow-up prompts land off-distribution.
        if (next >= 151643 && tok_.decode({next}).empty()) {
            if (getenv("MOSS_RT_DEBUG")) {
                std::vector<std::pair<float, int>> top(logits.size());
                for (int i = 0; i < (int)logits.size(); i++) top[i] = {logits[i], i};
                std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                                  [](auto & a, auto & b) { return a.first > b.first; });
                fprintf(stderr, "[rt-dbg] stop token %d; top5:", next);
                for (int i = 0; i < 5; i++)
                    fprintf(stderr, " %d(%.2f)", top[i].second, top[i].first);
                fprintf(stderr, "\n");
            }
            run_text(g, seg, {next}, n_past, logits);   // closing token enters the KV
            break;
        }
        if (next == id_silence) {
            run_text(g, seg, {next}, n_past, logits);
            break;
        }
        const std::string piece = tok_.decode({next});
        out_text += piece;
        if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
        n_dec++;
        if (step + 1 >= max_new) break;
        if (!run_text(g, seg, {next}, n_past, logits)) break;
        next = sample(logits, gp);
    }
    fprintf(stderr, "[perf] session turn: prefill %d tok in %.0f ms | decode %d tok in %.0f ms\n",
            (int)seg.ids.size(),
            std::chrono::duration<double, std::milli>(t1 - t0).count(), n_dec,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count());
    // carry the session forward: decode positions continue via seg.rope_delta
    sess_plan = seg;
    sess_vrows = seg.n_vtot;
    return true;
}

}  // namespace moss_vl

namespace vqa {
Model * create_moss_vl() { return new moss_vl::MossVL(); }
}  // namespace vqa
