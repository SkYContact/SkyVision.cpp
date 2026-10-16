// qwen3_5_model.cpp - Qwen3_5-2B backend (Qwen3.5 hybrid text + Qwen3-VL ViT).
//
// Text decoder: 24 layers, 18 gated-delta linear-attention layers + 6 full
// attention layers (3,7,11,15,19,23). Full layers use interleaved MRoPE
// (GGML_ROPE_TYPE_IMROPE, sections 11/11/10, 64 of 256 head dims), per-head
// q|gate interleaved projection and sigmoid output gating. Linear layers use
// the fused ggml_gated_delta_net op (chunk + single-token both covered) with
// persistent f32 conv [k-1, conv_dim] and state [128,128,16] buffers.
// Vision tower: 24 ViT blocks via the shared vqa::vit::block, learned absolute
// pos table, 2D rope (GGML_ROPE_TYPE_VISION, sections 16/16), per-temporal-step
// unmasked flash attention, 2x2 merger + 2-layer GELU(erf) projector.
//
// Weight-layout notes that differ from the VideoChat3 backend:
//   * Qwen3.5 RMSNorm applies x*(1+w): the converter already stored w+1.
//   * there is no separate ffn_norm - attn_post_norm feeds the FFN directly.
//   * text KV exists only for the 6 full-attention layers.
#include "vqa.hpp"
#include "vit_block.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#   include "ggml-cuda.h"
#endif
#include "tokenizer.hpp"
#include "attention.hpp"
#include "sampler.hpp"
#include "vqa_util.hpp"
#include "ggml_dump.hpp"
#include "stage_buf.hpp"
#include "../videochat3/vc3_preprocess.hpp"   // smart_resize / resize_bicubic_antialias
#include "qwen3_5_preprocess.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <filesystem>

namespace qwen3_5 {

struct VisLayer {
    ggml_tensor *norm0_w = nullptr, *norm0_b = nullptr;
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *wqkv_w = nullptr, *wqkv_b = nullptr;
    ggml_tensor *wo_w = nullptr, *wo_b = nullptr;
    ggml_tensor *fc0_w = nullptr, *fc0_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
};

struct TxtLayer {
    bool full = false;
    ggml_tensor *norm = nullptr, *post_norm = nullptr;
    // full attention
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    ggml_tensor *qn = nullptr, *kn = nullptr;
    // linear attention (gated delta net)
    ggml_tensor *la_qkv = nullptr, *la_gate = nullptr;
    ggml_tensor *la_beta = nullptr, *la_alpha = nullptr;
    ggml_tensor *la_dt = nullptr, *la_a = nullptr;       // [n_vh] f32
    ggml_tensor *la_conv1d = nullptr;                     // [k, conv_dim] f32
    ggml_tensor *la_norm = nullptr;                       // [head_v_dim]
    ggml_tensor *la_out = nullptr;
    // ffn
    ggml_tensor *gate = nullptr, *up = nullptr, *down = nullptr;
};

struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *mpos_in = nullptr, *mask_in = nullptr;
    ggml_tensor *img_in = nullptr, *img_pos_in = nullptr;
    ggml_tensor *logits = nullptr;
};

class Qwen3_5 : public vqa::Model {
public:
    ~Qwen3_5() override;
    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "qwen3.5"; }
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp,
             std::string & out_text) override;

    int n_ctx = 8192;
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool alloc_state();
    bool encode_vision(const vqa::MediaInput & media, std::vector<float> & feats,
                       int * n_tokens, int * grid /* t, h, w (merged) */, int token_budget = 0);
    bool build_text_graph(TextGraph & g, int T, int S, int n_img, int state_slot = 0, bool logits_all = false);
    // qwen3_5_mtp.inc
    int mtp_draft_step(int tok, const float * h, int slot, int axis,
                       std::vector<float> & logits_out, std::vector<float> & h_out);
    bool mtp_verify(const int * toks, int n_past, int axis0, int state_slot,
                    std::vector<float> & logits2, std::vector<float> & h_out);
    bool run_speculative(int next, int seed0, const std::vector<float> & logits_prev, float * h_seed,
                         int max_new, const vqa::GenParams & gp, std::string & raw, int & n_dec);
    bool run_text(TextGraph & g, const std::vector<int> & toks,
                  const std::vector<float> * img_feats, const std::vector<int> * img_pos,
                  const std::vector<int> & mpos, std::vector<float> & logits_out,
                  std::vector<float> * h_last = nullptr);
    std::string render_prompt(int n_img_tokens, const std::string & user_text, bool video) const;
    int sample(const std::vector<float> & logits, const vqa::GenParams & gp);
    void dump(const ggml_cgraph * gf, const char * name);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    int n_layer = 24, n_embd = 2048, n_head = 8, n_kv_head = 2, head_dim = 256;
    int n_ff = 6144, n_vocab = 248320;
    float rms_eps = 1e-6f, rope_theta = 1e7f;
    int rope_dim = 64;
    int mrope_sections[4] = {11, 11, 10, 0};
    int n_full_layers = 6;
    // gated-delta dims
    int ssm_d_conv = 4, ssm_d_state = 128, ssm_d_inner = 2048;
    int ssm_n_group = 16, ssm_dt_rank = 16;

    VisConfig vc;
    int id_image_pad = 248056, id_video_pad = 248057;
    int id_vision_start = 248053, id_vision_end = 248054;
    int id_eos = 0, id_endoftext = 248044;

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr, wbuf_plain = nullptr;
    std::vector<char> host_f32_storage;
    vqa::MMap mm;
    std::vector<float> patch_bias;       // v.patch_embd.bias, read on the host at load time
    vqa::StageBuffer tower_buf;          // device-side tower output, host never touched between steps
    std::vector<float> pos_table_host;   // [pos_side^2, n_embd]

    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr;
    std::vector<TxtLayer> tl;
    std::vector<VisLayer> vl;
    ggml_tensor *v_patch_w = nullptr, *v_patch_b = nullptr;
    ggml_tensor *mm_norm_w = nullptr, *mm_norm_b = nullptr;
    ggml_tensor *mm_fc1_w = nullptr, *mm_fc1_b = nullptr;
    ggml_tensor *mm_fc2_w = nullptr, *mm_fc2_b = nullptr;

    vqa::KvCache kv;                     // 6 full-attention layers only
    // speculative MTP head (official Qwen3.5 checkpoints; see dev-notes)
    bool has_mtp = false;
    bool marlin_skills = false;   // trained caption/find skills (Marlin fine-tunes only)
    ggml_tensor *mtp_fc = nullptr, *mtp_e_norm = nullptr, *mtp_h_norm = nullptr;
    ggml_tensor *mtp_head_norm = nullptr, *mtp_attn_norm = nullptr, *mtp_post_norm = nullptr;
    ggml_tensor *mtp_q = nullptr, *mtp_k = nullptr, *mtp_v = nullptr, *mtp_o = nullptr;
    ggml_tensor *mtp_qn = nullptr, *mtp_kn = nullptr;
    ggml_tensor *mtp_gate = nullptr, *mtp_up = nullptr, *mtp_down = nullptr;
    vqa::KvCache mtp_kv;                 // the head's single attention layer
    // persistent gated-delta states (one per linear layer)
    ggml_context * sctx = nullptr;
    ggml_backend_buffer_t sbuf = nullptr;
    // two rollback slots per linear layer: slot 1 holds the state one token
    // back (written by the speculative verify graph, read when a draft token
    // is rejected); slot 0 is the current state everywhere else
    std::vector<ggml_tensor *> ssm_state, conv_state;          // slot 0
    std::vector<ggml_tensor *> ssm_state_b, conv_state_b;      // slot 1
    std::vector<ggml_tensor *> ssm_state_c, conv_state_c;      // slot 2 (pre-round save)

    ggml_gallocr_t allocr = nullptr;
    int n_past = 0;
    std::vector<int> last_mpos;          // 3-axis position of the last prompt token
    int mrope_delta = 0;                 // continuation offset (get_rope_index)

    // static decode graph (T=1): same gf object across steps -> CUDA graph capture
    ggml_context * dec_ctx = nullptr;
    ggml_cgraph * dec_gf = nullptr;
    int dec_blk = 0;
    ggml_tensor * dec_tok = nullptr, * dec_mpos = nullptr, * dec_mask = nullptr;

    // speculative graph caches (qwen3_5_mtp.inc): one verify graph per state
    // slot + one draft graph, keyed on the 512-block KV window; allocated once
    // and reused across rounds so per-round cost is input-set + compute only
    struct SpecCache {
        ggml_context * ctx = nullptr;
        ggml_cgraph * gf = nullptr;
        int blk = -1, slot = -1;
        ggml_tensor * tok = nullptr, * mpos = nullptr, * mask = nullptr;
        ggml_tensor * logits = nullptr, * hout = nullptr;
    };
    SpecCache spec_v[2];
    SpecCache spec_d;

    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

// ---------------------------------------------------------------- helpers
static float gguf_f32(struct gguf_context * g, const char * key, float def) {
    int64_t id = gguf_find_key(g, key);
    return id >= 0 ? gguf_get_val_f32(g, id) : def;
}
static int gguf_u32(struct gguf_context * g, const char * key, int def) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return def;
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return (int)gguf_get_val_u32(g, id);
        case GGUF_TYPE_BOOL:   return gguf_get_val_bool(g, id) ? 1 : 0;
        case GGUF_TYPE_INT32:  return (int)gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (int)gguf_get_val_u64(g, id);
        case GGUF_TYPE_INT64:  return (int)gguf_get_val_i64(g, id);
        default: return def;
    }
}

static void tensor_to_f32(ggml_tensor * t, float * dst) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        memcpy(dst, t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n; i++) dst[i] = ggml_fp16_to_fp32(s[i]);
    } else {
        const struct ggml_type_traits * tt = ggml_get_type_traits(t->type);
        tt->to_float(t->data, dst, n);
    }
}

static float randf_(uint64_t & s) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return (float)(((s * 0x2545F4914F6CDD1Dull) >> 11) * (1.0 / 900719254740992.0));
}

Qwen3_5::~Qwen3_5() {
    if (dec_ctx) ggml_free(dec_ctx);
    for (auto & c : spec_v) if (c.ctx) ggml_free(c.ctx);
    if (spec_d.ctx) ggml_free(spec_d.ctx);
    if (sctx) ggml_free(sctx);
    if (sbuf) ggml_backend_buffer_free(sbuf);
    kv.free();
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wbuf_plain) ggml_backend_buffer_free(wbuf_plain);
    if (wctx) ggml_free(wctx);
    if (allocr) ggml_gallocr_free(allocr);
    if (backend) ggml_backend_free(backend);
}

bool Qwen3_5::init_backend() {
    if (use_gpu) {
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
#endif
    }
    if (backend) {
        fprintf(stderr, "backend: %s\n", ggml_backend_name(backend));
    } else {
        backend = ggml_backend_cpu_init();
        if (!backend) { fprintf(stderr, "no backend available\n"); return false; }
        if (n_threads <= 0) n_threads = (int)std::max(1u, std::thread::hardware_concurrency() / 2);
        ggml_backend_cpu_set_n_threads(backend, n_threads);
        fprintf(stderr, "backend: %s (%d threads)\n", ggml_backend_name(backend), n_threads);
    }
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

// ---------------------------------------------------------------- load
bool Qwen3_5::load(const std::string & gguf_path, const std::string & tokenizer_path,
                  const vqa::LoadOptions & opt) {
    n_ctx = opt.n_ctx > 0 ? opt.n_ctx : n_ctx;
    n_threads = opt.n_threads;
    use_gpu = opt.use_gpu;
    dump_dir = opt.dump_dir;
    if (const char * d = getenv("QWEN35_DUMP_DIR")) dump_dir = d;
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::Qwen2Regex)) return false;
    if (!alloc_state()) return false;

    auto pick = [&](const char * s, int def) {
        int v = tok_.id_of(s);
        return v >= 0 ? v : def;
    };
    id_image_pad = pick("<|image_pad|>", id_image_pad);
    id_video_pad = pick("<|video_pad|>", id_video_pad);
    id_vision_start = pick("<|vision_start|>", id_vision_start);
    id_vision_end = pick("<|vision_end|>", id_vision_end);
    id_eos = pick("<|im_end|>", 248046);
    id_endoftext = pick("<|endoftext|>", id_endoftext);

    caps_ = {};
    caps_.image = true;
    caps_.video = true;
    caps_.timestamps_prompt = true;
    fprintf(stderr, "marlin: ready (%d text layers [%d full-attn], %d vision layers, ctx %d)\n",
            n_layer, n_full_layers, vc.n_layer, n_ctx);
    return true;
}

bool Qwen3_5::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f GB\n", mm.size / 1e9);

    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer   = gguf_u32(gu, "marlin.block_count", n_layer);
    n_embd    = gguf_u32(gu, "marlin.embedding_length", n_embd);
    n_head    = gguf_u32(gu, "marlin.attention.head_count", n_head);
    n_kv_head = gguf_u32(gu, "marlin.attention.head_count_kv", n_kv_head);
    head_dim  = gguf_u32(gu, "marlin.attention.key_length", head_dim);
    n_ff      = gguf_u32(gu, "marlin.feed_forward_length", n_ff);
    n_vocab   = gguf_u32(gu, "marlin.vocab_size", n_vocab);
    rms_eps   = gguf_f32(gu, "marlin.attention.layer_norm_rms_epsilon", rms_eps);
    rope_theta = gguf_f32(gu, "marlin.rope.freq_base", rope_theta);
    rope_dim  = gguf_u32(gu, "marlin.rope.dimension_count", rope_dim);
    mrope_sections[0] = gguf_u32(gu, "marlin.rope.dimension_section_0", mrope_sections[0]);
    mrope_sections[1] = gguf_u32(gu, "marlin.rope.dimension_section_1", mrope_sections[1]);
    mrope_sections[2] = gguf_u32(gu, "marlin.rope.dimension_section_2", mrope_sections[2]);
    mrope_sections[3] = gguf_u32(gu, "marlin.rope.dimension_section_3", mrope_sections[3]);
    ssm_d_conv = gguf_u32(gu, "marlin.ssm.conv_kernel", ssm_d_conv);
    ssm_d_state = gguf_u32(gu, "marlin.ssm.state_size", ssm_d_state);
    ssm_d_inner = gguf_u32(gu, "marlin.ssm.inner_size", ssm_d_inner);
    ssm_n_group = gguf_u32(gu, "marlin.ssm.group_count", ssm_n_group);
    ssm_dt_rank = gguf_u32(gu, "marlin.ssm.time_step_rank", ssm_dt_rank);

    vc.n_layer = gguf_u32(gu, "marlin.vision.block_count", vc.n_layer);
    vc.n_embd  = gguf_u32(gu, "marlin.vision.embedding_length", vc.n_embd);
    vc.n_head  = gguf_u32(gu, "marlin.vision.attention.head_count", vc.n_head);
    vc.n_ff    = gguf_u32(gu, "marlin.vision.feed_forward_length", vc.n_ff);
    vc.patch   = gguf_u32(gu, "marlin.vision.patch_size", vc.patch);
    vc.temporal_patch = gguf_u32(gu, "marlin.vision.temporal_patch_size", vc.temporal_patch);
    vc.merge   = gguf_u32(gu, "marlin.vision.merge_size", vc.merge);
    vc.pos_side = (int)std::lround(sqrtf((float)gguf_u32(gu, "marlin.vision.pos_table_size", 2304)));
    vc.layernorm_eps = gguf_f32(gu, "marlin.vision.layer_norm_epsilon", vc.layernorm_eps);
    vc.rope_theta = gguf_f32(gu, "marlin.vision.rope_theta", vc.rope_theta);
    vc.out_hidden = gguf_u32(gu, "marlin.vision.out_hidden_size", vc.out_hidden);
    has_mtp = gguf_u32(gu, "marlin.has_mtp", 0) != 0;
    marlin_skills = gguf_u32(gu, "marlin.skills", 0) != 0;
    id_image_pad = gguf_u32(gu, "marlin.image_token_id", id_image_pad);
    id_video_pad = gguf_u32(gu, "marlin.video_token_id", id_video_pad);
    id_vision_start = gguf_u32(gu, "marlin.vision_start_token_id", id_vision_start);
    id_vision_end = gguf_u32(gu, "marlin.vision_end_token_id", id_vision_end);
    id_endoftext = gguf_u32(gu, "marlin.pad_token_id", id_endoftext);

    const size_t data_off = gguf_get_data_offset(gu);
    const int64_t nt = gguf_get_n_tensors(gu);
    for (int64_t i = 0; i < nt; i++) {
        const char * name = gguf_get_tensor_name(gu, i);
        ggml_tensor * t = ggml_get_tensor(wctx, name);
        if (!t) continue;
        t->data = (char *)mm.ptr + data_off + gguf_get_tensor_offset(gu, i);
    }
    gguf_free(gu);

    // 1D tensors (norms, biases, dt/a vectors, conv kernels) -> host F32
    std::vector<char> host_f32;
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
            tensor_to_f32(t, dst);
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
        pos_table_host.resize(ggml_nelements(pe));
        tensor_to_f32(pe, pos_table_host.data());
    }

    {
        ggml_tensor * pb = ggml_get_tensor(wctx, "v.patch_embd.bias");
        if (pb && pb->type == GGML_TYPE_F32 && pb->ne[0] == vc.n_embd && pb->ne[1] == 1)
            patch_bias.assign((const float *)pb->data, (const float *)pb->data + vc.n_embd);
    }

    {
        std::vector<std::pair<ggml_tensor *, const void *>> uploads;
        size_t total = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t)) {
            uploads.push_back({t, t->data});
            total += ggml_nbytes(t);
        }
        if (ggml_backend_is_cpu(backend)) {
            wbuf = ggml_backend_cpu_buffer_from_ptr(mm.ptr, mm.size);
            if (!host_f32.empty())
                wbuf_plain = ggml_backend_cpu_buffer_from_ptr(host_f32.data(), host_f32.size());
            for (auto & u : uploads) {
                const char * pdata = (const char *)u.first->data;
                const char * fb = host_f32.empty() ? nullptr : host_f32.data();
                const char * fe = fb ? fb + host_f32.size() : nullptr;
                u.first->buffer = (fb && pdata >= fb && pdata < fe) ? wbuf_plain : wbuf;
            }
            fprintf(stderr, "weights: %.2f GB zero-copy from mmap\n", total / 1e9);
        } else {
            for (auto & u : uploads) u.first->data = nullptr;
            wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
            if (!wbuf) { fprintf(stderr, "weight allocation failed (need %.2f GB)\n", total / 1e9); return false; }
            for (auto & u : uploads) ggml_backend_tensor_set(u.first, u.second, 0, ggml_nbytes(u.first));
            fprintf(stderr, "uploaded %.2f GB of weights\n", total / 1e9);
        }
        if (ggml_backend_is_cpu(backend)) {
            host_f32_storage = std::move(host_f32);
        } else {
            host_f32.clear();
            host_f32.shrink_to_fit();
            mm.release();
        }
    }

    auto gt = [&](const char * name) {
        ggml_tensor * t = ggml_get_tensor(wctx, name);
        if (!t) fprintf(stderr, "missing tensor: %s\n", name);
        return t;
    };
    tok_embd = gt("token_embd.weight");
    out_norm = gt("output_norm.weight");
    v_patch_w = gt("v.patch_embd.weight");
    v_patch_b = gt("v.patch_embd.bias");
    mm_norm_w = gt("mm.norm.weight");
    mm_norm_b = gt("mm.norm.bias");
    mm_fc1_w = gt("mm.fc1.weight");
    mm_fc1_b = gt("mm.fc1.bias");
    mm_fc2_w = gt("mm.fc2.weight");
    mm_fc2_b = gt("mm.fc2.bias");

    // per-layer types: read the recurrent flag array when present
    std::vector<uint8_t> recurrent(n_layer, 1);
    {
        int64_t id = -1;
        struct gguf_context * g2 = gguf_init_from_file(path, {true, nullptr});
        if (g2) {
            id = gguf_find_key(g2, "marlin.recurrent_layers");
            if (id >= 0 && gguf_get_kv_type(g2, id) == GGUF_TYPE_ARRAY) {
                const uint8_t * arr = (const uint8_t *)gguf_get_arr_data(g2, id);
                for (int i = 0; i < n_layer && i < (int)gguf_get_arr_n(g2, id); i++) recurrent[i] = arr[i];
            }
            gguf_free(g2);
        }
    }
    n_full_layers = 0;
    tl.resize(n_layer);
    for (int i = 0; i < n_layer; i++) {
        char buf[128];
        auto g = [&](const char * suffix) {
            snprintf(buf, sizeof(buf), "blk.%d.%s", i, suffix);
            return gt(buf);
        };
        TxtLayer & L = tl[i];
        L.full = !recurrent[i];
        if (L.full) n_full_layers++;
        L.norm = g("attn_norm.weight");
        L.post_norm = g("attn_post_norm.weight");
        if (L.full) {
            L.q = g("attn_q.weight");
            L.k = g("attn_k.weight");
            L.v = g("attn_v.weight");
            L.o = g("attn_output.weight");
            L.qn = g("attn_q_norm.weight");
            L.kn = g("attn_k_norm.weight");
        } else {
            L.la_qkv = g("la_qkv.weight");
            L.la_gate = g("la_gate.weight");
            L.la_beta = g("la_beta.weight");
            L.la_alpha = g("la_alpha.weight");
            L.la_dt = g("la_dt");
            L.la_a = g("la_a");
            L.la_conv1d = g("la_conv1d");
            L.la_norm = g("la_norm.weight");
            L.la_out = g("la_out.weight");
        }
        L.gate = g("ffn_gate.weight");
        L.up = g("ffn_up.weight");
        L.down = g("ffn_down.weight");
    }
    if (has_mtp) {
        mtp_fc = gt("mtp.fc");          mtp_e_norm = gt("mtp.e_norm");
        mtp_h_norm = gt("mtp.h_norm");  mtp_head_norm = gt("mtp.head_norm");
        mtp_attn_norm = gt("mtp.attn_norm"); mtp_post_norm = gt("mtp.post_norm");
        mtp_q = gt("mtp.q"); mtp_k = gt("mtp.k"); mtp_v = gt("mtp.v"); mtp_o = gt("mtp.o");
        mtp_qn = gt("mtp.q_norm"); mtp_kn = gt("mtp.k_norm");
        mtp_gate = gt("mtp.ffn_gate"); mtp_up = gt("mtp.ffn_up"); mtp_down = gt("mtp.ffn_down");
        if (!mtp_fc || !mtp_q || !mtp_head_norm) { fprintf(stderr, "mtp tensors incomplete\n"); return false; }
        fprintf(stderr, "MTP draft head present (speculative decode, QWEN35_SPEC=0 to disable)\n");
    }
    vl.resize(vc.n_layer);
    for (int i = 0; i < vc.n_layer; i++) {
        char buf[128];
        auto g = [&](const char * suffix) {
            snprintf(buf, sizeof(buf), "v.blk.%d.%s", i, suffix);
            return gt(buf);
        };
        VisLayer & L = vl[i];
        L.norm0_w = g("norm0.weight");
        L.norm0_b = g("norm0.bias");
        L.norm1_w = g("norm1.weight");
        L.norm1_b = g("norm1.bias");
        L.wqkv_w = g("wqkv.weight");
        L.wqkv_b = g("wqkv.bias");
        L.wo_w = g("wo.weight");
        L.wo_b = g("wo.bias");
        L.fc0_w = g("mlp.fc0.weight");
        L.fc0_b = g("mlp.fc0.bias");
        L.fc1_w = g("mlp.fc1.weight");
        L.fc1_b = g("mlp.fc1.bias");
    }
    if (!tok_embd || !out_norm || !v_patch_w) return false;
    if ((int)tok_embd->ne[0] != n_embd) {
        fprintf(stderr, "token_embd row length %lld != n_embd %d (tied head expected)\n",
                (long long)tok_embd->ne[0], n_embd);
        return false;
    }
    return true;
}

bool Qwen3_5::alloc_state() {
    if (!kv.alloc(backend, n_full_layers, n_kv_head, head_dim, n_ctx)) return false;
    if (has_mtp && !mtp_kv.alloc(backend, 1, n_kv_head, head_dim, n_ctx)) return false;
    const int n_lin = n_layer - n_full_layers;
    const int conv_dim = 2 * ssm_n_group * ssm_d_state + ssm_d_inner;   // 6144
    const int n_rec = 6 * n_lin + 8;   // 2 buffers x 3 rollback slots per linear layer
    struct ggml_init_params ip = { (size_t)n_rec * ggml_tensor_overhead(), NULL, true };
    sctx = ggml_init(ip);
    if (!sctx) return false;
    ssm_state.resize(n_lin);
    conv_state.resize(n_lin);
    ssm_state_b.resize(n_lin);
    conv_state_b.resize(n_lin);
    ssm_state_c.resize(n_lin);
    conv_state_c.resize(n_lin);
    for (int i = 0; i < n_lin; i++) {
        ssm_state[i] = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, ssm_d_state, ssm_d_state, ssm_dt_rank);
        conv_state[i] = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, ssm_d_conv - 1, conv_dim);
        ssm_state_b[i] = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, ssm_d_state, ssm_d_state, ssm_dt_rank);
        conv_state_b[i] = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, ssm_d_conv - 1, conv_dim);
        ssm_state_c[i] = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, ssm_d_state, ssm_d_state, ssm_dt_rank);
        conv_state_c[i] = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, ssm_d_conv - 1, conv_dim);
    }
    sbuf = ggml_backend_alloc_ctx_tensors(sctx, backend);
    if (!sbuf) { fprintf(stderr, "ssm state allocation failed\n"); return false; }
    ggml_backend_buffer_clear(sbuf, 0);
    fprintf(stderr, "ssm states: %.1f MB (%d linear layers)\n",
            (double)(n_lin * (128 * 128 * 16 + 3 * 6144) * 4) / 1e6, n_lin);
    return true;
}

void Qwen3_5::dump(const ggml_cgraph * gf, const char * name) {
    if (dump_dir.empty()) return;
    vqa::dump::write((ggml_cgraph *)gf, dump_dir.c_str(), name);
}

// ---------------------------------------------------------------- vision
namespace {
struct MarlinVitEnv {
    ggml_context *gctx;
    ggml_cgraph *gf;
    VisConfig *vc = nullptr;
    ggml_tensor *t_vpos = nullptr;
    int N = 0, vhd = 0;
    const char *stop_after = nullptr;
    std::function<bool(const char *)> run;
    int state = 0;
};

static ggml_tensor * marlin_vit_attention(void * ud_, ggml_tensor * qkv) {
    MarlinVitEnv & e = *(MarlinVitEnv *)ud_;
    ggml_context * gctx = e.gctx;
    const VisConfig & vc = *e.vc;
    const int N = e.N, vhd = e.vhd;

    const size_t off = (size_t)vc.n_head * vhd * qkv->nb[0];
    auto qkv_head_view = [&](int block) {
        return ggml_view_3d(gctx, qkv, vhd, vc.n_head, N,
                            (size_t)vhd * qkv->nb[0], qkv->nb[1], (size_t)block * off);
    };
    ggml_tensor * q = qkv_head_view(0);
    ggml_tensor * k = qkv_head_view(1);
    ggml_tensor * v = qkv_head_view(2);

    // native pair layout (h-block | w-block, NEOX halves) is already the
    // ggml GGML_ROPE_TYPE_VISION layout; sections 16|16 over 32 pairs
    int sections[GGML_MROPE_SECTIONS] = { vhd / 4, vhd / 4, 0, 0 };
    q = ggml_rope_multi(gctx, q, e.t_vpos, nullptr, vhd / 2, sections, GGML_ROPE_TYPE_VISION,
                        32768, vc.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_multi(gctx, k, e.t_vpos, nullptr, vhd / 2, sections, GGML_ROPE_TYPE_VISION,
                        32768, vc.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    if (e.stop_after && !strcmp(e.stop_after, "v_blk0_q_rope")) {
        ggml_set_name(q, "v_blk0_q_rope");
        ggml_build_forward_expand(e.gf, q);
        if (!e.run("v_blk0_q_rope")) e.state = 2; else e.state = 1;
        return nullptr;
    }

    // one temporal step per graph -> one unmasked FA over the whole stream
    auto fa_in = [&](ggml_tensor * t) {
        return ggml_view_3d(gctx, t, vhd, N, vc.n_head, t->nb[2], t->nb[1], 0);
    };
    ggml_tensor * fa = ggml_flash_attn_ext(gctx, fa_in(q), fa_in(k), fa_in(v), nullptr,
                                           1.0f / sqrtf((float)vhd), 0.0f, 0.0f);
    return ggml_reshape_2d(gctx, ggml_cont(gctx, fa), (int64_t)vc.n_embd, N);
}

static bool marlin_vit_probe(void * ud_, const char * stage, ggml_tensor * t) {
    MarlinVitEnv & e = *(MarlinVitEnv *)ud_;
    char nm[40];
    snprintf(nm, sizeof(nm), "v_blk0_%s", stage);
    ggml_set_name(t, nm);
    if (e.stop_after && !strcmp(e.stop_after, nm)) {
        ggml_build_forward_expand(e.gf, t);
        if (!e.run(nm)) e.state = 2; else e.state = 1;
        return true;
    }
    return false;
}
}  // namespace

bool Qwen3_5::encode_vision(const vqa::MediaInput & media, std::vector<float> & feats,
                           int * n_tokens, int * grid, int token_budget) {
    *n_tokens = 0;
    grid[0] = grid[1] = grid[2] = 0;
    if (media.empty()) return true;
    const int T = (int)media.frames.size();
    const int src_h = media.height, src_w = media.width;
    const int factor = vc.patch * vc.merge;                       // 32

    // frame count to a multiple of temporal_patch by repeating the last frame
    int T2 = T;
    std::vector<const vqa::Frame *> frames;
    for (int i = 0; i < T; i++) frames.push_back(&media.frames[i]);
    while (T2 % vc.temporal_patch != 0) { frames.push_back(&media.frames[T - 1]); T2++; }
    const int gt = T2 / vc.temporal_patch;

    long long max_pixels = media.kind == vqa::MediaKind::Video ? 200704 : 16384LL * 32 * 32;
    if (media.kind == vqa::MediaKind::Video) {
        if (const char * e = getenv("QWEN35_VIDEO_MP"))
            max_pixels = (long long)(atof(e) * 1e6 + 0.5);
    }
    if (token_budget > 0) {
        // LLM tokens per temporal step = (gh/2)*(gw/2) = H*W/(32*32)
        const long long min_total = 4LL * gt;
        if (token_budget < min_total) {
            fprintf(stderr, "video cannot fit in n_ctx: needs >= %d vision tokens (%d step%s), "
                            "budget %d -- raise --n-ctx or reduce --max-tokens\n",
                    (int)min_total, gt, gt == 1 ? "" : "s", token_budget);
            return false;
        }
        const long long per_step = token_budget / gt;
        max_pixels = std::min(max_pixels, per_step * factor * factor);
        if (max_pixels < 200704)
            fprintf(stderr, "vision budget %d tokens (%d step%s): max_pixels clamped to %.2f MP\n",
                    token_budget, gt, gt == 1 ? "" : "s", max_pixels / 1e6);
    }
    int H = 0, W = 0;
    if (!vc3::smart_resize(src_h, src_w, factor, 4 * factor * factor,
                           (int)std::min(max_pixels, (long long)INT_MAX), &H, &W)) {
        fprintf(stderr, "image too small (%dx%d, factor %d)\n", src_w, src_h, factor);
        return false;
    }
    if (media.kind == vqa::MediaKind::Video)
        fprintf(stderr, "video %d frame%s %dx%d -> %dx%d each (%d temporal step%s)\n",
                T, T == 1 ? "" : "s", src_w, src_h, W, H, gt, gt == 1 ? "" : "s");
    else
        fprintf(stderr, "image %dx%d -> %dx%d\n", src_w, src_h, W, H);

    // resize once, patchify per temporal step (block-major, with pos + rope coords)
    std::vector<std::vector<float>> rgb(T2);
    for (int i = 0; i < T2; i++)
        vc3::resize_bicubic_antialias(frames[i]->rgb.data(), src_w, src_h, W, H, rgb[i]);

    std::vector<float> patches, pos;         // per-step blocks appended
    std::vector<int> coords;                 // (h, w) per token, block-major
    int gh = 0, gw = 0;
    for (int s = 0; s < gt; s++) {
        const float * fs[2] = { rgb[s * vc.temporal_patch].data(), rgb[s * vc.temporal_patch + 1].data() };
        std::vector<float> p, po;
        std::vector<int> co;
        qwen3_5::patchify3d(fs, H, W, vc, pos_table_host.data(), p, po, co, &gh, &gw);
        patches.insert(patches.end(), p.begin(), p.end());
        pos.insert(pos.end(), po.begin(), po.end());
        coords.insert(coords.end(), co.begin(), co.end());
    }
    const int hw = gh * gw;
    const int vhd = vc.n_embd / vc.n_head;
    const int C3 = 3 * vc.temporal_patch * vc.patch * vc.patch;
    if (!patch_bias.empty())
        vqa::vit::fold_patch_bias(pos.data(), pos.size(), vc.n_embd, patch_bias.data());

    const char * stop_after = getenv("QWEN35_STOP_AFTER");

    // ---- tower: one graph per temporal step (attention is step-local);
    // every step writes its [n_embd, hw] output into a slice of a persistent
    // DEVICE buffer (core/stage_buf.hpp) - the merger then reads the whole
    // buffer as a view, so tower outputs never round-trip through the host
    if (!tower_buf.ensure(backend, vc.n_embd, (int64_t)hw * gt)) return false;
    auto t_now = []() { return std::chrono::steady_clock::now(); };
    for (int s = 0; s < gt; s++) {
        const int Nc = hw;
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
            vpos[i] = coords[((size_t)s * hw + i) * 2];             // h  (section 0)
            vpos[(size_t)Nc + i] = coords[((size_t)s * hw + i) * 2 + 1];   // w  (section 1)
        }

        ggml_tensor * x = ggml_mul_mat(gctx, v_patch_w, t_patches);
        if (patch_bias.empty())
            x = ggml_add(gctx, x, v_patch_b);
        x = ggml_add(gctx, x, t_pos);
        ggml_set_name(x, "v_patch_out");

        auto run_chunk = [&](const char * out_name, std::vector<float> * dest) -> bool {
            if (!dump_dir.empty()) {
                static const char * const dump_names[] = {
                    "patches", "v_pos_in", "v_patch_out", "v_blk0_attn_out", "v_blk0_out",
                    "v_final", nullptr,
                };
                vqa::dump::mark_sinks(gctx, gf, dump_names);
            }
            if (!ggml_gallocr_alloc_graph(allocr, gf)) {
                fprintf(stderr, "vision graph allocation failed\n");
                return false;
            }
            if (t_patches->buffer) ggml_backend_tensor_set(t_patches, patches.data() + (size_t)s * hw * C3, 0, sizeof(float) * (size_t)Nc * C3);
            if (t_pos->buffer)     ggml_backend_tensor_set(t_pos, pos.data() + (size_t)s * hw * vc.n_embd, 0, sizeof(float) * (size_t)Nc * vc.n_embd);
            if (t_vpos->buffer)    ggml_backend_tensor_set(t_vpos, vpos.data(), 0, sizeof(int) * vpos.size());
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "vision compute failed\n");
                return false;
            }
            if (s == 0) {
                dump(gf, "patches");
                dump(gf, "v_pos_in");
                dump(gf, "v_patch_out");
                dump(gf, "v_blk0_attn_out");
                dump(gf, "v_blk0_out");
            }
            dump(gf, out_name);
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
        { const char * e = getenv("QWEN35_VL"); if (e) vl_max = std::min(vl_max, atoi(e)); }
        MarlinVitEnv env;
        env.gctx = gctx; env.gf = gf; env.vc = &vc; env.t_vpos = t_vpos;
        env.N = Nc; env.vhd = vhd; env.stop_after = stop_s; env.state = 0;
        env.run = [&run_chunk](const char * name) -> bool { std::vector<float> sink; return run_chunk(name, &sink); };
        for (int il = 0; il < vl_max; il++) {
            VisLayer & L = vl[il];
            vqa::vit::BlockWeights bw{L.norm0_w, L.norm0_b, L.wqkv_w, L.wqkv_b, L.wo_w, L.wo_b,
                                      L.norm1_w, L.norm1_b, L.fc0_w, L.fc0_b, L.fc1_w, L.fc1_b};
            vqa::vit::BlockHooks bh;
            bh.attention = marlin_vit_attention;
            bh.probe_layer0 = marlin_vit_probe;
            bh.ud = &env;
            x = vqa::vit::block(gctx, x, bw, vc.layernorm_eps, bh, il);
            if (!x) { ggml_free(gctx); return env.state == 1; }
        }
        // no final norm in this tower
        ggml_set_name(x, "v_final");
        ggml_tensor * out_sink = tower_buf.sink(gctx, x, (int64_t)s * hw);
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

    // ---- merger: LayerNorm(1024) per patch -> 2x2 fold (block-major stream)
    //      -> fc1 + GELU(erf) -> fc2
    const int N = hw * gt;
    const int mh = gh / vc.merge, mw = gw / vc.merge;
    {
        struct ggml_init_params ip2 = { 32ull * 1024 * 1024, NULL, true };
        ggml_context * mctx = ggml_init(ip2);
        if (!mctx) return false;
        struct ggml_cgraph * mgf = ggml_new_graph_custom(mctx, 8192, false);
        ggml_tensor * t_x = tower_buf.view(mctx, N, 0);   // device-side, no host round trip
        ggml_tensor * x = t_x;

        auto run_merger = [&](const char * out_name) -> bool {
            if (!dump_dir.empty()) {
                static const char * const dump_names[] = { "v_merge", "v_proj_out", nullptr };
                vqa::dump::mark_sinks(mctx, mgf, dump_names);
            }
            if (!ggml_gallocr_alloc_graph(allocr, mgf)) {
                fprintf(stderr, "merger graph allocation failed\n");
                return false;
            }
            if (ggml_backend_graph_compute(backend, mgf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "merger compute failed\n");
                return false;
            }
            dump(mgf, "v_merge");
            dump(mgf, "v_proj_out");
            dump(mgf, out_name);
            ggml_tensor * out = ggml_graph_get_tensor(mgf, out_name);
            const int n = (int)out->ne[1], d = (int)out->ne[0];
            feats.resize((size_t)n * d);
            ggml_backend_tensor_get(out, feats.data(), 0, sizeof(float) * feats.size());
            *n_tokens = n;
            return true;
        };

        ggml_tensor * xn = ggml_norm(mctx, x, vc.layernorm_eps);
        xn = ggml_mul(mctx, xn, mm_norm_w);
        xn = ggml_add(mctx, xn, mm_norm_b);
        // the stream is block-major: plain reshape folds each 2x2 neighbourhood
        ggml_tensor * x4 = ggml_reshape_2d(mctx, xn, (int64_t)vc.n_embd * vc.merge * vc.merge,
                                           (int64_t)mh * mw * gt);
        ggml_set_name(x4, "v_merge");
        if (stop_after && !strcmp(stop_after, "v_merge")) {
            ggml_build_forward_expand(mgf, x4);
            if (!run_merger("v_merge")) { ggml_free(mctx); return false; }
            ggml_free(mctx);
            *n_tokens = mh * mw * gt;
            grid[0] = gt; grid[1] = mh; grid[2] = mw;
            return true;
        }
        ggml_tensor * h = ggml_mul_mat(mctx, mm_fc1_w, x4);
        h = ggml_add(mctx, h, mm_fc1_b);
        h = ggml_gelu_erf(mctx, h);                    // nn.GELU() == exact erf
        h = ggml_mul_mat(mctx, mm_fc2_w, h);
        h = ggml_add(mctx, h, mm_fc2_b);
        ggml_set_name(h, "v_proj_out");
        ggml_build_forward_expand(mgf, h);
        if (!run_merger("v_proj_out")) { ggml_free(mctx); return false; }
        ggml_free(mctx);
    }
    grid[0] = gt; grid[1] = mh; grid[2] = mw;
    fprintf(stderr, "vision: %d patches (%dx%d, %d step%s) -> %d tokens%s\n", N, gh, gw,
            gt, gt == 1 ? "" : "s", *n_tokens, stop_after ? " (truncated for parity)" : "");
    return true;
}

// ---------------------------------------------------------------- text
bool Qwen3_5::build_text_graph(TextGraph & g, int T, int S, int n_img, int state_slot, bool logits_all) {
    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 32768, false);

    const int D = n_embd, HD = head_dim, NH = n_head, NKV = n_kv_head;
    const int Sk = ssm_d_state, Hv = ssm_dt_rank;         // 128, 16
    const int kv_dim = 2 * ssm_n_group * Sk + ssm_d_inner;    // 6144
    const int k_conv = ssm_d_conv;

    g.tok_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    // 3-axis (t, h, w) mrope positions + 1D absolute slot positions interleaved:
    // layout [t(T), h(T), w(T), slot(T)]
    g.mpos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, 4 * T);
    ggml_set_name(g.mpos_in, "mpos_in");
    g.mask_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, S, T);
    ggml_set_name(g.mask_in, "mask_in");

    ggml_tensor * x = ggml_get_rows(g.ctx, tok_embd, g.tok_in);      // [D, T]
    if (n_img > 0) {
        g.img_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, n_img);
        ggml_set_name(g.img_in, "img_in");
        g.img_pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_img);
        ggml_set_name(g.img_pos_in, "img_pos_in");
        x = ggml_set_rows(g.ctx, ggml_dup(g.ctx, x), g.img_in, g.img_pos_in);
    }
    ggml_set_name(x, "emb");

    // rope position views over mpos_in (b must be a [4N] vector for rope_multi)
    ggml_tensor * pos_rope = ggml_view_1d(g.ctx, g.mpos_in, 4 * T, 0);
    ggml_tensor * pos_slot = ggml_view_1d(g.ctx, g.mpos_in, T, (size_t)3 * T * sizeof(int));

    int kv_idx = 0, ssm_idx = 0;
    int max_layers = n_layer;
    if (const char * e = getenv("QWEN35_MAX_LAYERS")) max_layers = std::min(max_layers, atoi(e));
    for (int il = 0; il < max_layers; il++) {
        TxtLayer & L = tl[il];
        ggml_tensor * l = ggml_rms_norm(g.ctx, x, rms_eps);
        l = ggml_mul(g.ctx, l, L.norm);

        ggml_tensor * attn_out = nullptr;
        if (L.full) {
            const int ki = kv_idx++;
            ggml_tensor * q_full = ggml_mul_mat(g.ctx, L.q, l);      // [NH*2*HD, T]
            ggml_tensor * k = ggml_mul_mat(g.ctx, L.k, l);
            ggml_tensor * v = ggml_mul_mat(g.ctx, L.v, l);
            if (il == 0) { ggml_set_name(q_full, "t_l0_q"); ggml_set_name(k, "t_l0_k"); }

            // per-head interleaved [q(HD) | gate(HD)]
            ggml_tensor * q3 = ggml_view_3d(g.ctx, q_full, HD, NH, T,
                                            (size_t)2 * HD * ggml_element_size(q_full),
                                            q_full->nb[1], 0);
            q3 = ggml_rms_norm(g.ctx, q3, rms_eps);
            q3 = ggml_mul(g.ctx, q3, L.qn);
            ggml_tensor * gate = ggml_view_3d(g.ctx, q_full, HD, NH, T,
                                              (size_t)2 * HD * ggml_element_size(q_full),
                                              q_full->nb[1],
                                              (size_t)HD * ggml_element_size(q_full));
            gate = ggml_cont_2d(g.ctx, gate, (int64_t)NH * HD, T);
            ggml_tensor * k3 = ggml_reshape_3d(g.ctx, k, HD, NKV, T);
            k3 = ggml_rms_norm(g.ctx, k3, rms_eps);
            k3 = ggml_mul(g.ctx, k3, L.kn);
            if (il == 0) { ggml_set_name(q3, "t_l0_q_norm"); ggml_set_name(k3, "t_l0_k_norm"); }

            // interleaved MRoPE: 64 of 256 head dims, sections 11/11/10/0
            q3 = ggml_rope_multi(g.ctx, q3, pos_rope, nullptr, rope_dim, mrope_sections,
                                 GGML_ROPE_TYPE_IMROPE, n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k3 = ggml_rope_multi(g.ctx, k3, pos_rope, nullptr, rope_dim, mrope_sections,
                                 GGML_ROPE_TYPE_IMROPE, n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            if (il == 0) { ggml_set_name(q3, "t_l0_q_rope"); ggml_set_name(k3, "t_l0_k_rope"); }

            ggml_tensor * k2 = ggml_reshape_2d(g.ctx, k3, NKV * HD, T);
            ggml_tensor * v2 = ggml_reshape_2d(g.ctx, v, NKV * HD, T);
            ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.k[ki], k2, pos_slot));
            ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.v[ki], v2, pos_slot));

            ggml_tensor * kf = ggml_view_3d(g.ctx, kv.k[ki], HD, S, NKV,
                                            kv.k[ki]->nb[1], HD * sizeof(ggml_fp16_t), 0);
            ggml_tensor * vf = ggml_view_3d(g.ctx, kv.v[ki], HD, S, NKV,
                                            kv.v[ki]->nb[1], HD * sizeof(ggml_fp16_t), 0);
            ggml_tensor * attn = nullptr;
            if (logits_all && T > 1) {
                // speculative verify: run flash attention PER TOKEN (T=1
                // queries) instead of one fused T=2 call. The fused call
                // lands on the mma-f16 kernel whose f16 Q*K costs ~2% logits
                // vs the vector kernel the static decode path uses; per-token
                // calls with the same 512-block-padded S take the identical
                // vector-kernel path and keep verify bit-identical to plain
                // greedy decoding.
                for (int t = 0; t < T; t++) {
                    ggml_tensor * qt = ggml_view_3d(g.ctx, q3, HD, 1, NH,
                                                    q3->nb[1], q3->nb[1], (size_t)t * q3->nb[2]);
                    ggml_tensor * mt = ggml_view_2d(g.ctx, g.mask_in, S, 1,
                                                    g.mask_in->nb[1], (size_t)t * g.mask_in->nb[1]);
                    ggml_tensor * fa_t = ggml_flash_attn_ext(g.ctx, qt, kf, vf, mt,
                                                             1.0f / sqrtf((float)HD), 0.0f, 0.0f);
                    ggml_tensor * col = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, fa_t), (int64_t)NH * HD, 1);
                    attn = attn ? ggml_concat(g.ctx, attn, col, 1) : col;
                }
            } else {
                ggml_tensor * qf = ggml_view_3d(g.ctx, q3, HD, T, NH, q3->nb[2], q3->nb[1], 0);
                ggml_tensor * fa = ggml_flash_attn_ext(g.ctx, qf, kf, vf, g.mask_in,
                                                       1.0f / sqrtf((float)HD), 0.0f, 0.0f);
                attn = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, fa), NH * HD, T);
            }
            if (il == 0) ggml_set_name(attn, "t_l0_attn");
            attn = ggml_mul(g.ctx, attn, ggml_sigmoid(g.ctx, gate));    // output gating
            attn_out = ggml_mul_mat(g.ctx, L.o, attn);
        } else {
            const int si = ssm_idx++;
            // projections
            ggml_tensor * qkv = ggml_mul_mat(g.ctx, L.la_qkv, l);      // [kv_dim, T]
            ggml_tensor * z = ggml_mul_mat(g.ctx, L.la_gate, l);       // [d_inner, T]
            if (il == 0) { ggml_set_name(qkv, "t_l0_la_qkv"); ggml_set_name(z, "t_l0_la_z"); }

            ggml_tensor * beta = ggml_mul_mat(g.ctx, L.la_beta, l);    // [Hv, T]
            beta = ggml_sigmoid(g.ctx, beta);
            beta = ggml_reshape_4d(g.ctx, beta, 1, Hv, T, 1);
            ggml_tensor * alpha = ggml_mul_mat(g.ctx, L.la_alpha, l);  // [Hv, T]
            alpha = ggml_reshape_3d(g.ctx, alpha, Hv, T, 1);
            ggml_tensor * a_bias = ggml_add(g.ctx, alpha, L.la_dt);    // bcast [Hv] over T
            ggml_tensor * gk = ggml_mul(g.ctx, ggml_softplus(g.ctx, a_bias), L.la_a);
            gk = ggml_reshape_4d(g.ctx, gk, 1, Hv, T, 1);
            if (il == 0) { ggml_set_name(beta, "t_l0_la_beta"); ggml_set_name(gk, "t_l0_la_g"); }

            // causal conv: prepend the persistent state, roll it forward
            // (ggml_ssm_conv: sx 3d [k-1+T, conv_dim, 1], c [k, conv_dim],
            //  result [conv_dim, T, 1])
            ggml_tensor * qkv3 = ggml_reshape_3d(g.ctx, qkv, kv_dim, T, 1);
            ggml_tensor * qkv_t = ggml_transpose(g.ctx, qkv3);              // [T, kv_dim, 1]
            const bool spec = logits_all && T > 1;   // speculative verify graph
            ggml_tensor * conv_in_state = (state_slot == 1 ? conv_state_b : state_slot == 2 ? conv_state_c : conv_state)[si];
            ggml_tensor * conv_st = ggml_reshape_3d(g.ctx, conv_in_state, k_conv - 1, kv_dim, 1);
            ggml_tensor * conv_in = ggml_concat(g.ctx, conv_st, qkv_t, 0);  // [k-1+T, kv_dim, 1]
            ggml_tensor * conv_state_last =
                ggml_view_3d(g.ctx, conv_in, k_conv - 1, kv_dim, 1,
                             conv_in->nb[1], conv_in->nb[2],
                             (size_t)(conv_in->ne[0] - (k_conv - 1)) * conv_in->nb[0]);
            // state writes hang off the layer chain; without a dependency edge
            // into the main path, CUDA multi-stream scheduling can overlap the
            // copy with later nodes that REUSE the source buffer (observed as
            // ~1e-2-slot-state corruption from the first post-attention layer
            // on). Tie every state copy into the layer output via +0.
            ggml_tensor * state_dep = nullptr;
            auto tie_state = [&](ggml_tensor * cpy_out) {
                ggml_build_forward_expand(g.gf, cpy_out);
                ggml_tensor * s = ggml_scale(g.ctx, ggml_sum(g.ctx, cpy_out), 0.0f);
                state_dep = state_dep ? ggml_add(g.ctx, state_dep, s) : s;
            };
            // the rolled conv window ALWAYS goes to the MAIN conv_state (the
            // pre-fix code wrote it into `conv_st`, which on the slot-1
            // rejection path is a view of conv_state_b - so main never
            // advanced and the next round read a stale window)
            tie_state(ggml_cpy(g.ctx, conv_state_last,
                               ggml_reshape_3d(g.ctx, conv_state[si], k_conv - 1, kv_dim, 1)));
            if (spec) {
                // rollback windows: state after 1 token (slot 1) and after 0 (slot 2,
                // the untouched input state, needed when the whole draft is rejected)
                ggml_tensor * cs1 = ggml_view_3d(g.ctx, conv_in, k_conv - 1, kv_dim, 1,
                                                 conv_in->nb[1], conv_in->nb[2], 1 * conv_in->nb[0]);
                tie_state(ggml_cpy(g.ctx, cs1, ggml_reshape_3d(g.ctx, conv_state_b[si], k_conv - 1, kv_dim, 1)));
                tie_state(ggml_cpy(g.ctx, conv_st,
                                                         ggml_reshape_3d(g.ctx, conv_state_c[si], k_conv - 1, kv_dim, 1)));
            }
            ggml_tensor * conv = ggml_ssm_conv(g.ctx, conv_in, L.la_conv1d); // [kv_dim, T, 1]
            conv = ggml_silu(g.ctx, conv);
            if (il == 0) ggml_set_name(conv, "t_l0_la_conv");

            // split q/k/v heads (channels: q 2*group*Sk? no: q,k = group*Sk each, v = inner)
            const int qk_dim = ssm_n_group * Sk;
            auto head_view = [&](int block, int dim, int heads) {
                return ggml_view_3d(g.ctx, conv, dim, heads, T,
                                    (size_t)dim * ggml_element_size(conv),
                                    (size_t)kv_dim * ggml_element_size(conv),
                                    (size_t)block * qk_dim * ggml_element_size(conv));
            };
            ggml_tensor * q_conv = head_view(0, Sk, ssm_n_group);
            ggml_tensor * k_conv2 = head_view(1, Sk, ssm_n_group);
            ggml_tensor * v_conv = head_view(2, Sk, Hv);               // v block offset 2*qk_dim
            // l2 norm over the head dim (FLA kernel semantics)
            auto l2n = [&](ggml_tensor * t) {
                const float n = (float)t->ne[0];
                return ggml_scale(g.ctx, ggml_rms_norm(g.ctx, t, 1e-6f / n), 1.0f / sqrtf(n));
            };
            q_conv = l2n(q_conv);
            k_conv2 = l2n(k_conv2);
            q_conv = ggml_reshape_4d(g.ctx, q_conv, Sk, ssm_n_group, T, 1);
            k_conv2 = ggml_reshape_4d(g.ctx, k_conv2, Sk, ssm_n_group, T, 1);
            v_conv = ggml_cont(g.ctx, v_conv);      // head_view leaves a strided view
            v_conv = ggml_reshape_4d(g.ctx, v_conv, Sk, Hv, T, 1);

            // fused gated delta net; the result packs scores + K=1 state snapshot
            ggml_tensor * ssm_in = (state_slot == 1 ? ssm_state_b : state_slot == 2 ? ssm_state_c : ssm_state)[si];
            ggml_tensor * s0 = ggml_reshape_4d(g.ctx, ssm_in, Sk, Sk, Hv, 1);
            ggml_tensor * res = ggml_gated_delta_net(g.ctx, q_conv, k_conv2, v_conv, gk, beta,
                                                     s0, /*K=*/(spec && getenv("QWEN35_SPEC_K1") == nullptr) ? 2 : 1);
            ggml_tensor * out = ggml_view_4d(g.ctx, res, Sk, Hv, T, 1,
                                             ggml_row_size(res->type, Sk),
                                             ggml_row_size(res->type, Sk * Hv),
                                             ggml_row_size(res->type, Sk * Hv * T), 0);
            ggml_tensor * new_state = ggml_view_3d(g.ctx, res, Sk, Sk, Hv,
                                                   ggml_row_size(res->type, Sk),
                                                   ggml_row_size(res->type, Sk * Sk),
                                                   ggml_row_size(res->type, Sk * Hv * T));
            tie_state(ggml_cpy(g.ctx, new_state, ssm_state[si]));
            if (spec) {
                // snapshot layout: [scores T cols][slot0 = final][slot1 = one token back]
                ggml_tensor * snap1 = ggml_view_3d(g.ctx, res, Sk, Sk, Hv,
                        Sk * ggml_element_size(res), Sk * Sk * ggml_element_size(res),
                        ggml_row_size(res->type, Sk * Hv * (T + Sk)));
                tie_state(ggml_cpy(g.ctx, snap1, ssm_state_b[si]));
                tie_state(ggml_cpy(g.ctx, s0,
                                                         ggml_reshape_3d(g.ctx, ssm_state_c[si], Sk, Sk, Hv)));
            }
            if (il == 0) ggml_set_name(out, "t_l0_la_core");

            // gated RMSNorm over the head dim, gated by silu(z)
            ggml_tensor * zn = ggml_reshape_3d(g.ctx, z, Sk, Hv, T);
            ggml_tensor * on = ggml_rms_norm(g.ctx, out, rms_eps);
            on = ggml_mul(g.ctx, on, L.la_norm);
            on = ggml_mul(g.ctx, on, ggml_silu(g.ctx, zn));
            ggml_tensor * flat = ggml_reshape_2d(g.ctx, on, ssm_d_inner, T);
            attn_out = ggml_mul_mat(g.ctx, L.la_out, flat);
            if (state_dep)   // serialize the state copies into the main chain (+0)
                attn_out = ggml_add(g.ctx, attn_out, state_dep);
        }

        x = ggml_add(g.ctx, x, attn_out);
        ggml_tensor * ffn_inp = x;
        if (il == 0) ggml_set_name(x, "t_l0_resid");

        // attn_post_norm doubles as the FFN norm (no separate ffn_norm)
        ggml_tensor * m = ggml_rms_norm(g.ctx, x, rms_eps);
        m = ggml_mul(g.ctx, m, L.post_norm);
        ggml_tensor * gate = ggml_mul_mat(g.ctx, L.gate, m);
        ggml_tensor * up = ggml_mul_mat(g.ctx, L.up, m);
        m = ggml_swiglu_split(g.ctx, gate, up);
        if (il == 0) ggml_set_name(m, "t_l0_swiglu");
        m = ggml_mul_mat(g.ctx, L.down, m);
        x = ggml_add(g.ctx, ffn_inp, m);
        if (il == 0) ggml_set_name(x, "t_l0_out");
    }

    x = ggml_rms_norm(g.ctx, x, rms_eps);
    x = ggml_mul(g.ctx, x, out_norm);
    ggml_set_name(x, "t_final_norm");

    if (logits_all) {
        g.logits = ggml_mul_mat(g.ctx, tok_embd, x);       // [vocab, T] speculative verify
    } else if (T > 1) {
        ggml_tensor * last = ggml_view_2d(g.ctx, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
        g.logits = ggml_mul_mat(g.ctx, tok_embd, last);    // tied lm head
    } else {
        g.logits = ggml_mul_mat(g.ctx, tok_embd, x);
    }
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);
    if (!dump_dir.empty()) {
        static const char * const dump_names[] = {
            "emb", "t_l0_q", "t_l0_q_norm", "t_l0_q_rope", "t_l0_k_rope", "t_l0_attn",
            "t_l0_la_qkv", "t_l0_la_z", "t_l0_la_beta", "t_l0_la_g", "t_l0_la_conv",
            "t_l0_la_core", "t_l0_swiglu", "t_l0_out", "t_final_norm", nullptr,
        };
        vqa::dump::mark_sinks(g.ctx, g.gf, dump_names);
    }
    return true;
}

bool Qwen3_5::run_text(TextGraph & g, const std::vector<int> & toks,
                      const std::vector<float> * img_feats, const std::vector<int> * img_pos,
                      const std::vector<int> & mpos, std::vector<float> & logits_out,
                      std::vector<float> * h_last) {
    const int T = (int)toks.size();
    const int S = n_past + T;
    if (S > n_ctx) { fprintf(stderr, "context overflow (%d > %d)\n", S, n_ctx); return false; }
    const int n_img = (img_feats && img_pos) ? (int)(img_feats->size() / n_embd) : 0;
    static const bool prof = getenv("QWEN35_PROF") != nullptr;
    static double tb = 0, ta = 0, tc = 0; static int ns = 0;
    auto tick = []() { return std::chrono::steady_clock::now(); };
    auto tp0 = tick();

    auto fill_inputs = [&](ggml_tensor * tok_t, ggml_tensor * mpos_t, ggml_tensor * mask_t) {
        ggml_backend_tensor_set(tok_t, toks.data(), 0, sizeof(int) * T);
        ggml_backend_tensor_set(mpos_t, mpos.data(), 0, sizeof(int) * 4 * T);
        vqa::set_causal_mask(mask_t, n_past, T);
    };

    // static single-token decode graph (CUDA graph capture friendly)
    const bool use_static = (T == 1) && n_img == 0 && !ggml_backend_is_cpu(backend);
    if (use_static) {
        const int blk = std::min(vqa::block_for(S), n_ctx);
        if (!dec_gf || dec_blk != blk) {
            if (dec_ctx) ggml_free(dec_ctx);
            struct ggml_init_params ip = { 96ull * 1024 * 1024, NULL, true };
            dec_ctx = ggml_init(ip);
            TextGraph dg;
            dg.ctx = dec_ctx;
            if (!build_text_graph(dg, 1, blk, 0)) { ggml_free(dec_ctx); dec_ctx = nullptr; dec_gf = nullptr; return false; }
            dec_gf = dg.gf;
            if (!ggml_gallocr_alloc_graph(allocr, dec_gf)) {
                fprintf(stderr, "static decode graph alloc failed\n");
                ggml_free(dec_ctx); dec_ctx = nullptr; dec_gf = nullptr;
                return false;
            }
            dec_blk = blk;
            dec_tok = ggml_graph_get_tensor(dec_gf, "tok_in");
            dec_mpos = ggml_graph_get_tensor(dec_gf, "mpos_in");
            dec_mask = ggml_graph_get_tensor(dec_gf, "mask_in");
        }
        fill_inputs(dec_tok, dec_mpos, dec_mask);
        auto tp2 = tick();
        if (ggml_backend_graph_compute(backend, dec_gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "text compute failed\n");
            return false;
        }
        if (prof) {
            ns++;
            tb += std::chrono::duration<double,std::milli>(tp2-tp0).count();
            tc += std::chrono::duration<double,std::milli>(tick()-tp2).count();
            if (ns % 50 == 0)
                fprintf(stderr, "[prof] steps=%d input %.2fms compute %.2fms (avg, static)\n", ns, tb/ns, tc/ns);
        }
        logits_out.resize(n_vocab);
        ggml_backend_tensor_get(ggml_graph_get_tensor(dec_gf, "logits"), logits_out.data(), 0, sizeof(float) * n_vocab);
        n_past = S;
        return true;
    }

    if (!build_text_graph(g, T, S, n_img)) return false;
    auto tp1 = tick();
    if (!ggml_gallocr_alloc_graph(allocr, g.gf)) {
        fprintf(stderr, "text graph allocation failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }
    ggml_backend_tensor_set(g.tok_in, toks.data(), 0, sizeof(int) * T);
    ggml_backend_tensor_set(g.mpos_in, mpos.data(), 0, sizeof(int) * 4 * T);
    if (n_img > 0) {
        ggml_backend_tensor_set(g.img_in, img_feats->data(), 0, sizeof(float) * img_feats->size());
        ggml_backend_tensor_set(g.img_pos_in, img_pos->data(), 0, sizeof(int) * n_img);
    }
    vqa::set_causal_mask(g.mask_in, n_past, T);
    auto tp2 = tick();
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "text compute failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }
    if (prof) {
        ns++;
        tb += std::chrono::duration<double,std::milli>(tp1-tp0).count();
        ta += std::chrono::duration<double,std::milli>(tp2-tp1).count();
        tc += std::chrono::duration<double,std::milli>(tick()-tp2).count();
        if (ns % 50 == 0)
            fprintf(stderr, "[prof] steps=%d build %.2fms alloc+input %.2fms compute %.2fms (avg)\n",
                    ns, tb/ns, ta/ns, tc/ns);
    }
    dump(g.gf, "emb");
    dump(g.gf, "t_l0_q_rope");
    dump(g.gf, "t_l0_out");
    dump(g.gf, "t_final_norm");
    dump(g.gf, "logits");

    logits_out.resize(n_vocab);
    ggml_backend_tensor_get(g.logits, logits_out.data(), 0, sizeof(float) * n_vocab);
    if (h_last) {
        ggml_tensor * hn = ggml_graph_get_tensor(g.gf, "t_final_norm");
        h_last->resize(n_embd);
        ggml_backend_tensor_get(hn, h_last->data(), (size_t)(T - 1) * hn->nb[1], sizeof(float) * n_embd);
    }
    n_past = S;
    ggml_free(g.ctx);
    g.ctx = nullptr;
    return true;
}

std::string Qwen3_5::render_prompt(int n_img_tokens, const std::string & user_text, bool video) const {
    // chat_template.jinja: single vision block for the whole video, plain text
    // prompt after it (Qwen3_5's training format; timestamps live in the mrope
    // t-axis, not in the prompt text)
    std::string s = "<|im_start|>user\n";
    if (n_img_tokens > 0) {
        const char * pad = video ? "<|video_pad|>" : "<|image_pad|>";
        s += "<|vision_start|>";
        for (int i = 0; i < n_img_tokens; i++) s += pad;
        s += "<|vision_end|>";
    }
    s += user_text;
    // official chat template: the assistant prefix carries the ms-swift
    // non-thinking artifact, so generations start clean
    s += "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return s;
}

int Qwen3_5::sample(const std::vector<float> & logits, const vqa::GenParams & gp) {
    return vqa::sample_logits(logits, gp, rng_state);
}

// strip <think>...</think> blocks / bare <think> prefixes (training artifact)
static std::string strip_thinking(const std::string & s) {
    std::string out = s;
    for (size_t p; (p = out.find("<think>")) != std::string::npos;) {
        size_t close = out.find("</think>", p);
        if (close == std::string::npos) { out.erase(p, 7); break; }
        out.erase(p, close + 8 - p);
    }
    for (size_t p; (p = out.find("</think>")) != std::string::npos;) out.erase(p, 8);
    // trim leading whitespace/newlines left behind
    size_t b = out.find_first_not_of(" \t\r\n");
    return b == std::string::npos ? std::string() : out.substr(b);
}

bool Qwen3_5::ask(const vqa::MediaInput & media, const std::string & prompt,
                 const std::string & mode, const vqa::GenParams & gp, std::string & out_text) {
    // the two trained skills (modeling_marlin.py canonical prompts - DO NOT EDIT):
    // "caption" ignores the user text and emits the Scene:/Events: format,
    // "find" treats the user text as the event to temporally ground
    static const char * CAPTION_PROMPT =
        "Provide a spatial description of this clip followed by time-ranged events.\n"
        "For each event, give the time range as <start - end> and a short description.";
    std::string prompt_eff = prompt;
    if (mode == "caption" && marlin_skills) prompt_eff = CAPTION_PROMPT;
    else if (mode == "find" && marlin_skills)
        prompt_eff = "Identify the timestamps during which \"" + prompt + "\" takes place. "
                     "Output the time range as \"From <start> to <end>.\" (numbers in seconds).";
    const std::string & user_text = prompt_eff;
    std::vector<float> feats;
    int n_img = 0;
    int grid[3] = {0, 0, 0};     // (temporal steps, merged h, merged w)
    const auto tv0 = std::chrono::steady_clock::now();
    const bool video_kind = media.kind == vqa::MediaKind::Video;
    {
        const int text_overhead = (int)tok_.encode(render_prompt(1, user_text, video_kind)).size();
        const int budget = n_ctx - gp.max_tokens - text_overhead - 16;
        if (budget <= 0) {
            fprintf(stderr, "n_ctx %d too small: prompt %d tokens + max_new %d leaves nothing for the image\n",
                    n_ctx, text_overhead, gp.max_tokens);
            return false;
        }
        // media cache keyed on frame bytes (same picture, follow-up question)
        static struct {
            uint64_t key = 0;
            std::vector<float> f;
            int n = 0;
            int g[3] = {0, 0, 0};
        } vcache;
        uint64_t key = 1469598103934665603ull;
        for (const auto & fr : media.frames) {
            key ^= fr.rgb.size(); key *= 1099511628211ull;
            for (size_t k = 0; k < fr.rgb.size(); k++) { key ^= fr.rgb[k]; key *= 1099511628211ull; }
        }
        key ^= (uint64_t)media.width << 32 ^ (uint64_t)media.height;
        if (vcache.key == key && vcache.n > 0) {
            feats = vcache.f; n_img = vcache.n; memcpy(grid, vcache.g, sizeof(grid));
            fprintf(stderr, "vision: cache hit (%d tokens)\n", n_img);
        } else if (encode_vision(media, feats, &n_img, grid, budget)) {
            vcache.key = key; vcache.f = feats; vcache.n = n_img; memcpy(vcache.g, grid, sizeof(grid));
        } else {
            return false;
        }
    }
    fprintf(stderr, "[perf] vision encode: %.0f ms (%d tokens)\n",
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tv0).count(), n_img);

    const std::string text = render_prompt(n_img, user_text, video_kind);
    const std::vector<int> ids = tok_.encode(text);
    const int pad_id = video_kind ? id_video_pad : id_image_pad;
    std::vector<int> img_pos;
    for (int i = 0; i < (int)ids.size(); i++) if (ids[i] == pad_id) img_pos.push_back(i);
    if ((int)img_pos.size() != n_img) {
        fprintf(stderr, "placeholder mismatch: %d pad tokens vs %d vision tokens\n",
                (int)img_pos.size(), n_img);
        if (n_img > 0) return false;
    }

    // mrope position ids (get_rope_index): text runs share (p,p,p); the vision
    // run uses (t, h, w) over the merged grid; after a vision segment the text
    // counter advances by max(h, w)
    std::vector<int> mpos;
    mpos.reserve((size_t)4 * ids.size());
    {
        std::vector<int> axis[3];
        int cur = 0;
        for (size_t i = 0; i < ids.size();) {
            if ((int)i == (img_pos.empty() ? -1 : img_pos[0]) && !img_pos.empty()) {
                // whole vision run
                const int gt = grid[0], mh = grid[1], mw = grid[2];
                for (int t = 0; t < gt; t++)
                    for (int h = 0; h < mh; h++)
                        for (int w = 0; w < mw; w++) {
                            axis[0].push_back(cur + t);
                            axis[1].push_back(cur + h);
                            axis[2].push_back(cur + w);
                        }
                cur += std::max(mh, mw);
                i += img_pos.size();
                img_pos.clear();
                continue;
            }
            axis[0].push_back(cur); axis[1].push_back(cur); axis[2].push_back(cur);
            cur++;
            i++;
        }
        mpos.insert(mpos.end(), axis[0].begin(), axis[0].end());
        mpos.insert(mpos.end(), axis[1].begin(), axis[1].end());
        mpos.insert(mpos.end(), axis[2].begin(), axis[2].end());
        for (int i = 0; i < (int)ids.size(); i++) mpos.push_back(i);    // absolute KV slots
        last_mpos = {axis[0].back(), axis[1].back(), axis[2].back()};
        const int mx = std::max({axis[0].back(), axis[1].back(), axis[2].back()});
        mrope_delta = (mx + 1) - (int)ids.size();
    }

    fprintf(stderr, "prompt: %d tokens (%d image), max_new %d\n", (int)ids.size(), n_img, gp.max_tokens);
    int max_new = gp.max_tokens;
    if ((int)ids.size() + max_new > n_ctx) {
        max_new = n_ctx - (int)ids.size();
        fprintf(stderr, "warning: prompt %d + max_new %d > n_ctx %d, clamping max_new to %d\n",
                (int)ids.size(), gp.max_tokens, n_ctx, max_new);
        if (max_new <= 0) return false;
    }

    // fresh state per question
    n_past = 0;
    if (sbuf) ggml_backend_buffer_clear(sbuf, 0);
    kv.clear();
    auto t0 = std::chrono::steady_clock::now();
    TextGraph g;
    std::vector<float> logits;
    std::vector<float> h_seed;
    if (!run_text(g, ids, n_img > 0 ? &feats : nullptr, n_img > 0 ? &img_pos : nullptr, mpos, logits,
                  has_mtp ? &h_seed : nullptr))
        return false;
    auto t1 = std::chrono::steady_clock::now();

    std::string raw;
    int next = sample(logits, gp);
    int n_dec = 0;
    // opt-in while the accept-path snapshot semantics are being fixed (dev-notes)
    const char * spec_env = getenv("QWEN35_SPEC");
    static const bool spec_on = spec_env != nullptr && strcmp(spec_env, "0") != 0;
    if (has_mtp && spec_on && !h_seed.empty()) {
        if (!run_speculative(next, ids.back(), logits, h_seed.data(), max_new, gp, raw, n_dec))
            return false;
    } else {
    std::vector<int> step_pos(4), step_tok(1);
    for (int step = 0; step < max_new; step++) {
        if (next == id_eos || next == id_endoftext) break;
        const std::string piece = tok_.decode({next});
        raw += piece;
        if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
        n_dec++;
        if (step + 1 >= max_new) break;
        // mrope axis of the NEW token = previous token's axis + 1 (matches the
        // prefill convention axis == KV column; the old `+ mrope_delta` froze
        // every generated token at the last prompt token's axis)
        step_pos[0] = last_mpos[0] + 1;
        step_pos[1] = last_mpos[1] + 1;
        step_pos[2] = last_mpos[2] + 1;
        step_pos[3] = n_past;               // KV slot
        step_tok[0] = next;
        if (!run_text(g, step_tok, nullptr, nullptr, step_pos, logits)) break;
        last_mpos = {step_pos[0], step_pos[1], step_pos[2]};
        next = sample(logits, gp);
    }
    }   // non-speculative decode loop
    auto t2 = std::chrono::steady_clock::now();
    const double ms_prefill = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_decode = std::chrono::duration<double, std::milli>(t2 - t1).count();
    fprintf(stderr, "[perf] prefill %d tok in %.0f ms (%.1f tok/s) | decode %d tok in %.0f ms (%.2f tok/s)\n",
            (int)ids.size(), ms_prefill, ms_prefill > 0 ? ids.size() * 1000.0 / ms_prefill : 0.0,
            n_dec, ms_decode, ms_decode > 0 ? n_dec * 1000.0 / ms_decode : 0.0);
    out_text = strip_thinking(raw);
    return true;
}


#include "qwen3_5_mtp.inc"

}  // namespace qwen3_5

namespace vqa {
Model * create_qwen3_5() { return new qwen3_5::Qwen3_5(); }
}  // namespace vqa
