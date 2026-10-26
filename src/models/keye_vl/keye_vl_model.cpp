// keye_vl_model.cpp - Keye-VL-2.0-30B-A3B backend (Kwai Keye VL 2.0).
//
// Ported from Kwai-Keye's official llama.cpp fork (branch
// keye-vl-v2-30b-release-video): a Qwen3-style MoE text trunk (48 layers,
// 128 routed experts / 8 active, softmax gating with top-k renormalization)
// with mRoPE [16,24,24] theta 1e7 + per-head q/k RMSNorm, plus a DSA
// "lightning indexer" per layer that selects the top-k KV positions each
// token may attend to (sparse attention), and a SigLIP-so400m vision tower
// (1152/16h/27L, patch 14, 2D mRoPE inside the ViT, learned pos-emb
// bilinearly interpolated to the actual grid) with a 2x2 spatial-merge +
// 2-layer GELU projector into text space.
//
// Layout notes (all cross-checked against the fork's keye.cpp / keyevl2.cpp):
//   * indexer: q = W_qb*h [1024] -> [64,16,T], per-head RMSNorm, rope with
//     sections scaled by 64/128 = 0.5 ([8,12,12]); k = W_k*h [64] (MQA),
//     LayerNorm(w+b), same rope; both multiplied by a fixed 64x64
//     Walsh-Hadamard/sqrt(64) matrix, then k is cached in a per-layer
//     [64, n_ctx] F16 "lid" cache. Score = sum_h ReLU(k.q) * (W_proj*h/8)
//     -> [S,T], + causal mask, top_k (min(S, 2048)) indices.
//   * final attention mask = causal AND selected: fill(-inf) tensor with
//     zeros written at the top-k indices (ggml_set_rows, same shapes as the
//     fork), added to the plain causal mask.
//   * MoE: softmax gating over all 128 experts, top-8 logits, softmax over
//     the selected logits (= norm_topk_prob renormalization), two
//     mul_mat_id calls (up and gate are separate expert tensors).
//   * vision: host-side raster patchify + one matmul (engine convention,
//     qwen2vl/siglip towers); video = the OFFICIAL default path: per-frame
//     independent tower passes ("<|frame|><hh:mm:ss>" markers between
//     frames). Verified against the official wrapper
//     (modeling_keye_topk_mask_30ba3b.py): the official video call passes
//     interpolate_pos_encoding=True (spatial position table, NOT the
//     packing_position_embedding - that table is dead code on this path)
//     and per-frame cu_seqlens, i.e. no cross-frame attention. mRoPE
//     positions follow the official get_rope_index (bitwise-parity,
//     2026-10-25). The opt-in slow/fast dual-stream frame_types feature
//     (fast frames wrapped in <|fast_start|>/<|fast_end|> at lower
//     resolution) is not implemented - the official default is all-slow.
//
// The 19 GB Q4_K_M checkpoint does not fit a 12 GB GPU, so this backend
// defaults to CPU; set KEYE_GPU=1 to try the CUDA backend anyway.
#include "core/vqa.hpp"

#include "core/engine/attention.hpp"
#include "core/engine/backend.hpp"
#include "core/engine/decode_graph.hpp"
#include "core/blocks/vit_block.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/gguf_util.hpp"
#include "core/runtime/load_util.hpp"
#include "core/media/resample.hpp"
#include "core/engine/sampler.hpp"
#include "core/engine/weight_load.hpp"
#include "core/engine/sched.hpp"
#include "core/engine/expert_cache.hpp"
#include "core/engine/moe_compute.hpp"
#include "core/engine/moe_decode_driver.hpp"
#include "core/engine/step_graph.hpp"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

// step_graph.cpp: the cgraph under construction during StepGraph::build(),
// used by the graph-A builder in keye_expert_decode.inc (declared at GLOBAL
// scope - opening `namespace vqa` inside keye_vl would shadow ::vqa)
namespace vqa {
ggml_cgraph * step_graph_builder_gf();
namespace moe {
// residual-splitting form of build_expert_ffn (same ops; out = res + Σ ...,
// the frozen header's version passes res = x)
ggml_tensor * build_expert_ffn_res(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * x,
                                   ggml_tensor * res,
                                   const std::vector<ggml_tensor *> & gate,
                                   const std::vector<ggml_tensor *> & up,
                                   const std::vector<ggml_tensor *> & down,
                                   ggml_tensor * w, ggml_unary_op act);
// v3 slot-table form (ExpertCache table layout): mul_mat_id over the cache's
// 3D slot tables; residual-splitting like build_expert_ffn_res above
ggml_tensor * build_expert_ffn_table_res(ggml_context * ctx, ggml_cgraph * gf,
                                         ggml_tensor * x, ggml_tensor * res,
                                         ggml_tensor * gate_table, ggml_tensor * up_table,
                                         ggml_tensor * down_table,
                                         ggml_tensor * gate_up_ids, ggml_tensor * down_ids,
                                         ggml_tensor * w, ggml_unary_op act);
}
}

namespace keye_vl {

struct TxtLayer {
    ggml_tensor *attn_norm = nullptr;
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    ggml_tensor *qn = nullptr, *kn = nullptr;              // per-head [128]
    // DSA indexer
    ggml_tensor *ix_q_b = nullptr;    // [2048, 1024] q projection
    ggml_tensor *ix_q_norm = nullptr; // [64]
    ggml_tensor *ix_k = nullptr;      // [2048, 64]
    ggml_tensor *ix_k_norm = nullptr, *ix_k_norm_b = nullptr; // [64]
    ggml_tensor *ix_proj = nullptr;   // [2048, 16] head weights
    // MoE
    ggml_tensor *ffn_norm = nullptr;
    ggml_tensor *gate_inp = nullptr;                        // [2048, 128]
    ggml_tensor *up_exps = nullptr, *gate_exps = nullptr, *down_exps = nullptr;
};

struct TextGraph {
    ggml_context *ctx = nullptr;
    ggml_cgraph *gf = nullptr;
    ggml_tensor *tok_in = nullptr, *pos_in = nullptr, *slot_in = nullptr, *mask_in = nullptr;
    ggml_tensor *img_in = nullptr, *img_pos_in = nullptr;
    ggml_tensor *logits = nullptr;
};

struct VisLayer {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr, *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *wq = nullptr, *bq = nullptr, *wk = nullptr, *bk = nullptr;
    ggml_tensor *wv = nullptr, *bv = nullptr, *wo = nullptr, *bo = nullptr;
    ggml_tensor *fc0_w = nullptr, *fc0_b = nullptr, *fc1_w = nullptr, *fc1_b = nullptr;
};

class KeyeVL2 : public vqa::Model {
public:
    ~KeyeVL2() override;
    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "keye-vl2"; }
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp, std::string & out_text) override;

    int n_ctx = 8192;
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

private:
    bool init_backend();
    bool load_text_weights(const char * path);
    bool load_vision_weights(const char * path);
    bool alloc_kv();

    bool encode_frame(const vqa::Frame & frame, int w, int h, std::vector<float> & feats,
                      int * n_tokens, int * gh, int * gw);
    bool build_vision_graph(int ph, int pw);
    bool run_vision(const std::vector<float> & patches, std::vector<float> & feats);

    bool build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all);
    bool run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<int> & pos4,
                  const std::vector<float> * img_feats, const std::vector<int> * img_pos,
                  std::vector<float> & logits, int slot_base = 0,
                  const std::vector<float> * mask = nullptr);
    void dump(const ggml_cgraph * gf, const char * name);

    // KEYE_EXPERTCACHE=1: expert-granularity GPU-resident dynamic decode
    // (see keye_expert_decode.inc). ec_enabled is the request, ec_ok the
    // successfully-initialized state; decode falls back to the existing path
    // whenever ec_ok is false.
    bool ec_enabled = false, ec_ok = false;
    // one cache per quant-layout group (the GGUF mixes Q4_K/Q6_K across layers)
    std::vector<vqa::ExpertCache> ec_caches;
    std::vector<int> ec_group;                 // layer -> group index
    int ec_n_groups = 0;
    vqa::moe_decode::State ec_st;              // driver-owned decode state
    vqa::moe_decode::Hooks ec_hooks;           // filled once in ec_init
    TextGraph * ec_xchk_rg = nullptr;          // KEYE_XCHK reference graph
    int ec_ref_mode = 0;                       // KEYE_EC_REF mode
    std::vector<float> ec_mask;                // per-step causal mask cache
    std::vector<float> ec_lc_m, ec_lc_xmid, ec_lc_w8;   // layer IO staging for KEYE_EC_REF
    std::vector<int> ec_lc_sel;
    int ec_mask_n = -1;
    // v2: per-layer FIXED graph-A holders (StepGraph, built once, re-fed every
    // step - S is the block-aligned n_ctx so shapes never change) + async
    // prefetch bookkeeping (prev step's selections per layer).
    struct EcGraphA {
        ggml_tensor *tok_in = nullptr, *hid_in = nullptr, *pos_in = nullptr,
                    *slot_in = nullptr, *mask_in = nullptr, *had = nullptr;
        ggml_tensor *w8_o = nullptr, *sel_o = nullptr, *m_o = nullptr, *x_o = nullptr;
    };
    struct EcGaUd { KeyeVL2 * m; int il; EcGraphA * io; };
    std::vector<EcGraphA> ec_ga_io;            // model leaves per layer
    bool ec_init();
    bool ec_build_graphs();
    static ggml_tensor * ec_ga_builder(ggml_context * c, void * ud);    static bool ec_build_a(void * m, int il, vqa::StepGraph & sg,
                           vqa::moe_decode::AIO & io);
    static bool ec_feed_a(void * m, int il, vqa::moe_decode::AIO & io,
                          vqa::StepGraph & sg, const void * step_ctx,
                          int cur_tok, int n_past, const float * hid);
    static void ec_post_a(void * m, int il, const float * xmid, const float * mn,
                          const int * sel, const float * w8);
    static bool ec_attn_only(void * m, int il);
    static void ec_trace(void * m, int il, int phase);
    static bool ec_layer_check(void * m, int il, float * hid_io);
    bool ec_decode_step(int cur_tok, int pos, int n_past, std::vector<float> & logits);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    // text hparams (defaults = the released 30B-A3B checkpoint)
    int n_layer = 48, n_embd = 2048, n_head = 32, n_kv_head = 4, head_dim = 128;
    int n_vocab = 151936, n_expert = 128, n_expert_used = 8, n_ff_exp = 768;
    float rms_eps = 1e-6f, rope_theta = 1e7f;
    int mrope_section[3] = { 16, 24, 24 };
    int ix_head = 16, ix_head_dim = 64, ix_top_k = 2048, ix_q_dim = 1024;

    // vision hparams (SigLIP-so400m lineage)
    int v_layer = 27, v_embd = 1152, v_head = 16, v_hd = 72, v_ff = 4304;
    int v_patch = 14, v_merge = 2, v_out = 2048, v_pos_side = 27;
    float v_eps = 1e-6f, v_rope_theta = 10000.0f;
    float v_mean = 0.5f, v_std = 0.5f;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;  // split mode: computes the expert mul_mat_ids
    bool split_mode = false;               // Strata-style: dense on GPU, experts in host mmap
    ggml_backend_buffer_t host_wbuf = nullptr;   // host weight buffer (mmap wrap, split mode)
    vqa::SplitSched split_sched;
    ggml_context * wctx = nullptr;          // text weights
    ggml_backend_buffer_t wbuf = nullptr;
    vqa::MMap mm;
    vqa::GraphSlot<TextGraph> dec_slot;     // reused decode graph (decode_graph.hpp)
    ggml_context * vctx = nullptr;          // vision weights (mmproj)
    ggml_backend_buffer_t vbuf = nullptr;
    vqa::MMap vmm;
    vqa::AlignedBuf host_f32;
    ggml_gallocr_t allocr = nullptr;

    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr, *out_w = nullptr;
    std::vector<TxtLayer> tl;
    std::vector<VisLayer> vl;
    ggml_tensor *v_patch_w = nullptr, *v_patch_b = nullptr;
    ggml_tensor *v_pos_embd = nullptr;      // [1152, 729] source, interpolated on host
    ggml_tensor *v_post_ln_w = nullptr, *v_post_ln_b = nullptr;
    ggml_tensor *mm_norm_w = nullptr, *mm_norm_b = nullptr;
    ggml_tensor *mm0_w = nullptr, *mm0_b = nullptr, *mm2_w = nullptr, *mm2_b = nullptr;
    std::vector<float> hadamard64;          // [64*64] Walsh-Hadamard / 8

    vqa::KvCache kv;
    ggml_context * lid_ctx = nullptr;       // per-layer indexer K [64, n_ctx]
    ggml_backend_buffer_t lid_buf = nullptr;
    std::vector<ggml_tensor *> lid_k;

    // per-(ph,pw) vision graph cache (rebuilt on size change)
    ggml_context * vg_ctx = nullptr;
    ggml_cgraph * vg = nullptr;
    int vg_ph = 0, vg_pw = 0;
    ggml_tensor *vg_in = nullptr, *vg_pos_in = nullptr, *vg_rope_pos = nullptr, *vg_out = nullptr;
    vqa::vit::VitHookEnv vit_env;

    int n_past = 0;
    int next_pos = 0;

    int id_image_pad = 151655, id_vision_start = 151652, id_vision_end = 151653;
    int id_frame = 151677, id_im_start = 151644, id_im_end = 151645, id_eos = 151645;
    bool no_dsa = false;              // KEYE_NO_DSA=1: plain causal attention

    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

// ---------------------------------------------------------------- lifecycle

KeyeVL2::~KeyeVL2() {
    dec_slot.reset();
    kv.free();
    split_sched.free_sched();
    if (host_wbuf) ggml_backend_buffer_free(host_wbuf);
    if (lid_buf) ggml_backend_buffer_free(lid_buf);
    if (lid_ctx) ggml_free(lid_ctx);
    if (allocr) ggml_gallocr_free(allocr);
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (vbuf) ggml_backend_buffer_free(vbuf);
    if (vctx) ggml_free(vctx);
    if (vg_ctx) ggml_free(vg_ctx);
    if (backend) ggml_backend_free(backend);
    if (backend_cpu) ggml_backend_free(backend_cpu);
}

bool KeyeVL2::init_backend() {
    // KEYE_CPU=1 forces the pure CPU path. Otherwise --gpu (the CLI default on
    // CUDA builds) tries the GPU: the 19 GB checkpoint does not fit a 12 GB
    // card whole, so by default the expert tensors stay in the host mmap and
    // only the dense weights go up (Strata-style split, see sched.hpp);
    // KEYE_GPU=1 forces the all-on-GPU placement.
    if (getenv("KEYE_CPU")) use_gpu = false;
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!ggml_backend_is_cpu(backend)) {
        // split partner: a CPU backend for the expert mul_mat_ids
        backend_cpu = vqa::init_backend(false, n_threads);
        if (!backend_cpu) return false;
    }
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

bool KeyeVL2::load_text_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer   = vqa::gguf::u32(gu, "keye-vl2.block_count", n_layer);
    n_embd    = vqa::gguf::u32(gu, "keye-vl2.embedding_length", n_embd);
    n_head    = vqa::gguf::u32(gu, "keye-vl2.attention.head_count", n_head);
    n_kv_head = vqa::gguf::u32(gu, "keye-vl2.attention.head_count_kv", n_kv_head);
    head_dim  = vqa::gguf::u32(gu, "keye-vl2.attention.key_length", head_dim);
    n_expert  = vqa::gguf::u32(gu, "keye-vl2.expert_count", n_expert);
    n_expert_used = vqa::gguf::u32(gu, "keye-vl2.expert_used_count", n_expert_used);
    n_ff_exp  = vqa::gguf::u32(gu, "keye-vl2.expert_feed_forward_length", n_ff_exp);
    rms_eps   = vqa::gguf::f32(gu, "keye-vl2.attention.layer_norm_rms_epsilon", rms_eps);
    rope_theta = vqa::gguf::f32(gu, "keye-vl2.rope.freq_base", rope_theta);
    {
        const std::vector<int> sec = vqa::gguf::arr_u32(gu, "keye-vl2.rope.dimension_sections",
                                                        { 16, 24, 24 });
        for (int i = 0; i < 3 && i < (int)sec.size(); i++) mrope_section[i] = sec[(size_t)i];
    }
    ix_head     = vqa::gguf::u32(gu, "keye-vl2.attention.indexer.head_count", ix_head);
    ix_head_dim = vqa::gguf::u32(gu, "keye-vl2.attention.indexer.key_length", ix_head_dim);
    ix_top_k    = vqa::gguf::u32(gu, "keye-vl2.attention.indexer.top_k", ix_top_k);
    ix_q_dim    = vqa::gguf::u32(gu, "keye-vl2.attention.indexer.q_dim", ix_q_dim);

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);

    vqa::weights::promote_1d_to_f32(wctx, host_f32);
    {
        // placement: the checkpoint (18.6 GB) exceeds a 12 GB card. Options,
        // fastest first on this class of machine: pure CPU (default; the
        // driver's sysmem fallback for a full-GPU upload runs at ~1 tok/s),
        // VQA_SPLIT=1 (dense on GPU + experts in host mmap, ~10 tok/s),
        // KEYE_GPU=1 (force the full upload anyway).
        bool want_split = backend_cpu && !getenv("KEYE_GPU") &&
                          (getenv("VQA_SPLIT") != nullptr || ec_enabled);
        if (!want_split && backend_cpu && !getenv("KEYE_GPU")) {
            size_t total_all = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t))
                total_all += ggml_nbytes(t);
            size_t free_vram = 0, total_vram = 0;
#ifdef GGML_USE_CUDA
            ggml_backend_cuda_get_device_memory(0, &free_vram, &total_vram);
            if (total_all + (size_t)2.0e9 > free_vram) {
                fprintf(stderr, "keye: weights %.1f GB exceed free VRAM %.1f GB -> CPU backend "
                        "(VQA_SPLIT=1: dense on GPU + experts in RAM; KEYE_GPU=1: force GPU)\n",
                        total_all / 1e9, free_vram / 1e9);
                ggml_gallocr_free(allocr); allocr = nullptr;
                ggml_backend_free(backend);
                backend = backend_cpu;             // reuse the CPU partner
                backend_cpu = nullptr;
                allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                if (!allocr) return false;
            }
#endif
        }
        if (want_split) {
            size_t dev_total = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t))
                if (!vqa::weights::moe_expert_pred(t->name)) dev_total += ggml_nbytes(t);
            size_t free_vram = 0, total_vram = 0;
#ifdef GGML_USE_CUDA
            if (!ggml_backend_is_cpu(backend)) {
                ggml_backend_cuda_get_device_memory(0, &free_vram, &total_vram);
                // dense weights + KV (~0.9 GB) + compute headroom; an explicit
                // VQA_GPULAYERS overrides the budget check
                const bool forced = getenv("VQA_GPULAYERS") != nullptr || ec_enabled;
                if (!forced && dev_total + (size_t)2.5e9 > free_vram) want_split = false;
                fprintf(stderr, "keye split: dense %.2f GB, free VRAM %.2f GB -> %s\n",
                        dev_total / 1e9, free_vram / 1e9,
                        want_split ? "experts on CPU" : "no split");
            } else want_split = false;
#else
            want_split = false;
#endif
        }
        if (want_split) {
            vqa::weights::SplitBound sb = vqa::weights::split_bind_to_backends(
                wctx, backend, mm, vqa::weights::moe_expert_pred,
                { { host_f32.data(), host_f32.size() } });
            if (!sb.ok) return false;
            wbuf = sb.dev_buf;
            host_wbuf = sb.host_buf;
            ggml_backend_buffer_set_usage(sb.dev_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_set_usage(sb.host_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            split_mode = true;
            if (!split_sched.init(backend, backend_cpu)) return false;
        } else {
            vqa::weights::Bound bound = vqa::weights::bind_to_backend(
                wctx, backend, mm, { { host_f32.data(), host_f32.size() } });
            if (!bound.ok) return false;
            wbuf = bound.main_buf;
            if (!ggml_backend_is_cpu(backend)) mm.release();
        }
    }

    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) fprintf(stderr, "missing tensor: %s\n", n);
        return t;
    };
    tok_embd = gt("token_embd.weight");
    out_norm = gt("output_norm.weight");
    out_w    = gt("output.weight");
    if (tok_embd) n_vocab = (int)tok_embd->ne[1];   // GGUF has no vocab_size key

    // KEYE_WCHECK: layer-0 weight byte sums (binding sanity vs a numpy oracle)
    if (getenv("KEYE_WCHECK")) {
        auto ck = [&](const char * n) {
            ggml_tensor * w = ggml_get_tensor(wctx, n);
            if (!w) { printf("WCHECK %s missing\n", n); return; }
            const size_t bytes = ggml_nbytes(w);
            std::vector<uint8_t> buf(bytes);
            ggml_backend_tensor_get(w, buf.data(), 0, bytes);
            uint64_t s = 0;
            for (uint8_t b : buf) s += b;
            printf("WCHECK %s bytes %zu bytessum %llu\n", n, bytes, (unsigned long long)s);
        };
        ck("blk.0.attn_q.weight");
        ck("blk.0.ffn_gate_inp.weight");
        ck("blk.0.ffn_gate_exps.weight");
        ck("blk.0.ffn_up_exps.weight");
        ck("blk.0.ffn_down_exps.weight");
    }
    tl.resize((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "blk.%d.%s", i, suf); return gt(b); };
        TxtLayer & L = tl[(size_t)i];
        L.attn_norm = g("attn_norm.weight");
        L.wq = g("attn_q.weight"); L.wk = g("attn_k.weight");
        L.wv = g("attn_v.weight"); L.wo = g("attn_output.weight");
        L.qn = g("attn_q_norm.weight"); L.kn = g("attn_k_norm.weight");
        L.ix_q_b = g("indexer.attn_q_b.weight");
        L.ix_q_norm = g("indexer.q_norm.weight");
        L.ix_k = g("indexer.attn_k.weight");
        L.ix_k_norm = g("indexer.k_norm.weight"); L.ix_k_norm_b = g("indexer.k_norm.bias");
        L.ix_proj = g("indexer.proj.weight");
        L.ffn_norm = g("ffn_norm.weight");
        L.gate_inp = g("ffn_gate_inp.weight");
        L.up_exps = g("ffn_up_exps.weight");
        L.gate_exps = g("ffn_gate_exps.weight");
        L.down_exps = g("ffn_down_exps.weight");
        if (!L.attn_norm || !L.wq || !L.wk || !L.wv || !L.wo || !L.qn || !L.kn ||
            !L.ix_q_b || !L.ix_q_norm || !L.ix_k || !L.ix_k_norm || !L.ix_proj ||
            !L.ffn_norm || !L.gate_inp || !L.up_exps || !L.gate_exps || !L.down_exps)
            return false;
    }

    // Walsh-Hadamard / sqrt(64) (same construction as the fork's ggml_gen_hadamard)
    hadamard64.assign((size_t)ix_head_dim * ix_head_dim, 0.0f);
    hadamard64[0] = 1.0f / sqrtf((float)ix_head_dim);
    for (int s = 1; s < ix_head_dim; s *= 2)
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++) {
                const float v = hadamard64[(size_t)i * ix_head_dim + j];
                hadamard64[(size_t)(i + s) * ix_head_dim + j] = v;
                hadamard64[(size_t)i * ix_head_dim + (j + s)] = v;
                hadamard64[(size_t)(i + s) * ix_head_dim + (j + s)] = -v;
            }

    fprintf(stderr, "keye-vl2: %d layers d=%d experts %d/%d (ff %d), indexer %dx%d top %d\n",
            n_layer, n_embd, n_expert_used, n_expert, n_ff_exp, ix_head, ix_head_dim, ix_top_k);
    return tok_embd && out_norm && out_w;
}

bool KeyeVL2::load_vision_weights(const char * path) {
    if (!vmm.load(path)) { fprintf(stderr, "mmproj mmap failed: %s\n", path); return false; }
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &vctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "mmproj gguf init failed\n"); return false; }

    v_layer = vqa::gguf::u32(gu, "clip.vision.block_count", v_layer);
    v_embd  = vqa::gguf::u32(gu, "clip.vision.embedding_length", v_embd);
    v_ff    = vqa::gguf::u32(gu, "clip.vision.feed_forward_length", v_ff);
    v_head  = vqa::gguf::u32(gu, "clip.vision.attention.head_count", v_head);
    v_patch = vqa::gguf::u32(gu, "clip.vision.patch_size", v_patch);
    v_out   = (int)vqa::gguf::u32(gu, "clip.vision.projection_dim", v_out);
    v_eps   = vqa::gguf::f32(gu, "clip.vision.attention.layer_norm_epsilon", v_eps);
    v_hd = v_embd / (v_head > 0 ? v_head : 1);

    vqa::weights::bind_from_mmap(gu, vctx, vmm);
    gguf_free(gu);
    vqa::weights::promote_1d_to_f32(vctx, host_f32);
    {
        vqa::weights::Bound bound = vqa::weights::bind_to_backend(
            vctx, backend, vmm, { { host_f32.data(), host_f32.size() } });
        if (!bound.ok) return false;
        vbuf = bound.main_buf;
        if (!ggml_backend_is_cpu(backend)) vmm.release();
    }

    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(vctx, n);
        if (!t) fprintf(stderr, "missing mmproj tensor: %s\n", n);
        return t;
    };
    v_patch_w = gt("v.patch_embd.weight");
    v_patch_b = gt("v.patch_embd.bias");
    v_pos_embd = gt("v.position_embd.weight");
    v_post_ln_w = gt("v.post_ln.weight");
    v_post_ln_b = gt("v.post_ln.bias");
    mm_norm_w = gt("mm.input_norm.weight");
    mm_norm_b = gt("mm.input_norm.bias");
    mm0_w = gt("mm.0.weight"); mm0_b = gt("mm.0.bias");
    mm2_w = gt("mm.2.weight"); mm2_b = gt("mm.2.bias");
    vl.resize((size_t)v_layer);
    for (int i = 0; i < v_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "v.blk.%d.%s", i, suf); return gt(b); };
        VisLayer & L = vl[(size_t)i];
        L.ln1_w = g("ln1.weight"); L.ln1_b = g("ln1.bias");
        L.ln2_w = g("ln2.weight"); L.ln2_b = g("ln2.bias");
        L.wq = g("attn_q.weight"); L.bq = g("attn_q.bias");
        L.wk = g("attn_k.weight"); L.bk = g("attn_k.bias");
        L.wv = g("attn_v.weight"); L.bv = g("attn_v.bias");
        L.wo = g("attn_out.weight"); L.bo = g("attn_out.bias");
        L.fc0_w = g("ffn_up.weight"); L.fc0_b = g("ffn_up.bias");
        L.fc1_w = g("ffn_down.weight"); L.fc1_b = g("ffn_down.bias");
        if (!L.ln1_w || !L.wq || !L.wk || !L.wv || !L.wo || !L.fc0_w || !L.fc1_w) return false;
    }
    v_pos_side = (int)sqrtf((float)v_pos_embd->ne[1]);
    return v_patch_w && v_patch_b && v_pos_embd && v_post_ln_w && mm0_w && mm2_w && mm_norm_w;
}

bool KeyeVL2::alloc_kv() {
    if (!kv.alloc(backend, n_layer, n_kv_head, head_dim, n_ctx)) return false;
    // indexer K cache: per-layer [64, n_ctx] F16
    struct ggml_init_params ip = { (size_t)(n_layer + 8) * ggml_tensor_overhead(), NULL, true };
    lid_ctx = ggml_init(ip);
    if (!lid_ctx) return false;
    lid_k.resize((size_t)n_layer);
    for (int i = 0; i < n_layer; i++)
        lid_k[(size_t)i] = ggml_new_tensor_2d(lid_ctx, GGML_TYPE_F16, ix_head_dim, n_ctx);
    lid_buf = ggml_backend_alloc_ctx_tensors(lid_ctx, backend);
    if (!lid_buf) return false;
    ggml_backend_buffer_clear(lid_buf, 0);
    return true;
}

bool KeyeVL2::load(const std::string & gguf_path, const std::string & tokenizer_path,
                   const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "KEYE", n_threads, use_gpu, dump_dir);
    no_dsa = getenv("KEYE_PARITY_NO_DSA") != nullptr;
    ec_enabled = getenv("KEYE_EXPERTCACHE") != nullptr && getenv("KEYE_CPU") == nullptr;
    if (!init_backend()) return false;
    if (!load_text_weights(gguf_path.c_str())) return false;

    // mmproj sibling: --vision, else mmproj*.gguf next to the main gguf
    std::string mmproj = opt.vision_path;
    if (mmproj.empty()) {
        const size_t slash = gguf_path.find_last_of("/\\");
        const std::string dir = slash == std::string::npos ? "." : gguf_path.substr(0, slash);
        for (const char * cand : { "/mmproj-F16.gguf", "/mmproj-BF16.gguf", "/mmproj-Q8_0.gguf" }) {
            std::string f = dir + cand;
            FILE * fp = fopen(f.c_str(), "rb");
            if (fp) { fclose(fp); mmproj = f; break; }
        }
    }
    if (mmproj.empty()) { fprintf(stderr, "keye-vl2: no mmproj sibling found\n"); return false; }
    fprintf(stderr, "keye-vl2: vision weights: %s\n", mmproj.c_str());
    if (!load_vision_weights(mmproj.c_str())) return false;

    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::Qwen2Regex)) return false;
    if (!alloc_kv()) return false;
    if (ec_enabled && !ec_init()) return false;

    vqa::assign_token_ids(tok_, {
        { "<|image_pad|>",    &id_image_pad },
        { "<|vision_start|>", &id_vision_start },
        { "<|vision_end|>",   &id_vision_end },
        { "<|frame|>",        &id_frame },
        { "<|im_start|>",     &id_im_start },
        { "<|im_end|>",       &id_im_end },
    });
    id_eos = id_im_end;
    caps_.image = true;
    caps_.video = true;
    fprintf(stderr, "keye-vl2: ready (%dL d=%d hd=%d vocab=%d; vision %dL d=%d hd=%d)\n",
            n_layer, n_embd, head_dim, n_vocab, v_layer, v_embd, v_hd);
    return true;
}

// ---------------------------------------------------------------- vision

bool KeyeVL2::build_vision_graph(int ph, int pw) {
    if (vg_ctx && vg_ph == ph && vg_pw == pw) return true;
    if (vg_ctx) { ggml_free(vg_ctx); vg_ctx = nullptr; vg = nullptr; }

    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
    struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    vg_ctx = ggml_init(ip);
    if (!vg_ctx) return false;
    ggml_context * c = vg_ctx;
    const int N = ph * pw;

    ggml_cgraph * gf = ggml_new_graph_custom(c, 8192, false);
    vg = gf;

    // inputs: raster patches [3*14*14, N] (mean/std on host); interpolated
    // position embedding [1152, N]; 4-plane rope positions (y,x,y,x)
    vg_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, 3 * v_patch * v_patch, N);
    ggml_set_name(vg_in, "v_patches");
    vg_pos_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, v_embd, N);
    ggml_set_name(vg_pos_in, "v_pos_in");
    vg_rope_pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * N);
    ggml_set_name(vg_rope_pos, "v_rope_pos");

    // patch embedding: conv weight stored as [14,14,3,1152]; flat = [588, 1152]
    ggml_tensor * patch_w2 = ggml_reshape_2d(c, v_patch_w, 3 * v_patch * v_patch, v_embd);
    ggml_tensor * x = ggml_mul_mat(c, patch_w2, vg_in);
    x = ggml_add(c, x, v_patch_b);
    x = ggml_add(c, x, vg_pos_in);
    ggml_set_name(x, "v_patch_out");

    // KEYE_VSTOP: parity truncation - stop after patch embed (0), ViT layer k
    // (1..27), post_ln (28), merge (29), or run the full projector (unset)
    const int vstop = getenv("KEYE_VSTOP") ? atoi(getenv("KEYE_VSTOP")) : -1;
    if (vstop == 0) {
        ggml_build_forward_expand(gf, x);
        ggml_build_forward_expand(gf, vg_rope_pos);   // keep the input allocated
        vg_out = x;
        vg_ph = ph; vg_pw = pw;
        if (!ggml_gallocr_alloc_graph(allocr, gf)) return false;
        return true;
    }

    vit_env = vqa::vit::VitHookEnv();
    vit_env.gctx = c;
    vit_env.gf = gf;
    vit_env.n_head = v_head;
    vit_env.n_embd = v_embd;
    vit_env.N = N;
    vit_env.vhd = v_hd;
    vit_env.vhd_f = v_hd;
    vit_env.n_ctx = 32768;
    vit_env.rope_theta = v_rope_theta;
    vit_env.dbg_names = true;
    vit_env.t_vpos = vg_rope_pos;

    for (int i = 0; i < v_layer; i++) {
        const VisLayer & L = vl[(size_t)i];
        vit_env.layer = i;
        if (vstop == i + 1) {
            // truncate after i layers: output the current stream
            ggml_tensor * out = ggml_cont(c, x);
            ggml_build_forward_expand(gf, out);
            ggml_build_forward_expand(gf, vg_rope_pos);
            vg_out = out; vg_ph = ph; vg_pw = pw;
            if (!ggml_gallocr_alloc_graph(allocr, gf)) return false;
            return true;
        }
        ggml_tensor * ln = ggml_norm(c, x, v_eps);
        ln = ggml_mul(c, ln, L.ln1_w);
        ln = ggml_add(c, ln, L.ln1_b);
        // separate q/k/v projections fused into q|k|v along ne0 for the hook
        ggml_tensor *q = ggml_add(c, ggml_mul_mat(c, L.wq, ln), L.bq);
        ggml_tensor *k = ggml_add(c, ggml_mul_mat(c, L.wk, ln), L.bk);
        ggml_tensor *v = ggml_add(c, ggml_mul_mat(c, L.wv, ln), L.bv);
        ggml_tensor *qkv = ggml_concat(c, ggml_concat(c, q, k, 0), v, 0);
        ggml_tensor *attn = vqa::vit::vit_attention_hook((void *)&vit_env, qkv);
        if (!attn) return false;
        attn = ggml_mul_mat(c, L.wo, ggml_cont(c, attn));
        attn = ggml_add(c, attn, L.bo);
        x = ggml_add(c, x, attn);
        // MLP (GELU tanh; SigLIP convention, matches clip.use_gelu)
        ggml_tensor *l2 = ggml_norm(c, x, v_eps);
        l2 = ggml_mul(c, l2, L.ln2_w);
        l2 = ggml_add(c, l2, L.ln2_b);
        if (vstop == 100 + i) {   // truncate after layer i attention residual
            ggml_tensor * out = ggml_cont(c, x);
            ggml_build_forward_expand(gf, out);
            ggml_build_forward_expand(gf, vg_rope_pos);
            vg_out = out; vg_ph = ph; vg_pw = pw;
            if (!ggml_gallocr_alloc_graph(allocr, gf)) return false;
            return true;
        }
        ggml_tensor *f = ggml_add(c, ggml_mul_mat(c, L.fc0_w, l2), L.fc0_b);
        f = ggml_gelu(c, f);
        f = ggml_add(c, ggml_mul_mat(c, L.fc1_w, f), L.fc1_b);
        x = ggml_add(c, x, f);
        if (i == v_layer - 1) ggml_set_name(x, "v_vit_out");
    }

    // post_ln (LayerNorm with bias)
    x = ggml_cont(c, x);
    x = ggml_norm(c, x, v_eps);
    x = ggml_add(c, ggml_mul(c, x, v_post_ln_w), v_post_ln_b);

    // 2x2 spatial merge via the fork's two-step permute/cont merge
    // (token raster (y,x); merged token (my,mx) = patches (2my+dy, 2mx+dx),
    //  dy outer then dx inner, matching sglang (p1 p2 d))
    ggml_tensor *grid = ggml_reshape_4d(c, x, v_embd, pw, ph, 1);      // [D, x, y, 1]
    ggml_tensor *tr = ggml_permute(c, grid, 0, 2, 1, 3);               // [D, y, x, 1]
    tr = ggml_cont_4d(c, tr, v_embd * 2, ph / 2, pw, 1);               // merge along y
    tr = ggml_permute(c, tr, 0, 2, 1, 3);                              // [D*2, x, y/2, 1]
    tr = ggml_cont_4d(c, tr, v_embd * 4, pw / 2, ph / 2, 1);           // merge along x
    ggml_tensor *mrg = ggml_reshape_2d(c, tr, v_embd * 4, (pw / 2) * (ph / 2));
    ggml_set_name(mrg, "v_merge");

    // projector: LayerNorm (pre_norm, eps 1e-5) -> mm.0 -> GELU -> mm.2
    ggml_tensor *pr = ggml_norm(c, mrg, 1e-5f);
    pr = ggml_add(c, ggml_mul(c, pr, mm_norm_w), mm_norm_b);
    pr = ggml_add(c, ggml_mul_mat(c, mm0_w, pr), mm0_b);
    pr = ggml_gelu(c, pr);
    pr = ggml_add(c, ggml_mul_mat(c, mm2_w, pr), mm2_b);
    ggml_set_name(pr, "v_feats");

    ggml_build_forward_expand(gf, pr);
    vg_out = pr;
    vg_ph = ph; vg_pw = pw;

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "v_patch_out", "v_blk0_q_rope", "v_blk0_attn_out",
                                           "v_vit_out", "v_merge", "v_feats", nullptr };
        vqa::dump::mark_sinks(c, gf, dn);
        vqa::dump::finalize_sinks(c, gf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "vision graph allocation failed\n");
        return false;
    }
    return true;
}

bool KeyeVL2::run_vision(const std::vector<float> & patches, std::vector<float> & feats) {
    ggml_backend_tensor_set(vg_in, patches.data(), 0, sizeof(float) * patches.size());
    if (ggml_backend_graph_compute(backend, vg) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "vision compute failed\n");
        return false;
    }
    if (!dump_dir.empty()) {
        static const char * const dn[] = { "v_patch_out", "v_vit_out", "v_merge", "v_feats", nullptr };
        for (int i = 0; dn[i]; i++) dump(vg, dn[i]);
    }
    if (const char * fp = getenv("KEYE_VFEATS")) {   // graph output readback: trustworthy
        const int64_t n_el = ggml_nelements(vg_out);
        feats.resize((size_t)n_el);
        ggml_backend_tensor_get(vg_out, feats.data(), 0, sizeof(float) * feats.size());
        FILE * f = fopen(fp, "wb");
        if (f) { fwrite(feats.data(), 4, feats.size(), f); fclose(f); }
        fprintf(stderr, "keye vision: wrote %lld feats to %s\n", (long long)n_el, fp);
        return true;
    }
    const int n_tok = (vg_pw / 2) * (vg_ph / 2);
    feats.resize((size_t)n_tok * v_out);
    ggml_backend_tensor_get(vg_out, feats.data(), 0, sizeof(float) * feats.size());
    return true;
}

bool KeyeVL2::encode_frame(const vqa::Frame & frame, int w, int h,
                           std::vector<float> & feats, int * n_tokens, int * gh, int * gw) {
    const int factor = v_patch * v_merge;                   // 28
    const int min_px = 8 * factor * factor;                 // 8 merged tokens
    const int max_px = 2048 * factor * factor / 4;          // cap merged tokens (CPU speed)
    int rh = 0, rw = 0;
    if (!vqa::img::smart_resize(h, w, factor, min_px, max_px, &rh, &rw)) {
        fprintf(stderr, "image too small for the vision tower\n");
        return false;
    }
    const int ph = rh / v_patch, pw = rw / v_patch;
    fprintf(stderr, "keye vision: %dx%d -> %dx%d (grid %dx%d, %d tokens)\n",
            w, h, rw, rh, pw, ph, (pw / 2) * (ph / 2));

    std::vector<float> resized;   // RGB float 0..255
    vqa::img::resize_antialias(frame.rgb.data(), w, h, rw, rh, resized, /*pil=BILINEAR*/2);
    if (getenv("KEYE_RAWDBG")) {
        fprintf(stderr, "raw[0..5]: %d %d %d %d %d %d | resized[0..5]: %.1f %.1f %.1f %.1f %.1f %.1f | n=%zu\n",
                frame.rgb[0], frame.rgb[1], frame.rgb[2], frame.rgb[3], frame.rgb[4], frame.rgb[5],
                resized[0], resized[1], resized[2], resized[3], resized[4], resized[5], resized.size());
    }
    // raster patchify [3*14*14, ph*pw], channel-planar per patch, mean/std applied
    const int P = v_patch;
    std::vector<float> patches((size_t)3 * P * P * ph * pw);
    for (int py = 0; py < ph; py++)
        for (int px = 0; px < pw; px++)
            for (int c = 0; c < 3; c++)
                for (int dy = 0; dy < P; dy++)
                    for (int dx = 0; dx < P; dx++) {
                        const size_t sp = (((size_t)(py * P + dy) * rw) + (px * P + dx)) * 3;
                        const size_t dp = ((size_t)(py * pw + px) * 3 + c) * P * P
                                        + (size_t)dy * P + dx;
                        patches[dp] = ((resized[sp + c] / 255.0f) - v_mean) / v_std;
                    }

    // bilinear interpolation of the learned position embedding [1152, side^2]
    // to (ph, pw) on host (F.interpolate align_corners=False)
    if (const char * fp = getenv("KEYE_PATCHES")) {
        FILE * f = fopen(fp, "wb");
        if (f) { fwrite(patches.data(), 4, patches.size(), f); fclose(f); }
        fprintf(stderr, "keye vision: wrote patches (%d x %d)\n",
                3 * v_patch * v_patch, ph * pw);
    }
    std::vector<float> pos_e((size_t)v_embd * ph * pw);
    {
        std::vector<float> src((size_t)v_embd * v_pos_side * v_pos_side);
        ggml_backend_tensor_get(v_pos_embd, src.data(), 0, sizeof(float) * src.size());
        const float sy = (float)v_pos_side / ph, sx = (float)v_pos_side / pw;
        for (int py = 0; py < ph; py++) {
            const float fy = (py + 0.5f) * sy - 0.5f;
            const int y0 = std::max(0, (int)floorf(fy));
            const int y1 = std::min(v_pos_side - 1, y0 + 1);
            const float wy = std::min(1.0f, std::max(0.0f, fy - y0));
            for (int px_ = 0; px_ < pw; px_++) {
                const float fx = (px_ + 0.5f) * sx - 0.5f;
                const int x0 = std::max(0, (int)floorf(fx));
                const int x1 = std::min(v_pos_side - 1, x0 + 1);
                const float wx = std::min(1.0f, std::max(0.0f, fx - x0));
                const size_t d = ((size_t)py * pw + px_) * v_embd;
                const float * a = &src[((size_t)y0 * v_pos_side + x0) * v_embd];
                const float * b = &src[((size_t)y0 * v_pos_side + x1) * v_embd];
                const float * cc = &src[((size_t)y1 * v_pos_side + x0) * v_embd];
                const float * dd = &src[((size_t)y1 * v_pos_side + x1) * v_embd];
                for (int e = 0; e < v_embd; e++)
                    pos_e[d + e] = (a[e] * (1 - wx) + b[e] * wx) * (1 - wy)
                                 + (cc[e] * (1 - wx) + dd[e] * wx) * wy;
            }
        }
    }
    // rope positions: 4 planes (y, x, y, x) per patch, raster order
    std::vector<int> rp;
    {
        const int N = ph * pw;
        rp.resize((size_t)4 * N);
        for (int y = 0; y < ph; y++)
            for (int x = 0; x < pw; x++) {
                const size_t i = (size_t)y * pw + x;
                rp[i] = y; rp[(size_t)N + i] = x;
                rp[(size_t)2 * N + i] = y; rp[(size_t)3 * N + i] = x;
            }
    }

    if (!build_vision_graph(ph, pw)) return false;
    ggml_backend_tensor_set(vg_rope_pos, rp.data(), 0, sizeof(int) * rp.size());
    ggml_backend_tensor_set(vg_pos_in, pos_e.data(), 0, sizeof(float) * pos_e.size());
    if (!run_vision(patches, feats)) return false;
    *n_tokens = (pw / 2) * (ph / 2);
    *gh = ph / 2;
    *gw = pw / 2;
    return true;
}

// ---------------------------------------------------------------- text

bool KeyeVL2::build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all) {
    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
    struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    ggml_context * c = g.ctx;
    g.gf = ggml_new_graph_custom(c, 131072, false);

    const int D = n_embd, HD = head_dim, NH = n_head, NKV = n_kv_head;
    const int IXH = ix_head, IXD = ix_head_dim;
    const int K_SEL = std::min<int>(S, ix_top_k);

    g.tok_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    g.pos_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * T);
    ggml_set_name(g.pos_in, "pos_in");
    g.slot_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_name(g.slot_in, "slot_in");
    // F32 causal mask [S, T]; the DSA top-k mask is added in-graph per layer
    g.mask_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, S, T);
    ggml_set_name(g.mask_in, "mask_in");

    ggml_tensor * had = ggml_new_tensor_2d(c, GGML_TYPE_F32, IXD, IXD);
    ggml_set_name(had, "hadamard");

    ggml_tensor * x = ggml_get_rows(c, tok_embd, g.tok_in);
    if (n_img > 0) {
        g.img_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n_img);
        ggml_set_name(g.img_in, "img_in");
        g.img_pos_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_img);
        ggml_set_name(g.img_pos_in, "img_pos_in");
        x = ggml_set_rows(c, ggml_dup(c, x), g.img_in, g.img_pos_in);
    }
    ggml_set_name(x, "t_emb");

    int sec[4] = { mrope_section[0], mrope_section[1], mrope_section[2], 0 };
    // indexer rope sections scale with head size: [16,24,24] * 64/128
    int isec[4] = { mrope_section[0] / 2, mrope_section[1] / 2, mrope_section[2] / 2, 0 };
    const float ix_scale = 1.0f / sqrtf((float)IXD);
    const bool xchk = getenv("KEYE_XCHK") != nullptr;

    for (int il = 0; il < n_layer; il++) {
        const TxtLayer & L = tl[(size_t)il];
        ggml_tensor * l = ggml_rms_norm(c, x, rms_eps);
        l = ggml_mul(c, l, L.attn_norm);
        if (il == 0) ggml_set_name(l, "t_l0_norm");

        // ---- DSA indexer (per fork keye.cpp) ----
        ggml_tensor * top_k = nullptr;
        if (!no_dsa) {
            ggml_tensor *iq = ggml_mul_mat(c, L.ix_q_b, l);              // [1024, T]
            iq = ggml_reshape_3d(c, iq, IXD, IXH, T);
            iq = ggml_rms_norm(c, iq, rms_eps);
            iq = ggml_mul(c, iq, L.ix_q_norm);
            iq = ggml_rope_multi(c, iq, g.pos_in, nullptr, IXD, isec,
                                 GGML_ROPE_TYPE_MROPE, n_ctx, rope_theta,
                                 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            iq = ggml_mul_mat(c, had, iq);                               // Hadamard

            ggml_tensor *ik = ggml_mul_mat(c, L.ix_k, l);                // [64, T] MQA
            ik = ggml_norm(c, ik, 1e-6f);
            ik = ggml_add(c, ggml_mul(c, ik, L.ix_k_norm), L.ix_k_norm_b);
            ik = ggml_reshape_3d(c, ik, IXD, 1, T);
            ik = ggml_rope_multi(c, ik, g.pos_in, nullptr, IXD, isec,
                                 GGML_ROPE_TYPE_MROPE, n_ctx, rope_theta,
                                 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ik = ggml_mul_mat(c, had, ik);
            ggml_tensor *ik2 = ggml_cast(c, ggml_reshape_2d(c, ik, IXD, T), GGML_TYPE_F16);
            ggml_build_forward_expand(g.gf, ggml_set_rows(c, lid_k[(size_t)il], ik2, g.slot_in));

            // scores against the cached indexer keys
            ggml_tensor *iw = ggml_scale(c, ggml_mul_mat(c, L.ix_proj, l), ix_scale); // [16, T]
            ggml_tensor *kc = ggml_view_2d(c, lid_k[(size_t)il], IXD, S,
                                           lid_k[(size_t)il]->nb[1], 0);  // [64, S]
            ggml_tensor *q3 = ggml_cont(c, ggml_permute(c, iq, 0, 2, 1, 3));   // [64, T, 16]
            ggml_tensor *k3 = ggml_reshape_3d(c, kc, IXD, S, 1);
            ggml_tensor *kq = ggml_mul_mat(c, k3, q3);                   // [S, T, 16]
            ggml_tensor *sc = ggml_cont(c, ggml_permute(c, kq, 2, 1, 0, 3));  // [16, T, S]
            sc = ggml_relu(c, sc);
            sc = ggml_mul(c, sc, ggml_reshape_3d(c, iw, IXH, T, 1));     // broadcast over S
            sc = ggml_sum_rows(c, sc);                                   // [1, T, S]
            sc = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, sc, 2, 1, 0, 3)), S, T);
            sc = ggml_add(c, sc, g.mask_in);                             // causal
            top_k = ggml_cont(c, ggml_top_k(c, sc, K_SEL));              // [K, T] I32
            if (il == 0) ggml_set_name(top_k, "t_l0_topk");
        }

        // ---- GQA attention with per-head q/k norm + mRoPE ----
        ggml_tensor *q = ggml_mul_mat(c, L.wq, l);
        ggml_tensor *k = ggml_mul_mat(c, L.wk, l);
        ggml_tensor *v = ggml_mul_mat(c, L.wv, l);
        ggml_tensor *q3 = ggml_reshape_3d(c, q, HD, NH, T);
        q3 = ggml_rms_norm(c, q3, rms_eps);
        q3 = ggml_mul(c, q3, L.qn);
        q3 = ggml_rope_multi(c, q3, g.pos_in, nullptr, HD, sec, GGML_ROPE_TYPE_MROPE,
                             n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor *k3 = ggml_reshape_3d(c, k, HD, NKV, T);
        k3 = ggml_rms_norm(c, k3, rms_eps);
        k3 = ggml_mul(c, k3, L.kn);
        k3 = ggml_rope_multi(c, k3, g.pos_in, nullptr, HD, sec, GGML_ROPE_TYPE_MROPE,
                             n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (il == 0) {
            ggml_set_name(q3, "t_l0_q_rope");
            ggml_set_name(k3, "t_l0_k_rope");
        }

        ggml_tensor *k2 = ggml_cast(c, ggml_reshape_2d(c, k3, NKV * HD, T), GGML_TYPE_F16);
        ggml_tensor *v2 = ggml_cast(c, ggml_reshape_2d(c, v, NKV * HD, T), GGML_TYPE_F16);
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, kv.k[(size_t)il], k2, g.slot_in));
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, kv.v[(size_t)il], v2, g.slot_in));

        // ---- DSA mask: -inf everywhere except the selected top-k rows ----
        // KEYE_NO_DSA=1: plain causal attention (isolation switch for debugging)
        ggml_tensor *dsa_mask = nullptr;
        if (!no_dsa) {
        ggml_tensor *mask_all = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, S, T);
        mask_all = ggml_fill(c, mask_all, -INFINITY);
        ggml_tensor *tk_idx = ggml_view_3d(c, top_k, K_SEL, T, 1,
                                           top_k->nb[1], top_k->nb[2], 0);
        ggml_tensor *zeros = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, K_SEL, T);
        zeros = ggml_fill(c, zeros, 0.0f);
        ggml_tensor *sparse = ggml_set_rows(c, mask_all, zeros, tk_idx);
        sparse = ggml_view_2d(c, sparse, S, T, sparse->nb[1], 0);        // [S, T]
        dsa_mask = ggml_add(c, sparse, g.mask_in);                       // AND causal
        } else {
            dsa_mask = ggml_cast(c, g.mask_in, GGML_TYPE_F16);
        }
        dsa_mask = ggml_cast(c, dsa_mask, GGML_TYPE_F16);                // FA wants F16

        ggml_tensor *qf = ggml_cont(c, ggml_permute(c, q3, 0, 2, 1, 3)); // [HD, T, NH]
        ggml_tensor *kf = ggml_view_3d(c, kv.k[(size_t)il], HD, S, NKV,
                                       kv.k[(size_t)il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor *vf = ggml_view_3d(c, kv.v[(size_t)il], HD, S, NKV,
                                       kv.v[(size_t)il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor *fa = ggml_flash_attn_ext(c, qf, kf, vf, dsa_mask,
                                              1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        ggml_tensor *attn = ggml_reshape_2d(c, ggml_cont(c, fa), NH * HD, T);
        if (il == 0) ggml_set_name(attn, "t_l0_attn");
        attn = ggml_mul_mat(c, L.wo, attn);
        x = ggml_add(c, x, attn);
        if (il == 0) ggml_set_name(x, "t_l0_resid");

        // ---- MoE FFN: softmax gating, top-8, renorm (softmax over selected) ----
        ggml_tensor *m = ggml_rms_norm(c, x, rms_eps);
        m = ggml_mul(c, m, L.ffn_norm);
        if (il == 0) ggml_set_name(m, "t_l0_m");
        ggml_tensor *rlogits = ggml_mul_mat(c, L.gate_inp, m);           // [E, T]
        if (il == 0) ggml_set_name(rlogits, "t_l0_rlogits");
        ggml_tensor *sel = ggml_top_k(c, rlogits, n_expert_used);        // [K, T] I32
        ggml_tensor *sel_logits = ggml_get_rows(c,
                ggml_reshape_3d(c, rlogits, 1, n_expert, T), sel);       // [1, K, T]
        sel_logits = ggml_reshape_2d(c, sel_logits, n_expert_used, T);
        ggml_tensor *w8 = ggml_soft_max(c, sel_logits);                  // [K, T]
        w8 = ggml_reshape_3d(c, w8, 1, n_expert_used, T);

        ggml_tensor *cur3 = ggml_reshape_3d(c, m, D, 1, T);
        ggml_tensor *gu = ggml_mul_mat_id(c, L.gate_exps, cur3, sel);    // [I, K, T]
        ggml_tensor *up = ggml_mul_mat_id(c, L.up_exps, cur3, sel);      // [I, K, T]
        // swiglu_split (GGML_OP_GLU) instead of mul(silu(gu), up): identical
        // math, and only the GLU form triggers the CUDA fused
        // mul_mat_id+mul_mat_id+GLU pass (see ggml-cuda.cu ~1680-1860)
        ggml_tensor *act = ggml_swiglu_split(c, gu, up);
        if (il == 0) { ggml_set_name(gu, "t_l0_gu"); ggml_set_name(up, "t_l0_up");
                       ggml_set_name(act, "t_l0_act"); ggml_set_name(sel, "t_l0_sel"); }
        if (il == 0 && getenv("KEYE_SELDBG"))
            ggml_build_forward_expand(g.gf, sel);   // keep alive for host readback
        ggml_tensor *ff3 = ggml_mul_mat_id(c, L.down_exps, act, sel);    // [D, K, T]
        ggml_tensor *wb = ggml_view_3d(c, w8, 1, n_expert_used, T, w8->nb[1], w8->nb[2], 0);
        ggml_tensor *prod = ggml_mul(c, ff3, wb);                        // [D, K, T]
        ggml_tensor *pk = ggml_cont(c, ggml_permute(c, prod, 1, 0, 2, 3)); // [K, D, T]
        ggml_tensor *moe = ggml_reshape_2d(c, ggml_sum_rows(c, pk), D, T);
        if (il == 0) ggml_set_name(moe, "t_l0_moe");
        x = ggml_add(c, x, moe);
        if (il == 0) ggml_set_name(x, "t_l0_out");
        if (xchk) {   // per-layer readback tails for the EC cross-check
            char nb[48];
            snprintf(nb, sizeof(nb), "t_l%d_out_o", il);
            ggml_tensor * xo = ggml_dup(c, x);
            ggml_set_name(xo, nb);
            ggml_build_forward_expand(g.gf, xo);
        }
    }

    x = ggml_rms_norm(c, x, rms_eps);
    x = ggml_mul(c, x, out_norm);
    ggml_set_name(x, "t_final_norm");

    if (logits_all) {
        g.logits = ggml_mul_mat(c, out_w, x);
    } else {
        ggml_tensor *last = ggml_view_2d(c, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
        g.logits = ggml_mul_mat(c, out_w, last);
    }
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "t_emb", "t_l0_norm", "t_l0_q_rope", "t_l0_k_rope",
                                           "t_l0_topk", "t_l0_attn", "t_l0_gu", "t_l0_up",
                                           "t_l0_act", "t_l0_sel", "t_l0_moe", "t_l0_out",
                                           "t_final_norm", nullptr };
        vqa::dump::mark_sinks(c, g.gf, dn);
        vqa::dump::finalize_sinks(c, g.gf, backend);
    }
    if (!split_mode) {
        if (!ggml_gallocr_alloc_graph(allocr, g.gf)) {
            fprintf(stderr, "text graph allocation failed\n");
            return false;
        }
        if (!no_dsa)
            ggml_backend_tensor_set(had, hadamard64.data(), 0, sizeof(float) * hadamard64.size());
    }
    return true;
}

// KEYE_PROFILE=1: per-op decode timing. A single-backend sched with an eval
// callback that always splits re-executes the graph one node at a time (the
// sched splits via internal graph views), timing each op as it goes. This is
// many times slower than one whole-graph compute and the per-node dispatch
// overhead inflates small ops, so read it for ratios, not absolutes.
namespace keye_profile {
std::map<std::string, double> ms_by_op;
double ms_all = 0.0, ms_mmid = 0.0;
std::chrono::steady_clock::time_point last;
bool started = false;
long calls = 0;

// split only at the ops that plausibly dominate a MoE decode step; everything
// between two splits is one segment attributed to the closing op. Splitting
// per node would measure only the sched's per-split dispatch overhead
// (~4400 nodes/step -> every op looked like ~20 ms of nothing).
bool is_split_op(ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_GLU:
        case GGML_OP_ROPE:
            return true;
        default:
            return false;
    }
}

bool cb(ggml_tensor * t, bool ask, void *) {
    if (ask) {
        if (!started) { last = std::chrono::steady_clock::now(); started = true; }
        return is_split_op(t);
    }
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - last).count();
    last = std::chrono::steady_clock::now();
    ms_all += ms;
    if (t->op == GGML_OP_MUL_MAT_ID) ms_mmid += ms;
    ms_by_op[ggml_op_name(t->op)] += ms;
    return true;
}
}  // namespace keye_profile

static bool profiled_text_compute(ggml_backend_t backend, ggml_cgraph * gf) {
    static const bool on = getenv("KEYE_PROFILE") != nullptr;
    static const bool timed = getenv("KEYE_TIME") != nullptr;
    if (!on) {
        auto t0 = std::chrono::steady_clock::now();
        enum ggml_status ec = ggml_backend_graph_compute(backend, gf);
        if (timed)
            fprintf(stderr, "keye-time: %.1f ms/step\n",
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count());
        return ec == GGML_STATUS_SUCCESS;
    }

    using namespace keye_profile;
    ggml_backend_sched_t sched =
        ggml_backend_sched_new(&backend, nullptr, 1, (size_t)ggml_graph_n_nodes(gf),
                               /*parallel*/ false, /*op_offload*/ false);
    if (!sched) return false;
    ggml_backend_sched_set_eval_callback(sched, cb, nullptr);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) { ggml_backend_sched_free(sched); return false; }
    enum ggml_status ec = ggml_backend_sched_graph_compute(sched, gf);
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_free(sched);
    if (ec != GGML_STATUS_SUCCESS) return false;

    calls++;
    fprintf(stderr, "keye-profile[call %ld]: nodes %d, whole-graph equivalent %.1f ms, mul_mat_id %.1f ms\n",
            calls, ggml_graph_n_nodes(gf), ms_all, ms_mmid);
    if (calls % 8 == 0) {
        std::vector<std::pair<double, std::string>> ranked;
        double sum = 0;
        for (const auto & kv : ms_by_op) { ranked.emplace_back(kv.second, kv.first); sum += kv.second; }
        std::sort(ranked.rbegin(), ranked.rend());
        fprintf(stderr, "keye-profile cumulative after %ld calls (%.0f ms total):\n", calls, sum);
        for (const auto & r : ranked)
            fprintf(stderr, "  %-12s %9.1f ms  %5.1f%%\n", r.second.c_str(), r.first, 100.0 * r.first / sum);
    }
    return true;
}

bool KeyeVL2::run_text(TextGraph & g, const std::vector<int> & toks,
                       const std::vector<int> & pos4, const std::vector<float> * img_feats,
                       const std::vector<int> * img_pos, std::vector<float> & logits,
                       int slot_base, const std::vector<float> * mask) {
    const int T = (int)toks.size();
    if (split_mode && !split_sched.alloc(g.gf)) return false;   // inputs valid after this
    if (split_mode && !no_dsa) {
        ggml_tensor * had = ggml_graph_get_tensor(g.gf, "hadamard");
        if (had) ggml_backend_tensor_set(had, hadamard64.data(), 0, sizeof(float) * hadamard64.size());
    }
    ggml_backend_tensor_set(g.tok_in, toks.data(), 0, sizeof(int) * toks.size());
    ggml_backend_tensor_set(g.pos_in, pos4.data(), 0, sizeof(int) * pos4.size());
    {
        std::vector<int> slots((size_t)T);
        for (int i = 0; i < T; i++) slots[(size_t)i] = slot_base + i;
        ggml_backend_tensor_set(g.slot_in, slots.data(), 0, sizeof(int) * slots.size());
    }
    if (mask)
        ggml_backend_tensor_set(g.mask_in, mask->data(), 0, sizeof(float) * mask->size());
    if (g.img_in && img_feats && img_pos) {
        const size_t n_up = std::min(img_feats->size(), (size_t)ggml_nelements(g.img_in));
        ggml_backend_tensor_set(g.img_in, img_feats->data(), 0, sizeof(float) * n_up);
        ggml_backend_tensor_set(g.img_pos_in, img_pos->data(), 0, sizeof(int) * img_pos->size());
    }
    if (split_mode) {
        if (!split_sched.run(g.gf)) {
            fprintf(stderr, "text compute failed (split)\n");
            return false;
        }
    } else if (!profiled_text_compute(backend, g.gf)) {
        fprintf(stderr, "text compute failed\n");
        return false;
    }
    if (getenv("KEYE_NANCHECK")) {
        // compute 后直接读回（SOP：sink dump 可能被 gallocr 复用污染，不可信）
        const int n_nodes = ggml_graph_n_nodes(g.gf);
        int reported = 0;
        for (int i = 0; i < n_nodes && reported < 12; i++) {
            ggml_tensor * n = ggml_graph_node(g.gf, i);
            if (!n || n->type != GGML_TYPE_F32 || n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE) continue;
            const int64_t total = ggml_nelements(n);
            const int64_t step = std::max<int64_t>(1, total / 256);
            int nn = 0;
            float v;
            for (int64_t j = 0; j < total; j += step) {
                ggml_backend_tensor_get(n, &v, (size_t)j * sizeof(float), sizeof(float));
                if (std::isnan(v) || std::isinf(v)) nn++;
            }
            if (nn > 0) {
                fprintf(stderr, "nancheck[%d] %-24s op=%-22s nan/inf %d/%lld\n",
                        i, n->name[0] ? n->name : "-", ggml_op_name(n->op), nn, (long long)(total / step));
                reported++;
            }
        }
    }
    if (!dump_dir.empty()) {
        static const char * const dn[] = { "t_l0_norm", "t_l0_q_rope", "t_l0_topk", "t_l0_attn",
                                           "t_l0_moe", "t_l0_out", nullptr };
        for (int i = 0; dn[i]; i++) dump(g.gf, dn[i]);
    }
    logits.resize((size_t)n_vocab);
    ggml_backend_tensor_get(g.logits, logits.data(), 0, sizeof(float) * logits.size());
    if (getenv("KEYE_SELDBG")) {
        ggml_tensor * sel = ggml_graph_get_tensor(g.gf, "t_l0_sel");
        if (sel) {
            int s8[16];
            ggml_backend_tensor_get(sel, s8, 0, sizeof(int) * n_expert_used);
            fprintf(stderr, "seldbg L0 sel:");
            for (int i = 0; i < n_expert_used; i++) fprintf(stderr, " %d", s8[i]);
            fprintf(stderr, "\n");
        }
        ggml_tensor * rl = ggml_graph_get_tensor(g.gf, "t_l0_rlogits");
        if (rl) {
            std::vector<float> rv((size_t)n_expert * T);
            ggml_backend_tensor_get(rl, rv.data(), 0, sizeof(float) * rv.size());
            int best = 0;
            for (int e = 1; e < n_expert; e++) if (rv[(size_t)e] > rv[(size_t)best]) best = e;
            fprintf(stderr, "seldbg L0 rlogits argmax %d (%.4f), rlogits[0..3]"
                    " %.4f %.4f %.4f %.4f\n", best, rv[(size_t)best],
                    rv[0], rv[1], rv[2], rv[3]);
        }
    }
    return true;
}

void KeyeVL2::dump(const ggml_cgraph * gf, const char * name) {
    if (dump_dir.empty()) return;
    vqa::dump::write((ggml_cgraph *)gf, dump_dir.c_str(), name);
}

// ---------------------------------------------------------------- ask

bool KeyeVL2::ask(const vqa::MediaInput & media, const std::string & prompt,
                  const std::string & /*mode*/, const vqa::GenParams & gp,
                  std::string & out_text) {
    out_text.clear();

    struct FrameFeats { std::vector<float> feats; int n_tok, gh, gw; double ts; };
    std::vector<FrameFeats> ff;
    if (!media.empty()) {
        if (gp.on_stage) gp.on_stage("vision", 0, (int)media.frames.size(), gp.on_stage_ud);
        int done = 0;
        for (size_t i = 0; i < media.frames.size(); i++) {
            FrameFeats f;
            f.ts = i < media.timestamps.size() ? media.timestamps[i] : -1.0;
            if (!encode_frame(media.frames[i], media.width, media.height,
                              f.feats, &f.n_tok, &f.gh, &f.gw)) return false;
            ff.push_back(std::move(f));
            if (gp.on_stage) gp.on_stage("vision", ++done, (int)media.frames.size(), gp.on_stage_ud);
        }
        if (const char * mi = getenv("KEYE_MAXIMG")) {  // parity knob: cap image tokens
            int n = atoi(mi);
            for (auto & f : ff) f.n_tok = std::min(f.n_tok, n);
            ff.resize(1);
        }
    }

    // prompt: <|vision_start|> (per frame: <|frame|>[<hh:mm:ss>] + n_tok image
    // pads) <|vision_end|>; the pad ids are inserted as ids because encode()
    // does not round-trip "<|image_pad|>" (the Qwen regex tokenizer splits it)
    std::string text_head = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
    if (!ff.empty()) text_head += "<|vision_start|>";

    // build the token stream segment by segment: encode() does not round-trip
    // "<|image_pad|>" (the Qwen regex tokenizer splits it), so the pad ids are
    // inserted directly instead of going through the text
    std::vector<int> toks = tok_.encode(text_head);
    if (!ff.empty()) {
        const bool is_video = ff.size() > 1;
        for (size_t i = 0; i < ff.size(); i++) {
            if (is_video) {
                std::string mark = "<|frame|>";
                if (ff[i].ts >= 0.0) {
                    int total = (int)ff[i].ts;
                    char b[32];
                    snprintf(b, sizeof(b), "<|frame|><%02d:%02d:%02d>",
                             total / 3600, (total % 3600) / 60, total % 60);
                    mark = b;
                }
                std::vector<int> mt = tok_.encode(mark);
                toks.insert(toks.end(), mt.begin(), mt.end());
            }
            toks.insert(toks.end(), (size_t)ff[i].n_tok, id_image_pad);
        }
        std::vector<int> et = tok_.encode("<|vision_end|>" + prompt + "<|im_end|>\n<|im_start|>assistant\n");
        toks.insert(toks.end(), et.begin(), et.end());
    } else {
        std::vector<int> et = tok_.encode(prompt + "<|im_end|>\n<|im_start|>assistant\n");
        toks.insert(toks.end(), et.begin(), et.end());
    }
    if ((int)toks.size() + gp.max_tokens + 8 > n_ctx) {
        fprintf(stderr, "prompt %zu + %d tokens exceed n_ctx %d\n", toks.size(), gp.max_tokens, n_ctx);
        return false;
    }
    const int T = (int)toks.size();

    // mRoPE positions - official get_rope_index semantics
    // (modeling_keye_topk_mask_30ba3b.py:788-935, verified against source):
    // the sequence alternates TEXT runs and per-frame PAD runs. Per frame:
    //   text run  -> arange(text_len) + st_idx on all three axes
    //   pad run   -> (t=0, y=row, x=col) + text_len + st_idx
    //   st_idx    = text_len + st_idx + max(gh-1, gw-1) + 1
    // The h/w axes raise the base for everything after the frame (mrope
    // delta). The previous fork semantics (absolute sequence base) diverge
    // from the second frame on.
    std::vector<int> pos((size_t)4 * T, 0);
    std::vector<int> img_pos;
    {
        int st = 0, st_idx = 0;
        size_t fi = 0;
        while (fi < ff.size()) {
            // find this frame's pad-run start from st
            int ed = st;
            while (ed < T && toks[(size_t)ed] != id_image_pad) ed++;
            const int text_len = ed - st;
            for (int k = 0; k < text_len; k++)
                pos[(size_t)(st + k)] = pos[(size_t)T + (st + k)] =
                pos[(size_t)2 * T + (st + k)] = st_idx + k;
            const FrameFeats & f = ff[fi];
            const int n = f.gh * f.gw;
            const int base = st_idx + text_len;
            for (int k = 0; k < n; k++) {
                const int idx = ed + k;
                pos[(size_t)idx] = base;
                pos[(size_t)T + idx] = base + k / f.gw;
                pos[(size_t)2 * T + idx] = base + k % f.gw;
                img_pos.push_back(idx);
            }
            st_idx = base + std::max(f.gh - 1, f.gw - 1) + 1;
            st = ed + n;
            fi++;
        }
        for (int k = st; k < T; k++)
            pos[(size_t)k] = pos[(size_t)T + k] = pos[(size_t)2 * T + k] = st_idx + (k - st);
    }
    if (getenv("KEYE_DUMP_POS")) {
        FILE * fp = fopen("keye_pos_eng.bin", "wb");
        if (fp) { fwrite(pos.data(), sizeof(int), pos.size(), fp); fclose(fp); }
    }
    std::vector<float> feats_all;
    for (const auto & f : ff) {
        feats_all.insert(feats_all.end(), f.feats.begin(),
                         f.feats.begin() + (size_t)f.n_tok * v_out);
    }
    if (getenv("KEYE_ZEROIMG"))   // isolation: zero embeddings through the same path
        std::fill(feats_all.begin(), feats_all.end(), 0.0f);

    TextGraph g;
    if (!build_text_graph(g, T, T, (int)img_pos.size(), false)) { ggml_free(g.ctx); return false; }
    std::vector<float> mask((size_t)T * T, -INFINITY);
    for (int t = 0; t < T; t++)
        for (int s = 0; s <= t; s++) mask[(size_t)t * T + s] = 0.0f;
    std::vector<float> logits;
    if (!run_text(g, toks, pos, img_pos.empty() ? nullptr : &feats_all,
                  img_pos.empty() ? nullptr : &img_pos, logits, 0, &mask)) { ggml_free(g.ctx); return false; }
    ggml_free(g.ctx);
    n_past = T;
    next_pos = 0;
    for (int i = 0; i < T; i++)
        next_pos = std::max({ next_pos, pos[(size_t)i] + 1, pos[(size_t)T + i] + 1,
                              pos[(size_t)2 * T + i] + 1 });
    if (getenv("KEYE_DEBUG")) {
        fprintf(stderr, "tokens(%d):", T);
        for (int t : toks) fprintf(stderr, " %d", t);
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
        if (ec_enabled && ec_ok && !getenv("KEYE_EC_OFF")) {
            // dynamic expert-cache decode path (Strata-style tier 2)
            std::vector<float> lg;
            if (!ec_decode_step(cur, next_pos, n_past, lg)) return false;
            n_past++;
            next_pos++;
            cur = vqa::sample_logits(lg, gp, rng_state);
            if (cur == id_eos) break;
            emit(cur);
            continue;
        }
        // Graph reuse + KV-window bucketing (see decode_graph.hpp), scoped by
        // what the A/B numbers support per placement:
        //  - split mode: the per-step rebuild stays - the two-backend sched
        //    rewrites node->src to its per-split copy tensors (ggml-backend.cpp
        //    split_graph), so a graph reused across steps is re-split in an
        //    already-rewritten state; window stays block_for(512) as before.
        //  - CPU (default): attention + the DSA indexer scale with S and own
        //    the step, so dead-key padding is pure loss (bucket-128 measured
        //    4.7x slower in the tail past n_past 128) while the per-step
        //    rebuild costs ~0.1 of ~9-11 ms - keep the exact window.
        //  - all-GPU (launch-bound): one graph per 1024 bucket, the
        //    decode_graph.hpp shape; VQA_DECODE_BUCKET overrides the width.
        const bool on_cpu = ggml_backend_is_cpu(backend);
        const int S = split_mode ? vqa::block_for(n_past + 1)
                   : on_cpu      ? n_past + 1
                                 : vqa::decode_bucket(n_past + 1, n_ctx, false);
        TextGraph step{};
        TextGraph * dp = &step;
        if (!split_mode && !on_cpu) {
            if (!dec_slot.prepare(S, [&](TextGraph & g) { return build_text_graph(g, 1, S, 0, false); }))
                return false;
            dp = &dec_slot.g;
        } else if (!build_text_graph(step, 1, S, 0, false)) {
            ggml_free(step.ctx);
            return false;
        }
        // decode step: every cached key is causally visible, padding hidden.
        // F32 mask (the DSA scores are F32; the shared set_causal_mask is F16).
        std::vector<float> mask((size_t)S, -INFINITY);
        for (int s = 0; s <= n_past; s++) mask[(size_t)s] = 0.0f;
        std::vector<int> p1((size_t)4, next_pos);
        std::vector<float> lg;
        const bool ok = run_text(*dp, { cur }, p1, nullptr, nullptr, lg, n_past, &mask);
        if (dp == &step) ggml_free(step.ctx);   // slot graphs live until the next bucket
        if (!ok) return false;
        n_past++;
        next_pos++;
        cur = vqa::sample_logits(lg, gp, rng_state);
        if (cur == id_eos) break;
        emit(cur);
    }
    out_text = tok_.decode(out);
    if (ec_enabled && ec_ok && ec_st.steps > 0) {
        vqa::ExpertCacheStats st;
        for (auto & c : ec_caches) {
            const vqa::ExpertCacheStats s = c.stats();
            st.hits += s.hits; st.misses += s.misses; st.evictions += s.evictions;
            st.swap_bytes += s.swap_bytes; st.resident += s.resident; st.n_slots += s.n_slots;
        }
        const uint64_t acc = st.hits + st.misses;
        int in_flight = 0;
        for (auto & c : ec_caches) in_flight += c.in_flight();
        fprintf(stderr, "keye ec: %d decode steps, %.1f ms/step (A %.2f + B %.2f + swap-wait %.2f"
                " + readback %.2f), %.1f tok/s | cache: %llu hits / %llu misses (%.1f%% hit),"
                " %llu evictions, %.2f GB swapped, %d/%d slots resident, in_flight %d (async %s)\n",
                ec_st.steps, ec_st.ms_fin / ec_st.steps, ec_st.ms_a / ec_st.steps, ec_st.ms_b / ec_st.steps,
                ec_st.ms_swap / ec_st.steps, ec_st.ms_rb / ec_st.steps,
                (double)(ec_st.steps * 1000) / ec_st.ms_fin,
                (unsigned long long)st.hits, (unsigned long long)st.misses,
                acc ? 100.0 * (double)st.hits / (double)acc : 0.0,
                (unsigned long long)st.evictions, st.swap_bytes / 1e9, st.resident, st.n_slots,
                in_flight, ec_hooks.async_ok ? "on" : "off/v1");
    }
    return true;
}

#include "keye_expert_decode.inc"

}  // namespace keye_vl

namespace vqa {
Model * create_keye_vl() { return new keye_vl::KeyeVL2(); }
}  // namespace vqa
