// qwen3omni_model.cpp - Qwen3-Omni-30B-A3B thinker backend (text-first port).
//
// Architecture truth (transcribed from the official transformers 5.20.1
// modeling_qwen3_omni_moe.py; do NOT re-derive from the Qwen3-Next family -
// this checkpoint is NOT a GDN hybrid):
//   text   48 decoder layers, ALL full attention. GQA 32:4, head_dim 128,
//          hidden 2048, per-head q/k RMSNorm (plain x*w), interleaved MRoPE
//          (GGML_ROPE_TYPE_IMROPE, sections 24/20/20 over all 128 rotated
//          dims, theta 1e6), untied lm_head (vocab 152064).
//          FFN = routed MoE every layer: 128 experts, top-8,
//          norm_topk_prob=True (softmax over the selected logits, same as the
//          Keye backend's formula), moe_inter 768, NO shared expert.
//   vision Qwen3-VL ViT (27 blocks, deepstack taps at 8/16/24) - wired in a
//          later stage; the weights are present in the GGUF but not bound yet.
//   audio  Qwen3-ASR-family tower (32x1280, chunked window attention) - same.
//
// This file implements the TEXT path only (SOP D3/D4): load, ChatML ask(),
// generation. Placement: the Q8_0 checkpoint is ~32 GB on a 32 GB-RAM host, so
// the default is the pure-CPU backend with every weight zero-copy mmap'd
// (bind_from_mmap; pages stream in on demand). QOMNI_GPU=1 tries a GPU upload
// with the Keye-style expert split for a later stage.
//
// Weight layout (convert/convert_qwen3omni.py):
//   token_embd.weight [2048, vocab] Q8   output_norm.weight [2048] F16
//   lm_head.weight    [2048, vocab] Q8   (untied)
//   blk.{i}.attn_norm / attn_q / attn_k / attn_v / attn_output
//          / attn_q_norm / attn_k_norm / ffn_norm
//   blk.{i}.ffn_router.weight [D, n_exp] F32 (NOTE: stored as HF [n_exp, D],
//          ggml ne = [D, E])
//   blk.{i}.ffn_{gate,up}_exps.weight  ne = [D, I, E]   Q8_0 stacked
//   blk.{i}.ffn_down_exps.weight       ne = [I, D, E]   Q8_0 stacked
#include "core/vqa.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#   include "ggml-cuda.h"
#endif
#include "core/text/tokenizer.hpp"
#include "core/engine/attention.hpp"
#include "core/engine/moe_decode_driver.hpp"
#include "core/blocks/omni_speech.hpp"
#include "core/engine/sampler.hpp"
#include "core/util/aligned_buf.hpp"
#include "core/util/mmap.hpp"
#include "core/engine/backend.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/gguf_util.hpp"
#include "core/engine/stage_buf.hpp"
#include "core/runtime/load_util.hpp"
#include "core/engine/weight_load.hpp"
#include "core/engine/sched.hpp"
#include "core/engine/expert_cache.hpp"
#include "core/engine/moe_compute.hpp"
#include "core/engine/step_graph.hpp"
#include "core/engine/op_profile.hpp"
#include "core/blocks/qwen3vl_tower.hpp"
#include "core/media/resample.hpp"
#include "qwen3vl_preprocess.hpp"     // patchify3d + VisConfig (shared preprocess)
#include "qwen3omni_audio.hpp"
#include "qwen3omni_talker.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// internal (not in the frozen headers): the cgraph under construction during
// vqa::StepGraph::build() - used by the graph-A builder in
// qwen3omni_expert_decode.inc - and the residual-splitting forms of the
// generic routed-FFN builders (defined in moe_compute.cpp)
namespace vqa {
ggml_cgraph * step_graph_builder_gf();
namespace moe {
ggml_tensor * build_expert_ffn_res(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * x,
                                   ggml_tensor * res,
                                   const std::vector<ggml_tensor *> & gate,
                                   const std::vector<ggml_tensor *> & up,
                                   const std::vector<ggml_tensor *> & down,
                                   ggml_tensor * w, ggml_unary_op act);
ggml_tensor * build_expert_ffn_table_res(ggml_context * ctx, ggml_cgraph * gf,
                                         ggml_tensor * x, ggml_tensor * res,
                                         ggml_tensor * gate_table, ggml_tensor * up_table,
                                         ggml_tensor * down_table,
                                         ggml_tensor * gate_up_ids, ggml_tensor * down_ids,
                                         ggml_tensor * w, ggml_unary_op act);
}
}

namespace qwen3omni {

struct TxtLayer {
    ggml_tensor *norm = nullptr, *post_norm = nullptr;
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    ggml_tensor *qn = nullptr, *kn = nullptr;
    ggml_tensor *ffn_norm = nullptr;
    ggml_tensor *router = nullptr;                    // [D, n_exp] F32
    ggml_tensor *gate_exps = nullptr, *up_exps = nullptr, *down_exps = nullptr;
};

struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *mpos_in = nullptr, *mask_in = nullptr;
    ggml_tensor *img_in = nullptr, *img_pos_in = nullptr;
    ggml_tensor *logits = nullptr;
    // talker trajectory capture (dup'd graph outputs: gallocr-safe)
    ggml_tensor *emb_out = nullptr;    // post-injection embedding row(s)
    ggml_tensor *h24_out = nullptr;    // hidden after decoder layer accept_hidden_layer-1
};

class Qwen3Omni : public vqa::Model {
public:
    Qwen3Omni() = default;
    ~Qwen3Omni() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "qwen3-omni"; }

    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp,
             std::string & out_text) override;

    std::string format_history_turn(const std::string & user,
                                    const std::string & assistant) const override {
        return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
             + assistant + "<|im_end|>\n";
    }
    std::string format_system(const std::string & sys) const override {
        return sys.empty() ? std::string()
                           : "<|im_start|>system\n" + sys + "<|im_end|>\n";
    }

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all = false,
                         bool for_talker = false);
    bool load_talker();
    bool encode_vision(const vqa::MediaInput & media, std::vector<float> & feats,
                       int * n_tokens, int * grid, int token_budget);
    bool run_text(TextGraph & g, const std::vector<int> & toks,
                  const std::vector<float> * img_feats, const std::vector<int> * img_pos,
                  const std::vector<int> & mpos, std::vector<float> & logits_out,
                  std::vector<float> * emb_all = nullptr,
                  std::vector<float> * h24_all = nullptr);
    bool generate(const std::vector<int> & ids, const std::vector<float> * feats,
                  const std::vector<int> * img_pos, const std::vector<int> & mpos,
                  const vqa::GenParams & gp, std::string & out_text);
    int sample(const std::vector<float> & logits, const vqa::GenParams & gp);
    void dump(ggml_cgraph * gf, const char * name);

    // QOMNI_EXPERTCACHE=1: expert-granularity GPU-resident dynamic decode
    // (see qwen3omni_expert_decode.inc, same design as the keye integration).
    // ec_enabled is the request, ec_ok the initialized state; decode falls
    // back to the existing path whenever ec_ok is false.
    bool ec_enabled = false, ec_ok = false;
    std::vector<vqa::ExpertCache> ec_caches;   // one per quant-layout group
    std::vector<int> ec_group;                 // layer -> group index
    int ec_n_groups = 0;
    vqa::moe_decode::State ec_st;              // driver-owned decode state
    vqa::moe_decode::Hooks ec_hooks;           // filled once in ec_init
    std::vector<float> * ec_traj_emb = nullptr, * ec_traj_h24 = nullptr;
    // fixed per-layer graph-A holders (StepGraph, built once, re-fed every
    // step; S is block-aligned so shapes never change mid-block)
    struct EcGraphA {
        ggml_tensor *tok_in = nullptr, *hid_in = nullptr, *mpos_in = nullptr,
                    *mask_in = nullptr;
        ggml_tensor *w8_o = nullptr, *sel_o = nullptr, *m_o = nullptr, *x_o = nullptr;
    };
    struct EcGaUd { Qwen3Omni * m; int il; EcGraphA * io; };
    std::vector<EcGraphA> ec_ga_io;            // model leaves per layer
    std::vector<ggml_fp16_t> ec_mask;   // per-step causal mask cache
    int ec_mask_n = -1;
    bool ec_init();
    static ggml_tensor * ec_ga_builder(ggml_context * c, void * ud);
    static bool ec_build_a(void * m, int il, vqa::StepGraph & sg,
                           vqa::moe_decode::AIO & io);
    static bool ec_feed_a(void * m, int il, vqa::moe_decode::AIO & io,
                          vqa::StepGraph & sg, const void * step_ctx,
                          int cur_tok, int n_past, const float * hid);
    static void ec_post_a(void * m, int il, const float * xmid, const float * mn,
                          const int * sel, const float * w8);
    static void ec_post_b(void * m, int il, const float * hid);
    bool ec_decode_step(int cur_tok, const int * mpos4, int n_past,
                        std::vector<float> & logits,
                        std::vector<float> * emb_out = nullptr,
                        std::vector<float> * h24_out = nullptr);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    int n_layer = 48, n_embd = 2048, n_head = 32, n_kv_head = 4, head_dim = 128;
    int n_vocab = 152064;
    int n_expert = 128, n_expert_used = 8, n_ff_exp = 768;
    float rms_eps = 1e-6f, rope_theta = 1e6f;
    int rope_dim = 128;
    int mrope_sections[4] = {24, 20, 20, 0};

    int n_ctx = 8192;
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;
    std::string model_dir_hint;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;  // split mode: runs the expert mul_mat_ids
    bool split_mode = false;               // dense on GPU, experts in host mmap
    ggml_backend_buffer_t host_wbuf = nullptr;   // host weight buffer (mmap wrap, split mode)
    vqa::SplitSched split_sched;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    vqa::AlignedBuf host_f32_storage;
    vqa::MMap mm;

    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr, *out_w = nullptr;
    // ec split mode: host copy of tok_embd for the (CPU) deferred talker's
    // rows_any reads - reading a CUDA-buffered table from the CPU segfaults
    std::vector<uint8_t> tok_embd_host_mem;
    ggml_context * tok_embd_host_ctx = nullptr;
    ggml_backend_buffer_t tok_embd_host_buf = nullptr;
    ggml_tensor * tok_embd_host = nullptr;
    std::vector<TxtLayer> tl;
    ggml_tensor *v_patch_w = nullptr;

    int id_eos = 151645, id_endoftext = 151643;

    // ---- vision (Qwen3-VL tower, shared block) ----
    qwen3vl::VisConfig vc;
    vqa::qwen3vl_tower::Weights vt;
    std::vector<float> pos_table_host;     // [pos_side^2, vc.n_embd] F32
    std::vector<float> patch_bias;         // [vc.n_embd] F32 (folded into pos)
    vqa::StageBuffer tower_buf;            // [vc.n_embd, N]
    vqa::StageBuffer ds_buf[8];            // [n_embd, N] per DeepStack tap
    int n_ds = 0;
    int id_image_pad = 151655, id_vision_start = 151652, id_vision_end = 151653;
    qwen3omni::AudioTower at;
    qwen3omni::Talker talker;
    bool talker_on = false;
    bool talker_deferred = false;   // ec mode: load on first --speak (VRAM)
    int id_audio_pad = 151675;
    int a_n_layer = 32, a_n_embd = 1280, a_n_head = 20, a_n_ff = 5120;
    int a_n_window = 50, a_n_window_infer = 800, a_ds_hidden = 480;

    vqa::KvCache kv;
    ggml_gallocr_t allocr = nullptr;
    int n_past = 0;
    int last_mpos[3] = {0, 0, 0};
    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

Qwen3Omni::~Qwen3Omni() {
    kv.free();
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (allocr) ggml_gallocr_free(allocr);
    split_sched.free_sched();
    if (host_wbuf) ggml_backend_buffer_free(host_wbuf);
    if (tok_embd_host_buf) ggml_backend_buffer_free(tok_embd_host_buf);
    if (tok_embd_host_ctx) ggml_free(tok_embd_host_ctx);
    if (backend_cpu) ggml_backend_free(backend_cpu);
    if (backend) ggml_backend_free(backend);
}

bool Qwen3Omni::init_backend() {
    // text stage: CPU is the default target. QOMNI_GPU=1 uploads to the GPU;
    // QOMNI_EXPERTCACHE=1 additionally implies the Strata-style split (dense
    // on the GPU, experts left in the host mmap for ExpertCache slots)
    if (!getenv("QOMNI_GPU") && !ec_enabled) use_gpu = false;
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!ggml_backend_is_cpu(backend)) {
        // split partner: a CPU backend for the expert mul_mat_ids (prefill)
        backend_cpu = vqa::init_backend(false, n_threads);
        if (!backend_cpu) return false;
    }
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

bool Qwen3Omni::load(const std::string & gguf_path, const std::string & tokenizer_path,
                     const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "QOMNI", n_threads, use_gpu, dump_dir);
    {
        const size_t slash = gguf_path.find_last_of("/\\");
        model_dir_hint = (slash == std::string::npos) ? "." : gguf_path.substr(0, slash);
    }
    ec_enabled = getenv("QOMNI_EXPERTCACHE") != nullptr;
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::Qwen2Regex)) return false;
    if (tok_.size() > 0) {
        const int64_t diff = tok_.size() > n_vocab ? tok_.size() - (int64_t)n_vocab
                                                   : (int64_t)n_vocab - tok_.size();
        if (diff * 5 > n_vocab) {
            fprintf(stderr, "qwen3omni: tokenizer vocab %d vs model vocab %d - wrong tokenizer.json\n",
                    tok_.size(), n_vocab);
            return false;
        }
    }
    if (!kv.alloc(backend, n_layer, n_kv_head, head_dim, n_ctx)) return false;

    vqa::assign_token_ids(tok_, {
        { "<|image_pad|>",    &id_image_pad },
        { "<|vision_start|>", &id_vision_start },
        { "<|vision_end|>",   &id_vision_end },
        { "<|audio_pad|>",    &id_audio_pad },
    });
    id_eos = vqa::pick_token(tok_, "<|im_end|>", 151645);
    id_endoftext = vqa::pick_token(tok_, "<|endoftext|>", 151643);

    // speech output: sibling talker GGUF (qwen25omni two-file precedent).
    // ec mode defers the load to the first --speak: the talker would eat the
    // VRAM the expert slots need (see load_talker)
    if (ec_enabled) talker_deferred = true;
    else load_talker();

    if (ec_enabled && !ec_init()) return false;

    caps_ = {};
    caps_.image = !pos_table_host.empty();
    caps_.audio = !at.mel_filters_host.empty();

    fprintf(stderr, "qwen3omni: ready (%d layers, %d/%d experts, ctx %d)\n",
            n_layer, n_expert_used, n_expert, n_ctx);
    return true;
}

bool Qwen3Omni::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f GB\n", mm.size / 1e9);

    struct ggml_init_params wip = { 512ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    wctx = ggml_init(wip);
    if (!wctx) return false;
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer       = vqa::gguf::u32(gu, "qwen3omni.block_count", n_layer);
    n_embd        = vqa::gguf::u32(gu, "qwen3omni.embedding_length", n_embd);
    n_head        = vqa::gguf::u32(gu, "qwen3omni.attention.head_count", n_head);
    n_kv_head     = vqa::gguf::u32(gu, "qwen3omni.attention.head_count_kv", n_kv_head);
    head_dim      = vqa::gguf::u32(gu, "qwen3omni.attention.key_length", head_dim);
    n_vocab       = vqa::gguf::u32(gu, "qwen3omni.vocab_size", n_vocab);
    rms_eps       = vqa::gguf::f32(gu, "qwen3omni.attention.layer_norm_rms_epsilon", rms_eps);
    rope_theta    = vqa::gguf::f32(gu, "qwen3omni.rope.freq_base", rope_theta);
    n_expert      = vqa::gguf::u32(gu, "qwen3omni.expert_count", n_expert);
    n_expert_used = vqa::gguf::u32(gu, "qwen3omni.expert_used_count", n_expert_used);
    n_ff_exp      = vqa::gguf::u32(gu, "qwen3omni.expert_feed_forward_length", n_ff_exp);
    mrope_sections[0] = vqa::gguf::u32(gu, "qwen3omni.rope.mrope_section_0", mrope_sections[0]);
    mrope_sections[1] = vqa::gguf::u32(gu, "qwen3omni.rope.mrope_section_1", mrope_sections[1]);
    mrope_sections[2] = vqa::gguf::u32(gu, "qwen3omni.rope.mrope_section_2", mrope_sections[2]);

    vc.n_layer        = vqa::gguf::u32(gu, "qwen3omni.vision.block_count", vc.n_layer);
    vc.n_embd         = vqa::gguf::u32(gu, "qwen3omni.vision.embedding_length", vc.n_embd);
    vc.n_ff           = vqa::gguf::u32(gu, "qwen3omni.vision.feed_forward_length", vc.n_ff);
    vc.n_head         = vqa::gguf::u32(gu, "qwen3omni.vision.attention.head_count", vc.n_head);
    vc.patch          = vqa::gguf::u32(gu, "qwen3omni.vision.patch_size", vc.patch);
    vc.temporal_patch = vqa::gguf::u32(gu, "qwen3omni.vision.temporal_patch_size", vc.temporal_patch);
    vc.merge          = vqa::gguf::u32(gu, "qwen3omni.vision.merge_size", vc.merge);
    vc.layernorm_eps  = vqa::gguf::f32(gu, "qwen3omni.vision.layer_norm_epsilon", vc.layernorm_eps);
    vc.out_hidden     = vqa::gguf::u32(gu, "qwen3omni.vision.out_hidden_size", vc.out_hidden);
    for (int k = 0; k < 8; k++) {
        char key[64];
        snprintf(key, sizeof(key), "qwen3omni.vision.deepstack_index_%d", k);
        const int64_t ki = gguf_find_key(gu, key);
        if (ki < 0) break;
        vc.deepstack.push_back((int)vqa::gguf::u32(gu, key, 0));
    }
    a_n_layer        = vqa::gguf::u32(gu, "qwen3omni.audio.block_count", a_n_layer);
    a_n_embd         = vqa::gguf::u32(gu, "qwen3omni.audio.embedding_length", a_n_embd);
    a_n_ff           = vqa::gguf::u32(gu, "qwen3omni.audio.feed_forward_length", a_n_ff);
    a_n_head         = vqa::gguf::u32(gu, "qwen3omni.audio.attention.head_count", a_n_head);
    a_n_window       = vqa::gguf::u32(gu, "qwen3omni.audio.n_window", a_n_window);
    a_n_window_infer = vqa::gguf::u32(gu, "qwen3omni.audio.n_window_infer", a_n_window_infer);
    a_ds_hidden      = vqa::gguf::u32(gu, "qwen3omni.audio.downsample_hidden_size", a_ds_hidden);

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);
    vqa::weights::promote_1d_to_f32(wctx, host_f32_storage);

    // CPU backend: wrap the mmap'd weights in a from_ptr buffer (no copy, the
    // 32 GB Q8_0 stream pages in on demand). GPU: materialize into VRAM/limits
    // are a later stage's problem. QOMNI_EXPERTCACHE on a CUDA backend takes
    // the Keye-style split instead: dense weights up, expert tensors kept in
    // the host mmap (the cache slots mirror them into VRAM per step).
    bool want_split = false;
#ifdef GGML_USE_CUDA
    if (backend_cpu && ec_enabled && !ggml_backend_is_cpu(backend)) {
        size_t dev_total = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(wctx); t; t = ggml_get_next_tensor(wctx, t))
            if (!vqa::weights::moe_expert_pred(t->name)) dev_total += ggml_nbytes(t);
        size_t free_vram = 0, total_vram = 0;
        ggml_backend_cuda_get_device_memory(0, &free_vram, &total_vram);
        // dense weights + KV (~0.9 GB) + talker (~4.9 GB on its own backend) +
        // compute headroom
        if (dev_total + (size_t)2.5e9 <= free_vram) want_split = true;
        fprintf(stderr, "qomni split: dense %.2f GB, free VRAM %.2f GB -> %s\n",
                dev_total / 1e9, free_vram / 1e9,
                want_split ? "experts on CPU" : "no split (too little VRAM)");
    }
#endif
    if (want_split) {
        vqa::weights::SplitBound sb = vqa::weights::split_bind_to_backends(
            wctx, backend, mm, vqa::weights::moe_expert_pred,
            { { host_f32_storage.data(), host_f32_storage.size() } });
        if (!sb.ok) return false;
        wbuf = sb.dev_buf;
        host_wbuf = sb.host_buf;
        ggml_backend_buffer_set_usage(sb.dev_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_set_usage(sb.host_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        split_mode = true;
        if (!split_sched.init(backend, backend_cpu, 32768)) return false;
    } else {
    vqa::weights::Bound bound = vqa::weights::bind_to_backend(
        wctx, backend, mm, { { host_f32_storage.data(), host_f32_storage.size() } });
    if (!bound.ok) return false;
    wbuf = bound.main_buf;
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (!ggml_backend_is_cpu(backend)) mm.release();
    }

    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) fprintf(stderr, "missing tensor: %s\n", n);
        return t;
    };
    tok_embd = gt("token_embd.weight");
    out_norm = gt("output_norm.weight");
    out_w    = gt("lm_head.weight");
    if (!out_w) out_w = gt("output.weight");     // converter alias

    tl.resize((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "blk.%d.%s", i, suf); return gt(b); };
        TxtLayer & L = tl[(size_t)i];
        L.norm      = g("attn_norm.weight");
        L.q = g("attn_q.weight");   L.k = g("attn_k.weight");
        L.v = g("attn_v.weight");   L.o = g("attn_output.weight");
        L.qn = g("attn_q_norm.weight"); L.kn = g("attn_k_norm.weight");
        L.ffn_norm  = g("ffn_norm.weight");
        L.router    = g("ffn_router.weight");
        L.gate_exps = g("ffn_gate_exps.weight");
        L.up_exps   = g("ffn_up_exps.weight");
        L.down_exps = g("ffn_down_exps.weight");
        if (!L.norm || !L.q || !L.k || !L.v || !L.o || !L.qn || !L.kn || !L.ffn_norm ||
            !L.router || !L.gate_exps || !L.up_exps || !L.down_exps)
            return false;
    }

    // ---------------- vision tower (shared qwen3vl_tower block) -------------
    v_patch_w = gt("v.patch_embd.weight");
    ggml_tensor * v_patch_b = gt("v.patch_embd.bias");
    ggml_tensor * v_pos = gt("v.pos_emb.weight");
    n_ds = 0;
    const bool has_vision = v_patch_w && v_patch_b && v_pos;
    if (has_vision) {
        vt.patch_w = v_patch_w;
        vt.layers.resize((size_t)vc.n_layer);
        for (int i = 0; i < vc.n_layer; i++) {
            char b[96];
            auto g = [&](const char * suf) { snprintf(b, sizeof(b), "v.blk.%d.%s", i, suf); return gt(b); };
            vqa::qwen3vl_tower::Layer & L = vt.layers[i];
            L.norm0_w = g("norm1_weight"); L.norm0_b = g("norm1_bias");
            L.norm1_w = g("norm2_weight"); L.norm1_b = g("norm2_bias");
            L.wqkv_w  = g("attn_qkv_weight"); L.wqkv_b = g("attn_qkv_bias");
            L.wo_w    = g("attn_proj_weight"); L.wo_b = g("attn_proj_bias");
            L.fc0_w   = g("mlp_linear_fc1_weight"); L.fc0_b = g("mlp_linear_fc1_bias");
            L.fc1_w   = g("mlp_linear_fc2_weight"); L.fc1_b = g("mlp_linear_fc2_bias");
            if (!L.norm0_w || !L.wqkv_w || !L.fc0_w) return false;
        }
        vt.main = {gt("mm.ln_q.weight"), gt("mm.ln_q.bias"),
                   gt("mm.fc1.weight"), gt("mm.fc1.bias"),
                   gt("mm.fc2.weight"), gt("mm.fc2.bias")};
        if (!vt.main.norm_w || !vt.main.fc1_w) return false;
        for (int k = 0; k < 8; k++) {
            char pre[32];
            snprintf(pre, sizeof(pre), "mm_ds.%d.", k);
            char n1[96]; snprintf(n1, sizeof(n1), "%sln_q.weight", pre);
            if (!ggml_get_tensor(wctx, n1)) break;
            vqa::qwen3vl_tower::Merger M;
            char n2[96], n3[96], n4[96], n5[96], n6[96];
            snprintf(n2, sizeof(n2), "%sln_q.bias", pre);
            snprintf(n3, sizeof(n3), "%sfc1.weight", pre);
            snprintf(n4, sizeof(n4), "%sfc1.bias", pre);
            snprintf(n5, sizeof(n5), "%sfc2.weight", pre);
            snprintf(n6, sizeof(n6), "%sfc2.bias", pre);
            M = {gt(n1), gt(n2), gt(n3), gt(n4), gt(n5), gt(n6)};
            if (!M.norm_w || !M.fc1_w) return false;
            vt.ds.push_back(M);
            n_ds++;
        }
        if (n_ds != (int)vc.deepstack.size()) {
            fprintf(stderr, "qwen3omni vision: %d ds mergers vs %d indexes\n",
                    n_ds, (int)vc.deepstack.size());
            return false;
        }
        // pos table rows live in ne[1] ([rows, width] ggml ne order)
        if ((int64_t)vc.pos_side * vc.pos_side != v_pos->ne[1]) {
            vc.pos_side = (int)std::lround(sqrtf((float)v_pos->ne[1]));
            if ((int64_t)vc.pos_side * vc.pos_side != v_pos->ne[1]) {
                fprintf(stderr, "qwen3omni vision: pos table %lld is not square\n",
                        (long long)v_pos->ne[1]);
                return false;
            }
        }
        pos_table_host.resize((size_t)v_pos->ne[0] * v_pos->ne[1]);
        ggml_backend_tensor_get(v_pos, pos_table_host.data(), 0,
                                sizeof(float) * pos_table_host.size());
        patch_bias.resize(ggml_nelements(v_patch_b));
        if (v_patch_b->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(v_patch_b, patch_bias.data(), 0,
                                    sizeof(float) * patch_bias.size());
        } else {
            std::vector<uint16_t> tmp(patch_bias.size());
            ggml_backend_tensor_get(v_patch_b, tmp.data(), 0, tmp.size() * 2);
            for (size_t i = 0; i < tmp.size(); i++)
                patch_bias[i] = ggml_fp16_to_fp32(tmp[i]);
        }
        fprintf(stderr, "qwen3omni vision: %d blocks d=%d patch=%d/%d merge=%d "
                "pos %lld + %d deepstack taps\n", vc.n_layer, vc.n_embd, vc.patch,
                vc.temporal_patch, vc.merge, (long long)v_pos->ne[1], n_ds);
    }

    // ---------------- audio tower (Qwen3-ASR family) ------------------------
    if (ggml_get_tensor(wctx, "a.blk.0.attn_q.weight")) {
        at.n_layer = a_n_layer; at.n_embd = a_n_embd; at.n_head = a_n_head;
        at.n_ff = a_n_ff; at.n_window = a_n_window;
        at.n_window_infer = a_n_window_infer; at.ds_hidden = a_ds_hidden;
        at.backend = backend; at.allocr = allocr;
        at.conv1_w = gt("a.conv1.weight"); at.conv1_b = gt("a.conv1.bias");
        at.conv2_w = gt("a.conv2.weight"); at.conv2_b = gt("a.conv2.bias");
        at.conv3_w = gt("a.conv3.weight"); at.conv3_b = gt("a.conv3.bias");
        at.conv_out_w = gt("a.conv_out.weight");
        at.ln_post_w = gt("a.ln_post.weight"); at.ln_post_b = gt("a.ln_post.bias");
        at.proj1_w = gt("a.proj1.weight"); at.proj1_b = gt("a.proj1.bias");
        at.proj2_w = gt("a.proj2.weight"); at.proj2_b = gt("a.proj2.bias");
        if (!at.conv1_w || !at.conv_out_w || !at.ln_post_w || !at.proj2_w) return false;
        at.layers.resize((size_t)a_n_layer);
        for (int i = 0; i < a_n_layer; i++) {
            char b[96];
            auto g = [&](const char * suf) { snprintf(b, sizeof(b), "a.blk.%d.%s", i, suf); return gt(b); };
            qwen3omni::AudLayer & L = at.layers[i];
            L.ln1_w = g("attn_ln.weight"); L.ln1_b = g("attn_ln.bias");
            L.wq = g("attn_q.weight"); L.bq = g("attn_q.bias");
            L.wk = g("attn_k.weight"); L.bk = g("attn_k.bias");
            L.wv = g("attn_v.weight"); L.bv = g("attn_v.bias");
            L.wo = g("attn_o.weight"); L.bo = g("attn_o.bias");
            L.ln2_w = g("ffn_ln.weight"); L.ln2_b = g("ffn_ln.bias");
            L.fc1_w = g("ffn_fc1.weight"); L.fc1_b = g("ffn_fc1.bias");
            L.fc2_w = g("ffn_fc2.weight"); L.fc2_b = g("ffn_fc2.bias");
            if (!L.ln1_w || !L.wq || !L.fc1_w) return false;
        }
        // host F32 copies for the conv stack (F16 in GGUF)
        // NOTE: promote_1d_to_f32 already turned every 1D F16 (the conv
        // biases) into F32 - read by tensor type, not by assumption
        auto hget = [&](ggml_tensor * t, std::vector<float> & dst) {
            dst.resize(ggml_nelements(t));
            if (t->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(t, dst.data(), 0, sizeof(float) * dst.size());
            } else {
                std::vector<uint16_t> tmp(dst.size());
                ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size() * 2);
                for (size_t i = 0; i < tmp.size(); i++)
                    dst[i] = ggml_fp16_to_fp32(tmp[i]);
            }
        };
        hget(at.conv1_w, at.host_c1w); hget(at.conv1_b, at.host_c1b);
        hget(at.conv2_w, at.host_c2w); hget(at.conv2_b, at.host_c2b);
        hget(at.conv3_w, at.host_c3w); hget(at.conv3_b, at.host_c3b);
        hget(at.conv_out_w, at.host_cow);
        // mel filterbank sidecar next to the GGUF
        {
            std::string p = model_dir_hint + "/mel_filters_128_201.f32";
            FILE * fp = fopen(p.c_str(), "rb");
            if (fp) {
                at.mel_filters_host.resize((size_t)128 * 201);
                if (fread(at.mel_filters_host.data(), 4, at.mel_filters_host.size(), fp)
                        == at.mel_filters_host.size())
                    fprintf(stderr, "qwen3omni audio: mel filters sidecar loaded\n");
                else
                    at.mel_filters_host.clear();
                fclose(fp);
            }
            if (at.mel_filters_host.empty())
                fprintf(stderr, "qwen3omni audio: mel_filters_128_201.f32 missing "
                        "next to the GGUF - audio input disabled\n");
        }
    }

    fprintf(stderr, "qwen3omni: %d layers d=%d, experts %d/%d (ff %d)%s\n",
            n_layer, n_embd, n_expert_used, n_expert, n_ff_exp,
            has_vision ? " + vision" : "");
    return tok_embd && out_norm && out_w;
}

bool Qwen3Omni::load_talker() {
    for (const char * suffix : {"-q8_0", "-q6_k", ""}) {
        std::string p = model_dir_hint + "/qwen3omni-talker" +
                        (suffix[0] ? std::string(suffix) : std::string()) + ".gguf";
        FILE * fp = fopen(p.c_str(), "rb");
        if (!fp) continue;
        fclose(fp);
        // talker on its own CUDA backend when the build has one: the 4.9GB
        // speech stack fits any modern card even though the 27GB thinker
        // stays on CPU. QOMNI_TALKER_CPU=1 forces the old shared placement.
        // Deferred (ec) load: the expert slots already hold the VRAM, so the
        // talker goes to the CPU partner backend instead.
        ggml_backend_t tb = backend;
        ggml_gallocr_t ta = allocr;
#ifdef GGML_USE_CUDA
        const bool cpu_talker = talker_deferred && ec_ok;
        if (!getenv("QOMNI_TALKER_CPU") && !cpu_talker) {
            ggml_backend_t cuda = ggml_backend_cuda_init(0);
            if (cuda) {
                fprintf(stderr, "qwen3omni: talker on its own %s\n", ggml_backend_name(cuda));
                tb = cuda;
                ta = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cuda));
            }
        } else if (cpu_talker && backend_cpu) {
            tb = backend_cpu;
            ta = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_cpu));
            fprintf(stderr, "qwen3omni: talker on CPU (expert slots hold the VRAM)\n");
        }
#endif
        if (talker.load(p.c_str(), tb, ta)) talker_on = true;
        talker.host = backend_cpu ? backend_cpu : backend;   // CPU partner for quantized-table reads
        break;
    }
    return talker_on;
}

// ---------------------------------------------------------------- text
bool Qwen3Omni::build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all,
                                 bool for_talker) {
    // TextGraph is reused across calls but tensors live in the per-call ctx:
    // stale pointers into the freed previous ctx would skip the rebuild
    g.emb_out = nullptr;
    g.h24_out = nullptr;
    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 32768, false);

    const int D = n_embd, HD = head_dim, NH = n_head, NKV = n_kv_head;
    ggml_context * c = g.ctx;

    g.tok_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    // [t(T), h(T), w(T), slot(T)] - interleaved MRoPE planes + absolute KV slot
    g.mpos_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * T);
    ggml_set_name(g.mpos_in, "mpos_in");
    g.mask_in = ggml_new_tensor_2d(c, GGML_TYPE_F16, S, T);
    ggml_set_name(g.mask_in, "mask_in");

    ggml_tensor * x = ggml_get_rows(c, tok_embd, g.tok_in);            // [D, T]
    if (n_img > 0) {
        g.img_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_embd, n_img);
        ggml_set_name(g.img_in, "img_in");
        g.img_pos_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_img);
        ggml_set_name(g.img_pos_in, "img_pos_in");
        x = ggml_set_rows(c, ggml_dup(c, x), g.img_in, g.img_pos_in);
    }
    ggml_set_name(x, "emb");
    if (for_talker) {
        g.emb_out = ggml_dup(c, x);
        ggml_set_name(g.emb_out, "t_emb_out");
        ggml_build_forward_expand(g.gf, g.emb_out);
    }

    // NOTE: pos_rope is mpos_in ITSELF, not an identity view - the split
    // scheduler only assigns graph-input leafs a backend when they are a DIRECT
    // src of a node (a view-wrapped input never gets one -> gallocr assert)
    ggml_tensor * pos_rope = g.mpos_in;
    ggml_tensor * pos_slot = ggml_view_1d(c, g.mpos_in, T, (size_t)3 * T * sizeof(int));

    int max_layers = n_layer;
    if (const char * e = getenv("QOMNI_MAX_LAYERS")) max_layers = std::min(max_layers, atoi(e));

    for (int il = 0; il < max_layers; il++) {
        TxtLayer & L = tl[il];
        ggml_tensor * l = ggml_rms_norm(c, x, rms_eps);
        l = ggml_mul(c, l, L.norm);
        if (il == 0) ggml_set_name(l, "t_l0_norm");

        ggml_tensor * q = ggml_mul_mat(c, L.q, l);                      // [NH*HD, T]
        ggml_tensor * k = ggml_mul_mat(c, L.k, l);
        ggml_tensor * v = ggml_mul_mat(c, L.v, l);

        ggml_tensor * q3 = ggml_reshape_3d(c, q, HD, NH, T);
        q3 = ggml_rms_norm(c, q3, rms_eps);
        q3 = ggml_mul(c, q3, L.qn);
        ggml_tensor * k3 = ggml_reshape_3d(c, k, HD, NKV, T);
        k3 = ggml_rms_norm(c, k3, rms_eps);
        k3 = ggml_mul(c, k3, L.kn);

        // interleaved MRoPE: all 128 head dims rotated, sections 24/20/20.
        // q and k share (pos, sections, theta): ONE rope over the
        // head-concatenated tensor, then view-split. NOTE view_3d(ctx, t,
        // ne0, ne1, ne2, s1, s2, off) - s1/s2 are the strides of THIS view's
        // dims 1/2, so for [HD, heads, T] they are (qk->nb[1], qk->nb[2]);
        // swapping them transposes head/T and silently garbles everything
        // (the first fusion attempt died exactly there - not a ggml bug).
        static const bool nofuse = getenv("QOMNI_NOFUSE") != nullptr;
        if (nofuse) {
        q3 = ggml_rope_multi(c, q3, pos_rope, nullptr, rope_dim, mrope_sections,
                             GGML_ROPE_TYPE_IMROPE, n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_multi(c, k3, pos_rope, nullptr, rope_dim, mrope_sections,
                             GGML_ROPE_TYPE_IMROPE, n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        } else {
        ggml_tensor * qk = ggml_concat(c, q3, k3, 1);              // [HD, NH+NKV, T]
        qk = ggml_rope_multi(c, qk, pos_rope, nullptr, rope_dim, mrope_sections,
                             GGML_ROPE_TYPE_IMROPE, n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        q3 = ggml_view_3d(c, qk, HD, NH, T, qk->nb[1], qk->nb[2], 0);
        k3 = ggml_view_3d(c, qk, HD, NKV, T, qk->nb[1], qk->nb[2],
                          (size_t)NH * qk->nb[1]);
        }
        if (il == 0) { ggml_set_name(q3, "t_l0_q_rope"); ggml_set_name(k3, "t_l0_k_rope"); }

        ggml_tensor * k2 = ggml_reshape_2d(c, ggml_cont(c, k3), NKV * HD, T);
        ggml_tensor * v2 = ggml_reshape_2d(c, v, NKV * HD, T);
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, kv.k[il], k2, pos_slot));
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, kv.v[il], v2, pos_slot));

        ggml_tensor * kf = ggml_view_3d(c, kv.k[il], HD, S, NKV,
                                        kv.k[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * vf = ggml_view_3d(c, kv.v[il], HD, S, NKV,
                                        kv.v[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * qf = ggml_view_3d(c, q3, HD, T, NH, q3->nb[2], q3->nb[1], 0);
        ggml_tensor * fa = ggml_flash_attn_ext(c, qf, kf, vf, g.mask_in,
                                               1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(c, ggml_cont(c, fa), NH * HD, T);
        if (il == 0) ggml_set_name(attn, "t_l0_attn");
        attn = ggml_mul_mat(c, L.o, attn);
        x = ggml_add(c, x, attn);
        if (il == 0) ggml_set_name(x, "t_l0_resid");

        // ---- MoE FFN: softmax gating over the top-k selected logits (norm_topk) ----
        ggml_tensor * m = ggml_rms_norm(c, x, rms_eps);
        m = ggml_mul(c, m, L.ffn_norm);
        if (il == 0) ggml_set_name(m, "t_l0_m");
        ggml_tensor * rlogits = ggml_mul_mat(c, L.router, m);           // [E, T]
        if (il == 0) ggml_set_name(rlogits, "t_l0_rlogits");
        ggml_tensor * sel = ggml_top_k(c, rlogits, n_expert_used);      // [K, T] I32
        ggml_tensor * sel_logits = ggml_get_rows(c,
                ggml_reshape_3d(c, rlogits, 1, n_expert, T), sel);      // [1, K, T]
        sel_logits = ggml_reshape_2d(c, sel_logits, n_expert_used, T);
        ggml_tensor * w8 = ggml_soft_max(c, sel_logits);                // [K, T]
        w8 = ggml_reshape_3d(c, w8, 1, n_expert_used, T);

        ggml_tensor * cur3 = ggml_reshape_3d(c, m, D, 1, T);
        ggml_tensor * gu = ggml_mul_mat_id(c, L.gate_exps, cur3, sel);  // [I, K, T]
        ggml_tensor * up = ggml_mul_mat_id(c, L.up_exps, cur3, sel);    // [I, K, T]
        ggml_tensor * act = ggml_swiglu_split(c, gu, up);
        if (il == 0) { ggml_set_name(gu, "t_l0_gu"); ggml_set_name(up, "t_l0_up");
                       ggml_set_name(act, "t_l0_act"); ggml_set_name(sel, "t_l0_sel"); }
        ggml_tensor * ff3 = ggml_mul_mat_id(c, L.down_exps, act, sel);  // [D, K, T]
        ggml_tensor * wb = ggml_view_3d(c, w8, 1, n_expert_used, T, w8->nb[1], w8->nb[2], 0);
        ggml_tensor * prod = ggml_mul(c, ff3, wb);                      // [D, K, T]
        ggml_tensor * pk = ggml_cont(c, ggml_permute(c, prod, 1, 0, 2, 3)); // [K, D, T]
        ggml_tensor * moe = ggml_reshape_2d(c, ggml_sum_rows(c, pk), D, T);
        if (il == 0) ggml_set_name(moe, "t_l0_moe");
        x = ggml_add(c, x, moe);
        if (il == 0) ggml_set_name(x, "t_l0_out");

        // talker trajectory: thinker hidden at accept_hidden_layer (HF takes
        // hidden_states[24] = output of decoder layer 23)
        if (for_talker && il == 23 && !g.h24_out) {
            g.h24_out = ggml_dup(c, x);
            ggml_set_name(g.h24_out, "t_h24_out");
            ggml_build_forward_expand(g.gf, g.h24_out);
        }

        // DeepStack: add tap il's merged tower features at the visual token
        // positions only (zero elsewhere), after decoder layer il
        if (n_ds > 0 && il < n_ds && n_img > 0 && ds_buf[il].t && ds_buf[il].t->ne[1] >= n_img) {
            ggml_tensor * dk = ggml_view_2d(c, ds_buf[il].t, n_embd, n_img,
                                            ds_buf[il].t->nb[1], 0);
            ggml_tensor * z = ggml_set_rows(c, ggml_scale(c, x, 0.0f), dk, g.img_pos_in);
            if (il == 0) ggml_set_name(z, "t_l0_deepstack");
            x = ggml_add(c, x, z);
        }
    }

    x = ggml_rms_norm(c, x, rms_eps);
    x = ggml_mul(c, x, out_norm);
    ggml_set_name(x, "t_final_norm");

    if (logits_all) {
        g.logits = ggml_mul_mat(c, out_w, x);
    } else {
        ggml_tensor * last = ggml_view_2d(c, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
        g.logits = ggml_mul_mat(c, out_w, last);
    }
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "emb", "t_l0_norm", "t_l0_q_rope", "t_l0_k_rope",
                                           "t_l0_attn", "t_l0_resid", "t_l0_m", "t_l0_rlogits", "t_l0_sel", "t_l0_gu",
                                           "t_l0_up", "t_l0_act", "t_l0_moe", "t_l0_out",
                                           "t_final_norm", nullptr };
        vqa::dump::mark_sinks(c, g.gf, dn);
        vqa::dump::finalize_sinks(c, g.gf, backend);
    }
    return true;
}

bool Qwen3Omni::run_text(TextGraph & g, const std::vector<int> & toks,
                         const std::vector<float> * img_feats, const std::vector<int> * img_pos,
                         const std::vector<int> & mpos, std::vector<float> & logits_out,
                         std::vector<float> * emb_all, std::vector<float> * h24_all) {
    const int T = (int)toks.size();
    const int S = n_past + T;
    if (S > n_ctx) { fprintf(stderr, "context overflow (%d > %d)\n", S, n_ctx); return false; }
    const int n_img = (img_feats && img_pos) ? (int)(img_feats->size() / n_embd) : 0;

    const auto tb0 = std::chrono::steady_clock::now();
    const bool for_talker = (emb_all || h24_all);
    if (for_talker) fprintf(stderr, "[dbg] run_text T=%d for_talker\n", T);
    if (!build_text_graph(g, T, S, n_img, false, for_talker)) return false;
    const auto tb1 = std::chrono::steady_clock::now();
    if (split_mode ? !split_sched.alloc(g.gf)
                   : !ggml_gallocr_alloc_graph(allocr, g.gf)) {
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
    const auto tb2 = std::chrono::steady_clock::now();
    vqa::set_causal_mask(g.mask_in, n_past, T);
    static const bool prof = getenv("QOMNI_PROFILE") != nullptr;
    static vqa::op_profile::State prof_state;
    static int prof_step = 0;
    bool ok;
    if (prof) {
        ok = vqa::op_profile::profile_graph(backend, g.gf, prof_state,
                                            T > 1 ? "prefill" : "decode",
                                            prof_step++, T > 1 ? 1000000 : 32);
    } else {
        ok = split_mode ? split_sched.run(g.gf)
                        : ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS;
    }
    if (!ok) {
        fprintf(stderr, "text compute failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }
    static const bool dump_prefill_only = getenv("QOMNI_PREFILL_DUMP_ONLY") != nullptr;
    if (!dump_prefill_only || T > 1) {
        dump(g.gf, "emb");
        dump(g.gf, "t_l0_attn");
        dump(g.gf, "t_l0_resid");
        dump(g.gf, "t_l0_m");
        dump(g.gf, "t_l0_rlogits");
        dump(g.gf, "t_l0_act");
        dump(g.gf, "t_l0_moe");
        dump(g.gf, "t_l0_out");
        dump(g.gf, "t_final_norm");
        dump(g.gf, "logits");
    }

    logits_out.resize(n_vocab);
    ggml_backend_tensor_get(g.logits, logits_out.data(), 0, sizeof(float) * n_vocab);
    if (emb_all && g.emb_out) {
        fprintf(stderr, "[dbg] read emb_out ne1=%lld\n", (long long)g.emb_out->ne[1]);
        emb_all->resize((size_t)n_embd * T);
        ggml_backend_tensor_get(g.emb_out, emb_all->data(), 0, sizeof(float) * emb_all->size());
    }
    if (h24_all && g.h24_out) {
        fprintf(stderr, "[dbg] read h24_out ne1=%lld\n", (long long)g.h24_out->ne[1]);
        h24_all->resize((size_t)n_embd * T);
        ggml_backend_tensor_get(g.h24_out, h24_all->data(), 0, sizeof(float) * h24_all->size());
    }
    if (getenv("QOMNI_TIME") && T == 1) {
        static double b_ms = 0, a_ms = 0, c_ms = 0; static int ns = 0;
        b_ms += std::chrono::duration<double, std::milli>(tb1 - tb0).count();
        a_ms += std::chrono::duration<double, std::milli>(tb2 - tb1).count();
        c_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tb2).count();
        if (++ns % 64 == 0)
            fprintf(stderr, "[qomni-time] build %.2fms alloc %.2fms compute+rest %.2fms\n",
                    b_ms / ns, a_ms / ns, c_ms / ns);
    }
    n_past = S;
    ggml_free(g.ctx);
    g.ctx = nullptr;
    return true;
}

int Qwen3Omni::sample(const std::vector<float> & logits, const vqa::GenParams & gp) {
    return vqa::sample_logits(logits, gp, rng_state);
}

void Qwen3Omni::dump(ggml_cgraph * gf, const char * name) {
    if (dump_dir.empty()) return;
    vqa::dump::write(gf, dump_dir.c_str(), name);
}

// host preprocessing + the shared tower graph (qwen3vl_tower.hpp)
bool Qwen3Omni::encode_vision(const vqa::MediaInput & media, std::vector<float> & feats,
                              int * n_tokens, int * grid, int token_budget) {
    *n_tokens = 0;
    grid[0] = grid[1] = grid[2] = 0;
    if (media.empty()) return true;
    const int T = (int)media.frames.size();
    const int src_h = media.height, src_w = media.width;
    const int factor = vc.patch * vc.merge;                     // 32

    long long max_pixels = media.kind == vqa::MediaKind::Video ? 200704 : 16384LL * 32 * 32;
    if (token_budget > 0) {
        const long long min_total = 4LL * T;
        if (token_budget < min_total) return false;
        const long long per_step = token_budget / T;
        max_pixels = std::min(max_pixels, per_step * factor * factor);
    }
    int H = 0, W = 0;
    if (!vqa::img::smart_resize(src_h, src_w, factor, 4 * factor * factor,
                                (int)std::min(max_pixels, (long long)INT_MAX), &H, &W)) {
        fprintf(stderr, "image too small (%dx%d, factor %d)\n", src_w, src_h, factor);
        return false;
    }
    fprintf(stderr, "image %dx%d -> %dx%d\n", src_w, src_h, W, H);

    std::vector<std::vector<float>> rgb(T);
    for (int i = 0; i < T; i++)
        vqa::img::resize_bicubic_antialias(media.frames[i].rgb.data(), src_w, src_h, W, H, rgb[i]);

    std::vector<float> patches, pos;
    std::vector<int> coords;
    int gh = 0, gw = 0;
    for (int s = 0; s < T; s++) {
        const float * fs[2] = { rgb[s].data(), rgb[s].data() };
        std::vector<float> p, po;
        std::vector<int> co;
        qwen3vl::patchify3d(fs, H, W, vc, pos_table_host.data(), p, po, co, &gh, &gw);
        patches.insert(patches.end(), p.begin(), p.end());
        pos.insert(pos.end(), po.begin(), po.end());
        coords.insert(coords.end(), co.begin(), co.end());
    }
    const int hw = gh * gw, mh = gh / vc.merge, mw = gw / vc.merge;
    const int C3 = 3 * vc.temporal_patch * vc.patch * vc.patch;
    vqa::vit::fold_patch_bias(pos.data(), pos.size(), vc.n_embd, patch_bias.data());

    vqa::qwen3vl_tower::Config tc;   // same fields as the preprocess VisConfig
    tc.n_layer = vc.n_layer; tc.n_embd = vc.n_embd; tc.n_head = vc.n_head;
    tc.n_ff = vc.n_ff; tc.patch = vc.patch; tc.temporal_patch = vc.temporal_patch;
    tc.merge = vc.merge; tc.layernorm_eps = vc.layernorm_eps;
    tc.rope_theta = vc.rope_theta;
    for (int k = 0; k < n_ds; k++) tc.deepstack.push_back(vc.deepstack[k]);
    vqa::qwen3vl_tower::RunCtx rc;
    rc.backend = backend; rc.allocr = allocr; rc.cfg = &tc; rc.w = &vt;
    rc.text_width = n_embd; rc.tower_buf = &tower_buf;
    rc.ds_buf = n_ds > 0 ? ds_buf : nullptr;
    rc.env_prefix = "QOMNI"; rc.dump_dir = dump_dir;
    std::vector<vqa::qwen3vl_tower::FrameInput> fi(T);
    for (int s = 0; s < T; s++) {
        fi[s].patches = patches.data() + (size_t)s * hw * C3;
        fi[s].pos = pos.data() + (size_t)s * hw * vc.n_embd;
        fi[s].coords = coords.data() + (size_t)s * hw * 2;
    }
    const auto tv0 = std::chrono::steady_clock::now();
    if (!vqa::qwen3vl_tower::run(rc, fi.data(), T, H, W, gh, gw, feats, n_tokens))
        return false;
    fprintf(stderr, "[perf] vision encode: %.0f ms (%d tokens)\n",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv0).count(),
            *n_tokens);
    grid[0] = T; grid[1] = mh; grid[2] = mw;
    fprintf(stderr, "vision: %d patches (%dx%d) -> %d tokens (%d deepstack taps)\n",
            hw * T, gh, gw, *n_tokens, n_ds);
    return true;
}

bool Qwen3Omni::generate(const std::vector<int> & ids, const std::vector<float> * feats,
                         const std::vector<int> * img_pos, const std::vector<int> & mpos,
                         const vqa::GenParams & gp, std::string & out_text) {
    const int n_img = (feats && img_pos && !img_pos->empty()) ? (int)(feats->size() / n_embd) : 0;
    int max_new = gp.max_tokens;
    if ((int)ids.size() + max_new > n_ctx) {
        max_new = n_ctx - (int)ids.size();
        fprintf(stderr, "warning: clamping max_new to %d\n", max_new);
        if (max_new <= 0) return false;
    }
    n_past = 0;
    kv.clear();

    // mpos comes from the caller: [t(T), h(T), w(T), slot(T)]
    {
        const size_t T = ids.size();
        last_mpos[0] = mpos[T - 1];
        last_mpos[1] = mpos[T + T - 1];
        last_mpos[2] = mpos[2 * T - 1];
    }

    auto t0 = std::chrono::steady_clock::now();
    TextGraph g;
    std::vector<float> logits;
    // talker trajectory: thinker embed + layer-24 hidden for every prompt and
    // generated token (exportable for the HF oracle cross-drive)
    std::vector<float> traj_emb, traj_h24;
    std::vector<float> step_emb, step_h24;   // scratch: run_text overwrites
    if (!run_text(g, ids, n_img > 0 ? feats : nullptr, n_img > 0 ? img_pos : nullptr,
                  mpos, logits, gp.speak_wav ? &traj_emb : nullptr,
                  gp.speak_wav ? &traj_h24 : nullptr)) return false;
    auto t1 = std::chrono::steady_clock::now();

    out_text.clear();
    int next = sample(logits, gp);
    int n_dec = 0;
    std::vector<int> step_pos(4), step_tok(1), gen_ids;
    for (int step = 0; step < max_new; step++) {
        if (next == id_eos || next == id_endoftext) break;
        if (gp.speak_wav) gen_ids.push_back(next);
        out_text += tok_.decode({next});
        if (gp.on_token) gp.on_token(tok_.decode({next}).c_str(), gp.on_token_ud);
        n_dec++;
        if (step + 1 >= max_new) break;
        step_pos[0] = last_mpos[0] + 1;
        step_pos[1] = last_mpos[1] + 1;
        step_pos[2] = last_mpos[2] + 1;
        step_pos[3] = n_past;
        step_tok[0] = next;
        if (ec_enabled && ec_ok && !getenv("QOMNI_EC_OFF")) {
            // dynamic expert-cache decode path (Strata-style tier 2);
            // captures the talker trajectory (emb = L0 input row, h24 = the
            // post-layer-23 hidden) the same way the for_talker graph does
            if (!ec_decode_step(next, step_pos.data(), n_past, logits,
                                gp.speak_wav ? &step_emb : nullptr,
                                gp.speak_wav ? &step_h24 : nullptr)) break;
            n_past += 1;   // run_text updates n_past itself; ec_decode_step does not
        } else if (!run_text(g, step_tok, nullptr, nullptr, step_pos, logits,
                      gp.speak_wav ? &step_emb : nullptr,
                      gp.speak_wav ? &step_h24 : nullptr)) break;
        if (gp.speak_wav) {
            traj_emb.insert(traj_emb.end(), step_emb.begin(), step_emb.end());
            traj_h24.insert(traj_h24.end(), step_h24.begin(), step_h24.end());
        }
        last_mpos[0] = step_pos[0]; last_mpos[1] = step_pos[1]; last_mpos[2] = step_pos[2];
        next = sample(logits, gp);
    }
    (void)n_dec;
    if (gp.speak_wav && talker_deferred) {
        load_talker();          // while talker_deferred is still set: the
        talker_deferred = false; // CPU-partner placement decision reads it
    }
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
        fprintf(stderr, "qomni ec: %d decode steps, %.1f ms/step (A %.2f + B %.2f"
                " + swap-wait %.2f + readback %.2f), %.1f tok/s | cache:"
                " %llu hits / %llu misses (%.1f%% hit), %llu evictions,"
                " %.2f GB swapped, %d/%d slots resident, in_flight %d (async %s)\n",
                ec_st.steps, ec_st.ms_fin / ec_st.steps, ec_st.ms_a / ec_st.steps, ec_st.ms_b / ec_st.steps,
                ec_st.ms_swap / ec_st.steps, ec_st.ms_rb / ec_st.steps,
                (double)(ec_st.steps * 1000) / ec_st.ms_fin,
                (unsigned long long)st.hits, (unsigned long long)st.misses,
                acc ? 100.0 * (double)st.hits / (double)acc : 0.0,
                (unsigned long long)st.evictions, st.swap_bytes / 1e9,
                st.resident, st.n_slots, in_flight, ec_hooks.async_ok ? "on" : "off/v1");
    }
    if (gp.speak_wav) {
        if (getenv("QOMNI_TRAJ_DUMP")) {
            const std::string dir = getenv("QOMNI_TRAJ_DUMP");
            FILE * fo = fopen((dir + "/traj.bin").c_str(), "wb");
            if (fo) {
                auto wv = [&](const std::vector<float> & v) {
                    uint64_t n = v.size(); fwrite(&n, 8, 1, fo);
                    if (n) fwrite(v.data(), 4, (size_t)n, fo);
                };
                uint64_t nid = ids.size();
                fwrite(&nid, 8, 1, fo); fwrite(ids.data(), 4, ids.size(), fo);
                uint64_t ng = (uint64_t)gen_ids.size();
                fwrite(&ng, 8, 1, fo); fwrite(gen_ids.data(), 4, gen_ids.size(), fo);
                wv(traj_emb); wv(traj_h24);
                fclose(fo);
            }
        }
        if (talker_on && !gen_ids.empty()) {
            SynthIn si;
            si.max_codes = gp.max_codes;
            if (!gp.speaker.empty()) {
                auto it = talker.speaker_ids.find(gp.speaker);
                if (it != talker.speaker_ids.end()) talker.id_speaker = it->second;
                else fprintf(stderr, "qwen3omni: unknown speaker '%s' (have:", gp.speaker.c_str());
                if (it == talker.speaker_ids.end()) {
                    for (auto & kv : talker.speaker_ids) fprintf(stderr, " %s", kv.first.c_str());
                    fprintf(stderr, ")\n");
                }
            }
            si.ids = &ids;
            si.gen_ids = &gen_ids;
            si.traj_emb = &traj_emb;
            si.traj_h24 = &traj_h24;
            if (split_mode && !tok_embd_host) {
                // one-time D2H of the whole table (~250 MB Q6_K)
                const size_t nb = ggml_nbytes(tok_embd);
                tok_embd_host_mem.resize(nb);
                ggml_backend_tensor_get(tok_embd, tok_embd_host_mem.data(), 0, nb);
                tok_embd_host_ctx = ggml_init({ 128ull << 10, NULL, /*no_alloc*/ true });
                tok_embd_host = ggml_new_tensor_2d(tok_embd_host_ctx, tok_embd->type,
                                                   tok_embd->ne[0], tok_embd->ne[1]);
                tok_embd_host_buf = ggml_backend_cpu_buffer_from_ptr(
                    tok_embd_host_mem.data(), nb);
                tok_embd_host->buffer = tok_embd_host_buf;
                tok_embd_host->data = tok_embd_host_mem.data();
            }
            si.thinker_embd = tok_embd_host ? tok_embd_host : tok_embd;
            si.thinker_embd_dim = n_embd;
            si.rng = &rng_state;
            std::vector<float> wav;
            const auto ts0 = std::chrono::steady_clock::now();
            if (synthesize(&talker, si, wav) && !wav.empty()) {
                vqa::omni::write_wav_24k(gp.speak_wav, wav);
                fprintf(stderr, "talker: %zu samples (%.2f s) in %.1f s\n", wav.size(),
                        wav.size() / 24000.0,
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts0).count() / 1e3);
            } else {
                fprintf(stderr, "talker: synthesize failed\n");
            }
        }
    }
    const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double pre_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "[perf] prefill %.0f ms (%d tok), decode %.0f ms (%d tok, %.1f tok/s)\n",
            pre_ms, (int)ids.size(), dt - pre_ms, n_dec,
            n_dec > 0 ? n_dec * 1000.0 / (dt - pre_ms) : 0.0);
    return true;
}

bool Qwen3Omni::ask(const vqa::MediaInput & media, const std::string & prompt,
                    const std::string & /*mode*/, const vqa::GenParams & gp,
                    std::string & out_text) {
    if (media.has_audio())
        fprintf(stderr, "qwen3omni: audio input lands in the audio stage\n");
    std::vector<float> feats, afeats;
    int n_aud = 0;
    if (media.has_audio()) {
        const float * pcm = media.audio.data();
        int n_samples = (int)media.audio.size();
        if (media.sample_rate != 16000) {
            fprintf(stderr, "qwen3omni: audio must be 16 kHz (got %d)\n", media.sample_rate);
            return false;
        }
        if (!at.encode(pcm, n_samples, afeats, &n_aud, dump_dir.empty() ? nullptr : dump_dir.c_str()))
            return false;
    }
    int n_img = 0;
    int grid[3] = {0, 0, 0};
    if (!media.empty()) {
        const int text_overhead = (int)tok_.encode(gp.history + prompt).size() + 32;
        const int budget = n_ctx - gp.max_tokens - text_overhead - 16;
        if (budget <= 0) { fprintf(stderr, "n_ctx too small for prompt\n"); return false; }
        if (!encode_vision(media, feats, &n_img, grid, budget)) return false;
    }

    // prompt: [vision blocks] text, ChatML
    std::string vision_blocks;
    if (n_img > 0 && grid[0] > 0) {
        const int per = grid[1] * grid[2];
        for (int b = 0; b < grid[0]; b++) {
            vision_blocks += "<|vision_start|>";
            for (int k = 0; k < per; k++) vision_blocks += "<|image_pad|>";
            vision_blocks += "<|vision_end|>";
        }
    }
    std::string audio_blocks;
    for (int k = 0; k < n_aud; k++) audio_blocks += "<|audio_pad|>";
    const std::string text = gp.history + "<|im_start|>user\n" + vision_blocks +
        audio_blocks + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    const std::vector<int> ids = tok_.encode(text);
    if (getenv("QOMNI_DUMP_IDS")) {
        fprintf(stderr, "prompt ids:");
        for (int id : ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }
    std::vector<int> img_pos;
    for (int i = 0; i < (int)ids.size(); i++) if (ids[i] == id_image_pad) img_pos.push_back(i);
    if ((int)img_pos.size() != n_img) {
        fprintf(stderr, "placeholder mismatch: %d pad tokens vs %d vision tokens\n",
                (int)img_pos.size(), n_img);
        if (n_img > 0) return false;
    }
    std::vector<int> aud_pos;
    for (int i = 0; i < (int)ids.size(); i++) if (ids[i] == id_audio_pad) aud_pos.push_back(i);
    if ((int)aud_pos.size() != n_aud) {
        fprintf(stderr, "audio placeholder mismatch: %d pads vs %d audio tokens\n",
                (int)aud_pos.size(), n_aud);
        if (n_aud > 0) return false;
    }
    // combined multimodal injection: one position/feature list (media tokens
    // share the set_rows path; mrope positions treat audio as t=h=w text walk
    // in the shared builder, so audio runs walk (p, p, p) exactly like HF)
    std::vector<int> mm_pos = img_pos;
    std::vector<float> mm_feats = feats;
    if (n_aud > 0) {
        mm_pos.insert(mm_pos.end(), aud_pos.begin(), aud_pos.end());
        mm_feats.insert(mm_feats.end(), afeats.begin(), afeats.end());
        if (n_img == 0) { feats.swap(mm_feats); img_pos.swap(mm_pos); }
        else { feats.swap(mm_feats); img_pos.swap(mm_pos); }
    }
    std::vector<int> mpos;
    vqa::build_mpos_qwen3vl(ids, img_pos, n_img, grid[1], grid[2], mpos);
    if (n_aud > 0) {
        // audio tokens were appended to img_pos after the image block(s):
        // rebuild the plain (t, t, t) walk for them
        // (build_mpos handled the image blocks; fix up the audio tail)
        // audio occupies the trailing positions of the multimodal block list
        // simplest correct approach: recompute the whole walk here
        mpos.clear();
        mpos.reserve((size_t)4 * ids.size());
        std::vector<int> axis[3];
        const int per = grid[1] * grid[2];
        int cur = 0;
        size_t i = 0;
        while (i < ids.size()) {
            if (ids[i] == id_image_pad && n_img > 0) {
                for (int b = 0; b < grid[0] && i < ids.size(); b++) {
                    for (int k = 0; k < per && i < ids.size(); k++, i++) {
                        axis[0].push_back(cur);
                        axis[1].push_back(cur + k / grid[2]);
                        axis[2].push_back(cur + k % grid[2]);
                    }
                }
                cur += std::max(grid[1], grid[2]);
            } else if (ids[i] == id_audio_pad && n_aud > 0) {
                for (int k = 0; k < n_aud && i < ids.size(); k++, i++) {
                    axis[0].push_back(cur); axis[1].push_back(cur); axis[2].push_back(cur);
                    cur++;
                }
            } else {
                axis[0].push_back(cur); axis[1].push_back(cur); axis[2].push_back(cur);
                cur++; i++;
            }
        }
        mpos.insert(mpos.end(), axis[0].begin(), axis[0].end());
        mpos.insert(mpos.end(), axis[1].begin(), axis[1].end());
        mpos.insert(mpos.end(), axis[2].begin(), axis[2].end());
        for (int k = 0; k < (int)ids.size(); k++) mpos.push_back(k);
    }
    return generate(ids, (n_img + n_aud) > 0 ? &feats : nullptr,
                    (n_img + n_aud) > 0 ? &img_pos : nullptr, mpos, gp, out_text);
}

#include "qwen3omni_expert_decode.inc"

}  // namespace qwen3omni

namespace vqa {
Model * create_qwen3omni() { return new qwen3omni::Qwen3Omni(); }
}
