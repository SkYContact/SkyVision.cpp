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
#include "core/vqa.hpp"
#include "core/blocks/vit_block.hpp"
#include "core/blocks/llm_block.hpp"

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
#include "core/engine/sampler.hpp"
#include "core/util/aligned_buf.hpp"
#include "core/util/mmap.hpp"
#include "core/engine/backend.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/gguf_util.hpp"
#include "core/engine/stage_buf.hpp"
#include "core/runtime/load_util.hpp"
#include "core/media/resample.hpp"
#include "core/engine/weight_load.hpp"
#include "qwen3_5_preprocess.hpp"
#include "minicpm_preprocess.hpp"   // MiniCPM-V 4.7 (qwen3_5_text trunk + SigLIP canvas tower)

#include <algorithm>
#include <chrono>
#include <climits>
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

// MiniCPM-V 4.7 window-attention merger (insert_layer_id=6) + the 2x2 projection
// MLP that follows the tower. The window half pools each 2x2 patch window into one
// token, the projection half merges 2x2 again (16x total).
struct McpmVitMerger {
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;    // layer_norm1
    ggml_tensor *attn_w = nullptr, *attn_b = nullptr;      // fused q|k|v (bias)
    ggml_tensor *attn_o_w = nullptr, *attn_o_b = nullptr;  // out_proj
    ggml_tensor *pre_norm_w = nullptr, *pre_norm_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;        // window_hidden -> window_intermediate
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;        // window_intermediate -> hidden
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
    // The front ends gate --frames on caps() *before* load(), so the video
    // capability has to be visible at construction; load() refines it per arch.
    Qwen3_5() { caps_.image = true; caps_.video = true; }
    ~Qwen3_5() override;
    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override {
        return minicpm ? "minicpm-v-4.7" : (has_head ? "videojev" : "qwen3.5");
    }
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp,
             std::string & out_text) override;
    // ChatML template knowledge for the server's multi-turn sessions (moss
    // pattern): gp.history is prepended verbatim before the new user turn in
    // ask(), so the front end stays template-agnostic
    std::string format_history_turn(const std::string & user,
                                    const std::string & assistant) const override {
        return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
             + assistant + "<|im_end|>\n";
    }
    // VideoJev: candidate scoring instead of decoding (see vqa::ScoreQuery)
    bool score(const vqa::MediaInput & media, const std::vector<vqa::ScoreQuery> & queries,
               float threshold, std::vector<vqa::ScoreResult> & out,
               const vqa::GenParams & gp) override;

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
    // MiniCPM-V 4.7 (see minicpm_preprocess.hpp): SigLIP tower -> window merger at
    // insert layer -> post LN -> 2x2 projection MLP, one pass per crop.
    bool encode_vision_minicpm(const vqa::MediaInput & media, std::vector<float> & feats,
                               int * n_tokens, int * grid);
    bool encode_crop_minicpm(const minicpm::Crop & crop, std::vector<float> & feats,
                             int * n_tokens);
    int mcpm_crop_no = 0;    // dump suffix when a plan has several crops
    int mcpm_ncrops = 1;
    std::string mcpm_prompt(const std::string & user_text, bool video) const;
    bool mcpm_mpos(const std::vector<int> & ids, std::vector<int> & mpos);
    // VideoJev decision graph tap: gather hidden rows by index and score them
    // with the cosine head, replacing the lm head as the graph output.
    struct ScoreTap {
        int n_max_cand = 0;   // candidates per query (padded)
        int n_query = 0;
    };
    bool build_text_graph(TextGraph & g, int T, int S, int n_img, int state_slot = 0,
                          bool logits_all = false, const ScoreTap * tap = nullptr);
    bool run_score_text(TextGraph & g, const std::vector<int> & toks,
                        const std::vector<float> & feats, const std::vector<int> & img_pos,
                        const std::vector<int> & mpos, const std::vector<int> & pair_idx,
                        const std::vector<int> & query_idx, const ScoreTap & tap,
                        std::vector<float> & logits_out);
    // qwen3_5_mtp.inc
    int mtp_draft_step(int tok, const float * h, int slot, int axis,
                       std::vector<float> & logits_out, std::vector<float> & h_out, int win_lo = -1);
    bool mtp_verify(const int * toks, int n_past, int axis0, int state_slot,
                    std::vector<float> & logits2, std::vector<float> & h_out, int am_out[2] = nullptr);
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
    // ---- MiniCPM-V 4.7: same text trunk, SigLIP canvas tower + 16x mergers ----
    bool minicpm = false;
    minicpm::VisConfig mvc;
    ggml_tensor *m_patch_w = nullptr, *m_patch_b = nullptr;
    ggml_tensor *m_post_norm_w = nullptr, *m_post_norm_b = nullptr;   // tower post_layernorm
    ggml_tensor *m_out_head = nullptr;                               // lm_head (untied)
    McpmVitMerger mv;
    int id_image_start = 248078, id_image_end = 248079;   // <image> / </image> for canvas mrope
    int id_slice_start = 248088, id_slice_end = 248089;   // <slice> / </slice>
    // per visual block (image or video frame): token counts per crop + canvas geometry
    std::vector<std::vector<int>> mcpm_crop_tokens;
    std::vector<minicpm::BlockGeom> mcpm_blocks;
    int id_image_pad = 248056, id_video_pad = 248057;
    int id_vision_start = 248053, id_vision_end = 248054;
    int id_eos = 0, id_endoftext = 248044;
    // VideoJev: the last text layer keeps attention only (reference deletes the
    // MLP + post-norm) and a cosine decision head replaces the lm head
    bool last_attn_only = false;
    bool has_head = false;
    int jev_head_dim = 512;
    float jev_head_scale = 10.0f, jev_threshold = 0.25f;
    int jev_smooth = 1, id_decide = 248077, id_option_end = 248078;
    ggml_tensor *jev_q_w = nullptr, *jev_q_b = nullptr;
    ggml_tensor *jev_k_w = nullptr, *jev_k_b = nullptr, *jev_b = nullptr;

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr, wbuf_plain = nullptr;
    vqa::AlignedBuf host_f32_storage;
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
    // the decode graph reserves from its OWN gallocr: a shared one re-reserves
    // (and frees) the buffer under dec_tok/dec_mpos/dec_mask whenever a later
    // turn's prefill graph outgrows it, leaving the cached input pointers
    // dangling (reproduced live as a server SIGSEGV on the turn after a
    // short ask - same bug vc3 hit, fixed there in ddf69d2, moss in 09c944a)
    ggml_gallocr_t dec_allocr = nullptr;

    // speculative graph caches (qwen3_5_mtp.inc): one verify graph per state
    // slot + one draft graph, keyed on the 512-block KV window; allocated once
    // and reused across rounds so per-round cost is input-set + compute only
    struct SpecCache {
        ggml_context * ctx = nullptr;
        ggml_cgraph * gf = nullptr;
        int blk = -1, slot = -1;
        ggml_tensor * tok = nullptr, * mpos = nullptr, * mask = nullptr;
        ggml_tensor * logits = nullptr, * hout = nullptr;
        ggml_tensor * am = nullptr;          // per-column argmax (I32), device-side
        // dedicated gallocr: a shared allocr re-reserves (and can relocate) its
        // buffer whenever a BIGGER graph is allocated, dangling every pointer
        // of the graphs allocated before it - hit as a segfault in tensor_set
        // on the second spec round when the T=2 verify graph outgrew the draft
        ggml_gallocr_t alloc = nullptr;
    };
    SpecCache spec_v[2];
    SpecCache spec_d;

    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

// ---------------------------------------------------------------- helpers

static float randf_(uint64_t & s) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return (float)(((s * 0x2545F4914F6CDD1Dull) >> 11) * (1.0 / 900719254740992.0));
}

Qwen3_5::~Qwen3_5() {
    if (dec_ctx) ggml_free(dec_ctx);
    if (dec_allocr) ggml_gallocr_free(dec_allocr);
    for (auto & c : spec_v) {
        if (c.ctx) ggml_free(c.ctx);
        if (c.alloc) ggml_gallocr_free(c.alloc);
    }
    if (spec_d.ctx) ggml_free(spec_d.ctx);
    if (spec_d.alloc) ggml_gallocr_free(spec_d.alloc);
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
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!dec_allocr) dec_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr && dec_allocr != nullptr;
}

// ---------------------------------------------------------------- load
bool Qwen3_5::load(const std::string & gguf_path, const std::string & tokenizer_path,
                  const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "QWEN35", n_threads, use_gpu, dump_dir);
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::Qwen2Regex)) return false;
    // a tokenizer from the wrong family (a 151k Qwen vocab picked for this
    // 248k model by a registry scanning a models dir) loads fine and turns
    // every answer into merged-garbage loops - refuse it here where the
    // mismatch is visible (same 20% family rule the registry discovery uses)
    if (tok_.size() > 0) {
        const int64_t diff = tok_.size() > n_vocab ? tok_.size() - (int64_t)n_vocab
                                                   : (int64_t)n_vocab - tok_.size();
        if (diff * 5 > n_vocab) {
            fprintf(stderr, "qwen3_5: tokenizer vocab %d vs model vocab %d - wrong tokenizer.json\n",
                    tok_.size(), n_vocab);
            return false;
        }
    }
    if (!alloc_state()) return false;

    vqa::assign_token_ids(tok_, {
        { "<|image_pad|>",    &id_image_pad },
        { "<|video_pad|>",    &id_video_pad },
        { "<|vision_start|>", &id_vision_start },
        { "<|vision_end|>",   &id_vision_end },
        { "<|endoftext|>",    &id_endoftext },
    });
    id_eos = vqa::pick_token(tok_, "<|im_end|>", 248046);

    caps_ = {};
    caps_.image = true;
    // MiniCPM-V 4.7: images and video frames both go through the canvas layout
    caps_.video = true;
    caps_.timestamps_prompt = !minicpm;
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

    n_layer   = vqa::gguf::u32(gu, "marlin.block_count", n_layer);
    n_embd    = vqa::gguf::u32(gu, "marlin.embedding_length", n_embd);
    n_head    = vqa::gguf::u32(gu, "marlin.attention.head_count", n_head);
    n_kv_head = vqa::gguf::u32(gu, "marlin.attention.head_count_kv", n_kv_head);
    head_dim  = vqa::gguf::u32(gu, "marlin.attention.key_length", head_dim);
    n_ff      = vqa::gguf::u32(gu, "marlin.feed_forward_length", n_ff);
    n_vocab   = vqa::gguf::u32(gu, "marlin.vocab_size", n_vocab);
    rms_eps   = vqa::gguf::f32(gu, "marlin.attention.layer_norm_rms_epsilon", rms_eps);
    rope_theta = vqa::gguf::f32(gu, "marlin.rope.freq_base", rope_theta);
    rope_dim  = vqa::gguf::u32(gu, "marlin.rope.dimension_count", rope_dim);
    mrope_sections[0] = vqa::gguf::u32(gu, "marlin.rope.dimension_section_0", mrope_sections[0]);
    mrope_sections[1] = vqa::gguf::u32(gu, "marlin.rope.dimension_section_1", mrope_sections[1]);
    mrope_sections[2] = vqa::gguf::u32(gu, "marlin.rope.dimension_section_2", mrope_sections[2]);
    mrope_sections[3] = vqa::gguf::u32(gu, "marlin.rope.dimension_section_3", mrope_sections[3]);
    ssm_d_conv = vqa::gguf::u32(gu, "marlin.ssm.conv_kernel", ssm_d_conv);
    ssm_d_state = vqa::gguf::u32(gu, "marlin.ssm.state_size", ssm_d_state);
    ssm_d_inner = vqa::gguf::u32(gu, "marlin.ssm.inner_size", ssm_d_inner);
    ssm_n_group = vqa::gguf::u32(gu, "marlin.ssm.group_count", ssm_n_group);
    ssm_dt_rank = vqa::gguf::u32(gu, "marlin.ssm.time_step_rank", ssm_dt_rank);

    vc.n_layer = vqa::gguf::u32(gu, "marlin.vision.block_count", vc.n_layer);
    vc.n_embd  = vqa::gguf::u32(gu, "marlin.vision.embedding_length", vc.n_embd);
    vc.n_head  = vqa::gguf::u32(gu, "marlin.vision.attention.head_count", vc.n_head);
    vc.n_ff    = vqa::gguf::u32(gu, "marlin.vision.feed_forward_length", vc.n_ff);
    vc.patch   = vqa::gguf::u32(gu, "marlin.vision.patch_size", vc.patch);
    vc.temporal_patch = vqa::gguf::u32(gu, "marlin.vision.temporal_patch_size", vc.temporal_patch);
    vc.merge   = vqa::gguf::u32(gu, "marlin.vision.merge_size", vc.merge);
    vc.pos_side = (int)std::lround(sqrtf((float)vqa::gguf::u32(gu, "marlin.vision.pos_table_size", 2304)));
    vc.layernorm_eps = vqa::gguf::f32(gu, "marlin.vision.layer_norm_epsilon", vc.layernorm_eps);
    vc.rope_theta = vqa::gguf::f32(gu, "marlin.vision.rope_theta", vc.rope_theta);
    vc.out_hidden = vqa::gguf::u32(gu, "marlin.vision.out_hidden_size", vc.out_hidden);
    // MiniCPM-V 4.7 rides the same text trunk but brings a SigLIP canvas tower:
    // no rope (learned position table addressed by nearest bucket ids), a window
    // merger inserted after vision layer `insert_layer`, and a projection MLP.
    // The tower's own sizes come from the shared marlin.vision.* keys.
    // Dispatch on the GGUF's own architecture field - minicpmv47 has its own
    // arch name in the registry, so no guessing can collide. Tensor probing for
    // this split is fragile: v.pos_emb.weight is carried by every qwen3_5
    // checkpoint (the VideoJev / Marlin / Qwen3.5 SigLIP towers all have it),
    // and each future qwen3_5-trunk model would add another colliding
    // signature. vm.norm1.weight stays only as the fallback for checkpoints
    // written with no arch field at all.
    {
        const std::string gguf_arch = vqa::gguf::str(gu, "general.architecture");
        minicpm = gguf_arch == "minicpmv47" || gguf_arch == "minicpm_v4_7" ||
                  (gguf_arch.empty() && ggml_get_tensor(wctx, "vm.norm1.weight") != nullptr);
    }
    if (minicpm) {
        mvc.n_embd  = vc.n_embd;
        mvc.n_head  = vc.n_head;
        mvc.n_layer = vc.n_layer;
        mvc.n_ff    = vc.n_ff;
        mvc.patch   = vc.patch;
        mvc.ln_eps  = vc.layernorm_eps;
        mvc.pos_side     = vqa::gguf::u32(gu, "minicpm.vision.pos_side", mvc.pos_side);
        mvc.insert_layer = vqa::gguf::u32(gu, "minicpm.vision.insert_layer", mvc.insert_layer);
        mvc.window       = vqa::gguf::u32(gu, "minicpm.vision.window", mvc.window);
        mvc.merge        = vqa::gguf::u32(gu, "minicpm.vision.merge", mvc.merge);
        mvc.downsample   = vqa::gguf::u32(gu, "minicpm.downsample", mvc.downsample);
        if (const char * e = getenv("MINICPM_DOWNSAMPLE"))   // "4x" / "16x" override
            mvc.downsample = (strncmp(e, "4", 1) == 0) ? 4 : 16;
        id_image_start   = vqa::gguf::u32(gu, "minicpm.image_start_id", id_image_start);
        id_image_end     = vqa::gguf::u32(gu, "minicpm.image_end_id", id_image_end);
        id_slice_start   = vqa::gguf::u32(gu, "minicpm.slice_start_id", id_slice_start);
        id_slice_end     = vqa::gguf::u32(gu, "minicpm.slice_end_id", id_slice_end);
        mvc.max_slice_nums = vqa::gguf::u32(gu, "minicpm.vision.max_slice_nums",
                                            mvc.max_slice_nums);
        if (const char * e = getenv("MINICPM_SCALE_RES")) mvc.scale_resolution = atoi(e);
        if (const char * e = getenv("MINICPM_NO_SLICES")) mvc.max_slice_nums = 1;
    }
    has_mtp = vqa::gguf::u32(gu, "marlin.has_mtp", 0) != 0;
    marlin_skills = vqa::gguf::u32(gu, "marlin.skills", 0) != 0;
    id_image_pad = vqa::gguf::u32(gu, "marlin.image_token_id", id_image_pad);
    id_video_pad = vqa::gguf::u32(gu, "marlin.video_token_id", id_video_pad);
    id_vision_start = vqa::gguf::u32(gu, "marlin.vision_start_token_id", id_vision_start);
    id_vision_end = vqa::gguf::u32(gu, "marlin.vision_end_token_id", id_vision_end);
    id_endoftext = vqa::gguf::u32(gu, "marlin.pad_token_id", id_endoftext);
    last_attn_only = vqa::gguf::u32(gu, "marlin.last_layer_attn_only", 0) != 0;
    // VideoJev decision head (absent on plain Qwen3.5 / Marlin checkpoints)
    jev_head_dim    = vqa::gguf::u32(gu, "videojev.head_dim", jev_head_dim);
    jev_head_scale  = vqa::gguf::f32(gu, "videojev.head_scale", jev_head_scale);
    jev_threshold   = vqa::gguf::f32(gu, "videojev.threshold", jev_threshold);
    jev_smooth      = vqa::gguf::u32(gu, "videojev.smooth", jev_smooth);
    id_decide       = vqa::gguf::u32(gu, "videojev.decide_token_id", id_decide);
    id_option_end   = vqa::gguf::u32(gu, "videojev.option_end_token_id", id_option_end);

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);

    // 1D tensors (norms, biases, dt/a vectors, conv kernels) -> host F32
    vqa::AlignedBuf host_f32;
    vqa::weights::promote_1d_to_f32(wctx, host_f32);

    {
        ggml_tensor * pe = ggml_get_tensor(wctx, "v.pos_emb.weight");
        if (!pe) { fprintf(stderr, "missing v.pos_emb.weight\n"); return false; }
        pos_table_host.resize(ggml_nelements(pe));
        vqa::gguf::tensor_to_f32(pe, pos_table_host.data());
    }

    {
        ggml_tensor * pb = ggml_get_tensor(wctx, "v.patch_embd.bias");
        if (pb && pb->type == GGML_TYPE_F32 && pb->ne[0] == vc.n_embd && pb->ne[1] == 1)
            patch_bias.assign((const float *)pb->data, (const float *)pb->data + vc.n_embd);
    }

    {
        vqa::weights::Bound bound = vqa::weights::bind_to_backend(
            wctx, backend, mm, { { host_f32.data(), host_f32.size() } });
        if (!bound.ok) return false;
        wbuf = bound.main_buf;
        if (!bound.host_bufs.empty()) wbuf_plain = bound.host_bufs[0];
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
        const bool attn_only = last_attn_only && i == n_layer - 1;
        L.post_norm = attn_only ? nullptr : g("attn_post_norm.weight");
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
        if (L.post_norm) {
            L.gate = g("ffn_gate.weight");
            L.up = g("ffn_up.weight");
            L.down = g("ffn_down.weight");
        }
    }
    if (last_attn_only)
        fprintf(stderr, "last layer %d is attention-only (VideoJev trim)\n", n_layer - 1);
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
    if (minicpm) {
        // SigLIP tower post-norm + the window merger + the untied lm head. The
        // tower blocks themselves reuse the v.blk.N.* bindings above, and the
        // 2x2 projection MLP reuses mm.norm/mm.fc1/mm.fc2 (different shapes from
        // the qwen3_5 merger; only one mode is ever active).
        m_post_norm_w = gt("v.post_norm.weight");
        m_post_norm_b = gt("v.post_norm.bias");
        mv.norm1_w  = gt("vm.norm1.weight");   mv.norm1_b  = gt("vm.norm1.bias");
        mv.attn_w   = gt("vm.attn.weight");    mv.attn_b   = gt("vm.attn.bias");
        mv.attn_o_w = gt("vm.wo.weight");      mv.attn_o_b = gt("vm.wo.bias");
        mv.pre_norm_w = gt("vm.pre_norm.weight"); mv.pre_norm_b = gt("vm.pre_norm.bias");
        mv.fc1_w    = gt("vm.fc1.weight");     mv.fc1_b    = gt("vm.fc1.bias");
        mv.fc2_w    = gt("vm.fc2.weight");     mv.fc2_b    = gt("vm.fc2.bias");
        m_out_head  = ggml_get_tensor(wctx, "output.weight");   // untied head (optional)
        if (!m_post_norm_w || !mv.norm1_w || !mv.fc1_w || !mm_fc1_w)
            { fprintf(stderr, "minicpm: merger/tower tensors incomplete\n"); return false; }
        if (!m_out_head) fprintf(stderr, "minicpm: no output.weight, falling back to tied head\n");
        fprintf(stderr, "minicpm-v tower: %d layers (d %d, h %d, patch %d), window merger "
                        "after layer %d, canvas %dx\n",
                mvc.n_layer, mvc.n_embd, mvc.n_head, mvc.patch, mvc.insert_layer,
                mvc.downsample);
    }

    // VideoJev decision head (optional): fp32 CosineDecisionHead q/k projections
    jev_q_w = ggml_get_tensor(wctx, "jev_head_q.weight");
    jev_q_b = ggml_get_tensor(wctx, "jev_head_q.bias");
    jev_k_w = ggml_get_tensor(wctx, "jev_head_k.weight");
    jev_k_b = ggml_get_tensor(wctx, "jev_head_k.bias");
    jev_b = ggml_get_tensor(wctx, "jev_head_b");
    has_head = jev_q_w && jev_q_b && jev_k_w && jev_k_b && jev_b;
    if (has_head) {
        if ((int)jev_q_w->ne[0] != n_embd || (int)jev_q_w->ne[1] != jev_head_dim) {
            fprintf(stderr, "decision head q_proj is [%lld, %lld], expected [%d, %d]\n",
                    (long long)jev_q_w->ne[0], (long long)jev_q_w->ne[1], n_embd, jev_head_dim);
            return false;
        }
        fprintf(stderr, "decision head: cosine %d-d, scale %g, threshold %g (score-only model)\n",
                jev_head_dim, (double)jev_head_scale, (double)jev_threshold);
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
// The attention hook (strided head views, ggml GGML_ROPE_TYPE_VISION rope
// with sections 16|16 over 32 pairs, one unmasked FA over the whole stream,
// layer-0 parity truncation) lives once in vqa::vit::vit_attention_hook
// (core/vit_block.hpp); this backend only supplies its parameters below.

// MiniCPM-V 4.7 tower (16x canvas path): SigLIP encoder -> window-attention
// merger after `insert_layer` -> post_layernorm -> 2x2 projection MLP into the
// text width. One image, one graph; the tower blocks are the shared
// vqa::vit::block with no rope (position info is a learned table that the host
// folds into the patch embedding input, exactly like qwen3_5 folds its table).
bool Qwen3_5::encode_vision(const vqa::MediaInput & media, std::vector<float> & feats,
                           int * n_tokens, int * grid, int token_budget) {
    *n_tokens = 0;
    grid[0] = grid[1] = grid[2] = 0;
    if (media.empty()) return true;
    if (minicpm) return encode_vision_minicpm(media, feats, n_tokens, grid);
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
    if (!vqa::img::smart_resize(src_h, src_w, factor, 4 * factor * factor,
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
        vqa::img::resize_bicubic_antialias(frames[i]->rgb.data(), src_w, src_h, W, H, rgb[i]);

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
        // per-step graph: drop the previous step's registry (and its sink buffers)
        if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
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

        bool sinks_ready = false;   // keep nodes attach once, at the graph tail
        auto run_chunk = [&](const char * out_name, std::vector<float> * dest) -> bool {
            if (!dump_dir.empty() && !sinks_ready) {
                static const char * const dump_names[] = {
                    "patches", "v_pos_in", "v_patch_out", "v_blk0_attn_out", "v_blk0_out",
                    "v_final", nullptr,
                };
                vqa::dump::mark_sinks(gctx, gf, dump_names);
                vqa::dump::finalize_sinks(gctx, gf, backend);
                sinks_ready = true;
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
        vqa::vit::VitHookEnv env;
        env.gctx = gctx; env.gf = gf; env.t_vpos = t_vpos;
        env.n_head = vc.n_head; env.n_embd = vc.n_embd;
        env.N = Nc; env.vhd = vhd; env.vhd_f = vhd;
        env.n_ctx = 32768; env.rope_theta = vc.rope_theta;
        env.rope_stop_all_layers = true;
        env.stop_after = stop_s; env.state = 0;
        env.run = [&run_chunk](const char * name) -> bool { std::vector<float> sink; return run_chunk(name, &sink); };
        for (int il = 0; il < vl_max; il++) {
            VisLayer & L = vl[il];
            vqa::vit::BlockWeights bw{L.norm0_w, L.norm0_b, L.wqkv_w, L.wqkv_b, L.wo_w, L.wo_b,
                                      L.norm1_w, L.norm1_b, L.fc0_w, L.fc0_b, L.fc1_w, L.fc1_b};
            vqa::vit::BlockHooks bh;
            bh.attention = vqa::vit::vit_attention_hook;
            bh.probe_layer0 = vqa::vit::vit_probe_layer0_hook;
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
        if (!dump_dir.empty()) vqa::dump::dump_registry_clear();   // new graph, new registry
        struct ggml_init_params ip2 = { 32ull * 1024 * 1024, NULL, true };
        ggml_context * mctx = ggml_init(ip2);
        if (!mctx) return false;
        struct ggml_cgraph * mgf = ggml_new_graph_custom(mctx, 8192, false);
        ggml_tensor * t_x = tower_buf.view(mctx, N, 0);   // device-side, no host round trip
        ggml_tensor * x = t_x;

        bool sinks_ready = false;
        auto run_merger = [&](const char * out_name) -> bool {
            if (!dump_dir.empty() && !sinks_ready) {
                static const char * const dump_names[] = { "v_merge", "v_proj_out", nullptr };
                vqa::dump::mark_sinks(mctx, mgf, dump_names);
                vqa::dump::finalize_sinks(mctx, mgf, backend);
                sinks_ready = true;
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

// One vision-encoder pass over one crop (thumbnail or a detail crop): the tower,
// the window merger at `insert_layer` (16x only) and the 2x2 projection MLP.
bool Qwen3_5::encode_crop_minicpm(const minicpm::Crop & crop, std::vector<float> & feats,
                                  int * n_tokens) {
    *n_tokens = 0;
    // multi-crop plans get per-crop dump names (patches0, patches1, ...) so the
    // parity tooling can compare every crop instead of just the last one
    const std::string sfx = mcpm_ncrops > 1 ? std::to_string(mcpm_crop_no) : std::string();
    const std::string nm_patches = "patches" + sfx, nm_pos = "v_pos_in" + sfx;
    const std::string nm_patch_out = "v_patch_out" + sfx, nm_vm = "vm_out" + sfx;
    const std::string nm_tower = "v_tower_out" + sfx, nm_proj = "v_proj_out" + sfx;
    const int p = mvc.patch, D = mvc.n_embd, NH = mvc.n_head, HD = D / NH;
    std::vector<float> pos;
    const std::vector<float> & patches = crop.patches;
    int gh = crop.gh, gw = crop.gw;     // halved by the 16x window merger below
    const int C3 = 3 * p * p;
    int N = gh * gw;
    {
        // learned position table: nearest bucket ids, folded on the host (the
        // table is [pos_side*pos_side, D], one row per table cell)
        std::vector<int> pids;
        minicpm::nearest_pos_ids(gh, gw, mvc.pos_side, pids);
        pos.resize((size_t)D * N);
        for (int i = 0; i < N; i++)
            memcpy(pos.data() + (size_t)i * D, pos_table_host.data() + (size_t)pids[i] * D,
                   sizeof(float) * D);
    }

    struct ggml_init_params ip = { 96ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    ggml_context * gctx = ggml_init(ip);
    if (!gctx) return false;
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 32768, false);

    ggml_tensor * t_patches = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, C3, N);
    ggml_set_name(t_patches, nm_patches.c_str());
    ggml_tensor * t_pos = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, D, N);
    ggml_set_name(t_pos, nm_pos.c_str());
    ggml_tensor * t_widx = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, N);
    ggml_set_name(t_widx, "vm_widx");
    // window attention mask: block diagonal over groups of window^2 tokens. The
    // patch count is final before the tower runs, so the tensor exists up front.
    // "4x" mode never runs the window merger, so it needs no mask.
    ggml_tensor * t_vmask = nullptr;
    if (mvc.downsample == 16) {
        t_vmask = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, N, N);
        ggml_set_name(t_vmask, "vm_mask");
    }

    ggml_tensor * x = ggml_mul_mat(gctx, v_patch_w, t_patches);
    x = ggml_add(gctx, x, v_patch_b);    // SigLIP conv bias stays in-graph
    x = ggml_add(gctx, x, t_pos);
    ggml_set_name(x, nm_patch_out.c_str());

    vqa::vit::VitHookEnv env;
    env.gctx = gctx; env.gf = gf;
    env.n_head = NH; env.n_embd = D;
    env.vhd = HD; env.vhd_f = HD;
    env.no_rope = true;                 // SigLIP: position comes from the table add
    env.N = N;
    vqa::vit::BlockHooks bh;
    bh.attention = vqa::vit::vit_attention_hook;
    bh.ud = &env;

    std::vector<int> widx;   // window gather order, built when the merger fires

    // window-attention merger: pool each 2x2 patch window into one token
    auto window_merge = [&](ggml_tensor * xin) -> ggml_tensor * {
        minicpm::window_index(gh, gw, mvc.window, widx);
        ggml_tensor * xr = ggml_get_rows(gctx, xin, t_widx);      // window-major stream
        ggml_tensor * l = ggml_norm(gctx, xr, mvc.ln_eps);
        l = ggml_mul(gctx, l, mv.norm1_w);
        l = ggml_add(gctx, l, mv.norm1_b);

        ggml_tensor * qkv = ggml_mul_mat(gctx, mv.attn_w, l);
        qkv = ggml_add(gctx, qkv, mv.attn_b);
        const size_t off = (size_t)NH * HD * qkv->nb[0];
        auto hv = [&](int blk) {
            return ggml_view_3d(gctx, qkv, HD, NH, N, (size_t)HD * qkv->nb[0], qkv->nb[1],
                                (size_t)blk * off);
        };
        ggml_tensor * q = hv(0), * k = hv(1), * v = hv(2);
        // one 4-token flash-attn per window: the stream is window-major, so the
        // mask is just a block diagonal over groups of `window^2`
        ggml_tensor * qf = ggml_view_3d(gctx, q, HD, N, NH, q->nb[2], q->nb[1], 0);
        ggml_tensor * kf = ggml_view_3d(gctx, k, HD, N, NH, k->nb[2], k->nb[1], 0);
        ggml_tensor * vf = ggml_view_3d(gctx, v, HD, N, NH, v->nb[2], v->nb[1], 0);
        ggml_tensor * fa = ggml_flash_attn_ext(gctx, qf, kf, vf, t_vmask,
                                               1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        ggml_tensor * ao = ggml_reshape_2d(gctx, ggml_cont(gctx, fa), D, N);
        ao = ggml_mul_mat(gctx, mv.attn_o_w, ao);
        ao = ggml_add(gctx, ao, mv.attn_o_b);
        ggml_tensor * h = ggml_add(gctx, xr, ao);

        // window mean as the residual of the pooling MLP
        const int ws = mvc.window * mvc.window, nw = N / ws;
        ggml_tensor * h3 = ggml_reshape_3d(gctx, h, D, ws, nw);
        ggml_tensor * mean = nullptr;
        for (int t = 0; t < ws; t++) {
            ggml_tensor * tv = ggml_view_3d(gctx, h3, D, 1, nw, h3->nb[1], h3->nb[2],
                                            (size_t)t * h3->nb[1]);
            mean = mean ? ggml_add(gctx, mean, tv) : tv;
        }
        mean = ggml_scale(gctx, mean, 1.0f / (float)ws);
        mean = ggml_reshape_2d(gctx, mean, D, nw);    // back to [D, nw] for the add

        ggml_tensor * flat = ggml_reshape_2d(gctx, h, (int64_t)D * ws, nw);
        flat = ggml_norm(gctx, flat, mvc.ln_eps);
        flat = ggml_mul(gctx, flat, mv.pre_norm_w);
        flat = ggml_add(gctx, flat, mv.pre_norm_b);
        ggml_tensor * o = ggml_mul_mat(gctx, mv.fc1_w, flat);
        o = ggml_add(gctx, o, mv.fc1_b);
        o = ggml_gelu(gctx, o);                  // gelu_pytorch_tanh
        o = ggml_mul_mat(gctx, mv.fc2_w, o);
        o = ggml_add(gctx, o, mv.fc2_b);
        return ggml_add(gctx, o, mean);
    };

    for (int il = 0; il < mvc.n_layer; il++) {
        vqa::vit::BlockWeights bw{vl[il].norm0_w, vl[il].norm0_b, vl[il].wqkv_w, vl[il].wqkv_b,
                                  vl[il].wo_w, vl[il].wo_b, vl[il].norm1_w, vl[il].norm1_b,
                                  vl[il].fc0_w, vl[il].fc0_b, vl[il].fc1_w, vl[il].fc1_b};
        bh.probe_layer0 = nullptr;               // no layer-0 parity sink for this tower yet
        x = vqa::vit::block(gctx, x, bw, mvc.ln_eps, bh, il);
        if (!x) { ggml_free(gctx); return false; }
        // "4x" mode skips the vit merger: the tower runs full resolution and only
        // the projection MLP merges 2x2 (4x more visual tokens than the default)
        if (mvc.downsample == 16 && il == mvc.insert_layer) {
            x = window_merge(x);
            ggml_set_name(x, nm_vm.c_str());
            gh /= mvc.window; gw /= mvc.window;
            N = gh * gw;
            env.N = N;
            fprintf(stderr, "minicpm: window merger after layer %d -> %d tokens\n", il, N);
        }
    }

    // post_layernorm, then the 2x2 projection MLP into the text width
    x = ggml_norm(gctx, x, mvc.ln_eps);
    x = ggml_mul(gctx, x, m_post_norm_w);
    x = ggml_add(gctx, x, m_post_norm_b);
    ggml_set_name(x, nm_tower.c_str());

    const int mh = gh / mvc.merge, mw = gw / mvc.merge;
    const int n_out = mh * mw;
    std::vector<int> midx;
    minicpm::merge_index(gh, gw, mvc.merge, midx);
    ggml_tensor * t_midx = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, (int64_t)midx.size());
    ggml_set_name(t_midx, "pm_idx");
    ggml_tensor * cat = nullptr;
    for (int k = 0; k < mvc.merge * mvc.merge; k++) {
        ggml_tensor * iv = ggml_view_1d(gctx, t_midx, n_out, (size_t)k * n_out * sizeof(int));
        ggml_tensor * g = ggml_get_rows(gctx, x, iv);     // [D, n_out]
        cat = cat ? ggml_concat(gctx, cat, g, 0) : g;      // dim0 = (ky, kx, dim)
    }
    cat = ggml_norm(gctx, cat, mvc.ln_eps);
    cat = ggml_mul(gctx, cat, mm_norm_w);
    cat = ggml_add(gctx, cat, mm_norm_b);
    ggml_tensor * h = ggml_mul_mat(gctx, mm_fc1_w, cat);
    h = ggml_add(gctx, h, mm_fc1_b);
    h = ggml_gelu_erf(gctx, h);                 // nn.GELU() == exact erf
    h = ggml_mul_mat(gctx, mm_fc2_w, h);
    h = ggml_add(gctx, h, mm_fc2_b);
    ggml_set_name(h, nm_proj.c_str());
    ggml_build_forward_expand(gf, h);

    if (!dump_dir.empty()) {
        const char * dump_names[] = { nm_patches.c_str(), nm_pos.c_str(), nm_patch_out.c_str(),
                                      nm_vm.c_str(), nm_tower.c_str(), nm_proj.c_str(), nullptr };
        vqa::dump::mark_sinks(gctx, gf, dump_names);
        vqa::dump::finalize_sinks(gctx, gf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "minicpm vision graph allocation failed\n");
        ggml_free(gctx);
        return false;
    }
    ggml_backend_tensor_set(t_patches, patches.data(), 0, sizeof(float) * patches.size());
    ggml_backend_tensor_set(t_pos, pos.data(), 0, sizeof(float) * pos.size());
    if (!widx.empty()) ggml_backend_tensor_set(t_widx, widx.data(), 0, sizeof(int) * widx.size());
    ggml_backend_tensor_set(t_midx, midx.data(), 0, sizeof(int) * midx.size());
    // window attention mask: block diagonal over groups of window^2 tokens
    if (t_vmask) {
        const int N0 = gh * gw * mvc.window * mvc.window;   // back to the patch count
        const int ws = mvc.window * mvc.window;
        std::vector<int> spans;
        for (int s = 0; s < N0; s += ws) spans.push_back(s);
        vqa::set_blockdiag_mask(t_vmask, spans);
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "minicpm vision compute failed\n");
        ggml_free(gctx);
        return false;
    }
    ggml_tensor * out = ggml_graph_get_tensor(gf, nm_proj.c_str());
    feats.resize((size_t)out->ne[0] * out->ne[1]);
    ggml_backend_tensor_get(out, feats.data(), 0, sizeof(float) * feats.size());
    *n_tokens = (int)out->ne[1];
    dump(gf, nm_patches.c_str());
    dump(gf, nm_pos.c_str());
    dump(gf, nm_patch_out.c_str());
    dump(gf, nm_vm.c_str());
    dump(gf, nm_tower.c_str());
    dump(gf, nm_proj.c_str());
    ggml_free(gctx);
    mcpm_crop_no++;
    if (getenv("MINICPM_VERBOSE"))
        fprintf(stderr, "minicpm crop: %dx%d patches -> %d tokens\n", gw, gh, *n_tokens);
    return true;
}

// Whole media input: one plan per image / video frame, each plan = thumbnail plus
// (slice_mode) up to max_slice_nums detail crops. Features concatenate in prompt
// order; the per-block geometry is cached for the canvas M-RoPE pass.
bool Qwen3_5::encode_vision_minicpm(const vqa::MediaInput & media, std::vector<float> & feats,
                                    int * n_tokens, int * grid) {
    *n_tokens = 0;
    grid[0] = grid[1] = grid[2] = 0;
    if (media.empty()) return true;
    const bool video = media.kind == vqa::MediaKind::Video;
    const int nframe = video ? (int)media.frames.size() : 1;
    const int merge_factor = (mvc.downsample == 4) ? 2 : 4;   // patches per LLM token
    feats.clear();
    mcpm_blocks.clear();
    mcpm_crop_tokens.clear();
    mcpm_crop_no = 0;
    mcpm_ncrops = 1;
    int total = 0;
    for (int f = 0; f < nframe; f++) {
        minicpm::ImagePlan plan;
        if (!minicpm::prepare_image(media.frames[f].rgb.data(), media.width, media.height, mvc,
                                    /*allow_slices=*/!video, plan))
            return false;
        mcpm_ncrops = (video ? nframe : 1) * (int)plan.crops.size();
        minicpm::BlockGeom geom;
        geom.rows = plan.rows;
        geom.cols = plan.cols;
        std::vector<int> crop_tokens;
        crop_tokens.reserve(plan.crops.size());
        for (size_t ci = 0; ci < plan.crops.size(); ci++) {
            std::vector<float> cf;
            int n = 0;
            if (!encode_crop_minicpm(plan.crops[ci], cf, &n)) return false;
            crop_tokens.push_back(n);
            feats.insert(feats.end(), cf.begin(), cf.end());
            total += n;
        }
        geom.thumb_h = plan.crops[0].gh / merge_factor;
        geom.thumb_w = plan.crops[0].gw / merge_factor;
        if (plan.rows && plan.crops.size() > 1) {
            geom.slice_h = plan.crops[1].gh / merge_factor;
            geom.slice_w = plan.crops[1].gw / merge_factor;
        }
        mcpm_blocks.push_back(geom);
        mcpm_crop_tokens.push_back(crop_tokens);
        if (f == 0) { grid[0] = 1; grid[1] = geom.thumb_h; grid[2] = geom.thumb_w; }
        if (!video)
            fprintf(stderr, "minicpm image %dx%d: %zu crop%s, %dx%d/%dx%d grids\n",
                    media.width, media.height, plan.crops.size(),
                    plan.crops.size() == 1 ? "" : "s", geom.thumb_w, geom.thumb_h,
                    geom.slice_w, geom.slice_h);
    }
    grid[0] = nframe;
    *n_tokens = total;
    return true;
}

// ---------------------------------------------------------------- text

// closure for the shared vqa::llm::block call on the linear-attention layers
// (build_text_graph): what LayerCtx cannot carry - the persistent conv/ssm
// states (caller asset: 2 buffers x 3 rollback slots), the speculative-verify
// flags and the running linear-layer index. The gated-delta branch has no
// rope/KV cache of its own, so only hooks.attention is overridden; norm + MLP
// stay inside llm_block.
struct LaHookCtx {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    const std::vector<TxtLayer> * tl = nullptr;
    const std::vector<ggml_tensor *> * conv_state = nullptr, * conv_state_b = nullptr, * conv_state_c = nullptr;
    const std::vector<ggml_tensor *> * ssm_state = nullptr, * ssm_state_b = nullptr, * ssm_state_c = nullptr;
    int si = 0;            // running linear-layer index (the old ssm_idx)
    int state_slot = 0;    // which rollback slot this graph reads its states from
    bool spec = false;     // speculative verify graph (logits_all && T > 1)
    int T = 0;
    int kv_dim = 0, k_conv = 0, Sk = 0, Hv = 0, d_inner = 0, n_group = 0;
};

// the linear-attention branch (gated delta net), moved VERBATIM out of the
// build_text_graph loop (same ops, same order - the tie_state +0 dependency
// edges and their interleaving with the main chain are load-bearing: the CUDA
// multi-stream scheduler may not reorder the persistent-state copies ahead of
// nodes that reuse their source buffers). block() adds the returned [D, T]
// branch to the residual stream. Layer-0 probe names: t_l0_la_* are stamped
// here; llm_block's dbg_layer0 covers t_l0_norm/t_l0_swiglu/t_l0_out (dbg_name
// stays null - the two name sources are mutually exclusive and the dump sink
// list needs t_l0_swiglu).
static ggml_tensor * qwen35_la_attn(void * ud, int il, ggml_tensor * l,
                                    const vqa::llm::LayerCtx & c) {
    LaHookCtx * hc = (LaHookCtx *)ud;
    ggml_context * ctx = hc->ctx;
    const TxtLayer & L = (*hc->tl)[il];
    const int si = hc->si++;
    const int T = hc->T, kv_dim = hc->kv_dim, k_conv = hc->k_conv;
    const int Sk = hc->Sk, Hv = hc->Hv, n_group = hc->n_group;
    const bool spec = hc->spec;
    const int state_slot = hc->state_slot;

    // projections
    ggml_tensor * qkv = ggml_mul_mat(ctx, L.la_qkv, l);      // [kv_dim, T]
    ggml_tensor * z = ggml_mul_mat(ctx, L.la_gate, l);       // [d_inner, T]
    if (il == 0) { ggml_set_name(qkv, "t_l0_la_qkv"); ggml_set_name(z, "t_l0_la_z"); }

    ggml_tensor * beta = ggml_mul_mat(ctx, L.la_beta, l);    // [Hv, T]
    beta = ggml_sigmoid(ctx, beta);
    beta = ggml_reshape_4d(ctx, beta, 1, Hv, T, 1);
    ggml_tensor * alpha = ggml_mul_mat(ctx, L.la_alpha, l);  // [Hv, T]
    alpha = ggml_reshape_3d(ctx, alpha, Hv, T, 1);
    ggml_tensor * a_bias = ggml_add(ctx, alpha, L.la_dt);    // bcast [Hv] over T
    ggml_tensor * gk = ggml_mul(ctx, ggml_softplus(ctx, a_bias), L.la_a);
    gk = ggml_reshape_4d(ctx, gk, 1, Hv, T, 1);
    if (il == 0) { ggml_set_name(beta, "t_l0_la_beta"); ggml_set_name(gk, "t_l0_la_g"); }

    // causal conv: prepend the persistent state, roll it forward
    // (ggml_ssm_conv: sx 3d [k-1+T, conv_dim, 1], c [k, conv_dim],
    //  result [conv_dim, T, 1])
    ggml_tensor * qkv3 = ggml_reshape_3d(ctx, qkv, kv_dim, T, 1);
    ggml_tensor * qkv_t = ggml_transpose(ctx, qkv3);              // [T, kv_dim, 1]
    ggml_tensor * conv_in_state =
        (state_slot == 1 ? *hc->conv_state_b : state_slot == 2 ? *hc->conv_state_c : *hc->conv_state)[si];
    ggml_tensor * conv_st = ggml_reshape_3d(ctx, conv_in_state, k_conv - 1, kv_dim, 1);
    ggml_tensor * conv_in = ggml_concat(ctx, conv_st, qkv_t, 0);  // [k-1+T, kv_dim, 1]
    ggml_tensor * conv_state_last =
        ggml_view_3d(ctx, conv_in, k_conv - 1, kv_dim, 1,
                     conv_in->nb[1], conv_in->nb[2],
                     (size_t)(conv_in->ne[0] - (k_conv - 1)) * conv_in->nb[0]);
    // state writes hang off the layer chain; without a dependency edge
    // into the main path, CUDA multi-stream scheduling can overlap the
    // copy with later nodes that REUSE the source buffer (observed as
    // ~1e-2-slot-state corruption from the first post-attention layer
    // on). Tie every state copy into the layer output via +0.
    ggml_tensor * state_dep = nullptr;
    auto tie_state = [&](ggml_tensor * cpy_out) {
        ggml_build_forward_expand(hc->gf, cpy_out);
        ggml_tensor * s = ggml_scale(ctx, ggml_sum(ctx, cpy_out), 0.0f);
        state_dep = state_dep ? ggml_add(ctx, state_dep, s) : s;
    };
    // the rolled conv window ALWAYS goes to the MAIN conv_state (the
    // pre-fix code wrote it into `conv_st`, which on the slot-1
    // rejection path is a view of conv_state_b - so main never
    // advanced and the next round read a stale window)
    tie_state(ggml_cpy(ctx, conv_state_last,
                       ggml_reshape_3d(ctx, (*hc->conv_state)[si], k_conv - 1, kv_dim, 1)));
    if (spec) {
        // rollback windows: state after 1 token (slot 1) and after 0 (slot 2,
        // the untouched input state, needed when the whole draft is rejected)
        ggml_tensor * cs1 = ggml_view_3d(ctx, conv_in, k_conv - 1, kv_dim, 1,
                                         conv_in->nb[1], conv_in->nb[2], 1 * conv_in->nb[0]);
        tie_state(ggml_cpy(ctx, cs1, ggml_reshape_3d(ctx, (*hc->conv_state_b)[si], k_conv - 1, kv_dim, 1)));
        tie_state(ggml_cpy(ctx, conv_st,
                           ggml_reshape_3d(ctx, (*hc->conv_state_c)[si], k_conv - 1, kv_dim, 1)));
    }
    ggml_tensor * conv = ggml_ssm_conv(ctx, conv_in, L.la_conv1d); // [kv_dim, T, 1]
    conv = ggml_silu(ctx, conv);
    if (il == 0) ggml_set_name(conv, "t_l0_la_conv");

    // split q/k/v heads (channels: q 2*group*Sk? no: q,k = group*Sk each, v = inner)
    const int qk_dim = n_group * Sk;
    auto head_view = [&](int block, int dim, int heads) {
        return ggml_view_3d(ctx, conv, dim, heads, T,
                            (size_t)dim * ggml_element_size(conv),
                            (size_t)kv_dim * ggml_element_size(conv),
                            (size_t)block * qk_dim * ggml_element_size(conv));
    };
    ggml_tensor * q_conv = head_view(0, Sk, n_group);
    ggml_tensor * k_conv2 = head_view(1, Sk, n_group);
    ggml_tensor * v_conv = head_view(2, Sk, Hv);               // v block offset 2*qk_dim
    // l2 norm over the head dim (FLA kernel semantics)
    auto l2n = [&](ggml_tensor * t) {
        const float n = (float)t->ne[0];
        return ggml_scale(ctx, ggml_rms_norm(ctx, t, 1e-6f / n), 1.0f / sqrtf(n));
    };
    q_conv = l2n(q_conv);
    k_conv2 = l2n(k_conv2);
    q_conv = ggml_reshape_4d(ctx, q_conv, Sk, n_group, T, 1);
    k_conv2 = ggml_reshape_4d(ctx, k_conv2, Sk, n_group, T, 1);
    v_conv = ggml_cont(ctx, v_conv);      // head_view leaves a strided view
    v_conv = ggml_reshape_4d(ctx, v_conv, Sk, Hv, T, 1);

    // fused gated delta net; the result packs scores + K=1 state snapshot
    ggml_tensor * ssm_in =
        (state_slot == 1 ? *hc->ssm_state_b : state_slot == 2 ? *hc->ssm_state_c : *hc->ssm_state)[si];
    ggml_tensor * s0 = ggml_reshape_4d(ctx, ssm_in, Sk, Sk, Hv, 1);
    ggml_tensor * res = ggml_gated_delta_net(ctx, q_conv, k_conv2, v_conv, gk, beta,
                                             s0, /*K=*/(spec && getenv("QWEN35_SPEC_K1") == nullptr) ? 2 : 1);
    ggml_tensor * out = ggml_view_4d(ctx, res, Sk, Hv, T, 1,
                                     ggml_row_size(res->type, Sk),
                                     ggml_row_size(res->type, Sk * Hv),
                                     ggml_row_size(res->type, Sk * Hv * T), 0);
    ggml_tensor * new_state = ggml_view_3d(ctx, res, Sk, Sk, Hv,
                                           ggml_row_size(res->type, Sk),
                                           ggml_row_size(res->type, Sk * Sk),
                                           ggml_row_size(res->type, Sk * Hv * T));
    tie_state(ggml_cpy(ctx, new_state, (*hc->ssm_state)[si]));
    if (spec) {
        // snapshot layout: [scores T cols][slot0 = final][slot1 = one token back]
        ggml_tensor * snap1 = ggml_view_3d(ctx, res, Sk, Sk, Hv,
                Sk * ggml_element_size(res), Sk * Sk * ggml_element_size(res),
                ggml_row_size(res->type, Sk * Hv * (T + Sk)));
        tie_state(ggml_cpy(ctx, snap1, (*hc->ssm_state_b)[si]));
        tie_state(ggml_cpy(ctx, s0,
                           ggml_reshape_3d(ctx, (*hc->ssm_state_c)[si], Sk, Sk, Hv)));
    }
    if (il == 0) ggml_set_name(out, "t_l0_la_core");

    // gated RMSNorm over the head dim, gated by silu(z)
    ggml_tensor * zn = ggml_reshape_3d(ctx, z, Sk, Hv, T);
    ggml_tensor * on = ggml_rms_norm(ctx, out, c.rms_eps);
    on = ggml_mul(ctx, on, L.la_norm);
    on = ggml_mul(ctx, on, ggml_silu(ctx, zn));
    ggml_tensor * flat = ggml_reshape_2d(ctx, on, hc->d_inner, T);
    ggml_tensor * attn_out = ggml_mul_mat(ctx, L.la_out, flat);
    if (state_dep)   // serialize the state copies into the main chain (+0)
        attn_out = ggml_add(ctx, attn_out, state_dep);
    return attn_out;
}

bool Qwen3_5::build_text_graph(TextGraph & g, int T, int S, int n_img, int state_slot,
                               bool logits_all, const ScoreTap * tap) {
    // new graph -> previous dump registry/sink buffers are stale
    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
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

    int kv_idx = 0;
    int max_layers = n_layer;
    if (const char * e = getenv("QWEN35_MAX_LAYERS")) max_layers = std::min(max_layers, atoi(e));

    // shared-block plumbing for the linear-attention layers; the persistent
    // states are the CALLER's asset, captured by the hook closure (never in
    // LayerCtx), and si restarts at 0 with every graph build like ssm_idx did
    LaHookCtx hc;
    hc.ctx = g.ctx;
    hc.gf = g.gf;
    hc.tl = &tl;
    hc.conv_state = &conv_state; hc.conv_state_b = &conv_state_b; hc.conv_state_c = &conv_state_c;
    hc.ssm_state = &ssm_state;   hc.ssm_state_b = &ssm_state_b;   hc.ssm_state_c = &ssm_state_c;
    hc.state_slot = state_slot;
    hc.spec = logits_all && T > 1;
    hc.T = T;
    hc.kv_dim = kv_dim; hc.k_conv = k_conv; hc.Sk = Sk; hc.Hv = Hv;
    hc.d_inner = ssm_d_inner; hc.n_group = ssm_n_group;

    vqa::llm::LayerCtx lc;
    lc.gf = g.gf;
    lc.T = T; lc.S = S;
    lc.n_head = NH; lc.n_kv_head = NKV; lc.head_dim = HD;
    lc.n_ctx = n_ctx;
    lc.rms_eps = rms_eps;
    lc.rope_theta = rope_theta;

    for (int il = 0; il < max_layers; il++) {
        TxtLayer & L = tl[il];
        if (!L.full) {
            // gated-delta linear-attention layer: norm + MLP come from the
            // shared vqa::llm::block (core/llm_block.hpp); the whole recurrent
            // attention branch - projections, conv/state rolling, delta scan,
            // gated norm and the persistent-state write-back - hangs off
            // hooks.attention (qwen35_la_attn above, moved verbatim).
            vqa::llm::LayerWeights w;
            w.norm = L.norm;
            w.ffn_norm = L.post_norm;   // attn_post_norm doubles as the FFN norm
            w.gate = L.gate; w.up = L.up; w.down = L.down;
            vqa::llm::BlockHooks lh;
            lh.ud = &hc;
            lh.dbg_layer0 = true;       // llm_block stamps t_l0_norm/swiglu/out
            lh.attention = qwen35_la_attn;
            x = vqa::llm::block(g.ctx, x, il, w, lc, lh);
            continue;
        }
        ggml_tensor * l = ggml_rms_norm(g.ctx, x, rms_eps);
        l = ggml_mul(g.ctx, l, L.norm);

        ggml_tensor * attn_out = nullptr;
        {   // full-attention layer, kept handwritten: interleaved q|gate
            // projection + IMROPE + the per-token verify FA are shapes the
            // block has no opinion on (core/llm_block.hpp header note)
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
            // QWEN35_SPEC_FUSED=1: verify uses ONE fused T=2 flash-attn call
            // (the mma-f16 kernel, ~2% logits drift vs the vector kernel) —
            // trades bit-exactness vs plain decode for per-round latency
            static const bool spec_fused_fa = getenv("QWEN35_SPEC_FUSED") != nullptr;
            if (logits_all && T > 1 && !spec_fused_fa) {
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
        }

        x = ggml_add(g.ctx, x, attn_out);
        ggml_tensor * ffn_inp = x;
        if (il == 0) ggml_set_name(x, "t_l0_resid");

        // attention-only layer (VideoJev trims the final MLP + post-norm)
        if (!L.post_norm) {
            if (il == 0) ggml_set_name(x, "t_l0_out");
            continue;
        }
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

    if (tap) {
        // VideoJev CosineDecisionHead: gather the readout rows (candidate and
        // query positions) out of the final hidden state and score every
        // (query, candidate) pair with a scaled cosine in FP32
        const int nq = tap->n_query, nk = tap->n_max_cand;
        if (!has_head) { fprintf(stderr, "score requested but the GGUF has no decision head\n"); return false; }
        ggml_tensor * pair_idx = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, (int64_t)nq * nk);
        ggml_set_name(pair_idx, "jev_pair_in");
        ggml_tensor * query_idx = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, nq);
        ggml_set_name(query_idx, "jev_query_in");

        ggml_tensor * hp = ggml_get_rows(g.ctx, x, pair_idx);       // [D, nq*nk]
        ggml_tensor * hq = ggml_get_rows(g.ctx, x, query_idx);      // [D, nq]
        auto l2n = [&](ggml_tensor * t) {
            const float n = (float)t->ne[0];
            return ggml_scale(g.ctx, ggml_rms_norm(g.ctx, t, 1e-6f / n), 1.0f / sqrtf(n));
        };
        ggml_tensor * keys = ggml_add(g.ctx, ggml_mul_mat(g.ctx, jev_k_w, hp), jev_k_b);  // [hd, nq*nk]
        keys = l2n(keys);
        ggml_tensor * qs = ggml_add(g.ctx, ggml_mul_mat(g.ctx, jev_q_w, hq), jev_q_b);    // [hd, nq]
        qs = l2n(qs);
        ggml_set_name(keys, "jev_keys");
        ggml_set_name(qs, "jev_queries");
        // per-query candidate sets are contiguous columns of `keys`
        ggml_tensor * keys3 = ggml_reshape_3d(g.ctx, keys, jev_head_dim, nk, nq);
        ggml_tensor * qs3 = ggml_reshape_3d(g.ctx, qs, jev_head_dim, 1, nq);
        ggml_tensor * logits = ggml_mul_mat(g.ctx, qs3, keys3);     // [1, nk, nq]
        logits = ggml_reshape_2d(g.ctx, logits, nk, nq);            // [nk, nq] (k fastest)
        logits = ggml_scale(g.ctx, logits, jev_head_scale);
        logits = ggml_add(g.ctx, logits, jev_b);
        ggml_set_name(logits, "jev_logits");
        g.logits = logits;
        ggml_build_forward_expand(g.gf, g.logits);
        return true;
    }

    // MiniCPM-V 4.7 ships an untied lm_head (tie_word_embeddings=false); every
    // other tenant ties the head to the token embedding.
    ggml_tensor * head = m_out_head ? m_out_head : tok_embd;
    if (logits_all) {
        g.logits = ggml_mul_mat(g.ctx, head, x);           // [vocab, T] speculative verify
    } else if (T > 1) {
        ggml_tensor * last = ggml_view_2d(g.ctx, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
        g.logits = ggml_mul_mat(g.ctx, head, last);        // tied lm head
    } else {
        g.logits = ggml_mul_mat(g.ctx, head, x);
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
        vqa::dump::finalize_sinks(g.ctx, g.gf, backend);   // keep nodes at the graph tail
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
            if (!ggml_gallocr_alloc_graph(dec_allocr, dec_gf)) {
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
        if (h_last) {
            // MTP spec's plain rounds resume the draft chain from here: the
            // h must come out of the same graph family the verify uses, or
            // the draft head sees inputs it never predicts from
            ggml_tensor * hn = ggml_graph_get_tensor(dec_gf, "t_final_norm");
            if (hn) {
                h_last->resize(n_embd);
                ggml_backend_tensor_get(hn, h_last->data(), (size_t)(T - 1) * hn->nb[1], sizeof(float) * n_embd);
            }
        }
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
    // MiniCPM-V: canvas markers (<image>/</image>) plus the per-image id tag, no
    // <|vision_start|> wrapper (chat_template.jinja + replace_image_token)
    if (minicpm) {
        (void)n_img_tokens;   // the visual block comes from the encoded crops
        return minicpm::wrap_prompt(mcpm_prompt(user_text, video), user_text);
    }
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

// Visual part of the prompt: for an image the id tag plus one block (thumbnail +
// detail crops), for a video one block per frame with no id tag (the processor
// only numbers images).
std::string Qwen3_5::mcpm_prompt(const std::string & user_text, bool video) const {
    (void)user_text;
    std::string vis;
    const char * pad = video ? "<|video_pad|>" : "<|image_pad|>";
    for (size_t b = 0; b < mcpm_crop_tokens.size(); b++) {
        const std::vector<int> & ct = mcpm_crop_tokens[b];
        if (ct.empty()) continue;
        if (!video && b == 0) vis += "<image_id>0</image_id>";   // image-only setting
        minicpm::append_image_block(vis, ct.data(), (int)ct.size(), mcpm_blocks[b].rows,
                                    mcpm_blocks[b].cols, pad, "<image>", "</image>",
                                    "<slice>", "</slice>");
    }
    return vis;
}

// Canvas M-RoPE for MiniCPM-V 4.7 (see minicpm::canvas_mpos): image tokens sit on a
// 2-D canvas instead of a 1-D run, so the three rope channels are not equal inside
// the visual span. Also refreshes last_mpos / mrope_delta for the decode steps.
bool Qwen3_5::mcpm_mpos(const std::vector<int> & ids, std::vector<int> & mpos) {
    if (!minicpm::canvas_mpos(ids, mcpm_blocks, id_image_start, id_image_end,
                              id_slice_start, id_slice_end, id_image_pad, id_video_pad, mpos))
        return false;
    const int T = (int)ids.size();
    last_mpos = { mpos[(size_t)T - 1], mpos[(size_t)2 * T - 1], mpos[(size_t)3 * T - 1] };
    const int mx = std::max({last_mpos[0], last_mpos[1], last_mpos[2]});
    mrope_delta = (mx + 1) - T;
    if (getenv("MINICPM_DUMP_MPOS")) {
        // canvas M-RoPE rows for the oracle comparison (ref --pos-only)
        fprintf(stderr, "minicpm mpos t:");
        for (int i = 0; i < T; i++) fprintf(stderr, " %d", mpos[i]);
        fprintf(stderr, "\nminicpm mpos h:");
        for (int i = 0; i < T; i++) fprintf(stderr, " %d", mpos[(size_t)T + i]);
        fprintf(stderr, "\nminicpm mpos w:");
        for (int i = 0; i < T; i++) fprintf(stderr, " %d", mpos[(size_t)2 * T + i]);
        fprintf(stderr, "\nminicpm mpos delta: %d\n", mrope_delta);
    }
    return true;
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
        const int text_overhead = (int)tok_.encode(gp.history +
            render_prompt(1, user_text, video_kind)).size();
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

    // multi-turn session: the server's history prefix (fully formatted prior
    // turns, moss pattern) goes in verbatim before the new user turn; the
    // final assistant turn keeps the non-thinking prefix from render_prompt
    const std::string text = gp.history + render_prompt(n_img, user_text, video_kind);
    const std::vector<int> ids = tok_.encode(text);
    if (minicpm && getenv("MINICPM_DUMP_IDS")) {
        // prompt id stream for the oracle comparison (scripts/ref_minicpmv47.py --ids-only)
        fprintf(stderr, "minicpm ids:");
        for (int id : ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }
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
    if (minicpm) {
        // canvas M-RoPE: (t, h, w) laid out per the reference get_rope_index
        if (!mcpm_mpos(ids, mpos)) return false;
    } else {
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


// ---------------------------------------------------------------- scoring
// VideoJev ("System One") does not decode: the whole query group goes through
// one causal forward pass and the decision head scores a fixed candidate list
// per question. Everything below mirrors the reference inference scripts
// (processing_videojev.prepare_query_groups / format_predictions).

bool Qwen3_5::run_score_text(TextGraph & g, const std::vector<int> & toks,
                             const std::vector<float> & feats, const std::vector<int> & img_pos,
                             const std::vector<int> & mpos, const std::vector<int> & pair_idx,
                             const std::vector<int> & query_idx, const ScoreTap & tap,
                             std::vector<float> & logits_out) {
    const int T = (int)toks.size();
    if (T > n_ctx) { fprintf(stderr, "context overflow (%d > %d)\n", T, n_ctx); return false; }
    const int n_img = (int)(feats.size() / n_embd);
    if (!build_text_graph(g, T, T, n_img, 0, false, &tap)) return false;
    if (!ggml_gallocr_alloc_graph(allocr, g.gf)) {
        fprintf(stderr, "score graph allocation failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }
    ggml_backend_tensor_set(g.tok_in, toks.data(), 0, sizeof(int) * T);
    ggml_backend_tensor_set(g.mpos_in, mpos.data(), 0, sizeof(int) * 4 * T);
    if (n_img > 0) {
        ggml_backend_tensor_set(g.img_in, feats.data(), 0, sizeof(float) * feats.size());
        ggml_backend_tensor_set(g.img_pos_in, img_pos.data(), 0, sizeof(int) * n_img);
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(g.gf, "jev_pair_in"),
                            pair_idx.data(), 0, sizeof(int) * pair_idx.size());
    ggml_backend_tensor_set(ggml_graph_get_tensor(g.gf, "jev_query_in"),
                            query_idx.data(), 0, sizeof(int) * query_idx.size());
    vqa::set_causal_mask(g.mask_in, 0, T);
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "score compute failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }
    const size_t n = (size_t)tap.n_max_cand * tap.n_query;
    logits_out.resize(n);
    ggml_backend_tensor_get(g.logits, logits_out.data(), 0, sizeof(float) * n);
    ggml_free(g.ctx);
    g.ctx = nullptr;
    return true;
}

namespace {

// pair_timeline(): one score per temporal patch, cell bounds halfway between
// adjacent patch centres; outer cells extend to the video endpoints
static void pair_bounds(double fps, int n_frames, double duration,
                        std::vector<double> & centers, std::vector<double> & lo,
                        std::vector<double> & hi) {
    const int gt = (n_frames + 1) / 2;
    std::vector<double> t((size_t)n_frames);
    for (int i = 0; i < n_frames; i++) t[(size_t)i] = i / fps;
    centers.resize(gt);
    for (int j = 0; j < gt; j++) {
        const double a = t[(size_t)std::min(2 * j, n_frames - 1)];
        const double b = t[(size_t)std::min(2 * j + 1, n_frames - 1)];
        centers[(size_t)j] = (a + b) / 2;
    }
    std::vector<double> edges;
    edges.push_back(0.0);
    for (int i = 1; i + 1 <= n_frames - 1; i += 2)
        edges.push_back((t[(size_t)i] + t[(size_t)(i + 1)]) / 2);
    edges.push_back(duration);
    // edges.size() == gt + 1 (last cell may be degenerate for tiny clips)
    lo.resize(gt);
    hi.resize(gt);
    for (int j = 0; j < gt; j++) {
        lo[(size_t)j] = edges[(size_t)std::min(j, (int)edges.size() - 1)];
        hi[(size_t)j] = j + 1 < (int)edges.size() ? edges[(size_t)j + 1] : duration;
        if (hi[(size_t)j] <= lo[(size_t)j]) hi[(size_t)j] = lo[(size_t)j] + 1.0 / fps;
    }
}

// videojev_temporal.decode(): smoothed, threshold-shifted Kadane over the
// Bernoulli logits, then round / repair / clamp
static void decode_interval(const std::vector<float> & logits, double threshold, int smooth,
                            const std::vector<double> & lo, const std::vector<double> & hi,
                            double duration, double & start, double & end) {
    const int n = (int)logits.size();
    std::vector<double> z((size_t)n);
    for (int i = 0; i < n; i++) z[(size_t)i] = (double)logits[(size_t)i];
    if (smooth > 1) {
        const int h = smooth / 2;
        std::vector<double> pad((size_t)(n + 2 * h));
        for (int i = 0; i < n + 2 * h; i++)
            pad[(size_t)i] = z[(size_t)std::min(std::max(i - h, 0), n - 1)];
        for (int i = 0; i < n; i++) {
            double acc = 0;
            for (int k = 0; k < smooth; k++) acc += pad[(size_t)(i + k)];
            z[(size_t)i] = acc / smooth;
        }
    }
    const double shift = std::log(threshold / (1.0 - threshold));
    for (int i = 0; i < n; i++) z[(size_t)i] -= shift;
    double best = z[0], cur = z[0];
    int best_s = 0, best_e = 0, s = 0;
    for (int i = 1; i < n; i++) {
        if (cur <= 0) { s = i; cur = z[(size_t)i]; }
        else cur += z[(size_t)i];
        if (cur > best) { best = cur; best_s = s; best_e = i; }
    }
    const double raw_s = lo[(size_t)best_s], raw_e = hi[(size_t)best_e];
    // round_repair_clamp_v1: round half to even, repair a collapsed interval
    // using the evidence overlap, then clamp into [0, duration]
    double rs = std::nearbyint(raw_s), re = std::nearbyint(raw_e);
    if (re <= rs) {
        const double cands[2][2] = {{rs - 1.0, rs}, {rs, rs + 1.0}};
        const double mid = (raw_s + raw_e) / 2;
        double best_lo = rs, best_hi = re;
        bool found = false;
        // max over (evidence overlap, -|midpoint distance|, -start), lexicographic
        double best_dot = 0, best_mid = 0, best_start = 0;
        for (auto & c : cands) {
            if (!(c[0] >= 0 && c[0] < hi.back() && c[1] > lo.front())) continue;
            double dot = 0;
            for (int i = 0; i < n; i++)
                dot += z[(size_t)i] * std::max(0.0, std::min(hi[(size_t)i], c[1]) -
                                                    std::max(lo[(size_t)i], c[0]));
            const double mid_pen = -std::fabs((c[0] + c[1]) / 2 - mid);
            const bool better = !found
                || dot > best_dot
                || (dot == best_dot && (mid_pen > best_mid
                    || (mid_pen == best_mid && c[0] < best_start)));
            if (!better) continue;
            found = true;
            best_dot = dot;
            best_mid = mid_pen;
            best_start = c[0];
            best_lo = c[0];
            best_hi = c[1];
        }
        if (found) { rs = best_lo; re = best_hi; }
    }
    start = std::min(duration, std::max(0.0, rs));
    end = std::min(duration, std::max(0.0, re));
}

}  // namespace

bool Qwen3_5::score(const vqa::MediaInput & media, const std::vector<vqa::ScoreQuery> & queries,
                    float threshold_in, std::vector<vqa::ScoreResult> & out,
                    const vqa::GenParams & gp) {
    (void)gp;
    out.clear();
    if (!has_head) {
        fprintf(stderr, "this checkpoint has no decision head; --score needs a VideoJev GGUF\n");
        return false;
    }
    if (queries.empty() || media.empty()) {
        fprintf(stderr, "score needs a video and at least one question\n");
        return false;
    }
    const double fps = media.fps > 0 ? media.fps : 2.0;
    const int n_frames = (int)media.frames.size();
    const double duration = n_frames / fps;
    const double threshold = threshold_in > 0 ? threshold_in : jev_threshold;

    std::vector<float> feats;
    int n_img = 0, grid[3] = {0, 0, 0};
    if (!encode_vision(media, feats, &n_img, grid, (int)((double)n_ctx * 0.8))) return false;
    const int gt = grid[0], mh = grid[1], mw = grid[2];
    const int spatial = mh * mw;
    if (gt * spatial != n_img) {
        fprintf(stderr, "vision tokens %d != %d*%d\n", n_img, gt, spatial);
        return false;
    }
    std::vector<double> centers, cell_lo, cell_hi;
    pair_bounds(fps, n_frames, duration, centers, cell_lo, cell_hi);
    if ((int)centers.size() != gt) {
        fprintf(stderr, "temporal patches %d != %d score slots\n", gt, (int)centers.size());
        return false;
    }

    // candidate text: the timestamps the vision block itself carries (":.1f")
    std::vector<std::string> stamps((size_t)gt);
    for (int j = 0; j < gt; j++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "<%.1f seconds>", centers[(size_t)j]);
        stamps[(size_t)j] = buf;
    }

    // ---- packed token ids: one user turn per question, video only in the first
    bool bad = false;
    for (const auto & q : queries) {
        if (q.task != "grounding" && q.task != "choice") {
            fprintf(stderr, "unknown task '%s' (grounding|choice)\n", q.task.c_str());
            return false;
        }
        if (q.query.find("<|") != std::string::npos) { bad = true; }
        for (const auto & o : q.options) if (o.find("<|") != std::string::npos) bad = true;
        if (q.task == "choice" && q.options.size() < 2) {
            fprintf(stderr, "choice question '%s' needs at least two options\n", q.id.c_str());
            return false;
        }
    }
    if (bad) { fprintf(stderr, "questions/options must not contain control tokens\n"); return false; }

    // the chat template's video placeholder (<|vision_start|><|video_pad|><|vision_end|>)
    // is replaced by the per-frame expansion, so its own brackets stay: one
    // <|vision_start|> before the first frame and one <|vision_end|> after the last
    std::string vblock = "<|vision_start|>";
    for (int j = 0; j < gt; j++) {
        vblock += stamps[(size_t)j];
        vblock += "<|vision_start|>";
        for (int k = 0; k < spatial; k++) vblock += "<|video_pad|>";
        vblock += "<|vision_end|>";
    }
    vblock += "<|vision_end|>";

    auto suffix = [&](const vqa::ScoreQuery & q, bool leading_newline) {
        const bool choice = q.task == "choice";
        std::string s;
        if (leading_newline) s += "\n";
        s += choice ? "Task: single-choice video question answering" : "Task: temporal grounding";
        s += "\nQuestion: " + q.query + "\nOptions:\n";
        if (choice) for (const auto & o : q.options) s += o + "<|option_end|>\n";
        else        for (const auto & st : stamps)  s += st + "<|option_end|>\n";
        s += "<|decide|>";
        return s;
    };

    std::vector<int> ids, pair_idx, query_idx;
    std::vector<int> counts;
    int n_max_cand = 0;
    for (size_t i = 0; i < queries.size(); i++) {
        std::string text = "<|im_start|>user\n";
        if (i == 0) text += vblock;
        text += suffix(queries[i], true);
        text += "<|im_end|>\n";
        const std::vector<int> row = tok_.encode(text);
        const int base = (int)ids.size();
        ids.insert(ids.end(), row.begin(), row.end());
        std::vector<int> opts, dec;
        for (int k = 0; k < (int)row.size(); k++) {
            if (row[(size_t)k] == id_option_end) opts.push_back(base + k);
            else if (row[(size_t)k] == id_decide) dec.push_back(base + k);
        }
        const int expect = queries[i].task == "choice" ? (int)queries[i].options.size() : gt;
        if ((int)opts.size() != expect || dec.size() != 1) {
            fprintf(stderr, "question %zu: %d candidates (expected %d) / %zu decide tokens\n",
                    i, (int)opts.size(), expect, dec.size());
            return false;
        }
        pair_idx.insert(pair_idx.end(), opts.begin(), opts.end());
        query_idx.push_back(dec[0]);
        counts.push_back(expect);
        n_max_cand = std::max(n_max_cand, expect);
    }
    // pad the candidate matrix per query (extra columns are masked out)
    {
        std::vector<int> padded;
        padded.reserve((size_t)n_max_cand * queries.size());
        size_t src = 0;
        for (size_t i = 0; i < queries.size(); i++) {
            for (int k = 0; k < n_max_cand; k++)
                padded.push_back(k < counts[i] ? pair_idx[src + (size_t)k] : pair_idx[src]);
            src += (size_t)counts[i];
        }
        pair_idx.swap(padded);
    }

    std::vector<int> img_pos;
    for (int i = 0; i < (int)ids.size(); i++) if (ids[i] == id_video_pad) img_pos.push_back(i);
    if ((int)img_pos.size() != n_img) {
        fprintf(stderr, "placeholder mismatch: %d pad tokens vs %d vision tokens\n",
                (int)img_pos.size(), n_img);
        return false;
    }

    // ---- 3-axis positions: text runs advance one slot each, a temporal patch
    // block contributes max(h, w) slots (Qwen3-VL timestamp layout: no t axis
    // advancement inside a frame, the timestamp text carries the time)
    std::vector<int> mpos;
    {
        std::vector<int> axis[3];
        axis[0].reserve(ids.size());
        axis[1].reserve(ids.size());
        axis[2].reserve(ids.size());
        int cur = 0;
        for (size_t i = 0; i < ids.size();) {
            if (ids[i] == id_video_pad) {
                const size_t end = std::min(ids.size(), i + (size_t)spatial);
                if (end - i != (size_t)spatial || (end < ids.size() && ids[end] == id_video_pad)) {
                    fprintf(stderr, "vision block is not %d tokens\n", spatial);
                    return false;
                }
                for (int l = 0; l < spatial; l++) {
                    axis[0].push_back(cur);
                    axis[1].push_back(cur + l / mw);
                    axis[2].push_back(cur + l % mw);
                }
                cur += std::max(mh, mw);
                i = end;
                continue;
            }
            axis[0].push_back(cur);
            axis[1].push_back(cur);
            axis[2].push_back(cur);
            cur++;
            i++;
        }
        mpos.reserve((size_t)4 * ids.size());
        for (int a = 0; a < 3; a++) mpos.insert(mpos.end(), axis[a].begin(), axis[a].end());
        for (int i = 0; i < (int)ids.size(); i++) mpos.push_back(i);
    }

    fprintf(stderr, "score: %d tokens (%d vision), %zu question%s, max %d candidates\n",
            (int)ids.size(), n_img, queries.size(), queries.size() == 1 ? "" : "s", n_max_cand);
    if ((int)ids.size() > n_ctx) {
        fprintf(stderr, "query group is %d tokens, n_ctx is %d -- raise --n-ctx or fewer questions\n",
                (int)ids.size(), n_ctx);
        return false;
    }

    n_past = 0;
    if (sbuf) ggml_backend_buffer_clear(sbuf, 0);
    kv.clear();
    ScoreTap tap;
    tap.n_max_cand = n_max_cand;
    tap.n_query = (int)queries.size();
    TextGraph g;
    std::vector<float> logits;
    const auto t0 = std::chrono::steady_clock::now();
    if (!run_score_text(g, ids, feats, img_pos, mpos, pair_idx, query_idx, tap, logits)) return false;
    fprintf(stderr, "[perf] score forward: %.0f ms\n",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());

    out.resize(queries.size());
    for (size_t i = 0; i < queries.size(); i++) {
        vqa::ScoreResult & r = out[i];
        r.id = queries[i].id.empty() ? std::to_string(i) : queries[i].id;
        r.task = queries[i].task;
        const int cnt = counts[i];
        std::vector<double> raw((size_t)cnt);
        for (int k = 0; k < cnt; k++) raw[(size_t)k] = (double)logits[i * (size_t)n_max_cand + (size_t)k];
        r.probabilities.resize((size_t)cnt);
        if (r.task == "choice") {
            double mx = raw[0];
            for (double v : raw) mx = std::max(mx, v);
            double sum = 0;
            for (int k = 0; k < cnt; k++) sum += std::exp(raw[(size_t)k] - mx);
            int best = 0;
            for (int k = 0; k < cnt; k++) {
                const double p = std::exp(raw[(size_t)k] - mx) / sum;
                r.probabilities[(size_t)k] = (float)p;
                if (p > r.probabilities[(size_t)best]) best = k;
            }
            r.answer_index = best;
            if (best < (int)queries[i].options.size()) r.answer = queries[i].options[(size_t)best];
        } else {
            for (int k = 0; k < cnt; k++) {
                const float p = (float)(1.0 / (1.0 + std::exp(-raw[(size_t)k])));
                r.probabilities[(size_t)k] = p;
                r.times.push_back(centers[(size_t)k]);
            }
            std::vector<float> z((size_t)cnt);
            for (int k = 0; k < cnt; k++) z[(size_t)k] = (float)raw[(size_t)k];
            decode_interval(z, threshold, jev_smooth, cell_lo, cell_hi, duration, r.start, r.end);
        }
    }
    return true;
}

#include "qwen3_5_mtp.inc"

}  // namespace qwen3_5

namespace vqa {
Model * create_qwen3_5() { return new qwen3_5::Qwen3_5(); }
}  // namespace vqa
