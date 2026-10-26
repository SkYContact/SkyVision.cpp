// qwen3vl_model.cpp - Qwen3-VL backend (OneStreamer-4B and plain Qwen3-VL densies).
//
// OneStreamer-4B is a streaming-video SFT of Qwen3-VL-4B-Instruct, so this
// backend implements the Qwen3-VL *dense* family properly instead of borrowing
// the Qwen3.5 hybrid trunk:
//
//   text   36 dense decoder layers (no linear/SSM layers), GQA 32:8, head_dim
//          128, hidden 2560, SwiGLU 9728, per-head q/k RMSNorm (plain x*w),
//          interleaved MRoPE (GGML_ROPE_TYPE_IMROPE, sections 24/20/20 over the
//          full 128 rotated dims, theta 5e6), tied embeddings, no attention
//          output gate.
//   vision 24 ViT blocks via the shared vqa::vit::block (patch 16, temporal 2,
//          learned 48x48 absolute pos table, GGML_ROPE_TYPE_VISION 2D rope,
//          final 2x2 merger: LayerNorm(hidden) -> fold -> fc1 -> GELU(erf) -> fc2).
//   deepstack (Qwen3-VL's signature): the tower's block 5/11/17 outputs go
//          through their own merger MLPs (LayerNorm over the POST-fold width
//          4*hidden, then fc1/GELU/fc2 into text width) and are added into the
//          text residual stream at the visual token positions after decoder
//          layers 0/1/2.
//
// Frames are encoded ONE per vision pass with the temporal patch duplicated
// (grid (1, h, w) each), which is the layout the processor produces for a single
// image and the one OneStreamer was trained on; the prompt then carries one
// timestamped vision block per frame ("<0.000s-0.250s>" + block), i.e. the
// streaming protocol of the reference inference script.
//
// Weight-layout notes (see convert/convert_onestreamer.py):
//   * RMSNorm is x*w (NOT Qwen3.5's x*(1+w)); the converter stores it verbatim.
//   * q_proj is a plain [n_head*head_dim] projection - no interleaved |gate half.
//   * mrope rope.dimension_count is 128 (all head dims rotated).
#include "core/vqa.hpp"
#include "core/blocks/vit_block.hpp"
#include "core/blocks/qwen3vl_tower.hpp"

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
#include "qwen3vl_preprocess.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace qwen3vl {

struct VisLayer {
    ggml_tensor *norm0_w = nullptr, *norm0_b = nullptr;
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *wqkv_w = nullptr, *wqkv_b = nullptr;
    ggml_tensor *wo_w = nullptr, *wo_b = nullptr;
    ggml_tensor *fc0_w = nullptr, *fc0_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
};

struct TxtLayer {
    ggml_tensor *norm = nullptr, *post_norm = nullptr;
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    ggml_tensor *qn = nullptr, *kn = nullptr;
    ggml_tensor *gate = nullptr, *up = nullptr, *down = nullptr;
};

// one merger MLP: LayerNorm -> fc1 -> GELU(erf) -> fc2
struct Merger {
    ggml_tensor *norm_w = nullptr, *norm_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
};

struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *mpos_in = nullptr, *mask_in = nullptr;
    ggml_tensor *img_in = nullptr, *img_pos_in = nullptr;
    ggml_tensor *logits = nullptr;
};

class Qwen3VL : public vqa::Model {
public:
    Qwen3VL() { caps_.image = true; caps_.video = true; }
    ~Qwen3VL() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "qwen3-vl"; }

    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp,
             std::string & out_text) override;

    // Realtime streaming session (OneStreamer protocol): turn 1 seeds the system
    // prompt + the question and every turn appends the newly arrived frames
    // (each with its [t0,t1) label) to a sliding window, answers, and keeps the
    // answer as an assistant turn of the next replay. Vision features of older
    // frames are cached, so the tower only ever runs on new frames.
    bool ask_session(const vqa::MediaInput & media, const std::string & prompt,
                     const vqa::GenParams & gp, std::string & out_text) override;
    void session_reset() override;

    // ChatML (Qwen3-VL chat_template.jinja): prior turns for multi-turn sessions
    std::string format_history_turn(const std::string & user,
                                    const std::string & assistant) const override {
        return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
             + assistant + "<|im_end|>\n";
    }
    // streaming / proactive front ends: a system turn carries the OVO-style
    // three-state protocol (</Silence> </Standby> </Response>)
    std::string format_system(const std::string & sys) const override {
        return sys.empty() ? std::string()
                           : "<|im_start|>system\n" + sys + "<|im_end|>\n";
    }
    const char * silence_piece() const override { return "</Silence>"; }
    // OneStreamer's trained protocol (verbatim from the reference
    // vlmeval/dataset/ovo_timing.py:STREAMING_SYS_CONSERVATIVE_THREE_PROTOCOL)
    const char * realtime_system_prompt() const override;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool alloc_state();

    // vision: one pass per frame (temporal patch = the frame twice)
    bool encode_vision(const vqa::MediaInput & media, std::vector<float> & feats,
                       int * n_tokens, int * grid /* steps, mh, mw */, int token_budget);
    bool build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all = false);
    bool run_text(TextGraph & g, const std::vector<int> & toks,
                  const std::vector<float> * img_feats, const std::vector<int> * img_pos,
                  const std::vector<int> & mpos, std::vector<float> & logits_out);
    std::string render_prompt(const std::string & user_text, bool video,
                             const std::vector<std::pair<double, double>> * marks,
                             const std::vector<int> * mark_tokens) const;
    // prompt ids + vision features -> answer (shared by ask / ask_session).
    // resume = the KV cache already holds a matching id prefix, so only the new
    // tail is prefilled and n_past is kept.
    bool generate(const std::vector<int> & ids, const std::vector<float> * feats,
                  const std::vector<int> & img_pos, const std::vector<int> & mpos,
                  const vqa::GenParams & gp, std::string & out_text, bool resume = false);
    // mrope (t, h, w) + absolute KV slot per prompt token, one visual run per frame
    void build_mpos(const std::vector<int> & ids, const std::vector<int> & img_pos,
                    int n_img, int mh, int mw, std::vector<int> & mpos);
    int sample(const std::vector<float> & logits, const vqa::GenParams & gp);
    void dump(const ggml_cgraph * gf, const char * name);

    // ---- realtime session state ------------------------------------------
    bool s_active = false;
    int s_max_frames = 64;                  // sliding window (reference default 120)
    int s_mh = 0, s_mw = 0, s_per_frame = 0;
    std::string s_sys;                      // formatted system turn (may be empty)
    std::string s_q;                        // the global question
    std::vector<float> s_feats;             // [n_embd, n_frames * per_frame]
    std::vector<std::pair<double, double>> s_marks;      // per retained frame
    // every token id actually written into the KV cache (prompt + generated),
    // so a later turn can prove its prompt is an extension and splice instead
    // of re-prefilling the whole window
    std::vector<int> s_kv_ids;
    // one entry per completed turn: how many frames it contributed + its answer
    // (the reference StreamingSession replays every turn's frames + answer)
    struct STurn { int frames = 0; std::string ans; };
    std::vector<STurn> s_turns;

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    // text hyperparameters (config.json of Qwen3-VL-4B)
    int n_layer = 36, n_embd = 2560, n_head = 32, n_kv_head = 8, head_dim = 128;
    int n_ff = 9728, n_vocab = 151936;
    float rms_eps = 1e-6f, rope_theta = 5e6f;
    int rope_dim = 128;
    int mrope_sections[4] = {24, 20, 20, 0};

    VisConfig vc;
    int n_ds = 0;

    int n_ctx = 8192;
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr, wbuf_plain = nullptr;
    vqa::AlignedBuf host_f32_storage;
    vqa::MMap mm;
    std::vector<float> patch_bias;        // v.patch_embd.bias (host f32, folded into pos)
    std::vector<float> pos_table_host;    // [pos_side^2, n_embd]
    vqa::StageBuffer tower_buf;           // device-side tower output [n_embd, N]
    vqa::StageBuffer ds_buf[8];           // DeepStack tap output [n_embd, N] each

    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr;
    std::vector<TxtLayer> tl;
    std::vector<VisLayer> vl;
    ggml_tensor *v_patch_w = nullptr, *v_patch_b = nullptr;
    Merger mm_;                           // main merger
    std::vector<Merger> dsm;              // DeepStack mergers (one per tap)
    int ds_idx[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    int id_image_pad = 151655, id_video_pad = 151656;
    int id_vision_start = 151652, id_vision_end = 151653;
    int id_eos = 0, id_endoftext = 151643;

    vqa::KvCache kv;                      // all n_layer layers are full attention
    ggml_gallocr_t allocr = nullptr;
    int n_past = 0;
    std::vector<int> last_mpos;           // 3-axis mrope position of the last token
    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

// ---------------------------------------------------------------- helpers
static float randf_(uint64_t & s) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return (float)(((s * 0x2545F4914F6CDD1Dull) >> 11) * (1.0 / 900719254740992.0));
}

Qwen3VL::~Qwen3VL() {
    kv.free();
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wbuf_plain) ggml_backend_buffer_free(wbuf_plain);
    if (wctx) ggml_free(wctx);
    if (allocr) ggml_gallocr_free(allocr);
    if (backend) ggml_backend_free(backend);
}

bool Qwen3VL::init_backend() {
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

// ---------------------------------------------------------------- load
bool Qwen3VL::load(const std::string & gguf_path, const std::string & tokenizer_path,
                   const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "QWEN3VL", n_threads, use_gpu, dump_dir);
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::Qwen2Regex)) return false;
    if (tok_.size() > 0) {
        const int64_t diff = tok_.size() > n_vocab ? tok_.size() - (int64_t)n_vocab
                                                   : (int64_t)n_vocab - tok_.size();
        if (diff * 5 > n_vocab) {
            fprintf(stderr, "qwen3vl: tokenizer vocab %d vs model vocab %d - wrong tokenizer.json\n",
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
    id_eos = vqa::pick_token(tok_, "<|im_end|>", 151645);

    caps_ = {};
    caps_.image = true;
    caps_.video = true;
    caps_.proactive_states = true;   // </Silence> </Standby> </Response> protocol
    caps_.timestamps_prompt = true;  // "<0.000s-0.250s>" per frame in the prompt
    fprintf(stderr, "qwen3vl: ready (%d text layers, %d vision layers%s, ctx %d)\n",
            n_layer, vc.n_layer, n_ds ? " + deepstack" : "", n_ctx);
    return true;
}

bool Qwen3VL::load_weights(const char * path) {
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

    id_image_pad = vqa::gguf::u32(gu, "marlin.image_token_id", id_image_pad);
    id_video_pad = vqa::gguf::u32(gu, "marlin.video_token_id", id_video_pad);
    id_vision_start = vqa::gguf::u32(gu, "marlin.vision_start_token_id", id_vision_start);
    id_vision_end = vqa::gguf::u32(gu, "marlin.vision_end_token_id", id_vision_end);
    id_endoftext = vqa::gguf::u32(gu, "marlin.pad_token_id", id_endoftext);

    // DeepStack tap indices (vision block per text layer 0..n-1)
    {
        int64_t id = gguf_find_key(gu, "marlin.deepstack_indexes");
        if (id >= 0 && gguf_get_kv_type(gu, id) == GGUF_TYPE_ARRAY) {
            const uint32_t * arr = (const uint32_t *)gguf_get_arr_data(gu, id);
            const int n = std::min((int)gguf_get_arr_n(gu, id),
                                   (int)(sizeof(ds_idx) / sizeof(ds_idx[0])));
            for (int i = 0; i < n; i++) ds_idx[i] = (int)arr[i];
            n_ds = n;
        }
    }

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);

    // 1D tensors (norms, biases) -> host F32
    vqa::AlignedBuf host_f32;
    vqa::weights::promote_1d_to_f32(wctx, host_f32);

    {
        ggml_tensor * pe = ggml_get_tensor(wctx, "v.pos_emb.weight");
        if (!pe) { fprintf(stderr, "missing v.pos_emb.weight\n"); return false; }
        pos_table_host.resize(ggml_nelements(pe));
        vqa::gguf::tensor_to_f32(pe, pos_table_host.data());
    }
    // patch embedding bias is folded into the host pos rows (one graph add less)
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
    v_patch_b = ggml_get_tensor(wctx, "v.patch_embd.bias");   // optional (folded on the host)
    mm_.norm_w = gt("mm.norm.weight"); mm_.norm_b = gt("mm.norm.bias");
    mm_.fc1_w  = gt("mm.fc1.weight");  mm_.fc1_b  = gt("mm.fc1.bias");
    mm_.fc2_w  = gt("mm.fc2.weight");  mm_.fc2_b  = gt("mm.fc2.bias");

    tl.resize(n_layer);
    for (int i = 0; i < n_layer; i++) {
        char buf[128];
        auto g = [&](const char * suffix) {
            snprintf(buf, sizeof(buf), "blk.%d.%s", i, suffix);
            return gt(buf);
        };
        TxtLayer & L = tl[i];
        L.norm = g("attn_norm.weight");
        L.post_norm = g("attn_post_norm.weight");
        L.q = g("attn_q.weight");
        L.k = g("attn_k.weight");
        L.v = g("attn_v.weight");
        L.o = g("attn_output.weight");
        L.qn = g("attn_q_norm.weight");
        L.kn = g("attn_k_norm.weight");
        L.gate = g("ffn_gate.weight");
        L.up = g("ffn_up.weight");
        L.down = g("ffn_down.weight");
    }

    vl.resize(vc.n_layer);
    for (int i = 0; i < vc.n_layer; i++) {
        char buf[128];
        auto g = [&](const char * suffix) {
            snprintf(buf, sizeof(buf), "v.blk.%d.%s", i, suffix);
            return gt(buf);
        };
        VisLayer & L = vl[i];
        L.norm0_w = g("norm0.weight"); L.norm0_b = g("norm0.bias");
        L.norm1_w = g("norm1.weight"); L.norm1_b = g("norm1.bias");
        L.wqkv_w  = g("wqkv.weight");  L.wqkv_b  = g("wqkv.bias");
        L.wo_w    = g("wo.weight");    L.wo_b    = g("wo.bias");
        L.fc0_w   = g("mlp.fc0.weight"); L.fc0_b = g("mlp.fc0.bias");
        L.fc1_w   = g("mlp.fc1.weight"); L.fc1_b = g("mlp.fc1.bias");
    }
    if (n_ds > 0) {
        dsm.resize(n_ds);
        for (int k = 0; k < n_ds; k++) {
            char buf[128];
            auto g = [&](const char * suffix) {
                snprintf(buf, sizeof(buf), "dsm.%d.%s", k, suffix);
                return gt(buf);
            };
            Merger & M = dsm[k];
            M.norm_w = g("norm.weight"); M.norm_b = g("norm.bias");
            M.fc1_w  = g("fc1.weight");  M.fc1_b  = g("fc1.bias");
            M.fc2_w  = g("fc2.weight");  M.fc2_b  = g("fc2.bias");
            if (!M.norm_w || !M.fc1_w || !M.fc2_w) {
                fprintf(stderr, "deepstack merger %d is incomplete\n", k);
                return false;
            }
        }
        fprintf(stderr, "deepstack: %d tap%s (vision blocks", n_ds, n_ds == 1 ? "" : "s");
        for (int k = 0; k < n_ds; k++) fprintf(stderr, " %d", ds_idx[k]);
        fprintf(stderr, ")\n");
    }

    if (!tok_embd || !out_norm || !v_patch_w) return false;
    if ((int)tok_embd->ne[0] != n_embd) {
        fprintf(stderr, "token_embd row length %lld != n_embd %d (tied head expected)\n",
                (long long)tok_embd->ne[0], n_embd);
        return false;
    }
    if (vc.out_hidden != n_embd) {
        fprintf(stderr, "vision out_hidden %d != text width %d\n", vc.out_hidden, n_embd);
        return false;
    }
    return true;
}

bool Qwen3VL::alloc_state() {
    return kv.alloc(backend, n_layer, n_kv_head, head_dim, n_ctx);
}

void Qwen3VL::dump(const ggml_cgraph * gf, const char * name) {
    if (dump_dir.empty()) return;
    vqa::dump::write((ggml_cgraph *)gf, dump_dir.c_str(), name);
}

// ---------------------------------------------------------------- vision
bool Qwen3VL::encode_vision(const vqa::MediaInput & media, std::vector<float> & feats,
                            int * n_tokens, int * grid, int token_budget) {
    *n_tokens = 0;
    grid[0] = grid[1] = grid[2] = 0;
    if (media.empty()) return true;
    const int T = (int)media.frames.size();
    const int src_h = media.height, src_w = media.width;
    const int factor = vc.patch * vc.merge;                     // 32

    long long max_pixels = media.kind == vqa::MediaKind::Video ? 200704 : 16384LL * 32 * 32;
    if (media.kind == vqa::MediaKind::Video) {
        if (const char * e = getenv("QWEN3VL_VIDEO_MP"))
            max_pixels = (long long)(atof(e) * 1e6 + 0.5);
    }
    if (token_budget > 0) {
        // text tokens per frame = (gh/2)*(gw/2) = H*W/(32*32); one step per frame
        const long long min_total = 4LL * T;
        if (token_budget < min_total) {
            fprintf(stderr, "video cannot fit in n_ctx: needs >= %d vision tokens, budget %d\n",
                    (int)min_total, token_budget);
            return false;
        }
        const long long per_step = token_budget / T;
        max_pixels = std::min(max_pixels, per_step * factor * factor);
        if (max_pixels < 200704)
            fprintf(stderr, "vision budget %d tokens (%d frame%s): max_pixels clamped to %.2f MP\n",
                    token_budget, T, T == 1 ? "" : "s", max_pixels / 1e6);
    }
    int H = 0, W = 0;
    if (!vqa::img::smart_resize(src_h, src_w, factor, 4 * factor * factor,
                                (int)std::min(max_pixels, (long long)INT_MAX), &H, &W)) {
        fprintf(stderr, "image too small (%dx%d, factor %d)\n", src_w, src_h, factor);
        return false;
    }
    if (media.kind == vqa::MediaKind::Video)
        fprintf(stderr, "video %d frame%s %dx%d -> %dx%d each\n",
                T, T == 1 ? "" : "s", src_w, src_h, W, H);
    else
        fprintf(stderr, "image %dx%d -> %dx%d\n", src_w, src_h, W, H);

    // resize once, patchify per frame with the temporal slice duplicated (the
    // processor's single-image layout: grid (1, gh, gw) per frame)
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
        patchify3d(fs, H, W, vc, pos_table_host.data(), p, po, co, &gh, &gw);
        patches.insert(patches.end(), p.begin(), p.end());
        pos.insert(pos.end(), po.begin(), po.end());
        coords.insert(coords.end(), co.begin(), co.end());
    }
    const int hw = gh * gw;
    const int mh = gh / vc.merge, mw = gw / vc.merge;
    const int vhd = vc.n_embd / vc.n_head;
    const int C3 = 3 * vc.temporal_patch * vc.patch * vc.patch;
    if (!patch_bias.empty())
        vqa::vit::fold_patch_bias(pos.data(), pos.size(), vc.n_embd, patch_bias.data());

    const char * stop_after = getenv("QWEN3VL_STOP_AFTER");

    // graph side lives in the shared tower block (qwen3vl_tower.hpp); this
    // backend supplies the preprocessed frames + weights + staging buffers
    {
        std::vector<vqa::qwen3vl_tower::FrameInput> fi(T);
        for (int s = 0; s < T; s++) {
            fi[s].patches = patches.data() + (size_t)s * hw * C3;
            fi[s].pos = pos.data() + (size_t)s * hw * vc.n_embd;
            fi[s].coords = coords.data() + (size_t)s * hw * 2;
        }
        vqa::qwen3vl_tower::Weights tw;
        tw.patch_w = v_patch_w;
        tw.layers.resize(vl.size());
        for (size_t k = 0; k < vl.size(); k++) {
            const VisLayer & L = vl[k];
            tw.layers[k] = {L.norm0_w, L.norm0_b, L.norm1_w, L.norm1_b,
                            L.wqkv_w, L.wqkv_b, L.wo_w, L.wo_b,
                            L.fc0_w, L.fc0_b, L.fc1_w, L.fc1_b};
        }
        tw.main = {mm_.norm_w, mm_.norm_b, mm_.fc1_w, mm_.fc1_b, mm_.fc2_w, mm_.fc2_b};
        tw.ds.resize(dsm.size());
        for (size_t k = 0; k < dsm.size(); k++)
            tw.ds[k] = {dsm[k].norm_w, dsm[k].norm_b, dsm[k].fc1_w, dsm[k].fc1_b, dsm[k].fc2_w, dsm[k].fc2_b};
        vqa::qwen3vl_tower::Config tc;
        tc.n_layer = vc.n_layer; tc.n_embd = vc.n_embd; tc.n_head = vc.n_head;
        tc.n_ff = vc.n_ff; tc.patch = vc.patch; tc.temporal_patch = vc.temporal_patch;
        tc.merge = vc.merge; tc.layernorm_eps = vc.layernorm_eps;
        tc.rope_theta = vc.rope_theta;
        for (int k = 0; k < n_ds; k++) tc.deepstack.push_back(ds_idx[k]);
        vqa::qwen3vl_tower::RunCtx rc;
        rc.backend = backend; rc.allocr = allocr; rc.cfg = &tc; rc.w = &tw;
        rc.text_width = n_embd; rc.tower_buf = &tower_buf;
        rc.ds_buf = n_ds > 0 ? ds_buf : nullptr;
        rc.env_prefix = "QWEN3VL"; rc.dump_dir = dump_dir;
        vqa::qwen3vl_tower::run(rc, fi.data(), T, H, W, gh, gw, feats, n_tokens);
        // parity stop points (v_patch_out / v_final) return early inside the
        // tower with a filled feats; mirror the old early-return semantics
        const char * stop_after = getenv("QWEN3VL_STOP_AFTER");
        if (stop_after && (!strcmp(stop_after, "v_patch_out") || !strcmp(stop_after, "v_final"))) {
            *n_tokens = (int)(feats.size() / vc.n_embd);
            grid[0] = T; grid[1] = mh; grid[2] = mw;
            return true;
        }
    }
    grid[0] = T; grid[1] = mh; grid[2] = mw;
    const int N = hw * T;
    fprintf(stderr, "vision: %d patch%s (%dx%d, %d frame%s) -> %d tokens%s\n",
            N, N == 1 ? "" : "es", gh, gw, T, T == 1 ? "" : "s", *n_tokens,
            stop_after ? " (truncated for parity)" : "");
    return true;
}

// ---------------------------------------------------------------- text
bool Qwen3VL::build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all) {
    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 32768, false);

    const int D = n_embd, HD = head_dim, NH = n_head, NKV = n_kv_head;

    g.tok_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    // 3-axis (t, h, w) mrope positions + 1D absolute slot positions interleaved:
    // layout [t(T), h(T), w(T), slot(T)]
    g.mpos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, 4 * T);
    ggml_set_name(g.mpos_in, "mpos_in");
    g.mask_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, S, T);
    ggml_set_name(g.mask_in, "mask_in");

    ggml_tensor * x = ggml_get_rows(g.ctx, tok_embd, g.tok_in);       // [D, T]
    if (n_img > 0) {
        g.img_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, n_img);
        ggml_set_name(g.img_in, "img_in");
        g.img_pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_img);
        ggml_set_name(g.img_pos_in, "img_pos_in");
        x = ggml_set_rows(g.ctx, ggml_dup(g.ctx, x), g.img_in, g.img_pos_in);
    }
    ggml_set_name(x, "emb");

    ggml_tensor * pos_rope = ggml_view_1d(g.ctx, g.mpos_in, 4 * T, 0);
    ggml_tensor * pos_slot = ggml_view_1d(g.ctx, g.mpos_in, T, (size_t)3 * T * sizeof(int));

    int max_layers = n_layer;
    if (const char * e = getenv("QWEN3VL_MAX_LAYERS")) max_layers = std::min(max_layers, atoi(e));

    for (int il = 0; il < max_layers; il++) {
        TxtLayer & L = tl[il];
        ggml_tensor * l = ggml_rms_norm(g.ctx, x, rms_eps);
        l = ggml_mul(g.ctx, l, L.norm);
        if (il == 0) ggml_set_name(l, "t_l0_norm");

        ggml_tensor * q = ggml_mul_mat(g.ctx, L.q, l);                // [NH*HD, T]
        ggml_tensor * k = ggml_mul_mat(g.ctx, L.k, l);
        ggml_tensor * v = ggml_mul_mat(g.ctx, L.v, l);
        if (il == 0) { ggml_set_name(q, "t_l0_q"); ggml_set_name(k, "t_l0_k"); }

        ggml_tensor * q3 = ggml_reshape_3d(g.ctx, q, HD, NH, T);
        q3 = ggml_rms_norm(g.ctx, q3, rms_eps);
        q3 = ggml_mul(g.ctx, q3, L.qn);
        ggml_tensor * k3 = ggml_reshape_3d(g.ctx, k, HD, NKV, T);
        k3 = ggml_rms_norm(g.ctx, k3, rms_eps);
        k3 = ggml_mul(g.ctx, k3, L.kn);
        if (il == 0) { ggml_set_name(q3, "t_l0_q_norm"); ggml_set_name(k3, "t_l0_k_norm"); }

        // interleaved MRoPE (Qwen3-VL): all 128 head dims rotated, sections 24/20/20
        q3 = ggml_rope_multi(g.ctx, q3, pos_rope, nullptr, rope_dim, mrope_sections,
                             GGML_ROPE_TYPE_IMROPE, n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_multi(g.ctx, k3, pos_rope, nullptr, rope_dim, mrope_sections,
                             GGML_ROPE_TYPE_IMROPE, n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (il == 0) { ggml_set_name(q3, "t_l0_q_rope"); ggml_set_name(k3, "t_l0_k_rope"); }

        ggml_tensor * k2 = ggml_reshape_2d(g.ctx, k3, NKV * HD, T);
        ggml_tensor * v2 = ggml_reshape_2d(g.ctx, v, NKV * HD, T);
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.k[il], k2, pos_slot));
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.v[il], v2, pos_slot));

        ggml_tensor * kf = ggml_view_3d(g.ctx, kv.k[il], HD, S, NKV,
                                        kv.k[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * vf = ggml_view_3d(g.ctx, kv.v[il], HD, S, NKV,
                                        kv.v[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * qf = ggml_view_3d(g.ctx, q3, HD, T, NH, q3->nb[2], q3->nb[1], 0);
        ggml_tensor * fa = ggml_flash_attn_ext(g.ctx, qf, kf, vf, g.mask_in,
                                              1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, fa), NH * HD, T);
        if (il == 0) ggml_set_name(attn, "t_l0_attn");
        attn = ggml_mul_mat(g.ctx, L.o, attn);
        x = ggml_add(g.ctx, x, attn);
        if (il == 0) ggml_set_name(x, "t_l0_resid");

        ggml_tensor * m = ggml_rms_norm(g.ctx, x, rms_eps);
        m = ggml_mul(g.ctx, m, L.post_norm);
        ggml_tensor * gate = ggml_mul_mat(g.ctx, L.gate, m);
        ggml_tensor * up = ggml_mul_mat(g.ctx, L.up, m);
        m = ggml_swiglu_split(g.ctx, gate, up);
        if (il == 0) ggml_set_name(m, "t_l0_swiglu");
        m = ggml_mul_mat(g.ctx, L.down, m);
        x = ggml_add(g.ctx, x, m);
        if (il == 0) ggml_set_name(x, "t_l0_out");

        // DeepStack: after decoder layer il, add tap il's merged tower features at
        // the visual token positions only (zero elsewhere)
        if (n_ds > 0 && il < n_ds && n_img > 0 && ds_buf[il].t && ds_buf[il].t->ne[1] >= n_img) {
            ggml_tensor * dk = ggml_view_2d(g.ctx, ds_buf[il].t, D, n_img,
                                            ds_buf[il].t->nb[1], 0);
            ggml_tensor * z = ggml_set_rows(g.ctx, ggml_scale(g.ctx, x, 0.0f), dk, g.img_pos_in);
            if (il == 0) ggml_set_name(z, "t_l0_deepstack");
            x = ggml_add(g.ctx, x, z);
        }
    }

    x = ggml_rms_norm(g.ctx, x, rms_eps);
    x = ggml_mul(g.ctx, x, out_norm);
    ggml_set_name(x, "t_final_norm");

    if (logits_all) {
        g.logits = ggml_mul_mat(g.ctx, tok_embd, x);          // tied head
    } else {
        ggml_tensor * last = ggml_view_2d(g.ctx, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
        g.logits = ggml_mul_mat(g.ctx, tok_embd, last);       // tied head
    }
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);

    if (!dump_dir.empty()) {
        static const char * const dump_names[] = {
            "emb", "t_l0_norm", "t_l0_q", "t_l0_q_norm", "t_l0_q_rope", "t_l0_attn",
            "t_l0_deepstack", "t_l0_swiglu", "t_l0_out", "t_final_norm", "logits", nullptr,
        };
        vqa::dump::mark_sinks(g.ctx, g.gf, dump_names);
        vqa::dump::finalize_sinks(g.ctx, g.gf, backend);
    }
    return true;
}

bool Qwen3VL::run_text(TextGraph & g, const std::vector<int> & toks,
                       const std::vector<float> * img_feats, const std::vector<int> * img_pos,
                       const std::vector<int> & mpos, std::vector<float> & logits_out) {
    const int T = (int)toks.size();
    const int S = n_past + T;
    if (S > n_ctx) { fprintf(stderr, "context overflow (%d > %d)\n", S, n_ctx); return false; }
    const int n_img = (img_feats && img_pos) ? (int)(img_feats->size() / n_embd) : 0;
    static const bool prof = getenv("QWEN3VL_PROF") != nullptr;
    static double tb = 0, ta = 0, tc = 0; static int ns = 0;
    auto tick = []() { return std::chrono::steady_clock::now(); };
    auto tp0 = tick();

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
    auto tp3 = tick();
    if (prof) {
        ns++;
        tb += std::chrono::duration<double,std::milli>(tp1-tp0).count();
        ta += std::chrono::duration<double,std::milli>(tp2-tp1).count();
        tc += std::chrono::duration<double,std::milli>(tp3-tp2).count();
        if (ns % 50 == 0)
            fprintf(stderr, "[prof] steps=%d build %.2fms alloc+input %.2fms compute %.2fms (avg)\n",
                    ns, tb/ns, ta/ns, tc/ns);
    }
    // parity dumps: only the prefill graph is interesting, each decode step would
    // overwrite the same sink names with a T=1 tensor
    static const bool dump_prefill_only = getenv("QWEN3VL_PREFILL_DUMP_ONLY") != nullptr;
    if (!dump_prefill_only || T > 1) {
        dump(g.gf, "emb");
        dump(g.gf, "t_l0_q_rope");
        dump(g.gf, "t_l0_out");
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

// Prompt layout: chat_template.jinja (Qwen3-VL-Instruct) - a user turn whose
// content is "<vision block(s)><text>"; video frames carry the streaming
// "<0.000s-0.250s>" interval label the model was trained with, and every frame
// gets its own vision block (per-frame image protocol, see the header note).
std::string Qwen3VL::render_prompt(const std::string & user_text, bool video,
                                   const std::vector<std::pair<double, double>> * marks,
                                   const std::vector<int> * mark_tokens) const {
    std::string s = "<|im_start|>user\n";
    // every frame is encoded as its own image (grid (1, h, w)), so the prompt
    // uses image blocks even for video - a <|video_pad|> run would tell the model
    // to expect one fused (t/2, h, w) grid, which this backend never builds
    const char * pad = "<|image_pad|>";
    (void)video;
    if (mark_tokens && !mark_tokens->empty()) {
        char ts[64];
        for (size_t i = 0; i < mark_tokens->size(); i++) {
            if (marks && i < marks->size()) {
                // "<0.000s-0.250s>\n<image>" - the reference StreamingSession puts
                // the interval label on its own line before the frame's block
                snprintf(ts, sizeof(ts), "<%.3fs-%.3fs>\n", (*marks)[i].first, (*marks)[i].second);
                s += ts;
            }
            s += "<|vision_start|>";
            for (int k = 0; k < (*mark_tokens)[i]; k++) s += pad;
            s += "<|vision_end|>";
        }
    }
    s += user_text;
    s += "<|im_end|>\n<|im_start|>assistant\n";
    return s;
}

// The tokenizer's decode() drops added-special tokens, but the streaming
// protocol IS those tokens - map the five OneStreamer control ids back to their
// text so silence/standby/response/memory markers survive into the output.
static const char * control_piece(int id) {
    switch (id) {
    case 151669: return "</Response>";
    case 151670: return "</Standby>";
    case 151671: return "</Silence>";
    case 151672: return "</Summary>";
    case 151673: return "</Observe>";
    default: return nullptr;
    }
}

int Qwen3VL::sample(const std::vector<float> & logits, const vqa::GenParams & gp) {
    return vqa::sample_logits(logits, gp, rng_state);
}

bool Qwen3VL::ask(const vqa::MediaInput & media, const std::string & prompt,
                  const std::string & /*mode*/, const vqa::GenParams & gp,
                  std::string & out_text) {
    std::vector<float> feats;
    int n_img = 0;
    int grid[3] = {0, 0, 0};     // (frames, merged h, merged w)
    const auto tv0 = std::chrono::steady_clock::now();
    const bool video_kind = media.kind == vqa::MediaKind::Video;

    // per-frame prompt timestamps: "<t0s-t1s>" costs ~7 tokens per frame
    const int n_frames = (int)media.frames.size();
    {
        const int text_overhead = (int)tok_.encode(gp.history +
            render_prompt(prompt, video_kind, nullptr, nullptr)).size();
        const int stamp_cost = video_kind ? 8 * n_frames : 0;
        const int budget = n_ctx - gp.max_tokens - text_overhead - stamp_cost - 16;
        if (budget <= 0) {
            fprintf(stderr, "n_ctx %d too small: prompt %d tokens + max_new %d leaves nothing for the image\n",
                    n_ctx, text_overhead, gp.max_tokens);
            return false;
        }
        // media cache: the same picture re-encoded on every follow-up question
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
        key ^= ((uint64_t)media.width << 32) ^ (uint64_t)media.height;
        if (vcache.key == key && vcache.n > 0) {
            feats = vcache.f; n_img = vcache.n; memcpy(grid, vcache.g, sizeof(grid));
            fprintf(stderr, "vision: cache hit (%d tokens)\n", n_img);
        } else if (encode_vision(media, feats, &n_img, grid, budget)) {
            vcache.key = key; vcache.f = feats; vcache.n = n_img;
            memcpy(vcache.g, grid, sizeof(grid));
        } else {
            return false;
        }
    }
    fprintf(stderr, "[perf] vision encode: %.0f ms (%d tokens)\n",
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tv0).count(),
            n_img);

    // one vision block per frame, each with its [t0, t1) interval label
    std::vector<std::pair<double, double>> marks;
    std::vector<int> mark_tokens;
    if (video_kind && grid[0] > 0) {
        const int per_frame = (grid[1] * grid[2]);
        const double def_fps = media.fps > 0 ? media.fps : 1.0;
        for (int i = 0; i < grid[0]; i++) {
            const double t0 = i < (int)media.timestamps.size() ? media.timestamps[i] : i / def_fps;
            const double t1 = t0 + 1.0 / def_fps;
            marks.push_back({ t0, t1 });
            mark_tokens.push_back(per_frame);
        }
    } else if (grid[0] == 1) {
        marks.clear();
        mark_tokens.assign(1, grid[1] * grid[2]);
    }

    const std::string text = gp.history +
        render_prompt(prompt, video_kind, &marks, &mark_tokens);
    const std::vector<int> ids = tok_.encode(text);
    if (getenv("QWEN3VL_DUMP_IDS")) {
        fprintf(stderr, "prompt ids:");
        for (int id : ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }
    const int pad_id = id_image_pad;    // per-frame image blocks (see render_prompt)
    std::vector<int> img_pos;
    for (int i = 0; i < (int)ids.size(); i++) if (ids[i] == pad_id) img_pos.push_back(i);
    if ((int)img_pos.size() != n_img) {
        fprintf(stderr, "placeholder mismatch: %d pad tokens vs %d vision tokens\n",
                (int)img_pos.size(), n_img);
        if (n_img > 0) return false;
    }

    std::vector<int> mpos;
    build_mpos(ids, img_pos, n_img, grid[1], grid[2], mpos);
    return generate(ids, n_img > 0 ? &feats : nullptr, img_pos, mpos, gp, out_text);
}

// mrope position ids (get_rope_index): text runs share (p, p, p); every vision
// block walks its merged grid as (t = cur, h, w) and the running counter then
// advances by max(mh, mw). The fourth plane carries the absolute KV slot.
void Qwen3VL::build_mpos(const std::vector<int> & ids, const std::vector<int> & img_pos,
                         int n_img, int mh, int mw, std::vector<int> & mpos) {
    mpos.clear();
    mpos.reserve((size_t)4 * ids.size());
    std::vector<int> axis[3];
    const int per = mh * mw;
    const int n_blk = (per > 0 && n_img > 0) ? n_img / per : 0;
    if (n_img > 0 && !img_pos.empty()) {
        const size_t v0 = (size_t)img_pos[0];
        int cur = 0;
        for (size_t i = 0; i < v0; i++) {
            axis[0].push_back(cur); axis[1].push_back(cur); axis[2].push_back(cur);
            cur++;
        }
        for (int b = 0; b < n_blk; b++) {
            for (int k = 0; k < per; k++) {
                axis[0].push_back(cur);
                axis[1].push_back(cur + k / mw);
                axis[2].push_back(cur + k % mw);
            }
            cur += std::max(mh, mw);
        }
        for (size_t i = v0 + (size_t)n_img; i < ids.size(); i++) {
            axis[0].push_back(cur); axis[1].push_back(cur); axis[2].push_back(cur);
            cur++;
        }
    } else {
        for (size_t i = 0; i < ids.size(); i++) {
            axis[0].push_back((int)i); axis[1].push_back((int)i); axis[2].push_back((int)i);
        }
    }
    mpos.insert(mpos.end(), axis[0].begin(), axis[0].end());
    mpos.insert(mpos.end(), axis[1].begin(), axis[1].end());
    mpos.insert(mpos.end(), axis[2].begin(), axis[2].end());
    for (int i = 0; i < (int)ids.size(); i++) mpos.push_back(i);    // absolute KV slots
    last_mpos = {axis[0].back(), axis[1].back(), axis[2].back()};
}

bool Qwen3VL::generate(const std::vector<int> & ids, const std::vector<float> * feats,
                       const std::vector<int> & img_pos, const std::vector<int> & mpos,
                       const vqa::GenParams & gp, std::string & out_text, bool resume) {
    const int n_img = (feats && !img_pos.empty()) ? (int)(feats->size() / n_embd) : 0;
    const int n_past0 = resume ? n_past : 0;
    fprintf(stderr, "prompt: %d tokens (%d image, %d new%s), max_new %d\n",
            n_past0 + (int)ids.size(), n_img, (int)ids.size(),
            resume ? ", KV splice" : "", gp.max_tokens);
    int max_new = gp.max_tokens;
    if (n_past0 + (int)ids.size() + max_new > n_ctx) {
        max_new = n_ctx - n_past0 - (int)ids.size();
        fprintf(stderr, "warning: prompt %d + max_new %d > n_ctx %d, clamping max_new to %d\n",
                n_past0 + (int)ids.size(), gp.max_tokens, n_ctx, max_new);
        if (max_new <= 0) return false;
    }

    if (!resume) {
        n_past = 0;
        kv.clear();
        s_kv_ids.clear();
    }
    auto t0 = std::chrono::steady_clock::now();
    TextGraph g;
    std::vector<float> logits;
    if (!run_text(g, ids, n_img > 0 ? feats : nullptr, n_img > 0 ? &img_pos : nullptr, mpos, logits))
        return false;
    s_kv_ids.insert(s_kv_ids.end(), ids.begin(), ids.end());
    auto t1 = std::chrono::steady_clock::now();

    out_text.clear();
    int next = sample(logits, gp);
    if (getenv("QWEN3VL_DEBUG")) fprintf(stderr, "first token: %d\n", next);
    int n_dec = 0;
    std::vector<int> step_pos(4), step_tok(1);
    for (int step = 0; step < max_new; step++) {
        if (next == id_eos || next == id_endoftext) break;
        std::string piece = tok_.decode({next});
        if (piece.empty())
            if (const char * cp = control_piece(next)) piece = cp;
        out_text += piece;
        if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
        n_dec++;
        if (step + 1 >= max_new) break;
        step_pos[0] = last_mpos[0] + 1;
        step_pos[1] = last_mpos[1] + 1;
        step_pos[2] = last_mpos[2] + 1;
        step_pos[3] = n_past;
        step_tok[0] = next;
        if (!run_text(g, step_tok, nullptr, nullptr, step_pos, logits)) break;
        s_kv_ids.push_back(next);
        last_mpos = {step_pos[0], step_pos[1], step_pos[2]};
        next = sample(logits, gp);
    }
    auto t2 = std::chrono::steady_clock::now();
    const double ms_prefill = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double ms_decode = std::chrono::duration<double, std::milli>(t2 - t1).count();
    fprintf(stderr, "[perf] prefill %d tok in %.0f ms (%.1f tok/s) | decode %d tok in %.0f ms (%.2f tok/s)\n",
            (int)ids.size(), ms_prefill, ms_prefill > 0 ? ids.size() * 1000.0 / ms_prefill : 0.0,
            n_dec, ms_decode, ms_decode > 0 ? n_dec * 1000.0 / ms_decode : 0.0);
    return true;
}

// ---------------------------------------------------------------- session
// The three-state streaming protocol the model was SFT'd on. Verbatim copy of
// STREAMING_SYS_CONSERVATIVE_THREE_PROTOCOL in the reference eval kit
// (Eval/VLMEvalKit/vlmeval/dataset/ovo_timing.py) - not a paraphrase, the
// wording is part of the training recipe.
static const char OVO_STREAMING_SYSTEM[] =
    "You are a conservative assistant for real-time streaming video event detection.\n"
    "You receive the video incrementally, with every update labeled by an absolute time interval such as <0s-1s>. Base each decision only on evidence observed so far.\n"
    "\n"
    "Every </Response> is a final detection of one distinct event boundary. An unmatched, premature, or duplicate response is an error, so prioritize precision over frequent responses.\n"
    "\n"
    "At every update, use exactly one of these three protocols:\n"
    "\n"
    "1. Use </Silence> when:\n"
    "   - No relevant event is currently supported by clear visual evidence.\n"
    "   - The requested event has not started, or a suspected event is no longer supported.\n"
    "   - The same event was already reported and no unmistakably new occurrence or step has completed.\n"
    "   - The scene is unchanged, shows only aftermath, or merely repeats evidence already used for a response.\n"
    "   Output exactly </Silence> and nothing else.\n"
    "\n"
    "2. Use </Standby> only when:\n"
    "   - A specific relevant event is visibly developing, but its completion or identity still needs confirmation.\n"
    "   - There is concrete new evidence, not just generic activity or uncertainty.\n"
    "   </Standby> is reversible. Return to </Silence> if later frames do not confirm a new event.\n"
    "   Output exactly </Standby> and nothing else.\n"
    "\n"
    "3. Use </Response> only when:\n"
    "   - The visual evidence clearly establishes that a new, distinct requested event or step has just completed.\n"
    "   - All essential entities, actions, attributes, and relationships required by the question are confirmed.\n"
    "   - This event has not already been reported.\n"
    "   Output </Response> followed by a concise answer.\n"
    "\n"
    "Task-specific rules:\n"
    "- For a single-event question, respond at most once in the entire stream. After responding, always use </Silence>.\n"
    "- For \"How many times\" questions, respond exactly once when each genuinely new occurrence completes. Do not repeat the same count while no new occurrence has happened.\n"
    "- For step-description questions, respond exactly once when each genuinely new requested step completes. Do not repeatedly report an ongoing or already completed step.\n"
    "\n"
    "Do not predict future actions, treat a precursor as completion, or use </Response> merely because the current frames are relevant. When uncertain, prefer </Standby> or </Silence>.\n";

const char * Qwen3VL::realtime_system_prompt() const { return OVO_STREAMING_SYSTEM; }

void Qwen3VL::session_reset() {
    s_active = false;
    s_mh = s_mw = s_per_frame = 0;
    s_sys.clear(); s_q.clear();
    s_feats.clear(); s_marks.clear(); s_turns.clear(); s_kv_ids.clear();
}

bool Qwen3VL::ask_session(const vqa::MediaInput & media, const std::string & prompt,
                          const vqa::GenParams & gp, std::string & out_text) {
    if (const char * e = getenv("QWEN3VL_SESSION_FRAMES"))
        s_max_frames = std::max(1, atoi(e));
    if (!s_active) {
        s_active = true;
        s_sys = gp.history;          // the front end passes format_system(...) here
        s_q = prompt;
        s_feats.clear(); s_marks.clear(); s_turns.clear();
        s_mh = s_mw = s_per_frame = 0;
        fprintf(stderr, "session: start (window %d frames)\n", s_max_frames);
    } else if (!prompt.empty() && s_q.empty()) {
        s_q = prompt;                // a later turn may carry the question
    }

    // ---- encode only the newly arrived frames -----------------------------
    if (!media.empty()) {
        std::vector<float> nf;
        int nn = 0, grid[3] = {0, 0, 0};
        const int budget = n_ctx - gp.max_tokens - 128;
        if (budget <= 0) return false;
        if (!encode_vision(media, nf, &nn, grid, budget)) return false;
        const int per = grid[1] * grid[2];
        if (s_per_frame == 0) { s_per_frame = per; s_mh = grid[1]; s_mw = grid[2]; }
        else if (per != s_per_frame || grid[1] != s_mh || grid[2] != s_mw) {
            fprintf(stderr, "session: frame geometry changed (%dx%d -> %dx%d), resetting window\n",
                    s_mh, s_mw, grid[1], grid[2]);
            s_feats.clear(); s_marks.clear(); s_turns.clear();
            s_per_frame = per; s_mh = grid[1]; s_mw = grid[2];
        }
        const double def_fps = media.fps > 0 ? media.fps : 1.0;
        const size_t cols = (size_t)n_embd * per;
        for (int f = 0; f < grid[0]; f++) {
            s_feats.insert(s_feats.end(), nf.begin() + (size_t)f * cols,
                           nf.begin() + (size_t)(f + 1) * cols);
            const double t0 = f < (int)media.timestamps.size() ? media.timestamps[f] : f / def_fps;
            s_marks.push_back({ t0, t0 + 1.0 / def_fps });
        }
    }
    if (s_marks.empty()) {
        fprintf(stderr, "session: no frames yet\n");
        return false;
    }

    // sliding window: drop the oldest TURNS (frames + their answer), like the
    // reference; an over-long single turn is trimmed frame by frame
    while ((int)s_marks.size() > s_max_frames && s_turns.size() > 1) {
        const int drop = s_turns.front().frames;
        s_feats.erase(s_feats.begin(), s_feats.begin() + (size_t)n_embd * s_per_frame * drop);
        s_marks.erase(s_marks.begin(), s_marks.begin() + drop);
        s_turns.erase(s_turns.begin());
    }
    while ((int)s_marks.size() > s_max_frames) {
        s_feats.erase(s_feats.begin(), s_feats.begin() + (size_t)n_embd * s_per_frame);
        s_marks.erase(s_marks.begin());
        if (!s_turns.empty()) s_turns.front().frames = std::max(0, s_turns.front().frames - 1);
    }

    // ---- replay: system, question, (frames + answer) per turn, this turn ----
    std::string text = s_sys;
    if (!s_q.empty()) text += "<|im_start|>user\n" + s_q + "<|im_end|>\n";
    std::string frame_block;
    frame_block.reserve(32 + (size_t)12 * s_per_frame);
    char ts[64];
    size_t fi = 0;
    auto append_frames = [&](int n, std::string & dst) {
        for (int k = 0; k < n && fi < s_marks.size(); k++, fi++) {
            snprintf(ts, sizeof(ts), "<%.3fs-%.3fs>\n", s_marks[fi].first, s_marks[fi].second);
            dst += ts;
            dst += "<|vision_start|>";
            // per-frame IMAGE blocks: the reference streaming session feeds each
            // frame as its own image, which is what the SFT was trained on
            for (int t = 0; t < s_per_frame; t++) dst += "<|image_pad|>";
            dst += "<|vision_end|>";
        }
    };
    for (const STurn & T : s_turns) {
        text += "<|im_start|>user\n";
        append_frames(T.frames, text);
        text += "<|im_end|>\n<|im_start|>assistant\n" + T.ans + "<|im_end|>\n";
    }
    const int cur_frames = (int)s_marks.size() - (int)fi;
    text += "<|im_start|>user\n";
    append_frames(cur_frames, text);
    text += "<|im_end|>\n<|im_start|>assistant\n";

    const std::vector<int> ids = tok_.encode(text);
    std::vector<int> img_pos;
    for (int i = 0; i < (int)ids.size(); i++) if (ids[i] == id_image_pad) img_pos.push_back(i);
    const int n_img = (int)(s_feats.size() / n_embd);
    if ((int)img_pos.size() != n_img) {
        fprintf(stderr, "session: placeholder mismatch (%d pads vs %d vision tokens)\n",
                (int)img_pos.size(), n_img);
        return false;
    }
    std::vector<int> mpos;
    build_mpos(ids, img_pos, n_img, s_mh, s_mw, mpos);

    std::string ans;
    // ---- KV splice: when this turn's prompt is exactly the prefix the cache
    // already holds plus a new tail, prefill only the tail (that is what makes a
    // live session cheaper than re-encoding the window). The frame layout must
    // line up: this turn's vision tokens are the tail of s_feats.
    const size_t kv_len = s_kv_ids.size();
    const bool can_splice = kv_len > 0 && ids.size() > kv_len && (int)kv_len == n_past &&
                            std::equal(s_kv_ids.begin(), s_kv_ids.end(), ids.begin());
    if (can_splice) {
        const int skip = (int)kv_len;
        const int T = (int)ids.size() - skip;
        std::vector<int> ids_t(ids.begin() + skip, ids.end());
        std::vector<int> pos_t;
        pos_t.reserve((size_t)4 * T);
        for (int plane = 0; plane < 4; plane++)
            pos_t.insert(pos_t.end(), mpos.begin() + (size_t)plane * ids.size() + skip,
                         mpos.begin() + (size_t)plane * ids.size() + ids.size());
        std::vector<int> ip_t;
        for (int p : img_pos) if (p >= skip) ip_t.push_back(p - skip);
        const size_t add_cols = (size_t)cur_frames * s_per_frame;
        if (ip_t.size() == add_cols) {
            std::vector<float> f_t;
            if (add_cols > 0)
                f_t.assign(s_feats.end() - (std::ptrdiff_t)(add_cols * (size_t)n_embd), s_feats.end());
            if (!generate(ids_t, f_t.empty() ? nullptr : &f_t, ip_t, pos_t, gp, ans, true))
                return false;
            out_text = ans;
            s_turns.push_back({ cur_frames, ans });
            return true;
        }
        fprintf(stderr, "session: splice rejected (frame layout), rebuilding the window\n");
    }
    if (!generate(ids, n_img > 0 ? &s_feats : nullptr, img_pos, mpos, gp, ans)) return false;
    out_text = ans;
    s_turns.push_back({ cur_frames, ans });
    return true;
}

}  // namespace qwen3vl

namespace vqa {
Model * create_qwen3vl() { return new qwen3vl::Qwen3VL(); }
}  // namespace vqa
