// minicpmo_model.cpp - MiniCPM-o 4.5 backend (openbmb/MiniCPM-o-4_5, arch key
// "minicpmo"): Qwen3-8B dense trunk + SigLIP-SO400M NaFiT vision tower with
// the MiniCPM-V 9-slice tiler + 64-query resampler + whisper-medium audio
// tower + the tk.* TTS sub-tower (codebook ids out; the CosyVoice2 token2wav
// vocoder is a later wave). Numerics contract at the top of minicpmo.hpp.
//
// Injection (official processor semantics, ref/processing_minicpmo.py):
//   image: <image_id>N</image_id><image><unk>x64</image> + one
//          <slice><unk>x64</slice> per grid row, rows joined "\n"
//   audio: <|audio_start|><unk>xN<|audio_end|>  (N = pooled frames)
//   message parts joined "\n"; ChatML user turn + assistant generation
//   prompt with the Qwen3 empty think block; the TTS template appends
//   <|tts_bos|> and the LLM generates text ... <|tts_eos|>.
//
// Gates (MINICPMO_* env, mirroring the nemotronomni surface):
//   MINICPMO_DUMP_DIR       intermediate dumps (ggml_dump registry)
//   MINICPMO_LOGITS_ALL=1   prefill fetches the whole [vocab, T] logits
//   MINICPMO_PREFILL_DUMP   file for the prefill logits block
//   MINICPMO_STEP_DUMP      prefix for per-step gen logits
//   MINICPMO_TOWER_ONLY=1   encode media + dump, no generation
//   MINICPMO_TTS_GREEDY=1   TTS argmax (logits-level gate; official
//                           sampling mode otherwise)
//   MINICPMO_TTS_SEED=N     torch CPU generator seed for the TTS loop (0)
#include "core/vqa.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "core/text/tokenizer.hpp"
#include "core/engine/media_inject.hpp"
#include "core/engine/attention.hpp"
#include "core/engine/decode_graph.hpp"
#include "core/engine/backend.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/gguf_util.hpp"
#include "core/engine/sampler.hpp"
#include "core/runtime/load_util.hpp"
#include "core/engine/weight_load.hpp"
#include "core/util/aligned_buf.hpp"
#include "core/util/mmap.hpp"
#include "minicpmo.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace mcpm {

struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *pos_in = nullptr, *mask_in = nullptr;
    ggml_tensor *mm_in = nullptr, *mm_pos_in = nullptr;   // media rows (prefill)
    ggml_tensor *logits = nullptr, *hidden = nullptr;
};

class MiniCPMO : public vqa::Model {
public:
    MiniCPMO() {
        caps_.image = true;
        caps_.audio = true;
        caps_.video = false;             // one image per ask (official path)
    }
    ~MiniCPMO() override;
    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "minicpm-o-4.5"; }
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp,
             std::string & out_text) override;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool load_towers();
    bool alloc_talker();
    bool build_text_graph(TextGraph & g, int T, int S, bool logits_all, int n_mm,
                          bool want_hidden);
    bool run_text(TextGraph & g, const std::vector<int> & toks,
                  const std::vector<int> & pos, std::vector<float> & logits_out,
                  std::vector<float> * hidden_out,
                  const std::vector<float> * mm_feats = nullptr,
                  const std::vector<int> * mm_pos = nullptr);
    bool generate(const std::vector<int> & ids, const vqa::GenParams & gp,
                  std::vector<int> & out_ids, std::string & out_text,
                  std::vector<int> * tts_text, std::vector<float> * tts_hidden,
                  const std::vector<float> * mm_feats = nullptr,
                  const std::vector<int> * mm_pos = nullptr);
    void dump(ggml_cgraph * gf, const char * name);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    // real-checkpoint dims (arch-spec; GGUF metadata overrides)
    int n_layer = 36, n_embd = 4096, n_head = 32, n_kv_head = 8, head_dim = 128;
    int n_vocab = 151748, n_ff = 12288;
    float rms_eps = 1e-6f, rope_theta = 1e6f;
    int n_ctx = 8192;
    int eos_id = 151645, eos2_id = 151643;
    // media / tts special ids (tokenizer-resolved where present)
    int id_unk = 128244, id_im_s = 151669, id_im_e = 151670;
    int id_sl_s = 151679, id_sl_e = 151680;
    int id_au_s = 151697, id_au_e = 151699;
    int id_tts_bos = 151703, id_tts_eos = 151704;

    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir, model_dir_hint;

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    vqa::AlignedBuf host_f32_storage;
    vqa::MMap mm;

    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr, *out_w = nullptr;
    struct Layer { minicpmo::LlmAttnW a; ggml_tensor *norm, *ffn_norm; minicpmo::LlmFfnW f; };
    std::vector<Layer> tl;
    minicpmo::VisionTower vt;
    minicpmo::AudioTower at;
    minicpmo::Talker tk;
    minicpmo::Vocoder voc;    // token2wav sibling (minicpmo-vocoder.gguf)
    vqa::KvCache kv;          // LLM trunk
    vqa::KvCache tkv;         // TTS sub-tower

    ggml_gallocr_t allocr = nullptr;
    vqa::GraphSlot<TextGraph> dec_slot;
    int n_past = 0;
    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

MiniCPMO::~MiniCPMO() {
    dec_slot.reset();
    kv.free();
    tkv.free();
    if (vt.rs_proj_buf) ggml_backend_buffer_free(vt.rs_proj_buf);
    if (vt.rs_proj_ctx) ggml_free(vt.rs_proj_ctx);
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (allocr) ggml_gallocr_free(allocr);
    if (backend) ggml_backend_free(backend);
}

bool MiniCPMO::init_backend() {
    if (getenv("MINICPMO_GPU")) use_gpu = true;
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr;
}

bool MiniCPMO::load(const std::string & gguf_path, const std::string & tokenizer_path,
                    const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "MINICPMO", n_threads, use_gpu, dump_dir);
    vt.dump_dir = dump_dir.c_str();
    at.dump_dir = dump_dir.c_str();
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tokenizer_path.empty() && !tok_.load(tokenizer_path.c_str(),
                                               vqa::TokenizerMode::Qwen2Regex))
        return false;
    if (tok_.size() > 0) {
        const int64_t diff = tok_.size() > n_vocab ? tok_.size() - (int64_t)n_vocab
                                                   : (int64_t)n_vocab - tok_.size();
        if (diff * 5 > n_vocab) {
            fprintf(stderr, "minicpmo: tokenizer vocab %d vs model vocab %d"
                            " - wrong tokenizer.json\n", tok_.size(), n_vocab);
            return false;
        }
    }
    vqa::assign_token_ids(tok_, {
        { "<unk>", &id_unk }, { "<image>", &id_im_s }, { "</image>", &id_im_e },
        { "<slice>", &id_sl_s }, { "</slice>", &id_sl_e },
        { "<|audio_start|>", &id_au_s }, { "<|audio_end|>", &id_au_e },
        { "<|tts_bos|>", &id_tts_bos }, { "<|tts_eos|>", &id_tts_eos },
        { "<|im_end|>", &eos_id }, { "<|endoftext|>", &eos2_id },
    });
    if (!kv.alloc(backend, n_layer, n_kv_head, head_dim, n_ctx)) return false;
    if (!load_towers()) return false;
    if (!alloc_talker()) return false;
    // optional token2wav sibling: minicpmo-vocoder.gguf next to the main GGUF
    // (env MINICPMO_VOCODER overrides the path; --speak then emits a wav)
    {
        std::string vp = getenv("MINICPMO_VOCODER") ? getenv("MINICPMO_VOCODER") : "";
        if (vp.empty()) {
            const size_t slash = gguf_path.find_last_of("/\\");
            std::string dir = slash == std::string::npos ? std::string(".")
                                                         : gguf_path.substr(0, slash);
            for (const std::string & cand :
                 { dir + "/minicpmo-vocoder.gguf" }) {
                if (FILE * f = fopen(cand.c_str(), "rb")) { fclose(f); vp = cand; break; }
            }
        }
        if (!vp.empty() && !getenv("MINICPMO_NO_VOCODER")) {
            if (!voc.load(backend, vp.c_str()))
                fprintf(stderr, "minicpmo: vocoder load failed - --speak falls"
                                " back to codes.txt only\n");
        }
    }
    fprintf(stderr, "minicpmo: %d layers d=%d (gqa %d:%d hd%d theta %.0e),"
                    " vision 27x1152 + rs(%d q), audio whisper %dx%d mel%d,"
                    " tts 20x768%s eos=%d\n",
            n_layer, n_embd, n_head, n_kv_head, head_dim, (double)rope_theta,
            vt.query_num, at.n_layer, at.d_model, at.n_mel,
            tk.tl.empty() ? "" : " [tts]", eos_id);
    return true;
}

bool MiniCPMO::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f GB\n", mm.size / 1e9);
    {
        const size_t slash = std::string(path).find_last_of("/\\");
        model_dir_hint = (slash == std::string::npos) ? "." : std::string(path).substr(0, slash);
    }
    struct ggml_init_params wip = { 512ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    wctx = ggml_init(wip);
    if (!wctx) return false;
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer = vqa::gguf::u32(gu, "minicpmo.block_count", n_layer);
    n_embd  = vqa::gguf::u32(gu, "minicpmo.embedding_length", n_embd);
    n_head  = vqa::gguf::u32(gu, "minicpmo.attention.head_count", n_head);
    n_kv_head = vqa::gguf::u32(gu, "minicpmo.attention.head_count_kv", n_kv_head);
    head_dim = vqa::gguf::u32(gu, "minicpmo.attention.key_length", head_dim);
    n_vocab = vqa::gguf::u32(gu, "minicpmo.vocab_size", n_vocab);
    n_ff    = vqa::gguf::u32(gu, "minicpmo.feed_forward_length", n_ff);
    rms_eps = vqa::gguf::f32(gu, "minicpmo.attention.layer_norm_rms_epsilon", rms_eps);
    rope_theta = vqa::gguf::f32(gu, "minicpmo.rope.freq_base", rope_theta);
    eos_id  = vqa::gguf::u32(gu, "minicpmo.eos_token_id", eos_id);

    vt.n_layer = vqa::gguf::u32(gu, "minicpmo.vision.block_count", vt.n_layer);
    vt.n_embd  = vqa::gguf::u32(gu, "minicpmo.vision.embedding_length", vt.n_embd);
    vt.n_head  = vqa::gguf::u32(gu, "minicpmo.vision.attention.head_count", vt.n_head);
    vt.n_ff    = vqa::gguf::u32(gu, "minicpmo.vision.feed_forward_length", vt.n_ff);
    vt.patch   = vqa::gguf::u32(gu, "minicpmo.vision.patch_size", vt.patch);
    vt.grid_side = vqa::gguf::u32(gu, "minicpmo.vision.image_size", 980) / vt.patch;
    vt.ln_eps  = vqa::gguf::f32(gu, "minicpmo.vision.layer_norm_epsilon", vt.ln_eps);
    vt.query_num = vqa::gguf::u32(gu, "minicpmo.vision.query_num", vt.query_num);
    vt.rs_embed  = vqa::gguf::u32(gu, "minicpmo.vision.resampler_embed", vt.rs_embed);

    at.n_layer = vqa::gguf::u32(gu, "minicpmo.audio.block_count", at.n_layer);
    at.d_model = vqa::gguf::u32(gu, "minicpmo.audio.embedding_length", at.d_model);
    at.n_head  = vqa::gguf::u32(gu, "minicpmo.audio.attention.head_count", at.n_head);
    at.n_mel   = vqa::gguf::u32(gu, "minicpmo.audio.num_mel_bins", at.n_mel);
    at.n_pos   = vqa::gguf::u32(gu, "minicpmo.audio.max_source_positions", at.n_pos);
    at.pool_step = vqa::gguf::u32(gu, "minicpmo.audio.pool_step", at.pool_step);
    at.chunk_s = vqa::gguf::f32(gu, "minicpmo.audio.chunk_length", at.chunk_s);

    tk.n_layer = vqa::gguf::u32(gu, "minicpmo.tts.block_count", tk.n_layer);
    tk.n_embd  = vqa::gguf::u32(gu, "minicpmo.tts.embedding_length", tk.n_embd);
    tk.n_head  = vqa::gguf::u32(gu, "minicpmo.tts.attention.head_count", tk.n_head);
    tk.n_kv_head = vqa::gguf::u32(gu, "minicpmo.tts.attention.head_count_kv", tk.n_kv_head);
    tk.head_dim = tk.n_embd / tk.n_head;
    tk.n_text_vocab = vqa::gguf::u32(gu, "minicpmo.tts.num_text_tokens", tk.n_text_vocab);
    tk.n_audio_vocab = vqa::gguf::u32(gu, "minicpmo.tts.num_audio_tokens", tk.n_audio_vocab);
    tk.audio_bos = vqa::gguf::u32(gu, "minicpmo.tts.audio_bos_token_id", tk.audio_bos);
    tk.text_eos  = vqa::gguf::u32(gu, "minicpmo.tts.text_eos_token_id", tk.text_eos);

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);

    // tensors used inside elementwise/graph-structural ops must be plain F32:
    // all norms/biases are 1D (auto-promoted); the audio conv kernels ride the
    // explicit path (ggml conv_1d is happiest with F32 rows, qwen3omni
    // precedent); vision pos_embd is already F32 in the GGUF.
    std::vector<std::string> extra;
    std::vector<const char *> extra_names;
    extra.push_back("a.conv1.weight");
    extra.push_back("a.conv2.weight");
    for (const std::string & s : extra) extra_names.push_back(s.c_str());
    extra_names.push_back(nullptr);
    vqa::weights::promote_1d_to_f32(wctx, host_f32_storage, extra_names.data());

    vqa::weights::Bound bound = vqa::weights::bind_to_backend(
        wctx, backend, mm, { { host_f32_storage.data(), host_f32_storage.size() } });
    if (!bound.ok) return false;
    wbuf = bound.main_buf;
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (!ggml_backend_is_cpu(backend)) mm.release();

    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) fprintf(stderr, "minicpmo: missing tensor %s\n", n);
        return t;
    };
    tok_embd = gt("token_embd.weight");
    out_norm = gt("output_norm.weight");
    out_w    = gt("lm_head.weight");
    if (!tok_embd || !out_norm || !out_w) return false;

    tl.resize((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "blk.%d.%s", i, suf); return gt(b); };
        Layer & L = tl[(size_t)i];
        L.norm = g("attn_norm.weight");
        L.a.q = g("attn_q.weight");
        L.a.k = g("attn_k.weight");
        L.a.v = g("attn_v.weight");
        L.a.o = g("attn_output.weight");
        L.a.qn = g("attn_q_norm.weight");
        L.a.kn = g("attn_k_norm.weight");
        L.ffn_norm = g("ffn_norm.weight");
        L.f.gate = g("ffn_gate.weight");
        L.f.up = g("ffn_up.weight");
        L.f.down = g("ffn_down.weight");
        if (!L.norm || !L.a.q || !L.a.k || !L.a.v || !L.a.o || !L.a.qn ||
            !L.a.kn || !L.ffn_norm || !L.f.gate || !L.f.up || !L.f.down)
            return false;
    }
    return true;
}

// (bind_from_mmap runs while the gguf handle is still open - see load_weights)

bool MiniCPMO::load_towers() {
    auto gt = [&](const char * n) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) fprintf(stderr, "minicpmo tower: missing tensor %s\n", n);
        return t;
    };
    vt.backend = backend;
    at.backend = backend;
    tk.backend = backend;
    vt.allocr = allocr;
    at.allocr = allocr;
    tk.allocr = allocr;

    // ---- vision (SigLIP NaFiT + resampler) --------------------------------
    vt.patch_w = gt("v.patch_embd.weight");
    vt.patch_b = gt("v.patch_embd.bias");
    vt.pos_embd = gt("v.pos_embd.weight");
    // grid_side: the "vision.image_size" metadata is the 448 SLICE resolution;
    // the NaFiT table is the 980px TRAINING grid - derive it from the table.
    vt.grid_side = (int)std::lround(std::sqrt((double)vt.pos_embd->ne[1]));
    if (vt.grid_side * vt.grid_side != (int)vt.pos_embd->ne[1]) {
        fprintf(stderr, "minicpmo: pos_embd rows %lld are not a square grid\n",
                (long long)vt.pos_embd->ne[1]);
        return false;
    }
    vt.post_ln_w = gt("v.post_ln.weight");
    vt.post_ln_b = gt("v.post_ln.bias");
    if (!vt.patch_w || !vt.patch_b || !vt.pos_embd || !vt.post_ln_w || !vt.post_ln_b)
        return false;
    vt.rs.query = gt("rs.query");
    vt.rs.kv_proj = gt("rs.kv_proj.weight");
    vt.rs.in_proj = gt("rs.attn.in_proj_weight");
    vt.rs.in_proj_b = gt("rs.attn.in_proj_bias");
    vt.rs.out_proj = gt("rs.attn.out_proj.weight");
    vt.rs.out_proj_b = gt("rs.attn.out_proj.bias");
    vt.rs.ln_q_w = gt("rs.ln_q.weight");
    vt.rs.ln_q_b = gt("rs.ln_q.bias");
    vt.rs.ln_kv_w = gt("rs.ln_kv.weight");
    vt.rs.ln_kv_b = gt("rs.ln_kv.bias");
    vt.rs.ln_post_w = gt("rs.ln_post.weight");
    vt.rs.ln_post_b = gt("rs.ln_post.bias");
    vt.rs.proj = gt("rs.proj.weight");
    if (!vt.rs.query || !vt.rs.kv_proj || !vt.rs.in_proj || !vt.rs.in_proj_b ||
        !vt.rs.out_proj || !vt.rs.out_proj_b || !vt.rs.ln_q_w || !vt.rs.ln_q_b ||
        !vt.rs.ln_kv_w || !vt.rs.ln_kv_b || !vt.rs.ln_post_w || !vt.rs.ln_post_b ||
        !vt.rs.proj)
        return false;
    vt.rs_heads = (int)(vt.rs_embed / 128);
    vt.vl.resize((size_t)vt.n_layer);
    for (int i = 0; i < vt.n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "v.blk.%d.%s", i, suf); return gt(b); };
        minicpmo::SiglipLayer & L = vt.vl[(size_t)i];
        L.ln1_w = g("ln1.weight"); L.ln1_b = g("ln1.bias");
        L.wq = g("wq.weight");     L.bq = g("bq");
        L.wk = g("wk.weight");     L.bk = g("bk");
        L.wv = g("wv.weight");     L.bv = g("bv");
        L.wo = g("wo.weight");     L.bo = g("bo");
        L.ln2_w = g("ln2.weight"); L.ln2_b = g("ln2.bias");
        L.fc1_w = g("fc1.weight"); L.fc1_b = g("fc1.bias");
        L.fc2_w = g("fc2.weight"); L.fc2_b = g("fc2.bias");
        if (!L.ln1_w || !L.ln1_b || !L.wq || !L.bq || !L.wk || !L.bk ||
            !L.wv || !L.bv || !L.wo || !L.bo || !L.ln2_w || !L.ln2_b ||
            !L.fc1_w || !L.fc1_b || !L.fc2_w || !L.fc2_b)
            return false;
    }

    // ---- audio (whisper-medium) -------------------------------------------
    at.conv1_w = gt("a.conv1.weight");
    at.conv1_b = gt("a.conv1.bias");
    at.conv2_w = gt("a.conv2.weight");
    at.conv2_b = gt("a.conv2.bias");
    at.pos_embd = gt("a.pos_embd.weight");
    at.ln_post_w = gt("a.ln_post.weight");
    at.ln_post_b = gt("a.ln_post.bias");
    at.proj_fc1 = gt("a.proj.fc1.weight");
    at.proj_fc1_b = gt("a.proj.fc1.bias");
    at.proj_fc2 = gt("a.proj.fc2.weight");
    at.proj_fc2_b = gt("a.proj.fc2.bias");
    if (!at.conv1_w || !at.conv1_b || !at.conv2_w || !at.conv2_b ||
        !at.pos_embd || !at.ln_post_w || !at.ln_post_b || !at.proj_fc1 ||
        !at.proj_fc1_b || !at.proj_fc2 || !at.proj_fc2_b)
        return false;
    at.al.resize((size_t)at.n_layer);
    for (int i = 0; i < at.n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "a.blk.%d.%s", i, suf); return gt(b); };
        minicpmo::AudLayer & L = at.al[(size_t)i];
        L.attn_ln_w = g("attn_ln.weight"); L.attn_ln_b = g("attn_ln.bias");
        L.wq = g("attn_q.weight"); L.bq = g("attn_q.bias");
        L.wk = g("attn_k.weight");                 // no bias (whisper)
        L.wv = g("attn_v.weight"); L.bv = g("attn_v.bias");
        L.wo = g("attn_o.weight"); L.bo = g("attn_o.bias");
        L.ffn_ln_w = g("ffn_ln.weight"); L.ffn_ln_b = g("ffn_ln.bias");
        L.fc1_w = g("ffn_fc1.weight"); L.fc1_b = g("ffn_fc1.bias");
        L.fc2_w = g("ffn_fc2.weight"); L.fc2_b = g("ffn_fc2.bias");
        if (!L.attn_ln_w || !L.attn_ln_b || !L.wq || !L.bq || !L.wk ||
            !L.wv || !L.bv || !L.wo || !L.bo || !L.ffn_ln_w || !L.ffn_ln_b ||
            !L.fc1_w || !L.fc1_b || !L.fc2_w || !L.fc2_b)
            return false;
    }
    // whisper 80-mel filterbank: sidecar next to the GGUF (transformers
    // WhisperFeatureExtractor mel_filters dumped [80, 201] row-major),
    // runtime HTK as the fallback
    {
        const std::string p = model_dir_hint + "/mel_filters_80_201.f32";
        FILE * fp = fopen(p.c_str(), "rb");
        if (fp) {
            at.mel_filters_host.resize((size_t)at.n_mel * 201);
            if (fread(at.mel_filters_host.data(), 4, at.mel_filters_host.size(), fp)
                != at.mel_filters_host.size())
                at.mel_filters_host.clear();
            fclose(fp);
            if (!at.mel_filters_host.empty())
                fprintf(stderr, "minicpmo audio: mel filters sidecar loaded\n");
        }
        if (!at.load_mel_filters()) return false;
    }

    // ---- talker (tk.*) -----------------------------------------------------
    tk.emb_text = gt("tk.emb_text.weight");
    tk.emb_code = gt("tk.emb_code.weight");
    tk.head_code = gt("tk.head_code.weight");
    tk.output_norm = gt("tk.output_norm.weight");
    tk.proj_sem_fc1 = gt("tk.proj_semantic.fc1.weight");
    tk.proj_sem_fc1_b = gt("tk.proj_semantic.fc1.bias");
    tk.proj_sem_fc2 = gt("tk.proj_semantic.fc2.weight");
    tk.proj_sem_fc2_b = gt("tk.proj_semantic.fc2.bias");
    if (!tk.emb_text || !tk.emb_code || !tk.head_code || !tk.output_norm ||
        !tk.proj_sem_fc1 || !tk.proj_sem_fc1_b || !tk.proj_sem_fc2 ||
        !tk.proj_sem_fc2_b)
        return false;
    tk.tl.resize((size_t)tk.n_layer);
    for (int i = 0; i < tk.n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "tk.blk.%d.%s", i, suf); return gt(b); };
        minicpmo::TkLayer & L = tk.tl[(size_t)i];
        L.attn_norm = g("attn_norm.weight");
        L.q = g("attn_q.weight"); L.k = g("attn_k.weight");
        L.v = g("attn_v.weight"); L.o = g("attn_output.weight");
        L.ffn_norm = g("ffn_norm.weight");
        L.gate = g("ffn_gate.weight"); L.up = g("ffn_up.weight");
        L.down = g("ffn_down.weight");
        if (!L.attn_norm || !L.q || !L.k || !L.v || !L.o || !L.ffn_norm ||
            !L.gate || !L.up || !L.down)
            return false;
    }
    return true;
}

bool MiniCPMO::alloc_talker() {
    if (tk.tl.empty()) return true;        // no tts tensors: speak disabled
    if (const char * e = getenv("MINICPMO_TTS_THETA"))
        tk.rope_theta = (float)atof(e);
    if (const char * e = getenv("MINICPMO_TTS_EPS"))
        tk.rms_eps = (float)atof(e);
    if (!tkv.alloc(backend, tk.n_layer, tk.n_kv_head, tk.head_dim, 4096))
        return false;
    tk.kv = &tkv;
    return true;
}

// ------------------------------------------------------------------ graph

bool MiniCPMO::build_text_graph(TextGraph & g, int T, int S, bool logits_all,
                                int n_mm, bool want_hidden) {
    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
    struct ggml_init_params ip = { 128ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 32768, false);
    ggml_context * c = g.ctx;
    const int D = n_embd;

    g.tok_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    ggml_set_input(g.tok_in);
    g.pos_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_set_name(g.pos_in, "pos_in");
    ggml_set_input(g.pos_in);
    g.mask_in = ggml_new_tensor_2d(c, GGML_TYPE_F16, S, T);
    ggml_set_name(g.mask_in, "mask_in");
    ggml_set_input(g.mask_in);

    ggml_tensor * x = ggml_get_rows(c, tok_embd, g.tok_in);
    ggml_set_name(x, "emb");
    x = vqa::engine::inject_media_rows(c, x, &g.mm_in, &g.mm_pos_in, D, n_mm);

    for (int il = 0; il < n_layer; il++) {
        const Layer & L = tl[(size_t)il];
        ggml_tensor * l = ggml_rms_norm(c, x, rms_eps);
        l = ggml_mul(c, l, L.norm);
        const char * dbg = (il == 0 && !dump_dir.empty()) ? "t0" : nullptr;
        ggml_tensor * a = minicpmo::llm_attn_branch(
            c, g.gf, L.a, kv.k[(size_t)il], kv.v[(size_t)il], l, g.mask_in,
            g.pos_in, n_head, n_kv_head, head_dim, rope_theta, n_ctx, rms_eps,
            dbg);
        x = ggml_add(c, x, a);
        l = ggml_rms_norm(c, x, rms_eps);
        l = ggml_mul(c, l, L.ffn_norm);
        ggml_tensor * f = minicpmo::llm_ffn_branch(c, L.f, l, rms_eps, dbg);
        x = ggml_add(c, x, f);
        if (il == 0) ggml_set_name(x, "t_l0_out");
    }
    x = ggml_rms_norm(c, x, rms_eps);
    x = ggml_mul(c, x, out_norm);
    ggml_set_name(x, "t_final_norm");

    if (logits_all) {
        g.logits = ggml_mul_mat(c, out_w, x);
    } else {
        ggml_tensor * last = ggml_view_2d(c, x, D, 1, x->nb[1],
                                          (size_t)(T - 1) * x->nb[1]);
        g.logits = ggml_mul_mat(c, out_w, last);
        if (want_hidden) {
            g.hidden = last;               // the row that produced the sample
        }
    }
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);
    if (g.hidden) ggml_build_forward_expand(g.gf, g.hidden);

    if (!dump_dir.empty()) {
        static const char * const dn[] = {
            "emb", "t0_q_norm", "t0_q_rope", "t0_attn", "t0_swiglu",
            "t_l0_out", "t_final_norm", nullptr,
        };
        vqa::dump::mark_sinks(c, g.gf, dn);
        vqa::dump::finalize_sinks(c, g.gf, backend);
    }
    return true;
}

bool MiniCPMO::run_text(TextGraph & g, const std::vector<int> & toks,
                        const std::vector<int> & pos,
                        std::vector<float> & logits_out,
                        std::vector<float> * hidden_out,
                        const std::vector<float> * mm_feats,
                        const std::vector<int> * mm_pos) {
    const int T = (int)toks.size();
    const int S = n_past + T;
    if (S > n_ctx) { fprintf(stderr, "context overflow (%d > %d)\n", S, n_ctx); return false; }
    if ((int)pos.size() != T) { fprintf(stderr, "pos/tok length mismatch\n"); return false; }
    const int n_mm = mm_feats ? (int)(mm_feats->size() / (size_t)n_embd) : 0;
    if (n_mm > 0 && (T == 1 || !mm_pos || (int)mm_pos->size() != n_mm)) {
        fprintf(stderr, "minicpmo: media rows only fit a T>1 prefill\n");
        return false;
    }
    const bool want_hidden = hidden_out != nullptr;

    const bool decode_step = (T == 1);
    if (decode_step) {
        const int window_S = vqa::decode_bucket(S, n_ctx, ggml_backend_is_cpu(backend));
        // key folds the hidden-capture flag: the slot graph must not be reused
        // across generate() calls that differ in the tts hidden output
        const int slot_key = window_S * 2 + (want_hidden ? 1 : 0);
        if (!dec_slot.prepare(slot_key, [&](TextGraph & d) {
                return build_text_graph(d, 1, window_S, false, 0, want_hidden);
            }))
            return false;
        TextGraph * dg = &dec_slot.g;
        if (!ggml_gallocr_alloc_graph(allocr, dg->gf)) {
            fprintf(stderr, "minicpmo: text graph allocation failed\n");
            dec_slot.reset();
            return false;
        }
        ggml_backend_tensor_set(dg->tok_in, toks.data(), 0, sizeof(int) * T);
        ggml_backend_tensor_set(dg->pos_in, pos.data(), 0, sizeof(int) * T);
        vqa::set_causal_mask(dg->mask_in, n_past, T);
        if (ggml_backend_graph_compute(backend, dg->gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "minicpmo: text compute failed\n");
            dec_slot.reset();
            return false;
        }
        ggml_tensor * lg = ggml_graph_get_tensor(dg->gf, "logits");
        logits_out.resize((size_t)n_vocab);
        ggml_backend_tensor_get(lg, logits_out.data(), 0,
                                sizeof(float) * logits_out.size());
        if (hidden_out) {
            // decode graphs run T=1: the final-norm tensor is the row itself
            ggml_tensor * hd = ggml_graph_get_tensor(dg->gf, "t_final_norm");
            hidden_out->resize((size_t)n_embd);
            ggml_backend_tensor_get(hd, hidden_out->data(), 0,
                                    sizeof(float) * n_embd);
        }
        n_past = S;
        return true;
    }

    const bool want_all = getenv("MINICPMO_LOGITS_ALL") != nullptr;
    if (!build_text_graph(g, T, S, want_all, n_mm, want_hidden)) return false;
    if (!ggml_gallocr_alloc_graph(allocr, g.gf)) {
        fprintf(stderr, "minicpmo: text graph allocation failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }
    ggml_backend_tensor_set(g.tok_in, toks.data(), 0, sizeof(int) * T);
    ggml_backend_tensor_set(g.pos_in, pos.data(), 0, sizeof(int) * T);
    vqa::set_causal_mask(g.mask_in, n_past, T);
    if (n_mm > 0) {
        ggml_backend_tensor_set(g.mm_in, mm_feats->data(), 0,
                                sizeof(float) * mm_feats->size());
        ggml_backend_tensor_set(g.mm_pos_in, mm_pos->data(), 0,
                                sizeof(int) * mm_pos->size());
    }
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "minicpmo: text compute failed\n");
        ggml_free(g.ctx);
        g.ctx = nullptr;
        return false;
    }
    ggml_tensor * lg = ggml_graph_get_tensor(g.gf, "logits");
    const int rows = want_all ? T : 1;
    logits_out.resize((size_t)n_vocab * (size_t)rows);
    ggml_backend_tensor_get(lg, logits_out.data(), 0,
                            sizeof(float) * logits_out.size());
    if (hidden_out) {
        ggml_tensor * hd = ggml_graph_get_tensor(g.gf, "t_final_norm");
        hidden_out->resize((size_t)n_embd);
        ggml_backend_tensor_get(hd, hidden_out->data(),
                                sizeof(float) * (size_t)(T - 1) * n_embd,
                                sizeof(float) * n_embd);
    }
    if (want_all && getenv("MINICPMO_PREFILL_DUMP")) {
        FILE * f = fopen(getenv("MINICPMO_PREFILL_DUMP"), "wb");
        if (f) {
            fwrite(logits_out.data(), sizeof(float), logits_out.size(), f);
            fclose(f);
        }
    }
    if (!dump_dir.empty()) {
        dump(g.gf, "emb");
        if (T > 1) {
            dump(g.gf, "t0_q_norm");
            dump(g.gf, "t0_q_rope");
            dump(g.gf, "t0_attn");
            dump(g.gf, "t0_swiglu");
            dump(g.gf, "t_l0_out");
        }
        dump(g.gf, "t_final_norm");
        dump(g.gf, "logits");
    }
    n_past = S;
    ggml_free(g.ctx);
    g.ctx = nullptr;
    return true;
}

void MiniCPMO::dump(ggml_cgraph * gf, const char * name) {
    if (dump_dir.empty()) return;
    vqa::dump::write(gf, dump_dir.c_str(), name);
}

// ------------------------------------------------------------------ generate

bool MiniCPMO::generate(const std::vector<int> & ids, const vqa::GenParams & gp,
                        std::vector<int> & out_ids, std::string & out_text,
                        std::vector<int> * tts_text,
                        std::vector<float> * tts_hidden,
                        const std::vector<float> * mm_feats,
                        const std::vector<int> * mm_pos) {
    out_ids.clear();
    out_text.clear();
    if ((int)ids.size() >= n_ctx) {
        fprintf(stderr, "minicpmo: prompt %zu tok fills n_ctx %d\n", ids.size(), n_ctx);
        return false;
    }
    int max_new = gp.max_tokens;
    if ((int)ids.size() + max_new > n_ctx) {
        max_new = n_ctx - (int)ids.size();
        fprintf(stderr, "minicpmo: clamping max_new to %d\n", max_new);
    }
    kv.clear();
    n_past = 0;

    TextGraph g;
    std::vector<float> logits, hidden;
    std::vector<int> pos(ids.size());
    for (size_t i = 0; i < ids.size(); i++) pos[i] = (int)i;
    const auto pf0 = std::chrono::steady_clock::now();
    if (!run_text(g, ids, pos, logits, tts_text ? &hidden : nullptr,
                  mm_feats, mm_pos))
        return false;
    if (ids.size() > 1)
        fprintf(stderr, "minicpmo: llm prefill %.0f ms (%zu tok)\n",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - pf0).count(),
                ids.size());
    auto last_row = [&]() {
        return std::vector<float>(logits.end() - (size_t)n_vocab, logits.end());
    };
    int cur = vqa::sample_logits(last_row(), gp, rng_state);
    const char * step_dump = getenv("MINICPMO_STEP_DUMP");
    for (int step = 0; step < max_new; step++) {
        if (step_dump) {
            char pth[512];
            snprintf(pth, sizeof(pth), "%s_step%d.f32", step_dump, step);
            FILE * f = fopen(pth, "wb");
            if (f) {
                fwrite(logits.data() + (logits.size() - (size_t)n_vocab),
                       sizeof(float), (size_t)n_vocab, f);
                fclose(f);
            }
        }
        if (cur == eos_id || cur == eos2_id) break;
        out_ids.push_back(cur);
        const bool is_tts_eos = tts_text && cur == id_tts_eos;
        if (!is_tts_eos) {
            if (tts_text) {
                tts_text->push_back(cur);
                tts_hidden->insert(tts_hidden->end(), hidden.begin(), hidden.end());
            }
            if (tok_.size() > 0) {
                const std::string piece = tok_.decode({ cur });
                out_text += piece;
                if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
            }
        }
        if (is_tts_eos || (tts_text && cur == eos_id)) break;
        if (step + 1 >= max_new) break;
        const int p = n_past;
        std::vector<float> step_hidden;
        std::vector<float> * hp = tts_text ? &step_hidden : nullptr;
        if (!run_text(g, { cur }, { p }, logits, hp)) return false;
        if (tts_text) hidden = std::move(step_hidden);
        cur = vqa::sample_logits(logits, gp, rng_state);
    }
    return true;
}

// ------------------------------------------------------------------ ask

bool MiniCPMO::ask(const vqa::MediaInput & media, const std::string & prompt,
                   const std::string & mode, const vqa::GenParams & gp,
                   std::string & out_text) {
    const bool tower_only = getenv("MINICPMO_TOWER_ONLY") != nullptr;
    const bool speak = gp.speak_wav != nullptr && gp.speak_wav[0] != '\0';
    std::vector<float> vfeats, afeats;
    int img_tokens = 0, aud_tokens = 0;

    if (!media.empty()) {
        if (media.frames.size() != 1) {
            fprintf(stderr, "minicpmo: one image per request\n");
            return false;
        }
        std::vector<minicpmo::VisionTower::TilePre> tiles;
        if (!vt.preprocess(media.frames[0].rgb.data(), media.width, media.height, tiles))
            return false;
        if (!vt.encode(tiles, vfeats)) return false;
        img_tokens = vt.n_tok;
    }
    if (media.has_audio()) {
        if (media.sample_rate != 16000) {
            fprintf(stderr, "minicpmo: audio must be 16 kHz (got %d)\n",
                    media.sample_rate);
            return false;
        }
        const auto a0 = std::chrono::steady_clock::now();
        int n_aud = 0;
        if (!at.encode(media.audio.data(), (int)media.audio.size(), afeats, &n_aud))
            return false;
        const auto a1 = std::chrono::steady_clock::now();
        fprintf(stderr, "minicpmo: audio encode %.0f ms -> %d tok\n",
                std::chrono::duration<double, std::milli>(a1 - a0).count(),
                n_aud);
        aud_tokens = n_aud;
    }
    if (tower_only) {
        fprintf(stderr, "minicpmo: TOWER_ONLY stop (image %d tok / audio %d tok)\n",
                img_tokens, aud_tokens);
        out_text.clear();
        if (gp.on_stage) gp.on_stage("done", 0, 0, gp.on_stage_ud);
        return true;
    }

    // ---- placeholder expansion (processor semantics) ----------------------
    std::string content;
    {
        std::vector<std::string> parts;
        if (img_tokens > 0) {
            // <image_id>0</image_id><image><unk>x64</image> + slice rows
            std::string ph = "<image_id>0</image_id><image>";
            for (int i = 0; i < 64; i++) ph += "<unk>";
            ph += "</image>";
            // one <slice> row per slice tile, rows joined by "\n" (the grid
            // geometry only affects the tiling order, not the token stream)
            const int n_sl = (int)vfeats.size() / 4096 / 64 - 1;
            for (int r = 0; r < n_sl; r++) {
                ph += "<slice>";
                for (int i = 0; i < 64; i++) ph += "<unk>";
                ph += "</slice>";
                if (r + 1 < n_sl) ph += "\n";
            }
            parts.push_back(ph);
        }
        if (aud_tokens > 0) {
            std::string ph = "<|audio_start|>";
            for (int i = 0; i < aud_tokens; i++) ph += "<unk>";
            ph += "<|audio_end|>";
            parts.push_back(ph);
        }
        parts.push_back(prompt);
        std::string joined;
        for (size_t i = 0; i < parts.size(); i++) {
            if (i) joined += "\n";
            joined += parts[i];
        }
        content = joined;
    }

    std::string full;
    if (mode == "raw") {
        full = gp.history + content;
    } else {
        full = gp.history + "<|im_start|>user\n" + content +
               "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        if (speak) full += "<|tts_bos|>";
    }
    const std::vector<int> ids = tok_.encode(full);
    if (getenv("MINICPMO_DUMP_IDS")) {
        fprintf(stderr, "minicpmo prompt ids (%zu):", ids.size());
        for (int id : ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }

    // ---- bounds + feature order (image bounds first, then audio) ---------
    std::vector<std::array<int, 2>> bounds;
    std::vector<int> kinds;                // 0 = image, 1 = audio
    for (size_t i = 0; i < ids.size(); i++) {
        if (ids[i] == id_im_s || ids[i] == id_sl_s) {
            size_t j = i + 1;
            while (j < ids.size() && ids[j] != id_im_e && ids[j] != id_sl_e) j++;
            bounds.push_back({ (int)i + 1, (int)j });
            kinds.push_back(0);
            i = j;
        } else if (ids[i] == id_au_s) {
            size_t j = i + 1;
            while (j < ids.size() && ids[j] != id_au_e) j++;
            bounds.push_back({ (int)i + 1, (int)j });
            kinds.push_back(1);
            i = j;
        }
    }
    {
        int got_img = 0, got_aud = 0;
        for (size_t b = 0; b < bounds.size(); b++)
            (kinds[b] == 0 ? got_img += bounds[b][1] - bounds[b][0]
                           : got_aud += bounds[b][1] - bounds[b][0]);
        if (got_img != img_tokens || got_aud != aud_tokens) {
            fprintf(stderr, "minicpmo: placeholder mismatch: %d/%d image,"
                            " %d/%d audio\n", got_img, img_tokens, got_aud,
                            aud_tokens);
            return false;
        }
    }
    std::vector<float> mm_feats;
    std::vector<int> mm_pos;
    mm_feats.reserve(vfeats.size() + afeats.size());
    {
        size_t voff = 0, aoff = 0;
        for (size_t b = 0; b < bounds.size(); b++) {
            const int len = bounds[b][1] - bounds[b][0];
            const std::vector<float> & src = kinds[b] == 0 ? vfeats : afeats;
            size_t & off = kinds[b] == 0 ? voff : aoff;
            for (int k = 0; k < len; k++) {
                for (int d = 0; d < n_embd; d++)
                    mm_feats.push_back(src[off + (size_t)k * n_embd + d]);
                mm_pos.push_back(bounds[b][0] + k);
            }
            off += (size_t)len * n_embd;
        }
    }

    std::vector<int> out_ids;
    std::vector<int> tts_text;
    std::vector<float> tts_hidden;
    if (!generate(ids, gp, out_ids, out_text, speak ? &tts_text : nullptr,
                  speak ? &tts_hidden : nullptr,
                  mm_feats.empty() ? nullptr : &mm_feats,
                  mm_pos.empty() ? nullptr : &mm_pos))
        return false;

    if (speak) {
        if (tk.tl.empty()) {
            fprintf(stderr, "minicpmo: tts tower not loaded\n");
            return false;
        }
        std::vector<int> codes;
        if (const char * cf = getenv("MINICPMO_VOC_CODES")) {
            // parity: teacher-force a codebook id list (one id per line)
            FILE * f = fopen(cf, "r");
            if (!f) { fprintf(stderr, "minicpmo: cannot open %s\n", cf); return false; }
            int v = 0;
            while (fscanf(f, "%d", &v) == 1) codes.push_back(v);
            fclose(f);
            fprintf(stderr, "minicpmo: teacher-forced %zu codes\n", codes.size());
        } else if (!tk.synthesize(tts_text, tts_hidden, codes)) return false;
        // land the codebook ids next to the requested wav path (gate hook)
        std::string path = gp.speak_wav;
        const size_t dot = path.find_last_of('.');
        std::string cpath = (dot == std::string::npos) ? path + ".codes.txt"
                                                       : path.substr(0, dot) + ".codes.txt";
        if (FILE * f = fopen(cpath.c_str(), "w")) {
            for (size_t i = 0; i < codes.size(); i++)
                fprintf(f, "%d%s", codes[i], i + 1 < codes.size() ? " " : "\n");
            fclose(f);
        }
        if (!voc.loaded()) {
            fprintf(stderr, "minicpmo: %zu speech codes -> %s (vocoder gguf not"
                            " loaded)\n", codes.size(), cpath.c_str());
        } else {
            std::vector<float> pcm;
            if (!voc.run(codes, pcm)) {
                fprintf(stderr, "minicpmo: token2wav failed\n");
                return false;
            }
            // 24 kHz 16-bit mono wav (torchaudio.save convention: *32768)
            FILE * f = fopen(path.c_str(), "wb");
            if (!f) { fprintf(stderr, "minicpmo: cannot write %s\n", path.c_str()); return false; }
            const uint32_t nb = (uint32_t)(pcm.size() * 2);
            const uint32_t rl = 36 + nb;
            fwrite("RIFF", 1, 4, f); fwrite(&rl, 4, 1, f); fwrite("WAVE", 1, 4, f);
            fwrite("fmt ", 1, 4, f);
            { const uint32_t fs = 16; fwrite(&fs, 4, 1, f);
              const uint16_t af = 1, ch = 1, bits = 16;
              const uint32_t sr = 24000, br = 48000; const uint16_t bl = 2;
              fwrite(&af, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&sr, 4, 1, f);
              fwrite(&br, 4, 1, f); fwrite(&bl, 2, 1, f); fwrite(&bits, 2, 1, f); }
            fwrite("data", 1, 4, f); fwrite(&nb, 4, 1, f);
            std::vector<int16_t> iq(pcm.size());
            for (size_t i = 0; i < pcm.size(); i++)
                iq[i] = (int16_t)std::lround(std::max(-1.0f,
                        std::min(1.0f, pcm[i])) * 32768.0f);
            fwrite(iq.data(), 2, iq.size(), f);
            fclose(f);
            fprintf(stderr, "minicpmo: %zu speech codes -> %s (%.2f s @ 24 kHz,"
                    " token2wav ggml)\n", codes.size(), path.c_str(),
                    pcm.size() / 24000.0);
        }
    }
    if (gp.on_stage) gp.on_stage("done", (int)out_ids.size(), (int)out_ids.size(),
                                 gp.on_stage_ud);
    return true;
}

}  // namespace mcpm

namespace vqa {
Model * create_minicpmo() { return new mcpm::MiniCPMO(); }
}  // namespace vqa
