// qwen25omni_model.cpp - Qwen2.5-Omni thinker backend.
//
// The thinker is a Qwen2.5 text trunk whose embedding table is fed by two
// encoder towers, so the graph layout follows the proven qwen2vl backend:
//   * the vision tower is Qwen2.5-VL down to the merger (the converter fuses
//     Omni's separate q/k/v into one wqkv, so the same graph runs unchanged);
//   * the audio tower is whisper-style with n_window=100 chunked attention -
//     see qwen25omni_audio.cpp for the chunking bookkeeping;
//   * the text trunk is plain Qwen2 with section MRoPE (16/24/24). Positions
//     follow the official get_rope_index: image tokens take the (t, h, w)
//     grid and advance the running index by max(grid_h, grid_w); audio tokens
//     are plain running positions, exactly like text.
// Talker (speech output) loads from the qwen25omni-talker-*.gguf sibling when
// present; until it is wired the backend answers with text only.
#include "qwen25omni_model.hpp"
#include "qwen25omni_talker.hpp"

#include "attention.hpp"
#include "backend.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "gguf_util.hpp"
#include "load_util.hpp"
#include "qwen2vl_tower.hpp"
#include "resample.hpp"
#include "sampler.hpp"
#include "weight_load.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace qwen25omni {
namespace {
void dump_bins(const char * path, const int * hdr, int n_hdr, const float * data, size_t n) {
    FILE * f = fopen(path, "wb");
    if (!f) return;
    if (n_hdr) fwrite(hdr, sizeof(int), (size_t)n_hdr, f);
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    fprintf(stderr, "dumped %s\n", path);
}
}  // namespace


struct TxtLayer {
    ggml_tensor *norm = nullptr;
    ggml_tensor *wq = nullptr, *bq = nullptr;
    ggml_tensor *wk = nullptr, *bk = nullptr;
    ggml_tensor *wv = nullptr, *bv = nullptr;
    ggml_tensor *wo = nullptr;
    ggml_tensor *ffn_norm = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
};

// one text graph: prefill (T tokens) or one decode step (T=1)
struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *pos_in = nullptr, *slot_in = nullptr, *mask_in = nullptr;
    ggml_tensor *img_in = nullptr, *img_pos_in = nullptr;
    ggml_tensor *aud_in = nullptr, *aud_pos_in = nullptr;
    ggml_tensor *logits = nullptr;
    // talker feed: hidden BEFORE the final norm + this step's token embedding
    ggml_tensor *hidden_out = nullptr, *embed_out = nullptr;
    // talker prefill: input embeddings with multimodal rows zeroed
    ggml_tensor *emb0_out = nullptr;
};

class Qwen25Omni : public vqa::Model {
public:
    ~Qwen25Omni() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "qwen2.5-omni"; }
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp, std::string & out_text) override;

    // tunables (set before load)
    int n_ctx = 8192;
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool alloc_kv();

    bool encode_vision(const vqa::MediaInput & media, std::vector<float> & feats, int * n_tokens,
                       int grid[3]);
    bool build_vision_graph(int gh, int gw);
    bool run_vision(const std::vector<float> & patches, std::vector<float> & feats);

    bool encode_audio(const vqa::MediaInput & media, std::vector<float> & feats, int * n_tokens);

    // shared-tower debug hooks (see qwen2vl_tower.hpp)
    static void tower_after_patch_(void * ud, ggml_context * ctx, ggml_cgraph * gf,
                                   int v_embd, int N, ggml_tensor * x);
    static void tower_before_layers_(void * ud, ggml_context * ctx, ggml_cgraph * gf,
                                     int v_layer, int v_embd, int N);
    static void tower_after_layer0_attn_(void * ud, ggml_context * ctx, ggml_cgraph * gf,
                                         int v_embd, int N, ggml_tensor * x);
    static void tower_after_layer_(void * ud, ggml_context * ctx, ggml_cgraph * gf,
                                   int il, int v_embd, int N, ggml_tensor * x);
    static void tower_after_run_(void * ud, ggml_cgraph * gf);

    bool build_text_graph(TextGraph & g, int T, int S, int n_img, int n_aud,
                          bool for_talker = false);
    bool run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<int> & pos4,
                  const std::vector<float> * img_feats, const std::vector<int> * img_pos,
                  const std::vector<float> * aud_feats, const std::vector<int> * aud_pos,
                  std::vector<float> & logits, int slot_base = 0,
                  std::vector<float> * hidden_out = nullptr,
                  std::vector<float> * embed_out = nullptr,
                  std::vector<float> * hidden_all = nullptr);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    // text
    int n_layer = 36, n_embd = 2048, n_head = 16, n_kv_head = 2, head_dim = 128;
    int n_ff = 11008, n_vocab = 151936;
    float rms_eps = 1e-6f, rope_theta = 1e6f;
    int mrope_section[3] = {16, 24, 24};

    // vision
    int v_layer = 32, v_embd = 1280, v_head = 16, v_hd = 80, v_ff = 3420;
    int v_patch = 14, v_temporal = 2, v_merge = 2, v_out = 2048;
    int v_window = 112, n_wa_pattern = 8;
    float v_eps = 1e-6f, v_rope_theta = 10000.0f;
    float v_mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };   // CLIP stats (processor config)
    float v_std[3] = { 0.26862954f, 0.26130258f, 0.27577711f };

    // shared Qwen2.5-VL vision tower (graph, weights, config) - see
    // core/qwen2vl_tower.hpp; the v_* fields above mirror tower_.cfg and stay
    // in sync from the gguf metadata in load_weights()
    vqa::qwen2vl_tower::Tower tower_;

    AudioTower audio;

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    vqa::MMap mm;
    vqa::AlignedBuf host_f32;
    ggml_gallocr_t allocr = nullptr;

    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr, *out_head = nullptr;
    std::vector<TxtLayer> tl;

    ggml_tensor *v_patch_sink = nullptr;   // guards the patch-embed dump read
    ggml_context * v_dump_ctx2 = nullptr, * v_dump_ctx3 = nullptr;   // layer-0 probes
    ggml_tensor * v_probe0 = nullptr, * v_probe1 = nullptr, * v_probe2 = nullptr;
    std::vector<std::string> v_layer_files;      // per-layer hidden dump paths
    ggml_context * v_dump_ctx = nullptr;         // persistent dump buffers (the
    ggml_backend_buffer_t v_dump_alloc = nullptr;// gallocr pool reuses dup sinks)
    std::vector<ggml_tensor *> v_dump_bufs;
    ggml_tensor * v_dump_rows = nullptr;

    vqa::KvCache kv;
    int n_past = 0;
    int next_pos = 0;      // MRoPE index for the next decoded token

    // optional speech synthesis (talker LM + token2wav, sibling gguf)
    TalkerLM talker;
    Token2Wav t2w;
    bool talker_loaded = false;
    vqa::MMap tmm;
    ggml_context * twctx = nullptr;
    ggml_backend_buffer_t twbuf = nullptr;
    vqa::AlignedBuf thost_f32;
    ggml_gallocr_t tallocr = nullptr;
    int spk_bos_token = 151872;
    std::vector<float> spk_cond;      // [192]
    std::vector<float> spk_ref_mel;   // [400*80]
    bool load_talker(const std::string & gguf_path);
    bool synthesize_speech(const std::vector<int> & toks, const std::vector<int> & pos,
                            const std::vector<float> & prompt_hidden,
                            const std::vector<float> & prompt_embed,
                            const std::vector<int> & gen_toks,
                            const std::vector<std::vector<float>> & gen_hidden,
                            const std::vector<std::vector<float>> & gen_embed,
                            int next_pos, const char * wav_path);

    int id_audio = 151646, id_audio_start = 151647, id_audio_end = 151648;
    int id_image_pad = 151655, id_video_pad = 151656;
    int id_vision_start = 151652, id_vision_end = 151653;
    int id_im_start = 151644, id_im_end = 151645, id_eos = 151645;

    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

// ---------------------------------------------------------------- lifecycle

Qwen25Omni::~Qwen25Omni() {
    kv.free();
    tower_.free();
    if (allocr) ggml_gallocr_free(allocr);
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wctx) ggml_free(wctx);
    if (backend) ggml_backend_free(backend);
}

bool Qwen25Omni::init_backend() {
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    audio.backend = backend;
    // separate graph allocator: sharing the vision/text gallocr corrupted the
    // audio graph's input buffers (compute read OOB gather indices)
    audio.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    audio.n_threads = n_threads;
    return allocr != nullptr;
}

bool Qwen25Omni::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    const char * A = "qwen25omni";
    n_layer   = vqa::gguf::u32(gu, "qwen25omni.block_count", n_layer);
    n_embd    = vqa::gguf::u32(gu, "qwen25omni.embedding_length", n_embd);
    n_head    = vqa::gguf::u32(gu, "qwen25omni.attention.head_count", n_head);
    n_kv_head = vqa::gguf::u32(gu, "qwen25omni.attention.head_count_kv", n_kv_head);
    head_dim  = vqa::gguf::u32(gu, "qwen25omni.attention.key_length", head_dim);
    n_ff      = vqa::gguf::u32(gu, "qwen25omni.feed_forward_length", n_ff);
    rms_eps   = vqa::gguf::f32(gu, "qwen25omni.attention.layer_norm_rms_epsilon", rms_eps);
    rope_theta = vqa::gguf::f32(gu, "qwen25omni.rope.freq_base", rope_theta);
    {
        const std::vector<int> sec = vqa::gguf::arr_u32(gu, "qwen25omni.rope.mrope_section",
                                                        {16, 24, 24});
        for (int i = 0; i < 3 && i < (int)sec.size(); i++) mrope_section[i] = sec[(size_t)i];
    }
    id_audio       = vqa::gguf::u32(gu, "qwen25omni.audio_token_id", id_audio);
    id_audio_start = vqa::gguf::u32(gu, "qwen25omni.audio_start_id", id_audio_start);
    id_audio_end   = vqa::gguf::u32(gu, "qwen25omni.audio_end_id", id_audio_end);
    id_image_pad   = vqa::gguf::u32(gu, "qwen25omni.image_token_id", id_image_pad);
    id_vision_start = vqa::gguf::u32(gu, "qwen25omni.vision_start_id", id_vision_start);
    id_vision_end  = vqa::gguf::u32(gu, "qwen25omni.vision_end_id", id_vision_end);
    id_video_pad   = vqa::gguf::u32(gu, "qwen25omni.video_token_id", id_video_pad);

    v_layer   = vqa::gguf::u32(gu, "qwen25omni.vision.block_count", v_layer);
    v_embd    = vqa::gguf::u32(gu, "qwen25omni.vision.embedding_length", v_embd);
    v_ff      = vqa::gguf::u32(gu, "qwen25omni.vision.feed_forward_length", v_ff);
    v_head    = vqa::gguf::u32(gu, "qwen25omni.vision.attention.head_count", v_head);
    v_patch   = vqa::gguf::u32(gu, "qwen25omni.vision.patch_size", v_patch);
    v_temporal = vqa::gguf::u32(gu, "qwen25omni.vision.temporal_patch_size", v_temporal);
    v_merge   = vqa::gguf::u32(gu, "qwen25omni.vision.spatial_merge_size", v_merge);
    v_out     = vqa::gguf::u32(gu, "qwen25omni.vision.out_hidden_size", v_out);
    v_window  = vqa::gguf::u32(gu, "qwen25omni.vision.window_size", v_window);
    v_eps     = vqa::gguf::f32(gu, "qwen25omni.vision.layer_norm_epsilon", v_eps);
    {
        int64_t id = gguf_find_key(gu, "qwen25omni.vision.image_mean");
        if (id >= 0 && gguf_get_arr_n(gu, id) == 3) {
            const float * a = (const float *)gguf_get_arr_data(gu, id);
            for (int c = 0; c < 3; c++) v_mean[c] = a[c];
        }
        id = gguf_find_key(gu, "qwen25omni.vision.image_std");
        if (id >= 0 && gguf_get_arr_n(gu, id) == 3) {
            const float * a = (const float *)gguf_get_arr_data(gu, id);
            for (int c = 0; c < 3; c++) v_std[c] = a[c];
        }
    }
    {
        const std::vector<int> fb = vqa::gguf::arr_u32(gu, "qwen25omni.vision.fullatt_blocks", {7});
        n_wa_pattern = (fb.empty() ? 8 : fb[0] + 1);
    }
    v_hd = v_embd / (v_head > 0 ? v_head : 1);

    tower_.cfg.n_layer = v_layer;  tower_.cfg.n_embd = v_embd;
    tower_.cfg.n_head = v_head;    tower_.cfg.n_hd = v_hd;
    tower_.cfg.n_patch = v_patch;  tower_.cfg.n_temporal = v_temporal;
    tower_.cfg.n_merge = v_merge;  tower_.cfg.n_out = v_out;
    tower_.cfg.n_window = v_window; tower_.cfg.n_wa_pattern = n_wa_pattern;
    tower_.cfg.eps = v_eps;        tower_.cfg.rope_theta = v_rope_theta;
    for (int c = 0; c < 3; c++) {
        tower_.cfg.img_mean[c] = v_mean[c];
        tower_.cfg.img_std[c] = v_std[c];
    }
    tower_.hooks.max_layers_env = "QOMNI_MAX_VLAYERS";
    tower_.hooks.ud = this;
    tower_.hooks.after_patch = &Qwen25Omni::tower_after_patch_;
    tower_.hooks.before_layers = &Qwen25Omni::tower_before_layers_;
    tower_.hooks.after_layer0_attn = &Qwen25Omni::tower_after_layer0_attn_;
    tower_.hooks.after_layer = &Qwen25Omni::tower_after_layer_;
    tower_.hooks.after_run = &Qwen25Omni::tower_after_run_;

    audio.n_layer = vqa::gguf::u32(gu, "qwen25omni.audio.block_count", audio.n_layer);
    audio.n_embd  = vqa::gguf::u32(gu, "qwen25omni.audio.embedding_length", audio.n_embd);
    audio.n_ff    = vqa::gguf::u32(gu, "qwen25omni.audio.feed_forward_length", audio.n_ff);
    audio.n_head  = vqa::gguf::u32(gu, "qwen25omni.audio.attention.head_count", audio.n_head);
    audio.n_mel   = vqa::gguf::u32(gu, "qwen25omni.audio.num_mel_bins", audio.n_mel);
    audio.n_pos   = vqa::gguf::u32(gu, "qwen25omni.audio.max_source_positions", audio.n_pos);
    audio.n_window = vqa::gguf::u32(gu, "qwen25omni.audio.n_window", audio.n_window);
    audio.out_dim = vqa::gguf::u32(gu, "qwen25omni.audio.output_dim", audio.out_dim);
    audio.ln_eps  = vqa::gguf::f32(gu, "qwen25omni.audio.layer_norm_epsilon", audio.ln_eps);
    audio.hop     = vqa::gguf::u32(gu, "qwen25omni.audio.hop_length", audio.hop);
    audio.n_fft   = vqa::gguf::u32(gu, "qwen25omni.audio.n_fft", audio.n_fft);

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);

    vqa::weights::promote_1d_to_f32(wctx, host_f32);
    {
        vqa::weights::Bound bound = vqa::weights::bind_to_backend(
            wctx, backend, mm, { { host_f32.data(), host_f32.size() } });
        if (!bound.ok) return false;
        wbuf = bound.main_buf;
        if (!ggml_backend_is_cpu(backend)) mm.release();
    }

    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) fprintf(stderr, "missing tensor: %s\n", n);
        return t;
    };
    tok_embd = gt("t.embed.weight");
    out_norm = gt("t.norm.weight");
    out_head = gt("t.output.weight");     // Omni's lm_head is not tied
    if (!out_head) out_head = tok_embd;
    tl.resize((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "t.blk.%d.%s", i, suf); return gt(b); };
        TxtLayer & L = tl[(size_t)i];
        L.norm = g("attn_norm.weight");
        L.wq = g("wq.weight"); L.bq = g("bq");
        L.wk = g("wk.weight"); L.bk = g("bk");
        L.wv = g("wv.weight"); L.bv = g("bv");
        L.wo = g("wo.weight");
        L.ffn_norm = g("ffn_norm.weight");
        L.gate = g("w_gate.weight"); L.up = g("w_up.weight"); L.down = g("w_down.weight");
        if (!L.norm || !L.wq || !L.wk || !L.wv || !L.wo || !L.ffn_norm || !L.gate || !L.up || !L.down)
            return false;
    }
    tower_.w.patch_w = gt("v.patch_embed.weight");
    tower_.w.merger_ln = gt("v.merger.ln.weight");
    tower_.w.mm1_w = gt("v.merger.fc1.weight"); tower_.w.mm1_b = gt("v.merger.fc1.bias");
    tower_.w.mm2_w = gt("v.merger.fc2.weight"); tower_.w.mm2_b = gt("v.merger.fc2.bias");
    tower_.w.layers.resize((size_t)v_layer);
    for (int i = 0; i < v_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "v.blk.%d.%s", i, suf); return gt(b); };
        vqa::qwen2vl_tower::Layer & L = tower_.w.layers[(size_t)i];
        L.ln1 = g("norm1.weight"); L.ln2 = g("norm2.weight");
        L.wqkv = g("wqkv.weight"); L.bqkv = g("wqkv.bias");
        L.wo = g("wo.weight"); L.bo = g("wo.bias");
        L.wg = g("w_gate.weight"); L.bg = g("w_gate.bias");
        L.wu = g("w_up.weight"); L.bu = g("w_up.bias");
        L.wd = g("w_down.weight"); L.bd = g("w_down.bias");
        if (!L.ln1 || !L.ln2 || !L.wqkv || !L.wo || !L.wg || !L.wu || !L.wd) return false;
    }

    // audio tower weights
    audio.conv1_w = gt("a.conv1.weight"); audio.conv1_b = gt("a.conv1.bias");
    audio.conv2_w = gt("a.conv2.weight"); audio.conv2_b = gt("a.conv2.bias");
    audio.pos_embd = gt("a.position_embd.weight");
    audio.mel_filters = gt("a.mel_filters");
    audio.ln_post_w = gt("a.ln_post.weight"); audio.ln_post_b = gt("a.ln_post.bias");
    audio.proj_w = gt("a.proj.weight"); audio.proj_b = gt("a.proj.bias");
    audio.layers.resize((size_t)audio.n_layer);
    for (int i = 0; i < audio.n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "a.blk.%d.%s", i, suf); return gt(b); };
        AudLayer & L = audio.layers[(size_t)i];
        L.ln1_w = g("ln_1.weight"); L.ln1_b = g("ln_1.bias");
        L.wq = g("wq.weight"); L.bq = g("bq");
        L.wk = g("wk.weight");
        L.wv = g("wv.weight"); L.bv = g("bv");
        L.wo = g("wo.weight"); L.bo = g("bo");
        L.ln2_w = g("ln_2.weight"); L.ln2_b = g("ln_2.bias");
        L.w_gate = g("w_gate.weight"); L.b_gate = g("b_gate");
        L.w_down = g("w_down.weight"); L.b_down = g("b_down");
        if (!L.ln1_w || !L.wq || !L.wk || !L.wv || !L.wo || !L.w_gate || !L.w_down) return false;
    }

    // keep a host copy of the mel filterbank: log_mel reads it directly, and
    // on a GPU backend the tensor data no longer lives on the host
    {
        // backend-aware copy: on GPU the filterbank bytes live on the device
        const int nf = audio.n_mel * (audio.n_fft / 2 + 1);
        audio.mel_filters_host.resize((size_t)nf);
        ggml_backend_tensor_get(audio.mel_filters, audio.mel_filters_host.data(), 0,
                                (size_t)nf * sizeof(float));
    }
    fprintf(stderr, "qwen2.5-omni: %d text layers, %d vision layers, %d audio layers\n",
            n_layer, v_layer, audio.n_layer);
    return tok_embd && out_norm && out_head;
}

bool Qwen25Omni::load_talker(const std::string & gguf_path) {
    if (!tmm.load(gguf_path.c_str())) return false;
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &twctx };
    struct gguf_context * gu = gguf_init_from_file(gguf_path.c_str(), p);
    if (!gu) return false;
    talker.n_layer = vqa::gguf::u32(gu, "qwen25omni-talker.block_count", talker.n_layer);
    talker.n_embd  = vqa::gguf::u32(gu, "qwen25omni-talker.embedding_length", talker.n_embd);
    talker.embw    = vqa::gguf::u32(gu, "qwen25omni-talker.embedding_width", talker.embw);
    talker.n_head  = vqa::gguf::u32(gu, "qwen25omni-talker.attention.head_count", talker.n_head);
    talker.n_kv_head = vqa::gguf::u32(gu, "qwen25omni-talker.attention.head_count_kv", talker.n_kv_head);
    talker.head_dim  = vqa::gguf::u32(gu, "qwen25omni-talker.attention.key_length", talker.head_dim);
    talker.n_ff    = vqa::gguf::u32(gu, "qwen25omni-talker.feed_forward_length", talker.n_ff);
    talker.n_vocab = vqa::gguf::u32(gu, "qwen25omni-talker.codec_vocab_size", talker.n_vocab);
    talker.eps     = vqa::gguf::f32(gu, "qwen25omni-talker.attention.layer_norm_rms_epsilon", talker.eps);
    talker.rope_theta = vqa::gguf::f32(gu, "qwen25omni-talker.rope.freq_base", talker.rope_theta);
    {
        const std::vector<int> sec = vqa::gguf::arr_u32(gu, "qwen25omni-talker.rope.mrope_section",
                                                        {16, 16, 0});
        for (int i = 0; i < 3 && i < (int)sec.size(); i++) talker.mrope_section[i] = sec[(size_t)i];
    }
    talker.id_codec_start = vqa::gguf::u32(gu, "qwen25omni-talker.codec_start", talker.id_codec_start);
    talker.id_codec_end   = vqa::gguf::u32(gu, "qwen25omni-talker.codec_end", talker.id_codec_end);
    talker.id_codec_pad   = vqa::gguf::u32(gu, "qwen25omni-talker.codec_pad", talker.id_codec_pad);
    talker.id_codec_mask  = vqa::gguf::u32(gu, "qwen25omni-talker.codec_mask", talker.id_codec_mask);
    t2w.d_depth = vqa::gguf::u32(gu, "qwen25omni-talker.dit.depth", t2w.d_depth);
    t2w.d_dim   = vqa::gguf::u32(gu, "qwen25omni-talker.dit.dim", t2w.d_dim);
    t2w.d_emb   = vqa::gguf::u32(gu, "qwen25omni-talker.dit.emb_dim", t2w.d_emb);
    t2w.d_head  = vqa::gguf::u32(gu, "qwen25omni-talker.dit.head_count", t2w.d_head);
    t2w.d_hd    = vqa::gguf::u32(gu, "qwen25omni-talker.dit.key_length", t2w.d_hd);
    t2w.d_mel   = vqa::gguf::u32(gu, "qwen25omni-talker.dit.mel_dim", t2w.d_mel);
    t2w.d_embeds= vqa::gguf::u32(gu, "qwen25omni-talker.dit.num_embeds", t2w.d_embeds);
    t2w.d_repeats = vqa::gguf::u32(gu, "qwen25omni-talker.dit.repeats", t2w.d_repeats);
    t2w.d_block = vqa::gguf::u32(gu, "qwen25omni-talker.dit.block_size", t2w.d_block);
    {
        const std::vector<int> la = vqa::gguf::arr_u32(gu, "qwen25omni-talker.dit.look_ahead_layers",
                                                       {10});
        t2w.look_ahead_layers.assign(la.begin(), la.end());
        const std::vector<int> lb = vqa::gguf::arr_u32(gu, "qwen25omni-talker.dit.look_backward_layers",
                                                       {0, 20});
        t2w.look_backward_layers.assign(lb.begin(), lb.end());
    }
    t2w.enc_dim = vqa::gguf::u32(gu, "qwen25omni-talker.dit.enc_dim", t2w.enc_dim);
    t2w.enc_emb_dim = vqa::gguf::u32(gu, "qwen25omni-talker.dit.enc_emb_dim", t2w.enc_emb_dim);

    vqa::weights::bind_from_mmap(gu, twctx, tmm);
    gguf_free(gu);
    vqa::weights::promote_1d_to_f32(twctx, thost_f32);
    vqa::weights::Bound bound = vqa::weights::bind_to_backend(twctx, backend, tmm,
                                                              { { thost_f32.data(), thost_f32.size() } });
    if (!bound.ok) return false;
    twbuf = bound.main_buf;

    if (!talker.load(backend, twctx)) return false;
    if (!t2w.load(backend, twctx)) return false;
    // baked speaker (Chelsie by default)
    char key[96];
    snprintf(key, sizeof(key), "spk.Chelsie.bos_token");
    ggml_tensor * t = ggml_get_tensor(twctx, "spk.Chelsie.cond");
    if (t) {
        t2w.spk_cond.resize((size_t)ggml_nelements(t));
        ggml_backend_tensor_get(t, t2w.spk_cond.data(), 0, ggml_nbytes(t));
    }
    t = ggml_get_tensor(twctx, "spk.Chelsie.ref_mel");
    if (t) {
        t2w.spk_ref_mel.resize((size_t)ggml_nelements(t));
        ggml_backend_tensor_get(t, t2w.spk_ref_mel.data(), 0, ggml_nbytes(t));
    }
    spk_bos_token = 151872;   // also present as kv metadata; the default voice
    talker_loaded = true;
    fprintf(stderr, "qwen2.5-omni: talker loaded (%s)\n", gguf_path.c_str());
    return true;
}

bool Qwen25Omni::alloc_kv() {
    return kv.alloc(backend, n_layer, n_kv_head, head_dim, n_ctx);
}

bool Qwen25Omni::load(const std::string & gguf_path, const std::string & tokenizer_path,
                      const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "QOMNI", n_threads, use_gpu, dump_dir);
    audio.dump_dir = dump_dir;
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::Qwen2Regex)) return false;
    if (!alloc_kv()) return false;

    vqa::assign_token_ids(tok_, {
        { "<|AUDIO|>",         &id_audio },
        { "<|audio_bos|>",     &id_audio_start },
        { "<|audio_eos|>",     &id_audio_end },
        { "<|IMAGE|>",         &id_image_pad },
        { "<|VIDEO|>",         &id_video_pad },
        { "<|vision_bos|>",    &id_vision_start },
        { "<|vision_eos|>",    &id_vision_end },
        { "<|im_start|>",      &id_im_start },
        { "<|im_end|>",        &id_im_end },
    });
    id_eos = id_im_end;
    caps_.image = true;
    caps_.video = false;
    caps_.audio = true;   // whisper encoder consumes audio tracks
    // optional talker sibling: any qwen25omni-talker-*.gguf next to the thinker
    {
        std::string dir = gguf_path;
        const size_t slash = dir.find_last_of("/\\");
        dir = slash == std::string::npos ? std::string(".") : dir.substr(0, slash);
        std::string talker;
        for (const std::string & cand :
             { dir + "/qwen25omni-talker-f16.gguf", dir + "/qwen25omni-talker-q8_0.gguf",
               dir + "/qwen25omni-talker-q4_0.gguf" }) {
            if (FILE * f = fopen(cand.c_str(), "rb")) { fclose(f); talker = cand; break; }
        }
        if (!talker.empty() && !getenv("QOMNI_NO_TALKER")) load_talker(talker);
    }
    fprintf(stderr, "qwen2.5-omni: ready (%dL d=%d nh=%d nkv=%d hd=%d vocab=%d)\n",
            n_layer, n_embd, n_head, n_kv_head, head_dim, n_vocab);
    return true;
}

// ---------------------------------------------------------------- vision
//
// Same tower as the qwen2vl backend (Qwen2.5-VL): window attention over 112px
// windows, 2D M-RoPE, merger. Kept inline here because the Omni converter
// bakes the same tensor names.

bool Qwen25Omni::build_vision_graph(int gh, int gw) {
    return tower_.build(gh, gw, allocr, backend);
}

bool Qwen25Omni::run_vision(const std::vector<float> & patches, std::vector<float> & feats) {
    if (const char * dp = getenv("QOMNI_VIN_DUMP")) {   // raw patchified input
        FILE * f = fopen(dp, "wb");
        if (f) { int r = 1, c = (int)patches.size();
                 fwrite(&r, 4, 1, f); fwrite(&c, 4, 1, f);
                 fwrite(patches.data(), 4, patches.size(), f); fclose(f); }
    }
    std::vector<float> patches_ov;
    if (const char * lp = getenv("QOMNI_VIN_LOAD")) {   // parity: exact HF pixels
        FILE * f = fopen(lp, "rb");
        if (f) {
            patches_ov.resize(patches.size());
            if (fread(patches_ov.data(), 4, patches_ov.size(), f) == patches_ov.size())
                const_cast<std::vector<float> &>(patches) = patches_ov;
            else fprintf(stderr, "vin load size mismatch\n");
            fclose(f);
        }
    }
    return tower_.run(backend, patches, feats);
}

// ------------------------------------------------------------------ tower hooks

void Qwen25Omni::tower_after_patch_(void * ud, ggml_context * ctx, ggml_cgraph * gf,
                                    int v_embd, int N, ggml_tensor * x) {
    Qwen25Omni * m = (Qwen25Omni *)ud;
    if (getenv("QOMNI_VB0_DUMP")) {   // layer-0 bisection probes (persistent
        m->v_dump_ctx2 = m->v_dump_ctx2 ? m->v_dump_ctx2 : ggml_init(   // buffer trick
            { (size_t)8 * ggml_tensor_overhead(), NULL, true });
        if (!m->v_probe0) {
            m->v_probe0 = ggml_new_tensor_2d(m->v_dump_ctx2, GGML_TYPE_F32, v_embd, N);
            m->v_probe1 = ggml_new_tensor_1d(m->v_dump_ctx2, GGML_TYPE_I32, N);
            ggml_backend_alloc_ctx_tensors(m->v_dump_ctx2, m->backend);
            std::vector<int> r(N);
            for (int i = 0; i < N; i++) r[i] = i;
            ggml_backend_tensor_set(m->v_probe1, r.data(), 0, sizeof(int) * N);
        }
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, m->v_probe0,
            ggml_reshape_2d(ctx, ggml_cont(ctx, x), v_embd, N), m->v_probe1));
    }
    if (getenv("QOMNI_VPATCH_DUMP")) {   // dup: the bare buffer gets reused
        m->v_patch_sink = ggml_dup(ctx, x);
        ggml_build_forward_expand(gf, m->v_patch_sink);
    }
}

void Qwen25Omni::tower_before_layers_(void * ud, ggml_context * /*ctx*/,
                                      ggml_cgraph * /*gf*/, int v_layer, int v_embd, int N) {
    Qwen25Omni * m = (Qwen25Omni *)ud;
    if (getenv("QOMNI_VLAYER_DUMP")) {   // persistent per-layer dump buffers,
        if (m->v_dump_ctx) { ggml_free(m->v_dump_ctx); m->v_dump_ctx = nullptr; }
        struct ggml_init_params dip = { (size_t)(v_layer * 2 + 8) * ggml_tensor_overhead(),
                                        NULL, true };
        m->v_dump_ctx = ggml_init(dip);
        m->v_dump_bufs.assign(v_layer, nullptr);
        for (int il = 0; il < v_layer; il++)
            m->v_dump_bufs[il] = ggml_new_tensor_2d(m->v_dump_ctx, GGML_TYPE_F32, v_embd, N);
        m->v_dump_rows = ggml_new_tensor_1d(m->v_dump_ctx, GGML_TYPE_I32, N);
        m->v_dump_alloc = ggml_backend_alloc_ctx_tensors(m->v_dump_ctx, m->backend);
        ggml_backend_buffer_clear(m->v_dump_alloc, 0);
        std::vector<int> r(N);
        for (int i = 0; i < N; i++) r[i] = i;
        ggml_backend_tensor_set(m->v_dump_rows, r.data(), 0, sizeof(int) * N);
    }
}

void Qwen25Omni::tower_after_layer0_attn_(void * ud, ggml_context * ctx, ggml_cgraph * gf,
                                          int v_embd, int N, ggml_tensor * x) {
    Qwen25Omni * m = (Qwen25Omni *)ud;
    if (getenv("QOMNI_VB0_DUMP")) {
        m->v_dump_ctx3 = ggml_init({ (size_t)4 * ggml_tensor_overhead(), NULL, true });
        m->v_probe2 = ggml_new_tensor_2d(m->v_dump_ctx3, GGML_TYPE_F32, v_embd, N);
        ggml_backend_alloc_ctx_tensors(m->v_dump_ctx3, m->backend);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, m->v_probe2,
            ggml_reshape_2d(ctx, ggml_cont(ctx, x), v_embd, N), m->v_probe1));
    }
}

void Qwen25Omni::tower_after_layer_(void * ud, ggml_context * ctx, ggml_cgraph * gf,
                                    int il, int v_embd, int N, ggml_tensor * x) {
    Qwen25Omni * m = (Qwen25Omni *)ud;
    if (m->v_dump_rows) {   // per-layer hidden copy into persistent buffers
        char fn[256];
        snprintf(fn, sizeof(fn), "%s_L%02d.bin", getenv("QOMNI_VLAYER_DUMP"), il);
        m->v_layer_files.push_back(fn);
        ggml_tensor * src2 = ggml_reshape_2d(ctx, ggml_cont(ctx, x), v_embd, N);
        ggml_build_forward_expand(gf,
            ggml_set_rows(ctx, m->v_dump_bufs[il], src2, m->v_dump_rows));
    }
}

void Qwen25Omni::tower_after_run_(void * ud, ggml_cgraph * /*gf*/) {
    Qwen25Omni * m = (Qwen25Omni *)ud;
    const int v_embd = m->tower_.cfg.n_embd;
    const int N = m->tower_.st.n_patches_x * m->tower_.st.n_patches_y;
    for (size_t i = 0; i < m->v_layer_files.size(); i++) {   // per-layer hiddens
        std::vector<float> hv((size_t)ggml_nelements(m->v_dump_bufs[i]));
        ggml_backend_tensor_get(m->v_dump_bufs[i], hv.data(), 0, sizeof(float) * hv.size());
        FILE * f = fopen(m->v_layer_files[i].c_str(), "wb");
        if (f) { int r = v_embd, c = N;
                 fwrite(&r, 4, 1, f); fwrite(&c, 4, 1, f);
                 fwrite(hv.data(), 4, hv.size(), f); fclose(f); }
    }
    m->v_layer_files.clear();
    if (m->v_probe0) {
        auto wr = [&](ggml_tensor * t, const char * fn) {
            std::vector<float> hv((size_t)ggml_nelements(t));
            ggml_backend_tensor_get(t, hv.data(), 0, sizeof(float) * hv.size());
            FILE * f = fopen(fn, "wb");
            if (f) { int r = v_embd, c = N;
                     fwrite(&r, 4, 1, f); fwrite(&c, 4, 1, f);
                     fwrite(hv.data(), 4, hv.size(), f); fclose(f); }
        };
        wr(m->v_probe0, "vb0_patch.bin");
        if (m->v_probe2) wr(m->v_probe2, "vb0_attn.bin");
    }
    if (m->v_patch_sink) {
        const char * dp = getenv("QOMNI_VPATCH_DUMP");
        ggml_tensor * pe = m->v_patch_sink;
        if (pe) {
            std::vector<float> hv((size_t)ggml_nelements(pe));
            ggml_backend_tensor_get(pe, hv.data(), 0, sizeof(float) * hv.size());
            FILE * f = fopen(dp, "wb");
            if (f) { int r = v_embd, c = N;
                     fwrite(&r, 4, 1, f); fwrite(&c, 4, 1, f);
                     fwrite(hv.data(), 4, hv.size(), f); fclose(f); }
        }
    }
}

bool Qwen25Omni::encode_vision(const vqa::MediaInput & media, std::vector<float> & feats,
                               int * n_tokens, int grid[3]) {
    if (media.empty()) return false;
    // smart_resize: multiples of patch*merge, total pixels within the official bounds
    const int factor = v_patch * v_merge;
    const int min_px = 4 * factor * factor;               // 4*28*28
    const int max_px = 16384 * factor * factor / 4;       // 4096 merged tokens cap
    int rh = 0, rw = 0;
    if (!vqa::img::smart_resize(media.height, media.width, factor, min_px, max_px, &rh, &rw)) {
        fprintf(stderr, "image too small for the vision tower\n");
        return false;
    }
    const int gh = rh / v_patch, gw = rw / v_patch;

    std::vector<float> resized;
    const uint8_t * src_rgb = media.frames[0].rgb.data();
    std::vector<uint8_t> raw_img;
    if (const char * lp = getenv("QOMNI_RAWIMG_LOAD")) {  // parity: PIL-decoded
        FILE * f = fopen(lp, "rb");                          // source pixels
        if (f) {
            const size_t n = (size_t)media.width * media.height * 3;
            raw_img.resize(n);
            if (fread(raw_img.data(), 1, n, f) == n) src_rgb = raw_img.data();
            else fprintf(stderr, "rawimg size mismatch\n");
            fclose(f);
        }
    }
    vqa::img::resize_tv_bicubic_aa(src_rgb, media.width, media.height, rw, rh, resized);
    const int N = gh * gw;
    // 2x2 group-order patch extraction + CLIP mean/std (see qwen2vl_tower.hpp)
    std::vector<float> patches;
    tower_.patchify(resized.data(), rw, rh, gh, gw, patches);

    if (!build_vision_graph(gh, gw)) return false;
    if (!run_vision(patches, feats)) return false;
    if (getenv("QOMNI_VIS_DUMP")) {
        const int hdr[2] = { (int)feats.size() / v_out, v_out };
        dump_bins(getenv("QOMNI_VIS_DUMP"), hdr, 2, feats.data(), feats.size());
    }
    *n_tokens = (int)feats.size() / v_out;
    grid[0] = 1;
    grid[1] = gh / v_merge;
    grid[2] = gw / v_merge;
    fprintf(stderr, "vision: %dx%d -> %d patches -> %d tokens\n", rw, rh, N, *n_tokens);
    return true;
}

// ---------------------------------------------------------------- audio

bool Qwen25Omni::encode_audio(const vqa::MediaInput & media, std::vector<float> & feats,
                              int * n_tokens) {
    if (media.audio.empty() || media.sample_rate <= 0) return false;
    // the tower is trained on 16 kHz input; the media layer already resamples,
    // so anything else here is a bug in the loader - just assert the rate
    if (media.sample_rate != 16000) {
        fprintf(stderr, "audio: expected 16 kHz input, got %d Hz\n", media.sample_rate);
        return false;
    }
    // keep the tower inside the position table / KV budget (25 tokens / second)
    const int max_samples = audio.n_pos * 2 * audio.hop;   // 1500 frames = 30 s
    const int n = std::min<int>((int)media.audio.size(), max_samples);
    if ((int)media.audio.size() > n) fprintf(stderr, "audio: truncated to 30 s\n");

    MelSpec mel = audio.log_mel(media.audio.data(), n);
    if (const char * fp = getenv("QOMNI_MEL")) {   // load a reference mel instead
        FILE * f = fopen(fp, "rb");
        if (f) {
            int shape[2];
            if (fread(shape, sizeof(int), 2, f) == 2 && shape[0] == audio.n_mel) {
                mel.n_mel = shape[0];
                mel.n_len = shape[1];
                mel.data.resize((size_t)mel.n_mel * mel.n_len);
                if (fread(mel.data.data(), sizeof(float), mel.data.size(), f) == mel.data.size())
                    fprintf(stderr, "audio: mel %dx%d from %s (FFT skipped)\n",
                            mel.n_mel, mel.n_len, fp);
            }
            fclose(f);
        }
    }
    fprintf(stderr, "audio: %d samples -> mel %dx%d\n", n, mel.n_mel, mel.n_len);
    if (const char * dp = getenv("QOMNI_MEL_DUMP")) {
        const int hdr[2] = { mel.n_mel, mel.n_len };
        dump_bins(dp, hdr, 2, mel.data.data(), mel.data.size());
    }
    const bool ok = audio.encode(mel, feats, n_tokens);
    if (ok && getenv("QOMNI_AUD_DUMP")) {
        const int hdr[2] = { (int)feats.size() / audio.out_dim, audio.out_dim };
        dump_bins(getenv("QOMNI_AUD_DUMP"), hdr, 2, feats.data(), feats.size());
    }
    return ok;
}

// ---------------------------------------------------------------- text

bool Qwen25Omni::build_text_graph(TextGraph & g, int T, int S, int n_img, int n_aud,
                                  bool for_talker) {
    struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 32768, false);

    const int D = n_embd, HD = head_dim, NH = n_head, NKV = n_kv_head;

    g.tok_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    // MRoPE positions: [t(T), h(T), w(T), unused(T)] — ggml reads 4 arrays
    g.pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, 4 * T);
    ggml_set_name(g.pos_in, "pos_in");
    g.slot_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.slot_in, "slot_in");
    g.mask_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, S, T);
    ggml_set_name(g.mask_in, "mask_in");

    ggml_tensor * x = ggml_get_rows(g.ctx, tok_embd, g.tok_in);   // [D, T]
    if (n_img > 0) {
        g.img_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, n_img);
        ggml_set_name(g.img_in, "img_in");
        g.img_pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_img);
        ggml_set_name(g.img_pos_in, "img_pos_in");
        // ggml_set_rows writes into its destination, so it needs its own copy
        x = ggml_set_rows(g.ctx, ggml_dup(g.ctx, x), g.img_in, g.img_pos_in);
    }
    if (n_aud > 0) {
        g.aud_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, n_aud);
        ggml_set_name(g.aud_in, "aud_in");
        g.aud_pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_aud);
        ggml_set_name(g.aud_pos_in, "aud_pos_in");
        x = ggml_set_rows(g.ctx, ggml_dup(g.ctx, x), g.aud_in, g.aud_pos_in);
    }
    ggml_set_name(x, "t_emb");
    if (for_talker) {
        // the talker sees zeros at multimodal positions (HF zeroes the
        // encoder features out of the embedding stream it consumes)
        g.emb0_out = x;   // read back with zero rows patched host-side
    }

    int sec[4] = { mrope_section[0], mrope_section[1], mrope_section[2], 0 };
    for (int il = 0; il < n_layer; il++) {
        TxtLayer & L = tl[(size_t)il];
        ggml_tensor * l = ggml_rms_norm(g.ctx, x, rms_eps);
        l = ggml_mul(g.ctx, l, L.norm);

        ggml_tensor * q = ggml_mul_mat(g.ctx, L.wq, l);
        q = ggml_add(g.ctx, q, L.bq);
        ggml_tensor * k = ggml_mul_mat(g.ctx, L.wk, l);
        k = ggml_add(g.ctx, k, L.bk);
        ggml_tensor * v = ggml_mul_mat(g.ctx, L.wv, l);
        v = ggml_add(g.ctx, v, L.bv);
        if (il == 0) { ggml_set_name(q, "t_l0_q"); ggml_set_name(k, "t_l0_k"); }

        ggml_tensor * q3 = ggml_reshape_3d(g.ctx, q, HD, NH, T);
        ggml_tensor * k3 = ggml_reshape_3d(g.ctx, k, HD, NKV, T);
        q3 = ggml_rope_multi(g.ctx, q3, g.pos_in, nullptr, HD, sec, GGML_ROPE_TYPE_MROPE,
                             n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_multi(g.ctx, k3, g.pos_in, nullptr, HD, sec, GGML_ROPE_TYPE_MROPE,
                             n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // the cache is F16, so cast explicitly: ggml_set_rows copies element-wise
        // and a mixed dtype silently writes garbage
        ggml_tensor * k2 = ggml_cast(g.ctx, ggml_reshape_2d(g.ctx, k3, NKV * HD, T),
                                     GGML_TYPE_F16);
        ggml_tensor * v2 = ggml_cast(g.ctx, ggml_reshape_2d(g.ctx, v, NKV * HD, T),
                                     GGML_TYPE_F16);
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.k[il], k2, g.slot_in));
        ggml_build_forward_expand(g.gf, ggml_set_rows(g.ctx, kv.v[il], v2, g.slot_in));

        ggml_tensor * qf = ggml_cont(g.ctx, ggml_permute(g.ctx, q3, 0, 2, 1, 3));   // [HD, T, NH]
        ggml_tensor * kf = ggml_view_3d(g.ctx, kv.k[il], HD, S, NKV,
                                        kv.k[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * vf = ggml_view_3d(g.ctx, kv.v[il], HD, S, NKV,
                                        kv.v[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * fa = ggml_flash_attn_ext(g.ctx, qf, kf, vf, g.mask_in,
                                               1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(g.ctx, ggml_cont(g.ctx, fa), NH * HD, T);
        attn = ggml_mul_mat(g.ctx, L.wo, attn);
        x = ggml_add(g.ctx, x, attn);

        ggml_tensor * m = ggml_rms_norm(g.ctx, x, rms_eps);
        m = ggml_mul(g.ctx, m, L.ffn_norm);
        ggml_tensor * gate = ggml_mul_mat(g.ctx, L.gate, m);
        ggml_tensor * up = ggml_mul_mat(g.ctx, L.up, m);
        m = ggml_swiglu_split(g.ctx, gate, up);
        m = ggml_mul_mat(g.ctx, L.down, m);
        x = ggml_add(g.ctx, x, m);
        if (il == 0) ggml_set_name(x, "t_l0_out");
    }

    if (for_talker) {
        g.embed_out = ggml_get_rows(g.ctx, tok_embd, g.tok_in);
        ggml_set_name(g.embed_out, "t_embed_out");
        // HF reply track: hidden_states[-1]; the 5.x capture machinery may or
        // may not include the final norm - switchable for parity bisecting
        // HF 5.x capture: hidden_states[-1] = last-layer output, final norm NOT
        // included - the talker reply track is the pre-norm hidden
        // (QOMNI_TALK_POSTNORM=1 switches back for bisecting)
        const bool pre = getenv("QOMNI_TALK_POSTNORM") == nullptr;
        if (for_talker) fprintf(stderr, "text: hidden track %s" "\n", pre ? "PRENORM" : "POSTNORM");
        // dup: without it gallocr reuses x's buffer for the final-norm result
        g.hidden_out = pre ? ggml_dup(g.ctx, x) : nullptr;
    }
    x = ggml_rms_norm(g.ctx, x, rms_eps);
    x = ggml_mul(g.ctx, x, out_norm);
    ggml_set_name(x, "t_final_norm");
    if (for_talker && !g.hidden_out) g.hidden_out = x;

    if (T > 1) {
        ggml_tensor * last = ggml_view_2d(g.ctx, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
        g.logits = ggml_mul_mat(g.ctx, out_head, last);
    } else {
        g.logits = ggml_mul_mat(g.ctx, out_head, x);
    }
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);
    if (g.hidden_out) ggml_build_forward_expand(g.gf, g.hidden_out);
    if (g.embed_out) ggml_build_forward_expand(g.gf, g.embed_out);
    if (g.emb0_out) ggml_build_forward_expand(g.gf, g.emb0_out);

    if (!ggml_gallocr_alloc_graph(allocr, g.gf)) {
        fprintf(stderr, "text graph allocation failed\n");
        return false;
    }
    return true;
}

bool Qwen25Omni::run_text(TextGraph & g, const std::vector<int> & toks,
                          const std::vector<int> & pos4, const std::vector<float> * img_feats,
                          const std::vector<int> * img_pos, const std::vector<float> * aud_feats,
                          const std::vector<int> * aud_pos, std::vector<float> & logits,
                          int slot_base, std::vector<float> * hidden_out,
                          std::vector<float> * embed_out,
                          std::vector<float> * hidden_all) {
    const int T = (int)toks.size();
    ggml_backend_tensor_set(g.tok_in, toks.data(), 0, sizeof(int) * toks.size());
    ggml_backend_tensor_set(g.pos_in, pos4.data(), 0, sizeof(int) * pos4.size());
    {
        std::vector<int> slots((size_t)T);
        for (int i = 0; i < T; i++) slots[(size_t)i] = slot_base + i;
        ggml_backend_tensor_set(g.slot_in, slots.data(), 0, sizeof(int) * slots.size());
    }
    if (g.img_in && img_feats && img_pos) {
        ggml_backend_tensor_set(g.img_in, img_feats->data(), 0,
                                sizeof(float) * img_feats->size());
        ggml_backend_tensor_set(g.img_pos_in, img_pos->data(), 0, sizeof(int) * img_pos->size());
    }
    if (g.aud_in && aud_feats && aud_pos) {
        ggml_backend_tensor_set(g.aud_in, aud_feats->data(), 0,
                                sizeof(float) * aud_feats->size());
        ggml_backend_tensor_set(g.aud_pos_in, aud_pos->data(), 0, sizeof(int) * aud_pos->size());
    }
    if (getenv("QOMNI_DBG_IDX")) {   // read back every index input after upload
        auto rb = [](ggml_tensor * t, const char * what) {
            std::vector<int> v((size_t)ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, sizeof(int) * v.size());
            int mn = v.empty() ? 0 : v[0], mx = mn;
            for (int x : v) { mn = std::min(mn, x); mx = std::max(mx, x); }
            fprintf(stderr, "dbg %s: n=%zu min=%d max=%d\n", what, v.size(), mn, mx);
        };
        rb(g.tok_in, "tok_in"); rb(g.pos_in, "pos_in"); rb(g.slot_in, "slot_in");
        if (g.img_pos_in) rb(g.img_pos_in, "img_pos_in");
        if (g.aud_pos_in) rb(g.aud_pos_in, "aud_pos_in");
    }
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "text compute failed\n");
        return false;
    }
    logits.resize((size_t)n_vocab);
    ggml_backend_tensor_get(g.logits, logits.data(), 0, sizeof(float) * logits.size());
    if (g.hidden_out && hidden_out) {
        hidden_out->resize((size_t)n_embd);
        const size_t off = g.hidden_out->ne[1] > 1
            ? (size_t)(g.hidden_out->ne[1] - 1) * g.hidden_out->nb[1] : 0;
        ggml_backend_tensor_get(g.hidden_out, hidden_out->data(), off, sizeof(float) * n_embd);
    }
    if (g.hidden_out && hidden_all) {
        hidden_all->resize((size_t)n_embd * g.hidden_out->ne[1]);
        ggml_backend_tensor_get(g.hidden_out, hidden_all->data(), 0,
                                sizeof(float) * hidden_all->size());
    }
    if (g.embed_out && embed_out) {
        embed_out->resize((size_t)n_embd);
        ggml_backend_tensor_get(g.embed_out, embed_out->data(), 0, sizeof(float) * n_embd);
    }
    return true;
}

// ---------------------------------------------------------------- ask

bool Qwen25Omni::ask(const vqa::MediaInput & media, const std::string & prompt,
                     const std::string & /*mode*/, const vqa::GenParams & gp,
                     std::string & out_text) {
    out_text.clear();

    std::vector<float> img_feats;
    int n_img = 0;
    int grid[3] = { 0, 0, 0 };
    if (!media.empty()) {
        if (gp.on_stage) gp.on_stage("vision", 0, 1, gp.on_stage_ud);
        if (!encode_vision(media, img_feats, &n_img, grid)) return false;
        if (gp.on_stage) gp.on_stage("vision", 1, 1, gp.on_stage_ud);
    }

    std::vector<float> aud_feats;
    int n_aud = 0;
    if (media.has_audio()) {
        if (gp.on_stage) gp.on_stage("audio", 0, 1, gp.on_stage_ud);
        if (!encode_audio(media, aud_feats, &n_aud)) return false;
        if (gp.on_stage) gp.on_stage("audio", 1, 1, gp.on_stage_ud);
    }

    std::string text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n";
    // official chat template order: vision block first, then audio, then text
    if (n_img > 0) {
        text += "<|vision_bos|>";
        for (int i = 0; i < n_img; i++) text += "<|IMAGE|>";
        text += "<|vision_eos|>";
    }
    if (n_aud > 0) {
        text += "<|audio_bos|>";
        for (int i = 0; i < n_aud; i++) text += "<|AUDIO|>";
        text += "<|audio_eos|>";
    }
    text += prompt;
    text += "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<int> toks = tok_.encode(text);
    if (getenv("QOMNI_DBG_IDX"))
        fprintf(stderr, "dbg ask: n_img=%d n_aud=%d T=%zu grid=%d,%d,%d\n",
                n_img, n_aud, toks.size(), grid[0], grid[1], grid[2]);
    if ((int)toks.size() + gp.max_tokens + 8 > n_ctx) {
        fprintf(stderr, "prompt %zu + %d tokens exceed n_ctx %d\n", toks.size(), gp.max_tokens, n_ctx);
        return false;
    }
    const int T = (int)toks.size();

    // official get_rope_index: audio tokens are plain running positions; image
    // tokens take (t, h, w) and the index advances by max(grid_h, grid_w)
    std::vector<int> pos((size_t)4 * T, 0);
    std::vector<int> img_pos, aud_pos;
    img_pos.reserve((size_t)n_img);
    aud_pos.reserve((size_t)n_aud);
    {
        const int gh = grid[0] > 0 ? grid[1] : 0, gw = grid[0] > 0 ? grid[2] : 0;
        int st = 0, seen_img = 0, seen_aud = 0;
        for (int i = 0; i < T; i++) {
            if (n_aud > 0 && toks[(size_t)i] == id_audio && seen_aud < n_aud) {
                pos[(size_t)i] = pos[(size_t)T + i] = pos[(size_t)2 * T + i] = st;
                aud_pos.push_back(i);
                seen_aud++;
                st++;
            } else if (n_img > 0 && toks[(size_t)i] == id_image_pad && seen_img < n_img) {
                const int k = seen_img % (gh * gw);
                pos[(size_t)i] = st;                          // t (grid_t == 1 -> 0)
                pos[(size_t)T + i] = st + k / gw;              // h
                pos[(size_t)2 * T + i] = st + k % gw;          // w
                img_pos.push_back(i);
                seen_img++;
                if (seen_img % (gh * gw) == 0) st += std::max(gh, gw);
            } else {
                pos[(size_t)i] = pos[(size_t)T + i] = pos[(size_t)2 * T + i] = st;
                st++;
            }
        }
    }
    // the placeholder tokens MUST all be single special tokens in the loaded
    // tokenizer: a wrong-but-same-size tokenizer (vocab-probe collision) splits
    // them, the gather below finds nothing, and the set_rows injection then
    // reads uninitialized device indices - fail loudly instead
    if ((n_img > 0 && img_pos.size() != (size_t)n_img) ||
        (n_aud > 0 && aud_pos.size() != (size_t)n_aud)) {
        fprintf(stderr, "qwen25omni: soft-token placeholders not found in the tokenizer "
                        "(img %zu/%d aud %zu/%d) - wrong tokenizer.json for this model\n",
                img_pos.size(), n_img, aud_pos.size(), n_aud);
        return false;
    }

    const bool want_talker = talker_loaded && gp.speak_wav;
    TextGraph g;
    if (!build_text_graph(g, T, T, n_img, n_aud, want_talker)) { ggml_free(g.ctx); return false; }
    vqa::set_causal_mask(g.mask_in, 0, T);
    std::vector<float> logits, prompt_hidden_all;
    if (!run_text(g, toks, pos, n_img > 0 ? &img_feats : nullptr,
                  n_img > 0 ? &img_pos : nullptr, n_aud > 0 ? &aud_feats : nullptr,
                  n_aud > 0 ? &aud_pos : nullptr, logits, 0, nullptr, nullptr,
                  want_talker ? &prompt_hidden_all : nullptr)) {
        ggml_free(g.ctx);
        return false;
    }
    ggml_free(g.ctx);
    if (getenv("QOMNI_LOGITS_DUMP")) {
        dump_bins(getenv("QOMNI_LOGITS_DUMP"), nullptr, 0, logits.data(), logits.size());
    }
    n_past = T;
    next_pos = 0;
    for (int i = 0; i < T; i++)
        next_pos = std::max({ next_pos, pos[(size_t)i] + 1, pos[(size_t)T + i] + 1,
                              pos[(size_t)2 * T + i] + 1 });

    std::vector<int> out;
    std::vector<std::vector<float>> gen_hidden, gen_embed;
    int cur = vqa::sample_logits(logits, gp, rng_state);
    auto emit = [&](int t) {
        std::string piece = tok_.decode({ t });
        out.push_back(t);
        if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
    };
    if (cur != id_eos) emit(cur);

    for (int i = 1; i < gp.max_tokens && cur != id_eos; i++) {
        const int S = n_past + 1;
        TextGraph d;
        if (!build_text_graph(d, 1, S, 0, 0, want_talker)) { ggml_free(d.ctx); return false; }
        vqa::set_causal_mask(d.mask_in, n_past, 1);
        std::vector<int> p1((size_t)4, next_pos);
        std::vector<float> lg, hid, emb;
        if (!run_text(d, { cur }, p1, nullptr, nullptr, nullptr, nullptr, lg, n_past,
                      want_talker ? &hid : nullptr, want_talker ? &emb : nullptr)) {
            ggml_free(d.ctx);
            return false;
        }
        ggml_free(d.ctx);
        n_past++;
        next_pos++;
        if (want_talker) {
            gen_hidden.push_back(std::move(hid));
            gen_embed.push_back(std::move(emb));
        }
        cur = vqa::sample_logits(lg, gp, rng_state);
        if (cur == id_eos) break;
        emit(cur);
    }
    if (want_talker) {
        // HF feeds the talker the prompt's post-norm hidden plus the plain
        // token embedding, with the image/audio rows of the embed track zeroed
        std::vector<float> prompt_embed((size_t)n_embd * T, 0.0f);
        if (prompt_hidden_all.size() == (size_t)n_embd * T) {
            for (int i = 0; i < T; i++) {
                if (toks[(size_t)i] == id_image_pad || toks[(size_t)i] == id_audio) continue;
                read_emb_row(tok_embd, toks[(size_t)i],
                             prompt_embed.data() + (size_t)n_embd * i);
            }
        }
        synthesize_speech(toks, pos, prompt_hidden_all, prompt_embed, out,
                          gen_hidden, gen_embed, next_pos, gp.speak_wav);
    }
    out_text = tok_.decode(out);
    return true;
}

bool Qwen25Omni::synthesize_speech(const std::vector<int> & toks, const std::vector<int> & pos,
                                   const std::vector<float> & prompt_hidden,
                                   const std::vector<float> & prompt_embed,
                                   const std::vector<int> & gen_toks,
                                   const std::vector<std::vector<float>> & gen_hidden,
                                   const std::vector<std::vector<float>> & gen_embed,
                                   int next_pos, const char * wav_path) {
    (void)gen_toks;
    // thinker embedding row of the speaker bos token
    std::vector<float> bos_emb((size_t)n_embd);
    read_emb_row(tok_embd, spk_bos_token, bos_emb.data());
    if (!tallocr) tallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    talker.allocr = tallocr;
    // t2w keeps its own gallocr (made in Token2Wav::load): sharing the pool
    // with the talker graphs made the allocator move t2w buffers between
    // graph computations, which forced the per-eval mask re-uploads and is
    // incompatible with the static DiT / BigVGAN graphs.
    float t_temp = 0.9f; int t_topk = 40; float t_topp = 0.8f, t_rep = 1.05f;
    if (const char * e = getenv("QOMNI_TALK_TEMP")) t_temp = atof(e);
    if (const char * e = getenv("QOMNI_TALK_TOPK")) t_topk = atoi(e);
    if (const char * e = getenv("QOMNI_TALK_TOPP")) t_topp = atof(e);
    if (const char * e = getenv("QOMNI_TALK_REP")) t_rep = atof(e);
    std::vector<int> codec;
    const auto t_tk0 = std::chrono::steady_clock::now();
    if (!talker.synthesize(toks, pos, prompt_hidden, prompt_embed, std::vector<int>(),
                           next_pos, gen_toks, gen_hidden, gen_embed, tok_embd, id_eos,
                           151859, bos_emb, t_temp, t_topk, t_topp, t_rep, rng_state, codec)) {
        fprintf(stderr, "talker: synthesis failed\n");
        return false;
    }
    if (const char * dp = getenv("QOMNI_CODEC_DUMP")) {
        FILE * f = fopen(dp, "wb");
        if (f) { int n = (int)codec.size(); fwrite(&n, 4, 1, f);
                 fwrite(codec.data(), 4, codec.size(), f); fclose(f); }
    }
    if (const char * lp = getenv("QOMNI_CODEC_LOAD")) {   // parity: run t2w on
        FILE * f = fopen(lp, "rb");                        // a reference codec stream
        if (f) {
            int n = 0;
            if (fread(&n, 4, 1, f) == 1 && n > 0 && n < 100000) {
                codec.resize(n);
                fread(codec.data(), 4, n, f);
            }
            fclose(f);
            fprintf(stderr, "t2w: loaded %d reference codec tokens\n", (int)codec.size());
        }
    }
    const auto t_tk1 = std::chrono::steady_clock::now();
    std::vector<float> pcm;
    if (!t2w.run(codec, pcm)) {
        fprintf(stderr, "token2wav: synthesis failed\n");
        return false;
    }
    const auto t_t2w1 = std::chrono::steady_clock::now();
    fprintf(stderr, "timing: talker %.2f s | t2w %.2f s (%d codes)\n",
            std::chrono::duration<double>(t_tk1 - t_tk0).count(),
            std::chrono::duration<double>(t_t2w1 - t_tk1).count(), (int)codec.size());
    // write a 24 kHz mono s16 wav
    FILE * f = fopen(wav_path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", wav_path); return false; }
    const uint32_t rate = 24000, n = (uint32_t)pcm.size();
    const uint16_t ch = 1, bits = 16;
    const uint32_t bytes = n * 2;
    fwrite("RIFF", 1, 4, f);
    uint32_t v32 = 36 + bytes; fwrite(&v32, 4, 1, f);
    fwrite("WAVEfmt ", 1, 8, f);
    v32 = 16; fwrite(&v32, 4, 1, f);
    uint16_t v16 = 1; fwrite(&v16, 2, 1, f);
    fwrite(&ch, 2, 1, f); fwrite(&rate, 4, 1, f);
    v32 = rate * 2; fwrite(&v32, 4, 1, f);
    v16 = 2; fwrite(&v16, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&bytes, 4, 1, f);
    for (uint32_t i = 0; i < n; i++) {
        float s = std::min(1.0f, std::max(-1.0f, pcm[i]));
        int16_t q = (int16_t)lrintf(s * 32767.0f);
        fwrite(&q, 2, 1, f);
    }
    fclose(f);
    fprintf(stderr, "speech: %s (%.2f s @ 24 kHz)\n", wav_path, n / 24000.0);
    return true;
}

}  // namespace qwen25omni

namespace vqa {
Model * create_qwen25omni() { return new qwen25omni::Qwen25Omni(); }
}  // namespace vqa
