// qwen25omni_talker.hpp - speech synthesis side of Qwen2.5-Omni:
//   TalkerLM  - 24-layer dual-track LM that turns the thinker's reply into
//               25 Hz audio codec tokens;
//   Token2Wav - flow-matching DiT (codec tokens -> mel) + BigVGAN (mel -> PCM).
// Weights come from the qwen25omni-talker-*.gguf sibling of the thinker GGUF.
#pragma once

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace qwen25omni {

// read one row (dst[0..ne0)) of a weight tensor, decoding f16 or quantized
// weights (q8_0/q4_0 rows decoded via the type traits; rows are block-aligned)
inline void read_emb_row(ggml_tensor * t, int row, float * dst) {
    const size_t n = (size_t)t->ne[0];
    if (row < 0 || (size_t)row >= (size_t)t->ne[1]) {
        fprintf(stderr, "read_emb_row OOB: %s row %d / %lld\n",
                t->name ? t->name : "?", row, (long long)t->ne[1]);
        row = 0;
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, dst, (size_t)row * t->nb[1], n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> raw(n);
        ggml_backend_tensor_get(t, raw.data(), (size_t)row * t->nb[1], n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; i++) dst[i] = ggml_fp16_to_fp32(raw[i]);
    } else {
        const size_t bs = t->nb[1];
        std::vector<uint8_t> raw(bs);
        ggml_backend_tensor_get(t, raw.data(), (size_t)row * bs, bs);
        const struct ggml_type_traits * tt = ggml_get_type_traits(t->type);
        tt->to_float(raw.data(), dst, n);
    }
}

struct TkLayer {
    ggml_tensor *norm = nullptr;
    ggml_tensor *wq = nullptr, *bq = nullptr;
    ggml_tensor *wk = nullptr, *bk = nullptr;
    ggml_tensor *wv = nullptr, *bv = nullptr;
    ggml_tensor *wo = nullptr;
    ggml_tensor *ffn_norm = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
};

// DiT transformer block (flow matching over mel frames)
struct DiTLayer {
    ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;   // AdaLN linear (dim*6)
    ggml_tensor *wq = nullptr, *bq = nullptr;
    ggml_tensor *wk = nullptr, *bk = nullptr;
    ggml_tensor *wv = nullptr, *bv = nullptr;
    ggml_tensor *wo = nullptr, *bo = nullptr;
    ggml_tensor *w_gate = nullptr, *b_gate = nullptr;             // ff.ff.0
    ggml_tensor *w_down = nullptr, *b_down = nullptr;             // ff.ff.3
};

class TalkerLM {
public:
    bool load(ggml_backend_t backend, struct ggml_context * wctx);   // tensors already bound

    // Synthesize codec tokens for one finished thinker turn. The thinker
    // embed table is borrowed for the text bos/eos/pad embeddings.
    //   prompt_toks / prompt_pos4 / mm_rows: the full prompt, its MRoPE
    //     positions and the rows whose embedding the talker must see as zero
    //   next_pos: the next free MRoPE index after the prompt
    //   prompt_hidden / prompt_embed: per prompt token, post-norm hidden and
    //     plain token embedding (zeros at image/audio rows, HF masks those)
    //   gen_hidden / gen_embed: per generated token, post-norm hidden + embedding
    //   thinker_eos / thinker_pad: token ids whose thinker embeddings close the
    //     reply stream
    bool synthesize(const std::vector<int> & prompt_toks, const std::vector<int> & prompt_pos4,
                    const std::vector<float> & prompt_hidden,
                    const std::vector<float> & prompt_embed,
                    const std::vector<int> & mm_rows, int next_pos,
                    const std::vector<int> & gen_toks,
                    const std::vector<std::vector<float>> & gen_hidden,
                    const std::vector<std::vector<float>> & gen_embed,
                    ggml_tensor * thinker_embed, int thinker_eos, int thinker_pad,
                    const std::vector<float> & spk_bos_embed,   // thinker embed of the speaker bos
                    float temperature, int top_k, float top_p, float rep_penalty,
                    uint64_t & rng, std::vector<int> & codec_out);

    int codec_vocab() const { return n_vocab; }
    int codec_eos() const { return id_codec_end; }
    int codec_pad() const { return id_codec_pad; }

    // hyperparams
    int n_layer = 24, n_embd = 896, embw = 2048, n_head = 14, n_kv_head = 2, head_dim = 64;
    int n_ff = 4864, n_vocab = 8448;
    float eps = 1e-6f, rope_theta = 1e6f;
    int mrope_section[3] = {16, 16, 0};
    int id_codec_start = 8293, id_codec_end = 8294, id_codec_pad = 8292, id_codec_mask = 8296;

    ggml_tensor *embed = nullptr;         // [embw, n_vocab]
    ggml_tensor *out_norm = nullptr;
    ggml_tensor *to_proj_w = nullptr, *to_proj_b = nullptr;
    ggml_tensor *codec_head = nullptr;    // [n_embd, n_vocab]
    std::vector<TkLayer> layers;

    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;

private:
    bool build_prefill(int T, int S);
    bool build_step(int S);

    ggml_context * tctx = nullptr;
    ggml_cgraph * tgf = nullptr;
    ggml_tensor *g_emb_in = nullptr, *g_codec_in = nullptr, *g_pos_in = nullptr;
    ggml_tensor *g_codec_emb = nullptr;   // host-fed codec embedding track [embw, T]
    ggml_tensor *g_slot_in = nullptr, *g_mask_in = nullptr, *g_reply_in = nullptr;
    ggml_tensor *g_logits = nullptr;

    struct Kv { ggml_tensor *k = nullptr, *v = nullptr; };
    ggml_context * kvctx = nullptr;
    ggml_backend_buffer_t kvbuf = nullptr;
    std::vector<Kv> kv;
    int n_past = 0;
    bool step_ready = false;   // static decode graph built & allocated
};

class Token2Wav {
public:
    bool load(ggml_backend_t backend, struct ggml_context * wctx);
    // codec tokens (25 / s) -> 24 kHz mono float PCM in [-1, 1]
    bool run(const std::vector<int> & codec, std::vector<float> & pcm);

    int d_depth = 22, d_dim = 1024, d_emb = 512, d_head = 16, d_hd = 64;
    int d_mel = 80, d_embeds = 8193, d_repeats = 2, d_block = 24;
    int enc_dim = 128;      // ECAPA output width
    int enc_emb_dim = 192;  // baked speaker cond width
    std::vector<int> look_ahead_layers = {10};
    std::vector<int> look_backward_layers = {0, 20};
    int num_steps = 10;
    float guidance = 0.5f, sway = -1.0f;

    ggml_tensor *time_mlp0_w = nullptr, *time_mlp0_b = nullptr;
    ggml_tensor *time_mlp1_w = nullptr, *time_mlp1_b = nullptr;
    ggml_tensor *codec_embd = nullptr;    // [d_emb, d_embeds+1]
    ggml_tensor *in_proj_w = nullptr, *in_proj_b = nullptr;   // [d_dim, 80+192+192+512]
    ggml_tensor *inv_freq = nullptr;      // [d_hd/2]
    std::vector<DiTLayer> layers;
    ggml_tensor *norm_out_w = nullptr, *norm_out_b = nullptr; // AdaLN linear (dim*2)
    ggml_tensor *proj_out_w = nullptr, *proj_out_b = nullptr;
    // ECAPA / CampPlus speaker encoder over the reference mel
    ggml_tensor *spk_tdnn0_w = nullptr, *spk_tdnn0_b = nullptr;
    struct SpkBlock {
        ggml_tensor *tdnn1_w = nullptr, *tdnn1_b = nullptr;
        std::vector<ggml_tensor *> res2net_w, res2net_b;      // scale-1 convs
        ggml_tensor *tdnn2_w = nullptr, *tdnn2_b = nullptr;
        ggml_tensor *se1_w = nullptr, *se1_b = nullptr;
        ggml_tensor *se2_w = nullptr, *se2_b = nullptr;
    };
    std::vector<SpkBlock> spk_blocks;
    ggml_tensor *spk_mfa_w = nullptr, *spk_mfa_b = nullptr;
    ggml_tensor *spk_asp_tdnn_w = nullptr, *spk_asp_tdnn_b = nullptr;
    ggml_tensor *spk_asp_conv_w = nullptr, *spk_asp_conv_b = nullptr;
    ggml_tensor *spk_fc_w = nullptr, *spk_fc_b = nullptr;
    std::vector<float> spk_cond;      // baked speaker cond [192]
    std::vector<float> spk_ref_mel;   // baked reference mel [T x 80]

    // BigVGAN (plain C++ host code - see qwen25omni_t2w.cpp)
    struct Conv1D { std::vector<float> w; std::vector<float> b; int oc = 0, ic = 0, k = 0, dil = 1; };
    struct ResBlock { std::vector<Conv1D> c1, c2; std::vector<std::vector<float>> s1a, s1b, s2a, s2b; };
    std::vector<float> bg_pre_w, bg_pre_b;
    struct UpLayer { std::vector<float> w; std::vector<float> b; int k = 0, stride = 0, oc = 0; };
    std::vector<UpLayer> ups;
    std::vector<ResBlock> rbs;
    std::vector<float> snake_pre_a, snake_pre_b;
    std::vector<float> post_w;   // no bias
    int b_ch = 1536, b_mel = 80;

    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;

private:
    // DiT ggml graph (one flow-matching evaluation, CFG batched)
    bool build_dit_graph(int T);
    bool ecapa_embed(const float * mel, int T, std::vector<float> & out);
    bool bigvgan_run(const std::vector<float> & mel, int L, std::vector<float> & pcm);
    // BigVGAN as a ggml graph on the same backend (T2W_HOST=1 keeps the host
    // path above as an oracle / fallback)
    bool build_bg_graph(int L0);
    bool bigvgan_run_ggml(const std::vector<float> & mel, int L0, std::vector<float> & pcm);
    bool dit_eval(const float * x, int T, float t, const float * codec_emb, bool drop_code,
                  const float * spk, const float * cond_vec, std::vector<float> & out);

    ggml_context * gctx = nullptr;
    ggml_cgraph * ggf = nullptr;
    // dedicated ctx + buffer for the per-run-constant graph inputs (masks,
    // pos, codec/spk/cond tracks): gallocr shares pool memory between
    // non-overlapping-liveness tensors, so mid-graph inputs must NOT live in
    // the pool (they'd be clobbered between evals)
    ggml_context * ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    ggml_tensor *g_x_in = nullptr, *g_t_in = nullptr, *g_code_emb = nullptr;
    ggml_tensor *g_spk_in = nullptr, *g_cond_in = nullptr, *g_out = nullptr;
    std::vector<ggml_tensor *> g_masks;   // per-layer block-diff masks
    std::vector<ggml_tensor *> g_sinks;   // debug: hidden after blocks 0/1/2
    ggml_tensor *g_pos = nullptr;

    // speaker embedding of the reference mel, cached (the mel never changes)
    std::vector<float> cond_cached;
    bool cond_ready = false;
    bool dit_masks_ready = false;   // block masks uploaded for the current graph
    bool dit_inputs_ready = false;  // constant codec/spk/cond tracks uploaded
    std::vector<std::vector<ggml_fp16_t>> dit_mask_cache;   // built once per graph
    std::vector<int> dit_pos_cache;

    // BigVGAN ggml graph state (own ctx + own gallocr so the pool never moves
    // between graphs; CUDA graph cache keys on cgraph->nodes[0], so bctx /
    // bg_allocr must outlive every compute of bgf)
    struct ggml_context * wctx_ref = nullptr;   // weight ctx (model lifetime)
    ggml_context * bctx = nullptr;
    ggml_cgraph * bgf = nullptr;
    ggml_gallocr_t bg_allocr = nullptr;
    ggml_context * bg_ictx = nullptr;           // graph-input constants
    ggml_backend_buffer_t bg_ibuf = nullptr;

    ggml_tensor * bg_mel_in = nullptr;   // [L0, 80] planar (ne0 = time)
    ggml_tensor * bg_out = nullptr;      // [L] clamped pcm
    int bg_L_cached = -1;
    std::vector<std::pair<ggml_tensor *, std::vector<char>>> bg_in_uploads;
    std::vector<ggml_tensor *> bg_dbg;   // T2W_BG_DBG: per-stage dup sinks
};

}  // namespace qwen25omni
