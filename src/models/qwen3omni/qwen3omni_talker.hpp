// qwen3omni_talker.hpp - Qwen3-Omni speech-output stack (talker + RQ code
// predictor + code2wav vocoder), loaded from the separate
// qwen3omni-talker-q8_0.gguf (qwen25omni two-file precedent).
//
// Protocol (official generate, transcribed from transformers 5.20.1):
//   thinker trajectory (embedding + layer-24 hidden per token)
//     -> text_projection / hidden_projection (ResizeMLP 2048->2048->1024)
//     -> talker 20-layer MoE trunk (Qwen3 attention + IMROPE 24/20/20, top-6
//        routed experts + sigmoid-gated shared expert) -> codec_head
//     -> first code token (vocab 3072)
//     -> code predictor: 5-layer dense trunk, per-group embedding tables and
//        heads (15 groups); steps 1..15 autoregressive
//     -> per step residual_codes [16] -> (after generation) code2wav:
//        mean of 16 group embeds -> 8-layer transformer (sliding window 72)
//        -> transconv/ConvNeXt upsample x4 -> snake/transconv decoder x(8,5,4)
//        -> 24 kHz wav. Output length is trimmed by each decoder transconv
//        (kernel=2r, stride=r): N codes -> N*1920 - (8+5+4) samples.
//
// HEAD TYING (settled 2026-10-26): the checkpoint stores DIFFERENT values for
// codec_head vs codec_embedding (and lm_head.i vs codec_embedding.i). HF's
// tie_weights explicitly does NOT tie when both sides exist with different
// values, so the heads are INDEPENDENT weights - the engine binds
// tk.codec_head / cp.head.{k} and uses them for the output projection.
#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdint>
#include <string>
#include <map>
#include <vector>

namespace vqa { class MMap; }

namespace qwen3omni {
extern int dump_c2w_mid;

struct TkLayer {
    ggml_tensor *norm = nullptr, *post_norm = nullptr;
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    ggml_tensor *qn = nullptr, *kn = nullptr;
    ggml_tensor *router = nullptr;
    ggml_tensor *gate_exps = nullptr, *up_exps = nullptr, *down_exps = nullptr;
    ggml_tensor *sh_gate = nullptr, *sh_up = nullptr, *sh_down = nullptr;
    ggml_tensor *sh_gate_scalar = nullptr;   // [1024] row of a 1-in linear
};

struct CpLayer {
    ggml_tensor *norm = nullptr, *post_norm = nullptr;
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    ggml_tensor *qn = nullptr, *kn = nullptr;
    ggml_tensor *gate = nullptr, *up = nullptr, *down = nullptr;
};

struct C2wLayer {
    ggml_tensor *attn_norm = nullptr, *ffn_norm = nullptr;
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr;
    ggml_tensor *o_w = nullptr, *o_b = nullptr;
    ggml_tensor *attn_scale = nullptr, *mlp_scale = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    ggml_tensor *up_w = nullptr;
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
};

struct C2wConv {   // host-side conv weight [Co, Ci, K] F32 + bias
    std::vector<float> w, b;
    int co = 0, ci = 0, k = 0, dilation = 1, stride = 1, groups = 1;
    bool has_bias = true;
};

class Talker {
public:
    int n_layer = 20, n_embd = 1024, n_head = 16, n_kv_head = 2, head_dim = 128;
    int n_expert = 128, n_expert_used = 6, n_ff_exp = 384, n_ff_shared = 768;
    int n_vocab = 3072;                       // codec token space
    float rms_eps = 1e-6f, rope_theta = 1e6f;
    int mrope_sections[4] = {24, 20, 20, 0};
    // thinker-side projection (ResizeMLP): fc1 2048->2048 SiLU, fc2 ->1024
    ggml_tensor *tp_fc1_w = nullptr, *tp_fc1_b = nullptr;
    ggml_tensor *tp_fc2_w = nullptr, *tp_fc2_b = nullptr;
    ggml_tensor *hp_fc1_w = nullptr, *hp_fc1_b = nullptr;
    ggml_tensor *hp_fc2_w = nullptr, *hp_fc2_b = nullptr;

    ggml_tensor *token_embd = nullptr;        // [1024, 3072] codec embedding
    ggml_tensor *codec_head = nullptr;       // [1024, 3072] INDEPENDENT head
    ggml_tensor *cp_head[15] = {};           // [1024, 2048] per-group heads
    ggml_tensor *out_norm = nullptr;
    std::vector<TkLayer> tl;

    // predictor
    int cp_n_layer = 5, cp_n_head = 16, cp_n_kv_head = 8, cp_head_dim = 128;
    int cp_n_ff = 3072, cp_n_vocab = 2048, n_groups = 16;
    float cp_rms_eps = 1e-6f;
    ggml_tensor *cp_emb[15] = {};             // group embedding tables
    ggml_tensor *cp_out_norm = nullptr;
    std::vector<CpLayer> cl;

    // code2wav
    int c2w_n_layer = 8, c2w_hidden = 1024, c2w_window = 72;
    int c2w_decoder_dim = 1536, c2w_codebook = 2048;
    ggml_tensor *c2w_emb = nullptr;           // [1024, 32768]
    std::vector<C2wLayer> c2l;
    ggml_tensor *c2w_norm = nullptr;
    // upsample (2 x [transconv k2s2 + ConvNeXt]) - host F32 copies
    struct ConvNeXt {
        C2wConv dw;                            // groups=dim
        std::vector<float> ln_w, ln_b;         // [dim]
        ggml_tensor *pw1_w = nullptr, *pw1_b = nullptr, *pw2_w = nullptr, *pw2_b = nullptr;
        std::vector<float> gamma;              // [dim]
    };
    C2wConv up_tc[2];
    ConvNeXt up_cnx[2];
    // decoder tree (host): entry conv, 3 blocks (snake+transconv+3 resunits), tail snake+conv
    struct ResUnit { C2wConv c1, c2; std::vector<float> a1_alpha, a1_beta, a2_alpha, a2_beta; };
    struct DecBlock { std::vector<float> s_alpha, s_beta; C2wConv tc; ResUnit ru[3]; };
    C2wConv dec_head;                          // 1024 -> 1536, k7
    DecBlock dec[4];
    std::vector<float> tail_alpha, tail_beta;
    C2wConv dec_tail;                          // 96 -> 1, k7

    // special ids
    int id_codec_eos = 2150, id_codec_bos = 2149, id_codec_pad = 2148;
    int id_codec_nothink = 2155, id_think_bos = 2156, id_think_eos = 2157;
    int id_tts_bos = 151672, id_tts_eos = 151673, id_tts_pad = 151671;
    int id_speaker = 2302;                     // ethan
    std::map<std::string, int> speaker_ids;    // from gguf metadata
    int accept_hidden_layer = 24;

    // runtime (backend/allocr borrowed from the thinker model)
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    vqa::MMap * mm = nullptr;
    void * host_f32 = nullptr;   // AlignedBuf owned by the cpp (opaque)
    bool loaded = false;
    const char * dump_dir = "";

    void run_c2_kernel(const std::vector<float> & in, std::vector<float> & out);
    bool load(const char * path, ggml_backend_t backend_, ggml_gallocr_t allocr_);
};

struct SynthIn {
    const std::vector<int> * ids = nullptr;
    const std::vector<int> * gen_ids = nullptr;
    const std::vector<float> * traj_emb = nullptr;
    const std::vector<float> * traj_h24 = nullptr;
    ggml_tensor * thinker_embd = nullptr;
    int thinker_embd_dim = 2048;
    uint64_t * rng = nullptr;
    int max_codes = 400;
};

bool synthesize(Talker * t, const SynthIn & in, std::vector<float> & wav);

// oracle parity probes (driven by tools/qwen3omni_pred_test.cpp):
// cp: prefill [2,1024] embeds then 14 greedy steps, heads 0..14; returns
//     15 rows of logits [15,2048] and last hidden [15,1024]
bool cp_parity(Talker * t, const float * emb2, std::vector<float> & logits15,
               std::vector<float> & hidden15);
// tk: one trunk forward of T rows at mpos (4T ints: t/h/w rows + kv slots);
//     stateful across calls (shared KvCache), mirrors prefill/step sequence
bool tk_parity(Talker * t, const float * emb, int T, const int * mpos,
               std::vector<float> & logits_last, std::vector<float> & hidden_all);
bool code2wav(Talker * t, const std::vector<std::vector<float>> & codes16, std::vector<float> & wav);

}  // namespace qwen3omni
