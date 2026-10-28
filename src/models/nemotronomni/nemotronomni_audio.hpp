// nemotronomni_audio.hpp - Parakeet/FastConformer audio tower for the
// Nemotron-3-Omni backend.
//
// Architecture truth (ref/ processing + transformers parakeet 5.9/5.21;
// oracle parity: nemotron3omni-port/oracle/dump_repo_audio):
//   mel        librosa slaney filterbank (computed, NOT the checkpoint's
//              BF16 fb - the oracle mel gate is the HF-extractor dump),
//              torch STFT semantics: preemph 0.97, center=True constant pad,
//              hann periodic=False (win 400), power spectrum, n_fft 512,
//              hop 160 -> T = 1 + L//160 frames, log(x + 2^-24).
//   CMVN       per utterance over the L//160 valid frames (var ddof=1,
//              eps 1e-5), padding frames zeroed.
//   conv front 3 stride-2 stages (8x time subsample): Conv2d(1->256,3x3)+ReLU,
//              [depthwise 3x3 + pointwise 1x1 + ReLU] x2, all with bias, then
//              flatten(256x16) -> Linear[1024, 4096]. After EVERY Conv2d
//              (pointwise included) the frames >= cur_len are re-zeroed
//              (cur_len folds (L-1)//2+1 per stride-2 conv).
//   encoder    24 Conformer blocks d1024: macaron FFN x0.5, Transformer-XL
//              relative-pos attention (bias_u/v, rel_shift implemented as the
//              exact skew view bd[t, j] = qv[t] . rel[(j - t) + T - 1], fed to
//              flash-attn as a per-head additive F16 mask), conv module
//              (pw1 1024->2048, GLU, depthwise k=9 + BN(folded), silu, pw2),
//              FFN2 x0.5, norm_out. inv_freq table rebuilt: 512 entries
//              base 10000, positions T-1..-(T-1), sin/cos interleaved.
//   projector  RMSNorm(1024, 1e-5) -> Linear[4096, 1024] -> relu^2 ->
//              Linear[2688, 4096] -> [T', 2688] (<so_embedding> contents).
#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <string>
#include <vector>

namespace nemo {

struct ParakeetBlk {
    ggml_tensor *ln_ff1_w = nullptr, *ln_ff1_b = nullptr;
    ggml_tensor *ff1_w1 = nullptr, *ff1_w2 = nullptr;      // [1024, 4096] / [4096, 1024]
    ggml_tensor *ln_att_w = nullptr, *ln_att_b = nullptr;
    ggml_tensor *att_q = nullptr, *att_k = nullptr, *att_v = nullptr, *att_o = nullptr;
    ggml_tensor *att_rel_k = nullptr;                      // [1024, 1024], no bias
    ggml_tensor *bias_u = nullptr, *bias_v = nullptr;      // [128, 8] F32
    ggml_tensor *ln_conv_w = nullptr, *ln_conv_b = nullptr;
    ggml_tensor *conv_pw1 = nullptr;                       // [1024, 2048], no bias
    ggml_tensor *conv_dw_raw = nullptr;                    // [9, 1024] F16 (gguf)
    ggml_tensor *bn_w = nullptr, *bn_b = nullptr;          // BN stats [1024] F32
    ggml_tensor *bn_mean = nullptr, *bn_var = nullptr;
    ggml_tensor *conv_dw_fold = nullptr;                   // [9, 1024] F32, BN folded
    ggml_tensor *conv_dw_bias = nullptr;                   // [1024] F32, BN folded beta
    ggml_tensor *conv_pw2 = nullptr;                       // [1024, 1024], no bias
    ggml_tensor *ln_ff2_w = nullptr, *ln_ff2_b = nullptr;
    ggml_tensor *ff2_w1 = nullptr, *ff2_w2 = nullptr;
    ggml_tensor *ln_out_w = nullptr, *ln_out_b = nullptr;
};

struct ParakeetTower {
    int n_layer = 24, d_model = 1024, n_head = 8, ffn = 4096, conv_k = 9;
    int n_mel = 128, sub_ch = 256, freq_final = 16;
    int llm_dim = 2688;
    float bn_eps = 1e-5f;

    // bound ggml weights
    std::vector<ParakeetBlk> al;
    ggml_tensor *proj_ln = nullptr;    // a.proj.ln  [1024] F32
    ggml_tensor *proj_fc1 = nullptr;   // a.proj.fc1 [1024, 4096]
    ggml_tensor *proj_fc2 = nullptr;   // a.proj.fc2 [4096, 2688]

    // subsampling conv host weights (F32 copies read from the bound tensors)
    std::vector<float> s0_w, s0_b;     // conv 1->256   w[co][kh][kw]
    std::vector<float> s1_dw, s1_dwb;  // dw 3x3 s2
    std::vector<float> s1_pw, s1_pwb;  // pw 1x1        w[co][ci]
    std::vector<float> s2_dw, s2_dwb;
    std::vector<float> s2_pw, s2_pwb;
    ggml_tensor *sub_proj_w = nullptr; // [4096, 1024] (in-graph)
    ggml_tensor *sub_proj_b = nullptr; // [1024] F32

    // derived-tensor context: BN-folded depthwise weights/biases per layer
    ggml_context * bctx = nullptr;
    ggml_backend_buffer_t bbuf = nullptr;

    // graph
    ggml_context * actx = nullptr;
    ggml_cgraph * agf = nullptr;
    ggml_gallocr_t allocr = nullptr;   // borrowed from the model
    ggml_backend_t backend = nullptr;  // borrowed from the model
    ggml_tensor *a_in = nullptr, *a_pos_in = nullptr, *a_feats = nullptr;
    ggml_tensor *a_mmask = nullptr, *a_rmask = nullptr;   // HF pad masks (finfo.min fill / row zero)
    int a_t_valid = -1;                // HF n_out_valid (frames after subsampling that are real)
    int a_t_valid_built = -1;
    int a_t = -1;                      // built frame count
    std::string dump_dir;
    std::vector<const char *> sink_names;  // dump names, set by build_graph
    int n_sink = 0;

    ~ParakeetTower();
    void free_graph();

    // BN eval semantics folded into the depthwise conv weights once at load.
    bool fold_bn(ggml_backend_t backend);

    // wav -> mel+CMVN -> conv frontend. Host side; fills `sub` [T', 1024]
    // (frame-major), `pos` [2T'-1, 1024], and returns T' (*mel_frames = the
    // pre-conv mel frame count).
    int frontend(const float * pcm, int n_samples, std::vector<float> & mel,
                 std::vector<float> & sub, std::vector<float> & pos,
                 int * mel_frames = nullptr);

    bool build_graph(int T);
    bool encode(ggml_backend_t backend, const float * pcm, int n_samples,
                std::vector<float> & embeds, int * n_tok, const char * dump);
};

}  // namespace nemo
