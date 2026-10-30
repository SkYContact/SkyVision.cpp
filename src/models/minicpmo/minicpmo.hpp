// minicpmo.hpp - the MiniCPM-o 4.5 backend surface (openbmb/MiniCPM-o-4_5).
//
// One header shared by the backend (minicpmo_model.cpp + the tower/ talker
// .cpp files) and the synthetic parity driver (tools/minicpmo_seg_test.cpp),
// so the branch builders have exactly one contract.
//
// Architecture truth (ref/modeling_minicpmo.py + config.json; the normative
// table is D:/S2TT/minicpmo45-port/arch-spec.md):
//
//   LLM     plain Qwen3-8B dense: 36L x 4096, 32q/8kv GQA, head_dim 128,
//           NEOX rope theta 1e6, per-head q/k RMSNorm, untied lm_head.
//   vision  SigLIP-SO400M NaFiT 27L x 1152 @448px, patch 14, gelu tanh.
//           Tiles: MiniCPM-V LPU 9-slice (max_slice_nums 9, scale 448),
//           thumbnail first + slices row-major. Patch order row-major
//           (h outer, w inner); position ids are BUCKETIZED fractional
//           coords into the 980px 70x70 training grid (SiglipVisionEmbeddi
//           mgs.forward, modeling_navit_siglip.py:349-364): for patch row i
//           of h: bucket = ceil(70*i/h) - 1 (integer exact, boundary
//           inclusive right=True), id = bucket_h*70 + bucket_w.
//   rs.     64-query perceiver resampler: kv_proj 1152->4096, ln_kv, keys =
//           kv + runtime 2D sincos (embed first half from the column grid,
//           second half from the row grid), fused in_proj q|k|v row blocks,
//           unmasked cross-attn 32 heads, out_proj, ln_post, x @ proj
//           (RIGHT multiply - bare Parameter). Per tile: 64 tokens.
//   audio   whisper-medium encoder 24L x 1024, 80 mel (whisper 16k/n_fft
//           400/hop 160, log10 -> max-8dB clamp -> (x+4)/4). conv k3 s1 p1
//           -> gelu(erf) -> conv k3 s2 p1 -> gelu -> pos 0..T2-1 -> post-LN
//           blocks (x += attn(LN(x)); x += fc2(gelu(fc1(LN(x)))), k_proj
//           WITHOUT bias) -> ln_post -> fc1/ReLU/fc2 -> AvgPool1d(5).
//           Attention: 1s chunk mask (audio_chunk_length 1.0 -> 50-frame
//           chunks after the stride-2 conv), query q sees keys
//           [0, min((q/50+1)*50, T2)) - cumulative, not plain blockdiag.
//   tk.     TTS sub-tower: LlamaModel 768 x 20L (12q/12kv, head_dim 64, no
//           biases, no q/k norm, no rope metadata -> HF LlamaConfig defaults
//           theta 5e5 / eps 1e-6), emb_text [152064, 768] independent table,
//           emb_code [6562, 768] single codebook (num_vq=1), head_code
//           768->6562 (converter recombined the weight_norm), condition =
//           emb_text(gen text) + L2(proj_semantic(llm last-layer hidden))
//           then emb_text(text_eos) + emb_text(audio_bos) appended; AR loop
//           emb_code(prev) at position cond_len + t - 1; eos = 6561.
//           The token2wav vocoder (CosyVoice2 flow+hift) is a later wave.
//
// GGUF tensor contract: convert/convert_minicpmo45.py (arch key "minicpmo")
// + the vocoder sibling GGUF from convert_minicpmo45_vocoder.py (arch key
// "token2wav", namespaces t2w./cfm./dit./hifi./spk.).
#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "core/engine/attention.hpp"    // vqa::KvCache (talker)

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace vqa { class MMap; }

namespace minicpmo {

// ------------------------------------------------------------------ host
// MiniCPM-V LPU slice grid (processing_minicpmo.py get_sliced_grid). Returns
// false when the image must not be sliced (grid null -> single upscaled tile).
bool slice_grid(int width, int height, int max_slice_nums, int scale_resolution,
                int * out_cols, int * out_rows);

// find_best_resize + ensure_divide (processing_minicpmo.py:148-160).
void best_resize(int width, int height, int scale_resolution, int patch_size,
                 bool allow_upscale, int * out_w, int * out_h);

// get_refine_size (processing_minicpmo.py:162-176) - the per-slice resize the
// patch grid is cut from.
void refine_size(int width, int height, int grid_cols, int grid_rows,
                 int scale_resolution, int patch_size, int * out_w, int * out_h);

// SiglipVisionEmbeddings bucketized position ids for one (h_patches,
// w_patches) tile: row-major ids into the 70x70 grid.
void vision_pos_ids(int h_patches, int w_patches, int grid_side,
                    std::vector<int> & ids);

// get_2d_sincos_pos_embed (modeling_minicpmo.py:3643-3691): row-major
// (h outer, w inner) [h*w, dim]; first half of dim encodes the COLUMN grid,
// second half the ROW grid; omega = 1/10000^(2i/dim).
void sincos_pos_2d(int h, int w, int dim, std::vector<float> & out);

// Subsequent-chunk visibility (modeling_minicpmo.py:372-402 + 676-684):
// query q attends keys [0, min((q/chunk + 1)*chunk, size)). F16 mask rows
// of `size` (column-major rows like set_causal_mask: mask[q*size + k]).
void set_chunk_mask(ggml_tensor * dst, int chunk);

// ------------------------------------------------------------------ LLM
// Qwen3 dense layer weights (ggml mul_mat orientation: ne[0] = IN).
struct LlmAttnW {
    ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr, * o = nullptr;
    ggml_tensor * qn = nullptr, * kn = nullptr;   // per-head RMSNorm [head_dim]
};
struct LlmFfnW {
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
};

// Qwen3 attention: GQA + per-head q/k norm + NEOX rope + causal FA over an
// F16 KV cache + o proj. Returns the [D, T] branch output (residual is the
// caller's). l = the post attention-norm stream [D, T].
ggml_tensor * llm_attn_branch(ggml_context * ctx, ggml_cgraph * gf,
                              const LlmAttnW & w, ggml_tensor * k_cache,
                              ggml_tensor * v_cache, ggml_tensor * l,
                              ggml_tensor * mask, ggml_tensor * pos,
                              int n_head, int n_kv_head, int head_dim,
                              float rope_theta, int n_ctx, float rms_eps,
                              const char * dbg);

// SwiGLU MLP on the post ffn-norm stream.
ggml_tensor * llm_ffn_branch(ggml_context * ctx, const LlmFfnW & w,
                             ggml_tensor * l, float rms_eps, const char * dbg);

// TTS condition ("hidden_text_merge", modeling_minicpmo.py:1308-1319):
// emb_text(tok_in) + L2(proj_semantic(hid_in)) rowwise. All tensors graph
// inputs except the weights. Returns [768, T].
ggml_tensor * tts_condition(ggml_context * ctx, ggml_tensor * emb_text,
                            ggml_tensor * tok_in, ggml_tensor * hid_in,
                            ggml_tensor * fc1, ggml_tensor * fc1_b,
                            ggml_tensor * fc2, ggml_tensor * fc2_b);

// ------------------------------------------------------------------ vision
struct SiglipLayer {
    ggml_tensor * ln1_w = nullptr, * ln1_b = nullptr;
    ggml_tensor * wq = nullptr, * bq = nullptr;
    ggml_tensor * wk = nullptr, * bk = nullptr;
    ggml_tensor * wv = nullptr, * bv = nullptr;
    ggml_tensor * wo = nullptr, * bo = nullptr;
    ggml_tensor * ln2_w = nullptr, * ln2_b = nullptr;
    ggml_tensor * fc1_w = nullptr, * fc1_b = nullptr;
    ggml_tensor * fc2_w = nullptr, * fc2_b = nullptr;
};

struct ResamplerW {
    ggml_tensor * query = nullptr;         // [4096, 64] F32
    ggml_tensor * kv_proj = nullptr;       // [1152, 4096]
    ggml_tensor * in_proj = nullptr;       // [4096, 12288] (q|k|v row blocks)
    ggml_tensor * in_proj_b = nullptr;     // [12288] F32
    ggml_tensor * out_proj = nullptr;      // [4096, 4096]
    ggml_tensor * out_proj_b = nullptr;    // [4096] F32
    // LayerNorm pairs, all F32 (eps 1e-6 in the graph)
    ggml_tensor * ln_q_w = nullptr, * ln_q_b = nullptr;
    ggml_tensor * ln_kv_w = nullptr, * ln_kv_b = nullptr;
    ggml_tensor * ln_post_w = nullptr, * ln_post_b = nullptr;
    ggml_tensor * proj = nullptr;          // [4096, 4096] RIGHT multiply; the
                                           // gguf bytes are the torch [in,out]
                                           // parameter o-major (ne0 = out) -
                                           // snapshot source only, the graph
                                           // multiplies with proj_f32 below
    ggml_tensor * proj_f32 = nullptr;      // [in, out] F32 contiguous replica
                                           // (mul_mat src0, K = in on ne0);
                                           // built once by VisionTower::encode
};

// One resampler pass over one tile: x = post- SigLIP hidden [1152, n],
// pos = the tile's 2D sincos [4096, n] F32 (dim-fast like every graph
// tensor). Returns [4096, 64] (dim-fast; 64 query tokens).
ggml_tensor * resampler_branch(ggml_context * ctx, const ResamplerW & w,
                               ggml_tensor * x, ggml_tensor * pos,
                               int n_head, const char * dbg);

class VisionTower {
public:
    int n_layer = 27, n_embd = 1152, n_head = 16, n_ff = 4304, patch = 14;
    int grid_side = 70;                    // pos_embd rows = 70*70
    int query_num = 64, rs_embed = 4096, rs_heads = 32;
    float ln_eps = 1e-6f;
    float img_mean[3] = {0.5f, 0.5f, 0.5f}, img_std[3] = {0.5f, 0.5f, 0.5f};

    ggml_tensor * patch_w = nullptr, * patch_b = nullptr, * pos_embd = nullptr;
    ggml_tensor * post_ln_w = nullptr, * post_ln_b = nullptr;
    ResamplerW rs;
    std::vector<SiglipLayer> vl;

    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;
    const char * dump_dir = "";
    int n_tok = 0;                         // 64 * n_tiles of the last encode
    // F32 [in, out] replica of rs.proj for the in-graph right multiply
    // (the raw F16/quantized weight mis-routes the CPU GEMM - see
    // resampler_branch; the old workaround was a single-thread host loop)
    ggml_context * rs_proj_ctx = nullptr;
    ggml_backend_buffer_t rs_proj_buf = nullptr;
    ggml_tensor * rs_proj_in = nullptr;    // [in, out] F32 contiguous

    // host front end: RGB8 -> tiled patch vectors + per-tile geometry.
    // tiles: one [588, n_patches] F32 block per tile (patch vectors packed
    // (c,y,x)); pos: per-tile bucketized position ids; sincos: per-tile
    // [4096, n] tables.
    struct TilePre {
        int h_p = 0, w_p = 0;              // patch grid
        std::vector<float> patches;        // [588, h_p*w_p]
        std::vector<int> pos_ids;          // [h_p*w_p]
        std::vector<float> sincos;         // [4096, h_p*w_p]
    };
    bool preprocess(const uint8_t * rgb, int width, int height,
                    std::vector<TilePre> & out);

    // graph: run all tiles (each its own tower pass + resampler), emit
    // [4096, 64*n_tiles] F32 token-major embedding.
    bool encode(std::vector<TilePre> & tiles, std::vector<float> & feats);
};

// ------------------------------------------------------------------ audio
struct AudLayer {
    ggml_tensor * attn_ln_w = nullptr, * attn_ln_b = nullptr;
    ggml_tensor * wq = nullptr, * bq = nullptr;
    ggml_tensor * wk = nullptr;            // whisper: k_proj has NO bias
    ggml_tensor * wv = nullptr, * bv = nullptr;
    ggml_tensor * wo = nullptr, * bo = nullptr;
    ggml_tensor * ffn_ln_w = nullptr, * ffn_ln_b = nullptr;
    ggml_tensor * fc1_w = nullptr, * fc1_b = nullptr;
    ggml_tensor * fc2_w = nullptr, * fc2_b = nullptr;
};

class AudioTower {
public:
    int n_layer = 24, d_model = 1024, n_head = 16, n_mel = 80;
    int n_pos = 1500;                      // max_source_positions
    int pool_step = 5;
    float chunk_s = 1.0f;                  // audio_chunk_length
    float ln_eps = 1e-5f;                  // WhisperConfig norm_epsilon default

    ggml_tensor * conv1_w = nullptr, * conv1_b = nullptr;
    ggml_tensor * conv2_w = nullptr, * conv2_b = nullptr;
    ggml_tensor * pos_embd = nullptr;      // [1024, 1500] F32
    ggml_tensor * ln_post_w = nullptr, * ln_post_b = nullptr;
    ggml_tensor * proj_fc1 = nullptr, * proj_fc1_b = nullptr;
    ggml_tensor * proj_fc2 = nullptr, * proj_fc2_b = nullptr;
    std::vector<AudLayer> al;
    std::vector<float> mel_filters_host;   // [80*201] sidecar (else runtime HTK)

    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;
    const char * dump_dir = "";

    bool load_mel_filters();               // sidecar next to the GGUF, else runtime
    // log-mel: whisper framing + log10 + max-8dB dynamic clamp + (x+4)/4.
    bool mel(const float * pcm, int n_samples, std::vector<float> & out, int * n_len);

    // one tower pass (one <=30s chunk): feats [4096, n_pool] F32 tokens.
    bool encode_chunk(const float * mel80, int n_len, std::vector<float> & feats,
                      int * n_pool);
    // full clip: 30s chunking (processing chunk_audio), concatenated tokens.
    bool encode(const float * pcm, int n_samples, std::vector<float> & feats,
                int * n_tokens);
};

// ------------------------------------------------------------------ talker
struct TkLayer {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr, * o = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
};

class Talker {
public:
    int n_layer = 20, n_embd = 768, n_head = 12, n_kv_head = 12, head_dim = 64;
    int n_text_vocab = 152064, n_audio_vocab = 6562;
    int audio_bos = 151687, text_eos = 151692;
    float rms_eps = 1e-6f;
    float rope_theta = 10000.0f;           // step-logits gate: 1e4 matches the
                                           // official trunk 20x closer than the
                                           // LlamaConfig-default 5e5

    ggml_tensor * emb_text = nullptr;      // [768, 152064]
    ggml_tensor * emb_code = nullptr;      // [768, 6562]
    ggml_tensor * head_code = nullptr;     // [768, 6562] (weight_norm recombined)
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * proj_sem_fc1 = nullptr, * proj_sem_fc1_b = nullptr;
    ggml_tensor * proj_sem_fc2 = nullptr, * proj_sem_fc2_b = nullptr;
    std::vector<TkLayer> tl;

    // runtime
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;
    vqa::KvCache * kv = nullptr;           // caller-owned (20 layers)
    const char * dump_dir = "";

    // L2(proj_semantic(hidden)) + emb_text(tokens) + eos/bos condition, then
    // the AR loop (see synthesize_cond). hidden: [4096] per generated-text
    // token (row-major); tokens: the LLM text ids inside the tts_bound.
    bool synthesize(const std::vector<int> & tokens,
                    const std::vector<float> & hidden,
                    std::vector<int> & codes, int max_new_token = 2048);
    // AR code generation from a prebuilt condition matrix [T, 768] row-major
    // (token rows; text_eos/audio_bos appended in-graph). Returns the code
    // ids WITHOUT the eos.
    bool synthesize_cond(const std::vector<float> & cond,
                         std::vector<int> & codes, int max_new_token = 2048);
};

// ---------------------------------------------------------------- vocoder
// CosyVoice2 token2wav (openbmb MiniCPM-o 4.5 voice), pure ggml: CAM++
// speaker encoder (campplus.onnx weights), the flow model (UpsampleConformer
// EncoderV2 + DiT-16 CFM estimator, euler x10 with CFG 0.7) and the HiFT
// vocoder (ConvRNNF0Predictor + ConvTranspose stack + Snake resblocks +
// NSF sine source + ISTFT head). The sine source, STFT and ISTFT run as host
// DSP (deterministic math, no weights); every learned layer is a ggml graph.
// Weights come from the minicpmo-vocoder.gguf sibling (arch "token2wav").
class Vocoder {
public:
    int n_mel = 80, codebook = 6561;
    int flow_d = 512, flow_heads = 8, flow_hd = 64;
    int n_cfm = 6, n_cfm_up = 4, n_dit = 16, dit_d = 512;
    float cfg_rate = 0.7f;
    int n_timesteps = 10;
    float dit_eps = 1e-6f, dit_qk_eps = 1e-5f, conf_ln_eps = 1e-12f,
          embed_ln_eps = 1e-5f;
    int n_prompt_tokens = 0, n_prompt_mels = 0, n_fbank = 0, n_rand = 0;

    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;

    // t2w.* prompt cache
    ggml_tensor * rand_noise = nullptr;        // [80, N] F32 CFM buffer
    ggml_tensor * prompt_tokens = nullptr;     // [n_pt] I32
    ggml_tensor * prompt_mels = nullptr;       // [n_pm, 80] F32 rows
    ggml_tensor * spk_fbank = nullptr;         // [n_f, 80] F32 rows
    ggml_tensor * spk_emb_ref = nullptr;       // [192] F32 (parity ref only)

    // cfm.* flow encoder
    ggml_tensor * input_emb = nullptr;
    ggml_tensor * spk_proj_w = nullptr, * spk_proj_b = nullptr;
    ggml_tensor * embed_w = nullptr, * embed_b = nullptr;
    ggml_tensor * embed_ln_w = nullptr, * embed_ln_b = nullptr;
    ggml_tensor * la_conv1_w = nullptr, * la_conv1_b = nullptr;
    ggml_tensor * la_conv2_w = nullptr, * la_conv2_b = nullptr;
    struct ConfLayer {
        ggml_tensor * ln_mha_w = nullptr, * ln_mha_b = nullptr;
        ggml_tensor * ln_ff_w = nullptr, * ln_ff_b = nullptr;
        ggml_tensor * wq = nullptr, * bq = nullptr;
        ggml_tensor * wk = nullptr, * bk = nullptr;
        ggml_tensor * wv = nullptr, * bv = nullptr;
        ggml_tensor * wo = nullptr, * bo = nullptr;
        ggml_tensor * wpos = nullptr;
        ggml_tensor * pos_u = nullptr, * pos_v = nullptr;
        ggml_tensor * ffn_gate_w = nullptr, * ffn_gate_b = nullptr;
        ggml_tensor * ffn_down_w = nullptr, * ffn_down_b = nullptr;
    };
    std::vector<ConfLayer> cl, cul;
    ggml_tensor * up_conv_w = nullptr, * up_conv_b = nullptr;
    ggml_tensor * up_embed_w = nullptr, * up_embed_b = nullptr;
    ggml_tensor * up_embed_ln_w = nullptr, * up_embed_ln_b = nullptr;
    ggml_tensor * after_norm_w = nullptr, * after_norm_b = nullptr;
    ggml_tensor * proj_w = nullptr, * proj_b = nullptr;

    // dit.* CFM estimator
    ggml_tensor * t_mlp0_w = nullptr, * t_mlp0_b = nullptr;
    ggml_tensor * t_mlp1_w = nullptr, * t_mlp1_b = nullptr;
    ggml_tensor * in_proj_w = nullptr, * in_proj_b = nullptr;
    struct DiTLayer {
        ggml_tensor * wq = nullptr, * bq = nullptr;
        ggml_tensor * wk = nullptr, * bk = nullptr;
        ggml_tensor * wv = nullptr, * bv = nullptr;
        ggml_tensor * wo = nullptr, * bo = nullptr;
        ggml_tensor * qn_w = nullptr, * qn_b = nullptr;
        ggml_tensor * kn_w = nullptr, * kn_b = nullptr;
        ggml_tensor * mlp_gate_w = nullptr, * mlp_gate_b = nullptr;
        ggml_tensor * mlp_down_w = nullptr, * mlp_down_b = nullptr;
        ggml_tensor * conv1_w = nullptr, * conv1_b = nullptr;
        ggml_tensor * conv_ln_w = nullptr, * conv_ln_b = nullptr;
        ggml_tensor * conv2_w = nullptr, * conv2_b = nullptr;
        ggml_tensor * adaln_w = nullptr, * adaln_b = nullptr;
    };
    std::vector<DiTLayer> dl;
    ggml_tensor * fin_adaln_w = nullptr, * fin_adaln_b = nullptr;
    ggml_tensor * fin_out_w = nullptr, * fin_out_b = nullptr;

    // hifi.* HiFT generator
    ggml_tensor * f0_c_w[5] = {}, * f0_c_b[5] = {};
    ggml_tensor * f0_head_w = nullptr, * f0_head_b = nullptr;
    ggml_tensor * src_lin_w = nullptr, * src_lin_b = nullptr;
    ggml_tensor * conv_pre_w = nullptr, * conv_pre_b = nullptr;
    struct Conv1 { ggml_tensor * w = nullptr, * b = nullptr; int k = 0;
        // per-tap reordered kernel [K*IC, OC] rows (k, ic) for strided convs
        ggml_tensor * w_ph = nullptr; };
    struct Up { ggml_tensor * w = nullptr, * b = nullptr; int k = 0, s = 0, p = 0; };
    Up ups[3];
    struct ResBlock {
        Conv1 c1[3], c2[3];
        ggml_tensor * a1[3] = {}, * a2[3] = {};
        int k[3] = {}, dil[3] = {1, 3, 5};
    };
    ResBlock rb[9];
    Conv1 sdown[3];
    ResBlock srb[3];
    ggml_tensor * conv_post_w = nullptr, * conv_post_b = nullptr;

    // spk.* campplus
    struct SpkConv2d { ggml_tensor * w = nullptr, * b = nullptr; };
    SpkConv2d head_conv1, l10c1, l10c2, l10sc, l11c1, l11c2,
              l20c1, l20c2, l20sc, l21c1, l21c2, head_conv2, tdnn;
    struct Tdnnd {
        // entry BN precomputed as x*iv + bb (iv = w/sqrt(v+eps),
        // bb = b - m*iv); tdnnd1 of each block has one too (block1
        // normalizes the relu'd tdnn output, blocks 2/3 the transit output)
        ggml_tensor * bn_iv = nullptr, * bn_bb = nullptr;
        ggml_tensor * linear1_w = nullptr, * linear1_b = nullptr;
        ggml_tensor * cam_ll_w = nullptr;
        ggml_tensor * cam1_w = nullptr, * cam1_b = nullptr;
        ggml_tensor * cam2_w = nullptr, * cam2_b = nullptr;
    };
    Tdnnd tdnnd[3][24];
    int n_tdnnd[3] = {12, 24, 16};
    struct Transit { SpkConv2d conv; ggml_tensor * bn_iv = nullptr, * bn_bb = nullptr; };
    Transit transit[3];
    ggml_tensor * dense_w = nullptr, * dense_b = nullptr;

    // filled by run() for the parity drivers
    std::vector<float> out_spk, out_mel, out_f0;

    bool load(ggml_backend_t backend_, const char * path);
    // codes (WITHOUT eos) -> 24 kHz mono PCM. run() is deterministic given
    // the GGUF (the CFM noise is the baked rand_noise buffer, exactly the
    // official module buffer); the HiFT sine RNG draws are engine-side
    // unless MINICPMO_VOC_RNG_DIR points at the parity dumps.
    bool run(const std::vector<int> & codes, std::vector<float> & pcm);

    bool loaded() const;

private:
    struct impl;
    impl * p = nullptr;
};

}  // namespace minicpmo
