// vc3_model.cpp - VideoChat3-4B backend (Qwen3 text decoder + VideoChat3 ViT).
//
// Conventions:
//   * GGUF linear weights are stored as ne = [in, out]; consumption is
//     ggml_mul_mat(W, x[in, T]) -> [out, T].
//   * the text decoder is fed a host-assembled F32 embedding matrix [n_embd, T]
//     (token rows read back from the possibly quantized token_embd, image rows
//     spliced in), so a single graph shape covers text and image prompts.
//   * text attention uses a per-layer F16 KV cache [n_kv_head*head_dim, n_ctx]
//     with flash attention and an explicit F16 causal mask.
//   * the vision tower runs as its own graph per clip: patch embed, 27 blocks
//     with 2D rope and full (non causal) attention, final LN, merge, projector.
#include "core/vqa.hpp"
#include "core/blocks/vit_block.hpp"
#include "core/blocks/llm_block.hpp"

#include <functional>
#include <array>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#   include "ggml-cuda.h"   // ggml_backend_cuda_init
#endif
#include "core/text/tokenizer.hpp"
#include "core/engine/backend.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/gguf_util.hpp"
#include "core/runtime/load_util.hpp"
#include "core/engine/weight_load.hpp"
#include "core/engine/stage_buf.hpp"
#include "vc3_preprocess.hpp"
#include "core/engine/attention.hpp"
#include "core/engine/sampler.hpp"
#include "core/util/aligned_buf.hpp"
#include "core/util/mmap.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace vc3 {

// ---------------------------------------------------------------- structures
struct VisLayer {
    ggml_tensor *norm0_w = nullptr, *norm0_b = nullptr;
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *wqkv_w = nullptr, *wqkv_b = nullptr;
    ggml_tensor *wo_w = nullptr, *wo_b = nullptr;
    ggml_tensor *fc0_w = nullptr, *fc0_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
};

struct TextGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor *tok_in = nullptr, *pos_in = nullptr, *mask_in = nullptr;
    ggml_tensor *img_in = nullptr, *img_pos_in = nullptr;
    ggml_tensor *logits = nullptr;
};

// batch attention group: (col_start, col_count, kv_base, window)
typedef std::array<int, 4> BatGroup;

class VideoChat3 : public vqa::Model {
public:
    ~VideoChat3() override;

    bool load(const std::string & gguf_path, const std::string & tokenizer_path,
              const vqa::LoadOptions & opt = vqa::LoadOptions()) override;
    const vqa::Capabilities & caps() const override { return caps_; }
    const char * name() const override { return "videochat3-4b"; }
    bool ask(const vqa::MediaInput & media, const std::string & prompt,
             const std::string & mode, const vqa::GenParams & gp, std::string & out_text) override;
    // ChatML template knowledge for the server's multi-turn sessions (moss
    // pattern): gp.history is prepended verbatim before the new user turn in
    // ask(), so the front end stays template-agnostic
    std::string format_history_turn(const std::string & user,
                                    const std::string & assistant) const override {
        return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
             + assistant + "<|im_end|>\n";
    }

    // tunables, set before load()
    int n_ctx = 8192;
    bool use_gpu = false;
    int n_threads = 0;
    std::string dump_dir;

private:
    bool init_backend();
    bool load_weights(const char * path);
    bool alloc_kv();
    // clip_marks: per temporal clip (timestamp, token count) for prompt building
    // token_budget > 0 clamps max_pixels so the merged vision token count fits the
    // text context; images that cannot fit even at minimum resolution fail here,
    // before the expensive vision encode (instead of the late run_text check)
    bool encode_vision(const vqa::MediaInput & media, std::vector<float> & feats, int * n_tokens,
                       std::vector<std::pair<float,int>> * clip_marks = nullptr, int token_budget = 0);
    bool build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all = false,
                          const std::vector<BatGroup> * bat_groups = nullptr);
    bool run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<float> * img_feats,
                  const std::vector<int> * img_pos, std::vector<float> & logits_out);
    std::string render_prompt(int n_img_tokens, const std::string & user_text, bool video,
                               const std::vector<std::pair<float,int>> * clip_marks = nullptr) const;
    int sample(const std::vector<float> & logits, const vqa::GenParams & gp);
    float randf();
    void dump(const ggml_cgraph * gf, const char * name);

    vqa::Capabilities caps_;
    vqa::Tokenizer tok_;

    int n_layer = 36, n_embd = 2560, n_head = 32, n_kv_head = 8, head_dim = 128;
    int n_ff = 9728, n_vocab = 151936;
    float rms_eps = 1e-6f, rope_theta = 5e6f;
    VisConfig vc;

    ggml_backend_t backend = nullptr;
    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;         // weights: mmap-backed (CPU) or uploaded
    ggml_backend_buffer_t wbuf_plain = nullptr;   // host f32 block holding the 1D tensors
    ggml_backend_buffer_t wbuf_fold = nullptr;    // host block holding LN-folded vision weights (CPU path)
    vqa::AlignedBuf host_f32_storage;           // keeps that block alive on the CPU path
    std::vector<char> fold_storage;               // keeps the folded weights alive (both paths)
    bool fold_bias = false;                       // LN bias absorbed into the matmul bias (exact, F32)
    bool fold_scale = false;                      // LN gain folded into the weight (VC3_FOLD_W=1; re-rounds F16, lossy)
    bool wo_padded = false;                       // wo input rows padded to 80/head (exact; kills the FA de-pad copy)
    void (*stage_cb)(const char *, int, int, void *) = nullptr;   // per-clip vision progress
    void * stage_ud = nullptr;
    std::vector<float> patch_bias;                // v.patch_embd.bias, folded into the host pos table
    vqa::StageBuffer tower_buf;          // device-side tower output, host never touched between clips
    vqa::MMap mm;

    std::vector<float> pos_emb_host, pos_time_host;

    ggml_tensor *tok_embd = nullptr, *out_norm = nullptr;
    std::vector<vqa::llm::LayerWeights> tl;   // core/llm_block.hpp
    std::vector<VisLayer> vl;
    ggml_tensor *v_patch_w = nullptr, *v_patch_b = nullptr;
    ggml_tensor *v_final_w = nullptr, *v_final_b = nullptr;
    ggml_tensor *mm_pre_w = nullptr, *mm_pre_b = nullptr;
    ggml_tensor *mm_l1_w = nullptr, *mm_l1_b = nullptr;
    ggml_tensor *mm_l2_w = nullptr, *mm_l2_b = nullptr;

    vqa::KvCache kv;
    // batch decode (vc3_batch.inc)
    int bat_n = 0;
    ggml_context * bat_ctx = nullptr;
    ggml_cgraph * bat_gf = nullptr;
    ggml_tensor * bat_tok = nullptr, * bat_pos = nullptr, * bat_mask = nullptr;
    int bat_win = 0;
    std::vector<BatGroup> bat_groups;
    int step_batch(const std::vector<int> & toks, const std::vector<int> & posv,
                   const std::vector<int> & lo, const std::vector<int> & hi,
                   const std::vector<BatGroup> & groups, int win, float * logits_out);
    int run_batch(const std::string & path, int n_gen);
    // static decode graph (T=1, fixed KV window per block): same gf object across
    // steps -> eligible for ggml CUDA graph capture (see moondream Engine::step)
    ggml_context * dec_ctx = nullptr;
    ggml_cgraph * dec_gf = nullptr;
    int dec_blk = 0;
    ggml_tensor * dec_tok = nullptr, * dec_pos = nullptr, * dec_mask = nullptr;
    // the decode graph gets its OWN gallocr: a shared one re-reserves (and
    // frees) the buffer under dec_tok/dec_pos/dec_mask whenever a later turn's
    // prefill graph outgrows it, leaving these cached input pointers dangling
    // (multi-turn crash: heap AV in the decode step from turn 2 on - same bug
    // moss_vl hit, fixed there in 09c944a)
    ggml_gallocr_t dec_allocr = nullptr;
    ggml_gallocr_t allocr = nullptr;
    int n_past = 0;

    int id_image_pad = 151655, id_video_pad = 151656;
    int id_vision_start = 151652, id_vision_end = 151653;
    int id_im_start = 151644, id_im_end = 151645, id_eos = 151645, id_endoftext = 151643;

    uint64_t rng_state = 0x9E3779B97F4A7C15ull;
};

// ---------------------------------------------------------------- helpers


// per-pair complex rotation on an interleaved head vector: pair p = (v[2p], v[2p+1]).
// cos_t/sin_t are [1, pairs, H, N] - pre-expanded on the host to re/im's exact shape,
// so every elementwise op is same-shape. ore/oim land in a preallocated interleaved
// tensor via two cpy ops - a dim0 concat here explodes into thousands of tiny kernel
// launches under WDDM.
// per-pair complex rotation on an interleaved head vector: pair p = (v[2p], v[2p+1]).
// cos_t/sin_t are [1, pairs, H*N] - pre-expanded on the host to re/im's exact shape.
// Everything is kept 3d: the CUDA concat launcher loops over ne[3], so a 4d layout
// with ne3=N explodes into thousands of tiny kernel launches under WDDM.
static ggml_tensor * rope_2d(ggml_context * ctx, struct ggml_cgraph * gf, ggml_tensor * x,
                             ggml_tensor * cos_t, ggml_tensor * sin_t) {
    const int64_t hd = x->ne[0], H = x->ne[1], N = x->ne[2];
    const int64_t pairs = hd / 2, M = H * N;
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, 2, pairs, M);
    ggml_tensor * re = ggml_cont(ctx, ggml_view_3d(ctx, x3, 1, pairs, M, x3->nb[1], x3->nb[2], 0));
    ggml_tensor * im = ggml_cont(ctx, ggml_view_3d(ctx, x3, 1, pairs, M, x3->nb[1], x3->nb[2], x3->nb[0]));
    ggml_tensor * c = ggml_reshape_3d(ctx, cos_t, 1, pairs, M);
    ggml_tensor * sn = ggml_reshape_3d(ctx, sin_t, 1, pairs, M);
    ggml_tensor * ore = ggml_sub(ctx, ggml_mul(ctx, re, c), ggml_mul(ctx, im, sn));
    ggml_tensor * oim = ggml_add(ctx, ggml_mul(ctx, re, sn), ggml_mul(ctx, im, c));
    return ggml_reshape_3d(ctx, ggml_concat(ctx, ore, oim, 0), hd, H, N);
}

float VideoChat3::randf() {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (float)(((rng_state * 0x2545F4914F6CDD1Dull) >> 11) * (1.0 / 9007199254740992.0));
}

VideoChat3::~VideoChat3() {
    if (allocr) ggml_gallocr_free(allocr);
    if (dec_allocr) ggml_gallocr_free(dec_allocr);
    if (bat_ctx) ggml_free(bat_ctx);
    if (dec_ctx) ggml_free(dec_ctx);
    kv.free();
    if (wbuf) ggml_backend_buffer_free(wbuf);
    if (wbuf_plain) ggml_backend_buffer_free(wbuf_plain);
    if (wbuf_fold) ggml_backend_buffer_free(wbuf_fold);
    if (wctx) ggml_free(wctx);
    if (backend) ggml_backend_free(backend);
}

bool VideoChat3::init_backend() {
    backend = vqa::init_backend(use_gpu, n_threads);
    if (!backend) return false;
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!dec_allocr) dec_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return allocr != nullptr && dec_allocr != nullptr;
}

// ---------------------------------------------------------------- load
bool VideoChat3::load(const std::string & gguf_path, const std::string & tokenizer_path,
                      const vqa::LoadOptions & opt) {
    n_ctx = vqa::resolve_n_ctx(opt, n_ctx);
    vqa::apply_load_options(opt, "VC3", n_threads, use_gpu, dump_dir);
    if (!init_backend()) return false;
    if (!load_weights(gguf_path.c_str())) return false;
    if (!tok_.load(tokenizer_path.c_str(), vqa::TokenizerMode::Qwen2Regex)) return false;
    if (!alloc_kv()) return false;

    vqa::assign_token_ids(tok_, {
        { "<|image_pad|>",    &id_image_pad },
        { "<|video_pad|>",    &id_video_pad },
        { "<|vision_start|>", &id_vision_start },
        { "<|vision_end|>",   &id_vision_end },
        { "<|im_start|>",     &id_im_start },
        { "<|im_end|>",       &id_im_end },
    });
    id_eos = id_im_end;

    caps_ = {};
    caps_.image = true;
    caps_.video = true;
    caps_.proactive_states = true;
    caps_.timestamps_prompt = true;
    fprintf(stderr, "videochat3: ready (%d text layers, %d vision layers, ctx %d)\n",
            n_layer, vc.n_layer, n_ctx);
    return true;
}

bool VideoChat3::load_weights(const char * path) {
    if (!mm.load(path)) { fprintf(stderr, "mmap failed: %s\n", path); return false; }
    fprintf(stderr, "model file: %.2f GB\n", mm.size / 1e9);

    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ &wctx };
    struct gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "gguf init failed\n"); return false; }

    n_layer   = vqa::gguf::u32(gu, "videochat3.block_count", n_layer);
    n_embd    = vqa::gguf::u32(gu, "videochat3.embedding_length", n_embd);
    n_head    = vqa::gguf::u32(gu, "videochat3.attention.head_count", n_head);
    n_kv_head = vqa::gguf::u32(gu, "videochat3.attention.head_count_kv", n_kv_head);
    head_dim  = vqa::gguf::u32(gu, "videochat3.attention.key_length", head_dim);
    n_ff      = vqa::gguf::u32(gu, "videochat3.feed_forward_length", n_ff);
    n_vocab   = vqa::gguf::u32(gu, "videochat3.vocab_size", n_vocab);
    rms_eps   = vqa::gguf::f32(gu, "videochat3.attention.layer_norm_rms_epsilon", rms_eps);
    rope_theta = vqa::gguf::f32(gu, "videochat3.rope.freq_base", rope_theta);

    vc.n_layer = vqa::gguf::u32(gu, "videochat3.vision.block_count", vc.n_layer);
    vc.n_embd  = vqa::gguf::u32(gu, "videochat3.vision.embedding_length", vc.n_embd);
    vc.n_head  = vqa::gguf::u32(gu, "videochat3.vision.attention.head_count", vc.n_head);
    vc.n_ff    = vqa::gguf::u32(gu, "videochat3.vision.feed_forward_length", vc.n_ff);
    vc.patch   = vqa::gguf::u32(gu, "videochat3.vision.patch_size", vc.patch);
    vc.merge   = vqa::gguf::u32(gu, "videochat3.vision.merge_size", vc.merge);
    vc.temporal_merge = vqa::gguf::u32(gu, "videochat3.vision.temporal_merge_size", vc.temporal_merge);
    vc.pos_h   = vqa::gguf::u32(gu, "videochat3.vision.pos_emb_height", vc.pos_h);
    vc.pos_w   = vqa::gguf::u32(gu, "videochat3.vision.pos_emb_width", vc.pos_w);
    vc.layernorm_eps = vqa::gguf::f32(gu, "videochat3.vision.layer_norm_epsilon", vc.layernorm_eps);
    vc.rope_theta = vqa::gguf::f32(gu, "videochat3.vision.rope_theta", vc.rope_theta);
    if (vqa::gguf::u32(gu, "videochat3.vision.rope_layout", 0) != 2) {
        fprintf(stderr, "this build expects a rope_layout=2 GGUF (wqkv rows permuted + zero-padded for the fused vision rope); re-run convert/convert_videochat3.py\n");
        return false;
    }

    vqa::weights::bind_from_mmap(gu, wctx, mm);
    gguf_free(gu);

    // 1D tensors (norms, biases) -> host F32: ggml elementwise ops need matching types
    vqa::AlignedBuf host_f32;
    vqa::weights::promote_1d_to_f32(wctx, host_f32);

    // --- vision LayerNorm affine folding (opt-in) -----------------------------
    //   W·(norm(x)*w + b) + B  ==  W·(norm(x)*w) + (W·b + B)   [bias fold]
    //                            ==  (W·diag(w))·norm(x) + (W·b + B)  [+ scale fold]
    // Neither variant is exact under ggml's f16 GEMM: mul_mat converts the
    // activation to the weight's vec_dot_type (F16), so moving arithmetic across
    // the GEMM boundary changes the rounding path - measured v_final drift ~1%
    // (mean) over 27 layers and greedy divergence. Off by default; enable with
    // VC3_FOLD_BIAS=1 / VC3_FOLD_W=1 to trade that precision for ~1/3 fewer
    // elementwise kernels in the vision graph.
    {
        struct FoldRef { std::string w, b, nw, nb; };
        std::vector<FoldRef> folds;
        for (int i = 0; i < vc.n_layer; i++) {
            const std::string p = "v.blk." + std::to_string(i) + ".";
            folds.push_back({p + "wqkv.weight",      p + "wqkv.bias",      p + "norm0.weight", p + "norm0.bias"});
            folds.push_back({p + "mlp.fc0.weight",   p + "mlp.fc0.bias",   p + "norm1.weight", p + "norm1.bias"});
        }
        folds.push_back({"mm.linear_1.weight", "mm.linear_1.bias", "mm.pre_norm.weight", "mm.pre_norm.bias"});

        const bool want_bias = getenv("VC3_FOLD_BIAS") != nullptr;
        const bool want_scale = getenv("VC3_FOLD_W") != nullptr;
        bool ok = want_bias || want_scale;
        if (ok) {
            size_t store_bytes = 0;
            for (auto & f : folds) {
                ggml_tensor * W = ggml_get_tensor(wctx, f.w.c_str());
                ggml_tensor * B = ggml_get_tensor(wctx, f.b.c_str());
                if (!W || !B || !ggml_get_tensor(wctx, f.nw.c_str()) || !ggml_get_tensor(wctx, f.nb.c_str()))
                    { ok = false; break; }
                store_bytes += (size_t)ggml_nbytes(B);
                if (want_scale) store_bytes += (size_t)ggml_nbytes(W);
            }
            if (ok) {
                fold_storage.resize(store_bytes);
                size_t off = 0;
                for (auto & f : folds) {
                    ggml_tensor * W  = ggml_get_tensor(wctx, f.w.c_str());
                    ggml_tensor * B  = ggml_get_tensor(wctx, f.b.c_str());
                    ggml_tensor * NW = ggml_get_tensor(wctx, f.nw.c_str());
                    ggml_tensor * NB = ggml_get_tensor(wctx, f.nb.c_str());
                    const int64_t in = W->ne[0], out = W->ne[1];
                    const float * w = (const float *)NW->data;    // F32 after the 1D pass
                    const float * b = (const float *)NB->data;
                    float * bnew = (float *)(fold_storage.data() + off);
                    off += (size_t)ggml_nbytes(B);
                    char * wnew = nullptr;
                    if (want_scale) { wnew = fold_storage.data() + off; off += (size_t)ggml_nbytes(W); }
                    // one pass per column: W·b for the new bias (+ optional
                    // w[i]·W[i,o]; input-dim scaling is orthogonal to the
                    // rope_layout=2 row permutation on the output side)
                    if (W->type == GGML_TYPE_F16) {
                        const ggml_fp16_t * src = (const ggml_fp16_t *)W->data;
                        ggml_fp16_t * dst = (ggml_fp16_t *)wnew;
                        for (int64_t o = 0; o < out; o++) {
                            const ggml_fp16_t * col = src + (size_t)o * in;
                            ggml_fp16_t * dcol = dst ? dst + (size_t)o * in : nullptr;
                            float acc = 0.0f;
                            for (int64_t i = 0; i < in; i++) {
                                const float v = ggml_fp16_to_fp32(col[i]);
                                acc += v * b[i];
                                if (dcol) dcol[i] = ggml_fp32_to_fp16(v * w[i]);
                            }
                            bnew[o] = acc + ((const float *)B->data)[o];
                        }
                    } else if (W->type == GGML_TYPE_F32) {
                        const float * src = (const float *)W->data;
                        float * dst = (float *)wnew;
                        for (int64_t o = 0; o < out; o++) {
                            const float * col = src + (size_t)o * in;
                            float * dcol = dst ? dst + (size_t)o * in : nullptr;
                            float acc = 0.0f;
                            for (int64_t i = 0; i < in; i++) {
                                acc += col[i] * b[i];
                                if (dcol) dcol[i] = col[i] * w[i];
                            }
                            bnew[o] = acc + ((const float *)B->data)[o];
                        }
                    } else { ok = false; break; }
                    B->data = bnew;
                    if (wnew) W->data = wnew;
                }
                if (!ok) fold_storage.clear();
            }
            if (ok) {
                fold_bias = true;
                fold_scale = want_scale;
                fprintf(stderr, "vision LN %sfolded into wqkv/fc0/mm_l1 biases%s (%.1f MB host block)\n",
                        want_scale ? "affines " : "bias ",
                        want_scale ? " + gains" : "",
                        (double)fold_storage.size() / 1e6);
            }
        }
        // patch bias -> host pos table: host-side F32, exact, always on
        {
            ggml_tensor * pb = ggml_get_tensor(wctx, "v.patch_embd.bias");
            if (pb && pb->type == GGML_TYPE_F32 && pb->ne[0] == vc.n_embd && pb->ne[1] == 1)
                patch_bias.assign((const float *)pb->data, (const float *)pb->data + vc.n_embd);
        }
        // wo input rows: 72 -> 80 per head with zero rows. FA's pad columns are
        // exact zeros (the converter zero-pads the wqkv rows and bias), so the
        // extra dot-product terms are exact zeros — bitwise identical, and the
        // graph drops the de-padding view+cont for single-clip inputs.
        if (vc.n_embd / vc.n_head == 72 && getenv("VC3_NO_FOLD") == nullptr) {
            const int64_t in_old = vc.n_embd, in_new = vc.n_head * 80;
            const size_t w_bytes = (size_t)in_new * vc.n_embd * sizeof(ggml_fp16_t);
            const size_t base = fold_storage.size();
            fold_storage.resize(base + (size_t)vc.n_layer * w_bytes);
            bool ok = true;
            for (int i = 0; i < vc.n_layer && ok; i++) {
                char buf[64];
                snprintf(buf, sizeof(buf), "v.blk.%d.wo.weight", i);
                ggml_tensor * W = ggml_get_tensor(wctx, buf);
                if (!W || W->type != GGML_TYPE_F16 || W->ne[0] != in_old) { ok = false; break; }
                const ggml_fp16_t * src = (const ggml_fp16_t *)W->data;
                ggml_fp16_t * dst = (ggml_fp16_t *)(fold_storage.data() + base + (size_t)i * w_bytes);
                const int64_t out = W->ne[1];
                for (int64_t o = 0; o < out; o++) {
                    for (int64_t h = 0; h < vc.n_head; h++) {
                        const ggml_fp16_t * srow = src + (size_t)o * in_old + h * 72;
                        ggml_fp16_t * drow = dst + (size_t)o * in_new + h * 80;
                        for (int j = 0; j < 72; j++) drow[j] = srow[j];
                        for (int j = 72; j < 80; j++) drow[j] = ggml_fp32_to_fp16(0.0f);
                    }
                }
                W->ne[0] = in_new;
                W->nb[0] = sizeof(ggml_fp16_t);
                W->nb[1] = (size_t)in_new * sizeof(ggml_fp16_t);
                W->nb[2] = W->nb[1];
                W->nb[3] = W->nb[2];
                W->data = dst;
            }
            if (ok) { wo_padded = true; fprintf(stderr, "wo input rows padded to %d/head (%.1f MB host block)\n",
                                                (int)in_new / vc.n_head, (double)(vc.n_layer * w_bytes) / 1e6); }
        }
    }

    {
        ggml_tensor * pe = ggml_get_tensor(wctx, "v.pos_emb.weight");
        ggml_tensor * tw = ggml_get_tensor(wctx, "v.pos_emb.time_weight");
        if (!pe || !tw) { fprintf(stderr, "missing positional embedding tensors\n"); return false; }
        pos_emb_host.resize(ggml_nelements(pe));
        pos_time_host.resize(ggml_nelements(tw));
        vqa::gguf::tensor_to_f32(pe, pos_emb_host.data());
        vqa::gguf::tensor_to_f32(tw, pos_time_host.data());
    }

    {
        // zero-copy (same trick as the moondream engine): the weights keep pointing
        // into the mmap, the f32-converted 1D tensors and the folded/padded vision
        // weights into their host blocks. Avoids a second multi-GB commit on
        // Windows, where the mmap already counts against the commit limit - load is
        // faster and the resident footprint is roughly halved.
        vqa::weights::Bound bound = vqa::weights::bind_to_backend(wctx, backend, mm,
            { { host_f32.data(), host_f32.size() }, { fold_storage.data(), fold_storage.size() } });
        if (!bound.ok) return false;
        wbuf = bound.main_buf;
        if (bound.host_bufs.size() > 0) wbuf_plain = bound.host_bufs[0];
        if (bound.host_bufs.size() > 1) wbuf_fold = bound.host_bufs[1];
    }
    if (ggml_backend_is_cpu(backend)) {
        host_f32_storage = std::move(host_f32);   // the 1D tensors point into it
    } else {
        host_f32.clear();
        host_f32.shrink_to_fit();
        mm.release();
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
    v_final_w = gt("v.final_norm.weight");
    v_final_b = gt("v.final_norm.bias");
    mm_pre_w = gt("mm.pre_norm.weight");
    mm_pre_b = gt("mm.pre_norm.bias");
    mm_l1_w = gt("mm.linear_1.weight");
    mm_l1_b = gt("mm.linear_1.bias");
    mm_l2_w = gt("mm.linear_2.weight");
    mm_l2_b = gt("mm.linear_2.bias");

    tl.resize(n_layer);
    for (int i = 0; i < n_layer; i++) {
        vqa::llm::LayerWeights & L = tl[i];
        vqa::gguf::bind_layer(wctx, "blk.%d", i, {
            {"attn_norm.weight",   &L.norm},
            {"attn_q.weight",      &L.q},
            {"attn_k.weight",      &L.k},
            {"attn_v.weight",      &L.v},
            {"attn_output.weight", &L.o},
            {"attn_q_norm.weight", &L.qn},
            {"attn_k_norm.weight", &L.kn},
            {"ffn_norm.weight",    &L.ffn_norm},
            {"ffn_gate.weight",    &L.gate},
            {"ffn_up.weight",      &L.up},
            {"ffn_down.weight",    &L.down},
        });
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
    return tok_embd != nullptr && out_norm != nullptr && v_patch_w != nullptr;
}

bool VideoChat3::alloc_kv() {
    return kv.alloc(backend, n_layer, n_kv_head, head_dim, n_ctx);
}

void VideoChat3::dump(const ggml_cgraph * gf, const char * name) {
    if (dump_dir.empty()) return;
    vqa::dump::write((ggml_cgraph *)gf, dump_dir.c_str(), name);   // core/ggml_dump.hpp
}

// ---- shared ViT block plumbing (vqa::vit::block) --------------------------
// The attention hook (strided padded head views, ggml GGML_ROPE_TYPE_VISION
// rope with the pad-rescaled freq base, clip-local unmasked FA, wo_padded /
// de-pad output paths, layer-0 parity truncation) lives once in
// vqa::vit::vit_attention_hook (core/vit_block.hpp); this backend only
// supplies its parameters in encode_vision. state = 1 after a successful
// truncated run.

// ---------------------------------------------------------------- vision
bool VideoChat3::encode_vision(const vqa::MediaInput & media, std::vector<float> & feats, int * n_tokens,
                               std::vector<std::pair<float,int>> * clip_marks, int token_budget) {
    *n_tokens = 0;
    if (media.empty()) return true;
    const int T = (int)media.frames.size();
    const int src_h = media.height, src_w = media.width;
    const int factor = vc.patch * vc.merge;

    int H = 0, W = 0;
    // qwen_vl_utils defaults used by the official demo: min 56*56, max 16384*28*28
    long long max_pixels = 16384LL * 28 * 28;
    if (token_budget > 0) {
        // merged tokens per clip = (H/28)*(W/28) = H*W/factor^2, and the merger
        // mean-pools the time axis of each temporal clip
        const int n_clips = (T + vc.temporal_merge - 1) / vc.temporal_merge;
        const long long min_total = 4LL * n_clips;             // min_pixels 56*56 -> 2x2 grid
        if (token_budget < min_total) {
            fprintf(stderr, "image/video cannot fit in n_ctx: needs >= %d merged vision tokens "
                            "(%d clip%s), budget %d -- raise --n-ctx or reduce --max-tokens\n",
                    (int)min_total, n_clips, n_clips == 1 ? "" : "s", token_budget);
            return false;
        }
        const long long per_grid = token_budget / n_clips;
        max_pixels = std::min(max_pixels, per_grid * factor * factor);
        if (max_pixels < 16384LL * 28 * 28)
            fprintf(stderr, "vision budget %d tokens (%d clip%s): max_pixels clamped to %.2f MP\n",
                    token_budget, n_clips, n_clips == 1 ? "" : "s", max_pixels / 1e6);
    }
    // per-frame resolution cap for videos: clip attention is quadratic over the
    // clip's (4 frames x patches) tokens, so full-HD frames make one clip cost
    // ~7.5s on a laptop GPU. ~0.75MP/frame keeps video QA accurate in practice
    // at a fraction of the cost; VC3_VIDEO_MP=0 restores full resolution up to
    // the context budget.
    if (media.kind == vqa::MediaKind::Video) {
        double vmp = 0.75;
        if (const char * e = getenv("VC3_VIDEO_MP")) vmp = atof(e);
        if (vmp > 0 && (double)max_pixels > vmp * 1e6) {
            max_pixels = (long long)(vmp * 1e6);
            fprintf(stderr, "video frame cap: %.2f MP/frame (VC3_VIDEO_MP to override)\n", vmp);
        }
    }
    if (!smart_resize(src_h, src_w, factor, 3136, (int)std::min(max_pixels, (long long)INT_MAX), &H, &W)) {
        fprintf(stderr, "image too small (%dx%d, factor %d)\n", src_w, src_h, factor);
        return false;
    }
    // VRAM guard: the tower runs at PATCH level, so the activation footprint
    // scales with (H/patch)*(W/patch)*T — the token budget above only bounds the
    // text context. A 16-frame 1820x840 clip is within budget but needs ~5 GB
    // of vision activations, which aborts the whole process on a 12 GB card.
    // Cap the patch count (~48k rows ≈ 2 GB of f32 activations) by shrinking
    // the per-frame resolution.
    {
        // the tower now runs per clip, so the footprint scales with ONE clip
        const long long n_patches = (long long)(H / vc.patch) * (W / vc.patch) * std::min(T, vc.temporal_merge);
        const long long n_max = 48000;
        if (n_patches > n_max) {
            const double scale = (double)n_max / (double)n_patches;
            max_pixels = std::min(max_pixels, (long long)(max_pixels * scale));
            int H2, W2;
            if (!smart_resize(src_h, src_w, factor, 3136, (int)std::min(max_pixels, (long long)INT_MAX), &H2, &W2)) {
                fprintf(stderr, "image too small (%dx%d, factor %d)\n", src_w, src_h, factor);
                return false;
            }
            fprintf(stderr, "vision VRAM guard: %lld patches > %lld, shrinking %dx%d -> %dx%d each\n",
                    n_patches, n_max, W, H, W2, H2);
            H = H2; W = W2;
        }
    }
    if (media.kind == vqa::MediaKind::Video)
        fprintf(stderr, "video %d frame%s %dx%d -> %dx%d each\n", T, T == 1 ? "" : "s", src_w, src_h, W, H);
    else
        fprintf(stderr, "image %dx%d -> %dx%d\n", src_w, src_h, W, H);

    // clips of temporal_merge frames each (modeling_videochat3.py split_grid_thws_clip_by_clip):
    // attention is block-diagonal per clip, the temporal pos index restarts per clip and
    // the merger mean-pools the time axis of each clip
    std::vector<int> clip_t;
    for (int off = 0; off < T; off += vc.temporal_merge)
        clip_t.push_back(std::min(vc.temporal_merge, T - off));
    // per-clip prompt timestamps: mean of the first/last frame time in the clip
    // (VideoChat3Processor._calculate_timestamps semantics)
    std::vector<std::pair<float,int>> marks;
    if (media.kind == vqa::MediaKind::Video) {
        const double def_fps = media.fps > 0 ? media.fps : 1.0;
        int off = 0;
        for (int t : clip_t) {
            const double t0 = off < (int)media.timestamps.size() ? media.timestamps[off] : off / def_fps;
            const double t1 = (off + t - 1) < (int)media.timestamps.size() ? media.timestamps[off + t - 1] : (off + t - 1) / def_fps;
            marks.push_back({ (float)((t0 + t1) / 2.0), 0 });   // token count filled after patchify
            off += t;
        }
    }

    std::vector<float> rgb, patches, pos, pos_clip;
    int gh = 0, gw = 0;
    for (int ti = 0; ti < T; ti++) {
        resize_bicubic_antialias(media.frames[ti].rgb.data(), src_w, src_h, W, H, rgb);
        std::vector<float> p;
        patchify(rgb.data(), H, W, vc, p, &gh, &gw);
        patches.insert(patches.end(), p.begin(), p.end());
    }
    const int hw = gh * gw;
    const int N = hw * T;
    for (int t : clip_t) {
        pos_embed_for_grid(pos_emb_host.data(), pos_time_host.data(), vc, t, gh, gw, pos_clip);
        pos.insert(pos.end(), pos_clip.begin(), pos_clip.end());
    }
    if (!patch_bias.empty())   // patch bias folded here: one fewer graph add
        for (size_t k = 0; k < pos.size(); k++) pos[k] += patch_bias[k % vc.n_embd];

    const int vhd = vc.n_embd / vc.n_head;
    const bool pad80 = vhd == 72;
    // 2D positions for ggml's fused GGML_ROPE_TYPE_VISION kernel: the converter
    // permutes the wqkv q/k rows into its (x-pairs | y-pairs, re|im) layout, so
    // rope is one fused kernel instead of the former ~9-node rotation subgraph
    // rope coordinates are built per chunk below (sequential over the token
    // stream, one grid per frame - official get_freqs_cis reshapes the
    // row-major table flat)

    // ---- tower: clips are attention-independent (block-diagonal attention,
    // per-clip position tables), so each clip runs through the ViT as its own
    // graph and the activation footprint scales with ONE clip instead of the
    // whole video: a 16-frame 1820x840 request needs ~5 GB as one graph but
    // ~1.3 GB per clip. Exact same ops, just executed per clip.
    const char * stop_after = getenv("VC3_STOP_AFTER");
    static const bool vprof = getenv("VC3_PROF") != nullptr;
    static double v_alloc = 0, v_input = 0, v_comp = 0; static int v_runs = 0;
    auto t_now = []() { return std::chrono::steady_clock::now(); };

    // tower outputs land in a persistent DEVICE buffer (core/stage_buf.hpp);
    // the merger reads it as a view - no per-clip host round trip
    if (!tower_buf.ensure(backend, vc.n_embd, (int64_t)hw * (int64_t)T)) return false;
    int frame0 = 0;
    // immediate first tick so the UI shows the total before clip 0 finishes
    // (one clip of a full-resolution video can take seconds)
    if (stage_cb) stage_cb("vision", 0, (int)clip_t.size(), stage_ud);
    for (size_t c = 0; c < clip_t.size(); c++) {
        const int t = clip_t[c];
        const int Nc = hw * t;
        // per-clip graph: drop the previous clip's registry (and its sink buffers)
        if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
        struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
        ggml_context * gctx = ggml_init(ip);
        if (!gctx) return false;
        struct ggml_cgraph * gf = ggml_new_graph_custom(gctx, 16384, false);

        ggml_tensor * t_patches = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 3 * vc.patch * vc.patch, Nc);
        ggml_tensor * t_pos = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, vc.n_embd, Nc);
        ggml_tensor * t_vpos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, (int64_t)Nc * 4);
        std::vector<int> vpos((size_t)Nc * 4, 0);
        for (int s = 0; s < Nc; s++) {
            const int g = s % hw;   // frame-local: clips restart the 2D grid
            vpos[s] = g % gw;
            vpos[(size_t)Nc + s] = g / gw;
        }
        const int C3 = 3 * vc.patch * vc.patch;
        const float * patches_c = patches.data() + (size_t)frame0 * hw * C3;
        const float * pos_c = pos.data() + (size_t)frame0 * hw * vc.n_embd;

        ggml_tensor * x = ggml_mul_mat(gctx, v_patch_w, t_patches);
        if (patch_bias.empty())   // bias folded into the host pos table at load time
            x = ggml_add(gctx, x, v_patch_b);
        x = ggml_add(gctx, x, t_pos);
        ggml_set_name(x, "v_patch_out");

        bool sinks_ready = false;   // keep nodes attach once, at the graph tail
        auto run_chunk = [&](const char * out_name, std::vector<float> * dest) -> bool {
            const auto ra = t_now();
            if (!dump_dir.empty() && !sinks_ready) {
                static const char * const dump_names[] = {
                    "v_patch_out", "v_blk0_qkv", "v_blk0_q_rope", "v_blk0_attn_out",
                    "v_blk0_out", "v_final", nullptr,
                };
                vqa::dump::mark_sinks(gctx, gf, dump_names);
                vqa::dump::finalize_sinks(gctx, gf, backend);
                sinks_ready = true;
            }
            if (!ggml_gallocr_alloc_graph(allocr, gf)) {
                fprintf(stderr, "vision graph allocation failed\n");
                return false;
            }
            const auto rb = t_now();
            if (t_patches->buffer) ggml_backend_tensor_set(t_patches, patches_c, 0, sizeof(float) * (size_t)Nc * C3);
            if (t_pos->buffer)     ggml_backend_tensor_set(t_pos, pos_c, 0, sizeof(float) * (size_t)Nc * vc.n_embd);
            if (t_vpos->buffer)    ggml_backend_tensor_set(t_vpos, vpos.data(), 0, sizeof(int) * vpos.size());
            const auto rc = t_now();
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "vision compute failed\n");
                return false;
            }
            const auto rd = t_now();
            if (vprof) {
                v_runs++;
                v_alloc += std::chrono::duration<double,std::milli>(rb-ra).count();
                v_input += std::chrono::duration<double,std::milli>(rc-rb).count();
                v_comp  += std::chrono::duration<double,std::milli>(rd-rc).count();
                fprintf(stderr, "[vprof] run=%d alloc %.1fms input %.1fms compute %.1fms\n",
                        v_runs, v_alloc/v_runs, v_input/v_runs, v_comp/v_runs);
            }
            if (c == 0) {
                dump(gf, "patches");
                dump(gf, "v_patch_out");
                dump(gf, "v_blk0_qkv");
                dump(gf, "v_blk0_q_rope");
                dump(gf, "v_blk0_attn_out");
                dump(gf, "v_blk0_out");
            }
            dump(gf, out_name);   // sink dump is always buffer-safe
            if (dest) {
                ggml_tensor * out = ggml_graph_get_tensor(gf, out_name);
                dest->resize((size_t)out->ne[1] * out->ne[0]);
                ggml_backend_tensor_get(out, dest->data(), 0, sizeof(float) * dest->size());
            }
            return true;
        };

        // parity truncation only applies to the first chunk (parity runs are images)
        const char * stop_c = (c == 0) ? stop_after : nullptr;
        if (stop_c && !strcmp(stop_c, "v_patch_out")) {
            ggml_build_forward_expand(gf, x);
            if (!run_chunk("v_patch_out", &feats)) { ggml_free(gctx); return false; }
            *n_tokens = (int)(feats.size() / vc.n_embd);
            ggml_free(gctx);
            return true;
        }

        int vl_max = vc.n_layer;
        { const char * e = getenv("VC3_VL"); if (e) vl_max = std::min(vl_max, atoi(e)); }
        std::vector<int> clip_one = { t };
        vqa::vit::VitHookEnv env;
        env.gctx = gctx; env.gf = gf; env.t_vpos = t_vpos;
        env.n_head = vc.n_head; env.n_embd = vc.n_embd;
        env.N = Nc; env.vhd = vhd; env.vhd_f = pad80 ? 80 : vhd; env.hw = hw; env.n_ctx = n_ctx;
        env.rope_theta = vc.rope_theta; env.rope_pad_rescale = pad80;
        env.wo_padded = wo_padded; env.clips = &clip_one;
        env.dbg_names = true;   // layer-0 rope/attn tensor names
        env.stop_after = stop_c;
        env.state = 0;
        env.run = [&run_chunk](const char * name) -> bool { std::vector<float> sink; return run_chunk(name, &sink); };
        for (int il = 0; il < vl_max; il++) {
            VisLayer & L = vl[il];
            vqa::vit::BlockWeights bw{L.norm0_w, L.norm0_b, L.wqkv_w, L.wqkv_b, L.wo_w, L.wo_b,
                                      L.norm1_w, L.norm1_b, L.fc0_w, L.fc0_b, L.fc1_w, L.fc1_b};
            vqa::vit::BlockHooks bh;
            bh.attention = vqa::vit::vit_attention_hook;
            bh.probe_layer0 = vqa::vit::vit_probe_layer0_hook;
            bh.drop_ln_mul = fold_scale;
            bh.drop_ln_add = fold_bias;
            bh.ud = &env;
            env.layer = il;
            x = vqa::vit::block(gctx, x, bw, vc.layernorm_eps, bh, il);
            if (!x) {   // layer-0 probe truncated the graph (state 1) or failed (2)
                ggml_free(gctx);
                return env.state == 1;
            }
        }
        x = ggml_norm(gctx, x, vc.layernorm_eps);
        x = ggml_mul(gctx, x, v_final_w);
        x = ggml_add(gctx, x, v_final_b);
        ggml_set_name(x, "v_final");
        if (stop_c && !strcmp(stop_c, "v_final")) {
            ggml_build_forward_expand(gf, x);
            if (!run_chunk("v_final", &feats)) { ggml_free(gctx); return false; }
            *n_tokens = (int)(feats.size() / vc.n_embd);
            ggml_free(gctx);
            return true;
        }
        ggml_tensor * out_sink = tower_buf.sink(gctx, x, (int64_t)frame0);
        ggml_build_forward_expand(gf, out_sink);
        if (!run_chunk("v_final", nullptr)) { ggml_free(gctx); return false; }
        ggml_free(gctx);
        frame0 += t;
        if (stage_cb) stage_cb("vision", (int)(c + 1), (int)clip_t.size(), stage_ud);
    }

    // ---- merger + projector over the concatenated tower output
    {
        if (!dump_dir.empty()) vqa::dump::dump_registry_clear();   // new graph, new registry
        struct ggml_init_params ip2 = { 32ull * 1024 * 1024, NULL, /*no_alloc*/ true };
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
            const int n = (int)out->ne[1];
            const int d = (int)out->ne[0];
            feats.resize((size_t)n * d);
            ggml_backend_tensor_get(out, feats.data(), 0, sizeof(float) * feats.size());
            *n_tokens = n;
            {
                const int per_clip_tok = (gh / vc.merge) * (gw / vc.merge);
                for (auto & m : marks) m.second = per_clip_tok;
                if (clip_marks) *clip_marks = marks;
            }
            fprintf(stderr, "vision: %d patches (%dx%d) -> %d tokens (%zu clip%s)%s\n", N, gh, gw, n,
                    clip_t.size(), clip_t.size() == 1 ? "" : "s",
                    stop_after ? " (truncated for parity)" : "");
            return true;
        };

        // merge: the reference reads the sequence as (t, nh, kh, nw, kw) and mean-pools
        // the time axis of each clip, then folds (kh,kw) into the feature dim
        // (patch_merger in modeling_videochat3.py); the processor emits (nh, nw, kh, kw)
        const int nh = gh / vc.merge, nw = gw / vc.merge;
        const int kh = vc.merge, kw = vc.merge;
        ggml_tensor * pooled = nullptr;
        size_t off = 0;
        for (int t : clip_t) {
            ggml_tensor * acc = ggml_view_2d(mctx, x, vc.n_embd, hw, x->nb[1], off);
            for (int ti = 1; ti < t; ti++)
                acc = ggml_add(mctx, acc,
                        ggml_view_2d(mctx, x, vc.n_embd, hw, x->nb[1], off + (size_t)ti * hw * x->nb[1]));
            acc = ggml_scale(mctx, acc, 1.0f / t);
            pooled = pooled ? ggml_concat(mctx, pooled, acc, 1) : acc;
            off += (size_t)t * hw * x->nb[1];
        }
        x = pooled;
        ggml_tensor * x4 = ggml_reshape_4d(mctx, x, (int64_t)vc.n_embd * kw, nw, kh, (int64_t)nh * clip_t.size());
        x4 = ggml_cont(mctx, ggml_permute(mctx, x4, 0, 2, 1, 3));
        ggml_tensor * x3 = ggml_reshape_3d(mctx, x4, vc.n_embd, kw * kh, (int64_t)nh * nw * clip_t.size());
        ggml_set_name(x3, "v_merge");
        if (stop_after && !strcmp(stop_after, "v_merge")) {
            ggml_build_forward_expand(mgf, x3);
            if (!run_merger("v_merge")) { ggml_free(mctx); return false; }
            ggml_free(mctx);
            return true;
        }
        ggml_tensor * pn = ggml_norm(mctx, x3, 1e-5f);
        if (!fold_scale) pn = ggml_mul(mctx, pn, mm_pre_w);   // gain folded into mm_l1 (VC3_FOLD_W)
        if (!fold_bias)  pn = ggml_add(mctx, pn, mm_pre_b);   // bias folded into mm_l1
        ggml_tensor * x2 = ggml_reshape_2d(mctx, pn, (int64_t)vc.n_embd * kw * kh, (int64_t)nh * nw * clip_t.size());
        ggml_tensor * h = ggml_mul_mat(mctx, mm_l1_w, x2);
        h = ggml_add(mctx, h, mm_l1_b);
        h = ggml_gelu_erf(mctx, h);                   // nn.GELU() == exact erf
        h = ggml_mul_mat(mctx, mm_l2_w, h);
        h = ggml_add(mctx, h, mm_l2_b);
        ggml_set_name(h, "v_proj_out");
        ggml_build_forward_expand(mgf, h);
        if (!run_merger("v_proj_out")) { ggml_free(mctx); return false; }
        ggml_free(mctx);
    }
    return true;
}

// ---------------------------------------------------------------- text// ---------------------------------------------------------------- text
bool VideoChat3::build_text_graph(TextGraph & g, int T, int S, int n_img, bool logits_all,
                               const std::vector<BatGroup> * bat_groups) {
    // a new graph invalidates the previous dump registry (its sink buffers are
    // freed here, so a stale entry can never be read as a live tensor)
    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 16384, false);

    const int D = n_embd, HD = head_dim, NH = n_head, NKV = n_kv_head;

    g.tok_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.tok_in, "tok_in");
    g.pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
    ggml_set_name(g.pos_in, "pos_in");
    if (!bat_groups) {
        g.mask_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, S, T);
        ggml_set_name(g.mask_in, "mask_in");
    }

    // token embedding stays on the device (works with a quantized token_embd and
    // avoids one host round-trip per prompt token); image rows are scattered into
    // it exactly like llama.cpp does for multimodal batches
    ggml_tensor * x = ggml_get_rows(g.ctx, tok_embd, g.tok_in);      // [D, T]
    if (n_img > 0) {
        g.img_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, D, n_img);
        ggml_set_name(g.img_in, "img_in");
        g.img_pos_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_img);
        ggml_set_name(g.img_pos_in, "img_pos_in");
        // ggml_set_rows writes into its destination, so it needs its own copy
        x = ggml_set_rows(g.ctx, ggml_dup(g.ctx, x), g.img_in, g.img_pos_in);
    }
    ggml_set_name(x, "emb");

    // batch group masks: ONE tensor per group, shared by every layer (creating
    // them inside the layer loop made 36 same-named tensors and only the first
    // ever got filled — the rest ran on uninitialized memory)
    std::vector<ggml_tensor *> bat_mask_ts;
    if (bat_groups) {
        char mn[24];
        for (const auto & gr : *bat_groups) {
            snprintf(mn, sizeof(mn), "mask_b%d", gr[0]);
            int w = (gr[3] + 31) & ~31;
            if (gr[2] + w > n_ctx) w = n_ctx - gr[2];
            ggml_tensor * m = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, w, gr[1]);
            ggml_set_name(m, mn);
            bat_mask_ts.push_back(m);
        }
    }
    int max_layers = n_layer;
    if (const char * e = getenv("VC3_MAX_LAYERS")) max_layers = std::min(max_layers, atoi(e));
    // ---- decoder layer loop: shared vqa::llm::block (core/llm_block.hpp) ----
    // the layer body below is VideoChat3's original build verbatim (same ops,
    // same operand order, same node creation order - f16 graphs are sensitive
    // to adjacent-node reordering); only the t_l0_* names, the fusion toggle
    // and the cache/mask/position plumbing stay parameterized here.
    static const bool no_fusion = getenv("VC3_NO_FUSION") != nullptr;
    vqa::llm::LayerCtx lc;
    lc.gf = g.gf;
    lc.pos = g.pos_in;
    lc.mask = g.mask_in;
    lc.T = T; lc.S = S;
    lc.n_head = NH; lc.n_kv_head = NKV; lc.head_dim = HD;
    lc.n_ctx = n_ctx;
    lc.rms_eps = rms_eps;
    lc.rope_theta = rope_theta;
    lc.bat_groups = bat_groups;
    lc.bat_masks = bat_groups ? &bat_mask_ts : nullptr;
    vqa::llm::BlockHooks lh;
    lh.swiglu_split = !no_fusion;
    for (int il = 0; il < max_layers; il++) {
        lc.k_cache = kv.k[il];
        lc.v_cache = kv.v[il];
        x = vqa::llm::block(g.ctx, x, il, tl[il], lc, lh);
    }

    x = ggml_rms_norm(g.ctx, x, rms_eps);
    x = ggml_mul(g.ctx, x, out_norm);
    ggml_set_name(x, "t_final_norm");

    if (logits_all) {
        g.logits = ggml_mul_mat(g.ctx, tok_embd, x);   // [vocab, T] for batch decode
    } else {
        ggml_tensor * last = ggml_view_2d(g.ctx, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
        g.logits = ggml_mul_mat(g.ctx, tok_embd, last);
    }
    ggml_set_name(g.logits, "logits");
    ggml_build_forward_expand(g.gf, g.logits);
    if (!dump_dir.empty()) {
        static const char * const dump_names[] = {
            "emb", "t_l0_norm", "t_l0_q", "t_l0_q_norm", "t_l0_q_rope", "t_l0_attn",
            "t_l0_out", "t_final_norm", nullptr,
        };
        vqa::dump::mark_sinks(g.ctx, g.gf, dump_names);
        vqa::dump::finalize_sinks(g.ctx, g.gf, backend);   // keep nodes at the graph tail
    }
    return true;
}

bool VideoChat3::run_text(TextGraph & g, const std::vector<int> & toks, const std::vector<float> * img_feats,
                          const std::vector<int> * img_pos, std::vector<float> & logits_out) {
    const int T = (int)toks.size();
    const int S = n_past + T;
    if (S > n_ctx) { fprintf(stderr, "context overflow (%d > %d)\n", S, n_ctx); return false; }
    const int n_img = (img_feats && img_pos) ? (int)(img_feats->size() / n_embd) : 0;
    static const bool prof = getenv("VC3_PROF") != nullptr;
    static double tb = 0, ta = 0, tc = 0; static int ns = 0;
    auto tick = [](){ return std::chrono::steady_clock::now(); };
    auto tp0 = tick();

    // single-token decode: a static graph with a block-rounded KV window.
    // The identical gf object across steps lets ggml capture a CUDA graph,
    // cutting per-step kernel-launch overhead (the moondream Engine pattern).
    ggml_tensor * logits_t = nullptr;
    const bool use_static = (T == 1) && n_img == 0 && !ggml_backend_is_cpu(backend);
    if (use_static) {
        const int blk = std::min(vqa::block_for(S), n_ctx);
        if (!dec_gf || dec_blk != blk) {
            if (dec_ctx) ggml_free(dec_ctx);
            struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, true };
            dec_ctx = ggml_init(ip);
            TextGraph dg;
            // build directly into the persistent context
            const int D = n_embd;
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
            dec_pos = ggml_graph_get_tensor(dec_gf, "pos_in");
            dec_mask = ggml_graph_get_tensor(dec_gf, "mask_in");
        }
        ggml_backend_tensor_set(dec_tok, toks.data(), 0, sizeof(int));
        ggml_backend_tensor_set(dec_pos, &n_past, 0, sizeof(int));
        vqa::set_causal_mask(dec_mask, n_past, 1);
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
                fprintf(stderr, "[prof] steps=%d build+input %.2fms compute %.2fms (avg, static)\n", ns, tb/ns, tc/ns);
        }
        logits_out.resize(n_vocab);
        ggml_backend_tensor_get(ggml_graph_get_tensor(dec_gf, "logits"), logits_out.data(), 0, sizeof(float) * n_vocab);
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
    if (n_img > 0) {
        ggml_backend_tensor_set(g.img_in, img_feats->data(), 0, sizeof(float) * img_feats->size());
        ggml_backend_tensor_set(g.img_pos_in, img_pos->data(), 0, sizeof(int) * n_img);
    }
    {
        std::vector<int> posv(T);
        for (int i = 0; i < T; i++) posv[i] = n_past + i;
        ggml_backend_tensor_set(g.pos_in, posv.data(), 0, sizeof(int) * T);
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
    dump(g.gf, "emb");
    dump(g.gf, "t_l0_norm");
    dump(g.gf, "t_l0_q_rope");
    dump(g.gf, "t_l0_out");
    dump(g.gf, "t_final_norm");
    dump(g.gf, "logits");

    logits_out.resize(n_vocab);
    ggml_backend_tensor_get(g.logits, logits_out.data(), 0, sizeof(float) * n_vocab);
    n_past = S;
    ggml_free(g.ctx);
    g.ctx = nullptr;
    return true;
}

std::string VideoChat3::render_prompt(int n_img_tokens, const std::string & user_text, bool video,
                                       const std::vector<std::pair<float,int>> * clip_marks) const {
    std::string s = "<|im_start|>user\n";
    if (n_img_tokens > 0) {
        const char * pad = video ? "<|video_pad|>" : "<|image_pad|>";
        if (video && clip_marks && !clip_marks->empty()) {
            // official processor format: one timestamped segment per temporal clip
            // ("<t seconds>" = mean of the clip's first/last frame times)
            char ts[32];
            for (const auto & m : *clip_marks) {
                snprintf(ts, sizeof(ts), "<%.1f seconds>", m.first);
                s += ts;
                s += "<|vision_start|>";
                for (int i = 0; i < m.second; i++) s += pad;
                s += "<|vision_end|>";
            }
        } else {
            s += "<|vision_start|>";
            for (int i = 0; i < n_img_tokens; i++) s += pad;
            s += "<|vision_end|>";
        }
    }
    s += user_text;
    s += "<|im_end|>\n<|im_start|>assistant\n";
    return s;
}

int VideoChat3::sample(const std::vector<float> & logits, const vqa::GenParams & gp) {
    return vqa::sample_logits(logits, gp, rng_state);
}

bool VideoChat3::ask(const vqa::MediaInput & media, const std::string & prompt,
                     const std::string & /*mode*/, const vqa::GenParams & gp, std::string & out_text) {
    std::vector<float> feats;
    int n_img = 0;
    stage_cb = gp.on_stage;
    stage_ud = gp.on_stage_ud;
    const auto tv0 = std::chrono::steady_clock::now();
    std::vector<std::pair<float,int>> clip_marks;
    const bool video_kind = media.kind == vqa::MediaKind::Video;
    // spend at most n_ctx - max_tokens - text on vision tokens; measure the text
    // overhead with a 1-placeholder prompt (clip timestamps add a few tokens more,
    // covered by the 16-token margin) so an oversized image is clamped or rejected
    // before the multi-second vision encode instead of after it
    {
        const int text_overhead = (int)tok_.encode(gp.history +
            render_prompt(1, prompt, video_kind, nullptr)).size();
        // per-clip timestamp prompt ("<12.5 seconds>" etc.) costs ~10 tokens; the
        // 1-placeholder probe above renders none of them, so reserve room for all clips
        const int n_clips = (int)((media.frames.size() + vc.temporal_merge - 1) / vc.temporal_merge);
        const int budget = n_ctx - gp.max_tokens - text_overhead - 16 - 12 * n_clips;
        if (budget <= 0) {
            fprintf(stderr, "n_ctx %d too small: prompt %d tokens + max_new %d leaves nothing for the image\n",
                    n_ctx, text_overhead, gp.max_tokens);
            return false;
        }
        // media cache: the same picture/video re-encoded on every follow-up
        // question wasted seconds - key on the frame bytes and reuse the feats
        // (weights are fixed for the resident model, so the cache never goes stale)
        static struct {
            uint64_t key = 0;
            std::vector<float> f;
            int n = 0;
            std::vector<std::pair<float,int>> marks;
        } vcache;
        uint64_t key = 1469598103934665603ull;
        for (const auto & fr : media.frames) {
            key ^= fr.rgb.size(); key *= 1099511628211ull;
            for (size_t k = 0; k < fr.rgb.size(); k++) { key ^= fr.rgb[k]; key *= 1099511628211ull; }
        }
        key ^= (uint64_t)media.width << 32 ^ (uint64_t)media.height;
        if (vcache.key == key && vcache.n > 0) {
            feats = vcache.f; n_img = vcache.n; clip_marks = vcache.marks;
            fprintf(stderr, "vision: cache hit (%d tokens)\n", n_img);
        } else if (encode_vision(media, feats, &n_img, &clip_marks, budget)) {
            vcache.key = key; vcache.f = feats; vcache.n = n_img; vcache.marks = clip_marks;
        } else {
            return false;
        }
    }
    fprintf(stderr, "[perf] vision encode: %.0f ms (%d tokens)\n",
            std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tv0).count(), n_img);

    const bool video = media.kind == vqa::MediaKind::Video;
    // multi-turn session: the server's history prefix (fully formatted prior
    // turns, moss pattern) goes in verbatim before the new user turn
    const std::string text = gp.history + render_prompt(n_img, prompt, video, &clip_marks);
    const std::vector<int> ids = tok_.encode(text);
    const int pad_id = video ? id_video_pad : id_image_pad;
    std::vector<int> img_pos;
    for (int i = 0; i < (int)ids.size(); i++) if (ids[i] == pad_id) img_pos.push_back(i);
    if ((int)img_pos.size() != n_img) {
        fprintf(stderr, "placeholder mismatch: %d image_pad tokens vs %d vision tokens\n",
                (int)img_pos.size(), n_img);
        if (n_img > 0) return false;
    }
    fprintf(stderr, "prompt: %d tokens (%d image), max_new %d\n", (int)ids.size(), n_img, gp.max_tokens);
    int max_new = gp.max_tokens;
    if ((int)ids.size() + max_new > n_ctx) {
        max_new = n_ctx - (int)ids.size();
        fprintf(stderr, "warning: prompt %d + max_new %d > n_ctx %d, clamping max_new to %d\n",
                (int)ids.size(), gp.max_tokens, n_ctx, max_new);
        if (max_new <= 0) return false;
    }

    n_past = 0;
    kv.clear();
    auto t0 = std::chrono::steady_clock::now();
    TextGraph g;
    std::vector<float> logits;
    if (!run_text(g, ids, n_img > 0 ? &feats : nullptr, n_img > 0 ? &img_pos : nullptr, logits))
        return false;
    auto t1 = std::chrono::steady_clock::now();

    out_text.clear();
    int next = sample(logits, gp);
    int n_dec = 0;
    for (int step = 0; step < max_new; step++) {
        if (next == id_eos || next == id_endoftext) break;
        const std::string piece = tok_.decode({next});
        out_text += piece;
        if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
        n_dec++;
        if (step + 1 >= max_new) break;
        if (!run_text(g, {next}, nullptr, nullptr, logits)) break;
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

#include "vc3_batch.inc"

}  // namespace vc3

namespace vqa {
Model * create_videochat3() { return new vc3::VideoChat3(); }
}  // namespace vqa
