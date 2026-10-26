// qwen3omni_talker.cpp - see qwen3omni_talker.hpp for the protocol contract.
#include "qwen3omni_talker.hpp"

#include "core/util/mmap.hpp"
#include "core/util/aligned_buf.hpp"
#include "core/engine/weight_load.hpp"
#include "core/engine/gguf_util.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/sampler.hpp"
#include "core/engine/attention.hpp"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "core/blocks/omni_speech.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qwen3omni {
int dump_c2w_mid = 0;  // test hook: write embed/transformer intermediates

// ---------------------------------------------------------------- helpers
static void f16_to_host(ggml_tensor * t, std::vector<float> & dst) {
    dst.resize(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, dst.data(), 0, sizeof(float) * dst.size());
    } else {
        std::vector<uint16_t> tmp(dst.size());
        ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size() * 2);
        for (size_t i = 0; i < tmp.size(); i++)
            dst[i] = ggml_fp16_to_fp32(tmp[i]);
    }
}

// host conv weight: torch [Co, Ci, K]
static bool conv_from_gguf(ggml_context * ctx, const char * wname, const char * bname,
                           int dilation, int stride, int groups, C2wConv & out,
                           bool transposed = false) {
    ggml_tensor * w = ggml_get_tensor(ctx, wname);
    ggml_tensor * b = bname ? ggml_get_tensor(ctx, bname) : nullptr;
    if (!w) { fprintf(stderr, "talker: missing conv %s\n", wname); return false; }
    // ggml ne order: [K, Ci, Co] (numpy [Co, Ci, K] reversed)
   // regular conv: numpy [Co, Ci, K] -> ne [K, Ci, Co]; transposed:
    // numpy [Ci, Co, K] -> ne [K, Co, Ci] (Co/Ci swapped!)
    out.k = (int)w->ne[0];
    if (transposed) { out.ci = (int)w->ne[2]; out.co = (int)w->ne[1]; }
    else            { out.ci = (int)w->ne[1]; out.co = (int)w->ne[2]; }
    out.dilation = dilation; out.stride = stride; out.groups = groups;
    out.has_bias = b != nullptr;
    f16_to_host(w, out.w);
    if (b) f16_to_host(b, out.b);
    return true;
}

// causal conv1d with the HF _get_extra_padding_for_conv1d right-pad formula
// `Ci` is the FULL input channel count; c.ci is per-group (weight layout
// [Co, Ci/groups, K]) and must NOT be used to size the input buffer
static void causal_conv(const C2wConv & c, const float * in, int Ci, int T, float * outv) {
    const int k_eff = (c.k - 1) * c.dilation + 1;
    const int pad_left = k_eff - c.stride;
    double n_frames = ((double)T - c.k + pad_left) / c.stride + 1.0;
    // HF: ideal_length = (ceil(n_frames) - 1) * stride + (k - pad)
    const int ideal = ((int)std::ceil(n_frames - 1e-9) - 1) * c.stride + (c.k - pad_left);
    const int extra = ideal - T;
    const int Tin = pad_left + T + std::max(0, extra);
    std::vector<float> xin((size_t)Ci * Tin, 0.0f);
    for (int ch = 0; ch < Ci; ch++)
        if (T > 0) memcpy(&xin[(size_t)ch * Tin + pad_left], in + (size_t)ch * T, sizeof(float) * T);
    const int Tout = (Tin - k_eff) / c.stride + 1;
    const int g = c.groups, cpg = c.co / g, cpi = (Ci / g) ? (Ci / g) : c.ci;
    if (dump_c2w_mid) fprintf(stderr, "[cc] co=%d cpg=%d cpi=%d Tout=%d Tin=%d wsize=%zu\n", c.co, cpg, cpi, Tout, Tin, c.w.size());
    for (int co = 0; co < c.co; co++) {
        const int gi = co / cpg;
        for (int t = 0; t < Tout; t++) {
            float acc = c.has_bias ? c.b[(size_t)co] : 0.0f;
            const float * wr = &c.w[((size_t)co * cpi) * c.k];
            for (int ci = 0; ci < cpi; ci++) {
                const float * xr = &xin[(size_t)(gi * cpi + ci) * Tin + (size_t)t * c.stride];
                for (int kk = 0; kk < c.k; kk++)
                    acc += wr[(size_t)ci * c.k + kk] * xr[kk * c.dilation];
            }
            outv[(size_t)co * Tout + t] = acc;
        }
    }
}

// transposed conv, torch layout [Ci, Co, K], stride r, (k-r)/2 trim at both
// ends. Dense (groups=1) throughout this vocoder. Parallel over output channels.


// ---------------------------------------------------------------- load
bool Talker::load(const char * path, ggml_backend_t backend_, ggml_gallocr_t allocr_) {
    backend = backend_; allocr = allocr_;
    mm = new vqa::MMap();
    if (!mm->load(path)) { fprintf(stderr, "talker mmap failed\n"); return false; }
    struct ggml_init_params wip = { 128ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    wctx = ggml_init(wip);
    if (!wctx) return false;
    struct gguf_init_params p = { true, &wctx };
    gguf_context * gu = gguf_init_from_file(path, p);
    if (!gu) { fprintf(stderr, "talker gguf init failed\n"); return false; }

    n_layer     = vqa::gguf::u32(gu, "qwen3omni.talker.block_count", n_layer);
    n_embd      = vqa::gguf::u32(gu, "qwen3omni.talker.embedding_length", n_embd);
    n_head      = vqa::gguf::u32(gu, "qwen3omni.talker.attention.head_count", n_head);
    n_kv_head   = vqa::gguf::u32(gu, "qwen3omni.talker.attention.head_count_kv", n_kv_head);
    n_expert    = vqa::gguf::u32(gu, "qwen3omni.talker.expert_count", n_expert);
    n_expert_used = vqa::gguf::u32(gu, "qwen3omni.talker.expert_used_count", n_expert_used);
    n_ff_exp    = vqa::gguf::u32(gu, "qwen3omni.talker.expert_feed_forward_length", n_ff_exp);
    n_vocab     = vqa::gguf::u32(gu, "qwen3omni.talker.vocab_size", n_vocab);
    accept_hidden_layer = vqa::gguf::u32(gu, "qwen3omni.talker.accept_hidden_layer", accept_hidden_layer);
    n_groups    = vqa::gguf::u32(gu, "qwen3omni.talker.num_code_groups", n_groups);
    id_codec_eos  = vqa::gguf::u32(gu, "qwen3omni.codec.eos", id_codec_eos);
    id_codec_bos  = vqa::gguf::u32(gu, "qwen3omni.codec.bos", id_codec_bos);
    id_codec_pad  = vqa::gguf::u32(gu, "qwen3omni.codec.pad", id_codec_pad);
    id_codec_nothink = vqa::gguf::u32(gu, "qwen3omni.codec.nothink", id_codec_nothink);
    id_tts_bos  = vqa::gguf::u32(gu, "qwen3omni.tts.bos", id_tts_bos);
    id_tts_eos  = vqa::gguf::u32(gu, "qwen3omni.tts.eos", id_tts_eos);
    id_tts_pad  = vqa::gguf::u32(gu, "qwen3omni.tts.pad", id_tts_pad);
    cp_n_layer  = vqa::gguf::u32(gu, "qwen3omni.cp.block_count", cp_n_layer);
    cp_n_vocab  = vqa::gguf::u32(gu, "qwen3omni.cp.vocab_size", cp_n_vocab);
    c2w_n_layer = vqa::gguf::u32(gu, "qwen3omni.c2w.block_count", c2w_n_layer);
    c2w_hidden  = vqa::gguf::u32(gu, "qwen3omni.c2w.hidden_size", c2w_hidden);
    c2w_window  = vqa::gguf::u32(gu, "qwen3omni.c2w.sliding_window", c2w_window);
    c2w_decoder_dim = vqa::gguf::u32(gu, "qwen3omni.c2w.decoder_dim", c2w_decoder_dim);
    c2w_codebook = vqa::gguf::u32(gu, "qwen3omni.c2w.codebook_size", c2w_codebook);
    for (auto & kv : vqa::gguf::u32_by_prefix(gu, "qwen3omni.speaker."))
        speaker_ids[kv.first] = kv.second;

    vqa::weights::bind_from_mmap(gu, wctx, *mm);
    gguf_free(gu);
    static vqa::AlignedBuf host_f32_storage;  // function-static: outlives load()
    host_f32 = &host_f32_storage;
    vqa::weights::promote_1d_to_f32(wctx, host_f32_storage);
    // bind FIRST: every host-side weight read below needs tensors in buffers
    vqa::weights::Bound bound = vqa::weights::bind_to_backend(wctx, backend, *mm, {});
    if (!bound.ok) return false;
    wbuf = bound.main_buf;
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) fprintf(stderr, "talker: missing tensor %s\n", n);
        return t;
    };
    token_embd = gt("tk.token_embd.weight");
    out_norm   = gt("tk.output_norm.weight");
    tp_fc1_w = gt("tk.text_projection.fc1.weight"); tp_fc1_b = gt("tk.text_projection.fc1.bias");
    tp_fc2_w = gt("tk.text_projection.fc2.weight"); tp_fc2_b = gt("tk.text_projection.fc2.bias");
    hp_fc1_w = gt("tk.hidden_projection.fc1.weight"); hp_fc1_b = gt("tk.hidden_projection.fc1.bias");
    hp_fc2_w = gt("tk.hidden_projection.fc2.weight"); hp_fc2_b = gt("tk.hidden_projection.fc2.bias");
    if (!token_embd || !out_norm || !tp_fc1_w || !hp_fc1_w) return false;
    codec_head = gt("tk.codec_head.weight");
    if (!codec_head) return false;
    for (int i = 0; i < 15; i++) {
        char b2[96];
        snprintf(b2, sizeof(b2), "cp.head.%d.weight", i);
        cp_head[i] = gt(b2);
    }

    tl.resize((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "tk.blk.%d.%s", i, suf); return gt(b); };
        TkLayer & L = tl[i];
        L.norm = g("attn_norm.weight");
        L.q = g("attn_q.weight"); L.k = g("attn_k.weight");
        L.v = g("attn_v.weight"); L.o = g("attn_output.weight");
        L.qn = g("attn_q_norm.weight"); L.kn = g("attn_k_norm.weight");
        L.post_norm = g("ffn_norm.weight");
        L.router = g("ffn_router.weight");
        L.gate_exps = g("ffn_gate_exps.weight");
        L.up_exps = g("ffn_up_exps.weight");
        L.down_exps = g("ffn_down_exps.weight");
        L.sh_gate = g("ffn_shared_gate_proj.weight");
        L.sh_up = g("ffn_shared_up_proj.weight");
        L.sh_down = g("ffn_shared_down_proj.weight");
        L.sh_gate_scalar = g("ffn_shared_gate.weight");
        if (!L.norm || !L.q || !L.router || !L.gate_exps || !L.sh_gate || !L.sh_gate_scalar)
            return false;
    }
    for (int k = 0; k < n_groups - 1; k++) {
        char b[96];
        snprintf(b, sizeof(b), "cp.token_embd.%d.weight", k);
        cp_emb[k] = gt(b);
        if (!cp_emb[k]) return false;
    }
    cp_out_norm = gt("cp.output_norm.weight");
    cl.resize((size_t)cp_n_layer);
    for (int i = 0; i < cp_n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "cp.blk.%d.%s", i, suf); return gt(b); };
        CpLayer & L = cl[i];
        L.norm = g("attn_norm.weight");
        L.q = g("attn_q.weight"); L.k = g("attn_k.weight");
        L.v = g("attn_v.weight"); L.o = g("attn_output.weight");
        L.qn = g("attn_q_norm.weight"); L.kn = g("attn_k_norm.weight");
        L.post_norm = g("ffn_norm.weight");
        L.gate = g("ffn_gate.weight"); L.up = g("ffn_up.weight"); L.down = g("ffn_down.weight");
        if (!L.norm || !L.q || !L.gate) return false;
    }

    c2w_emb = gt("c2w.code_embedding.weight");
    c2w_norm = gt("c2w.blk.0.attn_norm.weight");  // placeholder probe below
    c2l.resize((size_t)c2w_n_layer);
    bool have_c2w = c2w_emb != nullptr;
    if (have_c2w) {
        for (int i = 0; i < c2w_n_layer; i++) {
            char b[96];
            auto g = [&](const char * suf) { snprintf(b, sizeof(b), "c2w.blk.%d.%s", i, suf); return gt(b); };
            C2wLayer & L = c2l[i];
            L.attn_norm = g("attn_norm.weight");
            L.q = g("attn_q.weight"); L.k = g("attn_k.weight"); L.v = g("attn_v.weight");
            L.o_w = g("attn_output.weight"); L.o_b = g("attn_output.bias");
            L.attn_scale = g("attn_layer_scale.weight");
            L.mlp_scale = g("ffn_layer_scale.weight");
            L.ffn_norm = g("ffn_norm.weight");
            L.fc1_w = g("ffn_gate.weight"); L.fc1_b = g("ffn_gate.bias");
            L.up_w = g("ffn_up.weight");
            L.fc2_w = g("ffn_down.weight"); L.fc2_b = g("ffn_down.bias");
            if (!L.attn_norm || !L.q || !L.up_w || !L.attn_scale) { have_c2w = false; break; }
        }
    }
    if (have_c2w) {
        c2w_norm = gt("c2w.output_norm.weight");
        // upsample: transconv k2 s2 + ConvNeXt (dwconv k7 groups=dim, LN, pw 4d, GELU, gamma)
        for (int u = 0; u < 2; u++) {
            char b[96], w[128];
            snprintf(b, sizeof(b), "c2w.upsample.%d.0", u);
            snprintf(w, sizeof(w), "%s.conv.weight", b);
            char wb[136]; snprintf(wb, sizeof(wb), "%s.conv.bias", b);
            if (!conv_from_gguf(wctx, w, wb, 1, 2, 1, up_tc[u], true)) return false;
            ConvNeXt & cx = up_cnx[u];
            snprintf(w, sizeof(w), "c2w.upsample.%d.1.dwconv.conv.weight", u);
            snprintf(wb, sizeof(wb), "c2w.upsample.%d.1.dwconv.conv.bias", u);
            if (!conv_from_gguf(wctx, w, wb, 1, 1, c2w_hidden, cx.dw)) return false;
            snprintf(w, sizeof(w), "c2w.upsample.%d.1.norm.weight", u);
            snprintf(wb, sizeof(wb), "c2w.upsample.%d.1.norm.bias", u);
            f16_to_host(ggml_get_tensor(wctx, w), cx.ln_w);
            f16_to_host(ggml_get_tensor(wctx, wb), cx.ln_b);
            char pw[128];
            snprintf(pw, sizeof(pw), "c2w.upsample.%d.1", u);
            char t1[160]; snprintf(t1, sizeof(t1), "%s.pwconv1.weight", pw);
            char t2[160]; snprintf(t2, sizeof(t2), "%s.pwconv1.bias", pw);
            char t3[160]; snprintf(t3, sizeof(t3), "%s.pwconv2.weight", pw);
            char t4[160]; snprintf(t4, sizeof(t4), "%s.pwconv2.bias", pw);
            cx.pw1_w = gt(t1); cx.pw1_b = gt(t2); cx.pw2_w = gt(t3); cx.pw2_b = gt(t4);
            snprintf(t1, sizeof(t1), "c2w.upsample.%d.1.gamma", u);
            f16_to_host(ggml_get_tensor(wctx, t1), cx.gamma);
            if (!cx.pw1_w || !cx.pw2_w) return false;
        }
        // decoder tree (4 blocks: rates 8,5,4,3)
        if (!conv_from_gguf(wctx, "c2w.decoder.0.conv.weight", "c2w.decoder.0.conv.bias", 1, 1, 1, dec_head))
            return false;
        const int rates[4] = {8, 5, 4, 3};
        for (int d = 0; d < 4; d++) {
            DecBlock & D = dec[d];
            char b[96], t[160];
            snprintf(b, sizeof(b), "c2w.decoder.%d", d + 1);
            snprintf(t, sizeof(t), "%s.block.0.alpha", b); f16_to_host(ggml_get_tensor(wctx, t), D.s_alpha);
            snprintf(t, sizeof(t), "%s.block.0.beta", b);  f16_to_host(ggml_get_tensor(wctx, t), D.s_beta);
            snprintf(t, sizeof(t), "%s.block.1.conv.weight", b);
            char tb[176]; snprintf(tb, sizeof(tb), "%s.block.1.conv.bias", b);
            if (!conv_from_gguf(wctx, t, tb, 1, rates[d], 1, D.tc, true)) return false;
            const int dil[3] = {1, 3, 9};
            for (int r = 0; r < 3; r++) {
                snprintf(t, sizeof(t), "%s.block.%d.act1.alpha", b, r + 2);
                f16_to_host(ggml_get_tensor(wctx, t), D.ru[r].a1_alpha);
                snprintf(t, sizeof(t), "%s.block.%d.act1.beta", b, r + 2);
                f16_to_host(ggml_get_tensor(wctx, t), D.ru[r].a1_beta);
                snprintf(t, sizeof(t), "%s.block.%d.act2.alpha", b, r + 2);
                f16_to_host(ggml_get_tensor(wctx, t), D.ru[r].a2_alpha);
                snprintf(t, sizeof(t), "%s.block.%d.act2.beta", b, r + 2);
                f16_to_host(ggml_get_tensor(wctx, t), D.ru[r].a2_beta);
                snprintf(t, sizeof(t), "%s.block.%d.conv1.conv.weight", b, r + 2);
                snprintf(tb, sizeof(tb), "%s.block.%d.conv1.conv.bias", b, r + 2);
                if (!conv_from_gguf(wctx, t, tb, dil[r], 1, 1, D.ru[r].c1)) return false;
                snprintf(t, sizeof(t), "%s.block.%d.conv2.conv.weight", b, r + 2);
                snprintf(tb, sizeof(tb), "%s.block.%d.conv2.conv.bias", b, r + 2);
                if (!conv_from_gguf(wctx, t, tb, 1, 1, 1, D.ru[r].c2)) return false;
            }
        }
        f16_to_host(ggml_get_tensor(wctx, "c2w.decoder.5.alpha"), tail_alpha);
        f16_to_host(ggml_get_tensor(wctx, "c2w.decoder.5.beta"), tail_beta);
        if (!conv_from_gguf(wctx, "c2w.decoder.6.conv.weight", "c2w.decoder.6.conv.bias", 1, 1, 1, dec_tail))
            return false;
    }

    loaded = true;
    fprintf(stderr, "talker: ready (%d layers, %d/%d experts + shared, cp %dL, c2w %dL%s)\n",
            n_layer, n_expert_used, n_expert, cp_n_layer, c2w_n_layer,
            have_c2w ? "" : " (c2w MISSING)");
    return true;
}



static int D_(Talker * t) { return t->n_embd; }



// ---------------------------------------------------------------- talker trunk
struct TkGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * emb_in = nullptr;  // [1024, T] F32
    ggml_tensor * mpos_in = nullptr;  // [4T]
    ggml_tensor * mask_in = nullptr;
    ggml_tensor * logits = nullptr;  // [3072, last]
    ggml_tensor * hidden_out = nullptr;  // [1024, T] dup'd last-layer pre-norm? post final norm
};

struct TkRuntime {
    Talker * tk = nullptr;
    vqa::KvCache kv;          // talker trunk
    vqa::KvCache cp_kv;       // predictor
    int n_past = 0, cp_past = 0;
    TkGraph g;
    bool build(Talker * t, int T, int S);
    bool run(Talker * t, const float * emb, const int * mpos, std::vector<float> & logits,
             std::vector<float> * hidden_all);
};

bool TkRuntime::build(Talker * t, int T, int S) {
    tk = t;
    g.emb_in = nullptr; g.mpos_in = nullptr; g.mask_in = nullptr;
    g.logits = nullptr; g.hidden_out = nullptr;
    struct ggml_init_params ip = { 32ull * 1024 * 1024, NULL, true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    g.gf = ggml_new_graph_custom(g.ctx, 16384, false);
    ggml_context * c = g.ctx;
    const int D = t->n_embd, HD = t->head_dim, NH = t->n_head, NKV = t->n_kv_head;

    g.emb_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
    g.mpos_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * T);
    g.mask_in = ggml_new_tensor_2d(c, GGML_TYPE_F16, S, T);

    ggml_tensor * x = g.emb_in;
    ggml_tensor * pos_rope = ggml_view_1d(c, g.mpos_in, 4 * T, 0);
    ggml_tensor * pos_slot = ggml_view_1d(c, g.mpos_in, T, (size_t)3 * T * sizeof(int));

    for (int il = 0; il < t->n_layer; il++) {
        TkLayer & L = t->tl[il];
        ggml_tensor * l = ggml_rms_norm(c, x, t->rms_eps);
        l = ggml_mul(c, l, L.norm);
        ggml_tensor * q = ggml_mul_mat(c, L.q, l);
        ggml_tensor * k = ggml_mul_mat(c, L.k, l);
        ggml_tensor * v = ggml_mul_mat(c, L.v, l);
        ggml_tensor * q3 = ggml_reshape_3d(c, q, HD, NH, T);
        q3 = ggml_mul(c, ggml_rms_norm(c, q3, t->rms_eps), L.qn);
        ggml_tensor * k3 = ggml_reshape_3d(c, k, HD, NKV, T);
        k3 = ggml_mul(c, ggml_rms_norm(c, k3, t->rms_eps), L.kn);
        q3 = ggml_rope_multi(c, q3, pos_rope, nullptr, HD, t->mrope_sections,
                             GGML_ROPE_TYPE_IMROPE, 8192, t->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_multi(c, k3, pos_rope, nullptr, HD, t->mrope_sections,
                             GGML_ROPE_TYPE_IMROPE, 8192, t->rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * k2 = ggml_reshape_2d(c, k3, NKV * HD, T);
        ggml_tensor * v2 = ggml_reshape_2d(c, v, NKV * HD, T);
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, kv.k[il], k2, pos_slot));
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, kv.v[il], v2, pos_slot));
        ggml_tensor * kf = ggml_view_3d(c, kv.k[il], HD, S, NKV, kv.k[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * vf = ggml_view_3d(c, kv.v[il], HD, S, NKV, kv.v[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * qf = ggml_view_3d(c, q3, HD, T, NH, q3->nb[2], q3->nb[1], 0);
        ggml_tensor * fa = ggml_flash_attn_ext(c, qf, kf, vf, g.mask_in, 1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(c, ggml_cont(c, fa), NH * HD, T);
        attn = ggml_mul_mat(c, L.o, attn);
        x = ggml_add(c, x, attn);

        // MoE: top-6 routed + sigmoid-gated shared expert
        ggml_tensor * m = ggml_mul(c, ggml_rms_norm(c, x, t->rms_eps), L.post_norm);
        ggml_tensor * rlogits = ggml_mul_mat(c, L.router, m);
        ggml_tensor * sel = ggml_top_k(c, rlogits, t->n_expert_used);
        ggml_tensor * sel_logits = ggml_get_rows(c, ggml_reshape_3d(c, rlogits, 1, t->n_expert, T), sel);
        sel_logits = ggml_reshape_2d(c, sel_logits, t->n_expert_used, T);
        ggml_tensor * w6 = ggml_reshape_3d(c, ggml_soft_max(c, sel_logits), 1, t->n_expert_used, T);
        ggml_tensor * cur3 = ggml_reshape_3d(c, m, D, 1, T);
        ggml_tensor * gu = ggml_mul_mat_id(c, L.gate_exps, cur3, sel);
        ggml_tensor * up = ggml_mul_mat_id(c, L.up_exps, cur3, sel);
        ggml_tensor * act = ggml_swiglu_split(c, gu, up);
        ggml_tensor * ff3 = ggml_mul_mat_id(c, L.down_exps, act, sel);
        ggml_tensor * wb = ggml_view_3d(c, w6, 1, t->n_expert_used, T, w6->nb[1], w6->nb[2], 0);
        ggml_tensor * prod = ggml_mul(c, ff3, wb);
        ggml_tensor * pk = ggml_cont(c, ggml_permute(c, prod, 1, 0, 2, 3));
        ggml_tensor * moe = ggml_reshape_2d(c, ggml_sum_rows(c, pk), D, T);
        // shared expert: SwiGLU [gate,up,down] then sigmoid(scalar gate)*out
        ggml_tensor * sg = ggml_mul_mat(c, L.sh_gate, m);
        ggml_tensor * su = ggml_mul_mat(c, L.sh_up, m);
        ggml_tensor * sd = ggml_mul_mat(c, L.sh_down, ggml_swiglu_split(c, sg, su));
        ggml_tensor * gate_row = ggml_mul_mat(c, L.sh_gate_scalar, m);  // [1, T]
        gate_row = ggml_sigmoid(c, gate_row);
        sd = ggml_mul(c, sd, gate_row);
        x = ggml_add(c, x, ggml_add(c, moe, sd));
    }
    x = ggml_mul(c, ggml_rms_norm(c, x, t->rms_eps), t->out_norm);
    g.hidden_out = ggml_dup(c, x);
    ggml_set_name(g.hidden_out, "tk_hidden");
    ggml_build_forward_expand(g.gf, g.hidden_out);
    // head = INDEPENDENT codec_head from the checkpoint (HF tie_weights does
    // NOT tie when both sides exist with different values - maxdiff 0.148)
    ggml_tensor * last = ggml_view_2d(c, x, D, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
    g.logits = ggml_mul_mat(c, t->codec_head, last);
    ggml_set_name(g.logits, "tk_logits");
    ggml_build_forward_expand(g.gf, g.logits);
    return true;
}

bool TkRuntime::run(Talker * t, const float * emb, const int * mpos, std::vector<float> & logits,
                    std::vector<float> * hidden_all) {
    const int T = (int)g.emb_in->ne[1];
    const int S = n_past + T;
    if (!ggml_gallocr_alloc_graph(t->allocr, g.gf)) return false;
    ggml_backend_tensor_set(g.emb_in, emb, 0, sizeof(float) * (size_t)D_(t) * T);
    ggml_backend_tensor_set(g.mpos_in, mpos, 0, sizeof(int) * 4 * T);
    vqa::set_causal_mask(g.mask_in, n_past, T);
    if (ggml_backend_graph_compute(t->backend, g.gf) != GGML_STATUS_SUCCESS) return false;
    logits.resize(t->n_vocab);
    ggml_backend_tensor_get(g.logits, logits.data(), 0, sizeof(float) * logits.size());
    if (hidden_all) {
        hidden_all->resize((size_t)t->n_embd * T);
        ggml_backend_tensor_get(g.hidden_out, hidden_all->data(), 0, sizeof(float) * hidden_all->size());
    }
    n_past = S;
    ggml_free(g.ctx);
    g.ctx = nullptr;
    return true;
}



// ---------------------------------------------------------------- predictor
// 5-layer dense trunk, KV cached; per-group embedding table in, per-group
// head (tied to the same table, embedding side wins) out.

struct CpRuntime {
    Talker * tk = nullptr;
    vqa::KvCache kv;
    // static T=1 graph: the same graph serves prefill rows and all steps
    // (head picked by an I32 leaf through mul_mat_id, KV fixed to CP_KV_MAX
    // slots with a host mask) - no rebuild/alloc between steps
    ggml_context * ctx = nullptr, * ictx = nullptr;
    ggml_gallocr_t allocr = nullptr;  // dedicated pool: nothing else reallocates it
    ggml_backend_buffer_t ibuf = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * emb_in = nullptr, * pos_in = nullptr, * mask_in = nullptr;
    ggml_tensor * w_cur = nullptr;          // [D, vocab] head written per step from host
    std::vector<std::vector<float>> head_rows; // 15 x [D*vocab] host copies
    ggml_tensor * hidden_out = nullptr, * logits = nullptr;
    ggml_tensor * heads3d = nullptr;      // [D, vocab, 15] F32 stacked heads
    int n_past = 0;
    static constexpr int CP_KV_MAX = 16;

    bool init(Talker * t);                // build the static graph once
    bool run(Talker * t, const float * emb, int pos, int head_group,
             std::vector<float> & hidden_out, std::vector<float> * logits_out = nullptr);
    // RQ step: prefill [talker_hidden, emb0(code0)] then 14 more steps
    bool rq_block(Talker * t, const std::vector<float> & talker_hidden,
                  const float * code0_embed, std::vector<int> & codes,
                  std::vector<float> & mid_hiddens, uint64_t & rng);
};
bool CpRuntime::init(Talker * t) {
    tk = t;
    const int D = t->n_embd, HD = t->cp_head_dim, NH = t->cp_n_head, NKV = t->cp_n_kv_head;
   // stacked heads [D, vocab, 15] F32 (heads are F32 in the gguf; dequant
    // whatever is there via the get-rows trick so mul_mat_id sees one tensor)
    {
        std::vector<float> all((size_t)D * t->cp_n_vocab * 15);
        ggml_context * c2 = ggml_init({ 8 << 20, nullptr, true });
        ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(t->backend));
        head_rows.resize(15);
        for (int g = 0; g < 15; g++) {
            ggml_tensor * w = t->cp_head[g];
            head_rows[(size_t)g].resize((size_t)D * t->cp_n_vocab);
            std::vector<float> & rows = head_rows[(size_t)g];
            if (w->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(w, rows.data(), 0, rows.size() * 4);
            } else {
                const int nr = (int)w->ne[1];
                ggml_tensor * ids = ggml_new_tensor_1d(c2, GGML_TYPE_I32, nr);
                std::vector<int> idv(nr);
                for (int q = 0; q < nr; q++) idv[(size_t)q] = q;
                ggml_tensor * src = ggml_cast(c2, w, GGML_TYPE_F32);
                ggml_tensor * rr = ggml_get_rows(c2, src, ids);
                ggml_cgraph * g2 = ggml_new_graph(c2);
                ggml_build_forward_expand(g2, rr);
                ggml_gallocr_alloc_graph(al, g2);
                ggml_backend_tensor_set(ids, idv.data(), 0, sizeof(int) * nr);
                ggml_backend_graph_compute(t->backend, g2);
                ggml_backend_tensor_get(rr, rows.data(), 0, rows.size() * 4);
            }
        }
        ggml_gallocr_free(al);
        ggml_free(c2);
        ictx = ggml_init({ 16ull << 20, nullptr, true });
        if (!ictx) return false;
        const int64_t wne[2] = { D, (int64_t)t->cp_n_vocab };
        w_cur = ggml_new_tensor(ictx, GGML_TYPE_F32, 2, wne);
        ibuf = ggml_backend_alloc_ctx_tensors(ictx, t->backend);
        if (!ibuf) return false;
    }
    ctx = ggml_init({ (32ull << 20), nullptr, true });
    if (!ctx) return false;
    gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_context * c = ctx;

    emb_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, 1);
    pos_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    mask_in = ggml_new_tensor_2d(c, GGML_TYPE_F16, CP_KV_MAX, 1);

    ggml_tensor * x = emb_in;
    for (int il = 0; il < t->cp_n_layer; il++) {
        CpLayer & L = t->cl[il];
        ggml_tensor * l = ggml_mul(c, ggml_rms_norm(c, x, t->cp_rms_eps), L.norm);
        ggml_tensor * q = ggml_mul_mat(c, L.q, l);
        ggml_tensor * k = ggml_mul_mat(c, L.k, l);
        ggml_tensor * v = ggml_mul_mat(c, L.v, l);
        ggml_tensor * q3 = ggml_reshape_3d(c, q, HD, NH, 1);
        q3 = ggml_mul(c, ggml_rms_norm(c, q3, t->cp_rms_eps), L.qn);
        ggml_tensor * k3 = ggml_reshape_3d(c, k, HD, NKV, 1);
        k3 = ggml_mul(c, ggml_rms_norm(c, k3, t->cp_rms_eps), L.kn);
        q3 = ggml_rope_multi(c, q3, pos_in, nullptr, HD, nullptr,
                             GGML_ROPE_TYPE_NEOX, 8192, 1e6f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_multi(c, k3, pos_in, nullptr, HD, nullptr,
                             GGML_ROPE_TYPE_NEOX, 8192, 1e6f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * k2 = ggml_reshape_2d(c, k3, NKV * HD, 1);
        ggml_tensor * v2 = ggml_reshape_2d(c, v, NKV * HD, 1);
        ggml_build_forward_expand(gf, ggml_set_rows(c, kv.k[il], k2, pos_in));
        ggml_build_forward_expand(gf, ggml_set_rows(c, kv.v[il], v2, pos_in));
       // fixed CP_KV_MAX views; stale slots are hidden by the mask leaf
        ggml_tensor * kf = ggml_view_3d(c, kv.k[il], HD, CP_KV_MAX, NKV, kv.k[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * vf = ggml_view_3d(c, kv.v[il], HD, CP_KV_MAX, NKV, kv.v[il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * qf = ggml_view_3d(c, q3, HD, 1, NH, q3->nb[2], q3->nb[1], 0);
        ggml_tensor * fa = ggml_flash_attn_ext(c, qf, kf, vf, mask_in, 1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(c, ggml_cont(c, fa), NH * HD, 1);
        x = ggml_add(c, x, ggml_mul_mat(c, L.o, attn));
        ggml_tensor * m = ggml_mul(c, ggml_rms_norm(c, x, t->cp_rms_eps), L.post_norm);
        m = ggml_mul_mat(c, L.down, ggml_swiglu_split(c, ggml_mul_mat(c, L.gate, m), ggml_mul_mat(c, L.up, m)));
        x = ggml_add(c, x, m);
    }
    x = ggml_mul(c, ggml_rms_norm(c, x, t->cp_rms_eps), t->cp_out_norm);
    hidden_out = ggml_dup(c, x);
    ggml_set_name(hidden_out, "cp_hidden");
    ggml_build_forward_expand(gf, hidden_out);
    logits = ggml_mul_mat(c, w_cur, x);
    ggml_set_name(logits, "cp_logits");
    ggml_build_forward_expand(gf, logits);
    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(t->backend));
    if (!allocr || !ggml_gallocr_alloc_graph(allocr, gf)) return false;
    return true;
}

// one static-graph step: upload 4 leaves (emb row, pos, head id, 64-fp16
// mask), compute, read hidden + logits. No rebuild, no realloc.
bool CpRuntime::run(Talker * t, const float * emb, int pos, int head_group,
                    std::vector<float> & hidden_out, std::vector<float> * logits_out) {
    ggml_backend_tensor_set(emb_in, emb, 0, sizeof(float) * (size_t)t->n_embd);
    ggml_backend_tensor_set(pos_in, &pos, 0, sizeof(int));
    const auto h0 = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(w_cur, head_rows[(size_t)head_group].data(), 0,
                            head_rows[(size_t)head_group].size() * 4);
    {
       // attention over slots [0, pos]; everything else masked out
        std::vector<ggml_fp16_t> mk((size_t)CP_KV_MAX, ggml_fp32_to_fp16(-INFINITY));
        for (int i = 0; i <= pos; i++) mk[(size_t)i] = ggml_fp32_to_fp16(0.0f);
        ggml_backend_tensor_set(mask_in, mk.data(), 0, sizeof(ggml_fp16_t) * mk.size());
    }
    const auto s0 = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(t->backend, gf) != GGML_STATUS_SUCCESS) return false;
    hidden_out.resize((size_t)t->n_embd);
    ggml_backend_tensor_get(this->hidden_out, hidden_out.data(), 0, sizeof(float) * hidden_out.size());
    if (logits_out) {
        logits_out->resize(t->cp_n_vocab);
        ggml_backend_tensor_get(logits, logits_out->data(), 0, sizeof(float) * logits_out->size());
    }
    n_past = pos + 1;
    return true;
}


// predictor mirrors the talker's sampling (prepare_inputs: top_k 50, top_p
// 0.8); greedy available via temp 0 for deterministic parity runs
static int cp_sample(const std::vector<float> & logits, uint64_t & rng,
                     float temp = 0.0f) {
    vqa::omni::CodecSampler s(2048, temp, 50, 0.8f, 1.0f);
    return s.sample(logits, nullptr, rng);
}

bool CpRuntime::rq_block(Talker * t, const std::vector<float> & talker_hidden,
                         const float * code0_embed, std::vector<int> & codes,
                         std::vector<float> & mid_hiddens, uint64_t & rng) {
    // official prepare_inputs_for_generation stepping, on the STATIC T=1
    // graph: two prefill rows (talker hidden at slot 0, emb(code0) at slot 1,
    // head group 0), then 14 steps (input = table[g-1](prev), head = table[g])
    const int D = t->n_embd;
    mid_hiddens.clear();
    std::vector<float> logits, hid;
    if (!ctx && !init(t)) return false;

    if (!run(t, talker_hidden.data(), 0, 0, hid, nullptr)) return false;
    if (!run(t, code0_embed, 1, 0, hid, &logits)) return false;
    codes.push_back(cp_sample(logits, rng));

    for (int gstep = 1; gstep < 15; gstep++) {
        const int prev = codes.back();
        ggml_tensor * tbl = t->cp_emb[gstep - 1];
        std::vector<float> row((size_t)D);
        if (tbl->type == GGML_TYPE_F16) {
            std::vector<uint16_t> tmp(D);
            ggml_backend_tensor_get(tbl, tmp.data(),
                                    (size_t)prev * D * sizeof(uint16_t), D * sizeof(uint16_t));
            for (int i = 0; i < D; i++) row[(size_t)i] = ggml_fp16_to_fp32(tmp[(size_t)i]);
        } else {
            ggml_backend_tensor_get(tbl, row.data(), (size_t)prev * D * sizeof(float), sizeof(float) * D);
        }
        if (!run(t, row.data(), gstep + 1, gstep, hid, &logits)) return false;
        codes.push_back(cp_sample(logits, rng));
    }
    return true;
}

// ---------------------------------------------------------------- code2wav
// graph side: the 8-layer sliding-window transformer. Vocoder convs run host.
static bool c2w_transformer(Talker * t, const float * emb /* [1024, T] */, int T,
                            std::vector<float> & out /* [1024, T] */) {
    const int D = t->c2w_hidden, HD = t->c2w_hidden / 16, NH = 16, NKV = 16, W = t->c2w_window;
    struct ggml_init_params ip = { 32ull * 1024 * 1024, NULL, true };
    ggml_context * c = ggml_init(ip);
    if (!c) return false;
    ggml_cgraph * gf = ggml_new_graph_custom(c, 8192, false);
    ggml_tensor * in = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, T);
    ggml_tensor * pos_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, T);
    ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, T, T);

    ggml_tensor * x = in;
    for (int il = 0; il < t->c2w_n_layer; il++) {
        C2wLayer & L = t->c2l[il];
        ggml_tensor * l = ggml_mul(c, ggml_rms_norm(c, x, 1e-5f), L.attn_norm);
        ggml_tensor * q = ggml_mul_mat(c, L.q, l);
        ggml_tensor * k = ggml_mul_mat(c, L.k, l);
        ggml_tensor * v = ggml_mul_mat(c, L.v, l);
        ggml_tensor * q3 = ggml_reshape_3d(c, q, HD, NH, T);
        ggml_tensor * k3 = ggml_reshape_3d(c, k, HD, NKV, T);
        ggml_tensor * v3 = ggml_reshape_3d(c, v, HD, NKV, T);
        q3 = ggml_rope_multi(c, q3, pos_in, nullptr, HD, nullptr,
                             GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_multi(c, k3, pos_in, nullptr, HD, nullptr,
                             GGML_ROPE_TYPE_NEOX, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * qf = ggml_cont(c, ggml_permute(c, q3, 0, 2, 1, 3));
        ggml_tensor * kf = ggml_cont(c, ggml_permute(c, k3, 0, 2, 1, 3));
        ggml_tensor * vf = ggml_cont(c, ggml_permute(c, v3, 0, 2, 1, 3));
        ggml_tensor * fa = ggml_flash_attn_ext(c, qf, kf, vf, mask, 1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(c, ggml_cont(c, fa), NH * HD, T);
        attn = ggml_mul_mat(c, L.o_w, attn);
        attn = ggml_mul(c, attn, L.attn_scale);          // LayerScale
        x = ggml_add(c, x, attn);
        ggml_tensor * m = ggml_mul(c, ggml_rms_norm(c, x, 1e-5f), L.ffn_norm);
        m = ggml_mul_mat(c, L.fc2_w, ggml_swiglu_split(c, ggml_mul_mat(c, L.fc1_w, m),
                                                        ggml_mul_mat(c, L.up_w, m)));
        m = ggml_mul(c, m, L.mlp_scale);
        x = ggml_add(c, x, m);
    }
    x = ggml_mul(c, ggml_rms_norm(c, x, 1e-5f), t->c2w_norm);
    ggml_build_forward_expand(gf, x);
    if (!ggml_gallocr_alloc_graph(t->allocr, gf)) { ggml_free(c); return false; }
    ggml_backend_tensor_set(in, emb, 0, sizeof(float) * (size_t)D * T);
    {
        std::vector<int> pos(T);
        for (int i = 0; i < T; i++) pos[i] = i;
        ggml_backend_tensor_set(pos_in, pos.data(), 0, sizeof(int) * T);
        std::vector<ggml_fp16_t> mk((size_t)T * T, ggml_fp32_to_fp16(-INFINITY));
        for (int r = 0; r < T; r++) {
            const int lo = std::max(0, r - W + 1);
            for (int kc = lo; kc <= r; kc++) mk[(size_t)r * T + kc] = ggml_fp32_to_fp16(0.0f);
        }
        ggml_backend_tensor_set(mask, mk.data(), 0, sizeof(ggml_fp16_t) * mk.size());
    }
    if (ggml_backend_graph_compute(t->backend, gf) != GGML_STATUS_SUCCESS) { ggml_free(c); return false; }
    out.resize((size_t)D * T);
    ggml_tensor * xo = x;
    ggml_backend_tensor_get(xo, out.data(), 0, sizeof(float) * out.size());
    ggml_free(c);
    return true;
}


// full code2wav: codes [16][N] -> wav samples

// ---------------------------------------------------------------- graph vocoder
// The whole conv stack after the (already-graph) 8-layer transformer, as ONE
// ggml graph - the host loops were 96% of synthesize() wall time. Layout
// conventions follow the BigVGAN graph in qwen25omni_t2w.cpp:
//   * between blocks activations are CT: ne0=C channels contiguous per time
//     step (per-channel params broadcast over ne0);
//   * convs transpose to [T, C] (time contiguous per channel) first;
//   * dense/depthwise convs are im2col(F32) + mul_mat (ggml_conv_1d's F16
//     im2col caps parity), transposed convs use ggml_conv_transpose_1d with
//     kernels declared [K, OC, IC] - the host C2wConv buffers are stored
//     k-fastest, so the bytes drop in unchanged;
//   * causal left padding is a concat(zeros) along ne1 (ggml_pad only pads
//     the back), stride-1 convs keep T (ideal_length == T for all kernels).
struct C2wGraph {
    ggml_context * bctx = nullptr, * ictx = nullptr;
    ggml_backend_buffer_t ibuf = nullptr;
    ggml_gallocr_t allocr = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * h_in = nullptr;      // [D, N] token-major (ne0=D)
    ggml_tensor * wav_out = nullptr; // [1, T] F32
    int N = 0;
    std::vector<std::pair<ggml_tensor *, std::vector<char>>> uploads;
};

static bool c2w_vocoder_graph(Talker * t, C2wGraph & G, int N) {
    G.N = N;
    G.backend = t->backend;
    if (!G.allocr) G.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(G.backend));
    G.ictx = ggml_init({ 96ull << 20, nullptr, true });
    G.bctx = ggml_init({ 768ull << 20, nullptr, true });
    if (!G.ictx || !G.bctx) return false;
    G.gf = ggml_new_graph_custom(G.bctx, 131072, false);
    ggml_context * c = G.bctx, * ic = G.ictx;

    auto new_in = [&](const int64_t * ne, int nd, const void * data, size_t nb) {
        ggml_tensor * r = nd == 1 ? ggml_new_tensor_1d(ic, GGML_TYPE_F32, ne[0])
                                  : ggml_new_tensor(ic, GGML_TYPE_F32, nd, ne);
        G.uploads.emplace_back(r, std::vector<char>((const char *)data, (const char *)data + nb));
        return r;
    };
    auto vec_in = [&](const std::vector<float> & v) {
        const int64_t ne[1] = { (int64_t)v.size() };
        return new_in(ne, 1, v.data(), v.size() * 4);
    };
    auto zeros_ct = [&](int C, int T) {
        std::vector<float> z((size_t)C * T, 0.0f);
        const int64_t ne[2] = { C, T };
        return new_in(ne, 2, z.data(), z.size() * 4);
    };
    auto toCT = [&](ggml_tensor * x) { return ggml_cont(c, ggml_transpose(c, x)); };

    // dense causal conv: CT in -> CT out (stride 1, dilation d, groups=1)
    auto conv_causal = [&](const C2wConv & cc, ggml_tensor * x) {
        const int Ci = (int)x->ne[0];
        const int pad_left = (cc.k - 1) * cc.dilation;
        ggml_tensor * xp = x;
        if (pad_left > 0) xp = ggml_concat(c, zeros_ct(Ci, pad_left), x, 1);
        // host bytes are [OC, IC, K] k-fastest == ggml [K, IC, OC]
        const int64_t wne[3] = { cc.k, (int64_t)cc.ci, (int64_t)cc.co };
        ggml_tensor * w = new_in(wne, 3, cc.w.data(), cc.w.size() * 4);
        ggml_tensor * in = ggml_cont(c, ggml_transpose(c, xp));         // [T+pad, Ci]
        ggml_tensor * im = ggml_im2col(c, w, in, 1, 0, 0, 0, cc.dilation, 0, false, GGML_TYPE_F32);
        ggml_tensor * y = ggml_mul_mat(c,
            ggml_reshape_2d(c, im, im->ne[0], im->ne[2] * im->ne[1]),
            ggml_reshape_2d(c, w, w->ne[0] * w->ne[1], w->ne[2]));      // [T, OC]
        ggml_tensor * yc = toCT(y);
        if (cc.has_bias) yc = ggml_add(c, yc, vec_in(cc.b));
        return yc;
    };
    // depthwise causal conv (groups=C): CT in -> CT out
    auto dw_causal = [&](const C2wConv & cc, ggml_tensor * x) {
        const int Ci = (int)x->ne[0];
        const int pad_left = (cc.k - 1) * cc.dilation;
        ggml_tensor * xp = x;
        if (pad_left > 0) xp = ggml_concat(c, zeros_ct(Ci, pad_left), x, 1);
        const int64_t kne[3] = { cc.k, 1, (int64_t)cc.co };
        ggml_tensor * k = new_in(kne, 3, cc.w.data(), cc.w.size() * 4);
        ggml_tensor * in = ggml_cont(c, ggml_transpose(c, xp));          // [T+pad, C]
        ggml_tensor * nb4 = ggml_reshape_4d(c, in, in->ne[0], 1, in->ne[1], 1);
        ggml_tensor * im = ggml_im2col(c, k, nb4, 1, 0, 0, 0, cc.dilation, 0, false, GGML_TYPE_F32);
        ggml_tensor * r = ggml_mul_mat(c, im, k);                        // [OW, 1, C]
        ggml_tensor * y3 = ggml_reshape_3d(c, r, r->ne[0], r->ne[2], 1);  // [T, C]
        ggml_tensor * yc = toCT(y3);
        if (cc.has_bias) yc = ggml_add(c, yc, vec_in(cc.b));
        return yc;
    };
    // transposed conv: CT in -> CT out, trim (k-s) at BOTH ends (HF TransConvNet)
    auto transconv = [&](const C2wConv & cc, ggml_tensor * x) {
        const int r = cc.stride, k = cc.k;
        const int OC = cc.co;
        const int raw = ((int)x->ne[1] - 1) * r + k;
        const int trim = k - r, to = raw - 2 * trim;
        const int64_t wne[3] = { k, OC, (int64_t)cc.ci };                // [K, OC, IC] bytes
        ggml_tensor * w = new_in(wne, 3, cc.w.data(), cc.w.size() * 4);
        ggml_tensor * in = ggml_cont(c, ggml_transpose(c, x));           // [T, IC]
        ggml_tensor * yt = ggml_conv_transpose_1d(c, w, in, r, 0, 1);  // [raw, OC]
        ggml_tensor * yc = ggml_cont(c, ggml_view_2d(c, yt, to, OC, yt->nb[1],
                                                     (size_t)trim * yt->nb[0]));
        yc = toCT(yc);
        if (cc.has_bias) yc = ggml_add(c, yc, vec_in(cc.b));
        return yc;
    };
    // snake: x + sin(exp(a) x)^2  // (exp(b) + 1e-9), per channel
    auto snake_g = [&](const std::vector<float> & alpha, const std::vector<float> & beta, ggml_tensor * x) {
        std::vector<float> a(alpha.size()), ib(alpha.size());
        for (size_t i = 0; i < alpha.size(); i++) {
            a[i] = std::exp(alpha[i]);
            ib[i] = 1.0f / (std::exp(beta[i]) + 1e-9f);
        }
        ggml_tensor * sg = ggml_sin(c, ggml_mul(c, x, vec_in(a)));
        return ggml_add(c, x, ggml_mul(c, ggml_sqr(c, sg), vec_in(ib)));
    };
    // LayerNorm over channels (ne0), eps 1e-6
    auto ln_g = [&](const std::vector<float> & w, const std::vector<float> & b, ggml_tensor * x) {
        const float e6 = 1e-6f;
        const int64_t ne1[1] = { 1 };
        ggml_tensor * eps = new_in(ne1, 1, &e6, 4);
        ggml_tensor * m = ggml_mean(c, x);                               // [1, T]
        ggml_tensor * v = ggml_sub(c, ggml_mean(c, ggml_sqr(c, x)), ggml_sqr(c, m));
        ggml_tensor * sd = ggml_sqrt(c, ggml_add(c, v, eps));            // [1, T]
        ggml_tensor * xn = ggml_div(c, ggml_sub(c, x, m), sd);
        return ggml_add(c, ggml_mul(c, xn, vec_in(w)), vec_in(b));
    };

    // ---- chain -----------------------------------------------------------
    G.h_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, t->c2w_hidden, N);     // CT [1024, N]
    ggml_tensor * x = G.h_in;
    for (int u = 0; u < 2; u++) {
        ggml_tensor * up = transconv(t->up_tc[u], x);

        const Talker::ConvNeXt & cx = t->up_cnx[u];
        ggml_tensor * y = dw_causal(cx.dw, up);
        ggml_tensor * ln = ln_g(cx.ln_w, cx.ln_b, y);
        ggml_tensor * h1 = ggml_gelu(c, ggml_add(c, ggml_mul_mat(c, cx.pw1_w, ln), cx.pw1_b));
        ggml_tensor * pw = ggml_add(c, ggml_mul_mat(c, cx.pw2_w, h1), cx.pw2_b);
        // HF ConvNeXt residual is the BLOCK input (the transconv output),
        // not the dw-conv output
        x = ggml_add(c, up, ggml_mul(c, pw, vec_in(cx.gamma)));

    }
    x = conv_causal(t->dec_head, x);
    const int rates[4] = { 8, 5, 4, 3 };
    for (int d = 0; d < 4; d++) {
        const Talker::DecBlock & B = t->dec[d];
        (void)rates;
        x = snake_g(B.s_alpha, B.s_beta, x);
        x = transconv(B.tc, x);

        for (int ru = 0; ru < 3; ru++) {
            const Talker::ResUnit & U = B.ru[ru];
            ggml_tensor * resid = x;
            x = snake_g(U.a1_alpha, U.a1_beta, x);
            x = conv_causal(U.c1, x);

            x = snake_g(U.a2_alpha, U.a2_beta, x);

            x = ggml_add(c, conv_causal(U.c2, x), resid);

        }
    }
    x = snake_g(t->tail_alpha, t->tail_beta, x);
    x = conv_causal(t->dec_tail, x);        // [1, T] CT
    G.wav_out = ggml_clamp(c, x, -1.0f, 1.0f);
    ggml_set_name(G.wav_out, "c2w_wav");
    ggml_build_forward_expand(G.gf, G.wav_out);
    G.ibuf = ggml_backend_alloc_ctx_tensors(G.ictx, G.backend);
    if (!G.ibuf) return false;
    for (auto & up2 : G.uploads)
        ggml_backend_tensor_set(up2.first, up2.second.data(), 0, up2.second.size());
    return true;
}

static void c2w_graph_free(C2wGraph & G) {
    if (G.ibuf) { ggml_backend_buffer_free(G.ibuf); G.ibuf = nullptr; }
    if (G.ictx) { ggml_free(G.ictx); G.ictx = nullptr; }
    if (G.bctx) { ggml_free(G.bctx); G.bctx = nullptr; }
}

// graph-vocoder driver: embed + transformer (existing graphs) + one big conv
// graph. Numerically equivalent to the host path (pw graph-vs-host corr
// 0.9999 at Q8; see tools/qwen3omni_pw_test.cpp).
static bool code2wav_graph(Talker * t, const std::vector<std::vector<float>> & codes16,
                           std::vector<float> & wav, double & voc_ms) {
    const int N = (int)codes16[0].size();
    const int D = t->c2w_hidden, CB = t->c2w_codebook;
    std::vector<float> emb((size_t)D * N);
    {
        for (int n = 0; n < N; n++) {
            float * acc = &emb[(size_t)n * D];
            memset(acc, 0, sizeof(float) * D);
            for (int g = 0; g < 16; g++) {
                const int row = g * CB + (int)codes16[g][n];
                std::vector<uint16_t> tmp(D);
                ggml_backend_tensor_get(t->c2w_emb, tmp.data(), (size_t)row * D * sizeof(uint16_t),
                                        (size_t)D * sizeof(uint16_t));
                for (int i = 0; i < D; i++) acc[i] += ggml_fp16_to_fp32(tmp[i]);
            }
            for (int i = 0; i < D; i++) acc[i] /= 16.0f;
        }
    }
    std::vector<float> h;  // token-major rows [N, D] (n*D + i) == CT bytes
    if (!c2w_transformer(t, emb.data(), N, h)) return false;
    const auto t0 = std::chrono::steady_clock::now();
    C2wGraph G;
    if (!c2w_vocoder_graph(t, G, N)) { c2w_graph_free(G); return false; }
    if (!ggml_gallocr_alloc_graph(G.allocr, G.gf)) { c2w_graph_free(G); return false; }
    ggml_backend_tensor_set(G.h_in, h.data(), 0, sizeof(float) * h.size());
    const bool ok = ggml_backend_graph_compute(G.backend, G.gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        wav.resize((size_t)G.wav_out->ne[0] * G.wav_out->ne[1]);
        ggml_backend_tensor_get(G.wav_out, wav.data(), 0, sizeof(float) * wav.size());
    }
    voc_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    c2w_graph_free(G);
    return ok;
}

bool code2wav(Talker * t, const std::vector<std::vector<float>> & codes16, std::vector<float> & wav) {
    double voc_ms = 0;
    return code2wav_graph(t, codes16, wav, voc_ms);
}

// ---------------------------------------------------------------- synthesize
// Mirror of the official generate() speech path (text-only inputs; multimodal
// talker positions land with the video stage).
// SynthIn is declared in qwen3omni_talker.hpp

// ResizeMLP: fc1 -> SiLU -> fc2 as a one-shot graph (weights may be Q8_0)
static void resize_mlp(Talker * t, ggml_tensor * fc1_w, ggml_tensor * fc1_b,
                       ggml_tensor * fc2_w, ggml_tensor * fc2_b,
                       const float * in, int rows, int in_d, float * out) {
    struct ggml_init_params ip = { 4ull << 20, NULL, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, in_d, rows);
    ggml_tensor * h = ggml_mul_mat(c, fc1_w, x);
    if (fc1_b) h = ggml_add(c, h, fc1_b);
    h = ggml_silu(c, h);
    ggml_tensor * y = ggml_mul_mat(c, fc2_w, h);
    if (fc2_b) y = ggml_add(c, y, fc2_b);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(t->backend));
    ggml_gallocr_alloc_graph(al, gf);
    ggml_backend_tensor_set(x, in, 0, sizeof(float) * (size_t)in_d * rows);
    ggml_backend_graph_compute(t->backend, gf);
    ggml_backend_tensor_get(y, out, 0, sizeof(float) * (size_t)t->n_embd * rows);
    ggml_gallocr_free(al);
    ggml_free(c);
}

// read rows of an embedding table of ANY dtype (quantized thinker tables
// included) via a one-shot get_rows graph on the tensor's backend
static void rows_any(ggml_backend_t backend, ggml_tensor * t, const int * ids, int n, float * out, ggml_backend_t host = nullptr) {
    struct ggml_init_params ip = { 1ull << 20, NULL, true };
    ggml_context * c = ggml_init(ip);
    ggml_tensor * id_t = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    ggml_tensor * src = t;
    // quantized-table reads (thinker embed Q6_K) run on the CPU partner:
    // CUDA get_rows lacks Q6_K and the tensor lives in a host buffer anyway
    ggml_backend_t be = backend;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) {
        be = host ? host : backend;
        src = ggml_cast(c, t, GGML_TYPE_F32);
    }
    ggml_tensor * r = ggml_get_rows(c, src, id_t);
    ggml_cgraph * gf = ggml_new_graph(c);
    ggml_build_forward_expand(gf, r);
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(al, gf);
    ggml_backend_tensor_set(id_t, ids, 0, sizeof(int) * n);
    ggml_backend_graph_compute(be, gf);
    ggml_backend_tensor_get(r, out, 0, sizeof(float) * (size_t)n * (size_t)t->ne[0]);
    ggml_gallocr_free(al);
    ggml_free(c);
}

// read rows [r0, r0+n) of an F16 embedding-style tensor into host f32
static void rows_f16(ggml_tensor * t, int r0, int n, float * out) {
    const int d = (int)t->ne[0];
    if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> tmp((size_t)n * d);
        ggml_backend_tensor_get(t, tmp.data(), (size_t)r0 * d * sizeof(uint16_t),
                                (size_t)n * d * sizeof(uint16_t));
        for (size_t i = 0; i < tmp.size(); i++) out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        ggml_backend_tensor_get(t, out, (size_t)r0 * d * sizeof(float), sizeof(float) * n * d);
    }
}

// official generate(): temp 0.9, top_k 50, top_p 1.0, repetition_penalty
// 1.05, suppress the trailing 1024 special tokens except the codec eos
static int talker_sample(const std::vector<float> & logits, Talker * t,
                         const std::vector<int> & history, uint64_t & rng) {
    vqa::omni::CodecSampler s(t->n_vocab, 0.9f, 50, 1.0f, 1.05f,
                              t->n_vocab - 1024, t->id_codec_eos);
    return s.sample(logits, &history, rng);
}

bool synthesize(Talker * t, const SynthIn & in, std::vector<float> & wav) {
    fprintf(stderr, "[syn] enter\n");
    const int D = t->n_embd, TD = in.thinker_embd_dim;
    const std::vector<int> & ids = *in.ids;
    const std::vector<int> & gen = *in.gen_ids;
    const size_t T_total = in.traj_emb->size() / TD;
    // the LAST generated token never gets its own forward (no next-token
    // compute), so its trajectory row is legitimately absent
    if (T_total + 1 < ids.size() + gen.size()) return false;

    // ---- user part: text rows via text_projection, mm rows via hidden_projection
    const int id_mm[] = {151655, 151656, 151675};  // image  // video  // audio pad
    // user part: ONLY user-role tokens (official generate builds user_mask by
    // "role token after previous im_start" - system and assistant template
    // rows are NOT forwarded to the talker). ids[i]==872 <=> "user".
    std::vector<int> uidx;
    {
        int prev_im = -1;
        for (size_t i = 0; i < ids.size(); i++) {
            if (ids[i] == 151644 && i + 1 < ids.size()) prev_im = (int)i;
            if (prev_im >= 0 && ids[prev_im + 1] == 872) uidx.push_back((int)i);
        }
        if (uidx.empty()) for (size_t i = 0; i < ids.size(); i++) uidx.push_back((int)i);
    }
    std::vector<float> user((size_t)uidx.size() * D);
    {
        std::vector<int> mm_rows, tx_rows;  // indices into uidx
        for (size_t k = 0; k < uidx.size(); k++) {
            const int id = ids[uidx[k]];
            bool is_mm = id == id_mm[0] || id == id_mm[1] || id == id_mm[2];
            if (is_mm) mm_rows.push_back((int)k); else tx_rows.push_back((int)k);
        }
        std::vector<float> proj((size_t)tx_rows.size() * D);
        // gather embed rows for text
        std::vector<float> src((size_t)tx_rows.size() * TD);
        for (size_t k = 0; k < tx_rows.size(); k++)
            memcpy(&src[k * TD], &in.traj_emb->at((size_t)tx_rows[k] * TD), sizeof(float) * TD);
        resize_mlp(t, t->tp_fc1_w, t->tp_fc1_b, t->tp_fc2_w, t->tp_fc2_b,
                   src.data(), (int)tx_rows.size(), TD, proj.data());
        for (size_t k = 0; k < tx_rows.size(); k++)
            memcpy(&user[(size_t)tx_rows[k] * D], &proj[k * D], sizeof(float) * D);
        if (!mm_rows.empty()) {
            std::vector<float> src((size_t)mm_rows.size() * TD);
            for (size_t k = 0; k < mm_rows.size(); k++)
                memcpy(&src[k * TD], &in.traj_h24->at((size_t)mm_rows[k] * TD), sizeof(float) * TD);
            std::vector<float> proj((size_t)mm_rows.size() * D);
            resize_mlp(t, t->hp_fc1_w, t->hp_fc1_b, t->hp_fc2_w, t->hp_fc2_b,
                       src.data(), (int)mm_rows.size(), TD, proj.data());
            for (size_t k = 0; k < mm_rows.size(); k++)
                memcpy(&user[(size_t)mm_rows[k] * D], &proj[k * D], sizeof(float) * D);
        }
    }

    fprintf(stderr, "[syn] user done\n");
    // ---- assistant part (last im_start+assistant turn)
    int as = -1;
    for (size_t i = 0; i + 1 < ids.size(); i++)
        if (ids[i] == 151644 && ids[i + 1] == 77091) as = (int)i;
    if (as < 0) return false;
    std::vector<float> tts_special(3 * TD);
    {
        const int tts_ids[3] = {t->id_tts_bos, t->id_tts_eos, t->id_tts_pad};
        rows_any(t->backend, in.thinker_embd, tts_ids, 3, tts_special.data(), t->host);
    }
    std::vector<float> tts_proj(3 * D);
    resize_mlp(t, t->tp_fc1_w, t->tp_fc1_b, t->tp_fc2_w, t->tp_fc2_b,
               tts_special.data(), 3, TD, tts_proj.data());
    const float * bos_e = &tts_proj[0], *eos_e = &tts_proj[D], *pad_e = &tts_proj[2 * D];

    // full assistant text rows: [as, as+1, as+2] + gen tokens, projected
    const int n_atx = 3 + (int)gen.size();
    std::vector<float> atx_src((size_t)n_atx * TD);
    for (int k = 0; k < 3; k++)
        memcpy(&atx_src[k * TD], &in.traj_emb->at((size_t)(as + k) * TD), sizeof(float) * TD);
    const size_t n_rows = T_total;  // the last generated row may be absent
    for (size_t k = 0; k < gen.size(); k++) {
        const size_t row = std::min((size_t)(ids.size() + k), n_rows - 1);
        memcpy(&atx_src[(3 + k) * TD], &in.traj_emb->at(row * TD), sizeof(float) * TD);
    }
    std::vector<float> atx((size_t)n_atx * D);
    resize_mlp(t, t->tp_fc1_w, t->tp_fc1_b, t->tp_fc2_w, t->tp_fc2_b,
               atx_src.data(), n_atx, TD, atx.data());

    // trailing: gen rows 1..n-1 then eos
    std::vector<float> trailing(atx.begin() + 4 * D, atx.end());
    trailing.insert(trailing.end(), eos_e, eos_e + D);

    // codec special embedding rows (talker codec vocabulary)
    const int codec_ids[6] = {t->id_codec_nothink, 2156, 2157, t->id_speaker, t->id_codec_pad, t->id_codec_bos};
    std::vector<float> codec6(6 * D);
    for (int k = 0; k < 6; k++)
        rows_f16(t->token_embd, codec_ids[k], 1, &codec6[k * D]);

    // prefill matrix: user rows + [3 text, pad*4, bos, first-gen] + codec adds
    const int Tpre = (int)uidx.size() + 9;
    std::vector<float> pre((size_t)Tpre * D, 0.0f);
    memcpy(pre.data(), user.data(), sizeof(float) * user.size());
    float * a9 = pre.data() + (size_t)uidx.size() * D;
    memcpy(a9 + 0 * D, &atx[0], sizeof(float) * D);
    memcpy(a9 + 1 * D, atx.data() + 1 * D, sizeof(float) * D);
    memcpy(a9 + 2 * D, atx.data() + 2 * D, sizeof(float) * D);
    for (int k = 0; k < 4; k++) memcpy(a9 + (3 + k) * D, pad_e, sizeof(float) * D);
    memcpy(a9 + 7 * D, bos_e, sizeof(float) * D);
    memcpy(a9 + 8 * D, &atx[3 * D], sizeof(float) * D);
    for (int k = 0; k < 6; k++)
        for (int i = 0; i < D; i++) a9[(3 + k) * D + i] += codec6[k * D + i];

    std::vector<int> mpos((size_t)4 * Tpre);
    for (int i = 0; i < Tpre; i++)
        mpos[i] = mpos[Tpre + i] = mpos[2 * Tpre + i] = i, mpos[3 * Tpre + i] = i;

    fprintf(stderr, "[syn] prefill built Tpre=%d\n", Tpre);;
    if (const char * pd = getenv("QOMNI_ATX_DUMP")) {
        FILE * pf = fopen(pd, "wb");
        if (pf) { fwrite(atx.data(), 4, atx.size(), pf); fclose(pf); }
    }
    if (const char * pd = getenv("QOMNI_ATXSRC_DUMP")) {
        FILE * pf = fopen(pd, "wb");
        if (pf) { fwrite(atx_src.data(), 4, atx_src.size(), pf); fclose(pf); }
    }
    if (const char * pd = getenv("QOMNI_PRE_DUMP")) {
        FILE * pf = fopen(pd, "wb");
        if (pf) { fwrite(pre.data(), 4, pre.size(), pf); fclose(pf); }
    }
    TkRuntime tk;
    if (!tk.kv.alloc(t->backend, t->n_layer, t->n_kv_head, t->head_dim, 8192)) return false;
    if (!tk.build(t, Tpre, Tpre)) return false;
    std::vector<float> logits, hid_all;
    if (!tk.run(t, pre.data(), mpos.data(), logits, &hid_all)) return false;
    fprintf(stderr, "[syn] prefill ran\n");
    std::vector<int> codec_hist;  // recent first-group ids for rep penalty
    int c0 = talker_sample(logits, t, codec_hist, *in.rng);
    std::vector<std::vector<float>> codes16(16);
    std::vector<float> step_emb(D), step_hid;
    CpRuntime cp;  // static graph: built once, reused across frames
    if (!cp.kv.alloc(t->backend, t->cp_n_layer, t->cp_n_kv_head, t->cp_head_dim, CpRuntime::CP_KV_MAX)) return false;
    int step = 0;
    double t_tk_step = 0, t_pred = 0;
    while (c0 != t->id_codec_eos && step < in.max_codes) {
        fprintf(stderr, "[syn] step %d c0=%d\n", step, c0);
        const auto tq0 = std::chrono::steady_clock::now();
        // talker last hidden (post-norm) drives the predictor
        std::vector<float> th(hid_all.end() - D, hid_all.end());
        std::vector<float> c0e(D);
        rows_f16(t->token_embd, c0, 1, c0e.data());
        std::vector<int> codes{c0};
        std::vector<float> mid;
        if (!cp.rq_block(t, th, c0e.data(), codes, mid, *in.rng)) return false;
        t_pred += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tq0).count();
        for (int g = 0; g < 16; g++) codes16[g].push_back((float)codes[g]);

        // next talker input (official prepare_inputs_for_generation): sum of
        // 16 embedding rows - emb(c0) + cp_emb[g-1](codes[g]) for g=1..15 -
        // NOT the predictor's final-layer hiddens (HF sums hidden_states[.][0],
        // the per-step INPUT embeddings, and excludes the prefill step).
        if (step + 1 >= in.max_codes) break;
        std::vector<float> nxt(c0e);
        std::vector<float> grow(D);
        for (int g = 1; g <= 15; g++) {
            rows_f16(t->cp_emb[g - 1], codes[g], 1, grow.data());
            for (int i = 0; i < D; i++) nxt[i] += grow[i];
        }
        (void)mid;
        const float * trail_row = (step < (int)(trailing.size() / D))
                                      ? &trailing[(size_t)step * D] : pad_e;
        for (int i = 0; i < D; i++) nxt[i] += trail_row[i];

        std::vector<int> mpos1 = {Tpre + step, Tpre + step, Tpre + step, Tpre + step};
        const auto tt0 = std::chrono::steady_clock::now();
        if (!tk.build(t, 1, Tpre + step + 1)) return false;
        if (!tk.run(t, nxt.data(), mpos1.data(), logits, &hid_all)) return false;
        t_tk_step += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tt0).count();
        codec_hist.push_back(c0);
        c0 = talker_sample(logits, t, codec_hist, *in.rng);
        step++;
    }
    if (codes16[0].empty()) { wav.clear(); return true; }
    fprintf(stderr, "talker timing ms: trunk %.0f predictor %.0f\n", t_tk_step, t_pred);
    fprintf(stderr, "talker: %zu code frames\n", codes16[0].size());
    if (const char * cd = getenv("QOMNI_CODES_DUMP")) {
        FILE * cf = fopen(cd, "wb");
        if (cf) { for (int g = 0; g < 16; g++) fwrite(codes16[g].data(), 4, codes16[g].size(), cf); fclose(cf); }
    }
        return code2wav(t, codes16, wav);
}

// ---- oracle parity probes (tools/qwen3omni_pred_test.cpp) ----
bool cp_parity(Talker * t, const float * emb2, std::vector<float> & logits15,
               std::vector<float> & hidden15) {
    const int D = t->n_embd;
    logits15.clear(); hidden15.clear();
    CpRuntime cp;
    if (!cp.kv.alloc(t->backend, t->cp_n_layer, t->cp_n_kv_head, t->cp_head_dim, 64)) return false;
    if (!cp.init(t)) return false;
    std::vector<float> logits, hid;
    if (!cp.run(t, emb2, 0, 0, hid, nullptr)) return false;
    if (!cp.run(t, emb2 + D, 1, 0, hid, &logits)) return false;
    logits15.insert(logits15.end(), logits.begin(), logits.end());
    hidden15.insert(hidden15.end(), hid.begin(), hid.end());
    for (int gstep = 1; gstep < 15; gstep++) {
        int prev = 0;
        for (int i = 1; i < t->cp_n_vocab; i++) if (logits[i] > logits[prev]) prev = i;
        ggml_tensor * tbl = t->cp_emb[gstep - 1];
        std::vector<float> row((size_t)D);
        if (tbl->type == GGML_TYPE_F16) {
            std::vector<uint16_t> tmp(D);
            ggml_backend_tensor_get(tbl, tmp.data(), (size_t)prev * D * sizeof(uint16_t), D * sizeof(uint16_t));
            for (int i = 0; i < D; i++) row[(size_t)i] = ggml_fp16_to_fp32(tmp[(size_t)i]);
        } else {
            ggml_backend_tensor_get(tbl, row.data(), (size_t)prev * D * sizeof(float), sizeof(float) * D);
        }
        if (!cp.run(t, row.data(), gstep + 1, gstep, hid, &logits)) return false;
        logits15.insert(logits15.end(), logits.begin(), logits.end());
        hidden15.insert(hidden15.end(), hid.begin(), hid.end());
    }
    return true;
}

static TkRuntime * g_tk_probe = nullptr;
bool tk_parity(Talker * t, const float * emb, int T, const int * mpos,
               std::vector<float> & logits_last, std::vector<float> & hidden_all) {
    if (!g_tk_probe) {
        g_tk_probe = new TkRuntime();
        if (!g_tk_probe->kv.alloc(t->backend, t->n_layer, t->n_kv_head, t->head_dim, 8192)) return false;
    }
    if (!g_tk_probe->build(t, T, g_tk_probe->n_past + T)) return false;
    return g_tk_probe->run(t, emb, mpos, logits_last, &hidden_all);
}

// test hook: run one C2wConv through the shared causal_conv kernel
void Talker::run_c2_kernel(const std::vector<float> & in, std::vector<float> & out) {
    causal_conv(dec[0].ru[0].c2, in.data(), 768, 1272, out.data());
}

}  // namespace qwen3omni
