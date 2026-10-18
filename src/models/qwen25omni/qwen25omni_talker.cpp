// qwen25omni_talker.cpp - the talker LM: a 24-layer Qwen2-style decoder at
// width 896 that reads a dual-track stream (text-track embeddings from the
// thinker + codec-track embeddings) and emits 25 Hz audio codec tokens.
//
// Stream layout follows transformers' Qwen2_5OmniTalkerForConditionalGeneration:
//   prefill  = [prompt embeds (multimodal rows zeroed), speaker-bos embed,
//               h_1 + e_1]  with codec ids [mask x T_p, pad, bos]
//   step k   = to_proj(embed(codec_k) + (h_{k+2} + e_{k+2})), then the thinker
//              eos / text-pad embeddings once the reply runs out.
// The projector to_proj maps the 2048-wide track sum into the 896-wide trunk;
// codec_head turns the trunk output into logits over the 8448-entry codebook.
#include "qwen25omni_talker.hpp"

#include "attention.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace qwen25omni {

bool TalkerLM::load(ggml_backend_t backend_, struct ggml_context * wctx) {
    backend = backend_;
    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) fprintf(stderr, "talker: missing tensor %s\n", n);
        return t;
    };
    embed = gt("tk.embed.weight");
    out_norm = gt("tk.norm.weight");
    to_proj_w = gt("tk.to_proj.weight");
    to_proj_b = gt("tk.to_proj.bias");
    codec_head = gt("tk.codec_head.weight");
    layers.resize((size_t)n_layer);
    for (int i = 0; i < n_layer; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "tk.blk.%d.%s", i, suf); return gt(b); };
        TkLayer & L = layers[(size_t)i];
        L.norm = g("attn_norm.weight");
        L.wq = g("wq.weight"); L.bq = g("bq");
        L.wk = g("wk.weight"); L.bk = g("bk");
        L.wv = g("wv.weight"); L.bv = g("bv");
        L.wo = g("wo.weight");
        L.ffn_norm = g("ffn_norm.weight");
        L.gate = g("w_gate.weight"); L.up = g("w_up.weight"); L.down = g("w_down.weight");
        if (!L.norm || !L.wq || !L.wk || !L.wv || !L.wo || !L.ffn_norm || !L.gate || !L.up ||
            !L.down) return false;
    }
    // KV cache
    const int NKV = n_kv_head * head_dim;
    struct ggml_init_params ip = { (size_t)(n_layer * 2 + 8) * ggml_tensor_overhead(), NULL, true };
    kvctx = ggml_init(ip);
    kv.resize((size_t)n_layer);
    ggml_tensor * first = nullptr;
    {
        // allocate through one backend buffer of the same backend
        std::vector<ggml_tensor *> ts;
        for (int i = 0; i < n_layer; i++) {
            kv[(size_t)i].k = ggml_new_tensor_2d(kvctx, GGML_TYPE_F16, NKV, 8192);
            kv[(size_t)i].v = ggml_new_tensor_2d(kvctx, GGML_TYPE_F16, NKV, 8192);
            ts.push_back(kv[(size_t)i].k);
            ts.push_back(kv[(size_t)i].v);
        }
        first = ts[0];
        // one shared allocation: allocate the whole context at once
        kvbuf = ggml_backend_alloc_ctx_tensors(kvctx, backend);
        if (!kvbuf) { fprintf(stderr, "talker: kv alloc failed\n"); return false; }
        ggml_backend_buffer_clear(kvbuf, 0);
    }
    (void)first;
    fprintf(stderr, "talker: loaded (%dL d=%d vocab=%d)\n", n_layer, n_embd, n_vocab);
    return embed && out_norm && codec_head;
}

// one graph usable for both prefill (T>1) and a decode step (T==1); the codec
// ids ride their own embedding track and the reply stream comes in as [embw, T]
bool TalkerLM::build_prefill(int T, int S) {
    if (tctx) { ggml_free(tctx); tctx = nullptr; }
    struct ggml_init_params ip = { 128ull * 1024 * 1024, NULL, true };
    tctx = ggml_init(ip);
    if (!tctx) return false;
    tgf = ggml_new_graph_custom(tctx, 16384, false);

    const int HD = head_dim, NH = n_head, NKV = n_kv_head;
    g_emb_in = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, embw, T);      // text track
    ggml_set_name(g_emb_in, "tk_text");
    g_codec_in = ggml_new_tensor_1d(tctx, GGML_TYPE_I32, T);          // codec track ids
    ggml_set_name(g_codec_in, "tk_codec_ids");
    g_reply_in = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, embw, T);    // hidden part of track 1
    ggml_set_name(g_reply_in, "tk_reply");
    g_pos_in = ggml_new_tensor_1d(tctx, GGML_TYPE_I32, 4 * T);
    ggml_set_name(g_pos_in, "tk_pos");
    g_slot_in = ggml_new_tensor_1d(tctx, GGML_TYPE_I32, T);
    ggml_set_name(g_slot_in, "tk_slot");
    g_mask_in = ggml_new_tensor_2d(tctx, GGML_TYPE_F16, S, T);
    ggml_set_name(g_mask_in, "tk_mask");

    // dual-track sum: text embed + hidden reply + codec embed, then project
    ggml_tensor * x = ggml_add(tctx, g_emb_in, g_reply_in);
    x = ggml_add(tctx, x, ggml_get_rows(tctx, embed, g_codec_in));
    x = ggml_mul_mat(tctx, to_proj_w, x);
    x = ggml_add(tctx, x, to_proj_b);
    ggml_set_name(x, "tk_in_proj");

    int sec[4] = { mrope_section[0], mrope_section[1], mrope_section[2], 0 };
    for (int il = 0; il < n_layer; il++) {
        const TkLayer & L = layers[(size_t)il];
        ggml_tensor * h = ggml_rms_norm(tctx, x, eps);
        h = ggml_mul(tctx, h, L.norm);

        ggml_tensor * q = ggml_mul_mat(tctx, L.wq, h);
        q = ggml_add(tctx, q, L.bq);
        ggml_tensor * k = ggml_mul_mat(tctx, L.wk, h);
        k = ggml_add(tctx, k, L.bk);
        ggml_tensor * v = ggml_mul_mat(tctx, L.wv, h);
        v = ggml_add(tctx, v, L.bv);

        ggml_tensor * q3 = ggml_reshape_3d(tctx, q, HD, NH, T);
        ggml_tensor * k3 = ggml_reshape_3d(tctx, k, HD, NKV, T);
        q3 = ggml_rope_multi(tctx, q3, g_pos_in, nullptr, HD, sec, GGML_ROPE_TYPE_MROPE,
                             8192, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_multi(tctx, k3, g_pos_in, nullptr, HD, sec, GGML_ROPE_TYPE_MROPE,
                             8192, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor * k2 = ggml_cast(tctx, ggml_reshape_2d(tctx, k3, NKV * HD, T), GGML_TYPE_F16);
        ggml_tensor * v2 = ggml_cast(tctx, ggml_reshape_2d(tctx, v, NKV * HD, T), GGML_TYPE_F16);
        ggml_build_forward_expand(tgf, ggml_set_rows(tctx, kv[(size_t)il].k, k2, g_slot_in));
        ggml_build_forward_expand(tgf, ggml_set_rows(tctx, kv[(size_t)il].v, v2, g_slot_in));

        ggml_tensor * qf = ggml_cont(tctx, ggml_permute(tctx, q3, 0, 2, 1, 3));
        ggml_tensor * kf = ggml_view_3d(tctx, kv[(size_t)il].k, HD, S, NKV,
                                        kv[(size_t)il].k->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * vf = ggml_view_3d(tctx, kv[(size_t)il].v, HD, S, NKV,
                                        kv[(size_t)il].v->nb[1], HD * sizeof(ggml_fp16_t), 0);
        ggml_tensor * fa = ggml_flash_attn_ext(tctx, qf, kf, vf, g_mask_in,
                                               1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(tctx, ggml_cont(tctx, fa), NH * HD, T);
        attn = ggml_mul_mat(tctx, L.wo, attn);
        x = ggml_add(tctx, x, attn);

        ggml_tensor * m = ggml_rms_norm(tctx, x, eps);
        m = ggml_mul(tctx, m, L.ffn_norm);
        ggml_tensor * gate = ggml_mul_mat(tctx, L.gate, m);
        ggml_tensor * up = ggml_mul_mat(tctx, L.up, m);
        m = ggml_swiglu_split(tctx, gate, up);
        m = ggml_mul_mat(tctx, L.down, m);
        x = ggml_add(tctx, x, m);
    }

    x = ggml_rms_norm(tctx, x, eps);
    x = ggml_mul(tctx, x, out_norm);
    ggml_tensor * last = T > 1
        ? ggml_view_2d(tctx, x, n_embd, 1, x->nb[1], (size_t)(T - 1) * x->nb[1])
        : x;
    g_logits = ggml_mul_mat(tctx, codec_head, last);
    ggml_set_name(g_logits, "tk_logits");
    ggml_build_forward_expand(tgf, g_logits);

    if (!ggml_gallocr_alloc_graph(allocr, tgf)) {
        fprintf(stderr, "talker: graph alloc failed\n");
        return false;
    }
    return true;
}

bool TalkerLM::synthesize(const std::vector<int> & prompt_toks, const std::vector<int> & prompt_pos4,
                          const std::vector<int> & mm_rows, int next_pos,
                          const std::vector<int> & gen_toks,
                          const std::vector<std::vector<float>> & gen_hidden,
                          const std::vector<std::vector<float>> & gen_embed,
                          ggml_tensor * thinker_embed, int thinker_eos, int thinker_pad,
                          const std::vector<float> & spk_bos_embed,
                          float temperature, int top_k, float top_p, float rep_penalty,
                          uint64_t & rng, std::vector<int> & codec_out) {
    codec_out.clear();
    const int m = (int)gen_hidden.size();
    if (m < 2) { fprintf(stderr, "talker: need >=2 reply tokens\n"); return false; }

    // codec embedding rows (host copies)
    auto codec_row = [&](int tok, std::vector<float> & out) {
        out.resize((size_t)embw);
        read_emb_row(embed, tok, out.data());
    };
    auto thinker_row = [&](int tok, std::vector<float> & out) {
        out.resize((size_t)embw);
        read_emb_row(thinker_embed, tok, out.data());
    };

    // reply stream consumed at decode step k: h_{k+2}+e_{k+2}, then eos, pad
    std::vector<std::vector<float>> stream;
    for (int i = 2; i < m; i++) {
        std::vector<float> v((size_t)embw);
        for (int d = 0; d < embw; d++)
            v[(size_t)d] = gen_hidden[(size_t)i][(size_t)d] + gen_embed[(size_t)i][(size_t)d];
        stream.push_back(std::move(v));
    }
    {
        std::vector<float> e;
        thinker_row(thinker_eos, e);
        stream.push_back(e);
        thinker_row(thinker_pad, e);
        stream.push_back(e);
    }

    // ---- prefill: [h0+e0, bos + codec_pad, h1+e1 + codec_bos] ----
    const int T = 3;
    if (!build_prefill(T, T)) return false;
    {
        std::vector<float> text((size_t)embw * T, 0.0f);
        for (int d = 0; d < embw; d++) text[d] = gen_hidden[0][(size_t)d] + gen_embed[0][(size_t)d];
        memcpy(text.data() + (size_t)embw, spk_bos_embed.data(), (size_t)embw * sizeof(float));
        for (int d = 0; d < embw; d++)
            text[(size_t)2 * embw + d] = gen_hidden[1][(size_t)d] + gen_embed[1][(size_t)d];
        std::vector<float> cpad, cbos;
        codec_row(id_codec_pad, cpad);
        codec_row(id_codec_start, cbos);
        for (int d = 0; d < embw; d++) {
            text[(size_t)embw + d] += cpad[(size_t)d];
            text[(size_t)2 * embw + d] += cbos[(size_t)d];
        }
        ggml_backend_tensor_set(g_emb_in, text.data(), 0, sizeof(float) * text.size());
        std::vector<float> zero((size_t)embw * T, 0.0f);
        ggml_backend_tensor_set(g_reply_in, zero.data(), 0, sizeof(float) * zero.size());
        std::vector<int> ids = { id_codec_mask, id_codec_pad, id_codec_start };
        // position 0 must have NO codec track: feed the mask id and cancel it
        std::vector<float> cmask;
        codec_row(id_codec_mask, cmask);
        for (int d = 0; d < embw; d++) text[d] -= cmask[d];
        ggml_backend_tensor_set(g_emb_in, text.data(), 0, sizeof(float) * text.size());
        ggml_backend_tensor_set(g_codec_in, ids.data(), 0, sizeof(int) * ids.size());
        std::vector<int> pos((size_t)4 * T);
        for (int a = 0; a < 4; a++)
            for (int i = 0; i < T; i++) pos[(size_t)a * T + i] = next_pos + i;
        ggml_backend_tensor_set(g_pos_in, pos.data(), 0, sizeof(int) * pos.size());
        std::vector<int> slots = { 0, 1, 2 };
        ggml_backend_tensor_set(g_slot_in, slots.data(), 0, sizeof(int) * slots.size());
        vqa::set_causal_mask(g_mask_in, 0, T);
    }

    // temp/top-k/top-p sampling with repetition penalty, codec_bos suppressed
    auto sample = [&](std::vector<float> & lg, std::vector<int> & history) {
        for (int t : history) {
            if (t < 0 || t >= n_vocab) continue;
            float & v = lg[(size_t)t];
            v = v > 0 ? v / rep_penalty : v * rep_penalty;
        }
        lg[(size_t)id_codec_start] = -INFINITY;
        std::vector<std::pair<float, int>> cand((size_t)n_vocab);
        for (int i = 0; i < n_vocab; i++) cand[(size_t)i] = { lg[(size_t)i], i };
        std::sort(cand.begin(), cand.end(),
                  [](const std::pair<float, int> & a, const std::pair<float, int> & b) {
                      return a.first > b.first;
                  });
        int keep = top_k > 0 ? std::min(n_vocab, top_k) : n_vocab;
        const float inv_t = 1.0f / std::max(temperature, 1e-4f);
        std::vector<float> p((size_t)keep);
        float mx = cand[0].first, sum = 0.0f;
        for (int i = 0; i < keep; i++) {
            p[(size_t)i] = expf((cand[(size_t)i].first - mx) * inv_t);
            sum += p[(size_t)i];
        }
        for (int i = 0; i < keep; i++) p[(size_t)i] /= sum;
        if (top_p < 1.0f) {
            float acc = 0.0f;
            int n = 0;
            while (n < keep && acc < top_p) { acc += p[(size_t)n]; n++; }
            keep = std::max(1, n);
        }
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
        const float r = (float)(((rng * 0x2545F4914F6CDD1Dull) >> 11) * (1.0 / 9007199254740992.0));
        float acc = 0.0f;
        for (int i = 0; i < keep; i++) {
            acc += p[(size_t)i];
            if (r <= acc) return cand[(size_t)i].second;
        }
        return cand[(size_t)keep - 1].second;
    };

    std::vector<float> logits((size_t)n_vocab);
    if (ggml_backend_graph_compute(backend, tgf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "talker: prefill compute failed\n");
        return false;
    }
    ggml_backend_tensor_get(g_logits, logits.data(), 0, sizeof(float) * n_vocab);
    n_past = T;

    std::vector<int> history;
    int cur = sample(logits, history);
    codec_out.push_back(cur);
    history.push_back(cur);

    const int max_codec = 25 * 120;   // 2 minutes of speech
    for (int step = 0; step < (int)stream.size() && cur != id_codec_end && cur != id_codec_pad &&
                       (int)codec_out.size() < max_codec; step++) {
        if (!build_prefill(1, n_past + 1)) return false;
        std::vector<float> cemb;
        codec_row(cur, cemb);
        const std::vector<float> & rep = stream[(size_t)step];
        std::vector<float> in((size_t)embw);
        for (int d = 0; d < embw; d++) in[(size_t)d] = cemb[(size_t)d] + rep[(size_t)d];
        ggml_backend_tensor_set(g_emb_in, in.data(), 0, sizeof(float) * embw);
        std::vector<float> zero((size_t)embw, 0.0f);
        ggml_backend_tensor_set(g_reply_in, zero.data(), 0, sizeof(float) * embw);
        std::vector<int> id1 = { cur };
        ggml_backend_tensor_set(g_codec_in, id1.data(), 0, sizeof(int));
        std::vector<int> pos((size_t)4, next_pos + T + step);
        ggml_backend_tensor_set(g_pos_in, pos.data(), 0, sizeof(int) * pos.size());
        std::vector<int> slot = { n_past };
        ggml_backend_tensor_set(g_slot_in, slot.data(), 0, sizeof(int));
        vqa::set_causal_mask(g_mask_in, n_past, 1);

        if (ggml_backend_graph_compute(backend, tgf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "talker: step compute failed\n");
            return false;
        }
        ggml_backend_tensor_get(g_logits, logits.data(), 0, sizeof(float) * n_vocab);
        n_past++;
        cur = sample(logits, history);
        if (cur == id_codec_end || cur == id_codec_pad) break;
        codec_out.push_back(cur);
        history.push_back(cur);
    }
    fprintf(stderr, "talker: %d codec tokens (%.1f s)\n", (int)codec_out.size(),
            codec_out.size() / 25.0);
    return !codec_out.empty();
}

}  // namespace qwen25omni
