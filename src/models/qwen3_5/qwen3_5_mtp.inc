// qwen3_5_mtp.inc - one-token-recursion MTP speculative decoding for the
// Qwen3.5 checkpoints that ship an mtp.* head (marlin.has_mtp).
//
// Round layout (greedy, K = 2 drafted tokens), see docs/dev-notes.md:
//   draft:  MTP(t_last, h_target) -> d1, then MTP(d1, h_d1) -> d2
//   verify: target forward T = [d1, d2] in one graph -> logits for both
//           positions + the hidden after both (next draft seed)
//   accept: g1 = argmax(L_prev) decides d1; g2 = argmax(L1) decides d2;
//           emits d1..dn plus the correction token argmax(Ln). Rejected
//           tokens are rolled back by reading the gated-delta-net snapshot
//           slots written by the verify graph (slot1 = one token back,
//           slot2 = pre-round state); the next round's graphs are rebuilt
//           against the chosen slot, so no post-hoc device copying.
// This file is included inside namespace qwen3_5, after class Qwen3_5.
// ---------------------------------------------------------------------------
// MTP draft: one forward of the single-block head.
//   tok   : the last (accepted or drafted) token id
//   h     : the matching hidden state [n_embd] (target final-norm output,
//           or the head's own h from the previous recursion)
//   slot  : absolute KV position this draft RUNS AT (target-style position)
//   axis  : mrope axis position for the drafted token
// Writes logits_out [n_vocab] and h_out [n_embd]; returns argmax token.
int Qwen3_5::mtp_draft_step(int tok, const float * h, int slot, int axis,
                          std::vector<float> & logits_out, std::vector<float> & h_out) {
    ggml_context * ctx = nullptr;
    struct ggml_init_params ip = { 32ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    ctx = ggml_init(ip);
    if (!ctx) return -1;
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);

    const int D = n_embd, HD = head_dim, NH = n_head, NKV = n_kv_head;
    ggml_tensor * t_tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * t_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1);
    ggml_tensor * t_mpos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    const int S = std::min(n_ctx, slot + 1);
    ggml_tensor * t_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S, 1);

    ggml_tensor * e = ggml_get_rows(ctx, tok_embd, t_tok);        // [D,1]
    ggml_tensor * en = ggml_mul(ctx, ggml_rms_norm(ctx, e, rms_eps), mtp_e_norm);
    ggml_tensor * hn = ggml_mul(ctx, ggml_rms_norm(ctx, t_h, rms_eps), mtp_h_norm);
    ggml_tensor * x = ggml_mul_mat(ctx, mtp_fc, ggml_concat(ctx, en, hn, 0));

    ggml_tensor * l = ggml_mul(ctx, ggml_rms_norm(ctx, x, rms_eps), mtp_attn_norm);
    ggml_tensor * q_full = ggml_mul_mat(ctx, mtp_q, l);
    ggml_tensor * k = ggml_mul_mat(ctx, mtp_k, l);
    ggml_tensor * v = ggml_mul_mat(ctx, mtp_v, l);
    ggml_tensor * q3 = ggml_view_3d(ctx, q_full, HD, NH, 1,
                                    (size_t)2 * HD * ggml_element_size(q_full), q_full->nb[1], 0);
    q3 = ggml_mul(ctx, ggml_rms_norm(ctx, q3, rms_eps), mtp_qn);
    ggml_tensor * gate = ggml_view_3d(ctx, q_full, HD, NH, 1,
                                      (size_t)2 * HD * ggml_element_size(q_full), q_full->nb[1],
                                      (size_t)HD * ggml_element_size(q_full));
    gate = ggml_cont_2d(ctx, gate, (int64_t)NH * HD, 1);
    ggml_tensor * k3 = ggml_reshape_3d(ctx, k, HD, NKV, 1);
    k3 = ggml_mul(ctx, ggml_rms_norm(ctx, k3, rms_eps), mtp_kn);
    ggml_tensor * pos_rope = ggml_view_1d(ctx, t_mpos, 4, 0);
    ggml_tensor * pos_slot = ggml_view_1d(ctx, t_mpos, 1, (size_t)3 * sizeof(int));
    q3 = ggml_rope_multi(ctx, q3, pos_rope, nullptr, rope_dim, mrope_sections,
                         GGML_ROPE_TYPE_IMROPE, n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k3 = ggml_rope_multi(ctx, k3, pos_rope, nullptr, rope_dim, mrope_sections,
                         GGML_ROPE_TYPE_IMROPE, n_ctx, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k2 = ggml_reshape_2d(ctx, k3, NKV * HD, 1);
    ggml_tensor * v2 = ggml_reshape_2d(ctx, v, NKV * HD, 1);
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, mtp_kv.k[0], k2, pos_slot));
    ggml_build_forward_expand(gf, ggml_set_rows(ctx, mtp_kv.v[0], v2, pos_slot));
    ggml_tensor * qf = ggml_view_3d(ctx, q3, HD, 1, NH, q3->nb[2], q3->nb[1], 0);
    ggml_tensor * kf = ggml_view_3d(ctx, mtp_kv.k[0], HD, S, NKV,
                                    mtp_kv.k[0]->nb[1], HD * sizeof(ggml_fp16_t), 0);
    ggml_tensor * vf = ggml_view_3d(ctx, mtp_kv.v[0], HD, S, NKV,
                                    mtp_kv.v[0]->nb[1], HD * sizeof(ggml_fp16_t), 0);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, qf, kf, vf, t_mask,
                                           1.0f / sqrtf((float)HD), 0.0f, 0.0f);
    ggml_tensor * attn = ggml_reshape_2d(ctx, ggml_cont(ctx, fa), NH * HD, 1);
    attn = ggml_mul(ctx, attn, ggml_sigmoid(ctx, gate));
    attn = ggml_mul_mat(ctx, mtp_o, attn);
    x = ggml_add(ctx, x, attn);
    ggml_tensor * m = ggml_mul(ctx, ggml_rms_norm(ctx, x, rms_eps), mtp_post_norm);
    m = ggml_swiglu_split(ctx, ggml_mul_mat(ctx, mtp_gate, m), ggml_mul_mat(ctx, mtp_up, m));
    x = ggml_add(ctx, x, ggml_mul_mat(ctx, mtp_down, m));
    ggml_tensor * h_out_t = ggml_mul(ctx, ggml_rms_norm(ctx, x, rms_eps), mtp_head_norm);
    ggml_set_name(h_out_t, "mtp_h");
    ggml_tensor * logits = ggml_mul_mat(ctx, tok_embd, h_out_t);
    ggml_set_name(logits, "mtp_logits");
    ggml_build_forward_expand(gf, logits);

    if (!ggml_gallocr_alloc_graph(allocr, gf)) { ggml_free(ctx); return -1; }
    ggml_backend_tensor_set(t_tok, &tok, 0, sizeof(int));
    ggml_backend_tensor_set(t_h, h, 0, sizeof(float) * D);
    int mpos[4] = { axis, axis, axis, slot };
    ggml_backend_tensor_set(t_mpos, mpos, 0, sizeof(mpos));
    vqa::set_causal_mask(t_mask, slot, 1);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { ggml_free(ctx); return -1; }
    logits_out.resize(n_vocab);
    h_out.resize(D);
    ggml_backend_tensor_get(logits, logits_out.data(), 0, sizeof(float) * n_vocab);
    ggml_backend_tensor_get(h_out_t, h_out.data(), 0, sizeof(float) * D);
    ggml_free(ctx);
    int best = 0;
    for (int i = 1; i < n_vocab; i++) if (logits_out[i] > logits_out[best]) best = i;
    return best;
}

// verify forward: target over the two drafted tokens; returns logits for both
// positions [2][n_vocab] and the hidden after both (next draft seed).
bool Qwen3_5::mtp_verify(const int * toks, int n_past, int axis0, int state_slot, std::vector<float> & logits2, std::vector<float> & h_out) {
    TextGraph g;
    const int T = 2;
    const int S = n_past + T;
    if (!build_text_graph(g, T, S, 0, state_slot, /*logits_all*/ true)) return false;
    if (!ggml_gallocr_alloc_graph(allocr, g.gf)) { ggml_free(g.ctx); return false; }
    ggml_backend_tensor_set(g.tok_in, toks, 0, sizeof(int) * T);
    int mpos[8] = { axis0, axis0 + 1,          // t axis, both tokens
                    axis0, axis0 + 1,          // h axis
                    axis0, axis0 + 1,          // w axis
                    n_past, n_past + 1 };      // absolute KV slots
    ggml_backend_tensor_set(g.mpos_in, mpos, 0, sizeof(mpos));
    vqa::set_causal_mask(g.mask_in, n_past, T);
    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        ggml_free(g.ctx); return false;
    }
    logits2.resize((size_t)2 * n_vocab);
    ggml_backend_tensor_get(g.logits, logits2.data(), 0, sizeof(float) * logits2.size());
    ggml_tensor * hn = ggml_graph_get_tensor(g.gf, "t_final_norm");
    h_out.resize((size_t)2 * n_embd);   // col0 = after one token, col1 = after both
    ggml_backend_tensor_get(hn, h_out.data(), 0, sizeof(float) * 2 * n_embd);
    ggml_free(g.ctx);
    return true;
}


// greedy two-token rounds: verify processes [last emitted token, drafted d1]
// in one forward; emits both on match, or the correction token with a
// one-token state rollback otherwise.
bool Qwen3_5::run_speculative(int next, int seed0, const std::vector<float> & logits_prev, float * h_seed,
                             int max_new, const vqa::GenParams & gp, std::string & raw, int & n_dec) {
    std::vector<float> L_prev = logits_prev;      // distribution after the last processed token
    std::vector<float> L2((size_t)2 * n_vocab), h_d(n_embd * 2), dlog(n_vocab), dh(n_embd);
    // seed = the last PROCESSED token: its embedding pairs with the hidden h.
    // `pending` is the sampled-but-unprocessed token the round will verify.
    int seed_tok = seed0;
    int pending = next;
    std::vector<float> h(h_seed, h_seed + n_embd);
    int write_base = n_past;
    int cur = last_mpos[0] + mrope_delta;          // axis of the last prompt token
    int state_slot = 0;
    long accepted = 0, rounds = 0;
    if (getenv("QWEN35_SPEC_SELFTEST") && n_past >= 2) {
        // minimal reject-path probe: verify_a [pending, X] (rolled back),
        // then verify_b reading slot1 at write_base = np+1 - must equal the
        // sequential [pending, g1, Y] chain exactly
        const int n_lin = n_layer - n_full_layers;
        const size_t sst = (size_t)ssm_d_state * ssm_d_state * ssm_dt_rank * 4;
        const size_t cst = (size_t)(ssm_d_conv - 1) * (2 * ssm_n_group * ssm_d_state + ssm_d_inner) * 4;
        std::vector<std::vector<char>> sv(n_lin), cv(n_lin);
        for (int i = 0; i < n_lin; i++) {
            sv[i].resize(sst); cv[i].resize(cst);
            ggml_backend_tensor_get(ssm_state[i], sv[i].data(), 0, sst);
            ggml_backend_tensor_get(conv_state[i], cv[i].data(), 0, cst);
        }
        std::vector<float> L2x((size_t)2 * n_vocab), hx(2 * n_embd);
        const int save_np = n_past;
        const int ta[2] = { pending, 12345 };
        mtp_verify(ta, save_np, cur + 1, 0, L2x, hx);
        std::vector<float> Lb((size_t)2 * n_vocab), hb(2 * n_embd);
        const int tb[2] = { 999, 998 };
        mtp_verify(tb, save_np + 1, cur + 2, 1, Lb, hb);
        std::vector<char> stB(sst);
        ggml_backend_tensor_get(ssm_state[0], stB.data(), 0, sst);
        // restore, reference: [pending, 999, 998]
        for (int i = 0; i < n_lin; i++) {
            ggml_backend_tensor_set(ssm_state[i], sv[i].data(), 0, sst);
            ggml_backend_tensor_set(conv_state[i], cv[i].data(), 0, cst);
        }
        n_past = save_np;
        std::vector<float> Lr(n_vocab), dummy(n_vocab);
        TextGraph g;
        std::vector<int> p1 = { cur + 1, cur + 1, cur + 1, save_np };
        run_text(g, { pending }, nullptr, nullptr, p1, Lr, &dummy);
        std::vector<int> p2 = { cur + 2, cur + 2, cur + 2, save_np + 1 };
        run_text(g, { 999 }, nullptr, nullptr, p2, Lr, nullptr);
        std::vector<int> p3 = { cur + 3, cur + 3, cur + 3, save_np + 2 };
        run_text(g, { 998 }, nullptr, nullptr, p3, Lr, nullptr);
        {   const float * x = (const float *)stB.data(), * y = (const float *)sv[0].data();  // y unused norm base
            (void)y;
            std::vector<char> ref3(sst);
            ggml_backend_tensor_get(ssm_state[0], ref3.data(), 0, sst);
            const float * r = (const float *)ref3.data(), * b = (const float *)stB.data();
            double num = 0, den = 0;
            for (size_t i = 0; i < sst / 4; i++) { double d = r[i] - b[i]; num += d * d; den += (double)r[i] * r[i]; }
            int amB = (int)(std::max_element(Lb.begin(), Lb.begin() + n_vocab) - Lb.begin());
            int amR = (int)(std::max_element(Lr.begin(), Lr.end()) - Lr.begin());
            fprintf(stderr, "[selftest] REJECT-PATH: state rel=%.3e argmax %d vs %d\n", sqrt(num / (den + 1e-12)), amB, amR);
        }
        return false;
    }
    while (n_dec < max_new) {
        if (pending == id_eos || pending == id_endoftext) break;
        // two-step MTP recursion: step 1 predicts the PENDING position (already
        // decided - only its hidden feeds the chain), step 2 predicts d1
        const int d0 = mtp_draft_step(seed_tok, h.data(), write_base, cur + 1, dlog, dh);
        if (d0 < 0) return false;
        const int d1 = mtp_draft_step(d0, dh.data(), write_base + 1, cur + 2, dlog, dh);
        if (d1 < 0) return false;
        const int toks[2] = { pending, d1 };
        if (!mtp_verify(toks, write_base, cur + 1, state_slot, L2, h_d)) return false;
        const float * L1 = L2.data();
        const float * L2c = L2.data() + n_vocab;
        int g1 = 0;
        for (int i = 1; i < n_vocab; i++) if (L1[i] > L1[g1]) g1 = i;
        // emit the (already validated) token
        const std::string piece = tok_.decode({pending});
        raw += piece;
        if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
        n_dec++;
        rounds++;
        static const bool force_rej = getenv("QWEN35_SPEC_FORCE_REJ") != nullptr;
        if (g1 == d1 && !force_rej) {
            if (d1 == id_eos || d1 == id_endoftext || n_dec >= max_new) break;
            const std::string p2 = tok_.decode({d1});
            raw += p2;
            if (gp.on_token) gp.on_token(p2.c_str(), gp.on_token_ud);
            n_dec++; accepted++;
            int g2 = 0;
            for (int i = 1; i < n_vocab; i++) if (L2c[i] > L2c[g2]) g2 = i;
            seed_tok = d1; pending = g2; L_prev.assign(L2c, L2c + n_vocab);
            h.assign(h_d.begin() + n_embd, h_d.end());
            state_slot = 0; write_base += 2; cur += 2;
        } else {
            // rollback keeps the state after `pending`: that IS the new seed
            seed_tok = pending; pending = g1; L_prev.assign(L1, L1 + n_vocab);
            h.assign(h_d.begin(), h_d.begin() + n_embd);
            state_slot = 1; write_base += 1; cur += 1;
        }
    }
    n_past = write_base;
    last_mpos = { cur, cur, cur };
    fprintf(stderr, "[spec] %d rounds, %ld double-hits (%.0f%%), %d tokens, %.1f tok/round\n",
            (int)rounds, accepted, rounds ? 100.0 * accepted / rounds : 0.0f, n_dec,
            rounds ? (double)n_dec / rounds : 0.0);
    return true;
}
