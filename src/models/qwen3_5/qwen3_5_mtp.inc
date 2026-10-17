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
                          std::vector<float> & logits_out, std::vector<float> & h_out, int win_lo) {
    // graph is cached per 512-block window (mask/KV views padded to the block
    // so the tensor shapes are stable); per call we only set inputs + compute
    const int blk = std::min(vqa::block_for(slot + 1), n_ctx);
    const int D = n_embd, HD = head_dim, NH = n_head, NKV = n_kv_head;
    SpecCache & c = spec_d;
    ggml_tensor * t_tok, * t_h, * t_mpos, * t_mask, * logits, * h_out_t, * t_am;
    if (c.blk == blk && c.gf) {
        t_tok   = ggml_graph_get_tensor(c.gf, "dt_tok");
        t_h     = ggml_graph_get_tensor(c.gf, "dt_h");
        t_mpos  = ggml_graph_get_tensor(c.gf, "dt_mpos");
        t_mask  = ggml_graph_get_tensor(c.gf, "dt_mask");
        logits  = ggml_graph_get_tensor(c.gf, "mtp_logits");
        h_out_t = ggml_graph_get_tensor(c.gf, "mtp_h");
        t_am    = ggml_graph_get_tensor(c.gf, "dt_am");
        if (!t_tok || !t_h || !t_mpos || !t_mask || !logits || !h_out_t || !t_am) return -1;
    } else {
        if (c.ctx) ggml_free(c.ctx);
        struct ggml_init_params ip = { 32ull * 1024 * 1024, NULL, /*no_alloc*/ true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return -1;
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
        c.ctx = ctx; c.gf = gf; c.blk = blk;
        t_tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_name(t_tok, "dt_tok");
        t_h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 1);
        ggml_set_name(t_h, "dt_h");
        t_mpos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        ggml_set_name(t_mpos, "dt_mpos");
        t_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, blk, 1);
        ggml_set_name(t_mask, "dt_mask");
        const int S = blk;

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
    h_out_t = ggml_mul(ctx, ggml_rms_norm(ctx, x, rms_eps), mtp_head_norm);
    ggml_set_name(h_out_t, "mtp_h");
    logits = ggml_mul_mat(ctx, tok_embd, h_out_t);
    ggml_set_name(logits, "mtp_logits");
    // argmax on the device: the round loop only consumes the drafted token id,
    // so the [n_vocab] logits never cross back to the host (600+ KB of D2H per
    // draft step used to dominate the round)
    ggml_tensor * am = ggml_argmax(ctx, logits);
    ggml_set_name(am, "dt_am");
    t_am = am;
    ggml_build_forward_expand(gf, logits);
    ggml_build_forward_expand(gf, am);
        if (!c.alloc) c.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(c.alloc, gf)) { ggml_free(ctx); c.ctx = nullptr; c.blk = -1; return -1; }
    }
    ggml_backend_tensor_set(t_tok, &tok, 0, sizeof(int));
    ggml_backend_tensor_set(t_h, h, 0, sizeof(float) * D);
    int mpos[4] = { axis, axis, axis, slot };
    ggml_backend_tensor_set(t_mpos, mpos, 0, sizeof(mpos));
    static const bool selfkv = getenv("QWEN35_SPEC_SELFKV") != nullptr;
    static const bool freshkv = getenv("QWEN35_SPEC_FRESHKV") != nullptr;
    if (selfkv) {
        // attend only to the draft token's own KV
        const int S = (int)t_mask->ne[0];
        std::vector<ggml_fp16_t> mv((size_t)S, ggml_fp32_to_fp16(-INFINITY));
        if (slot >= 0 && slot < S) mv[(size_t)slot] = ggml_fp32_to_fp16(0.0f);
        ggml_backend_tensor_set(t_mask, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());
    } else if (freshkv) {
        // llama.cpp-equivalent draft-region reset, on the cheap: cols < the
        // round's write_base hold KV of drafts for OTHER tokens (drafted was
        // not always accepted), so start the attention window at write_base;
        // cols [write_base, slot] were written fresh by this round's steps
        std::vector<int> lo{win_lo}, hi{slot + 1}, pos{slot};
        vqa::set_window_mask(t_mask, lo, hi, pos);
    } else {
        vqa::set_causal_mask(t_mask, slot, 1);
    }
    if (ggml_backend_graph_compute(backend, c.gf) != GGML_STATUS_SUCCESS) return -1;
    h_out.resize(D);
    ggml_backend_tensor_get(h_out_t, h_out.data(), 0, sizeof(float) * D);
    int best = -1;
    ggml_backend_tensor_get(t_am, &best, 0, sizeof(int));
    return best;
}

// verify forward: target over the two drafted tokens; returns logits for both
// positions [2][n_vocab] and the hidden after both (next draft seed).
bool Qwen3_5::mtp_verify(const int * toks, int n_past, int axis0, int state_slot, std::vector<float> & logits2, std::vector<float> & h_out, int am_out[2]) {
    static const char * const sdump = getenv("QWEN35_SPEC_DUMP");
    // full-logits readback is debug-only: the accept decision needs just the
    // two per-column argmaxes, computed in-graph (2 ints instead of 2*n_vocab)
    static const bool want_logits = getenv("QWEN35_SPEC_TRACE") || getenv("QWEN35_SPEC_LIVECHK") || getenv("QWEN35_SPEC_SELFTEST");
    static const char * const dump_names[] = {
        "emb", "t_l0_la_qkv", "t_l0_la_z", "t_l0_la_conv", "t_l0_la_core",
        "t_l0_out", "t_final_norm", nullptr,
    };
    const int T = 2;
    // 512-block-padded KV window (same block_for as the static decode graph)
    // so the per-token FA calls take the vector-kernel path and stay
    // bit-identical to plain decoding. Graph cached per (state slot, block).
    const int S = std::min(vqa::block_for(n_past + T), n_ctx);
    SpecCache & c = spec_v[state_slot == 1 ? 1 : 0];
    if (c.blk != S || c.slot != state_slot || !c.gf) {
        if (c.ctx) {
            ggml_free(c.ctx);
            if (sdump) vqa::dump::dump_registry_clear();   // the cached graph's sinks die with it
        }
        TextGraph g;
        if (!build_text_graph(g, T, S, 0, state_slot, /*logits_all*/ true)) return false;
        if (sdump) vqa::dump::mark_sinks(g.ctx, g.gf, dump_names);
        // per-column argmax of the [n_vocab, T] logits, kept on the device
        // (must be expanded BEFORE gallocr_alloc, or its output never lands
        // in a buffer and the kernel writes through a null pointer)
        if (!want_logits) {
            c.am = ggml_argmax(g.ctx, g.logits);
            ggml_set_name(c.am, "v_am");
            ggml_build_forward_expand(g.gf, c.am);
        }
        if (sdump) vqa::dump::finalize_sinks(g.ctx, g.gf, backend);   // keeps at the graph tail
        if (!c.alloc) c.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(c.alloc, g.gf)) { ggml_free(g.ctx); return false; }
        c.ctx = g.ctx; c.gf = g.gf; c.blk = S; c.slot = state_slot;
        c.tok = g.tok_in; c.mpos = g.mpos_in; c.mask = g.mask_in; c.logits = g.logits;
        c.hout = ggml_graph_get_tensor(g.gf, "t_final_norm");
    }
    ggml_cgraph * gf = c.gf;
    ggml_tensor * g_tok = c.tok, * g_mpos = c.mpos, * g_mask = c.mask, * g_logits = c.logits, * g_hn = c.hout;
    ggml_backend_tensor_set(g_tok, toks, 0, sizeof(int) * T);
    int mpos[8] = { axis0, axis0 + 1,          // t axis, both tokens
                    axis0, axis0 + 1,          // h axis
                    axis0, axis0 + 1,          // w axis
                    n_past, n_past + 1 };      // absolute KV slots
    ggml_backend_tensor_set(g_mpos, mpos, 0, sizeof(mpos));
    vqa::set_causal_mask(g_mask, n_past, T);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return false;
    int am[2] = { 0, 0 };
    if (c.am) {
        ggml_backend_tensor_get(c.am, am, 0, sizeof(am));
    } else {
        logits2.resize((size_t)2 * n_vocab);
        ggml_backend_tensor_get(g_logits, logits2.data(), 0, sizeof(float) * logits2.size());
        for (int i = 1; i < n_vocab; i++) { if (logits2[i] > logits2[am[0]]) am[0] = i; if (logits2[n_vocab + i] > logits2[n_vocab + am[1]]) am[1] = i; }
    }
    if (am_out) { am_out[0] = am[0]; am_out[1] = am[1]; }
    if (want_logits) {
        if (logits2.empty()) { logits2.resize((size_t)2 * n_vocab); ggml_backend_tensor_get(g_logits, logits2.data(), 0, sizeof(float) * logits2.size()); }
    } else {
        logits2.clear();
    }
    h_out.resize((size_t)2 * n_embd);   // col0 = after one token, col1 = after both
    ggml_backend_tensor_get(g_hn, h_out.data(), 0, sizeof(float) * 2 * n_embd);
    if (sdump) {
        for (const char * const * n = dump_names; *n; ++n)
            vqa::dump::write(gf, sdump, *n);
    }
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
    // axis of the last prompt token: the three axes are equal for a trailing
    // text token (prompt ends with text); +1 per generated token matches the
    // non-speculative decode loop's step_pos = last_mpos + 1 convention
    int cur = std::max({last_mpos[0], last_mpos[1], last_mpos[2]});
    int state_slot = 0;
    long accepted = 0, rounds = 0;
    static const bool trace = getenv("QWEN35_SPEC_TRACE") != nullptr;
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
        const int ta[2] = { pending, 999 };   // same second token as the T=1
                                               // reference chain so KV col1 compares
        mtp_verify(ta, save_np, cur + 1, 0, L2x, hx);
        // snapshot the KV columns verify_a just wrote (cols save_np, +1)
        const size_t colb = kv.k[0]->nb[1];
        std::vector<std::vector<char>> kva(n_full_layers * 2), kvv(n_full_layers * 2);
        for (int i = 0; i < n_full_layers * 2; i++) {
            kva[i].resize(colb); kvv[i].resize(colb);
            ggml_backend_tensor_get(kv.k[i / 2], kva[i].data(), (size_t)(save_np + i % 2) * colb, colb);
            ggml_backend_tensor_get(kv.v[i / 2], kvv[i].data(), (size_t)(save_np + i % 2) * colb, colb);
        }
        // snapshot the MAIN conv+ssm state verify_a just wrote (after 2 tokens)
        std::vector<std::vector<char>> csv_a(n_lin), ssv_a(n_lin);
        // snapshot the ROLLBACK (slot-b) states verify_a wrote = after 1 token
        std::vector<std::vector<char>> csb_a(n_lin), ssb_a(n_lin);
        for (int i = 0; i < n_lin; i++) {
            csv_a[i].resize(cst); ssv_a[i].resize(sst);
            ggml_backend_tensor_get(conv_state[i], csv_a[i].data(), 0, cst);
            ggml_backend_tensor_get(ssm_state[i], ssv_a[i].data(), 0, sst);
            csb_a[i].resize(cst); ssb_a[i].resize(sst);
            ggml_backend_tensor_get(conv_state_b[i], csb_a[i].data(), 0, cst);
            ggml_backend_tensor_get(ssm_state_b[i], ssb_a[i].data(), 0, sst);
        }
        if (const char * rd0 = getenv("QWEN35_SPEC_DUMP")) {
            // stage-diff probe: restore the pre-verify state, T=1 reference
            // forward of `pending` at save_np, dumped to QWEN35_SPEC_DUMP/ref
            // (the root holds the verify_a [pending, 12345] dumps; snapshot
            // them to /v first - the later verify_b overwrites the root)
            std::error_code ec;
            std::filesystem::copy(rd0, std::string(rd0) + "/v",
                                  std::filesystem::copy_options::recursive, ec);
            for (int i = 0; i < n_lin; i++) {
                ggml_backend_tensor_set(ssm_state[i], sv[i].data(), 0, sst);
                ggml_backend_tensor_set(conv_state[i], cv[i].data(), 0, cst);
            }
            const std::string rd = std::string(rd0) + "/ref";
            std::filesystem::create_directories(rd, ec);
            static const char * const dn[] = { "emb", "t_l0_la_qkv", "t_l0_la_z",
                "t_l0_la_conv", "t_l0_la_core", "t_l0_out", "t_final_norm", nullptr };
            TextGraph g1;
            if (build_text_graph(g1, 1, save_np + 1, 0, 0, false)) {
                vqa::dump::mark_sinks(g1.ctx, g1.gf, dn);
                vqa::dump::finalize_sinks(g1.ctx, g1.gf, backend);
                if (ggml_gallocr_alloc_graph(allocr, g1.gf)) {
                    int pend = pending;
                    ggml_backend_tensor_set(g1.tok_in, &pend, 0, sizeof(int));
                    int mp1[4] = { cur + 1, cur + 1, cur + 1, save_np };
                    ggml_backend_tensor_set(g1.mpos_in, mp1, 0, sizeof(mp1));
                    vqa::set_causal_mask(g1.mask_in, save_np, 1);
                    if (ggml_backend_graph_compute(backend, g1.gf) == GGML_STATUS_SUCCESS)
                        for (const char * const * n = dn; *n; ++n)
                            vqa::dump::write(g1.gf, rd.c_str(), *n);
                }
                ggml_free(g1.ctx);
            }
        }
        std::vector<float> Lb((size_t)2 * n_vocab), hb(2 * n_embd);
        const int tb[2] = { 999, 998 };
        if (const char * rd0 = getenv("QWEN35_SPEC_DUMP")) {
            // T=1 reference chain [pending, 999] from the pre-verify state,
            // dumping the token-999 step to QWEN35_SPEC_DUMP/ref2; then run
            // verify_b so the ROOT dumps hold its slot1 [999, 998] tensors
            static const char * const dn2[] = { "emb", "t_l0_la_qkv", "t_l0_la_z",
                "t_l0_la_conv", "t_l0_la_core", "t_l0_out", "t_final_norm", nullptr };
            const std::string rd2 = std::string(rd0) + "/ref2";
            std::error_code ec2;
            std::filesystem::create_directories(rd2, ec2);
            for (int i = 0; i < n_lin; i++) {
                ggml_backend_tensor_set(ssm_state[i], sv[i].data(), 0, sst);
                ggml_backend_tensor_set(conv_state[i], cv[i].data(), 0, cst);
            }
            for (int step = 0; step < 2; step++) {
                TextGraph g1;
                const int tok = step ? 999 : pending;
                const int np = save_np + step;
                if (!build_text_graph(g1, 1, np + 1, 0, 0, false)) continue;
                vqa::dump::mark_sinks(g1.ctx, g1.gf, dn2);
                vqa::dump::finalize_sinks(g1.ctx, g1.gf, backend);
                if (!ggml_gallocr_alloc_graph(allocr, g1.gf)) { ggml_free(g1.ctx); continue; }
                ggml_backend_tensor_set(g1.tok_in, &tok, 0, sizeof(int));
                int mp[4] = { cur + 1 + step, cur + 1 + step, cur + 1 + step, np };
                ggml_backend_tensor_set(g1.mpos_in, mp, 0, sizeof(mp));
                vqa::set_causal_mask(g1.mask_in, np, 1);
                if (ggml_backend_graph_compute(backend, g1.gf) != GGML_STATUS_SUCCESS) { ggml_free(g1.ctx); continue; }
                if (step == 1)
                    for (const char * const * n = dn2; *n; ++n)
                        vqa::dump::write(g1.gf, rd2.c_str(), *n);
                ggml_free(g1.ctx);
            }
            // compare KV columns: verify_a writes (pending@save_np, 12345@+1)
            // vs T=1 chain (pending@save_np, 999@+1) - col save_np must match
            for (int i = 0; i < n_full_layers * 2; i++) {
                std::vector<char> kb(colb), vb(colb);
                ggml_backend_tensor_get(kv.k[i / 2], kb.data(), (size_t)(save_np + i % 2) * colb, colb);
                ggml_backend_tensor_get(kv.v[i / 2], vb.data(), (size_t)(save_np + i % 2) * colb, colb);
                auto relfn = [&](const std::vector<char> & a, const std::vector<char> & b) {
                    const ggml_fp16_t * x = (const ggml_fp16_t *)a.data();
                    const ggml_fp16_t * y = (const ggml_fp16_t *)b.data();
                    double num = 0, den = 0;
                    for (size_t j = 0; j < colb / 2; j++) {
                        double d = ggml_fp16_to_fp32(x[j]) - ggml_fp16_to_fp32(y[j]);
                        num += d * d; den += (double)ggml_fp16_to_fp32(y[j]) * ggml_fp16_to_fp32(y[j]);
                    }
                    return sqrt(num / (den + 1e-12));
                };
                fprintf(stderr, "[selftest] KV L%d col%d: k rel=%.3e v rel=%.3e\n",
                        i / 2, i % 2, relfn(kva[i], kb), relfn(kvv[i], vb));
            }
            // MAIN conv+ssm state after verify_a [pending, 999] vs T=1 chain
            for (int i = 0; i < n_lin; i++) {
                std::vector<char> cb(cst), sb(sst);
                ggml_backend_tensor_get(conv_state[i], cb.data(), 0, cst);
                ggml_backend_tensor_get(ssm_state[i], sb.data(), 0, sst);
                double cn = 0, cd = 0, sn = 0, sd2 = 0;
                const float * c0 = (const float *)csv_a[i].data(), * c1 = (const float *)cb.data();
                const float * s0 = (const float *)ssv_a[i].data(), * s1 = (const float *)sb.data();
                for (size_t j = 0; j < cst / 4; j++) { double d = c0[j] - c1[j]; cn += d * d; cd += (double)c1[j] * c1[j]; }
                for (size_t j = 0; j < sst / 4; j++) { double d = s0[j] - s1[j]; sn += d * d; sd2 += (double)s1[j] * s1[j]; }
                fprintf(stderr, "[selftest] MAIN L%d: conv rel=%.3e ssm rel=%.3e\n",
                        i, sqrt(cn / (cd + 1e-12)), sqrt(sn / (sd2 + 1e-12)));
            }
        }
        mtp_verify(tb, save_np + 1, cur + 2, 1, Lb, hb);
        // full-state snapshot after the slot1-start verify [999, 998]
        std::vector<std::vector<char>> csB(n_lin), ssB(n_lin);
        for (int i = 0; i < n_lin; i++) {
            csB[i].resize(cst); ssB[i].resize(sst);
            ggml_backend_tensor_get(conv_state[i], csB[i].data(), 0, cst);
            ggml_backend_tensor_get(ssm_state[i], ssB[i].data(), 0, sst);
        }
        // restore, reference: [pending, 999, 998]
        for (int i = 0; i < n_lin; i++) {
            ggml_backend_tensor_set(ssm_state[i], sv[i].data(), 0, sst);
            ggml_backend_tensor_set(conv_state[i], cv[i].data(), 0, cst);
        }
        n_past = save_np;
        std::vector<float> Lr(n_vocab), dummy(n_vocab), Lr2(n_vocab), Ldyn(n_vocab), Lstat(n_vocab);
        TextGraph g;
        std::vector<int> p1 = { cur + 1, cur + 1, cur + 1, save_np };
        // DYNAMIC T=1 step for the first reference token (run_text would take
        // the STATIC 512-block decode graph; comparing slot-b against that
        // mixes static-vs-dynamic FA differences into the check)
        {
            // variant A: dynamic graph with S = exact (np+1)
            // variant B: dynamic graph with S = 512 block (FA padding only,
            //            same window as the static decode graph)
            TextGraph g1;
            if (build_text_graph(g1, 1, save_np + 1, 0, 0, false) &&
                ggml_gallocr_alloc_graph(allocr, g1.gf)) {
                int pend = pending;
                ggml_backend_tensor_set(g1.tok_in, &pend, 0, sizeof(int));
                int mp[4] = { cur + 1, cur + 1, cur + 1, save_np };
                ggml_backend_tensor_set(g1.mpos_in, mp, 0, sizeof(mp));
                vqa::set_causal_mask(g1.mask_in, save_np, 1);
                if (ggml_backend_graph_compute(backend, g1.gf) == GGML_STATUS_SUCCESS)
                    ggml_backend_tensor_get(g1.logits, Ldyn.data(), 0, sizeof(float) * n_vocab);
            }
            ggml_free(g1.ctx);
            for (int i = 0; i < n_lin; i++) {
                ggml_backend_tensor_set(ssm_state[i], sv[i].data(), 0, sst);
                ggml_backend_tensor_set(conv_state[i], cv[i].data(), 0, cst);
            }
            TextGraph gb;
            const int blk = vqa::block_for(save_np + 1);
            if (build_text_graph(gb, 1, blk, 0, 0, false) &&
                ggml_gallocr_alloc_graph(allocr, gb.gf)) {
                int pend = pending;
                ggml_backend_tensor_set(gb.tok_in, &pend, 0, sizeof(int));
                int mp[4] = { cur + 1, cur + 1, cur + 1, save_np };
                ggml_backend_tensor_set(gb.mpos_in, mp, 0, sizeof(mp));
                vqa::set_causal_mask(gb.mask_in, save_np, 1);
                if (ggml_backend_graph_compute(backend, gb.gf) == GGML_STATUS_SUCCESS)
                    ggml_backend_tensor_get(gb.logits, Lstat.data(), 0, sizeof(float) * n_vocab);
            }
            ggml_free(gb.ctx);
        }
        {   // compare: exact-S vs 512-block-S vs true static run_text path
            for (int i = 0; i < n_lin; i++) {
                ggml_backend_tensor_set(ssm_state[i], sv[i].data(), 0, sst);
                ggml_backend_tensor_set(conv_state[i], cv[i].data(), 0, cst);
            }
            n_past = save_np;
            std::vector<float> Lst2(n_vocab);
            std::vector<int> p1 = { cur + 1, cur + 1, cur + 1, save_np };
            run_text(g, { pending }, nullptr, nullptr, p1, Lst2, nullptr);
            auto lrel = [&](const std::vector<float> & a, const std::vector<float> & b) {
                double ln = 0, ld = 0;
                for (int i = 0; i < n_vocab; i++) { double d = a[i] - b[i]; ln += d * d; ld += (double)b[i] * b[i]; }
                return sqrt(ln / (ld + 1e-12));
            };
            int amd = (int)(std::max_element(Ldyn.begin(), Ldyn.end()) - Ldyn.begin());
            int amb = (int)(std::max_element(Lstat.begin(), Lstat.end()) - Lstat.begin());
            fprintf(stderr, "[selftest] exact-S vs blk-S rel=%.3e (am %d/%d) | blk-S vs STATIC rel=%.3e\n",
                    lrel(Ldyn, Lstat), amd, amb, lrel(Lstat, Lst2));
            // S sweep: logits vs padded S, all from the same pristine state
            for (int Sx : { save_np + 1, save_np + 2, save_np + 8, 64, 256, 512 }) {
                for (int i = 0; i < n_lin; i++) {
                    ggml_backend_tensor_set(ssm_state[i], sv[i].data(), 0, sst);
                    ggml_backend_tensor_set(conv_state[i], cv[i].data(), 0, cst);
                }
                std::vector<float> Lx(n_vocab);
                TextGraph gx;
                if (build_text_graph(gx, 1, Sx, 0, 0, false) &&
                    ggml_gallocr_alloc_graph(allocr, gx.gf)) {
                    int pend = pending;
                    ggml_backend_tensor_set(gx.tok_in, &pend, 0, sizeof(int));
                    int mp[4] = { cur + 1, cur + 1, cur + 1, save_np };
                    ggml_backend_tensor_set(gx.mpos_in, mp, 0, sizeof(mp));
                    vqa::set_causal_mask(gx.mask_in, save_np, 1);
                    if (ggml_backend_graph_compute(backend, gx.gf) == GGML_STATUS_SUCCESS)
                        ggml_backend_tensor_get(gx.logits, Lx.data(), 0, sizeof(float) * n_vocab);
                }
                ggml_free(gx.ctx);
                fprintf(stderr, "[selftest] sweep S=%d rel-vs-exact=%.3e\n", Sx, lrel(Lx, Ldyn));
            }
            // T=2 [pending, 999] verify: col0 logits vs the VEC reference
            // (Lstat, T=1 padded S) and vs the mma-f16 reference (Ldyn)
            {
                for (int i = 0; i < n_lin; i++) {
                    ggml_backend_tensor_set(ssm_state[i], sv[i].data(), 0, sst);
                    ggml_backend_tensor_set(conv_state[i], cv[i].data(), 0, cst);
                }
                std::vector<float> L2p((size_t)2 * n_vocab), hh(2 * n_embd);
                const int t2[2] = { pending, 999 };
                mtp_verify(t2, save_np, cur + 1, 0, L2p, hh);
                fprintf(stderr, "[selftest] T2 col0: vs VEC=%.3e vs MMA=%.3e\n",
                        lrel(std::vector<float>(L2p.begin(), L2p.begin() + n_vocab), Lstat),
                        lrel(std::vector<float>(L2p.begin(), L2p.begin() + n_vocab), Ldyn));
            }
            // restore the pristine state, then one STATIC [pending] step so
            // the p2/p3 reference chain below is entirely static-path (the
            // verify graphs match the static decode numerics)
            for (int i = 0; i < n_lin; i++) {
                ggml_backend_tensor_set(ssm_state[i], sv[i].data(), 0, sst);
                ggml_backend_tensor_set(conv_state[i], cv[i].data(), 0, cst);
            }
            n_past = save_np;
            std::vector<int> p0 = { cur + 1, cur + 1, cur + 1, save_np };
            run_text(g, { pending }, nullptr, nullptr, p0, Lr, nullptr);
        }
        std::vector<int> p2 = { cur + 2, cur + 2, cur + 2, save_np + 1 };
        run_text(g, { 999 }, nullptr, nullptr, p2, Lr2, nullptr);
        std::vector<int> p3 = { cur + 3, cur + 3, cur + 3, save_np + 2 };
        run_text(g, { 998 }, nullptr, nullptr, p3, Lr, nullptr);
        {   // full conv+ssm comparison across ALL linear layers
            double worst_c = 0, worst_s = 0;
            for (int i = 0; i < n_lin; i++) {
                std::vector<char> cb(cst), sb2(sst);
                ggml_backend_tensor_get(conv_state[i], cb.data(), 0, cst);
                ggml_backend_tensor_get(ssm_state[i], sb2.data(), 0, sst);
                auto rel2 = [&](const std::vector<char> & a, const std::vector<char> & b) {
                    const float * x = (const float *)a.data(), * y = (const float *)b.data();
                    double num = 0, den = 0;
                    for (size_t j = 0; j < b.size() / 4; j++) { double d = x[j] - y[j]; num += d * d; den += (double)y[j] * y[j]; }
                    return sqrt(num / (den + 1e-12));
                };
                worst_c = std::max(worst_c, rel2(csB[i], cb));
                worst_s = std::max(worst_s, rel2(ssB[i], sb2));
            }
            int amB = (int)(std::max_element(Lb.begin(), Lb.begin() + n_vocab) - Lb.begin());
            int am2 = (int)(std::max_element(Lr2.begin(), Lr2.end()) - Lr2.begin());
            double ln = 0, ld = 0;
            for (int i = 0; i < n_vocab; i++) { double d = Lb[i] - Lr2[i]; ln += d * d; ld += (double)Lr2[i] * Lr2[i]; }
            fprintf(stderr, "[selftest] REJECT-PATH: conv rel=%.3e ssm rel=%.3e | logits rel=%.3e argmax %d vs %d\n",
                    worst_c, worst_s, sqrt(ln / (ld + 1e-12)), amB, am2);
        }
        return false;
    }
    // Adaptive drafting, llama.cpp speculative-style n_min/p_min gating: when
    // the recent double-hit rate drops too low, a plain T=1 decode round is
    // cheaper than paying the T=2 verify for a likely single-token round. The
    // floor is therefore base speed, and every PROBE_INTERVAL plain rounds a
    // real spec round re-measures acceptance so recovery is picked up.
    double ema = 0.5;                 // EMA of per-round double-hit
    int plain_streak = 0;
    int plain_streak_total = 0;
    const double ema_min = getenv("QWEN35_SPEC_EMA_MIN") ? atof(getenv("QWEN35_SPEC_EMA_MIN")) : 0.30;
    const int probe_interval = getenv("QWEN35_SPEC_PROBE") ? atoi(getenv("QWEN35_SPEC_PROBE")) : 32;
    // bisect helper: force the first k rounds plain, then pure-spec
    const int plain_first = getenv("QWEN35_SPEC_PLAIN_FIRST") ? atoi(getenv("QWEN35_SPEC_PLAIN_FIRST")) : -1;
    TextGraph g_plain;
    while (n_dec < max_new) {
        if (pending == id_eos || pending == id_endoftext) break;
        const bool force_plain = plain_first >= 0 && rounds < plain_first;
        if ((ema < ema_min && plain_streak < probe_interval) || force_plain) {
            // plain round: process `pending` through the static decode path.
            // after a reject the live SSM/conv state lives in the slot-b
            // rollback tensors (state_slot == 1); the static graph only reads
            // the main states, so restore them first or the round continues
            // from a stale state and the text degenerates
            if (state_slot == 1) {
                const int n_linp = n_layer - n_full_layers;
                for (int i = 0; i < n_linp; i++) {
                    ggml_backend_tensor_copy(ssm_state_b[i], ssm_state[i]);
                    ggml_backend_tensor_copy(conv_state_b[i], conv_state[i]);
                }
            }
            std::vector<float> Lp(n_vocab), hp(n_embd);
            std::vector<int> pp = { cur + 1, cur + 1, cur + 1, write_base };
            n_past = write_base;    // run_text's static path derives the KV
                                    // window + mask base from the member
            if (!run_text(g_plain, { pending }, nullptr, nullptr, pp, Lp, &hp)) return false;
            // debug: how far is the static path's h from the verify path's h
            // for the SAME token+state? (draft-chain sensitivity probe)
            static const bool hchk = getenv("QWEN35_SPEC_HCHK") != nullptr;
            if (hchk) {
                std::vector<float> L2x((size_t)2 * n_vocab), hx(2 * n_embd);
                const int tx[2] = { pending, 999 };
                int amx[2];
                n_past = write_base;
                if (mtp_verify(tx, write_base, cur + 1, state_slot, L2x, hx, amx)) {
                    double num = 0, den = 0, n1 = 0, n2 = 0;
                    for (int i = 0; i < n_embd; i++) { double d = hp[i] - hx[i]; num += d * d; den += (double)hx[i] * hx[i]; n1 += (double)hp[i] * hp[i]; }
                    for (int i = 0; i < n_embd; i++) n2 += (double)hx[i] * hx[i];
                    fprintf(stderr, "[hchk] wb=%d h rel=%.3e |hp|=%.3f |hx|=%.3f\n", write_base, sqrt(num / (den + 1e-12)), sqrt(n1 / n_embd), sqrt(n2 / n_embd));
                }
            }
            const std::string piece = tok_.decode({pending});
            raw += piece;
            if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
            n_dec++; rounds++;
            int gn = 0;
            for (int i = 1; i < n_vocab; i++) if (Lp[i] > Lp[gn]) gn = i;
            seed_tok = pending; pending = gn;
            h.assign(hp.begin(), hp.end());
            state_slot = 0; write_base += 1; cur += 1;
            plain_streak++; plain_streak_total++;
            continue;
        }
        plain_streak = 0;
        // two-step MTP recursion: step 1 predicts the PENDING position (already
        // decided - only its hidden feeds the chain), step 2 predicts d1
        const int d0 = mtp_draft_step(seed_tok, h.data(), write_base, cur + 1, dlog, dh, write_base);
        if (d0 < 0) return false;
        const int d1 = mtp_draft_step(d0, dh.data(), write_base + 1, cur + 2, dlog, dh, write_base);
        if (d1 < 0) return false;
        const int toks[2] = { pending, d1 };
        static const int livechk = getenv("QWEN35_SPEC_LIVECHK") ? atoi(getenv("QWEN35_SPEC_LIVECHK")) : 0;
        const bool do_chk = livechk > 0 && rounds < (size_t)livechk;
        const int n_lin2 = n_layer - n_full_layers;
        std::vector<std::vector<char>> sv_main, cv_main;
        if (do_chk) {
            sv_main.resize(n_lin2); cv_main.resize(n_lin2);
            // snapshot the state verify will READ (slot-b tensors on the
            // rejection path) so the reference chain starts identically
            for (int i = 0; i < n_lin2; i++) {
                sv_main[i].resize((size_t)ssm_d_state * ssm_d_state * ssm_dt_rank * 4);
                cv_main[i].resize((size_t)(ssm_d_conv - 1) * (2 * ssm_n_group * ssm_d_state + ssm_d_inner) * 4);
                ggml_backend_tensor_get((state_slot == 1 ? ssm_state_b : ssm_state)[i], sv_main[i].data(), 0, sv_main[i].size());
                ggml_backend_tensor_get((state_slot == 1 ? conv_state_b : conv_state)[i], cv_main[i].data(), 0, cv_main[i].size());
            }
        }
        int v_am[2] = { -1, -1 };
        if (!mtp_verify(toks, write_base, cur + 1, state_slot, L2, h_d, v_am)) return false;
        // snapshot verify's persistent writes for the state comparison below
        std::vector<std::vector<char>> sv_post, cv_post, kv_post;
        if (do_chk) {
            sv_post = std::vector<std::vector<char>>(n_lin2);
            cv_post = std::vector<std::vector<char>>(n_lin2);
            for (int i = 0; i < n_lin2; i++) {
                sv_post[i].resize((size_t)ssm_d_state * ssm_d_state * ssm_dt_rank * 4);
                cv_post[i].resize((size_t)(ssm_d_conv - 1) * (2 * ssm_n_group * ssm_d_state + ssm_d_inner) * 4);
                ggml_backend_tensor_get(ssm_state[i], sv_post[i].data(), 0, sv_post[i].size());
                ggml_backend_tensor_get(conv_state[i], cv_post[i].data(), 0, cv_post[i].size());
            }
            const size_t colb2 = kv.k[0]->nb[1];
            kv_post.resize(n_full_layers * 2 * 2);
            for (int i = 0; i < n_full_layers; i++)
                for (int c = 0; c < 2; c++) {
                    kv_post[(i * 2 + c) * 2].resize(colb2);
                    kv_post[(i * 2 + c) * 2 + 1].resize(colb2);
                    ggml_backend_tensor_get(kv.k[i], kv_post[(i * 2 + c) * 2].data(), (size_t)(write_base + c) * colb2, colb2);
                    ggml_backend_tensor_get(kv.v[i], kv_post[(i * 2 + c) * 2 + 1].data(), (size_t)(write_base + c) * colb2, colb2);
                }
        }
        if (do_chk) {
            // restore the pre-verify state and run the SAME two tokens through
            // the static T=1 decode path; logits must match verify bit-close
            for (int i = 0; i < n_lin2; i++) {
                ggml_backend_tensor_set(ssm_state[i], sv_main[i].data(), 0, sv_main[i].size());
                ggml_backend_tensor_set(conv_state[i], cv_main[i].data(), 0, cv_main[i].size());
            }
            n_past = write_base;
            std::vector<float> Lr1(n_vocab), Lr2b(n_vocab), hdummy;
            TextGraph gc;
            std::vector<int> pa = { cur + 1, cur + 1, cur + 1, write_base };
            run_text(gc, { pending }, nullptr, nullptr, pa, Lr1, nullptr);
            std::vector<int> pb = { cur + 2, cur + 2, cur + 2, write_base + 1 };
            run_text(gc, { d1 }, nullptr, nullptr, pb, Lr2b, nullptr);
            auto lrel = [&](const float * a, const float * b) {
                double ln = 0, ld = 0;
                for (int i = 0; i < n_vocab; i++) { double d = a[i] - b[i]; ln += d * d; ld += (double)b[i] * b[i]; }
                return sqrt(ln / (ld + 1e-12));
            };
            int ar1 = (int)(std::max_element(Lr1.begin(), Lr1.end()) - Lr1.begin());
            int ar2 = (int)(std::max_element(Lr2b.begin(), Lr2b.end()) - Lr2b.begin());
            const float * V1 = L2.data();
            const float * V2c = L2.data() + n_vocab;
            int av1 = 0, av2 = 0;
            for (int i = 1; i < n_vocab; i++) { if (V1[i] > V1[av1]) av1 = i; if (V2c[i] > V2c[av2]) av2 = i; }
            fprintf(stderr, "[livechk] r%ld wb=%d slot=%d: L1 rel=%.3e (am %d/%d) L2 rel=%.3e (am %d/%d)\n",
                    rounds, write_base, state_slot, lrel(V1, Lr1.data()), av1, ar1,
                    lrel(V2c, Lr2b.data()), av2, ar2);
            auto brel = [&](const std::vector<char> & a, const std::vector<char> & b) {
                const float * x = (const float *)a.data(), * y = (const float *)b.data();
                double ln = 0, ld = 0;
                for (size_t j = 0; j < b.size() / 4; j++) { double d = x[j] - y[j]; ln += d * d; ld += (double)y[j] * y[j]; }
                return sqrt(ln / (ld + 1e-12));
            };
            double ws = 0, wc2 = 0, wk = 0, wv = 0;
            for (int i = 0; i < n_lin2; i++) {
                std::vector<char> sb3(sv_post[i].size()), cb3(cv_post[i].size());
                ggml_backend_tensor_get(ssm_state[i], sb3.data(), 0, sb3.size());
                ggml_backend_tensor_get(conv_state[i], cb3.data(), 0, cb3.size());
                ws = std::max(ws, brel(sv_post[i], sb3));
                wc2 = std::max(wc2, brel(cv_post[i], cb3));
            }
            const size_t colb3 = kv.k[0]->nb[1];
            for (int i = 0; i < n_full_layers; i++)
                for (int c = 0; c < 2; c++) {
                    std::vector<char> kb3(colb3), vb3(colb3);
                    ggml_backend_tensor_get(kv.k[i], kb3.data(), (size_t)(write_base + c) * colb3, colb3);
                    ggml_backend_tensor_get(kv.v[i], vb3.data(), (size_t)(write_base + c) * colb3, colb3);
                    const ggml_fp16_t * k0 = (const ggml_fp16_t *)kv_post[(i * 2 + c) * 2].data();
                    const ggml_fp16_t * k1 = (const ggml_fp16_t *)kb3.data();
                    const ggml_fp16_t * v0 = (const ggml_fp16_t *)kv_post[(i * 2 + c) * 2 + 1].data();
                    const ggml_fp16_t * v1 = (const ggml_fp16_t *)vb3.data();
                    double kn = 0, kd = 0, vn = 0, vd = 0;
                    for (size_t j = 0; j < colb3 / 2; j++) {
                        double d = ggml_fp16_to_fp32(k0[j]) - ggml_fp16_to_fp32(k1[j]); kn += d * d; kd += (double)ggml_fp16_to_fp32(k1[j]) * ggml_fp16_to_fp32(k1[j]);
                        d = ggml_fp16_to_fp32(v0[j]) - ggml_fp16_to_fp32(v1[j]); vn += d * d; vd += (double)ggml_fp16_to_fp32(v1[j]) * ggml_fp16_to_fp32(v1[j]);
                    }
                    wk = std::max(wk, sqrt(kn / (kd + 1e-12)));
                    wv = std::max(wv, sqrt(vn / (vd + 1e-12)));
                }
            fprintf(stderr, "[livechk]     state: ssm=%.3e conv=%.3e | KV k=%.3e v=%.3e\n", ws, wc2, wk, wv);
            // continue on the reference chain (identical if verify is exact)
        }
        // fast path: the two argmaxes come from the verify graph (device-side);
        // the full-logits CPU argmax only exists under the debug switches
        const bool have_logits = (int)L2.size() == 2 * n_vocab;
        const float * L1 = have_logits ? L2.data() : nullptr;
        const float * L2c = have_logits ? L2.data() + n_vocab : nullptr;
        int g1 = v_am[0];
        if (have_logits) { g1 = 0; for (int i = 1; i < n_vocab; i++) if (L1[i] > L1[g1]) g1 = i; }
        if (trace) fprintf(stderr, "[trace] r%ld wb=%d cur=%d slot=%d seed=%d pend=%d d1=%d g1=%d",
                           rounds, write_base, cur, state_slot, seed_tok, pending, d1, g1);
        // emit the (already validated) token
        const std::string piece = tok_.decode({pending});
        raw += piece;
        if (gp.on_token) gp.on_token(piece.c_str(), gp.on_token_ud);
        n_dec++;
        rounds++;
        if (trace) fprintf(stderr, "%s", "\n");
        static const bool force_rej = getenv("QWEN35_SPEC_FORCE_REJ") != nullptr;
        if (g1 == d1 && !force_rej) {
            if (d1 == id_eos || d1 == id_endoftext || n_dec >= max_new) break;
            const std::string p2 = tok_.decode({d1});
            raw += p2;
            if (gp.on_token) gp.on_token(p2.c_str(), gp.on_token_ud);
            n_dec++; accepted++;
            int g2 = v_am[1];
            if (have_logits) { g2 = 0; for (int i = 1; i < n_vocab; i++) if (L2c[i] > L2c[g2]) g2 = i; }
            if (trace) fprintf(stderr, " ACC d1=%d g2=%d", d1, g2);
            seed_tok = d1; pending = g2;
            if (have_logits) L_prev.assign(L2c, L2c + n_vocab);
            h.assign(h_d.begin() + n_embd, h_d.end());
            state_slot = 0; write_base += 2; cur += 2;
            ema = 0.8 * ema + 0.2;
        } else {
            // rollback keeps the state after `pending`: that IS the new seed
            seed_tok = pending; pending = g1;
            if (have_logits) L_prev.assign(L1, L1 + n_vocab);
            h.assign(h_d.begin(), h_d.begin() + n_embd);
            state_slot = 1; write_base += 1; cur += 1;
            ema = 0.8 * ema;
        }
    }
    n_past = write_base;
    last_mpos = { cur, cur, cur };
    fprintf(stderr, "[spec] %d rounds (%d plain), %ld double-hits (%.0f%%), %d tokens, %.1f tok/round\n",
            (int)rounds, plain_streak_total, accepted, rounds ? 100.0 * accepted / rounds : 0.0f, n_dec,
            rounds ? (double)n_dec / rounds : 0.0);
    return true;
}
