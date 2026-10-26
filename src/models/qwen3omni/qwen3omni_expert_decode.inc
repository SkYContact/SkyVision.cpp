// qwen3omni_expert_decode.inc - included at the bottom of qwen3omni_model.cpp
// (inside namespace qwen3omni; member definitions of Qwen3Omni).
//
// Expert-granular GPU-resident dynamic decode path (Strata-style tier 2),
// enabled with QOMNI_EXPERTCACHE=1. The orchestration (KV-block rebuild,
// acquire/pin, one-layer-ahead prefetch, graph B slot tables, graph C, the
// timers) lives in the shared core/engine/moe_decode_driver.hpp; what stays
// here is the qwen3omni-specific half:
//   * graph A: the layer's own T=1 op sequence, copied VERBATIM from
//     build_text_graph (fused IMROPE default, QOMNI_NOFUSE honored) - the
//     math must not change
//   * input feeding: the 4-plane [t, h, w, slot] mpos row + F16 mask
//   * trajectory capture: layer-0's x_o is t_emb_out, layer-23's post-graph-B
//     hidden is t_h24_out (talker --speak unchanged)
//   * QOMNI_XCHK semantic cross-check vs the standard decode graph
//   * QOMNI_KVSNAP: after each xchk ref run, snapshot the ref-written KV
//     row of every layer and diff it against the EC step write (locates
//     the layer where the two paths first diverge)
//   * QOMNI_EC_LAYERCHK: per-layer parity of the slot-table MoE (graph B)
//     against a standard mul_mat_id over the original tensors, given the
//     same graph-A outputs (isolates MoE vs attention on any drift)
//
// GEOMETRY PARITY (load-bearing): the xchk reference walks the standard
// decode bucket, so the driver EC KV block must stay aligned with it
// (decode_bucket: 128 CPU / 1024 GPU) - flash_attn_ext is not bit-exact
// across padded-window widths, and a 1-ulp attention difference flips
// router top-k near-ties into visible drift.
//
// CPU validation mode: on a CPU backend the cache lives in host RAM and
// everything still runs - used to check the semantics/logits before any GPU
// placement (QOMNI_XCHK=1 must print maxdiff 0.00000).
//
// Failure policy: if the cache cannot be initialized (no CUDA / no VRAM),
// QOMNI_EXPERTCACHE is disabled and decode falls back to the existing path.

static Qwen3Omni * g_ec_self = nullptr;   // trampoline target for InitSpec::tensor

bool Qwen3Omni::ec_init() {
    ec_ok = false;
    bool cpu_mode = false;
    if (ggml_backend_is_cpu(backend)) {
        // CPU validation mode (whole model already in the host mmap)
        split_mode = false;
        cpu_mode = true;
    } else if (!split_mode) {
        fprintf(stderr, "qomni ec: no GPU split placement available,"
                " QOMNI_EXPERTCACHE disabled\n");
        ec_enabled = false;
        return true;
    }
    g_ec_self = this;
    // isolation knobs (must match the reference path: build_text_graph
    // honors QOMNI_MAX_LAYERS too, so the xchk comparison stays
    // layer-for-layer exact); QOMNI_EC_BUDGET resizes the slot pool.
    int ec_n = n_layer;
    size_t ec_budget = (size_t)4.0e9;
    if (const char * e = getenv("QOMNI_MAX_LAYERS")) ec_n = std::min(ec_n, atoi(e));
    if (const char * e = getenv("QOMNI_EC_BUDGET")) ec_budget = (size_t)atof(e);
    struct Tr { static const ggml_tensor * get(int l, int t) {
        Qwen3Omni & M = *g_ec_self;   // 0 up / 1 gate / 2 down
        const TxtLayer & L = M.tl[(size_t)l];
        return t == 0 ? L.up_exps : t == 1 ? L.gate_exps : L.down_exps;
    }};
    vqa::moe_decode::InitSpec sp{
        backend,
        ec_n, n_expert, n_expert_used,
        &Tr::get,
        cpu_mode ? ec_budget : (size_t)0,
        (size_t)2.0e9,
        true,               // stage_threads_16 (measured 4.1 -> 10.2 tok/s)
        "qomni ec",
        ec_enabled,
        ec_caches,
        ec_group,
        ec_n_groups,
        ec_hooks.async_ok,
    };
    if (!vqa::moe_decode::init_grouped(sp)) return false;
    if (!ec_enabled) return true;
    ec_hooks.n_layer = ec_n;
    ec_hooks.n_embd = n_embd;
    ec_hooks.n_expert_used = n_expert_used;
    ec_hooks.n_vocab = n_vocab;
    ec_hooks.rms_eps = rms_eps;
    ec_hooks.backend = backend;
    ec_hooks.allocr = allocr;
    ec_hooks.out_norm = out_norm;
    ec_hooks.out_w = out_w;
    ec_hooks.caches = &ec_caches;
    ec_hooks.group = &ec_group;
    ec_hooks.log_tag = "qomni ec";
    ec_hooks.pf_env = "QOMNI_EC_PF";
    ec_hooks.spec_env = "QOMNI_EC_SPEC";
    ec_hooks.build_a = &Qwen3Omni::ec_build_a;
    ec_hooks.feed_a = &Qwen3Omni::ec_feed_a;
    ec_hooks.post_a = &Qwen3Omni::ec_post_a;
    ec_hooks.post_b = &Qwen3Omni::ec_post_b;
    ec_ga_io.assign((size_t)ec_n, EcGraphA());
    ec_ok = true;
    return true;
}

// ------------------------------------------------------------------ graph A
// Fixed per-layer graph. The op sequence is copied VERBATIM from
// build_text_graph (T=1) - do not reorder.
ggml_tensor * Qwen3Omni::ec_ga_builder(ggml_context * ca, void * ud_v) {
    EcGaUd & ud = *(EcGaUd *) ud_v;
    Qwen3Omni & M = *ud.m;
    const int il = ud.il;
    EcGraphA & io = *ud.io;
    ggml_cgraph * fa = vqa::step_graph_builder_gf();

    const int D = M.n_embd, HD = M.head_dim, NH = M.n_head, NKV = M.n_kv_head;
    const int T = 1;
    const int S = M.ec_st.S;

    io.mpos_in = ggml_new_tensor_1d(ca, GGML_TYPE_I32, 4 * T);
    io.mask_in = ggml_new_tensor_2d(ca, GGML_TYPE_F16, S, T);
    io.tok_in  = ggml_new_tensor_1d(ca, GGML_TYPE_I32, T);
    io.hid_in  = ggml_new_tensor_2d(ca, GGML_TYPE_F32, D, T);

    ggml_tensor * x;
    if (il == 0) {
        x = ggml_get_rows(ca, M.tok_embd, io.tok_in);
    } else {
        x = io.hid_in;
    }

    const TxtLayer & L = M.tl[(size_t)il];

    ggml_tensor * l = ggml_rms_norm(ca, x, M.rms_eps);
    l = ggml_mul(ca, l, L.norm);

    ggml_tensor * q = ggml_mul_mat(ca, L.q, l);
    ggml_tensor * k = ggml_mul_mat(ca, L.k, l);
    ggml_tensor * v = ggml_mul_mat(ca, L.v, l);

    ggml_tensor * q3 = ggml_reshape_3d(ca, q, HD, NH, T);
    q3 = ggml_rms_norm(ca, q3, M.rms_eps);
    q3 = ggml_mul(ca, q3, L.qn);
    ggml_tensor * k3 = ggml_reshape_3d(ca, k, HD, NKV, T);
    k3 = ggml_rms_norm(ca, k3, M.rms_eps);
    k3 = ggml_mul(ca, k3, L.kn);

    // interleaved MRoPE via the [t,h,w,slot] mpos row (see build_text_graph)
    ggml_tensor * pos_rope = ggml_view_1d(ca, io.mpos_in, 4 * T, 0);
    ggml_tensor * pos_slot = ggml_view_1d(ca, io.mpos_in, T, (size_t)3 * T * sizeof(int));
    static const bool nofuse = getenv("QOMNI_NOFUSE") != nullptr;
    if (nofuse) {
        q3 = ggml_rope_multi(ca, q3, pos_rope, nullptr, M.rope_dim, M.mrope_sections,
                             GGML_ROPE_TYPE_IMROPE, M.n_ctx, M.rope_theta,
                             1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k3 = ggml_rope_multi(ca, k3, pos_rope, nullptr, M.rope_dim, M.mrope_sections,
                             GGML_ROPE_TYPE_IMROPE, M.n_ctx, M.rope_theta,
                             1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    } else {
        ggml_tensor * qk = ggml_concat(ca, q3, k3, 1);          // [HD, NH+NKV, T]
        qk = ggml_rope_multi(ca, qk, pos_rope, nullptr, M.rope_dim, M.mrope_sections,
                             GGML_ROPE_TYPE_IMROPE, M.n_ctx, M.rope_theta,
                             1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        q3 = ggml_view_3d(ca, qk, HD, NH, T, qk->nb[1], qk->nb[2], 0);
        k3 = ggml_view_3d(ca, qk, HD, NKV, T, qk->nb[1], qk->nb[2],
                          (size_t)NH * qk->nb[1]);
    }

    ggml_tensor * k2 = ggml_reshape_2d(ca, ggml_cont(ca, k3), NKV * HD, T);
    ggml_tensor * v2 = ggml_reshape_2d(ca, v, NKV * HD, T);
    ggml_build_forward_expand(fa, ggml_set_rows(ca, M.kv.k[(size_t)il], k2, pos_slot));
    ggml_build_forward_expand(fa, ggml_set_rows(ca, M.kv.v[(size_t)il], v2, pos_slot));

    ggml_tensor * kf = ggml_view_3d(ca, M.kv.k[(size_t)il], HD, S, NKV,
                                    M.kv.k[(size_t)il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
    ggml_tensor * vf = ggml_view_3d(ca, M.kv.v[(size_t)il], HD, S, NKV,
                                    M.kv.v[(size_t)il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
    ggml_tensor * qf = ggml_view_3d(ca, q3, HD, T, NH, q3->nb[2], q3->nb[1], 0);
    ggml_tensor * fa_attn = ggml_flash_attn_ext(ca, qf, kf, vf, io.mask_in,
                                                1.0f / sqrtf((float)HD), 0.0f, 0.0f);
    ggml_tensor * attn = ggml_reshape_2d(ca, ggml_cont(ca, fa_attn), NH * HD, T);
    attn = ggml_mul_mat(ca, L.o, attn);
    x = ggml_add(ca, x, attn);                       // x_mid (residual after attn)

    // ---- router: same softmax-over-selected renormalization as prefill ----
    ggml_tensor * m = ggml_rms_norm(ca, x, M.rms_eps);
    m = ggml_mul(ca, m, L.ffn_norm);
    ggml_tensor * rlogits = ggml_mul_mat(ca, L.router, m);              // [E, T]
    ggml_tensor * sel = ggml_top_k(ca, rlogits, M.n_expert_used);       // [K, T] I32
    ggml_tensor * sel_logits = ggml_get_rows(ca,
            ggml_reshape_3d(ca, rlogits, 1, M.n_expert, T), sel);
    sel_logits = ggml_reshape_2d(ca, sel_logits, M.n_expert_used, T);
    ggml_tensor * w8 = ggml_soft_max(ca, sel_logits);                   // [K, T]

    // dup'd tails: keep their buffers for the whole compute (see keye's note)
    io.w8_o = ggml_dup(ca, w8);
    io.sel_o = ggml_dup(ca, sel);
    io.m_o = ggml_dup(ca, m);
    io.x_o = ggml_dup(ca, x);
    ggml_set_name(io.w8_o, "qec_w8_o");
    ggml_set_name(io.sel_o, "qec_sel_o");
    ggml_set_name(io.m_o, "qec_m_o");
    ggml_set_name(io.x_o, "qec_x_o");
    ggml_build_forward_expand(fa, io.w8_o);
    ggml_build_forward_expand(fa, io.sel_o);
    ggml_build_forward_expand(fa, io.m_o);
    ggml_build_forward_expand(fa, io.x_o);
    return io.x_o;
}

bool Qwen3Omni::ec_build_a(void * m, int il, vqa::StepGraph & sg,
                           vqa::moe_decode::AIO & io) {
    Qwen3Omni & M = *(Qwen3Omni *) m;
    EcGaUd ud = { &M, il, &M.ec_ga_io[(size_t)il] };
    if (!sg.build(M.backend, ec_ga_builder, &ud)) return false;
    EcGraphA & eg = M.ec_ga_io[(size_t)il];
    io.tok_in = eg.tok_in;
    io.hid_in = eg.hid_in;
    io.m_o = eg.m_o;
    io.x_o = eg.x_o;
    io.sel_o = eg.sel_o;
    io.w8_o = eg.w8_o;
    io.model_io = &eg;
    return true;
}

bool Qwen3Omni::ec_feed_a(void * m, int il, vqa::moe_decode::AIO & io,
                          vqa::StepGraph & sg, const void * step_ctx,
                          int cur_tok, int n_past, const float * hid) {
    (void)cur_tok; (void)hid;
    Qwen3Omni & M = *(Qwen3Omni *) m;
    EcGraphA & eg = *(EcGraphA *) io.model_io;
    const int S = M.ec_st.S;
    if (M.ec_mask.size() != (size_t)S || M.ec_mask_n != n_past) {
        M.ec_mask.assign((size_t)S, ggml_fp32_to_fp16(-INFINITY));
        for (int sq = 0; sq <= n_past; sq++) M.ec_mask[(size_t)sq] = ggml_fp32_to_fp16(0.0f);
        M.ec_mask_n = n_past;
    }
    return sg.set_input(eg.mpos_in, step_ctx, sizeof(int) * 4) &&
           sg.set_input(eg.mask_in, M.ec_mask.data(),
                        sizeof(ggml_fp16_t) * M.ec_mask.size());
}

namespace qomni_lc { static float xo[2048], mm[2048], w8[16]; static int sel[16]; static int step = 0, bad = 0; }

void Qwen3Omni::ec_post_a(void * m, int il, const float * xmid, const float * mn,
                          const int * sel, const float * w8) {
    Qwen3Omni & M = *(Qwen3Omni *) m;
    if (il == 0 && M.ec_traj_emb)
        M.ec_traj_emb->assign(xmid, xmid + M.n_embd);
    // QOMNI_EC_LAYERCHK: stash the graph-A outputs so ec_post_b can rebuild
    // this layer MoE from the ORIGINAL mmap experts and compare.
    if (getenv("QOMNI_EC_LAYERCHK") && M.n_embd <= 2048 && M.n_expert_used <= 16) {
        memcpy(qomni_lc::xo, xmid, sizeof(float) * (size_t)M.n_embd);
        memcpy(qomni_lc::mm, mn, sizeof(float) * (size_t)M.n_embd);
        memcpy(qomni_lc::sel, sel, sizeof(int) * (size_t)M.n_expert_used);
        memcpy(qomni_lc::w8, w8, sizeof(float) * (size_t)M.n_expert_used);
    }
    (void)il;
}

void Qwen3Omni::ec_post_b(void * m, int il, const float * hid) {
    Qwen3Omni & M = *(Qwen3Omni *) m;
    if (il == 23 && M.ec_traj_h24)
        M.ec_traj_h24->assign(hid, hid + M.n_embd);

    // QOMNI_EC_LAYERCHK: per-layer parity of the EC slot-table MoE (graph B)
    // against a standard mul_mat_id over the original expert tensors, given
    // the exact same graph-A outputs. Locates the first diverging layer.
    static const bool lc_on = getenv("QOMNI_EC_LAYERCHK") != nullptr;
    if (!lc_on || M.n_embd > 2048 || M.n_expert_used > 16) return;
    if (il == 0) qomni_lc::step++;

    const int D = M.n_embd, K = M.n_expert_used;
    ggml_context * cc = ggml_init({ 32ull << 20, NULL, true });
    if (!cc) return;
    ggml_cgraph * fc = ggml_new_graph_custom(cc, 2048, false);
    ggml_tensor * m_in  = ggml_new_tensor_2d(cc, GGML_TYPE_F32, D, 1);
    ggml_tensor * xo_in = ggml_new_tensor_2d(cc, GGML_TYPE_F32, D, 1);
    ggml_tensor * sel_in = ggml_new_tensor_2d(cc, GGML_TYPE_I32, K, 1);
    ggml_tensor * w_in  = ggml_new_tensor_2d(cc, GGML_TYPE_F32, K, 1);
    const TxtLayer & L = M.tl[(size_t)il];
    ggml_tensor * cur3 = ggml_reshape_3d(cc, m_in, D, 1, 1);
    ggml_tensor * gu = ggml_mul_mat_id(cc, L.gate_exps, cur3, sel_in);
    ggml_tensor * up = ggml_mul_mat_id(cc, L.up_exps, cur3, sel_in);
    ggml_tensor * act = ggml_swiglu_split(cc, gu, up);
    ggml_tensor * ff3 = ggml_mul_mat_id(cc, L.down_exps, act, sel_in);
    ggml_tensor * w3 = ggml_reshape_3d(cc, w_in, 1, K, 1);
    ggml_tensor * wb = ggml_view_3d(cc, w3, 1, K, 1, w3->nb[1], w3->nb[2], 0);
    ggml_tensor * prod = ggml_mul(cc, ff3, wb);
    ggml_tensor * pk = ggml_cont(cc, ggml_permute(cc, prod, 1, 0, 2, 3));
    ggml_tensor * moe = ggml_reshape_2d(cc, ggml_sum_rows(cc, pk), D, 1);
    ggml_tensor * out = ggml_add(cc, xo_in, moe);
    ggml_build_forward_expand(fc, out);
    if (!ggml_gallocr_alloc_graph(M.allocr, fc)) { ggml_free(cc); return; }
    ggml_backend_tensor_set(m_in, qomni_lc::mm, 0, sizeof(float) * (size_t)D);
    ggml_backend_tensor_set(xo_in, qomni_lc::xo, 0, sizeof(float) * (size_t)D);
    ggml_backend_tensor_set(sel_in, qomni_lc::sel, 0, sizeof(int) * (size_t)K);
    ggml_backend_tensor_set(w_in, qomni_lc::w8, 0, sizeof(float) * (size_t)K);
    if (ggml_backend_graph_compute(M.backend, fc) == GGML_STATUS_SUCCESS) {
        std::vector<float> ref((size_t)D);
        ggml_backend_tensor_get(out, ref.data(), 0, sizeof(float) * (size_t)D);
        double md = 0; int am = 0;
        for (int v = 0; v < D; v++) {
            const double d = std::fabs((double)ref[(size_t)v] - (double)hid[v]);
            if (d > md) { md = d; am = v; }
        }
        if (md > 1e-4) {
            ++qomni_lc::bad;
            fprintf(stderr, "[lc] LAYER MISMATCH step %d layer %d maxdiff %.6g at %d (ec %.6f ref %.6f)\n",
                    qomni_lc::step, il, md, am, hid[am], ref[(size_t)am]);
        } else if (il < 2) {
            fprintf(stderr, "[lc] ok step %d layer %d maxdiff %.3g\n", qomni_lc::step, il, md);
        }
    }
    ggml_free(cc);
}

bool Qwen3Omni::ec_decode_step(int cur_tok, const int * mpos4, int n_past,
                               std::vector<float> & logits,
                               std::vector<float> * emb_out,
                               std::vector<float> * h24_out) {
    // QOMNI_XCHK: run the standard decode graph for the same step and compare
    // the logits against the EC path (CPU-mode semantic validation)
    static std::vector<uint8_t> kv_snap;
    bool kv_snap_on = false;
    std::vector<float> ref_logits;
    const bool xchk = getenv("QOMNI_XCHK") != nullptr;
    const bool kv_snap_on_req = getenv("QOMNI_KVSNAP") != nullptr;
    if (xchk) {
        const int n_past_saved = n_past;
        TextGraph rg;
        const std::vector<int> toks = { cur_tok };
        const std::vector<int> mp(mpos4, mpos4 + 4);
        if (!run_text(rg, toks, nullptr, nullptr, mp, ref_logits)) {
            ggml_free(rg.ctx);
            return false;
        }
        ggml_free(rg.ctx);
        this->n_past = n_past_saved;   // run_text advanced the MEMBER; undo it
        // (a bare n_past here assigns the shadowing parameter and leaks
        // +1/step into the KV position). It also keeps the xchk ref on the
        // exact same KV geometry as the EC step and the production path.
        kv_snap_on = kv_snap_on_req;
        if (kv_snap_on) {
            kv_snap.clear();
            for (int il2 = 0; il2 < n_layer; il2++) {
                const ggml_tensor * kt[2] = { kv.k[(size_t)il2], kv.v[(size_t)il2] };
                for (int t2 = 0; t2 < 2; t2++) {
                    const size_t row = kt[t2]->nb[1];
                    const size_t off = (size_t)n_past * row;
                    const size_t at = kv_snap.size();
                    kv_snap.resize(at + row);
                    ggml_backend_tensor_get(kt[t2], kv_snap.data() + at, off, row);
                }
            }
        }
    }
    ec_traj_emb = emb_out;
    ec_traj_h24 = h24_out;
    if (!vqa::moe_decode::decode_step(ec_hooks, ec_st, this, mpos4, cur_tok,
                                      n_past, logits))
        return false;
    if (xchk && kv_snap_on) {
        int first_bad_layer = -1;
        size_t at = 0;
        for (int il2 = 0; il2 < n_layer && first_bad_layer < 0; il2++) {
            const ggml_tensor * kt[2] = { kv.k[(size_t)il2], kv.v[(size_t)il2] };
            for (int t2 = 0; t2 < 2; t2++) {
                const size_t row = kt[t2]->nb[1];
                const size_t off = (size_t)n_past * row;
                std::vector<uint8_t> now(row);
                ggml_backend_tensor_get(kt[t2], now.data(), off, row);
                if (memcmp(now.data(), kv_snap.data() + at, row) != 0) {
                    first_bad_layer = il2;
                    size_t bd = 0;
                    while (bd < row && now[bd] == kv_snap[at + bd]) bd++;
                    fprintf(stderr, "[kvsnap] step %d layer %d %s row %d differs at byte %zu/%zu\n",
                            qomni_lc::step, il2, t2 == 0 ? "k" : "v", n_past, bd, row);
                }
                at += row;
            }
        }
        if (first_bad_layer < 0) fprintf(stderr, "[kvsnap] step %d all rows identical\n", qomni_lc::step);
    }
    if (xchk) {
        double dmax = 0; int im = 0;
        for (int v = 0; v < n_vocab; v++) {
            double d = fabs((double)logits[(size_t)v] - ref_logits[(size_t)v]);
            if (d > dmax) { dmax = d; im = v; }
        }
        fprintf(stderr, "xchk logits maxdiff %.5f at id %d (ec %.4f ref %.4f)\n",
                dmax, im, logits[(size_t)im], ref_logits[(size_t)im]);
    }
    return true;
}
