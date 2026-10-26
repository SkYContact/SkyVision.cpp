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
    struct Tr { static const ggml_tensor * get(int l, int t) {
        Qwen3Omni & M = *g_ec_self;   // 0 up / 1 gate / 2 down
        const TxtLayer & L = M.tl[(size_t)l];
        return t == 0 ? L.up_exps : t == 1 ? L.gate_exps : L.down_exps;
    }};
    vqa::moe_decode::InitSpec sp{
        backend,
        n_layer, n_expert, n_expert_used,
        &Tr::get,
        cpu_mode ? (size_t)4.0e9 : (size_t)0,
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
    ec_hooks.n_layer = n_layer;
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
    ec_ga_io.assign((size_t)n_layer, EcGraphA());
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
    std::vector<ggml_fp16_t> mask((size_t)S, ggml_fp32_to_fp16(-INFINITY));
    for (int sq = 0; sq <= n_past; sq++) mask[(size_t)sq] = ggml_fp32_to_fp16(0.0f);
    return sg.set_input(eg.mpos_in, step_ctx, sizeof(int) * 4) &&
           sg.set_input(eg.mask_in, mask.data(), sizeof(ggml_fp16_t) * mask.size());
}

void Qwen3Omni::ec_post_a(void * m, int il, const float * xmid, const float * mn,
                          const int * sel, const float * w8) {
    (void)mn; (void)sel; (void)w8;
    Qwen3Omni & M = *(Qwen3Omni *) m;
    if (il == 0 && M.ec_traj_emb)
        M.ec_traj_emb->assign(xmid, xmid + M.n_embd);
}

void Qwen3Omni::ec_post_b(void * m, int il, const float * hid) {
    Qwen3Omni & M = *(Qwen3Omni *) m;
    if (il == 23 && M.ec_traj_h24)
        M.ec_traj_h24->assign(hid, hid + M.n_embd);
}

bool Qwen3Omni::ec_decode_step(int cur_tok, const int * mpos4, int n_past,
                               std::vector<float> & logits,
                               std::vector<float> * emb_out,
                               std::vector<float> * h24_out) {
    // QOMNI_XCHK: run the standard decode graph for the same step and compare
    // the logits against the EC path (CPU-mode semantic validation)
    std::vector<float> ref_logits;
    const bool xchk = getenv("QOMNI_XCHK") != nullptr;
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
        n_past = n_past_saved;   // run_text advanced it; undo (the standard
        // path rewrote the exact same KV slot with the exact same k/v, so the
        // cache content itself is unchanged)
    }
    ec_traj_emb = emb_out;
    ec_traj_h24 = h24_out;
    if (!vqa::moe_decode::decode_step(ec_hooks, ec_st, this, mpos4, cur_tok,
                                      n_past, logits))
        return false;
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
