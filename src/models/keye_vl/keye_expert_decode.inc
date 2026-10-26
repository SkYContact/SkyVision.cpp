// keye_expert_decode.inc - included at the bottom of keye_vl_model.cpp
// (inside namespace keye_vl; member definitions of KeyeVL2).
//
// Expert-granularity GPU-resident dynamic decode path (Strata-style tier 2),
// enabled with KEYE_EXPERTCACHE=1. Implies the VQA_SPLIT placement (dense
// weights on the GPU, expert tensors left in the host mmap) but the experts
// are NOT computed on the CPU. Each decode step runs, per layer,
//   graph A  (FIXED StepGraph, built once by ec_build_graphs): embed/hid ->
//            attn_norm -> DSA indexer -> GQA attention (KV + lid set_rows) ->
//            residual -> ffn_norm -> router -> top-8 ids (+ renorm softmax w)
//   host     read the 8 expert ids back, ExpertCache::acquire -> device slots
//   graph B  (rebuilt per layer - slot tensor pointers change - but allocated
//            through the shared gallocr so same-shape buffers are reused):
//            moe::build_expert_ffn over the 8 slot experts -> residual
// followed by output_norm + output head for the logits.
//
// v2 changes (interface unchanged, KEYE_EXPERTCACHE=1 behavior upgrade):
//   * graph A is a per-layer vqa::StepGraph with S = block_for(n_ctx, 512)
//     FIXED for the whole session (mask leaf re-fed every step: 0 for the
//     n_past+1 live slots, -inf for the padding), eliminating the v1 per-step
//     48x graph-A rebuild + allocation. Each holder owns a PRIVATE gallocr:
//     ggml_gallocr re-reserves (and frees) its pool whenever a
//     differently-shaped graph is allocated on it, so a shared allocr would
//     invalidate the fixed graphs' tensor pointers when graph B alternates
//     through it every step.
//   * the MoE segment is the generic vqa::moe::build_expert_ffn; the router
//     weights are a [8, 1] graph input instead of ggml_scale constants (same
//     single F32 multiply per element - math unchanged).
//   * async prefetch (expert_cache v2): after begin_step the previous step's
//     selections for every layer are prefetched (LRU hot-expert speculation),
//     and each layer's fresh selections are prefetched right after the graph-A
//     readback, just before acquire. If enable_async() failed (no CUDA / stub
//     cache), prefetch is a no-op and everything degrades to v1 sync behavior.
//
// The op sequence inside graph A is copied verbatim from build_text_graph
// (T=1); the math must not change (SOP rule). Prefill is untouched: it still
// runs the existing single big graph (split scheduler in this mode).
//
// Failure policy: if the cache cannot be initialized (no CUDA / no VRAM),
// KEYE_EXPERTCACHE is disabled and decode falls back to the existing path.

// vqa::step_graph_builder_gf() (step_graph.cpp: the cgraph under construction
// during StepGraph::build(), used below to attach the ggml_set_rows KV-write
// side effects) and vqa::moe::build_expert_ffn_res (residual-splitting
// overload: the frozen header's build_expert_ffn uses x itself as the
// residual; keye's residual is x_mid, not the ffn_norm output m) are both
// declared at GLOBAL scope in keye_vl_model.cpp - opening `namespace vqa`
// here would shadow ::vqa and break every vqa:: reference below.

static inline double ec_ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t).count();
}

static KeyeVL2 * g_keye_ec_self = nullptr;   // trampoline target for InitSpec::tensor

bool KeyeVL2::ec_init() {
    ec_ok = false;
    if (!split_mode || ggml_backend_is_cpu(backend)) {
        fprintf(stderr, "keye ec: no GPU split placement available, KEYE_EXPERTCACHE disabled\n");
        ec_enabled = false;
        return true;
    }
    g_keye_ec_self = this;
    struct Tr { static const ggml_tensor * get(int l, int t) {
        KeyeVL2 & M = *g_keye_ec_self;   // 0 up / 1 gate / 2 down
        const TxtLayer & L = M.tl[(size_t)l];
        return t == 0 ? L.up_exps : t == 1 ? L.gate_exps : L.down_exps;
    }};
    vqa::moe_decode::InitSpec sp{
        backend,
        n_layer, n_expert, n_expert_used,
        &Tr::get,
        (size_t)0,          // no CPU validation mode for keye
        (size_t)2.0e9,
        false,              // staging threads stay at the library default
        "keye ec",
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
    ec_hooks.log_tag = "keye ec";
    ec_hooks.pf_env = "KEYE_EC_PF";
    ec_hooks.spec_env = "KEYE_EC_SPEC";
    ec_hooks.build_a = &KeyeVL2::ec_build_a;
    ec_hooks.feed_a = &KeyeVL2::ec_feed_a;
    ec_hooks.post_a = &KeyeVL2::ec_post_a;
    ec_hooks.attn_only = &KeyeVL2::ec_attn_only;
    ec_hooks.trace = &KeyeVL2::ec_trace;
    ec_hooks.layer_check = &KeyeVL2::ec_layer_check;
    ec_ga_io.assign((size_t)n_layer, EcGraphA());
    ec_ok = true;
    return true;
}

// ------------------------------------------------------------------ v2 graph A
// Fixed per-layer graph. The op sequence is copied VERBATIM from the v1
// inline builder (which itself copies build_text_graph T=1) - do not reorder.
ggml_tensor * KeyeVL2::ec_ga_builder(ggml_context * ca, void * ud_v) {
    EcGaUd & ud = *(EcGaUd *) ud_v;
    KeyeVL2 & M = *ud.m;
    const int il = ud.il;
    EcGraphA & io = *ud.io;
    ggml_cgraph * fa = vqa::step_graph_builder_gf();

    const int D = M.n_embd, HD = M.head_dim, NH = M.n_head, NKV = M.n_kv_head;
    const int IXH = M.ix_head, IXD = M.ix_head_dim;
    const int T = 1;
    const int S = M.ec_st.S;
    const int K_SEL = std::min<int>(S, M.ix_top_k);

    io.pos_in  = ggml_new_tensor_1d(ca, GGML_TYPE_I32, 4 * T);
    io.slot_in = ggml_new_tensor_1d(ca, GGML_TYPE_I32, T);
    io.mask_in = ggml_new_tensor_2d(ca, GGML_TYPE_F32, S, T);
    io.had     = ggml_new_tensor_2d(ca, GGML_TYPE_F32, IXD, IXD);
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
    l = ggml_mul(ca, l, L.attn_norm);

    // ---- DSA indexer (op sequence copied verbatim from build_text_graph) ----
    ggml_tensor * top_k = nullptr;
    if (!M.no_dsa) {
        int isec[4] = { M.mrope_section[0] / 2, M.mrope_section[1] / 2,
                        M.mrope_section[2] / 2, 0 };
        const float ix_scale = 1.0f / sqrtf((float)IXD);
        ggml_tensor *iq = ggml_mul_mat(ca, L.ix_q_b, l);              // [1024, T]
        iq = ggml_reshape_3d(ca, iq, IXD, IXH, T);
        iq = ggml_rms_norm(ca, iq, M.rms_eps);
        iq = ggml_mul(ca, iq, L.ix_q_norm);
        iq = ggml_rope_multi(ca, iq, io.pos_in, nullptr, IXD, isec,
                             GGML_ROPE_TYPE_MROPE, M.n_ctx, M.rope_theta,
                             1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        iq = ggml_mul_mat(ca, io.had, iq);

        ggml_tensor *ik = ggml_mul_mat(ca, L.ix_k, l);                // [64, T] MQA
        ik = ggml_norm(ca, ik, 1e-6f);
        ik = ggml_add(ca, ggml_mul(ca, ik, L.ix_k_norm), L.ix_k_norm_b);
        ik = ggml_reshape_3d(ca, ik, IXD, 1, T);
        ik = ggml_rope_multi(ca, ik, io.pos_in, nullptr, IXD, isec,
                             GGML_ROPE_TYPE_MROPE, M.n_ctx, M.rope_theta,
                             1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ik = ggml_mul_mat(ca, io.had, ik);
        ggml_tensor *ik2 = ggml_cast(ca, ggml_reshape_2d(ca, ik, IXD, T), GGML_TYPE_F16);
        ggml_build_forward_expand(fa, ggml_set_rows(ca, M.lid_k[(size_t)il], ik2, io.slot_in));

        ggml_tensor *iw = ggml_scale(ca, ggml_mul_mat(ca, L.ix_proj, l), ix_scale);
        ggml_tensor *kc = ggml_view_2d(ca, M.lid_k[(size_t)il], IXD, S,
                                       M.lid_k[(size_t)il]->nb[1], 0);
        ggml_tensor *q3 = ggml_cont(ca, ggml_permute(ca, iq, 0, 2, 1, 3));
        ggml_tensor *k3 = ggml_reshape_3d(ca, kc, IXD, S, 1);
        ggml_tensor *kq = ggml_mul_mat(ca, k3, q3);
        ggml_tensor *sc = ggml_cont(ca, ggml_permute(ca, kq, 2, 1, 0, 3));
        sc = ggml_relu(ca, sc);
        sc = ggml_mul(ca, sc, ggml_reshape_3d(ca, iw, IXH, T, 1));
        sc = ggml_sum_rows(ca, sc);
        sc = ggml_reshape_2d(ca, ggml_cont(ca, ggml_permute(ca, sc, 2, 1, 0, 3)), S, T);
        sc = ggml_add(ca, sc, io.mask_in);
        top_k = ggml_cont(ca, ggml_top_k(ca, sc, K_SEL));
    }

    // ---- GQA attention with per-head q/k norm + mRoPE ----
    int sec[4] = { M.mrope_section[0], M.mrope_section[1], M.mrope_section[2], 0 };
    ggml_tensor *q = ggml_mul_mat(ca, L.wq, l);
    ggml_tensor *k = ggml_mul_mat(ca, L.wk, l);
    ggml_tensor *v = ggml_mul_mat(ca, L.wv, l);
    ggml_tensor *q3 = ggml_reshape_3d(ca, q, HD, NH, T);
    q3 = ggml_rms_norm(ca, q3, M.rms_eps);
    q3 = ggml_mul(ca, q3, L.qn);
    q3 = ggml_rope_multi(ca, q3, io.pos_in, nullptr, HD, sec, GGML_ROPE_TYPE_MROPE,
                         M.n_ctx, M.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor *k3 = ggml_reshape_3d(ca, k, HD, NKV, T);
    k3 = ggml_rms_norm(ca, k3, M.rms_eps);
    k3 = ggml_mul(ca, k3, L.kn);
    k3 = ggml_rope_multi(ca, k3, io.pos_in, nullptr, HD, sec, GGML_ROPE_TYPE_MROPE,
                         M.n_ctx, M.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor *k2 = ggml_cast(ca, ggml_reshape_2d(ca, k3, NKV * HD, T), GGML_TYPE_F16);
    ggml_tensor *v2 = ggml_cast(ca, ggml_reshape_2d(ca, v, NKV * HD, T), GGML_TYPE_F16);
    ggml_build_forward_expand(fa, ggml_set_rows(ca, M.kv.k[(size_t)il], k2, io.slot_in));
    ggml_build_forward_expand(fa, ggml_set_rows(ca, M.kv.v[(size_t)il], v2, io.slot_in));

    ggml_tensor *dsa_mask = nullptr;
    if (!M.no_dsa) {
        ggml_tensor *mask_all = ggml_new_tensor_3d(ca, GGML_TYPE_F32, 1, S, T);
        mask_all = ggml_fill(ca, mask_all, -INFINITY);
        ggml_tensor *tk_idx = ggml_view_3d(ca, top_k, K_SEL, T, 1,
                                           top_k->nb[1], top_k->nb[2], 0);
        ggml_tensor *zeros = ggml_new_tensor_3d(ca, GGML_TYPE_F32, 1, K_SEL, T);
        zeros = ggml_fill(ca, zeros, 0.0f);
        ggml_tensor *sparse = ggml_set_rows(ca, mask_all, zeros, tk_idx);
        sparse = ggml_view_2d(ca, sparse, S, T, sparse->nb[1], 0);
        dsa_mask = ggml_add(ca, sparse, io.mask_in);
    } else {
        dsa_mask = ggml_cast(ca, io.mask_in, GGML_TYPE_F16);
    }
    dsa_mask = ggml_cast(ca, dsa_mask, GGML_TYPE_F16);

    ggml_tensor *qf = ggml_cont(ca, ggml_permute(ca, q3, 0, 2, 1, 3));
    ggml_tensor *kf = ggml_view_3d(ca, M.kv.k[(size_t)il], HD, S, NKV,
                                   M.kv.k[(size_t)il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
    ggml_tensor *vf = ggml_view_3d(ca, M.kv.v[(size_t)il], HD, S, NKV,
                                   M.kv.v[(size_t)il]->nb[1], HD * sizeof(ggml_fp16_t), 0);
    ggml_tensor *fa_attn = ggml_flash_attn_ext(ca, qf, kf, vf, dsa_mask,
                                               1.0f / sqrtf((float)HD), 0.0f, 0.0f);
    ggml_tensor *attn = ggml_reshape_2d(ca, ggml_cont(ca, fa_attn), NH * HD, T);
    attn = ggml_mul_mat(ca, L.wo, attn);
    x = ggml_add(ca, x, attn);                       // x_mid (residual after attn)

    // ---- router: same softmax-over-selected renormalization as prefill ----
    ggml_tensor *m = ggml_rms_norm(ca, x, M.rms_eps);
    m = ggml_mul(ca, m, L.ffn_norm);
    ggml_tensor *rlogits = ggml_mul_mat(ca, L.gate_inp, m);          // [E, T]
    ggml_tensor *sel = ggml_top_k(ca, rlogits, M.n_expert_used);     // [K, T] I32
    ggml_tensor *sel_logits = ggml_get_rows(ca,
            ggml_reshape_3d(ca, rlogits, 1, M.n_expert, T), sel);
    sel_logits = ggml_reshape_2d(ca, sel_logits, M.n_expert_used, T);
    ggml_tensor *w8 = ggml_soft_max(ca, sel_logits);                 // [K, T]

    // outputs as fresh trailing nodes: build_forward_expand on a tensor
    // that already sits mid-graph is a no-op, and gallocr may then reuse
    // its buffer after its last consumer, so the host readback would see
    // aliased garbage. dup'd tails are never a src of anything later and
    // therefore keep their buffers for the whole compute.
    io.w8_o = ggml_dup(ca, w8);
    io.sel_o = ggml_dup(ca, sel);
    io.m_o = ggml_dup(ca, m);
    io.x_o = ggml_dup(ca, x);
    ggml_set_name(io.w8_o, "ec_w8_o");
    ggml_set_name(io.sel_o, "ec_sel_o");
    ggml_set_name(io.m_o, "ec_m_o");
    ggml_set_name(io.x_o, "ec_x_o");
    ggml_build_forward_expand(fa, io.w8_o);
    ggml_build_forward_expand(fa, io.sel_o);
    ggml_build_forward_expand(fa, io.m_o);
    ggml_build_forward_expand(fa, io.x_o);
    return io.x_o;
}

bool KeyeVL2::ec_build_a(void * m, int il, vqa::StepGraph & sg,
                           vqa::moe_decode::AIO & io) {
    KeyeVL2 & M = *(KeyeVL2 *) m;
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

bool KeyeVL2::ec_feed_a(void * m, int il, vqa::moe_decode::AIO & io,
                          vqa::StepGraph & sg, const void * step_ctx,
                          int cur_tok, int n_past, const float * hid) {
    (void)step_ctx; (void)cur_tok; (void)hid;
    KeyeVL2 & M = *(KeyeVL2 *) m;
    EcGraphA & eg = *(EcGraphA *) io.model_io;
    const int S = M.ec_st.S;
    if (M.ec_mask.size() != (size_t)S || M.ec_mask_n != n_past) {
        M.ec_mask.assign((size_t)S, -INFINITY);
        for (int sq = 0; sq <= n_past; sq++) M.ec_mask[(size_t)sq] = 0.0f;
        M.ec_mask_n = n_past;
    }
    const int pos4[4] = { n_past, n_past, n_past, n_past };
    const int slot1[1] = { n_past };
    return sg.set_input(eg.pos_in, pos4, sizeof(int) * 4) &&
           sg.set_input(eg.slot_in, slot1, sizeof(int)) &&
           sg.set_input(eg.mask_in, M.ec_mask.data(),
                        sizeof(float) * M.ec_mask.size());
}

void KeyeVL2::ec_post_a(void * m, int il, const float * xmid, const float * mn,
                          const int * sel, const float * w8) {
    KeyeVL2 & M = *(KeyeVL2 *) m;
    // stash the layer IO for the REF hook (ec_layer_check)
    M.ec_lc_m.assign(mn, mn + M.n_embd);
    M.ec_lc_xmid.assign(xmid, xmid + M.n_embd);
    M.ec_lc_sel.assign(sel, sel + M.n_expert_used);
    M.ec_lc_w8.assign(w8, w8 + M.n_expert_used);
    if (getenv("KEYE_EC_DBG") && il < 2) {
        fprintf(stderr, "ec L%02d sel:", il);
        for (int ke = 0; ke < M.n_expert_used; ke++)
            fprintf(stderr, " %d(%.3f)", sel[ke], w8[ke]);
        fprintf(stderr, "\n");
    }
    if (getenv("KEYE_EC_DBG")) {
        double xm = 0, mm = 0;
        for (int j = 0; j < M.n_embd; j++) {
            double a = fabs((double)xmid[(size_t)j]); if (a > xm) xm = a;
            double b = fabs((double)mn[(size_t)j]); if (b > mm) mm = b;
        }
        fprintf(stderr, "ec L%02d graphA: max|xmid| %.4f max|m| %.4f\n", il, xm, mm);
    }
}

bool KeyeVL2::ec_attn_only(void * m, int il) {
    (void)m; (void)il;
    return getenv("KEYE_EC_ATTN") != nullptr;   // isolation: skip the MoE
}

void KeyeVL2::ec_trace(void * m, int il, int phase) {
    (void)m;
    if (!getenv("KEYE_EC_TRACE")) return;
    static const char * ph[5] = { "post-readback", "acquired", "ffn built",
                                  "alloc ok", "inputs set" };
    if (phase >= 0 && phase < 5)
        fprintf(stderr, "trace L%02d: %s\n", il, ph[phase]);
}

bool KeyeVL2::ec_layer_check(void * m, int il, float * hid_io) {
    // KEYE_XCHK per-layer hidden compare against the reference graph
    KeyeVL2 & M = *(KeyeVL2 *) m;
    if (M.ec_xchk_rg) {
        char nb[48];
        snprintf(nb, sizeof(nb), "t_l%d_out_o", il);
        ggml_tensor * xo = ggml_graph_get_tensor(M.ec_xchk_rg->gf, nb);
        if (xo) {
            std::vector<float> ref((size_t)M.n_embd);
            ggml_backend_tensor_get(xo, ref.data(), 0, sizeof(float) * M.n_embd);
            double dmax = 0, rmax = 0; int im = 0;
            for (int j = 0; j < M.n_embd; j++) {
                double d = fabs((double)hid_io[(size_t)j] - ref[(size_t)j]);
                if (d > dmax) { dmax = d; im = j; }
                double rr = fabs((double)ref[(size_t)j]);
                if (rr > rmax) rmax = rr;
            }
            fprintf(stderr, "xchk L%02d maxdiff %.5f (ref max %.4f, ec[%d] %.4f ref[%d] %.4f)\n",
                    il, dmax, rmax, im, hid_io[(size_t)im], im, ref[(size_t)im]);
        }
    }
    // KEYE_EC_REF: CPU reference of the same MoE (mul_mat_id over the host
    // expert tensors, exactly like the prefill graph). The byte-level
    // slotcheck that needed the in-loop slot handles retired with the
    // driver migration.
    if (M.ec_ref_mode) {
        const int D = M.n_embd, T = 1, K = M.n_expert_used;
        const TxtLayer & L = M.tl[(size_t)il];
        ggml_context * cr = ggml_init({ 16ull << 20, NULL, true });
        ggml_cgraph * fr = ggml_new_graph_custom(cr, 4096, false);
        ggml_tensor * rin = ggml_new_tensor_2d(cr, GGML_TYPE_F32, D, T);
        ggml_tensor * rsel = ggml_new_tensor_2d(cr, GGML_TYPE_I32, K, T);
        ggml_tensor * rmid = ggml_new_tensor_2d(cr, GGML_TYPE_F32, D, T);
        ggml_tensor * rcur3 = ggml_reshape_3d(cr, rin, D, 1, T);
        ggml_tensor * rgu = ggml_mul_mat_id(cr, L.gate_exps, rcur3, rsel);
        ggml_tensor * rup = ggml_mul_mat_id(cr, L.up_exps, rcur3, rsel);
        ggml_tensor * ract = ggml_mul(cr, ggml_silu(cr, rgu), rup);
        ggml_tensor * rff3 = ggml_mul_mat_id(cr, L.down_exps, ract, rsel);
        ggml_tensor * rwb = ggml_new_tensor_2d(cr, GGML_TYPE_F32, K, T);
        ggml_tensor * rprod = ggml_mul(cr, rff3,
            ggml_reshape_3d(cr, rwb, 1, K, T));
        ggml_tensor * rpk = ggml_cont(cr, ggml_permute(cr, rprod, 1, 0, 2, 3));
        ggml_tensor * rmoe = ggml_reshape_2d(cr, ggml_sum_rows(cr, rpk), D, T);
        ggml_tensor * rout = ggml_add(cr, rmid, rmoe);
        ggml_build_forward_expand(fr, rout);
        ggml_gallocr_t ra = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(M.backend_cpu));
        if (ggml_gallocr_alloc_graph(ra, fr)) {
            ggml_backend_tensor_set(rin, M.ec_lc_m.data(), 0, sizeof(float) * D);
            ggml_backend_tensor_set(rsel, M.ec_lc_sel.data(), 0, sizeof(int) * K);
            ggml_backend_tensor_set(rmid, M.ec_lc_xmid.data(), 0, sizeof(float) * D);
            ggml_backend_tensor_set(rwb, M.ec_lc_w8.data(), 0, sizeof(float) * K);
            if (ggml_backend_graph_compute(M.backend_cpu, fr) == GGML_STATUS_SUCCESS) {
                std::vector<float> ref((size_t)D);
                ggml_backend_tensor_get(rout, ref.data(), 0, sizeof(float) * D);
                double dmax = 0, rmax = 0; int im = 0;
                for (int j = 0; j < D; j++) {
                    double d = fabs((double)hid_io[(size_t)j] - ref[(size_t)j]);
                    if (d > dmax) { dmax = d; im = j; }
                    double rr = fabs((double)ref[(size_t)j]);
                    if (rr > rmax) rmax = rr;
                }
                fprintf(stderr, "ec L%02d REF maxdiff %.5f (ref max %.4f at %d, got %.4f)\n",
                        il, dmax, rmax, im, hid_io[(size_t)im]);
                if (M.ec_ref_mode == 2)
                    memcpy(hid_io, ref.data(), sizeof(float) * D);   // isolation
            }
            ggml_gallocr_free(ra);
        }
        ggml_free(cr);
    }
    return true;
}

bool KeyeVL2::ec_decode_step(int cur_tok, int pos, int n_past, std::vector<float> & logits) {
    const int S = vqa::block_for(n_past + 1, 512);
    std::vector<float> mask((size_t)S, -INFINITY);
    for (int sq = 0; sq <= n_past; sq++) mask[(size_t)sq] = 0.0f;

    // KEYE_XCHK: run the standard split decode graph for the same step and
    // compare every layer hidden + the logits against the EC path
    TextGraph rg;
    std::vector<float> ref_logits;
    const bool xchk = getenv("KEYE_XCHK") != nullptr;
    if (xchk) {
        if (!build_text_graph(rg, 1, S, 0, false)) { ggml_free(rg.ctx); return false; }
        if (!run_text(rg, { cur_tok }, { pos, pos, pos, pos }, nullptr, nullptr,
                      ref_logits, n_past, &mask)) { ggml_free(rg.ctx); return false; }
    }
    ec_xchk_rg = xchk ? &rg : nullptr;
    const char * ec_ref_env = getenv("KEYE_EC_REF");
    ec_ref_mode = ec_ref_env ? atoi(ec_ref_env) : 0;

    if (!vqa::moe_decode::decode_step(ec_hooks, ec_st, this, nullptr, cur_tok,
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
        ggml_free(rg.ctx);
        ec_xchk_rg = nullptr;
    }
    return true;
}
