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

bool KeyeVL2::ec_init() {
    ec_ok = false;
#ifdef GGML_USE_CUDA
    if (!split_mode || ggml_backend_is_cpu(backend)) {
        fprintf(stderr, "keye ec: no GPU split placement available, KEYE_EXPERTCACHE disabled\n");
        ec_enabled = false;
        return true;
    }
    {
        // the GGUF mixes quant types across layers (e.g. Q4_K_M-style per-tensor
        // quants); the cache spec assumes one layout per tensor index, so group
        // layers by their (type, shape) signature and use one cache per group
        std::vector<int> grp((size_t)n_layer, 0);
        int n_grp = 0;
        for (int l = 0; l < n_layer; l++) {
            ggml_tensor * ex[3] = { tl[(size_t)l].up_exps, tl[(size_t)l].gate_exps,
                                    tl[(size_t)l].down_exps };
            int g = -1;
            for (int l2 = 0; l2 < l; l2++) {
                ggml_tensor * ey[3] = { tl[(size_t)l2].up_exps, tl[(size_t)l2].gate_exps,
                                        tl[(size_t)l2].down_exps };
                bool same = true;
                for (int t = 0; t < 3; t++)
                    if (ex[t]->type != ey[t]->type || ex[t]->ne[0] != ey[t]->ne[0]
                        || ex[t]->ne[1] != ey[t]->ne[1]) { same = false; break; }
                if (same) { g = grp[(size_t)l2]; break; }
            }
            if (g < 0) g = n_grp++;
            grp[(size_t)l] = g;
        }
        for (int g = 0; g < n_grp; g++) {
            int cnt = 0, first = -1;
            for (int l = 0; l < n_layer; l++)
                if (grp[(size_t)l] == g) { ++cnt; if (first < 0) first = l; }
            fprintf(stderr, "keye ec: layout group %d: %d layers (first L%d:"
                    " up %s gate %s down %s)\n", g, cnt, first,
                    ggml_type_name(tl[(size_t)first].up_exps->type),
                    ggml_type_name(tl[(size_t)first].gate_exps->type),
                    ggml_type_name(tl[(size_t)first].down_exps->type));
        }
        ec_group = grp;
        ec_n_groups = n_grp;
    }
        size_t free_vram = 0, total_vram = 0;
        ggml_backend_cuda_get_device_memory(0, &free_vram, &total_vram);
        const size_t budget = free_vram > (size_t)2.0e9 ? free_vram - (size_t)2.0e9 : 0;
        if (budget == 0) {
            fprintf(stderr, "keye ec: < 2 GB free VRAM -> KEYE_EXPERTCACHE disabled\n");
            ec_enabled = false;
            return true;
        }
        ec_caches.assign((size_t)ec_n_groups, vqa::ExpertCache());
        int total_slots = 0;
        for (int g = 0; g < ec_n_groups; g++) {
            int first = -1, layers = 0;
            for (int l = 0; l < n_layer; l++)
                if (ec_group[(size_t)l] == g) { ++layers; if (first < 0) first = l; }
            ggml_tensor * ex0[3] = { tl[(size_t)first].up_exps, tl[(size_t)first].gate_exps,
                                     tl[(size_t)first].down_exps };
            vqa::ExpertCacheSpec spec;
            spec.n_layers = n_layer;         // global layer indexing (layer_base
            spec.n_experts = n_expert;        // entries filled only for this
            spec.tensors_per_expert = 3;      // group's layers; others unused)
            spec.layer_base.assign((size_t)n_layer,
                                   std::vector<const uint8_t *>(3, nullptr));
            spec.desc.resize(3);
            size_t slot_bytes = 0, est_slot = 0;
            for (int t = 0; t < 3; t++) {
                spec.desc[(size_t)t].ne[0] = ex0[t]->ne[0];
                spec.desc[(size_t)t].ne[1] = ex0[t]->ne[1];
                spec.desc[(size_t)t].ne[2] = 1;
                spec.desc[(size_t)t].ne[3] = 1;
                spec.desc[(size_t)t].type = ex0[t]->type;
                spec.desc[(size_t)t].host_stride = ggml_nbytes(ex0[t]) / ex0[t]->ne[2];
                slot_bytes += spec.desc[(size_t)t].host_stride;
                // the cache's internal slot estimate (element count x
                // ggml_type_size) overcounts block-quants; see below
                est_slot += (size_t)ex0[t]->ne[0] * ex0[t]->ne[1]
                          * ggml_type_size(ex0[t]->type);
            }
            for (int l = 0; l < n_layer; l++) {
                if (ec_group[(size_t)l] != g) continue;
                ggml_tensor * ex[3] = { tl[(size_t)l].up_exps, tl[(size_t)l].gate_exps,
                                        tl[(size_t)l].down_exps };
                for (int t = 0; t < 3; t++)
                    spec.layer_base[(size_t)l][(size_t)t] = (const uint8_t *) ex[t]->data;
            }
            // budget share proportional to the group's layer count; rescale so
            // the slot COUNT matches what the real bytes allow (the cache's
            // device allocation uses real ggml_nbytes, so true VRAM usage stays
            // within the share)
            const size_t gbudget = budget * (size_t)layers / (size_t)n_layer;
            if (gbudget < (size_t)8 * slot_bytes) {
                fprintf(stderr, "keye ec: group %d budget %.2f GB < 8 slots x %.1f MB"
                        " -> KEYE_EXPERTCACHE disabled\n", g, gbudget / 1e9,
                        slot_bytes / 1e6);
                ec_enabled = false;
                return true;
            }
            int n_want = (int)(gbudget / slot_bytes);
            if (n_want > layers * n_expert) n_want = layers * n_expert;
            // a decode step pins up to 8 acquires per layer of this group
            const int n_min = layers * n_expert_used * 2;
            if (n_want < n_min) n_want = std::min(layers * n_expert, n_min);
            if (!ec_caches[(size_t)g].init(backend, spec, (size_t)n_want * est_slot)) {
                fprintf(stderr, "keye ec: ExpertCache init failed (group %d)"
                        " -> falling back\n", g);
                ec_enabled = false;
                return true;
            }
            total_slots += ec_caches[(size_t)g].stats().n_slots;
            fprintf(stderr, "keye ec: group %d (%d layers, slot %.2f MB) ready\n",
                    g, layers, slot_bytes / 1e6);
        }
        fprintf(stderr, "keye ec: ready, %d slots total (free VRAM %.2f GB,"
                " budget %.2f GB)\n", total_slots, free_vram / 1e9, budget / 1e9);
        if (total_slots < n_layer * n_expert_used) {
            fprintf(stderr, "keye ec: WARNING %d slots < %d experts per decode"
                    " step, expect swapping\n", total_slots, n_layer * n_expert_used);
        }
        ec_ok = true;
        // v2: start the prefetch workers; a failure anywhere degrades every
        // cache to v1 synchronous semantics (prefetch stays a safe no-op)
        ec_async_ok = true;
        for (auto & c : ec_caches)
            if (!c.enable_async()) ec_async_ok = false;
        fprintf(stderr, "keye ec: async prefetch %s\n",
                ec_async_ok ? "enabled" : "unavailable -> synchronous (v1)");
#else
    fprintf(stderr, "keye ec: not a CUDA build, KEYE_EXPERTCACHE disabled\n");
    ec_enabled = false;
#endif
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
    const int S = M.ec_S;
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

bool KeyeVL2::ec_build_graphs() {
    if (ec_S <= 0) ec_S = vqa::block_for(n_past + 1, 512);
    ec_ga.assign((size_t)n_layer, vqa::StepGraph());
    ec_ga_io.assign((size_t)n_layer, EcGraphA());
    ec_prev_sel.assign((size_t)n_layer, std::vector<int>());
    for (int il = 0; il < n_layer; il++) {
        EcGaUd ud = { this, il, &ec_ga_io[(size_t)il] };
        // private gallocr per holder: the pool must never be re-reserved by a
        // differently-shaped graph (see file header), and 48 same-shaped pools
        // cost only ~a few MB of VRAM each
        if (!ec_ga[(size_t)il].build(backend, ec_ga_builder, &ud)) {
            fprintf(stderr, "keye ec: graph A build failed (layer %d)\n", il);
            return false;
        }
        // constant for the whole session (private pool is never reset)
        ggml_backend_tensor_set(ec_ga_io[(size_t)il].had, hadamard64.data(), 0,
                                sizeof(float) * hadamard64.size());
        if (getenv("KEYE_EC_DBG")) {   // leaf wiring sanity
            EcGraphA & io2 = ec_ga_io[(size_t)il];
            ggml_tensor * chk[10] = { io2.tok_in, io2.hid_in, io2.pos_in, io2.slot_in,
                                      io2.mask_in, io2.had, io2.w8_o, io2.sel_o,
                                      io2.m_o, io2.x_o };
            for (int i = 0; i < 10; i++)
                if (!chk[i] || (chk[i]->view_src ? chk[i]->view_src->buffer
                                                 : chk[i]->buffer) == nullptr)
                    fprintf(stderr, "keye ec: L%d leaf %d has NO buffer\n", il, i);
        }
    }
    ec_ga_ready = true;
    fprintf(stderr, "keye ec: %d fixed layer-A step graphs ready (S=%d)\n",
            n_layer, ec_S);
    return true;
}

bool KeyeVL2::ec_decode_step(int cur_tok, int pos, int n_past, std::vector<float> & logits) {
    const int D = n_embd;
    const int T = 1;
    const int K = n_expert_used;

    const int pos4[4] = { pos, pos, pos, pos };
    const int slot1[1] = { n_past };
    const bool ec_dbg_attn = getenv("KEYE_EC_ATTN") != nullptr;
    const char * ec_ref_env = getenv("KEYE_EC_REF");
    const int ec_ref_mode = ec_ref_env ? atoi(ec_ref_env) : 0;

    // KV window rounded to a 512 block so the graph shape only changes every
    // 512 positions (mask hides the padding): the per-layer graphs are rebuilt
    // just once per block instead of once per step, while attention/DSA work
    // stays proportional to the actual context, not the full n_ctx
    const int S_need = vqa::block_for(n_past + 1, 512);
    if (!ec_ga_ready || S_need != ec_S) {
        if (ec_ga_ready) {
            fprintf(stderr, "keye ec: KV window crossed a 512 block (%d -> %d),"
                    " rebuilding %d layer-A graphs\n", ec_S, S_need, n_layer);
            ec_ga_ready = false;
        }
        ec_S = S_need;
        if (!ec_build_graphs()) return false;
    }
    const int S = ec_S;

    // decode: every cached position is causally visible; the padding slots up
    // to the fixed S are hidden with -inf (same convention as the split
    // scheduler's block-aligned decode graphs)
    std::vector<float> mask((size_t)S, -INFINITY);
    for (int s = 0; s <= n_past; s++) mask[(size_t)s] = 0.0f;

    int sel_host[16];
    float w8_host[16];

    for (auto & c : ec_caches) c.begin_step();
    std::vector<float> hid((size_t)D);
    std::vector<float> m_host((size_t)D), xmid_host((size_t)D);
    // NOTE: no whole-cache speculative prefetch at begin_step - firing all 48
    // layers' prev selections at once (384 experts ~ >1 GB/step) thrashes
    // every slot through the PCIe pipe (measured 227 MB swapped per step at
    // 9.3 tok/s). Speculation is one layer ahead instead, below.

    // KEYE_XCHK: run the standard split decode graph for the same step and
    // compare every layer's hidden state + the logits against the EC path
    TextGraph rg;
    std::vector<float> ref_logits;
    const bool xchk = getenv("KEYE_XCHK") != nullptr;
    if (xchk) {
        if (!build_text_graph(rg, 1, S, 0, false)) { ggml_free(rg.ctx); return false; }
        if (!run_text(rg, { cur_tok }, { pos, pos, pos, pos }, nullptr, nullptr,
                      ref_logits, n_past, &mask)) { ggml_free(rg.ctx); return false; }
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int il = 0; il < n_layer; il++) {
        const TxtLayer & L = tl[(size_t)il];
        vqa::StepGraph & sg = ec_ga[(size_t)il];
        EcGraphA & io = ec_ga_io[(size_t)il];

        // ---------------- graph A: fixed StepGraph, re-fed ----------------
        const auto ta0 = std::chrono::steady_clock::now();
        if (!sg.set_input(io.pos_in, pos4, sizeof(int) * 4) ||
            !sg.set_input(io.slot_in, slot1, sizeof(int)) ||
            !sg.set_input(io.mask_in, mask.data(), sizeof(float) * mask.size())) {
            fprintf(stderr, "keye ec: graph A input upload failed (layer %d)\n", il);
            return false;
        }
        if (il == 0) {
            if (!sg.set_input(io.tok_in, &cur_tok, sizeof(int))) return false;
        } else {
            if (!sg.set_input(io.hid_in, hid.data(), sizeof(float) * D)) return false;
        }
        if (!sg.run()) {
            fprintf(stderr, "keye ec: graph A compute failed (layer %d)\n", il);
            return false;
        }
        ec_ms_a += ec_ms_since(ta0);
        const auto trb0 = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(io.m_o, m_host.data(), 0, sizeof(float) * D);
        ggml_backend_tensor_get(io.x_o, xmid_host.data(), 0, sizeof(float) * D);
        ggml_backend_tensor_get(io.sel_o, sel_host, 0, sizeof(int) * K);
        ggml_backend_tensor_get(io.w8_o, w8_host, 0, sizeof(float) * K);
        ec_ms_rb += ec_ms_since(trb0);
        if (getenv("KEYE_EC_DBG") && il < 2) {
            fprintf(stderr, "ec L%02d sel:", il);
            for (int ke = 0; ke < K; ke++) fprintf(stderr, " %d(%.3f)", sel_host[ke], w8_host[ke]);
            fprintf(stderr, "\n");
        }
        if (getenv("KEYE_EC_DBG")) {
            double xm = 0, mm = 0;
            for (int j = 0; j < D; j++) {
                double a = fabs((double)xmid_host[(size_t)j]); if (a > xm) xm = a;
                double b = fabs((double)m_host[(size_t)j]); if (b > mm) mm = b;
            }
            fprintf(stderr, "ec L%02d graphA: max|xmid| %.4f max|m| %.4f\n", il, xm, mm);
        }

        // prefetch: this layer's fresh selections (the worker pipelines the
        // copies while acquire/graph-B-build run) plus ONE-LAYER-AHEAD
        // speculation (layer il+1's previous-step selections - hot experts
        // repeat across steps, and the copies overlap layer il's graph B).
        // No-op without enable_async.
        // KEYE_EC_PF=0 disables every prefetch call (pure v1 sync acquires)
        if (ec_async_ok && getenv("KEYE_EC_PF") == nullptr) {
            // KEYE_EC_SPEC=0 disables the one-layer-ahead speculation (keep
            // only the current layer's prefetch)
            static const int ec_spec = getenv("KEYE_EC_SPEC")
                                     ? atoi(getenv("KEYE_EC_SPEC")) : 1;
            std::vector<std::pair<int, int>> pf;
            for (int ke = 0; ke < K; ke++) pf.push_back({ il, sel_host[ke] });
            const bool has_next = ec_spec && il + 1 < n_layer
                                && !ec_prev_sel[(size_t)il + 1].empty();
            const bool next_same_grp = has_next
                && ec_group[(size_t)il + 1] == ec_group[(size_t)il];
            if (next_same_grp)
                for (int e : ec_prev_sel[(size_t)il + 1]) pf.push_back({ il + 1, e });
            ec_caches[(size_t)ec_group[(size_t)il]].prefetch(pf);
            // next-layer pairs may belong to the other layout group's cache
            if (has_next && !next_same_grp) {
                std::vector<std::pair<int, int>> pf2;
                for (int e : ec_prev_sel[(size_t)il + 1]) pf2.push_back({ il + 1, e });
                ec_caches[(size_t)ec_group[(size_t)il + 1]].prefetch(pf2);
            }
        }

        // ---------------- graph B: the 8 slot experts ----------------
        if (ec_dbg_attn) {   // isolation: skip the MoE, pass xmid through
            hid = xmid_host;
            if (getenv("KEYE_EC_DBG")) {
                double s = 0, mx = 0;
                for (float v : hid) { s += (double)v * v; double a = v < 0 ? -(double)v : v; if (a > mx) mx = a; }
                fprintf(stderr, "ec L%02d |h| %.3f max %.3f (attn-only)\n", il, sqrt(s), mx);
            }
            continue;
        }
        const auto tsw0 = std::chrono::steady_clock::now();
        const bool ec_trace = getenv("KEYE_EC_TRACE") != nullptr;
        if (ec_trace) fprintf(stderr, "trace L%02d: post-readback\n", il);
        std::vector<ggml_tensor *> gate(K), up(K), down(K);
        ggml_tensor * sl_dbg[16][3] = {};
        // v3: if the cache runs the table layout, the 8 experts go through 3
        // ggml_mul_mat_id launches over the slot tables (24 GEMV -> 3) and the
        // gate/up pair + GLU hits the CUDA fused pass
        vqa::ExpertCache & ec_c = ec_caches[(size_t)ec_group[(size_t)il]];
        const bool ec_tbl_avail = ec_c.table(0) && ec_c.table(1) && ec_c.table(2);
        int ids_gu_host[16] = {}, ids_dn_host[16] = {};
        bool ec_tbl = ec_tbl_avail;
        for (int ke = 0; ke < K; ke++) {
            std::vector<ggml_tensor *> sl = ec_c.acquire(il, sel_host[ke]);
            if (sl.size() != 3) {
                fprintf(stderr, "keye ec: acquire failed (layer %d expert %d)\n", il, sel_host[ke]);
                return false;
            }
            if (ec_ref_mode) { sl_dbg[ke][0] = sl[0]; sl_dbg[ke][1] = sl[1]; sl_dbg[ke][2] = sl[2]; }
            gate[(size_t)ke] = sl[1];
            up[(size_t)ke]   = sl[0];
            down[(size_t)ke] = sl[2];
            ids_gu_host[ke] = ec_c.slot_index(sl[1]);   // gate/up ids
            ids_dn_host[ke] = ec_c.slot_index(sl[2]);   // down ids
            if (ids_gu_host[ke] < 0 || ids_dn_host[ke] < 0) ec_tbl = false;
        }
        ec_ms_swap += ec_ms_since(tsw0);
        ec_prev_sel[(size_t)il].assign(sel_host, sel_host + K);
        if (ec_trace) fprintf(stderr, "trace L%02d: acquired\n", il);

        ggml_context * cb = ggml_init({ 16ull << 20, NULL, true });
        if (!cb) return false;
        ggml_cgraph * fb = ggml_new_graph_custom(cb, 8192, false);
        ggml_tensor * moe_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, D, T);
        ggml_tensor * xmid_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, D, T);
        ggml_tensor * w_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, K, T);

        ggml_tensor * ids_gu_in = nullptr, * ids_dn_in = nullptr;
        ggml_tensor * hout;
        if (ec_tbl) {
            // v3 table path: 3 mul_mat_id kernels; gate/up share ids_gu_in so
            // the CUDA mul_mat_id+mul_mat_id+GLU fusion matcher accepts them
            ids_gu_in = ggml_new_tensor_2d(cb, GGML_TYPE_I32, K, T);
            ids_dn_in = ggml_new_tensor_2d(cb, GGML_TYPE_I32, K, T);
            hout = vqa::moe::build_expert_ffn_table_res(cb, fb, moe_in, xmid_in,
                    ec_c.table(1), ec_c.table(0), ec_c.table(2),
                    ids_gu_in, ids_dn_in, w_in, GGML_UNARY_OP_SILU);
        } else {
            hout = vqa::moe::build_expert_ffn_res(cb, fb, moe_in, xmid_in,
                                                  gate, up, down, w_in,
                                                  GGML_UNARY_OP_SILU);
        }
        if (!hout) {
            fprintf(stderr, "keye ec: build_expert_ffn failed (layer %d)\n", il);
            ggml_free(cb);
            return false;
        }
        if (ec_trace) fprintf(stderr, "trace L%02d: ffn built\n", il);
        if (ec_ref_mode && (il == 6 || il == 8)) {   // slot integrity BEFORE compute
            ggml_tensor * ex[3] = { tl[(size_t)il].up_exps, tl[(size_t)il].gate_exps,
                                    tl[(size_t)il].down_exps };
            for (int ke = 0; ke < K; ke++) {
                const int eid = sel_host[ke];
                for (int t = 0; t < 3; t++) {
                    const size_t stride = ggml_nbytes(ex[t]) / ex[t]->ne[2];
                    const uint8_t * host = (const uint8_t *) ex[t]->data + (size_t)eid * stride;
                    std::vector<uint8_t> dev(std::min<size_t>(stride, 4096));
                    ggml_backend_tensor_get(sl_dbg[ke][t], dev.data(), 0, dev.size());
                    size_t first = (size_t)-1;
                    for (size_t b = 0; b < dev.size(); b++)
                        if (dev[b] != host[b]) { first = b; break; }
                    fprintf(stderr, "  precompute slotcheck L%d ke%d t%d eid=%d: %s"
                            " (dev0 %02x host0 %02x)\n", il, ke, t, eid,
                            first == (size_t)-1 ? "OK" : "MISMATCH", dev[0], host[0]);
                }
            }
        }
        const auto tb0 = std::chrono::steady_clock::now();
        if (!ggml_gallocr_alloc_graph(allocr, fb)) {
            fprintf(stderr, "keye ec: graph B alloc failed (layer %d)\n", il);
            ggml_free(cb);
            return false;
        }
        if (ec_trace) fprintf(stderr, "trace L%02d: alloc ok buf moe=%p xmid=%p w=%p\n",
                              il, (void *) moe_in->buffer, (void *) xmid_in->buffer,
                              (void *) w_in->buffer);
        ggml_backend_tensor_set(moe_in, m_host.data(), 0, sizeof(float) * D);
        ggml_backend_tensor_set(xmid_in, xmid_host.data(), 0, sizeof(float) * D);
        ggml_backend_tensor_set(w_in, w8_host, 0, sizeof(float) * K);
        if (ec_tbl) {   // slot ids are valid only until the next acquire
            ggml_backend_tensor_set(ids_gu_in, ids_gu_host, 0, sizeof(int) * K);
            ggml_backend_tensor_set(ids_dn_in, ids_dn_host, 0, sizeof(int) * K);
        }
        if (ec_trace) fprintf(stderr, "trace L%02d: inputs set\n", il);
        if (ggml_backend_graph_compute(backend, fb) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "keye ec: graph B compute failed (layer %d)\n", il);
            ggml_free(cb);
            return false;
        }
        ec_ms_b += ec_ms_since(tb0);
        const auto trb1 = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(hout, hid.data(), 0, sizeof(float) * D);
        ec_ms_rb += ec_ms_since(trb1);
        ggml_free(cb);
        if (getenv("KEYE_EC_DBG")) {
            double s = 0, mx = 0;
            for (float v : hid) { s += (double)v * v; double a = v < 0 ? -(double)v : v; if (a > mx) mx = a; }
            fprintf(stderr, "ec L%02d |h| %.3f max %.3f\n", il, sqrt(s), mx);
        }
        if (xchk) {
            char nb[48];
            snprintf(nb, sizeof(nb), "t_l%d_out_o", il);
            ggml_tensor * xo = ggml_graph_get_tensor(rg.gf, nb);
            if (xo) {
                std::vector<float> ref((size_t)D);
                ggml_backend_tensor_get(xo, ref.data(), 0, sizeof(float) * D);
                double dmax = 0, rmax = 0; int im = 0;
                for (int j = 0; j < D; j++) {
                    double d = fabs((double)hid[(size_t)j] - ref[(size_t)j]);
                    if (d > dmax) { dmax = d; im = j; }
                    double rr = fabs((double)ref[(size_t)j]);
                    if (rr > rmax) rmax = rr;
                }
                fprintf(stderr, "xchk L%02d maxdiff %.5f (ref max %.4f, ec[%d] %.4f ref[%d] %.4f)\n",
                        il, dmax, rmax, im, hid[(size_t)im], im, ref[(size_t)im]);
            }
        }
        if (ec_ref_mode) {
            // CPU reference of the same MoE (mul_mat_id over the host expert
            // tensors, exactly like the prefill graph) - parity cross-check
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
            ggml_gallocr_t ra = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_cpu));
            if (ggml_gallocr_alloc_graph(ra, fr)) {
                ggml_backend_tensor_set(rin, m_host.data(), 0, sizeof(float) * D);
                ggml_backend_tensor_set(rsel, sel_host, 0, sizeof(int) * K);
                ggml_backend_tensor_set(rmid, xmid_host.data(), 0, sizeof(float) * D);
                ggml_backend_tensor_set(rwb, w8_host, 0, sizeof(float) * K);
                if (ggml_backend_graph_compute(backend_cpu, fr) == GGML_STATUS_SUCCESS) {
                    std::vector<float> ref((size_t)D);
                    ggml_backend_tensor_get(rout, ref.data(), 0, sizeof(float) * D);
                    double dmax = 0, rmax = 0; int im = 0;
                    for (int j = 0; j < D; j++) {
                        double d = fabs((double)hid[(size_t)j] - ref[(size_t)j]);
                        if (d > dmax) { dmax = d; im = j; }
                        double rr = fabs((double)ref[(size_t)j]);
                        if (rr > rmax) rmax = rr;
                    }
                    fprintf(stderr, "ec L%02d REF maxdiff %.5f (ref max %.4f at %d,"
                            " got %.4f)\n", il, dmax, rmax, im, hid[(size_t)im]);
                    if (dmax > 1.0) {   // slot-bytes verification for the bad layer
                        ggml_tensor * ex[3] = { tl[(size_t)il].up_exps, tl[(size_t)il].gate_exps,
                                                tl[(size_t)il].down_exps };
                        for (int ke = 0; ke < K; ke++) {
                            const int eid = sel_host[ke];
                            for (int t = 0; t < 3; t++) {
                                const size_t stride = ggml_nbytes(ex[t]) / ex[t]->ne[2];
                                const uint8_t * host = (const uint8_t *) ex[t]->data
                                                     + (size_t)eid * stride;
                                std::vector<uint8_t> dev(std::min<size_t>(stride, 65536));
                                ggml_backend_tensor_get(sl_dbg[ke][t], dev.data(), 0, dev.size());
                                size_t first = (size_t)-1;
                                for (size_t b = 0; b < dev.size(); b++)
                                    if (dev[b] != host[b]) { first = b; break; }
                                fprintf(stderr, "  slotcheck L%d e%d t%d eid=%d:"
                                        " %s (first diff @%zu)\n", il, ke, t, eid,
                                        first == (size_t)-1 ? "OK" : "MISMATCH", first);
                            }
                        }
                    }
                    if (ec_ref_mode == 2)
                        hid = ref;   // isolation: graph A only, exact CPU MoE
                }
                ggml_gallocr_free(ra);
            }
            ggml_free(cr);
        }
    }

    // ---------------- final norm + output head ----------------
    {
        ggml_context * cc = ggml_init({ 8ull << 20, NULL, true });
        if (!cc) return false;
        ggml_cgraph * fc = ggml_new_graph_custom(cc, 1024, false);
        ggml_tensor * h_in = ggml_new_tensor_2d(cc, GGML_TYPE_F32, D, T);
        ggml_tensor * fn = ggml_rms_norm(cc, h_in, rms_eps);
        fn = ggml_mul(cc, fn, out_norm);
        ggml_tensor * lg = ggml_mul_mat(cc, out_w, fn);
        ggml_build_forward_expand(fc, lg);
        if (!ggml_gallocr_alloc_graph(allocr, fc)) { ggml_free(cc); return false; }
        ggml_backend_tensor_set(h_in, hid.data(), 0, sizeof(float) * D);
        if (ggml_backend_graph_compute(backend, fc) != GGML_STATUS_SUCCESS) {
            ggml_free(cc);
            return false;
        }
        logits.resize((size_t)n_vocab);
        ggml_backend_tensor_get(lg, logits.data(), 0, sizeof(float) * logits.size());
        ggml_free(cc);
        if (getenv("KEYE_EC_DBG")) {
            int top[5] = { 0, 0, 0, 0, 0 };
            for (int v = 1; v < n_vocab; v++)
                for (int j = 0; j < 5; j++)
                    if (logits[(size_t)v] > logits[(size_t)top[j]]) {
                        for (int q = 4; q > j; q--) top[q] = top[q - 1];
                        top[j] = v;
                        break;
                    }
            fprintf(stderr, "ec logits top5:");
            for (int j = 0; j < 5; j++) fprintf(stderr, " %d(%.3f)", top[j], logits[(size_t)top[j]]);
            fprintf(stderr, "\n");
        }
    }
    ec_ms_fin += ec_ms_since(t0);
    ec_steps++;
    if (xchk) {
        double dmax = 0; int im = 0;
        for (int v = 0; v < n_vocab; v++) {
            double d = fabs((double)logits[(size_t)v] - ref_logits[(size_t)v]);
            if (d > dmax) { dmax = d; im = v; }
        }
        fprintf(stderr, "xchk logits maxdiff %.5f at id %d (ec %.4f ref %.4f)\n",
                dmax, im, logits[(size_t)im], ref_logits[(size_t)im]);
        ggml_free(rg.ctx);
    }
    return true;
}
