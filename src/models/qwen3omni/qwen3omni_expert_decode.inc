// qwen3omni_expert_decode.inc - included at the bottom of qwen3omni_model.cpp
// (inside namespace qwen3omni; member definitions of Qwen3Omni).
//
// Expert-granularity GPU-resident dynamic decode path (Strata-style tier 2),
// enabled with QOMNI_EXPERTCACHE=1. Structure is copied from the verified
// keye integration (keye_expert_decode.inc):
//   graph A  (FIXED StepGraph, built once by ec_build_graphs): embed/hid ->
//            per-head q/k norm + fused IMROPE -> GQA attention (KV set_rows)
//            -> residual -> ffn_norm -> router -> top-8 ids + renorm softmax w
//   host     read the 8 ids back, ExpertCache::acquire -> device slots
//   graph B  (rebuilt per layer): moe::build_expert_ffn_table_res over the
//            v3 slot tables (3 mul_mat_id launches) -> residual
//   graph C  final out_norm + lm_head
// The op sequence inside graph A is copied VERBATIM from build_text_graph
// (T=1, fused-rope default; QOMNI_NOFUSE honored); the math must not change.
//
// q3-specific adaptations vs keye:
//   * IMROPE: mpos input is the 4-plane [t, h, w, slot] row already used by
//     build_text_graph (NOT keye's pos x4); the decode caller passes the same
//     step_pos vector the standard path uses.
//   * mask is F16 [S, 1] (the main graph's convention), not F32.
//   * talker trajectory: layer-0's x_o IS the embedding row the standard
//     for_talker graph dups as t_emb_out, and the post-graph-B hidden of layer
//     23 IS t_h24_out - both captured here so --speak works unchanged.
//   * prefill (multimodal injection, DeepStack, img set_rows) stays on the
//     existing single big graph (split scheduler in this mode).
//
// CPU validation mode: on a CPU backend (no CUDA in the build, or the model
// did not go to the GPU) the cache lives in host RAM and everything still
// runs - used to check the semantics/logits before any GPU placement.
//
// Failure policy: if the cache cannot be initialized (no CUDA / no VRAM),
// QOMNI_EXPERTCACHE is disabled and decode falls back to the existing path.

static inline double qec_ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t).count();
}

bool Qwen3Omni::ec_init() {
    ec_ok = false;
    if (ggml_backend_is_cpu(backend)) {
        // CPU validation mode (whole model already in the host mmap)
        split_mode = false;
    } else if (!split_mode) {
        fprintf(stderr, "qomni ec: no GPU split placement available,"
                " QOMNI_EXPERTCACHE disabled\n");
        ec_enabled = false;
        return true;
    }
    {
        // group layers by their (type, shape) signature (the GGUF could mix
        // quant types across layers); copied from the keye integration
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
        ec_group = grp;
        ec_n_groups = n_grp;
    }
        size_t budget = 0;
#ifdef GGML_USE_CUDA
    if (!ggml_backend_is_cpu(backend)) {
        size_t free_vram = 0, total_vram = 0;
        ggml_backend_cuda_get_device_memory(0, &free_vram, &total_vram);
        budget = free_vram > (size_t)2.0e9 ? free_vram - (size_t)2.0e9 : 0;
        if (budget == 0) {
            fprintf(stderr, "qomni ec: < 2 GB free VRAM -> QOMNI_EXPERTCACHE disabled\n");
            ec_enabled = false;
            return true;
        }
        fprintf(stderr, "qomni ec: free VRAM %.2f GB, slot budget %.2f GB\n",
                free_vram / 1e9, budget / 1e9);
    }
#endif
        if (budget == 0) {
            if (!ggml_backend_is_cpu(backend)) {
                fprintf(stderr, "qomni ec: CUDA unavailable -> QOMNI_EXPERTCACHE disabled\n");
                ec_enabled = false;
                return true;
            }
            // CPU validation: cap the resident slot set at ~4 GB of RAM
            budget = (size_t)4.0e9;
            fprintf(stderr, "qomni ec: CPU validation mode (host-resident slots,"
                    " budget %.1f GB)\n", budget / 1e9);
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
            spec.n_layers = n_layer;
            spec.n_experts = n_expert;
            spec.tensors_per_expert = 3;
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
            const size_t gbudget = budget * (size_t)layers / (size_t)n_layer;
            if (gbudget < (size_t)8 * slot_bytes) {
                fprintf(stderr, "qomni ec: group %d budget %.2f GB < 8 slots x %.1f MB"
                        " -> QOMNI_EXPERTCACHE disabled\n", g, gbudget / 1e9,
                        slot_bytes / 1e6);
                ec_enabled = false;
                return true;
            }
            int n_want = (int)(gbudget / slot_bytes);
            if (n_want > layers * n_expert) n_want = layers * n_expert;
            const int n_min = layers * n_expert_used * 2;
            if (n_want < n_min) n_want = std::min(layers * n_expert, n_min);
            if (!ec_caches[(size_t)g].init(backend, spec, (size_t)n_want * est_slot)) {
                fprintf(stderr, "qomni ec: ExpertCache init failed (group %d)"
                        " -> falling back\n", g);
                ec_enabled = false;
                return true;
            }
            total_slots += ec_caches[(size_t)g].stats().n_slots;
            fprintf(stderr, "qomni ec: group %d (%d layers, slot %.2f MB) ready\n",
                    g, layers, slot_bytes / 1e6);
        }
        fprintf(stderr, "qomni ec: ready, %d slots total\n", total_slots);
        if (total_slots < n_layer * n_expert_used) {
            fprintf(stderr, "qomni ec: WARNING %d slots < %d experts per decode"
                    " step, expect swapping\n", total_slots, n_layer * n_expert_used);
        }
        ec_ok = true;
        // the async staging pool defaults to min(4, hw/2) threads; the
        // pageable->pinned memcpy is the swap bottleneck on this model
        // (measured 4.1 tok/s @4 vs 10.2 tok/s @16 threads) - raise the
        // default here unless the user set EC_STAGE_THREADS
        if (!getenv("EC_STAGE_THREADS")) {
#ifdef _WIN32
            _putenv_s("EC_STAGE_THREADS", "16");
#else
            setenv("EC_STAGE_THREADS", "16", 0);
#endif
        }
        ec_async_ok = true;
        for (auto & c : ec_caches)
            if (!c.enable_async()) ec_async_ok = false;
        fprintf(stderr, "qomni ec: async prefetch %s\n",
                ec_async_ok ? "enabled" : "unavailable -> synchronous (v1)");
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
    const int S = M.ec_S;

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

bool Qwen3Omni::ec_build_graphs() {
    if (ec_S <= 0) ec_S = vqa::block_for(n_past + 1, 512);
    ec_ga.assign((size_t)n_layer, vqa::StepGraph());
    ec_ga_io.assign((size_t)n_layer, EcGraphA());
    ec_prev_sel.assign((size_t)n_layer, std::vector<int>());
    for (int il = 0; il < n_layer; il++) {
        EcGaUd ud = { this, il, &ec_ga_io[(size_t)il] };
        // private gallocr per holder (pool must never be re-reserved by the
        // per-step graph B; see the keye integration's notes)
        if (!ec_ga[(size_t)il].build(backend, ec_ga_builder, &ud)) {
            fprintf(stderr, "qomni ec: graph A build failed (layer %d)\n", il);
            return false;
        }
    }
    ec_ga_ready = true;
    fprintf(stderr, "qomni ec: %d fixed layer-A step graphs ready (S=%d)\n",
            n_layer, ec_S);
    return true;
}

bool Qwen3Omni::ec_decode_step(int cur_tok, const int * mpos4, int n_past,
                               std::vector<float> & logits,
                               std::vector<float> * emb_out,
                               std::vector<float> * h24_out) {
    const int D = n_embd;
    const int T = 1;
    const int K = n_expert_used;

    const int S_need = vqa::block_for(n_past + 1, 512);
    if (!ec_ga_ready || S_need != ec_S) {
        if (ec_ga_ready) {
            fprintf(stderr, "qomni ec: KV window crossed a 512 block (%d -> %d),"
                    " rebuilding %d layer-A graphs\n", ec_S, S_need, n_layer);
            ec_ga_ready = false;
        }
        ec_S = S_need;
        if (!ec_build_graphs()) return false;
    }
    const int S = ec_S;

    std::vector<ggml_fp16_t> mask((size_t)S, ggml_fp32_to_fp16(-INFINITY));
    for (int s = 0; s <= n_past; s++) mask[(size_t)s] = ggml_fp32_to_fp16(0.0f);

    int sel_host[16];
    float w8_host[16];

    for (auto & c : ec_caches) c.begin_step();
    std::vector<float> hid((size_t)D);
    std::vector<float> m_host((size_t)D), xmid_host((size_t)D);

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

    auto t0 = std::chrono::steady_clock::now();
    for (int il = 0; il < n_layer; il++) {
        const TxtLayer & L = tl[(size_t)il];
        vqa::StepGraph & sg = ec_ga[(size_t)il];
        EcGraphA & io = ec_ga_io[(size_t)il];

        // ---------------- graph A: fixed StepGraph, re-fed ----------------
        const auto ta0 = std::chrono::steady_clock::now();
        if (!sg.set_input(io.mpos_in, mpos4, sizeof(int) * 4) ||
            !sg.set_input(io.mask_in, mask.data(), sizeof(ggml_fp16_t) * mask.size())) {
            fprintf(stderr, "qomni ec: graph A input upload failed (layer %d)\n", il);
            return false;
        }
        if (il == 0) {
            if (!sg.set_input(io.tok_in, &cur_tok, sizeof(int))) return false;
        } else {
            if (!sg.set_input(io.hid_in, hid.data(), sizeof(float) * D)) return false;
        }
        if (!sg.run()) {
            fprintf(stderr, "qomni ec: graph A compute failed (layer %d)\n", il);
            return false;
        }
        ec_ms_a += qec_ms_since(ta0);
        const auto trb0 = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(io.m_o, m_host.data(), 0, sizeof(float) * D);
        ggml_backend_tensor_get(io.x_o, xmid_host.data(), 0, sizeof(float) * D);
        ggml_backend_tensor_get(io.sel_o, sel_host, 0, sizeof(int) * K);
        ggml_backend_tensor_get(io.w8_o, w8_host, 0, sizeof(float) * K);
        ec_ms_rb += qec_ms_since(trb0);
        if (il == 0 && emb_out)
            emb_out->assign(xmid_host.begin(), xmid_host.end());

        // prefetch: this layer's fresh selections + one-layer-ahead
        // speculation (no-op without enable_async; QOMNI_EC_PF=0 disables)
        if (ec_async_ok && getenv("QOMNI_EC_PF") == nullptr) {
            static const int ec_spec = getenv("QOMNI_EC_SPEC")
                                     ? atoi(getenv("QOMNI_EC_SPEC")) : 1;
            std::vector<std::pair<int, int>> pf;
            for (int ke = 0; ke < K; ke++) pf.push_back({ il, sel_host[ke] });
            const bool has_next = ec_spec && il + 1 < n_layer
                                && !ec_prev_sel[(size_t)il + 1].empty();
            const bool next_same_grp = has_next
                && ec_group[(size_t)il + 1] == ec_group[(size_t)il];
            if (next_same_grp)
                for (int e : ec_prev_sel[(size_t)il + 1]) pf.push_back({ il + 1, e });
            ec_caches[(size_t)ec_group[(size_t)il]].prefetch(pf);
            if (has_next && !next_same_grp) {
                std::vector<std::pair<int, int>> pf2;
                for (int e : ec_prev_sel[(size_t)il + 1]) pf2.push_back({ il + 1, e });
                ec_caches[(size_t)ec_group[(size_t)il + 1]].prefetch(pf2);
            }
        }

        // ---------------- graph B: the 8 slot experts ----------------
        const auto tsw0 = std::chrono::steady_clock::now();
        std::vector<ggml_tensor *> gate(K), up(K), down(K);
        vqa::ExpertCache & ec_c = ec_caches[(size_t)ec_group[(size_t)il]];
        const bool ec_tbl_avail = ec_c.table(0) && ec_c.table(1) && ec_c.table(2);
        int ids_gu_host[16] = {}, ids_dn_host[16] = {};
        bool ec_tbl = ec_tbl_avail;
        for (int ke = 0; ke < K; ke++) {
            std::vector<ggml_tensor *> sl = ec_c.acquire(il, sel_host[ke]);
            if (sl.size() != 3) {
                fprintf(stderr, "qomni ec: acquire failed (layer %d expert %d)\n",
                        il, sel_host[ke]);
                return false;
            }
            gate[(size_t)ke] = sl[1];
            up[(size_t)ke]   = sl[0];
            down[(size_t)ke] = sl[2];
            ids_gu_host[ke] = ec_c.slot_index(sl[1]);   // gate/up ids
            ids_dn_host[ke] = ec_c.slot_index(sl[2]);   // down ids
            if (ids_gu_host[ke] < 0 || ids_dn_host[ke] < 0) ec_tbl = false;
        }
        ec_ms_swap += qec_ms_since(tsw0);
        ec_prev_sel[(size_t)il].assign(sel_host, sel_host + K);

        ggml_context * cb = ggml_init({ 16ull << 20, NULL, true });
        if (!cb) return false;
        ggml_cgraph * fb = ggml_new_graph_custom(cb, 8192, false);
        ggml_tensor * moe_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, D, T);
        ggml_tensor * xmid_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, D, T);
        ggml_tensor * w_in = ggml_new_tensor_2d(cb, GGML_TYPE_F32, K, T);

        ggml_tensor * ids_gu_in = nullptr, * ids_dn_in = nullptr;
        ggml_tensor * hout;
        if (ec_tbl) {
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
            fprintf(stderr, "qomni ec: build_expert_ffn failed (layer %d)\n", il);
            ggml_free(cb);
            return false;
        }
        const auto tb0 = std::chrono::steady_clock::now();
        if (!ggml_gallocr_alloc_graph(allocr, fb)) {
            fprintf(stderr, "qomni ec: graph B alloc failed (layer %d)\n", il);
            ggml_free(cb);
            return false;
        }
        ggml_backend_tensor_set(moe_in, m_host.data(), 0, sizeof(float) * D);
        ggml_backend_tensor_set(xmid_in, xmid_host.data(), 0, sizeof(float) * D);
        ggml_backend_tensor_set(w_in, w8_host, 0, sizeof(float) * K);
        if (ec_tbl) {   // slot ids are valid only until the next acquire
            ggml_backend_tensor_set(ids_gu_in, ids_gu_host, 0, sizeof(int) * K);
            ggml_backend_tensor_set(ids_dn_in, ids_dn_host, 0, sizeof(int) * K);
        }
        if (ggml_backend_graph_compute(backend, fb) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "qomni ec: graph B compute failed (layer %d)\n", il);
            ggml_free(cb);
            return false;
        }
        ec_ms_b += qec_ms_since(tb0);
        const auto trb1 = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(hout, hid.data(), 0, sizeof(float) * D);
        ec_ms_rb += qec_ms_since(trb1);
        ggml_free(cb);
        if (il == 23 && h24_out)
            h24_out->assign(hid.begin(), hid.end());
        (void)L;
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
    }
    ec_ms_fin += qec_ms_since(t0);
    ec_steps++;
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
