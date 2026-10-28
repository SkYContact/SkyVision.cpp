// nemotronomni_expert_decode.inc - included at the bottom of
// nemotronomni_model.cpp (inside namespace nemo; member definitions of
// NemotronOmni).
//
// Expert-granular dynamic decode path (Strata-style tier 2), enabled with
// NEMOTRONOMNI_EXPERTCACHE=1. THIRD consumer of the shared
// core/engine/moe_decode_driver.hpp orchestration (after keye_vl and
// qwen3omni). What differs from the two uniform-MoE consumers, and which
// default-neutral driver extension carries it:
//
//   * the 52 layers are hybrid (23x Mamba2 'M' / 6x attention '*' / 23x MoE
//     'E'): Hooks::moe_of maps driver layers to the MoE slot index; the M and
//     * layers run ENTIRELY inside their fixed graph A (Mamba2 conv/ssm state
//     writes included, via the same tie_state +0 chain as build_text_graph)
//     and the driver skips their (empty) MoE half via moe_of == -1 +
//     attn_only-style pass-through of the residual row.
//   * the MoE is NON-gated (relu^2 experts, no gate branch) plus a dense
//     shared expert added on the same normed row: Hooks::build_b replaces the
//     driver's fixed gated-SILU graph B with the moe_branch tail replicated
//     VERBATIM over the ExpertCache slot tables (2 mul_mat_id launches:
//     up-exps -> relu^2 -> down-exps) + the dense shared expert, keeping the
//     standard path's exact op order and reduction order
//     x + (routed_sum + shexp) for bit-parity.
//   * tensors_per_expert = 2 (up/down; no gate tensors exist in the GGUF).
//   * CUDA form = EC split (NEMOTRONOMNI_GPU=1): dense trunk + towers + KV +
//     Mamba2 states on the GPU, the 128-expert tables left in the host mmap
//     (split_bind_to_backends), mirrored into bounded VRAM slots per decode
//     step. Prefill rides the two-backend SplitSched (fresh graphs only -
//     decode_graph.hpp SPLIT-PLACEMENT CONTRACT); EC decode never touches the
//     scheduler (single-backend graphs only).
//
// Mamba2 x EC interaction: the persistent states live in the model's sbuf and
// are written by the M-layer graph A exactly like the standard decode graph;
// cross-layer ordering holds through the +0 state_dep ties inside each graph,
// cross-step ordering through the backend's synchronous graph_compute. The
// graphs rebuild wholesale when the KV window crosses a block (the driver's
// ec_block stays pinned to the model decode bucket: 128 CPU / 1024 GPU), so
// the Mamba graphs are rebuilt too - harmless: states are external tensors,
// never graph-owned.
//
// CPU validation mode: on a CPU backend the whole model is host-resident and
// the cache lives in host RAM (pure semantics check; NEMO_XCHK must print
// maxdiff 0.00000 and the greedy streams must match the EC-off run byte for
// byte). Failure policy: if the cache cannot be initialized (no CUDA / no
// VRAM) EC is disabled and decode falls back to the existing path.
//
// Env knobs: NEMOTRONOMNI_EXPERTCACHE (enable), NEMOTRONOMNI_EC_OFF (route
// decode back to the standard path at runtime), NEMO_EC_BUDGET (slot pool
// bytes; GPU default = free VRAM - NEMO_EC_RESERVE, CPU default 4 GB),
// NEMO_EC_RESERVE (GPU VRAM headroom kept out of the slot budget, 2 GB
// default), NEMO_XCHK (per-step logits compare vs the standard decode path),
// NEMO_EC_DBG (per-layer selected-expert dump), NEMO_EC_PF / NEMO_EC_SPEC
// (driver prefetch / one-layer-ahead knobs).

static NemotronOmni * g_ec_self = nullptr;   // trampoline target for InitSpec::tensor

bool NemotronOmni::ec_init() {
    ec_ok = false;
    bool cpu_mode = false;
    if (ggml_backend_is_cpu(backend)) {
        // CPU validation mode (whole model already in the host mmap)
        split_mode = false;
        cpu_mode = true;
    } else if (!split_mode) {
        fprintf(stderr, "nemo ec: no GPU split placement available,"
                        " NEMOTRONOMNI_EXPERTCACHE disabled\n");
        ec_enabled = false;
        return true;
    }
    g_ec_self = this;
    // layer maps (NEMOTRONOMNI_MAX_LAYERS truncation mirrors build_text_graph
    // so the xchk comparison stays layer-for-layer exact)
    int max_layers = n_layer;
    if (const char * e = getenv("NEMOTRONOMNI_MAX_LAYERS"))
        max_layers = std::min(max_layers, atoi(e));
    ec_m_idx.assign((size_t)n_layer, -1);
    ec_kv_idx.assign((size_t)n_layer, -1);
    ec_moe_of.assign((size_t)n_layer, -1);
    ec_e_layer.assign((size_t)n_layer, -1);
    int si = 0, ki = 0, ei = 0;
    for (int il = 0; il < max_layers; il++) {
        if (pattern[(size_t)il] == 'M') ec_m_idx[(size_t)il] = si++;
        else if (pattern[(size_t)il] == '*') ec_kv_idx[(size_t)il] = ki++;
        else {
            ec_e_layer[(size_t)ei] = il;   // MoE slot -> driver layer
            ec_moe_of[(size_t)il] = ei++;  // driver layer -> MoE slot
        }
    }
    const int n_e = ei;
    for (int il = 0; il < max_layers; il++)
        if (pattern[(size_t)il] == 'E') tl[(size_t)il].e.f_eps = f_eps;

    struct Tr { static const ggml_tensor * get(int l, int t) {
        // l = MoE slot index; t: 0 up / 1 down (tensors_per_expert = 2)
        NemotronOmni & M = *g_ec_self;
        const MoEW & w = M.tl[(size_t)M.ec_e_layer[(size_t)l]].e;
        return t == 0 ? w.up_exps : w.down_exps;
    }};
    size_t cpu_budget = cpu_mode ? (size_t)4.0e9 : 0;
    size_t reserve = (size_t)2.0e9;
    if (const char * e = getenv("NEMO_EC_BUDGET")) cpu_budget = (size_t)atof(e);
    if (const char * e = getenv("NEMO_EC_RESERVE")) reserve = (size_t)atof(e);
    vqa::moe_decode::InitSpec sp{
        backend,
        n_e, n_expert, n_expert_used,
        &Tr::get,
        cpu_budget,
        reserve,
        true,               // stage_threads_16 (qwen3omni-measured)
        "nemo ec",
        ec_enabled,
        ec_caches,
        ec_group,
        ec_n_groups,
        ec_hooks.async_ok,
    };
    sp.tensors_per_expert = 2;
    // GPU: the slot count must come strictly from the budget - the legacy
    // n_min raise (23 layers x 6 used x 2 = 276 slots x ~20 MB) would ask for
    // 5.5 GB of VRAM the F16 dense trunk's leftovers cannot cover
    sp.budget_is_hard = !cpu_mode;
    if (!vqa::moe_decode::init_grouped(sp)) return false;
    if (!ec_enabled) return true;
    ec_hooks.n_layer = max_layers;
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
    ec_hooks.moe_of = ec_moe_of.data();
    ec_hooks.tensors_per_expert = 2;
    ec_hooks.log_tag = "nemo ec";
    ec_hooks.pf_env = "NEMO_EC_PF";
    ec_hooks.spec_env = "NEMO_EC_SPEC";
    ec_hooks.build_a = &NemotronOmni::ec_build_a;
    ec_hooks.feed_a = &NemotronOmni::ec_feed_a;
    ec_hooks.post_a = &NemotronOmni::ec_post_a;
    ec_hooks.build_b = &NemotronOmni::ec_build_b;
    ec_hooks.feed_b = &NemotronOmni::ec_feed_b;
    ec_hooks.layer_check = &NemotronOmni::ec_layer_check;
    ec_ga_io.assign((size_t)max_layers, EcGraphA());
    ec_mask_n = -1;
#ifdef GGML_USE_CUDA
    if (!ggml_backend_is_cpu(backend)) {
        size_t free_vram = 0, total_vram = 0;
        ggml_backend_cuda_get_device_memory(0, &free_vram, &total_vram);
        fprintf(stderr, "nemo ec: split mode, %.2f GB VRAM free after placement\n",
                free_vram / 1e9);
    }
#endif
    ec_ok = true;
    return true;
}

// ------------------------------------------------------------------ graph A
// Fixed per-layer graph. The op sequences are copied VERBATIM from
// build_text_graph (T=1) through the shared branch builders (mamba2_branch /
// attn_branch / the moe router tail) - do not reorder.
ggml_tensor * NemotronOmni::ec_ga_builder(ggml_context * ca, void * ud_v) {
    EcGaUd & ud = *(EcGaUd *) ud_v;
    NemotronOmni & M = *ud.m;
    const int il = ud.il;
    EcGraphA & io = *ud.io;
    ggml_cgraph * fa = vqa::step_graph_builder_gf();

    const int D = M.n_embd, HD = M.head_dim, NH = M.n_head, NKV = M.n_kv_head;
    const int T = 1;
    const int S = M.ec_st.S;
    const char kind = M.pattern[(size_t)il];
    const TxtLayer & L = M.tl[(size_t)il];

    io.tok_in = ggml_new_tensor_1d(ca, GGML_TYPE_I32, T);
    io.hid_in = ggml_new_tensor_2d(ca, GGML_TYPE_F32, D, T);
    ggml_tensor * x;
    if (il == 0) {
        x = ggml_get_rows(ca, M.tok_embd, io.tok_in);
    } else {
        x = io.hid_in;
    }
    if (kind == '*') {
        io.pos_in = ggml_new_tensor_1d(ca, GGML_TYPE_I32, T);
        io.mask_in = ggml_new_tensor_2d(ca, GGML_TYPE_F16, S, T);
    } else if (kind == 'M') {
        io.ssm_ids = ggml_new_tensor_1d(ca, GGML_TYPE_I32, 1);
    }

    ggml_tensor * l = ggml_rms_norm(ca, x, M.rms_eps);
    l = ggml_mul(ca, l, L.norm);
    if (kind == 'M') {
        ggml_tensor * mo = mamba2_branch(ca, fa, L.m,
                                         M.st[(size_t)M.ec_m_idx[(size_t)il]], l,
                                         io.ssm_ids, M.rms_eps,
                                         M.dt_clamp ? M.f_one : nullptr,
                                         M.dt_min, nullptr);
        x = ggml_add(ca, x, mo);
    } else if (kind == '*') {
        ggml_tensor * ao = attn_branch(ca, fa, L.a,
                                       M.kv.k[(size_t)M.ec_kv_idx[(size_t)il]],
                                       M.kv.v[(size_t)M.ec_kv_idx[(size_t)il]],
                                       l, io.mask_in, io.pos_in,
                                       M.rope_partial84 ? io.pos_in : nullptr,
                                       M.rope_dims, M.rope_theta, M.n_ctx,
                                       M.n_head, M.n_kv_head, M.head_dim,
                                       nullptr);
        x = ggml_add(ca, x, ao);
    } else {
        // MoE layer: attention-free - the residual add happens in graph B
        // (xmid + (routed + shexp)). Router tail copied VERBATIM from
        // moe_branch (sigmoid + selection-only bias, unbiased weights
        // renormalized, x routed_scaling).
        const MoEW & w = L.e;
        ggml_tensor * rlogits = ggml_mul_mat(ca, w.router, l);          // [E, T]
        ggml_tensor * sig = ggml_sigmoid(ca, rlogits);
        ggml_tensor * sel_score = w.probs_b ? ggml_add(ca, sig, w.probs_b) : sig;
        ggml_tensor * sel = ggml_top_k(ca, sel_score, M.n_expert_used); // [K, T]
        ggml_tensor * wsel = ggml_get_rows(ca,
                ggml_reshape_3d(ca, sig, 1, M.n_expert, T), sel);
        wsel = ggml_reshape_2d(ca, wsel, M.n_expert_used, T);
        ggml_tensor * wsum = ggml_sum_rows(ca, wsel);
        if (w.f_eps) wsum = ggml_add(ca, wsum, w.f_eps);
        ggml_tensor * w8 = ggml_div(ca, wsel, wsum);
        w8 = ggml_scale(ca, w8, M.routed_scaling);
        io.m_o = ggml_dup(ca, l);
        io.w8_o = ggml_dup(ca, w8);
        io.sel_o = ggml_dup(ca, sel);
        io.x_o = ggml_dup(ca, x);
    }
    if (kind != 'E') {
        // whole layer already computed: the driver still reads the four AIO
        // leaves back - m/x are the layer's output row, sel/w8 are zero-fed
        // dummies whose values are never used (moe_of == -1 skips graph B)
        io.sel_o = ggml_new_tensor_2d(ca, GGML_TYPE_I32, M.n_expert_used, T);
        io.w8_o = ggml_new_tensor_2d(ca, GGML_TYPE_F32, M.n_expert_used, T);
        io.m_o = ggml_dup(ca, x);
        io.x_o = ggml_dup(ca, x);
    }

    // dup'd tails: keep their buffers for the whole compute (see keye's note:
    // a tensor that sits mid-graph may get its buffer reused after its last
    // consumer, so a plain node would alias garbage at readback time)
    ggml_set_name(io.w8_o, "nec_w8_o");
    ggml_set_name(io.sel_o, "nec_sel_o");
    ggml_set_name(io.m_o, "nec_m_o");
    ggml_set_name(io.x_o, "nec_x_o");
    ggml_build_forward_expand(fa, io.w8_o);
    ggml_build_forward_expand(fa, io.sel_o);
    ggml_build_forward_expand(fa, io.m_o);
    ggml_build_forward_expand(fa, io.x_o);
    return io.x_o;
}

bool NemotronOmni::ec_build_a(void * m, int il, vqa::StepGraph & sg,
                              vqa::moe_decode::AIO & io) {
    NemotronOmni & M = *(NemotronOmni *) m;
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
    if (M.pattern[(size_t)il] != 'E') {
        // zero the dummy router readbacks once (values unused on M/* layers)
        const std::vector<int> zi((size_t)M.n_expert_used, 0);
        const std::vector<float> zf((size_t)M.n_expert_used, 0.0f);
        sg.set_input(eg.sel_o, zi.data(), sizeof(int) * zi.size());
        sg.set_input(eg.w8_o, zf.data(), sizeof(float) * zf.size());
    }
    return true;
}

bool NemotronOmni::ec_feed_a(void * m, int il, vqa::moe_decode::AIO & io,
                             vqa::StepGraph & sg, const void * step_ctx,
                             int cur_tok, int n_past, const float * hid) {
    (void)step_ctx; (void)hid;
    NemotronOmni & M = *(NemotronOmni *) m;
    EcGraphA & eg = *(EcGraphA *) io.model_io;
    const int S = M.ec_st.S;
    static const bool dbg_feed = getenv("NEMO_EC_DBG") != nullptr;
    if (dbg_feed)
        fprintf(stderr, "ec feed L%d kind %c tok=%p hid=%p pos=%p mask=%p ids=%p\n",
                il, M.pattern[(size_t)il], (void *)eg.tok_in, (void *)eg.hid_in,
                (void *)eg.pos_in, (void *)eg.mask_in, (void *)eg.ssm_ids);
    bool ok = true;
    if (il == 0) {
        ok = sg.set_input(eg.tok_in, &cur_tok, sizeof(int));
    } else {
        ok = sg.set_input(eg.hid_in, hid, sizeof(float) * M.n_embd);
    }
    if (eg.pos_in) {
        const int p1[1] = { n_past };   // KV slot == absolute position (NoPE)
        ok = ok && sg.set_input(eg.pos_in, p1, sizeof(int));
    }
    if (eg.ssm_ids) {
        const int32_t z = 0;            // single-sequence state slot
        ok = ok && sg.set_input(eg.ssm_ids, &z, sizeof(int32_t));
    }
    if (eg.mask_in) {
        // same row as set_causal_mask(mask, n_past, 1): 0 for s <= n_past
        if (M.ec_mask.size() != (size_t)S || M.ec_mask_n != n_past) {
            M.ec_mask.assign((size_t)S, ggml_fp32_to_fp16(-INFINITY));
            for (int sq = 0; sq <= n_past; sq++)
                M.ec_mask[(size_t)sq] = ggml_fp32_to_fp16(0.0f);
            M.ec_mask_n = n_past;
        }
        ok = ok && sg.set_input(eg.mask_in, M.ec_mask.data(),
                                sizeof(ggml_fp16_t) * M.ec_mask.size());
    }
    return ok;
}

void NemotronOmni::ec_post_a(void * m, int il, const float * xmid,
                             const float * mn, const int * sel,
                             const float * w8) {
    NemotronOmni & M = *(NemotronOmni *) m;
    if (getenv("NEMO_EC_DBG") && M.pattern[(size_t)il] == 'E') {
        fprintf(stderr, "ec L%02d sel:", il);
        for (int ke = 0; ke < M.n_expert_used; ke++)
            fprintf(stderr, " %d(%.3f)", sel[ke], w8[ke]);
        fprintf(stderr, "\n");
    }
    // M/* layers: x_o IS the layer output - probe it here (the driver skips
    // graph B and layer_check for them)
    if (M.pattern[(size_t)il] != 'E' && !M.ec_ref_layers.empty()) {
        const float * ref = M.ec_ref_layers.data() + (size_t)il * M.n_embd;
        double dmax = 0; int im = 0;
        for (int j = 0; j < M.n_embd; j++) {
            const double d = fabs((double)xmid[j] - (double)ref[j]);
            if (d > dmax) { dmax = d; im = j; }
        }
        fprintf(stderr, "xchk L%02d maxdiff %.6g at %d (ec %.5f ref %.5f)\n",
                il, dmax, im, xmid[im], ref[im]);
    }
    (void)mn;
}

// ------------------------------------------------------------------ graph B
// The MoE tail, built over the acquired slot tensors. Default (table layout):
// up_table -> relu^2 -> down_table through 2 mul_mat_id launches + the dense
// shared expert; the op sequence and the reduction order replicate
// moe_branch VERBATIM so the EC bits match the standard path exactly
// (mul_mat_id per selected expert is id-sliced but byte-identical math).
ggml_tensor * NemotronOmni::ec_build_b(void * m, int il, ggml_context * cb,
                                       ggml_cgraph * fb, ggml_tensor * moe_in,
                                       ggml_tensor * xmid_in,
                                       const std::vector<ggml_tensor *> & slots,
                                       ggml_tensor * w_in) {
    NemotronOmni & M = *(NemotronOmni *) m;
    const int D = M.n_embd, T = 1, K = M.n_expert_used;
    const MoEW & w = M.tl[(size_t)il].e;
    vqa::ExpertCache & c =
        M.ec_caches[(size_t)M.ec_group[(size_t)M.ec_moe_of[(size_t)il]]];

    ggml_tensor * ff3 = nullptr;
    if (c.table(0) && c.table(1)) {
        // slot views are [up, down] per expert (tensors_per_expert = 2); ids
        // are read right here - after this layer's acquires, before the next
        // (the bytes go into the graph input only in ec_feed_b: tensors in
        // the scratch ctx have no buffer until the driver's gallocr alloc)
        for (int ke = 0; ke < K; ke++)
            M.ec_b_ids[(size_t)ke] = c.slot_index(slots[(size_t)ke * 2]);
        ggml_tensor * ids = ggml_new_tensor_2d(cb, GGML_TYPE_I32, K, T);
        M.ec_ga_io[(size_t)il].ids_in = ids;
        ggml_tensor * m3 = ggml_reshape_3d(cb, moe_in, D, 1, T);
        ggml_tensor * gu = ggml_mul_mat_id(cb, c.table(0), m3, ids);  // [I, K, T]
        ggml_tensor * act = ggml_sqr(cb, ggml_relu(cb, gu));          // relu^2
        ff3 = ggml_mul_mat_id(cb, c.table(1), act, ids);              // [D, K, T]
    } else {
        // EC_TABLE=0 fallback: per-expert GEMVs over the acquired slots,
        // accumulated in expert order like the sum_rows reduction
        ggml_tensor * acc = nullptr;
        for (int ke = 0; ke < K; ke++) {
            ggml_tensor * gu = ggml_mul_mat(cb, slots[(size_t)ke * 2], moe_in);
            gu = ggml_sqr(cb, ggml_relu(cb, gu));
            ggml_tensor * f = ggml_mul_mat(cb, slots[(size_t)ke * 2 + 1], gu);
            ggml_tensor * wk = ggml_view_2d(cb, w_in, 1, w_in->ne[1],
                                            w_in->nb[1], (size_t)ke * w_in->nb[0]);
            ggml_tensor * term = ggml_mul(cb, f, wk);
            acc = acc ? ggml_add(cb, acc, term) : term;
        }
        ff3 = ggml_reshape_3d(cb, acc, D, K, T);
    }

    // weighted sum + shared expert + residual - VERBATIM moe_branch tail
    // (out = xmid + (routed + shexp), the exact standard-path order)
    ggml_tensor * wb = ggml_reshape_3d(cb, w_in, 1, K, T);
    ggml_tensor * prod = ggml_mul(cb, ff3, wb);                         // [D, K, T]
    ggml_tensor * pk = ggml_cont(cb, ggml_permute(cb, prod, 1, 0, 2, 3));
    ggml_tensor * routed = ggml_reshape_2d(cb, ggml_sum_rows(cb, pk), D, T);
    ggml_tensor * sh = ggml_mul_mat(cb, w.shexp_up, moe_in);
    sh = ggml_sqr(cb, ggml_relu(cb, sh));
    sh = ggml_mul_mat(cb, w.shexp_down, sh);
    ggml_tensor * out = ggml_add(cb, xmid_in, ggml_add(cb, routed, sh));
    ggml_build_forward_expand(fb, out);
    return out;
}

void NemotronOmni::ec_feed_b(void * m, int il) {
    NemotronOmni & M = *(NemotronOmni *) m;
    EcGraphA & eg = M.ec_ga_io[(size_t)il];
    if (eg.ids_in)
        ggml_backend_tensor_set(eg.ids_in, M.ec_b_ids.data(), 0,
                                sizeof(int) * (size_t)M.n_expert_used);
}

// NEMO_XCHK per-layer probe: diff the EC layer output against the reference
// decode graph's t_l<il>_out_o. The reference values are SNAPSHOT right after
// the ref run (they live in the shared gallocr arena and the EC step's graph
// B allocations clobber them mid-step - reading them live compares garbage).
bool NemotronOmni::ec_layer_check(void * m, int il, float * hid_io) {
    NemotronOmni & M = *(NemotronOmni *) m;
    if (M.ec_ref_layers.empty()) return true;
    const float * ref = M.ec_ref_layers.data() + (size_t)il * M.n_embd;
    double dmax = 0; int im = 0;
    for (int j = 0; j < M.n_embd; j++) {
        const double d = fabs((double)hid_io[j] - (double)ref[j]);
        if (d > dmax) { dmax = d; im = j; }
    }
    fprintf(stderr, "xchk L%02d maxdiff %.6g at %d (ec %.5f ref %.5f)\n",
            il, dmax, im, hid_io[im], ref[im]);
    return true;
}

bool NemotronOmni::ec_decode_step(int cur_tok, int n_past,
                                  std::vector<float> & logits) {
    // NEMO_XCHK: run the standard decode path for the same step first and
    // compare the logits (CPU validation: maxdiff must print 0.00000). The
    // ref run shares the KV/Mamba state assets and writes the same n_past
    // row the EC step is about to write.
    std::vector<float> ref_logits;
    const bool xchk = getenv("NEMO_XCHK") != nullptr;
    if (xchk) {
        TextGraph rg;
        const int p_saved = n_past;
        // KV is idempotent under a re-run (both paths rewrite the same
        // n_past row), but the Mamba2 states are NOT: the ref run consumes
        // + advances every conv/ssm state, so the EC step must see the exact
        // same "before" state. Snapshot before the ref, restore after it
        // (xchk-only cost: ~50 MB of state copy per step).
        ec_state_snap.clear();
        for (int i = 0; i < n_mamba; i++) {
            const ggml_tensor * ts[2] = { st[(size_t)i].conv, st[(size_t)i].ssm };
            for (int t = 0; t < 2; t++) {
                const size_t nb = ggml_nbytes(ts[t]);
                const size_t at = ec_state_snap.size();
                ec_state_snap.resize(at + nb);
                ggml_backend_tensor_get(ts[t], ec_state_snap.data() + at, 0, nb);
            }
        }
        if (!run_text(rg, { cur_tok }, { n_past }, ref_logits)) {
            if (rg.ctx) ggml_free(rg.ctx);
            return false;
        }
        this->n_past = p_saved;   // run_text advanced the MEMBER; undo it
                                  // (qwen3omni lesson: a bare n_past here
                                  // assigns the shadowing parameter and
                                  // leaks +1/step into the KV position)
        {
            size_t at = 0;
            for (int i = 0; i < n_mamba; i++) {
                ggml_tensor * ts[2] = { st[(size_t)i].conv, st[(size_t)i].ssm };
                for (int t = 0; t < 2; t++) {
                    const size_t nb = ggml_nbytes(ts[t]);
                    ggml_backend_tensor_set(ts[t], ec_state_snap.data() + at, 0, nb);
                    at += nb;
                }
            }
        }
        // keep the ref layer outputs for the per-layer probe: snapshot them
        // NOW - they sit in the shared gallocr arena and the EC step's graph
        // B allocations clobber them mid-step
        {
            TextGraph * rgp = rg.gf ? &rg : &dec_slot.g;
            ec_ref_layers.assign((size_t)ec_hooks.n_layer * (size_t)n_embd, 0.0f);
            for (int il = 0; il < ec_hooks.n_layer; il++) {
                const ggml_tensor * xo = ggml_graph_get_tensor(rgp->gf,
                        ("t_l" + std::to_string(il) + "_out_o").c_str());
                if (xo) ggml_backend_tensor_get(xo,
                        ec_ref_layers.data() + (size_t)il * (size_t)n_embd, 0,
                        sizeof(float) * (size_t)n_embd);
            }
        }
        if (!vqa::moe_decode::decode_step(ec_hooks, ec_st, this, nullptr,
                                          cur_tok, n_past, logits)) {
            ec_ref_layers.clear();
            return false;
        }
        ec_ref_layers.clear();
        double dmax = 0; int im = 0;
        for (int v = 0; v < n_vocab; v++) {
            double d = fabs((double)logits[(size_t)v] - ref_logits[(size_t)v]);
            if (d > dmax) { dmax = d; im = v; }
        }
        fprintf(stderr, "nemo xchk step %d logits maxdiff %.5f at id %d"
                " (ec %.4f ref %.4f)\n", (int)ec_st.steps, dmax, im,
                logits[(size_t)im], ref_logits[(size_t)im]);
        return true;
    }
    if (!vqa::moe_decode::decode_step(ec_hooks, ec_st, this, nullptr, cur_tok,
                                      n_past, logits))
        return false;
    return true;
}
