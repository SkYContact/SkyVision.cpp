// moondream_batch.inc — batched text decoding for the moondream CLI.
// Included from src/models/moondream/moondream.cpp AFTER the Engine definition; defines
// Engine::step_batch + a standalone run_batch() driver. Kept separate to avoid
// touching the (working) single-sequence paths.
//
// Design: the compute graphs are reused unchanged — positions and the attention
// mask are already graph inputs. Each sequence gets a private slot range in the
// shared KV cache; per-column mask windows [lo, hi] express per-sequence visibility.
// Prefill runs one one-shot fixed_ctx graph per sequence (logits for the last
// column only); decode runs one static T=B graph per step (logits for all columns),
// which is also eligible for ggml CUDA graph capture.

struct BatchSeq {
    std::vector<int> toks, out;
    int base = 0, len = 0, next = 0;
    bool done = false;
};

int Engine::step_batch(const std::vector<int> & toks, const std::vector<int> & posv,
                       const std::vector<int> & lo, const std::vector<int> & hi, float * logits_out) {
    const int T = (int)toks.size();
    bool static_bat = false;
    struct ggml_context * gctx = nullptr;
    struct ggml_cgraph * gf = nullptr;

    if (bat_n > 0 && T == bat_n && !bat_gf) {
        // first batch-decode step: build the static T=B graph (full-window KV views,
        // logits for every column)
        struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, true };
        bat_ctx = ggml_init(ip);
        int * hb[2] = { nullptr, nullptr };
        bat_gf = build_graph(bat_ctx, toks.data(), T, hb, /*fixed_ctx*/ true,
                             /*emb_input*/ false, /*logits_all*/ true);
        if (!gallocr) gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(gallocr, bat_gf)) { fprintf(stderr, "batch graph alloc failed\n"); return -1; }
        bat_tok_in = ggml_graph_get_tensor(bat_gf, "tok_in");
        bat_pos_in = ggml_graph_get_tensor(bat_gf, "pos_in");
        bat_mask_in = ggml_graph_get_tensor(bat_gf, "mask_in");
    }
    if (bat_n > 0 && T == bat_n && bat_gf) {
        gctx = bat_ctx; gf = bat_gf; static_bat = true;
    } else {
        // one-shot prefill graph: fixed_ctx so the KV views cover the full window
        // (batch sequences live in slot ranges beyond n_past)
        struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, true };
        gctx = ggml_init(ip);
        int * hb[2] = { nullptr, nullptr };
        gf = build_graph(gctx, toks.data(), T, hb, /*fixed_ctx*/ true,
                         /*emb_input*/ false, /*logits_all*/ false);
        if (!gallocr) gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(gallocr, gf)) { fprintf(stderr, "batch graph alloc failed\n"); ggml_free(gctx); return -1; }
    }

    {
        struct ggml_tensor * tok_in = static_bat ? bat_tok_in : ggml_graph_get_tensor(gf, "tok_in");
        struct ggml_tensor * pos_in = static_bat ? bat_pos_in : ggml_graph_get_tensor(gf, "pos_in");
        ggml_backend_tensor_set(tok_in, toks.data(), 0, sizeof(int) * T);
        ggml_backend_tensor_set(pos_in, posv.data(), 0, sizeof(int) * T);
        struct ggml_tensor * mask = static_bat ? bat_mask_in : ggml_graph_get_tensor(gf, "mask_in");
        if (mask) {
            const int S = (int)mask->ne[0];
            std::vector<ggml_fp16_t> mv((size_t)S * T);
            for (int t = 0; t < T; t++)
                for (int s = 0; s < S; s++)
                    mv[(size_t)t * S + s] = ggml_fp32_to_fp16((s >= lo[t] && s <= hi[t]) ? 0.0f : -INFINITY);
            ggml_backend_tensor_set(mask, mv.data(), 0, sizeof(ggml_fp16_t) * mv.size());
        }
        const int H = model.n_head;
        std::vector<float> psv(H * T);
        char nm[32];
        for (int il = 0; il < model.n_layer; il++) {
            snprintf(nm, sizeof(nm), "tau_ps_%d", il);
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, nm);
            if (!t) continue;
            for (int k = 0; k < T; k++) {
                const float p = posv[k] + 1;
                for (int h = 0; h < H; h++)
                    psv[k * H + h] = 1.0f / (1.0f + expf(-model.tau_alpha_host[il * H + h] * logf(p))) + 0.5f;
            }
            ggml_backend_tensor_set(t, psv.data(), 0, sizeof(float) * psv.size());
        }
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "batch compute failed\n");
        if (!static_bat) ggml_free(gctx);
        return -1;
    }
    if (logits_out) {
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        // prefill graphs compute logits for the LAST column only ([vocab, 1]); the
        // static batch graph produces [vocab, T]. Take exactly what the graph has —
        // never replicate columns into the caller buffer (that overflowed the heap
        // in an earlier revision and looked like a prompt-dependent segfault).
        const int64_t ncols = logits->ne[1];
        ggml_backend_tensor_get(logits, logits_out, 0,
                sizeof(float) * (size_t)model.n_vocab * ncols);
    }
    if (!static_bat) ggml_free(gctx);
    return 0;
}

// driver: returns process exit code
static int run_batch(Engine & e, const char * batch_path, int n_ctx, int n_gen) {
    std::vector<std::string> prompts;
    {
        FILE * bf = fopen(batch_path, "rb");
        if (!bf) { fprintf(stderr, "cannot open %s\n", batch_path); return 1; }
        std::string line; int c;
        while ((c = fgetc(bf)) != EOF) {
            if (c == '\n' || c == '\r') {
                if (!line.empty()) { prompts.push_back(line); line.clear(); }
            } else line += (char)c;
        }
        if (!line.empty()) prompts.push_back(line);
        fclose(bf);
    }
    if (prompts.empty()) { fprintf(stderr, "no prompts in %s\n", batch_path); return 1; }
    const int B = (int)std::min<size_t>(prompts.size(), 8);
    const int cap = n_ctx / B;   // per-sequence KV slot range
    e.bat_n = B;
    fprintf(stderr, "batch: %d sequences, %d slots each\n", B, cap);

    std::vector<BatchSeq> seqs(B);
    std::vector<float> lv((size_t)e.model.n_vocab * B);
    const int V = e.model.n_vocab;
    auto col_argmax = [&](const float * col) {
        return vqa::greedy_argmax(col, V, 1, 9);   // 1..9 are reserved control tokens
    };

    // prefill each sequence in its own KV slot range
    for (int i = 0; i < B; i++) {
        BatchSeq & s = seqs[i];
        s.base = i * cap;
        s.toks = { 0, 1, 15381, 2 };   // bos + query prefix
        auto q = e.tok.encode(prompts[i]);
        s.toks.insert(s.toks.end(), q.begin(), q.end());
        s.toks.push_back(3);
        const int T = (int)s.toks.size();
        std::vector<int> pos(T), lo(T), hi(T);
        for (int t = 0; t < T; t++) { pos[t] = s.base + t; lo[t] = s.base; hi[t] = s.base + t; }
        e.step_batch(s.toks, pos, lo, hi, lv.data());   // single column -> lv[0]
        s.next = col_argmax(lv.data());
        s.len = T;
        s.done = (s.next == 0);
    }

    // batched decode: every sequence advances in one graph per step
    auto t0 = std::chrono::steady_clock::now();
    int steps = 0;
    while (steps < n_gen) {
        bool any = false;
        for (auto & s : seqs) if (!s.done) any = true;
        if (!any) break;
        std::vector<int> ct(B), pos(B), lo(B), hi(B);
        for (int i = 0; i < B; i++) {
            BatchSeq & s = seqs[i];
            ct[i] = s.done ? 0 : s.next;   // finished columns keep decoding a pad token
            pos[i] = s.base + s.len;
            lo[i] = s.base;
            hi[i] = s.base + s.len;        // includes the token written this step
        }
        e.step_batch(ct, pos, lo, hi, lv.data());
        steps++;
        for (int i = 0; i < B; i++) {
            BatchSeq & s = seqs[i];
            if (s.done) continue;
            s.out.push_back(s.next);
            s.len++;
            s.next = col_argmax(lv.data() + (size_t)i * V);
            if (s.next == 0) s.done = true;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    size_t ntok = 0;
    for (int i = 0; i < B; i++) {
        ntok += seqs[i].out.size();
        printf("[%d] %s\n", i, e.tok.decode(seqs[i].out).c_str());
    }
    fprintf(stderr, "batch: %zu tokens / %d seqs in %.1fs (%.1f tok/s aggregate)\n",
            ntok, B, secs, ntok / secs);
    return 0;
}
