// vc3_batch.inc - batched text decoding for the VideoChat3 backend.
// Included from vc3_model.cpp after the VideoChat3 class definition.
//
// Same design as moondream_batch.inc: the compute graph is unchanged, positions
// and the attention mask are graph inputs. Each sequence gets a private slot
// range in the shared KV cache; per-column mask windows [lo, hi) express
// per-sequence visibility. Prefill runs one full-window graph per sequence
// (logits for the last column only); decode runs one static T=B graph per step
// (logits for all columns), eligible for ggml CUDA graph capture.

#include "attention.hpp"


struct BatchSeq {
    std::vector<int> toks, out;
    int base = 0, len = 0, next = 0;
    bool done = false;
};

int VideoChat3::step_batch(const std::vector<int> & toks, const std::vector<int> & posv,
                           const std::vector<int> & lo, const std::vector<int> & hi, float * logits_out) {
    const int T = (int)toks.size();
    bool static_bat = false;
    struct ggml_context * gctx = nullptr;
    struct ggml_cgraph * gf = nullptr;

    if (bat_n > 0 && T == bat_n && bat_gf) {
        gctx = bat_ctx; gf = bat_gf; static_bat = true;
    } else {
        struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, true };
        gctx = ggml_init(ip);
        TextGraph g;
        g.ctx = gctx;
        const bool all = (bat_n > 0 && T == bat_n);   // first batch-decode step
        if (!build_text_graph(g, T, n_ctx, 0, all)) { ggml_free(gctx); return -1; }
        gf = g.gf;
        if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(allocr, gf)) { fprintf(stderr, "batch graph alloc failed\n"); ggml_free(gctx); return -1; }
        if (all) {
            bat_ctx = gctx;
            bat_gf = gf;
            bat_tok = ggml_graph_get_tensor(gf, "tok_in");
            bat_pos = ggml_graph_get_tensor(gf, "pos_in");
            bat_mask = ggml_graph_get_tensor(gf, "mask_in");
            static_bat = true;
        }
    }

    ggml_backend_tensor_set(static_bat ? bat_tok : ggml_graph_get_tensor(gf, "tok_in"),
                            toks.data(), 0, sizeof(int) * T);
    ggml_backend_tensor_set(static_bat ? bat_pos : ggml_graph_get_tensor(gf, "pos_in"),
                            posv.data(), 0, sizeof(int) * T);
    ggml_tensor * mask = static_bat ? bat_mask : ggml_graph_get_tensor(gf, "mask_in");
    if (mask) vqa::set_window_mask(mask, lo, hi, posv);

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "batch compute failed\n");
        if (!static_bat) ggml_free(gctx);
        return -1;
    }
    if (logits_out) {
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
        // prefill graphs carry one logits column, the static batch graph [vocab, T];
        // copy exactly what the graph produced (see moondream_batch.inc heap lesson)
        const int64_t ncols = logits->ne[1];
        ggml_backend_tensor_get(logits, logits_out, 0, sizeof(float) * (size_t)n_vocab * ncols);
    }
    if (!static_bat) ggml_free(gctx);
    return 0;
}

int VideoChat3::run_batch(const std::string & path, int n_gen) {
    std::vector<std::string> prompts;
    {
        FILE * bf = fopen(path.c_str(), "rb");
        if (!bf) { fprintf(stderr, "cannot open %s\n", path.c_str()); return 1; }
        std::string line; int c;
        while ((c = fgetc(bf)) != EOF) {
            if (c == '\n' || c == '\r') {
                if (!line.empty()) { prompts.push_back(line); line.clear(); }
            } else line += (char)c;
        }
        if (!line.empty()) prompts.push_back(line);
        fclose(bf);
    }
    if (prompts.empty()) { fprintf(stderr, "no prompts in %s\n", path.c_str()); return 1; }
    const int B = (int)std::min<size_t>(prompts.size(), 8);
    const int cap = n_ctx / B;
    bat_n = B;
    fprintf(stderr, "batch: %d sequences, %d slots each\n", B, cap);

    std::vector<BatchSeq> seqs(B);
    std::vector<float> lv((size_t)n_vocab * B);
    auto col_argmax = [&](const float * col) {
        int best = 0; float bv = -INFINITY;
        for (int i = 0; i < n_vocab; i++) { if (col[i] > bv) { bv = col[i]; best = i; } }
        return best;
    };

    // prefill each sequence in its own KV slot range
    for (int i = 0; i < B; i++) {
        BatchSeq & sq = seqs[i];
        sq.base = i * cap;
        // chat template without a vision block
        sq.toks = tok_.encode("<|im_start|>user\n" + prompts[i] + "<|im_end|>\n<|im_start|>assistant\n");
        const int T = (int)sq.toks.size();
        std::vector<int> pos(T), lo(T), hi(T);
        for (int t = 0; t < T; t++) { pos[t] = sq.base + t; lo[t] = sq.base; hi[t] = sq.base + t + 1; }
        step_batch(sq.toks, pos, lo, hi, lv.data());
        sq.next = col_argmax(lv.data());
        sq.len = T;
        sq.done = (sq.next == id_eos || sq.next == id_endoftext);
    }

    // batched decode: every sequence advances in one graph per step
    auto t0 = std::chrono::steady_clock::now();
    int steps = 0;
    while (steps < n_gen) {
        bool any = false;
        for (auto & sq : seqs) if (!sq.done) any = true;
        if (!any) break;
        std::vector<int> ct(B), pos(B), lo(B), hi(B);
        for (int i = 0; i < B; i++) {
            BatchSeq & sq = seqs[i];
            ct[i] = sq.done ? id_eos : sq.next;   // finished columns decode a pad token
            pos[i] = sq.base + sq.len;
            lo[i] = sq.base;
            hi[i] = sq.base + sq.len + 1;         // includes the key written this step
        }
        step_batch(ct, pos, lo, hi, lv.data());
        steps++;
        for (int i = 0; i < B; i++) {
            BatchSeq & sq = seqs[i];
            if (sq.done) continue;
            sq.out.push_back(sq.next);
            sq.len++;
            sq.next = col_argmax(lv.data() + (size_t)i * n_vocab);
            if (sq.next == id_eos || sq.next == id_endoftext) sq.done = true;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    size_t ntok = 0;
    for (int i = 0; i < B; i++) {
        ntok += seqs[i].out.size();
        printf("[%d] %s\n", i, tok_.decode(seqs[i].out).c_str());
    }
    fprintf(stderr, "batch: %zu tokens / %d seqs in %.1fs (%.1f tok/s aggregate)\n",
            ntok, B, secs, ntok / secs);
    return 0;
}

