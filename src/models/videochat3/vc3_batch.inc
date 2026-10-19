// vc3_batch.inc - batched text decoding for the VideoChat3 backend.
// Included from vc3_model.cpp after the VideoChat3 class definition.
//
// Attention runs per sequence over its OWN windowed KV view (see the groups
// layout in build_text_graph): prefill sweeps exactly its prompt length
// instead of all n_ctx slots, decode keeps a block-rounded per-column window
// that grows in rare rebuilds (static shapes in between, so the graph stays
// CUDA-capture eligible).

#include "core/engine/attention.hpp"

struct BatchSeq {
    std::vector<int> toks, out;
    int base = 0, len = 0, next = 0;
    bool done = false;
};

static void vc3_set_group_masks(ggml_cgraph * gf, const std::vector<BatGroup> & groups,
                                const std::vector<int> & lo, const std::vector<int> & hi,
                                const std::vector<int> & pos) {
    // group masks are WINDOW-RELATIVE [0, win): shift the absolute slot
    // windows by the group's kv base before filling them
    for (const auto & gr : groups) {
        char mn[24];
        snprintf(mn, sizeof(mn), "mask_b%d", gr[0]);
        ggml_tensor * m = ggml_graph_get_tensor(gf, mn);
        if (!m) continue;
        std::vector<int> glo(gr[1]), ghi(gr[1]), gpos(gr[1]);
        for (int k = 0; k < gr[1]; k++) {
            glo[k]  = lo[gr[0] + k]  - gr[2];
            ghi[k]  = hi[gr[0] + k]  - gr[2];
            gpos[k] = pos[gr[0] + k] - gr[2];
        }
        vqa::set_window_mask(m, glo, ghi, gpos);
    }
}

int VideoChat3::step_batch(const std::vector<int> & toks, const std::vector<int> & posv,
                           const std::vector<int> & lo, const std::vector<int> & hi,
                           const std::vector<BatGroup> & groups, int win, float * logits_out) {
    const int T = (int)toks.size();
    bool static_bat = false;
    struct ggml_context * gctx = nullptr;
    struct ggml_cgraph * gf = nullptr;

    // per-column windowed FA measured 3.4x SLOWER for decode (36 layers x B
    // tiny FA launches are launch-bound); decode keeps the single full-window
    // masked FA. Prefill keeps the windowed groups - exact and cheap there.
    const bool windowed = !groups.empty() && groups[0][1] > 1;   // prefill groups span columns
    const bool decode_shape = (bat_n > 0 && T == bat_n);
    if (decode_shape && bat_gf && bat_win == win) {
        gctx = bat_ctx; gf = bat_gf; static_bat = true;
    } else {
        struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, true };
        gctx = ggml_init(ip);
        TextGraph g;
        g.ctx = gctx;
        if (!build_text_graph(g, T, windowed ? win : n_ctx, 0, decode_shape,
                              windowed ? &groups : nullptr)) { ggml_free(gctx); return -1; }
        gf = g.gf;
        if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(allocr, gf)) { fprintf(stderr, "batch graph alloc failed\n"); ggml_free(gctx); return -1; }
        if (decode_shape) {
            if (bat_ctx) ggml_free(bat_ctx);
            bat_ctx = gctx;
            bat_gf = gf;
            bat_win = win;
            bat_groups = groups;
            bat_tok = ggml_graph_get_tensor(gf, "tok_in");
            bat_pos = ggml_graph_get_tensor(gf, "pos_in");
            static_bat = true;
        }
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tok_in"), toks.data(), 0, sizeof(int) * T);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_in"), posv.data(), 0, sizeof(int) * T);
    if (windowed) {
        vc3_set_group_masks(gf, groups, lo, hi, posv);
    } else {
        ggml_tensor * m = ggml_graph_get_tensor(gf, "mask_in");
        if (m) vqa::set_window_mask(m, lo, hi, posv);
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "batch compute failed\n");
        if (!static_bat) ggml_free(gctx);
        return -1;
    }
    if (logits_out) {
        struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
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
        return vqa::greedy_argmax(col, n_vocab);
    };

    // prefill: one sequence per call, attention windowed to the prompt length
    // (the old graphs swept all n_ctx slots for every sequence)
    for (int i = 0; i < B; i++) {
        BatchSeq & sq = seqs[i];
        sq.base = i * cap;
        sq.toks = tok_.encode("<|im_start|>user\n" + prompts[i] + "<|im_end|>\n<|im_start|>assistant\n");
        const int T = (int)sq.toks.size();
        std::vector<int> pos(T), lo(T), hi(T);
        for (int t = 0; t < T; t++) { pos[t] = sq.base + t; lo[t] = sq.base; hi[t] = sq.base + t + 1; }
        std::vector<BatGroup> groups = { { 0, T, sq.base, T } };
        step_batch(sq.toks, pos, lo, hi, groups, T, lv.data());
        sq.next = col_argmax(lv.data());
        sq.len = T;
        sq.done = (sq.next == id_eos || sq.next == id_endoftext);
    }

    // batched decode: every sequence advances in one graph per step; the
    // per-column window is block-rounded over the longest sequence and only
    // rebuilt when it grows past the block edge
    auto t0 = std::chrono::steady_clock::now();
    int steps = 0;
    while (steps < n_gen) {
        bool any = false;
        for (auto & sq : seqs) if (!sq.done) any = true;
        if (!any) break;
        std::vector<int> ct(B), pos(B), lo(B), hi(B);
        int max_len = 0;
        for (int i = 0; i < B; i++) max_len = std::max(max_len, seqs[i].len);
        const int win = cap; (void)max_len;   // TEMP DEBUG
        std::vector<BatGroup> groups;
        for (int i = 0; i < B; i++) {
            BatchSeq & sq = seqs[i];
            ct[i] = sq.done ? id_eos : sq.next;
            pos[i] = sq.base + sq.len;
            lo[i] = sq.base;
            hi[i] = sq.base + sq.len + 1;
            groups.push_back({ i, 1, sq.base, win });
        }
        step_batch(ct, pos, lo, hi, groups, win, lv.data());
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
