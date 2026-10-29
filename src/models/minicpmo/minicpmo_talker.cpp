// minicpmo_talker.cpp - the MiniCPM-o 4.5 TTS sub-tower (tk.*): Llama-768 x
// 20L AR speech-code generator, from the hidden_text_merge condition to the
// codebook id sequence. The CosyVoice2 token2wav vocoder is a later wave.
//
// Sampling: the official non-streaming MiniCPMTTS.generate (modeling_minicpmo
// .py:4305-4501) - logits /= temperature, repetition-penalty processor (last
// 16-token window, penalty^freq, sign-aware), top-p 0.85 (softmax-cumsum,
// keep >= 3), top-k 25 (keep >= 3), eos masked before min_new_token 50,
// softmax + torch.multinomial. The oracle gate replays the official default
// TTSSamplingParams with torch CPU seed 0 (dump_tts: 102 codes, 86 unique);
// torch.multinomial is emulated with the exact CPU generator semantics
// (std::mt19937 seeded by manual_seed; uniform_real<float> = (draw & 2^24-1)
// * 2^-24; one draw per step; upper_bound over the float cumsum).
#include "minicpmo.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

namespace minicpmo {

// ------------------------------------------------------------------ torch CPU RNG

namespace {

// torch's CPU generator is a std::mt19937 seeded directly by manual_seed.
struct TorchRng {
    std::mt19937 mt;
    explicit TorchRng(uint64_t seed) : mt((uint32_t)seed) {}
    uint32_t random32() { return mt(); }
    // at::uniform_real_distribution<float>(0, 1): (draw & (2^24 - 1)) / 2^24
    float uniform_f32() {
        return (float)(random32() & 0xFFFFFFu) * (1.0f / 16777216.0f);
    }
};

int multinomial_one(const float * probs, int n, TorchRng & rng) {
    // torch CPU multinomial (num_samples = 1, with replacement): cumsum of the
    // (already normalized) probs, one uniform draw, std::upper_bound.
    double total = 0.0;
    for (int i = 0; i < n; i++) total += probs[i];
    const float r = rng.uniform_f32();
    const float target = (float)((double)r * total);
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        // upper_bound semantics: first prefix element strictly greater
        if (cum > target) return i;
    }
    return n - 1;
}

}  // namespace

// ------------------------------------------------------------------ condition

ggml_tensor * tts_condition(ggml_context * c, ggml_tensor * emb_text,
                            ggml_tensor * tok_in, ggml_tensor * hid_in,
                            ggml_tensor * fc1, ggml_tensor * fc1_b,
                            ggml_tensor * fc2, ggml_tensor * fc2_b) {
    // proj_semantic: fc1 -> ReLU -> fc2 (MultiModalProjector mlp)
    ggml_tensor * p = ggml_mul_mat(c, fc1, hid_in);
    p = ggml_add(c, p, fc1_b);
    p = ggml_relu(c, p);
    p = ggml_mul_mat(c, fc2, p);
    p = ggml_add(c, p, fc2_b);
    // F.normalize(p, p=2, dim=-1) - rowwise L2 (torch eps guard)
    ggml_tensor * sq = ggml_sum_rows(c, ggml_mul(c, p, p));      // [1, T]
    ggml_tensor * norm = ggml_sqrt(c, sq);
    norm = ggml_clamp(c, norm, 1e-12f, 1e30f);
    p = ggml_div(c, p, norm);
    // emb_text(tok) + normalized projection
    ggml_tensor * e = ggml_get_rows(c, emb_text, tok_in);
    return ggml_add(c, e, p);
}

// ------------------------------------------------------------------ graphs

// One tk llama-layer stack forward (prefill or step) over the caller's KV
// caches. emb [768, T]; pos [T] i32; mask [S, T] F16 (null for the causal
// prefill of a fresh cache is NOT allowed - always pass a mask).
static ggml_tensor * tk_forward(ggml_context * c, ggml_cgraph * gf, Talker & t,
                                ggml_tensor * emb, ggml_tensor * pos,
                                ggml_tensor * mask, int T, int S) {
    ggml_tensor * x = emb;
    for (int il = 0; il < (int)t.tl.size(); il++) {
        const TkLayer & L = t.tl[(size_t)il];
        minicpmo::LlmAttnW aw{ L.q, L.k, L.v, L.o, nullptr, nullptr };
        ggml_tensor * l = ggml_rms_norm(c, x, t.rms_eps);
        l = ggml_mul(c, l, L.attn_norm);
        ggml_tensor * a = llm_attn_branch(c, gf, aw, t.kv->k[(size_t)il],
                                          t.kv->v[(size_t)il], l, mask, pos,
                                          t.n_head, t.n_kv_head, t.head_dim,
                                          t.rope_theta, 4096, t.rms_eps,
                                          nullptr);
        x = ggml_add(c, x, a);
        minicpmo::LlmFfnW fw{ L.gate, L.up, L.down };
        l = ggml_rms_norm(c, x, t.rms_eps);
        l = ggml_mul(c, l, L.ffn_norm);
        ggml_tensor * f = llm_ffn_branch(c, fw, l, t.rms_eps, nullptr);
        x = ggml_add(c, x, f);
    }
    x = ggml_rms_norm(c, x, t.rms_eps);
    x = ggml_mul(c, x, t.output_norm);
    (void)S;
    return x;
}

bool Talker::synthesize(const std::vector<int> & tokens,
                        const std::vector<float> & hidden,
                        std::vector<int> & codes, int max_new_token) {
    codes.clear();
    // MINICPMO_TTS_ORACLE_COND=<prefix>: replay the oracle inputs_embeds
    // (f32 raw bin [T, 768]) + bound tokens (one id per line) - isolates the
    // AR sub-tower from LLM drift
    if (const char * oc = getenv("MINICPMO_TTS_ORACLE_COND")) {
        std::string ep = std::string(oc) + ".emb.bin";
        std::string tp = std::string(oc) + ".tok.txt";
        FILE * fe = fopen(ep.c_str(), "rb");
        FILE * ft = fopen(tp.c_str(), "r");
        if (!fe || !ft) {
            fprintf(stderr, "minicpmo tts: oracle cond files missing (%s)\n",
                    ep.c_str());
            if (fe) fclose(fe);
            if (ft) fclose(ft);
            return false;
        }
        fseek(fe, 0, SEEK_END);
        const long sz = ftell(fe);
        fseek(fe, 0, SEEK_SET);
        std::vector<float> cond((size_t)sz / 4);
        if (fread(cond.data(), 4, cond.size(), fe) != cond.size()) {
            fclose(fe);
            fclose(ft);
            return false;
        }
        fclose(fe);
        std::vector<int> otoks;
        int v = 0;
        while (fscanf(ft, "%d", &v) == 1) otoks.push_back(v);
        fclose(ft);
        if (cond.size() != otoks.size() * (size_t)n_embd) {
            fprintf(stderr, "minicpmo tts: oracle cond %zu floats vs %zu tokens\n",
                    cond.size(), otoks.size());
            return false;
        }
        return synthesize_cond(cond, codes, max_new_token);
    }
    if (tokens.empty() || hidden.size() != tokens.size() * (size_t)4096) {
        fprintf(stderr, "minicpmo tts: condition shape mismatch (%zu tokens,"
                        " %zu floats)\n", tokens.size(), hidden.size());
        return false;
    }
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    {   // condition ("hidden_text_merge") in a small graph, direct read
        struct ggml_init_params ip = { 32ull * 1024 * 1024, NULL, true };
        ggml_context * cc = ggml_init(ip);
        if (!cc) return false;
        ggml_cgraph * cf = ggml_new_graph_custom(cc, 4096, false);
        ggml_tensor * tok_in = ggml_new_tensor_1d(cc, GGML_TYPE_I32,
                                                  (int64_t)tokens.size());
        ggml_set_name(tok_in, "t_cond_tok");
        ggml_set_input(tok_in);
        ggml_tensor * hid_in = ggml_new_tensor_2d(cc, GGML_TYPE_F32, 4096,
                                                  (int64_t)tokens.size());
        ggml_set_name(hid_in, "t_cond_hid");
        ggml_set_input(hid_in);
        ggml_tensor * cond = tts_condition(cc, emb_text, tok_in, hid_in,
                                           proj_sem_fc1, proj_sem_fc1_b,
                                           proj_sem_fc2, proj_sem_fc2_b);
        ggml_build_forward_expand(cf, cond);
        if (!ggml_gallocr_alloc_graph(allocr, cf)) {
            fprintf(stderr, "minicpmo tts: cond alloc failed\n");
            ggml_free(cc);
            return false;
        }
        ggml_backend_tensor_set(tok_in, tokens.data(), 0,
                                sizeof(int) * tokens.size());
        ggml_backend_tensor_set(hid_in, hidden.data(), 0,
                                sizeof(float) * hidden.size());
        if (ggml_backend_graph_compute(backend, cf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "minicpmo tts: cond compute failed\n");
            ggml_free(cc);
            return false;
        }
        std::vector<float> condv((size_t)tokens.size() * n_embd);
        ggml_backend_tensor_get(cond, condv.data(), 0,
                                sizeof(float) * condv.size());
        ggml_free(cc);
        return synthesize_cond(condv, codes, max_new_token);
    }

}

bool Talker::synthesize_cond(const std::vector<float> & cond,
                             std::vector<int> & codes, int max_new_token) {    // sampling params: official default TTSSamplingParams (the oracle gate
    // replays exactly these); MINICPMO_TTS_GREEDY=1 = logits-gate mode
    const float temperature = 0.8f, top_p = 0.85f, rep_pen = 1.05f;
    const int top_k = 25, min_new_token = 50, win_size = 16;
    const bool greedy = getenv("MINICPMO_TTS_GREEDY") != nullptr;
    TorchRng rng(0);                       // official default seed 0
    if (const char * s = getenv("MINICPMO_TTS_SEED")) rng = TorchRng((uint64_t)atoll(s));

    const int T1 = (int)cond.size() / (size_t)n_embd;
    const int n_cond = T1 + 2;                    // + text_eos + audio_bos
    const int eos = n_audio_vocab - 1;            // 6561

    // ---- prefill: condition + eos/bos embeddings --------------------------
    std::vector<float> logits;
    int n_past = 0;
    {
        struct ggml_init_params ip = { 128ull * 1024 * 1024, NULL, true };
        ggml_context * c = ggml_init(ip);
        if (!c) return false;
        ggml_cgraph * gf = ggml_new_graph_custom(c, 32768, false);

        ggml_tensor * cond_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_embd, T1);
        ggml_set_name(cond_in, "t_cond");
        ggml_set_input(cond_in);
        ggml_tensor * bos_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, 2);
        ggml_set_name(bos_in, "t_bos");
        ggml_set_input(bos_in);
        ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_cond);
        ggml_set_name(pos, "t_pos");
        ggml_set_input(pos);
        ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_cond, n_cond);
        ggml_set_name(mask, "t_mask");
        ggml_set_input(mask);

        ggml_tensor * bos = ggml_get_rows(c, emb_text, bos_in);
        ggml_tensor * emb = ggml_concat(c, cond_in, bos, 1);   // [768, n_cond]
        ggml_tensor * x = tk_forward(c, gf, *this, emb, pos, mask, n_cond, n_cond);
        ggml_tensor * last = ggml_view_2d(c, x, n_embd, 1, x->nb[1],
                                          (size_t)(n_cond - 1) * x->nb[1]);
        ggml_tensor * lg = ggml_mul_mat(c, head_code, last);   // [6562, 1]
        ggml_set_name(lg, "t_logits");
        ggml_build_forward_expand(gf, lg);
        if (!ggml_gallocr_alloc_graph(allocr, gf)) {
            fprintf(stderr, "minicpmo tts: prefill alloc failed\n");
            ggml_free(c);
            return false;
        }
        ggml_backend_tensor_set(cond_in, cond.data(), 0,
                                sizeof(float) * cond.size());
        const int32_t bos2[2] = { text_eos, audio_bos };
        ggml_backend_tensor_set(bos_in, bos2, 0, sizeof(bos2));
        {
            std::vector<int32_t> pv((size_t)n_cond);
            for (int i = 0; i < n_cond; i++) pv[(size_t)i] = i;
            ggml_backend_tensor_set(pos, pv.data(), 0, sizeof(int) * n_cond);
            vqa::set_causal_mask(mask, 0, n_cond);
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "minicpmo tts: prefill compute failed\n");
            ggml_free(c);
            return false;
        }
        logits.resize((size_t)n_audio_vocab);
        ggml_backend_tensor_get(lg, logits.data(), 0,
                                sizeof(float) * n_audio_vocab);
        if (const char * sd = getenv("MINICPMO_TTS_STEP_DUMP")) {
            FILE * fp = fopen((std::string(sd) + "_step0.f32").c_str(), "wb");
            if (fp) { fwrite(logits.data(), 4, n_audio_vocab, fp); fclose(fp); }
        }
        n_past = n_cond;
        ggml_free(c);
    }

    // ---- AR loop ----------------------------------------------------------
    std::vector<float> work((size_t)n_audio_vocab);
    std::vector<int> gen;
    std::vector<int> teacher;   // MINICPMO_TTS_TEACHER (logits-gate only)
    for (int t = 0; t < max_new_token; t++) {
        memcpy(work.data(), logits.data(), sizeof(float) * n_audio_vocab);
        // logits /= temperature (greedy mode skips straight to the argmax)
        if (!greedy) {
            const float inv_t = 1.0f / temperature;
            for (auto & v : work) v *= inv_t;
            // repetition penalty over the last win_size generated tokens
            {
                const int lo = std::max(0, (int)gen.size() - win_size);
                std::vector<float> freq((size_t)n_audio_vocab, 0.0f);
                for (size_t i = lo; i < gen.size(); i++) freq[(size_t)gen[i]] += 1.0f;
                for (int i = 0; i < n_audio_vocab; i++) {
                    if (freq[(size_t)i] == 0.0f) continue;
                    const float alpha = std::pow(rep_pen, freq[(size_t)i]);
                    work[(size_t)i] = work[(size_t)i] < 0 ? work[(size_t)i] * alpha
                                                          : work[(size_t)i] / alpha;
                }
            }
            // top-p 0.85 (softmax cumsum over the sorted logits, keep >= 3)
            {
                std::vector<int> idx((size_t)n_audio_vocab);
                for (int i = 0; i < n_audio_vocab; i++) idx[(size_t)i] = i;
                std::sort(idx.begin(), idx.end(),
                          [&](int a, int b) { return work[(size_t)a] > work[(size_t)b]; });
                float mx = -1e30f;
                for (int i = 0; i < n_audio_vocab; i++)
                    mx = std::max(mx, work[(size_t)idx[(size_t)i]]);
                double sum = 0.0;
                std::vector<double> sp((size_t)n_audio_vocab);
                for (int i = 0; i < n_audio_vocab; i++) {
                    sp[(size_t)idx[(size_t)i]] = std::exp((double)work[(size_t)idx[(size_t)i]] - mx);
                    sum += sp[(size_t)idx[(size_t)i]];
                }
                double cum = 0.0;
                int keep = (int)n_audio_vocab;
                for (int i = 0; i < n_audio_vocab; i++) {
                    cum += sp[(size_t)idx[(size_t)i]] / sum;
                    if (cum <= (double)top_p) continue;
                    // cumulative_probs <= top_p removed: the first token whose
                    // running cumsum exceeds top_p stays (shifted set)
                    keep = i + 1;
                    break;
                }
                keep = std::max(keep, 3);
                for (int i = keep; i < n_audio_vocab; i++) work[(size_t)idx[(size_t)i]] = -INFINITY;
            }
            // top-k 25 (keep >= 3)
            {
                const int k = std::max(top_k, 3);
                std::vector<int> idx((size_t)n_audio_vocab);
                for (int i = 0; i < n_audio_vocab; i++) idx[(size_t)i] = i;
                std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(),
                                 [&](int a, int b) { return work[(size_t)a] > work[(size_t)b]; });
                const float th = work[(size_t)idx[(size_t)k - 1]];
                for (int i = 0; i < n_audio_vocab; i++)
                    if (work[(size_t)i] < th) work[(size_t)i] = -INFINITY;
            }
        }
        if (t < min_new_token) work[(size_t)eos] = -INFINITY;

        int next;
        if (greedy) {
            next = 0;
            for (int i = 1; i < n_audio_vocab; i++)
                if (work[(size_t)i] > work[(size_t)next]) next = i;
        } else {
            std::vector<float> probs((size_t)n_audio_vocab);
            double mx = -1e300;
            for (int i = 0; i < n_audio_vocab; i++)
                mx = std::max(mx, (double)work[(size_t)i]);
            double sum = 0.0;
            for (int i = 0; i < n_audio_vocab; i++) {
                probs[(size_t)i] = (float)std::exp((double)work[(size_t)i] - mx);
                sum += probs[(size_t)i];
            }
            for (auto & v : probs) v = (float)((double)v / sum);
            next = multinomial_one(probs.data(), n_audio_vocab, rng);
        }
        if (next == eos) break;
        gen.push_back(next);

        // step: emb_code(prev) at position n_cond + t - 1... the official
        // loop feeds the NEXT token embedding with position cond+t-1 for the
        // following step's forward: step t (t >= 1) consumes new_tokens[t-1]
        if (t + 1 >= max_new_token) break;
        {
            // MINICPMO_TTS_TEACHER=<codes.txt>: teacher-force the oracle code
            // history as the AR input (logits-gate alignment)
            if (const char * tf = getenv("MINICPMO_TTS_TEACHER")) {
                if (teacher.empty()) {
                    FILE * ft = fopen(tf, "r");
                    if (ft) {
                        int v2 = 0;
                        while (fscanf(ft, "%d", &v2) == 1) teacher.push_back(v2);
                        fclose(ft);
                    }
                }
            }
            struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, true };
            ggml_context * c = ggml_init(ip);
            if (!c) return false;
            ggml_cgraph * gf = ggml_new_graph_custom(c, 16384, false);
            const int T = 1;
            const int S = n_past + 1;
            ggml_tensor * code_in = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
            ggml_set_name(code_in, "t_code_in");
            ggml_set_input(code_in);
            ggml_tensor * pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
            ggml_set_name(pos, "t_pos");
            ggml_set_input(pos);
            ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, S, T);
            ggml_set_name(mask, "t_mask");
            ggml_set_input(mask);
            ggml_tensor * emb = ggml_get_rows(c, emb_code, code_in);
            ggml_tensor * x = tk_forward(c, gf, *this, emb, pos, mask, T, S);
            ggml_tensor * lg = ggml_mul_mat(c, head_code, x);
            ggml_set_name(lg, "t_logits");
            ggml_build_forward_expand(gf, lg);
            if (!ggml_gallocr_alloc_graph(allocr, gf)) {
                fprintf(stderr, "minicpmo tts: step alloc failed\n");
                ggml_free(c);
                return false;
            }
            const int feed = !teacher.empty() ? teacher[(size_t)t]
                                              : gen.back();
            ggml_backend_tensor_set(code_in, &feed, 0, sizeof(int));
            const int32_t p = n_cond + (t + 1) - 1;
            ggml_backend_tensor_set(pos, &p, 0, sizeof(int));
            vqa::set_causal_mask(mask, n_past, T);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "minicpmo tts: step compute failed\n");
                ggml_free(c);
                return false;
            }
            logits.resize((size_t)n_audio_vocab);
            ggml_backend_tensor_get(lg, logits.data(), 0,
                                    sizeof(float) * n_audio_vocab);
            if (const char * sd = getenv("MINICPMO_TTS_STEP_DUMP")) {
                char pth[512];
                snprintf(pth, sizeof(pth), "%s_step%d.f32", sd, t + 1);
                FILE * fp = fopen(pth, "wb");
                if (fp) { fwrite(logits.data(), 4, n_audio_vocab, fp); fclose(fp); }
            }
            ggml_free(c);
        }
        n_past += 1;
    }
    codes = std::move(gen);
    return true;
}

}  // namespace minicpmo
