// qwen3omni_audio.cpp - see qwen3omni_audio.hpp for the architecture contract.
#include "qwen3omni_audio.hpp"

#include "core/media/audio_feats.hpp"
#include "core/engine/gguf_util.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/op_profile.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qwen3omni {

static float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

bool AudioTower::encode(const float * pcm, int n_samples, std::vector<float> & feats,
                        int * n_tokens, const char * dump_dir) {
    *n_tokens = 0;
    if (n_samples <= 0 || mel_filters_host.empty()) return false;

    // ---- mel (whisper frontend, shared with qwen25omni) -------------------
    std::vector<float> mel;
    vqa::audio::log_mel_whisper(pcm, n_samples, 400, 160, n_mel,
                                mel_filters_host.data(), mel);
    vqa::audio::normalize_log_mel_whisper(mel);
    const int T = (int)mel.size() / n_mel;

    // ---- chunking: n_window*2 = 100-frame chunks --------------------------
    const int window2 = 2 * n_window;
    const int n_chunks = (T + window2 - 1) / window2;
    std::vector<int> clen(n_chunks);
    for (int c = 0; c < n_chunks; c++)
        clen[(size_t)c] = std::min(window2, T - c * window2);

    // host weight reads (weights live in the mmap / backend buffer; the model
    // passes host F32 copies it keeps alongside - see bind_audio in the model)
    // The model guarantees *_h host pointers exist; we read through them.
    // (declared here as members-less externs would be ugly - they ride along
    //  in this->h_* set by the caller before encode())
    const float *c1w = host_c1w.data(), *c1b = host_c1b.data(),
                *c2w = host_c2w.data(), *c2b = host_c2b.data(),
                *c3w = host_c3w.data(), *c3b = host_c3b.data(), *cow = host_cow.data();
    const int D = n_embd, NH = n_head, HD = D / NH;

    // ---- per-chunk conv stack + conv_out + sinusoid pos --------------------
    // after-cnn length: ((Lc + 1) / 2 + 1) / 2 + 1) / 2 with k3s2p1
    auto out_len = [](int L) { int a = (L + 1) / 2; a = (a + 1) / 2; return (a + 1) / 2; };
    int max_after = 0;
    for (int c = 0; c < n_chunks; c++) max_after = std::max(max_after, out_len(clen[(size_t)c]));

    std::vector<float> emb;                                 // [n_valid, 1280]
    std::vector<int> after(n_chunks);
    for (int c = 0; c < n_chunks; c++) {
        const int L = clen[(size_t)c];
        after[(size_t)c] = out_len(L);
        // stage A: [480, 128, L] from conv2d1 over [1, 128, L]
        const int H1 = 64, W1 = (L + 1) / 2;
        std::vector<float> s1((size_t)ds_hidden * H1 * W1);
        for (int co = 0; co < ds_hidden; co++) {
            for (int ho = 0; ho < H1; ho++) {
                for (int wo = 0; wo < W1; wo++) {
                    float acc = c1b ? c1b[(size_t)co] : 0.0f;
                    for (int ci = 0; ci < 1; ci++) {
                        for (int kh = 0; kh < 3; kh++) {
                            const int hi = ho * 2 + kh - 1;
                            if (hi < 0 || hi >= 128) continue;
                            for (int kw = 0; kw < 3; kw++) {
                                const int wi = wo * 2 + kw - 1;
                                if (wi < 0 || wi >= L) continue;
                                acc += c1w[((size_t)co * 1 + (size_t)ci) * 9 + (size_t)kh * 3 + kw]
                                       * mel[(size_t)hi * T + (size_t)(c * window2 + wi)];
                            }
                        }
                    }
                    s1[((size_t)co * H1 + (size_t)ho) * W1 + (size_t)wo] = gelu_erf(acc);
                }
            }
        }
        // stage B: [480, 64, W1] -> [480, 32, W2]
        const int H2 = 32, W2 = (W1 + 1) / 2;
        std::vector<float> s2((size_t)ds_hidden * H2 * W2);
        for (int co = 0; co < ds_hidden; co++) {
            for (int ho = 0; ho < H2; ho++) {
                for (int wo = 0; wo < W2; wo++) {
                    float acc = c2b ? c2b[(size_t)co] : 0.0f;
                    for (int ci = 0; ci < ds_hidden; ci++) {
                        for (int kh = 0; kh < 3; kh++) {
                            const int hi = ho * 2 + kh - 1;
                            if (hi < 0 || hi >= H1) continue;
                            for (int kw = 0; kw < 3; kw++) {
                                const int wi = wo * 2 + kw - 1;
                                if (wi < 0 || wi >= W1) continue;
                                acc += c2w[(((size_t)co * ds_hidden + (size_t)ci) * 9)
                                            + (size_t)kh * 3 + kw]
                                       * s1[((size_t)ci * H1 + (size_t)hi) * W1 + (size_t)wi];
                            }
                        }
                    }
                    s2[((size_t)co * H2 + (size_t)ho) * W2 + (size_t)wo] = gelu_erf(acc);
                }
            }
        }
        // stage C: [480, 32, W2] -> [480, 16, W3]
        const int H3 = 16, W3 = (W2 + 1) / 2;
        std::vector<float> s3((size_t)ds_hidden * H3 * W3);
        for (int co = 0; co < ds_hidden; co++) {
            for (int ho = 0; ho < H3; ho++) {
                for (int wo = 0; wo < W3; wo++) {
                    float acc = c3b ? c3b[(size_t)co] : 0.0f;
                    for (int ci = 0; ci < ds_hidden; ci++) {
                        for (int kh = 0; kh < 3; kh++) {
                            const int hi = ho * 2 + kh - 1;
                            if (hi < 0 || hi >= H2) continue;
                            for (int kw = 0; kw < 3; kw++) {
                                const int wi = wo * 2 + kw - 1;
                                if (wi < 0 || wi >= W2) continue;
                                acc += c3w[(((size_t)co * ds_hidden + (size_t)ci) * 9)
                                            + (size_t)kh * 3 + kw]
                                       * s2[((size_t)ci * H2 + (size_t)hi) * W2 + (size_t)wi];
                            }
                        }
                    }
                    s3[((size_t)co * H3 + (size_t)ho) * W3 + (size_t)wo] = gelu_erf(acc);
                }
            }
        }
        // conv_out: flatten (c, f) c-major per HF permute(0,3,1,2).view + Linear
        // HF: padded_embed [C=480, F=16, W3] -> permute to [W3, C, F] -> view
        // [W3, 480*16]; conv_out weight [1280, 480*16] (torch [out, in])
        const int n_pos = after[(size_t)c];
        for (int t = 0; t < n_pos; t++) {
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                const float * wr = cow + (size_t)d * (size_t)(ds_hidden * H3);
                for (int ci = 0; ci < ds_hidden; ci++)
                    for (int f = 0; f < H3; f++)
                        acc += wr[(size_t)ci * H3 + (size_t)f]
                               * s3[((size_t)ci * H3 + (size_t)f) * W3 + (size_t)t];
                emb.push_back(acc);
            }
        }
    }
    const int n_tok = (int)(emb.size() / D);
    // per-chunk sinusoid positions (sin|cos halves, restart at 0 per chunk)
    {
        std::vector<float> pos((size_t)n_tok * D);
        const int half = D / 2;
        const float log_inc = (float)std::log(10000.0) / (half - 1);
        int off = 0;
        for (int c = 0; c < n_chunks; c++) {
            const int n_pos = after[(size_t)c];
            for (int t = 0; t < n_pos; t++) {
                float * p = &pos[(size_t)(off + t) * D];
                for (int i = 0; i < half; i++) {
                    const float v = std::sin((float)t * std::exp(-(float)i * log_inc));
                    p[i] = v; p[half + i] = std::cos((float)t * std::exp(-(float)i * log_inc));
                }
            }
            off += n_pos;
        }
        for (size_t i = 0; i < emb.size(); i++) emb[i] += pos[i];
    }

    // ---- window mask: whole chunks grouped, up to 8 chunks per window ------
    const int ratio = std::max(1, n_window_infer / window2);   // 8
    const int win_after = max_after * ratio;
    std::vector<ggml_fp16_t> mask((size_t)n_tok * n_tok,
                                  ggml_fp32_to_fp16(-INFINITY));
    {
        int off = 0;
        while (off < n_tok) {
            const int w = std::min(win_after, n_tok - off);
            for (int r = 0; r < w; r++)
                for (int s = 0; s < w; s++)
                    mask[(size_t)(off + r) * n_tok + (size_t)(off + s)] = ggml_fp32_to_fp16(0.0f);
            off += w;
        }
    }

    // ---- graph: 32 pre-norm layers + ln_post + proj1/GELU/proj2 ------------
    free_graph();
    struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    actx = ggml_init(ip);
    if (!actx) return false;
    agf = ggml_new_graph_custom(actx, 32768, false);
    if (dump_dir && *dump_dir) {
        // raw mel as a graph-external file: shapes differ from the graph
        FILE * mf = nullptr;
        char mp[512];
        snprintf(mp, sizeof(mp), "%s/cpp_a_mel.bin", dump_dir);
        mf = fopen(mp, "wb");
        if (mf) {
            fwrite(mel.data(), 4, mel.size(), mf);
            fclose(mf);
            snprintf(mp, sizeof(mp), "%s/cpp_a_mel.shape", dump_dir);
            mf = fopen(mp, "w");
            if (mf) { fprintf(mf, "%d %d\n", n_mel, T); fclose(mf); }
        }
    }
    a_in = ggml_new_tensor_2d(actx, GGML_TYPE_F32, D, n_tok);
    ggml_set_name(a_in, "a_emb");
    a_mask = ggml_new_tensor_2d(actx, GGML_TYPE_F16, n_tok, n_tok);
    ggml_set_name(a_mask, "a_attn_mask");

    ggml_tensor * x = a_in;
    const float scale = 1.0f / sqrtf((float)HD);
    for (int il = 0; il < n_layer; il++) {
        const AudLayer & L = layers[(size_t)il];
        ggml_tensor * hh = ggml_norm(actx, x, ln_eps);
        hh = ggml_add(actx, ggml_mul(actx, hh, L.ln1_w), L.ln1_b);
        ggml_tensor * q = ggml_add(actx, ggml_mul_mat(actx, L.wq, hh), L.bq);
        ggml_tensor * k = ggml_add(actx, ggml_mul_mat(actx, L.wk, hh), L.bk);
        ggml_tensor * v = ggml_add(actx, ggml_mul_mat(actx, L.wv, hh), L.bv);
        if (il == 0) { ggml_set_name(q, "a_l0_q"); }
        ggml_tensor * q3 = ggml_reshape_3d(actx, q, HD, NH, n_tok);
        ggml_tensor * k3 = ggml_reshape_3d(actx, k, HD, NH, n_tok);
        ggml_tensor * v3 = ggml_reshape_3d(actx, v, HD, NH, n_tok);
        ggml_tensor * qf = ggml_cont(actx, ggml_permute(actx, q3, 0, 2, 1, 3));
        ggml_tensor * kf = ggml_cont(actx, ggml_permute(actx, k3, 0, 2, 1, 3));
        ggml_tensor * vf = ggml_cont(actx, ggml_permute(actx, v3, 0, 2, 1, 3));
        ggml_tensor * fa = ggml_flash_attn_ext(actx, qf, kf, vf, a_mask, scale, 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(actx, ggml_cont(actx, fa), D, n_tok);
        if (il == 0) ggml_set_name(attn, "a_l0_attn");
        attn = ggml_add(actx, ggml_mul_mat(actx, L.wo, attn), L.bo);
        x = ggml_add(actx, x, attn);

        hh = ggml_norm(actx, x, ln_eps);
        hh = ggml_add(actx, ggml_mul(actx, hh, L.ln2_w), L.ln2_b);
        ggml_tensor * g = ggml_add(actx, ggml_mul_mat(actx, L.fc1_w, hh), L.fc1_b);
        g = ggml_gelu_erf(actx, g);
        g = ggml_add(actx, ggml_mul_mat(actx, L.fc2_w, g), L.fc2_b);
        x = ggml_add(actx, x, g);
        if (il == 0) ggml_set_name(x, "a_l0_out");
    }
    ggml_set_name(x, "a_final");
    ggml_tensor * h2 = ggml_norm(actx, x, ln_eps);
    h2 = ggml_add(actx, ggml_mul(actx, h2, ln_post_w), ln_post_b);
    h2 = ggml_add(actx, ggml_mul_mat(actx, proj1_w, h2), proj1_b);
    h2 = ggml_gelu_erf(actx, h2);
    ggml_tensor * out = ggml_add(actx, ggml_mul_mat(actx, proj2_w, h2), proj2_b);
    ggml_set_name(out, "a_feats");
    a_feats = out;
    ggml_build_forward_expand(agf, out);

    if (dump_dir && *dump_dir) {
        static const char * const dn[] = { "a_emb", "a_l0_attn", "a_l0_out",
                                           "a_final", "a_feats", nullptr };
        vqa::dump::mark_sinks(actx, agf, dn);
        vqa::dump::finalize_sinks(actx, agf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, agf)) {
        fprintf(stderr, "audio graph allocation failed\n");
        return false;
    }
    ggml_backend_tensor_set(a_in, emb.data(), 0, sizeof(float) * emb.size());
    ggml_backend_tensor_set(a_mask, mask.data(), 0, sizeof(ggml_fp16_t) * mask.size());
    if (getenv("QOMNI_PROFILE")) {
        vqa::op_profile::State st;
        if (!vqa::op_profile::profile_graph(backend, agf, st, "audio", 0, 1))
            return false;
    } else if (ggml_backend_graph_compute(backend, agf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "audio compute failed\n");
        return false;
    }
    if (dump_dir && *dump_dir) {
        vqa::dump::write(agf, dump_dir, "a_emb");
        vqa::dump::write(agf, dump_dir, "a_l0_out");
        vqa::dump::write(agf, dump_dir, "a_final");
        vqa::dump::write(agf, dump_dir, "a_feats");
    }
    const int n = (int)a_feats->ne[1], d = (int)a_feats->ne[0];
    feats.resize((size_t)n * d);
    ggml_backend_tensor_get(a_feats, feats.data(), 0, sizeof(float) * feats.size());
    *n_tokens = n;
    fprintf(stderr, "audio: %d mel frames -> %d chunks -> %d tokens\n",
            T, n_chunks, n);
    cached_n_tok = n_tok;
    return true;
}

void AudioTower::free_graph() {
    if (actx) { ggml_free(actx); actx = nullptr; agf = nullptr; }
    cached_n_tok = -1;
}

}  // namespace qwen3omni
