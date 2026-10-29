// minicpmo_audio.cpp - MiniCPM-o 4.5 audio tower: whisper-standard 80-mel log
// features (log10 -> max-8dB dynamic clamp -> (x+4)/4, processing_minicpmo.py
// MiniCPMAAudioProcessor), the whisper-medium encoder (24L x 1024, post-LN,
// k_proj without bias, 1s cumulative chunk attention) and the 2-layer
// projection + AvgPool1d(5) head. Numerics contract in minicpmo.hpp.
#include "minicpmo.hpp"

#include "core/media/audio_feats.hpp"
#include "core/engine/ggml_dump.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace minicpmo {

// ---------------------------------------------------------------- mel

bool AudioTower::load_mel_filters() {
    // whisper's HTK mel bank (transformers WhisperFeatureExtractor defaults:
    // 201 bins, 80 mel, 0-8000 Hz @16k, norm=None, htk). Sidecar first (the
    // qwen3omni mel_filters_128_201.f32 precedent), runtime second.
    if (!mel_filters_host.empty()) return true;
    mel_filters_host.assign((size_t)n_mel * 201, 0.0f);
    vqa::audio::mel_filterbank_htk(n_mel, 400, 0.0f, 8000.0f, 16000,
                                   mel_filters_host);
    return true;
}

bool AudioTower::mel(const float * pcm, int n_samples, std::vector<float> & out,
                     int * n_len) {
    if (mel_filters_host.empty()) load_mel_filters();
    vqa::audio::log_mel_whisper(pcm, n_samples, 400, 160, n_mel,
                                mel_filters_host.data(), out);
    // MiniCPMAAudioProcessor: dynamic max-8dB clamp (dynamic_log_norm default)
    vqa::audio::normalize_log_mel_whisper(out);
    *n_len = n_samples / 160;
    return *n_len > 0;
}

// ---------------------------------------------------------------- graph

bool AudioTower::encode_chunk(const float * mel80, int n_len,
                              std::vector<float> & feats, int * n_pool) {
    const int T = n_len;                            // mel frames
    const int T2 = (T + 1) / 2;                     // conv2 k3 s2 p1
    if (T2 > n_pos) {
        fprintf(stderr, "minicpmo audio: chunk %d frames > %d positions\n",
                T2, n_pos);
        return false;
    }
    if (!allocr) allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    const bool want_dump = dump_dir && dump_dir[0];

    ggml_context * c = nullptr;
    ggml_cgraph * gf = nullptr;
    {
        struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, true };
        c = ggml_init(ip);
        if (!c) return false;
        gf = ggml_new_graph_custom(c, 32768, false);
    }

    // input [T, n_mel]: element (t, m) at t + m*T - mel80 is [bin, time],
    // exactly this layout
    ggml_tensor * a_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, n_mel);
    ggml_set_name(a_in, "a_mel");
    ggml_set_input(a_in);

    // ggml conv_1d: kernel [K, Cin, Cout], data [T, Cin] channel-planar,
    // output [T_out, Cout]; whisper pads both convs by 1
    ggml_tensor * x = ggml_conv_1d(c, conv1_w, a_in, 1, 1, 1);
    x = ggml_cont(c, ggml_transpose(c, x));         // [d_model, T]
    x = ggml_add(c, x, conv1_b);
    x = ggml_gelu_erf(c, x);
    if (want_dump) ggml_set_name(x, "a_conv1");
    x = ggml_cont(c, ggml_transpose(c, x));         // [T, d_model]
    x = ggml_conv_1d(c, conv2_w, x, 2, 1, 1);
    x = ggml_cont(c, ggml_transpose(c, x));         // [d_model, T2]
    x = ggml_add(c, x, conv2_b);
    x = ggml_gelu_erf(c, x);
    {
        ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, T2);
        ggml_set_name(ids, "a_pos_ids");
        ggml_set_input(ids);
        x = ggml_add(c, x, ggml_get_rows(c, pos_embd, ids));
    }
    if (want_dump) ggml_set_name(x, "a_embed");

    ggml_tensor * mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, T2, T2);
    ggml_set_name(mask, "a_mask");
    ggml_set_input(mask);

    const int HD = d_model / n_head;
    for (int il = 0; il < (int)al.size(); il++) {
        const AudLayer & L = al[(size_t)il];
        ggml_tensor * h = ggml_norm(c, x, ln_eps);
        h = ggml_add(c, ggml_mul(c, h, L.attn_ln_w), L.attn_ln_b);
        ggml_tensor * q = ggml_mul_mat(c, L.wq, h);
        q = ggml_add(c, q, L.bq);
        ggml_tensor * k = ggml_mul_mat(c, L.wk, h);   // no bias (whisper)
        ggml_tensor * v = ggml_mul_mat(c, L.wv, h);
        v = ggml_add(c, v, L.bv);
        ggml_tensor * q3 = ggml_reshape_3d(c, q, HD, n_head, T2);
        ggml_tensor * k3 = ggml_reshape_3d(c, k, HD, n_head, T2);
        ggml_tensor * v3 = ggml_reshape_3d(c, v, HD, n_head, T2);
        ggml_tensor * qf = ggml_cont(c, ggml_permute(c, q3, 0, 2, 1, 3));
        ggml_tensor * kf = ggml_cont(c, ggml_permute(c, k3, 0, 2, 1, 3));
        ggml_tensor * vf = ggml_cont(c, ggml_permute(c, v3, 0, 2, 1, 3));
        ggml_tensor * fa = ggml_flash_attn_ext(c, qf, kf, vf, mask,
                                               1.0f / std::sqrt((float)HD),
                                               0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(c, ggml_cont(c, fa), d_model, T2);
        attn = ggml_mul_mat(c, L.wo, attn);
        attn = ggml_add(c, attn, L.bo);
        x = ggml_add(c, x, attn);
        if (want_dump && il == 0) ggml_set_name(x, "a_l0_attn_out");

        h = ggml_norm(c, x, ln_eps);
        h = ggml_add(c, ggml_mul(c, h, L.ffn_ln_w), L.ffn_ln_b);
        ggml_tensor * f = ggml_mul_mat(c, L.fc1_w, h);
        f = ggml_add(c, f, L.fc1_b);
        f = ggml_gelu_erf(c, f);
        f = ggml_mul_mat(c, L.fc2_w, f);
        f = ggml_add(c, f, L.fc2_b);
        x = ggml_add(c, x, f);
        if (want_dump && il == 0) ggml_set_name(x, "a_l0_out");
    }
    x = ggml_norm(c, x, ln_eps);
    x = ggml_mul(c, x, ln_post_w);
    x = ggml_add(c, x, ln_post_b);
    if (want_dump) ggml_set_name(x, "a_apm_out");

    // projector: fc1 -> ReLU -> fc2, then AvgPool1d(pool_step) over time
    ggml_tensor * p = ggml_mul_mat(c, proj_fc1, x);
    p = ggml_add(c, p, proj_fc1_b);
    p = ggml_relu(c, p);
    p = ggml_mul_mat(c, proj_fc2, p);
    p = ggml_add(c, p, proj_fc2_b);
    if (want_dump) ggml_set_name(p, "a_proj_out");

    const int npool = T2 >= pool_step ? (T2 - pool_step) / pool_step + 1 : 0;
    const int pdim = (int)p->ne[0];    // proj out dim (= llm dim 4096)
    ggml_tensor * pooled = nullptr;
    if (npool > 0) {
        // AvgPool1d(pool_step, stride pool_step) over time = matmul with the
        // block-averaging matrix [T2, npool] (the view+sum_rows formulation
        // mis-reads a strided src0 in this ggml - repro'd in the seg-test
        // pool probe)
        ggml_tensor * avg = ggml_new_tensor_2d(c, GGML_TYPE_F32, T2, npool);
        ggml_set_name(avg, "a_pool_avg");
        ggml_set_input(avg);
        pooled = ggml_mul_mat(c, avg, ggml_cont(c, ggml_transpose(c, p)));   // [npool, pdim]
        ggml_set_name(pooled, "a_pool_out");
    } else {
        pooled = p;
    }
    ggml_build_forward_expand(gf, pooled);

    if (want_dump) {
        static const char * const dn[] = {
            "a_conv1", "a_embed", "a_l0_attn_out", "a_l0_out",
            "a_apm_out", "a_proj_out", "a_pool_out", nullptr,
        };
        vqa::dump::mark_sinks(c, gf, dn);
        vqa::dump::finalize_sinks(c, gf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, gf)) {
        fprintf(stderr, "minicpmo audio: graph alloc failed\n");
        ggml_free(c);
        return false;
    }
    ggml_backend_tensor_set(a_in, mel80, 0, sizeof(float) * (size_t)n_mel * T);
    {
        std::vector<int32_t> ids((size_t)T2);
        for (int i = 0; i < T2; i++) ids[(size_t)i] = i;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "a_pos_ids"),
                                ids.data(), 0, sizeof(int) * T2);
        set_chunk_mask(mask, (int)(chunk_s * 50.0f));
        {   // block-averaging matrix: column j = 1/5 over rows [5j, 5j+5)
            std::vector<float> am((size_t)T2 * npool, 0.0f);
            for (int j = 0; j < npool; j++)
                for (int i = 0; i < pool_step; i++)
                    am[(size_t)(j * pool_step + i) + (size_t)T2 * j] =
                        1.0f / (float)pool_step;   // avg[t, j], ggml (t, j) order
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "a_pool_avg"),
                                    am.data(), 0, sizeof(float) * am.size());
        }
    }
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "minicpmo audio: compute failed\n");
        ggml_free(c);
        return false;
    }
    if (want_dump) {
        vqa::dump::write(gf, dump_dir, "a_conv1");
        vqa::dump::write(gf, dump_dir, "a_embed");
        vqa::dump::write(gf, dump_dir, "a_l0_attn_out");
        vqa::dump::write(gf, dump_dir, "a_l0_out");
        vqa::dump::write(gf, dump_dir, "a_apm_out");
        vqa::dump::write(gf, dump_dir, "a_proj_out");
        vqa::dump::write(gf, dump_dir, "a_pool_out");
    }
    feats.resize((size_t)pdim * npool);
    {   // pooled ne = [npool, pdim] token-fast: flat j + npool*d -> transpose
        std::vector<float> raw((size_t)npool * pdim);
        ggml_backend_tensor_get(pooled, raw.data(), 0,
                                sizeof(float) * raw.size());
        for (int j = 0; j < npool; j++)
            for (int d = 0; d < pdim; d++)
                feats[(size_t)j * pdim + d] = raw[(size_t)j + (size_t)npool * d];
    }
    if (dump_dir && dump_dir[0]) {
        // parity surface: the chunk's final pooled embeddings
        FILE * fp = fopen(std::string(std::string(dump_dir) +
                                      "/host_pool.bin").c_str(), "wb");
        if (fp) { fwrite(feats.data(), 4, feats.size(), fp); fclose(fp); }
    }
    ggml_free(c);
    *n_pool = npool;
    return true;
}

bool AudioTower::encode(const float * pcm, int n_samples,
                        std::vector<float> & feats, int * n_tokens) {
    feats.clear();
    *n_tokens = 0;
    // processing chunk_audio: <=30s stays whole, longer clips split every 30s
    const int max_samples = 30 * 16000;
    std::vector<float> all;
    int total = 0;
    for (int off = 0; off < n_samples; off += max_samples) {
        const int n = std::min(max_samples, n_samples - off);
        std::vector<float> mel80;
        int n_len = 0;
        if (!mel(pcm + off, n, mel80, &n_len)) {
            fprintf(stderr, "minicpmo audio: chunk %d too short\n", off);
            return false;
        }
        std::vector<float> chunk_feats;
        int n_pool = 0;
        if (!encode_chunk(mel80.data(), n_len, chunk_feats, &n_pool)) return false;
        all.insert(all.end(), chunk_feats.begin(), chunk_feats.end());
        total += n_pool;
    }
    feats = std::move(all);
    *n_tokens = total;
    return total > 0;
}

}  // namespace minicpmo
