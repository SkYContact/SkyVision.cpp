// qwen25omni_audio.cpp - Whisper-style audio tower of Qwen2.5-Omni.
//
// Mel preprocessing is whisper.cpp's FFT pipeline (log-mel, hop 160, n_fft 400,
// reflect pads); the filterbank itself comes from the GGUF (the converter dumps
// the exact table the HF feature extractor uses).
//
// The tower deviates from vanilla whisper exactly where transformers'
// Qwen2_5OmniAudioEncoder does:
//   * the mel is cut into n_window*2 = 200-frame chunks and each chunk is
//     convolved independently (zero padding at the chunk edges). Reproduced
//     here by inserting 4 zero mel columns between chunks and zeroing the two
//     conv1 output columns adjacent to every gap, so the strided conv2 taps
//     see exactly what the per-chunk zero padding produces;
//   * attention is chunk-local (block-diagonal mask, one block per chunk);
//   * the sinusoid position table restarts at 0 within each chunk;
//   * HF drops chunk-padding rows before attention (valid_indices) and pools
//     stride-2 pairs inside each chunk, dropping an odd trailing frame - here
//     the padding rows stay in the graph (self-attention only, finite values)
//     and are gathered away after the last layer.
//
// Column bookkeeping: chunk c starts at mel column 204c (200 + 4 gap), its
// conv2 outputs land at graph columns 102c + j (conv2 center 2j is even), and
// each 204-wide mel slot maps to 102 conv2 columns, of which the last
// 102 - n_out are gap garbage.
#include "qwen25omni_model.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qwen25omni {

namespace {

#define SIN_COS_N_COUNT 400

struct WhisperCache {
    float sin_vals[SIN_COS_N_COUNT];
    float cos_vals[SIN_COS_N_COUNT];
    float hann_window[SIN_COS_N_COUNT];

    WhisperCache() {
        for (int i = 0; i < SIN_COS_N_COUNT; i++) {
            const double theta = (2 * M_PI * i) / SIN_COS_N_COUNT;
            sin_vals[i] = sinf(theta);
            cos_vals[i] = cosf(theta);
            hann_window[i] = 0.5f * (1.0f - cosf((2.0f * M_PI * i) / SIN_COS_N_COUNT));
        }
    }
};

const WhisperCache & whisper_cache() {
    static WhisperCache c;
    return c;
}

// naive DFT (odd sizes), Cooley-Tukey otherwise - whisper.cpp's portable FFT
void fft(float * in, int N, float * out) {
    const WhisperCache & c = whisper_cache();
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }
    const int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        const int sin_cos_step = SIN_COS_N_COUNT / N;
        for (int k = 0; k < N; k++) {
            float re = 0, im = 0;
            for (int n = 0; n < N; n++) {
                const int idx = (k * n * sin_cos_step) % SIN_COS_N_COUNT;
                re += in[n] * c.cos_vals[idx];
                im -= in[n] * c.sin_vals[idx];
            }
            out[k * 2 + 0] = re;
            out[k * 2 + 1] = im;
        }
        return;
    }

    float * even = in + N;
    for (int i = 0; i < half_N; ++i) even[i] = in[2 * i];
    float * even_fft = out + 2 * N;
    fft(even, half_N, even_fft);

    float * odd = even;
    for (int i = 0; i < half_N; ++i) odd[i] = in[2 * i + 1];
    float * odd_fft = even_fft + N;
    fft(odd, half_N, odd_fft);

    const int sin_cos_step = SIN_COS_N_COUNT / N;
    for (int k = 0; k < half_N; k++) {
        const int idx = k * sin_cos_step;
        const float re = c.cos_vals[idx], im = -c.sin_vals[idx];
        const float re_odd = odd_fft[2 * k + 0], im_odd = odd_fft[2 * k + 1];
        out[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;
        out[2 * (k + half_N) + 0] = even_fft[2 * k + 0] - re * re_odd + im * im_odd;
        out[2 * (k + half_N) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
}

}  // namespace

// ---------------------------------------------------------------- mel
//
// transformers' WhisperFeatureExtractor: reflect-pad n_fft/2 samples at both
// ends, frame with the Hann window, power spectrum, mel bank, log10 clamped to
// [max - 8, max], then (x + 4) / 4. No 30 s zero padding - the Omni processor
// keeps the variable length (feature_attention_mask carries it).
MelSpec AudioTower::log_mel(const float * pcm, int n_samples) const {
    const WhisperCache & c = whisper_cache();
    const int n_len = 1 + n_samples / hop;              // center=True frames
    const int n_fft_bins = n_fft / 2 + 1;
    MelSpec mel;
    mel.n_mel = n_mel;
    mel.n_len = n_len;
    mel.data.assign((size_t)n_mel * n_len, 0.0f);

    const float * filt = mel_filters_host.empty()
                              ? (const float *)mel_filters->data
                              : mel_filters_host.data();

    std::vector<float> fft_in(n_fft * 2, 0.0f);
    std::vector<float> fft_out(n_fft * 2 * 2 * 2);
    std::vector<float> power(n_fft_bins);

    const int pad = n_fft / 2;
    auto sample_at = [&](int i) {
        if (n_samples == 0) return 0.0f;
        if (i < 0) i = std::min(-i, n_samples - 1);      // reflect at the start
        if (i >= n_samples) return 0.0f;                 // torch.stft zero-pads the tail
        return pcm[i];
    };

    for (int t = 0; t < n_len; t++) {
        const int off = t * hop - pad;
        for (int j = 0; j < n_fft; j++) fft_in[j] = c.hann_window[j] * sample_at(off + j);
        fft(fft_in.data(), n_fft, fft_out.data());
        for (int j = 0; j < n_fft_bins; j++)
            power[j] = fft_out[2 * j + 0] * fft_out[2 * j + 0] +
                       fft_out[2 * j + 1] * fft_out[2 * j + 1];
        for (int m = 0; m < n_mel; m++) {
            double sum = 0.0;
            int k = 0;
            for (; k < n_fft_bins - 3; k += 4)
                sum += power[k + 0] * filt[m * n_fft_bins + k + 0] +
                       power[k + 1] * filt[m * n_fft_bins + k + 1] +
                       power[k + 2] * filt[m * n_fft_bins + k + 2] +
                       power[k + 3] * filt[m * n_fft_bins + k + 3];
            for (; k < n_fft_bins; k++) sum += power[k] * filt[m * n_fft_bins + k];
            mel.data[(size_t)m * n_len + t] = log10(std::max(sum, 1e-10));
        }
    }

    double mmax = -1e20;
    for (size_t i = 0; i < mel.data.size(); i++) mmax = std::max(mmax, (double)mel.data[i]);
    mmax -= 8.0;
    for (size_t i = 0; i < mel.data.size(); i++) {
        mel.data[i] = (float)std::max((double)mel.data[i], mmax);
        mel.data[i] = (mel.data[i] + 4.0f) / 4.0f;
    }
    return mel;
}

// ---------------------------------------------------------------- geometry

// ---------------------------------------------------------------- graph

bool AudioTower::build_graph(const AudioGeometry & geo) {
    if (actx) { ggml_free(actx); actx = nullptr; }
    const int T_pad = geo.T_pad, T2 = geo.cols;

    struct ggml_init_params ip = { 256ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    actx = ggml_init(ip);
    if (!actx) return false;
    agf = ggml_new_graph_custom(actx, 32768, false);

    a_in = ggml_new_tensor_2d(actx, GGML_TYPE_F32, T_pad, n_mel);
    ggml_set_name(a_in, "a_mel");
    a_mask_col = ggml_new_tensor_2d(actx, GGML_TYPE_F32, 1, T_pad);
    ggml_set_name(a_mask_col, "a_mask_col");
    a_pos_idx = ggml_new_tensor_1d(actx, GGML_TYPE_I32, T2);
    ggml_set_name(a_pos_idx, "a_pos_idx");
    a_attn_mask = ggml_new_tensor_2d(actx, GGML_TYPE_F16, T2, T2);
    ggml_set_name(a_attn_mask, "a_attn_mask");
    a_valid_idx = ggml_new_tensor_1d(actx, GGML_TYPE_I32, geo.n_valid);
    ggml_set_name(a_valid_idx, "a_valid_idx");
    a_pool_idx = ggml_new_tensor_1d(actx, GGML_TYPE_I32, geo.n_pool);
    ggml_set_name(a_pool_idx, "a_pool_idx");
    a_poolr_idx = ggml_new_tensor_1d(actx, GGML_TYPE_I32, geo.n_pool);
    ggml_set_name(a_poolr_idx, "a_poolr_idx");

    // ggml conv_1d conventions: kernel [K, C_in, C_out] (torch bytes), data
    // [T, C_in] read channel-planar, output [T_out, C_out]. Biases and the
    // boundary mask broadcast along ne[0], so each conv is followed by a
    // transpose into [C, T] before the elementwise ops - conv_1d output bias
    // cannot repeat along ne[0] (ggml_can_repeat needs exact divisibility).
    ggml_tensor * x = ggml_conv_1d(actx, conv1_w, a_in, 1, 1, 1);
    x = ggml_cont(actx, ggml_transpose(actx, x));      // [n_embd, T_pad]
    x = ggml_gelu_erf(actx, ggml_add(actx, x, conv1_b));
    x = ggml_mul(actx, x, a_mask_col);                 // [1, T_pad] broadcasts
    ggml_set_name(x, "a_conv1");
    x = ggml_cont(actx, ggml_transpose(actx, x));      // back to [T_pad, n_embd]
    x = ggml_conv_1d(actx, conv2_w, x, 2, 1, 1);
    x = ggml_cont(actx, ggml_transpose(actx, x));      // [n_embd, T2]
    x = ggml_gelu_erf(actx, ggml_add(actx, x, conv2_b));
    ggml_set_name(x, "a_conv2");
    // per-chunk sinusoid positions: get_rows([n_pos, n_embd], ids) already
    // yields [n_embd, T2]
    x = ggml_add(actx, x, ggml_get_rows(actx, pos_embd, a_pos_idx));
    ggml_set_name(x, "a_embed");

    const float scale = 1.0f / sqrtf((float)(n_embd / n_head));
    for (int il = 0; il < n_layer; il++) {
        const AudLayer & L = layers[(size_t)il];
        ggml_tensor * h = ggml_norm(actx, x, ln_eps);   // LayerNorm, carries bias
        h = ggml_add(actx, ggml_mul(actx, h, L.ln1_w), L.ln1_b);

        ggml_tensor * q = ggml_mul_mat(actx, L.wq, h);
        q = ggml_add(actx, q, L.bq);
        ggml_tensor * k = ggml_mul_mat(actx, L.wk, h);
        ggml_tensor * v = ggml_mul_mat(actx, L.wv, h);
        v = ggml_add(actx, v, L.bv);
        if (il == 0) { ggml_set_name(q, "a_l0_q"); ggml_set_name(k, "a_l0_k"); }

        ggml_tensor * q3 = ggml_reshape_3d(actx, q, n_embd / n_head, n_head, T2);
        ggml_tensor * k3 = ggml_reshape_3d(actx, k, n_embd / n_head, n_head, T2);
        ggml_tensor * v3 = ggml_reshape_3d(actx, v, n_embd / n_head, n_head, T2);
        ggml_tensor * qf = ggml_cont(actx, ggml_permute(actx, q3, 0, 2, 1, 3));
        ggml_tensor * kf = ggml_cont(actx, ggml_permute(actx, k3, 0, 2, 1, 3));
        ggml_tensor * vf = ggml_cont(actx, ggml_permute(actx, v3, 0, 2, 1, 3));
        ggml_tensor * fa = ggml_flash_attn_ext(actx, qf, kf, vf, a_attn_mask, scale, 0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(actx, ggml_cont(actx, fa), n_embd, T2);
        if (il == 0) ggml_set_name(attn, "a_l0_attn");
        attn = ggml_mul_mat(actx, L.wo, attn);
        attn = ggml_add(actx, attn, L.bo);
        x = ggml_add(actx, x, attn);

        h = ggml_norm(actx, x, ln_eps);
        h = ggml_add(actx, ggml_mul(actx, h, L.ln2_w), L.ln2_b);
        ggml_tensor * g = ggml_mul_mat(actx, L.w_gate, h);
        g = ggml_add(actx, g, L.b_gate);
        h = ggml_gelu_erf(actx, g);
        h = ggml_mul_mat(actx, L.w_down, h);
        h = ggml_add(actx, h, L.b_down);
        x = ggml_add(actx, x, h);
        if (il == 0) ggml_set_name(x, "a_l0_out");
    }
    ggml_set_name(x, "a_final");

    // drop the padding columns (chunk-major gather), then pool stride-2 pairs
    x = ggml_get_rows(actx, x, a_valid_idx);
    ggml_set_name(x, "a_valid");
    ggml_tensor * left = ggml_get_rows(actx, x, a_pool_idx);
    ggml_tensor * right = ggml_get_rows(actx, x, a_poolr_idx);
    ggml_tensor * pooled = ggml_scale(actx, ggml_add(actx, left, right), 0.5f);
    ggml_set_name(pooled, "a_pooled");

    ggml_tensor * h = ggml_norm(actx, pooled, ln_eps);
    h = ggml_add(actx, ggml_mul(actx, h, ln_post_w), ln_post_b);
    ggml_tensor * out = ggml_mul_mat(actx, proj_w, h);
    out = ggml_add(actx, out, proj_b);
    ggml_set_name(out, "a_feats");
    a_feats = out;
    ggml_build_forward_expand(agf, out);

    if (!ggml_gallocr_alloc_graph(allocr, agf)) {
        fprintf(stderr, "audio graph allocation failed\n");
        return false;
    }
    return true;
}

bool AudioTower::encode(const MelSpec & mel, std::vector<float> & feats, int * n_tokens) {
    if (mel.n_len <= 0) return false;
    const int window2 = 2 * n_window;
    AudioGeometry geo(mel.n_len, window2);
    fprintf(stderr, "audio: chunks=%d cols=%d valid=%d\n", geo.n_chunks, geo.cols, geo.n_valid);
    if (!build_graph(geo)) return false;
    fprintf(stderr, "audio: graph built\n");

    // ---- inputs ----------------------------------------------------------
    std::vector<float> in((size_t)n_mel * geo.T_pad, 0.0f);
    std::vector<float> cm((size_t)geo.T_pad, 1.0f);
    std::vector<int> pos((size_t)geo.cols, 0);
    std::vector<ggml_fp16_t> mask((size_t)geo.cols * geo.cols, ggml_fp32_to_fp16(-INFINITY));
    std::vector<int> valid, pool, poolr;
    valid.reserve((size_t)geo.n_valid);
    pool.reserve((size_t)geo.n_pool);

    for (int c = 0; c < geo.n_chunks; c++) {
        const int ms = geo.mel_start(c), ss = geo.slot_start(c), no = geo.n_out[(size_t)c];

        // mel into the padded slot. ggml conv_1d reads the [T, n_mel] input
        // channel-planar (element (m, t) at flat m*T + t), so each bin is a
        // contiguous run: mel.data is already [bin, time]
        for (int m = 0; m < n_mel; m++)
            memcpy(&in[(size_t)m * geo.T_pad + ms],
                   &mel.data[(size_t)m * mel.n_len + c * window2],
                   sizeof(float) * geo.chunk_len[(size_t)c]);

        // conv1 output columns touching the zero gaps must read as zero
        if (ms - 1 >= 0) cm[(size_t)(ms - 1)] = 0.0f;
        if (ms + geo.chunk_len[(size_t)c] < geo.T_pad) cm[(size_t)(ms + geo.chunk_len[(size_t)c])] = 0.0f;

        for (int j = 0; j < no; j++) {
            pos[(size_t)(ss + j)] = j;                   // positions restart per chunk
            valid.push_back(ss + j);
        }
        // chunk-local attention; padding rows (outside every chunk) self-attend
        for (int r = 0; r < no; r++)
            for (int s = 0; s < no; s++)
                mask[(size_t)(ss + r) * geo.cols + (ss + s)] = ggml_fp32_to_fp16(0.0f);
    }
    std::vector<char> in_chunk((size_t)geo.cols, 0);
    for (int c = 0; c < geo.n_chunks; c++)
        for (int j = 0; j < geo.n_out[(size_t)c]; j++) in_chunk[(size_t)(geo.slot_start(c) + j)] = 1;
    for (int col = 0; col < geo.cols; col++)
        if (!in_chunk[(size_t)col]) mask[(size_t)col * geo.cols + col] = ggml_fp32_to_fp16(0.0f);
    {
        int off = 0;
        for (int c = 0; c < geo.n_chunks; c++) {
            const int np = AudioGeometry::pooled_pairs(geo.n_out[(size_t)c]);
            for (int j = 0; j < np; j++) {
                pool.push_back(off + 2 * j);
                poolr.push_back(off + 2 * j + 1);
            }
            off += geo.n_out[(size_t)c];
        }
    }

    ggml_backend_tensor_set(a_in, in.data(), 0, sizeof(float) * in.size());
    ggml_backend_tensor_set(a_mask_col, cm.data(), 0, sizeof(float) * cm.size());
    ggml_backend_tensor_set(a_pos_idx, pos.data(), 0, sizeof(int) * pos.size());
    ggml_backend_tensor_set(a_attn_mask, mask.data(), 0, sizeof(ggml_fp16_t) * mask.size());
    ggml_backend_tensor_set(a_valid_idx, valid.data(), 0, sizeof(int) * valid.size());
    ggml_backend_tensor_set(a_pool_idx, pool.data(), 0, sizeof(int) * pool.size());
    ggml_backend_tensor_set(a_poolr_idx, poolr.data(), 0, sizeof(int) * poolr.size());
    if (getenv("QOMNI_VERIFY_IDX")) {   // read back and verify the gather tables
        std::vector<int> chk((size_t)geo.n_valid);
        ggml_backend_tensor_get(a_valid_idx, chk.data(), 0, sizeof(int) * chk.size());
        fprintf(stderr, "audio: valid readback max=%d (expect %d)\n",
                *std::max_element(chk.begin(), chk.end()), valid[(size_t)geo.n_valid - 1]);
        chk.resize((size_t)geo.n_pool);
        ggml_backend_tensor_get(a_poolr_idx, chk.data(), 0, sizeof(int) * chk.size());
        fprintf(stderr, "audio: poolr readback max=%d (expect %d)\n",
                *std::max_element(chk.begin(), chk.end()), poolr[(size_t)geo.n_pool - 1]);
    }
    if (ggml_backend_graph_compute(backend, agf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "audio compute failed\n");
        return false;
    }
    const int n = (int)a_feats->ne[1], d = (int)a_feats->ne[0];
    feats.resize((size_t)n * d);
    ggml_backend_tensor_get(a_feats, feats.data(), 0, sizeof(float) * feats.size());
    *n_tokens = n;
    return true;
}

}  // namespace qwen25omni
