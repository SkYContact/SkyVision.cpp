// codec_dsp.hpp - host-side DSP helpers shared by the codec (token2wav)
// stacks: the work-stealing parallel-for, the backend-aware host weight
// reader, the plain conv1d / transposed-conv1d loops, and the BigVGAN
// anti-aliased activation machinery (kaiser-sinc filter, aa snake).
//
// Pure code motion out of qwen25omni_t2w.cpp (the HF BigVGAN host oracle,
// which minicpmo's HiFT host DSP family also shares shapes with); loop
// order, accumulation order and the filter tables are load-bearing for the
// T2W parity gates - nothing here was "cleaned up". The minicpmo STFT/ISTFT
// and sine-source tables stay in minicpmo_vocoder.cpp (model contract).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace vqa {
namespace codec {

// minimal work-stealing parallel-for for the host-side vocoder loops; the
// per-index writes are disjoint in every call site (channels or positions)
template <typename F>
static void par_for(int n, F f) {
    unsigned nt = std::thread::hardware_concurrency();
    if (nt > 16) nt = 16;
    if (n <= 1 || nt <= 1) {
        for (int i = 0; i < n; i++) f(i);
        return;
    }
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (unsigned t = 0; t < nt; t++)
        pool.emplace_back([&] {
            int i;
            while ((i = next.fetch_add(1)) < n) f(i);
        });
    for (auto & th : pool) th.join();
}

// backend-aware host copy of a bound weight tensor (works on CPU and GPU)
inline void host_f32(ggml_tensor * t, std::vector<float> & out) {
    const size_t n = (size_t)ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> raw(n);
        ggml_backend_tensor_get(t, raw.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(raw[i]);
    }
}

// ---- host conv helpers (x laid out [C][T], weights torch [oc][ic][k]) -----

inline void conv1d_full(const float * x, int ic, int T, const float * w, const float * b,
                 int oc, int kw, int dil, int pad, float * y, int Ty) {
    par_for(oc, [&](int o) {
        float * yo = y + (size_t)o * Ty;
        for (int n = 0; n < Ty; n++) yo[n] = b ? b[(size_t)o] : 0.0f;
        for (int i = 0; i < ic; i++) {
            const float * xc = x + (size_t)i * T;
            const float * wo = w + ((size_t)o * ic + i) * kw;
            for (int kk = 0; kk < kw; kk++) {
                // torch conv1d is correlation: out[n] += w[k] * x[n + k - pad]
                const int off = kk * dil - pad;
                const float * wk = wo + kk;
                for (int n = 0; n < Ty; n++) {
                    const int idx = n + off;   // output stride is 1; dilation
                    if (idx >= 0 && idx < T)   // only spreads the kernel taps
                        yo[n] += wk[0] * xc[idx];
                }
            }
        }
    });
}

inline void conv_transpose1d_full(const float * x, int ic, int T, const float * w, const float * b,
                           int oc, int kw, int stride, int pad, float * y, int Ty) {
    std::fill(y, y + (size_t)oc * Ty, 0.0f);
    par_for(oc, [&](int o) {
        float * yrow = y + (size_t)o * Ty;
        for (int i = 0; i < ic; i++) {
            const float * xc = x + (size_t)i * T;
            const float * wo = w + ((size_t)i * oc) * kw;
            for (int n = 0; n < T; n++) {
                const float xv = xc[n];
                for (int kk = 0; kk < kw; kk++) {
                    const int idx = n * stride - pad + kk;
                    if (idx >= 0 && idx < Ty) yrow[idx] += wo[(size_t)o * kw + kk] * xv;
                }
            }
        }
    });
    if (b)
        for (int o = 0; o < oc; o++) {
            float * yo = y + (size_t)o * Ty;
            for (int n = 0; n < Ty; n++) yo[n] += b[o];
        }
}

inline float bessel_i0(float x) {
    float s = 1.0f, t = 1.0f;
    const float h = x * x / 4.0f;
    for (int m = 1; m < 40; m++) {
        t *= h / ((float)m * (float)m);
        s += t;
        if (t < 1e-10f * s) break;
    }
    return s;
}

// transformers' kaiser_sinc_filter1d (even kernels)
inline std::vector<float> kaiser_sinc_filter1d(int kernel, float cutoff, float half_width) {
    const int half = kernel / 2;
    const float delta_f = 4 * half_width;
    const float atten = 2.285f * (half - 1) * (float)M_PI * delta_f + 7.95f;
    float beta = 0.0f;
    if (atten > 50.0f) beta = 0.1102f * (atten - 8.7f);
    else if (atten >= 21.0f)
        beta = 0.5842f * powf(atten - 21.0f, 0.4f) + 0.07886f * (atten - 21.0f);
    const float i0b = bessel_i0(beta);
    std::vector<float> f((size_t)kernel);
    for (int n = 0; n < kernel; n++) {
        const float r = ((float)n - (float)(kernel - 1) / 2.0f) / ((float)(kernel - 1) / 2.0f);
        const float kwin = bessel_i0(beta * sqrtf(std::max(0.0f, 1.0f - r * r))) / i0b;
        const float t = (float)(n - half) + 0.5f;
        // torch.sinc(2*cutoff*t) * 2*cutoff, then normalized to sum 1 (HF)
        const float a = 2.0f * (float)M_PI * cutoff * t;
        const float v = (t == 0.0f) ? 1.0f : sinf(a) / a;
        f[(size_t)n] = 2.0f * cutoff * v * kwin;
    }
    double sum = 0.0;
    for (float v : f) sum += v;
    if (sum != 0.0)
        for (auto & v : f) v = (float)(v / sum);
    return f;
}

// anti-aliased activation resampler: replicate pad, depthwise filter, crop
// (UpSample1d: ratio*x transposed-conv; DownSample1d: strided conv)
inline void aa_resample(const float * x, int C, int T, const std::vector<float> & filter,
                 int kernel, int ratio, bool up, float * y, int Ty) {
    const int pad = kernel / ratio - 1;
    const int Tp = T + 2 * pad;
    std::vector<float> xp((size_t)C * Tp);
    for (int c = 0; c < C; c++) {
        const float * xc = x + (size_t)c * T;
        float * dst = xp.data() + (size_t)c * Tp;
        for (int i = 0; i < pad; i++) dst[i] = xc[0];
        memcpy(dst + pad, xc, (size_t)T * sizeof(float));
        for (int i = 0; i < pad; i++) dst[T + pad + i] = xc[T - 1];
    }
    std::fill(y, y + (size_t)C * Ty, 0.0f);
    if (up) {
        for (int c = 0; c < C; c++) {
            const float * xc = xp.data() + (size_t)c * Tp;
            float * yc = y + (size_t)c * Ty;
            for (int n = 0; n < Tp; n++) {
                const float xv = xc[n];
                for (int kk = 0; kk < kernel; kk++) {
                    const int idx = n * ratio + kk - (kernel - ratio) / 2;
                    if (idx >= 0 && idx < Ty) yc[idx] += ratio * filter[(size_t)kk] * xv;
                }
            }
        }
    } else {
        conv1d_full(xp.data(), C, Tp, filter.data(), nullptr, C, kernel, ratio, 0, y, Ty);
    }
}

// anti-aliased SnakeBeta: up2 (replicate pad 5, transposed-conv scatter,
// crop 15/16) -> snake -> down2 (replicate pad 5/6, strided conv), all with
// the 12-tap kaiser filter - exactly HF's AntiAliasedActivation1d
inline const std::vector<float> aa_filter = kaiser_sinc_filter1d(12, 0.25f, 0.3f);

inline void aa_snake_act(const float * x, int C, int T, const std::vector<float> & alpha,
                  const std::vector<float> & beta, const std::vector<float> & filter,
                  float * y) {
    const int K = (int)filter.size();          // 12
    const int pad = K / 2 - 1;                 // 5
    const int pad_l = pad * 2 + (K - 2) / 2;   // 15
    const int pad_r = pad * 2 + (K - 2 + 1) / 2;  // 16
    const int Tu = 2 * T - 1;
    std::vector<float> up((size_t)C * Tu);
    par_for(C, [&](int c) {   // C is small late in the stack; the inner loops
                              // are long enough that channel-level split works
        const float * xc = x + (size_t)c * T;
        float * yu = up.data() + (size_t)c * Tu;
        auto at = [&](int i) { return xc[i < 0 ? 0 : (i >= T ? T - 1 : i)]; };
        for (int n = -pad; n < T + pad; n++) {          // padded input positions
            const float xv = at(n);
            for (int kk = 0; kk < K; kk++) {
                const int idx = (n + pad) * 2 + kk;      // full transposed-conv index
                const int o = idx - pad_l;
                if (o >= 0 && o < Tu) yu[o] += 2.0f * filter[(size_t)kk] * xv;
            }
        }
        // snake on the upsampled grid
        const float a = expf(alpha[(size_t)c]);
        const float b = expf(beta[(size_t)c]) + 1e-9f;
        float * xu = up.data() + (size_t)c * Tu;
        for (int n = 0; n < Tu; n++) xu[n] += sinf(xu[n] * a) * sinf(xu[n] * a) / b;
        // down2: replicate pad left 5 / right 6, strided depthwise conv
        const int dl = K / 2 - 1, dr = K / 2;
        float * yc = y + (size_t)c * T;
        auto at2 = [&](int i) { return xu[i < 0 ? 0 : (i >= Tu ? Tu - 1 : i)]; };
        for (int n = 0; n < T; n++) {
            float sm = 0.0f;
            for (int kk = 0; kk < K; kk++) sm += filter[(size_t)kk] * at2(n * 2 - dl + kk);
            yc[n] = sm;
        }
    });
}

inline void snake_act(const float * x, int C, int T, const std::vector<float> & alpha,
               const std::vector<float> & beta, float * y) {
    for (int c = 0; c < C; c++) {
        const float a = expf(alpha[(size_t)c]);
        const float b = expf(beta[(size_t)c]) + 1e-9f;
        const float * xc = x + (size_t)c * T;
        float * yc = y + (size_t)c * T;
        for (int n = 0; n < T; n++) {
            const float s = sinf(xc[n] * a);
            yc[n] = xc[n] + s * s / b;
        }
    }
}

}  // namespace codec
}  // namespace vqa
