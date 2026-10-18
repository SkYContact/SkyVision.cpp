// qwen25omni_t2w.cpp - Token2Wav: flow-matching DiT (codec tokens -> mel)
// and the BigVGAN vocoder (mel -> PCM), both as static ggml graphs on the
// compute backend. The ECAPA speaker encoder stays host code (runs once per
// process). The original host BigVGAN implementation is kept as an oracle /
// fallback: T2W_HOST=1. Parity hooks: T2W_MEL_DUMP / T2W_MEL_LOAD feed a
// reference mel to the vocoder, T2W_PCM_DUMP saves the PCM.
//
// Semantics follow transformers' Qwen2_5OmniToken2Wav* exactly:
//   * one flow-matching trajectory over the WHOLE reply (no chunking): mel
//     length = codec tokens x repeats(2), RK4 over linspace(0,1,num_steps)
//     with the sway sampling correction and CFG guidance;
//   * DiT blocks use AdaLN-zero modulation, block-difference attention masks
//     (block 24; look-ahead on layer 10, look-back on layers 0 and 20) + RoPE;
//   * BigVGAN: mel -> exp -> dB(-115) - 20 -> normalize to [-1, 1], conv_pre,
//     6x [ConvTranspose1d + mean of 3 AMP res-blocks], snake, conv_post, clamp.
#include <chrono>
#include <thread>
#include <atomic>
#include <string>
#include <utility>
#include <vector>

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
#include "qwen25omni_talker.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qwen25omni {

namespace {

// backend-aware host copy of a bound weight tensor (works on CPU and GPU)
void host_vec(ggml_tensor * t, std::vector<float> & out) {
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

}  // namespace

namespace {

// ---- host conv helpers (x laid out [C][T], weights torch [oc][ic][k]) -----

void conv1d_full(const float * x, int ic, int T, const float * w, const float * b,
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

void conv_transpose1d_full(const float * x, int ic, int T, const float * w, const float * b,
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

float bessel_i0(float x) {
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
std::vector<float> kaiser_sinc_filter1d(int kernel, float cutoff, float half_width) {
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
void aa_resample(const float * x, int C, int T, const std::vector<float> & filter,
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
static const std::vector<float> aa_filter = kaiser_sinc_filter1d(12, 0.25f, 0.3f);

void aa_snake_act(const float * x, int C, int T, const std::vector<float> & alpha,
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

void snake_act(const float * x, int C, int T, const std::vector<float> & alpha,
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

// host GEMV for the small time-embedding MLPs: y[o] = b + W[o][:] . x
void linear_host(const std::vector<float> & w, const std::vector<float> & b, int in, int out,
                 const float * x, float * y) {
    for (int o = 0; o < out; o++) {
        float s = b.empty() ? 0.0f : b[(size_t)o];
        const float * wr = w.data() + (size_t)o * in;
        for (int i = 0; i < in; i++) s += wr[i] * x[i];
        y[o] = s;
    }
}

}  // namespace

// ---------------------------------------------------------------- loading

bool Token2Wav::load(ggml_backend_t backend_, struct ggml_context * wctx) {
    backend = backend_;
    wctx_ref = wctx;
    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(wctx, n);
        if (!t) fprintf(stderr, "t2w: missing tensor %s\n", n);
        return t;
    };
    time_mlp0_w = gt("d.time_mlp.0.weight"); time_mlp0_b = gt("d.time_mlp.0.bias");
    time_mlp1_w = gt("d.time_mlp.1.weight"); time_mlp1_b = gt("d.time_mlp.1.bias");
    codec_embd = gt("d.codec_embed.weight");
    in_proj_w = gt("d.in_proj.weight"); in_proj_b = gt("d.in_proj.bias");
    inv_freq = gt("d.inv_freq");
    norm_out_w = gt("d.norm_out.weight"); norm_out_b = gt("d.norm_out.bias");
    proj_out_w = gt("d.proj_out.weight"); proj_out_b = gt("d.proj_out.bias");
    layers.resize((size_t)d_depth);
    for (int i = 0; i < d_depth; i++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "d.blk.%d.%s", i, suf); return gt(b); };
        DiTLayer & L = layers[(size_t)i];
        L.attn_norm_w = g("attn_norm.weight"); L.attn_norm_b = g("attn_norm.bias");
        L.wq = g("wq.weight"); L.bq = g("bq");
        L.wk = g("wk.weight"); L.bk = g("bk");
        L.wv = g("wv.weight"); L.bv = g("bv");
        L.wo = g("wo.weight"); L.bo = g("bo");
        L.w_gate = g("w_gate.weight"); L.b_gate = g("b_gate");
        L.w_down = g("w_down.weight"); L.b_down = g("b_down");
        if (!L.wq || !L.wk || !L.wv || !L.wo || !L.w_gate || !L.w_down) return false;
    }
    spk_tdnn0_w = gt("d.spk.tdnn0.weight"); spk_tdnn0_b = gt("d.spk.tdnn0.bias");
    spk_blocks.resize(3);
    for (int bi = 0; bi < 3; bi++) {
        char b[96];
        auto g = [&](const char * suf) { snprintf(b, sizeof(b), "d.spk.blk.%d.%s", bi, suf); return gt(b); };
        SpkBlock & B = spk_blocks[(size_t)bi];
        B.tdnn1_w = g("tdnn1.weight"); B.tdnn1_b = g("tdnn1.bias");
        B.res2net_w.push_back(g("res2net.0.weight"));
        B.res2net_b.push_back(g("res2net.0.bias"));
        B.tdnn2_w = g("tdnn2.weight"); B.tdnn2_b = g("tdnn2.bias");
        B.se1_w = g("se1.weight"); B.se1_b = g("se1.bias");
        B.se2_w = g("se2.weight"); B.se2_b = g("se2.bias");
        if (!B.tdnn1_w || !B.res2net_w[0] || !B.tdnn2_w) return false;
    }
    spk_mfa_w = gt("d.spk.mfa.weight"); spk_mfa_b = gt("d.spk.mfa.bias");
    spk_asp_tdnn_w = gt("d.spk.asp.tdnn.weight"); spk_asp_tdnn_b = gt("d.spk.asp.tdnn.bias");
    spk_asp_conv_w = gt("d.spk.asp.conv.weight"); spk_asp_conv_b = gt("d.spk.asp.conv.bias");
    spk_fc_w = gt("d.spk.fc.weight"); spk_fc_b = gt("d.spk.fc.bias");

    // BigVGAN host weights
    auto gf = [&](const char * n, std::vector<float> & out) {
        ggml_tensor * t = gt(n);
        if (!t) return false;
        out.resize((size_t)ggml_nelements(t));
        host_vec(t, out);
        return true;
    };
    gf("bg.conv_pre.weight", bg_pre_w);
    gf("bg.conv_pre.bias", bg_pre_b);
    gf("bg.snake_pre.alpha", snake_pre_a);
    gf("bg.snake_pre.beta", snake_pre_b);
    gf("bg.conv_post.weight", post_w);
    const int n_up = 6;
    ups.resize((size_t)n_up);
    const int up_k[6] = {11, 7, 4, 4, 4, 4}, up_s[6] = {5, 3, 2, 2, 2, 2};
    for (int u = 0; u < n_up; u++) {
        char b[64];
        snprintf(b, sizeof(b), "bg.up.%d.weight", u);
        ggml_tensor * t = gt(b);
        ups[(size_t)u].w.resize((size_t)ggml_nelements(t));
        host_vec(t, ups[(size_t)u].w);
        snprintf(b, sizeof(b), "bg.up.%d.bias", u);
        gf(b, ups[(size_t)u].b);
        ups[(size_t)u].k = up_k[u];
        ups[(size_t)u].stride = up_s[u];
        ups[(size_t)u].oc = b_ch >> (u + 1);
    }
    const int n_rb = 18;
    rbs.resize((size_t)n_rb);
    for (int r = 0; r < n_rb; r++) {
        ResBlock & R = rbs[(size_t)r];
        for (int kk = 0; kk < 3; kk++) {
            char b[64];
            snprintf(b, sizeof(b), "bg.rb.%d.c1.%d.weight", r, kk);
            {
                ggml_tensor * t = gt(b);
                Conv1D c;
                c.w.resize((size_t)ggml_nelements(t));
                host_vec(t, c.w);
                snprintf(b, sizeof(b), "bg.rb.%d.c1.%d.bias", r, kk);
                gf(b, c.b);
                R.c1.push_back(std::move(c));
            }
            snprintf(b, sizeof(b), "bg.rb.%d.c2.%d.weight", r, kk);
            {
                ggml_tensor * t = gt(b);
                Conv1D c;
                c.w.resize((size_t)ggml_nelements(t));
                host_vec(t, c.w);
                snprintf(b, sizeof(b), "bg.rb.%d.c2.%d.bias", r, kk);
                gf(b, c.b);
                R.c2.push_back(std::move(c));
            }
            snprintf(b, sizeof(b), "bg.rb.%d.s1.%d.alpha", r, kk);
            gf(b, R.s1a.emplace_back());
            snprintf(b, sizeof(b), "bg.rb.%d.s1.%d.beta", r, kk);
            gf(b, R.s1b.emplace_back());
            snprintf(b, sizeof(b), "bg.rb.%d.s2.%d.alpha", r, kk);
            gf(b, R.s2a.emplace_back());
            snprintf(b, sizeof(b), "bg.rb.%d.s2.%d.beta", r, kk);
            gf(b, R.s2b.emplace_back());
        }
    }
    if (getenv("T2W_DEBUG")) {
        std::vector<float> chk((size_t)ggml_nelements(in_proj_w));
        vqa::gguf::tensor_to_f32(in_proj_w, chk.data());
        bool nan = false;
        for (float v : chk) if (std::isnan(v)) nan = true;
        fprintf(stderr, "t2w: in_proj_w %zu floats nan=%d\n", chk.size(), (int)nan);
    }
    fprintf(stderr, "t2w: loaded (dit %dL d=%d, bigvgan %d ups)\n", d_depth, d_dim, n_up);
    return true;
}


// ---------------------------------------------------------------- ECAPA

// speaker embedding of a reference mel [80 x T]; runs once per turn
bool Token2Wav::ecapa_embed(const float * mel, int T, std::vector<float> & out) {
    std::vector<float> w0, b0;
    host_vec(spk_tdnn0_w, w0); host_vec(spk_tdnn0_b, b0);
    std::vector<float> cur((size_t)80 * T);
    memcpy(cur.data(), mel, (size_t)80 * T * sizeof(float));
    int cc = 80;
    auto conv = [&](std::vector<float> & x, int ic, const float * wt, const std::vector<float> & bt,
                    int oc, int kw, int dil) {
        const int pad = (kw * dil - dil) / 2;
        std::vector<float> y((size_t)oc * T);
        // bt may live in a temp; decode biases to f32 once per call site
        conv1d_full(x.data(), ic, T, wt, bt.empty() ? nullptr : bt.data(), oc, kw, dil, pad,
                    y.data(), T);
        x.swap(y);
        cc = oc;
    };
    auto bias_vec = [&](ggml_tensor * t) {
        std::vector<float> v;
        host_vec(t, v);
        return v;
    };
    conv(cur, cc, w0.data(), bias_vec(spk_tdnn0_b), 256,
        (int)(w0.size() / (256u * 80u)), 1);
    if (getenv("T2W_DEBUG")) {
        float mnm = cur[0], mxm = cur[0]; bool nan = false;
        for (float v : cur) { if (std::isnan(v)) nan = true; if (v < mnm) mnm = v; if (v > mxm) mxm = v; }
        float melmn = mel[0], melmx = mel[0];
        for (int i = 0; i < 80 * T; i++) { if (mel[i] < melmn) melmn = mel[i]; if (mel[i] > melmx) melmx = mel[i]; }
        fprintf(stderr, "t2w: mel [%.3f..%.3f]; after tdnn0 nan=%d [%.2f..%.2f]\n",
                melmn, melmx, (int)nan, mnm, mxm);
    }

    std::vector<std::vector<float>> block_outs;
    for (int bi = 0; bi < 3; bi++) {
        const SpkBlock & B = spk_blocks[(size_t)bi];
        std::vector<float> w1, b1, w2, b2, wr, br, se1w, se1b, se2w, se2b;
        host_vec(B.tdnn1_w, w1); host_vec(B.tdnn1_b, b1);
        host_vec(B.tdnn2_w, w2); host_vec(B.tdnn2_b, b2);
        host_vec(B.res2net_w[0], wr); host_vec(B.res2net_b[0], br);
        host_vec(B.se1_w, se1w); host_vec(B.se1_b, se1b);
        host_vec(B.se2_w, se2w); host_vec(B.se2_b, se2b);
        const int dil = bi + 2;   // enc_dilations 1,2,3,4 for blocks 0..3
        // kernel widths from the actual checkpoint (this CampPlus variant uses
        // 1x1 convs in the SE-Res2Net blocks despite the config table)
        const int kw1 = (int)(w1.size() / (256u * 256u));
        const int kw2 = (int)(w2.size() / (256u * 256u));
        std::vector<float> residual = cur;
        auto mx = [](const std::vector<float> & v) {
            float m = 0; for (float a : v) m = std::max(m, fabsf(a)); return m;
        };
        if (getenv("T2W_DEBUG") && bi == 0) fprintf(stderr, "t2w: blk0 in |x|=%.2f\n", mx(cur));
        conv(cur, cc, w1.data(), b1, 256, kw1, dil);
        if (getenv("T2W_DEBUG") && bi == 0) fprintf(stderr, "t2w: blk0 tdnn1 |x|=%.2f\n", mx(cur));
        // res2net scale=2: branch0 identity, branch1 conv, concat on channels
        const int half = cc / 2;
        std::vector<float> y((size_t)cc * T);
        memcpy(y.data(), cur.data(), (size_t)half * T * sizeof(float));
        {
            std::vector<float> branch((size_t)half * T);
            conv1d_full(cur.data() + (size_t)half * T, half, T, wr.data(), br.data(),
                        half, 1, 1, 0, branch.data(), T);
            memcpy(y.data() + (size_t)half * T, branch.data(), (size_t)half * T * sizeof(float));
        }
        cur.swap(y);
        conv(cur, cc, w2.data(), b2, 256, kw2, dil);
        // squeeze-excite: global mean -> se1 -> relu -> se2 -> sigmoid -> scale
        std::vector<float> se1((size_t)64), se2((size_t)cc);
        {
            std::vector<float> mean(cc, 0.0f);
            for (int c = 0; c < cc; c++) {
                const float * xc = cur.data() + (size_t)c * T;
                for (int n = 0; n < T; n++) mean[(size_t)c] += xc[n];
                mean[(size_t)c] /= (float)T;
            }
            linear_host(se1w, se1b, cc, 64, mean.data(), se1.data());
            for (auto & v : se1) v = std::max(0.0f, v);
            linear_host(se2w, se2b, 64, cc, se1.data(), se2.data());
            for (auto & v : se2) v = 1.0f / (1.0f + expf(-v));
            for (int c = 0; c < cc; c++) {
                float * xc = cur.data() + (size_t)c * T;
                for (int n = 0; n < T; n++) xc[n] *= se2[(size_t)c];
            }
        }
        if (getenv("T2W_DEBUG") && bi == 0) fprintf(stderr, "t2w: blk0 se |x|=%.2f (resid %.2f)\n", mx(cur), mx(residual));
        for (size_t i = 0; i < cur.size(); i++) cur[i] += residual[i];
        if (getenv("T2W_DEBUG")) {
            bool nan = false;
            for (float v : cur) if (std::isnan(v)) nan = true;
            fprintf(stderr, "t2w: ecapa block %d nan=%d\n", bi, (int)nan);
        }
        block_outs.push_back(cur);
    }
    // MFA: concat blocks 1..3 -> conv k1
    std::vector<float> mfa_in((size_t)(cc * 3) * T);
    for (int bi = 0; bi < 3; bi++)
        memcpy(mfa_in.data() + (size_t)bi * cc * T, block_outs[(size_t)bi].data(),
               (size_t)cc * T * sizeof(float));
    std::vector<float> wmfa, bmfa;
    host_vec(spk_mfa_w, wmfa); host_vec(spk_mfa_b, bmfa);
    conv(mfa_in, cc * 3, wmfa.data(), bmfa, 768, 1, 1);
    if (getenv("T2W_DEBUG")) {
        bool nan = false;
        for (float v : mfa_in) if (std::isnan(v)) nan = true;
        fprintf(stderr, "t2w: mfa_in nan=%d\n", (int)nan);
    }
    // attentive statistics pooling
    std::vector<float> wattn, battn, wconv, bconv, wfc, bfc;
    host_vec(spk_asp_tdnn_w, wattn); host_vec(spk_asp_tdnn_b, battn);
    host_vec(spk_asp_conv_w, wconv); host_vec(spk_asp_conv_b, bconv);
    host_vec(spk_fc_w, wfc); host_vec(spk_fc_b, bfc);
    const int ac = 64;   // enc_attention_channels
    std::vector<float> att((size_t)768 * T);
    {
        std::vector<float> t1((size_t)ac * T);
        conv1d_full(mfa_in.data(), 768, T, wattn.data(), battn.data(), ac, 1, 1, 0,
                    t1.data(), T);
        for (auto & v : t1) v = tanhf(v);
        conv1d_full(t1.data(), ac, T, wconv.data(), bconv.data(), 768, 1, 1, 0, att.data(), T);
        for (int c = 0; c < 768; c++) {
            float * arow = att.data() + (size_t)c * T;
            float mx = -INFINITY;
            for (int n = 0; n < T; n++) mx = std::max(mx, arow[n]);
            float sum = 0.0f;
            for (int n = 0; n < T; n++) { arow[n] = expf(arow[n] - mx); sum += arow[n]; }
            for (int n = 0; n < T; n++) arow[n] /= sum;
        }
    }
    std::vector<float> mean(768, 0.0f), var(768, 0.0f);
    for (int c = 0; c < 768; c++) {
        const float * xc = mfa_in.data() + (size_t)c * T;
        const float * arow = att.data() + (size_t)c * T;
        for (int n = 0; n < T; n++) {
            mean[(size_t)c] += arow[n] * xc[n];
            var[(size_t)c] += arow[n] * xc[n] * xc[n];
        }
        var[(size_t)c] = sqrtf(std::max(1e-12f, var[(size_t)c] - mean[(size_t)c] * mean[(size_t)c]));
    }
    std::vector<float> pooled((size_t)768 * 2);
    for (int c = 0; c < 768; c++) {
        pooled[(size_t)c] = mean[(size_t)c];
        pooled[(size_t)768 + c] = var[(size_t)c];
    }
    out.resize((size_t)enc_dim);
    linear_host(wfc, bfc, 768 * 2, enc_dim, pooled.data(), out.data());
    return true;
}

// ---------------------------------------------------------------- DiT graph

bool Token2Wav::build_dit_graph(int T) {
    if (gctx) { ggml_free(gctx); gctx = nullptr; }
    g_masks.clear();   // stale pointers from the freed ctx
    g_sinks.clear();
    // The per-run-constant inputs (masks / pos / codec / spk / cond) live in
    // their own context + backend buffer, OUTSIDE the gallocr pool: gallocr
    // shares memory between tensors with non-overlapping liveness inside one
    // graph, so a mid-graph input (a per-layer mask) gets clobbered by later
    // intermediates before the next eval reads it. A dedicated buffer never
    // moves and is uploaded exactly once per (re)build.
    if (ibuf) { ggml_backend_buffer_free(ibuf); ibuf = nullptr; }
    if (ictx) { ggml_free(ictx); ictx = nullptr; }
    struct ggml_init_params iip = { 64ull * 1024 * 1024, NULL, true };
    ictx = ggml_init(iip);
    if (!ictx) return false;
    struct ggml_init_params ip = { 128ull * 1024 * 1024, NULL, true };
    gctx = ggml_init(ip);
    if (!gctx) return false;
    ggf = ggml_new_graph_custom(gctx, 16384, false);

    g_x_in = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, d_mel, T);
    ggml_set_name(g_x_in, "dit_x");
    g_cond_in = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, enc_dim, T);      // ECAPA out
    ggml_set_name(g_cond_in, "dit_cond");
    g_code_emb = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, d_emb, T);
    ggml_set_name(g_code_emb, "dit_code");
    g_spk_in = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, enc_emb_dim, T);   // baked cond
    ggml_set_name(g_spk_in, "dit_spk");
    g_t_in = ggml_new_tensor_1d(ictx, GGML_TYPE_F32, d_dim);
    ggml_set_name(g_t_in, "dit_t");

    // input projection: [x | ecapa(cond) | code | speaker] -> d_dim
    ggml_tensor * cat = ggml_concat(gctx, g_x_in, g_cond_in, 0);
    cat = ggml_concat(gctx, cat, g_code_emb, 0);
    cat = ggml_concat(gctx, cat, g_spk_in, 0);
    fprintf(stderr, "dit: mm in_proj cat=[%lld,%lld] w=[%lld,%lld]\n", (long long)cat->ne[0], (long long)cat->ne[1], (long long)in_proj_w->ne[0], (long long)in_proj_w->ne[1]);
    ggml_tensor * x = ggml_mul_mat(gctx, in_proj_w, cat);
    x = ggml_add(gctx, x, in_proj_b);
    ggml_set_name(x, "dit_embed");
    if (getenv("T2W_DEBUG")) {
        ggml_tensor * snk = ggml_dup(gctx, cat);
        ggml_build_forward_expand(ggf, snk);
        g_sinks.push_back(snk);   // sink 0: the concatenated input
        ggml_tensor * snk2 = ggml_dup(gctx, x);
        ggml_build_forward_expand(ggf, snk2);
        g_sinks.push_back(snk2);  // sink 1: input projection
    }

    g_pos = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, T);
    ggml_set_name(g_pos, "dit_pos");
    const int HD = d_hd, NH = d_head;
    for (int il = 0; il < d_depth; il++) {
        const DiTLayer & L = layers[(size_t)il];
        const bool ahead = std::find(look_ahead_layers.begin(), look_ahead_layers.end(), il)
                               != look_ahead_layers.end();
        const bool behind = std::find(look_backward_layers.begin(), look_backward_layers.end(), il)
                                != look_backward_layers.end();
        ggml_tensor * amask = ggml_new_tensor_2d(ictx, GGML_TYPE_F16, T, T);
        ggml_set_name(amask, "dit_mask");
        g_masks.push_back(amask);

        // AdaLN-zero modulation vectors from the (host-computed) time embedding
        if (il == 0) fprintf(stderr, "dit: mm mod w=[%lld,%lld]\n", (long long)L.attn_norm_w->ne[0], (long long)L.attn_norm_w->ne[1]);
        ggml_tensor * mod = ggml_mul_mat(gctx, L.attn_norm_w, g_t_in);
        mod = ggml_add(gctx, mod, L.attn_norm_b);          // [6*dim]
        const int64_t row = mod->nb[1];
        ggml_tensor * shift = ggml_view_2d(gctx, mod, d_dim, 1, row, 0);
        ggml_tensor * scale = ggml_view_2d(gctx, mod, d_dim, 1, row, (size_t)d_dim * sizeof(float));
        ggml_tensor * gate = ggml_view_2d(gctx, mod, d_dim, 1, row, (size_t)2 * d_dim * sizeof(float));
        ggml_tensor * shift_m = ggml_view_2d(gctx, mod, d_dim, 1, row, (size_t)3 * d_dim * sizeof(float));
        ggml_tensor * scale_m = ggml_view_2d(gctx, mod, d_dim, 1, row, (size_t)4 * d_dim * sizeof(float));
        ggml_tensor * gate_m = ggml_view_2d(gctx, mod, d_dim, 1, row, (size_t)5 * d_dim * sizeof(float));

        ggml_tensor * h = ggml_norm(gctx, x, 1e-6f);
        // norm * (1 + scale) + shift  ==  norm + norm*scale + shift
        h = ggml_add(gctx, ggml_add(gctx, h, ggml_mul(gctx, h, scale)), shift);
        ggml_tensor * q = ggml_mul_mat(gctx, L.wq, h);
        q = ggml_add(gctx, q, L.bq);
        ggml_tensor * k = ggml_mul_mat(gctx, L.wk, h);
        k = ggml_add(gctx, k, L.bk);
        ggml_tensor * v = ggml_mul_mat(gctx, L.wv, h);
        v = ggml_add(gctx, v, L.bv);
        ggml_tensor * q3 = ggml_rope_ext(gctx, ggml_reshape_3d(gctx, q, HD, NH, T), g_pos, nullptr,
                                         HD, 0, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * k3 = ggml_rope_ext(gctx, ggml_reshape_3d(gctx, k, HD, NH, T), g_pos, nullptr,
                                         HD, 0, 8192, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * qf = ggml_cont(gctx, ggml_permute(gctx, q3, 0, 2, 1, 3));
        ggml_tensor * kf = ggml_cont(gctx, ggml_permute(gctx, k3, 0, 2, 1, 3));
        ggml_tensor * vf = ggml_cont(gctx, ggml_permute(gctx, ggml_reshape_3d(gctx, v, HD, NH, T),
                                                        0, 2, 1, 3));
        ggml_tensor * fa = ggml_flash_attn_ext(gctx, qf, kf, vf, amask, 1.0f / sqrtf((float)HD),
                                               0.0f, 0.0f);
        ggml_tensor * attn = ggml_reshape_2d(gctx, ggml_cont(gctx, fa), NH * HD, T);
        attn = ggml_mul_mat(gctx, L.wo, attn);
        attn = ggml_add(gctx, attn, L.bo);
        x = ggml_add(gctx, x, ggml_mul(gctx, attn, gate));
        if (getenv("T2W_DEBUG") && il == 0) {
            ggml_tensor * snk = ggml_dup(gctx, x);
            ggml_build_forward_expand(ggf, snk);
            g_sinks.push_back(snk);   // sink 1: after first attention
        }

        ggml_tensor * h2 = ggml_norm(gctx, x, 1e-6f);
        h2 = ggml_add(gctx, ggml_add(gctx, h2, ggml_mul(gctx, h2, scale_m)), shift_m);
        ggml_tensor * ff = ggml_mul_mat(gctx, L.w_gate, h2);
        ff = ggml_add(gctx, ff, L.b_gate);
        ff = ggml_gelu(gctx, ff);                    // GELU(tanh) like HF
        ff = ggml_mul_mat(gctx, L.w_down, ff);
        ff = ggml_add(gctx, ff, L.b_down);
        x = ggml_add(gctx, x, ggml_mul(gctx, ff, gate_m));
        if (getenv("T2W_DEBUG") && il < 3) {
            ggml_tensor * snk = ggml_dup(gctx, x);
            ggml_set_name(snk, "dit_snk");
            ggml_build_forward_expand(ggf, snk);
            g_sinks.push_back(snk);
        }
    }
    // final adaLN + projection
    ggml_tensor * mod = ggml_mul_mat(gctx, norm_out_w, g_t_in);
    mod = ggml_add(gctx, mod, norm_out_b);
    ggml_tensor * shift = ggml_view_2d(gctx, mod, d_dim, 1, mod->nb[1], 0);
    ggml_tensor * scale = ggml_view_2d(gctx, mod, d_dim, 1, mod->nb[1],
                                       (size_t)d_dim * sizeof(float));
    ggml_tensor * h = ggml_norm(gctx, x, 1e-6f);
    h = ggml_add(gctx, ggml_add(gctx, h, ggml_mul(gctx, h, scale)), shift);
    ggml_tensor * out = ggml_mul_mat(gctx, proj_out_w, h);
    out = ggml_add(gctx, out, proj_out_b);
    ggml_set_name(out, "dit_out");
    g_out = out;
    ggml_build_forward_expand(ggf, out);
    ibuf = ggml_backend_alloc_ctx_tensors(ictx, backend);
    if (!ibuf) {
        fprintf(stderr, "t2w: input tensor alloc failed\n");
        return false;
    }
    if (!ggml_gallocr_alloc_graph(allocr, ggf)) {
        fprintf(stderr, "t2w: dit graph alloc failed\n");
        return false;
    }
    return true;
}

bool Token2Wav::dit_eval(const float * x, int T, float t, const float * codec_emb, bool drop_code,
                         const float * spk, const float * cond_vec, std::vector<float> & out) {  // spk: baked cond (192), cond_vec: ECAPA (128)
    // host time embedding: sinus(256) -> mlp0 -> silu -> mlp1 -> [d_dim]
    static std::vector<float> w0, b0, w1, b1;
    if (w0.empty()) { host_vec(time_mlp0_w, w0); host_vec(time_mlp0_b, b0);
                      host_vec(time_mlp1_w, w1); host_vec(time_mlp1_b, b1); }
    std::vector<float> te((size_t)256), th((size_t)d_dim);
    for (int i = 0; i < 128; i++) {
        const float f = expf(-logf(10000.0f) * (float)i / 127.0f);
        te[(size_t)i] = sinf(1000.0f * t * f);   // SinusPositionEmbedding scale=1000
        te[(size_t)128 + i] = cosf(1000.0f * t * f);
    }
    {
        std::vector<float> h0((size_t)d_dim);
        linear_host(w0, b0, 256, d_dim, te.data(), h0.data());
        for (auto & v : h0) v = v / (1.0f + expf(-v));
        linear_host(w1, b1, d_dim, d_dim, h0.data(), th.data());
    }
    // codec embedding rows for the CFG batch (cond real, uncond zeroed when
    // drop_code; both halves share the real one here - the caller decides)
    const int T0 = T / 2;
    std::vector<float> cemb((size_t)d_emb * T);
    for (int i = 0; i < T; i++) {
        if (drop_code && i >= T0) continue;
        const int code_row = (i % T0) / d_repeats;   // repeated code indices
        (void)code_row;
        memcpy(cemb.data() + (size_t)i * d_emb, codec_emb + (size_t)(i % T0) * d_emb,
               (size_t)d_emb * sizeof(float));
    }
    if (!g_x_in || g_x_in->ne[1] != T) {
        dit_masks_ready = false;
        dit_inputs_ready = false;
        if (!build_dit_graph(T)) return false;
    }
    std::vector<float> spk_in((size_t)enc_emb_dim * T, 0.0f);
    std::vector<float> cond_in((size_t)enc_dim * T, 0.0f);
    for (int i = 0; i < T0; i++) {
        if (spk) memcpy(spk_in.data() + (size_t)i * enc_emb_dim, spk,
                        (size_t)enc_emb_dim * sizeof(float));
        if (cond_vec) memcpy(cond_in.data() + (size_t)i * enc_dim, cond_vec,
                             (size_t)enc_dim * sizeof(float));
    }
    auto has_nan = [](const float * v, size_t n) {
        for (size_t i = 0; i < n; i++) if (std::isnan(v[i])) return true;
        return false;
    };
    if (getenv("T2W_DEBUG"))
        fprintf(stderr, "t2w: inputs nan x=%d cemb=%d spk=%d cond=%d t=%d\n",
                (int)has_nan(x, (size_t)d_mel * T), (int)has_nan(cemb.data(), cemb.size()),
                (int)has_nan(spk_in.data(), spk_in.size()), (int)has_nan(cond_in.data(), cond_in.size()),
                (int)has_nan(th.data(), th.size()));
    ggml_backend_tensor_set(g_x_in, x, 0, sizeof(float) * d_mel * T);
    if (!dit_inputs_ready) {   // codec/spk/cond tracks are identical per eval
        ggml_backend_tensor_set(g_code_emb, cemb.data(), 0, sizeof(float) * d_emb * T);
        ggml_backend_tensor_set(g_spk_in, spk_in.data(), 0, sizeof(float) * enc_emb_dim * T);
        ggml_backend_tensor_set(g_cond_in, cond_in.data(), 0, sizeof(float) * enc_dim * T);
        dit_inputs_ready = true;
    }
    ggml_backend_tensor_set(g_t_in, th.data(), 0, sizeof(float) * d_dim);
    // block-diff masks are identical for every RK4 eval - upload them once
    // right after the graph is (re)built, not per eval (22 x T^2 fp16 per
    // eval used to dominate the whole vocoder)
    // block-diff masks are identical for every RK4 eval - build and upload
    // them exactly once per (re)built graph. This is safe now that the t2w
    // graph has its own dedicated gallocr: nothing else allocates on that
    // pool, so the mask buffers cannot move between computations. (When the
    // pool was shared with the talker, buffers moved and we had to re-upload
    // 22 x T^2 fp16 per eval, which dominated the whole vocoder.)
    if (!dit_masks_ready) {
        const int T0b = T0;   // blocks live on the un-batched frame index
        dit_mask_cache.assign(layers.size(), {});
        for (size_t il = 0; il < layers.size(); il++) {
            const bool ahead = std::find(look_ahead_layers.begin(), look_ahead_layers.end(),
                                         (int)il) != look_ahead_layers.end();
            const bool behind = std::find(look_backward_layers.begin(), look_backward_layers.end(),
                                          (int)il) != look_backward_layers.end();
            dit_mask_cache[il].assign((size_t)T * T, ggml_fp32_to_fp16(-INFINITY));
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++) {
                    const int bi = (i % T0b) / d_block, bj = (j % T0b) / d_block;
                    const int diff = bj - bi;
                    if (diff == 0 || (ahead && diff == 1) || (behind && diff == -1))
                        dit_mask_cache[il][(size_t)i * T + j] = ggml_fp32_to_fp16(0.0f);
                }
        }
        std::vector<int> p((size_t)T);
        for (int i = 0; i < T; i++) p[(size_t)i] = i;
        dit_pos_cache = std::move(p);
        for (size_t il = 0; il < layers.size(); il++)
            ggml_backend_tensor_set(g_masks[il], dit_mask_cache[il].data(), 0,
                                    sizeof(ggml_fp16_t) * dit_mask_cache[il].size());
        ggml_backend_tensor_set(g_pos, dit_pos_cache.data(), 0, sizeof(int) * dit_pos_cache.size());
        dit_masks_ready = true;
    }
    const auto tc0 = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(backend, ggf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "t2w: dit compute failed\n");
        return false;
    }
    static int eval_no = 0;
    if ((eval_no++ % 12) == 0 || getenv("T2W_DEBUG"))
        fprintf(stderr, "t2w: dit eval %d (T=%d) compute %.2f s\n", eval_no, T,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count());
    out.resize((size_t)d_mel * T);
    for (size_t si = 0; si < g_sinks.size(); si++) {
        std::vector<float> sv((size_t)ggml_nelements(g_sinks[si]));
        ggml_backend_tensor_get(g_sinks[si], sv.data(), 0, sizeof(float) * sv.size());
        float mn = sv[0], mx = sv[0]; bool nan = false;
        for (float v : sv) { if (std::isnan(v)) nan = true; if (v < mn) mn = v; if (v > mx) mx = v; }
        fprintf(stderr, "t2w: dit sink %zu nan=%d [%.3f..%.3f]\n", si, (int)nan, mn, mx);
    }
    ggml_backend_tensor_get(g_out, out.data(), 0, sizeof(float) * out.size());
    if (getenv("T2W_DEBUG")) {
        float mn = out[0], mx = out[0];
        bool nan = false;
        for (float v : out) { if (std::isnan(v)) nan = true; if (v < mn) mn = v; if (v > mx) mx = v; }
        fprintf(stderr, "t2w: dit_eval t=%.3f nan=%d [%.3f..%.3f]\n", t, (int)nan, mn, mx);
    }
    return true;
}

bool Token2Wav::run(const std::vector<int> & codec, std::vector<float> & pcm) {
    if (!cond_ready) {
        if (spk_ref_mel.empty()) { fprintf(stderr, "t2w: no baked speaker\n"); return false; }
        // spk ref mel is stored flat [T][mel]; the encoder wants [mel][T]
        const int L = (int)spk_ref_mel.size() / d_mel;
        std::vector<float> cm((size_t)d_mel * L);
        for (int t = 0; t < L; t++)
            for (int mbin = 0; mbin < d_mel; mbin++)
                cm[(size_t)mbin * L + t] = spk_ref_mel[(size_t)t * d_mel + mbin];
        fprintf(stderr, "t2w: ecapa on %d frames\n", L);
        if (!ecapa_embed(cm.data(), L, cond_cached)) return false;
        float ec_mn = cond_cached[0], ec_mx = cond_cached[0];
        for (float v : cond_cached) { if (v < ec_mn) ec_mn = v; if (v > ec_mx) ec_mx = v; }
        fprintf(stderr, "t2w: ecapa done cond [%.4f..%.4f] n=%zu\n", ec_mn, ec_mx, cond_cached.size());
        cond_ready = true;
    }
    if (getenv("T2W_SKIP")) { fprintf(stderr, "t2w: SKIPPED\n"); return false; }
    // DiT cost grows ~quadratically with T0 and 4096 is the hard graph cap:
    // instead of dropping the WHOLE reply's audio ("bad length"), truncate the
    // codec stream so long answers speak their first N seconds
    int n_code = (int)codec.size();
    std::vector<int> codec_trunc;
    const int * codes = codec.data();
    {
        int cap = 400;   // 25 Hz codecs ~= 16 s of speech
        if (const char * e = getenv("QOMNI_T2W_MAX_CODES")) cap = atoi(e);
        const int t0_cap = 4096 / d_repeats;
        if (cap <= 0 || cap > t0_cap) cap = t0_cap;
        if (n_code > cap) {
            fprintf(stderr, "t2w: truncating %d codecs to %d (T0 cap; raise QOMNI_T2W_MAX_CODES)\n",
                    n_code, cap);
            codec_trunc.assign(codec.begin(), codec.begin() + cap);
            codes = codec_trunc.data();
            n_code = cap;
        }
    }
    const int T0 = n_code * d_repeats;
    if (T0 < 1 || T0 > 4096) { fprintf(stderr, "t2w: bad length %d\n", T0); return false; }
    const int T = T0 * 2;   // CFG batch
    // initial noise [mel][T0], tiled into the batch
    std::vector<float> x0((size_t)d_mel * T);
    {
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> noise((size_t)d_mel * T0);
        for (auto & v : noise) v = nd(rng);
        for (int i = 0; i < 2; i++)   // two CFG halves share the noise
            memcpy(x0.data() + (size_t)i * d_mel * T0, noise.data(),
                   (size_t)d_mel * T0 * sizeof(float));
    }
    // codec embeddings (each code repeated d_repeats times), per frame
    std::vector<float> cemb((size_t)d_emb * T0);
    {
        std::vector<float> row;
        for (int i = 0; i < n_code; i++) {
            row.resize((size_t)d_emb);
            read_emb_row(codec_embd, codes[(size_t)i], row.data());
            for (int r = 0; r < d_repeats; r++)
                memcpy(cemb.data() + (size_t)(i * d_repeats + r) * d_emb, row.data(),
                       (size_t)d_emb * sizeof(float));
        }
    }
    // new turn: the codec embedding track changes even when T is unchanged
    dit_inputs_ready = false;
    // RK4 over t in [0, 1] with the sway-corrected grid
    std::vector<float> tgrid((size_t)num_steps);
    for (int i = 0; i < num_steps; i++) {
        const float t = (float)i / (num_steps - 1);
        tgrid[(size_t)i] = t + sway * (cosf((float)M_PI / 2.0f * t) - 1.0f + t);
    }
    std::vector<float> val = x0;
    static std::chrono::steady_clock::time_point g_rk4_t0;
    g_rk4_t0 = std::chrono::steady_clock::now();
    std::vector<float> k1, k2, k3, k4, tmp;
    if (getenv("T2W_DEBUG")) fprintf(stderr, "t2w: rk4 T0=%d (%d evals)\n", T0, (num_steps - 1) * 4);
    bool mel_loaded = false;
    if (const char * mp = getenv("T2W_MEL_LOAD")) {   // parity: feed the vocoder
        FILE * f = fopen(mp, "rb");                     // a reference mel
        if (f) {                                        // [t][bin] row-major
            std::vector<float> ref((size_t)d_mel * T0);
            if (fread(ref.data(), 4, ref.size(), f) == ref.size()) {
                std::vector<float> tr((size_t)d_mel * T0);
                for (int t = 0; t < T0; t++)
                    for (int b = 0; b < d_mel; b++)
                        tr[(size_t)b * T0 + t] = ref[(size_t)t * d_mel + b];
                val = tr;
                mel_loaded = true;
            } else fprintf(stderr, "t2w: mel load size mismatch\n");
            fclose(f);
        }
    }
    if (mel_loaded) {
        fprintf(stderr, "t2w: rk4 SKIPPED (T2W_MEL_LOAD)\n");
    } else {
    auto evalf = [&](float t, const std::vector<float> & v, std::vector<float> & o) {
        // v is [mel][T0] per batch half; build the [mel][2*T0] batch
        std::vector<float> batch((size_t)d_mel * T);
        for (int i = 0; i < 2; i++)   // two CFG halves
            memcpy(batch.data() + (size_t)i * d_mel * T0, v.data(), (size_t)d_mel * T0 * sizeof(float));
        if (!dit_eval(batch.data(), T, t, cemb.data(), false, spk_cond.data(),
                      cond_cached.data(), o)) return false;
        // guided = cond + g*(cond - uncond), computed over the first half
        for (int i = 0; i < d_mel * T0; i++) {
            const float c = o[i], u = o[(size_t)T0 * d_mel + i];
            o[i] = c + guidance * (c - u);
        }
        o.resize((size_t)d_mel * T0);
        return true;
    };
    for (int s = 0; s + 1 < num_steps; s++) {
        const float t0 = tgrid[(size_t)s], t1 = tgrid[(size_t)s + 1];
        const float h = t1 - t0;
        if (!evalf(t0, val, k1)) return false;
        tmp.resize(val.size());
        for (size_t i = 0; i < val.size(); i++) tmp[i] = val[i] + h * k1[i] / 3.0f;
        if (!evalf(t0 + h / 3.0f, tmp, k2)) return false;
        for (size_t i = 0; i < val.size(); i++) tmp[i] = val[i] + h * (k2[i] - k1[i] / 3.0f);
        if (!evalf(t0 + 2.0f * h / 3.0f, tmp, k3)) return false;
        for (size_t i = 0; i < val.size(); i++) tmp[i] = val[i] + h * (k1[i] - k2[i] + k3[i]);
        if (!evalf(t1, tmp, k4)) return false;
        for (size_t i = 0; i < val.size(); i++)
            val[i] += h * (k1[i] + 3.0f * (k2[i] + k3[i]) + k4[i]) / 8.0f;
    }
    fprintf(stderr, "t2w: rk4 done mel [%.3f..%.3f] mean %.3f\n",
            *std::min_element(val.begin(), val.end()), *std::max_element(val.begin(), val.end()),
            std::accumulate(val.begin(), val.end(), 0.0) / val.size());
    }
    if (const char * dp = getenv("T2W_MEL_DUMP")) {   // parity hook: the mel
        FILE * f = fopen(dp, "wb");                   // actually fed to the vocoder
        if (f) {                                      // as [t][bin] row-major
            std::vector<float> tr((size_t)d_mel * T0);
            for (int t = 0; t < T0; t++)
                for (int b = 0; b < d_mel; b++)
                    tr[(size_t)t * d_mel + b] = val[(size_t)b * T0 + t];
            fwrite(tr.data(), 4, tr.size(), f);
            fclose(f);
        }
    }
    // val: [mel][T0] mel spectrogram -> BigVGAN
    const auto tb0 = std::chrono::steady_clock::now();
    fprintf(stderr, "t2w: rk4 loop %.2f s\n",
            std::chrono::duration<double>(tb0 - g_rk4_t0).count());
    const bool host_oracle = getenv("T2W_HOST") && atoi(getenv("T2W_HOST"));
    if (host_oracle)
        bigvgan_run(val, T0, pcm);
    else if (!bigvgan_run_ggml(val, T0, pcm)) {
        fprintf(stderr, "t2w: bigvgan graph failed, falling back to host\n");
        bigvgan_run(val, T0, pcm);
    }
    fprintf(stderr, "t2w: bigvgan(%s) %.2f s\n", host_oracle ? "host" : "gpu",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - tb0).count());
    if (const char * dp = getenv("T2W_PCM_DUMP")) {
        FILE * f = fopen(dp, "wb");
        if (f) { int n = (int)pcm.size(); fwrite(&n, 4, 1, f);
                 fwrite(pcm.data(), 4, pcm.size(), f); fclose(f); }
    }
    return !pcm.empty();
}

bool Token2Wav::bigvgan_run(const std::vector<float> & mel, int L0, std::vector<float> & pcm) {
    int L = L0;
    // preprocess: exp -> dB(-115) - 20 -> normalize to [-1, 1]
    std::vector<float> x((size_t)b_mel * L);
    const float min_amp = powf(10.0f, -115.0f / 20.0f);
    for (size_t i = 0; i < x.size(); i++) {
        const float amp = expf(mel[i]);
        const float db = 20.0f * log10f(std::max(amp, min_amp)) - 20.0f;
        x[i] = std::max(-1.0f, std::min(1.0f, (db + 115.0f) / 57.5f - 1.0f));
    }
    auto conv_app = [&](const std::vector<float> & in, int ic, int Tin, int oc, int kwidth,
                        const std::vector<float> & w, const std::vector<float> & b,
                        std::vector<float> & out) {
        const int pad = kwidth / 2;
        out.assign((size_t)oc * Tin, 0.0f);
        conv1d_full(in.data(), ic, Tin, w.data(), b.empty() ? nullptr : b.data(), oc, kwidth,
                    1, pad, out.data(), Tin);
    };
    int cc = b_ch;
    std::vector<float> cur;
    fprintf(stderr, "t2w: bigvgan L=%d\n", L);
    conv_app(x, b_mel, L, cc, 7, bg_pre_w, bg_pre_b, cur);
    if (getenv("T2W_DEBUG")) {
        float mn = cur[0], mx = cur[0];
        for (float v : cur) { if (v < mn) mn = v; if (v > mx) mx = v; }
        fprintf(stderr, "t2w: x ch0 t0-3 = %.6f %.6f %.6f\n", x[0], x[1], x[2]);
        fprintf(stderr, "t2w: convpre ch0 t0-3 = %.6f %.6f %.6f %.6f w0=%.6f b0=%.6f\n",
                cur[0], cur[1], cur[2], cur[3], bg_pre_w[0], bg_pre_b.empty() ? 0.f : bg_pre_b[0]);
        fprintf(stderr, "t2w: bg x [%.2f..%.2f] conv_pre [%.2f..%.2f]\n",
                *std::min_element(x.begin(), x.end()), *std::max_element(x.begin(), x.end()), mn, mx);
    }
    // 6x [conv transpose + mean of 3 res-blocks]
    for (size_t u = 0; u < ups.size(); u++) {
        const UpLayer & U = ups[u];
        const int kin = L;
        const int p = (U.k - U.stride) / 2;
        const int Lout = (kin - 1) * U.stride + U.k - 2 * p;
        std::vector<float> up((size_t)U.oc * Lout);
        conv_transpose1d_full(cur.data(), cc, kin, U.w.data(), U.b.data(), U.oc, U.k,
                              U.stride, p, up.data(), Lout);
        if (getenv("T2W_DEBUG") && u == 0)
            fprintf(stderr, "t2w: up0 ch0 t0-3 = %.6f %.6f %.6f (cur ch0 t0 = %.6f)\n",
                    up[0], up[(size_t)1], up[(size_t)2], cur[0]);
        // 3 res-blocks at U.oc channels, summed and averaged
        if (const char * dp = getenv("T2W_BLK_DUMP"); dp && u == 0) {
            char fn[256]; snprintf(fn, sizeof(fn), "%s_up.bin", dp);
            FILE * f = fopen(fn, "wb");
            if (f) { fwrite(&cc, 4, 1, f); fwrite(&kin, 4, 1, f);
                     fwrite(up.data(), 4, up.size(), f); fclose(f); }
        }
        std::vector<float> acc((size_t)U.oc * Lout, 0.0f);
        for (int r = 0; r < 3; r++) {
            const ResBlock & R = rbs[(size_t)u * 3 + (size_t)r];
            std::vector<float> h = up;
            int hc = U.oc;
            for (int kk = 0; kk < 3; kk++) {
                const int k1 = (int)(R.c1[(size_t)kk].w.size() / (size_t)(hc * hc));   // exact: [oc][ic][k]
                const int k2 = (int)(R.c2[(size_t)kk].w.size() / (size_t)(hc * hc));
                const int dil[3] = {1, 3, 5};
                std::vector<float> t((size_t)hc * Lout), t2((size_t)hc * Lout);
                aa_snake_act(h.data(), hc, Lout, R.s1a[(size_t)kk], R.s1b[(size_t)kk],
                             aa_filter, t.data());
                if (const char * dp = getenv("T2W_AA_DUMP"); dp && u == 0 && r == 0 && kk == 0) {
                    FILE * f = fopen(dp, "wb");
                    if (f) { int c2 = hc, t2_ = Lout; fwrite(&c2, 4, 1, f); fwrite(&t2_, 4, 1, f);
                             fwrite(t.data(), 4, t.size(), f); fclose(f); }
                }
                if (getenv("T2W_DEBUG") && u == 0 && r == 0 && kk == 0)
                    fprintf(stderr, "t2w: aa_snake0 |y|=%.4f y0=%.6f h0=%.6f a=%.4f b=%.4f\n",
                            fabsf(t[0]), t[1], h[1],
                            expf(R.s1a[0][0]), expf(R.s1b[0][0]) + 1e-9f);
                std::vector<float> y((size_t)hc * Lout);
                conv1d_full(t.data(), hc, Lout, R.c1[(size_t)kk].w.data(),
                            R.c1[(size_t)kk].b.data(), hc, k1, dil[kk], (k1 * dil[kk] - dil[kk]) / 2,
                            y.data(), Lout);
                aa_snake_act(y.data(), hc, Lout, R.s2a[(size_t)kk], R.s2b[(size_t)kk],
                             aa_filter, t2.data());
                conv1d_full(t2.data(), hc, Lout, R.c2[(size_t)kk].w.data(),
                            R.c2[(size_t)kk].b.data(), hc, k2, 1, (k2 - 1) / 2, y.data(), Lout);
                for (size_t i = 0; i < h.size(); i++) h[i] += y[i];
                if (getenv("T2W_DEBUG") && u == 3)
                    fprintf(stderr, "t2w: L3 r%d kk%d |t|=%.4f |c1y|=%.4f |y|=%.4f |h|=%.4f\n",
                            r, kk, fabsf(t[100]), fabsf(y[100]), fabsf(y[500]),
                            fabsf(h[100]));
            }
            for (size_t i = 0; i < acc.size(); i++) acc[i] += h[i];
            if (const char * dp = getenv("T2W_BLK_DUMP"); dp && u == 0) {
                char fn[256]; snprintf(fn, sizeof(fn), "%s_r%d.bin", dp, r);
                FILE * f = fopen(fn, "wb");
                if (f) { fwrite(&hc, 4, 1, f); fwrite(&Lout, 4, 1, f);
                         fwrite(h.data(), 4, h.size(), f); fclose(f); }
            }
        }
        const float inv = 1.0f / 3.0f;
        for (auto & v : acc) v *= inv;
        if (getenv("T2W_DEBUG")) {
            float mup = 0, mrb = 0;
            for (float v : up) mup = std::max(mup, fabsf(v));
            for (float v : acc) mrb = std::max(mrb, fabsf(v));
            float wrb = 0; for (float v : rbs[0].c1[0].w) wrb += fabsf(v);
            fprintf(stderr, "t2w: up0 |y|=%.3f rb |y|=%.3f rb.w absmean %.6f (k=%d)\n",
                    mup, mrb, wrb / rbs[0].c1[0].w.size(),
                    (int)(rbs[0].c1[0].w.size() / (size_t)(U.oc * U.oc)));
        }
        cur.swap(acc);
        if (const char * dp = getenv("T2W_LAYER_DUMP")) {
            char fn[256];
            snprintf(fn, sizeof(fn), "%s_L%d.bin", dp, (int)u);
            FILE * f = fopen(fn, "wb");
            if (f) { fwrite(&cc, 4, 1, f); fwrite(&L, 4, 1, f);
                     fwrite(cur.data(), 4, cur.size(), f); fclose(f); }
        }
        cc = U.oc;
        L = Lout;
    }
    // final snake + conv_post (no bias) + clamp
    std::vector<float> sf((size_t)cc * L);
    if (getenv("T2W_DEBUG")) fprintf(stderr, "t2w: bigvgan pre-snake [%.3f..%.3f]\n",
            *std::min_element(cur.begin(), cur.end()), *std::max_element(cur.begin(), cur.end()));
    aa_snake_act(cur.data(), cc, L, snake_pre_a, snake_pre_b, aa_filter, sf.data());
    pcm.assign((size_t)L, 0.0f);
    {
        // post_w is [1, cc, 7] torch layout
        const int kw = 7;
        par_for(L, [&](int n) {
            float s = 0.0f;
            for (int c = 0; c < cc; c++) {
                const float * xc = sf.data() + (size_t)c * L;
                const float * wc = post_w.data() + (size_t)c * kw;
                for (int kk = 0; kk < kw; kk++) {
                    const int idx = n + kk - kw / 2;   // correlation, like torch
                    if (idx >= 0 && idx < L) s += wc[kk] * xc[idx];
                }
            }
            pcm[(size_t)n] = std::max(-1.0f, std::min(1.0f, s));
        });
    }
    return true;
}

// ------------------------------------------------------------- BigVGAN graph
//
// Same pipeline as bigvgan_run, expressed as one ggml graph on `backend`.
// Layout conventions (matching ggml's conv ops):
//   * activations [T, C] planar (ne0 = time, contiguous per channel) for the
//     conv inputs/outputs; transposed to [C, T] (ne0 = channel) for bias /
//     snake / padding, where per-channel vectors broadcast along ne[1];
//   * dense conv kernels are referenced straight from wctx: the gguf stores
//     them [K, IC, OC] (torch bytes of [OC][IC][K]) which is exactly what
//     ggml_conv_1d wants; the ConvTranspose upsampler kernels are stored
//     [K, OC, IC] which is what ggml_conv_transpose_1d wants, but that op is
//     F32-only on CUDA, so they are dequantized once per (re)build into F32
//     graph inputs;
//   * the anti-aliased snake (aa_snake_act above) is reproduced as:
//       replicate-pad 5/5 -> zero-stuff x2 -> depthwise kaiser conv (the
//       transposed up-filter, taps flipped because ggml convs are
//       correlations) -> crop to 2T-1 -> snake -> replicate-pad 5/6 ->
//       strided depthwise kaiser conv (the down-filter, taps as-is).
//     The zero-stuffing is exact: the host transposed conv only scatters
//     into even w-lags, so the odd (zero) samples never contribute.
//   * ggml's im2col buffers activations in F16 on the conv path, so the GPU
//     result is not bit-identical to the host f32 oracle (see T2W parity).
bool Token2Wav::build_bg_graph(int L0) {
    if (bctx && bg_L_cached == L0) return true;
    if (bctx) { ggml_free(bctx); bctx = nullptr; }
    if (bg_ibuf) { ggml_backend_buffer_free(bg_ibuf); bg_ibuf = nullptr; }
    if (bg_ictx) { ggml_free(bg_ictx); bg_ictx = nullptr; }
    if (!bg_allocr) bg_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bg_in_uploads.clear();
    bg_L_cached = L0;
    // graph-input constants (filters / gathers / dequantized up-kernels) get
    // their own ctx + buffer, same reasoning as the DiT inputs: gallocr pool
    // memory is reused across non-overlapping lifetimes, which would clobber
    // them between vocoder runs (or even mid-graph for the gather indices)
    struct ggml_init_params bip = { 64ull * 1024 * 1024, NULL, true };
    bg_ictx = ggml_init(bip);
    struct ggml_init_params ip = { 512ull * 1024 * 1024, NULL, true };
    bctx = ggml_init(ip);
    if (!bctx || !bg_ictx) return false;
    bgf = ggml_new_graph_custom(bctx, 65536, false);
    const bool dbg_on = getenv("T2W_BG_DBG") != nullptr;
    bg_dbg.clear();

    auto new_in = [&](const int64_t * ne, int nd, ggml_type ty, const void * data, size_t nbytes) {
        ggml_tensor * t = nd == 1 ? ggml_new_tensor_1d(bg_ictx, ty, ne[0])
                                  : ggml_new_tensor(bg_ictx, ty, nd, ne);
        bg_in_uploads.emplace_back(t, std::vector<char>((const char *)data, (const char *)data + nbytes));
        return t;
    };
    auto gt = [&](const char * n) {
        ggml_tensor * t = ggml_get_tensor(wctx_ref, n);
        if (!t) fprintf(stderr, "t2w/bg: missing tensor %s\n", n);
        return t;
    };
    auto toCT = [&](ggml_tensor * t) {   // [T, C] -> [C, T]
        return ggml_cont(bctx, ggml_transpose(bctx, t));
    };
    // replicate-pad along time in [C, T] layout via a clamp()ed row gather
    auto reppad = [&](ggml_tensor * x, int lp, int rp) {
        const int64_t C = x->ne[0], T = x->ne[1];
        const int64_t Tp = T + lp + rp;
        std::vector<int> h((size_t)Tp);
        for (int t = 0; t < Tp; t++) {
            int v = t - lp;
            h[(size_t)t] = v < 0 ? 0 : (v >= (int)T ? (int)T - 1 : v);
        }
        const int64_t ne[1] = { Tp };
        ggml_tensor * idx = new_in(ne, 1, GGML_TYPE_I32, h.data(), sizeof(int) * (size_t)Tp);
        return ggml_get_rows(bctx, x, idx);   // [C, Tp]
    };
    // depthwise kaiser kernels, one pair per channel count (12 taps)
    std::map<int, ggml_tensor *> k_up, k_dn;
    auto dw_up = [&](int64_t C) {
        auto it = k_up.find((int)C);
        if (it != k_up.end()) return it->second;
        std::vector<float> h((size_t)12 * C);
        for (int64_t c = 0; c < C; c++)
            for (int k = 0; k < 12; k++)
                h[(size_t)c * 12 + (size_t)k] = 2.0f * aa_filter[11 - k];
        const int64_t ne[3] = { 12, 1, C };
        ggml_tensor * t = new_in(ne, 3, GGML_TYPE_F32, h.data(), h.size() * 4);
        k_up[(int)C] = t;
        return t;
    };
    auto dw_dn = [&](int64_t C) {
        auto it = k_dn.find((int)C);
        if (it != k_dn.end()) return it->second;
        std::vector<float> h((size_t)12 * C);
        for (int64_t c = 0; c < C; c++)
            for (int k = 0; k < 12; k++)
                h[(size_t)c * 12 + (size_t)k] = aa_filter[k];
        const int64_t ne[3] = { 12, 1, C };
        ggml_tensor * t = new_in(ne, 3, GGML_TYPE_F32, h.data(), h.size() * 4);
        k_dn[(int)C] = t;
        return t;
    };
    // dense conv: in [T, IC] cont, kernel F32 [K, IC, OC] -> out [OC, T'].
    // Built as im2col(F32) + mul_mat instead of ggml_conv_1d: the latter
    // hardcodes an F16 im2col buffer, which quantizes every activation and
    // caps the host-parity corr at ~0.995
    auto conv1 = [&](ggml_tensor * w, ggml_tensor * b, ggml_tensor * in, int dil) {
        const int K = (int)w->ne[0];
        const int pad = (K * dil - dil) / 2;
        ggml_tensor * im = ggml_im2col(bctx, w, in, 1, 0, pad, 0, dil, 0, false,
                                       GGML_TYPE_F32);              // [IC*K, OW]
        ggml_tensor * y = ggml_mul_mat(
            bctx, ggml_reshape_2d(bctx, im, im->ne[0], im->ne[2] * im->ne[1]),
            ggml_reshape_2d(bctx, w, w->ne[0] * w->ne[1], w->ne[2]));   // [OW, OC]
        ggml_tensor * yc = toCT(y);
        if (b) yc = ggml_add(bctx, yc, b);                          // [OC] broadcasts
        return yc;
    };
    // depthwise conv with an F32 im2col, same reasoning
    auto conv_dw_f32 = [&](ggml_tensor * k /*[K,1,C] F32*/, ggml_tensor * in /*[T,C]*/,
                           int st, int pad) {
        ggml_tensor * nb4 = ggml_reshape_4d(bctx, in, in->ne[0], 1, in->ne[1], 1);
        ggml_tensor * im = ggml_im2col(bctx, k, nb4, st, 0, pad, 0, 1, 0, false,
                                       GGML_TYPE_F32);              // [K, OW, C]
        ggml_tensor * r = ggml_mul_mat(bctx, im, k);                // [OW, 1, C]
        return ggml_reshape_3d(bctx, r, r->ne[0], r->ne[2], 1);     // [OW, C]
    };
    // scalar constant broadcastable anywhere
    auto fconst = [&](float v) {
        const int64_t ne[1] = { 1 };
        return new_in(ne, 1, GGML_TYPE_F32, &v, sizeof(float));
    };
    ggml_tensor * c_beta = fconst(1e-9f);

    // ---- anti-aliased snake: x [C,T] (cont) -> y [C,T] -------------------
    auto aa_snake_g = [&](ggml_tensor * h, ggml_tensor * a_t, ggml_tensor * b_t) {
        const int64_t C = h->ne[0], T = h->ne[1];
        ggml_tensor * xp = reppad(h, 5, 5);                        // [C, T+10]
        // zero-stuff: [C,Tp] -> [C,2Tp] with z[2t]=x[t], z[2t+1]=0.
        // concat along channels makes each time row [xp[:,t]; 0]; the flat
        // [C,2,Tp] order (c + C*j + 2C*t) reinterpreted as [C,2Tp] is exactly
        // (c, 2t + j) - no permute needed
        ggml_tensor * zs = ggml_scale(bctx, xp, 0.0f);
        ggml_tensor * cat = ggml_concat(bctx, xp, zs, 0);          // [2C, Tp]
        ggml_tensor * r3 = ggml_reshape_3d(bctx, cat, C, 2, T + 10);
        ggml_tensor * z = ggml_reshape_2d(bctx, r3, C, 2 * (T + 10));
        // up: y[o] = sum_k h'[k] z[o + 4 + k] - correlation with no padding,
        // read off from output row 4 (host: y[o] = sum_kk 2f[kk] xrep[(o+5-kk)/2])
        ggml_tensor * cu = conv_dw_f32(dw_up(C), toCT(z), 1, 0);
        ggml_tensor * cut = toCT(cu);                              // [C, 2Tp-11]
        ggml_tensor * yu = ggml_cont(bctx, ggml_view_2d(bctx, cut, C, 2 * T - 1,
                                                        cut->nb[1], 4 * cut->nb[1]));
        if (dbg_on && getenv("AA_TRACE")) {
            ggml_tensor * sk = ggml_dup(bctx, yu);
            ggml_build_forward_expand(bgf, sk);
            bg_dbg.push_back(sk);   // up-conv result yu
        }
        // snake: y + sin(a y)^2 / (exp(b) + eps)
        ggml_tensor * ar = ggml_reshape_2d(bctx, a_t, C, 1);
        ggml_tensor * br = ggml_reshape_2d(bctx, b_t, C, 1);
        ggml_tensor * sa = ggml_sin(bctx, ggml_mul(bctx, yu, ggml_exp(bctx, ar)));
        ggml_tensor * s2 = ggml_add(bctx, yu,
                                    ggml_div(bctx, ggml_mul(bctx, sa, sa),
                                             ggml_add(bctx, ggml_exp(bctx, br), c_beta)));
        // down: replicate pad 5/6, strided depthwise conv with the filter
        if (dbg_on && getenv("AA_TRACE")) {
            ggml_tensor * sk = ggml_dup(bctx, s2);
            ggml_build_forward_expand(bgf, sk);
            bg_dbg.push_back(sk);   // snaked yu
        }
        ggml_tensor * xd = reppad(s2, 5, 6);                       // [C, 2T+10]
        ggml_tensor * cd = conv_dw_f32(dw_dn(C), toCT(xd), 2, 0);  // [T, C]
        return toCT(cd);
    };

    // ---- mel preprocess: exp -> dB(-115) - 20 -> [-1, 1] -----------------
    const int64_t mel_ne[2] = { L0, b_mel };
    bg_mel_in = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, L0, b_mel);
    ggml_set_name(bg_mel_in, "bg_mel");
    ggml_tensor * x = bg_mel_in;
    x = ggml_clamp(bctx, ggml_exp(bctx, x), powf(10.0f, -115.0f / 20.0f), 1e30f);
    x = ggml_log(bctx, x);
    // (20*log10(v) - 20 + 115)/57.5 - 1 == (20/ln10 * ln(v) + 57.5)/57.5 - 1
    //                           ==  (20/ln10 * ln(v) + 37.5)/57.5... keep steps:
    x = ggml_scale(bctx, x, 20.0f / logf(10.0f));   // dB re min amplitude
    x = ggml_add(bctx, x, fconst(-20.0f));
    x = ggml_scale(bctx, x, 1.0f / 57.5f);
    x = ggml_add(bctx, x, fconst(115.0f / 57.5f - 1.0f));
    x = ggml_clamp(bctx, x, -1.0f, 1.0f);
    if (dbg_on) {
        ggml_tensor * snk = ggml_dup(bctx, x);
        ggml_build_forward_expand(bgf, snk);
        bg_dbg.push_back(snk);   // dbg 0: preprocessed mel [L0, 80] planar
    }

    // ---- conv_pre ---------------------------------------------------------
    ggml_tensor * cur = conv1(gt("bg.conv_pre.weight"), gt("bg.conv_pre.bias"), x, 1);
    if (dbg_on) {
        ggml_tensor * snk = ggml_dup(bctx, cur);
        ggml_build_forward_expand(bgf, snk);
        bg_dbg.push_back(snk);   // dbg 1: conv_pre out [C, L] planar
    }
    int64_t Cc = b_ch;
    int L = L0;

    // ---- 6x [conv_transpose + mean of 3 AMP res-blocks] -------------------
    const int n_up = 6, n_rb = 18;
    const int up_k[6] = {11, 7, 4, 4, 4, 4}, up_s[6] = {5, 3, 2, 2, 2, 2};
    for (int u = 0; u < n_up; u++) {
        const int OC = b_ch >> (u + 1);
        const int kin = L, p = (up_k[u] - up_s[u]) / 2;
        const int Lout = (kin - 1) * up_s[u] + up_k[u] - 2 * p;
        // F32 dequantized [K, OC, IC] transposed-conv kernel (CUDA op is F32)
        char wn[64];
        snprintf(wn, sizeof(wn), "bg.up.%d.weight", u);
        std::vector<float> uw;
        host_vec(gt(wn), uw);
        snprintf(wn, sizeof(wn), "bg.up.%d.bias", u);
        ggml_tensor * ub = gt(wn);
        const int64_t uw_ne[3] = { up_k[u], OC, (int64_t)Cc };
        ggml_tensor * uwt = new_in(uw_ne, 3, GGML_TYPE_F32, uw.data(), uw.size() * sizeof(float));
        ggml_tensor * in = ggml_cont(bctx, ggml_transpose(bctx, cur));       // [T, IC]
        ggml_tensor * yt = ggml_conv_transpose_1d(bctx, uwt, in, up_s[u], 0, 1);  // [Tfull, OC]
        ggml_tensor * yc = ggml_cont(bctx, ggml_view_2d(bctx, yt, Lout, OC,
                                                        yt->nb[1], (size_t)p * yt->nb[0]));
        yc = ggml_add(bctx, toCT(yc), ub);                                   // [OC, Lout]
        if (dbg_on && u == 0) {
            ggml_tensor * snk = ggml_dup(bctx, yc);
            ggml_build_forward_expand(bgf, snk);
            bg_dbg.push_back(snk);   // after conv_transpose + bias
        }
        // 3 res-blocks, summed and averaged
        ggml_tensor * acc = nullptr;
        for (int r = 0; r < 3; r++) {
            const int rbi = u * 3 + r;
            ggml_tensor * h = yc;
            for (int kk = 0; kk < 3; kk++) {
                const int dil[3] = {1, 3, 5};
                char b1[64], b2[64], b3[64], b4[64];
                snprintf(b1, sizeof(b1), "bg.rb.%d.c1.%d.weight", rbi, kk);
                snprintf(b2, sizeof(b2), "bg.rb.%d.c1.%d.bias", rbi, kk);
                snprintf(b3, sizeof(b3), "bg.rb.%d.c2.%d.weight", rbi, kk);
                snprintf(b4, sizeof(b4), "bg.rb.%d.c2.%d.bias", rbi, kk);
                snprintf(wn, sizeof(wn), "bg.rb.%d.s1.%d.alpha", rbi, kk);
                ggml_tensor * s1a = gt(wn);
                snprintf(wn, sizeof(wn), "bg.rb.%d.s1.%d.beta", rbi, kk);
                ggml_tensor * s1b = gt(wn);
                snprintf(wn, sizeof(wn), "bg.rb.%d.s2.%d.alpha", rbi, kk);
                ggml_tensor * s2a = gt(wn);
                snprintf(wn, sizeof(wn), "bg.rb.%d.s2.%d.beta", rbi, kk);
                ggml_tensor * s2b = gt(wn);
                ggml_tensor * t1 = aa_snake_g(h, s1a, s1b);
                if (dbg_on && u == 0 && r == 0 && kk == 0) {
                    ggml_tensor * snk = ggml_dup(bctx, t1);
                    ggml_build_forward_expand(bgf, snk);
                    bg_dbg.push_back(snk);   // first aa_snake out
                }
                ggml_tensor * y1 = conv1(gt(b1), gt(b2),
                                         ggml_cont(bctx, ggml_transpose(bctx, t1)), dil[kk]);
                ggml_tensor * t2 = aa_snake_g(y1, s2a, s2b);
                ggml_tensor * y2 = conv1(gt(b3), gt(b4),
                                         ggml_cont(bctx, ggml_transpose(bctx, t2)), 1);
                h = ggml_add(bctx, h, y2);
            }
            acc = acc ? ggml_add(bctx, acc, h) : h;
        }
        cur = ggml_scale(bctx, acc, 1.0f / 3.0f);
        if (dbg_on) {
            ggml_tensor * snk = ggml_dup(bctx, cur);
            ggml_build_forward_expand(bgf, snk);
            bg_dbg.push_back(snk);   // dbg 1..6: per up-layer out [C, L]
        }
        (void)n_rb;
        Cc = OC;
        L = Lout;
    }

    // ---- final aa-snake + conv_post (no bias) + clamp ---------------------
    ggml_tensor * sf = aa_snake_g(cur, gt("bg.snake_pre.alpha"), gt("bg.snake_pre.beta"));
    ggml_tensor * out = ggml_conv_1d(bctx, gt("bg.conv_post.weight"),
                                     ggml_cont(bctx, ggml_transpose(bctx, sf)), 1, 3, 1);
    out = ggml_clamp(bctx, out, -1.0f, 1.0f);   // [L, 1]
    // keep the debug sinks alive to the very end: gallocr reuses pool memory
    // across non-overlapping lifetimes, so a mid-graph sink would be clobbered
    // by later stages before it can be read back
    for (size_t i = 0; i < bg_dbg.size(); i++)
        out = ggml_add(bctx, out, ggml_scale(bctx, bg_dbg[i], 0.0f));
    ggml_set_name(out, "bg_pcm");
    bg_out = out;
    ggml_build_forward_expand(bgf, out);
    bg_ibuf = ggml_backend_alloc_ctx_tensors(bg_ictx, backend);
    if (!bg_ibuf) {
        fprintf(stderr, "t2w/bg: input tensor alloc failed\n");
        return false;
    }
    if (!ggml_gallocr_alloc_graph(bg_allocr, bgf)) {
        fprintf(stderr, "t2w/bg: graph alloc failed\n");
        return false;
    }
    for (auto & up : bg_in_uploads) {
        ggml_backend_tensor_set(up.first, up.second.data(), 0, up.second.size());
    }
    fprintf(stderr, "t2w/bg: graph built L0=%d nodes=%d\n", L0, (int)ggml_graph_n_nodes(bgf));
    return true;
}

bool Token2Wav::bigvgan_run_ggml(const std::vector<float> & mel, int L0, std::vector<float> & pcm) {
    if (!bg_mel_in || bg_L_cached != L0)
        if (!build_bg_graph(L0)) return false;
    // mel may be longer than the graph input (the RK4 state keeps the CFG
    // batch halves); the vocoder only consumes the first [b_mel, L0] block
    ggml_backend_tensor_set(bg_mel_in, mel.data(), 0,
                            sizeof(float) * (size_t)b_mel * L0);
    if (ggml_backend_graph_compute(backend, bgf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "t2w/bg: compute failed\n");
        return false;
    }
    pcm.assign((size_t)ggml_nelements(bg_out), 0.0f);
    ggml_backend_tensor_get(bg_out, pcm.data(), 0, sizeof(float) * pcm.size());
    if (getenv("T2W_BG_DBG")) {
        for (size_t i = 0; i < bg_dbg.size(); i++) {
            char fn[256];
            snprintf(fn, sizeof(fn), "%s%zu.bin", getenv("T2W_BG_DBG"), i);
            FILE * f = fopen(fn, "wb");
            if (f) {
                const size_t n = (size_t)ggml_nelements(bg_dbg[i]);
                std::vector<float> v(n);
                ggml_backend_tensor_get(bg_dbg[i], v.data(), 0, sizeof(float) * n);
                int c = (int)bg_dbg[i]->ne[0], t = (int)bg_dbg[i]->ne[1];
                fwrite(&c, 4, 1, f); fwrite(&t, 4, 1, f);
                fwrite(v.data(), 4, n, f);
                fclose(f);
            }
        }
    }
    return true;
}

}  // namespace qwen25omni
