// mage_gate.hpp - StreamMind cognition gate (Mage-VL System 1), host-side.
//
// The gate watches the perception token stream (one 2560-d mean-pooled token
// per frame) and outputs silent/speak logits per frame. Architecture (from
// the checkpoint's streammind_gate.py):
//   pre_net: fc + leaky_relu
//   1x Mamba-1 (d_model 2560, d_inner 5120, dt_rank 160, n_state 16, conv k4)
//     - mixer pre-norm = RMSNorm(2560); residual semantics per mamba_ssm Block
//   post_net: fc(leaky_relu(x))
//   cls_net: 4-layer vocab-2 Qwen3 over a [perception, embed(0)] token pair;
//     logits at position 0 = (silent, speak)
//
// The Mamba parts run as plain host F32 loops (one step per frame, ~50 MFLOP -
// no graph needed). The cls Qwen3 runs as a tiny ggml graph on a CPU backend.
#pragma once

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace mage {

struct MambaWeights {
    // host F32
    std::vector<float> in_proj_w;    // [10240, 2560] (x 5120 | z 5120)
    std::vector<float> conv_w;       // [5120, 4]
    std::vector<float> conv_b;       // [5120]
    std::vector<float> x_proj_w;     // [192, 5120]
    std::vector<float> dt_proj_w;    // [5120, 160]
    std::vector<float> dt_proj_b;    // [5120]
    std::vector<float> A;            // [5120, 16]  = -exp(A_log)
    std::vector<float> D;            // [5120]
    std::vector<float> out_proj_w;   // [2560, 5120]
    std::vector<float> norm_w, norm_b;   // mixer RMSNorm(2560)
    int d_model = 2560, d_inner = 5120, dt_rank = 160, n_state = 16, conv_k = 4;
    float rms_eps = 1e-5f;   // mamba_ssm RMSNorm default

    void load(struct gguf_context * gu, struct ggml_context * ctx);
};

// F32 read helper for any ggml tensor type (small host tensors only).
static inline void tensor_to_f32_host(ggml_tensor * t, std::vector<float> & out) {
    const int64_t n = ggml_nelements(t);
    out.resize((size_t)n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n; i++) out[(size_t)i] = ggml_fp16_to_fp32(s[i]);
    } else {
        const struct ggml_type_traits * tt = ggml_get_type_traits(t->type);
        tt->to_float(t->data, out.data(), n);
    }
}

// [out, in] row-major numpy/GGUF-style F32 reader: ggml ne = [in, out].
static inline void tensor_to_f32_mat(ggml_tensor * t, std::vector<float> & out) {
    mage::tensor_to_f32_host(t, out);
}

inline void MambaWeights::load(struct gguf_context * /*gu*/, struct ggml_context * ctx) {
    // NOTE: tensor->data must already point into the mmap (the caller wires it
    // through the wctx pass before calling load()).
    auto f32 = [&](const char * name, std::vector<float> & out) {
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) { fprintf(stderr, "mage_gate: missing %s\n", name); return; }
        mage::tensor_to_f32_host(t, out);
    };
    f32("gm.mamba.in_proj.weight", in_proj_w);
    f32("gm.mamba.conv1d.weight", conv_w);
    f32("gm.mamba.conv1d.bias", conv_b);
    f32("gm.mamba.x_proj.weight", x_proj_w);
    f32("gm.mamba.dt_proj.weight", dt_proj_w);
    f32("gm.mamba.dt_proj.bias", dt_proj_b);
    f32("gm.mamba.A_log.weight", A);
    for (float & v : A) v = -std::exp(v);
    f32("gm.mamba.D.weight", D);
    f32("gm.mamba.out_proj.weight", out_proj_w);
    f32("gm.mamba.norm.weight", norm_w);
    f32("gm.mamba.norm.bias", norm_b);
}

// One gate instance: rolling conv + SSM state across frames.
struct GateStream {
    const MambaWeights * w = nullptr;
    std::vector<float> conv_state;   // [d_inner, 3]  (oldest..newest)
    std::vector<float> ssm_state;    // [d_inner, n_state]
    bool fresh = true;

    void reset(const MambaWeights * weights) {
        w = weights;
        conv_state.assign((size_t)w->d_inner * (w->conv_k - 1), 0.0f);
        ssm_state.assign((size_t)w->d_inner * w->n_state, 0.0f);
        fresh = true;
    }

    // one perception token in, one [silent, speak] logit pair out
    void step(const float * x_t_raw, float * logits2,
              std::vector<float> & scratch) {
        const int D = w->d_model, DI = w->d_inner, R = w->dt_rank, NS = w->n_state;
        // mixer pre-norm: RMSNorm(x) before in_proj
        float x2[2560];
        double ss = 0;
        for (int d = 0; d < D; d++) ss += (double)x_t_raw[d] * x_t_raw[d];
        const float is = 1.0f / std::sqrt((float)(ss / D) + w->rms_eps);
        for (int d = 0; d < D; d++) x2[d] = x_t_raw[d] * is * w->norm_w[d] + w->norm_b[d];
        const float * x_t = x2;
        float * xz = scratch.data();                    // [2*DI]
        gemv(xz, w->in_proj_w.data(), x_t, 2 * DI, D);
        const float * x = xz;
        const float * z = xz + DI;

        // causal conv1d over the rolling window (kernel 4: 3 old + current)
        float * cx = scratch.data() + 2 * DI;           // [DI] conv output
        for (int i = 0; i < DI; i++) {
            float acc = w->conv_b[i];
            for (int j = 0; j < w->conv_k - 1; j++)
                acc += w->conv_w[(size_t)i * w->conv_k + j] * conv_state[(size_t)i * 3 + j];
            acc += w->conv_w[(size_t)i * w->conv_k + 3] * x[i];
            cx[i] = acc / (1.0f + std::exp(-acc));      // silu
        }
        // shift window
        for (int i = 0; i < DI; i++) {
            conv_state[(size_t)i * 3 + 0] = conv_state[(size_t)i * 3 + 1];
            conv_state[(size_t)i * 3 + 1] = conv_state[(size_t)i * 3 + 2];
            conv_state[(size_t)i * 3 + 2] = x[i];
        }

        float * dbc = cx + DI;                          // [R + 2*NS]
        gemv(dbc, w->x_proj_w.data(), cx, R + 2 * NS, DI);
        const float * B = dbc + R;
        const float * C = dbc + R + NS;
        float * dt = dbc + R + 2 * NS;                  // [DI]
        gemv(dt, w->dt_proj_w.data(), dbc, DI, R);
        for (int i = 0; i < DI; i++) dt[i] += w->dt_proj_b[i];
        for (int i = 0; i < DI; i++) {
            // numerically stable softplus (the naive exp() overflows to inf,
            // which then poisons the SSM state with NaNs)
            const float v = dt[i];
            dt[i] = v > 20.0f ? v : std::log1p(std::exp(v));
        }

        // selective-scan step
        for (int i = 0; i < DI; i++) {
            for (int j = 0; j < NS; j++) {
                const float a = dt[i] * w->A[(size_t)i * NS + j];
                const float dAe = std::exp(a);
                float & h = ssm_state[(size_t)i * NS + j];
                h = dAe * h + (dt[i] * B[j]) * cx[i];
            }
        }
        float * y = dbc + R + 2 * NS + DI;              // reuse: needs DI floats
        for (int i = 0; i < DI; i++) {
            float acc = w->D[i] * cx[i];
            for (int j = 0; j < NS; j++) acc += ssm_state[(size_t)i * NS + j] * C[j];
            y[i] = acc;
        }
        for (int i = 0; i < DI; i++) y[i] *= z[i] / (1.0f + std::exp(-z[i]));   // silu(z)
        gemv(logits2, w->out_proj_w.data(), y, D, DI);  // block out (+ residual done by caller)
    }

    size_t scratch_size() const {
        return (size_t)5 * w->d_inner + (size_t)w->dt_rank + 2 * (size_t)w->n_state + 8;
    }

    static void gemv(float * out, const float * W, const float * x, int rows, int cols) {
        for (int r = 0; r < rows; r++) {
            const float * row = W + (size_t)r * cols;
            float acc = 0.0f;
            for (int c = 0; c < cols; c++) acc += row[c] * x[c];
            out[r] = acc;
        }
    }
};

// cls_net: 4-layer vocab-2 Qwen3 over a [perception, embed(0)] pair - plain
// host implementation (the ggml-graph variant tripped over flash-attn dtype
// constraints and gallocr buffer recycling for a 2-token graph).
struct ClsWeights {
    std::vector<float> embed;      // [2, 2560]
    std::vector<float> out_w;      // [2, 2560]
    std::vector<float> out_norm;   // [2560]
    struct Layer {
        std::vector<float> ln, q, k, v, o, qn, kn, ffn_ln, gate, up, down;
    };
    std::vector<Layer> layers;
    int D = 2560, NH = 32, NKV = 8, HD = 128, FF = 12288;
    float eps = 1e-6f, rope_theta = 1000000.0f;

    void load(struct ggml_context * ctx);
    // logits[2] = (silent, speak) for one perception token
    void forward(const float * tok, float * logits, std::vector<float> & scratch) const;
};

inline void ClsWeights::load(struct ggml_context * ctx) {
    layers.resize(4);
    auto f32 = [&](const char * name, std::vector<float> & out) {
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) { fprintf(stderr, "mage_gate: missing %s\n", name); return; }
        mage::tensor_to_f32_host(t, out);
    };
    char buf[128];
    f32("cls.token_embd.weight", embed);
    f32("cls.output.weight", out_w);
    f32("cls.output_norm.weight", out_norm);
    for (int i = 0; i < 4; i++) {
        auto g = [&](const char * sfx) {
            snprintf(buf, sizeof(buf), "cls.blk.%d.%s", i, sfx);
            return buf;
        };
        Layer & L = layers[i];
        f32(g("attn_norm.weight"), L.ln);
        f32(g("attn_q.weight"), L.q);
        f32(g("attn_k.weight"), L.k);
        f32(g("attn_v.weight"), L.v);
        f32(g("attn_output.weight"), L.o);
        f32(g("attn_q_norm.weight"), L.qn);
        f32(g("attn_k_norm.weight"), L.kn);
        f32(g("ffn_norm.weight"), L.ffn_ln);
        f32(g("ffn_gate.weight"), L.gate);
        f32(g("ffn_up.weight"), L.up);
        f32(g("ffn_down.weight"), L.down);
    }
}

inline void rms(std::vector<float> & x, const std::vector<float> & w, float eps) {
    const int D = (int)w.size();
    double ss = 0;
    for (int d = 0; d < D; d++) ss += (double)x[d] * x[d];
    const float is = 1.0f / std::sqrt((float)(ss / D) + eps);
    for (int d = 0; d < D; d++) x[d] *= is * w[d];
}

inline void ClsWeights::forward(const float * tok, float * logits, std::vector<float> & scratch) const {
    const int D = this->D, HD = this->HD, NH = this->NH, NKV = this->NKV, FF = this->FF;
    // x[2][D]: token 0 = perception, token 1 = embed(0)
    float * x = scratch.data();
    for (int d = 0; d < D; d++) { x[d] = tok[d]; x[D + d] = embed[d]; }
    // rope tables for positions 0,1 (NEOX half-split HD=128 -> 64 freqs)
    static thread_local std::vector<float> cosT, sinT;
    cosT.resize(2 * HD); sinT.resize(2 * HD);
    for (int p = 0; p < 2; p++)
        for (int i = 0; i < HD / 2; i++) {
            const float a = (float)p * std::pow(rope_theta, -(float)i / (float)(HD / 2));
            cosT[p * HD + i] = std::cos(a);
            cosT[p * HD + i + HD / 2] = std::cos(a);
            sinT[p * HD + i] = std::sin(a);
            sinT[p * HD + i + HD / 2] = std::sin(a);
        }
    std::vector<float> att(2 * NH * HD), tmp(FF);
    for (int il = 0; il < 4; il++) {
        const Layer & L = layers[il];
        std::vector<float> h2(2 * D);
        for (int t = 0; t < 2; t++) {
            std::vector<float> h(x + t * D, x + t * D + D);
            rms(h, L.ln, eps);
            for (int d = 0; d < D; d++) h2[t * D + d] = h[d];
        }
        // q/k/v projections: q [2, NH*HD], k/v [2, NKV*HD]
        std::vector<float> Q(2 * NH * HD), K(2 * NKV * HD), V(2 * NKV * HD);
        for (int t = 0; t < 2; t++) {
            const float * in = &h2[t * D];
            for (int r = 0; r < NH * HD; r++) {
                const float * w = L.q.data() + (size_t)r * D;
                float acc = 0; for (int c = 0; c < D; c++) acc += w[c] * in[c];
                Q[t * NH * HD + r] = acc;
            }
            for (int r = 0; r < NKV * HD; r++) {
                const float * w = L.k.data() + (size_t)r * D;
                float acc = 0; for (int c = 0; c < D; c++) acc += w[c] * in[c];
                K[t * NKV * HD + r] = acc;
                const float * wv = L.v.data() + (size_t)r * D;
                float accv = 0; for (int c = 0; c < D; c++) accv += wv[c] * in[c];
                V[t * NKV * HD + r] = accv;
            }
        }
        // per-head qk-norm + rope (NEOX, pos = t)
        for (int t = 0; t < 2; t++) {
            for (int hh = 0; hh < NH; hh++) {
                float * qh = &Q[t * NH * HD + hh * HD];
                double ss = 0; for (int d = 0; d < HD; d++) ss += (double)qh[d] * qh[d];
                const float is = 1.0f / std::sqrt((float)(ss / HD) + eps);
                for (int d = 0; d < HD; d++) qh[d] *= is * L.qn[d];
                // rope NEOX halves
                for (int i = 0; i < HD / 2; i++) {
                    const float c = cosT[t * HD + i], s = sinT[t * HD + i];
                    const float e = qh[i], o = qh[i + HD / 2];
                    qh[i] = e * c - o * s;
                    qh[i + HD / 2] = o * c + e * s;
                }
            }
            for (int hh = 0; hh < NKV; hh++) {
                float * kh = &K[t * NKV * HD + hh * HD];
                double ss = 0; for (int d = 0; d < HD; d++) ss += (double)kh[d] * kh[d];
                const float is = 1.0f / std::sqrt((float)(ss / HD) + eps);
                for (int d = 0; d < HD; d++) kh[d] *= is * L.kn[d];
                for (int i = 0; i < HD / 2; i++) {
                    const float c = cosT[t * HD + i], s = sinT[t * HD + i];
                    const float e = kh[i], o = kh[i + HD / 2];
                    kh[i] = e * c - o * s;
                    kh[i + HD / 2] = o * c + e * s;
                }
            }
        }
        // causal attention over 2 tokens, GQA (4 q heads per kv head)
        for (int t = 0; t < 2; t++)
            for (int hh = 0; hh < NH; hh++) {
                const int khh = hh / 4;
                const float * kh = &K[0 * NKV * HD + khh * HD];
                const float * kh1 = &K[1 * NKV * HD + khh * HD];
                const float * qh = &Q[t * NH * HD + hh * HD];
                float s0 = 0, s1 = 0;
                for (int d = 0; d < HD; d++) { s0 += qh[d] * kh[d]; s1 += qh[d] * kh1[d]; }
                s0 /= std::sqrt((float)HD); s1 /= std::sqrt((float)HD);
                float w0, w1;
                if (t == 0) { w0 = 1.0f; w1 = 0.0f; }
                else {
                    const float m = std::max(s0, s1);
                    w0 = std::exp(s0 - m); w1 = std::exp(s1 - m);
                    const float sum = w0 + w1; w0 /= sum; w1 /= sum;
                }
                const float * v0 = &V[0 * NKV * HD + khh * HD];
                const float * v1 = &V[1 * NKV * HD + khh * HD];
                float * oh = &att[t * D + hh * HD];
                for (int d = 0; d < HD; d++) oh[d] = w0 * v0[d] + w1 * v1[d];
            }
        // o proj + residual
        for (int t = 0; t < 2; t++) {
            float * xr = x + t * D;
            for (int r = 0; r < D; r++) {
                const float * w = L.o.data() + (size_t)r * (NH * HD);
                float acc = 0;
                for (int c = 0; c < NH * HD; c++) acc += w[c] * att[t * NH * HD + c];
                xr[r] += acc;
            }
        }
        // ffn
        for (int t = 0; t < 2; t++) {
            float * xr = x + t * D;
            std::vector<float> h(xr, xr + D);
            rms(h, L.ffn_ln, eps);
            for (int r = 0; r < FF; r++) {
                const float * wg = L.gate.data() + (size_t)r * D;
                const float * wu = L.up.data() + (size_t)r * D;
                float a = 0, b = 0;
                for (int c = 0; c < D; c++) { a += wg[c] * h[c]; b += wu[c] * h[c]; }
                tmp[r] = a / (1.0f + std::exp(-a)) * b;
            }
            for (int r = 0; r < D; r++) {
                const float * wd = L.down.data() + (size_t)r * FF;
                float acc = 0;
                for (int c = 0; c < FF; c++) acc += wd[c] * tmp[c];
                xr[r] += acc;
            }
        }
    }
    // final norm + lm_head on token 0
    std::vector<float> h(x, x + D);
    rms(h, out_norm, eps);
    for (int r = 0; r < 2; r++) {
        const float * w = out_w.data() + (size_t)r * D;
        float acc = 0;
        for (int c = 0; c < D; c++) acc += w[c] * h[c];
        logits[r] = acc;
    }
}

}  // namespace mage
