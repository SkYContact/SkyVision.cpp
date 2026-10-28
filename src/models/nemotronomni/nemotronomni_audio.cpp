// nemotronomni_audio.cpp - see nemotronomni_audio.hpp for the contract.
#include "nemotronomni_audio.hpp"

#include "core/engine/ggml_dump.hpp"
#include "core/media/audio_feats.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace nemo {

ParakeetTower::~ParakeetTower() {
    free_graph();
    if (bbuf) ggml_backend_buffer_free(bbuf);
    if (bctx) ggml_free(bctx);
}

void ParakeetTower::free_graph() {
    if (actx) { ggml_free(actx); actx = nullptr; agf = nullptr; }
    a_in = a_pos_in = a_feats = nullptr;
    a_mmask = a_rmask = nullptr;
    a_t = -1;
    a_t_valid_built = -1;
}

// BatchNorm1d eval folded into the (bias-free) depthwise conv weights once at
// load: y = dw*x + 0 -> BN -> y' = (dw*gamma*invstd) * x + (beta - mean*gamma*invstd).
bool ParakeetTower::fold_bn(ggml_backend_t backend) {
    const size_t n_t = (size_t)n_layer * 2 + 8;
    struct ggml_init_params ip = { n_t * ggml_tensor_overhead(), NULL, true };
    bctx = ggml_init(ip);
    if (!bctx) return false;
    for (int il = 0; il < n_layer; il++) {
        ParakeetBlk & L = al[(size_t)il];
        L.conv_dw_fold = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, d_model, conv_k);
        L.conv_dw_bias = ggml_new_tensor_1d(bctx, GGML_TYPE_F32, d_model);
    }
    bbuf = ggml_backend_alloc_ctx_tensors(bctx, backend);
    if (!bbuf) { fprintf(stderr, "nemotronomni: audio fold alloc failed\n"); return false; }

    std::vector<float> dw((size_t)conv_k * d_model);
    std::vector<float> gw(d_model), gb(d_model), gm(d_model), gv(d_model);
    std::vector<float> foldw(dw.size()), foldb(d_model);
    std::vector<ggml_fp16_t> dwh(dw.size());
    for (int il = 0; il < n_layer; il++) {
        ParakeetBlk & L = al[(size_t)il];
        // conv.dw is F16 on disk (mul_mat operand dtype); the BN stats are F32
        ggml_backend_tensor_get(L.conv_dw_raw, dwh.data(), 0, sizeof(ggml_fp16_t) * dwh.size());
        for (size_t i = 0; i < dwh.size(); i++) dw[i] = ggml_fp16_to_fp32(dwh[i]);
        ggml_backend_tensor_get(L.bn_w, gw.data(), 0, sizeof(float) * d_model);
        ggml_backend_tensor_get(L.bn_b, gb.data(), 0, sizeof(float) * d_model);
        ggml_backend_tensor_get(L.bn_mean, gm.data(), 0, sizeof(float) * d_model);
        ggml_backend_tensor_get(L.bn_var, gv.data(), 0, sizeof(float) * d_model);
        for (int c = 0; c < d_model; c++) {
            const float invstd = 1.0f / std::sqrt(gv[(size_t)c] + bn_eps);
            const float s = gw[(size_t)c] * invstd;
            foldb[(size_t)c] = gb[(size_t)c] - gm[(size_t)c] * s;
            for (int k = 0; k < conv_k; k++)
                foldw[(size_t)c + (size_t)k * d_model] = dw[(size_t)k + (size_t)c * conv_k] * s;
        }
        ggml_backend_tensor_set(L.conv_dw_fold, foldw.data(), 0, sizeof(float) * foldw.size());
        ggml_backend_tensor_set(L.conv_dw_bias, foldb.data(), 0, sizeof(float) * foldb.size());
    }
    return true;
}

// ---- conv frontend (host, fp32) --------------------------------------------
//
// conv2d stride 2 pad 1 over (time, freq); [C][F][T] plane layout. After every
// Conv2d (the 1x1 pointwise included) the frames >= cur_len are re-zeroed,
// where cur_len folds (L-1)//2 + 1 per stride-2 conv from the VALID frame
// count (oracle pitfall: skipping the pointwise reset flips the last frame of
// the 4 s clip).

namespace {

struct CState {
    std::vector<float> x;   // [C][F][T]
    int C = 0, F = 0, T = 0;
};

static void conv3x3s2(const float * w, const float * b, int C_out, int C_in,
                      const CState & in, CState & out, int w_ci = -1) {
    // w_ci: input channels the WEIGHT carries - 0 marks a DEPTHWISE conv
    // (channel co's taps applied to input channel co only), C_in for a full
    // conv. Defaults to C_in. (A sentinel of 1 would be ambiguous with a
    // full conv over one channel - stage 1 is exactly that.)
    const bool dw = (w_ci == 0);
    if (w_ci < 0) w_ci = C_in;
    out.C = C_out;
    out.F = (in.F + 1) / 2;             // (F + 2*1 - 3)/2 + 1
    out.T = (in.T - 1) / 2 + 1;
    out.x.assign((size_t)C_out * out.F * out.T, 0.0f);
    for (int co = 0; co < C_out; co++)
        for (int hf = 0; hf < out.F; hf++)
            for (int wt = 0; wt < out.T; wt++) {
                float acc = b ? b[co] : 0.0f;
                for (int ci = 0; ci < C_in; ci++) {
                    if (dw && ci != co) continue;   // depthwise: ci == co only
                    const float * wk =
                        w + ((size_t)co * (dw ? 1 : w_ci) + (dw ? 0 : ci)) * 9;
                    for (int kh = 0; kh < 3; kh++) {       // kh: TIME axis (torch H)
                        const int ti = wt * 2 + kh - 1;
                        if (ti < 0 || ti >= in.T) continue;
                        for (int kw = 0; kw < 3; kw++) {   // kw: FREQ axis (torch W)
                            const int fi = hf * 2 + kw - 1;
                            if (fi < 0 || fi >= in.F) continue;
                            acc += wk[(size_t)kh * 3 + kw]
                                   * in.x[((size_t)ci * in.F + fi) * in.T + ti];
                        }
                    }
                }
                out.x[((size_t)co * out.F + hf) * out.T + wt] = acc;
            }
}

static void conv1x1(const float * w, const float * b, const CState & in, CState & out) {
    out.C = in.C; out.F = in.F; out.T = in.T;
    out.x.assign(in.x.size(), 0.0f);
    for (int co = 0; co < out.C; co++)
        for (int f = 0; f < out.F; f++)
            for (int t = 0; t < out.T; t++) {
                float acc = b ? b[co] : 0.0f;
                for (int ci = 0; ci < out.C; ci++)
                    acc += w[(size_t)co * out.C + ci]
                           * in.x[((size_t)ci * in.F + f) * in.T + t];
                out.x[((size_t)co * out.F + f) * out.T + t] = acc;
            }
}

static void zero_tail(CState & s, int cur_len) {
    for (int c = 0; c < s.C; c++)
        for (int f = 0; f < s.F; f++)
            for (int t = 0; t < s.T; t++) {
                float & v = s.x[((size_t)c * s.F + f) * s.T + t];
                if (t >= cur_len) v = 0.0f;
            }
}
static void relu_zero_tail(CState & s, int cur_len) {
    for (int c = 0; c < s.C; c++)
        for (int f = 0; f < s.F; f++)
            for (int t = 0; t < s.T; t++) {
                float & v = s.x[((size_t)c * s.F + f) * s.T + t];
                v = (t >= cur_len) ? 0.0f : (v > 0.0f ? v : 0.0f);
            }
}

// debug dumps of the CState planes (parity bisection; gated by NEMO_DUMP_DIR)
static bool g_dump_env = false;
static FILE * g_dump_fp = nullptr;
static char g_dump_dir[512] = { 0 };
static bool dump_env() { return g_dump_env; }
static void raw_dump_state(const char * tag, const CState & s) {
    char pth[600];
    snprintf(pth, sizeof(pth), "%s/cpp_%s.bin", g_dump_dir, tag);
    FILE * f = fopen(pth, "wb");
    if (!f) return;
    fwrite(s.x.data(), 4, s.x.size(), f);
    fclose(f);
    snprintf(pth, sizeof(pth), "%s/cpp_%s.shape", g_dump_dir, tag);
    if ((f = fopen(pth, "w")) != nullptr) {
        fprintf(f, "%d %d %d\n", s.C, s.F, s.T);
        fclose(f);
    }
}

}  // namespace

int ParakeetTower::frontend(const float * pcm, int n_samples, std::vector<float> & mel,
                            std::vector<float> & sub, std::vector<float> & pos,
                            int * mel_frames) {
    // ---- mel (librosa slaney, computed - the checkpoint's BF16 fb is NOT
    // used: the oracle mel gate is the HF-extractor dump; the checkpoint fb
    // disagrees at cos 0.964 on pure tones)
    std::vector<float> fb;
    vqa::audio::mel_filterbank_slaney(n_mel, 512, 0.0f, 8000.0f, 16000, fb);
    vqa::audio::log_mel_parakeet(pcm, n_samples, 512, 160, 400, n_mel, 0.97f,
                                 fb.data(), mel);
    const int T = (int)mel.size() / n_mel;
    if (mel_frames) *mel_frames = T;

    // ---- per-utterance CMVN over the L//160 valid frames (ddof=1, eps 1e-5)
    const int valid = n_samples / 160;
    for (int j = 0; j < n_mel; j++) {
        double mean = 0.0;
        for (int t = 0; t < valid; t++) mean += mel[(size_t)t * n_mel + j];
        mean /= valid;
        double var = 0.0;
        for (int t = 0; t < valid; t++) {
            const double d = mel[(size_t)t * n_mel + j] - mean;
            var += d * d;
        }
        var /= (valid - 1);
        const float inv = 1.0f / (std::sqrt((float)var) + 1e-5f);
        for (int t = 0; t < T; t++) {
            float & v = mel[(size_t)t * n_mel + j];
            v = (t < valid) ? (v - (float)mean) * inv : 0.0f;
        }
    }

    // ---- conv frontend: [C][F][T] planes
    CState cur;
    cur.C = 1; cur.F = n_mel; cur.T = T;
    cur.x.assign((size_t)n_mel * T, 0.0f);
    for (int t = 0; t < T; t++)
        for (int j = 0; j < n_mel; j++)
            cur.x[(size_t)j * T + t] = mel[(size_t)t * n_mel + j];

    int cur_len = valid;                 // the mask tracks VALID, not tensor, length
    auto fold = [&cur_len]() { cur_len = (cur_len - 1) / 2 + 1; };

    CState s1;
    conv3x3s2(s0_w.data(), s0_b.data(), sub_ch, 1, cur, s1);
    fold();
    relu_zero_tail(s1, cur_len);
    if (dump_env()) raw_dump_state("a_s1", s1);
    CState sd1, sp1;
    conv3x3s2(s1_dw.data(), s1_dwb.data(), sub_ch, sub_ch, s1, sd1, /*w_ci=*/0);
    fold();
    zero_tail(sd1, cur_len);             // mask after dw (NO relu - HF's layer 4 relu comes after the pw)
    if (dump_env()) raw_dump_state("a_sd1", sd1);
    conv1x1(s1_pw.data(), s1_pwb.data(), sd1, sp1);
    relu_zero_tail(sp1, cur_len);        // relu + mask after pw (HF layer 4)
    if (dump_env()) raw_dump_state("a_sp1", sp1);
    CState sd2, sp2;
    conv3x3s2(s2_dw.data(), s2_dwb.data(), sub_ch, sub_ch, sp1, sd2, /*w_ci=*/0);
    fold();
    zero_tail(sd2, cur_len);
    conv1x1(s2_pw.data(), s2_pwb.data(), sd2, sp2);
    relu_zero_tail(sp2, cur_len);

    a_t_valid = cur_len;                 // HF n_out_valid (50 of 51 when L%hop==0)



    const int Tp = sp2.T;                // == T'
    // flatten (C, F) c-major per time step; the Linear runs in-graph
    sub.assign((size_t)Tp * sub_ch * freq_final, 0.0f);
    for (int t = 0; t < Tp; t++)
        for (int c = 0; c < sub_ch; c++)
            for (int f = 0; f < freq_final; f++)
                sub[(size_t)t * (sub_ch * freq_final) + (size_t)c * freq_final + f] =
                    sp2.x[((size_t)c * sp2.F + f) * sp2.T + t];

    // ---- relative-position table: inv_freq 512 @ base 10000, positions
    // T'-1 .. -(T'-1) (descending), sin/cos interleaved on the last axis
    const int P = 2 * Tp - 1;
    pos.assign((size_t)P * d_model, 0.0f);
    std::vector<double> inv_freq((size_t)d_model / 2);
    for (int i = 0; i < d_model / 2; i++)
        inv_freq[(size_t)i] = 1.0 / std::pow(10000.0, (double)(2 * i) / (double)d_model);
    for (int p = 0; p < P; p++) {
        const double position = (double)(Tp - 1 - p);   // arange(T-1, -T, -1)
        float * row = pos.data() + (size_t)p * d_model;
        for (int i = 0; i < d_model / 2; i++) {
            const float fr = (float)((double)position * inv_freq[(size_t)i]);
            row[2 * i + 0] = std::sin(fr);
            row[2 * i + 1] = std::cos(fr);
        }
    }
    return Tp;
}

bool ParakeetTower::build_graph(int T) {
    free_graph();
    if (!dump_dir.empty()) vqa::dump::dump_registry_clear();
    struct ggml_init_params ip = { 64ull * 1024 * 1024, NULL, /*no_alloc*/ true };
    actx = ggml_init(ip);
    if (!actx) return false;
    agf = ggml_new_graph_custom(actx, 32768, false);
    ggml_context * c = actx;
    const int D = d_model, NH = n_head, HD = D / NH, P = 2 * T - 1;
    const float scale = 1.0f / std::sqrt((float)HD);

    a_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, sub_ch * freq_final, T);
    ggml_set_name(a_in, "a_sub_in");
    ggml_set_input(a_in);
    a_pos_in = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, P);
    ggml_set_name(a_pos_in, "a_pos_in");
    ggml_set_input(a_pos_in);


    // HF padding masks: bd fill -1e30 at masked keys/rows (finfo.min semantics),
    // and 0/1 row mask zeroing fully-masked rows before the depthwise conv.
    a_mmask = ggml_new_tensor_2d(c, GGML_TYPE_F32, T, T);
    ggml_set_input(a_mmask);
    a_rmask = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, T);
    ggml_set_input(a_rmask);

    // subsampling linear -> [D, T]
    ggml_tensor * x = ggml_add(c, ggml_mul_mat(c, sub_proj_w, a_in), sub_proj_b);
    ggml_set_name(x, "a_sub");

    for (int il = 0; il < n_layer; il++) {
        const ParakeetBlk & L = al[(size_t)il];
        char nm[48];

        // macaron FFN (x0.5)
        ggml_tensor * h = ggml_norm(c, x, 1e-5f);
        h = ggml_add(c, ggml_mul(c, h, L.ln_ff1_w), L.ln_ff1_b);
        h = ggml_mul_mat(c, L.ff1_w2, ggml_silu(c, ggml_mul_mat(c, L.ff1_w1, h)));
        x = ggml_add(c, x, ggml_scale(c, h, 0.5f));
        if (il == 0) ggml_set_name(x, "a_l0_ff1");

        // relative-position attention
        ggml_tensor * a = ggml_norm(c, x, 1e-5f);
        a = ggml_add(c, ggml_mul(c, a, L.ln_att_w), L.ln_att_b);
        ggml_tensor * q3 = ggml_reshape_3d(c, ggml_mul_mat(c, L.att_q, a), HD, NH, T);
        ggml_tensor * q3b = ggml_mul_mat(c, L.att_q, a);
        if (il == 0) {
            ggml_set_name(a, "a_l0_ain");
            ggml_set_name(q3, "a_l0_q3");
            ggml_set_name(q3b, "a_l0_q3b");
        }
        ggml_tensor * k3 = ggml_reshape_3d(c, ggml_mul_mat(c, L.att_k, a), HD, NH, T);
        ggml_tensor * v3 = ggml_reshape_3d(c, ggml_mul_mat(c, L.att_v, a), HD, NH, T);
        ggml_tensor * bu = ggml_reshape_3d(c, L.bias_u, HD, NH, 1);   // [hd, 8, 1]
        ggml_tensor * bv = ggml_reshape_3d(c, L.bias_v, HD, NH, 1);
        ggml_tensor * qu = ggml_add(c, q3, bu);
        ggml_tensor * qv = ggml_add(c, q3, bv);
        if (il == 0) ggml_set_name(qu, "a_l0_qu");
        if (il == 0) ggml_set_name(qv, "a_l0_qv");

        // matrix_bd = (qv . rel^T) * scale; the HF pad-left-1 rel_shift +
        // [..., :T] crop collapses to the exact skew view
        //   out[t, j] = bd[t, (T - 1) + j - t]
        // fed to flash-attn as a per-kv-head additive F16 mask. (Padding is
        // impossible for this tower's single unpadded clip, so the HF
        // finfo.min fill / GLU-row-zeroing never engage.)
        ggml_tensor * rel = ggml_reshape_3d(c, ggml_mul_mat(c, L.att_rel_k, a_pos_in),
                                            HD, NH, P);
        ggml_tensor * qv4 = ggml_cont(c, ggml_permute(c, qv, 0, 2, 1, 3));   // [hd, T, 8]
        ggml_tensor * rl4 = ggml_cont(c, ggml_permute(c, rel, 0, 2, 1, 3));  // [hd, P, 8]
        ggml_tensor * bd = ggml_mul_mat(c, rl4, qv4);                        // [p, t, h]
        snprintf(nm, sizeof(nm), "a_l%d_bdraw", il);
        ggml_set_name(bd, nm);
        snprintf(nm, sizeof(nm), "a_l%d_qv4", il);
        ggml_set_name(qv4, nm);
        snprintf(nm, sizeof(nm), "a_l%d_rl4", il);
        ggml_set_name(rl4, nm);
        // HF pad-left-1 rel_shift + [..., :T] crop == per-query-row window
        // over p: element (s(key), t(query)) = bd[s + 21 - t + 43*t] - an
        // all-positive-stride view over the [p, t, h] result.
        ggml_tensor * bdv = ggml_view_3d(c, bd, T, T, NH,
                                         (size_t)(P - 1) * bd->nb[0],          // s stride (42)
                                         bd->nb[2],                            // h stride (P*T); t stride = (P-1)*s
                                         (size_t)(T - 1) * bd->nb[0]);         // offset (p=21)
        snprintf(nm, sizeof(nm), "a_l%d_bdskew", il);
        ggml_set_name(bdv, nm);
        ggml_tensor * bds = ggml_scale(c, ggml_cont(c, bdv), scale);
        snprintf(nm, sizeof(nm), "a_l%d_bd", il);
        ggml_set_name(bds, nm);

        // explicit attention (T <= a few dozen frames; FA with a per-head
        // additive mask carrying large finite bd values was the wave-4 bring-up
        // suspect, so this path avoids FA entirely)
        ggml_tensor * qu_t = ggml_cont(c, ggml_permute(c, qu, 0, 2, 1, 3));  // [hd, t, h]
        ggml_tensor * k_t = ggml_cont(c, ggml_permute(c, k3, 0, 2, 1, 3));   // [hd, t, h]
        ggml_tensor * w = ggml_mul_mat(c, k_t, qu_t);                        // [j, t, h]
        w = ggml_scale(c, w, scale);
        w = ggml_add(c, w, bds);                // + per-head bd mask
        w = ggml_add(c, w, ggml_repeat(c, ggml_reshape_3d(c, a_mmask, T, T, 1), w));   // HF key/row mask
        w = ggml_soft_max(c, w);                                              // over j (ne0)
        ggml_tensor * v_t = ggml_cont(c, ggml_permute(c, v3, 1, 2, 0, 3));   // [j, d, h]
        ggml_tensor * av = ggml_mul_mat(c, v_t, w);                          // [d, t, h]
        ggml_tensor * attn = ggml_mul_mat(c, L.att_o,
            ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, av, 0, 2, 1, 3)), D, T));
        x = ggml_add(c, x, attn);
        if (il == 0) ggml_set_name(x, "a_l0_attnout");
        if (il > 0) { char bn[32]; snprintf(bn, sizeof(bn), "a_l%d_attnres", il); ggml_set_name(x, bn); }

        // conv module: pw1 -> GLU -> depthwise k9 (BN folded) -> silu -> pw2
        h = ggml_norm(c, x, 1e-5f);
        h = ggml_add(c, ggml_mul(c, h, L.ln_conv_w), L.ln_conv_b);
        ggml_tensor * g = ggml_mul_mat(c, L.conv_pw1, h);              // [2D, T]
        ggml_tensor * ga = ggml_view_2d(c, g, D, T, g->nb[1], 0);
        ggml_tensor * gb2 = ggml_view_2d(c, g, D, T, g->nb[1], (size_t)D * g->nb[0]);
        g = ggml_mul(c, ga, ggml_sigmoid(c, gb2));
        g = ggml_mul(c, g, ggml_repeat(c, a_rmask, g));   // HF: zero fully-masked rows before the dw conv
        // zero-pad the time axis 4 left / 4 right, then the 9 shifted taps
        ggml_tensor * zpad = ggml_scale(c,
            ggml_cont(c, ggml_view_2d(c, g, D, conv_k / 2, g->nb[1], 0)), 0.0f);
        ggml_tensor * xext = ggml_concat(c, ggml_concat(c, zpad, g, 1), zpad, 1);
        ggml_tensor * dwsum = nullptr;
        for (int k = 0; k < conv_k; k++) {
            ggml_tensor * tap = ggml_view_2d(c, xext, D, T, xext->nb[1],
                                             (size_t)k * xext->nb[1]);
            ggml_tensor * wk = ggml_view_1d(c, L.conv_dw_fold, D,
                                            (size_t)k * L.conv_dw_fold->nb[1]);
            ggml_tensor * term = ggml_mul(c, tap, wk);   // b (wk) broadcasts into a
            dwsum = dwsum ? ggml_add(c, dwsum, term) : term;
        }
        if (il == 0) ggml_set_name(g, "a_l0_glu");
        if (il == 0) ggml_set_name(dwsum, "a_l0_dwsum");
        g = ggml_silu(c, ggml_add(c, dwsum, L.conv_dw_bias));
        g = ggml_mul_mat(c, L.conv_pw2, g);
        x = ggml_add(c, x, g);
        if (il == 0) ggml_set_name(x, "a_l0_convout");
        if (il > 0) { char bn[32]; snprintf(bn, sizeof(bn), "a_l%d_convout", il); ggml_set_name(x, bn); }

        // macaron FFN 2 (x0.5) + norm_out (the block output IS normed)
        h = ggml_norm(c, x, 1e-5f);
        h = ggml_add(c, ggml_mul(c, h, L.ln_ff2_w), L.ln_ff2_b);
        h = ggml_mul_mat(c, L.ff2_w2, ggml_silu(c, ggml_mul_mat(c, L.ff2_w1, h)));
        x = ggml_add(c, x, ggml_scale(c, h, 0.5f));
        x = ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), L.ln_out_w), L.ln_out_b);
        if (il == 0) ggml_set_name(x, "a_l0_out");
    }

    // projector: RMSNorm -> Linear[4096, 1024] -> relu^2 -> Linear[2688, 4096]
    ggml_set_name(x, "a_enc_out");              // [D, T] pre-projection truth
    ggml_tensor * p = ggml_rms_norm(c, x, 1e-5f);
    p = ggml_mul(c, p, proj_ln);
    p = ggml_sqr(c, ggml_relu(c, ggml_mul_mat(c, proj_fc1, p)));
    a_feats = ggml_mul_mat(c, proj_fc2, p);                            // [2688, T]
    ggml_set_name(a_feats, "a_final_embeds");
    ggml_build_forward_expand(agf, a_feats);

    if (!dump_dir.empty()) {
        static const char * const dn[] = { "a_sub_in", "a_pos_in", "a_sub",
                                           "a_l0_ff1", "a_l0_ain", "a_l0_q3", "a_l0_qv4",
                                           "a_l0_qv", "a_l0_rl4", "a_l0_bdraw",
                                           "a_l0_glu", "a_l0_dwsum",
                                           "a_l0_bd",
                                           "a_l0_attnout", "a_l0_convout",
                                           "a_l1_attnres", "a_l2_attnres",
                                           "a_l3_attnres", "a_l1_convout",
                                           "a_l2_convout", "a_l3_convout",
                                           "a_l0_out", "a_enc_out",
                                           "a_final_embeds", nullptr };
        int ndn = 0;
        while (dn[ndn]) ndn++;
        std::vector<char *> names;
        for (int i = 0; dn[i]; i++) names.push_back(const_cast<char *>(dn[i]));
        static char extra[40][2][32];
        for (int il = 4; il < n_layer; il++) {
            snprintf(extra[il][0], sizeof(extra[il][0]), "a_l%d_attnres", il);
            snprintf(extra[il][1], sizeof(extra[il][1]), "a_l%d_convout", il);
            names.push_back(extra[il][0]); names.push_back(extra[il][1]);
        }
        names.push_back(nullptr);
        sink_names.assign(names.begin(), names.end() - 1);   // drop the nullptr terminator
        n_sink = (int)sink_names.size();
        vqa::dump::mark_sinks(c, agf, (const char * const *)names.data());
        vqa::dump::finalize_sinks(c, agf, backend);
    }
    if (!ggml_gallocr_alloc_graph(allocr, agf)) {
        fprintf(stderr, "nemotronomni: audio graph allocation failed\n");
        return false;
    }
    a_t = T;
    a_t_valid_built = a_t_valid;
    return true;
}

bool ParakeetTower::encode(ggml_backend_t backend, const float * pcm, int n_samples,
                           std::vector<float> & embeds, int * n_tok, const char * dump) {
    dump_dir = dump ? dump : "";
    this->backend = backend;
    g_dump_env = !dump_dir.empty();
    snprintf(g_dump_dir, sizeof(g_dump_dir), "%s", dump_dir.c_str());
    if (g_dump_env) std::filesystem::create_directories(dump_dir);
    std::vector<float> mel, sub, pos;
    int mel_T = 0;
    const int T = frontend(pcm, n_samples, mel, sub, pos, &mel_T);
    if (!agf || a_t != T || a_t_valid != a_t_valid_built) {
        if (!build_graph(T)) return false;
    }
    if (!dump_dir.empty()) {
        std::filesystem::create_directories(dump_dir);
        FILE * f = nullptr;
        char pth[512];
        auto raw_dump = [&](const char * tag, const std::vector<float> & v, int rows, int cc) {
            snprintf(pth, sizeof(pth), "%s/cpp_%s.bin", dump_dir.c_str(), tag);
            if ((f = fopen(pth, "wb")) != nullptr) {
                fwrite(v.data(), 4, v.size(), f);
                fclose(f);
                snprintf(pth, sizeof(pth), "%s/cpp_%s.shape", dump_dir.c_str(), tag);
                if ((f = fopen(pth, "w")) != nullptr) { fprintf(f, "%d %d\n", rows, cc); fclose(f); }
            }
        };
        raw_dump("a_mel", mel, mel_T, n_mel);
        raw_dump("a_flat", sub, T, sub_ch * freq_final);   // host conv flatten (pre-linear)
        raw_dump("a_pos", pos, 2 * T - 1, d_model);
    }
    const auto t0 = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(a_in, sub.data(), 0, sizeof(float) * sub.size());
    ggml_backend_tensor_set(a_pos_in, pos.data(), 0, sizeof(float) * pos.size());
    {   // HF pad masks: -1e30 at masked (row,key) pairs, 0/1 row mask
        const int nv = a_t_valid;
        std::vector<float> mm((size_t)T * T, 0.0f), rm((size_t)T, 1.0f);
        for (int t = 0; t < T; t++) {
            if (t >= nv) rm[(size_t)t] = 0.0f;
            float * row = &mm[(size_t)t * T];
            for (int j = 0; j < T; j++)
                if (t >= nv || j >= nv) row[j] = -1e30f;
        }
        ggml_backend_tensor_set(a_mmask, mm.data(), 0, sizeof(float) * mm.size());
        ggml_backend_tensor_set(a_rmask, rm.data(), 0, sizeof(float) * rm.size());
    }
    if (ggml_backend_graph_compute(backend, agf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "nemotronomni: audio compute failed\n");
        return false;
    }
    if (!dump_dir.empty()) {
        for (int i = 0; i < n_sink; i++) vqa::dump::write(agf, dump_dir.c_str(), sink_names[(size_t)i]);
    }
    embeds.resize((size_t)T * (size_t)llm_dim);
    ggml_backend_tensor_get(a_feats, embeds.data(), 0, sizeof(float) * embeds.size());
    *n_tok = T;
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "nemotronomni audio: %d samples -> %d mel frames -> %d tokens (%.0f ms)\n",
            n_samples, mel_T, T, ms);
    return true;
}

}  // namespace nemo
