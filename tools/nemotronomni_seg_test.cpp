// nemotronomni_seg_test.cpp - wave-3a unit parity driver for the
// Nemotron-3-Nano-Omni text chain (nemotronomni backend).
//
// Builds seeded-RNG synthetic weights at scaled-down-but-proportional shapes,
// runs each branch through its real ggml graph segment, and writes every
// tensor the numpy reference needs (weights + activations + persistent
// states) as cpp_<name>.bin/.shape into a dump directory:
//
//   mamba  : prefill T=8 in one graph, then 4 T=1 decode steps (recurrent
//            state chaining through the persistent conv/ssm slots)
//   attn   : NoPE GQA prefill T=6 + 3 decode steps into an F16 KV cache
//   moe    : sigmoid+bias routing, top-3 of 8, relu^2 experts + shared
//   chain  : norm->Mamba2 / norm->attn / norm->MoE layers plus final norm
//            and untied lm_head (the 52-layer dispatch in miniature)
//
// Compare with nemotron3omni-port/3a/parity_nemotronomni_seg.py (numpy
// reference: step-recursion AND chunk-SSD Mamba2, naive GQA attention, HF
// router math).
//
// Build (repo root):
//   cmake --build build --config Release --target nemotronomni-seg-test
// Run:
//   build/Release/nemotronomni-seg-test.exe [dump_dir]
//     (default: D:/S2TT/nemotron3omni-port/3a/dump)
//   then: python D:/S2TT/nemotron3omni-port/3a/parity_nemotronomni_seg.py
#include "nemotronomni_text.hpp"
#include "core/engine/ggml_dump.hpp"
#include "core/engine/attention.hpp"
#include "core/test/seg_test.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

// ------------------------------------------------------------------ mamba
// D=96, 8 heads x head_dim 64 (real head shape), n_groups 2, d_state 128
// (real), conv k=4 (real); hidden and group count scaled down.
static void scenario_mamba(ggml_backend_t backend, ggml_gallocr_t allocr,
                           const char * dir) {
    const int D = 96, NH = 8, HD = 64, G = 2, S = 128, K = 4;
    const int I = NH * HD, C = I + 2 * G * S;                 // 512, 1024
    const int DIN = I + C + NH;                               // 1544
    const float EPS = 1e-5f, DT_MIN = 1e-3f;

    WW w;
    w.init(backend, 24);
    ggml_tensor * in_proj  = w.make("wm_inproj", D, DIN);
    ggml_tensor * conv1d   = w.make("wm_conv", K, C);
    ggml_tensor * conv1d_b = w.make("wm_convb", C);
    ggml_tensor * dt_bias  = w.make("wm_dtb", NH);
    ggml_tensor * avec     = w.make("wm_a", 1, NH);
    ggml_tensor * dvec     = w.make("wm_d", NH);
    ggml_tensor * norm     = w.make("wm_norm", I / G, G);
    ggml_tensor * out_proj = w.make("wm_out", I, D);
    auto finchk = [&](Run & r, const char * tag) {
        if (!getenv("SEG_NOSINK")) return;
        ggml_tensor * t = vqa::dump::lookup(r.gf, tag);
        if (!t || !t->buffer) { fprintf(stderr, "nosink %s: no buffer\n", tag); return; }
        std::vector<float> v((size_t)ggml_nelements(t));
        ggml_backend_tensor_get(t, v.data(), 0, sizeof(float) * v.size());
        size_t bad = 0;
        for (float f : v) if (!std::isfinite(f)) bad++;
        fprintf(stderr, "nosink %s: nonfinite %zu / %zu\n", tag, bad, v.size());
    };
    ggml_tensor * conv_st  = w.make("ms_conv", K - 1, C);
    ggml_tensor * ssm_st   = w.make("ms_ssm", S, HD, NH);
    ggml_tensor * f_one    = w.make("wm_one", 1);
    ggml_tensor * ids      = w.make("wm_ids", 1, 1, 1, GGML_TYPE_I32);
    w.alloc();
    w.fill(in_proj,  rnd((size_t)DIN * D, 0.3f));
    w.fill(conv1d,   rnd((size_t)K * C, 0.5f));
    w.fill(conv1d_b, rnd(C, 0.5f));
    {
        // official init: dt0 ~ log-spaced U[time_step_min, time_step_max],
        // dt_bias = inverse softplus (modeling_nemotron_h.py:1086-1095)
        std::vector<float> b(NH);
        for (int h = 0; h < NH; h++) {
            const float u = 0.5f * (g_rng.u() + 1.0f);        // [0, 1]
            const float dt0 = 0.001f * std::pow(100.0f, u);   // U[1e-3, 0.1]
            b[(size_t)h] = std::log(std::expm1(dt0));
        }
        w.fill(dt_bias, b);
    }
    {
        std::vector<float> av(NH);
        for (int h = 0; h < NH; h++)
            av[(size_t)h] = -std::exp(std::log(1.0f + (float)h) + 0.1f * g_rng.u());
        w.fill(avec, av);
    }
    w.fill(dvec, rnd(NH, 1.0f));
    {
        std::vector<float> nv((size_t)I);
        for (size_t i = 0; i < nv.size(); i++) nv[i] = 1.0f + 0.2f * g_rng.u();
        w.fill(norm, nv);
    }
    w.fill(out_proj, rnd((size_t)D * I, 0.2f));
    w.fill(f_one, { 1.0f });
    { int32_t z = 0; ggml_backend_tensor_set(ids, &z, 0, sizeof(int32_t)); }

    dt(in_proj, dir, "wm_inproj");  dt(conv1d,  dir, "wm_conv");
    dt(conv1d_b, dir, "wm_convb");  dt(dt_bias, dir, "wm_dtb");
    dt(avec, dir, "wm_a");          dt(dvec,    dir, "wm_d");
    dt(norm, dir, "wm_norm");       dt(out_proj, dir, "wm_out");

    nemo::Mamba2W mw;
    mw.in_proj = in_proj; mw.conv1d = conv1d; mw.conv1d_b = conv1d_b;
    mw.dt_bias = dt_bias; mw.a = avec; mw.d = dvec; mw.norm = norm;
    mw.out_proj = out_proj;
    nemo::Mamba2State st{ conv_st, ssm_st };

    // ---- prefill: T=8 in one graph
    const int T = 8;
    {
        Run r; r.backend = backend; r.allocr = allocr; r.dir = dir;
        r.begin();
        ggml_tensor * l = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, D, T);
        ggml_set_name(l, "m0_in");
        ggml_set_input(l);
        ggml_tensor * out = nemo::mamba2_branch(r.ctx, r.gf, mw, st, l, ids,
                                                EPS, f_one, DT_MIN, "m0");
        (void)out;
        r.sinks = { "m0_in", "m0_inproj", "m0_dt", "m0_xbc",
                    "m0_yd", "m0_yn", "m0_out" };
        ggml_build_forward_expand(r.gf, out);   // production expands its tail
        if (!r.alloc()) exit(1);
        ggml_backend_tensor_set(l, rnd((size_t)D * T, 1.0f).data(), 0,
                                sizeof(float) * D * T);
        if (!r.compute()) exit(1);
        finchk(r, "m0_out");
        r.dump_all();
        r.end();
        dt(conv_st, dir, "m0_state_conv");
        dt(ssm_st, dir, "m0_state_ssm");
    }

    // ---- 4 decode steps: T=1 graphs, states persist in place
    for (int step = 0; step < 4; step++) {
        char nm[64];
        Run r; r.backend = backend; r.allocr = allocr; r.dir = dir;
        r.begin();
        ggml_tensor * l = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, D, 1);
        snprintf(nm, sizeof(nm), "m1_s%d_in", step);
        ggml_set_name(l, nm);
        ggml_set_input(l);
        ggml_tensor * out = nemo::mamba2_branch(r.ctx, r.gf, mw, st, l, ids,
                                                EPS, f_one, DT_MIN, nullptr);
        char nin[64], nout[64];
        snprintf(nin, sizeof(nin), "m1_s%d_in", step);
        snprintf(nout, sizeof(nout), "m1_s%d_out", step);
        ggml_set_name(out, nout);
        r.sinks = { nout, nin };
        ggml_build_forward_expand(r.gf, out);   // production expands its tail
        if (!r.alloc()) exit(1);
        ggml_backend_tensor_set(l, rnd((size_t)D, 1.0f).data(), 0,
                                sizeof(float) * D);
        if (!r.compute()) exit(1);
        finchk(r, nout);
        r.dump_all();
        r.end();
        snprintf(nm, sizeof(nm), "m1_s%d_state_conv", step);
        dt(conv_st, dir, nm);
        snprintf(nm, sizeof(nm), "m1_s%d_state_ssm", step);
        dt(ssm_st, dir, nm);
    }
    fprintf(stderr, "mamba scenario done\n");
}

// ------------------------------------------------------------------ attn
// GQA 8:2 (ratio 4:1), head_dim 16, NoPE; F16 KV cache like production.
static void scenario_attn(ggml_backend_t backend, ggml_gallocr_t allocr,
                          const char * dir) {
    const int D = 128, NH = 8, NKV = 2, HD = 16, S = 16;
    const int QD = NH * HD, KVD = NKV * HD;

    WW w;
    w.init(backend, 16);
    ggml_tensor * wq = w.make("aw_q", D, QD);
    ggml_tensor * wk = w.make("aw_k", D, KVD);
    ggml_tensor * wv = w.make("aw_v", D, KVD);
    ggml_tensor * wo = w.make("aw_o", QD, D);
    ggml_tensor * kc = w.make("ak_c", KVD, S, 1, GGML_TYPE_F16);
    ggml_tensor * vc = w.make("av_c", KVD, S, 1, GGML_TYPE_F16);
    w.alloc();
    w.fill(wq, rnd((size_t)QD * D, 0.3f));
    w.fill(wk, rnd((size_t)KVD * D, 0.3f));
    w.fill(wv, rnd((size_t)KVD * D, 0.3f));
    w.fill(wo, rnd((size_t)D * QD, 0.2f));

    dt(wq, dir, "aw_q"); dt(wk, dir, "aw_k");
    dt(wv, dir, "aw_v"); dt(wo, dir, "aw_o");

    nemo::AttnW aw;
    aw.q = wq; aw.k = wk; aw.v = wv; aw.o = wo;

    // ---- prefill: T=6, positions 0..5
    const int T = 6;
    {
        Run r; r.backend = backend; r.allocr = allocr; r.dir = dir;
        r.begin();
        ggml_tensor * l = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, D, T);
        ggml_set_name(l, "a0_in");
        ggml_set_input(l);
        ggml_tensor * pos = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, T);
        ggml_set_input(pos);
        ggml_tensor * mask = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F16, S, T);
        ggml_set_input(mask);
        ggml_tensor * out = nemo::attn_branch(r.ctx, r.gf, aw, kc, vc, l, mask,
                                              pos, nullptr, 84, 1e4f, S,
                                              NH, NKV, HD, "a0");
        (void)out;
        r.sinks = { "a0_in", "a0_q", "a0_q_pre_fa", "a0_k_pre_fa",
                    "a0_attn", "a0_out" };
        if (!r.alloc()) exit(1);
        ggml_backend_tensor_set(l, rnd((size_t)D * T, 1.0f).data(), 0,
                                sizeof(float) * D * T);
        std::vector<int> pv(T);
        for (int i = 0; i < T; i++) pv[i] = i;
        ggml_backend_tensor_set(pos, pv.data(), 0, sizeof(int) * T);
        vqa::set_causal_mask(mask, 0, T);
        if (!r.compute()) exit(1);
        r.dump_all();
        r.end();
        dt(kc, dir, "a0_kcache");
        dt(vc, dir, "a0_vcache");
    }

    // ---- 3 decode steps: positions 6..8
    for (int step = 0; step < 3; step++) {
        const int p = T + step;
        char nm[64];
        Run r; r.backend = backend; r.allocr = allocr; r.dir = dir;
        r.begin();
        ggml_tensor * l = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, D, 1);
        snprintf(nm, sizeof(nm), "a1_s%d_in", step);
        ggml_set_name(l, nm);
        ggml_set_input(l);
        ggml_tensor * pos = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, 1);
        ggml_set_input(pos);
        ggml_tensor * mask = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F16, S, 1);
        ggml_set_input(mask);
        char ndbg[16];
        snprintf(ndbg, sizeof(ndbg), "a1_s%d", step);
        ggml_tensor * out = nemo::attn_branch(r.ctx, r.gf, aw, kc, vc, l, mask,
                                              pos, nullptr, 84, 1e4f, S,
                                              NH, NKV, HD, ndbg);
        char nout[64];
        snprintf(nout, sizeof(nout), "a1_s%d_out", step);
        ggml_set_name(out, nout);
        char nat[64];
        snprintf(nat, sizeof(nat), "a1_s%d_attn", step);
        r.sinks = { nout, nm, nat };
        if (!r.alloc()) exit(1);
        ggml_backend_tensor_set(l, rnd((size_t)D, 1.0f).data(), 0,
                                sizeof(float) * D);
        ggml_backend_tensor_set(pos, &p, 0, sizeof(int));
        vqa::set_causal_mask(mask, p, 1);
        if (!r.compute()) exit(1);
        r.dump_all();
        r.end();
        snprintf(nm, sizeof(nm), "a1_s%d_kcache", step);
        dt(kc, dir, nm);
        snprintf(nm, sizeof(nm), "a1_s%d_vcache", step);
        dt(vc, dir, nm);
    }
    fprintf(stderr, "attn scenario done\n");
}

// ------------------------------------------------------------------ moe
// 8 experts top-3 (selection logic identical at 128/6), relu^2 experts,
// shared expert on top.
static void scenario_moe(ggml_backend_t backend, ggml_gallocr_t allocr,
                         const char * dir) {
    const int D = 64, E = 8, KU = 3, I = 24, ISH = 32, T = 4;

    WW w;
    w.init(backend, 16);
    ggml_tensor * router = w.make("ew_rt", D, E);
    ggml_tensor * probs_b = w.make("ew_b", E);
    ggml_tensor * up = w.make("ew_up", D, I, E);
    ggml_tensor * down = w.make("ew_dn", I, D, E);
    ggml_tensor * shu = w.make("ew_shu", D, ISH);
    ggml_tensor * shd = w.make("ew_shd", ISH, D);
    ggml_tensor * f_eps = w.make("ew_eps", 1);
    w.alloc();
    w.fill(router, rnd((size_t)D * E, 0.5f));
    w.fill(probs_b, rnd(E, 0.3f));
    w.fill(up, rnd((size_t)D * I * E, 0.3f));
    w.fill(down, rnd((size_t)I * D * E, 0.3f));
    w.fill(shu, rnd((size_t)D * ISH, 0.3f));
    w.fill(shd, rnd((size_t)D * ISH, 0.3f));
    w.fill(f_eps, { 1e-20f });

    dt(router, dir, "ew_rt");    dt(probs_b, dir, "ew_b");
    dt(up, dir, "ew_up");        dt(down, dir, "ew_dn");
    dt(shu, dir, "ew_shu");      dt(shd, dir, "ew_shd");

    nemo::MoEW mw;
    mw.router = router; mw.probs_b = probs_b; mw.up_exps = up;
    mw.down_exps = down; mw.shexp_up = shu; mw.shexp_down = shd;
    mw.f_eps = f_eps;

    Run r; r.backend = backend; r.allocr = allocr; r.dir = dir;
    r.begin();
    ggml_tensor * m = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, D, T);
    ggml_set_name(m, "e0_in");
    ggml_set_input(m);
    ggml_tensor * out = nemo::moe_branch(r.ctx, mw, m, KU, 2.5f, "e0");
    (void)out;
    r.sinks = { "e0_in", "e0_rlogits", "e0_sel", "e0_w8",
                "e0_routed", "e0_shexp", "e0_out" };
    r.raw = { "e0_sel" };
    if (!r.alloc()) exit(1);
    ggml_backend_tensor_set(m, rnd((size_t)D * T, 1.0f).data(), 0,
                            sizeof(float) * D * T);
    if (!r.compute()) exit(1);
    r.dump_all();
    dt_i32(vqa::dump::lookup(r.gf, "e0_sel"), dir, "e0_sel");
    r.end();
    fprintf(stderr, "moe scenario done\n");
}

// ------------------------------------------------------------------ chain
// norm->Mamba2 / norm->attn / norm->MoE + final norm + untied lm_head, the
// 52-layer dispatch in miniature (fresh zero states, one graph).
static void scenario_chain(ggml_backend_t backend, ggml_gallocr_t allocr,
                           const char * dir) {
    const int D = 128, V = 48, T = 5, S = 16;
    const int NH = 8, NKV = 2, HD = 16;         // attention
    const int MH = 8, MHD = 64, G = 2, SS = 128, KC = 4;   // mamba
    const int I = MH * MHD, C = I + 2 * G * SS; // 512, 1024
    const int DIN = I + C + MH;                 // 1544
    const int E = 8, KU = 3, IEXP = 32, ISH = 48;
    const float EPS = 1e-5f;

    WW w;
    w.init(backend, 48);
    ggml_tensor * emb   = w.make("w_emb", D, V);
    ggml_tensor * n0    = w.make("w_n0", D);
    ggml_tensor * n1    = w.make("w_n1", D);
    ggml_tensor * n2    = w.make("w_n2", D);
    // layer 0: mamba
    ggml_tensor * inproj = w.make("w_l0_inproj", D, DIN);
    ggml_tensor * conv   = w.make("w_l0_conv", KC, C);
    ggml_tensor * convb  = w.make("w_l0_convb", C);
    ggml_tensor * dtb    = w.make("w_l0_dtb", MH);
    ggml_tensor * avec   = w.make("w_l0_a", 1, MH);
    ggml_tensor * dvec   = w.make("w_l0_d", MH);
    ggml_tensor * mnorm  = w.make("w_l0_mnorm", I / G, G);
    ggml_tensor * mout   = w.make("w_l0_out", I, D);
    // layer 1: attention
    ggml_tensor * wq = w.make("w_l1_q", D, NH * HD);
    ggml_tensor * wk = w.make("w_l1_k", D, NKV * HD);
    ggml_tensor * wv = w.make("w_l1_v", D, NKV * HD);
    ggml_tensor * wo = w.make("w_l1_o", NH * HD, D);
    // layer 2: moe
    ggml_tensor * rt  = w.make("w_l2_rt", D, E);
    ggml_tensor * pb  = w.make("w_l2_b", E);
    ggml_tensor * up  = w.make("w_l2_up", D, IEXP, E);
    ggml_tensor * dn  = w.make("w_l2_dn", IEXP, D, E);
    ggml_tensor * shu = w.make("w_l2_shu", D, ISH);
    ggml_tensor * shd = w.make("w_l2_shd", ISH, D);
    ggml_tensor * fn  = w.make("w_fn", D);
    ggml_tensor * head = w.make("w_head", D, V);
    ggml_tensor * conv_st = w.make("cw_conv", KC - 1, C);
    ggml_tensor * ssm_st  = w.make("cw_ssm", SS, MHD, MH);
    ggml_tensor * kc = w.make("ck_c", NKV * HD, S, 1, GGML_TYPE_F16);
    ggml_tensor * vc = w.make("cv_c", NKV * HD, S, 1, GGML_TYPE_F16);
    ggml_tensor * f_one = w.make("w_one", 1);
    ggml_tensor * f_eps = w.make("w_eps", 1);
    ggml_tensor * ids = w.make("c_ids", 1, 1, 1, GGML_TYPE_I32);
    w.alloc();
    w.fill(emb, rnd((size_t)D * V, 1.0f));
    w.fill(n0, rnd(D, 1.0f));
    w.fill(n1, rnd(D, 1.0f));
    w.fill(n2, rnd(D, 1.0f));
    w.fill(inproj, rnd((size_t)DIN * D, 0.3f));
    w.fill(conv, rnd((size_t)KC * C, 0.5f));
    w.fill(convb, rnd(C, 0.5f));
    {
        std::vector<float> b(MH);
        for (int h = 0; h < MH; h++) {
            const float u = 0.5f * (g_rng.u() + 1.0f);
            const float dt0 = 0.001f * std::pow(100.0f, u);
            b[(size_t)h] = std::log(std::expm1(dt0));
        }
        w.fill(dtb, b);
    }
    {
        std::vector<float> av(MH);
        for (int h = 0; h < MH; h++)
            av[(size_t)h] = -std::exp(std::log(1.0f + (float)h) + 0.1f * g_rng.u());
        w.fill(avec, av);
    }
    w.fill(dvec, rnd(MH, 1.0f));
    {
        std::vector<float> nv((size_t)I);
        for (size_t i = 0; i < nv.size(); i++) nv[i] = 1.0f + 0.2f * g_rng.u();
        w.fill(mnorm, nv);
    }
    w.fill(mout, rnd((size_t)D * I, 0.2f));
    w.fill(wq, rnd((size_t)NH * HD * D, 0.3f));
    w.fill(wk, rnd((size_t)NKV * HD * D, 0.3f));
    w.fill(wv, rnd((size_t)NKV * HD * D, 0.3f));
    w.fill(wo, rnd((size_t)D * NH * HD, 0.2f));
    w.fill(rt, rnd((size_t)D * E, 0.5f));
    w.fill(pb, rnd(E, 0.3f));
    w.fill(up, rnd((size_t)D * IEXP * E, 0.3f));
    w.fill(dn, rnd((size_t)IEXP * D * E, 0.3f));
    w.fill(shu, rnd((size_t)D * ISH, 0.3f));
    w.fill(shd, rnd((size_t)D * ISH, 0.3f));
    w.fill(fn, rnd(D, 1.0f));
    w.fill(head, rnd((size_t)D * V, 0.2f));
    w.fill(f_one, { 1.0f });
    w.fill(f_eps, { 1e-20f });
    { int32_t z = 0; ggml_backend_tensor_set(ids, &z, 0, sizeof(int32_t)); }

    dt(emb, dir, "w_emb");
    dt(n0, dir, "w_n0"); dt(n1, dir, "w_n1"); dt(n2, dir, "w_n2");
    dt(inproj, dir, "w_l0_inproj"); dt(conv, dir, "w_l0_conv");
    dt(convb, dir, "w_l0_convb");   dt(dtb, dir, "w_l0_dtb");
    dt(avec, dir, "w_l0_a");        dt(dvec, dir, "w_l0_d");
    dt(mnorm, dir, "w_l0_mnorm");   dt(mout, dir, "w_l0_out");
    dt(wq, dir, "w_l1_q"); dt(wk, dir, "w_l1_k");
    dt(wv, dir, "w_l1_v"); dt(wo, dir, "w_l1_o");
    dt(rt, dir, "w_l2_rt"); dt(pb, dir, "w_l2_b");
    dt(up, dir, "w_l2_up"); dt(dn, dir, "w_l2_dn");
    dt(shu, dir, "w_l2_shu"); dt(shd, dir, "w_l2_shd");
    dt(fn, dir, "w_fn"); dt(head, dir, "w_head");

    nemo::Mamba2W mw;
    mw.in_proj = inproj; mw.conv1d = conv; mw.conv1d_b = convb;
    mw.dt_bias = dtb; mw.a = avec; mw.d = dvec; mw.norm = mnorm;
    mw.out_proj = mout;
    nemo::Mamba2State mst{ conv_st, ssm_st };
    nemo::AttnW aw;
    aw.q = wq; aw.k = wk; aw.v = wv; aw.o = wo;
    nemo::MoEW ew;
    ew.router = rt; ew.probs_b = pb; ew.up_exps = up; ew.down_exps = dn;
    ew.shexp_up = shu; ew.shexp_down = shd; ew.f_eps = f_eps;

    Run r; r.backend = backend; r.allocr = allocr; r.dir = dir;
    r.begin();
    ggml_tensor * tok = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, T);
    ggml_set_input(tok);
    ggml_tensor * pos = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, T);
    ggml_set_input(pos);
    ggml_tensor * mask = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F16, S, T);
    ggml_set_input(mask);

    ggml_tensor * x = ggml_get_rows(r.ctx, emb, tok);
    ggml_set_name(x, "emb");
    // layer 0: Mamba2
    ggml_tensor * l0 = ggml_rms_norm(r.ctx, x, EPS);
    l0 = ggml_mul(r.ctx, l0, n0);
    ggml_set_name(l0, "c0_in");
    ggml_tensor * b0 = nemo::mamba2_branch(r.ctx, r.gf, mw, mst, l0, ids,
                                           EPS, f_one, 1e-3f, nullptr);
    ggml_set_name(b0, "c0_out");
    x = ggml_add(r.ctx, x, b0);
    // layer 1: NoPE attention
    ggml_tensor * l1 = ggml_rms_norm(r.ctx, x, EPS);
    l1 = ggml_mul(r.ctx, l1, n1);
    ggml_set_name(l1, "c1_in");
    ggml_tensor * b1 = nemo::attn_branch(r.ctx, r.gf, aw, kc, vc, l1, mask,
                                         pos, nullptr, 84, 1e4f, S,
                                         NH, NKV, HD, nullptr);
    ggml_set_name(b1, "c1_out");
    x = ggml_add(r.ctx, x, b1);
    // layer 2: MoE
    ggml_tensor * l2 = ggml_rms_norm(r.ctx, x, EPS);
    l2 = ggml_mul(r.ctx, l2, n2);
    ggml_set_name(l2, "c2_in");
    ggml_tensor * b2 = nemo::moe_branch(r.ctx, ew, l2, KU, 2.5f, nullptr);
    ggml_set_name(b2, "c2_out");
    x = ggml_add(r.ctx, x, b2);

    x = ggml_rms_norm(r.ctx, x, EPS);
    x = ggml_mul(r.ctx, x, fn);
    ggml_set_name(x, "t_final_norm");
    ggml_tensor * last = ggml_view_2d(r.ctx, x, D, 1, x->nb[1],
                                      (size_t)(T - 1) * x->nb[1]);
    ggml_tensor * logits = ggml_mul_mat(r.ctx, head, last);
    ggml_set_name(logits, "logits");
    ggml_build_forward_expand(r.gf, logits);

    r.sinks = { "emb", "c0_in", "c0_out", "c1_in", "c1_out",
                "c2_in", "c2_out", "t_final_norm", "logits" };
    if (!r.alloc()) exit(1);
    {
        const int toks[T] = { 3, 17, 9, 41, 2 };
        std::vector<int> pv(T);
        for (int i = 0; i < T; i++) pv[i] = i;
        ggml_backend_tensor_set(tok, toks, 0, sizeof(int) * T);
        ggml_backend_tensor_set(pos, pv.data(), 0, sizeof(int) * T);
        vqa::set_causal_mask(mask, 0, T);
    }
    if (!r.compute()) exit(1);
    if (getenv("SEG_NOSINK")) {
        // quick finiteness probe of the no-sink (production-style) layout;
        // mid-chain reads may hit reused buffers - the tail tensors are the
        // meaningful ones
        auto chk = [&](const char * tag) {
            ggml_tensor * t = vqa::dump::lookup(r.gf, tag);
            if (!t) { fprintf(stderr, "nosink %s: not found\n", tag); return; }
            std::vector<float> v((size_t)ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, sizeof(float) * v.size());
            size_t bad = 0;
            for (float f : v) if (!std::isfinite(f)) bad++;
            fprintf(stderr, "nosink %s: nonfinite %zu / %zu\n", tag, bad, v.size());
        };
        chk("c0_out");
        chk("c1_out");
        chk("c2_out");
        chk("t_final_norm");
        chk("logits");
    }
    r.dump_all();
    r.end();
    dt(kc, dir, "c_kcache");
    dt(vc, dir, "c_vcache");
    fprintf(stderr, "chain scenario done\n");
}

int main(int argc, char ** argv) {
    g_rng = Rng(0xA11CEu);   // harness RNG seed - parity-stream contract
    const char * dir = argc > 1 ? argv[1] : "D:/S2TT/nemotron3omni-port/3a/dump";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (getenv("SEG_NTHREADS"))
        ggml_backend_cpu_set_n_threads(backend, 1);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    // SEG_ONLY=chain: the mamba/attn/moe scenario graphs are only fully
    // expanded through their sinks (the dump machinery does the final
    // build_forward_expand), so they cannot run under SEG_NOSINK
    if (getenv("SEG_ONLY") && !strcmp(getenv("SEG_ONLY"), "chain")) {
        scenario_chain(backend, allocr, dir);
    } else {
        scenario_mamba(backend, allocr, dir);
        scenario_attn(backend, allocr, dir);
        scenario_moe(backend, allocr, dir);
        scenario_chain(backend, allocr, dir);
    }
    ggml_gallocr_free(allocr);
    ggml_backend_free(backend);
    fprintf(stderr, "all scenarios written to %s\n", dir);
    return 0;
}
