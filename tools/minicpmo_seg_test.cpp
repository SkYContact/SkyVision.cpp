// minicpmo_seg_test.cpp - component parity driver for the MiniCPM-o 4.5
// backend (synthetic seeded weights, scaled-down-but-proportional shapes):
//
//   rs      : the 64-query resampler branch at D=256/R/8 heads, one 5x8 tile
//             (kv_proj -> ln_kv -> +2D sincos -> fused in_proj q|k|v ->
//              cross-attn -> out_proj -> ln_post -> RIGHT multiply proj)
//   slice   : host-only tables - slice grids, bucketized position ids, the
//             2D sincos builder (numpy recomputes the MiniCPM-V/PIL math)
//   audio   : whisper conv geometry (k3 s1 p1 / k3 s2 p1) + the cumulative
//             1s chunk mask (set_chunk_mask)
//   tts     : the tk. sub-tower in miniature - condition (proj_semantic +
//             L2), 2 llama layers at 96x12q/12kv theta 5e5, prefill + 2 AR
//             steps through the shared llm branches, head_code logits
//   llm     : one Qwen3 layer (per-head q/k norm + NEOX theta 1e6 + GQA 8:2)
//             + final norm + untied lm_head, prefill T=6 + 2 decode steps
//
// Every tensor the numpy reference needs lands as cpp_<name>.bin/.shape in
// the dump directory. Compare with scripts/parity_minicpmo_seg.py.
#include "minicpmo.hpp"
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

// ------------------------------------------------------------------ rs
static void scenario_rs(ggml_backend_t backend, ggml_gallocr_t allocr,
                        const char * dir) {
    // production: D 4096, in 1152, 32 heads, Q 64; scaled 4x down
    const int D = 256, IN = 96, NH = 8, Q = 16, N = 40;   // 5x8 tile
    WW w;
    w.init(backend, 16);
    ggml_tensor * query   = w.make("rs_query", D, Q);
    ggml_tensor * kv_proj = w.make("rs_wkv", IN, D);   // sink "rs_kv" is the graph output
    ggml_tensor * in_proj = w.make("rs_ip", D, 3 * D, 1, GGML_TYPE_F16);
    ggml_tensor * in_b    = w.make("rs_ipb", 3 * D);
    ggml_tensor * out_p   = w.make("rs_op", D, D, 1, GGML_TYPE_F16);
    ggml_tensor * out_b   = w.make("rs_opb", D);
    ggml_tensor * lnq_w   = w.make("rs_lnq_w", D);
    ggml_tensor * lnq_b   = w.make("rs_lnq_b", D);
    ggml_tensor * lnkv_w  = w.make("rs_lnkv_w", D);
    ggml_tensor * lnkv_b  = w.make("rs_lnkv_b", D);
    ggml_tensor * lnp_w   = w.make("rs_lnp_w", D);
    ggml_tensor * lnp_b   = w.make("rs_lnp_b", D);
    ggml_tensor * proj    = w.make("rs_proj", D, D);   // F32, mirrors the
                                                       // production replica;
                                                       // mul_mat src0
    w.alloc();
    w.fill(query,   rnd((size_t)D * Q, 1.0f));
    w.fill(kv_proj, rnd((size_t)D * IN, 0.2f));
    w.fill(in_proj, rnd((size_t)D * 3 * D, 0.08f));
    w.fill(in_b,    rnd(3 * (size_t)D, 0.3f));
    w.fill(out_p,   rnd((size_t)D * D, 0.08f));
    w.fill(out_b,   rnd((size_t)D, 0.3f));
    w.fill(lnq_w,   rnd((size_t)D, 1.0f));
    w.fill(lnq_b,   rnd((size_t)D, 0.2f));
    w.fill(lnkv_w,  rnd((size_t)D, 1.0f));
    w.fill(lnkv_b,  rnd((size_t)D, 0.2f));
    w.fill(lnp_w,   rnd((size_t)D, 1.0f));
    w.fill(lnp_b,   rnd((size_t)D, 0.2f));
    w.fill(proj,    rnd((size_t)D * D, 0.06f));

    dt(query, dir, "rs_query");   dt(kv_proj, dir, "rs_wkv");
    dt(in_proj, dir, "rs_ip");    dt(in_b, dir, "rs_ipb");
    dt(out_p, dir, "rs_op");      dt(out_b, dir, "rs_opb");
    dt(lnq_w, dir, "rs_lnq_w");   dt(lnq_b, dir, "rs_lnq_b");
    dt(lnkv_w, dir, "rs_lnkv_w"); dt(lnkv_b, dir, "rs_lnkv_b");
    dt(lnp_w, dir, "rs_lnp_w");   dt(lnp_b, dir, "rs_lnp_b");
    dt(proj, dir, "rs_proj");

    minicpmo::ResamplerW rw;
    rw.query = query; rw.kv_proj = kv_proj;
    rw.in_proj = in_proj; rw.in_proj_b = in_b;
    rw.out_proj = out_p; rw.out_proj_b = out_b;
    rw.ln_q_w = lnq_w; rw.ln_q_b = lnq_b;
    rw.ln_kv_w = lnkv_w; rw.ln_kv_b = lnkv_b;
    rw.ln_post_w = lnp_w; rw.ln_post_b = lnp_b;
    rw.proj = proj;
    rw.proj_f32 = proj;   // in-graph right multiply (branch output = x@proj)

    // tile input + host sincos (5x8 grid)
    std::vector<float> x = rnd((size_t)IN * N, 1.0f);
    std::vector<int> ids;
    minicpmo::vision_pos_ids(5, 8, 10, ids);
    std::vector<float> pos;
    minicpmo::sincos_pos_2d(5, 8, D, pos);
    dt_f(x, { IN, N }, dir, "rs_x");
    dt_f(std::vector<float>(ids.begin(), ids.end()), { N }, dir, "rs_pos_ids");
    dt_f(pos, { D, N }, dir, "rs_pos");

    Run r; r.backend = backend; r.allocr = allocr; r.dir = dir;
    r.begin();
    ggml_tensor * xt = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, IN, N);
    ggml_set_name(xt, "rs_x");
    ggml_set_input(xt);
    ggml_tensor * pt = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, D, N);
    ggml_set_name(pt, "rs_pos");
    ggml_set_input(pt);
    ggml_tensor * out = minicpmo::resampler_branch(r.ctx, rw, xt, pt, NH, "rs");
    ggml_set_name(out, "rs_out");
    r.sinks = { "rs_kv", "rs_keys", "rs_iq", "rs_ik", "rs_iv", "rs_fa",
                "rs_attn", "rs_lout", "rs_out" };
    if (!r.alloc()) exit(1);
    ggml_backend_tensor_set(xt, x.data(), 0, sizeof(float) * x.size());
    ggml_backend_tensor_set(pt, pos.data(), 0, sizeof(float) * pos.size());
    if (!r.compute()) exit(1);
    {
        // direct readback of the rs FA node (sink-dup A/B)
        ggml_tensor * fan = vqa::dump::lookup(r.gf, "rs_fa");
        std::vector<float> fv((size_t)32 * 8 * 16);
        ggml_backend_tensor_get(fan, fv.data(), 0, sizeof(float) * fv.size());
        FILE * fp = fopen((std::string(dir) + "/cpp_rs_fa_direct.bin").c_str(), "wb");
        if (fp) { fwrite(fv.data(), 4, fv.size(), fp); fclose(fp); }
    }
    r.dump_all();
    r.end();
    fprintf(stderr, "rs scenario done\n");
}

// ------------------------------------------------------------------ slice
static void scenario_slice(const char * dir) {
    // grids over a spread of aspect ratios (numpy recomputes the LPU math)
    struct { int w, h; } sizes[] = {
        { 512, 512 }, { 800, 600 }, { 300, 300 }, { 1000, 300 },
        { 640, 480 }, { 200, 800 }, { 448, 448 }, { 1920, 1080 },
    };
    std::vector<float> grids;    // [w, h, cols, rows(0 when unsliced)] per row
    for (const auto & s : sizes) {
        int cols = 0, rows = 0;
        const bool sliced = minicpmo::slice_grid(s.w, s.h, 9, 448, &cols, &rows);
        grids.push_back((float)s.w);
        grids.push_back((float)s.h);
        grids.push_back((float)cols);
        grids.push_back(sliced ? (float)rows : 0.0f);
    }
    dt_f(grids, { 4, (int64_t)(grids.size() / 4) }, dir, "slice_grids");

    // bucketized ids + sincos for a few tile shapes
    std::vector<float> all_ids, all_sc;
    int64_t n_ids = 0, n_sc_rows = 0;
    struct { int h, w; } tiles[] = { { 32, 32 }, { 23, 45 }, { 5, 8 } };
    for (const auto & t : tiles) {
        std::vector<int> ids;
        minicpmo::vision_pos_ids(t.h, t.w, 70, ids);
        for (int v : ids) all_ids.push_back((float)v);
        n_ids += (int64_t)t.h * t.w;
        std::vector<float> sc;
        minicpmo::sincos_pos_2d(t.h, t.w, 16, sc);   // dim 16 variant for diffing
        all_sc.insert(all_sc.end(), sc.begin(), sc.end());
        n_sc_rows += (int64_t)t.h * t.w;
    }
    dt_f(all_ids, { n_ids }, dir, "slice_pos_ids");
    dt_f(all_sc, { 16, n_sc_rows }, dir, "slice_sincos16");
    fprintf(stderr, "slice scenario done\n");
}

// ------------------------------------------------------------------ audio
static void scenario_audio(ggml_backend_t backend, ggml_gallocr_t allocr,
                           const char * dir) {
    const int C = 32, T = 40, T2 = (T + 1) / 2;
    WW w;
    w.init(backend, 8);
    ggml_tensor * c1 = w.make("a_c1", 3, C, C, GGML_TYPE_F32);   // [K, Cin, Cout]
    ggml_tensor * c2 = w.make("a_c2", 3, C, C, GGML_TYPE_F32);
    w.alloc();
    w.fill(c1, rnd((size_t)3 * C * C, 0.2f));
    w.fill(c2, rnd((size_t)3 * C * C, 0.2f));
    dt(c1, dir, "a_c1");
    dt(c2, dir, "a_c2");

    Run r; r.backend = backend; r.allocr = allocr; r.dir = dir;
    r.begin();
    ggml_tensor * in = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, T, C);
    ggml_set_name(in, "a_in");
    ggml_set_input(in);
    // conv chaining: conv_1d output [T_out, Cout] feeds the next conv_1d
    // directly (both convs pad 1, whisper geometry)
    ggml_tensor * x = ggml_conv_1d(r.ctx, c1, in, 1, 1, 1);
    x = ggml_conv_1d(r.ctx, c2, x, 2, 1, 1);
    ggml_set_name(x, "a_conv2");
    r.sinks = { "a_conv2" };
    if (!r.alloc()) exit(1);
    {
        std::vector<float> in_data = rnd((size_t)T * C, 1.0f);
        ggml_backend_tensor_set(in, in_data.data(), 0, sizeof(float) * T * C);
        dt_f(in_data, { T, C }, dir, "a_in");
    }
    if (!r.compute()) exit(1);
    r.dump_all();
    r.end();

    // pool probe: p [32, 20] -> view [5, 4, 32] -> sum_rows -> *0.2
    {
        Run rp; rp.backend = backend; rp.allocr = allocr; rp.dir = dir;
        rp.begin();
        const int TT = 20, PP = 32;
        ggml_tensor * pp = ggml_new_tensor_2d(rp.ctx, GGML_TYPE_F32, PP, TT);
        ggml_set_name(pp, "pl_p");
        ggml_set_input(pp);
        ggml_tensor * pl = ggml_view_3d(rp.ctx, pp, 5, 4, PP,
                                        sizeof(float) * 5, sizeof(float) * TT, 0);
        pl = ggml_scale(rp.ctx, ggml_sum_rows(rp.ctx, pl), 0.2f);
        ggml_set_name(pl, "pl_out");
        rp.sinks = { "pl_out" };
        if (!rp.alloc()) exit(1);
        {
            std::vector<float> pv((size_t)PP * TT);
            for (size_t i = 0; i < pv.size(); i++) pv[i] = g_rng.u();
            ggml_backend_tensor_set(pp, pv.data(), 0, sizeof(float) * pv.size());
            dt_f(pv, { PP, TT }, dir, "pl_p");
        }
        if (!rp.compute()) exit(1);
        rp.dump_all();
        rp.end();
    }

    // chunk mask: chunk 3 over 10 keys
    WW wm;
    wm.init(backend, 2);
    ggml_tensor * mask = wm.make("a_mask", 10, 10, 1, GGML_TYPE_F16);
    wm.alloc();
    minicpmo::set_chunk_mask(mask, 3);
    dt(mask, dir, "a_chunk_mask");
    (void)T2;
    fprintf(stderr, "audio scenario done\n");
}

// ------------------------------------------------------------------ fa-probe
static void scenario_fa_probe(ggml_backend_t backend, ggml_gallocr_t allocr,
                              const char * dir) {
    // minimal FA probe: dh=4, H=2, n_q=3, n_kv=5, no mask
    const int dh = 32, H = 8, NQ = 16, NKV = 40;
    Run r; r.backend = backend; r.allocr = allocr; r.dir = dir;
    r.begin();
    ggml_tensor * q = ggml_new_tensor_3d(r.ctx, GGML_TYPE_F32, dh, NQ, H);
    ggml_set_name(q, "fa_q");
    ggml_set_input(q);
    ggml_tensor * k32 = ggml_new_tensor_3d(r.ctx, GGML_TYPE_F32, dh, NKV, H);
    ggml_set_name(k32, "fa_k");
    ggml_set_input(k32);
    ggml_tensor * v32 = ggml_new_tensor_3d(r.ctx, GGML_TYPE_F32, dh, NKV, H);
    ggml_set_name(v32, "fa_v");
    ggml_set_input(v32);
    // A/B: FA_KV16=1 uses the production F16 KV-cache dtype, default F32
    const bool kv16 = getenv("FA_KV16") != nullptr;
    ggml_tensor * k = kv16 ? ggml_cast(r.ctx, k32, GGML_TYPE_F16) : k32;
    ggml_tensor * v = kv16 ? ggml_cast(r.ctx, v32, GGML_TYPE_F16) : v32;
    ggml_tensor * fa = ggml_flash_attn_ext(r.ctx, q, k, v, nullptr,
                                           1.0f / std::sqrt((float)dh),
                                           0.0f, 0.0f);
    ggml_set_name(fa, "fa_out");
    r.sinks = { "fa_out" };
    if (!r.alloc()) exit(1);
    std::vector<float> qv((size_t)dh * NQ * H), kv_((size_t)dh * NKV * H),
        vv((size_t)dh * NKV * H);
    for (size_t i = 0; i < qv.size(); i++) qv[i] = g_rng.u();
    for (size_t i = 0; i < kv_.size(); i++) kv_[i] = g_rng.u();
    for (size_t i = 0; i < vv.size(); i++) vv[i] = g_rng.u();
    ggml_backend_tensor_set(q, qv.data(), 0, sizeof(float) * qv.size());
    ggml_backend_tensor_set(k32, kv_.data(), 0, sizeof(float) * kv_.size());
    ggml_backend_tensor_set(v32, vv.data(), 0, sizeof(float) * vv.size());
    fprintf(stderr, "probe: inputs set\n");
    if (!r.compute()) exit(1);
    fprintf(stderr, "probe: computed\n");
    r.dump_all();
    {
        // direct readback of the FA result (independent of the sink dups)
        std::vector<float> direct((size_t)dh * NQ * H);
        ggml_backend_tensor_get(fa, direct.data(), 0, sizeof(float) * direct.size());
        FILE * fp = fopen((std::string(dir) + "/cpp_fa_out_direct.bin").c_str(), "wb");
        if (fp) { fwrite(direct.data(), 4, direct.size(), fp); fclose(fp); }
        fprintf(stderr, "fa direct [h0 t0..2][0..1]: %f %f | %f %f | %f %f\n",
                direct[0], direct[1], direct[4], direct[5], direct[8], direct[9]);
    }
    // inputs as f32 bins too
    dt_f(qv, { dh, NQ, H }, dir, "fa_q");
    dt_f(kv_, { dh, NKV, H }, dir, "fa_k");
    dt_f(vv, { dh, NKV, H }, dir, "fa_v");
    r.end();
    fprintf(stderr, "fa-probe scenario done\n");
}

// ------------------------------------------------------------------ tts
// 2 llama layers at D=96 (12 heads x 8), theta 5e5, eps 1e-6: condition +
// prefill T=6 + 2 AR steps + head logits.
static void scenario_tts(ggml_backend_t backend, ggml_gallocr_t allocr,
                         const char * dir) {
    const int D = 96, NH = 12, NKV = 12, HD = 8, V = 40, LL = 4096;
    const int T1 = 6;
    const float EPS = 1e-6f, THETA = 500000.0f;

    WW w;
    w.init(backend, 40);
    ggml_tensor * emb_text = w.make("tk_emb_text", D, V);
    ggml_tensor * emb_code = w.make("tk_emb_code", D, 16);
    ggml_tensor * head     = w.make("tk_head", D, 16);
    ggml_tensor * out_norm = w.make("tk_onorm", D);
    ggml_tensor * fc1 = w.make("tk_fc1", LL, D);
    ggml_tensor * fc1b = w.make("tk_fc1b", D);
    ggml_tensor * fc2 = w.make("tk_fc2", D, D);
    ggml_tensor * fc2b = w.make("tk_fc2b", D);
    ggml_tensor * ln[2], *q[2], *k[2], *v[2], *o[2], *fl[2], *g[2], *up[2], *dn[2];
    char nm[32];
    for (int i = 0; i < 2; i++) {
        snprintf(nm, sizeof(nm), "tk_l%d_ln", i);   ln[i] = w.make(nm, D);
        snprintf(nm, sizeof(nm), "tk_l%d_q", i);    q[i]  = w.make(nm, D, NH * HD);
        snprintf(nm, sizeof(nm), "tk_l%d_k", i);    k[i]  = w.make(nm, D, NKV * HD);
        snprintf(nm, sizeof(nm), "tk_l%d_v", i);    v[i]  = w.make(nm, D, NKV * HD);
        snprintf(nm, sizeof(nm), "tk_l%d_o", i);    o[i]  = w.make(nm, NH * HD, D);
        snprintf(nm, sizeof(nm), "tk_l%d_fl", i);   fl[i] = w.make(nm, D);
        snprintf(nm, sizeof(nm), "tk_l%d_g", i);    g[i]  = w.make(nm, D, 4 * D);
        snprintf(nm, sizeof(nm), "tk_l%d_u", i);    up[i] = w.make(nm, D, 4 * D);
        snprintf(nm, sizeof(nm), "tk_l%d_d", i);    dn[i] = w.make(nm, 4 * D, D);
    }
    vqa::KvCache kv;
    kv.alloc(backend, 2, NKV, HD, 32);
    w.alloc();
    w.fill(emb_text, rnd((size_t)D * V, 1.0f));
    w.fill(emb_code, rnd((size_t)D * 16, 1.0f));
    w.fill(head,     rnd((size_t)D * 16, 0.15f));
    w.fill(out_norm, rnd((size_t)D, 1.0f));
    w.fill(fc1, rnd((size_t)LL * D, 0.03f));
    w.fill(fc1b, rnd((size_t)D, 0.3f));
    w.fill(fc2, rnd((size_t)D * D, 0.1f));
    w.fill(fc2b, rnd((size_t)D, 0.3f));
    for (int i = 0; i < 2; i++) {
        w.fill(ln[i], rnd((size_t)D, 1.0f));
        w.fill(q[i],  rnd((size_t)NH * HD * D, 0.2f));
        w.fill(k[i],  rnd((size_t)NKV * HD * D, 0.2f));
        w.fill(v[i],  rnd((size_t)NKV * HD * D, 0.2f));
        w.fill(o[i],  rnd((size_t)NH * HD * D, 0.2f));
        w.fill(fl[i], rnd((size_t)D, 1.0f));
        w.fill(g[i],  rnd((size_t)4 * D * D, 0.2f));
        w.fill(up[i], rnd((size_t)4 * D * D, 0.2f));
        w.fill(dn[i], rnd((size_t)4 * D * D, 0.2f));
    }
    dt(emb_text, dir, "tk_emb_text");
    dt(emb_code, dir, "tk_emb_code");
    dt(head, dir, "tk_head");
    dt(out_norm, dir, "tk_onorm");
    dt(fc1, dir, "tk_fc1"); dt(fc1b, dir, "tk_fc1b");
    dt(fc2, dir, "tk_fc2"); dt(fc2b, dir, "tk_fc2b");
    for (int i = 0; i < 2; i++) {
        char n2[32];
#define DUMP1(v, tag) do { snprintf(n2, sizeof(n2), "tk_l%d_" tag, i); dt(v, dir, n2); } while (0)
        DUMP1(ln[i], "ln"); DUMP1(q[i], "q"); DUMP1(k[i], "k"); DUMP1(v[i], "v");
        DUMP1(o[i], "o"); DUMP1(fl[i], "fl"); DUMP1(g[i], "g"); DUMP1(up[i], "u");
        DUMP1(dn[i], "d");
#undef DUMP1
    }

    // ---- prefill: condition over T1 fake tokens + 2 bos rows -------------
    std::vector<int> toks(T1);
    std::vector<float> hid((size_t)LL * T1);
    for (int i = 0; i < T1; i++) toks[(size_t)i] = (i * 7 + 3) % V;
    for (auto & f : hid) f = g_rng.u() * 1.0f;
    dt_f(std::vector<float>(toks.begin(), toks.end()), { T1 }, dir, "tk_toks");
    dt_f(hid, { LL, T1 }, dir, "tk_hid");

    Run r; r.backend = backend; r.allocr = allocr; r.dir = dir;
    r.begin();
    ggml_tensor * tok_in = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, T1);
    ggml_set_name(tok_in, "tk_tokin");
    ggml_set_input(tok_in);
    ggml_tensor * hid_in = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, LL, T1);
    ggml_set_name(hid_in, "tk_hidin");
    ggml_set_input(hid_in);
    ggml_tensor * bos_in = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, 2);
    ggml_set_name(bos_in, "tk_bosin");
    ggml_set_input(bos_in);
    ggml_tensor * pos = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, T1 + 2);
    ggml_set_name(pos, "tk_pos");
    ggml_set_input(pos);
    ggml_tensor * mask = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F16, T1 + 2, T1 + 2);
    ggml_set_name(mask, "tk_mask");
    ggml_set_input(mask);

    ggml_tensor * cond = minicpmo::tts_condition(r.ctx, emb_text, tok_in, hid_in,
                                                 fc1, fc1b, fc2, fc2b);
    ggml_set_name(cond, "tk_cond");
    ggml_tensor * emb = ggml_concat(r.ctx, cond, ggml_get_rows(r.ctx, emb_text, bos_in), 1);
    ggml_set_name(emb, "tk_emb");
    ggml_tensor * x = emb;
    for (int il = 0; il < 2; il++) {
        minicpmo::LlmAttnW awl{ q[il], k[il], v[il], o[il], nullptr, nullptr };
        ggml_tensor * l = ggml_rms_norm(r.ctx, x, EPS);
        l = ggml_mul(r.ctx, l, ln[il]);
        ggml_tensor * a = minicpmo::llm_attn_branch(r.ctx, r.gf, awl, kv.k[(size_t)il],
                                                    kv.v[(size_t)il], l, mask, pos,
                                                    NH, NKV, HD, THETA, 4096, EPS,
                                                    il == 0 ? "tk0" : nullptr);
        x = ggml_add(r.ctx, x, a);
        minicpmo::LlmFfnW fwl{ g[il], up[il], dn[il] };
        l = ggml_rms_norm(r.ctx, x, EPS);
        l = ggml_mul(r.ctx, l, fl[il]);
        ggml_tensor * f = minicpmo::llm_ffn_branch(r.ctx, fwl, l, EPS, nullptr);
        x = ggml_add(r.ctx, x, f);
    }
    x = ggml_rms_norm(r.ctx, x, EPS);
    x = ggml_mul(r.ctx, x, out_norm);
    ggml_tensor * last = ggml_view_2d(r.ctx, x, D, 1, x->nb[1],
                                      (size_t)(T1 + 1) * x->nb[1]);
    ggml_tensor * logits = ggml_mul_mat(r.ctx, head, last);
    ggml_set_name(logits, "tk_logits");
    ggml_build_forward_expand(r.gf, logits);
    r.sinks = { "tk_cond", "tk_emb", "tk0_attn", "tk_logits" };
    if (!r.alloc()) exit(1);
    ggml_backend_tensor_set(tok_in, toks.data(), 0, sizeof(int) * T1);
    ggml_backend_tensor_set(hid_in, hid.data(), 0, sizeof(float) * hid.size());
    {
        const int32_t bos2[2] = { V - 2, V - 1 };   // valid emb_text rows
        ggml_backend_tensor_set(bos_in, bos2, 0, sizeof(bos2));
        std::vector<int32_t> pv((size_t)T1 + 2);
        for (int i = 0; i < T1 + 2; i++) pv[(size_t)i] = i;
        ggml_backend_tensor_set(pos, pv.data(), 0, sizeof(int) * pv.size());
        vqa::set_causal_mask(mask, 0, T1 + 2);
    }
    if (!r.compute()) exit(1);
    r.dump_all();
    r.end();

    // ---- 2 AR steps: emb_code(prev) at position T1+2+t -------------------
    int prev = 5, n_past = T1 + 2;
    for (int step = 0; step < 2; step++) {
        char npre[32];
        snprintf(npre, sizeof(npre), "tk_s%d", step);
        Run s; s.backend = backend; s.allocr = allocr; s.dir = dir;
        s.begin();
        ggml_tensor * cin = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, 1);
        ggml_set_name(cin, "tk_cin");
        ggml_set_input(cin);
        ggml_tensor * spos = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, 1);
        ggml_set_name(spos, "tk_spos");
        ggml_set_input(spos);
        const int S = n_past + 1;
        ggml_tensor * smask = ggml_new_tensor_2d(s.ctx, GGML_TYPE_F16, S, 1);
        ggml_set_name(smask, "tk_smask");
        ggml_set_input(smask);
        ggml_tensor * sx = ggml_get_rows(s.ctx, emb_code, cin);
        for (int il = 0; il < 2; il++) {
            minicpmo::LlmAttnW awl{ q[il], k[il], v[il], o[il], nullptr, nullptr };
            ggml_tensor * l = ggml_rms_norm(s.ctx, sx, EPS);
            l = ggml_mul(s.ctx, l, ln[il]);
            ggml_tensor * a = minicpmo::llm_attn_branch(s.ctx, s.gf, awl,
                                                        kv.k[(size_t)il],
                                                        kv.v[(size_t)il], l, smask,
                                                        spos, NH, NKV, HD, THETA,
                                                        4096, EPS, nullptr);
            sx = ggml_add(s.ctx, sx, a);
            minicpmo::LlmFfnW fwl{ g[il], up[il], dn[il] };
            l = ggml_rms_norm(s.ctx, sx, EPS);
            l = ggml_mul(s.ctx, l, fl[il]);
            ggml_tensor * f = minicpmo::llm_ffn_branch(s.ctx, fwl, l, EPS, nullptr);
            sx = ggml_add(s.ctx, sx, f);
        }
        sx = ggml_rms_norm(s.ctx, sx, EPS);
        sx = ggml_mul(s.ctx, sx, out_norm);
        ggml_tensor * lg = ggml_mul_mat(s.ctx, head, sx);
        ggml_set_name(lg, "tk_slogits");
        s.sinks = { "tk_slogits" };
        if (!s.alloc()) exit(1);
        ggml_backend_tensor_set(cin, &prev, 0, sizeof(int));
        const int32_t p = n_past;
        ggml_backend_tensor_set(spos, &p, 0, sizeof(int));
        vqa::set_causal_mask(smask, n_past, 1);
        if (!s.compute()) exit(1);
        s.dump_all();
        s.end();
        dt(kv.k[1], dir, (std::string("tk_s") + std::to_string(step) + "_kcache").c_str());
        dt(kv.v[1], dir, (std::string("tk_s") + std::to_string(step) + "_vcache").c_str());
        n_past += 1;
        prev = (prev * 3 + 1) % 16;
    }
    fprintf(stderr, "tts scenario done\n");
}

// ------------------------------------------------------------------ llm
// one Qwen3 layer: per-head q/k norm + NEOX theta 1e6 + GQA 8:2 + swiglu,
// prefill T=6 then 2 decode steps, final norm + untied head.
static void scenario_llm(ggml_backend_t backend, ggml_gallocr_t allocr,
                         const char * dir) {
    const int D = 128, NH = 8, NKV = 2, HD = 16, V = 48, T = 6;
    const float EPS = 1e-6f, THETA = 1e6f;

    WW w;
    w.init(backend, 24);
    ggml_tensor * emb = w.make("l_emb", D, V);
    ggml_tensor * norm = w.make("l_norm", D);
    ggml_tensor * ffn = w.make("l_ffn", D);
    ggml_tensor * wq = w.make("l_q", D, NH * HD);
    ggml_tensor * wk = w.make("l_k", D, NKV * HD);
    ggml_tensor * wv = w.make("l_v", D, NKV * HD);
    ggml_tensor * wo = w.make("l_o", NH * HD, D);
    ggml_tensor * qn = w.make("l_qn", HD);
    ggml_tensor * kn = w.make("l_kn", HD);
    ggml_tensor * gate = w.make("l_g", D, 4 * D);
    ggml_tensor * up = w.make("l_u", D, 4 * D);
    ggml_tensor * down = w.make("l_d", 4 * D, D);
    ggml_tensor * fn = w.make("l_fn", D);
    ggml_tensor * head = w.make("l_head", D, V);
    vqa::KvCache kv;
    kv.alloc(backend, 1, NKV, HD, 32);
    w.alloc();
    w.fill(emb, rnd((size_t)D * V, 1.0f));
    w.fill(norm, rnd((size_t)D, 1.0f));
    w.fill(ffn, rnd((size_t)D, 1.0f));
    w.fill(wq, rnd((size_t)NH * HD * D, 0.2f));
    w.fill(wk, rnd((size_t)NKV * HD * D, 0.2f));
    w.fill(wv, rnd((size_t)NKV * HD * D, 0.2f));
    w.fill(wo, rnd((size_t)NH * HD * D, 0.2f));
    w.fill(qn, rnd((size_t)HD, 1.0f));
    w.fill(kn, rnd((size_t)HD, 1.0f));
    w.fill(gate, rnd((size_t)4 * D * D, 0.2f));
    w.fill(up, rnd((size_t)4 * D * D, 0.2f));
    w.fill(down, rnd((size_t)4 * D * D, 0.2f));
    w.fill(fn, rnd((size_t)D, 1.0f));
    w.fill(head, rnd((size_t)D * V, 0.15f));
    dt(emb, dir, "l_emb"); dt(norm, dir, "l_norm"); dt(ffn, dir, "l_ffn");
    dt(wq, dir, "l_q"); dt(wk, dir, "l_k"); dt(wv, dir, "l_v"); dt(wo, dir, "l_o");
    dt(qn, dir, "l_qn"); dt(kn, dir, "l_kn");
    dt(gate, dir, "l_g"); dt(up, dir, "l_u"); dt(down, dir, "l_d");
    dt(fn, dir, "l_fn"); dt(head, dir, "l_head");

    const int toks[T] = { 3, 17, 9, 41, 2, 30 };
    int n_past = 0;
    for (int step = 0; step < 3; step++) {
        const int Tc = step == 0 ? T : 1;
        char pre[16];
        snprintf(pre, sizeof(pre), "l%d", step);
        Run r; r.backend = backend; r.allocr = allocr; r.dir = dir;
        r.begin();
        ggml_tensor * tok = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, Tc);
        ggml_set_name(tok, "l_tok");
        ggml_set_input(tok);
        ggml_tensor * pos = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, Tc);
        ggml_set_name(pos, "l_pos");
        ggml_set_input(pos);
        const int S = n_past + Tc;
        ggml_tensor * mask = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F16, S, Tc);
        ggml_set_name(mask, "l_mask");
        ggml_set_input(mask);

        ggml_tensor * x = ggml_get_rows(r.ctx, emb, tok);
        ggml_set_name(x, (std::string(pre) + "_emb").c_str());
        minicpmo::LlmAttnW aw{ wq, wk, wv, wo, qn, kn };
        ggml_tensor * l = ggml_rms_norm(r.ctx, x, EPS);
        l = ggml_mul(r.ctx, l, norm);
        ggml_tensor * a = minicpmo::llm_attn_branch(r.ctx, r.gf, aw, kv.k[0], kv.v[0],
                                                    l, mask, pos, NH, NKV, HD,
                                                    THETA, 4096, EPS,
                                                    step == 0 ? "l0" : nullptr);
        x = ggml_add(r.ctx, x, a);
        minicpmo::LlmFfnW fw{ gate, up, down };
        l = ggml_rms_norm(r.ctx, x, EPS);
        l = ggml_mul(r.ctx, l, ffn);
        ggml_tensor * f = minicpmo::llm_ffn_branch(r.ctx, fw, l, EPS, nullptr);
        x = ggml_add(r.ctx, x, f);
        x = ggml_rms_norm(r.ctx, x, EPS);
        x = ggml_mul(r.ctx, x, fn);
        ggml_tensor * last = ggml_view_2d(r.ctx, x, D, 1, x->nb[1],
                                          (size_t)(Tc - 1) * x->nb[1]);
        ggml_tensor * logits = ggml_mul_mat(r.ctx, head, last);
        char nlg[32];
        snprintf(nlg, sizeof(nlg), "%s_logits", pre);
        ggml_set_name(logits, nlg);
        ggml_build_forward_expand(r.gf, logits);
        char nemb[32], nmask[32];
        snprintf(nemb, sizeof(nemb), "%s_emb", pre);
        snprintf(nmask, sizeof(nmask), "%s_mask", pre);
        r.sinks = { nemb, nlg, "l0_q_norm", "l0_q_rope", "l0_attn" };
        if (!r.alloc()) exit(1);
        ggml_backend_tensor_set(tok, toks + (step == 0 ? 0 : step), 0,
                                sizeof(int) * Tc);
        std::vector<int32_t> pv((size_t)Tc);
        for (int i = 0; i < Tc; i++) pv[(size_t)i] = n_past + i;
        ggml_backend_tensor_set(pos, pv.data(), 0, sizeof(int) * Tc);
        vqa::set_causal_mask(mask, n_past, Tc);
        if (!r.compute()) exit(1);
        r.dump_all();
        r.end();
        n_past += Tc;
    }
    dt(kv.k[0], dir, "l_kcache");
    dt(kv.v[0], dir, "l_vcache");
    fprintf(stderr, "llm scenario done\n");
}

// ------------------------------------------------------------------ rs-real
static void scenario_rs_real(ggml_backend_t backend, ggml_gallocr_t allocr,
                             const char * dir) {
    // production resampler dims: D 4096 / in 1152 / 32 heads / Q 64 / n 1024
    const int D = 4096, IN = 1152, NH = 32, Q = 64, N = 1024;
    WW w;
    w.init(backend, 16);
    ggml_tensor * query   = w.make("rr_query", D, Q);
    ggml_tensor * kv_proj = w.make("rr_wkv", IN, D, 1, GGML_TYPE_F16);
    ggml_tensor * in_proj = w.make("rr_ip", D, 3 * D, 1, GGML_TYPE_F16);
    ggml_tensor * in_b    = w.make("rr_ipb", 3 * D);
    ggml_tensor * out_p   = w.make("rr_op", D, D, 1, GGML_TYPE_F16);
    ggml_tensor * out_b   = w.make("rr_opb", D);
    ggml_tensor * lnq_w   = w.make("rr_lnq_w", D);
    ggml_tensor * lnq_b   = w.make("rr_lnq_b", D);
    ggml_tensor * lnkv_w  = w.make("rr_lnkv_w", D);
    ggml_tensor * lnkv_b  = w.make("rr_lnkv_b", D);
    ggml_tensor * lnp_w   = w.make("rr_lnp_w", D);
    ggml_tensor * lnp_b   = w.make("rr_lnp_b", D);
    ggml_tensor * proj    = w.make("rr_proj", D, D);   // F32, mirrors the
                                                       // production load-time
                                                       // replica (see
                                                       // resampler_branch)
    w.alloc();
    {
        std::vector<float> v((size_t)D * Q);
        for (size_t i = 0; i < v.size(); i++) v[i] = g_rng.u() * 0.3f;
        w.fill(query, v);
        dt_f(v, { D, Q }, dir, "rr_query");
    }
    {
        std::vector<float> v((size_t)D * IN);
        for (size_t i = 0; i < v.size(); i++) v[i] = g_rng.u() * 0.05f;
        w.fill(kv_proj, v);
        dt_f(v, { D, IN }, dir, "rr_wkv");
    }
    {
        std::vector<float> v((size_t)D * 3 * D);
        for (size_t i = 0; i < v.size(); i++) v[i] = g_rng.u() * 0.02f;
        w.fill(in_proj, v);
        dt_f(v, { 3 * D, D }, dir, "rr_ip");
    }
    {
        std::vector<float> v((size_t)3 * D);
        for (size_t i = 0; i < v.size(); i++) v[i] = g_rng.u() * 0.3f;
        w.fill(in_b, v);
        dt_f(v, { 3 * D }, dir, "rr_ipb");
    }
    {
        std::vector<float> v((size_t)D * D);
        for (size_t i = 0; i < v.size(); i++) v[i] = g_rng.u() * 0.02f;
        w.fill(out_p, v);
        w.fill(proj, v);
        dt_f(v, { D, D }, dir, "rr_op");
    }
    {
        std::vector<float> v((size_t)D);
        for (size_t i = 0; i < v.size(); i++) v[i] = g_rng.u() * 0.3f;
        w.fill(out_b, v);
        dt_f(v, { D }, dir, "rr_opb");
    }
    {
        std::vector<float> v((size_t)D);
        for (size_t i = 0; i < v.size(); i++) v[i] = 1.0f + 0.2f * g_rng.u();
        w.fill(lnq_w, v);
        w.fill(lnkv_w, v);
        w.fill(lnp_w, v);
        dt_f(v, { D }, dir, "rr_ln_w");
    }
    {
        std::vector<float> v((size_t)D);
        for (size_t i = 0; i < v.size(); i++) v[i] = g_rng.u() * 0.2f;
        w.fill(lnq_b, v);
        w.fill(lnkv_b, v);
        w.fill(lnp_b, v);
        dt_f(v, { D }, dir, "rr_ln_b");
    }
    {
        std::vector<float> v((size_t)D * D);
        for (size_t i = 0; i < v.size(); i++) v[i] = g_rng.u() * 0.05f;
        w.fill(proj, v);
        dt_f(v, { D, D }, dir, "rr_proj");   // distinct from op - dump it!
    }

    minicpmo::ResamplerW rw;
    rw.query = query; rw.kv_proj = kv_proj;
    rw.in_proj = in_proj; rw.in_proj_b = in_b;
    rw.out_proj = out_p; rw.out_proj_b = out_b;
    rw.ln_q_w = lnq_w; rw.ln_q_b = lnq_b;
    rw.ln_kv_w = lnkv_w; rw.ln_kv_b = lnkv_b;
    rw.ln_post_w = lnp_w; rw.ln_post_b = lnp_b;
    rw.proj = proj;
    rw.proj_f32 = proj;   // in-graph right multiply (branch output = x@proj)

    std::vector<float> x = rnd((size_t)IN * N, 1.0f);
    std::vector<float> pos;
    minicpmo::sincos_pos_2d(32, 32, D, pos);
    dt_f(x, { IN, N }, dir, "rr_x");
    dt_f(pos, { D, N }, dir, "rr_pos");

    Run r; r.backend = backend; r.allocr = allocr; r.dir = dir;
    r.begin();
    ggml_tensor * xt = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, IN, N);
    ggml_set_name(xt, "rr_x");
    ggml_set_input(xt);
    ggml_tensor * pt = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, D, N);
    ggml_set_name(pt, "rr_pos");
    ggml_set_input(pt);
    ggml_tensor * out = minicpmo::resampler_branch(r.ctx, rw, xt, pt, NH, "rr");
    ggml_set_name(out, "rr_out");
    r.sinks = { "rr_out", "rr_kv", "rr_keys", "rr_iq", "rr_fa", "rr_attn", "rr_o1", "rr_lout" };
    ggml_build_forward_expand(r.gf, out);
    if (!r.alloc()) exit(1);
    ggml_backend_tensor_set(xt, x.data(), 0, sizeof(float) * x.size());
    ggml_backend_tensor_set(pt, pos.data(), 0, sizeof(float) * pos.size());
    if (!r.compute()) exit(1);
    r.dump_all();
    {
        // direct read of the graph output (no sink dup), written LAST
        std::vector<float> ov((size_t)D * Q);
        ggml_backend_tensor_get(out, ov.data(), 0, sizeof(float) * ov.size());
        dt_f(ov, { D, Q }, dir, "rr_out");
    }
    r.end();
    fprintf(stderr, "rs-real scenario done' + NL + '");
}

// ------------------------------------------------------------------ mulmat
static void scenario_mulmat(ggml_backend_t backend, ggml_gallocr_t allocr,
                            const char * dir) {
    // real-shape mul_mat probe: A [4096, 4096] F16, B [4096, 1024] F32
    const int K = 4096, M = 4096, N = 1024;
    WW w;
    w.init(backend, 4);
    ggml_tensor * A = w.make("mm_A", K, M, 1, GGML_TYPE_F16);
    w.alloc();
    {
        std::vector<float> av((size_t)K * M);
        for (size_t i = 0; i < av.size(); i++) av[i] = g_rng.u() * 0.05f;
        std::vector<ggml_fp16_t> ah(av.size());
        for (size_t i = 0; i < av.size(); i++) ah[i] = ggml_fp32_to_fp16(av[i]);
        ggml_backend_tensor_set(A, ah.data(), 0, sizeof(ggml_fp16_t) * ah.size());
        dt_f(av, { M, K }, dir, "mm_A");
    }
    Run r; r.backend = backend; r.allocr = allocr; r.dir = dir;
    r.begin();
    ggml_tensor * B = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(B, "mm_B");
    ggml_set_input(B);
    ggml_tensor * C = ggml_mul_mat(r.ctx, A, B);
    ggml_set_name(C, "mm_C");
    r.sinks = { "mm_C" };
    if (!r.alloc()) exit(1);
    {
        std::vector<float> bv((size_t)K * N);
        for (size_t i = 0; i < bv.size(); i++) bv[i] = g_rng.u() * 1.0f;
        ggml_backend_tensor_set(B, bv.data(), 0, sizeof(float) * bv.size());
        dt_f(bv, { K, N }, dir, "mm_B");
    }
    if (!r.compute()) exit(1);
    r.dump_all();
    r.end();
    fprintf(stderr, "mulmat scenario done\n");
}

int main(int argc, char ** argv) {
    g_rng = Rng(0xB0C5Eu);   // harness RNG seed - parity-stream contract
    const char * dir = argc > 1 ? argv[1] : "D:/S2TT/minicpmo45-port/seg-dump";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (getenv("SEG_NTHREADS"))
        ggml_backend_cpu_set_n_threads(backend, 1);
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    scenario_rs_real(backend, allocr, dir);
    scenario_mulmat(backend, allocr, dir);
    scenario_fa_probe(backend, allocr, dir);
    scenario_rs(backend, allocr, dir);
    scenario_slice(dir);
    scenario_audio(backend, allocr, dir);
    scenario_tts(backend, allocr, dir);
    scenario_llm(backend, allocr, dir);
    ggml_gallocr_free(allocr);
    ggml_backend_free(backend);
    fprintf(stderr, "all scenarios written to %s\n", dir);
    return 0;
}
