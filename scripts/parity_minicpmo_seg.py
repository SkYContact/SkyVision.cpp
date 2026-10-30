#!/usr/bin/env python3
# parity_minicpmo_seg.py - numpy reference for tools/minicpmo_seg_test.cpp
# (MiniCPM-o 4.5 component parity: resampler / slice tables / whisper conv +
# chunk mask / tk. sub-tower step / one Qwen3 layer).
#
# Usage: python scripts/parity_minicpmo_seg.py <dump_dir>
# (default D:/S2TT/minicpmo45-port/seg-dump). Dumps are ggml-order flat f32
# bins with a .shape sidecar (ne order, d0 fastest).
import struct
import sys
import os

import numpy as np

TOL = 2e-3          # f16-weight relative tolerance for matmul-heavy chains


def load(d, name):
    data = np.fromfile(os.path.join(d, f"cpp_{name}.bin"), dtype=np.float32)
    with open(os.path.join(d, f"cpp_{name}.shape")) as f:
        ne = [int(x) for x in f.read().split()]
    return data.reshape(list(reversed(ne)))   # ne (ggml) -> numpy (d0 fastest)


def flat(a):
    return a.reshape(-1)


def rms_norm(x, g, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def ln(x, g, b, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def check(name, a, b, tol=TOL):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    denom = np.maximum(np.abs(b).max(), 1e-6)
    diff = np.abs(a - b).max()
    ok = diff <= tol * denom
    print(f"  {name:24s} max_abs={diff:.3e} rel={diff / denom:.3e} "
          f"{'OK' if ok else 'FAIL'}")
    return ok


# ------------------------------------------------------------ resampler
def scenario_rs(d):
    print("rs:")
    D = 256
    x = load(d, "rs_x").reshape(40, 96)      # numpy (tokens, dim)
    pos = load(d, "rs_pos").reshape(40, D)
    ref_out = load(d, "rs_out").reshape(-1, D)
    q16 = flat(load(d, "rs_query")).reshape(16, D)    # shared queries
    kv_w = flat(load(d, "rs_wkv")).reshape(D, 96)   # numpy [256, 96] = (out, in)
    ip = flat(load(d, "rs_ip")).reshape(3 * D, D)   # (out=3D, in=D)
    ib = flat(load(d, "rs_ipb"))
    op = flat(load(d, "rs_op")).reshape(D, D)
    ob = flat(load(d, "rs_opb"))
    lnq_w, lnq_b = flat(load(d, "rs_lnq_w")), flat(load(d, "rs_lnq_b"))
    lnkv_w, lnkv_b = flat(load(d, "rs_lnkv_w")), flat(load(d, "rs_lnkv_b"))
    lnp_w, lnp_b = flat(load(d, "rs_lnp_w")), flat(load(d, "rs_lnp_b"))
    proj = flat(load(d, "rs_proj")).reshape(D, D)
    # ggml ne [in, out] == numpy (out, in): mul_mat(A, X) = X @ A.T
    kv = ln(x @ kv_w.T, lnkv_w, lnkv_b)
    keys = kv + pos
    qq = ln(q16, lnq_w, lnq_b)
    H = 8
    dh = D // H
    qw, kw, vw = ip[:D], ip[D:2 * D], ip[2 * D:]
    qb, kb, vb = ib[:D], ib[D:2 * D], ib[2 * D:]

    def heads(t, n):
        return t.reshape(n, H, dh).transpose(1, 0, 2)
    qp = qq @ qw.T + qb                 # (16, D)
    qh = heads(qp, 16)
    kh = heads(keys @ kw.T + kb, 40)
    vh = heads(kv @ vw.T + vb, 40)
    att = qh @ kh.transpose(0, 2, 1) / np.sqrt(dh)
    att = softmax(att)
    fa_ref = (att @ vh).transpose(1, 0, 2).reshape(16, D)
    ok0 = check("rs_fa", fa_ref, load(d, "rs_fa").reshape(16, D))
    out = fa_ref
    out = out @ op.T + ob
    out = ln(out, lnp_w, lnp_b)
    # the branch ends with the in-graph RIGHT multiply x @ proj (proj_f32
    # replica). A mul_mat src0's numpy view is (ne1, ne0) = (out, in), so
    # engine rs_out == ln_out @ P.T; rs_lout is the pre-proj ln_out stage.
    ok_l = check("rs_lout", out, load(d, "rs_lout").reshape(16, D))
    mine = out @ proj.T                     # [16, D] ln_out @ P.T == x @ proj
    ok1 = check("rs_out", mine, ref_out)
    ok2 = check("rs_keys", keys, load(d, "rs_keys").reshape(40, D))
    ok3 = check("rs_iq", qp, load(d, "rs_iq").reshape(16, D))
    ok4 = check("rs_kv", kv, load(d, "rs_kv").reshape(40, D))
    return ok0 and ok1 and ok2 and ok3 and ok4 and ok_l


# ------------------------------------------------------------ slice
def scenario_slice(d):
    print("slice:")
    ok = True
    grids = load(d, "slice_grids").reshape(-1, 4)
    ref = []
    for w_, h_, _, _ in grids:
        w_, h_ = int(w_), int(h_)
        ref.extend(slice_grid_ref(w_, h_))
    ok &= check("slice_grids", np.array(ref, np.float32), flat(grids), 0)

    # bucketized ids: boundaries k/70, right=True
    ids = load(d, "slice_pos_ids")
    shapes = [(32, 32), (23, 45), (5, 8)]
    mine = []
    for h_, w_ in shapes:
        for i in range(h_):
            bh = 70 * i // h_
            for j in range(w_):
                bw = 70 * j // w_
                mine.append(bh * 70 + bw)
    ok &= check("slice_pos_ids", np.array(mine, np.float32), flat(ids), 0)

    # sincos dim-16 tables (first half = column grid, second = row grid)
    sc = load(d, "slice_sincos16")
    rows = []
    for h_, w_ in shapes:
        for i in range(h_):
            for j in range(w_):
                om = 10000.0 ** (-np.arange(4) / 4.0)
                r = np.concatenate([np.sin(j * om), np.cos(j * om),
                                    np.sin(i * om), np.cos(i * om)])
                rows.append(r)
    ok &= check("slice_sincos16", np.array(rows).reshape(-1), flat(sc), 1e-5)
    return ok


def slice_grid_ref(w, h):
    log_ratio = np.log(w / h)
    ratio = w * h / (448.0 * 448.0)
    multiple = min(int(np.ceil(ratio)), 9)
    if multiple <= 1:
        cols, rows = 0, 0
    else:
        cands = []
        for n in {multiple - 1, multiple, multiple + 1}:
            if n == 1 or n > 9:
                continue
            for m in range(1, n + 1):
                if n % m == 0:
                    cands.append((m, n // m))
        best, err = (1, 1), 1e300
        for g in cands:
            e = abs(log_ratio - np.log(g[0] / g[1]))
            if e < err:
                best, err = g, e
        cols, rows = best
    # thumbnail + refine tile sizes
    def best_resize(w0, h0, up):
        if w0 * h0 > 448 * 448 or up:
            r = w0 / h0
            h0 = int(448 / np.sqrt(r))
            w0 = int(h0 * r)
        return (max(round(w0 / 14) * 14, 14), max(round(h0 / 14) * 14, 14))
    tw, th = best_resize(w, h, up=cols == 0)
    if cols == 0:
        rw, rh = tw, th
    else:
        rwd = max(round(w / cols) * cols, cols)
        rhd = max(round(h / rows) * rows, rows)
        gw, gh = best_resize(rwd // cols, rhd // rows, up=True)
        rw, rh = gw * cols, gh * rows
    return [w, h, cols, rows if cols else 0]


# ------------------------------------------------------------ audio
def scenario_audio(d):
    print("audio:")
    ok = True
    x = load(d, "a_in").reshape(32, 40).T    # ggml ne [T, C]: t fastest
    c1 = load(d, "a_c1").reshape(32, 32, 3)  # numpy [out, in, k]
    c2 = load(d, "a_c2").reshape(32, 32, 3)
    ref = load(d, "a_conv2").reshape(32, 20).T.reshape(-1)
    # torch conv1d: weight (out, in, k); x (t, in); pad 1 both convs
    def conv(x, w, stride):
        k = w.shape[2]
        pad = 1
        xt = np.pad(x, ((pad, pad), (0, 0)))
        t_out = (xt.shape[0] - k) // stride + 1
        w_ = w.transpose(2, 1, 0)            # (k, in, out)
        out = np.zeros((t_out, w.shape[0]), np.float64)
        for t in range(t_out):
            win = xt[t * stride:t * stride + k]          # (k, in)
            out[t] = (win[:, :, None] * w_).sum((0, 1))
        return out
    y = conv(x.astype(np.float64), c1.astype(np.float64), 1)
    y = conv(y, c2.astype(np.float64), 2)
    ok &= check("a_conv2", y.reshape(-1), flat(ref), 2e-2)

    # chunk mask (exact: -inf vs 0)
    m = flat(load(d, "a_chunk_mask"))
    ref_mask = np.ones((10, 10), np.float32) * -np.inf
    for q in range(10):
        for k in range(min((q // 3 + 1) * 3, 10)):
            ref_mask[q, k] = 0.0
    same = (m == ref_mask.reshape(-1)) | (np.isneginf(m) & np.isneginf(ref_mask.reshape(-1)))
    print(f"  {'a_chunk_mask':24s} exact {'OK' if same.all() else 'FAIL'}")
    ok &= bool(same.all())
    return ok


# ------------------------------------------------------------ tts
def scenario_tts(d):
    print("tts:")
    ok = True
    D, NH, HD, LL, V = 96, 12, 8, 4096, 40
    EPS, THETA = 1e-6, 5e5
    T1 = 6
    toks = load(d, "tk_toks").astype(int)
    hid = load(d, "tk_hid")
    emb_text = flat(load(d, "tk_emb_text")).reshape(V, D)
    fc1 = flat(load(d, "tk_fc1")).reshape(D, LL)
    fc1b = flat(load(d, "tk_fc1b"))
    fc2 = flat(load(d, "tk_fc2")).reshape(D, D)
    fc2b = flat(load(d, "tk_fc2b"))
    cond_ref = load(d, "tk_cond")
    p = np.maximum(hid @ fc1.T + fc1b, 0.0) @ fc2.T + fc2b
    p = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12)
    cond = emb_text[toks] + p
    cond = cond.reshape(6, 96)
    ok &= check("tk_cond", cond.reshape(-1), flat(cond_ref), 2e-2)

    # full prefill forward through the 2 layers
    emb = np.concatenate([cond, emb_text[[V - 2, V - 1]]], axis=0)
    w = {n: flat(load(d, f"tk_{n}")) for n in
         ["onorm", "fc1", "fc1b", "fc2", "fc2b"]}
    layers = []
    for i in range(2):
        layers.append(dict(
            ln=flat(load(d, f"tk_l{i}_ln")),
            q=flat(load(d, f"tk_l{i}_q")).reshape(NH * HD, D),
            k=flat(load(d, f"tk_l{i}_k")).reshape(NH * HD, D),
            v=flat(load(d, f"tk_l{i}_v")).reshape(NH * HD, D),
            o=flat(load(d, f"tk_l{i}_o")).reshape(D, NH * HD),
            fl=flat(load(d, f"tk_l{i}_fl")),
            g=flat(load(d, f"tk_l{i}_g")).reshape(4 * D, D),
            u=flat(load(d, f"tk_l{i}_u")).reshape(4 * D, D),
            dn=flat(load(d, f"tk_l{i}_d")).reshape(D, 4 * D)))
    onorm = flat(load(d, "tk_onorm"))
    head = flat(load(d, "tk_head")).reshape(16, D)
    emb_ref = load(d, "tk_emb")
    ok &= check("tk_emb", emb.reshape(-1), flat(emb_ref), 2e-2)

    def rope(x, pos, hd, theta):
        # x (T, nh, hd); NEOX pair rotation
        half = hd // 2
        om = theta ** (-np.arange(half) / half)
        angs = pos[:, None] * om[None, :]
        c, s = np.cos(angs), np.sin(angs)
        out = x.copy()
        a, b = x[..., :half], x[..., half:]
        out[..., :half] = a * c[:, None, :] - b * s[:, None, :]
        out[..., half:] = a * s[:, None, :] + b * c[:, None, :]
        return out

    T = T1 + 2
    pos = np.arange(T)
    x = emb.astype(np.float64)
    for L in layers:
        h = rms_norm(x, L["ln"], EPS)
        q = h @ L["q"].T
        k = h @ L["k"].T
        v = h @ L["v"].T
        q = rope(q.reshape(T, NH, HD), pos, HD, THETA).transpose(1, 0, 2)
        k = rope(k.reshape(T, NH, HD), pos, HD, THETA).transpose(1, 0, 2)
        vv = v.reshape(T, NH, HD).transpose(1, 0, 2)
        att = q @ k.transpose(0, 2, 1) / np.sqrt(HD)
        mask = np.triu(np.ones((T, T), bool), 1)
        att = np.where(mask[None], -np.inf, att)
        att = softmax(att)
        o = (att @ vv).transpose(1, 0, 2).reshape(T, NH * HD) @ L["o"].T
        x = x + o
        h = rms_norm(x, L["fl"], EPS)
        x = x + ((silu(h @ L["g"].T) * (h @ L["u"].T)) @ L["dn"].T)
    x = rms_norm(x, onorm, EPS)
    ok &= check("tk_logits", (x[-1] @ head.T).reshape(-1),
                load(d, "tk_logits").reshape(-1), 2e-2)
    return ok


# ------------------------------------------------------------ llm
def scenario_llm(d):
    print("llm:")
    ok = True
    D, NH, NKV, HD, V = 128, 8, 2, 16, 48
    EPS, THETA = 1e-6, 1e6
    emb = flat(load(d, "l_emb")).reshape(V, D).astype(np.float64)
    norm = flat(load(d, "l_norm"))
    ffn = flat(load(d, "l_ffn"))
    W = dict(
        q=flat(load(d, "l_q")).reshape(NH * HD, D),
        k=flat(load(d, "l_k")).reshape(NKV * HD, D),
        v=flat(load(d, "l_v")).reshape(NKV * HD, D),
        o=flat(load(d, "l_o")).reshape(D, NH * HD),
        qn=flat(load(d, "l_qn")), kn=flat(load(d, "l_kn")),
        g=flat(load(d, "l_g")).reshape(4 * D, D),
        u=flat(load(d, "l_u")).reshape(4 * D, D),
        dn=flat(load(d, "l_d")).reshape(D, 4 * D))
    fn = flat(load(d, "l_fn"))
    head = flat(load(d, "l_head")).reshape(V, D)
    toks = [3, 17, 9, 41, 2, 30]
    n_past = 0
    kn_all = np.zeros((NKV, 0, HD))
    vv_all = np.zeros((NKV, 0, HD))
    for step in range(3):
        ids = toks if step == 0 else [toks[step]]
        x = emb[np.array(ids)].astype(np.float64)
        pos = np.arange(n_past, n_past + len(ids))
        h = rms_norm(x, norm, EPS)
        q = (h @ W["q"].T).reshape(len(ids), NH, HD)
        k = (h @ W["k"].T).reshape(len(ids), NKV, HD)
        v = (h @ W["v"].T).reshape(len(ids), NKV, HD)
        # official order: per-head q/k RMSNorm BEFORE rope (a non-uniform
        # per-dim norm scale does NOT commute with the pair rotation)
        qn = rms_norm(q, W["qn"], EPS)
        kn = rms_norm(k, W["kn"], EPS)
        q = rope_llm(qn, pos, HD, THETA).transpose(1, 0, 2)
        k = rope_llm(kn, pos, HD, THETA).transpose(1, 0, 2)
        vv = v.transpose(1, 0, 2)
        kn_all = np.concatenate([kn_all, k], 1)
        vv_all = np.concatenate([vv_all, vv], 1)
        kk = np.repeat(kn_all, NH // NKV, axis=0)   # (NKV, S, HD) -> (NH, S, HD)
        vv2 = np.repeat(vv_all, NH // NKV, axis=0)
        att = q @ kk.transpose(0, 2, 1) / np.sqrt(HD)
        S = n_past + len(ids)
        m = np.zeros((len(ids), S), bool)
        for t in range(len(ids)):
            m[t, n_past + t + 1:] = True
        att = np.where(m[None], -np.inf, att)
        att = softmax(att)
        o = (att @ vv2).transpose(1, 0, 2).reshape(len(ids), NH * HD) @ W["o"].T
        x = x + o
        h = rms_norm(x, ffn, EPS)
        x = x + ((silu(h @ W["g"].T) * (h @ W["u"].T)) @ W["dn"].T)
        x = rms_norm(x, fn, EPS)
        ok &= check(f"l{step}_logits", (x[-1] @ head.T).reshape(-1),
                    load(d, f"l{step}_logits").reshape(-1), 2e-2)
        n_past += len(ids)
    return ok


def rope_llm(x, pos, hd, theta):
    half = hd // 2
    om = theta ** (-np.arange(half) / half)
    angs = pos[:, None] * om[None, :]
    c, s = np.cos(angs), np.sin(angs)
    out = x.copy()
    a, b = x[..., :half], x[..., half:]
    out[..., :half] = a * c[:, None, :] - b * s[:, None, :]
    out[..., half:] = a * s[:, None, :] + b * c[:, None, :]
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "D:/S2TT/minicpmo45-port/seg-dump"
    ok = True
    ok &= scenario_rs(d)
    ok &= scenario_slice(d)
    ok &= scenario_audio(d)
    ok &= scenario_tts(d)
    ok &= scenario_llm(d)
    print("PARITY " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
