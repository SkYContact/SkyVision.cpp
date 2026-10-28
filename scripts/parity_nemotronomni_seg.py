#!/usr/bin/env python3
# parity_nemotronomni_seg.py - wave-3a numpy reference for the Nemotron-3-Nano-Omni
# text-chain segments (Mamba2 / NoPE GQA attention / MoE), run against the dumps
# written by tools/nemotronomni_seg_test.exe.
#
#   python parity_nemotronomni_seg.py [dump_dir]
#     default dump dir: D:/S2TT/nemotron3omni-port/3a/dump
#
# Conventions (SOP checklist #5/#6): every dump is ggml ne order, d0 fastest ->
# read flat then reshape (ne1, ne0, ne2, ne3). After that reshape a columnar
# activation dump [D, T] reads as (T, D) and a 2D weight reads as (out, in)
# (the converter contract: ggml ne = HF shape reversed). All references are
# token-major (T, ...) and matmul as x @ W.T.
# References are computed in float64; the engine graphs are fp32, so structural
# errors (layout / split order / missing op) show up at O(1e-1..1) and pure
# rounding at O(1e-6).
#
# Mamba2 gets TWO independent references (SOP: numpy refs must cross-check
# before judging the engine):
#   ref_mamba(chunk=False) - per-token recurrence (decode semantics)
#   ref_mamba(chunk=True)  - the HF chunk-SSD prefill (modeling_nemotron_h.py:600-677)
# The dt clamp question is reported both ways: HF clamps softplus(dt+bias) at
# 1e-3 (:601-602); the ggml_ssm_scan kernel has no clamp. The engine
# pre-compensates (inverse-softplus round trip), so the CLAMPED variant must
# match; the UNCLAMPED delta quantifies what upstream llama.cpp drops.
import os
import sys
import numpy as np

DIR = sys.argv[1] if len(sys.argv) > 1 else "D:/S2TT/nemotron3omni-port/3a/dump"

FAILS = []
REPORT = []


def load(name):
    """read cpp_<name>.bin + .shape -> float64 array (d0 fastest)."""
    binp = os.path.join(DIR, f"cpp_{name}.bin")
    shp = os.path.join(DIR, f"cpp_{name}.shape")
    if not os.path.exists(binp):
        raise FileNotFoundError(binp)
    a = np.fromfile(binp, dtype=np.float32)
    ne = [int(x) for x in open(shp).read().split()]
    assert len(ne) == 4
    assert int(np.prod(ne)) == a.size, (name, ne, a.size)
    # ggml flat order is d0-fastest: the C-order reshape that reproduces it
    # is ne REVERSED (SOP checklist #6 - a plain (ne1, ne0, ...) reshape
    # silently permutes dims 2+ of 3d+ dumps)
    arr = a.reshape(tuple(reversed(ne)))
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[..., 0, :, ...] if False else arr[0]
    return arr.astype(np.float64)   # 1d dumps read back as (1, N); ravel there


def report(tag, got, ref, tol=1e-4):
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert got.shape == ref.shape, (tag, got.shape, ref.shape)
    d = np.abs(got - ref)
    max_abs = float(d.max())
    l2 = float(np.linalg.norm((got - ref).ravel()) /
               (np.linalg.norm(ref.ravel()) + 1e-30))
    ok = l2 < tol
    REPORT.append((tag, max_abs, l2, "PASS" if ok else "FAIL"))
    if not ok:
        FAILS.append(tag)
    return l2


# ------------------------------------------------------------------ math
def rms_norm(x, eps, axis=-1):
    v = np.mean(x * x, axis=axis, keepdims=True)
    return x / np.sqrt(v + eps)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)   # stable


def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def gqa_attention(qt, kt, vt, hd, nkv, q_pos=None):
    """qt [T, NH, HD]; kt/vt [S, NKV, HD] (the cached values the engine's FA
    reads). q head h attends kv head h // (nh/nkv) - transformers'
    repeat_interleave mapping. Query g sits at absolute position q_pos[g]
    (default: 0..T-1) and sees keys 0..pos."""
    T, NH, _ = qt.shape
    S = kt.shape[0]
    grp = NH // nkv
    if q_pos is None:
        q_pos = np.arange(T)
    out = np.zeros((T, NH, hd))
    for h in range(NH):
        kv = h // grp
        s = qt[:, h, :] @ kt[:, kv, :].T / np.sqrt(hd)        # [T, S]
        for t in range(T):
            row = np.full(S, -np.inf)
            row[: q_pos[t] + 1] = s[t, : q_pos[t] + 1]
            out[t, h, :] = softmax(row) @ vt[:, kv, :]
    return out


# ------------------------------------------------------------- Mamba2 ref
def mamba_split(w):
    """in_proj out-row contract: [z I | xBC C | dt NH]; head_dim = I // NH.
    Weight reads (out, in): total out rows = shape[0]."""
    d_total = w["in_proj"].shape[0]
    NH = w["dt_bias"].size
    G = w["norm"].shape[0]                      # norm reads (G, I/G)
    I = w["norm"].shape[0] * w["norm"].shape[1]
    HD = I // NH
    C = d_total - I - NH
    return I, C, NH, HD


def causal_conv(xBC, conv_w, conv_b, state):
    """xBC [T, C] raw; state [K-1, C] time-major (engine dump transposed) or
    None; conv_w (C, K). Returns post conv+bias+silu [T, C]."""
    T, C = xBC.shape
    K = conv_w.shape[1]
    rows = xBC if state is None else np.concatenate([state, xBC], axis=0)
    off = 0 if state is None else state.shape[0]
    cc = np.zeros((T, C))
    for t in range(T):
        for j in range(K):
            src = t - (K - 1) + j + off
            if 0 <= src < rows.shape[0]:
                cc[t] += conv_w[:, j] * rows[src]
        cc[t] += np.ravel(conv_b)
    return silu(cc)


def ref_mamba(l, w, conv_state=None, ssm_state=None, clamp_dt=True, chunk=False):
    """Full branch. l [T, D] token-major. Returns dict with token-major
    activations and the final SSM state [NH, HD, S]. chunk=True is the HF
    chunk-SSD prefill (single chunk, zero initial state)."""
    I, C, NH, HD = mamba_split(w)
    GS = (C - I) // 2
    G = w["norm"].shape[0]                      # norm reads (G, I/G)
    S = ssm_state.shape[2] if ssm_state is not None else GS // G
    T = l.shape[0]
    if chunk:
        assert conv_state is None, "chunk-SSD ref covers the zero-state prefill"
    else:
        conv_state = np.zeros((3, C)) if conv_state is None else conv_state
    ssm_state = np.zeros((NH, HD, S)) if ssm_state is None else ssm_state

    zx = l @ w["in_proj"].T                                   # [T, DIN]
    z = zx[:, :I]
    xBC = zx[:, I:I + C]
    dtr = zx[:, I + C:]
    xBCc = causal_conv(xBC, w["conv"], w["convb"], conv_state)
    x = xBCc[:, :I]
    B = xBCc[:, I:I + GS].reshape(T, G, S)
    Cc = xBCc[:, I + GS:].reshape(T, G, S)
    dt = softplus(dtr + w["dt_bias"].ravel())
    if clamp_dt:
        dt = np.maximum(dt, 1e-3)
    A = np.ravel(w["a"])                                     # [NH]
    grp = NH // G

    if chunk:
        x_dt = x.reshape(T, NH, HD) * dt[:, :, None]         # [T, NH, HD]
        cum = np.cumsum(A[None, :] * dt, axis=0)             # [T, NH]
        mask = np.tril(np.ones((T, T)))
        L = np.exp((cum[:, None, :] - cum[None, :, :]) * mask[:, :, None]
                   - 1e30 * (1.0 - mask[:, :, None]))
        Gm = np.zeros((T, T, NH))
        for h in range(NH):
            g = h // grp
            Gm[:, :, h] = Cc[:, g, :] @ B[:, g, :].T
        Y = np.einsum("lsh,shd->lhd", Gm * L, x_dt)
        y = Y.reshape(T, I)
        st = np.zeros((NH, HD, S))
        for h in range(NH):
            g = h // grp
            decay = np.exp(cum[-1, h] - cum[:, h])
            st[h] = np.einsum("s,sd,se->de", decay, x_dt[:, h, :], B[:, g, :])
    else:
        st = ssm_state.copy()
        y = np.zeros((T, I))
        for t in range(T):
            for h in range(NH):
                g = h // grp
                dA = np.exp(dt[t, h] * A[h])
                x_dt = x[t, h * HD:(h + 1) * HD] * dt[t, h]
                st[h] = st[h] * dA + np.outer(x_dt, B[t, g])
                y[t, h * HD:(h + 1) * HD] = st[h] @ Cc[t, g]

    yd = y + x * np.repeat(np.ravel(w["d"]), HD)                         # + D*x per head
    yg = yd * silu(z)                                        # gate BEFORE norm
    yn = (rms_norm(yg.reshape(T, G, I // G), 1e-5).reshape(T, I)
         * w["norm"].ravel()[None, :])
    out = yn @ w["out"].T
    conv_out = None
    if conv_state is not None:
        cs = np.concatenate([conv_state, xBC], axis=0)       # [K-1+T, C]
        conv_out = cs[-(w["conv"].shape[1] - 1):, :]
    return {"out": out, "state": st, "xbc": xBCc, "dt": dt, "yd": yd,
            "yn": yn, "inproj": zx, "conv_state": conv_out}


# ------------------------------------------------------------------ MoE ref
def ref_moe(m, w_rt, w_b, w_up, w_dn, w_shu, w_shd, k_used, scale, f_eps,
            with_routed=True, with_shared=True):
    """HF router math: sigmoid fp32 -> selection on sigmoid+bias -> weights =
    unbiased sigmoid / (sum+eps) * scale; relu^2 non-gated experts; shared
    expert on the same normed input, unscaled. m [T, D]."""
    T = m.shape[0]
    routed = np.zeros((T, m.shape[1]))
    shared = np.zeros((T, m.shape[1]))
    for t in range(T):
        rl = w_rt @ m[t]
        sig = sigmoid(rl)
        choice = sig + (np.ravel(w_b) if w_b is not None else 0.0)
        sel = np.argsort(-choice, kind="stable")[:k_used]
        wsel = sig[sel]
        w8 = wsel / (wsel.sum() + (f_eps if f_eps is not None else 0.0)) * scale
        if with_routed:
            acc = np.zeros(m.shape[1])
            for j, e in enumerate(sel):
                # expert weights read (E, I, D) / (E, D, I): w_up[e] is [I, D]
                hh = np.maximum(w_up[int(e)] @ m[t], 0.0)
                acc += w8[j] * (w_dn[int(e)] @ (hh * hh))
            routed[t] = acc
        if with_shared:
            sh = np.maximum(w_shu @ m[t], 0.0)
            shared[t] = w_shd @ (sh * sh)
    return {"out": routed + shared, "routed": routed, "shared": shared}


# =================================================================== mamba
def scenario_mamba():
    names = ["wm_inproj", "wm_conv", "wm_convb", "wm_dtb", "wm_a", "wm_d",
             "wm_norm", "wm_out"]
    key = {"wm_inproj": "in_proj", "wm_conv": "conv", "wm_convb": "convb",
           "wm_dtb": "dt_bias", "wm_a": "a", "wm_d": "d", "wm_norm": "norm",
           "wm_out": "out"}
    w = {key[n]: load(n) for n in names}
    l = load("m0_in")                                        # [T, D]
    conv_dump = load("m0_state_conv").T                      # [K-1, C] post-prefill
    ssm_dump = load("m0_state_ssm")                          # (NH, HD, S)

    # prefill runs on fresh zero states (buffer cleared at alloc)
    ref = ref_mamba(l, w, None, None, clamp_dt=True)

    # --- SOP #6 layout canary: prove this harness is sensitive to the d0
    # fastest orientation by scoring a deliberately transposed conv weight
    got_xbc = load("m0_xbc")
    l2_right = report("mamba/xbc (conv+bias+silu)", got_xbc, ref["xbc"])
    I, C, NH, HD = mamba_split(w)
    T = l.shape[0]
    xBC_raw = ref["inproj"][:, I:I + C]                      # [T, C]
    cw_wrong = w["conv"].T.copy()                            # pretend [K, C]
    ccw = np.zeros((T, C))
    for t in range(T):
        for j in range(4):
            if t - j >= 0:
                ccw[t] += cw_wrong[j] * xBC_raw[t - j]
    l2_wrong = np.linalg.norm((silu(ccw + w["convb"]) - got_xbc).ravel()) / \
        (np.linalg.norm(got_xbc) + 1e-30)
    assert l2_wrong > 100 * max(l2_right, 1e-9), \
        f"orientation canary insensitive: wrong {l2_wrong:.2e} vs right {l2_right:.2e}"
    print(f"  canary: wrong-orientation rel {l2_wrong:.3e} vs right "
          f"{l2_right:.3e} (sensitivity OK)")

    report("mamba/inproj", load("m0_inproj"), ref["inproj"], tol=1e-5)
    report("mamba/dt (engine inv-softplus round trip vs HF clamped)",
           np.log1p(np.exp(load("m0_dt"))), ref["dt"], tol=1e-3)
    report("mamba/y+Dx", load("m0_yd").reshape(T, I), ref["yd"])
    report("mamba/group-norm out", load("m0_yn"), ref["yn"])
    report("mamba/prefill out vs step-recursion (dt clamped=HF)",
           load("m0_out"), ref["out"])
    report("mamba/state after prefill (conv window)", conv_dump, ref["conv_state"])
    report("mamba/state after prefill (ssm)", ssm_dump, ref["state"], tol=1e-3)
    ref_c = ref_mamba(l, w, None, None, clamp_dt=True, chunk=True)
    report("mamba/ref_selfcheck step-vs-chunk (out)", ref["out"], ref_c["out"],
           tol=1e-9)
    report("mamba/ref_selfcheck step-vs-chunk (state)", ref["state"],
           ref_c["state"], tol=1e-9)
    ref_u = ref_mamba(l, w, None, None, clamp_dt=False)
    report("mamba/prefill out vs UNCLAMPED dt (llama.cpp semantics)",
           load("m0_out"), ref_u["out"], tol=1.0)

    # ---- decode chain: 4 steps, states carried through the dumps
    st_conv, st_ssm = conv_dump, ssm_dump
    for s in range(4):
        lin = load(f"m1_s{s}_in")
        got_o = load(f"m1_s{s}_out")
        r = ref_mamba(lin, w, st_conv, st_ssm, clamp_dt=True)
        report(f"mamba/decode s{s} out", got_o, r["out"])
        got_st = load(f"m1_s{s}_state_ssm")
        report(f"mamba/decode s{s} state", got_st, r["state"], tol=1e-3)
        xBC_new = (lin @ w["in_proj"].T)[:, I:I + C]         # raw xBC window row
        st_conv = np.concatenate([st_conv[1:], xBC_new], axis=0)
        st_ssm = r["state"]


# =================================================================== attn
def scenario_attn():
    wq, wk, wv, wo = (load(n) for n in ["aw_q", "aw_k", "aw_v", "aw_o"])
    l = load("a0_in")                                        # [T, D]
    T = l.shape[0]
    NH, NKV, HD = 8, 2, 16
    report("attn/prefill q (matmul only)", load("a0_q"), l @ wq.T, tol=1e-5)
    # NoPE assertion: engine pre-FA q/k must equal the raw projections exactly
    NH2 = NH * HD
    report("attn/NoPE: q_pre_fa == q (any rope would rotate)",
           load("a0_q_pre_fa").reshape(T, NH2), l @ wq.T, tol=1e-6)
    report("attn/NoPE: k_pre_fa == k",
           load("a0_k_pre_fa").reshape(T, NKV * HD), l @ wk.T, tol=1e-6)
    # attention through the F16 cache the engine's FA actually reads
    kc = load("a0_kcache")                                     # [S, KVD]
    vc = load("a0_vcache")
    kf = kc.reshape(kc.shape[0], NKV, HD)
    vf = vc.reshape(vc.shape[0], NKV, HD)
    q3 = (l @ wq.T).reshape(T, NH, HD)
    attn = gqa_attention(q3, kf, vf, HD, NKV).reshape(T, NH * HD)
    report("attn/prefill FA vs naive numpy (F16 cache)", load("a0_attn"), attn,
           tol=1e-3)
    report("attn/prefill out", load("a0_out"), attn @ wo.T, tol=1e-3)
    # decode steps: KV write-back row + output
    for s in range(3):
        lin = load(f"a1_s{s}_in")
        p = 6 + s
        kc = load(f"a1_s{s}_kcache")
        vc = load(f"a1_s{s}_vcache")
        kf = kc.reshape(kc.shape[0], NKV, HD)
        vf = vc.reshape(vc.shape[0], NKV, HD)
        q1 = (lin @ wq.T).reshape(NH, HD)
        out = gqa_attention(q1[None, :, :], kf, vf, HD, NKV,
                            q_pos=[p])[0].reshape(1, NH * HD)
        report(f"attn/decode s{s} out", load(f"a1_s{s}_out"), out @ wo.T,
               tol=1e-3)   # fattn-vs-numpy noise, same scale as the prefill FA
        report(f"attn/decode s{s} K row {p} (F16)", kf[p].ravel(),
               (lin @ wk.T).ravel(), tol=1e-3)
        report(f"attn/decode s{s} V row {p} (F16)", vf[p].ravel(),
               (lin @ wv.T).ravel(), tol=1e-3)


# =================================================================== moe
def scenario_moe():
    rt, pb, up, dn, shu, shd = (load(n) for n in
                                ["ew_rt", "ew_b", "ew_up", "ew_dn",
                                 "ew_shu", "ew_shd"])
    m = load("e0_in")                                        # [T, D]
    rl = m @ rt.T
    report("moe/router logits (fp32)", load("e0_rlogits"), rl, tol=1e-5)
    got_sel = load("e0_sel")
    assert np.allclose(got_sel, np.round(got_sel)), "sel must be integer ids"
    ref_sel = np.argsort(-(sigmoid(rl) + pb.ravel()), axis=1, kind="stable")[:, :3]
    report("moe/top-k ids (biased selection)", got_sel, ref_sel.astype(float),
           tol=0.5)
    r = ref_moe(m, rt, pb, up, dn, shu, shd, 3, 2.5, 1e-20)
    report("moe/out (routed + shared)", load("e0_out"), r["out"])
    report("moe/routed sum", load("e0_routed"), r["routed"])
    report("moe/shared expert (isolation)", load("e0_shexp"), r["shared"])
    T, K = m.shape[0], 3
    got_w8 = load("e0_w8")                    # (T, K) aligned with e0_sel
    got_sel = load("e0_sel").astype(int)
    bad = 0.0
    for t in range(T):
        sig = sigmoid(rl[t])
        sel = np.argsort(-(sig + pb.ravel()), kind="stable")[:K]
        ws = sig[sel] / (sig[sel].sum() + 1e-20) * 2.5
        ref_by_id = {int(e): float(wv) for e, wv in zip(sel, ws)}
        got_by_id = {int(e): float(wv) for e, wv in zip(got_sel[t], got_w8[t])}
        if set(ref_by_id) != set(got_by_id):
            bad = 1.0
            continue
        for e in ref_by_id:
            bad = max(bad, abs(ref_by_id[e] - got_by_id[e]))
    REPORT.append(("moe/weights by expert id (set + value)", bad, bad,
                   "PASS" if bad < 1e-4 else "FAIL"))
    if bad >= 1e-4:
        FAILS.append("moe/weights by expert id")


# =================================================================== chain
def scenario_chain():
    names = ["w_emb", "w_n0", "w_n1", "w_n2", "w_l0_inproj", "w_l0_conv",
             "w_l0_convb", "w_l0_dtb", "w_l0_a", "w_l0_d", "w_l0_mnorm",
             "w_l0_out", "w_l1_q", "w_l1_k", "w_l1_v", "w_l1_o", "w_l2_rt",
             "w_l2_b", "w_l2_up", "w_l2_dn", "w_l2_shu", "w_l2_shd", "w_fn",
             "w_head"]
    w = {n: load(n) for n in names}
    toks = np.array([3, 17, 9, 41, 2])
    NH, NKV, HD = 8, 2, 16
    emb = w["w_emb"][toks, :]                                # [T, D]
    report("chain/emb (get_rows)", load("emb"), emb, tol=1e-6)
    # layer 0: Mamba2 (norm -> branch -> residual)
    l0 = rms_norm(emb, 1e-5) * w["w_n0"].ravel()[None, :]
    report("chain/c0_in (rms_norm*n)", load("c0_in"), l0, tol=1e-5)
    mref = ref_mamba(l0, {"in_proj": w["w_l0_inproj"], "conv": w["w_l0_conv"],
                          "convb": w["w_l0_convb"], "dt_bias": w["w_l0_dtb"],
                          "a": w["w_l0_a"], "d": w["w_l0_d"],
                          "norm": w["w_l0_mnorm"], "out": w["w_l0_out"]},
                     clamp_dt=True)
    report("chain/c0_out (mamba branch)", load("c0_out"), mref["out"], tol=1e-4)
    x1 = emb + mref["out"]
    # layer 1: NoPE attention through the F16 caches
    l1 = rms_norm(x1, 1e-5) * w["w_n1"].ravel()[None, :]
    report("chain/c1_in (rms_norm*n)", load("c1_in"), l1, tol=1e-5)
    kc = load("c_kcache")
    vc = load("c_vcache")
    kf = kc.reshape(kc.shape[0], NKV, HD)
    vf = vc.reshape(vc.shape[0], NKV, HD)
    T = l1.shape[0]
    q3 = (l1 @ w["w_l1_q"].T).reshape(T, NH, HD)
    attn = gqa_attention(q3, kf, vf, HD, NKV).reshape(T, NH * HD)
    b1 = attn @ w["w_l1_o"].T
    report("chain/c1_out (attn branch, F16 cache)", load("c1_out"), b1, tol=1e-3)
    x2 = x1 + b1
    # layer 2: MoE
    l2 = rms_norm(x2, 1e-5) * w["w_n2"].ravel()[None, :]
    report("chain/c2_in (rms_norm*n)", load("c2_in"), l2, tol=1e-3)
    b2 = ref_moe(l2, w["w_l2_rt"], w["w_l2_b"], w["w_l2_up"], w["w_l2_dn"],
                 w["w_l2_shu"], w["w_l2_shd"], 3, 2.5, 1e-20)
    report("chain/c2_out (moe branch)", load("c2_out"), b2["out"], tol=1e-3)
    x3 = x2 + b2["out"]
    fin = rms_norm(x3, 1e-5) * w["w_fn"].ravel()[None, :]
    report("chain/final_norm", load("t_final_norm"), fin, tol=1e-3)
    report("chain/logits (untied head, last position)", load("logits").ravel(),
           w["w_head"] @ fin[-1], tol=1e-3)


if __name__ == "__main__":
    print(f"dumps: {DIR}")
    scenario_mamba()
    scenario_attn()
    scenario_moe()
    scenario_chain()
    print()
    print(f"{'check':58s} {'max_abs':>10s} {'rel_L2':>10s}  verdict")
    for tag, ma, l2, v in REPORT:
        print(f"{tag:58s} {ma:10.3e} {l2:10.3e}  {v}")
    print()
    if FAILS:
        print(f"RESULT: FAIL ({len(FAILS)}): {FAILS}")
        sys.exit(1)
    print("RESULT: ALL PASS")
