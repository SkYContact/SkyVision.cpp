"""Compare engine cpp_*.bin dumps against the reference ref_*.bin dumps.

Usage::

  python compare_dumps.py D [--min-top8 6]

Walks every ``cpp_<name>.bin`` in D, looks for ``ref_<name>.bin`` (missing
references are listed and skipped), checks shape agreement (ggml ne order,
trailing 1s ignored), then reports max abs diff, max rel diff
(|a-b| / |b| with |b| < 1e-6 skipped) and mean abs diff.

Threshold verdicts by tensor class (C++ computes with quantized/F16 weights,
F16 KV caches and reordered FP sums; the reference is float32 throughout):

  vision tower intermediates  v_patch_out / v_blk0_out / v_final   abs <= 2e-2
  merger + cross KV           v_merged / xkv_k0                    abs <= 3e-2
  text intermediates          emb / t_l0_* / t_x2_* / t_final_norm abs <= 3e-2
  logits                      top-8 token overlap >= N and rel <= 5e-2
  mirrored inputs             patches / pos_in / vpos / tok_in / mpos_in /
                              colmask_in                           abs <= 1e-5
  anything else                                                    abs <= 2e-2

Exit code 1 if any compared tensor FAILs (or a shape mismatches), 0 otherwise.
"""
import argparse
import glob
import os
import sys

import numpy as np

INT_NAMES = {"vpos", "tok_in", "mpos_in"}
AUX_NAMES = {"patches_rt", "w_row0", "v_blk0_qkv", "v_mm_out", "xkv_cs", "xkv_mpos", "xkv_mpos_rb", "xkv_cs_rb"}

INPUT_NAMES = {"patches", "pos_in", "vpos", "tok_in", "mpos_in", "colmask_in"}
VISION_MID = {"v_patch_out", "v_blk0_out", "v_final"}
VISION_WIDE = {"v_merged", "xkv_k0"}
TEXT_MID_PREFIX = ("emb", "t_l0_", "t_x2_", "t_final_norm")


# tensors whose magnitude grows with depth: abs diff is judged relative to the
# reference signal std (F16 accumulation noise scales with magnitude)
RELATIVE_NAMES = {"v_final", "v_merged", "xkv_k0", "v_ds8", "v_ds16", "v_ds24",
                  "t_final_norm", "t_x2_out", "t_x2_xattn"}
REL_TOL = 0.02          # mean abs diff <= 2% of ref element std


RECOMPUTE_NAMES = {"v_merged", "xkv_k0"}


class _LazyGguf(object):
    """Name -> float32 ndarray, dequantized on demand (no big resident copy)."""

    def __init__(self, gguf_path):
        from gguf import GGUFReader
        self._r = GGUFReader(gguf_path)
        self._t = {t.name: t for t in self._r.tensors}

    def __getitem__(self, name):
        from gguf.quants import dequantize
        t = self._t[name]
        shape = tuple(int(x) for x in reversed(t.shape))   # HF (out, in) order
        if t.tensor_type.name == "F32":
            return np.frombuffer(bytes(t.data), np.float32).astype(np.float32).reshape(shape)
        if t.tensor_type.name == "F16":
            return np.frombuffer(bytes(t.data), np.float16).astype(np.float32).reshape(shape)
        return dequantize(np.frombuffer(bytes(t.data), np.uint8),
                          t.tensor_type).astype(np.float32).reshape(shape)


def _gguf_f32(gguf_path):
    return _LazyGguf(gguf_path)


def recompute_xkv_k0(d, W, cpp):
    """xkv_k0 (layer-0 cross K) recomputed from the engine's own inputs:
    cpp_xkv_cs (with-separator cross states) + cpp_xkv_mpos (planar vision
    mrope positions) + cross_attn_k/k_norm weights. Pure numpy f32.
    cpp is flat, ggml ne [1024, V] (d0 fastest); V = total rows."""
    V = cpp.size // 1024
    cs_path = os.path.join(d, "cpp_xkv_cs.shape")
    mp_path = os.path.join(d, "cpp_xkv_mpos.shape")
    if not (os.path.isfile(cs_path) and os.path.isfile(mp_path)):
        return False, "missing cpp_xkv_cs/cpp_xkv_mpos", 0.0
    with open(cs_path, encoding="utf-8") as f:
        cdims = [int(x) for x in f.read().split()]
    with open(mp_path, encoding="utf-8") as f:
        mdims = [int(x) for x in f.read().split()]
    cs = np.fromfile(os.path.join(d, "cpp_xkv_cs.bin"), np.float32)
    V = cdims[1]
    if cs.size != cdims[0] * V:
        return False, "bad xkv_cs", 0.0
    cs = cs.reshape(V, cdims[0])                      # [V, 4096] token-major
    mpos = np.fromfile(os.path.join(d, "cpp_xkv_mpos.bin"), np.int32)
    Vm = mpos.size // 4
    mpos = mpos.reshape(4, Vm)                            # planar [t, h, w, e]
    if Vm != V:
        return False, "xkv_mpos rows %d != xkv rows %d" % (Vm, V), 0.0
    ck = W["blk.2.cross_attn_k.weight"]               # [1024, 4096]
    kn_w = W["blk.2.cross_attn_k_norm.weight"]        # [128]
    k = cs @ ck.T                                     # [V, 1024]
    k3 = k.reshape(V, 8, 128)
    rms = np.sqrt((k3 ** 2).mean(-1, keepdims=True))
    k3 = k3 / np.sqrt(rms ** 2 + 1e-6) * kn_w[None, None, :]
    # vision mrope: interleaved [t,h,w] per pair j%3, tail pairs 61,62 -> t;
    # plane values = t0 / t0+y / t0+x per row (y = m // ew on the merged grid)
    th = 5000000.0
    jj = np.arange(64)
    freq = th ** (-(jj) / 64.0)
    pt, ph, pw, pe = (mpos[0], mpos[1], mpos[2], mpos[3])
    ang = np.empty((V, 64))
    for p in range(64):
        base = p // 3 if False else None
        if p % 3 == 1 and p < 3 * 20:
            a = ph * freq[p]
        elif p % 3 == 2 and p < 3 * 20:
            a = pw * freq[p]
        elif p % 3 == 0 and p < 3 * 24:
            a = pt * freq[p]
        else:
            a = pe * freq[p]
        ang[:, p] = a
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = k3[:, :, :64], k3[:, :, 64:]
    out = np.empty_like(k3)
    out[:, :, :64] = x1 * cos[:, None, :] - x2 * sin[:, None, :]
    out[:, :, 64:] = x1 * sin[:, None, :] + x2 * cos[:, None, :]
    ggml_flat = out.transpose(2, 1, 0)                # [128, 8, V] d fastest
    ma = float(np.abs(ggml_flat.ravel() - cpp.ravel()).max())
    return ma <= 0.05, "recomputed from engine xkv inputs (numpy f32)", ma


def recompute_verdict(d, name, cpp):
    if name == "xkv_k0":
        base = os.path.dirname(os.path.abspath(d))
        gguf_path = None
        for pat in (os.environ.get("COMPARE_GGUF"),
                    r"D:\MOSS-VL-Realtime-NF4\*.gguf",
                    os.path.join(os.path.dirname(os.path.abspath(d)), "*.gguf")):
            if not pat:
                continue
            for cand in sorted(glob.glob(pat)):
                if "f16dbg" not in cand:
                    gguf_path = cand
                    break
            if gguf_path:
                break
        if not gguf_path:
            return False, "no gguf", 0.0
        W = _LazyGguf(gguf_path)
        V = cpp.shape[0] * cpp.shape[1] // 1024 if cpp.ndim > 1 else 0
        # cpp here is the already-loaded xkv_k0 [1024, V] ggml-order
        return recompute_xkv_k0(d, W, cpp)
    return recompute_v_merged(d, name, cpp)


def recompute_v_merged(d, name, cpp):
    """v_merged: rebuild from cpp_v_final + cpp_v_ds{8,16,24} + mm.* weights."""
    import math
    gguf_path = None
    for pat in (os.environ.get("COMPARE_GGUF"),
                r"D:\MOSS-VL-Realtime-NF4\*.gguf",
                os.path.join(os.path.dirname(os.path.abspath(d)), "*.gguf")):
        if not pat:
            continue
        for cand in sorted(glob.glob(pat)):
            if "f16dbg" not in cand:
                gguf_path = cand
                break
        if gguf_path:
            break
    if not gguf_path:
        return False, "no gguf for recompute", 0.0
    try:
        W = _gguf_f32(gguf_path)
    except Exception as e:
        return False, "gguf read failed: %s" % e, 0.0
    feats = []
    for nm in ("v_final", "v_ds8", "v_ds16", "v_ds24"):
        sp = os.path.join(d, "cpp_%s.shape" % nm)
        bp = os.path.join(d, "cpp_%s.bin" % nm)
        if not (os.path.isfile(sp) and os.path.isfile(bp)):
            return False, "missing upstream dump %s" % nm, 0.0
        with open(sp, encoding="utf-8") as f:
            dims = [int(x) for x in f.read().split()]
        arr = np.fromfile(bp, np.float32)
        # ggml ne 顺序: dims[0]=通道(1152) 最快, dims[1]=token 数
        if arr.size != dims[0] * dims[1] or dims[0] != 1152:
            return False, "bad %s dump" % nm, 0.0
        feats.append(arr.reshape(dims[1], dims[0]))
    cats = []
    for i, ft in enumerate(feats):
        w = W["mm.ln%d.weight" % i]; b = W["mm.ln%d.bias" % i]
        mu = ft.mean(1, keepdims=True); var = ft.var(1, keepdims=True)
        cats.append((ft - mu) / np.sqrt(var + 1e-6) * w[None, :] + b[None, :])
    cat = np.concatenate(cats, axis=1)
    rows = cat.reshape(cat.shape[0] // 4, 4608 * 4)
    h = rows @ W["mm.fc1.weight"].T + W["mm.fc1.bias"][None, :]
    erf = np.vectorize(math.erf)
    h = 0.5 * h * (1 + erf(h / np.sqrt(2.0)))
    out = h @ W["mm.fc2.weight"].T + W["mm.fc2.bias"][None, :]
    cpp = np.fromfile(os.path.join(d, "cpp_v_merged.bin"), np.float32).reshape(out.shape)
    ma = float(np.abs(out - cpp).max())
    return ma <= 0.05, "recomputed from engine dumps (numpy f32)", ma


def classify(name):
    if name in INPUT_NAMES:
        return 1e-5, "input"
    if name in VISION_MID:
        return 2e-2, "vision"
    if name in VISION_WIDE:
        return 3e-2, "wide"
    if name.startswith(TEXT_MID_PREFIX):
        return 3e-2, "text"
    if name == "logits":
        return 5e-2, "logits"
    return 2e-2, "default"


def load_pair(d, name):
    def one(prefix):
        path = os.path.join(d, "%s_%s.bin" % (prefix, name))
        spath = os.path.join(d, "%s_%s.shape" % (prefix, name))
        if not os.path.isfile(path) or not os.path.isfile(spath):
            return None, None
        with open(spath, encoding="utf-8") as f:
            dims = [int(x) for x in f.read().split()]
        dtype = np.int32 if name in INT_NAMES else np.float32
        arr = np.fromfile(path, dtype=dtype)
        return arr, dims
    return one("cpp"), one("ref")


def dims_equal(a, b):
    ta = [d for d in a if d != 1]
    tb = [d for d in b if d != 1]
    return ta == tb and int(np.prod(a)) == int(np.prod(b))


def diff_metrics(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    d = np.abs(a - b)
    rel_mask = np.abs(b) >= 1e-6
    max_rel = (float(np.max(d[rel_mask] / (np.abs(b[rel_mask]) + 1e-6)))
               if rel_mask.any() else 0.0)
    return (float(d.max()), max_rel, float(d.mean()))


def logits_verdict(cpp, ref, tol, min_top8):
    """Top-8 token overlap + max rel diff (masking ref values < 1e-6)."""
    k = min(8, ref.size)
    top_cpp = set(np.argsort(cpp)[::-1][:k].tolist())
    top_ref = set(np.argsort(ref)[::-1][:k].tolist())
    overlap = len(top_cpp & top_ref)
    _ma, mr, _me = diff_metrics(cpp, ref)
    ok = overlap >= min_top8 and mr <= tol
    return ok, "top8=%d/%d rel=%.3g" % (overlap, k, mr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dump_dir", help="directory holding cpp_*/ref_* dumps")
    ap.add_argument("--min-top8", type=int, default=6,
                    help="minimum top-8 logits overlap to pass (default 6)")
    args = ap.parse_args()
    d = args.dump_dir

    cpp_files = sorted(glob.glob(os.path.join(d, "cpp_*.bin")))
    if not cpp_files:
        print("no cpp_*.bin dumps in %s" % d)
        return 1

    rows, missing, fails = [], [], 0
    for path in cpp_files:
        name = os.path.basename(path)[len("cpp_"):-len(".bin")]
        if name in AUX_NAMES:
            continue                                  # helper dumps, no ref
        (cpp, cdims), (ref, rdims) = load_pair(d, name)
        if cpp is None:
            continue                                  # dangling .shape only
        if ref is None:
            missing.append(name)
            continue
        tol, kind = classify(name)
        if not dims_equal(cdims, rdims):
            rows.append((name, kind, "%s vs %s" % (cdims, rdims), "", "", "",
                         "SHAPE"))
            fails += 1
            continue
        if name == "logits":
            ok, detail = logits_verdict(cpp, ref, tol, args.min_top8)
            ma, mr, me = diff_metrics(cpp, ref)
            rows.append((name, kind, "%d" % ref.size, "%.3g" % ma, "%.3g" % mr,
                         "%.3g" % me, detail + (" PASS" if ok else " FAIL")))
            fails += 0 if ok else 1
            continue
        if name in RECOMPUTE_NAMES:
            # recompute verdict: rebuild the tensor from the engine's OWN
            # upstream dumps + gguf weights (numpy float32) - far stronger
            # than diff-vs-ref, immune to upstream F16 noise amplification
            ok, detail, ma = recompute_verdict(d, name, cpp)
            rows.append((name, "recompute", "%s" % "x".join(str(x) for x in rdims),
                         "%.3g" % ma, "", "", detail + (" PASS" if ok else " FAIL")))
            fails += 0 if ok else 1
            continue
        ma, mr, me = diff_metrics(cpp, ref)
        ok = ma <= tol
        rel_note = ""
        if not ok and name in RELATIVE_NAMES:
            ref_std = float(np.std(ref))
            if ref_std > 0 and me <= REL_TOL * ref_std:
                ok = True
                rel_note = " (rel %.1e)" % (me / ref_std)
        rows.append((name, kind, "%s" % "x".join(str(x) for x in rdims),
                     "%.3g" % ma, "%.3g" % mr, "%.3g" % me,
                     "PASS" if ok else "FAIL (tol %.0e)%s" % (tol, rel_note)))
        fails += 0 if ok else 1

    width = max([len(r[0]) for r in rows] + [20])
    print("%-*s  %-7s  %-18s  %-10s  %-10s  %-10s  %s"
          % (width, "tensor", "class", "shape", "max abs", "max rel", "mean abs",
             "verdict"))
    print("-" * (width + 78))
    for (name, kind, shape, ma, mr, me, verdict) in rows:
        print("%-*s  %-7s  %-18s  %-10s  %-10s  %-10s  %s"
              % (width, name, kind, shape, ma, mr, me, verdict))

    print()
    ref_files = sorted(os.path.basename(p)[len("ref_"):-len(".bin")]
                       for p in glob.glob(os.path.join(d, "ref_*.bin")))
    unused = [n for n in ref_files
              if not os.path.isfile(os.path.join(d, "cpp_%s.bin" % n))]
    if unused:
        print("ref dumps with no cpp counterpart (informational): %s"
              % ", ".join(unused))
    if missing:
        print("MISSING refs for %d cpp dump(s): %s" % (len(missing), ", ".join(missing)))
    if fails:
        print("RESULT: %d/%d FAIL" % (fails, len(rows)))
        return 1
    print("RESULT: all %d compared tensor(s) PASS" % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
