"""Compare C++ (vqa-cli --dump-dir) tensors against the PyTorch reference dumps.

  python scripts/compare_vc3.py --ref <ref_dir> --cpp <dump_dir>

Tensors are matched by semantic role; each side's ggml/numpy layout is
transposed as needed. Reports max abs diff, mean abs diff and cosine sim.
"""
import argparse
import os

import numpy as np


def load_cpp(d, name):
    p = os.path.join(d, f"cpp_{name}.bin")
    if not os.path.exists(p):
        return None
    shape_p = os.path.join(d, f"cpp_{name}.shape")
    ne = [int(x) for x in open(shape_p).read().split()]
    ne = [x for x in ne if x > 1] or [1]
    a = np.fromfile(p, dtype=np.float32)
    try:
        return a.reshape(tuple(reversed(ne)))   # ggml ne -> numpy shape
    except ValueError:
        return a


def report(tag, ref, cpp):
    if cpp is None or ref is None:
        print(f"{tag:14s} SKIP (missing)")
        return
    r = np.asarray(ref, dtype=np.float64).ravel()
    c = np.asarray(cpp, dtype=np.float64).ravel()
    if r.size != c.size:
        print(f"{tag:14s} SHAPE MISMATCH ref={np.asarray(ref).shape} cpp={np.asarray(cpp).shape}")
        return
    d = np.abs(r - c)
    denom = (np.linalg.norm(r) * np.linalg.norm(c))
    cos = float(np.dot(r, c) / denom) if denom > 0 else float("nan")
    rng = float(np.abs(r).max()) or 1.0
    print(f"{tag:14s} n={r.size:8d} maxabs={d.max():.3e} rel={d.max()/rng:.3e} mean={d.mean():.3e} cos={cos:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cpp", required=True)
    args = ap.parse_args()

    def npy(name):
        p = os.path.join(args.ref, name)
        return np.load(p) if os.path.exists(p) else None

    # ---- preprocessing: ref [N, 588] vs cpp ggml [588, N]
    rp = npy("ref_patches.npy")
    cp = load_cpp(args.cpp, "patches")
    if rp is not None and cp is not None:
        report("patches.T", rp.T, cp)
        report("patches(flat)", rp.ravel(), cp.ravel())   # sane only if both row-major

    # ---- vision tower stage by stage (ref layout [N, D] vs cpp ggml [D, N])
    report("v_patch_out", (npy("ref_vpatch.npy") or np.zeros(0)).T if npy("ref_vpatch.npy") is not None else None,
           load_cpp(args.cpp, "v_patch_out"))
    rb0 = npy("ref_vblk0.npy")
    report("v_blk0_out", rb0.T if rb0 is not None else None, load_cpp(args.cpp, "v_blk0_out"))

    rm = npy("ref_vmerge.npy")       # [n, 4608]
    cm = load_cpp(args.cpp, "v_merge")   # [1152, 4, n]
    if rm is not None and cm is not None:
        cm2 = np.transpose(cm, (2, 1, 0))            # [n, 4, 1152]
        report("v_merge", rm.reshape(rm.shape[0], 4, 1152), cm2)

    rp2 = npy("ref_vproj.npy")       # [n, 2560]
    report("v_proj_out", rp2.T if rp2 is not None else None, load_cpp(args.cpp, "v_proj_out"))

    # ---- text
    rl0 = npy("ref_layer0_in.npy")   # [1, T, D]
    if rl0 is not None:
        report("emb", rl0[0].T, load_cpp(args.cpp, "emb"))
    report("logits", npy("ref_logits.npy"), load_cpp(args.cpp, "logits"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
