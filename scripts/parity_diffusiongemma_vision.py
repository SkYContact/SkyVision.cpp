#!/usr/bin/env python3
"""parity_diffusiongemma_vision.py - compare the engine's vision tower against
the official transformers implementation.

    # engine side (writes host_*.bin - the trustworthy dumps)
    DG_DUMP_DIR=dgvis vqa-cli -m <gguf> -t models/diffusiongemma-26B-A4B-it \\
        --image test_shapes.png --ask x -n 8 --cpu
    # oracle
    python scripts/ref_diffusiongemma.py --src models/diffusiongemma-26B-A4B-it \\
        --engine-dump dgvis --out refvis
    # compare
    python scripts/parity_diffusiongemma_vision.py --ref refvis --cpp dgvis

Why `host_*` and not `cpp_*`: the tower's first tensor (`v_patches`) is a graph
INPUT, and gallocr recycles input buffers after compute, so a dump sink attached
to it reads back whatever ran last. The backend therefore writes the values that
actually enter/leave each stage straight from host memory.

Compared levels (engine -> oracle):
  host_v_patches      vs  (numpy recompute)  the patch embedding is checked
                                             separately by ref_diffusiongemma.py
  host_v_pooled       vs  ref_v_final        3x3 mean pool + sqrt(1152) + standardize
  host_v_soft         vs  ref_v_soft         + weightless RMSNorm + embed_vision projection
"""
import argparse
import os
import sys

import numpy as np

# (engine dump name, oracle dump name, tolerance on mean|d| / mean|ref|)
PAIRS = [
    ("host_v_pooled", "ref_v_final", 3e-2),
    ("host_v_soft", "ref_v_soft", 3e-2),
]


def load(d, name):
    b = os.path.join(d, name + ".bin")
    s = os.path.join(d, name + ".shape")
    if not os.path.exists(b):
        return None
    a = np.fromfile(b, dtype=np.float32)
    if os.path.exists(s):
        ne = [int(x) for x in open(s).read().split()]
        a = a.reshape([x for x in ne if x != 1][::-1])
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", default="refvis")
    ap.add_argument("--cpp", default="dgvis")
    args = ap.parse_args()

    bad = 0
    seen = 0
    for cpp_name, ref_name, thr in PAIRS:
        cpp = load(args.cpp, cpp_name)
        ref = load(args.ref, ref_name)
        if cpp is None:
            print(f"{cpp_name:16s} MISSING engine dump")
            bad += 1
            continue
        if ref is None:
            print(f"{cpp_name:16s} MISSING oracle dump (run scripts/ref_diffusiongemma.py)")
            bad += 1
            continue
        seen += 1
        if cpp.size != ref.size:
            print(f"{cpp_name:16s} SIZE MISMATCH engine={cpp.shape} oracle={ref.shape}")
            bad += 1
            continue
        a = cpp.reshape(-1).astype(np.float64)
        b = ref.reshape(-1).astype(np.float64)
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        diff = np.abs(a - b)
        rel = float(diff.mean() / (np.abs(b).mean() + 1e-9))
        ok = rel <= thr
        bad += 0 if ok else 1
        print(f"{cpp_name:16s} vs {ref_name:12s} {str(cpp.shape):18s} "
              f"corr={corr:.6f} mean|d|={diff.mean():.3e} max|d|={diff.max():.3e} "
              f"rel={rel:.3e} thr={thr:g} {'OK' if ok else 'FAIL'}")
    if seen == 0:
        sys.exit("nothing compared - check --cpp / --ref")
    print("PARITY:", "PASS" if bad == 0 else f"FAIL ({bad} level(s))")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
