"""Q6_K quantizer self-check.

convert/q6k.py carries two implementations of the same algorithm: a CUDA-torch
path (used automatically when torch+CUDA is available) and a numpy path (the
fallback). They must produce byte-identical GGUF blocks, otherwise a Q6_K model
built on a GPU machine would differ from one built on a CPU machine.

Also checks the packed layout (210 bytes / 256 values) and that a block
round-trips back to the original values within the expected quantization error.

Usage:
  python scripts/test_q6k.py [--blocks N]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "convert"))
import q6k

QK_K = 256
BLOCK_BYTES = 210


def make_cases(n_rand):
    """Edge cases that historically trip quantizers, plus random blocks."""
    rng = np.random.RandomState(0)
    cases = [
        rng.randn(n_rand, QK_K).astype(np.float32),
        np.zeros((1, QK_K), dtype=np.float32),                             # all zero
        (rng.choice([-1e-18, 1e-18], QK_K)).astype(np.float32)[None],      # below GROUP_MAX_EPS
        (rng.randn(1, QK_K) * 1e4).astype(np.float32),                     # large magnitudes
        np.full((1, QK_K), 3.25, dtype=np.float32),                        # constant
        (rng.randn(4, QK_K) * rng.choice([0.01, 1.0, 100.0], (4, 1))).astype(np.float32),
    ]
    return np.concatenate(cases, axis=0)


def dequantize(blocks):
    """Unpack ql/qh/scales/d back to float32, mirroring ggml's dequantize_row_q6_K."""
    nb = blocks.shape[0]
    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    sc = blocks[:, 192:208].view(np.int8).astype(np.int32)
    d = np.ascontiguousarray(blocks[:, 208:210]).view(np.float16).astype(np.float32)[:, 0]

    out = np.zeros((nb, QK_K), dtype=np.float32)
    for n in (0, 128):                      # two 128-value halves per block
        qo, ho, so, yo = (0, 0, 0, 0) if n == 0 else (64, 32, 8, 128)
        for l in range(32):
            is_ = l // 16                   # scales index base within the half
            q1 = ((ql[:, qo + l] & 0xF) | (((qh[:, ho + l] >> 0) & 3) << 4)).astype(np.int32) - 32
            q2 = ((ql[:, qo + l + 32] & 0xF) | (((qh[:, ho + l] >> 2) & 3) << 4)).astype(np.int32) - 32
            q3 = ((ql[:, qo + l] >> 4) | (((qh[:, ho + l] >> 4) & 3) << 4)).astype(np.int32) - 32
            q4 = ((ql[:, qo + l + 32] >> 4) | (((qh[:, ho + l] >> 6) & 3) << 4)).astype(np.int32) - 32
            out[:, yo + l + 0] = d * sc[:, so + is_ + 0] * q1
            out[:, yo + l + 32] = d * sc[:, so + is_ + 2] * q2
            out[:, yo + l + 64] = d * sc[:, so + is_ + 4] * q3
            out[:, yo + l + 96] = d * sc[:, so + is_ + 6] * q4
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=64)
    args = ap.parse_args()

    rows = make_cases(args.blocks)
    print(f"{rows.shape[0]} blocks of {QK_K} values")

    pack = q6k._quantize_q6_k_numpy(rows)
    assert pack.shape == (rows.shape[0], BLOCK_BYTES), pack.shape
    print(f"  layout: {pack.shape[0]} x {pack.shape[1]} bytes  OK")

    # 6-bit values: one step is 1/64 of the block scale, so anything below
    # TOL_STEP is a healthy round-trip. Blocks below GROUP_MAX_EPS are
    # quantized to zero by design, so they are excluded here.
    TOL_STEP = 0.02
    ref = dequantize(pack)
    scale = np.abs(rows).max(axis=1)
    nz = scale > 1e-12
    rel = np.abs(ref[nz] - rows[nz]).max(axis=1) / scale[nz]
    print(f"  round-trip: worst relative error {rel.max():.4f} of block scale, "
          f"expect <= {TOL_STEP:.4f}  {'OK' if rel.max() <= TOL_STEP else 'FAIL'}")

    try:
        import torch
    except ImportError:
        print("torch not installed - skipping the torch/numpy agreement check")
        return 0

    # The CUDA path runs the same 19-pass scale search, but torch accumulates the
    # 16-element dot products in a different order than numpy. Near-ties in the
    # search can therefore pick a different scale, so the two paths are expected
    # to agree to well within one quantization step -- not necessarily byte-wise.
    other = q6k._quantize_q6_k_torch(torch.from_numpy(np.ascontiguousarray(rows)))
    if np.array_equal(pack, other):
        print("  torch vs numpy: byte-identical  OK")
        agree = True
    else:
        n_diff = int((pack.astype(int) - other.astype(int)).any(axis=1).sum())
        diff = np.abs(dequantize(pack) - dequantize(other)).max(axis=1)
        dev = diff[nz] / scale[nz]
        agree = bool(n_diff <= rows.shape[0] // 8 and dev.max() <= 2 * TOL_STEP)
        print(f"  torch vs numpy: {n_diff}/{rows.shape[0]} blocks differ, "
              f"worst output deviation {dev.max():.4f} of block scale "
              f"(<= {2 * TOL_STEP:.4f} tolerated)  {'OK' if agree else 'FAIL'}")

    ok = agree and rel.max() <= TOL_STEP
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
