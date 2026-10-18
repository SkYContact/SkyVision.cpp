"""Compare the SkyVision qwen25omni backend against the HF reference dumps.

Usage:
  python scripts/compare_qwen25omni.py --ref ~/skyvision/models/qwen25omni/ref --cpp /tmp/cpp
"""
import argparse
import json
import os

import numpy as np


def load_bins(path):
    raw = open(path, "rb").read()
    n1, n2 = np.frombuffer(raw[:8], dtype=np.int32)
    return np.frombuffer(raw[8:], dtype=np.float32).reshape(n1, n2)


def report(name, a, b):
    if a is None or b is None:
        print(f"{name:16s} MISSING ({'cpp' if a is None else 'ref'})")
        return None
    n = min(a.shape[0], b.shape[0])
    a2, b2 = a[:n].astype(np.float64), b[:n].astype(np.float64)
    if a.shape != b.shape and a.ndim == b.ndim:
        pass
    d = np.abs(a2 - b2)
    denom = np.maximum(np.abs(b2), 1e-3)
    print(f"{name:16s} shape cpp{a.shape} ref{b.shape}  absmax {d.max():.5f}  "
          f"absmean {d.mean():.6f}  relmean {(d/denom).mean():.5f}")
    return d.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cpp", required=True)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.ref, "meta.json")))

    if os.path.exists(os.path.join(args.ref, "mel.npy")) and \
       os.path.exists(os.path.join(args.cpp, "mel.bin")):
        report("mel", load_bins(os.path.join(args.cpp, "mel.bin")).T,
               np.load(os.path.join(args.ref, "mel.npy")))
    report("audio_feats", load_bins(os.path.join(args.cpp, "aud.bin")) if
           os.path.exists(os.path.join(args.cpp, "aud.bin")) else None,
           np.load(os.path.join(args.ref, "audio_feats.npy")) if
           os.path.exists(os.path.join(args.ref, "audio_feats.npy")) else None)
    report("vision_feats", load_bins(os.path.join(args.cpp, "vis.bin")) if
           os.path.exists(os.path.join(args.cpp, "vis.bin")) else None,
           np.load(os.path.join(args.ref, "vision_feats.npy")) if
           os.path.exists(os.path.join(args.ref, "vision_feats.npy")) else None)
    report("logits0", (np.frombuffer(open(os.path.join(args.cpp, "logits.bin"), "rb").read(),
                                      dtype=np.float32) if
           os.path.exists(os.path.join(args.cpp, "logits.bin")) else None),
           np.load(os.path.join(args.ref, "logits0.npy")) if
           os.path.exists(os.path.join(args.ref, "logits0.npy")) else None)

    pos_ref = np.load(os.path.join(args.ref, "positions.npy")) if \
        os.path.exists(os.path.join(args.ref, "positions.npy")) else None
    if pos_ref is not None:
        print("positions first 12 (t,h,w):")
        print("  ref:", pos_ref[:12, :].T.tolist())
    print("ref gen text:", meta.get("gen_text", ""))


if __name__ == "__main__":
    main()
