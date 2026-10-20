"""Stage-by-stage ablation: MiniCPM-V 4.7 C++ backend vs the transformers oracle.

Runs vqa-cli with --dump-dir, then compares every stage the engine dumps
(v_patch_out / vm_out / v_tower_out / v_proj_out / logits) against the tensors
scripts/ref_minicpmv47.py captured. A first-stage mismatch localizes the bug:
preprocessing -> patches, patch embed, pos table -> tower -> merger -> text.

  python scripts/ref_minicpmv47.py --image photo.jpg --out ref_cpm.npz
  python scripts/parity_minicpmv47.py --ref ref_cpm.npz --cli build/vqa-cli \
      --model cpm-f16.gguf --tokenizer models/MiniCPM-V-4.7-1B --image photo.jpg

Canvas M-RoPE only (fast: the oracle loads no weights, the engine skips the text
graph, so this works for sliced images AND frame directories):

  python scripts/ref_minicpmv47.py --image big.jpg --pos-only --out ref_pos.npz
  python scripts/parity_minicpmv47.py --ref ref_pos.npz --pos --image big.jpg ...
  python scripts/ref_minicpmv47.py --frames-dir frames --pos-only --out ref_vid.npz
  python scripts/parity_minicpmv47.py --ref ref_vid.npz --pos --frames-dir frames ...

# 4x: add --4x here and to the ref script (plus MINICPM_DOWNSAMPLE=4x)
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

STAGES = [("v_patch_out", "emb", "patch embed (conv+pos)"),
          ("vm_out", "vm", "vit window merger (16x only)"),
          ("v_tower_out", "tower", "tower out (post-LN)"),
          ("v_proj_out", "feats", "projection feats"),
          ("logits", "logits", "last-token logits")]


def load_dump(dump_dir, name):
    shape_file = os.path.join(dump_dir, f"cpp_{name}.shape")
    if not os.path.isfile(shape_file):
        return None
    ne = [int(x) for x in open(shape_file).read().split()]
    return np.fromfile(os.path.join(dump_dir, f"cpp_{name}.bin"),
                       dtype=np.float32).reshape(ne[::-1]).squeeze()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", required=True, help="npz from scripts/ref_minicpmv47.py")
    ap.add_argument("--cli", default="build/vqa-cli")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--image", default="")
    ap.add_argument("--ask", default="Describe the image.")
    ap.add_argument("--dump-dir", default="cpp_minicpm_dump")
    ap.add_argument("--4x", dest="ds4", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="pass --gpu to the CLI (CUDA build)")
    ap.add_argument("--frames-dir", default="", help="compare a video prompt (--frames)")
    ap.add_argument("--pos", action="store_true",
                    help="only compare the canvas M-RoPE position ids "
                         "(ref must be produced with --pos-only)")
    args = ap.parse_args()

    env = dict(os.environ)
    if args.ds4:
        env["MINICPM_DOWNSAMPLE"] = "4x"
    if args.pos:
        env["MINICPM_DUMP_MPOS"] = "1"
    cmd = [args.cli, "--model", args.model, "--ask", args.ask,
           "--max-tokens", "1", "--quiet", "--dump-dir", args.dump_dir]
    if args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if args.gpu:
        cmd += ["--gpu"]
    if args.frames_dir:
        cmd += ["--frames", args.frames_dir]
    elif args.image:
        cmd += ["--image", args.image]
    run = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True,
                         errors="replace")
    stderr = run.stderr

    ref = np.load(args.ref)
    if args.pos:
        # canvas M-RoPE: exact integer comparison of the (t, h, w) rows
        if "pos" not in ref.files:
            print("ref npz has no 'pos' (run ref_minicpmv47.py --pos-only)")
            return 1
        mpos = {}
        for line in stderr.splitlines():
            m = re.match(r"minicpm mpos (t|h|w|delta): ?(.*)", line.strip())
            if m:
                mpos[m.group(1)] = [int(x) for x in m.group(2).split()]
        if len(mpos) != 4:
            print("engine printed no mpos (need MINICPM_DUMP_MPOS=1)")
            return 1
        pos = ref["pos"]
        okp = pos.shape[1] == len(mpos["t"])
        for k, name in enumerate(("t", "h", "w")):
            a = np.array(mpos[name])
            bad = 0 if a.size != pos.shape[1] else int(np.count_nonzero(a != pos[k]))
            print(f"mpos {name}: {bad} mismatch(es) over {pos.shape[1]} tokens")
            okp = okp and bad == 0
        print(f"mpos delta: engine {mpos['delta']} oracle {ref['delta'].reshape(-1).tolist()}")
        print("CANVAS M-ROPE IDENTICAL" if okp else "CANVAS MISMATCH")
        return 0 if okp else 1
    print(f"# ref stages: { {k: ref[k].shape for k in ref.files if k != 'gen'} }")
    ok = True
    for cpp_name, ref_key, label in STAGES:
        a = load_dump(args.dump_dir, cpp_name)
        if a is None:
            print(f"{label:28s} MISSING dump (cpp_{cpp_name})")
            ok = False
            continue
        b = ref[ref_key]
        if b.size <= 1152 and ref_key == "vm" and ref_key not in ref.files:
            continue
        a = np.asarray(a, np.float64).reshape(-1)
        b = np.asarray(b, np.float64).reshape(-1)
        if a.size != b.size:
            print(f"{label:28s} shape mismatch cpp={a.size} ref={b.size}")
            ok = False
            continue
        md = float(np.abs(a - b).max())
        corr = float(np.corrcoef(a, b)[0, 1])
        print(f"{label:28s} n={a.size:8d} corr={corr:.6f} maxdiff={md:.4g}")
        ok = ok and corr > 0.999

    lg = load_dump(args.dump_dir, "logits")
    if lg is not None:
        top_cpp = np.argsort(-lg.reshape(-1))[:8].tolist()
        top_ref = np.argsort(-ref["logits"].reshape(-1))[:8].tolist()
        print(f"top8 cpp {top_cpp}")
        print(f"top8 ref {top_ref}")
        print(f"ref greedy {ref['gen'][:8].tolist()}")
        ok = ok and top_cpp == top_ref
    print("PARITY OK" if ok else "PARITY MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
