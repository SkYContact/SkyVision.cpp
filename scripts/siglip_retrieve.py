"""Image retrieval over a SigLIP embedding gallery.

Builds (and caches) a gallery of L2-normalized image embeddings by driving
`vqa-cli --image <img> --embed ""` once per image, then embeds the query the
same way and ranks the gallery by cosine similarity:

  python scripts/siglip_retrieve.py --gallery parity/siglp/gallery \
      --query parity/siglp/ff_gradients.png [--topk 5]

  python scripts/siglip_retrieve.py --gallery parity/siglp/gallery \
      --query test_image.png --rebuild   # ignore the cache

The gallery cache lives next to the gallery as gallery.jsonl (one
{"file", "embedding"} JSON line per image), so adding images only embeds the
new ones.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def embed(cli, gguf, path):
    """One image -> normalized embedding via vqa-cli --embed."""
    env = dict(os.environ)
    dll = os.path.join(ROOT, "build-h", "bin", "Release")
    if os.path.isdir(dll):
        env["PATH"] = dll + os.pathsep + env.get("PATH", "")
    r = subprocess.run([cli, "--model", gguf, "--image", path, "--embed", "",
                        "--cpu", "--quiet"],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit(f"vqa-cli failed on {path}:\n{r.stderr[-800:]}")
    obj = json.loads(r.stdout)
    return obj["embedding"]


def load_gallery(cli, gguf, gallery, rebuild):
    cache = os.path.join(gallery, "gallery.jsonl")
    rows = {}
    if os.path.isfile(cache) and not rebuild:
        with open(cache, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    rows[o["file"]] = o["embedding"]
    files = sorted(f for f in os.listdir(gallery)
                   if f.lower().endswith(IMG_EXTS))
    new = [f for f in files if f not in rows]
    for name in new:
        path = os.path.join(gallery, name)
        print(f"[gallery] embedding {name}", file=sys.stderr)
        rows[name] = embed(cli, gguf, path)
    if new:
        with open(cache, "w", encoding="utf-8") as f:
            for name in files:
                f.write(json.dumps({"file": name, "embedding": rows[name]}) + "\n")
    return [(name, rows[name]) for name in files if name in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--cli", default=os.path.join(ROOT, "build-h", "Release", "vqa-cli.exe"))
    ap.add_argument("--gguf", default=os.path.join(ROOT, "models", "siglip-base", "siglip-f16.gguf"))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    gallery = load_gallery(args.cli, args.gguf, args.gallery, args.rebuild)
    q = np.asarray(embed(args.cli, args.gguf, args.query), dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)

    scored = []
    for name, vec in gallery:
        v = np.asarray(vec, dtype=np.float64)
        v /= max(np.linalg.norm(v), 1e-12)
        scored.append((float(q @ v), name))
    scored.sort(reverse=True)

    print(f"query: {os.path.basename(args.query)}   gallery: {len(gallery)} images")
    print(f"{'rank':>4}  {'cosine':>8}  file")
    for i, (s, name) in enumerate(scored[: args.topk]):
        print(f"{i:>4}  {s:8.6f}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
