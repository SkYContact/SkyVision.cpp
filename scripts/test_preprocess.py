"""Preprocess parity: official PIL pipeline (image_crops.py overlap_crop_image +
vision.py prepare_crops) vs C++ moondream-preprocess-test.

The C++ Lanczos replicates PIL's fixpoint 8bpc resampler, so crops should be
bit-exact (maxabs 0); tolerance 1e-3 kept for safety.
"""
import importlib.util
import os
import struct
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = r"D:/moondream.cpp"
EXE = ROOT + "/build/Release/moondream-preprocess-test.exe"
PARITY_DIR = ROOT + "/parity_preprocess"

_crops_py = None
for _p in ("/tmp/md31_image_crops.py",
           os.path.expandvars(r"%LOCALAPPDATA%/Temp/md31_image_crops.py"),
           r"C:/Users/41601/AppData/Local/Temp/md31_image_crops.py"):
    if os.path.exists(_p):
        _crops_py = _p
        break
if not _crops_py:
    sys.exit("md31_image_crops.py not found")
spec = importlib.util.spec_from_file_location("md31_image_crops", _crops_py)
image_crops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_crops)

MAX_CROPS, OVERLAP, CROP = 12, 4, 378


def make_test_image(w, h, seed):
    img = Image.new("RGB", (w, h), (30, 90, 160))
    d = ImageDraw.Draw(img)
    rng = np.random.RandomState(seed)
    for _ in range(12):
        x0, y0 = rng.randint(0, w - 80), rng.randint(0, h - 80)
        x1, y1 = x0 + rng.randint(30, 80), y0 + rng.randint(30, 80)
        col = tuple(int(c) for c in rng.randint(0, 255, 3))
        if rng.rand() < 0.5:
            d.ellipse([x0, y0, x1, y1], fill=col)
        else:
            d.rectangle([x0, y0, x1, y1], fill=col)
    for _ in range(6):  # high-frequency detail (stress test for Lanczos)
        x0, y0 = rng.randint(0, w - 1), rng.randint(0, h - 1)
        x1, y1 = rng.randint(0, w - 1), rng.randint(0, h - 1)
        d.line([x0, y0, x1, y1], fill=tuple(int(c) for c in rng.randint(0, 255, 3)), width=3)
    return img


def maxabs(a, b):
    return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).max())


def run_case(name, img):
    np_img = np.array(img.convert("RGB"))
    h, w = np_img.shape[:2]

    oc = image_crops.overlap_crop_image(np_img, max_crops=MAX_CROPS, overlap_margin=OVERLAP)
    tiling = oc["tiling"]
    ref = np.transpose(oc["crops"], (0, 3, 1, 2)).astype(np.float32) / 255.0
    ref = (ref - 0.5) / 0.5

    os.makedirs(PARITY_DIR, exist_ok=True)
    img_bin, out_bin = PARITY_DIR + "/img.bin", PARITY_DIR + "/out.bin"
    np_img.tofile(img_bin)
    r = subprocess.run([EXE, "--img", img_bin, "--w", str(w), "--h", str(h),
                        "--out", out_bin], capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        return False, 0.0, 0
    n, th, tw = struct.unpack("<3i", open(out_bin, "rb").read(12))
    cpp = np.fromfile(out_bin, dtype=np.float32, offset=12)

    ok = (n, th, tw) == (ref.shape[0], tiling[0], tiling[1])
    d = maxabs(cpp.reshape(ref.shape), ref) if ok else float("inf")
    print(f"  {name} ({w}x{h}): cpp n={n} tiling=({th},{tw})  ref n={ref.shape[0]} "
          f"tiling={tiling}  maxabs {d:.3e}  {'OK' if ok and d < 1e-3 else 'FAIL'}")
    return ok and d < 1e-3, d, n


def main():
    cases = [
        ("square 768", make_test_image(768, 768, 1)),
        ("wide 1024x700", make_test_image(1024, 700, 2)),
        ("tall 640x1024", make_test_image(640, 1024, 3)),
        ("small (single crop) 300x300", make_test_image(300, 300, 4)),
        ("huge 2048x2048", make_test_image(2048, 2048, 5)),
        ("extreme wide 2200x400", make_test_image(2200, 400, 6)),
    ]
    ok = True
    for name, img in cases:
        c, _, _ = run_case(name, img)
        ok = ok and c
    print("PARITY PASS" if ok else "PARITY FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
