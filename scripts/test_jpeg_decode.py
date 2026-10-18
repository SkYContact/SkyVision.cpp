#!/usr/bin/env python3
"""Byte-exact JPEG decode comparison: SkyVision.cpp engine vs PIL.

Both stack onto the libjpeg family, so RGB output must be BIT-IDENTICAL for
any baseline/progressive JPEG with any subsampling (the IDCT is the same
islow kernel). The old stb_image path differs by 1-3 gray levels, so the
script also dumps through stb as a control: if stb matches too, the engine
switch did not happen.

Usage:
    python scripts/test_jpeg_decode.py \
        --dump <path-to media-decode-dump(.exe)> [--keep <dir>]

Test media: generated from repo PNGs (baseline/progressive/4:2:0/4:2:2/gray)
plus real pipeline JPEGs via --extra <file> (repeatable). CMYK is reported as
an informational known difference (PIL applies Adobe inverted-CMYK semantics,
our decoder is plain 255-x), not a failure.
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def engine_dump(dump_exe, img_path, out_path, mode):
    """mode: engine (production dispatch) or stb (forced legacy path)."""
    r = subprocess.run([dump_exe, img_path, out_path, mode],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{mode} dump failed for {img_path}: {r.stderr.strip()}")
    with open(out_path, "rb") as f:
        blob = f.read()
    w, h = struct.unpack("<ii", blob[:8])
    return w, h, blob[8:]


def pil_rgb(img_path):
    im = Image.open(img_path)
    im = im.convert("RGB")
    return im.size, im.tobytes()


def make_synthetic(tmpdir):
    """Baseline/progressive/4:2:0/4:2:2/grayscale JPEGs from the repo PNG."""
    src = Image.open(os.path.join(ROOT, "test_image.png")).convert("RGB")
    src = src.resize((src.width * 2, src.height * 2))  # 96x64 -> enough DCT blocks
    cases = []
    specs = [
        ("synth_baseline_444.jpg", dict(quality=92, subsampling=0)),
        ("synth_progressive_420.jpg", dict(quality=88, subsampling=2, progressive=True)),
        ("synth_422.jpg", dict(quality=95, subsampling=1)),
        ("syntr_opt.jpg", dict(quality=80, subsampling=2, optimize=True)),
    ]
    for name, kw in specs:
        p = os.path.join(tmpdir, name)
        src.save(p, **kw)
        cases.append((name, p))
    gray = Image.open(os.path.join(ROOT, "test_image.png")).convert("L")
    p = os.path.join(tmpdir, "synth_gray.jpg")
    gray.save(p, quality=95)
    cases.append(("gray", p))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(
        ROOT, "build", "Release", "media-decode-dump.exe"))
    ap.add_argument("--extra", action="append", default=[],
                    help="real JPEG to test (repeatable)")
    ap.add_argument("--keep", default="", help="dir to keep generated jpegs")
    args = ap.parse_args()
    if not os.path.isfile(args.dump):
        sys.exit(f"dump tool not found: {args.dump} (build target media-decode-dump)")

    tmpdir = args.keep or tempfile.mkdtemp(prefix="jpeg_ab_")
    os.makedirs(tmpdir, exist_ok=True)

    cases = make_synthetic(tmpdir)
    for i, p in enumerate(args.extra):
        if not os.path.isfile(p):
            sys.exit(f"--extra not found: {p}")
        cases.append((os.path.basename(p), p))

    print(f"{'case':32} {'size':12} {'engine vs PIL':16} {'stb vs PIL':16} verdict")
    print("-" * 96)
    n_ok = n_stb_diff = 0
    failures = []
    for name, path in cases:
        out_e = os.path.join(tmpdir, "dump_engine.rgb")
        out_s = os.path.join(tmpdir, "dump_stb.rgb")
        (w, h), pil = pil_rgb(path)
        we, he, px_e = engine_dump(args.dump, path, out_e, "engine")
        ws, hs, px_s = engine_dump(args.dump, path, out_s, "stb")
        if (we, he) != (w, h) or (ws, hs) != (w, h):
            failures.append((name, "size mismatch "
                             f"pil={w}x{h} engine={we}x{he} stb={ws}x{hs}"))
            print(f"{name:32} {w}x{h:<7} SIZE MISMATCH")
            continue
        d_e = sum(a != b for a, b in zip(px_e, pil))
        d_s = sum(a != b for a, b in zip(px_s, pil))
        if d_e == 0:
            n_ok += 1
            stb_note = f"{d_s} B diff" if d_s else "identical (suspicious)"
            if d_s:
                n_stb_diff += 1
            print(f"{name:32} {w}x{h:<7} {'EXACT':16} {stb_note:16} PASS")
        else:
            failures.append((name, f"{d_e}/{len(pil)} bytes differ vs PIL"))
            print(f"{name:32} {w}x{h:<7} {d_e} B diff      {d_s} B diff       FAIL")

    print("-" * 96)
    print(f"engine==PIL byte-exact: {n_ok}/{len(cases)}")
    print(f"stb differs (proves engine switch): {n_stb_diff} case(s)")
    if failures:
        print("FAILURES:")
        for n, why in failures:
            print(f"  {n}: {why}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
