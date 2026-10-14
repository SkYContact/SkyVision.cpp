"""Region head parity: official torch (ref_region.py math) vs C++ engine.

1. converts region weights to moondream3.1-region.gguf (convert_region.py)
2. runs the reference encode/decode functions in torch F32 (BF16 ckpt upcast)
3. runs build/Release/moondream-region-test.exe on the same inputs
4. compares embeddings / logits

target: maxabs < 1e-3 (both sides F32, same weights)
"""
import glob
import os
import struct
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = r"D:/moondream.cpp"
GGUF = ROOT + "/moondream3.1-region.gguf"
EXE = ROOT + "/build/Release/moondream-region-test.exe"
PARITY_DIR = ROOT + "/parity_region"
ST = glob.glob(r"C:/Users/41601/.cache/huggingface/hub/models--moondream--moondream3.1-9B-A2B/snapshots/*/model.safetensors")[0]

DIM, COORD_FEAT, COORD_OUT = 2048, 256, 1024
SIZE_FEAT, SIZE_OUT = 512, 2048


class SfReader:
    def __init__(self, path):
        import json as _json
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.hdr = _json.loads(f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def get_f32(self, name):
        info = self.hdr[name]
        assert info["dtype"] == "BF16", (name, info["dtype"])
        s, e = info["data_offsets"]
        bits = self.mm[self.base + s:self.base + e].view(np.uint16)
        f32 = (bits.astype(np.uint32) << 16).view(np.float32)
        return torch.from_numpy(f32.reshape(info["shape"]).copy())


def load_ref():
    rd = SfReader(ST)
    P = "region_model."
    w = type("W", (), {})()
    # torch math convention: fourier uses x @ W with W [in, freqs]
    w.coord_features = rd.get_f32(P + "coordinate_features.weight").reshape(1, -1)  # [1,128]
    w.size_features = rd.get_f32(P + "size_features.weight").T.contiguous()        # [2,256]
    w.coord_encoder_w = rd.get_f32(P + "coordinate_encoder.weight")  # [2048,256]
    w.coord_encoder_b = rd.get_f32(P + "coordinate_encoder.bias")
    w.coord_decoder_w = rd.get_f32(P + "coordinate_head.weight")    # [1024,2048]
    w.coord_decoder_b = rd.get_f32(P + "coordinate_head.bias")
    w.size_encoder_w = rd.get_f32(P + "size_encoder.weight")
    w.size_encoder_b = rd.get_f32(P + "size_encoder.bias")
    w.size_decoder_w = rd.get_f32(P + "size_head.weight")           # [2048,2048]
    w.size_decoder_b = rd.get_f32(P + "size_head.bias")
    w.ln_w = rd.get_f32(P + "ln.weight")
    w.ln_b = rd.get_f32(P + "ln.bias")
    return w


# ---- ref_region.py math, F32 -------------------------------------------------
def fourier_features(x, wf):
    f = 2 * np.pi * x @ wf
    return torch.cat([f.cos(), f.sin()], dim=-1)


def ref_encode_coordinate(coords, w):
    return F.linear(fourier_features(coords, w.coord_features), w.coord_encoder_w, w.coord_encoder_b)


def ref_decode_coordinate(hidden, w):
    return F.linear(F.layer_norm(hidden, (DIM,), w.ln_w, w.ln_b), w.coord_decoder_w, w.coord_decoder_b)


def ref_encode_size(sizes, w):
    return F.linear(fourier_features(sizes, w.size_features), w.size_encoder_w, w.size_encoder_b)


def ref_decode_size(hidden, w):
    return F.linear(F.layer_norm(hidden, (DIM,), w.ln_w, w.ln_b), w.size_decoder_w, w.size_decoder_b).view(2, -1)


def maxabs(a, b):
    return float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).max())


def main():
    os.makedirs(PARITY_DIR, exist_ok=True)

    print("converting region gguf ...")
    r = subprocess.run([sys.executable, ROOT + "/convert/convert_region.py", "-o", GGUF],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        sys.exit("convert_region.py failed")

    torch.manual_seed(0)
    w = load_ref()

    coords = torch.rand(7, dtype=torch.float32)                     # 7 points
    sizes = torch.rand(5, 2, dtype=torch.float32)                   # 5 (w,h)
    hidden = torch.randn(DIM, dtype=torch.float32) * 3.0

    ref_cemb = ref_encode_coordinate(coords.view(-1, 1), w).numpy()
    ref_semb = ref_encode_size(sizes, w).numpy()
    ref_clog = ref_decode_coordinate(hidden, w).numpy()
    ref_slog = ref_decode_size(hidden, w).numpy().reshape(-1)

    inp = PARITY_DIR + "/in.bin"
    with open(inp, "wb") as f:
        f.write(struct.pack("<3i", 7, 5, 1))
        f.write(coords.numpy().astype(np.float32).tobytes())
        f.write(sizes.numpy().astype(np.float32).tobytes())
        f.write(hidden.numpy().astype(np.float32).tobytes())

    out = PARITY_DIR + "/out.bin"
    r = subprocess.run([EXE, "-m", GGUF, "--in", inp, "--out", out],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        sys.exit("region test exe failed")

    blob = np.fromfile(out, dtype=np.float32)
    o = 0
    cpp_cemb = blob[o:o + 7 * DIM].reshape(7, DIM); o += 7 * DIM
    cpp_semb = blob[o:o + 5 * DIM].reshape(5, DIM); o += 5 * DIM
    cpp_clog = blob[o:o + COORD_OUT]; o += COORD_OUT
    cpp_slog = blob[o:o + SIZE_OUT]

    checks = [
        ("encode_coordinate [7,2048]", maxabs(cpp_cemb, ref_cemb), 1e-3),
        ("encode_size      [5,2048]", maxabs(cpp_semb, ref_semb), 1e-3),
        ("decode_coordinate [1024]", maxabs(cpp_clog, ref_clog), 1e-3),
        ("decode_size       [2,1024]", maxabs(cpp_slog, ref_slog), 1e-3),
    ]
    ok = True
    for name, d, tol in checks:
        tag = "OK " if d < tol else "FAIL"
        print(f"  {name}: maxabs {d:.3e}  {tag}")
        ok = ok and d < tol
    print("PARITY PASS" if ok else "PARITY FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
