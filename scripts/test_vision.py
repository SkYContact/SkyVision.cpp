"""Vision encoder parity: official torch (moondream/torch) vs C++ ggml engine.

1. generates a 768x768 test image (solid background + shapes)
2. prepares crops exactly like the official prepare_crops (PIL path)
3. runs the reference SigLIP encoder + projection in torch (bf16, CPU),
   capturing every block's hidden state
4. runs build/Release/moondream-vision-test.exe with --dump-dir
5. compares per-layer hiddens, post_ln and the final [729,2048] embedding

target: maxabs < 0.05 (F16 weights, bf16 reference)
"""
import importlib.util
import math
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "convert"))
import hf_source

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
GGUF = os.path.join(ROOT, "moondream3.1-vision.gguf")
EXE = os.path.join(ROOT, "build", "Release", "moondream-vision-test.exe")
PARITY_DIR = os.path.join(ROOT, "parity_vision")
ST = hf_source.find_model_safetensors()

# official image_crops.py (from vikhyat/moondream; no relative imports)
IMAGE_CROPS_ENV = "MD31_IMAGE_CROPS"
_crops_py = None
for _p in (os.environ.get(IMAGE_CROPS_ENV),
           os.path.join(ROOT, "md31_image_crops.py"),
           "md31_image_crops.py",
           os.path.join(tempfile.gettempdir(), "md31_image_crops.py"),
           "/tmp/md31_image_crops.py"):
    if _p and os.path.exists(_p):
        _crops_py = _p
        break
if not _crops_py:
    sys.exit(f"md31_image_crops.py not found - fetch it from vikhyat/moondream and either "
             f"drop it in the repo root or set {IMAGE_CROPS_ENV}=<path>")
spec = importlib.util.spec_from_file_location("md31_image_crops", _crops_py)
image_crops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_crops)

# ---------------------------------------------------------------- config
ENC_DIM, PATCH, N_LAYERS, FF, N_HEADS = 1152, 14, 27, 4304, 16
PROJ_OUT, CROP, OVERLAP = 2048, 378, 4
GRID = CROP // PATCH  # 27


def make_test_image():
    img = Image.new("RGB", (768, 768), (30, 90, 160))
    d = ImageDraw.Draw(img)
    d.ellipse([80, 100, 320, 340], fill=(240, 200, 60), outline=(255, 255, 255), width=6)
    d.rectangle([420, 120, 700, 380], fill=(180, 60, 60), outline=(0, 0, 0), width=8)
    d.polygon([(384, 480), (220, 700), (548, 700)], fill=(60, 180, 90))
    d.line([(60, 60), (720, 720)], fill=(255, 255, 255), width=5)
    return img


# ---------------------------------------------------------------- reference model
class W:  # flat attribute bag mirroring nn.ModuleDict field names
    pass


# torch-free BF16 safetensors reader (torch lazy-loader segfaults under commit pressure)
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
        return torch.from_numpy(f32.reshape(info["shape"]).copy())  # torch [out,in]


def load_ref():
    rd = SfReader(ST)
    m = W()
    P = "vision_encoder.encoder.model.visual."
    m.patch_emb_w = rd.get_f32(P + "patch_embed.linear.weight")
    m.patch_emb_b = rd.get_f32(P + "patch_embed.linear.bias")
    m.pos_emb = rd.get_f32(P + "pos_embed")  # [1,729,1152]
    m.blocks = []
    for i in range(N_LAYERS):
        b = W()
        b.ln1_w = rd.get_f32(f"{P}blocks.{i}.norm1.weight"); b.ln1_b = rd.get_f32(f"{P}blocks.{i}.norm1.bias")
        b.qkv_w = rd.get_f32(f"{P}blocks.{i}.attn.qkv.weight"); b.qkv_b = rd.get_f32(f"{P}blocks.{i}.attn.qkv.bias")
        b.out_w = rd.get_f32(f"{P}blocks.{i}.attn.proj.weight"); b.out_b = rd.get_f32(f"{P}blocks.{i}.attn.proj.bias")
        b.ln2_w = rd.get_f32(f"{P}blocks.{i}.norm2.weight"); b.ln2_b = rd.get_f32(f"{P}blocks.{i}.norm2.bias")
        b.fc1_w = rd.get_f32(f"{P}blocks.{i}.mlp.fc1.weight"); b.fc1_b = rd.get_f32(f"{P}blocks.{i}.mlp.fc1.bias")
        b.fc2_w = rd.get_f32(f"{P}blocks.{i}.mlp.fc2.weight"); b.fc2_b = rd.get_f32(f"{P}blocks.{i}.mlp.fc2.bias")
        m.blocks.append(b)
    m.post_ln_w = rd.get_f32(P + "norm.weight"); m.post_ln_b = rd.get_f32(P + "norm.bias")
    m.pfc1_w = rd.get_f32("vision_encoder.projection.mlp.fc1.weight"); m.pfc1_b = rd.get_f32("vision_encoder.projection.mlp.fc1.bias")
    m.pfc2_w = rd.get_f32("vision_encoder.projection.mlp.fc2.weight"); m.pfc2_b = rd.get_f32("vision_encoder.projection.mlp.fc2.bias")
    return m


def ln(x, w, b):
    return F.layer_norm(x, (x.shape[-1],), w, b)


def attn(x, blk):
    B, T, C = x.shape
    H = N_HEADS
    qkv = F.linear(x, blk.qkv_w, blk.qkv_b)
    q, k, v = [t.view(B, T, H, C // H).transpose(1, 2) for t in qkv.chunk(3, dim=-1)]
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(B, T, C)
    return F.linear(out, blk.out_w, blk.out_b)


def gelu_approx(x):
    return F.gelu(x, approximate="tanh")


def ref_forward(crops, m):
    """full encoder, returns (hiddens per block [N,729,1152] f32 list, post_ln f32, final emb f32)"""
    B, C, H, Wd = crops.shape
    P = PATCH
    x = crops.reshape(B, C, H // P, P, Wd // P, P).permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, (H // P) * (Wd // P), C * P * P)
    x = F.linear(x, m.patch_emb_w, m.patch_emb_b) + m.pos_emb
    hiddens = []
    with torch.no_grad():
        for blk in m.blocks:
            x = x + attn(ln(x, blk.ln1_w, blk.ln1_b), blk)
            h = gelu_approx(F.linear(ln(x, blk.ln2_w, blk.ln2_b), blk.fc1_w, blk.fc1_b))
            x = x + F.linear(h, blk.fc2_w, blk.fc2_b)
            hiddens.append(x.float().numpy())
        post = ln(x, m.post_ln_w, m.post_ln_b).float()
    return hiddens, post.numpy()


def ref_projection(m, post_f32, tiling):
    th, tw = tiling
    global_features = torch.from_numpy(post_f32[0])  # [729,1152] f32
    local = torch.from_numpy(post_f32[1:]).view(-1, GRID, GRID, ENC_DIM)
    recon = image_crops.reconstruct_from_crops(
        local, tiling, patch_size=1, overlap_margin=OVERLAP)
    r = recon.permute(2, 0, 1)[None]  # [1,1152,H,W] bf16
    r = F.adaptive_avg_pool2d(r, (GRID, GRID))     # [1,C,27,27]
    r = r[0].permute(1, 2, 0).reshape(GRID * GRID, ENC_DIM).float()
    cat = torch.cat([global_features, r], dim=-1)[None]
    h = gelu_approx(F.linear(cat, m.pfc1_w, m.pfc1_b))
    out = F.linear(h, m.pfc2_w, m.pfc2_b)
    return out[0].float().numpy()


# ---------------------------------------------------------------- main
def maxabs(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).max())


def main():
    os.makedirs(PARITY_DIR, exist_ok=True)
    torch.manual_seed(0)

    img = make_test_image()
    np_img = np.array(img.convert("RGB"))
    oc = image_crops.overlap_crop_image(np_img, max_crops=12, overlap_margin=OVERLAP)
    tiling = oc["tiling"]
    all_crops = np.transpose(oc["crops"], (0, 3, 1, 2))
    crops = torch.from_numpy(all_crops).float().div_(255).sub_(0.5).div_(0.5)
    n = crops.shape[0]
    print(f"test image 768x768 -> tiling {tiling}, {n} crops")

    # save crops.bin for the C++ exe
    with open(PARITY_DIR + "/crops.bin", "wb") as f:
        f.write(struct.pack("<i", n))
        f.write(crops.float().numpy().astype(np.float32).tobytes())

    m = load_ref()
    print("running torch reference (bf16 CPU)...")
    hiddens, post = ref_forward(crops, m)
    final = ref_projection(m, post, tiling)
    np.save(PARITY_DIR + "/py_post_ln.npy", post)
    np.save(PARITY_DIR + "/py_final.npy", final)

    print("running C++ engine...")
    env = dict(os.environ, MD_CPU="1")
    r = subprocess.run([EXE, "-m", GGUF, "--crops", PARITY_DIR + "/crops.bin",
                        "--out", PARITY_DIR + "/cpp_final.bin",
                        "--tiling", str(tiling[0]), str(tiling[1]),
                        "--dump-dir", PARITY_DIR],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        sys.exit("C++ vision test failed")

    # per-layer parity via --max-layer runs (in-graph intermediate dumps are
    # unreliable: gallocr reuses buffers after a tensor's last consumer)
    print("\n=== per-layer parity, crop0 (maxabs) ===")
    worst = 0.0
    crop0_bin = PARITY_DIR + "/crop0.bin"
    with open(crop0_bin, "wb") as f:
        f.write(struct.pack("<i", 1))
        f.write(crops[0].float().numpy().astype(np.float32).tobytes())
    for i in range(N_LAYERS - 1):   # layer 26 raw hidden is unreachable: max_layer==27 applies post_ln
        out_bin = PARITY_DIR + f"/cpp_h{i}.bin"
        r2 = subprocess.run([EXE, "-m", GGUF, "--crops", crop0_bin, "--out", out_bin,
                             "--max-layer", str(i + 1)], capture_output=True, text=True,
                            env=env, cwd=ROOT)
        if r2.returncode != 0:
            sys.exit(f"C++ run for layer {i} failed")
        cpp = np.fromfile(out_bin, dtype=np.float32).reshape(729, 1152)
        ref = hiddens[i][0]
        d = maxabs(cpp, ref)
        rel = d / max(np.abs(ref).max(), 1e-9)   # unnormalized SigLIP hiddens reach O(1000)
        worst = max(worst, rel)
        tag = "OK " if rel < 5e-4 else "FAIL"
        print(f"  layer {i:2d}: maxabs {d:.4f}  rel {rel:.2e}  {tag}")
    d_post = max(maxabs(np.fromfile(f"{PARITY_DIR}/cpp_post_ln_c{c}.bin", dtype=np.float32).reshape(729, 1152),
                        post[c]) for c in range(n))
    print(f"  post_ln (all crops): {d_post:.4f}  {'OK' if d_post < 0.05 else 'FAIL'}")

    cpp_final = np.fromfile(PARITY_DIR + "/cpp_final.bin", dtype=np.float32).reshape(729, 2048)
    d_final = maxabs(cpp_final, final)
    print(f"  final embedding [729,2048]: {d_final:.4f}  {'OK' if d_final < 0.05 else 'FAIL'}")

    if max(worst, d_post, d_final) < 0.05:
        print("\nPARITY PASS (all < 0.05)")
    else:
        print("\nPARITY FAIL")


if __name__ == "__main__":
    main()
