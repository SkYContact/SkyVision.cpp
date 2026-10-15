# parity_vc3_video.py - compare the engine's multi-frame VideoChat3 vision path
# against the official PyTorch implementation.
#
#   1. vqa-cli --frames <dir> --dump-dir <dir> dumps v_final (vision tower out,
#      pre-merge), v_merge (post temporal-pool + spatial fold) and v_proj_out
#      (post projector) as row-major f32
#   2. this script loads the same frames, runs the official VideoChat3VisionModel
#      + patch_merger + projector in fp32, and reports max-abs error
import argparse
import glob
import importlib
import math
import sys
import types

import numpy as np
import torch
import safetensors.torch
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--frames", required=True)
parser.add_argument("--ckpt", default="models/VideoChat3-4B")
parser.add_argument("--dump", required=True, help="engine --dump-dir")
args = parser.parse_args()

# import the checkpoint's modeling code as a package (relative imports)
pkg = types.ModuleType("vc3pkg")
pkg.__path__ = [args.ckpt]
sys.modules["vc3pkg"] = pkg
mv = importlib.import_module("vc3pkg.modeling_videochat3")

files = sorted(glob.glob(args.frames + "/*.png") + glob.glob(args.frames + "/*.jpg"))
assert files, "no frames"
frames = [Image.open(f).convert("RGB") for f in files]
T = len(frames)

from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(args.ckpt, trust_remote_code=True)

# smart_resize exactly like the engine (qwen_vl_utils demo defaults)
def smart_resize(h, w, factor, min_pixels, max_pixels):
    def ceil_by(n, f): return int(math.ceil(n / f)) * f
    def floor_by(n, f): return int(math.floor(n / f)) * f
    h_bar = max(factor, ceil_by(h, factor))
    w_bar = max(factor, ceil_by(w, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h_bar = max(factor, floor_by(h / beta, factor))
        w_bar = max(factor, floor_by(w / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = max(factor, ceil_by(h * beta, factor))
        w_bar = max(factor, ceil_by(w * beta, factor))
    return h_bar, w_bar

w0, h0 = frames[0].size
H, W = smart_resize(h0, w0, 28, 3136, 16384 * 28 * 28)
print(f"{T} frames {w0}x{h0} -> {W}x{H}")

patch = cfg.vision_config.patch_size
gh, gw = H // patch, W // patch

# per-frame preprocessing identical to the official image processor:
# resize (bicubic, antialias) + (x/255 - 0.5)/0.5
pix = []
for fr in frames:
    fr = fr.resize((W, H), Image.BICUBIC)
    a = torch.from_numpy(np.asarray(fr, dtype=np.float32) / 255.0)  # HWC
    a = (a - 0.5) / 0.5
    pix.append(a.permute(2, 0, 1))
pixel_values = torch.stack(pix, 0).cuda()  # [T, 3, H, W]

# --- official weights
sd = {}
for fn in sorted(glob.glob(args.ckpt + "/*.safetensors")):
    sd.update(safetensors.torch.load_file(fn))
vis_sd = {k[len("model.vision_tower."):]: v.float() for k, v in sd.items() if k.startswith("model.vision_tower.")}
mm_sd = {k[len("model.multi_modal_projector."):]: v.float() for k, v in sd.items() if k.startswith("model.multi_modal_projector.")}
assert vis_sd and mm_sd, "weight keys not found"

vc = cfg.vision_config
vc.attn_impl = "eager"   # fp32 parity; flash-attn path is fp16/bf16-only
vis = mv.VideoChat3VisionModel._from_config(vc, torch_dtype=torch.float32).cuda().eval()
missing, unexpected = vis.load_state_dict(vis_sd, strict=False)
assert not unexpected, unexpected
assert not missing, missing

with torch.no_grad():
    grid = torch.tensor([[T, gh, gw]]).cuda()
    clips = vis.split_grid_thws_clip_by_clip(grid)

    hidden = vis.patch_embed(pixel_values, clips)
    hidden = vis.encoder(hidden, clips)
    hidden = vis.encoder.final_layernorm(hidden)             # pre-merge tower output
    merged = mv.patch_merger(hidden, clips, merge_kernel_size=vc.merge_kernel_size)

    import torch.nn as nn
    proj = nn.Linear(0, 0)  # placeholder; build exact projector below
    hidden_size = vc.hidden_size * vc.merge_kernel_size[0] * vc.merge_kernel_size[1]
    class Proj(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_norm = nn.LayerNorm(vc.hidden_size, eps=1e-5)
            self.linear_1 = nn.Linear(hidden_size, hidden_size, bias=True)
            self.act = nn.GELU()
            self.linear_2 = nn.Linear(hidden_size, cfg.text_config.hidden_size, bias=True)
        def forward(self, x):
            h = self.pre_norm(x).view(-1, hidden_size)
            return self.linear_2(self.act(self.linear_1(h)))
    projector = Proj().cuda().eval()
    projector.load_state_dict(mm_sd)
    py_proj = projector(torch.cat(merged, 0))        # [n_tokens, D_text]

py_final = hidden.cpu().float()                      # [T*gh*gw, D]
merged_all = torch.cat(merged, 0).cpu().float()      # [n_tokens, kh*kw, D]

# --- engine dumps (ggml [D, rows] -> [rows, D])
def load(name):
    a = np.fromfile(f"{args.dump}/cpp_{name}.bin", dtype=np.float32)
    s = open(f"{args.dump}/cpp_{name}.shape").read().split()
    ne = [int(x) for x in s]
    d = ne[0]
    rows = a.size // d
    return torch.from_numpy(a.reshape(rows, d))

def report(name, py, cpp):
    if py.numel() != cpp.numel():
        print(f"{name}: SIZE MISMATCH py {tuple(py.shape)} ({py.numel()}) vs cpp {tuple(cpp.shape)} ({cpp.numel()})")
        return
    a, b = py.flatten(), cpp.flatten()
    diff = (a - b).abs()
    scale = a.abs().max().clamp(min=1e-6)
    print(f"{name}: max-abs {diff.max().item():.5e}  rel-to-max {(diff.max()/scale).item():.3e}")

cpp_final = load("v_final")
report("v_final", py_final, cpp_final)

# engine v_merge: [tokens, kw*kh*D]; official: [tokens, kh*kw, D]
cpp_merge = load("v_merge")
if cpp_merge.numel() == merged_all.numel():
    tok = merged_all.shape[0]
    cpp_t = cpp_merge.reshape(tok, -1)
    py_t = merged_all.reshape(tok, -1)
    D = merged_all.shape[-1]
    cpp_alt = cpp_t.reshape(tok, 2, 2, D).permute(0, 2, 1, 3).reshape(tok, -1)
    print(f"v_merge: max-abs as-is {(cpp_t - py_t).abs().max().item():.5e}"
          f"  fold-swapped {(cpp_alt.float() - py_t).abs().max().item():.5e}")
else:
    print(f"v_merge: SIZE MISMATCH py {merged_all.numel()} vs cpp {cpp_merge.numel()}")

report("v_proj_out", py_proj.cpu(), load("v_proj_out"))
