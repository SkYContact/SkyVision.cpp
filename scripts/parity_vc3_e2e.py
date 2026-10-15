# parity_vc3_e2e.py - end-to-end parity: official VideoChat3-4B (bf16) vs engine.
# Builds the prompt manually (chat template + N video_pad tokens) and feeds
# pixel_values/video_grid_thw directly, mirroring the engine's preprocessing.
import argparse, glob, importlib, math, sys, types
import numpy as np
import torch
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--frames", required=True)
parser.add_argument("--ckpt", default="models/VideoChat3-4B")
parser.add_argument("--ask", required=True)
parser.add_argument("--max-tokens", type=int, default=48)
parser.add_argument("--image", action="store_true", help="single-image mode (image_pad)")
args = parser.parse_args()

pkg = types.ModuleType("vc3pkg")
pkg.__path__ = [args.ckpt]
sys.modules["vc3pkg"] = pkg

from transformers import AutoConfig, AutoTokenizer
cfg = AutoConfig.from_pretrained(args.ckpt, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(args.ckpt)
cfg.vision_config.attn_impl = "eager"   # no flash-attn installed; math-exact path

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

files = sorted(glob.glob(args.frames + "/*.png") + glob.glob(args.frames + "/*.jpg"))
assert files, "no frames"
frames = [Image.open(f).convert("RGB") for f in files]
w0, h0 = frames[0].size
H, W = smart_resize(h0, w0, 28, 3136, 16384 * 28 * 28)
patch = cfg.vision_config.patch_size
merge = cfg.vision_config.merge_kernel_size[0]
gh, gw = H // patch, W // patch
T = len(frames)
n_vis = (gh // merge) * (gw // merge) * (T if args.image else math.ceil(T / 4))
print(f"{T} frames {w0}x{h0} -> {W}x{H}, gh x gw {gh}x{gw}, vis tokens {n_vis}")

pix = []
for fr in frames:
    fr = fr.resize((W, H), Image.BICUBIC)
    a = torch.from_numpy(np.asarray(fr, np.float32) / 255.0)
    a = ((a - 0.5) / 0.5).permute(2, 0, 1)
    pix.append(a)
pixel_values = torch.stack(pix, 0).cuda()  # [T, 3, H, W]

pad = "<|image_pad|>" if args.image else "<|video_pad|>"
prompt = ("<|im_start|>user\n<|vision_start|>" + pad * n_vis + "<|vision_end|>"
          + args.ask + "<|im_end|>\n<|im_start|>assistant\n")
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
print("prompt tokens:", ids.shape[1])

mv = importlib.import_module("vc3pkg.modeling_videochat3")
model = mv.VideoChat3ForConditionalGeneration._from_config(cfg, dtype=torch.bfloat16).cuda().eval()
import safetensors.torch
sd = {}
for fn in sorted(glob.glob(args.ckpt + "/*.safetensors")):
    sd.update(safetensors.torch.load_file(fn))
missing, unexpected = model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("lm_head.")}, strict=False)
print("missing:", missing[:5], "unexpected:", unexpected[:5])

with torch.no_grad():
    out = model.generate(
        input_ids=ids,
        pixel_values_videos=pixel_values,
        video_grid_thw=torch.tensor([[T, gh, gw]]).cuda(),
        max_new_tokens=args.max_tokens,
        do_sample=False,
        temperature=None, top_p=None, top_k=None,
    )
text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
print("OFFICIAL:", text)
