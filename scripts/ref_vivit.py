"""Reference activations for a ViViT checkpoint (PyTorch / transformers).

The engine reuses the videomae video-tower backend for ViViT (cls token + learned
position table + gelu_fast), so the tensor names are the same and the two sides can
be compared stage by stage:

  python scripts/ref_vivit.py --model models/vivit-b-16x2 --image photo.jpg \\
      --out ref_vivit
  # -> ref_vivit/ref_v_patch_out.npy, ref_v_v_final.npy, ref_v_pooled.npy,
  #    ref_v_probs.npy  (+ printed top-5 labels)

Then run the engine on the same input with `--dump-dir ref_vivit` and compare:

  a = np.fromfile('ref_vivit/cpp_v_probs.bin', dtype=np.float32)
  b = np.load('ref_vivit/ref_v_probs.npy')
  np.abs(a - b).max()

A still image is repeated `num_frames` times, exactly like the engine's frame
sampler does when the media is shorter than the clip.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, VivitForVideoClassification


def read_video(path, num_frames):
    import av   # pip install av

    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    return [Image.fromarray(frames[i]) for i in idx]


def read_frames_dir(path, num_frames):
    names = sorted(f for f in os.listdir(path)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    imgs = [Image.open(os.path.join(path, n)).convert("RGB") for n in names]
    idx = np.linspace(0, len(imgs) - 1, num_frames).astype(int)
    return [imgs[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF checkpoint dir (or hub id)")
    ap.add_argument("--video", default=None, help="video file (needs the av package)")
    ap.add_argument("--frames", default=None, help="directory of pre-extracted frames")
    ap.add_argument("--image", default=None, help="still image (repeated to fill the clip)")
    ap.add_argument("--out", required=True, help="output directory for the .npy dumps")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    if not (args.video or args.frames or args.image):
        ap.error("pass --video, --frames or --image")
    os.makedirs(args.out, exist_ok=True)

    proc = AutoImageProcessor.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    num_frames = int(getattr(cfg, "num_frames", 32))
    if args.video:
        clip = read_video(args.video, num_frames)
    elif args.frames:
        clip = read_frames_dir(args.frames, num_frames)
    else:
        img = Image.open(args.image).convert("RGB")
        clip = [img] * num_frames
    pixel_values = proc(clip, return_tensors="pt")["pixel_values"]   # [1, T, C, H, W]
    print(f"clip: {num_frames} frames, pixel_values {tuple(pixel_values.shape)} "
          f"range [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")

    model = VivitForVideoClassification.from_pretrained(args.model).eval()

    seen = {}

    def hook(name):
        def fn(_m, _i, out):
            # encoder layers / embeddings return a tensor, the model returns a
            # BaseModelOutputWithPooling - normalize to the last hidden state
            if hasattr(out, "last_hidden_state"):
                out = out.last_hidden_state
            elif isinstance(out, (tuple, list)):
                out = out[0]
            seen[name] = out.detach().float()
        return fn

    base = model.vivit
    base.embeddings.register_forward_hook(hook("v_patch_out"))
    base.encoder.layer[0].register_forward_hook(hook("v_blk0_out"))
    base.register_forward_hook(hook("v_final"))

    with torch.no_grad():
        out = model(pixel_values=pixel_values)

    patch_out = seen["v_patch_out"][0].numpy().reshape(-1, cfg.hidden_size)   # [tokens, dim]
    hidden = seen["v_final"][0].numpy()                                       # post-LayerNorm
    pooled = hidden[0]                                                        # cls token

    def save(name, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        np.save(os.path.join(args.out, f"ref_{name}.npy"), a)
        print(f"{name:14s} shape {a.shape} mean {a.mean():+.6f} absmax {np.abs(a).max():.6f}")

    save("v_patch_out", patch_out.T)                       # ggml order: [dim, tokens]
    save("v_blk0_out", seen["v_blk0_out"][0].numpy().T)
    save("v_final", hidden.T)
    save("v_pooled", pooled)

    logits = out.logits[0].float().numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    save("v_logits", logits)
    save("v_probs", probs)

    for i in np.argsort(-probs)[: args.topk]:
        label = model.config.id2label.get(int(i), str(i))
        print(f"  {probs[i]:.6f}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
