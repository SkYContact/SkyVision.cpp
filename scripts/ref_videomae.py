"""Reference activations for a VideoMAE checkpoint (PyTorch / transformers).

Runs the official implementation on the same clip the engine sees and dumps the
tensors the C++ backend names, so the two can be compared stage by stage:

  python scripts/ref_videomae.py --model models/videomae-base-ft \\
      --video clip.mp4 --out ref_videomae
  # -> ref_videomae/ref_v_patch_out.npy, ref_v_final.npy, ref_v_pooled.npy,
  #    ref_v_logits.npy, ref_v_probs.npy  (+ printed top-5 labels)

The engine writes the matching `cpp_<name>.bin` / `cpp_<name>.shape` pair when
run with `--dump-dir ref_videomae`; both are [ne0, ne1, ...] C-order float32, so

  a = np.fromfile('ref_videomae/cpp_v_final.bin', dtype=np.float32)
  b = np.load('ref_videomae/ref_v_final.npy').T.reshape(-1)   # ggml ne order
  np.abs(a - b).max()

is the per-stage drift. Frame sampling matches the engine (uniform linspace
over the clip, the last frame repeating when the video is shorter).
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import (AutoConfig, AutoImageProcessor, VideoMAEForPreTraining,
                          VideoMAEForVideoClassification)


def read_video(path, num_frames):
    """Uniformly sample `num_frames` RGB frames from a file or a frame dir."""
    import av   # pip install av (the model card's own example uses it)

    container = av.open(path)
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    return [Image.fromarray(frames[i]) for i in idx]


def read_frames_dir(path, num_frames):
    names = sorted(f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    imgs = [Image.open(os.path.join(path, n)).convert("RGB") for n in names]
    idx = np.linspace(0, len(imgs) - 1, num_frames).astype(int)
    return [imgs[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF checkpoint dir (or hub id)")
    ap.add_argument("--video", default=None, help="video file (needs the av package)")
    ap.add_argument("--frames", default=None, help="directory of pre-extracted frames")
    ap.add_argument("--out", required=True, help="output directory for the .npy dumps")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    if not args.video and not args.frames:
        ap.error("pass --video or --frames")
    os.makedirs(args.out, exist_ok=True)

    proc = AutoImageProcessor.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    num_frames = int(getattr(cfg, "num_frames", 16))
    clip = (read_video(args.video, num_frames) if args.video
            else read_frames_dir(args.frames, num_frames))
    pixel_values = proc(clip, return_tensors="pt")["pixel_values"]   # [1, T, C, H, W]

    try:
        model = VideoMAEForVideoClassification.from_pretrained(args.model).eval()
    except Exception:
        model = VideoMAEForPreTraining.from_pretrained(args.model).eval()

    seen = {}

    def hook(name):
        def fn(_m, _i, out):
            seen[name] = out.detach().float()
        return fn

    base = model.videomae
    base.embeddings.register_forward_hook(hook("v_patch_out"))
    base.encoder.layer[0].register_forward_hook(hook("v_blk0_out"))
    base.register_forward_hook(hook("v_final"))

    with torch.no_grad():
        out = model(pixel_values=pixel_values)

    patch_out = seen["v_patch_out"][0].numpy()              # [tokens, dim]
    hidden = seen["v_final"][0].numpy()                     # already post-LayerNorm
    if getattr(model, "fc_norm", None) is not None:         # use_mean_pooling fine-tunes
        pooled = model.fc_norm(torch.from_numpy(hidden.mean(0))).numpy()
    else:
        pooled = hidden[0]                                  # no cls token: token 0

    def save(name, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        np.save(os.path.join(args.out, f"ref_{name}.npy"), a)
        print(f"{name:14s} shape {a.shape} mean {a.mean():+.6f} "
              f"absmax {np.abs(a).max():.6f}")

    save("v_patch_out", patch_out.T)     # ggml order: [dim, tokens]
    save("v_blk0_out", seen["v_blk0_out"][0].numpy().T)
    save("v_final", hidden.T)
    save("v_pooled", pooled)

    logits = out.logits[0].float().numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    save("v_logits", logits)
    save("v_probs", probs)

    for i in np.argsort(-probs)[: args.topk]:
        label = model.config.id2label.get(int(i), str(i)) if hasattr(model.config, "id2label") else str(i)
        print(f"  {probs[i]:.6f}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
