"""Reference activations for a V-JEPA 2 checkpoint (PyTorch / transformers).

Runs the official implementation on the same clip the engine sees and dumps the
tensors the C++ backend names, so the two can be compared stage by stage:

  python scripts/ref_vjepa2.py --model models/vjepa2-vitl-fpc64-256 \\
      --video clip.mp4 --out ref_vjepa2
  # -> ref_vjepa2/ref_v_patch_out.npy, ref_v_blk0_out.npy, ref_v_final.npy,
  #    ref_v_pooled.npy (+ ref_v_logits.npy / ref_v_probs.npy for a fine-tune,
  #    + ref_p_latents.npy and predict_spec.txt for the masked predictor)

The engine writes the matching `cpp_<name>.bin` / `cpp_<name>.shape` pair when
run with `--dump-dir ref_vjepa2`; both are [ne0, ne1, ...] C-order float32, so

  a = np.fromfile('ref_vjepa2/cpp_v_final.bin', dtype=np.float32)
  b = np.load('ref_vjepa2/ref_v_final.npy').T.reshape(-1)   # ggml ne order
  np.abs(a - b).max()

is the per-stage drift. Frame sampling matches the engine (uniform linspace over
the clip, the last frame repeating when the video is shorter).

`VJEPA2ForVideoClassification` checkpoints are handled too: the attentive pooler
output becomes `v_pooled`, the classifier adds `v_logits` / `v_probs`. The masked
predictor is always dumped (split at the middle of the token sequence) together
with `predict_spec.txt`, which can be fed straight to
`vqa-cli --mode predict --ask "$(cat predict_spec.txt)"`.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoVideoProcessor


def read_video(path, num_frames):
    """Uniformly sample `num_frames` RGB frames from a file."""
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
    ap.add_argument("--out", required=True, help="output directory for the .npy dumps")
    ap.add_argument("--mask-token", type=int, default=1,
                    help="predictor mask token index (transformers always uses 1)")
    args = ap.parse_args()
    if not args.video and not args.frames:
        ap.error("pass --video or --frames")
    os.makedirs(args.out, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model)
    num_frames = int(getattr(cfg, "frames_per_clip", 64))
    proc = AutoVideoProcessor.from_pretrained(args.model)
    clip = (read_video(args.video, num_frames) if args.video
            else read_frames_dir(args.frames, num_frames))
    pixel_values = proc(videos=clip, return_tensors="pt")["pixel_values_videos"]
    print(f"pixel_values {tuple(pixel_values.shape)}")

    arch = (cfg.architectures or ["VJEPA2Model"])[0]
    print(f"checkpoint class: {arch}")
    if arch == "VJEPA2ForVideoClassification":
        from transformers import VJEPA2ForVideoClassification as Cls
        model = Cls.from_pretrained(args.model).eval()
        encoder = model.vjepa2.encoder
    else:
        from transformers import VJEPA2Model
        model = VJEPA2Model.from_pretrained(args.model).eval()
        encoder = model.encoder

    seen = {}

    def hook(name):
        def fn(_m, _i, out):
            if isinstance(out, tuple):
                out = out[0]                  # encoder layers return (hidden, attentions)
            elif not torch.is_tensor(out):
                out = out.last_hidden_state   # newer transformers wraps activations
            seen[name] = out.detach().float()
        return fn

    encoder.embeddings.register_forward_hook(hook("v_patch_out"))
    encoder.layer[0].register_forward_hook(hook("v_blk0_out"))
    encoder.register_forward_hook(hook("v_final"))

    with torch.no_grad():
        out = model(pixel_values_videos=pixel_values)

    hidden = seen["v_final"][0].numpy()                 # [tokens, dim], post-LayerNorm
    n_tok = hidden.shape[0]

    def save(name, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        np.save(os.path.join(args.out, f"ref_{name}.npy"), a)
        print(f"{name:14s} shape {a.shape} mean {a.mean():+.6f} "
              f"absmax {np.abs(a).max():.6f}")

    save("v_patch_out", seen["v_patch_out"][0].numpy().T)     # ggml order: [dim, tokens]
    save("v_blk0_out", seen["v_blk0_out"][0].numpy().T)
    save("v_final", hidden.T)

    if arch == "VJEPA2ForVideoClassification":
        with torch.no_grad():
            pooled = model.pooler(torch.from_numpy(hidden).unsqueeze(0))[0]
            logits = model.classifier(pooled)
            probs = torch.softmax(logits, -1)
        save("v_pooled", pooled.numpy())
        save("v_logits", logits.numpy())
        save("v_probs", probs.numpy())
        top = int(np.argmax(probs.numpy()))
        label = cfg.id2label.get(top, cfg.id2label.get(str(top), "?"))
        print(f"top-1: {probs.numpy()[top]:.6f}  [{top}] {label}")
    else:
        pooled = hidden.mean(0)                         # mean over the token axis
        save("v_pooled", pooled)
        print(f"pooled: dim {pooled.size} l2 {np.linalg.norm(pooled):.6f}")

    # ---- masked predictor: context = first half, target = second half
    pred = model.predictor if arch != "VJEPA2ForVideoClassification" else model.vjepa2.predictor
    n_ctx = n_tok // 2
    ctx = list(range(0, n_ctx))
    tgt = list(range(n_ctx, n_tok))
    cm = torch.tensor([ctx])
    tm = torch.tensor([tgt])
    with torch.no_grad():
        pr = pred(seen["v_final"], [cm], [tm], mask_index=args.mask_token)
    save("p_latents", pr.last_hidden_state[0].numpy().T)   # [dim, n_target]
    spec = f"ctx=0:{n_ctx},tgt={n_ctx}:{n_tok},mask={args.mask_token}"
    with open(os.path.join(args.out, "predict_spec.txt"), "w", encoding="utf-8") as f:
        f.write(spec + "\n")
    print(f"predictor: {n_tok - n_ctx} target tokens, spec \"{spec}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
