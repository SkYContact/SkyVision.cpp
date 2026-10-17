"""Reference dumps for a Qwen2.5-VL checkpoint (PyTorch / transformers).

Produces the same tensors the engine writes with `--dump-dir`, so the two sides can
be diffed stage by stage:

  python scripts/ref_qwen2vl.py --model models/qwen2.5-vl-3b \
      --image photo.jpg --ask "How many people are in the image?" --out ref_q2v

  ref_q2v/ref_v_patches.npy     [n_patches, C*T*P*P]  processor output (2x2 group order)
  ref_q2v/ref_v_blk0.npy        [n_patches, 1280]     vision block 0
  ref_q2v/ref_v_blk15.npy       [n_patches, 1280]     block 15 (after the 2nd full layer)
  ref_q2v/ref_v_blk31.npy       [n_patches, 1280]     last block (= tower output)
  ref_q2v/ref_v_feats.npy       [n_merged, 2048]      merged features (raster order)
  ref_q2v/ref_t_emb.npy         [2048, T]             text input embeddings, image scattered in
  ref_q2v/ref_t_logits.npy      [vocab]               last-position logits

then run the engine on the same input with `--dump-dir ref_q2v` and compare
`cpp_*.bin` (the engine writes ggml order, [dim, n] per file plus a .shape).

Notes on the two traps this backend hit (worth knowing when diffing):
  * the vision merger RMS-normalizes **per token over 1280** and only then folds the
    2x2 group into one 5120 row - normalizing after the fold silently destroys the
    features;
  * the KV cache slot is the absolute sequence position, while the MRoPE `t` axis is
    not (every image token shares t = st), so they cannot share one index array.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF checkpoint dir (or hub id)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--ask", default="How many people are in the image?")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    proc = AutoProcessor.from_pretrained(args.model, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32).eval()
    vis = model.model.visual

    img = Image.open(args.image).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": args.ask}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ins = proc(text=[text], images=[img], return_tensors="pt")
    n_img = int((ins["input_ids"] == model.config.image_token_id).sum())
    print(f"prompt: {ins['input_ids'].shape[1]} tokens ({n_img} image), "
          f"grid {ins['image_grid_thw'].tolist()}")

    seen = {}

    def grab(name):
        def fn(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            seen[name] = t.detach().float()
        return fn

    for idx in (0, 15, 31):
        vis.blocks[idx].register_forward_hook(grab(f"blk{idx}"))

    def grab_layer0(m, i, o):
        seen["emb"] = i[0].detach().float()

    model.model.language_model.layers[0].register_forward_hook(grab_layer0)

    with torch.no_grad():
        out = model(**ins)

    np.save(os.path.join(args.out, "ref_v_patches.npy"), ins["pixel_values"].numpy())
    for k, v in seen.items():
        if k == "emb":
            continue
        a = v.reshape(-1, v.shape[-1]).numpy()
        np.save(os.path.join(args.out, f"ref_v_{k}.npy"), a)
        print(f"  ref_v_{k:10s} {a.shape} absmax {np.abs(a).max():9.2f}")
    emb = seen["emb"][0].numpy().T
    np.save(os.path.join(args.out, "ref_t_emb.npy"), emb)
    print(f"  ref_t_emb       {emb.shape} absmax {np.abs(emb).max():9.2f}")

    lg = out.logits[0, -1].float().numpy()
    np.save(os.path.join(args.out, "ref_t_logits.npy"), lg)
    top = np.argsort(-lg)[:5]
    print("  first-token top5:", [(int(i), round(float(lg[i]), 2)) for i in top])
    print("  decoded:", repr(proc.tokenizer.decode([int(top[0])])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
