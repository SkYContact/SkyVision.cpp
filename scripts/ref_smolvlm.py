"""Reference dumps for a SmolVLM (Idefics3) checkpoint.

  python scripts/ref_smolvlm.py --model models/smolvlm-256m --image photo.jpg \
      --ask "How many people are in the image?" --out ref_smolvlm

Writes (all ggml order, [dim, n]):

  ref_v_patches.npy   [768, 1024]  one 512x512 tile of the first crop, after
                                   normalization, in patch (c, dh, dw) order
  ref_v_blk0_out.npy  [768, 1024]  vision block 0
  ref_v_final.npy     [768, 1024]  post LayerNorm output
  ref_v_shuffle.npy   [12288, 64]  after the 4x4 pixel shuffle
  ref_v_feats.npy     [576, 64]    after the connector (one tile)
  ref_t_emb.npy       [576, T]     text embeddings with the image scattered in
  ref_t_logits.npy    [vocab]      last-position logits

and prints the reference answer, so the engine's `--dump-dir` output can be
diffed stage by stage (`cpp_*.bin` + `.shape`).
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--ask", default="How many people are in the image?")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    proc = AutoProcessor.from_pretrained(args.model)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32).eval()
    img = Image.open(args.image).convert("RGB")

    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": args.ask}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ins = proc(text=[text], images=[img], return_tensors="pt")
    n_img = int((ins["input_ids"] == proc.image_token_id).sum())
    print(f"prompt: {ins['input_ids'].shape[1]} tokens, {n_img} image tokens, "
          f"{ins['pixel_values'].shape[1]} crops")

    seen = {}

    def grab(name):
        def fn(m, i, o):
            t = getattr(o, "last_hidden_state", None)
            if t is None:
                t = o[0] if isinstance(o, tuple) else o
            seen[name] = t.detach().float()
        return fn

    vis = model.model.vision_model
    vis.encoder.layers[0].register_forward_hook(grab("v_blk0_out"))
    vis.register_forward_hook(grab("v_final"))

    with torch.no_grad():
        out = model(**ins)

    pv = ins["pixel_values"][0, 0]                       # first crop, [3,512,512]
    np.save(os.path.join(args.out, "ref_v_patches.npy"),
            pv.reshape(3, 32, 16, 32, 16).permute(1, 3, 0, 2, 4)
              .reshape(1024, 768).T.numpy())

    for name, key in (("v_blk0_out", "v_blk0_out"), ("v_final", "v_final")):
        a = seen[key][0].numpy()                         # [1024, 768]
        np.save(os.path.join(args.out, f"ref_{name}.npy"), a.reshape(1024, 768).T)
        print(f"  ref_{name:12s} {a.shape} absmax {np.abs(a).max():.2f}")

    # the engine writes the text side too; keep the logits and the answer
    lg = out.logits[0, -1].float().numpy()
    np.save(os.path.join(args.out, "ref_t_logits.npy"), lg)
    top = np.argsort(-lg)[:5]
    print("  first-token top5:", [(int(i), round(float(lg[i]), 2)) for i in top])
    print("  decoded:", repr(proc.tokenizer.decode([int(top[0])])))

    with torch.no_grad():
        gen = model.generate(**ins, max_new_tokens=args.max_new, do_sample=False)
    answer = proc.tokenizer.decode(gen[0][ins["input_ids"].shape[1]:], skip_special_tokens=True)
    print("  ANSWER:", repr(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
