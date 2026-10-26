"""Reference dumps for a Nemotron-3.5-Parse checkpoint.

  python scripts/ref_nemotron_parse.py --model models/Nemotron-3.5-Parse \
      --image page.png --out ref_np

Writes ``ref_<name>.bin`` + ``ref_<name>.shape`` in the same layout the engine's
``--dump-dir`` produces (raw float32, ggml ``ne`` order: last dim slowest), so
``python convert/compare_dumps.py ref_np`` works unchanged:

  v_patches      [768, n_patch]        processor patches (CLIP-normalized, c,dy,dx)
  v_positions    [1152, n_patch]       content-normalized 2D position field
  v_patch_out    [1152, n_patch]       patch embedder + positions (no class tokens)
  v_final        [1152, 10 + n_patch]  ViT stack output (class tokens first)
  v_neck         [1024, n2]            after conv1/LN/width-merge/LN
  v_feats        [1024, n2 + 1]        + the summary token (last)
  t_emb          [1024, T]             prefix + text, after layernorm_embedding
  t_l0_out       [1024, T]             decoder layer 0 output
  t_final_norm   [1024, T]             final LayerNorm
  logits         [vocab]               last-position logits

and prints the prompt ids + the greedy answer for comparison.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

DEFAULT_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"


def dump(out_dir, name, arr, dims):
    a = np.ascontiguousarray(arr, dtype=np.float32)
    assert a.size == int(np.prod(dims)), (name, a.shape, dims)
    a.tofile(os.path.join(out_dir, f"ref_{name}.bin"))
    with open(os.path.join(out_dir, f"ref_{name}.shape"), "w", encoding="utf-8") as f:
        f.write(" ".join(str(int(d)) for d in dims) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--max-edge", type=int, default=0,
                    help="cap the canvas height (mirrors NEMOTRON_PARSE_MAX_EDGE)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True,
                                      dtype=torch.float32).eval()
    img = Image.open(args.image).convert("RGB")

    if args.max_edge:
        # same box scaling the backend does before the aspect resize
        ip = proc.image_processor
        s = float(args.max_edge) / float(ip.final_size[0])
        ip.final_size = (max(16, int(round(ip.final_size[0] * s))),
                         max(64, int(round(ip.final_size[1] * s))))
        ip._create_transforms()

    ins = proc(images=[img], text=args.prompt, return_tensors="pt",
               add_special_tokens=False)
    pv = ins["pixel_values"]
    image_sizes = ins["image_sizes"]
    content_sizes = ins["content_sizes"]
    n_patch = int(pv.shape[0])
    rows, cols = [int(x) for x in image_sizes[0]]
    ch, cw = [int(x) for x in content_sizes[0]]
    print(f"patches {n_patch} page {rows}x{cols} content {ch}x{cw}")
    print("prompt ids:", ins["input_ids"][0].tolist(),
          "->", proc.tokenizer.decode(ins["input_ids"][0], skip_special_tokens=False))

    seen = {}

    def grab(name, kind="out"):
        def fn(m, i, o):
            if kind == "in":
                t = i[0]
            elif isinstance(o, torch.Tensor):
                t = o
            else:
                t = getattr(o, "last_hidden_state", None)
                if t is None:
                    t = o[0] if isinstance(o, (tuple, list)) else o
            seen[name] = t.detach().float()
        return fn

    enc = model.encoder
    vit = enc.model_encoder.radio_model.model
    enc.register_forward_hook(grab("v_feats"))
    vit.register_forward_hook(grab("v_final"))
    vit.patch_generator.embedder.register_forward_hook(grab("v_patch_out"))
    enc.layer_norm2.register_forward_hook(grab("v_neck"))
    model.decoder.layernorm_embedding.register_forward_hook(grab("t_emb"))
    model.decoder.layers[0].register_forward_hook(grab("t_l0_out"))
    model.decoder.layer_norm.register_forward_hook(grab("t_final_norm"))

    with torch.no_grad():
        out = model(**ins)

    dump(args.out, "v_patches", pv.numpy(), (768, n_patch, 1, 1))
    patch_emb = seen["v_patch_out"]                        # [n_patch, 1152] (no batch dim)
    if patch_emb.dim() == 3:
        patch_emb = patch_emb[0]

    # position field: the official helper adds it to zero patches
    sys.path.insert(0, os.path.abspath(args.model))
    from hf_prefix_geometry import _normalized_2d_position_pages
    pos = _normalized_2d_position_pages(
        torch, patches=torch.zeros(1, n_patch, 1152), pos_embed=vit.patch_generator.pos_embed,
        source_rows=128, source_cols=128, image_sizes=image_sizes,
        patch_size=16, content_sizes=content_sizes, content_normalized=True).detach()[0]
    dump(args.out, "v_positions", pos.numpy(), (1152, n_patch, 1, 1))
    # the engine's v_patch_out is embedder + positions (the hook above only sees
    # the Linear, which runs before the position add)
    dump(args.out, "v_patch_out", (patch_emb + pos).numpy(), (1152, n_patch, 1, 1))

    fin = seen["v_final"][0]                               # [10 + n_patch, 1152]
    dump(args.out, "v_final", fin.numpy(), (1152, fin.shape[0], 1, 1))
    n2 = seen["v_neck"][0].shape[0]
    dump(args.out, "v_neck", seen["v_neck"][0].numpy(), (1024, n2, 1, 1))
    feats = seen["v_feats"][0]                             # [n2 + 1, 1024]
    dump(args.out, "v_feats", feats.numpy(), (1024, feats.shape[0], 1, 1))

    T = seen["t_emb"][0].shape[0]
    for name in ("t_emb", "t_l0_out", "t_final_norm"):
        dump(args.out, name, seen[name][0].numpy(), (1024, T, 1, 1))

    lg = out.logits[0, -1].float().numpy()
    dump(args.out, "logits", lg, (lg.size, 1, 1, 1))
    top = np.argsort(-lg)[:8]
    print("  first-token top8:", [(int(i), proc.tokenizer.decode([int(i)]),
                                   round(float(lg[i]), 2)) for i in top])

    with torch.no_grad():
        gen = model.generate(**ins, max_new_tokens=args.max_new, do_sample=False,
                             repetition_penalty=1.1, num_beams=1)
    ids = gen[0][ins["input_ids"].shape[1]:].tolist()
    print("  greedy ids:", ids)
    print("  ANSWER:", repr(proc.tokenizer.decode(ids, skip_special_tokens=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
