"""Reference activations for VideoChat3-4B (PyTorch / transformers).

Runs the official implementation on an image and dumps intermediate tensors so
the C++ engine can be compared stage by stage.

Memory: only the first --layers text layers are instantiated (the rest are not
loaded), so a 4B model can be inspected on a modest machine.

Usage:
  python scripts/ref_videochat3.py --model <hf_dir> --out <dir> --image x.png \
         --prompt "What shapes are in this image?" [--layers 2]
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor


def load_selected(mdir, layers, want_prefixes):
    """Load bf16 weights as float32 for the requested name prefixes."""
    idx = os.path.join(mdir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = {}
    files = sorted(set(weight_map.values())) or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(mdir, "*.safetensors")))
    handles = {f: safe_open(os.path.join(mdir, f), framework="pt") for f in files}

    def keep(name):
        for p in want_prefixes:
            if name.startswith(p):
                return True
        # text layers: only the first `layers`
        if name.startswith("model.language_model.layers."):
            i = int(name.split(".")[3])
            return i < layers
        return False

    out = {}
    for name, f in weight_map.items():
        if not keep(name):
            continue
        t = handles[f].get_tensor(name)
        out[name] = t.to(torch.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default="Describe this image.")
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.image:
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        cfg.text_config.num_hidden_layers = args.layers
        cfg.vision_config.attn_impl = "sdpa"   # flash_attention_2 needs bf16 + the flash-attn package
        cfg.dtype = "float32"
        cfg.text_config.dtype = "float32"
        cfg.vision_config.dtype = "float32"
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True,
                                                 attn_implementation="eager")
        sd = load_selected(args.model, args.layers,
                           ["model.vision_tower.", "model.multi_modal_projector.",
                            "model.language_model.embed_tokens.", "model.language_model.norm."])
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("loaded", len(sd), "tensors; missing:", len(missing), "unexpected:", len(unexpected))
        model.eval()
    else:
        model = None

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    content = []
    if args.image:
        content.append({"type": "image", "image": args.image})
    content.append({"type": "text", "text": args.prompt})
    messages = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    kwargs = {}
    if args.image:
        from PIL import Image
        kwargs["images"] = [Image.open(args.image).convert("RGB")]
    inputs = proc(text=[text], return_tensors="pt", **kwargs)
    print("input_ids:", inputs["input_ids"][0].tolist())
    with open(os.path.join(args.out, "ref_input_ids.json"), "w") as f:
        json.dump(inputs["input_ids"][0].tolist(), f)
    with open(os.path.join(args.out, "ref_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    if not args.image:
        return 0

    pv = inputs["pixel_values"]
    thw = inputs["image_grid_thw"]
    print("pixel_values", tuple(pv.shape), "grid_thw", thw.tolist())
    np.save(os.path.join(args.out, "ref_patches.npy"), pv.float().numpy())
    np.save(os.path.join(args.out, "ref_grid_thw.npy"), thw.numpy())

    with torch.no_grad():
        pvf = pv.to(torch.float32)
        vt = model.model.vision_tower
        vhooks = {}

        def hook_patch(_m, _i, o):
            vhooks["patch"] = o.detach().clone()

        def hook_blk0(_m, _i, o):
            vhooks["blk0"] = o.detach().clone()

        vt.patch_embed.register_forward_hook(hook_patch)
        vt.encoder.blocks[0].register_forward_hook(hook_blk0)

        vfeat = vt(pixel_values=pvf, grid_thws=thw)
        merged = torch.cat(vfeat, dim=0) if isinstance(vfeat, (list, tuple)) else vfeat
        np.save(os.path.join(args.out, "ref_vpatch.npy"), vhooks["patch"].float().numpy())
        np.save(os.path.join(args.out, "ref_vblk0.npy"), vhooks["blk0"].float().numpy())
        np.save(os.path.join(args.out, "ref_vmerge.npy"), merged.float().numpy())
        proj = model.model.multi_modal_projector(merged)
        np.save(os.path.join(args.out, "ref_vproj.npy"), proj.float().numpy())

        hooks = {}
        def mk(i):
            def fn(_m, inp, _out):
                hooks[i] = inp[0].detach().clone()
            return fn
        model.model.language_model.layers[0].register_forward_hook(mk(0))

        out = model(**inputs)
        np.save(os.path.join(args.out, "ref_layer0_in.npy"), hooks[0].float().numpy())
        np.save(os.path.join(args.out, "ref_logits.npy"), out.logits[:, -1, :].float().numpy())
        nxt = int(out.logits[:, -1, :].argmax(-1).item())
        print("ref next token:", nxt, proc.tokenizer.decode([nxt]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
