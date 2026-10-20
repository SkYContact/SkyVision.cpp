"""Reference oracle for MiniCPM-V 4.7 (transformers >= 5.20, CPU bf16).

Captures the tensors scripts/parity_minicpmv47.py compares the C++ backend
against, stage by stage:

  patches  pixel_values, reshaped to the engine's (c, ky, kx) patch layout
  emb      vision_tower.embeddings output (patch conv + learned position table)
  vm       vit_merger output (16x only; empty in 4x)
  tower    vision_tower output = post_layernorm, i.e. after the insert merger
  feats    merger output (what the LLM consumes), [n_img, d_text]
  logits   last-position logits of the prefill
  gen      greedy 24-token continuation

Usage:
  python scripts/ref_minicpmv47.py --model models/MiniCPM-V-4.7-1B \
      --image photo.jpg --ask "Describe the image." --out ref_cpm.npz [--4x]

Needs a transformers build that ships models/minicpmv4_7 (5.20+); the repo's
main .venv is pinned to 4.57 for the other backends, so keep a separate env.
Careful: `pip install transformers` also pulls a torch wheel, and a torch that
does not match the interpreter's torchvision breaks the *video* processor with
"operator torchvision::nms does not exist". Install a matching pair:

  python -m venv .venv-cpm --system-site-packages
  .venv-cpm/Scripts/pip install transformers==5.20.2 --no-deps
  .venv-cpm/Scripts/pip install tokenizers huggingface_hub safetensors
  # CUDA oracle (minutes -> seconds on the 7-crop slice case):
  .venv-cpm/Scripts/pip install torch torchvision --index-url \
      https://download.pytorch.org/whl/cu130
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, MiniCPMV4_7ForConditionalGeneration


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="models/MiniCPM-V-4.7-1B")
    ap.add_argument("--image", default="",
                    help="single image (omit when using --frames-dir)")
    ap.add_argument("--frames-dir", default="",
                    help="video smoke test: one frame per image file in that directory; "
                         "the prompt uses the video placeholder/canvas path")
    ap.add_argument("--ask", default="Describe the image.")
    ap.add_argument("--out", default="ref_minicpmv47.npz")
    ap.add_argument("--4x", dest="ds4", action="store_true",
                    help="downsample_mode='4x' (must match MINICPM_DOWNSAMPLE=4x on the C++ side)")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--no-gen", action="store_true",
                    help="skip the greedy generate (the per-token forwards dominate runtime)")
    ap.add_argument("--attn", default="sdpa", help="attn implementation (sdpa|eager|flash_attention_2)")
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda (a GPU oracle turns the 7-crop case from minutes into seconds)")
    ap.add_argument("--ids-only", action="store_true",
                    help="only tokenize (no weights loaded): prompt ids + patches, seconds")
    ap.add_argument("--pos-only", action="store_true",
                    help="also compute the canvas M-RoPE position ids (model built on the "
                         "meta device: no weights, no forward)")
    args = ap.parse_args()
    ds = "4x" if args.ds4 else "16x"

    proc = AutoProcessor.from_pretrained(args.model)

    if args.frames_dir:
        import glob
        files = sorted(glob.glob(os.path.join(args.frames_dir, "*")))
        frames = [Image.open(f).convert("RGB") for f in files]
        msgs = [{"role": "user", "content": [{"type": "video"},
                                             {"type": "text", "text": args.ask}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                          enable_thinking=False)
        # do_sample_frames=False: use every frame we hand over (no fps/duration
        # metadata here, and the engine also feeds its full frame list)
        inputs = proc(text=[prompt], videos=[frames], return_tensors="pt", downsample_mode=ds,
                      do_sample_frames=False)
        print(f"[ref] {len(frames)} frames from {args.frames_dir}")
    else:
        img = Image.open(args.image).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": args.ask}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                          enable_thinking=False)
        inputs = proc(text=[prompt], images=[img], return_tensors="pt", downsample_mode=ds)

    if args.ids_only or args.pos_only:
        data = {"ids": inputs["input_ids"][0].cpu().numpy()}
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"][0].float().cpu().numpy()
            data["patches"] = pv.reshape(3, pv.shape[1], -1, pv.shape[1])
        if "target_sizes" in inputs:
            data["target_sizes"] = inputs["target_sizes"].cpu().numpy()
        if args.pos_only:
            # canvas M-RoPE is pure integer arithmetic on config + input ids, so the
            # model can live on the meta device: no weights, no forward, seconds
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(args.model)
            with torch.device("meta"):
                m = MiniCPMV4_7ForConditionalGeneration(cfg)
            m.config.downsample_mode = ds
            pos, delta = m.model.get_rope_index(
                inputs["input_ids"], mm_token_type_ids=inputs.get("mm_token_type_ids"),
                target_sizes=inputs.get("target_sizes"),
                target_sizes_videos=inputs.get("target_sizes_videos"), downsample_mode=ds)
            data["pos"] = pos[:, 0, :].cpu().numpy()      # (3, T) for batch 0
            data["delta"] = delta.cpu().numpy()
            print(f"[ref] pos shape {tuple(pos.shape)} delta {delta.tolist()}")
        np.savez(args.out, **data)
        ts = inputs.get("target_sizes")
        crops = 0 if ts is None else ts.shape[0]
        print(f"[ref] ids={inputs['input_ids'].shape[1]} crops={crops} "
              f"target_sizes={[] if ts is None else ts.tolist()}")
        print(f"[ref] wrote {args.out} (no weights loaded)")
        return

    model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn).eval()
    if args.device != "cpu":
        model = model.to(args.device)
        inputs = {k: (v.to(args.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    cap = {}
    model.model.vision_tower.register_forward_hook(
        lambda m, i, o: cap.__setitem__("tower", o.last_hidden_state))
    model.model.vision_tower.embeddings.register_forward_hook(
        lambda m, i, o: cap.__setitem__("emb", o))
    model.model.vision_tower.vit_merger.register_forward_hook(
        lambda m, i, o: cap.__setitem__("vm", o))
    # the merger yields the features of one crop (or a list of them): accumulate
    # across calls so a sliced prompt ends up with every crop, thumbnail first
    def merger_hook(m, i, o):
        got = list(o) if isinstance(o, (list, tuple)) else [o]
        cap["merger"] = cap.get("merger", []) + got
    model.model.merger.register_forward_hook(merger_hook)

    with torch.no_grad():
        out = model(**inputs, downsample_mode=ds)

    feats = cap["merger"]
    if isinstance(feats, (list, tuple)):
        feats = torch.cat([f.reshape(-1, f.shape[-1]) for f in feats], dim=0)
    feats = feats.float().cpu().numpy().reshape(-1, feats.shape[-1])
    tower = cap["tower"].float().cpu().numpy().reshape(-1, cap["tower"].shape[-1])
    emb = cap["emb"].float().cpu().numpy().reshape(-1, cap["emb"].shape[-1])
    vm = cap.get("vm")
    vm = np.zeros((1, 1152), np.float32) if vm is None else \
        (vm[0] if isinstance(vm, (list, tuple)) else vm).float().cpu().numpy().reshape(-1, 1152)

    # patches in the engine layout: (c, ky, token, kx)
    pv = inputs["pixel_values"][0].float().cpu().numpy()
    p = pv.shape[1]
    patches = pv.reshape(3, p, -1, p)

    logits = out.logits[0, -1].float().cpu().numpy()
    gen_ids = []
    if not args.no_gen:
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False,
                                 downsample_mode=ds)
        gen_ids = gen[0, inputs["input_ids"].shape[1]:].tolist()

    np.savez(args.out, ids=inputs["input_ids"][0].cpu().numpy(), patches=patches,
             emb=emb, vm=vm, tower=tower, feats=feats, logits=logits,
             gen=np.array(gen_ids), target_sizes=inputs["target_sizes"].cpu().numpy())
    print(f"[ref] downsample={ds} ids={inputs['input_ids'].shape[1]} "
          f"patches={emb.shape[0]} tower={tower.shape[0]} feats={feats.shape[0]}")
    if gen_ids:
        print("[ref] answer: " + proc.batch_decode([gen_ids], skip_special_tokens=True)[0][:300])
    print(f"[ref] wrote {args.out}")


if __name__ == "__main__":
    main()
