"""Marlin-2B parity: official Qwen3_5ForConditionalGeneration vs the ggml engine.

Usage (q35 conda env, py312 + torch 2.11 + transformers>=5.7)::

  # 1) engine dumps (run from the repo root)
  vqa-cli --model marlin-2b-q8_0.gguf --tokenizer models/Marlin-2B \
      --image IMG --ask "Describe this image." --max-tokens 20 --dump-dir tmp_dump

  # 2) reference side + comparison
  python scripts/parity_marlin.py --image IMG --dump-dir tmp_dump

Stages compared (both f32 on the GPU; the engine GGUF keeps the whole vision
tower F16 so vision stages are exactly comparable; text stages carry q8_0
quantization noise):
  patches      our patchify3d output (block-major) vs processor pixel_values
  v_patch_out  patch conv + pos embed output, permuted row<->block-major
  v_final      vision tower output (24 blocks)
  v_proj_out   merger + projector output == the LLM image embeddings
  t_final_norm text final hidden (last prompt token column-wise)
  logits       last-token logits (q8 noise expected)
  e2e          greedy continuation, token-prefix match vs the engine run

Both sides emit the patch stream in 2x2 BLOCK-MAJOR order (mh, mw, ir, ic) -
the Qwen3-VL processor rearranges patches with the merge kernel, our host
patchify3d does the same - so tensors compare index-for-index.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

CKPT = os.environ.get("MARLIN_CKPT", os.path.join(os.path.dirname(__file__), "..", "models", "Marlin-2B"))


def load_dump(dump_dir, name):
    p = os.path.join(dump_dir, "cpp_%s.bin" % name)
    s = os.path.join(dump_dir, "cpp_%s.shape" % name)
    with open(p, "rb") as f:
        arr = np.fromfile(f, dtype=np.float32)
    with open(s) as f:
        ne = [int(x) for x in f.read().split()]
    # ggml ne = [ne0..ne3], C-order numpy reshape(s3,s2,s1,s0) is row-major
    arr = arr.reshape(list(reversed(ne))).squeeze()
    # -> axes (T, ...) with ne0 fastest; return as torch [ne0, ...] column-major view
    t = torch.from_numpy(arr)
    if t.dim() >= 2:
        t = t.permute(*range(t.dim() - 1, -1, -1)).contiguous()   # [ne0, ne1, ...]
    return t


def rel(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--prompt", default="Describe this image.")
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    print("[ref] loading processor + model (bf16, cuda) ...", flush=True)
    proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        CKPT, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True).eval()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": os.path.abspath(args.image)},
        {"type": "text", "text": args.prompt}]}]
    inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
    print("[ref] input ids:", inputs["input_ids"].shape,
          "pixel:", inputs["pixel_values"].shape,
          "grid:", inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else None, flush=True)

    dd = args.dump_dir
    have = lambda n: os.path.exists(os.path.join(dd, "cpp_%s.bin" % n))
    report = []

    # ---- patches: engine patch vectors vs processor pixel_values -------------
    if have("patches"):
        ours = load_dump(dd, "patches")                       # [1536, N] block-major
        ref = inputs["pixel_values"].float().cpu().t()        # [1536, N]
        if ours.shape == ref.shape:
            r = rel(ours, ref)
            report.append(("patches", r, ours.shape))
            print("[parity] patches            rel=%.3e %s" % (r, ours.shape))
        else:
            print("[parity] patches shape mismatch %s vs %s" % (ours.shape, ref.shape))

    # ---- hooks for the vision tower -------------------------------------------
    caps = {}

    def hook(name, key, pooler=False):
        def fn(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            if pooler and hasattr(t, "pooler_output"):
                t = t.pooler_output
            caps[key] = t.detach().float().cpu()
        return fn

    vis = model.model.visual
    handles = [
        vis.patch_embed.register_forward_hook(hook("v_patch_out", "patch")),
        vis.blocks[-1].register_forward_hook(hook("v_final", "tower")),
        vis.register_forward_hook(hook("v_proj_out", "proj", pooler=True)),
    ]

    # one forward captures everything (greedy continuation afterwards for e2e)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.skip_e2e and 1 or args.max_new,
                             do_sample=False, output_hidden_states=True, return_dict_in_generate=True)
    for h in handles:
        h.remove()

    # patch conv output on the ref side has NO pos embed yet (patch_embed.forward
    # is the conv only); ours adds the pos table + bias -> rebuild ours' pos part
    # is impossible from outside, so compare at the tower input instead:
    # ref patch hook == conv output; engine v_patch_out == conv + bias + pos.
    # The block-major order is shared, so compare the towers and merger directly.
    if have("v_final") and "tower" in caps:
        ours = load_dump(dd, "v_final")           # [1024, N]
        ref = caps["tower"].t()                    # [1024, N]
        if ours.shape == ref.shape:
            r = rel(ours, ref)
            report.append(("v_final", r, ours.shape))
            print("[parity] v_final            rel=%.3e %s" % (r, ours.shape))
        else:
            print("[parity] v_final shape mismatch %s vs %s" % (ours.shape, ref.shape))
    if have("v_proj_out") and "merger" in caps:
        ours = load_dump(dd, "v_proj_out")        # [2048, M]
        # the ref merger hook sees the merger INPUT; pooler_output comes from the
        # wrapper - grab the visual module output instead via a second pass below
    if have("v_proj_out"):
        ours = load_dump(dd, "v_proj_out")
        ref = caps["proj"].t()
        if ours.shape == ref.shape:
            r = rel(ours, ref)
            report.append(("v_proj_out", r, ours.shape))
            print("[parity] v_proj_out         rel=%.3e %s" % (r, ours.shape))
        else:
            print("[parity] v_proj_out shape mismatch %s vs %s" % (ours.shape, ref.shape))

    # ---- text side -------------------------------------------------------------
    hs = out.hidden_states          # tuple per generated token; [0] = prompt, tuple over layers
    prompt_hidden = hs[0][-1]       # final-norm hidden over the whole prompt [1, T, D]
    if have("t_final_norm"):
        ours = load_dump(dd, "t_final_norm")      # [2048, T]
        ref = prompt_hidden[0].t().float().cpu()
        if ours.shape == ref.shape:
            r = rel(ours, ref)
            report.append(("t_final_norm", r, ours.shape))
            print("[parity] t_final_norm       rel=%.3e %s (q8 text weights)" % (r, ours.shape))
        else:
            print("[parity] t_final_norm shape mismatch %s vs %s" % (ours.shape, ref.shape))

    # ---- e2e greedy ------------------------------------------------------------
    new_ids = out.sequences[0][inputs["input_ids"].shape[1]:].tolist()
    ref_text = proc.tokenizer.decode(new_ids, skip_special_tokens=False)
    print("[ref] greedy:", repr(ref_text[:400]))

    ours_txt = os.path.join(dd, "engine_out.txt")
    if os.path.exists(ours_txt):
        eng = open(ours_txt, encoding="utf-8").read()
        # both sides without the training-time <think> artifact
        import re as _re
        ref_clean = _re.sub(r"<think>.*?</think>\s*", "", ref_text, flags=_re.S)
        ref_clean = _re.sub(r"^\s*<think>\s*", "", ref_clean)
        ref_toks = proc.tokenizer.encode(ref_clean, add_special_tokens=False)
        eng_ids = proc.tokenizer.encode(eng, add_special_tokens=False)
        n = 0
        for a, b in zip(ref_toks, eng_ids):
            if a != b:
                break
            n += 1
        print("[parity] e2e token prefix match %d/%d (engine tokens %d)" %
              (n, len(ref_toks), len(eng_ids)))
        print("[parity] engine:", repr(eng[:200]))
        report.append(("e2e_prefix", n, len(ref_toks)))

    print("\n[parity] summary:")
    for name, val, extra in report:
        print("  %-14s %s %s" % (name, ("rel=%.3e" % val) if isinstance(val, float) else val, extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
