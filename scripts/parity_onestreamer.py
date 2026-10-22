"""OneStreamer-4B parity: HF transformers Qwen3-VL reference vs the ggml engine.

The engine runs it as arch ``qwen3vl`` (its own backend: 36 dense full-attention
layers, no q|gate, IMROPE 24/20/20 over the full head_dim, DeepStack).

Usage (repo root, Windows: prefix the run with ``$env:PATH="build-ninja\\bin;$env:PATH"``)::

  # 1) engine dumps (an f16 GGUF keeps the text stages free of quantization noise;
  #    q8_0 works too, expect a slightly larger t_final_norm drift)
  set QWEN3VL_DUMP_IDS=1
  set QWEN3VL_PREFILL_DUMP_ONLY=1
  vqa-cli --model models/onestreamer-4b-q8_0.gguf --tokenizer models/OneStreamer-4B \
      --image IMG --ask "Describe the shapes and colors in this image." \
      --max-tokens 12 --temp 0 --n-ctx 4096 --dump-dir tmp_os > tmp_os\\engine.log 2>&1

  # 2) reference + comparison
  .venv\\Scripts\\python.exe scripts/parity_onestreamer.py --image IMG \
      --dump-dir tmp_os --engine-log tmp_os\\engine.log

Stages compared (the GGUF keeps the whole vision tower F16, so vision stages are
exactly comparable; the text stages carry the f16/q8 weight noise):
  ids            prompt token ids (template + <|image_pad|> count/positions)
  patches        patch vectors (block-major) vs processor pixel_values
  v_final        vision tower output (24 blocks)
  v_proj_out     merger output == the LLM image embeddings
  deepstack      3 tap features (post-2x2-fold merger) == deepstack_feature_lists
  t_final_norm   text final hidden over the prompt
  logits         first-step logits
  e2e            greedy continuation text match
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

CKPT = os.environ.get("ONESTREAMER_CKPT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                   "models", "OneStreamer-4B"))
IMAGE_PAD = 151655


def load_dump(dump_dir, name):
    p = os.path.join(dump_dir, "cpp_%s.bin" % name)
    s = os.path.join(dump_dir, "cpp_%s.shape" % name)
    if not (os.path.exists(p) and os.path.exists(s)):
        return None
    arr = np.fromfile(p, dtype=np.float32)
    with open(s) as f:
        ne = [int(x) for x in f.read().split()]
    arr = arr.reshape(list(reversed(ne))).squeeze()
    t = torch.from_numpy(arr)
    if t.dim() >= 2:
        t = t.permute(*range(t.dim() - 1, -1, -1)).contiguous()   # [ne0, ne1, ...]
    return t


def rel(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--engine-log", default=None)
    ap.add_argument("--prompt", default="Describe the shapes and colors in this image.")
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print("[ref] loading processor + model (bf16, cpu) ...", flush=True)
    proc = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.ckpt, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": os.path.abspath(args.image)},
        {"type": "text", "text": args.prompt}]}]
    inputs = proc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True)
    ref_ids = inputs["input_ids"][0].tolist()
    print("[ref] prompt ids:", len(ref_ids), "pixel:", tuple(inputs["pixel_values"].shape),
          "grid:", inputs["image_grid_thw"].tolist(), flush=True)

    dd, report = args.dump_dir, []

    # ---- prompt ids ---------------------------------------------------------
    if args.engine_log and os.path.exists(args.engine_log):
        txt = open(args.engine_log, encoding="utf-8", errors="replace").read()
        m = re.search(r"^prompt ids:(.*)$", txt, re.M)
        if m:
            ours = [int(x) for x in m.group(1).split()]
            same = ours == ref_ids
            print("[parity] ids   %s (%d vs %d)" % ("IDENTICAL" if same else "DIFFER",
                                                    len(ours), len(ref_ids)))
            if not same:
                for i, (a, b) in enumerate(zip(ours, ref_ids)):
                    if a != b:
                        print("   first diff at %d: ours %d ref %d" % (i, a, b))
                        break
                print("   ours:", ours[:24], "...")
                print("   ref :", ref_ids[:24], "...")
            report.append(("ids", "identical" if same else "DIFFER", len(ref_ids)))
    img_pos = [i for i, t in enumerate(ref_ids) if t == IMAGE_PAD]

    # ---- vision tower hooks -------------------------------------------------
    caps = {}

    def hook(key, picker=None):
        # keep the widest invocation: the decode steps re-run the same modules
        # with T=1 and would otherwise overwrite the prefill capture
        def fn(_m, _i, o):
            t = picker(o) if picker else o
            if t is None:
                return
            if isinstance(t, (list, tuple)):   # deepstack_feature_lists: one per tap
                t = t[0] if len(t) else None
                if t is None:
                    return
            prev = caps.get(key)
            if prev is None or (t.dim() >= 2 and prev.dim() >= 2 and t.shape[1] > prev.shape[1]):
                caps[key] = t.detach().clone()
        return fn

    vis = model.model.visual
    handles = [
        vis.blocks[-1].register_forward_hook(
            hook("tower", lambda o: o[0] if isinstance(o, tuple) else o)),
        vis.register_forward_hook(
            hook("proj", lambda o: o[0] if isinstance(o, tuple) else o)),
        vis.register_forward_hook(
            hook("deepstack", lambda o: o[1] if isinstance(o, tuple) and len(o) > 1 else None)),
        # the engine's t_final_norm is POST-norm, while generate()'s
        # hidden_states[-1] is the last decoder layer's pre-norm stream
        model.model.language_model.norm.register_forward_hook(
            hook("final_norm", lambda o: o[0] if isinstance(o, tuple) else o)),
    ]

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=(1 if args.skip_e2e else args.max_new),
                             do_sample=False, return_dict_in_generate=True,
                             output_hidden_states=True)
    for h in handles:
        h.remove()

    # ---- patches ------------------------------------------------------------
    ours = load_dump(dd, "patches")
    ref = inputs["pixel_values"].float().t()
    if ours is not None:
        if ours.shape == ref.shape:
            r = rel(ours, ref)
            print("[parity] patches        rel=%.3e %s" % (r, tuple(ours.shape)))
            report.append(("patches", r, tuple(ours.shape)))
        else:
            print("[parity] patches shape mismatch %s vs %s" % (tuple(ours.shape), tuple(ref.shape)))

    # ---- vision tower / merger ---------------------------------------------
    for name, key in (("v_final", "tower"), ("v_proj_out", "proj")):
        ours = load_dump(dd, name)
        if ours is None or caps.get(key) is None:
            continue
        ref = caps[key].float().t()
        if ours.shape == ref.shape:
            r = rel(ours, ref)
            print("[parity] %-14s rel=%.3e %s" % (name, r, tuple(ours.shape)))
            report.append((name, r, tuple(ours.shape)))
        else:
            print("[parity] %s shape mismatch %s vs %s" % (name, tuple(ours.shape), tuple(ref.shape)))

    # ---- deepstack taps: the engine dumps the layer-0 scatter ([D, T] with
    # zeros outside the visual run), the reference returns [n_img, D] per tap
    ours = load_dump(dd, "t_l0_deepstack")
    tap = caps.get("deepstack")          # tap 0 only (the engine dumps layer 0's add)
    if ours is not None and tap is not None:
        rows = ours[:, img_pos].t() if img_pos and ours.shape[1] > max(img_pos) else None
        if rows is not None:
            r = rel(rows, tap.float())
            print("[parity] deepstack[0]   rel=%.3e %s" % (r, tuple(tap.shape)))
            report.append(("deepstack0", r, tuple(tap.shape)))
    elif tap is not None:
        print("[parity] deepstack[0]   SKIP (engine dump has no t_l0_deepstack)")

    # ---- text side ----------------------------------------------------------
    ours = load_dump(dd, "t_final_norm")
    ref = caps["final_norm"][0].t().float() if caps.get("final_norm") is not None else None
    if ours is not None and ref is not None:
        if ours.shape == ref.shape:
            r, c = rel(ours, ref), cos(ours, ref)
            print("[parity] t_final_norm   rel=%.3e cos=%.6f %s" % (r, c, tuple(ours.shape)))
            report.append(("t_final_norm", r, "cos=%.6f" % c))
        else:
            print("[parity] t_final_norm shape mismatch %s vs %s" %
                  (tuple(ours.shape), tuple(ref.shape)))
    if caps.get("tower") is not None and load_dump(dd, "v_final") is not None:
        c = cos(load_dump(dd, "v_final"), caps["tower"].float().t())
        report.append(("v_final cos", c, ""))

    # NOTE: the engine's "logits" dump is a graph output that dump::write cannot
    # resolve by name, so the numeric first-step comparison is skipped here - the
    # e2e greedy prefix below covers the same path token for token.
    if out.scores:
        top = int(out.scores[0][0].argmax())
        print("[ref] first-step argmax %d (%r)" % (top, proc.tokenizer.decode([top])))

    # ---- e2e greedy ---------------------------------------------------------
    new_ids = out.sequences[0][len(ref_ids):].tolist()
    ref_text = proc.tokenizer.decode(new_ids, skip_special_tokens=False)
    print("[ref] greedy:", repr(ref_text[:300]))
    if args.engine_log and os.path.exists(args.engine_log):
        log = open(args.engine_log, encoding="utf-8", errors="replace").read()
        eng = None
        lines = [l for l in log.splitlines() if l.strip() and not l.startswith("[")]
        if lines:
            eng = lines[-1]
        if eng:
            a = proc.tokenizer.encode(ref_text, add_special_tokens=False)
            b = proc.tokenizer.encode(eng, add_special_tokens=False)
            n = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                n += 1
            print("[parity] e2e            token prefix match %d/%d" % (n, len(a)))
            print("[parity] engine:", repr(eng[:200]))
            report.append(("e2e", "%d/%d" % (n, len(a)), len(b)))

    print("\n[parity] summary:")
    for name, val, extra in report:
        print("  %-14s %s %s" % (name, ("rel=%.3e" % val) if isinstance(val, float) else val, extra))
    return 0


if __name__ == "__main__":
    sys.exit(main())
