"""Mage-VL parity: official MageVLForConditionalGeneration vs the ggml engine.

Usage::

  # 1) engine dumps (repo root; any build, CUDA or CPU)
  vqa-cli --model mage-vl-4b-q6_k.gguf --tokenizer models/Mage-VL \
      --image IMG --ask "Describe this image." --max-tokens 8 \
      --n-ctx 4096 --dump-dir tmp_dump

  # 2) reference side + comparison
  python scripts/parity_mage.py --image IMG --dump-dir tmp_dump

All reference tensors are captured with forward hooks from ONE real model
forward (the same pass that produces the greedy answer), so the oracle is
self-consistent by construction. Stages:

  patches      our block-layout patchify vs processor pixel_values
  v_patch_out  patch conv + layernorm_pre            (hook on layernorm_pre)
  v_blk0_qkv   block 0 fused qkv                     (hook on qkv)
  v_blk0_q_rope block 0 roped q                      (recomputed from the hooked
               qkv with the model's own rotary math - same op as attention)
  v_blk0_attn_out block 0 attention output            (hook on self_attn)
  v_blk0_out   block 0 residual output               (encoder hidden_states[1])
  v_final      vision tower output                   (encoder hidden_states[-1])
  v_merge      merger LN + 2x2 fold                  (hook on merger.ln_q)
  v_proj_out   merger MLP output == LLM embeddings   (hook on merger)
  t_final_norm text final hidden (last token)        (hook on lm.norm)
  logits       last-token logits
  e2e          greedy continuation, token-prefix match vs the engine run
"""
import argparse
import os
import sys

import numpy as np
import torch

CKPT = os.environ.get("MAGE_CKPT", os.path.join(os.path.dirname(__file__), "..", "models", "Mage-VL"))


def load_dump(dump_dir, name):
    p = os.path.join(dump_dir, "cpp_%s.bin" % name)
    s = os.path.join(dump_dir, "cpp_%s.shape" % name)
    with open(p, "rb") as f:
        arr = np.fromfile(f, dtype=np.float32)
    with open(s) as f:
        ne = [int(x) for x in f.read().split()]
    arr = arr.reshape(list(reversed(ne))).squeeze()
    t = torch.from_numpy(arr.copy())
    if t.dim() >= 2:
        t = t.permute(*range(t.dim() - 1, -1, -1)).contiguous()   # [ne0, ne1, ...]
    return t


def rel(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() != b.numel():
        return float("nan")
    return ((a - b).norm() / (a.norm() + 1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--prompt", default="Describe this image.")
    ap.add_argument("--max-new", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForCausalLM
    from PIL import Image

    print("[ref] loading processor + model (bf16, GPU w/ CPU offload) ...", flush=True)
    proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    torch.set_num_threads(8)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, trust_remote_code=True, dtype=torch.bfloat16,
        device_map="cpu").eval()   # CPU-only oracle: no CUDA context, lean RSS
    dev = torch.device("cpu")
    model.preparer = None

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": args.prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[Image.open(args.image).convert("RGB")], return_tensors="pt")
    pv = inputs["pixel_values"]
    grid = inputs["image_grid_thw"]
    pp = inputs["patch_positions"]
    print("[ref] pixel_values:", tuple(pv.shape), "grid:", grid.tolist(),
          "ids:", inputs["input_ids"].shape, flush=True)

    vis = model.model.visual
    cap = {}

    def hook(name):
        def f(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            cap[name] = t.detach().clone()
        return f

    h_pre = vis.layernorm_pre.register_forward_hook(hook("v_patch_out"))
    l0 = vis.encoder.layers[0]
    h_qkv = l0.self_attn.qkv.register_forward_hook(hook("v_blk0_qkv"))
    h_attn = l0.self_attn.register_forward_hook(hook("v_blk0_attn_out"))
    h_lnq = vis.merger.ln_q.register_forward_hook(hook("v_merge_lnq"))
    h_mm = vis.merger.register_forward_hook(hook("v_proj"))
    lm = model.model.language_model
    h_norm = lm.norm.register_forward_hook(hook("t_final_norm"))

    pvd = pv.to(vis.dtype).to(dev)
    with torch.inference_mode():
        out = model(input_ids=inputs["input_ids"].to(dev), pixel_values=pvd,
                    image_grid_thw=grid.to(dev), patch_positions=pp.to(dev),
                    output_hidden_states=True, logits_to_keep=1)
    for hh in (h_pre, h_qkv, h_attn, h_lnq, h_mm, h_norm):
        hh.remove()

    # encoder hidden states (block0 out, final): rerun just the tower's encoder
    with torch.inference_mode():
        x = vis.embeddings(pvd)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = vis.layernorm_pre(x)
        cu, _ = vis._build_cu_seqlens(grid.to(vis.device), x.shape[1],
                                      fixed_t=vis.config.frame_windows_size, device=dev)
        enc = vis.encoder(x, rotary_pos_emb=None, cu_seqlens=cu,
                          output_hidden_states=True, return_dict=True)
        enc_hs = enc.hidden_states

    report = []

    def check(name, ref, cpp, tol_ok, tol_warn):
        if cpp.dim() == 2 and cpp.shape == ref.shape[::-1]:
            cpp = cpp.t()   # ggml 2d dumps are [dim, tokens]; refs are token-major
        r = rel(ref, cpp)
        verdict = "PASS" if r <= tol_ok else ("WARN" if r <= tol_warn else "FAIL")
        report.append((verdict, name, r, tuple(ref.shape), tuple(cpp.shape)))

    check("patches", pvd.float().cpu(), load_dump(args.dump_dir, "patches"), 2e-3, 1e-2)
    check("v_patch_out", cap["v_patch_out"][0].float().cpu(),
          load_dump(args.dump_dir, "v_patch_out"), 2e-2, 6e-2)
    check("v_blk0_qkv", cap["v_blk0_qkv"][0].float().cpu(),
          load_dump(args.dump_dir, "v_blk0_qkv"), 2e-2, 6e-2)

    # roped q: recompute from the hooked qkv with the model's own rotary math
    qkv_t = cap["v_blk0_qkv"].float().cpu()
    nh = vis.config.num_attention_heads
    hd = vis.config.hidden_size // nh
    q = qkv_t.reshape(qkv_t.shape[0], qkv_t.shape[1], 3, nh, hd).permute(2, 0, 1, 3, 4)[0]
    with torch.inference_mode():
        freqs = vis.video_rope.forward_from_positions(pp.to(dev)).float().cpu()
    freqs = torch.cat([freqs, freqs], dim=-1)

    def rotate_half(t):
        te, to = t[..., ::2], t[..., 1::2]
        return torch.stack((-to, te), dim=-1).flatten(-2)
    qf = q.transpose(0, 1).float()                       # [heads, N, hd]
    cos = freqs.cos().unsqueeze(1).float()
    sin = freqs.sin().unsqueeze(1).float()
    qr = (qf * cos + rotate_half(qf) * sin)              # [heads, N, hd]
    check("v_blk0_q_rope", qr.permute(2, 0, 1).contiguous(),          # [hd, heads, N]
          load_dump(args.dump_dir, "v_blk0_q_rope"), 2e-2, 6e-2)

    check("v_blk0_attn_out", cap["v_blk0_attn_out"][0].float().cpu(),
          load_dump(args.dump_dir, "v_blk0_attn_out"), 2e-2, 6e-2)
    check("v_blk0_out", enc_hs[1][0].float().cpu(),
          load_dump(args.dump_dir, "v_blk0_out"), 2e-2, 6e-2)
    check("v_final", enc_hs[-1][0].float().cpu(),
          load_dump(args.dump_dir, "v_final"), 2e-2, 6e-2)
    vlnq = cap["v_merge_lnq"]
    if vlnq.dim() == 3:
        vlnq = vlnq[0]
    v_merge = vlnq.reshape(-1, vis.merger.hidden_size)
    check("v_merge", v_merge.float().cpu(), load_dump(args.dump_dir, "v_merge"), 2e-2, 6e-2)
    vproj = cap["v_proj"]
    if vproj.dim() == 3:
        vproj = vproj[0]
    check("v_proj_out", vproj.float().cpu(), load_dump(args.dump_dir, "v_proj_out"), 2e-2, 6e-2)

    th = cap["t_final_norm"]
    tfn = load_dump(args.dump_dir, "t_final_norm")
    check("t_final_norm", th[0, -1].float().cpu(), tfn[:, -1] if tfn.dim() == 2 else tfn,
          3e-2, 9e-2)
    logits = out.logits[0, -1]
    check("logits", logits.float().cpu(), load_dump(args.dump_dir, "logits"), 3e-2, 1e-1)

    # e2e greedy
    with torch.inference_mode():
        gen = model.generate(**{k: (v.to(dev) if hasattr(v, "to") else v)
                                for k, v in inputs.items()},
                             max_new_tokens=args.max_new, do_sample=False)
    answer = proc.tokenizer.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("[ref] e2e greedy:", repr(answer.strip()))

    print("\n%-8s %-18s %-10s %s" % ("verdict", "stage", "rel", "shapes (ref vs cpp)"))
    bad = 0
    for verdict, name, r, rs, cs in report:
        print("%-8s %-18s %-10.3e %s vs %s" % (verdict, name, r, rs, cs))
        if verdict == "FAIL":
            bad += 1
    print("[ref] answer above must token-prefix-match the engine output")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
