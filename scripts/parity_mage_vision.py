"""Mage-VL VISION parity: official Mage-ViT vs the ggml engine - vision-only oracle.

Loads ONLY the visual submodule (~1 GB in bf16) so the whole thing fits a
14 GB host next to the desktop environment (the full-model oracle OOM-kills
the box). The text stack is a straight port of the already-verified VideoChat3
Qwen3 text graph, so vision is where the parity risk lives.

Usage::

  vqa-cli --model mage-vl-4b-q6_k.gguf --tokenizer models/Mage-VL \
      --image IMG --ask "Describe this image." --max-tokens 8 \
      --n-ctx 4096 --dump-dir tmp_dump
  python scripts/parity_mage_vision.py --image IMG --dump-dir tmp_dump
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) // 1024
    return -1

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


def load_visual():
    """Instantiate MageVLVisionPretrainedModel with ONLY visual weights resident."""
    torch.set_num_threads(8)
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = AutoConfig.from_pretrained(CKPT, trust_remote_code=True)
    cls = get_class_from_dynamic_module("modeling_mage_vl.MageVLVisionPretrainedModel", CKPT)
    vis = cls._from_config(cfg.vision_config).to(torch.bfloat16).eval()

    index = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    want = {k: fn for k, fn in index.items() if k.startswith("model.visual.")}
    sd = {}
    by_file = {}
    for k, fn in want.items():
        by_file.setdefault(fn, []).append(k)
    for fn, keys in by_file.items():
        from safetensors.torch import safe_open
        with safe_open(os.path.join(CKPT, fn), framework="pt") as sf:
            for k in keys:
                sd[k[len("model.visual."):]] = sf.get_tensor(k)
    missing, unexpected = vis.load_state_dict(sd, strict=False)
    print("[ref] visual weights: %d tensors loaded (missing %d, unexpected %d)"
          % (len(sd), len(missing), len(unexpected)), flush=True)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--dump-dir", required=True)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from PIL import Image

    proc = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[Image.open(args.image).convert("RGB")], return_tensors="pt")
    pv, grid, pp = inputs["pixel_values"], inputs["image_grid_thw"], inputs["patch_positions"]

    vis = load_visual()
    report = []

    def check(name, ref, cpp, tol_ok, tol_warn):
        if cpp.dim() == 2 and cpp.shape == ref.shape[::-1]:
            cpp = cpp.t()   # ggml 2d dumps are [dim, tokens]; refs are token-major
        r = rel(ref, cpp)
        verdict = "PASS" if r <= tol_ok else ("WARN" if r <= tol_warn else "FAIL")
        report.append((verdict, name, r, tuple(ref.shape), tuple(cpp.shape)))

    with torch.inference_mode():
        print("[rss", rss_mb(), "MB]", flush=True)
        check("patches", pv.float(), load_dump(args.dump_dir, "patches"), 2e-3, 1e-2)

        print("[rss", rss_mb(), "MB]", flush=True)
        x = vis.embeddings(pv.to(vis.dtype))
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = vis.layernorm_pre(x)
        check("v_patch_out", x[0].float(), load_dump(args.dump_dir, "v_patch_out"), 2e-2, 6e-2)

        # roped q for block 0, from the module's own qkv + rotary math
        print("[rss", rss_mb(), "MB]", flush=True)
        l0 = vis.encoder.layers[0]
        h = l0.layer_norm1(x)
        qkv_t = l0.self_attn.qkv(h).float()   # Linear already adds the bias
        nh, hd = vis.config.num_attention_heads, vis.config.hidden_size // vis.config.num_attention_heads
        qkv2 = qkv_t.reshape(-1, 3, nh, hd)          # [N, 3, nh, hd]
        q = qkv2[:, 0]                                # [N, nh, hd]
        freqs = vis.video_rope.forward_from_positions(pp).float()
        freqs = torch.cat([freqs, freqs], dim=-1)

        def rotate_half(t):
            te, to = t[..., ::2], t[..., 1::2]
            return torch.stack((-to, te), dim=-1).flatten(-2)
        qf = q.permute(1, 0, 2).float()               # [nh, N, hd]
        cos, sin = freqs.cos().unsqueeze(0), freqs.sin().unsqueeze(0)
        qr = qf * cos + rotate_half(qf) * sin         # [nh, N, hd]
        check("v_blk0_qkv", qkv_t.reshape(-1, qkv_t.shape[-1]),
              load_dump(args.dump_dir, "v_blk0_qkv"), 2e-2, 6e-2)
        check("v_blk0_q_rope", qr.permute(2, 0, 1).contiguous(),   # [hd, nh, N]
              load_dump(args.dump_dir, "v_blk0_q_rope"), 2e-2, 6e-2)

        # attention out via the module (eager path, same inputs)
        orig_fwd = l0.self_attn.forward
        def cap_attn(hidden_states, rotary_pos_emb=None, cu_seqlens=None, **kw):
            out, _ = orig_fwd(hidden_states, rotary_pos_emb=rotary_pos_emb,
                              cu_seqlens=cu_seqlens, **kw)
            cap_attn.out = out
            return out, None
        l0.self_attn.forward = cap_attn
        cu, _ = vis._build_cu_seqlens(grid, x.shape[1],
                                      fixed_t=vis.config.frame_windows_size, device="cpu")
        print("[rss", rss_mb(), "MB]", flush=True)
        enc = vis.encoder(x, rotary_pos_emb=freqs.unsqueeze(0), cu_seqlens=cu,
                          output_hidden_states=True, return_dict=True)
        l0.self_attn.forward = orig_fwd
        check("v_blk0_attn_out", cap_attn.out[0].float(),
              load_dump(args.dump_dir, "v_blk0_attn_out"), 2e-2, 6e-2)
        check("v_blk0_out", enc.hidden_states[1][0].float(),
              load_dump(args.dump_dir, "v_blk0_out"), 2e-2, 6e-2)
        print("[rss", rss_mb(), "MB] post-encoder", flush=True)
        check("v_final", enc.hidden_states[-1][0].float(),
              load_dump(args.dump_dir, "v_final"), 2e-2, 6e-2)

        vlnq = vis.merger.ln_q(enc.hidden_states[-1])
        if vlnq.dim() == 3:
            vlnq = vlnq[0]
        check("v_merge", vlnq.reshape(-1, vis.merger.hidden_size).float(),
              load_dump(args.dump_dir, "v_merge"), 2e-2, 6e-2)
        print("[rss", rss_mb(), "MB]", flush=True)
        vproj = vis.merger(enc.hidden_states[-1], patch_positions=pp)
        if vproj.dim() == 3:
            vproj = vproj[0]
        check("v_proj_out", vproj.float(), load_dump(args.dump_dir, "v_proj_out"), 2e-2, 6e-2)

    print("\n%-8s %-18s %-10s %s" % ("verdict", "stage", "rel", "shapes (ref vs cpp)"))
    bad = 0
    for verdict, name, r, rs, cs in report:
        print("%-8s %-18s %-10.3e %s vs %s" % (verdict, name, r, rs, cs))
        if verdict == "FAIL":
            bad += 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
