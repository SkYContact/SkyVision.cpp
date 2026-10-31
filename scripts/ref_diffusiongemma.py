#!/usr/bin/env python3
"""ref_diffusiongemma.py - oracle dumps for the DiffusionGemma vision tower.

Runs the *official* transformers implementation of the Gemma4 vision tower (the
same module DiffusionGemma embeds as `model.encoder.vision_tower`) on the exact
inputs the engine used, so a `--dump-dir` run of the engine can be compared
level by level.

Only the tower + `embed_vision` projection are loaded (~2.2 GB), not the 48 GB
language model: the tower is a self-contained submodule of the checkpoint.

    # 1) engine side (writes cpp_*.bin next to v_pxpy.txt into ./dgvis)
    DG_DUMP_DIR=dgvis vqa-cli -m models/diffusiongemma-26b-a4b-it-q8_0.gguf \\
        -t models/diffusiongemma-26B-A4B-it --image test_shapes.png \\
        --ask x -n 8 --cpu

    # 2) oracle (needs the checkpoint dir; transformers >= 5.8)
    python scripts/ref_diffusiongemma.py --src models/diffusiongemma-26B-A4B-it \\
        --engine-dump dgvis --out refvis

    # 3) compare
    python scripts/parity_diffusiongemma_vision.py --ref refvis --cpp dgvis

Levels written (ref_<name>.bin + ref_<name>.shape, ggml order [ne0, ne1, ...]):

    v_patch_out   patch embedding + learned 2-D position table   [1152, np]
    v_l0_out      output of encoder layer 0                      [1152, np]
    v_final       tower output after the 3x3 pooler + standardization
                                                                 [1152, n_soft]
    v_soft        after the weightless RMSNorm + embed_vision projection
                                                                 [2816, n_soft]

The engine feeds patches in [np, 768] HF layout (dy, dx, c per patch), so the
dumped `cpp_v_patches` is transposed back here - that keeps the comparison honest
about the preprocessing as well.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


def unwrap(arr, ne):
    """ggml ne -> numpy shape (drop the trailing 1s, reverse)."""
    dims = [d for d in ne if d != 1] or [1]
    return arr.reshape(list(reversed(dims)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="HF checkpoint dir (config.json + shards)")
    ap.add_argument("--engine-dump", required=True, help="engine --dump-dir with cpp_v_patches.bin")
    ap.add_argument("--out", default="refvis", help="output dir for ref_*.bin")
    args = ap.parse_args()

    from safetensors.torch import load_file
    from transformers import Gemma4VisionConfig, Gemma4VisionModel

    cfg_all = json.load(open(os.path.join(args.src, "config.json"), encoding="utf-8"))
    vcfg_dict = cfg_all["vision_config"]
    vcfg = Gemma4VisionConfig(**{k: v for k, v in vcfg_dict.items()
                                 if k not in ("architectures", "model_type", "dtype",
                                              "_name_or_path", "id2label", "label2id",
                                              "problem_type", "is_encoder_decoder",
                                              "chunk_size_feed_forward", "output_attentions",
                                              "output_hidden_states", "return_dict")})
    vcfg._attn_implementation = "eager"

    # ---- inputs: exactly what the engine used
    #
    # Prefer the host-side dump: the graph-side `cpp_v_patches` sink points at a
    # graph INPUT, whose buffer gallocr recycles after compute, so it reads back
    # garbage (this is why the first parity attempt showed corr ~0.002).
    host_p = os.path.join(args.engine_dump, "host_v_patches.shape")
    if os.path.exists(host_p):
        ne_p = [int(x) for x in open(host_p).read().split()]
        flat = np.fromfile(os.path.join(args.engine_dump, "host_v_patches.bin"), dtype=np.float32)
        patches = unwrap(flat, ne_p)                    # [np, 768], ggml order
    else:
        ne_p = [int(x) for x in open(os.path.join(args.engine_dump, "cpp_v_patches.shape")).read().split()]
        flat = np.fromfile(os.path.join(args.engine_dump, "cpp_v_patches.bin"), dtype=np.float32)
        patches = unwrap(flat, ne_p)
    if patches.shape[-1] != 768:
        patches = patches.T
    npatch = patches.shape[0]
    # The engine applies Gemma4VisionPatchEmbedder's `2 * (x - 0.5)` during
    # preprocessing (so its graph gets [-1,1]); HF applies it INSIDE the module,
    # so hand the oracle the original [0,1] pixels to compare like with like.
    patches = (patches + 1.0) * 0.5

    pxpy = [l.split() for l in open(os.path.join(args.engine_dump, "v_pxpy.txt")).read().split("\n")[1:] if l.strip()]
    assert len(pxpy) == npatch, f"{len(pxpy)} positions vs {npatch} patches"
    pos = np.array([[int(a), int(b)] for a, b in pxpy], dtype=np.int64)

    pixel_values = torch.from_numpy(np.ascontiguousarray(patches)).float().unsqueeze(0)   # [1, np, 768]
    position_ids = torch.from_numpy(pos).unsqueeze(0)                                    # [1, np, 2]

    # ---- weights: only the tower + the projection
    weighprefix = "model.encoder.vision_tower."
    proj_prefix = "model.encoder.embed_vision."
    state = {}
    proj = {}
    for f in sorted(os.listdir(args.src)):
        if not f.endswith(".safetensors"):
            continue
        sd = load_file(os.path.join(args.src, f))
        for k, v in sd.items():
            if k.startswith(weighprefix):
                state[k[len(weighprefix):]] = v
            elif k.startswith(proj_prefix):
                proj[k[len(proj_prefix):]] = v
        del sd
    if not state:
        sys.exit("no vision_tower tensors found - wrong --src?")

    model = Gemma4VisionModel(vcfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
    if real_missing:
        print(f"  warning: missing tower tensors: {real_missing[:5]}")
    if unexpected:
        print(f"  warning: unexpected tower tensors: {unexpected[:5]}")
    model.eval()

    outdir = args.out
    os.makedirs(outdir, exist_ok=True)

    def dump(name, t):
        a = t.detach().float().cpu().numpy()
        # write in ggml order: ne0 is the last numpy dim
        ne = list(reversed(a.shape))
        a.reshape(-1).astype(np.float32).tofile(os.path.join(outdir, f"ref_{name}.bin"))
        with open(os.path.join(outdir, f"ref_{name}.shape"), "w") as f:
            f.write(" ".join(str(d) for d in ne) + " " + " ".join("1" for _ in range(4 - len(ne))) + "\n")
        print(f"  wrote ref_{name}.bin  {a.shape}")

    hooks = {}

    def grab(name):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            hooks[name] = t
        return fn

    model.patch_embedder.register_forward_hook(grab("v_patch_out"))
    model.encoder.layers[0].register_forward_hook(grab("v_l0_out"))
    # one hook per block: paired with the engine's DG_VISION_LAYERS=N (whose
    # host_v_final is block N-1's output) this bisects a divergence inside the
    # tower without trusting any in-graph dump
    for i, layer in enumerate(model.encoder.layers):
        layer.register_forward_hook(grab(f"v_blk{i}"))
    # tower body output BEFORE the pooler: compares against the engine's
    # host_v_final, so a mismatch here isolates the 27 blocks from the 3x3 pool
    model.encoder.register_forward_hook(grab("v_enc_out"))

    k = int(vcfg.pooling_kernel_size)
    print(f"  pixel_values {tuple(pixel_values.shape)}  positions {tuple(position_ids.shape)}  "
          f"pool {k}x{k} -> expect {npatch // (k * k)} soft tokens")
    with torch.no_grad():
        res = model(pixel_values=pixel_values, pixel_position_ids=position_ids, return_dict=True)
    last = res.last_hidden_state if hasattr(res, "last_hidden_state") else res[0]
    print(f"  oracle last_hidden_state {tuple(last.shape)}")

    dump("v_patch_out", hooks["v_patch_out"][0])
    dump("v_l0_out", hooks["v_l0_out"][0])
    enc_out = hooks["v_enc_out"][0]
    dump("v_enc_out", enc_out)
    for i in range(len(model.encoder.layers)):
        dump(f"v_blk{i}", hooks[f"v_blk{i}"][0])
    dump("v_final", last)

    # embed_vision: weightless RMSNorm -> projection
    w = proj.get("embedding_projection.weight")
    if w is None:
        sys.exit("no embed_vision.embedding_projection.weight in the checkpoint")
    eps = float(vcfg_dict.get("rms_norm_eps", 1e-6))
    x = last.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    soft = x @ w.float().T
    dump("v_soft", soft)
    print(f"oracle done: {npatch} patches -> {last.shape[1]} soft tokens")


if __name__ == "__main__":
    main()
