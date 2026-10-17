"""Convert a ViViT checkpoint (transformers `vivit`) to GGUF.

Usage::

  python convert/convert_vivit.py [--src DIR] [-o OUT.gguf] [--outtype f16|q8_0|q4_0]

The checkpoint is located like the other converters (``--src``, then
``$VIVIT_SRC``, the HF cache, then ``models/vivit-b-16x2``); download one with::

  hf download google/vivit-b-16x2 --local-dir models/vivit-b-16x2
  hf download google/vivit-b-16x2-kinetics400 --local-dir models/vivit-kinetics400

ViViT-B/16x2 is the joint space-time variant, i.e. a plain pre-LN ViT over
spatio-temporal tubelet tokens - the same graph the videomae backend implements.
Only these differ, and all of them are metadata knobs in the engine:

* a learned **cls token** is prepended (the position table has a row for it);
* the position table is a **learned parameter** (VideoMAE generates sin-cos);
* ``hidden_act = gelu_fast`` = ``x * sigmoid(1.702 x)`` (ggml's gelu_quick);
* ``layer_norm_eps = 1e-6`` and **q/k/v all carry a bias**;
* preprocessing resizes the shortest edge to 256 before the 224 center crop and
  zero-centers (equivalent to mean = std = 0.5).

Produced layout (``general.architecture = vivit``)::

  v.patch_embd.weight  [dim, C*tubelet*patch*patch]   Conv3d folded into a matmul
  v.patch_embd.bias    [dim]
  v.cls_token.weight   [dim]                           learned cls token
  v.pos_emb.weight     [n_patches + 1, dim]            learned table (includes cls)
  v.blk.{i}.ln1/wqkv/wo/ln2/fc1/fc2                    same block layout as videomae
  v.norm.{weight,bias}                                 final LayerNorm
  v.cls.{weight,bias}   [num_labels, dim]              Kinetics-400 classifier
"""
import argparse
import json
import os

import numpy as np
import gguf

import _lib
from _lib import log, warn, fail
import hf_source

MODEL_ID = "google/vivit-b-16x2"
ENV_VAR = "VIVIT_SRC"
PATTERNS = ("model.safetensors", "model.safetensors.index.json")

CFG = dict(n_layer=12, n_embd=768, n_head=12, n_ff=3072, patch=16, tubelet=2,
           image_size=224, num_frames=32, num_channels=3, eps=1e-6,
           hidden_act="gelu_fast", num_labels=0, labels=[],
           resize_shortest=256, resample=2,          # VivitImageProcessor: shortest 256, PIL BILINEAR
           mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q4_0": gguf.GGMLQuantizationType.Q4_0}


def ensure_safetensors(src_dir):
    """The google/vivit-* releases ship pytorch_model.bin only; TensorSource reads
    safetensors. Convert once (needs torch + safetensors) and reuse afterwards."""
    if any(f.endswith(".safetensors") for f in os.listdir(src_dir)):
        return
    bin_path = os.path.join(src_dir, "pytorch_model.bin")
    if not os.path.isfile(bin_path):
        fail(f"no *.safetensors and no pytorch_model.bin in {src_dir}")
    log("converting pytorch_model.bin -> model.safetensors (one-off, needs torch)")
    import torch
    from safetensors.torch import save_file

    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    sd = {k: v.contiguous().clone() for k, v in sd.items()}
    save_file(sd, os.path.join(src_dir, "model.safetensors"))
    log(f"wrote {os.path.join(src_dir, 'model.safetensors')}")


def read_config(src_dir):
    cfg = CFG.copy()
    path = os.path.join(src_dir, "config.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        tube = raw.get("tubelet_size") or [2, 16, 16]
        if isinstance(tube, int):
            tube = [tube, raw.get("patch_size", 16), raw.get("patch_size", 16)]
        cfg.update({k: v for k, v in dict(
            n_layer=raw.get("num_hidden_layers"), n_embd=raw.get("hidden_size"),
            n_head=raw.get("num_attention_heads"), n_ff=raw.get("intermediate_size"),
            patch=int(tube[1]), tubelet=int(tube[0]),
            image_size=raw.get("image_size"), num_frames=raw.get("num_frames"),
            num_channels=raw.get("num_channels"), eps=raw.get("layer_norm_eps"),
            hidden_act=raw.get("hidden_act"),
        ).items() if v is not None})
        id2label = raw.get("id2label") or {}
        if id2label:
            pairs = sorted(((int(k), v) for k, v in id2label.items()), key=lambda p: p[0])
            cfg["labels"] = [name for _, name in pairs]
            cfg["num_labels"] = len(cfg["labels"])
        log(f"hidden_act from the checkpoint: {cfg['hidden_act']}")
    else:
        warn("no config.json next to the weights, using built-in defaults")

    # preprocessing numbers (HF VivitImageProcessor): shortest edge + zero centering
    pp = os.path.join(src_dir, "preprocessor_config.json")
    if os.path.isfile(pp):
        with open(pp, encoding="utf-8") as f:
            p = json.load(f)
        size = p.get("size") or {}
        if p.get("do_center_crop"):
            crop = p.get("crop_size") or {}
            if isinstance(crop, dict) and crop.get("height"):
                cfg["image_size"] = int(crop["height"])
        if isinstance(size, dict) and size.get("shortest_edge"):
            cfg["resize_shortest"] = int(size["shortest_edge"])
        if p.get("resample") is not None:
            cfg["resample"] = int(p["resample"])   # PILImageResampling (2 bilinear, 3 bicubic)
        if p.get("do_zero_centering") and not p.get("do_normalize"):
            cfg["mean"], cfg["std"] = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        elif p.get("do_normalize"):
            cfg["mean"] = [float(x) for x in p.get("image_mean", cfg["mean"])]
            cfg["std"] = [float(x) for x in p.get("image_std", cfg["std"])]
    return cfg


def write_metadata(gw, cfg, has_classifier, outtype):
    gw.add_name("vivit")
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32("vivit.block_count", cfg["n_layer"])
    gw.add_uint32("vivit.embedding_length", cfg["n_embd"])
    gw.add_uint32("vivit.feed_forward_length", cfg["n_ff"])
    gw.add_uint32("vivit.attention.head_count", cfg["n_head"])
    gw.add_uint32("vivit.num_channels", cfg["num_channels"])
    gw.add_uint32("vivit.image_size", cfg["image_size"])
    gw.add_uint32("vivit.resize_shortest", cfg["resize_shortest"])
    # HF keeps `resample` as a PILImageResampling value (2 = bilinear, 3 = bicubic)
    gw.add_uint32("vivit.resize_resample", cfg["resample"])
    gw.add_uint32("vivit.patch_size", cfg["patch"])
    gw.add_uint32("vivit.tubelet_size", cfg["tubelet"])
    gw.add_uint32("vivit.num_frames", cfg["num_frames"])
    gw.add_float32("vivit.layer_norm_epsilon", float(cfg["eps"]))
    gw.add_string("vivit.hidden_act", str(cfg["hidden_act"]))
    gw.add_bool("vivit.use_mean_pooling", False)
    gw.add_bool("vivit.has_final_norm", True)
    gw.add_bool("vivit.has_cls_token", True)
    gw.add_bool("vivit.has_classifier", has_classifier)
    gw.add_uint32("vivit.num_labels", cfg["num_labels"] if has_classifier else 0)
    if has_classifier and cfg["labels"]:
        gw.add_array("vivit.labels", cfg["labels"])
    gw.add_array("vivit.image_mean", [float(x) for x in cfg["mean"]])
    gw.add_array("vivit.image_std", [float(x) for x in cfg["std"]])


def verify(path, required):
    names = {t.name for t in gguf.GGUFReader(path).tensors}
    missing = [n for n in required if n not in names]
    if missing:
        fail(f"written GGUF is missing {missing}")
    log(f"verified {len(names)} tensors in {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the linear weights (default f16, ~350MB)")
    ap.add_argument("-o", "--out", default=None,
                    help="output file (default vivit-<outtype>.gguf)")
    ap.add_argument("-q", "--quiet", action="store_true", help="less progress output")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"vivit-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS + ("pytorch_model.bin",))
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")
    ensure_safetensors(src)

    qtype = OUTTYPES[args.outtype]
    cfg = read_config(src)
    n_patches = ((cfg["num_frames"] // cfg["tubelet"]) * (cfg["image_size"] // cfg["patch"])
                 * (cfg["image_size"] // cfg["patch"]))
    log(f"encoder: {cfg['n_layer']}L d={cfg['n_embd']} h={cfg['n_head']} ff={cfg['n_ff']} "
        f"patch={cfg['patch']} tubelet={cfg['tubelet']} frames={cfg['num_frames']} "
        f"-> {n_patches} patch tokens (+1 cls)")
    log(f"preprocess: shortest edge {cfg['resize_shortest']} -> crop {cfg['image_size']}, "
        f"resample {cfg['resample']} ({'bilinear' if cfg['resample'] == 2 else 'bicubic'}), "
        f"mean/std {cfg['mean'][0]:g}/{cfg['std'][0]:g}, eps {cfg['eps']:g}, "
        f"act {cfg['hidden_act']}")

    weights = _lib.TensorSource(src)
    log(f"{len(weights)} tensors in the checkpoint")

    has_classifier = "classifier.weight" in weights and cfg["num_labels"] > 0
    if not has_classifier:
        warn("checkpoint has no classifier head: the GGUF only exposes the pooled "
             "(cls) embedding")

    gw = gguf.GGUFWriter(out, "vivit")
    write_metadata(gw, cfg, has_classifier, args.outtype)

    def linear(name, hf_name):
        arr = weights.get(hf_name)
        if qtype is not None and arr.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, name, _lib.quantize(arr, qtype), qtype)
        else:
            _lib.add_f16(gw, name, arr)

    def f32(name, hf_name):
        gw.add_tensor(name, np.ascontiguousarray(weights.get(hf_name), dtype=np.float32))

    # ------------------------------------------------------------- embeddings
    conv = weights.get("vivit.embeddings.patch_embeddings.projection.weight")
    dim, cin, kt, kh, kw = conv.shape
    if (cin, kt, kh, kw) != (cfg["num_channels"], cfg["tubelet"], cfg["patch"], cfg["patch"]):
        fail(f"unexpected patch_embeddings shape {conv.shape}")
    w2d = conv.reshape(dim, cin * kt * kh * kw)
    if qtype is not None and w2d.shape[-1] % 256 == 0:
        _lib.add_quantized(gw, "v.patch_embd.weight", _lib.quantize(w2d, qtype), qtype)
    else:
        _lib.add_f16(gw, "v.patch_embd.weight", w2d)
    f32("v.patch_embd.bias", "vivit.embeddings.patch_embeddings.projection.bias")

    cls = weights.get("vivit.embeddings.cls_token").reshape(dim)
    gw.add_tensor("v.cls_token.weight", np.ascontiguousarray(cls, dtype=np.float32))

    # learned position table [1, n_patches + 1, dim] -> [n_patches+1, dim] (F32)
    pos = weights.get("vivit.embeddings.position_embeddings").reshape(-1, dim)
    if pos.shape[0] != n_patches + 1:
        fail(f"position table has {pos.shape[0]} rows, expected {n_patches + 1}")
    gw.add_tensor("v.pos_emb.weight", np.ascontiguousarray(pos, dtype=np.float32))

    # -------------------------------------------------------------- encoder
    p = "vivit.encoder.layer"
    for i in range(cfg["n_layer"]):
        b, h = f"v.blk.{i}", f"{p}.{i}"
        f32(f"{b}.ln1.weight", f"{h}.layernorm_before.weight")
        f32(f"{b}.ln1.bias", f"{h}.layernorm_before.bias")
        wq = weights.get(f"{h}.attention.attention.query.weight")
        wk = weights.get(f"{h}.attention.attention.key.weight")
        wv = weights.get(f"{h}.attention.attention.value.weight")
        wqkv = np.concatenate([wq, wk, wv], axis=0)
        if qtype is not None and wqkv.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, f"{b}.wqkv.weight", _lib.quantize(wqkv, qtype), qtype)
        else:
            _lib.add_f16(gw, f"{b}.wqkv.weight", wqkv)
        # unlike VideoMAE, ViViT carries a bias on all three projections
        qb = weights.get(f"{h}.attention.attention.query.bias")
        kb = weights.get(f"{h}.attention.attention.key.bias")
        vb = weights.get(f"{h}.attention.attention.value.bias")
        gw.add_tensor(f"{b}.wqkv.bias",
                      np.ascontiguousarray(np.concatenate([qb, kb, vb]), dtype=np.float32))
        linear(f"{b}.wo.weight", f"{h}.attention.output.dense.weight")
        f32(f"{b}.wo.bias", f"{h}.attention.output.dense.bias")
        f32(f"{b}.ln2.weight", f"{h}.layernorm_after.weight")
        f32(f"{b}.ln2.bias", f"{h}.layernorm_after.bias")
        linear(f"{b}.fc1.weight", f"{h}.intermediate.dense.weight")
        f32(f"{b}.fc1.bias", f"{h}.intermediate.dense.bias")
        linear(f"{b}.fc2.weight", f"{h}.output.dense.weight")
        f32(f"{b}.fc2.bias", f"{h}.output.dense.bias")
        _lib.progress(i + 1, cfg["n_layer"], "layers")

    f32("v.norm.weight", "vivit.layernorm.weight")
    f32("v.norm.bias", "vivit.layernorm.bias")

    required = ["v.patch_embd.weight", "v.pos_emb.weight", "v.cls_token.weight",
                f"v.blk.{cfg['n_layer'] - 1}.fc2.weight", "v.norm.weight"]
    if has_classifier:
        linear("v.cls.weight", "classifier.weight")
        f32("v.cls.bias", "classifier.bias")
        required.append("v.cls.weight")
        log(f"classifier: {cfg['num_labels']} labels ({cfg['labels'][0]} ... )")

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    verify(out, required)

    size = os.path.getsize(out) / 1e6
    log(f"done: {out} ({size:.0f} MB, {args.outtype})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
