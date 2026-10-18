"""Convert a SigLIP checkpoint (transformers `siglip`) to GGUF.

Usage::

  python convert/convert_siglip.py [--src DIR] [-o OUT.gguf] [--outtype f16|q8_0|q4_0]

The checkpoint is located like the other converters (``--src``, then
``$SIGLIP_SRC``, the HF cache, then ``models/siglip-base``); download one with::

  hf download google/siglip-base-patch16-224 --local-dir models/siglip-base

Produced layout (one self-contained file, ``general.architecture = siglip``)::

  v.patch_embd.weight  [dim, C*patch*patch]      Conv2d folded into a matmul
  v.patch_embd.bias    [dim]
  v.pos_emb.weight     [n_patches, dim]          learned, written F32
  v.blk.{i}.ln1.{weight,bias}                    LayerNorm before attention
  v.blk.{i}.wqkv.weight [3*dim, dim]             q | k | v fused rows
  v.blk.{i}.wqkv.bias   [3*dim]
  v.blk.{i}.wo.{weight,bias}
  v.blk.{i}.ln2.{weight,bias}                    LayerNorm before the MLP
  v.blk.{i}.fc1.{weight,bias} [n_ff, dim] gelu_pytorch_tanh / fc2
  v.norm.{weight,bias}                           post_layernorm
  v.head.probe         [dim]                     attention-pooling head
  v.head.wqkv.{weight,bias} [3*dim, dim]         torch.nn.MultiheadAttention in_proj
  v.head.wo.{weight,bias}                        MultiheadAttention out_proj
  v.head.ln.{weight,bias}                        head layernorm
  v.head.fc1.{weight,bias} / v.head.fc2.{weight,bias}   head MLP
  siglip.logit_scale / siglip.logit_bias         retrieval logits scalars (F32)

Round-1 scope notes:

* The image tower + pooling head is everything `SiglipModel.get_image_features`
  returns: in this release lineage there is **no** projection linear - the
  model card's image features ARE the pooler output (transformers >= 4.44
  removed the no-op projections; this checkpoint never had them).
* The text tower is NOT exported yet (round 2: zero-shot retrieval). The
  logit_scale / logit_bias scalars ride along because they are two F32 values
  and round 2 needs them for `logit_scale * cos + logit_bias` sigmoid scoring.
"""

import argparse
import json
import os
import sys

import numpy as np
import gguf

import _lib
from _lib import log, warn, fail, progress
import hf_source

MODEL_ID = "google/siglip-base-patch16-224"
ENV_VAR = "SIGLIP_SRC"
PATTERNS = ("model.safetensors", "model.safetensors.index.json")

# defaults straight from transformers SiglipVisionConfig (config.json overrides)
CFG = dict(n_layer=12, n_embd=768, n_head=12, n_ff=3072, patch=16,
           image_size=224, num_channels=3, eps=1e-6, hidden_act="gelu_pytorch_tanh",
           has_head=True, num_labels=0, labels=[])

# SiglipImageProcessor defaults: square resize (PIL antialiased), rescale
# 1/255, normalize (x - 0.5) / 0.5. resample is a PILImageResampling id
# (3 = BICUBIC, 2 = BILINEAR); preprocessor_config.json overrides.
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
RESAMPLE = 3

VM = "vision_model."   # checkpoint key prefix, re-detected in main()

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q4_0": gguf.GGMLQuantizationType.Q4_0}


def read_config(src_dir):
    cfg = CFG.copy()
    path = os.path.join(src_dir, "config.json")
    if not os.path.isfile(path):
        warn("no config.json next to the weights, using built-in defaults")
        return cfg
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    v = raw.get("vision_config") or {}
    cfg.update({k: val for k, val in dict(
        n_layer=v.get("num_hidden_layers"), n_embd=v.get("hidden_size"),
        n_head=v.get("num_attention_heads"), n_ff=v.get("intermediate_size"),
        patch=v.get("patch_size"), image_size=v.get("image_size"),
        num_channels=v.get("num_channels"), eps=v.get("layer_norm_eps"),
        hidden_act=v.get("hidden_act"),
    ).items() if val is not None})
    cfg["has_head"] = bool(v.get("vision_use_head", True))
    # preprocessor_config.json: resize resample id + normalization stats
    pp = os.path.join(src_dir, "preprocessor_config.json")
    global RESAMPLE
    if os.path.isfile(pp):
        with open(pp, encoding="utf-8") as f:
            pc = json.load(f)
        RESAMPLE = int(pc.get("resample", 3))
        if pc.get("image_mean"):
            MEAN[:] = [float(x) for x in pc["image_mean"]]
        if pc.get("image_std"):
            STD[:] = [float(x) for x in pc["image_std"]]
    # SiglipForImageClassification fine-tunes carry the label table at the top
    # level (videomae.labels precedent)
    id2label = raw.get("id2label") or {}
    if id2label:
        pairs = sorted(((int(k), val) for k, val in id2label.items()), key=lambda p: p[0])
        cfg["labels"] = [name for _, name in pairs]
        cfg["num_labels"] = len(cfg["labels"])
    return cfg


def write_metadata(gw, cfg, outtype):
    gw.add_architecture()
    gw.add_name("SigLIP")
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32("siglip.block_count", cfg["n_layer"])
    gw.add_uint32("siglip.embedding_length", cfg["n_embd"])
    gw.add_uint32("siglip.feed_forward_length", cfg["n_ff"])
    gw.add_uint32("siglip.attention.head_count", cfg["n_head"])
    gw.add_uint32("siglip.num_channels", cfg["num_channels"])
    gw.add_uint32("siglip.image_size", cfg["image_size"])
    gw.add_uint32("siglip.patch_size", cfg["patch"])
    gw.add_float32("siglip.layer_norm_epsilon", float(cfg["eps"]))
    gw.add_string("siglip.hidden_act", str(cfg["hidden_act"]))
    gw.add_bool("siglip.has_head", bool(cfg["has_head"]))
    gw.add_bool("siglip.has_classifier", bool(cfg["num_labels"] > 0))
    gw.add_uint32("siglip.num_labels", int(cfg["num_labels"]))
    if cfg["labels"]:
        gw.add_array("siglip.labels", cfg["labels"])
    # release lineage note: get_image_features has no projection linear here
    gw.add_bool("siglip.has_projection", False)
    gw.add_array("siglip.image_mean", [float(x) for x in MEAN])
    gw.add_array("siglip.image_std", [float(x) for x in STD])
    gw.add_uint32("siglip.resize_resample", RESAMPLE)


def verify(path, required):
    names = {t.name for t in gguf.GGUFReader(path).tensors}
    missing = [n for n in required if n not in names]
    if missing:
        fail(f"written GGUF is missing {missing}")
    log(f"verified {len(names)} tensors in {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the linear weights (default f16; the vision "
                         "tower is ~86M params, ~180MB in f16)")
    ap.add_argument("-o", "--out", default=None,
                    help="output file (default siglip-<outtype>.gguf)")
    ap.add_argument("-q", "--quiet", action="store_true", help="less progress output")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"siglip-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    qtype = OUTTYPES[args.outtype]
    cfg = read_config(src)
    grid = cfg["image_size"] // cfg["patch"]
    n_patches = grid * grid
    log(f"encoder: {cfg['n_layer']}L d={cfg['n_embd']} h={cfg['n_head']} ff={cfg['n_ff']} "
        f"patch={cfg['patch']} -> {grid}x{grid} = {n_patches} tokens, eps={cfg['eps']}, "
        f"act={cfg['hidden_act']}")

    weights = _lib.TensorSource(src)
    log(f"{len(weights)} tensors in the checkpoint")

    # SiglipForImageClassification fine-tunes nest the towers under "model."
    global VM
    VM = "model.vision_model." if "model.vision_model.embeddings.patch_embedding.weight" in weights         else "vision_model."
    cfg["has_head"] = VM + "head.attention.in_proj_weight" in weights
    has_cls = "classifier.weight" in weights
    if has_cls and not cfg["labels"]:
        n = int(weights.get("classifier.weight").shape[0])
        cfg["num_labels"] = n
        cfg["labels"] = [str(i) for i in range(n)]
    if has_cls and (not cfg["num_labels"] or len(cfg["labels"]) != cfg["num_labels"]):
        fail(f"classifier shape disagrees with id2label ({cfg['num_labels']} labels)")
    if cfg["labels"]:
        log(f"classifier: {cfg['num_labels']} labels {cfg['labels'][:6]}"
            f"{' ...' if cfg['num_labels'] > 6 else ''}")
    if "text_model.embeddings.token_embedding.weight" in weights:
        warn("checkpoint also carries the text tower; the GGUF exports the "
             "image tower + pooling head + classifier only (text tower is "
             "future work: needs the BPE text side)")

    gw = gguf.GGUFWriter(out, "siglip")
    write_metadata(gw, cfg, args.outtype)

    def linear(name, arr):
        """2D linear weight: quantized per --outtype (row length must fit)."""
        if qtype is not None and arr.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, name, _lib.quantize(arr, qtype), qtype)
        else:
            _lib.add_f16(gw, name, arr)

    def f32(name, hf_name):
        gw.add_tensor(name, np.ascontiguousarray(weights.get(hf_name), dtype=np.float32))

    # ------------------------------------------------------------- embeddings
    conv = weights.get(VM + "embeddings.patch_embedding.weight")
    dim, cin, kh, kw = conv.shape
    if (cin, kh, kw) != (cfg["num_channels"], cfg["patch"], cfg["patch"]):
        fail(f"unexpected patch_embedding shape {conv.shape}")
    # Conv2d weight [dim, C, patch, patch] -> the same matmul rows
    w2d = conv.reshape(dim, cin * kh * kw)
    if qtype is not None and w2d.shape[-1] % 256 == 0:
        _lib.add_quantized(gw, "v.patch_embd.weight", _lib.quantize(w2d, qtype), qtype)
    else:
        _lib.add_f16(gw, "v.patch_embd.weight", w2d)
    f32("v.patch_embd.bias", VM + "embeddings.patch_embedding.bias")
    # learned position table [n_patches, dim]: gguf reverses the numpy shape, so
    # the engine sees ne = [dim, n_patches] and can add it to the token columns
    f32("v.pos_emb.weight", VM + "embeddings.position_embedding.weight")

    # -------------------------------------------------------------- encoder
    p = VM + "encoder.layers"
    for i in range(cfg["n_layer"]):
        b, h = f"v.blk.{i}", f"{p}.{i}"
        f32(f"{b}.ln1.weight", f"{h}.layer_norm1.weight")
        f32(f"{b}.ln1.bias", f"{h}.layer_norm1.bias")
        # q/k/v projections fused to match the q|k|v weight rows
        wqkv = np.concatenate([
            weights.get(f"{h}.self_attn.q_proj.weight"),
            weights.get(f"{h}.self_attn.k_proj.weight"),
            weights.get(f"{h}.self_attn.v_proj.weight"),
        ], axis=0)
        if qtype is not None and wqkv.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, f"{b}.wqkv.weight", _lib.quantize(wqkv, qtype), qtype)
        else:
            _lib.add_f16(gw, f"{b}.wqkv.weight", wqkv)
        bqkv = np.concatenate([
            weights.get(f"{h}.self_attn.q_proj.bias"),
            weights.get(f"{h}.self_attn.k_proj.bias"),
            weights.get(f"{h}.self_attn.v_proj.bias"),
        ])
        gw.add_tensor(f"{b}.wqkv.bias", np.ascontiguousarray(bqkv, dtype=np.float32))
        linear_name = f"{b}.wo.weight"
        wo = weights.get(f"{h}.self_attn.out_proj.weight")
        if qtype is not None and wo.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, linear_name, _lib.quantize(wo, qtype), qtype)
        else:
            _lib.add_f16(gw, linear_name, wo)
        f32(f"{b}.wo.bias", f"{h}.self_attn.out_proj.bias")
        f32(f"{b}.ln2.weight", f"{h}.layer_norm2.weight")
        f32(f"{b}.ln2.bias", f"{h}.layer_norm2.bias")
        linear(f"{b}.fc1.weight", weights.get(f"{h}.mlp.fc1.weight"))
        f32(f"{b}.fc1.bias", f"{h}.mlp.fc1.bias")
        linear(f"{b}.fc2.weight", weights.get(f"{h}.mlp.fc2.weight"))
        f32(f"{b}.fc2.bias", f"{h}.mlp.fc2.bias")
        if not args.quiet:
            progress(i, cfg["n_layer"], "encoder layer", every=4)

    f32("v.norm.weight", VM + "post_layernorm.weight")
    f32("v.norm.bias", VM + "post_layernorm.bias")

    # --------------------------------------------------- attention-pool head
    if cfg["has_head"]:
        hp = VM + "head"
        if f"{hp}.attention.in_proj_weight" not in weights:
            fail("config says vision_use_head but the checkpoint has no head weights")
        f32("v.head.probe", f"{hp}.probe")
        linear("v.head.wqkv.weight", weights.get(f"{hp}.attention.in_proj_weight"))
        f32("v.head.wqkv.bias", f"{hp}.attention.in_proj_bias")
        wo = weights.get(f"{hp}.attention.out_proj.weight")
        if qtype is not None and wo.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, "v.head.wo.weight", _lib.quantize(wo, qtype), qtype)
        else:
            _lib.add_f16(gw, "v.head.wo.weight", wo)
        f32("v.head.wo.bias", f"{hp}.attention.out_proj.bias")
        f32("v.head.ln.weight", f"{hp}.layernorm.weight")
        f32("v.head.ln.bias", f"{hp}.layernorm.bias")
        linear("v.head.fc1.weight", weights.get(f"{hp}.mlp.fc1.weight"))
        f32("v.head.fc1.bias", f"{hp}.mlp.fc1.bias")
        linear("v.head.fc2.weight", weights.get(f"{hp}.mlp.fc2.weight"))
        f32("v.head.fc2.bias", f"{hp}.mlp.fc2.bias")

    # retrieval logits scalars (zero-shot scoring needs the text tower too);
    # classification fine-tunes often strip them
    if "logit_scale" in weights:
        f32("siglip.logit_scale", "logit_scale")
    if "logit_bias" in weights:
        f32("siglip.logit_bias", "logit_bias")

    # ------------------------------------------- classification head + labels
    if has_cls:
        _lib.add_f16(gw, "v.cls.weight", weights.get("classifier.weight"))
        f32("v.cls.bias", "classifier.bias")

    # ------------------------------------------------------------ write + check
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    log(f"done: {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    required = ["v.patch_embd.weight", "v.pos_emb.weight",
                f"v.blk.{cfg['n_layer'] - 1}.fc2.weight", "v.norm.weight"]
    if has_cls:
        required += ["v.cls.weight", "v.cls.bias"]
    if cfg["has_head"]:
        required += ["v.head.probe", "v.head.wqkv.weight", "v.head.wo.weight"]
    verify(out, required)
    return 0


if __name__ == "__main__":
    sys.exit(main())
