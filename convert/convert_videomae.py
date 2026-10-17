"""Convert a VideoMAE checkpoint (transformers `videomae`) to GGUF.

Usage::

  python convert/convert_videomae.py [--src DIR] [-o OUT.gguf] [--outtype f16|q8_0|q4_0]

The checkpoint is located like the other converters (``--src``, then
``$VIDEOMAE_SRC``, the HF cache, then ``models/videomae-base``); download one
with::

  hf download MCG-NJU/videomae-base --local-dir models/videomae-base
  # or a fine-tune that carries a Kinetics-400 head:
  hf download MCG-NJU/videomae-base-finetuned-kinetics --local-dir models/videomae-base-ft

Produced layout (one self-contained file, ``general.architecture = videomae``)::

  v.patch_embd.weight  [dim, C*tubelet*patch*patch]   Conv3d folded into a matmul
  v.patch_embd.bias    [dim]
  v.pos_emb.weight     [n_patches, dim]                F32 - generated sin-cos table
  v.blk.{i}.ln1.{weight,bias}                          LayerNorm before attention
  v.blk.{i}.wqkv.weight [3*dim, dim]                   q | k | v fused rows
  v.blk.{i}.wqkv.bias   [3*dim]                        q_bias | 0 | v_bias
  v.blk.{i}.wo.{weight,bias}
  v.blk.{i}.ln2.{weight,bias}                          LayerNorm before the MLP
  v.blk.{i}.fc1.{weight,bias} [n_ff, dim] + GELU(erf) / fc2
  v.norm.{weight,bias}                                 final LayerNorm
  v.cls.{weight,bias}   [num_labels, dim]              only when a head exists
  v.fc_norm.{weight,bias}                              only with use_mean_pooling

Two checkpoint details that are easy to get wrong:

* VideoMAE has **no cls token** - the "pooled" vector is token 0 (the first
  spatio-temporal patch) unless ``use_mean_pooling``; the engine follows suit.
* The position table is the fixed sin-cos table from the original code
  (``get_sinusoid_encoding_table``), never a learned parameter. Keys and
  queries carry a bias (``q_bias`` / ``v_bias``), keys do not.

``MCG-NJU/videomae-base`` itself is a *pretraining* checkpoint
(``VideoMAEForPreTraining``): it has no classifier, so the converted file only
produces the pooled embedding. Fine-tunes such as
``MCG-NJU/videomae-base-finetuned-kinetics`` bring ``classifier.*`` plus the
``id2label`` table, which is written into the GGUF as ``videomae.labels``.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import gguf

import _lib
from _lib import log, warn, fail, progress
import hf_source

MODEL_ID = "MCG-NJU/videomae-base"
ENV_VAR = "VIDEOMAE_SRC"
PATTERNS = ("model.safetensors", "model.safetensors.index.json")

# defaults straight from MCG-NJU/videomae-base config.json (config overrides them)
CFG = dict(n_layer=12, n_embd=768, n_head=12, n_ff=3072, patch=16, tubelet=2,
           image_size=224, num_frames=16, num_channels=3, eps=1e-12,
           use_mean_pooling=False, qkv_bias=True, hidden_act="gelu",
           num_labels=0, labels=[])

# ImageNet statistics from the released preprocessor_config.json
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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
    cfg.update({k: v for k, v in dict(
        n_layer=raw.get("num_hidden_layers"), n_embd=raw.get("hidden_size"),
        n_head=raw.get("num_attention_heads"), n_ff=raw.get("intermediate_size"),
        patch=raw.get("patch_size"), tubelet=raw.get("tubelet_size"),
        image_size=raw.get("image_size"), num_frames=raw.get("num_frames"),
        num_channels=raw.get("num_channels"), eps=raw.get("layer_norm_eps"),
        use_mean_pooling=raw.get("use_mean_pooling"), qkv_bias=raw.get("qkv_bias"),
        hidden_act=raw.get("hidden_act"), num_labels=raw.get("num_labels"),
    ).items() if v is not None})
    cfg["hidden_act"] = str(cfg["hidden_act"]).replace("nn.GELU", "gelu")
    id2label = raw.get("id2label") or {}
    if id2label:
        pairs = sorted(((int(k), v) for k, v in id2label.items()), key=lambda p: p[0])
        cfg["labels"] = [name for _, name in pairs]
        n = len(cfg["labels"])
        if cfg["num_labels"] and cfg["num_labels"] != n:
            warn(f"num_labels {cfg['num_labels']} != len(id2label) {n}")
        cfg["num_labels"] = n
    elif cfg["num_labels"] is None:
        cfg["num_labels"] = 0
    return cfg


def sinusoid_table(n_patches, dim):
    """transformers.models.videomae.get_sinusoid_encoding_table, verbatim maths."""
    pos = np.arange(n_patches, dtype=np.float64)[:, None]
    exp = (2.0 * (np.arange(dim) // 2)) / float(dim)
    angle = pos / np.power(10000.0, exp)[None, :]
    table = np.empty_like(angle)
    table[:, 0::2] = np.sin(angle[:, 0::2])   # dim 2i
    table[:, 1::2] = np.cos(angle[:, 1::2])   # dim 2i+1
    return table.astype(np.float32)


def write_metadata(gw, cfg, has_classifier, has_final_norm, outtype):
    gw.add_architecture()
    gw.add_name("VideoMAE")
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32("videomae.block_count", cfg["n_layer"])
    gw.add_uint32("videomae.embedding_length", cfg["n_embd"])
    gw.add_uint32("videomae.feed_forward_length", cfg["n_ff"])
    gw.add_uint32("videomae.attention.head_count", cfg["n_head"])
    gw.add_uint32("videomae.num_channels", cfg["num_channels"])
    gw.add_uint32("videomae.image_size", cfg["image_size"])
    gw.add_uint32("videomae.patch_size", cfg["patch"])
    gw.add_uint32("videomae.tubelet_size", cfg["tubelet"])
    gw.add_uint32("videomae.num_frames", cfg["num_frames"])
    gw.add_float32("videomae.layer_norm_epsilon", float(cfg["eps"]))
    gw.add_bool("videomae.use_mean_pooling", bool(cfg["use_mean_pooling"]))
    gw.add_bool("videomae.has_final_norm", has_final_norm)
    gw.add_bool("videomae.has_classifier", has_classifier)
    gw.add_uint32("videomae.num_labels", cfg["num_labels"] if has_classifier else 0)
    if has_classifier and cfg["labels"]:
        gw.add_array("videomae.labels", cfg["labels"])
    gw.add_array("videomae.image_mean", [float(x) for x in IMAGENET_MEAN])
    gw.add_array("videomae.image_std", [float(x) for x in IMAGENET_STD])


def verify(path, required):
    names = {t.name for t in gguf.GGUFReader(path).tensors}
    missing = [n for n in required if n not in names]
    if missing:
        fail(f"written GGUF is missing {missing}")
    log(f"verified {len(names)} tensors in {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the linear weights (default f16; the model is "
                         "only ~94M params, so f16 costs ~190MB)")
    ap.add_argument("-o", "--out", default=None,
                    help="output file (default videomae-<outtype>.gguf)")
    ap.add_argument("-q", "--quiet", action="store_true", help="less progress output")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"videomae-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    qtype = OUTTYPES[args.outtype]
    cfg = read_config(src)
    n_patches = ((cfg["num_frames"] // cfg["tubelet"]) * (cfg["image_size"] // cfg["patch"])
                 * (cfg["image_size"] // cfg["patch"]))
    log(f"encoder: {cfg['n_layer']}L d={cfg['n_embd']} h={cfg['n_head']} ff={cfg['n_ff']} "
        f"patch={cfg['patch']} tubelet={cfg['tubelet']} frames={cfg['num_frames']} "
        f"-> {n_patches} tokens")

    weights = _lib.TensorSource(src)
    log(f"{len(weights)} tensors in the checkpoint")

    has_classifier = "classifier.weight" in weights and cfg["num_labels"] > 0
    if not has_classifier:
        warn("checkpoint has no classifier head (VideoMAEForPreTraining?): the GGUF "
             "will only expose the pooled embedding - use a *-finetuned-* checkpoint "
             "for labels")
    # use_mean_pooling fine-tunes have no final encoder LayerNorm (fc_norm instead)
    has_final_norm = "videomae.layernorm.weight" in weights
    if cfg["use_mean_pooling"] and has_final_norm:
        warn("use_mean_pooling with a final encoder LayerNorm: the engine expects "
             "fc_norm only, the encoder norm will be ignored")
    log(f"head: {'classifier' if has_classifier else 'none'}, "
        f"pooling: {'mean' if cfg['use_mean_pooling'] else 'token 0'}, "
        f"final norm: {'yes' if has_final_norm else 'no'}")

    gw = gguf.GGUFWriter(out, "videomae")
    write_metadata(gw, cfg, has_classifier, has_final_norm, args.outtype)

    def linear(name, hf_name):
        """2D linear weight: quantized per --outtype (row length must fit)."""
        arr = weights.get(hf_name)
        if qtype is not None and arr.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, name, _lib.quantize(arr, qtype), qtype)
        else:
            _lib.add_f16(gw, name, arr)

    def f32(name, hf_name):
        gw.add_tensor(name, np.ascontiguousarray(weights.get(hf_name), dtype=np.float32))

    # ------------------------------------------------------------- embeddings
    conv = weights.get("videomae.embeddings.patch_embeddings.projection.weight")
    dim, cin, kt, kh, kw = conv.shape
    if (cin, kt, kh, kw) != (cfg["num_channels"], cfg["tubelet"], cfg["patch"], cfg["patch"]):
        fail(f"unexpected patch_embeddings shape {conv.shape}")
    # Conv3d weight [dim, C, tubelet, patch, patch] -> the same matmul rows
    w2d = conv.reshape(dim, cin * kt * kh * kw)
    if qtype is not None and w2d.shape[-1] % 256 == 0:
        _lib.add_quantized(gw, "v.patch_embd.weight", _lib.quantize(w2d, qtype), qtype)
    else:
        _lib.add_f16(gw, "v.patch_embd.weight", w2d)
    f32("v.patch_embd.bias", "videomae.embeddings.patch_embeddings.projection.bias")
    # fixed sin-cos table [n_patches, dim]: gguf reverses the numpy shape, so the
    # engine sees ne = [dim, n_patches] and can add it to the token rows directly
    pos = sinusoid_table(n_patches, dim)
    gw.add_tensor("v.pos_emb.weight", np.ascontiguousarray(pos, dtype=np.float32))

    # -------------------------------------------------------------- encoder
    p = "videomae.encoder.layer"
    for i in range(cfg["n_layer"]):
        b, h = f"v.blk.{i}", f"{p}.{i}"
        f32(f"{b}.ln1.weight", f"{h}.layernorm_before.weight")
        f32(f"{b}.ln1.bias", f"{h}.layernorm_before.bias")
        # q_bias / k_bias(=0) / v_bias fused to match the q|k|v weight rows
        wq = weights.get(f"{h}.attention.attention.query.weight")
        wk = weights.get(f"{h}.attention.attention.key.weight")
        wv = weights.get(f"{h}.attention.attention.value.weight")
        wqkv = np.concatenate([wq, wk, wv], axis=0)
        if qtype is not None and wqkv.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, f"{b}.wqkv.weight", _lib.quantize(wqkv, qtype), qtype)
        else:
            _lib.add_f16(gw, f"{b}.wqkv.weight", wqkv)
        qb = (weights.get(f"{h}.attention.attention.q_bias") if cfg["qkv_bias"]
              else np.zeros(dim, dtype=np.float32))
        vb = (weights.get(f"{h}.attention.attention.v_bias") if cfg["qkv_bias"]
              else np.zeros(dim, dtype=np.float32))
        kb = np.zeros(dim, dtype=np.float32)
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
        if not args.quiet:
            progress(i, cfg["n_layer"], "encoder layer", every=4)

    if cfg["hidden_act"] != "gelu":
        warn(f"hidden_act is '{cfg['hidden_act']}', the engine always uses GELU(erf)")
    if has_final_norm:
        f32("v.norm.weight", "videomae.layernorm.weight")
        f32("v.norm.bias", "videomae.layernorm.bias")

    # ------------------------------------------------------------ heads
    if has_classifier:
        # the head stays F16: 400x768 is tiny and quantization only adds noise
        _lib.add_f16(gw, "v.cls.weight", weights.get("classifier.weight"))
        f32("v.cls.bias", "classifier.bias")
    if cfg["use_mean_pooling"] and "fc_norm.weight" in weights:
        f32("v.fc_norm.weight", "fc_norm.weight")
        f32("v.fc_norm.bias", "fc_norm.bias")

    # ------------------------------------------------------------ write + check
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    log(f"done: {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    required = ["v.patch_embd.weight", "v.pos_emb.weight", f"v.blk.{cfg['n_layer'] - 1}.fc2.weight"]
    if has_final_norm:
        required.append("v.norm.weight")
    if has_classifier:
        required.append("v.cls.weight")
    verify(out, required)
    return 0


if __name__ == "__main__":
    sys.exit(main())
