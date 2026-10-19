"""Convert a V-JEPA 2 checkpoint (transformers `vjepa2`) to GGUF.

Usage::

  python convert/convert_vjepa2.py [--src DIR] [-o OUT.gguf] [--outtype f16|q8_0|q4_0]

The checkpoint is located like the other converters (``--src``, then
``$VJEPA2_SRC``, the HF cache, then ``models/vjepa2-vitl-fpc64-256``); download
one with::

  hf download facebook/vjepa2-vitl-fpc64-256 --local-dir models/vjepa2-vitl-fpc64-256

Produced layout (one self-contained file, ``general.architecture = vjepa2``)::

  v.patch_embd.weight  [dim, C*tubelet*patch*patch]   Conv3d folded into a matmul
  v.patch_embd.bias    [dim]
  v.blk.{i}.ln1.{weight,bias}                          LayerNorm before attention
  v.blk.{i}.wqkv.weight [3*dim, dim]                   q | k | v fused rows
  v.blk.{i}.wqkv.bias   [3*dim]                        q | k | v biases fused
  v.blk.{i}.wo.{weight,bias}                           attention.proj
  v.blk.{i}.ln2.{weight,bias}                          LayerNorm before the MLP
  v.blk.{i}.fc1.{weight,bias} [n_ff, dim] + GELU(erf) / fc2
  v.norm.{weight,bias}                                 encoder.layernorm

What is deliberately dropped: the masked **predictor** head
(``predictor.*``, ~230MB: it only exists for self-supervised pretraining /
latent rollout) and any ``pooler.*`` / ``classifier.*`` of a fine-tune. This
converter exports the *encoder*: video -> 1024-d mean-pooled feature vector.

Three checkpoint details that are easy to get wrong:

* There is **no position table** - V-JEPA 2 uses 3-D decomposed RoPE inside
  every attention layer. The engine reconstructs the (t', h', w') token grid
  from ``frames_per_clip``/``tubelet_size``/``crop_size`` and rotates the three
  segments with one host-built cos/sin table per axis.
* Head dim 64 splits as 20 (temporal) + 20 (height) + 20 (width) + 4 untouched;
  the split rule is ``seg = 2*((head_dim//3)//2)``. It is *not* recorded in the
  GGUF because it is derivable, but the engine logs it at load time.
* ``rotate_queries_or_keys`` duplicates its cos/sin tables with
  ``repeat(1, 1, 1, 2)``, which **tiles** instead of interleaving, so a dimension
  pair (2u, 2u+1) rotates its even member on frequency 2u and its odd member on
  2u+1. Upstream keeps that on purpose ("fixing the bug would break
  compatibility with the pretrained model", vjepa2 PR #15), so the converter
  permutes every rotated q/k segment into ``[even dims | odd dims]`` and records
  ``vjepa2.rope_layout = "even-odd"``; the engine refuses other layouts instead
  of silently producing garbage.
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

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
ENV_VAR = "VJEPA2_SRC"
PATTERNS = ("model.safetensors", "model.safetensors.index.json")

# defaults straight from facebook/vjepa2-vitl-fpc64-256 config.json
CFG = dict(n_layer=24, n_embd=1024, n_head=16, n_ff=4096, patch=16, tubelet=2,
           image_size=256, num_frames=64, num_channels=3, eps=1e-6,
           mlp_ratio=4.0, use_rope=True, rope_theta=10000.0, hidden_act="gelu",
           # masked-latent predictor (world model half; not used by the encoder)
           pred_dim=384, pred_head=12, pred_layer=12, pred_n_mask=10,
           # attentive pooler + linear classifier (the SSv2 fine-tunes)
           pooler_layer=3, n_labels=0, labels=None)

# ImageNet statistics from video_preprocessor_config.json
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# preprocessing: shortest edge before the center crop + PIL resample id
PRE = {"resize_shortest": 0, "resample": 2}

OUTTYPES = {"f16": None, "f32": None,
            "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q4_0": gguf.GGMLQuantizationType.Q4_0}


def read_config(src_dir):
    cfg = CFG.copy()
    path = os.path.join(src_dir, "config.json")
    if not os.path.isfile(path):
        warn("no config.json next to the weights, using built-in defaults")
        return cfg
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    crop = raw.get("crop_size")
    cfg.update({k: v for k, v in dict(
        n_layer=raw.get("num_hidden_layers"), n_embd=raw.get("hidden_size"),
        n_head=raw.get("num_attention_heads"), patch=raw.get("patch_size"),
        tubelet=raw.get("tubelet_size"), image_size=crop,
        num_frames=raw.get("frames_per_clip"), num_channels=raw.get("in_chans"),
        eps=raw.get("layer_norm_eps"), mlp_ratio=raw.get("mlp_ratio"),
        use_rope=raw.get("use_rope"), hidden_act=raw.get("hidden_act"),
        pred_dim=raw.get("pred_hidden_size"),
        pred_head=raw.get("pred_num_attention_heads"),
        pred_layer=raw.get("pred_num_hidden_layers"),
        pred_n_mask=raw.get("pred_num_mask_tokens"),
        pooler_layer=raw.get("num_pooler_layers"),
    ).items() if v is not None})
    # the encoder MLP width is hidden_size * mlp_ratio in V-JEPA2Config
    cfg["n_ff"] = int(round(cfg["n_embd"] * float(cfg["mlp_ratio"])))
    cfg["pred_ff"] = int(round(cfg["pred_dim"] * float(raw.get("pred_mlp_ratio", 4.0))))
    # id2label is the label table the fine-tuned classifier was trained with
    id2label = raw.get("id2label") or {}
    if id2label:
        cfg["labels"] = [str(id2label[str(i)]) for i in range(len(id2label))]
        cfg["n_labels"] = len(cfg["labels"])

    pp = os.path.join(src_dir, "video_preprocessor_config.json")
    if os.path.isfile(pp):
        with open(pp, encoding="utf-8") as f:
            p = json.load(f)
        size = p.get("size") or {}
        PRE["resize_shortest"] = int(size.get("shortest_edge") or 0)
        PRE["resample"] = int(p.get("resample", 2))
        crop_size = p.get("crop_size") or {}
        if isinstance(crop_size, dict) and crop_size.get("height"):
            cfg["image_size"] = int(crop_size["height"])
        mean = p.get("image_mean")
        std = p.get("image_std")
        if mean and std:
            globals()["IMAGENET_MEAN"] = [float(x) for x in mean]
            globals()["IMAGENET_STD"] = [float(x) for x in std]
    return cfg


def write_metadata(gw, cfg, outtype, heads):
    gw.add_architecture()
    gw.add_name("V-JEPA 2")
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32("vjepa2.block_count", cfg["n_layer"])
    gw.add_uint32("vjepa2.embedding_length", cfg["n_embd"])
    gw.add_uint32("vjepa2.feed_forward_length", cfg["n_ff"])
    gw.add_uint32("vjepa2.attention.head_count", cfg["n_head"])
    gw.add_uint32("vjepa2.num_channels", cfg["num_channels"])
    gw.add_uint32("vjepa2.image_size", cfg["image_size"])
    gw.add_uint32("vjepa2.patch_size", cfg["patch"])
    gw.add_uint32("vjepa2.tubelet_size", cfg["tubelet"])
    gw.add_uint32("vjepa2.num_frames", cfg["num_frames"])
    gw.add_float32("vjepa2.layer_norm_epsilon", float(cfg["eps"]))
    gw.add_bool("vjepa2.use_rope", bool(cfg["use_rope"]))
    gw.add_float32("vjepa2.rope_theta", float(cfg["rope_theta"]))
    # the engine's rope math assumes the [pair-first | pair-second] row layout
    gw.add_string("vjepa2.rope_layout", "even-odd")
    gw.add_bool("vjepa2.has_final_norm", True)
    gw.add_bool("vjepa2.mean_pool", not heads["pooler"])
    gw.add_array("vjepa2.image_mean", [float(x) for x in IMAGENET_MEAN])
    gw.add_array("vjepa2.image_std", [float(x) for x in IMAGENET_STD])
    gw.add_uint32("vjepa2.resize_shortest", int(PRE["resize_shortest"]))
    gw.add_uint32("vjepa2.resize_resample", int(PRE["resample"]))
    gw.add_string("vjepa2.hidden_act", str(cfg["hidden_act"]))

    # masked-latent predictor (the world-model half: encoder features + mask
    # tokens -> predicted target latents). Optional: skipped with --no-predictor.
    gw.add_bool("vjepa2.has_predictor", heads["predictor"])
    if heads["predictor"]:
        gw.add_uint32("vjepa2.predictor.block_count", cfg["pred_layer"])
        gw.add_uint32("vjepa2.predictor.embedding_length", cfg["pred_dim"])
        gw.add_uint32("vjepa2.predictor.feed_forward_length", cfg["pred_ff"])
        gw.add_uint32("vjepa2.predictor.attention.head_count", cfg["pred_head"])
        gw.add_uint32("vjepa2.predictor.num_mask_tokens", cfg["pred_n_mask"])

    # attentive pooler + linear classifier (the *-ssv2 / *-diving48 fine-tunes)
    gw.add_bool("vjepa2.has_pooler", heads["pooler"])
    if heads["pooler"]:
        gw.add_uint32("vjepa2.pooler.layers", cfg["pooler_layer"])
    gw.add_bool("vjepa2.has_classifier", heads["classifier"])
    if heads["classifier"]:
        gw.add_uint32("vjepa2.num_labels", cfg["n_labels"])
        gw.add_array("vjepa2.labels", list(cfg["labels"]))


def rope_row_permutation(n_head, head_dim, seg, tail):
    """Row order the engine expects for q/k: [pair-first-half | pair-second-half].

    Meta's `rotate_queries_or_keys` (vjepa2 PR #15) duplicates the cos/sin table
    with `repeat(..., 2)`, which *tiles* the frequencies instead of interleaving
    them, so an adjacent pair (2u, 2u+1) rotates its first member with freq 2u and
    its second member with freq 2u+1. Upstream keeps that on purpose (fixing it
    breaks the pretrained weights).

    Splitting the head dim into "first half of every pair" then "second half of
    every pair" (with the unrotated tail dims tucked into the two halves) turns
    the whole thing into the generic pair rotation the shared vision hook
    (core/vit_block.hpp) already implements, just with a second angle table for
    the odd member. q and k get the same permutation, so attention scores are
    unaffected; v and the output projection are untouched.
    """
    if tail % 2:
        fail(f"head_dim {head_dim}: the {tail} unrotated trailing dims must be even")
    half = head_dim // 2
    idx = []
    for h in range(n_head):
        base = h * head_dim
        first, second = [], []
        for a in range(3):
            s = base + a * seg
            first += [s + 2 * u for u in range(seg // 2)]           # pair member 1
            second += [s + 2 * u + 1 for u in range(seg // 2)]      # pair member 2
        tail_idx = [base + 3 * seg + t for t in range(tail)]
        first += tail_idx[: tail // 2]
        second += tail_idx[tail // 2:]
        if len(first) != half or len(second) != half:
            fail(f"head_dim {head_dim}: rope split {seg}x3 + {tail} does not halve evenly")
        idx += first + second
    return np.asarray(idx, dtype=np.int64)


def rope_angle_columns(head_dim, seg):
    """(axis, freq index) of every pair-rotation table column.

    Column k rotates pair u = k % (seg/2) of rotated segment a = k / (seg/2);
    with the duplicated (tiled) frequency table the first pair member uses index
    `(2u) % (seg/2)` and the second `(2u + 1) % (seg/2)` - the second is *not*
    always the first + 1 (it wraps whenever seg/2 is odd, e.g. the 384-d
    predictor whose head_dim 32 gives seg 10). Returns (axis, freq0, freq1);
    the unrotated tail columns come back as (-1, -1, -1): angle 0, cos 1 / sin 0.
    """
    half = head_dim // 2
    per_seg = seg // 2
    cols = []
    for k in range(half):
        if k >= 3 * per_seg:
            cols.append((-1, -1, -1))
            continue
        a, u = divmod(k, per_seg)
        cols.append((a, (2 * u) % per_seg, (2 * u + 1) % per_seg))
    return cols


def verify(path, required):
    names = {t.name for t in gguf.GGUFReader(path).tensors}
    missing = [n for n in required if n not in names]
    if missing:
        fail(f"written GGUF is missing {missing}")
    log(f"verified {len(names)} tensors in {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the linear weights (default f16; the ViT-L "
                         "encoder is ~326M params, so f16 costs ~650MB)")
    ap.add_argument("-o", "--out", default=None,
                    help="output file (default vjepa2-<outtype>.gguf)")
    ap.add_argument("-q", "--quiet", action="store_true", help="less progress output")
    ap.add_argument("--no-predictor", action="store_true",
                    help="skip the masked-latent predictor (the world-model half, "
                         "~50MB at f16; the encoder is unaffected)")
    ap.add_argument("--no-pooler", action="store_true",
                    help="skip the attentive pooler and the classifier of a "
                         "fine-tune checkpoint (encode-only GGUF)")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"vjepa2-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    qtype = OUTTYPES[args.outtype]
    cfg = read_config(src)
    n_patches = ((cfg["num_frames"] // cfg["tubelet"]) * (cfg["image_size"] // cfg["patch"])
                 * (cfg["image_size"] // cfg["patch"]))
    hd = cfg["n_embd"] // cfg["n_head"]
    seg = 2 * ((hd // 3) // 2)
    tail = hd - 3 * seg
    perm = rope_row_permutation(cfg["n_head"], hd, seg, tail)
    log(f"encoder: {cfg['n_layer']}L d={cfg['n_embd']} h={cfg['n_head']} ff={cfg['n_ff']} "
        f"patch={cfg['patch']} tubelet={cfg['tubelet']} frames={cfg['num_frames']} "
        f"-> {n_patches} tokens")
    log(f"3d rope: head_dim {hd} = {seg} (t) + {seg} (h) + {seg} (w) + {tail} (none), "
        f"q/k rows permuted to [pair-first | pair-second]")
    log(f"preprocess: resize shortest edge {PRE['resize_shortest'] or '-'} "
        f"(resample {PRE['resample']}) -> center crop {cfg['image_size']}")

    raw = _lib.TensorSource(src)
    log(f"{len(raw)} tensors in the checkpoint")

    if not cfg["use_rope"]:
        fail("this checkpoint has use_rope=false (a position table instead of 3-D "
             "rope); the engine only implements the rope variant")

    # VJEPA2ForVideoClassification wraps the base model: `vjepa2.encoder.*`,
    # `vjepa2.predictor.*`, then the pooler/classifier heads. The pretraining
    # releases use the bare `encoder.` prefix, so normalize once here.
    pfx = "vjepa2." if any(n.startswith("vjepa2.encoder.") for n in raw.names()) else ""

    class _Prefixed:
        """Reads `vjepa2.<name>` when present, else the bare `<name>`.

        The fine-tune wrapper prefixes the encoder and the predictor
        (`vjepa2.encoder.*`, `vjepa2.predictor.*`) but keeps the pooler and the
        classifier at the top level (`pooler.*`, `classifier.*`).
        """

        def __init__(self, src_, prefix):
            self.src, self.prefix = src_, prefix

        def get(self, name, dtype=np.float32):
            if self.prefix + name in self.src:
                return self.src.get(self.prefix + name, dtype)
            return self.src.get(name, dtype)

        def __contains__(self, name):
            return (self.prefix + name) in self.src or name in self.src

        def names(self):
            return list(self.src.names())

    weights = _Prefixed(raw, pfx)
    if pfx:
        log("checkpoint is a fine-tune wrapper (vjepa2.* keys)")

    has_predictor = ("predictor.embeddings.predictor_embeddings.weight" in weights
                     and not args.no_predictor)
    has_pooler = "pooler.query_tokens" in weights and not args.no_pooler
    has_classifier = has_pooler and "classifier.weight" in weights
    if "pooler.query_tokens" in weights and args.no_pooler:
        warn("--no-pooler: dropping the attentive pooler and its classifier")
    if has_classifier and not cfg["labels"]:
        fail("the checkpoint has a classifier but config.json carries no id2label "
             "table; refusing to write a head whose labels would be unknown")
    heads = dict(predictor=has_predictor, pooler=has_pooler, classifier=has_classifier)
    if any(n.startswith("encoder.embeddings.register_tokens") for n in weights.names()):
        fail("checkpoint has register tokens; the engine does not implement them yet")

    gw = gguf.GGUFWriter(out, "vjepa2")
    write_metadata(gw, cfg, args.outtype, heads)

    def linear(name, hf_name):
        """2D linear weight: quantized per --outtype (row length must fit)."""
        arr = weights.get(hf_name)
        add_linears(name, arr)

    def f32(name, hf_name):
        gw.add_tensor(name, np.ascontiguousarray(weights.get(hf_name), dtype=np.float32))

    def add_linears(name, arr):
        """Store one linear weight at the precision --outtype asks for."""
        if args.outtype == "f32":
            # full-precision export: ~2x the file, but removes the only
            # non-deterministic error source (parity debugging)
            gw.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
        elif qtype is not None and arr.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, name, _lib.quantize(arr, qtype), qtype)
        else:
            _lib.add_f16(gw, name, arr)

    # ------------------------------------------------------------- embeddings
    conv = weights.get("encoder.embeddings.patch_embeddings.proj.weight")
    dim, cin, kt, kh, kw = conv.shape
    if (cin, kt, kh, kw) != (cfg["num_channels"], cfg["tubelet"], cfg["patch"], cfg["patch"]):
        fail(f"unexpected patch_embeddings shape {conv.shape}")
    # Conv3d weight [dim, C, tubelet, patch, patch] -> the same matmul rows
    w2d = conv.reshape(dim, cin * kt * kh * kw)
    add_linears("v.patch_embd.weight", w2d)
    f32("v.patch_embd.bias", "encoder.embeddings.patch_embeddings.proj.bias")

    # -------------------------------------------------------------- encoder
    p = "encoder.layer"
    for i in range(cfg["n_layer"]):
        b, h = f"v.blk.{i}", f"{p}.{i}"
        f32(f"{b}.ln1.weight", f"{h}.norm1.weight")
        f32(f"{b}.ln1.bias", f"{h}.norm1.bias")
        # q | k | v fused rows, matching the engine's [3*dim, dim] matmul. q/k go
        # through the rope layout permutation, v stays in the checkpoint order.
        wq = weights.get(f"{h}.attention.query.weight")[perm]
        wk = weights.get(f"{h}.attention.key.weight")[perm]
        wv = weights.get(f"{h}.attention.value.weight")
        wqkv = np.concatenate([wq, wk, wv], axis=0)
        add_linears(f"{b}.wqkv.weight", wqkv)
        bq = weights.get(f"{h}.attention.query.bias")[perm]
        bk = weights.get(f"{h}.attention.key.bias")[perm]
        bv = weights.get(f"{h}.attention.value.bias")
        gw.add_tensor(f"{b}.wqkv.bias",
                      np.ascontiguousarray(np.concatenate([bq, bk, bv]), dtype=np.float32))
        linear(f"{b}.wo.weight", f"{h}.attention.proj.weight")
        f32(f"{b}.wo.bias", f"{h}.attention.proj.bias")
        f32(f"{b}.ln2.weight", f"{h}.norm2.weight")
        f32(f"{b}.ln2.bias", f"{h}.norm2.bias")
        fc1 = weights.get(f"{h}.mlp.fc1.weight")
        if fc1.shape[0] != cfg["n_ff"]:
            fail(f"layer {i}: fc1 is [{fc1.shape[0]}, {fc1.shape[1]}], expected n_ff "
                 f"{cfg['n_ff']} (hidden_size * mlp_ratio)")
        linear(f"{b}.fc1.weight", f"{h}.mlp.fc1.weight")
        f32(f"{b}.fc1.bias", f"{h}.mlp.fc1.bias")
        linear(f"{b}.fc2.weight", f"{h}.mlp.fc2.weight")
        f32(f"{b}.fc2.bias", f"{h}.mlp.fc2.bias")
        if not args.quiet:
            progress(i, cfg["n_layer"], "encoder layer", every=4)

    if cfg["hidden_act"] != "gelu":
        warn(f"hidden_act is '{cfg['hidden_act']}', the engine always uses GELU(erf)")
    f32("v.norm.weight", "encoder.layernorm.weight")
    f32("v.norm.bias", "encoder.layernorm.bias")

    # -------------------------------------------------------------- predictor
    # masked-latent predictor: 12 narrow layers (384-d) that take the encoder
    # stream at the context tokens plus learned mask tokens at the target
    # positions and predict the target latents. Same block body as the encoder,
    # so it reuses every engine code path (rope included, with its own head dim).
    if has_predictor:
        pd, ph = cfg["pred_dim"], cfg["pred_head"]
        p_hd = pd // ph
        p_seg = 2 * ((p_hd // 3) // 2)
        p_tail = p_hd - 3 * p_seg
        p_perm = rope_row_permutation(ph, p_hd, p_seg, p_tail)
        pw = "predictor"
        log(f"predictor: {cfg['pred_layer']}L d={pd} h={ph} ff={cfg['pred_ff']} "
            f"mask tokens={cfg['pred_n_mask']} (rope head_dim {p_hd} -> "
            f"{p_seg}+{p_seg}+{p_seg}+{p_tail})")
        linear("p.emb.weight", f"{pw}.embeddings.predictor_embeddings.weight")
        f32("p.emb.bias", f"{pw}.embeddings.predictor_embeddings.bias")
        mt = weights.get(f"{pw}.embeddings.mask_tokens")          # [n_mask, 1, 1, d]
        if mt.shape[0] != cfg["pred_n_mask"] or mt.shape[-1] != pd:
            fail(f"mask_tokens shape {mt.shape} disagrees with the config "
                 f"({cfg['pred_n_mask']} x {pd})")
        # ggml rows are the fast axis: the engine gathers a row per target token,
        # so the tensor is [pred_dim, n_mask] == numpy (n_mask, pred_dim)
        gw.add_tensor("p.mask_tokens",
                      np.ascontiguousarray(mt.reshape(mt.shape[0], pd), dtype=np.float32))
        for i in range(cfg["pred_layer"]):
            b, h = f"p.blk.{i}", f"{pw}.layer.{i}"
            f32(f"{b}.ln1.weight", f"{h}.norm1.weight")
            f32(f"{b}.ln1.bias", f"{h}.norm1.bias")
            wq = weights.get(f"{h}.attention.query.weight")[p_perm]
            wk = weights.get(f"{h}.attention.key.weight")[p_perm]
            wv = weights.get(f"{h}.attention.value.weight")
            wqkv = np.concatenate([wq, wk, wv], axis=0)
            add_linears(f"{b}.wqkv.weight", wqkv)
            gw.add_tensor(f"{b}.wqkv.bias", np.ascontiguousarray(np.concatenate([
                weights.get(f"{h}.attention.query.bias")[p_perm],
                weights.get(f"{h}.attention.key.bias")[p_perm],
                weights.get(f"{h}.attention.value.bias")]), dtype=np.float32))
            linear(f"{b}.wo.weight", f"{h}.attention.proj.weight")
            f32(f"{b}.wo.bias", f"{h}.attention.proj.bias")
            f32(f"{b}.ln2.weight", f"{h}.norm2.weight")
            f32(f"{b}.ln2.bias", f"{h}.norm2.bias")
            linear(f"{b}.fc1.weight", f"{h}.mlp.fc1.weight")
            f32(f"{b}.fc1.bias", f"{h}.mlp.fc1.bias")
            linear(f"{b}.fc2.weight", f"{h}.mlp.fc2.weight")
            f32(f"{b}.fc2.bias", f"{h}.mlp.fc2.bias")
        f32("p.norm.weight", f"{pw}.layernorm.weight")
        f32("p.norm.bias", f"{pw}.layernorm.bias")
        linear("p.proj.weight", f"{pw}.proj.weight")
        f32("p.proj.bias", f"{pw}.proj.bias")

    # ------------------------------------------------------------ pooler + head
    # Attentive pooler (official get_vision_features uses it when present):
    # `num_pooler_layers` plain pre-norm self-attention blocks over the token
    # stream, then ONE learned query token cross-attends to it (that cross
    # attention has no output projection - upstream calls this out explicitly).
    # The fine-tuned classifier is a single linear layer on the pooled vector.
    if has_pooler:
        log(f"pooler: {cfg['pooler_layer']} self-attention layers + 1 cross-attention "
            f"query, then a {cfg['n_labels']}-way linear classifier"
            if has_classifier else
            f"pooler: {cfg['pooler_layer']} self-attention layers + 1 cross-attention query")
        gw.add_tensor("pool.query",
                      np.ascontiguousarray(
                          weights.get("pooler.query_tokens").reshape(cfg["n_embd"]),
                          dtype=np.float32))
        for i in range(cfg["pooler_layer"]):
            b, h = f"pool.self.{i}", f"pooler.self_attention_layers.{i}"
            f32(f"{b}.ln1.weight", f"{h}.layer_norm1.weight")
            f32(f"{b}.ln1.bias", f"{h}.layer_norm1.bias")
            # no rope in the pooler: q/k/v keep the checkpoint order
            wq = weights.get(f"{h}.self_attn.q_proj.weight")
            wk = weights.get(f"{h}.self_attn.k_proj.weight")
            wv = weights.get(f"{h}.self_attn.v_proj.weight")
            wqkv = np.concatenate([wq, wk, wv], axis=0)
            add_linears(f"{b}.wqkv.weight", wqkv)
            gw.add_tensor(f"{b}.wqkv.bias", np.ascontiguousarray(np.concatenate([
                weights.get(f"{h}.self_attn.q_proj.bias"),
                weights.get(f"{h}.self_attn.k_proj.bias"),
                weights.get(f"{h}.self_attn.v_proj.bias")]), dtype=np.float32))
            linear(f"{b}.wo.weight", f"{h}.self_attn.out_proj.weight")
            f32(f"{b}.wo.bias", f"{h}.self_attn.out_proj.bias")
            f32(f"{b}.ln2.weight", f"{h}.layer_norm2.weight")
            f32(f"{b}.ln2.bias", f"{h}.layer_norm2.bias")
            linear(f"{b}.fc1.weight", f"{h}.mlp.fc1.weight")
            f32(f"{b}.fc1.bias", f"{h}.mlp.fc1.bias")
            linear(f"{b}.fc2.weight", f"{h}.mlp.fc2.weight")
            f32(f"{b}.fc2.bias", f"{h}.mlp.fc2.bias")
        c, h = "pool.cross", "pooler.cross_attention_layer"
        f32(f"{c}.ln1.weight", f"{h}.layer_norm1.weight")
        f32(f"{c}.ln1.bias", f"{h}.layer_norm1.bias")
        linear(f"{c}.wq.weight", f"{h}.cross_attn.q_proj.weight")
        f32(f"{c}.wq.bias", f"{h}.cross_attn.q_proj.bias")
        wkv = np.concatenate([weights.get(f"{h}.cross_attn.k_proj.weight"),
                              weights.get(f"{h}.cross_attn.v_proj.weight")], axis=0)
        add_linears(f"{c}.wkv.weight", wkv)
        gw.add_tensor(f"{c}.wkv.bias", np.ascontiguousarray(np.concatenate([
            weights.get(f"{h}.cross_attn.k_proj.bias"),
            weights.get(f"{h}.cross_attn.v_proj.bias")]), dtype=np.float32))
        f32(f"{c}.ln2.weight", f"{h}.layer_norm2.weight")
        f32(f"{c}.ln2.bias", f"{h}.layer_norm2.bias")
        linear(f"{c}.fc1.weight", f"{h}.mlp.fc1.weight")
        f32(f"{c}.fc1.bias", f"{h}.mlp.fc1.bias")
        linear(f"{c}.fc2.weight", f"{h}.mlp.fc2.weight")
        f32(f"{c}.fc2.bias", f"{h}.mlp.fc2.bias")
    if has_classifier:
        linear("cls.weight", "classifier.weight")
        f32("cls.bias", "classifier.bias")

    # ------------------------------------------------------------ write + check
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    log(f"done: {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    required = ["v.patch_embd.weight", "v.blk.0.wqkv.weight", "v.norm.weight",
                f"v.blk.{cfg['n_layer'] - 1}.fc2.weight"]
    if has_predictor:
        required += ["p.emb.weight", "p.mask_tokens", "p.proj.weight",
                     f"p.blk.{cfg['pred_layer'] - 1}.fc2.weight"]
    if has_pooler:
        required += ["pool.query", "pool.cross.wq.weight",
                     f"pool.self.{cfg['pooler_layer'] - 1}.fc2.weight"]
    if has_classifier:
        required += ["cls.weight", "cls.bias"]
    verify(out, required)
    return 0


if __name__ == "__main__":
    sys.exit(main())
