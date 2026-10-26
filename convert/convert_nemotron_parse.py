"""Convert NVIDIA Nemotron-3.5-Parse to GGUF.

Usage::

  python convert/convert_nemotron_parse.py [--src DIR] [-o OUT.gguf] [--outtype f16|f32|q8_0]

Checkpoint discovery follows the other converters (``--src``, then
``$NEMOTRON_PARSE_SRC``, the HF cache, then ``models/Nemotron-3.5-Parse``); fetch
one with::

  hf download nvidia/Nemotron-3.5-Parse --local-dir models/Nemotron-3.5-Parse

Layout (``general.architecture = nemotron_parse``): a C-RADIOv4-SO400M vision
tower + a 1-D/2-D conv neck + a 10 layer mBART-style causal decoder that consumes
the visual features as a causally-masked prefix and emits the document markup.

  v.patch_embd.weight      [1152, 768]      Linear(768 -> 1152) over 16x16 patches
  v.cls_token.weight       [1152, 10]       the 10 learned class tokens
  v.pos_embd.weight        [1152, 16384]    learned 128x128 position field (f32)
  v.blk.{i}.ln1/ln2        LayerNorm eps 1e-6 (+bias)
  v.blk.{i}.wqkv/bqkv      fused [3456, 1152] q|k|v + bias (timm Block, qkv_bias)
  v.blk.{i}.wo/bo, fc0/fc1 plain GELU(erf) MLP, hidden 4304
  v.neck.conv1.weight      [1024, 1152]     1x1 Conv1d over the feature dim
  v.neck.ln1, v.neck.ln2   LayerNorm eps 1e-6
  v.neck.conv2.weight      [1024, 4096]     (1,4)/stride 4 width merge, folded:
                                            index = k*1024 + in_channel (k fastest
                                            in torch, so the converter permutes)
  v.neck.sum_proj / ln3    Linear(2304 -> 1024) over class tokens 0,1 + LayerNorm
  t.embed.weight           [72512, 1024]    tied with lm_head (no separate tensor)
  t.emb_norm               LayerNorm applied to the concatenated prefix+text stream
  t.blk.{i}.ln1/wq..wo     pre-LN GQA decoder layer (16 q / 8 kv, head_dim 64),
                           no rope and no positional embedding at all
  t.blk.{i}.ln2/fc1/fc2    pre-LN MLP, GELU (erf), hidden 4096
  t.norm                   final LayerNorm

The decoder's only notion of order is the causal mask; `pos` in the graph exists
purely to index the KV cache write-back.
"""
import argparse
import json
import math
import os

import numpy as np
import gguf

import _lib
from _lib import log, fail, progress
import hf_source

MODEL_ID = "nvidia/Nemotron-3.5-Parse"
ENV_VAR = "NEMOTRON_PARSE_SRC"
PATTERNS = ("model.safetensors", "model.safetensors.index.json")

CFG = dict(
    # decoder
    n_layer=10, n_embd=1024, n_head=16, n_kv_head=8, head_dim=64, n_ff=4096,
    vocab=72512, eps=1e-5, max_seq=16384, class_token_start=52315,
    bos=0, eos=2, pad=1, embed_scale=math.sqrt(1024),
    # vision tower
    v_layer=27, v_embd=1152, v_head=16, v_ff=4304, v_patch=16, v_eps=1e-6,
    v_pos_source=128, v_class_tokens=10, v_summary_tokens=2,
    # neck
    neck_in=1152, neck_dim=1024, neck_merge=4,
    # preprocessing
    image_height=2048, image_width=1664,
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711],
)

OUTTYPES = {"f16": None, "f32": None,
            "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q4_0": gguf.GGMLQuantizationType.Q4_0}

# the encoder module prefix is stable in the released checkpoint, but resolve it
# by suffix so a re-upload under a different wrapper still converts
ENC_SUFFIX = "model_encoder.radio_model.model"


def read_config(src_dir, cfg):
    path = os.path.join(src_dir, "config.json")
    if not os.path.isfile(path):
        fail(f"no config.json in {src_dir}")
    raw = json.load(open(path, encoding="utf-8"))
    dec = raw.get("decoder", {}) or {}
    enc = raw.get("encoder", {}) or {}
    cfg["n_layer"] = int(dec.get("decoder_layers", cfg["n_layer"]))
    cfg["n_embd"] = int(dec.get("d_model", cfg["n_embd"]))
    cfg["n_head"] = int(dec.get("decoder_attention_heads", cfg["n_head"]))
    cfg["n_kv_head"] = int(dec.get("self_attention_kv_heads") or raw.get("self_attention_kv_heads", cfg["n_kv_head"]))
    cfg["head_dim"] = cfg["n_embd"] // cfg["n_head"]
    cfg["n_ff"] = int(dec.get("decoder_ffn_dim", cfg["n_ff"]))
    cfg["vocab"] = int(dec.get("vocab_size", cfg["vocab"]))
    cfg["max_seq"] = int(raw.get("max_sequence_length", cfg["max_seq"]))
    cfg["class_token_start"] = int(raw.get("class_token_start_idx", cfg["class_token_start"]))
    cfg["bos"] = int(raw.get("bos_token_id", cfg["bos"]))
    cfg["eos"] = int(raw.get("eos_token_id", cfg["eos"]))
    cfg["pad"] = int(raw.get("pad_token_id", cfg["pad"]))
    cfg["embed_scale"] = math.sqrt(cfg["n_embd"]) if dec.get("scale_embedding", True) else 1.0
    cfg["v_embd"] = int(raw.get("vision_spatial_dim", cfg["v_embd"]))
    cfg["v_patch"] = int(enc.get("patch_size", cfg["v_patch"]))
    cfg["v_eps"] = 1e-6
    args = enc.get("args") or {}
    cfg["v_class_tokens"] = 10 if args.get("cls_token_per_teacher") else cfg["v_class_tokens"]
    size = raw.get("image_size") or [cfg["image_height"], cfg["image_width"]]
    cfg["image_height"], cfg["image_width"] = int(size[0]), int(size[1])

    pp = os.path.join(src_dir, "preprocessor_config.json")
    if os.path.isfile(pp):
        p = json.load(open(pp, encoding="utf-8"))
        fs = p.get("final_size") or [cfg["image_height"], cfg["image_width"]]
        cfg["image_height"], cfg["image_width"] = int(fs[0]), int(fs[1])
        cfg["image_mean"] = [float(x) for x in (p.get("image_mean") or cfg["image_mean"])]
        cfg["image_std"] = [float(x) for x in (p.get("image_std") or cfg["image_std"])]
        # the processor must be the NaViT content-normalized variant - the fixed
        # 2048x1664 canvas path would need a different position geometry
        if not p.get("navit_packing", True):
            fail("this release expects navit_packing=true (fixed-canvas checkpoints "
                 "use a different position geometry)")
        if p.get("navit_coordinate_frame", "content_normalized") != "content_normalized":
            fail("only the content-normalized NaViT coordinate frame is implemented")
    return cfg


def write_metadata(gw, cfg, outtype):
    arch = "nemotron_parse"
    gw.add_name(arch)
    gw.add_file_type({
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "f32": gguf.LlamaFileType.ALL_F32,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }[outtype])
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32(f"{arch}.vocab_size", cfg["vocab"])
    gw.add_uint32(f"{arch}.embedding_length", cfg["n_embd"])
    gw.add_uint32(f"{arch}.block_count", cfg["n_layer"])
    gw.add_uint32(f"{arch}.feed_forward_length", cfg["n_ff"])
    gw.add_uint32(f"{arch}.attention.head_count", cfg["n_head"])
    gw.add_uint32(f"{arch}.attention.head_count_kv", cfg["n_kv_head"])
    gw.add_uint32(f"{arch}.attention.key_length", cfg["head_dim"])
    gw.add_float32(f"{arch}.attention.layer_norm_epsilon", float(cfg["eps"]))
    gw.add_uint32(f"{arch}.context_length", cfg["max_seq"])
    gw.add_float32(f"{arch}.embedding_scale", float(cfg["embed_scale"]))
    gw.add_uint32(f"{arch}.class_token_start_idx", cfg["class_token_start"])
    gw.add_uint32(f"{arch}.bos_token_id", cfg["bos"])
    gw.add_uint32(f"{arch}.eos_token_id", cfg["eos"])
    gw.add_uint32(f"{arch}.pad_token_id", cfg["pad"])

    gw.add_uint32(f"{arch}.vision.block_count", cfg["v_layer"])
    gw.add_uint32(f"{arch}.vision.embedding_length", cfg["v_embd"])
    gw.add_uint32(f"{arch}.vision.feed_forward_length", cfg["v_ff"])
    gw.add_uint32(f"{arch}.vision.attention.head_count", cfg["v_head"])
    gw.add_uint32(f"{arch}.vision.patch_size", cfg["v_patch"])
    gw.add_float32(f"{arch}.vision.layer_norm_epsilon", float(cfg["v_eps"]))
    gw.add_uint32(f"{arch}.vision.pos_source", cfg["v_pos_source"])
    gw.add_uint32(f"{arch}.vision.class_tokens", cfg["v_class_tokens"])
    gw.add_uint32(f"{arch}.vision.summary_tokens", cfg["v_summary_tokens"])
    gw.add_uint32(f"{arch}.vision.image_height", cfg["image_height"])
    gw.add_uint32(f"{arch}.vision.image_width", cfg["image_width"])
    gw.add_array(f"{arch}.vision.image_mean", [float(x) for x in cfg["image_mean"]])
    gw.add_array(f"{arch}.vision.image_std", [float(x) for x in cfg["image_std"]])

    gw.add_uint32(f"{arch}.vision.neck_length", cfg["neck_dim"])
    gw.add_uint32(f"{arch}.vision.neck_merge", cfg["neck_merge"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the 2D weights (default f16)")
    ap.add_argument("-o", "--out", default=None,
                    help="output file (default nemotron-parse-<outtype>.gguf)")
    ap.add_argument("-q", "--quiet", action="store_true")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"nemotron-parse-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    cfg = read_config(src, dict(CFG))
    qtype = OUTTYPES[args.outtype]
    log(f"decoder: {cfg['n_layer']}L d={cfg['n_embd']} nh={cfg['n_head']}/"
        f"{cfg['n_kv_head']} hd={cfg['head_dim']} ff={cfg['n_ff']} "
        f"vocab={cfg['vocab']} eps={cfg['eps']:g} scale={cfg['embed_scale']:g} "
        f"max_seq={cfg['max_seq']}")
    log(f"vision: {cfg['v_layer']}L d={cfg['v_embd']} heads={cfg['v_head']} "
        f"ff={cfg['v_ff']} patch={cfg['v_patch']} pos={cfg['v_pos_source']}^2 "
        f"cls={cfg['v_class_tokens']} + summary={cfg['v_summary_tokens']}")
    log(f"neck: {cfg['neck_in']} -> {cfg['neck_dim']}, width merge {cfg['neck_merge']}; "
        f"canvas {cfg['image_width']}x{cfg['image_height']} (aspect resize, 16/64 pad)")

    weights = _lib.TensorSource(src)
    names = weights.names()
    log(f"{len(names)} tensors in the checkpoint")

    # ---- resolve the encoder module prefix from the patch embedder name
    embed_name = next((n for n in names if n.endswith(".patch_generator.embedder.weight")), None)
    if embed_name is None:
        fail("no '*patch_generator.embedder.weight' in the checkpoint "
             "(wrong --src? expected the Nemotron-3.5-Parse release)")
    enc_prefix = embed_name[:-len(".patch_generator.embedder.weight")]
    log(f"encoder prefix: {enc_prefix}")
    if ENC_SUFFIX not in enc_prefix:
        log(f"warning: unexpected encoder prefix (wanted '*{ENC_SUFFIX}')")

    def have(name):
        if name not in weights:
            fail(f"checkpoint has no tensor '{name}'")
        return name

    gw = gguf.GGUFWriter(out, "nemotron_parse")
    write_metadata(gw, cfg, args.outtype)

    # ggml blocks: Q8_0 / Q4_0 quantise in 32-element rows (QK8_0 = QK4_0 = 32),
    # only the K-quants need 256. The tower is 1152-wide, so a 256 rule would keep
    # every vision matmul at f16 and leave the CPU prefill on the slow path.
    row_mult = 256 if qtype in (gguf.GGMLQuantizationType.Q6_K,) else 32
    stat = {"q": 0, "f16": 0, "f32": 0}

    def linear(name, hf_name, quant_ok=True):
        arr = weights.get(have(hf_name))
        if qtype is not None and quant_ok and arr.ndim == 2 and arr.shape[1] % row_mult == 0:
            _lib.add_quantized(gw, name, _lib.quantize(arr, qtype), qtype)
            stat["q"] += int(arr.size)
        elif args.outtype == "f32":
            gw.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
            stat["f32"] += int(arr.size)
        else:
            _lib.add_f16(gw, name, arr)
            stat["f16"] += int(arr.size)

    def f32(name, hf_name, shape=None):
        arr = np.ascontiguousarray(weights.get(have(hf_name)), dtype=np.float32)
        if shape is not None:
            arr = arr.reshape(shape)
        gw.add_tensor(name, arr)

    # ------------------------------------------------------------- vision tower
    linear("v.patch_embd.weight", f"{enc_prefix}.patch_generator.embedder.weight")
    f32("v.cls_token.weight", f"{enc_prefix}.patch_generator.cls_token.token")
    pos = np.ascontiguousarray(weights.get(have(f"{enc_prefix}.patch_generator.pos_embed")),
                               dtype=np.float32)
    if pos.ndim == 3:
        pos = pos[0]
    if pos.shape != (cfg["v_pos_source"] ** 2, cfg["v_embd"]):
        fail(f"position field shape {pos.shape} is not "
             f"({cfg['v_pos_source']}^2, {cfg['v_embd']})")
    gw.add_tensor("v.pos_embd.weight", pos)

    for i in range(cfg["v_layer"]):
        h, b = f"{enc_prefix}.blocks.{i}", f"v.blk.{i}"
        f32(f"{b}.ln1.weight", f"{h}.norm1.weight")
        f32(f"{b}.ln1.bias", f"{h}.norm1.bias")
        linear(f"{b}.wqkv.weight", f"{h}.attn.qkv.weight")
        f32(f"{b}.bqkv", f"{h}.attn.qkv.bias")
        linear(f"{b}.wo.weight", f"{h}.attn.proj.weight")
        f32(f"{b}.bo", f"{h}.attn.proj.bias")
        f32(f"{b}.ln2.weight", f"{h}.norm2.weight")
        f32(f"{b}.ln2.bias", f"{h}.norm2.bias")
        linear(f"{b}.fc0.weight", f"{h}.mlp.fc1.weight")
        f32(f"{b}.fc0.bias", f"{h}.mlp.fc1.bias")
        linear(f"{b}.fc1.weight", f"{h}.mlp.fc2.weight")
        f32(f"{b}.fc1.bias", f"{h}.mlp.fc2.bias")
        progress(i + 1, cfg["v_layer"], "vision blocks")

    # --------------------------------------------------------------------- neck
    conv1 = weights.get(have(f"encoder.conv1.weight"))          # [1024, 1152, 1]
    if conv1.shape[:2] != (cfg["neck_dim"], cfg["neck_in"]):
        fail(f"unexpected conv1 shape {conv1.shape}")
    # 1x1 Conv1d == one matmul: [out, in] (the trailing kernel dim is 1)
    w1 = np.ascontiguousarray(conv1.reshape(cfg["neck_dim"], cfg["neck_in"]), dtype=np.float32)
    gw.add_tensor("v.neck.conv1.weight", w1)
    f32("v.neck.conv1.bias", "encoder.conv1.bias")
    f32("v.neck.ln1.weight", "encoder.layer_norm1.weight")
    f32("v.neck.ln1.bias", "encoder.layer_norm1.bias")
    # conv2 [out, in, 1, k] -> [out, k, in] flattened: the graph reshapes the
    # token stream to (in*merge, tokens/merge), i.e. element order in_channel +
    # neck_in * k, so the weight must be laid out the same way
    conv2 = weights.get(have("encoder.conv2.weight"))          # [1024, 1024, 1, 4]
    k = conv2.shape[3]
    if conv2.shape[:3] != (cfg["neck_dim"], cfg["neck_dim"], 1) or k != cfg["neck_merge"]:
        fail(f"unexpected conv2 shape {conv2.shape}")
    w2 = np.ascontiguousarray(conv2.reshape(cfg["neck_dim"], cfg["neck_dim"], k)
                              .transpose(0, 2, 1).reshape(cfg["neck_dim"], k * cfg["neck_dim"]),
                              dtype=np.float32)
    gw.add_tensor("v.neck.conv2.weight", w2)
    f32("v.neck.ln2.weight", "encoder.layer_norm2.weight")
    f32("v.neck.ln2.bias", "encoder.layer_norm2.bias")
    linear("v.neck.sum_proj.weight", "encoder.sum_proj.weight")
    f32("v.neck.sum_proj.bias", "encoder.sum_proj.bias")
    f32("v.neck.ln3.weight", "encoder.layer_norm3.weight")
    f32("v.neck.ln3.bias", "encoder.layer_norm3.bias")

    # ------------------------------------------------------------------ decoder
    linear("t.embed.weight", "decoder.embed_tokens.weight")
    f32("t.emb_norm.weight", "decoder.layernorm_embedding.weight")
    f32("t.emb_norm.bias", "decoder.layernorm_embedding.bias")
    f32("t.norm.weight", "decoder.layer_norm.weight")
    f32("t.norm.bias", "decoder.layer_norm.bias")
    for i in range(cfg["n_layer"]):
        h, b = f"decoder.layers.{i}", f"t.blk.{i}"
        f32(f"{b}.ln1.weight", f"{h}.self_attn_layer_norm.weight")
        f32(f"{b}.ln1.bias", f"{h}.self_attn_layer_norm.bias")
        linear(f"{b}.wq.weight", f"{h}.self_attn.q_proj.weight")
        f32(f"{b}.bq", f"{h}.self_attn.q_proj.bias")
        linear(f"{b}.wk.weight", f"{h}.self_attn.k_proj.weight")
        f32(f"{b}.bk", f"{h}.self_attn.k_proj.bias")
        linear(f"{b}.wv.weight", f"{h}.self_attn.v_proj.weight")
        f32(f"{b}.bv", f"{h}.self_attn.v_proj.bias")
        linear(f"{b}.wo.weight", f"{h}.self_attn.out_proj.weight")
        f32(f"{b}.bo", f"{h}.self_attn.out_proj.bias")
        f32(f"{b}.ln2.weight", f"{h}.final_layer_norm.weight")
        f32(f"{b}.ln2.bias", f"{h}.final_layer_norm.bias")
        linear(f"{b}.fc1.weight", f"{h}.fc1.weight")
        f32(f"{b}.fc1.bias", f"{h}.fc1.bias")
        linear(f"{b}.fc2.weight", f"{h}.fc2.weight")
        f32(f"{b}.fc2.bias", f"{h}.fc2.bias")
        progress(i + 1, cfg["n_layer"], "decoder blocks")

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    written = {t.name: [int(x) for x in t.shape] for t in gguf.GGUFReader(out).tensors}
    # ggml shape == reversed numpy shape, so a stray 3-D Conv weight (ne0 == 1)
    # would silently break the first mul_mat: check the matmul operands' ne0
    required = {
        "v.patch_embd.weight": (3 * cfg["v_patch"] ** 2, cfg["v_embd"]),
        "v.cls_token.weight": (cfg["v_embd"], cfg["v_class_tokens"]),
        "v.pos_embd.weight": (cfg["v_embd"], cfg["v_pos_source"] ** 2),
        "v.neck.conv1.weight": (cfg["neck_in"], cfg["neck_dim"]),
        "v.neck.conv2.weight": (cfg["neck_dim"] * cfg["neck_merge"], cfg["neck_dim"]),
        # the summary is the FIRST TWO tower class tokens (2 * tower dim), not neck dims
        "v.neck.sum_proj.weight": (cfg["v_embd"] * cfg["v_summary_tokens"], cfg["neck_dim"]),
        "v.neck.ln3.weight": (cfg["neck_dim"],),
        "t.embed.weight": (cfg["n_embd"], cfg["vocab"]),
        "t.emb_norm.weight": (cfg["n_embd"],),
        "t.norm.weight": (cfg["n_embd"],),
        f"v.blk.{cfg['v_layer'] - 1}.fc1.weight": (cfg["v_ff"], cfg["v_embd"]),
        f"t.blk.{cfg['n_layer'] - 1}.fc2.weight": (cfg["n_ff"], cfg["n_embd"]),
    }
    missing = [n for n in required if n not in written]
    if missing:
        fail(f"written GGUF is missing {missing}")
    bad = {n: (written[n], want) for n, want in required.items() if tuple(written[n]) != want}
    if bad:
        fail(f"written GGUF has wrong tensor shapes: {bad}")
    log(f"precision: {stat['q'] / 1e6:.1f}M params quantised, {stat['f16'] / 1e6:.1f}M f16, "
        f"{stat['f32'] / 1e6:.1f}M f32")
    log(f"verified {len(written)} tensors, {os.path.getsize(out) / 1e6:.0f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
