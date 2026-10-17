"""Convert a Qwen2.5-VL checkpoint (transformers `qwen2_5_vl`) to GGUF.

Usage::

  python convert/convert_qwen2vl.py [--src DIR] [-o OUT.gguf] [--outtype f16|q8_0|q4_0]

The checkpoint is located like the other converters (``--src``, then
``$QWEN2VL_SRC``, the HF cache, then ``models/qwen2.5-vl-3b``); download one with::

  hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir models/qwen2.5-vl-3b

Layout (``general.architecture = qwen2vl``):

  t.embed.weight        [vocab, hidden]                lm_head is tied to it
  t.blk.{i}.attn_norm.weight / ffn_norm.weight         RMSNorm
  t.blk.{i}.wq.weight   [hidden, nh*hd]  + .bq         q/k/v carry a bias (Qwen2)
  t.blk.{i}.wk.weight   [hidden, nkv*hd] + .bk
  t.blk.{i}.wv.weight   [hidden, nkv*hd] + .bv
  t.blk.{i}.wo.weight   [hidden, nh*hd]
  t.blk.{i}.w_gate/w_up/w_down                         SwiGLU MLP
  t.norm.weight                                        final RMSNorm

  v.patch_embed.weight  [1280, 3*temporal*patch*patch] Conv3d folded into a matmul
  v.blk.{i}.norm1/norm2.weight                         LayerNorm **without bias**
  v.blk.{i}.wqkv.weight [3*1280, 1280] + .b            fused, no per-head split
  v.blk.{i}.wo.weight + .bo
  v.blk.{i}.w_gate/w_up/w_down + biases                SwiGLU MLP (this tower has biases)
  v.merger.ln.weight / fc1 / fc2                       2x2 patch merger into the text width

Nothing rope-specific is stored: the text side builds section MRoPE (16/24/24 of
head_dim/2) and the vision tower builds its own 2D RoPE at run time.
"""
import argparse
import json
import os

import numpy as np
import gguf

import _lib
from _lib import log, warn, fail, progress
import hf_source

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ENV_VAR = "QWEN2VL_SRC"
PATTERNS = ("model.safetensors", "model.safetensors.index.json")

CFG = dict(n_layer=36, n_embd=2048, n_head=16, n_kv_head=2, head_dim=128, n_ff=11008,
           vocab=151936, eps=1e-6, rope_theta=1e6, mrope_section=(16, 24, 24),
           v_depth=32, v_embd=1280, v_head=16, v_ff=3420, v_patch=14, v_temporal=2,
           v_merge=2, v_out=2048, v_eps=1e-6,            v_full_blocks=(7, 15, 23, 31),
           v_window=112, image_token_id=151655, vision_start_id=151652,
           vision_end_id=151653, video_token_id=151656,
           v_mean=[0.48145466, 0.4578275, 0.40821073],
           v_std=[0.26862954, 0.26130258, 0.27577711])

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q4_0": gguf.GGMLQuantizationType.Q4_0}


def read_config(src_dir):
    cfg = dict(CFG)
    path = os.path.join(src_dir, "config.json")
    if not os.path.isfile(path):
        fail(f"no config.json in {src_dir}")
    raw = json.load(open(path, encoding="utf-8"))
    vc = raw.get("vision_config", {}) or {}
    rs = raw.get("rope_scaling", {}) or {}
    n_head = int(raw.get("num_attention_heads", cfg["n_head"]))
    cfg.update(
        n_layer=int(raw.get("num_hidden_layers", cfg["n_layer"])),
        n_embd=int(raw.get("hidden_size", cfg["n_embd"])),
        n_head=n_head,
        n_kv_head=int(raw.get("num_key_value_heads", cfg["n_kv_head"])),
        head_dim=int(raw.get("head_dim") or cfg["n_embd"] // n_head),
        n_ff=int(raw.get("intermediate_size", cfg["n_ff"])),
        vocab=int(raw.get("vocab_size", cfg["vocab"])),
        eps=float(raw.get("rms_norm_eps", cfg["eps"])),
        rope_theta=float(raw.get("rope_theta", cfg["rope_theta"])),
        mrope_section=tuple(rs.get("mrope_section") or cfg["mrope_section"]),
        image_token_id=int(raw.get("image_token_id", cfg["image_token_id"])),
        vision_start_id=int(raw.get("vision_start_token_id", cfg["vision_start_id"])),
        vision_end_id=int(raw.get("vision_end_token_id", cfg["vision_end_id"])),
        video_token_id=int(raw.get("video_token_id", cfg["video_token_id"])),
        v_depth=int(vc.get("depth", cfg["v_depth"])),
        v_embd=int(vc.get("hidden_size", cfg["v_embd"])),
        v_head=int(vc.get("num_heads", cfg["v_head"])),
        v_ff=int(vc.get("intermediate_size", cfg["v_ff"])),
        v_patch=int(vc.get("patch_size", cfg["v_patch"])),
        v_temporal=int(vc.get("temporal_patch_size", cfg["v_temporal"])),
        v_merge=int(vc.get("spatial_merge_size", cfg["v_merge"])),
        v_out=int(vc.get("out_hidden_size", cfg["v_out"])),
        v_eps=float(vc.get("layer_norm_eps", cfg["v_eps"])),
        v_full_blocks=tuple(vc.get("fullatt_block_indexes", cfg["v_full_blocks"])),
        v_window=int(vc.get("window_size", cfg["v_window"])),
    )
    cfg["tie_word_embeddings"] = bool(raw.get("tie_word_embeddings", True))
    # image normalization lives in the processor config (CLIP statistics for 2.5-VL)
    pp = os.path.join(src_dir, "preprocessor_config.json")
    if os.path.isfile(pp):
        p = json.load(open(pp, encoding="utf-8"))
        if p.get("image_mean"):
            cfg["v_mean"] = [float(x) for x in p["image_mean"]]
        if p.get("image_std"):
            cfg["v_std"] = [float(x) for x in p["image_std"]]
    return cfg


def write_metadata(gw, cfg, outtype):
    gw.add_name("qwen2vl")
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32("qwen2vl.block_count", cfg["n_layer"])
    gw.add_uint32("qwen2vl.embedding_length", cfg["n_embd"])
    gw.add_uint32("qwen2vl.feed_forward_length", cfg["n_ff"])
    gw.add_uint32("qwen2vl.attention.head_count", cfg["n_head"])
    gw.add_uint32("qwen2vl.attention.head_count_kv", cfg["n_kv_head"])
    gw.add_uint32("qwen2vl.attention.key_length", cfg["head_dim"])
    gw.add_float32("qwen2vl.attention.layer_norm_rms_epsilon", float(cfg["eps"]))
    gw.add_float32("qwen2vl.rope.freq_base", float(cfg["rope_theta"]))
    gw.add_array("qwen2vl.rope.mrope_section", [int(x) for x in cfg["mrope_section"]])
    gw.add_uint32("qwen2vl.vocab_size", cfg["vocab"])
    gw.add_bool("qwen2vl.tie_word_embeddings", cfg["tie_word_embeddings"])
    for key in ("image_token_id", "vision_start_id", "vision_end_id", "video_token_id"):
        gw.add_uint32(f"qwen2vl.{key}", int(cfg[key]))

    gw.add_uint32("qwen2vl.vision.block_count", cfg["v_depth"])
    gw.add_uint32("qwen2vl.vision.embedding_length", cfg["v_embd"])
    gw.add_uint32("qwen2vl.vision.feed_forward_length", cfg["v_ff"])
    gw.add_uint32("qwen2vl.vision.attention.head_count", cfg["v_head"])
    gw.add_uint32("qwen2vl.vision.patch_size", cfg["v_patch"])
    gw.add_uint32("qwen2vl.vision.temporal_patch_size", cfg["v_temporal"])
    gw.add_uint32("qwen2vl.vision.spatial_merge_size", cfg["v_merge"])
    gw.add_uint32("qwen2vl.vision.out_hidden_size", cfg["v_out"])
    gw.add_uint32("qwen2vl.vision.window_size", cfg["v_window"])
    gw.add_float32("qwen2vl.vision.layer_norm_epsilon", float(cfg["v_eps"]))
    gw.add_array("qwen2vl.vision.image_mean", [float(x) for x in cfg["v_mean"]])
    gw.add_array("qwen2vl.vision.image_std", [float(x) for x in cfg["v_std"]])
    gw.add_array("qwen2vl.vision.fullatt_blocks", [int(x) for x in cfg["v_full_blocks"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the 2D weights (default f16)")
    ap.add_argument("-o", "--out", default=None, help="output file (default qwen2vl-<outtype>.gguf)")
    ap.add_argument("--vision-f32", action="store_true",
                    help="keep the vision tower in F32: its activations reach ~2e4 and "
                         "F16 weights push the merged features ~2%% off, which flips "
                         "count answers on this model")
    ap.add_argument("-q", "--quiet", action="store_true")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"qwen2vl-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    cfg = read_config(src)
    qtype = OUTTYPES[args.outtype]
    log(f"text: {cfg['n_layer']}L d={cfg['n_embd']} nh={cfg['n_head']} nkv={cfg['n_kv_head']} "
        f"hd={cfg['head_dim']} ff={cfg['n_ff']} vocab={cfg['vocab']} eps={cfg['eps']:g} "
        f"theta={cfg['rope_theta']:g} mrope={cfg['mrope_section']}")
    log(f"vision: {cfg['v_depth']}L d={cfg['v_embd']} heads={cfg['v_head']} ff={cfg['v_ff']} "
        f"patch={cfg['v_patch']} temporal={cfg['v_temporal']} merge={cfg['v_merge']} "
        f"window={cfg['v_window']} full={cfg['v_full_blocks']}")

    weights = _lib.TensorSource(src)
    log(f"{len(weights)} tensors in the checkpoint")

    gw = gguf.GGUFWriter(out, "qwen2vl")
    write_metadata(gw, cfg, args.outtype)

    def linear(name, hf_name, dim_out=None):
        arr = weights.get(hf_name)
        if qtype is not None and arr.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, name, _lib.quantize(arr, qtype), qtype)
        else:
            _lib.add_f16(gw, name, arr)

    def f32(name, hf_name):
        gw.add_tensor(name, np.ascontiguousarray(weights.get(hf_name), dtype=np.float32))

    def vlinear(name, hf_name):
        # the tower is numerically fragile: F32 by default (see --vision-f32)
        arr = weights.get(hf_name)
        if args.vision_f32:
            gw.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
        elif qtype is not None and arr.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, name, _lib.quantize(arr, qtype), qtype)
        else:
            _lib.add_f16(gw, name, arr)

    # --------------------------------------------------------------- vision
    conv = weights.get("visual.patch_embed.proj.weight")
    dim, cin, kt, kh, kw = conv.shape
    if (cin, kt, kh, kw) != (3, cfg["v_temporal"], cfg["v_patch"], cfg["v_patch"]):
        fail(f"unexpected patch_embed shape {conv.shape}")
    w2d = conv.reshape(dim, cin * kt * kh * kw)
    if args.vision_f32:
        gw.add_tensor("v.patch_embed.weight", np.ascontiguousarray(w2d, dtype=np.float32))
    elif qtype is not None and w2d.shape[-1] % 256 == 0:
        _lib.add_quantized(gw, "v.patch_embed.weight", _lib.quantize(w2d, qtype), qtype)
    else:
        _lib.add_f16(gw, "v.patch_embed.weight", w2d)

    for i in range(cfg["v_depth"]):
        h, b = f"visual.blocks.{i}", f"v.blk.{i}"
        f32(f"{b}.norm1.weight", f"{h}.norm1.weight")          # LayerNorm, no bias
        f32(f"{b}.norm2.weight", f"{h}.norm2.weight")
        vlinear(f"{b}.wqkv.weight", f"{h}.attn.qkv.weight")
        f32(f"{b}.wqkv.bias", f"{h}.attn.qkv.bias")
        vlinear(f"{b}.wo.weight", f"{h}.attn.proj.weight")
        f32(f"{b}.wo.bias", f"{h}.attn.proj.bias")
        vlinear(f"{b}.w_gate.weight", f"{h}.mlp.gate_proj.weight")
        f32(f"{b}.w_gate.bias", f"{h}.mlp.gate_proj.bias")
        vlinear(f"{b}.w_up.weight", f"{h}.mlp.up_proj.weight")
        f32(f"{b}.w_up.bias", f"{h}.mlp.up_proj.bias")
        vlinear(f"{b}.w_down.weight", f"{h}.mlp.down_proj.weight")
        f32(f"{b}.w_down.bias", f"{h}.mlp.down_proj.bias")
        progress(i + 1, cfg["v_depth"], "vision blocks")

    f32("v.merger.ln.weight", "visual.merger.ln_q.weight")
    vlinear("v.merger.fc1.weight", "visual.merger.mlp.0.weight")
    f32("v.merger.fc1.bias", "visual.merger.mlp.0.bias")
    vlinear("v.merger.fc2.weight", "visual.merger.mlp.2.weight")
    f32("v.merger.fc2.bias", "visual.merger.mlp.2.bias")

    # ----------------------------------------------------------- text trunk
    linear("t.embed.weight", "model.embed_tokens.weight")
    for i in range(cfg["n_layer"]):
        h, b = f"model.layers.{i}", f"t.blk.{i}"
        f32(f"{b}.attn_norm.weight", f"{h}.input_layernorm.weight")
        f32(f"{b}.ffn_norm.weight", f"{h}.post_attention_layernorm.weight")
        linear(f"{b}.wq.weight", f"{h}.self_attn.q_proj.weight")
        f32(f"{b}.bq", f"{h}.self_attn.q_proj.bias")
        linear(f"{b}.wk.weight", f"{h}.self_attn.k_proj.weight")
        f32(f"{b}.bk", f"{h}.self_attn.k_proj.bias")
        linear(f"{b}.wv.weight", f"{h}.self_attn.v_proj.weight")
        f32(f"{b}.bv", f"{h}.self_attn.v_proj.bias")
        linear(f"{b}.wo.weight", f"{h}.self_attn.o_proj.weight")
        linear(f"{b}.w_gate.weight", f"{h}.mlp.gate_proj.weight")
        linear(f"{b}.w_up.weight", f"{h}.mlp.up_proj.weight")
        linear(f"{b}.w_down.weight", f"{h}.mlp.down_proj.weight")
        progress(i + 1, cfg["n_layer"], "text blocks")
    f32("t.norm.weight", "model.norm.weight")

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    names = {t.name for t in gguf.GGUFReader(out).tensors}
    required = ["t.embed.weight", "t.norm.weight",
                f"t.blk.{cfg['n_layer'] - 1}.w_down.weight",
                "v.patch_embed.weight", "v.merger.fc2.weight",
                f"v.blk.{cfg['v_depth'] - 1}.w_down.weight"]
    missing = [n for n in required if n not in names]
    if missing:
        fail(f"written GGUF is missing {missing}")
    log(f"verified {len(names)} tensors, {os.path.getsize(out) / 1e9:.2f} GB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
