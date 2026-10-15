"""Convert a VideoChat3-4B HF release (BF16 safetensors) to GGUF.

Usage::

  python convert/convert_videochat3.py [--src DIR] [-o OUT.gguf]
                                       [--outtype f16|q8_0|q6_k|q4_0]

The checkpoint is located like the other converters (``--src``, then
``$VIDEOCHAT3_SRC``, the HF cache, then ``models/VideoChat3-4B``); download it
with::

  hf download qqceqqq/VideoChat3-4B --local-dir models/VideoChat3-4B

Produced layout (one self-contained file; the engine reads it plus the
checkpoint's ``tokenizer.json``)::

  --- text decoder, Qwen3: quantized per --outtype ---
  token_embd.weight                  [vocab, embd]   (lm_head is tied to this)
  output_norm.weight                 [embd]
  blk.{i}.attn_norm.weight           [embd]           RMSNorm
  blk.{i}.attn_q/k/v.weight                           q: heads*hd, kv: kv_heads*hd
  blk.{i}.attn_q_norm.weight         [head_dim]       per-head RMSNorm (Qwen3)
  blk.{i}.attn_k_norm.weight         [head_dim]
  blk.{i}.attn_output.weight
  blk.{i}.ffn_norm.weight / ffn_gate / ffn_up / ffn_down   SwiGLU

  --- VideoChat3 ViT + projector, always F16 ---
  v.patch_embd.weight  [out=1152, c*kh*kw=588]   conv folded into a matmul
  v.patch_embd.bias    [1152]
  v.pos_emb.weight     [64, 64, 1152]   F32 - interpolated per image on the host
  v.pos_emb.time_weight[4, 1, 1152]     F32
  v.blk.{i}.norm0/norm1.{weight,bias}   LayerNorm
  v.blk.{i}.wqkv/wo.{weight,bias}
  v.blk.{i}.mlp.fc0/fc1.{weight,bias}   GELU(tanh) MLP
  v.final_norm.{weight,bias}
  mm.pre_norm.{weight,bias}             LayerNorm(1e-5)
  mm.linear_1.{weight,bias} + GELU(erf) / mm.linear_2.{weight,bias}

The vision tower and the projector stay F16 (0.4B params, precision sensitive),
the learned position tables stay F32 because the host interpolates them.
"""
import argparse
import json
import os
import sys

import numpy as np
import gguf

import _lib
import hf_source
from _lib import log, warn, fail, progress

MODEL_ID = "qqceqqq/VideoChat3-4B"
ENV_VAR = "VIDEOCHAT3_SRC"
PATTERNS = ("model.safetensors.index.json", "model.safetensors")

# defaults straight from the released config.json (overridden by it when present)
TEXT_CFG = dict(n_layer=36, n_embd=2560, n_head=32, n_kv_head=8, head_dim=128,
                n_ff=9728, n_vocab=151936, rms_eps=1e-6, rope_theta=5e6, ctx=262144)
VIS_CFG = dict(n_layer=27, n_embd=1152, n_head=16, n_ff=4304, patch=14, merge=2,
               temporal_merge=4, pos_h=64, pos_w=64, eps=1e-5)
SPECIAL = dict(image_token_id=151655, video_token_id=151656, vision_start_token_id=151652,
               vision_end_token_id=151653, im_start_token_id=151644, im_end_token_id=151645,
               eos_token_id=151645, pad_token_id=151643)

# quantization for the text decoder;  None == keep F16
OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q6_k": gguf.GGMLQuantizationType.Q6_K,
            "q4_0": gguf.GGMLQuantizationType.Q4_0,
            "q4_k": gguf.GGMLQuantizationType.Q4_K}


def read_config(src_dir):
    """Merge the checkpoint's config.json into the built-in defaults."""
    path = os.path.join(src_dir, "config.json")
    if not os.path.isfile(path):
        warn("no config.json next to the weights, using built-in defaults")
        return TEXT_CFG.copy(), VIS_CFG.copy()
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    tc, vc = cfg.get("text_config", {}), cfg.get("vision_config", {})
    text = TEXT_CFG.copy()
    text.update({k: v for k, v in dict(
        n_layer=tc.get("num_hidden_layers"), n_embd=tc.get("hidden_size"),
        n_head=tc.get("num_attention_heads"), n_kv_head=tc.get("num_key_value_heads"),
        head_dim=tc.get("head_dim"), n_ff=tc.get("intermediate_size"),
        n_vocab=tc.get("vocab_size"), rms_eps=tc.get("rms_norm_eps"),
        rope_theta=tc.get("rope_theta"), ctx=tc.get("max_position_embeddings"),
    ).items() if v is not None})
    vis = VIS_CFG.copy()
    vis.update({k: v for k, v in dict(
        n_layer=vc.get("num_hidden_layers"), n_embd=vc.get("hidden_size"),
        n_head=vc.get("num_attention_heads"), n_ff=vc.get("intermediate_size"),
        patch=vc.get("patch_size"), merge=(vc.get("merge_kernel_size") or [2, 2])[0],
        temporal_merge=vc.get("temporal_merge_size"),
        pos_h=vc.get("init_pos_emb_height"), pos_w=vc.get("init_pos_emb_width"),
    ).items() if v is not None})
    return text, vis


def write_metadata(gw, text, vis, outtype):
    gw.add_architecture()
    gw.add_name("VideoChat3-4B")
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q6_k": gguf.LlamaFileType.MOSTLY_Q6_K,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
        "q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_M,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32("videochat3.block_count", text["n_layer"])
    gw.add_uint32("videochat3.embedding_length", text["n_embd"])
    gw.add_uint32("videochat3.feed_forward_length", text["n_ff"])
    gw.add_uint32("videochat3.attention.head_count", text["n_head"])
    gw.add_uint32("videochat3.attention.head_count_kv", text["n_kv_head"])
    gw.add_uint32("videochat3.attention.key_length", text["head_dim"])
    gw.add_uint32("videochat3.attention.value_length", text["head_dim"])
    gw.add_uint32("videochat3.vocab_size", text["n_vocab"])
    gw.add_uint32("videochat3.context_length", text["ctx"])
    gw.add_float32("videochat3.attention.layer_norm_rms_epsilon", text["rms_eps"])
    gw.add_float32("videochat3.rope.freq_base", float(text["rope_theta"]))
    gw.add_bool("videochat3.tie_word_embeddings", True)

    gw.add_uint32("videochat3.vision.block_count", vis["n_layer"])
    gw.add_uint32("videochat3.vision.embedding_length", vis["n_embd"])
    gw.add_uint32("videochat3.vision.feed_forward_length", vis["n_ff"])
    gw.add_uint32("videochat3.vision.attention.head_count", vis["n_head"])
    gw.add_uint32("videochat3.vision.patch_size", vis["patch"])
    gw.add_uint32("videochat3.vision.merge_size", vis["merge"])
    gw.add_uint32("videochat3.vision.temporal_merge_size", vis["temporal_merge"])
    gw.add_uint32("videochat3.vision.pos_emb_height", vis["pos_h"])
    gw.add_uint32("videochat3.vision.pos_emb_width", vis["pos_w"])
    gw.add_float32("videochat3.vision.layer_norm_epsilon", vis["eps"])
    gw.add_float32("videochat3.vision.rope_theta", 10000.0)
    for key, value in SPECIAL.items():
        gw.add_uint32(f"videochat3.{key}", value)


def verify(path, expected_tensors, required):
    """Reopen the file and check the tensor table (headers only, cheap)."""
    names = {t.name for t in gguf.GGUFReader(path).tensors}
    missing = [n for n in required if n not in names]
    if missing:
        fail(f"written GGUF is missing {missing}")
    if len(names) != expected_tensors:
        warn(f"tensor count {len(names)} != expected {expected_tensors}")
    log(f"verified {len(names)} tensors in {os.path.basename(path)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="q8_0", choices=sorted(OUTTYPES),
                    help="quantization for the text decoder (default q8_0); the "
                         "vision tower and projector always stay F16")
    ap.add_argument("-o", "--out", default=None,
                    help="output file (default videochat3-4b-<outtype>.gguf)")
    ap.add_argument("-q", "--quiet", action="store_true", help="less progress output")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"videochat3-4b-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    qtype = OUTTYPES[args.outtype]
    text, vis = read_config(src)
    log(f"text: {text['n_layer']}L d={text['n_embd']} h={text['n_head']}/{text['n_kv_head']} "
        f"hd={text['head_dim']} ff={text['n_ff']} vocab={text['n_vocab']} theta={text['rope_theta']}")
    log(f"vision: {vis['n_layer']}L d={vis['n_embd']} h={vis['n_head']} ff={vis['n_ff']} "
        f"patch={vis['patch']} merge={vis['merge']} tmerge={vis['temporal_merge']} "
        f"pos={vis['pos_h']}x{vis['pos_w']}")

    def quantize_text(arr):
        """(out, in) float32 -> pre-quantized blocks, or None to keep F16."""
        if qtype is None:
            return None
        if arr.shape[-1] % 256:                 # quant blocks are 256 wide
            warn(f"row length {arr.shape[-1]} is not a multiple of 256, keeping F16")
            return None
        if qtype == gguf.GGMLQuantizationType.Q6_K:
            import q6k                           # gguf-py decodes Q6_K only
            return q6k.quantize_q6_k(np.ascontiguousarray(arr, dtype=np.float32))
        return _lib.quantize(arr, qtype)

    weights = _lib.TensorSource(src)
    log(f"{len(weights)} tensors in the checkpoint")

    gw = gguf.GGUFWriter(out, "videochat3")
    write_metadata(gw, text, vis, args.outtype)

    stats = {"quantized": 0, "f16": 0}

    def add_tensor(name, hf_name, quantize=False):
        """Read once, then store quantized (text linears) or F16."""
        arr = weights.get(hf_name)
        blocks = quantize_text(arr) if quantize else None
        if blocks is None:
            _lib.add_f16(gw, name, arr)
            stats["f16"] += 1
        else:
            _lib.add_quantized(gw, name, blocks, qtype)
            stats["quantized"] += 1

    # ---------------------------------------------------------- text decoder
    lm = "model.language_model"
    log("text decoder: embeddings")
    add_tensor("token_embd.weight", f"{lm}.embed_tokens.weight", quantize=True)
    add_tensor("output_norm.weight", f"{lm}.norm.weight")

    for i in range(text["n_layer"]):
        p, b = f"{lm}.layers.{i}", f"blk.{i}"
        add_tensor(f"{b}.attn_norm.weight", f"{p}.input_layernorm.weight")
        add_tensor(f"{b}.attn_q.weight", f"{p}.self_attn.q_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_k.weight", f"{p}.self_attn.k_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_v.weight", f"{p}.self_attn.v_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_output.weight", f"{p}.self_attn.o_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_q_norm.weight", f"{p}.self_attn.q_norm.weight")
        add_tensor(f"{b}.attn_k_norm.weight", f"{p}.self_attn.k_norm.weight")
        add_tensor(f"{b}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight")
        add_tensor(f"{b}.ffn_gate.weight", f"{p}.mlp.gate_proj.weight", quantize=True)
        add_tensor(f"{b}.ffn_up.weight", f"{p}.mlp.up_proj.weight", quantize=True)
        add_tensor(f"{b}.ffn_down.weight", f"{p}.mlp.down_proj.weight", quantize=True)
        if not args.quiet:
            progress(i, text["n_layer"], "text layer", every=6)

    # ---------------------------------------------------------- vision tower
    vt = "model.vision_tower"
    conv = weights.get(f"{vt}.patch_embed.proj.weight")     # (1152, 3, P, P)
    out_dim, channels, kh, kw = conv.shape
    if (channels, kh, kw) != (3, vis["patch"], vis["patch"]):
        fail(f"unexpected patch_embed shape {conv.shape}")
    _lib.add_f16(gw, "v.patch_embd.weight", conv.reshape(out_dim, channels * kh * kw))
    stats["f16"] += 1
    add_tensor("v.patch_embd.bias", f"{vt}.patch_embed.proj.bias")
    # position tables stay F32: the host interpolates them per image
    for name, hf_suffix in (("v.pos_emb.weight", "pos_emb.weight"),
                            ("v.pos_emb.time_weight", "pos_emb.time_weight")):
        gw.add_tensor(name, np.ascontiguousarray(
            weights.get(f"{vt}.patch_embed.{hf_suffix}"), dtype=np.float32))

    vision_suffixes = ("norm0.weight", "norm0.bias", "norm1.weight", "norm1.bias",
                       "wqkv.weight", "wqkv.bias", "wo.weight", "wo.bias",
                       "mlp.fc0.weight", "mlp.fc0.bias", "mlp.fc1.weight", "mlp.fc1.bias")
    for i in range(vis["n_layer"]):
        p, b = f"{vt}.encoder.blocks.{i}", f"v.blk.{i}"
        for suffix in vision_suffixes:
            add_tensor(f"{b}.{suffix}", f"{p}.{suffix}")
        if not args.quiet:
            progress(i, vis["n_layer"], "vision layer", every=6)

    add_tensor("v.final_norm.weight", f"{vt}.encoder.final_layernorm.weight")
    add_tensor("v.final_norm.bias", f"{vt}.encoder.final_layernorm.bias")

    # ---------------------------------------------------------- projector
    pj = "model.multi_modal_projector"
    log("projector")
    for suffix in ("pre_norm.weight", "pre_norm.bias", "linear_1.weight", "linear_1.bias",
                   "linear_2.weight", "linear_2.bias"):
        add_tensor(f"mm.{suffix}", f"{pj}.{suffix}")

    # ---------------------------------------------------------- write + check
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    log(f"done: {out} ({os.path.getsize(out) / 1e9:.2f} GB, "
        f"{stats['quantized']} quantized / {stats['f16']} f16 tensors)")
    verify(out, len(weights), required=[
        "token_embd.weight", "output_norm.weight",
        "blk.0.attn_q.weight", f"blk.{text['n_layer'] - 1}.ffn_down.weight",
        "v.patch_embd.weight", "v.pos_emb.weight",
        f"v.blk.{vis['n_layer'] - 1}.mlp.fc1.weight", "mm.linear_2.weight",
    ])
    return 0


if __name__ == "__main__":
    sys.exit(main())
