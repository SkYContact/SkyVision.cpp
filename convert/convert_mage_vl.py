"""Convert a Microsoft Mage-VL HF release (BF16 safetensors) to GGUF.

Usage::

  python convert/convert_mage_vl.py [--src DIR] [-o OUT.gguf]
                                    [--outtype f16|q8_0|q6_k|q4_0]

The checkpoint is located like the other converters (``--src``, then
``$MAGE_VL_SRC``, the HF cache, then ``models/Mage-VL``); download it with::

  modelscope download microsoft/Mage-VL --local_dir models/Mage-VL
  # or: hf download microsoft/Mage-VL --local-dir models/Mage-VL

Produced layout (one self-contained file; the engine reads it plus the
checkpoint's ``tokenizer.json``)::

  --- text decoder, Qwen3-4B-Instruct-2507: quantized per --outtype ---
  token_embd.weight                  [vocab, embd]   (UNtied: lm_head is separate)
  output.weight                      [vocab, embd]
  output_norm.weight                 [embd]
  blk.{i}.attn_norm.weight           [embd]           RMSNorm
  blk.{i}.attn_q/k/v/output.weight                    q: heads*hd, kv: kv_heads*hd
  blk.{i}.attn_q_norm.weight         [head_dim]       per-head RMSNorm (Qwen3)
  blk.{i}.attn_k_norm.weight         [head_dim]
  blk.{i}.ffn_norm.weight / ffn_gate / ffn_up / ffn_down   SwiGLU

  --- Mage-ViT (Codec-ViT) + patch merger, always F16 ---
  v.patch_embd.weight  [out=1024, c*P*P=768]   conv folded into a matmul (no bias)
  v.pre_norm.{weight,bias}              LayerNorm(1e-6) before the encoder
  v.blk.{i}.norm0/norm1.{weight,bias}   LayerNorm(1e-6)
  v.blk.{i}.wqkv/wo.{weight,bias}       fused qkv [3*1024, 1024], q|k|v blocks
  v.blk.{i}.mlp.fc0/fc1.{weight,bias}   GELU(erf) MLP
  mm.ln_q.{weight,bias}                 LayerNorm(1e-6) before the merger
  mm.fc1.{weight,bias} + GELU(erf) / mm.fc2.{weight,bias}   merger MLP [4096 -> 2560]

RoPE note: Mage-ViT uses per-dim rotary frequencies (a 4:6:6 T:H:W split where
each of the 32 frequencies is applied to ONE dim, not shared per pair), applied
with an interleaved rotate_half. ggml's rope kernels pair two dims at one
angle, so the engine builds the rotation from explicit cos/sin tables in the
graph (mage_model.cpp rope_mage) - no weight permutation is needed here, the
fused qkv rows stay in reference order.

The vision tower and the merger stay F16 (0.5B params, precision sensitive).
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

MODEL_ID = "microsoft/Mage-VL"
ENV_VAR = "MAGE_VL_SRC"
PATTERNS = ("model.safetensors.index.json", "model.safetensors")

# defaults straight from the released config.json (overridden by it when present)
TEXT_CFG = dict(n_layer=36, n_embd=2560, n_head=32, n_kv_head=8, head_dim=128,
                n_ff=9728, n_vocab=151936, rms_eps=1e-6, rope_theta=5e6, ctx=262144)
VIS_CFG = dict(n_layer=24, n_embd=1024, n_head=16, n_ff=4096, patch=16, merge=2,
               window=4, eps=1e-6, rope_theta=10000.0,
               head_dim=64,            # 1024 / 16
               freq_t=8, freq_h=12, freq_w=12)   # per-dim freq split (4:6:6 of half=32)
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
        rope_theta=tc.get("rope_parameters", {}).get("rope_theta", tc.get("rope_theta")),
        ctx=tc.get("max_position_embeddings"),
    ).items() if v is not None})
    vis = VIS_CFG.copy()
    vis.update({k: v for k, v in dict(
        n_layer=vc.get("num_hidden_layers"), n_embd=vc.get("hidden_size"),
        n_head=vc.get("num_attention_heads"), n_ff=vc.get("intermediate_size"),
        patch=vc.get("patch_size"), merge=vc.get("spatial_merge_size"),
        window=vc.get("frame_windows_size"), eps=vc.get("layer_norm_eps"),
        rope_theta=vc.get("rope_theta"),
    ).items() if v is not None})
    hd = vis["n_embd"] // vis["n_head"]
    half = hd // 2
    unit = half // 16
    vis["head_dim"] = hd
    vis["freq_t"], vis["freq_h"], vis["freq_w"] = 4 * unit, 6 * unit, 6 * unit
    return text, vis


def write_metadata(gw, text, vis, outtype):
    gw.add_architecture()
    gw.add_name("Mage-VL-4B")
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q6_k": gguf.LlamaFileType.MOSTLY_Q6_K,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
        "q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_M,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32("mage_vl.block_count", text["n_layer"])
    gw.add_uint32("mage_vl.embedding_length", text["n_embd"])
    gw.add_uint32("mage_vl.feed_forward_length", text["n_ff"])
    gw.add_uint32("mage_vl.attention.head_count", text["n_head"])
    gw.add_uint32("mage_vl.attention.head_count_kv", text["n_kv_head"])
    gw.add_uint32("mage_vl.attention.key_length", text["head_dim"])
    gw.add_uint32("mage_vl.attention.value_length", text["head_dim"])
    gw.add_uint32("mage_vl.vocab_size", text["n_vocab"])
    gw.add_uint32("mage_vl.context_length", text["ctx"])
    gw.add_float32("mage_vl.attention.layer_norm_rms_epsilon", text["rms_eps"])
    gw.add_float32("mage_vl.rope.freq_base", float(text["rope_theta"]))
    gw.add_bool("mage_vl.tie_word_embeddings", False)

    gw.add_uint32("mage_vl.vision.block_count", vis["n_layer"])
    gw.add_uint32("mage_vl.vision.embedding_length", vis["n_embd"])
    gw.add_uint32("mage_vl.vision.feed_forward_length", vis["n_ff"])
    gw.add_uint32("mage_vl.vision.attention.head_count", vis["n_head"])
    gw.add_uint32("mage_vl.vision.patch_size", vis["patch"])
    gw.add_uint32("mage_vl.vision.merge_size", vis["merge"])
    gw.add_uint32("mage_vl.vision.frame_window", vis["window"])
    gw.add_float32("mage_vl.vision.layer_norm_epsilon", vis["eps"])
    gw.add_float32("mage_vl.vision.rope_theta", float(vis["rope_theta"]))
    # per-dim rotary frequency split (dims of one half): t | h | w
    gw.add_uint32("mage_vl.vision.freq_dims_t", vis["freq_t"])
    gw.add_uint32("mage_vl.vision.freq_dims_h", vis["freq_h"])
    gw.add_uint32("mage_vl.vision.freq_dims_w", vis["freq_w"])
    # Qwen2VL-style image preprocessing (slow processor semantics)
    for i, v in enumerate((0.48145466, 0.4578275, 0.40821073)):
        gw.add_float32(f"mage_vl.vision.image_mean.{i}", v)
    for i, v in enumerate((0.26862954, 0.26130258, 0.27577711)):
        gw.add_float32(f"mage_vl.vision.image_std.{i}", v)
    gw.add_uint32("mage_vl.vision.min_pixels", 3136)
    gw.add_uint32("mage_vl.vision.max_pixels", 4000000)
    for key, value in SPECIAL.items():
        gw.add_uint32(f"mage_vl.{key}", value)


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
                         "vision tower and merger always stay F16")
    ap.add_argument("-o", "--out", default=None,
                    help="output file (default mage-vl-4b-<outtype>.gguf)")
    ap.add_argument("-q", "--quiet", action="store_true", help="less progress output")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"mage-vl-4b-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    qtype = OUTTYPES[args.outtype]
    text, vis = read_config(src)
    log(f"text: {text['n_layer']}L d={text['n_embd']} h={text['n_head']}/{text['n_kv_head']} "
        f"hd={text['head_dim']} ff={text['n_ff']} vocab={text['n_vocab']} theta={text['rope_theta']}")
    log(f"vision: {vis['n_layer']}L d={vis['n_embd']} h={vis['n_head']} ff={vis['n_ff']} "
        f"patch={vis['patch']} merge={vis['merge']} window={vis['window']} "
        f"freqs(t/h/w)={vis['freq_t']}/{vis['freq_h']}/{vis['freq_w']}")

    def quantize_text(arr):
        """(out, in) float32 -> pre-quantized blocks, or None to keep F16."""
        if qtype is None:
            return None
        if arr.shape[-1] % 256:                 # quant blocks are 256 wide
            warn(f"row length {arr.shape[-1]} is not a multiple of 256, keeping F16")
            return None
        if qtype == gguf.GGMLQuantizationType.Q6_K:
            import q6k                           # gguf-py decodes Q6_K only
            # chunked: the quantizer materializes several [rows, 256] f32 work
            # arrays, which for token_embd-sized tensors OOMs a 14GB host.
            # Preallocate the packed output and quantize 64K rows at a time.
            a = np.ascontiguousarray(arr, dtype=np.float32)
            mat = a.reshape(-1, 256)
            out = np.empty((mat.shape[0], 210), dtype=np.uint8)
            step = 65536
            for i in range(0, mat.shape[0], step):
                out[i:i + step] = q6k.quantize_q6_k(mat[i:i + step])
            return out.reshape(a.shape[:-1] + ((a.shape[-1] // 256) * 210,))
        return _lib.quantize(arr, qtype)

    weights = _lib.TensorSource(src)
    log(f"{len(weights)} tensors in the checkpoint")

    gw = gguf.GGUFWriter(out, "mage_vl")
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
    add_tensor("output.weight", "lm_head.weight", quantize=True)
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
    vt = "model.visual"
    conv = weights.get(f"{vt}.embeddings.patch_embedding.weight")   # (1024, 3, P, P)
    out_dim, channels, kh, kw = conv.shape
    if (channels, kh, kw) != (3, vis["patch"], vis["patch"]):
        fail(f"unexpected patch_embedding shape {conv.shape}")
    _lib.add_f16(gw, "v.patch_embd.weight", conv.reshape(out_dim, channels * kh * kw))
    stats["f16"] += 1
    add_tensor("v.pre_norm.weight", f"{vt}.layernorm_pre.weight")
    add_tensor("v.pre_norm.bias", f"{vt}.layernorm_pre.bias")

    vision_suffixes = ("norm0.weight", "norm0.bias", "norm1.weight", "norm1.bias",
                       "wqkv.weight", "wqkv.bias", "wo.weight", "wo.bias",
                       "mlp.fc0.weight", "mlp.fc0.bias", "mlp.fc1.weight", "mlp.fc1.bias")
    for i in range(vis["n_layer"]):
        p, b = f"{vt}.encoder.layers.{i}", f"v.blk.{i}"
        for suffix in vision_suffixes:
            if suffix.startswith("wqkv"):
                hf = f"{p}.self_attn.qkv.{suffix.split('.', 1)[1]}"
            elif suffix.startswith("wo."):
                hf = f"{p}.self_attn.proj.{suffix.split('.', 1)[1]}"
            elif suffix.startswith("mlp.fc0."):
                hf = f"{p}.mlp.fc1.{suffix.split('.', 2)[2]}"    # SiglipMLP: fc1 = in -> ffn
            elif suffix.startswith("mlp.fc1."):
                hf = f"{p}.mlp.fc2.{suffix.split('.', 2)[2]}"    # SiglipMLP: fc2 = ffn -> in
            elif suffix.startswith("norm0."):
                hf = f"{p}.layer_norm1.{suffix.split('.', 1)[1]}"
            elif suffix.startswith("norm1."):
                hf = f"{p}.layer_norm2.{suffix.split('.', 1)[1]}"
            else:
                hf = f"{p}.{suffix}"
            add_tensor(f"{b}.{suffix}", hf)
        if not args.quiet:
            progress(i, vis["n_layer"], "vision layer", every=6)

    # ---------------------------------------------------------- patch merger
    log("patch merger")
    add_tensor("mm.ln_q.weight", f"{vt}.merger.ln_q.weight")
    add_tensor("mm.ln_q.bias", f"{vt}.merger.ln_q.bias")
    add_tensor("mm.fc1.weight", f"{vt}.merger.mlp.0.weight")
    add_tensor("mm.fc1.bias", f"{vt}.merger.mlp.0.bias")
    add_tensor("mm.fc2.weight", f"{vt}.merger.mlp.2.weight")
    add_tensor("mm.fc2.bias", f"{vt}.merger.mlp.2.bias")

    # ---------------------------------------------------------- write + check
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    log(f"done: {out} ({os.path.getsize(out) / 1e9:.2f} GB, "
        f"{stats['quantized']} quantized / {stats['f16']} f16 tensors)")
    verify(out, len(weights), required=[
        "token_embd.weight", "output.weight", "output_norm.weight",
        "blk.0.attn_q.weight", f"blk.{text['n_layer'] - 1}.ffn_down.weight",
        "v.patch_embd.weight", "v.pre_norm.weight",
        f"v.blk.{vis['n_layer'] - 1}.mlp.fc1.weight",
        "mm.ln_q.weight", "mm.fc2.weight",
    ])
    return 0


if __name__ == "__main__":
    sys.exit(main())
