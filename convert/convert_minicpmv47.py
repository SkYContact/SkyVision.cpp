"""openbmb/MiniCPM-V-4.7-* -> GGUF (text trunk on the qwen3_5 backend).

MiniCPM-V 4.7 is a SigLIP canvas tower bolted onto a **Qwen3.5 text trunk**
(``text_config.model_type == "qwen3_5_text"``: 3 gated-delta linear layers + 1
full-attention layer per group of 4, IMROPE 11/11/10 over 64 of 256 head dims,
interleaved q|gate projection). The text half therefore uses the exact tensor
names the qwen3_5 backend already binds (``blk.N.*``), and the vision half adds:

  * ``v.*``   SigLIP tower (27 layers, d=1152, patch 14, GELU-tanh, no rope)
  * ``vm.*``  window-attention merger inserted after vision layer 6 (16x mode):
              per-2x2-window attention + LN/Linear/GELU/Linear over 4*d
  * ``mm.*``  2x2 projection MLP (4*1152 -> 4*1152 -> text width)
  * ``minicpm.*``  canvas M-RoPE constants (position-table side, insert layer,
                   window kernel) and the ``<image>``/``</image>`` marker ids

Text layout notes (identical to convert_qwen3_5.py, which this file deliberately
mirrors so the two stay diffable):
  * Qwen3.5 RMSNorm applies ``x * (1 + w)`` -- every text RMSNorm weight except
    the gated ``linear_attn.norm`` gets +1 at conversion time.
  * full-attn ``q_proj`` is already per-head interleaved [q(head_dim) | gate(head_dim)].
  * ``conv1d.weight`` [conv_dim, 1, k] -> [k, conv_dim] (ggml_ssm_conv layout),
    ``A_log`` -> ``-exp(A_log)``.
  * MiniCPM 1B ships an **untied** lm_head (tie_word_embeddings=false) -> ``output.weight``.
  * the SigLIP conv weight [out, c, ky, kx] flattens to (c, ky, kx), which is the
    patch-vector order minicpm_preprocess.hpp emits, so patch_embedding is one matmul.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import gguf

import _lib
import hf_source

MODEL_ID = "openbmb/MiniCPM-V-4.7-1B"
ENV_VAR = "MINICPM_SRC"
PATTERNS = ["*.safetensors", "config.json"]

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q6_k": gguf.GGMLQuantizationType.Q6_K,
            "q4_k": gguf.GGMLQuantizationType.Q4_K}

log, warn, fail, progress = _lib.log, _lib.warn, _lib.fail, _lib.progress


def read_config(src):
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    # slice_mode / normalization live in the processor config, not config.json
    pre = {}
    pre_path = os.path.join(src, "preprocessor_config.json")
    if os.path.isfile(pre_path):
        with open(pre_path, encoding="utf-8") as f:
            pre = json.load(f)
    tc, vc = cfg.get("text_config", {}), cfg.get("vision_config", {})
    if tc.get("model_type") != "qwen3_5_text":
        fail(f"text_config.model_type is {tc.get('model_type')!r}, expected 'qwen3_5_text' "
             f"(a future MiniCPM-V on another backbone needs its own mapping here)")
    rope = tc.get("rope_parameters", {})
    head_dim = int(tc.get("head_dim", 256))
    text = {
        "n_layer": int(tc["num_hidden_layers"]),
        "n_embd": int(tc["hidden_size"]),
        "n_head": int(tc["num_attention_heads"]),
        "n_kv_head": int(tc["num_key_value_heads"]),
        "head_dim": head_dim,
        "n_ff": int(tc["intermediate_size"]),
        "n_vocab": int(tc["vocab_size"]),
        "rms_eps": float(tc.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(rope.get("rope_theta", 1e7)),
        "rope_dim": int(head_dim * float(rope.get("partial_rotary_factor",
                                                  tc.get("partial_rotary_factor", 0.25)))),
        "mrope_sections": list(rope.get("mrope_section", [11, 11, 10])) + [0],
        "full_attn_interval": int(tc.get("full_attention_interval", 4)),
        "layer_types": tc.get("layer_types"),
        # gated delta net dims
        "ssm_d_conv": int(tc.get("linear_conv_kernel_dim", 4)),
        "ssm_d_state": int(tc.get("linear_key_head_dim", 128)),
        "ssm_n_group": int(tc.get("linear_num_key_heads", 16)),
        "ssm_dt_rank": int(tc.get("linear_num_value_heads", 16)),
        "ssm_d_inner": int(tc.get("linear_num_value_heads", 16)) * int(tc.get("linear_value_head_dim", 128)),
        "tie": bool(tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False))),
    }
    vis = {
        "n_layer": int(vc["num_hidden_layers"]),
        "n_embd": int(vc["hidden_size"]),
        "n_head": int(vc["num_attention_heads"]),
        "n_ff": int(vc["intermediate_size"]),
        "patch": int(vc["patch_size"]),
        "pos_side": int(vc["image_size"]) // int(vc["patch_size"]),
        "eps": float(vc.get("layer_norm_eps", 1e-6)),
    }
    top = {
        "insert_layer": int(cfg.get("insert_layer_id", 6)),
        "window": 2,                                  # config.window_kernel_size
        "merge": int(cfg.get("merge_kernel_size", [2, 2])[0]),
        "merger_times": int(cfg.get("merger_times", 1)),
        "image_token_id": cfg.get("image_token_id"),
        "video_token_id": cfg.get("video_token_id"),
        "image_start_id": cfg.get("image_start_id"),
        "image_end_id": cfg.get("image_end_id"),
        "slice_start_id": cfg.get("slice_start_id"),
        "slice_end_id": cfg.get("slice_end_id"),
        "max_slice_nums": int(pre.get("max_slice_nums", 9)),
        "downsample": cfg.get("downsample_mode", "16x"),
    }
    if top["downsample"] != "16x":
        fail("only downsample_mode='16x' is wired (the 4x path skips the window merger); "
             "convert with the default checkpoint config")
    if top["merger_times"] != 1:
        fail(f"merger_times={top['merger_times']} is not wired (only 1 merge round)")
    return text, vis, top


def full_attn_layers(text):
    if text["layer_types"]:
        return {i for i, t in enumerate(text["layer_types"]) if t == "full_attention"}
    return {i for i in range(text["n_layer"]) if (i + 1) % text["full_attn_interval"] == 0}


def write_metadata(gw, text, vis, top, outtype):
    gw.add_uint32("general.file_type", 1 if outtype == "f16" else 7)
    gw.add_uint32("marlin.block_count", text["n_layer"])
    gw.add_uint32("marlin.embedding_length", text["n_embd"])
    gw.add_uint32("marlin.feed_forward_length", text["n_ff"])
    gw.add_uint32("marlin.attention.head_count", text["n_head"])
    gw.add_uint32("marlin.attention.head_count_kv", text["n_kv_head"])
    gw.add_uint32("marlin.attention.key_length", text["head_dim"])
    gw.add_uint32("marlin.attention.value_length", text["head_dim"])
    gw.add_uint32("marlin.vocab_size", text["n_vocab"])
    gw.add_uint32("marlin.context_length", 262144)
    gw.add_float32("marlin.attention.layer_norm_rms_epsilon", text["rms_eps"])
    gw.add_float32("marlin.rope.freq_base", text["rope_theta"])
    gw.add_uint32("marlin.rope.dimension_count", text["rope_dim"])
    for i, s in enumerate(text["mrope_sections"]):
        gw.add_uint32(f"marlin.rope.dimension_section_{i}", s)
    gw.add_bool("marlin.tie_word_embeddings", text["tie"])
    # 1 = gated-delta linear-attention layer, 0 = full attention (3:1 pattern)
    recurrent = [i not in full_attn_layers(text) for i in range(text["n_layer"])]
    gw.add_array("marlin.recurrent_layers", recurrent)
    gw.add_uint32("marlin.ssm.conv_kernel", text["ssm_d_conv"])
    gw.add_uint32("marlin.ssm.state_size", text["ssm_d_state"])
    gw.add_uint32("marlin.ssm.inner_size", text["ssm_d_inner"])
    gw.add_uint32("marlin.ssm.group_count", text["ssm_n_group"])
    gw.add_uint32("marlin.ssm.time_step_rank", text["ssm_dt_rank"])
    gw.add_bool("marlin.has_mtp", False)
    gw.add_bool("marlin.skills", False)
    gw.add_uint32("marlin.image_token_id", top["image_token_id"])
    gw.add_uint32("marlin.video_token_id", top["video_token_id"])

    # vision tower (shared key names the qwen3_5 loader already reads)
    gw.add_uint32("marlin.vision.block_count", vis["n_layer"])
    gw.add_uint32("marlin.vision.embedding_length", vis["n_embd"])
    gw.add_uint32("marlin.vision.feed_forward_length", vis["n_ff"])
    gw.add_uint32("marlin.vision.attention.head_count", vis["n_head"])
    gw.add_uint32("marlin.vision.patch_size", vis["patch"])
    gw.add_uint32("marlin.vision.temporal_patch_size", 1)      # SigLIP: single frame
    gw.add_uint32("marlin.vision.merge_size", top["merge"])
    gw.add_uint32("marlin.vision.pos_table_size", vis["pos_side"] ** 2)
    gw.add_float32("marlin.vision.layer_norm_epsilon", vis["eps"])
    gw.add_uint32("marlin.vision.out_hidden_size", text["n_embd"])

    # MiniCPM-specific: the presence of v.pos_emb.weight is what flips the backend
    # into its minicpm mode, these keys parameterize the rest of that mode
    gw.add_uint32("minicpm.vision.pos_side", vis["pos_side"])
    gw.add_uint32("minicpm.vision.insert_layer", top["insert_layer"])
    gw.add_uint32("minicpm.vision.window", top["window"])
    gw.add_uint32("minicpm.vision.merge", top["merge"])
    gw.add_uint32("minicpm.image_start_id", top["image_start_id"])
    gw.add_uint32("minicpm.image_end_id", top["image_end_id"])
    gw.add_uint32("minicpm.slice_start_id", top["slice_start_id"])
    gw.add_uint32("minicpm.slice_end_id", top["slice_end_id"])
    gw.add_uint32("minicpm.vision.max_slice_nums", top["max_slice_nums"])
    # 16 = window merger + 2x2 projection (default); 4 = projection only. The runtime
    # can still switch per call (MINICPM_DOWNSAMPLE=4x), same as the official API.
    gw.add_uint32("minicpm.downsample", 4 if top["downsample"] == "4x" else 16)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="q8_0", choices=sorted(OUTTYPES),
                    help="quantization for the text decoder (default q8_0); the "
                         "vision tower and both mergers always stay F16")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-q", "--quiet", action="store_true")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                       patterns=PATTERNS)
    src_name = src.replace(chr(92), "/").rstrip("/").split("/")[-1]
    out = args.out or f"{src_name.lower()}-{args.outtype}.gguf"
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    qtype = OUTTYPES[args.outtype]
    text, vis, top = read_config(src)
    full = full_attn_layers(text)
    log(f"text: {text['n_layer']}L ({len(full)} full-attn) d={text['n_embd']} "
        f"vocab={text['n_vocab']} rope={text['rope_dim']}d "
        f"sections={text['mrope_sections']} theta={text['rope_theta']:g}")
    log(f"vision: {vis['n_layer']}L d={vis['n_embd']} h={vis['n_head']} ff={vis['n_ff']} "
        f"patch={vis['patch']} pos={vis['pos_side']}x{vis['pos_side']} "
        f"window merger after layer {top['insert_layer']}")

    def quantize_text(arr):
        if qtype is None or arr.shape[-1] % 256:
            return None
        if qtype == gguf.GGMLQuantizationType.Q6_K:
            import q6k
            return q6k.quantize_q6_k(np.ascontiguousarray(arr, dtype=np.float32))
        return _lib.quantize(arr, qtype)

    weights = _lib.TensorSource(src)
    log(f"{len(weights)} tensors in the checkpoint")

    gw = gguf.GGUFWriter(out, "minicpmv47")
    gw.add_name(src_name)
    write_metadata(gw, text, vis, top, args.outtype)

    stats = {"quantized": 0, "f16": 0, "f32": 0}

    def add_tensor(name, hf_name, quantize=False, plus_one=False, dtype=None):
        arr = weights.get(hf_name)
        if plus_one:
            arr = arr + np.float32(1.0)
        if dtype is not None:
            gw.add_tensor(name, np.ascontiguousarray(arr, dtype=dtype))
            stats["f32"] += 1
            return
        blocks = quantize_text(arr) if quantize else None
        if blocks is None:
            _lib.add_f16(gw, name, arr)
            stats["f16"] += 1
        else:
            _lib.add_quantized(gw, name, blocks, qtype)
            stats["quantized"] += 1

    # ------------------------------------------------------------ text decoder
    lm = "model.language_model"
    log("text decoder: embeddings")
    emb = weights.get(f"{lm}.embed_tokens.weight")
    blocks = quantize_text(emb)
    if blocks is None:
        _lib.add_f16(gw, "token_embd.weight", emb); stats["f16"] += 1
    else:
        _lib.add_quantized(gw, "token_embd.weight", blocks, qtype); stats["quantized"] += 1
    if not text["tie"]:
        add_tensor("output.weight", "lm_head.weight", quantize=True)
    add_tensor("output_norm.weight", f"{lm}.norm.weight", plus_one=True)

    for i in range(text["n_layer"]):
        p, b = f"{lm}.layers.{i}", f"blk.{i}"
        add_tensor(f"{b}.attn_norm.weight", f"{p}.input_layernorm.weight", plus_one=True)
        add_tensor(f"{b}.attn_post_norm.weight", f"{p}.post_attention_layernorm.weight", plus_one=True)
        if i in full:
            add_tensor(f"{b}.attn_q.weight", f"{p}.self_attn.q_proj.weight", quantize=True)
            add_tensor(f"{b}.attn_k.weight", f"{p}.self_attn.k_proj.weight", quantize=True)
            add_tensor(f"{b}.attn_v.weight", f"{p}.self_attn.v_proj.weight", quantize=True)
            add_tensor(f"{b}.attn_output.weight", f"{p}.self_attn.o_proj.weight", quantize=True)
            add_tensor(f"{b}.attn_q_norm.weight", f"{p}.self_attn.q_norm.weight", plus_one=True)
            add_tensor(f"{b}.attn_k_norm.weight", f"{p}.self_attn.k_norm.weight", plus_one=True)
        else:
            add_tensor(f"{b}.la_qkv.weight", f"{p}.linear_attn.in_proj_qkv.weight", quantize=True)
            add_tensor(f"{b}.la_gate.weight", f"{p}.linear_attn.in_proj_z.weight", quantize=True)
            add_tensor(f"{b}.la_beta.weight", f"{p}.linear_attn.in_proj_b.weight", quantize=True)
            add_tensor(f"{b}.la_alpha.weight", f"{p}.linear_attn.in_proj_a.weight", quantize=True)
            add_tensor(f"{b}.la_norm.weight", f"{p}.linear_attn.norm.weight")   # gated: NO +1
            add_tensor(f"{b}.la_out.weight", f"{p}.linear_attn.out_proj.weight", quantize=True)
            gw.add_tensor(f"{b}.la_dt", np.ascontiguousarray(
                weights.get(f"{p}.linear_attn.dt_bias"), dtype=np.float32)); stats["f32"] += 1
            gw.add_tensor(f"{b}.la_a", np.ascontiguousarray(
                -np.exp(weights.get(f"{p}.linear_attn.A_log")), dtype=np.float32)); stats["f32"] += 1
            conv = weights.get(f"{p}.linear_attn.conv1d.weight").squeeze(1)     # [conv_dim, k]
            gw.add_tensor(f"{b}.la_conv1d", np.ascontiguousarray(conv, dtype=np.float32)); stats["f32"] += 1
        add_tensor(f"{b}.ffn_gate.weight", f"{p}.mlp.gate_proj.weight", quantize=True)
        add_tensor(f"{b}.ffn_up.weight", f"{p}.mlp.up_proj.weight", quantize=True)
        add_tensor(f"{b}.ffn_down.weight", f"{p}.mlp.down_proj.weight", quantize=True)
        if not args.quiet:
            progress(i, text["n_layer"], "text layer", every=6)

    # ---------------------------------------------------------- SigLIP tower
    vt = "model.vision_tower"
    log("vision tower")
    conv = weights.get(f"{vt}.embeddings.patch_embedding.weight")   # [d, 3, p, p]
    d, c, kh, kw = conv.shape
    if (c, kh, kw) != (3, vis["patch"], vis["patch"]):
        fail(f"unexpected patch_embedding shape {conv.shape}")
    _lib.add_f16(gw, "v.patch_embd.weight", conv.reshape(d, c * kh * kw))
    stats["f16"] += 1
    add_tensor("v.patch_embd.bias", f"{vt}.embeddings.patch_embedding.bias")
    gw.add_tensor("v.pos_emb.weight", np.ascontiguousarray(
        weights.get(f"{vt}.embeddings.position_embedding.weight"), dtype=np.float32))
    stats["f32"] += 1     # F32: the host folds table rows into the patch input

    def qkv(dst, prefix, qk, kk, vk, fused="wqkv"):
        """fuse q/k/v (weight + bias) into one [3*d, d] / [3*d] block, q|k|v"""
        w = np.concatenate([weights.get(prefix + qk), weights.get(prefix + kk),
                            weights.get(prefix + vk)], axis=0)
        b = np.concatenate([weights.get(prefix + qk.replace("weight", "bias")),
                            weights.get(prefix + kk.replace("weight", "bias")),
                            weights.get(prefix + vk.replace("weight", "bias"))], axis=0)
        _lib.add_f16(gw, f"{dst}.{fused}.weight", w); stats["f16"] += 1
        _lib.add_f16(gw, f"{dst}.{fused}.bias", b); stats["f16"] += 1
        _lib.add_f16(gw, dst + ".wo.weight", weights.get(prefix + "out_proj.weight")); stats["f16"] += 1
        _lib.add_f16(gw, dst + ".wo.bias", weights.get(prefix + "out_proj.bias")); stats["f16"] += 1

    for i in range(vis["n_layer"]):
        p, b = f"{vt}.encoder.layers.{i}", f"v.blk.{i}"
        add_tensor(f"{b}.norm0.weight", f"{p}.layer_norm1.weight")
        add_tensor(f"{b}.norm0.bias", f"{p}.layer_norm1.bias")
        add_tensor(f"{b}.norm1.weight", f"{p}.layer_norm2.weight")
        add_tensor(f"{b}.norm1.bias", f"{p}.layer_norm2.bias")
        qkv(b, f"{p}.self_attn.", "q_proj.weight", "k_proj.weight", "v_proj.weight")
        add_tensor(f"{b}.mlp.fc0.weight", f"{p}.mlp.fc1.weight")
        add_tensor(f"{b}.mlp.fc0.bias", f"{p}.mlp.fc1.bias")
        add_tensor(f"{b}.mlp.fc1.weight", f"{p}.mlp.fc2.weight")
        add_tensor(f"{b}.mlp.fc1.bias", f"{p}.mlp.fc2.bias")
        if not args.quiet:
            progress(i, vis["n_layer"], "vision layer", every=6)

    add_tensor("v.post_norm.weight", f"{vt}.post_layernorm.weight")
    add_tensor("v.post_norm.bias", f"{vt}.post_layernorm.bias")

    # ------------------------------------------------- window-attention merger
    # window_kernel_size^2 * d flat per window -> block-diagonal-ish MLP whose
    # linear_1 is initialized block diagonal (4 x [4304, 1152] stacked)
    vm = f"{vt}.vit_merger"
    add_tensor("vm.norm1.weight", f"{vm}.layer_norm1.weight")
    add_tensor("vm.norm1.bias", f"{vm}.layer_norm1.bias")
    qkv("vm", f"{vm}.self_attn.", "q_proj.weight", "k_proj.weight", "v_proj.weight", fused="attn")
    add_tensor("vm.pre_norm.weight", f"{vm}.pre_norm.weight")
    add_tensor("vm.pre_norm.bias", f"{vm}.pre_norm.bias")
    add_tensor("vm.fc1.weight", f"{vm}.linear_1.weight")
    add_tensor("vm.fc1.bias", f"{vm}.linear_1.bias")
    add_tensor("vm.fc2.weight", f"{vm}.linear_2.weight")
    add_tensor("vm.fc2.bias", f"{vm}.linear_2.bias")

    # ------------------------------------------------- 2x2 projection MLP -> d_text
    pm = "model.merger.mlp.0"
    add_tensor("mm.norm.weight", f"{pm}.pre_norm.weight")
    add_tensor("mm.norm.bias", f"{pm}.pre_norm.bias")
    add_tensor("mm.fc1.weight", f"{pm}.linear_1.weight")
    add_tensor("mm.fc1.bias", f"{pm}.linear_1.bias")
    add_tensor("mm.fc2.weight", f"{pm}.linear_2.weight")
    add_tensor("mm.fc2.bias", f"{pm}.linear_2.bias")

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    log(f"done: {stats['quantized']} quantized, {stats['f16']} f16, {stats['f32']} f32")
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
