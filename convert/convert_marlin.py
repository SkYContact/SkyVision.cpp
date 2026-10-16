"""Convert NemoStation/Marlin-2B (HF safetensors) to GGUF for the marlin backend.

Marlin = Qwen3.5-2B hybrid text backbone (18 gated-delta linear-attention layers
+ 6 full-attention layers at 3,7,11,15,19,23) + Qwen3-VL style vision tower.

Usage::

  python convert/convert_marlin.py [--src DIR] [-o OUT.gguf]
                                   [--outtype f16|q8_0|q6_k|q4_0]

Checkpoint (ModelScope, no gating)::

  models/Marlin-2B   (NemoStation/Marlin-2B)

Layout notes (semantics verified against transformers 5.7 modeling_qwen3_5.py
and llama.cpp conversion/qwen.py; see docs/dev-notes.md "Marlin-2B"):

* Qwen3.5 RMSNorm applies ``x * (1 + w)`` -- every text RMSNorm weight except
  the gated ``linear_attn.norm`` gets +1 at conversion time.
* ``A_log`` is stored precomputed as ``-exp(A_log)``; ``dt_bias`` as-is.
* full-attn ``q_proj`` outputs per-head interleaved [q(head_dim) | gate(head_dim)];
  the engine splits the 2*head_dim-wide rows (no conversion needed).
* linear-attn: separate in_proj_qkv/z/a/b tensors, num_k_heads == num_v_heads
  (16 == 16) so the llama.cpp V-head reorder is a no-op.
* conv1d.weight [conv_dim, 1, k] -> transposed [k, conv_dim] (ggml_ssm_conv layout).
* vision tower: head_dim 64 and the native rope pair layout (h-block | w-block,
  NEOX re|im halves) is already the ggml GGML_ROPE_TYPE_VISION layout -- the
  wqkv tensors are stored unpermuted (unlike the VideoChat3 converter).
* pos_embed is a learned 48x48 table; the tower F16, the table F32 (the host
  bilinearly interpolates it per frame, ALIGN_CORNERS semantics).
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

# accepts both Marlin-2B and the official Qwen3.5 checkpoints (identical
# structure - Marlin is a plain SFT of Qwen3.5, tensor names unchanged)
MODEL_ID = "NemoStation/Marlin-2B"
ALT_MODEL_IDS = ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-7B", "Qwen/Qwen3.5-2B-Instruct", "Qwen/Qwen3.5-7B-Instruct")
ENV_VAR = "MARLIN_SRC"
PATTERNS = ("model.safetensors.index.json", "model.safetensors")

# defaults from the released config.json (overridden by it when present)
TEXT_CFG = dict(n_layer=24, n_embd=2048, n_head=8, n_kv_head=2, head_dim=256,
                n_ff=6144, n_vocab=248320, rms_eps=1e-6, rope_theta=1e7,
                rope_dim=64, mrope_sections=(11, 11, 10), ctx=262144,
                full_attn_interval=4,
                ssm_d_conv=4, ssm_d_state=128, ssm_d_inner=2048,
                ssm_n_group=16, ssm_dt_rank=16)
VIS_CFG = dict(n_layer=24, n_embd=1024, n_head=16, n_ff=4096, patch=16,
               temporal_patch=2, merge=2, pos_table=2304, eps=1e-6)
SPECIAL = dict(image_token_id=248056, video_token_id=248057,
               vision_start_token_id=248053, vision_end_token_id=248054,
               pad_token_id=248044)

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q6_k": gguf.GGMLQuantizationType.Q6_K,
            "q4_0": gguf.GGMLQuantizationType.Q4_0,
            "q4_k": gguf.GGMLQuantizationType.Q4_K}


def read_config(src_dir):
    path = os.path.join(src_dir, "config.json")
    if not os.path.isfile(path):
        warn("no config.json next to the weights, using built-in defaults")
        return TEXT_CFG.copy(), VIS_CFG.copy()
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    tc, vc = cfg.get("text_config", {}), cfg.get("vision_config", {})
    text = TEXT_CFG.copy()
    rp = tc.get("rope_parameters", {})
    text.update({k: v for k, v in dict(
        n_layer=tc.get("num_hidden_layers"), n_embd=tc.get("hidden_size"),
        n_head=tc.get("num_attention_heads"), n_kv_head=tc.get("num_key_value_heads"),
        head_dim=tc.get("head_dim"), n_ff=tc.get("intermediate_size"),
        n_vocab=tc.get("vocab_size"), rms_eps=tc.get("rms_norm_eps"),
        rope_theta=rp.get("rope_theta"), ctx=tc.get("max_position_embeddings"),
        rope_dim=int(tc.get("head_dim", 256) * rp.get("partial_rotary_factor", 0.25)),
        mrope_sections=tuple(rp.get("mrope_section", (11, 11, 10))),
        full_attn_interval=tc.get("full_attention_interval", 4),
        ssm_d_conv=tc.get("linear_conv_kernel_dim"), ssm_d_state=tc.get("linear_key_head_dim"),
        ssm_d_inner=tc.get("linear_value_head_dim") * tc.get("linear_num_value_heads"),
        ssm_n_group=tc.get("linear_num_key_heads"), ssm_dt_rank=tc.get("linear_num_value_heads"),
    ).items() if v is not None})
    if "layer_types" in tc:                       # explicit per-layer types win
        full = [i for i, t in enumerate(tc["layer_types"]) if t != "linear_attention"]
        text["_full_layers"] = full
    vis = VIS_CFG.copy()
    vis.update({k: v for k, v in dict(
        n_layer=vc.get("depth"), n_embd=vc.get("hidden_size"),
        n_head=vc.get("num_heads"), n_ff=vc.get("intermediate_size"),
        patch=vc.get("patch_size"), temporal_patch=vc.get("temporal_patch_size"),
        merge=vc.get("spatial_merge_size"), pos_table=vc.get("num_position_embeddings"),
    ).items() if v is not None})
    return text, vis


def full_attn_layers(text):
    if "_full_layers" in text:
        return set(text["_full_layers"])
    return {i for i in range(text["n_layer"]) if (i + 1) % text["full_attn_interval"] == 0}


def write_metadata(gw, text, vis, outtype):
    gw.add_architecture()   # name set by the caller from the checkpoint dir
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q6_k": gguf.LlamaFileType.MOSTLY_Q6_K,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
        "q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_M,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    # ---- text (hybrid qwen3.5) ----
    gw.add_uint32("marlin.block_count", text["n_layer"])
    gw.add_uint32("marlin.embedding_length", text["n_embd"])
    gw.add_uint32("marlin.feed_forward_length", text["n_ff"])
    gw.add_uint32("marlin.attention.head_count", text["n_head"])
    gw.add_uint32("marlin.attention.head_count_kv", text["n_kv_head"])
    gw.add_uint32("marlin.attention.key_length", text["head_dim"])
    gw.add_uint32("marlin.attention.value_length", text["head_dim"])
    gw.add_uint32("marlin.vocab_size", text["n_vocab"])
    gw.add_uint32("marlin.context_length", text["ctx"])
    gw.add_float32("marlin.attention.layer_norm_rms_epsilon", text["rms_eps"])
    gw.add_float32("marlin.rope.freq_base", float(text["rope_theta"]))
    gw.add_uint32("marlin.rope.dimension_count", text["rope_dim"])
    for i, s in enumerate(text["mrope_sections"]):
        gw.add_uint32(f"marlin.rope.dimension_section_{i}", s)
    gw.add_bool("marlin.tie_word_embeddings", True)
    recurrent = [i not in full_attn_layers(text) for i in range(text["n_layer"])]
    gw.add_array("marlin.recurrent_layers", recurrent)
    # gated-delta (ssm) hyperparameters
    gw.add_uint32("marlin.ssm.conv_kernel", text["ssm_d_conv"])
    gw.add_uint32("marlin.ssm.state_size", text["ssm_d_state"])
    gw.add_uint32("marlin.ssm.inner_size", text["ssm_d_inner"])
    gw.add_uint32("marlin.ssm.group_count", text["ssm_n_group"])
    gw.add_uint32("marlin.ssm.time_step_rank", text["ssm_dt_rank"])

    # ---- vision ----
    gw.add_uint32("marlin.vision.block_count", vis["n_layer"])
    gw.add_uint32("marlin.vision.embedding_length", vis["n_embd"])
    gw.add_uint32("marlin.vision.feed_forward_length", vis["n_ff"])
    gw.add_uint32("marlin.vision.attention.head_count", vis["n_head"])
    gw.add_uint32("marlin.vision.patch_size", vis["patch"])
    gw.add_uint32("marlin.vision.temporal_patch_size", vis["temporal_patch"])
    gw.add_uint32("marlin.vision.merge_size", vis["merge"])
    gw.add_uint32("marlin.vision.pos_table_size", vis["pos_table"])
    gw.add_float32("marlin.vision.layer_norm_epsilon", vis["eps"])
    gw.add_float32("marlin.vision.rope_theta", 10000.0)
    gw.add_uint32("marlin.vision.out_hidden_size", text["n_embd"])
    for key, value in SPECIAL.items():
        gw.add_uint32(f"marlin.{key}", value)


def verify(path, expected_tensors, required):
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
                         "vision tower always stays F16")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-q", "--quiet", action="store_true")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    src_name = (args.src or "").replace(chr(92), "/").rstrip("/").split("/")[-1]
    out = args.out or f"{(src_name or 'marlin-2b').lower()}-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    qtype = OUTTYPES[args.outtype]
    text, vis = read_config(src)
    full = full_attn_layers(text)
    log(f"text: {text['n_layer']}L ({len(full)} full-attn) "
        f"d={text['n_embd']} vocab={text['n_vocab']} rope={text['rope_dim']}d "
        f"sections={text['mrope_sections']} theta={text['rope_theta']:g}")
    log(f"vision: {vis['n_layer']}L d={vis['n_embd']} h={vis['n_head']} ff={vis['n_ff']} "
        f"patch={vis['patch']}x{vis['temporal_patch']}t merge={vis['merge']} "
        f"pos={vis['pos_table']}")

    def quantize_text(arr):
        if qtype is None:
            return None
        if arr.shape[-1] % 256:
            warn(f"row length {arr.shape[-1]} is not a multiple of 256, keeping F16")
            return None
        if qtype == gguf.GGMLQuantizationType.Q6_K:
            import q6k
            return q6k.quantize_q6_k(np.ascontiguousarray(arr, dtype=np.float32))
        return _lib.quantize(arr, qtype)

    weights = _lib.TensorSource(src)
    log(f"{len(weights)} tensors in the checkpoint")

    std = src.replace(chr(92), "/").rstrip("/").split("/")[-1]
    gw = gguf.GGUFWriter(out, "marlin")
    gw.add_name(std)
    write_metadata(gw, text, vis, args.outtype)

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

    # ---------------------------------------------------------- text decoder
    lm = "model.language_model"
    log("text decoder: embeddings")
    emb = weights.get(f"{lm}.embed_tokens.weight")
    if f"{lm}.lm_head.weight" in weights or "lm_head.weight" in weights:
        lmw = weights.get("lm_head.weight")
        if lmw.shape == emb.shape and not np.array_equal(lmw, emb):
            warn("lm_head differs from embed_tokens despite tie_word_embeddings")
    blocks = quantize_text(emb)
    if blocks is None:
        _lib.add_f16(gw, "token_embd.weight", emb); stats["f16"] += 1
    else:
        _lib.add_quantized(gw, "token_embd.weight", blocks, qtype); stats["quantized"] += 1
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
            add_tensor(f"{b}.la_norm.weight", f"{p}.linear_attn.norm.weight")   # gated norm: NO +1
            add_tensor(f"{b}.la_out.weight", f"{p}.linear_attn.out_proj.weight", quantize=True)
            gw.add_tensor(f"{b}.la_dt", np.ascontiguousarray(
                weights.get(f"{p}.linear_attn.dt_bias"), dtype=np.float32)); stats["f32"] += 1
            gw.add_tensor(f"{b}.la_a", np.ascontiguousarray(
                -np.exp(weights.get(f"{p}.linear_attn.A_log")), dtype=np.float32)); stats["f32"] += 1
            conv = weights.get(f"{p}.linear_attn.conv1d.weight").squeeze(1)   # [conv_dim, k]
            # HF layout is already ggml's: ne = [k, conv_dim] (k contiguous per channel)
            gw.add_tensor(f"{b}.la_conv1d", np.ascontiguousarray(conv, dtype=np.float32)); stats["f32"] += 1
        add_tensor(f"{b}.ffn_gate.weight", f"{p}.mlp.gate_proj.weight", quantize=True)
        add_tensor(f"{b}.ffn_up.weight", f"{p}.mlp.up_proj.weight", quantize=True)
        add_tensor(f"{b}.ffn_down.weight", f"{p}.mlp.down_proj.weight", quantize=True)
        if not args.quiet:
            progress(i, text["n_layer"], "text layer", every=6)

    # ---------------------------------------------------------- vision tower
    vt = "model.visual"
    conv = weights.get(f"{vt}.patch_embed.proj.weight")    # (1024, 3, 2, 16, 16)
    out_dim, channels, kt, kh, kw = conv.shape
    if (channels, kt, kh, kw) != (3, vis["temporal_patch"], vis["patch"], vis["patch"]):
        fail(f"unexpected patch_embed shape {conv.shape}")
    _lib.add_f16(gw, "v.patch_embd.weight", conv.reshape(out_dim, channels * kt * kh * kw))
    stats["f16"] += 1
    add_tensor("v.patch_embd.bias", f"{vt}.patch_embed.proj.bias")
    gw.add_tensor("v.pos_emb.weight", np.ascontiguousarray(
        weights.get(f"{vt}.pos_embed.weight"), dtype=np.float32))             # F32, host interpolates
    stats["f32"] += 1

    vision_suffixes = ("norm0.weight", "norm0.bias", "norm1.weight", "norm1.bias",
                       "wqkv.weight", "wqkv.bias", "wo.weight", "wo.bias",
                       "mlp.fc0.weight", "mlp.fc0.bias", "mlp.fc1.weight", "mlp.fc1.bias")
    hf_vision = {"norm0": "norm1", "norm1": "norm2", "wqkv": "attn.qkv", "wo": "attn.proj",
                 "mlp.fc0": "mlp.linear_fc1", "mlp.fc1": "mlp.linear_fc2"}
    for i in range(vis["n_layer"]):
        p, b = f"{vt}.blocks.{i}", f"v.blk.{i}"
        for suffix in vision_suffixes:
            mod, leaf = suffix.rsplit(".", 1)
            add_tensor(f"{b}.{suffix}", f"{p}.{hf_vision[mod]}.{leaf}")
        if not args.quiet:
            progress(i, vis["n_layer"], "vision layer", every=6)

    # ---------------------------------------------------------- merger (projector)
    log("merger")
    add_tensor("mm.norm.weight", f"{vt}.merger.norm.weight")
    add_tensor("mm.norm.bias", f"{vt}.merger.norm.bias")
    add_tensor("mm.fc1.weight", f"{vt}.merger.linear_fc1.weight")
    add_tensor("mm.fc1.bias", f"{vt}.merger.linear_fc1.bias")
    add_tensor("mm.fc2.weight", f"{vt}.merger.linear_fc2.weight")
    add_tensor("mm.fc2.bias", f"{vt}.merger.linear_fc2.bias")

    # ---------------------------------------------------------- write + check
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    log(f"done: {out} ({os.path.getsize(out) / 1e9:.2f} GB, "
        f"{stats['quantized']} quantized / {stats['f16']} f16 / {stats['f32']} f32 tensors)")
    verify(out, len(weights), required=[
        "token_embd.weight", "output_norm.weight",
        "blk.0.la_qkv.weight", "blk.3.attn_q.weight",
        f"blk.{text['n_layer'] - 1}.ffn_down.weight",
        "v.patch_embd.weight", "v.pos_emb.weight",
        f"v.blk.{vis['n_layer'] - 1}.mlp.fc1.weight", "mm.fc2.weight",
    ])
    return 0


if __name__ == "__main__":
    sys.exit(main())
