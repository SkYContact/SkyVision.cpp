"""Convert ZhuYuhan/VideoJev-4B (HF safetensors) to GGUF for the qwen3_5 engine.

VideoJev = Qwen3.5-4B backbone (32 layers: 24 gated-delta linear + 8 full
attention) + the Qwen3-VL style vision tower + a trained cosine decision head.

What differs from convert_qwen3_5.py (all verified against the released
config.json / model.safetensors.index.json):

* the HF weights are wrapped: every tensor sits under ``backbone.`` and the
  decision head under ``head.``; the backbone config is nested in
  ``backbone_config`` (top level carries the head/decoder fields instead).
* the LAST text layer is attention-only: the reference implementation deletes
  ``last.mlp`` and ``last.post_attention_layernorm`` before inference, and the
  checkpoint consequently ships neither.  ``marlin.last_layer_attn_only`` tells
  the engine to skip the FFN block on that layer.
* linear attention is GQA here: 16 key heads but 32 value heads.  ggml's
  ``gated_delta_net`` pairs value head ``j`` with key head ``j % n_k`` (tile),
  while the reference repeats q/k heads with ``repeat_interleave`` — so the
  value-side tensors are reordered at conversion time (see ``perm_v_heads``).
* ``head.{q,k}.{weight,bias}`` + a scalar ``head.bias`` form
  ``CosineDecisionHead``: unnormalized logits ``scale * cos(q(h_q), k(h_c)) + b``
  over (candidate, query) hidden-state pairs, scored in FP32.
* no lm_head (tied embeddings) and no MTP draft head.

Usage::

  python convert/convert_videojev.py --src models/VideoJev-4B \\
                                     -o videojev-4b-q8_0.gguf [--outtype q8_0]

Prompt format (must match the engine, see ``Qwen3_5::score``)::

  <|im_start|>user\\n
    "<|vision_start|>"                                   template's own video placeholder
    {"<t.t seconds><|vision_start|>" + video_pad * spatial + "<|vision_end|>"} per temporal patch
    "<|vision_end|>"                                     its closing bracket, after the last patch
    "\\nTask: temporal grounding\\nQuestion: Q\\nOptions:\\n"
    {"<t.t seconds>" + "<|option_end|>" + "\\n"} per candidate
    "<|decide|>"
  <|im_end|>\\n
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

MODEL_ID = "ZhuYuhan/VideoJev-4B"
ALT_MODEL_IDS = ("MCG-NJU/VideoJev-4B",)
ENV_VAR = "VIDEOJEV_SRC"
PATTERNS = ("model.safetensors.index.json", "model.safetensors")

# defaults from the released config.json (overridden by it when present)
TEXT_CFG = dict(n_layer=32, n_embd=2560, n_head=16, n_kv_head=4, head_dim=256,
                n_ff=10240, n_vocab=248320, rms_eps=1e-6, rope_theta=1e7,
                rope_dim=64, mrope_sections=(11, 11, 10), ctx=262144,
                full_attn_interval=4,
                ssm_d_conv=4, ssm_d_state=128, ssm_d_inner=4096,
                ssm_n_group=16, ssm_dt_rank=32)
VIS_CFG = dict(n_layer=24, n_embd=1024, n_head=16, n_ff=4096, patch=16,
               temporal_patch=2, merge=2, pos_table=2304, eps=1e-6)
SPECIAL = dict(image_token_id=248056, video_token_id=248057,
               vision_start_token_id=248053, vision_end_token_id=248054,
               pad_token_id=248044, eos_token_id=248044)
HEAD_CFG = dict(head_dim=512, head_scale=10.0, threshold=0.25, smooth=1,
                decide_token_id=248077, option_end_token_id=248078)

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q6_k": gguf.GGMLQuantizationType.Q6_K,
            "q4_0": gguf.GGMLQuantizationType.Q4_0,
            "q4_k": gguf.GGMLQuantizationType.Q4_K}


def tensor_prefixes(src_dir):
    """(language prefix, visual prefix) — VideoJev nests them under 'backbone.'."""
    try:
        with open(os.path.join(src_dir, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    if "backbone_config" in cfg:
        return "backbone.language_model", "backbone.visual", cfg
    return "model.language_model", "model.visual", cfg


def read_config(src_dir):
    lm, vt, cfg = tensor_prefixes(src_dir)
    bb = cfg.get("backbone_config", cfg)
    tc, vc = bb.get("text_config", {}), bb.get("vision_config", {})
    text = TEXT_CFG.copy()
    rp = tc.get("rope_parameters", tc.get("rope_scaling", {})) or {}
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
        ssm_d_inner=(tc.get("linear_value_head_dim") or 128) * (tc.get("linear_num_value_heads") or 32),
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
        eps=vc.get("layer_norm_eps"),
    ).items() if v is not None})
    head = HEAD_CFG.copy()
    head.update({k: v for k, v in dict(
        head_dim=cfg.get("head_dim"), head_scale=cfg.get("head_logit_scale"),
        decide_token_id=cfg.get("decide_token_id"),
        option_end_token_id=cfg.get("option_end_token_id"),
    ).items() if v is not None})
    dec = cfg.get("decoder", {}) or {}
    head.update({k: v for k, v in dict(
        threshold=dec.get("threshold"), smooth=dec.get("smooth")).items() if v is not None})
    if dec.get("postprocess", "round_repair_clamp_v1") != "round_repair_clamp_v1":
        warn(f"unknown decoder postprocess {dec.get('postprocess')!r}; "
             f"the engine implements round_repair_clamp_v1 only")
    special = SPECIAL.copy()
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        if bb.get(key) is not None:
            special[key] = bb[key]
    if tc.get("eos_token_id") is not None:
        special["eos_token_id"] = tc["eos_token_id"]
    return text, vis, head, special, lm, vt


def full_attn_layers(text):
    if "_full_layers" in text:
        return set(text["_full_layers"])
    return {i for i in range(text["n_layer"]) if (i + 1) % text["full_attn_interval"] == 0}


# ---------------------------------------------------------------- v-head reorder
def head_order(n_k, n_v, mode="interleave"):
    """Permutation that turns ggml's tile broadcast into HF repeat_interleave.

    ggml gated_delta_net pairs value head j with key head ``j % n_k``; the
    reference implementation pairs it with ``j // n_rep``.  Placing old value
    head ``n_rep*(j%n_k) + j//n_k`` at position j makes both agree.

    ``mode`` exists for ablation runs: ``identity`` keeps the checkpoint order
    (what a naive conversion does), ``inverse`` applies the opposite pairing.
    """
    if n_v == n_k:
        return None
    if n_v % n_k:
        fail(f"linear attention: {n_v} value heads is not a multiple of {n_k} key heads")
    n_rep = n_v // n_k
    if mode == "interleave":
        return [n_rep * (j % n_k) + j // n_k for j in range(n_v)]
    if mode == "identity":
        return None
    if mode == "inverse":
        return [n_k * (j % n_rep) + j // n_rep for j in range(n_v)]
    fail(f"unknown --vhead-order {mode}")


def permute_rows(arr, order, block):
    """Reorder row blocks of ``block`` rows each (weights are [rows, cols])."""
    if order is None:
        return arr
    out = np.empty_like(arr)
    for j, src in enumerate(order):
        out[j * block:(j + 1) * block] = arr[src * block:(src + 1) * block]
    return out


def permute_cols(arr, order, block):
    if order is None:
        return arr
    out = np.empty_like(arr)
    for j, src in enumerate(order):
        out[:, j * block:(j + 1) * block] = arr[:, src * block:(src + 1) * block]
    return out


src_hint = "."


def write_metadata(gw, text, vis, head, special, outtype):
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
    # the reference implementation deletes the last layer's MLP + post-norm
    gw.add_bool("marlin.last_layer_attn_only", True)

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
    for key, value in special.items():
        gw.add_uint32(f"marlin.{key}", value)

    # ---- decision head ----
    gw.add_uint32("videojev.head_dim", head["head_dim"])
    gw.add_float32("videojev.head_scale", float(head["head_scale"]))
    gw.add_float32("videojev.threshold", float(head["threshold"]))
    gw.add_uint32("videojev.smooth", int(head["smooth"]))
    gw.add_uint32("videojev.decide_token_id", int(head["decide_token_id"]))
    gw.add_uint32("videojev.option_end_token_id", int(head["option_end_token_id"]))


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
                         "vision tower and the decision head stay F16/F32")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--vhead-order", default="interleave",
                    choices=["interleave", "identity", "inverse"],
                    help="ablation switch: how linear-attention value heads are mapped "
                         "onto ggml's tile pairing (default interleave = correct)")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    src_name = (args.src or "").replace(chr(92), "/").rstrip("/").split("/")[-1]
    out = args.out or f"{(src_name or 'videojev-4b').lower()}-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    global src_hint
    src_hint = src
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    qtype = OUTTYPES[args.outtype]
    text, vis, head, special, lm, vt = read_config(src)
    full = full_attn_layers(text)
    last = text["n_layer"] - 1
    if last not in full:
        fail(f"last layer {last} is not a full-attention layer; the attention-only "
             f"trim the reference applies would not apply")
    log(f"text: {text['n_layer']}L ({len(full)} full-attn) d={text['n_embd']} "
        f"vocab={text['n_vocab']} rope={text['rope_dim']}d sections={text['mrope_sections']} "
        f"theta={text['rope_theta']:g} ssm k={text['ssm_n_group']} v={text['ssm_dt_rank']}")
    log(f"vision: {vis['n_layer']}L d={vis['n_embd']} h={vis['n_head']} ff={vis['n_ff']} "
        f"patch={vis['patch']}x{vis['temporal_patch']}t merge={vis['merge']} pos={vis['pos_table']}")
    log(f"head: dim={head['head_dim']} scale={head['head_scale']:g} "
        f"threshold={head['threshold']:g} smooth={head['smooth']}")

    order = head_order(text["ssm_n_group"], text["ssm_dt_rank"], args.vhead_order)
    if order is not None:
        log(f"linear attention GQA: {text['ssm_n_group']} key / {text['ssm_dt_rank']} value heads, "
            f"value side reordered for ggml's tile broadcast ({args.vhead_order})")
    else:
        log(f"linear attention GQA: {text['ssm_n_group']} key / {text['ssm_dt_rank']} value heads, "
            f"value order left as-is ({args.vhead_order})")

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
    gw = gguf.GGUFWriter(out, "videojev")
    gw.add_name(std)
    write_metadata(gw, text, vis, head, special, args.outtype)

    stats = {"quantized": 0, "f16": 0, "f32": 0}

    def add_tensor(name, hf_name, quantize=False, plus_one=False, dtype=None, arr=None):
        if arr is None:
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
    log("text decoder: embeddings")
    emb = weights.get(f"{lm}.embed_tokens.weight")
    if f"{lm}.lm_head.weight" in weights:
        warn("checkpoint carries a separate lm_head; the engine ties it to the embedding")
    blocks = quantize_text(emb)
    if blocks is None:
        _lib.add_f16(gw, "token_embd.weight", emb); stats["f16"] += 1
    else:
        _lib.add_quantized(gw, "token_embd.weight", blocks, qtype); stats["quantized"] += 1
    add_tensor("output_norm.weight", f"{lm}.norm.weight", plus_one=True)

    Sk = text["ssm_d_state"]
    n_vh = text["ssm_dt_rank"]
    n_kh = text["ssm_n_group"]
    for i in range(text["n_layer"]):
        p, b = f"{lm}.layers.{i}", f"blk.{i}"
        attn_only = (i == last)
        add_tensor(f"{b}.attn_norm.weight", f"{p}.input_layernorm.weight", plus_one=True)
        if not attn_only:
            add_tensor(f"{b}.attn_post_norm.weight", f"{p}.post_attention_layernorm.weight", plus_one=True)
        if i in full:
            add_tensor(f"{b}.attn_q.weight", f"{p}.self_attn.q_proj.weight", quantize=True)
            add_tensor(f"{b}.attn_k.weight", f"{p}.self_attn.k_proj.weight", quantize=True)
            add_tensor(f"{b}.attn_v.weight", f"{p}.self_attn.v_proj.weight", quantize=True)
            add_tensor(f"{b}.attn_output.weight", f"{p}.self_attn.o_proj.weight", quantize=True)
            add_tensor(f"{b}.attn_q_norm.weight", f"{p}.self_attn.q_norm.weight", plus_one=True)
            add_tensor(f"{b}.attn_k_norm.weight", f"{p}.self_attn.k_norm.weight", plus_one=True)
        else:
            qkv = weights.get(f"{p}.linear_attn.in_proj_qkv.weight")
            qk2 = 2 * n_kh * Sk
            if qkv.shape[0] != qk2 + n_vh * Sk:
                fail(f"{b}: in_proj_qkv has {qkv.shape[0]} rows, expected "
                     f"2*{n_kh}*{Sk} + {n_vh}*{Sk}")
            if order is not None:
                qkv = np.concatenate([qkv[:qk2], permute_rows(qkv[qk2:], order, Sk)], axis=0)
            add_tensor(f"{b}.la_qkv.weight", None, quantize=True, arr=qkv)
            add_tensor(f"{b}.la_gate.weight", f"{p}.linear_attn.in_proj_z.weight", quantize=True,
                       arr=permute_rows(weights.get(f"{p}.linear_attn.in_proj_z.weight"), order, Sk))
            add_tensor(f"{b}.la_beta.weight", f"{p}.linear_attn.in_proj_b.weight", quantize=True,
                       arr=permute_rows(weights.get(f"{p}.linear_attn.in_proj_b.weight"), order, 1))
            add_tensor(f"{b}.la_alpha.weight", f"{p}.linear_attn.in_proj_a.weight", quantize=True,
                       arr=permute_rows(weights.get(f"{p}.linear_attn.in_proj_a.weight"), order, 1))
            add_tensor(f"{b}.la_norm.weight", f"{p}.linear_attn.norm.weight")   # gated norm: NO +1
            out_proj = weights.get(f"{p}.linear_attn.out_proj.weight")
            add_tensor(f"{b}.la_out.weight", None, quantize=True,
                       arr=permute_cols(out_proj, order, Sk))
            dt_bias = weights.get(f"{p}.linear_attn.dt_bias")
            a_log = -np.exp(weights.get(f"{p}.linear_attn.A_log"))
            if order is not None:
                dt_bias = np.asarray(dt_bias)[order]
                a_log = np.asarray(a_log)[order]
            gw.add_tensor(f"{b}.la_dt", np.ascontiguousarray(dt_bias, dtype=np.float32)); stats["f32"] += 1
            gw.add_tensor(f"{b}.la_a", np.ascontiguousarray(a_log, dtype=np.float32)); stats["f32"] += 1
            conv = weights.get(f"{p}.linear_attn.conv1d.weight").squeeze(1)   # [conv_dim, k]
            if conv.shape[0] != qk2 + n_vh * Sk:
                fail(f"{b}: conv1d has {conv.shape[0]} channels, expected {qk2 + n_vh * Sk}")
            if order is not None:
                conv = np.concatenate([conv[:qk2], permute_rows(conv[qk2:], order, Sk)], axis=0)
            # HF layout is already ggml's: ne = [k, conv_dim] (k contiguous per channel)
            gw.add_tensor(f"{b}.la_conv1d", np.ascontiguousarray(conv, dtype=np.float32)); stats["f32"] += 1
        if not attn_only:
            add_tensor(f"{b}.ffn_gate.weight", f"{p}.mlp.gate_proj.weight", quantize=True)
            add_tensor(f"{b}.ffn_up.weight", f"{p}.mlp.up_proj.weight", quantize=True)
            add_tensor(f"{b}.ffn_down.weight", f"{p}.mlp.down_proj.weight", quantize=True)
        if not args.quiet:
            progress(i, text["n_layer"], "text layer", every=6)

    # ---------------------------------------------------------- vision tower
    log("vision tower")
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

    # ---------------------------------------------------------- decision head
    # CosineDecisionHead scores in FP32 (the reference keeps it in fp32 too)
    log("decision head")
    add_tensor("jev_head_q.weight", "head.q.weight", dtype=np.float32)
    add_tensor("jev_head_q.bias", "head.q.bias", dtype=np.float32)
    add_tensor("jev_head_k.weight", "head.k.weight", dtype=np.float32)
    add_tensor("jev_head_k.bias", "head.k.bias", dtype=np.float32)
    add_tensor("jev_head_b", None, dtype=np.float32,
               arr=np.asarray(weights.get("head.bias"), dtype=np.float32).reshape(1))

    gw.add_bool("marlin.has_mtp", False)
    gw.add_bool("marlin.skills", False)

    # ---------------------------------------------------------- write + check
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    log(f"done: {out} ({os.path.getsize(out) / 1e9:.2f} GB, "
        f"{stats['quantized']} quantized / {stats['f16']} f16 / {stats['f32']} f32 tensors)")
    verify(out, len(weights), required=[
        "token_embd.weight", "output_norm.weight",
        "blk.0.la_qkv.weight", f"blk.{min(full)}.attn_q.weight",
        f"blk.{last}.attn_q.weight",
        "v.patch_embd.weight", "v.pos_emb.weight",
        f"v.blk.{vis['n_layer'] - 1}.mlp.fc1.weight", "mm.fc2.weight",
        "jev_head_q.weight", "jev_head_k.weight", "jev_head_b",
    ])
    return 0


if __name__ == "__main__":
    sys.exit(main())
