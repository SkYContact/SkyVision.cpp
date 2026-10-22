"""Convert MCG-NJU/OneStreamer-4B (HF safetensors) to GGUF for the qwen3vl backend.

OneStreamer-4B is a streaming-video SFT of **Qwen3-VL-4B-Instruct**: a dense
Qwen3-VL text trunk (36 full-attention layers, IMROPE 24/20/20, per-head q/k
RMSNorm, tied embeddings) plus the 24-layer Qwen3-VL ViT (patch 16, temporal 2,
2x2 merge, learned 48x48 absolute pos table) and Qwen3-VL **DeepStack**: the
tower's intermediate outputs at blocks 5/11/17 are merged to text width by three
extra merger MLPs and added into the residual stream after decoder layers 0/1/2.

Usage::

  python convert/convert_onestreamer.py [--src DIR] [-o OUT.gguf]
                                       [--outtype f16|q8_0|q6_k|q4_0|q4_k]

Checkpoint (Hugging Face, no gating)::

  models/OneStreamer-4B   (MCG-NJU/OneStreamer-4B)

Layout notes (verified against transformers 4.57 modeling_qwen3_vl.py and the
Qwen3.5 converter, whose vision tower is the same Qwen3-VL tower):

* Qwen3-VL RMSNorm is the plain ``x * w`` form - **no +1** on any norm weight
  (Qwen3.5's ``x * (1 + w)`` must not be copied here).
* text ``q_proj`` is [num_heads * head_dim, hidden] with no q|gate interleave and
  no sigmoid output gate; q/k/v/o are stored unpermuted (HF rotate_half pairs ==
  ggml NEOX pairs).
* mrope: ``rope.dimension_count`` is the *rotated* dim count = head_dim (128, full
  rotary) and the sections (24/20/20) are pair counts of the interleaved Qwen3-VL
  layout == GGML_ROPE_TYPE_IMROPE.
* vision ``qkv`` is stored unpermuted: Qwen3-VL's native [row_re | col_re | row_im
  | col_im] pair layout is already ggml's GGML_ROPE_TYPE_VISION layout.
* pos_embed is a learned 48x48 table; the tower F16, the table F32 (the host
  bilinearly interpolates it per frame, ALIGN_CORNERS semantics).
* the main merger normalises **before** the 2x2 shuffle; the three DeepStack
  mergers normalise **after** it (LayerNorm width 4*hidden, not hidden).
* the checkpoint carries the streaming control tokens </Silence> </Standby>
  </Response> </Observe> </Summary> inside the 151936-entry vocab.
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

MODEL_ID = "MCG-NJU/OneStreamer-4B"
ALT_MODEL_IDS = ("Qwen/Qwen3-VL-4B-Instruct",)
ENV_VAR = "ONESTREAMER_SRC"
PATTERNS = ("model.safetensors.index.json", "*.safetensors")

ARCH = "qwen3vl"

# defaults from the released config.json (overridden by it when present)
TEXT_CFG = dict(n_layer=36, n_embd=2560, n_head=32, n_kv_head=8, head_dim=128,
                n_ff=9728, n_vocab=151936, rms_eps=1e-6, rope_theta=5e6,
                rope_dim=128, mrope_sections=(24, 20, 20), ctx=262144)
VIS_CFG = dict(n_layer=24, n_embd=1024, n_head=16, n_ff=4096, patch=16,
               temporal_patch=2, merge=2, pos_table=2304, eps=1e-6,
               deepstack=(5, 11, 17))
SPECIAL = dict(image_token_id=151655, video_token_id=151656,
               vision_start_token_id=151652, vision_end_token_id=151653,
               pad_token_id=151643)

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
    # Qwen3-VL keeps the mrope settings under rope_scaling (transformers <4.58)
    # or rope_parameters (newer); accept both spellings
    rp = tc.get("rope_scaling") or tc.get("rope_parameters") or {}
    head_dim = tc.get("head_dim") or (
        tc.get("hidden_size") // tc.get("num_attention_heads", 1) or text["head_dim"])
    text.update({k: v for k, v in dict(
        n_layer=tc.get("num_hidden_layers"), n_embd=tc.get("hidden_size"),
        n_head=tc.get("num_attention_heads"), n_kv_head=tc.get("num_key_value_heads"),
        head_dim=head_dim, n_ff=tc.get("intermediate_size"),
        n_vocab=tc.get("vocab_size"), rms_eps=tc.get("rms_norm_eps"),
        rope_theta=tc.get("rope_theta", rp.get("rope_theta")),
        ctx=tc.get("max_position_embeddings"),
        # full rotary: rope dim == head_dim (partial_rotary_factor defaults to 1)
        rope_dim=int(head_dim * rp.get("partial_rotary_factor", 1.0)),
        mrope_sections=tuple(rp.get("mrope_section") or text["mrope_sections"]),
    ).items() if v is not None})
    if rp.get("mrope_interleaved") is False:
        fail("mrope_interleaved=false is not supported (ggml IMROPE is the "
             "interleaved Qwen3-VL layout)")
    vis = VIS_CFG.copy()
    vis.update({k: v for k, v in dict(
        n_layer=vc.get("depth"), n_embd=vc.get("hidden_size"),
        n_head=vc.get("num_heads"), n_ff=vc.get("intermediate_size"),
        patch=vc.get("patch_size"), temporal_patch=vc.get("temporal_patch_size"),
        merge=vc.get("spatial_merge_size"), pos_table=vc.get("num_position_embeddings"),
        deepstack=tuple(vc.get("deepstack_visual_indexes") or vis["deepstack"]),
    ).items() if v is not None})
    return text, vis


src_hint = "."


def write_metadata(gw, text, vis, outtype):
    gw.add_architecture()   # "qwen3vl"
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q6_k": gguf.LlamaFileType.MOSTLY_Q6_K,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
        "q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_M,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    # ---- text (dense qwen3-vl trunk) ----
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
    # every layer is full attention (0 = not recurrent in the shared loader)
    gw.add_array("marlin.recurrent_layers", [False] * text["n_layer"])

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
    # DeepStack: vision block indices whose output feeds an extra merger MLP and
    # is added to the text residual stream after decoder layer 0, 1, 2
    gw.add_array("marlin.deepstack_indexes", [int(i) for i in vis["deepstack"]])
    gw.add_bool("marlin.has_mtp", False)
    gw.add_bool("marlin.skills", False)


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
    out = args.out or f"{(src_name or 'onestreamer-4b').lower()}-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    global src_hint
    src_hint = src
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    qtype = OUTTYPES[args.outtype]
    text, vis = read_config(src)
    if not vis["deepstack"]:
        warn("no deepstack_visual_indexes in config.json - DeepStack will be off")
    log(f"text: {text['n_layer']}L dense d={text['n_embd']} vocab={text['n_vocab']} "
        f"rope={text['rope_dim']}d sections={text['mrope_sections']} theta={text['rope_theta']:g}")
    log(f"vision: {vis['n_layer']}L d={vis['n_embd']} h={vis['n_head']} ff={vis['n_ff']} "
        f"patch={vis['patch']}x{vis['temporal_patch']}t merge={vis['merge']} "
        f"pos={vis['pos_table']} deepstack={vis['deepstack']}")

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
    gw = gguf.GGUFWriter(out, ARCH)
    gw.add_name(std)
    write_metadata(gw, text, vis, args.outtype)

    stats = {"quantized": 0, "f16": 0, "f32": 0}

    def add_tensor(name, hf_name, quantize=False, dtype=None):
        arr = weights.get(hf_name)
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
    add_tensor("output_norm.weight", f"{lm}.norm.weight")

    for i in range(text["n_layer"]):
        p, b = f"{lm}.layers.{i}", f"blk.{i}"
        add_tensor(f"{b}.attn_norm.weight", f"{p}.input_layernorm.weight")
        add_tensor(f"{b}.attn_post_norm.weight", f"{p}.post_attention_layernorm.weight")
        add_tensor(f"{b}.attn_q.weight", f"{p}.self_attn.q_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_k.weight", f"{p}.self_attn.k_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_v.weight", f"{p}.self_attn.v_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_output.weight", f"{p}.self_attn.o_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_q_norm.weight", f"{p}.self_attn.q_norm.weight")
        add_tensor(f"{b}.attn_k_norm.weight", f"{p}.self_attn.k_norm.weight")
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
    # main merger: LayerNorm over the *unshuffled* hidden (1024), then the 2x2 fold
    log("merger")
    add_tensor("mm.norm.weight", f"{vt}.merger.norm.weight")
    add_tensor("mm.norm.bias", f"{vt}.merger.norm.bias")
    add_tensor("mm.fc1.weight", f"{vt}.merger.linear_fc1.weight")
    add_tensor("mm.fc1.bias", f"{vt}.merger.linear_fc1.bias")
    add_tensor("mm.fc2.weight", f"{vt}.merger.linear_fc2.weight")
    add_tensor("mm.fc2.bias", f"{vt}.merger.linear_fc2.bias")

    # DeepStack mergers: use_postshuffle_norm=True - LayerNorm width 4*hidden,
    # applied AFTER the fold (see modeling_qwen3_vl.Qwen3VLVisionPatchMerger)
    log(f"deepstack mergers ({len(vis['deepstack'])})")
    for k in range(len(vis["deepstack"])):
        p, b = f"{vt}.deepstack_merger_list.{k}", f"dsm.{k}"
        add_tensor(f"{b}.norm.weight", f"{p}.norm.weight")
        add_tensor(f"{b}.norm.bias", f"{p}.norm.bias")
        add_tensor(f"{b}.fc1.weight", f"{p}.linear_fc1.weight")
        add_tensor(f"{b}.fc1.bias", f"{p}.linear_fc1.bias")
        add_tensor(f"{b}.fc2.weight", f"{p}.linear_fc2.weight")
        add_tensor(f"{b}.fc2.bias", f"{p}.linear_fc2.bias")
        expect = vis["n_embd"] * vis["merge"] * vis["merge"]
        got = weights.get(f"{p}.norm.weight").shape[0]
        if got != expect:
            fail(f"deepstack merger {k}: norm width {got}, expected {expect}")

    # ---------------------------------------------------------- write + check
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    log(f"done: {out} ({os.path.getsize(out) / 1e9:.2f} GB, "
        f"{stats['quantized']} quantized / {stats['f16']} f16 / {stats['f32']} f32 tensors)")
    verify(out, len(weights), required=[
        "token_embd.weight", "output_norm.weight",
        "blk.0.attn_q.weight", "blk.0.attn_q_norm.weight",
        f"blk.{text['n_layer'] - 1}.ffn_down.weight",
        "v.patch_embd.weight", "v.pos_emb.weight",
        f"v.blk.{vis['n_layer'] - 1}.mlp.fc1.weight", "mm.fc2.weight",
        "dsm.0.fc2.weight", f"dsm.{len(vis['deepstack']) - 1}.fc2.weight",
    ])
    return 0


if __name__ == "__main__":
    sys.exit(main())
