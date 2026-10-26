"""Convert Qwen3-Omni-30B-A3B-Instruct (BF16 safetensors, thinker only) to GGUF.

Usage::

  python convert/convert_qwen3omni.py [--src DIR] [-o OUT.gguf]
                                     [--outtype q8_0|f16]

Checkpoint: E:/SkyVision-weights/qwen3-omni/hf-instruct (default --src), i.e.
``hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir ...``.

Layout notes (all conventions verified against the official transformers
5.20.1 modeling source and the Keye GGUF stacked-expert byte order):

* text decoder (quantized per --outtype; norms/router stay F16/F32)
    token_embd / lm_head (untied) / output_norm
    blk.{i}.attn_norm, attn_q/k/v/o, attn_q_norm, attn_k_norm, ffn_norm
    blk.{i}.ffn_router.weight            [n_exp, n_embd] F32 (top-k router)
    blk.{i}.ffn_{gate,up,down}_exps      [n_exp, out, in] Q8_0 stacked;
        byte order = per-expert HF arrays concatenated on the slow axis, the
        same ne=[in,out,n_exp] layout the Keye GGUF uses (round-trip probe
        verified against gguf-py read/write).

* vision tower (always F16; learned pos table F32)
    v.patch_embd.weight [1152, 3*2*16*16] (Conv3d folded), v.patch_embd.bias
    v.pos_emb.weight [2304,1152] F32 (48x48 grid; host interpolates)
    v.blk.{i}.norm1/norm2.{w,b}  LayerNorm; attn_qkv/attn_proj (bias=True);
    mlp_fc1/fc2 (gelu-tanh)
    mm.ln_q + mm.fc1 [4608,1152] + mm.fc2 [2048,1152]      final merger
    mm_ds.{k}.ln_q + fc1 [4608,4608] + fc2 [2048,4608]     deepstack mergers
                                                            (k = 0,1,2 for
                                                            blocks 8,16,24)

* audio tower (always F16; sinusoidal positions are computed, not weights)
    a.conv1/conv2/conv3.{weight,bias}  Conv2d k3 s2 [480, Cin, 3, 3]
    a.conv_out.weight [1280, 480*16]
    a.blk.{i}.attn_ln / attn_q/k/v/o / ffn_ln / ffn_fc1/fc2   (Whisper style,
        LayerNorm pre-norm, biases everywhere, GELU)
    a.ln_post, a.proj1, a.proj2
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

_lib.torch = None   # torch 2.11 win: storage __getitem__ access-violates on this
                    # checkpoint's mmap; the numpy bf16 decode path is exact and
                    # fast enough

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
ENV_VAR = "QWEN3OMNI_SRC"
PATTERNS = ("model.safetensors.index.json",)

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q6_k": gguf.GGMLQuantizationType.Q6_K}


def read_config(src_dir):
    with open(os.path.join(src_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    th = cfg["thinker_config"]
    return th["text_config"], th["vision_config"], th["audio_config"], cfg


def write_metadata(gw, text, vis, aud, root, outtype):
    gw.add_architecture()
    gw.add_name("Qwen3-Omni-30B-A3B-Instruct-thinker")
    gw.add_file_type({"q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
                      "q6_k": gguf.LlamaFileType.MOSTLY_Q6_K}.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32("qwen3omni.block_count", text["num_hidden_layers"])
    gw.add_uint32("qwen3omni.embedding_length", text["hidden_size"])
    gw.add_uint32("qwen3omni.attention.head_count", text["num_attention_heads"])
    gw.add_uint32("qwen3omni.attention.head_count_kv", text["num_key_value_heads"])
    gw.add_uint32("qwen3omni.attention.key_length", text["head_dim"])
    gw.add_uint32("qwen3omni.attention.value_length", text["head_dim"])
    gw.add_uint32("qwen3omni.vocab_size", text["vocab_size"])
    gw.add_uint32("qwen3omni.context_length", text.get("max_position_embeddings", 32768))
    gw.add_float32("qwen3omni.attention.layer_norm_rms_epsilon", text["rms_norm_eps"])
    gw.add_float32("qwen3omni.rope.freq_base", float(text["rope_theta"]))
    gw.add_bool("qwen3omni.tie_word_embeddings", False)
    for i, s in enumerate(text["rope_scaling"]["mrope_section"]):
        gw.add_uint32(f"qwen3omni.rope.mrope_section_{i}", s)
    gw.add_uint32("qwen3omni.expert_count", text["num_experts"])
    gw.add_uint32("qwen3omni.expert_used_count", text["num_experts_per_tok"])
    gw.add_uint32("qwen3omni.expert_feed_forward_length", text["moe_intermediate_size"])
    gw.add_bool("qwen3omni.expert_norm_topk_prob", text.get("norm_topk_prob", True))

    gw.add_uint32("qwen3omni.vision.block_count", vis["depth"])
    gw.add_uint32("qwen3omni.vision.embedding_length", vis["hidden_size"])
    gw.add_uint32("qwen3omni.vision.feed_forward_length", vis["intermediate_size"])
    gw.add_uint32("qwen3omni.vision.attention.head_count", vis["num_heads"])
    gw.add_uint32("qwen3omni.vision.patch_size", vis["patch_size"])
    gw.add_uint32("qwen3omni.vision.temporal_patch_size", vis["temporal_patch_size"])
    gw.add_uint32("qwen3omni.vision.merge_size", vis["spatial_merge_size"])
    gw.add_uint32("qwen3omni.vision.out_hidden_size", vis["out_hidden_size"])
    if "num_position_embeddings" in vis:
        gw.add_uint32("qwen3omni.vision.num_position_embeddings", vis["num_position_embeddings"])
    for i, b in enumerate(vis["deepstack_visual_indexes"]):
        gw.add_uint32(f"qwen3omni.vision.deepstack_index_{i}", b)
    for k in ("image_token_id", "video_token_id", "vision_start_token_id",
              "vision_end_token_id", "image_token_id"):
        if k in root:
            gw.add_uint32(f"qwen3omni.{k}", root[k])
    for k in ("audio_token_id",):
        for src in (root, vis, aud):
            if k in src:
                gw.add_uint32(f"qwen3omni.{k}", src[k])
                break

    gw.add_uint32("qwen3omni.audio.block_count", aud["encoder_layers"])
    gw.add_uint32("qwen3omni.audio.embedding_length", aud["d_model"])
    gw.add_uint32("qwen3omni.audio.feed_forward_length", aud["encoder_ffn_dim"])
    gw.add_uint32("qwen3omni.audio.attention.head_count", aud["encoder_attention_heads"])
    gw.add_uint32("qwen3omni.audio.num_mel_bins", aud["num_mel_bins"])
    gw.add_uint32("qwen3omni.audio.n_window", aud["n_window"])
    gw.add_uint32("qwen3omni.audio.n_window_infer", aud["n_window_infer"])
    gw.add_uint32("qwen3omni.audio.conv_chunksize", aud["conv_chunksize"])
    gw.add_uint32("qwen3omni.audio.downsample_hidden_size", aud["downsample_hidden_size"])
    gw.add_uint32("qwen3omni.audio.output_dim", aud["output_dim"])
    gw.add_uint32("qwen3omni.audio.max_source_positions", aud["max_source_positions"])
    if eos := root.get("eos_token_id"):
        gw.add_uint32("qwen3omni.eos_token_id", eos)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="q8_0", choices=sorted(OUTTYPES))
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-q", "--quiet", action="store_true")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"qwen3omni-30b-thinker-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR, patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    qtype = OUTTYPES[args.outtype]
    text, vis, aud, root = read_config(src)
    n_l, n_exp = text["num_hidden_layers"], text["num_experts"]
    moe_i = text["moe_intermediate_size"]
    log(f"text: {n_l}L d={text['hidden_size']} moe={n_exp}x{text['num_experts_per_tok']} "
        f"I={moe_i} vocab={text['vocab_size']}")
    log(f"vision: {vis['depth']}L d={vis['hidden_size']} patch={vis['patch_size']}")
    log(f"audio: {aud['encoder_layers']}L d={aud['d_model']} mel={aud['num_mel_bins']}")

    import q6k
    def quantize_text(arr):
        if qtype is None:
            return None
        if arr.shape[-1] % 256:
            warn(f"row length {arr.shape[-1]} not multiple of 256, keeping F16: {arr.shape}")
            return None
        if qtype == gguf.GGMLQuantizationType.Q6_K:
            return q6k.quantize_q6_k(arr)          # gguf-py decodes Q6_K only
        return _lib.quantize(arr, qtype)

    weights = _lib.TensorSource(src)
    log(f"{len(weights)} tensors in the checkpoint (thinker subset will be used)")

    gw = gguf.GGUFWriter(out, "qwen3omni")
    write_metadata(gw, text, vis, aud, root, args.outtype)

    stats = {"quantized": 0, "f16": 0, "f32": 0}

    def add_tensor(name, hf_name, quantize=False, f32=False):
        arr = weights.get(hf_name)
        if f32:
            gw.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
            stats["f32"] += 1
            return
        blocks = quantize_text(arr) if quantize else None
        if blocks is None:
            _lib.add_f16(gw, name, arr)
            stats["f16"] += 1
        else:
            _lib.add_quantized(gw, name, blocks, qtype)
            stats["quantized"] += 1

    def add_exps(name_fmt, p, leaf, n_in):
        """Stack n_experts per-expert HF [out, in] arrays and write one
        quantized tensor. Byte order: expert-major concatenation of each
        expert's flat ne=[in,out] block (= np.stack axis 0, C order)."""
        parts = [weights.get(f"{p}.experts.{e}.{leaf}.weight") for e in range(n_exp)]
        arr = np.stack(parts, axis=0).astype(np.float32)          # [E, out, in]
        assert arr.shape[-1] == n_in and n_in % 256 == 0
        flat = arr.reshape(-1, n_in)
        q = q6k.quantize_q6_k(flat) if qtype == gguf.GGMLQuantizationType.Q6_K             else _lib.quantize(flat, qtype)
        gw.add_tensor(name_fmt, q, raw_shape=(n_exp, arr.shape[1], q.shape[-1]), raw_dtype=qtype)
        stats["quantized"] += 1

    # ---------------------------------------------------------- text decoder
    tm = "thinker.model"
    log("text decoder")
    add_tensor("token_embd.weight", f"{tm}.embed_tokens.weight", quantize=True)
    add_tensor("output_norm.weight", f"{tm}.norm.weight")
    add_tensor("lm_head.weight", "thinker.lm_head.weight", quantize=True)
    for i in range(n_l):
        p, b = f"{tm}.layers.{i}", f"blk.{i}"
        add_tensor(f"{b}.attn_norm.weight", f"{p}.input_layernorm.weight")
        add_tensor(f"{b}.attn_q.weight", f"{p}.self_attn.q_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_k.weight", f"{p}.self_attn.k_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_v.weight", f"{p}.self_attn.v_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_output.weight", f"{p}.self_attn.o_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_q_norm.weight", f"{p}.self_attn.q_norm.weight")
        add_tensor(f"{b}.attn_k_norm.weight", f"{p}.self_attn.k_norm.weight")
        add_tensor(f"{b}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight")
        add_tensor(f"{b}.ffn_router.weight", f"{p}.mlp.gate.weight", f32=True)
        add_exps(f"{b}.ffn_gate_exps.weight", f"{p}.mlp", "gate_proj", text["hidden_size"])
        add_exps(f"{b}.ffn_up_exps.weight", f"{p}.mlp", "up_proj", text["hidden_size"])
        add_exps(f"{b}.ffn_down_exps.weight", f"{p}.mlp", "down_proj", moe_i)
        if not args.quiet:
            progress(i, n_l, "text layer", every=4)

    # ---------------------------------------------------------- vision tower
    vt = "thinker.visual"
    log("vision tower")
    conv = weights.get(f"{vt}.patch_embed.proj.weight")      # [1152, 3, 2, 16, 16]
    out_dim, c, kt, kh, kw = conv.shape
    _lib.add_f16(gw, "v.patch_embd.weight", conv.reshape(out_dim, c * kt * kh * kw))
    stats["f16"] += 1
    add_tensor("v.patch_embd.bias", f"{vt}.patch_embed.proj.bias")
    add_tensor("v.pos_emb.weight", f"{vt}.pos_embed.weight", f32=True)   # [2304,1152], host interpolates
    vision_suffixes = ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                       "attn.qkv.weight", "attn.qkv.bias",
                       "attn.proj.weight", "attn.proj.bias",
                       "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
                       "mlp.linear_fc2.weight", "mlp.linear_fc2.bias")
    for i in range(vis["depth"]):
        p, b = f"{vt}.blocks.{i}", f"v.blk.{i}"
        for suffix in vision_suffixes:
            add_tensor(f"{b}.{suffix.replace('.', '_')}", f"{p}.{suffix}")
        if not args.quiet:
            progress(i, vis["depth"], "vision layer", every=6)
    # mergers: main (pre-shuffle norm) + deepstack (post-shuffle norm)
    add_tensor("mm.ln_q.weight", f"{vt}.merger.ln_q.weight")
    add_tensor("mm.ln_q.bias", f"{vt}.merger.ln_q.bias")
    add_tensor("mm.fc1.weight", f"{vt}.merger.mlp.0.weight")
    add_tensor("mm.fc1.bias", f"{vt}.merger.mlp.0.bias")
    add_tensor("mm.fc2.weight", f"{vt}.merger.mlp.2.weight")
    add_tensor("mm.fc2.bias", f"{vt}.merger.mlp.2.bias")
    for k in range(len(vis["deepstack_visual_indexes"])):
        add_tensor(f"mm_ds.{k}.ln_q.weight", f"{vt}.merger_list.{k}.ln_q.weight")
        add_tensor(f"mm_ds.{k}.ln_q.bias", f"{vt}.merger_list.{k}.ln_q.bias")
        add_tensor(f"mm_ds.{k}.fc1.weight", f"{vt}.merger_list.{k}.mlp.0.weight")
        add_tensor(f"mm_ds.{k}.fc1.bias", f"{vt}.merger_list.{k}.mlp.0.bias")
        add_tensor(f"mm_ds.{k}.fc2.weight", f"{vt}.merger_list.{k}.mlp.2.weight")
        add_tensor(f"mm_ds.{k}.fc2.bias", f"{vt}.merger_list.{k}.mlp.2.bias")

    # ---------------------------------------------------------- audio tower
    at = "thinker.audio_tower"
    log("audio tower")
    # (conv weights keep their [out, in, 3, 3] shape; engine does im2col folding)
    add_tensor("a.conv1.weight", f"{at}.conv2d1.weight")
    add_tensor("a.conv1.bias", f"{at}.conv2d1.bias")
    add_tensor("a.conv2.weight", f"{at}.conv2d2.weight")
    add_tensor("a.conv2.bias", f"{at}.conv2d2.bias")
    add_tensor("a.conv3.weight", f"{at}.conv2d3.weight")
    add_tensor("a.conv3.bias", f"{at}.conv2d3.bias")
    add_tensor("a.conv_out.weight", f"{at}.conv_out.weight")
    audio_suffixes = ("self_attn_layer_norm.weight", "self_attn_layer_norm.bias",
                      "self_attn.q_proj.weight", "self_attn.q_proj.bias",
                      "self_attn.k_proj.weight", "self_attn.k_proj.bias",
                      "self_attn.v_proj.weight", "self_attn.v_proj.bias",
                      "self_attn.out_proj.weight", "self_attn.out_proj.bias",
                      "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
                      "final_layer_norm.weight", "final_layer_norm.bias")
    amap = {"self_attn_layer_norm": "attn_ln", "final_layer_norm": "ffn_ln",
            "fc1": "ffn_fc1", "fc2": "ffn_fc2",
            "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
            "self_attn.v_proj": "attn_v", "self_attn.out_proj": "attn_o"}
    for i in range(aud["encoder_layers"]):
        p, b = f"{at}.layers.{i}", f"a.blk.{i}"
        for suffix in audio_suffixes:
            mod, leaf = suffix.rsplit(".", 1)
            add_tensor(f"{b}.{amap[mod]}.{leaf}", f"{p}.{suffix}")
        if not args.quiet:
            progress(i, aud["encoder_layers"], "audio layer", every=8)
    add_tensor("a.ln_post.weight", f"{at}.ln_post.weight")
    add_tensor("a.ln_post.bias", f"{at}.ln_post.bias")
    add_tensor("a.proj1.weight", f"{at}.proj1.weight")
    add_tensor("a.proj1.bias", f"{at}.proj1.bias")
    add_tensor("a.proj2.weight", f"{at}.proj2.weight")
    add_tensor("a.proj2.bias", f"{at}.proj2.bias")

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    log(f"done: {out} ({os.path.getsize(out) / 1e9:.2f} GB, "
        f"{stats['quantized']} quantized / {stats['f16']} f16 / {stats['f32']} f32 tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
