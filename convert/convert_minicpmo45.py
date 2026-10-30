"""Convert openbmb/MiniCPM-o-4.5 (BF16 safetensors) to GGUF.

Usage::

  python convert/convert_minicpmo45.py [--src DIR] [-o OUT.gguf]
                                       [--out-type f16|q8_0|q6_k]

Source: E:/SkyVision-weights/minicpmo45 (4 BF16 shards, 1414 tensors,
18.74 GB) -- openbmb/MiniCPM-o-4_5. The converter is strictly streaming:
shards are mmap'd, every tensor is transformed and flushed to the GGUF data
section one at a time (peak RSS stays in the low GBs -- the 18.7 GB
checkpoint must never be resident).

Layout (``general.architecture = minicpmo``, the checkpoint's model_type),
per minicpmo45-port/arch-spec.md (tensor contract section at the end):

  llm (plain Qwen3-8B dense: 36L, 4096, 32q/8kv GQA, head_dim 128, theta 1e6,
       untied, no biases, q/k RMSNorm per head_dim)
    token_embd.weight        [151748, 4096]
    lm_head.weight           [151748, 4096]   separate tensor (tie=false)
    output_norm.weight       [4096]  F32      RMSNorm eps 1e-6
    blk.{i}.attn_norm.weight [4096]  F32
    blk.{i}.attn_q/k/v.weight    [4096|1024|1024, 4096]
    blk.{i}.attn_output.weight   [4096, 4096]
    blk.{i}.attn_q_norm/attn_k_norm.weight [128] F32
    blk.{i}.ffn_norm.weight  [4096]  F32
    blk.{i}.ffn_gate/up.weight   [12288, 4096]
    blk.{i}.ffn_down.weight      [4096, 12288]

  vision (SigLIP-SO400M NaFiT, 27L x 1152 @448px, patch 14; names follow
          convert_smolvlm.py, the engine's SigLIP-family precedent)
    v.patch_embd.weight      [1152, 588]      Conv2d [1152,3,14,14] folded
    v.patch_embd.bias        [1152]  F32
    v.pos_embd.weight        [4900, 1152] F32 learned 70x70 NaFiT table
                             (host slices [h_patch, w_patch] per image)
    v.blk.{i}.ln1/ln2.{weight,bias}           LayerNorm F32 (eps 1e-6)
    v.blk.{i}.wq/bq, wk/bk, wv/bv             separate q/k/v + bias
    v.blk.{i}.wo/bo                           attention out + bias
    v.blk.{i}.fc1.weight/bias [4304,1152]     gelu_pytorch_tanh MLP
    v.blk.{i}.fc2.weight/bias [1152,4304]
    v.post_ln.weight/bias    [1152]  F32

  resampler (64-query perceiver, the vision->LLM projector; no engine
             precedent, fresh `rs.` namespace)
    rs.query                 [64, 4096]  F32   learned queries
    rs.kv_proj.weight        [4096, 1152]      Linear, no bias
    rs.attn.in_proj_weight   [12288, 4096]     nn.MultiheadAttention fused
    rs.attn.in_proj_bias     [12288]  F32      q|k|v packed rows 0:4096:8192
    rs.attn.out_proj.weight  [4096, 4096]
    rs.attn.out_proj.bias    [4096]  F32
    rs.ln_q/ln_kv/ln_post.{weight,bias} [4096] F32 (eps 1e-6)
    rs.proj.weight           [4096, 4096]      bare Parameter; applied as
                                               x @ proj (RIGHT multiply)

  audio (whisper-medium encoder, 24L x 1024, 80 mel; names follow
         convert_qwen3omni.py audio + nemotronomni a.proj)
    a.conv1.weight           [1024, 80, 3]     Conv1d k3 s1, kept 3-D (the
    a.conv1.bias             [1024]  F32       engine im2col-folds, qwen3omni
    a.conv2.weight           [1024, 1024, 3]   precedent)
    a.conv2.bias             [1024]  F32
    a.pos_embd.weight        [1500, 1024] F32  sinusoid table (registered buffer)
    a.blk.{i}.attn_ln.{weight,bias}           post-attn LayerNorm F32
    a.blk.{i}.attn_q/k/v/o.{weight,bias}      1024x1024, bias everywhere
    a.blk.{i}.ffn_ln.{weight,bias}            post-ffn LayerNorm F32
    a.blk.{i}.ffn_fc1/ffn_fc2.{weight,bias}   4096 GELU MLP
    a.ln_post.weight/bias    [1024]  F32       final encoder LayerNorm
    a.proj.fc1.weight/bias   [4096, 1024]      audio_projection_layer.linear1
    a.proj.fc2.weight/bias   [4096, 4096]      .linear2 (ReLU in between);
                                               engine adds AvgPool1d(5) after

  tts (speech-out sub-tower, in the main safetensors; names follow the
       convert_qwen3omni_talker.py `tk.` precedent)
    tk.emb_text.weight       [152064, 768]     separate text-embedding table
    tk.emb_code.weight       [6562, 768]       speech codebook (6561+eos, 1 VQ)
    tk.head_code.weight      [6562, 768]       weight_norm Linear RECOMBINED
                                               from parametrizations.weight
                                               .original0 [6562,1] (g) and
                                               .original1 [6562,768] (v):
                                               w = g * v / ||v||_row
    tk.output_norm.weight    [768]  F32        tts.model.norm
    tk.blk.{i}.attn_norm / attn_q/k/v / attn_output / ffn_norm /
    tk.blk.{i}.ffn_gate/up/down                plain Llama-768 x 20L, no bias,
                                               no q/k-norm
    tk.proj_semantic.fc1/fc2.{weight,bias}    4096 -> 768 -> 768 ReLU MLP
    tk.proj_spk.fc1/fc2.{weight,bias}         speaker-condition twin

  deliberately NOT converted (documented skip, diff set stays empty):
    tts.model.embed_tokens.weight  [32000, 768] -- the LlamaModel backbone's
      default-vocab embedding, never referenced at inference (the backbone is
      driven by inputs_embeds: tk.emb_text merges + tk.emb_code; the official
      code only ever calls self.model(inputs_embeds=...))

  also NOT in this GGUF (different conversion wave): the CosyVoice2
  token2wav vocoder assets under assets/token2wav/ (flow.pt 623MB + hift.pt
  83MB + speech_tokenizer_v2_25hz.onnx 496MB + campplus.onnx 28MB). They are
  .pt/.onnx, independent of the safetensors; when the engine ports them they
  become a SECOND GGUF (flow.yaml conformer 6+4 + DiT16 CFM + HiFiGAN) --
  this file's layout deliberately leaves the `rs.`/`tk.`/`v.`/`a.`/`blk.`
  namespaces unambiguous so a later token2wav-* GGUF can coexist.

The tokenizer is not embedded in the GGUF: the engine reads the checkpoint's
tokenizer.json directly (model_registry), same as qwen3omni/nemotronomni.

Precision contract: every matmul operand follows --out-type; all
norms/biases/pos-embed tables/learned queries are F32 regardless of
--out-type; spatial convs and the folded patch embed (last-dim 3/588, not
%32) are F16 in every tier. Row-width note for the quant tiers: the LLM
(4096/12288/1024), audio (1024/4096) and TTS (768/3072) widths are %256, but
the vision tower is NOT (1152 = 4.5x256; mlp fc2 in-width 4304 is not even
%32), so a uniform Q8_0 tier must fall back to F16 for v.blk.fc2 and a
uniform Q6_K also for every other 1152-in tensor. Tiers:
  f16        18.7 GB parity baseline (everything 2-D at F16)
  q8_0       ~9.5 GB uniform tier (v.blk.*.fc2.weight stays F16)
  q6_k       ~7.5 GB, additionally F16 fallbacks at every %256 != 0 row
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import gguf

import _lib
import _plan
from _lib import log, warn, fail, progress
from _plan import (QUANT_BLOCK, Shards, expand_bf16 as _expand_bf16,
                   peak_rss_mib)

_lib.torch = None   # numpy bf16 decode path: exact and proven (convert_qwen3omni)

ARCH = "minicpmo"   # = config.json model_type
MODEL_ID = "openbmb/MiniCPM-o-4_5"
DEFAULT_SRC = r"E:\SkyVision-weights\minicpmo45"
DEFAULT_OUT_DIR = r"E:\SkyVision-weights\minicpmo45"

OUTTYPES = {"f16": (None, gguf.LlamaFileType.MOSTLY_F16),
            "q8_0": (None, gguf.LlamaFileType.MOSTLY_Q8_0),
            "q6_k": (None, gguf.LlamaFileType.MOSTLY_Q6_K)}

SKIP_EXACT = ("tts.model.embed_tokens.weight",)   # dead LlamaModel default embedding
SKIP_SUFFIX = ()

LP = "llm.model"
VP = "vpm"
AP = "apm"
TP = "tts.model"


def quant_policy(ops, out_type):
    """Assign per-op quantized types (op["q"], kind "q") for the quant tiers.
    f16 mode is a no-op; F32/F16-contract ops always keep their precision."""
    if out_type not in ("q8_0", "q6_k"):
        return
    for op in ops:
        if op["k"] not in ("f16", "wn"):
            continue                              # f32/conv precision contract
        row = op["s"][-1]
        if out_type == "q8_0" and row % 32 == 0:
            qt = gguf.GGMLQuantizationType.Q8_0
        elif out_type == "q6_k" and row % 256 == 0:
            qt = gguf.GGMLQuantizationType.Q6_K
        else:
            warn(f"{op['g']}: row {row} not quantizable in {out_type}, keeping F16")
            continue
        if op["k"] == "wn":
            op["q"] = qt       # kind stays "wn"; materialize quantizes on op["q"]
        else:
            op["k"] = "q"
            op["q"] = qt


def read_config(src):
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    v = cfg["vision_config"]
    a = cfg["audio_config"]
    t = cfg["tts_config"]
    return dict(
        n_layer=cfg["num_hidden_layers"], n_embd=cfg["hidden_size"],
        n_head=cfg["num_attention_heads"], n_kv_head=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"], ff=cfg["intermediate_size"],
        vocab=cfg["vocab_size"], eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"], ctx=cfg["max_position_embeddings"],
        tie=cfg["tie_word_embeddings"], bos=cfg["bos_token_id"], eos=cfg["eos_token_id"],
        pad=cfg.get("bos_token_id", 151643),
        v_blocks=v["num_hidden_layers"], v_embd=v["hidden_size"],
        v_ff=v["intermediate_size"], v_heads=v["num_attention_heads"],
        v_patch=v["patch_size"], v_eps=v["layer_norm_eps"],
        image_size=cfg["image_size"], query_num=cfg["query_num"],
        a_blocks=a["encoder_layers"], a_embd=a["d_model"], a_ff=a["encoder_ffn_dim"],
        a_heads=a["encoder_attention_heads"], a_mel=a["num_mel_bins"],
        a_pos=a["max_source_positions"],
        audio_chunk=cfg["audio_chunk_length"], pool_step=cfg["audio_pool_step"],
        t_layers=t["num_hidden_layers"], t_embd=t["hidden_size"],
        t_ff=t["intermediate_size"], t_heads=t["num_attention_heads"],
        t_kv=t["num_key_value_heads"], t_vocab=t["num_text_tokens"],
        t_audio_vocab=t["num_audio_tokens"], t_vq=t["num_vq"],
        t_audio_bos=t["audio_bos_token_id"], t_text_eos=t["text_eos_token_id"],
    )


# --------------------------------------------------------------------- plan
# op kinds: f16 = matmul operand (out-type precision, quantizable),
# f32 = precision-critical small tensor (norms/biases/pos tables/queries),
# conv = spatial conv or folded patch embed, F16 in every tier (last dim
#        3/588 is not %32, never quantizable),
# wn   = weight_norm recombine (matmul operand, quantizable). t = transform.
def build_ops(cfg):
    ops = []
    f16, f32, conv = "f16", "f32", "conv"

    def add(g, h, k, s, t=None):
        ops.append(dict(g=g, h=h, k=k, s=tuple(s) if s is not None else None, t=t))

    # ---------------------------------------------------------- llm (Qwen3-8B)
    add("token_embd.weight", f"{LP}.embed_tokens.weight", f16,
        (cfg["vocab"], cfg["n_embd"]))
    add("output_norm.weight", f"{LP}.norm.weight", f32, (cfg["n_embd"],))
    for i in range(cfg["n_layer"]):
        p, b = f"{LP}.layers.{i}", f"blk.{i}"
        add(f"{b}.attn_norm.weight", f"{p}.input_layernorm.weight", f32, (cfg["n_embd"],))
        add(f"{b}.attn_q.weight", f"{p}.self_attn.q_proj.weight", f16,
            (cfg["n_head"] * cfg["head_dim"], cfg["n_embd"]))
        add(f"{b}.attn_k.weight", f"{p}.self_attn.k_proj.weight", f16,
            (cfg["n_kv_head"] * cfg["head_dim"], cfg["n_embd"]))
        add(f"{b}.attn_v.weight", f"{p}.self_attn.v_proj.weight", f16,
            (cfg["n_kv_head"] * cfg["head_dim"], cfg["n_embd"]))
        add(f"{b}.attn_output.weight", f"{p}.self_attn.o_proj.weight", f16,
            (cfg["n_embd"], cfg["n_head"] * cfg["head_dim"]))
        add(f"{b}.attn_q_norm.weight", f"{p}.self_attn.q_norm.weight", f32,
            (cfg["head_dim"],))
        add(f"{b}.attn_k_norm.weight", f"{p}.self_attn.k_norm.weight", f32,
            (cfg["head_dim"],))
        add(f"{b}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight", f32,
            (cfg["n_embd"],))
        add(f"{b}.ffn_gate.weight", f"{p}.mlp.gate_proj.weight", f16,
            (cfg["ff"], cfg["n_embd"]))
        add(f"{b}.ffn_up.weight", f"{p}.mlp.up_proj.weight", f16,
            (cfg["ff"], cfg["n_embd"]))
        add(f"{b}.ffn_down.weight", f"{p}.mlp.down_proj.weight", f16,
            (cfg["n_embd"], cfg["ff"]))
    if not cfg["tie"]:
        add("lm_head.weight", "llm.lm_head.weight", f16,
            (cfg["vocab"], cfg["n_embd"]))
    else:
        fail("tie_word_embeddings=true unexpected for MiniCPM-o-4.5; extend the plan")

    # ------------------------------------------------------- vision (SigLIP NaFiT)
    add("v.patch_embd.weight", f"{VP}.embeddings.patch_embedding.weight", conv,
        (cfg["v_embd"], 3 * cfg["v_patch"] * cfg["v_patch"]), t="fold")
    add("v.patch_embd.bias", f"{VP}.embeddings.patch_embedding.bias", f32,
        (cfg["v_embd"],))
    add("v.pos_embd.weight", f"{VP}.embeddings.position_embedding.weight", f32,
        None)   # shape verified against the checkpoint in main (70x70 grid)
    for i in range(cfg["v_blocks"]):
        p, b = f"{VP}.encoder.layers.{i}", f"v.blk.{i}"
        for ln, hf in (("ln1", "layer_norm1"), ("ln2", "layer_norm2")):
            add(f"{b}.{ln}.weight", f"{p}.{hf}.weight", f32, (cfg["v_embd"],))
            add(f"{b}.{ln}.bias", f"{p}.{hf}.bias", f32, (cfg["v_embd"],))
        add(f"{b}.wq.weight", f"{p}.self_attn.q_proj.weight", f16,
            (cfg["v_embd"], cfg["v_embd"]))
        add(f"{b}.bq", f"{p}.self_attn.q_proj.bias", f32, (cfg["v_embd"],))
        add(f"{b}.wk.weight", f"{p}.self_attn.k_proj.weight", f16,
            (cfg["v_embd"], cfg["v_embd"]))
        add(f"{b}.bk", f"{p}.self_attn.k_proj.bias", f32, (cfg["v_embd"],))
        add(f"{b}.wv.weight", f"{p}.self_attn.v_proj.weight", f16,
            (cfg["v_embd"], cfg["v_embd"]))
        add(f"{b}.bv", f"{p}.self_attn.v_proj.bias", f32, (cfg["v_embd"],))
        add(f"{b}.wo.weight", f"{p}.self_attn.out_proj.weight", f16,
            (cfg["v_embd"], cfg["v_embd"]))
        add(f"{b}.bo", f"{p}.self_attn.out_proj.bias", f32, (cfg["v_embd"],))
        add(f"{b}.fc1.weight", f"{p}.mlp.fc1.weight", f16,
            (cfg["v_ff"], cfg["v_embd"]))
        add(f"{b}.fc1.bias", f"{p}.mlp.fc1.bias", f32, (cfg["v_ff"],))
        add(f"{b}.fc2.weight", f"{p}.mlp.fc2.weight", f16,
            (cfg["v_embd"], cfg["v_ff"]))
        add(f"{b}.fc2.bias", f"{p}.mlp.fc2.bias", f32, (cfg["v_embd"],))
    add("v.post_ln.weight", f"{VP}.post_layernorm.weight", f32, (cfg["v_embd"],))
    add("v.post_ln.bias", f"{VP}.post_layernorm.bias", f32, (cfg["v_embd"],))

    # ---------------------------------------------------------- resampler
    add("rs.query", "resampler.query", f32, (cfg["query_num"], cfg["n_embd"]))
    add("rs.kv_proj.weight", "resampler.kv_proj.weight", f16,
        (cfg["n_embd"], cfg["v_embd"]))
    add("rs.attn.in_proj_weight", "resampler.attn.in_proj_weight", f16,
        (3 * cfg["n_embd"], cfg["n_embd"]))
    add("rs.attn.in_proj_bias", "resampler.attn.in_proj_bias", f32,
        (3 * cfg["n_embd"],))
    add("rs.attn.out_proj.weight", "resampler.attn.out_proj.weight", f16,
        (cfg["n_embd"], cfg["n_embd"]))
    add("rs.attn.out_proj.bias", "resampler.attn.out_proj.bias", f32,
        (cfg["n_embd"],))
    for ln in ("ln_q", "ln_kv", "ln_post"):
        add(f"rs.{ln}.weight", f"resampler.{ln}.weight", f32, (cfg["n_embd"],))
        add(f"rs.{ln}.bias", f"resampler.{ln}.bias", f32, (cfg["n_embd"],))
    add("rs.proj.weight", "resampler.proj", f16, (cfg["n_embd"], cfg["n_embd"]))

    # ---------------------------------------------------------- audio (whisper-medium)
    add("a.conv1.weight", f"{AP}.conv1.weight", conv,
        (cfg["a_embd"], cfg["a_mel"], 3))
    add("a.conv1.bias", f"{AP}.conv1.bias", f32, (cfg["a_embd"],))
    add("a.conv2.weight", f"{AP}.conv2.weight", conv,
        (cfg["a_embd"], cfg["a_embd"], 3))
    add("a.conv2.bias", f"{AP}.conv2.bias", f32, (cfg["a_embd"],))
    add("a.pos_embd.weight", f"{AP}.embed_positions.weight", f32,
        (cfg["a_pos"], cfg["a_embd"]))
    for i in range(cfg["a_blocks"]):
        p, b = f"{AP}.layers.{i}", f"a.blk.{i}"
        add(f"{b}.attn_ln.weight", f"{p}.self_attn_layer_norm.weight", f32,
            (cfg["a_embd"],))
        add(f"{b}.attn_ln.bias", f"{p}.self_attn_layer_norm.bias", f32,
            (cfg["a_embd"],))
        # whisper-medium here carries NO k_proj bias (q/v/out do): 15 tensors
        # per layer = 2x2 LayerNorms + fc1/fc2 + q,v,out (w+b) + k (w only)
        add(f"{b}.attn_q.weight", f"{p}.self_attn.q_proj.weight", f16,
            (cfg["a_embd"], cfg["a_embd"]))
        add(f"{b}.attn_q.bias", f"{p}.self_attn.q_proj.bias", f32, (cfg["a_embd"],))
        add(f"{b}.attn_k.weight", f"{p}.self_attn.k_proj.weight", f16,
            (cfg["a_embd"], cfg["a_embd"]))
        add(f"{b}.attn_v.weight", f"{p}.self_attn.v_proj.weight", f16,
            (cfg["a_embd"], cfg["a_embd"]))
        add(f"{b}.attn_v.bias", f"{p}.self_attn.v_proj.bias", f32, (cfg["a_embd"],))
        add(f"{b}.attn_o.weight", f"{p}.self_attn.out_proj.weight", f16,
            (cfg["a_embd"], cfg["a_embd"]))
        add(f"{b}.attn_o.bias", f"{p}.self_attn.out_proj.bias", f32, (cfg["a_embd"],))
        add(f"{b}.ffn_ln.weight", f"{p}.final_layer_norm.weight", f32,
            (cfg["a_embd"],))
        add(f"{b}.ffn_ln.bias", f"{p}.final_layer_norm.bias", f32, (cfg["a_embd"],))
        add(f"{b}.ffn_fc1.weight", f"{p}.fc1.weight", f16,
            (cfg["a_ff"], cfg["a_embd"]))
        add(f"{b}.ffn_fc1.bias", f"{p}.fc1.bias", f32, (cfg["a_ff"],))
        add(f"{b}.ffn_fc2.weight", f"{p}.fc2.weight", f16,
            (cfg["a_embd"], cfg["a_ff"]))
        add(f"{b}.ffn_fc2.bias", f"{p}.fc2.bias", f32, (cfg["a_embd"],))
    add("a.ln_post.weight", f"{AP}.layer_norm.weight", f32, (cfg["a_embd"],))
    add("a.ln_post.bias", f"{AP}.layer_norm.bias", f32, (cfg["a_embd"],))
    add("a.proj.fc1.weight", "audio_projection_layer.linear1.weight", f16,
        (cfg["n_embd"], cfg["a_embd"]))
    add("a.proj.fc1.bias", "audio_projection_layer.linear1.bias", f32,
        (cfg["n_embd"],))
    add("a.proj.fc2.weight", "audio_projection_layer.linear2.weight", f16,
        (cfg["n_embd"], cfg["n_embd"]))
    add("a.proj.fc2.bias", "audio_projection_layer.linear2.bias", f32,
        (cfg["n_embd"],))

    # ---------------------------------------------------------- tts sub-tower
    add("tk.emb_text.weight", "tts.emb_text.weight", f16,
        (cfg["t_vocab"], cfg["t_embd"]))
    add("tk.emb_code.weight", "tts.emb_code.0.weight", f16,
        (cfg["t_audio_vocab"], cfg["t_embd"]))
    add("tk.head_code.weight", "tts.head_code.0", "wn",
        (cfg["t_audio_vocab"], cfg["t_embd"]))
    add("tk.output_norm.weight", f"{TP}.norm.weight", f32, (cfg["t_embd"],))
    for i in range(cfg["t_layers"]):
        p, b = f"{TP}.layers.{i}", f"tk.blk.{i}"
        add(f"{b}.attn_norm.weight", f"{p}.input_layernorm.weight", f32,
            (cfg["t_embd"],))
        for w, hf in (("attn_q", "q_proj"), ("attn_k", "k_proj"),
                      ("attn_v", "v_proj"), ("attn_output", "o_proj")):
            add(f"{b}.{w}.weight", f"{p}.self_attn.{hf}.weight", f16,
                (cfg["t_embd"], cfg["t_embd"]))
        add(f"{b}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight", f32,
            (cfg["t_embd"],))
        add(f"{b}.ffn_gate.weight", f"{p}.mlp.gate_proj.weight", f16,
            (cfg["t_ff"], cfg["t_embd"]))
        add(f"{b}.ffn_up.weight", f"{p}.mlp.up_proj.weight", f16,
            (cfg["t_ff"], cfg["t_embd"]))
        add(f"{b}.ffn_down.weight", f"{p}.mlp.down_proj.weight", f16,
            (cfg["t_embd"], cfg["t_ff"]))
    for pj in ("semantic", "spk"):
        add(f"tk.proj_{pj}.fc1.weight", f"tts.projector_{pj}.linear1.weight", f16,
            (cfg["t_embd"], cfg["n_embd"]))
        add(f"tk.proj_{pj}.fc1.bias", f"tts.projector_{pj}.linear1.bias", f32,
            (cfg["t_embd"],))
        add(f"tk.proj_{pj}.fc2.weight", f"tts.projector_{pj}.linear2.weight", f16,
            (cfg["t_embd"], cfg["t_embd"]))
        add(f"tk.proj_{pj}.fc2.bias", f"tts.projector_{pj}.linear2.bias", f32,
            (cfg["t_embd"],))
    return ops


def materialize(op, sh):
    """Build the exact final tensor for one op (mmap -> transform -> dtype)."""
    if op["k"] == "wn":                        # weight_norm recombine
        g_raw, g_dt, _ = sh.raw(op["h"] + ".parametrizations.weight.original0")
        v_raw, v_dt, v_shape = sh.raw(op["h"] + ".parametrizations.weight.original1")
        g = np.empty(g_raw.size, dtype=np.float32)
        if g_dt == "BF16":
            _expand_bf16(g_raw, g)
        else:
            g[:] = g_raw.reshape(-1)
        v = np.empty(v_raw.size, dtype=np.float32).reshape(v_shape)
        if v_dt == "BF16":
            _expand_bf16(v_raw, v)
        else:
            v[:] = v_raw.reshape(v_shape)
        w = g.reshape(-1, 1) * v / np.sqrt((v * v).sum(-1, keepdims=True))
        del v
        if op.get("q") is not None:
            return _lib.quantize(np.ascontiguousarray(w), op["q"])
        return w.astype(np.float16)

    raw, dt, hf_shape = sh.raw(op["h"])
    if op["t"] == "fold":                      # Conv2d [out,3,14,14] -> [out, 588]
        n = int(np.prod(hf_shape[1:]))
        raw, hf_shape = raw.reshape(hf_shape[0], n), (hf_shape[0], n)

    if op["s"] is not None and tuple(hf_shape) != op["s"]:
        fail(f"{op['h']}: shape {hf_shape} != planned {op['s']}")

    if op["k"] in ("f16", "conv"):
        out = np.empty(raw.size, dtype=np.float16)
        if dt == "BF16":
            _expand_bf16(raw, out)
        else:
            out[:] = raw.reshape(-1)
        return out.reshape(op["s"])
    if op["k"] == "q":                         # per-op quantized matmul operand
        out = np.empty(raw.size, dtype=np.float32)
        if dt == "BF16":
            _expand_bf16(raw, out)
        else:
            out[:] = raw.reshape(-1)
        blocks = _lib.quantize(np.ascontiguousarray(out.reshape(op["s"])), op["q"])
        del out
        return blocks
    if op["k"] == "f32":
        out = np.empty(raw.size, dtype=np.float32)
        if dt == "BF16":
            _expand_bf16(raw, out)
        else:
            out[:] = raw.reshape(-1)
        return out.reshape(op["s"])
    fail(f"unknown op kind {op['k']}")


# ---------------------------------------------------------------- metadata
def write_metadata(gw, cfg, outtype):
    ftype = OUTTYPES[outtype][1]
    gw.add_architecture()
    gw.add_name("MiniCPM-o-4.5")
    gw.add_file_type(ftype)
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32(f"{ARCH}.block_count", cfg["n_layer"])
    gw.add_uint32(f"{ARCH}.embedding_length", cfg["n_embd"])
    gw.add_uint32(f"{ARCH}.vocab_size", cfg["vocab"])
    gw.add_uint32(f"{ARCH}.context_length", cfg["ctx"])
    gw.add_uint32(f"{ARCH}.attention.head_count", cfg["n_head"])
    gw.add_uint32(f"{ARCH}.attention.head_count_kv", cfg["n_kv_head"])
    gw.add_uint32(f"{ARCH}.attention.key_length", cfg["head_dim"])
    gw.add_uint32(f"{ARCH}.attention.value_length", cfg["head_dim"])
    gw.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", float(cfg["eps"]))
    gw.add_uint32(f"{ARCH}.rope.dimension_count", cfg["head_dim"])
    gw.add_float32(f"{ARCH}.rope.freq_base", float(cfg["rope_theta"]))
    gw.add_bool(f"{ARCH}.tie_word_embeddings", cfg["tie"])
    gw.add_uint32(f"{ARCH}.bos_token_id", cfg["bos"])
    gw.add_uint32(f"{ARCH}.eos_token_id", cfg["eos"])          # 151645 <|im_end|>
    gw.add_uint32(f"{ARCH}.pad_token_id", cfg["pad"])

    gw.add_uint32(f"{ARCH}.vision.block_count", cfg["v_blocks"])
    gw.add_uint32(f"{ARCH}.vision.embedding_length", cfg["v_embd"])
    gw.add_uint32(f"{ARCH}.vision.feed_forward_length", cfg["v_ff"])
    gw.add_uint32(f"{ARCH}.vision.attention.head_count", cfg["v_heads"])
    gw.add_uint32(f"{ARCH}.vision.patch_size", cfg["v_patch"])
    gw.add_uint32(f"{ARCH}.vision.image_size", cfg["image_size"])
    gw.add_float32(f"{ARCH}.vision.layer_norm_epsilon", float(cfg["v_eps"]))

    gw.add_uint32(f"{ARCH}.vision.query_num", cfg["query_num"])
    gw.add_uint32(f"{ARCH}.vision.resampler_embed", cfg["n_embd"])

    gw.add_uint32(f"{ARCH}.audio.block_count", cfg["a_blocks"])
    gw.add_uint32(f"{ARCH}.audio.embedding_length", cfg["a_embd"])
    gw.add_uint32(f"{ARCH}.audio.feed_forward_length", cfg["a_ff"])
    gw.add_uint32(f"{ARCH}.audio.attention.head_count", cfg["a_heads"])
    gw.add_uint32(f"{ARCH}.audio.num_mel_bins", cfg["a_mel"])
    gw.add_uint32(f"{ARCH}.audio.max_source_positions", cfg["a_pos"])
    gw.add_float32(f"{ARCH}.audio.chunk_length", float(cfg["audio_chunk"]))
    gw.add_uint32(f"{ARCH}.audio.pool_step", cfg["pool_step"])

    gw.add_uint32(f"{ARCH}.tts.block_count", cfg["t_layers"])
    gw.add_uint32(f"{ARCH}.tts.embedding_length", cfg["t_embd"])
    gw.add_uint32(f"{ARCH}.tts.feed_forward_length", cfg["t_ff"])
    gw.add_uint32(f"{ARCH}.tts.attention.head_count", cfg["t_heads"])
    gw.add_uint32(f"{ARCH}.tts.attention.head_count_kv", cfg["t_kv"])
    gw.add_uint32(f"{ARCH}.tts.num_text_tokens", cfg["t_vocab"])
    gw.add_uint32(f"{ARCH}.tts.num_audio_tokens", cfg["t_audio_vocab"])
    gw.add_uint32(f"{ARCH}.tts.num_vq", cfg["t_vq"])
    gw.add_uint32(f"{ARCH}.tts.audio_bos_token_id", cfg["t_audio_bos"])
    gw.add_uint32(f"{ARCH}.tts.text_eos_token_id", cfg["t_text_eos"])


def op_decl(op):
    """(shape_for_info, np_dtype, nbytes, raw_dtype) of the final tensor."""
    # quantizable kinds: "wn" keeps its kind once the tier assigns a type; the
    # float16-dtype kinds are the f16 matmul operands, the conv contract, and
    # unquantized "wn"
    return _plan.op_decl(op, quant_kinds=("wn",), f16_kinds=("f16", "conv", "wn"))


def hf_names(op):
    """HF tensors consumed by one op (head_code merges two)."""
    if op["k"] == "wn":
        return [op["h"] + ".parametrizations.weight.original0",
                op["h"] + ".parametrizations.weight.original1"]
    return [op["h"]]


def verify_source(src, weight_map, sh):
    """Download-integrity guard: refuse partial/mixed shards before planning.

    .done markers are used when the downloader drops them (nemotronomni
    convention); regardless, every shard's file size must cover its
    safetensors header's last data offset."""
    files = sorted(set(weight_map.values()))
    missing_files = [f for f in files if not os.path.exists(os.path.join(src, f))]
    if missing_files:
        fail(f"{len(missing_files)} shard file(s) missing, e.g. {missing_files[:2]}")
    done = {f + ".done" for f in files}
    has_any_done = any(os.path.exists(os.path.join(src, d)) for d in done)
    if has_any_done:
        no_done = [f for f in files
                   if not os.path.exists(os.path.join(src, f + ".done"))]
        if no_done:
            fail(f"{len(no_done)} shard(s) missing .done markers "
                 f"(download incomplete?), e.g. {no_done[:2]}")
    import mmap, struct
    for fname in files:
        f = open(os.path.join(src, fname), "rb")
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        n = struct.unpack("<Q", m[:8])[0]
        meta = json.loads(m[8:8 + n])
        end = max(meta[k]["data_offsets"][1] for k in meta if k != "__metadata__")
        size = os.path.getsize(os.path.join(src, fname))
        m.close()
        f.close()
        if size < 8 + n + end:
            fail(f"{fname}: {size} B < header data end {8 + n + end} B "
                 f"(download incomplete -- wait for the downloader, never "
                 f"re-download from this script)")


def bind_pos_embd(ops, sh, cfg):
    """The stored NaFiT position table keeps the original SigLIP training grid
    (980px -> 70x70 = 4900 rows), not the 448px runtime image size; pin the
    plan's shape to what the checkpoint actually holds."""
    _, pos_shape = sh.meta("vpm.embeddings.position_embedding.weight")
    op = next(op for op in ops if op["g"] == "v.pos_embd.weight")
    if len(pos_shape) != 2 or pos_shape[1] != cfg["v_embd"]:
        fail(f"unexpected position_embedding shape {pos_shape}")
    op["s"] = tuple(pos_shape)


def write_gguf(out, cfg, ops, sh, outtype, quiet=True):
    """Stream the GGUF through the shared skeleton (pre-declared header, then
    the data section one op at a time); metadata + materialize stay here."""
    return _plan.write_gguf(out, ARCH, ops, sh,
                            lambda gw: write_metadata(gw, cfg, outtype),
                            op_decl, materialize, quiet=quiet)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-type", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the matmul operands (default f16; norms/"
                         "biases/pos tables stay F32, convs F16 regardless)")
    ap.add_argument("-o", "--out", default=None,
                    help=f"output file (default {DEFAULT_OUT_DIR}\\minicpmo45-"
                         f"<out-type>.gguf)")
    ap.add_argument("--src", default=os.environ.get("MINICPMO45_SRC", DEFAULT_SRC),
                    help="checkpoint dir (4 model-*.safetensors + index json)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    src = os.path.abspath(args.src)
    index = os.path.join(src, "model.safetensors.index.json")
    if not os.path.isfile(index):
        fail(f"no model.safetensors.index.json in {src}")
    with open(index, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    log(f"source: {src} ({len(set(weight_map.values()))} shards, "
        f"{len(weight_map)} tensors)")

    cfg = read_config(src)
    ops = build_ops(cfg)
    quant_policy(ops, args.out_type)
    _plan.log_quant_policy(ops, args.out_type)

    sh = Shards(src, weight_map)
    verify_source(src, weight_map, sh)
    bind_pos_embd(ops, sh, cfg)

    consumed, skipped = _plan.check_coverage(ops, weight_map, sh, hf_names,
                                             SKIP_EXACT, SKIP_SUFFIX)
    log(f"plan: {len(ops)} GGUF tensors <- {len(consumed)} HF tensors "
        f"({len(skipped)} skipped: tts.model.embed_tokens, dead backbone "
        f"embedding)")

    out = args.out or os.path.join(DEFAULT_OUT_DIR, f"minicpmo45-{args.out_type}.gguf")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    elapsed = write_gguf(out, cfg, ops, sh, args.out_type, args.quiet)
    sh.close()

    size = os.path.getsize(out)
    log(f"done: {out} ({size / 1e9:.2f} GB, {len(ops)} tensors, "
        f"{elapsed:.0f}s, peak RSS {peak_rss_mib():.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
