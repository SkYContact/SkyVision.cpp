"""Convert NVIDIA Nemotron-3-Nano-Omni-30B-A3B (BF16 safetensors) to GGUF.

Usage::

  python convert/convert_nemotronomni.py [--src DIR] [-o OUT.gguf]
                                         [--out-type f16|q8_0|q6k_mixed]
                                         [--rope off|partial84]

Source: E:/SkyVision-weights/nemotron3omni (17 BF16 shards, 7349 tensors,
66.03 GB) -- nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16. The converter
is strictly streaming: shards are mmap'd, every tensor is transformed and
flushed to the GGUF data section one at a time (peak RSS stays in the low
GBs -- the 66 GB checkpoint must never be resident).

Layout (``general.architecture = nemotronomni``), per port-plan §3/D2, tensor
names following llama.cpp nemotron-h(-moe) with the repo's custom bindings:

  text (52 hybrid layers, pattern "MEMEM*..."; 23 Mamba2 / 6 attn / 23 MoE)
    token_embd.weight        [2688, 131072]   untied
    lm_head.weight           [2688, 131072]   separate tensor (tie=false)
    output_norm.weight       [2688]  F32      RMSNorm eps 1e-5
    blk.{i}.attn_norm.weight [2688]  F32      every layer (pre-norm)
    Mamba2 (z|xBC|dt = 4096|6144|64 slice order is inside ssm_in):
      blk.{i}.ssm_in.weight     [10304, 2688]  in_proj, one piece
      blk.{i}.ssm_conv1d.weight [6144, 4]      depthwise k=4, flattened
      blk.{i}.ssm_conv1d.bias   [6144]  F32    added outside the conv op
      blk.{i}.ssm_dt.bias       [64]    F32    softplus(+dt) clamp(min 1e-3)
      blk.{i}.ssm_a             [64, 1] F32    A = -exp(A_log), ne[0]==1
      blk.{i}.ssm_d             [64, 1] F32
      blk.{i}.ssm_norm.weight   [512, 8] F32   grouped RMSNorm (8 groups x 512)
      blk.{i}.ssm_out.weight    [2688, 4096]
    attention (layers 5,12,19,26,33,42; GQA 32q/2kv hd128; NoPE):
      blk.{i}.attn_q/k/v.weight  [4096|256|256, 2688]
      blk.{i}.attn_o.weight      [2688, 4096]
    MoE (sigmoid x2.5 router, relu^2 experts, 1 shared):
      blk.{i}.ffn_gate_inp.weight  [2688, 128] F32  router
      blk.{i}.ffn_exp_probs_b      [128]       F32  e_score_correction_bias
      blk.{i}.ffn_up_exps.weight   [128, 1856, 2688]  expert-stacked
      blk.{i}.ffn_down_exps.weight [128, 2688, 1856]
      blk.{i}.ffn_up_shexp.weight  [3712, 2688]  shared expert (no 2.5)
      blk.{i}.ffn_down_shexp.weight [2688, 3712]

  vision (C-RADIOv4-H, 32 blocks d1280; names follow convert_nemotron_parse):
    v.patch_embd.weight      [1280, 768]      patch 16x16 linear
    v.video_embd.weight      [1280, 1536]     T=2 tubelet patch linear
    v.cls_token.weight       [10, 1280]  F32  cls+registers, dropped after
    v.pos_embd.weight        [16384, 1280] F32  learned 128x128 field
    v.blk.{i}.ln1/ln2.{weight,bias}           LayerNorm (F32)
    v.blk.{i}.wqkv/bqkv      [3840, 1280]     timm fused qkv + bias
    v.blk.{i}.wo/bo, fc0/fc1 (+bias)          GELU MLP 5120
    mm.ln / mm.fc0 / mm.fc1  mlp1: RMSNorm(5120) -> 20480 -> relu^2 -> 2688

  audio (Parakeet/FastConformer encoder + projection):
    a.mel.fb / a.mel.window                   F32 (checkpoint BF16 filterbank)
    a.sub.{0,2,4}.weight   conv2d 3x3 (0: 1->256, 2/4: depthwise groups=256)
    a.sub.{1,3}.weight     conv2d 1x1 pointwise (HF layers.3/6)
    a.sub.proj.weight      [1024, 4096]      flatten 256x16 -> 1024
    a.blk.{i}.ln_ff1/ln_att/ln_conv/ln_ff2/ln_out.{weight,bias}  LayerNorm F32
    a.blk.{i}.ff1.w1/w2, ff2.w1/w2           macaron FFN 4096 (silu)
    a.blk.{i}.att.q/k/v/o, att.rel_k, att.bias_u/v   rel-pos attention
    a.blk.{i}.conv.pw1 [2048,1024] -> GLU -> conv.dw [1024,9] -> BN -> silu
    a.blk.{i}.conv.pw2 [1024,1024]; conv.bn.{weight,bias,mean,var} F32 raw
                           (BatchNorm folding is done engine-side, D5b)
    a.proj.ln / a.proj.fc1 [4096,1024] / a.proj.fc2 [2688,4096]

  deliberately NOT converted (documented in port-plan §10):
    vision_model.radio_model.input_conditioner.norm_mean/std  (2 tensors;
      preprocessor normalization, Identity at runtime)
    *.conv.norm.num_batches_tracked  (24 BN counters)

The tokenizer is not embedded in the GGUF: the engine reads the checkpoint's
tokenizer.json directly (model_registry), same as qwen3omni.

Precision contract: every matmul operand follows --out-type; all
norms/biases/router/exp_probs_b/pos_embed/cls_token/mel/BN are F32
regardless of --out-type. Tiers:
  f16        66 GB parity baseline (everything 2-D at F16)
  q8_0       ~35 GB uniform full-compat tier
  q6k_mixed  ~23 GB per-tensor hybrid standing in for a uniform k-quant tier
             (trunk widths 2688/1856/3712 are not %256, so k-quants can only
             cover ~5% of the bytes): token_embd/lm_head Q8_0, expert stacks
             Q5_0, in%256 tensors Q6_K, remaining %32 tensors Q5_0, and the
             host-read tails (4-tap ssm conv, 4D audio subsampling convs) F16.
             The Q6_K encoder is the ported ggml reference (q6k.py), verified
             bit-exact against the engine dequantizer.
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

ARCH = "nemotronomni"
MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
DEFAULT_SRC = r"E:\SkyVision-weights\nemotron3omni"
DEFAULT_OUT_DIR = r"E:\SkyVision-weights\nemotronomni"

# quantized tiers. "q8_0" is the uniform full-compat baseline; "q6k_mixed" is
# the per-tensor hybrid that stands in for a uniform Q4_K/Q6_K tier (the trunk
# widths 2688/1856/3712 are not multiples of 256, so k-quants can only cover
# 5% of the bytes -- see 6prep/quant-compat.md):
#   token_embd / lm_head      -> Q8_0 (output head + registry precision)
#   expert stacks (exps_*)    -> Q5_0
#   2D with in % 256 == 0     -> Q6_K  (towers, ssm_out/attn_o, audio linears)
#   2D with in % 32  == 0     -> Q5_0  (ssm_in, attn_q/k/v, ffn_up_shexp)
#   everything else           -> F16/F32 (1-D precision contract + row<32 tails
#                               like the 4-tap ssm conv and 4D audio subsampling,
#                               which the engine reads raw on the host)
OUTTYPES = {"f16": (None, gguf.LlamaFileType.MOSTLY_F16),
            "q8_0": (None, gguf.LlamaFileType.MOSTLY_Q8_0),
            "q6k_mixed": (None, gguf.LlamaFileType.MOSTLY_Q6_K)}
EMBEDDINGS = ("token_embd.weight", "lm_head.weight")


def quant_policy(ops, out_type):
    """Assign per-op quantized types (op["q"], kind "q") for the quant tiers.
    f16 mode is a no-op; F32/1-D ops always keep their precision contract."""
    if out_type not in ("q8_0", "q6k_mixed"):
        return
    for op in ops:
        if op["k"] not in ("f16", "exps_up", "exps_down"):
            continue                                  # F32/1-D contract
        row = op["s"][-1]
        if out_type == "q8_0" and row % 32 == 0:
            qt = gguf.GGMLQuantizationType.Q8_0
        elif out_type == "q8_0":
            warn(f"{op['g']}: row {row} not multiple of 32, keeping F16")
            continue
        elif op["g"] in EMBEDDINGS:
            qt = gguf.GGMLQuantizationType.Q8_0
        elif op["k"] in ("exps_up", "exps_down"):
            qt = gguf.GGMLQuantizationType.Q5_0
        elif row % 256 == 0:
            qt = gguf.GGMLQuantizationType.Q6_K
        elif row % 32 == 0:
            qt = gguf.GGMLQuantizationType.Q5_0
        else:
            warn(f"{op['g']}: row {row} not quantizable, keeping F16")
            continue
        if op["k"] == "f16":
            op["k"] = "q"
        op["q"] = qt


SKIP_EXACT = ("vision_model.radio_model.input_conditioner.norm_mean",
              "vision_model.radio_model.input_conditioner.norm_std")
SKIP_SUFFIX = (".conv.norm.num_batches_tracked",)

LP = "language_model.backbone"
VP = "vision_model.radio_model.model"
AP = "sound_encoder.encoder"


def read_config(src):
    with open(os.path.join(src, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    llm = cfg["llm_config"]
    pat = llm["hybrid_override_pattern"]
    if len(pat) != llm["num_hidden_layers"]:
        fail(f"hybrid_override_pattern has {len(pat)} chars, want {llm['num_hidden_layers']}")
    return dict(
        n_layer=llm["num_hidden_layers"], pattern=pat,
        n_embd=llm["hidden_size"], n_head=llm["num_attention_heads"],
        n_kv_head=llm["num_key_value_heads"], head_dim=llm["head_dim"],
        vocab=llm["vocab_size"], eps=llm["norm_eps"],
        n_expert=llm["n_routed_experts"], n_used=llm["num_experts_per_tok"],
        n_shared=llm["n_shared_experts"], ff_exp=llm["moe_intermediate_size"],
        ff_shexp=llm["moe_shared_expert_intermediate_size"],
        mamba_heads=llm["mamba_num_heads"], mamba_hd=llm["mamba_head_dim"],
        n_groups=llm["n_groups"], d_state=llm["ssm_state_size"],
        conv_kernel=llm["conv_kernel"], ctx=llm["max_position_embeddings"],
        v_blocks=32, v_embd=1280, v_ff=5120, v_heads=16, v_patch=16,
        mean=cfg["norm_mean"], std=cfg["norm_std"],
        a_blocks=cfg["sound_config"]["num_hidden_layers"],
        a_embd=cfg["sound_config"]["hidden_size"],
        a_ff=cfg["sound_config"]["intermediate_size"],
        a_heads=cfg["sound_config"]["num_attention_heads"],
        a_mel=cfg["sound_config"]["num_mel_bins"],
        bos=1, eos=11, pad=0,   # generation_config: eos [2, 11], stop on 11
    )


# --------------------------------------------------------------------- plan
# op kinds: f16 = matmul operand (out-type precision, quantizable),
# f32 = precision-critical small tensor, a = -exp(A_log), d = D,
# exps_up/exps_down = stacked experts. t = post-transform code.
def build_ops(cfg):
    ops = []
    f16, f32 = "f16", "f32"

    def add(g, h, k, s, t=None):
        ops.append(dict(g=g, h=h, k=k, s=tuple(s), t=t))

    add("token_embd.weight", f"{LP}.embeddings.weight", f16, (cfg["vocab"], cfg["n_embd"]))
    add("output_norm.weight", f"{LP}.norm_f.weight", f32, (cfg["n_embd"],))

    d_inner = cfg["mamba_heads"] * cfg["mamba_hd"]              # 4096
    conv_dim = d_inner + 2 * cfg["n_groups"] * cfg["d_state"]   # 6144
    d_in_proj = d_inner + conv_dim + cfg["mamba_heads"]         # 10304
    for i in range(cfg["n_layer"]):
        p, b = f"{LP}.layers.{i}", f"blk.{i}"
        add(f"{b}.attn_norm.weight", f"{p}.norm.weight", f32, (cfg["n_embd"],))
        c = cfg["pattern"][i]
        if c == "M":
            add(f"{b}.ssm_in.weight", f"{p}.mixer.in_proj.weight", f16, (d_in_proj, cfg["n_embd"]))
            add(f"{b}.ssm_conv1d.weight", f"{p}.mixer.conv1d.weight", f16,
                (conv_dim, cfg["conv_kernel"]), t="c14")
            add(f"{b}.ssm_conv1d.bias", f"{p}.mixer.conv1d.bias", f32, (conv_dim,))
            add(f"{b}.ssm_dt.bias", f"{p}.mixer.dt_bias", f32, (cfg["mamba_heads"],))
            add(f"{b}.ssm_a", f"{p}.mixer.A_log", "a", (cfg["mamba_heads"], 1))
            add(f"{b}.ssm_d", f"{p}.mixer.D", "d", (cfg["mamba_heads"], 1))
            # grouped RMSNorm: HF flat [4096] is group-major (view(-1, 8, 512)),
            # stored as numpy (8, 512) so the GGUF ne == llama.cpp {512, 8}
            add(f"{b}.ssm_norm.weight", f"{p}.mixer.norm.weight", f32,
                (cfg["n_groups"], d_inner // cfg["n_groups"]), t="gn")
            add(f"{b}.ssm_out.weight", f"{p}.mixer.out_proj.weight", f16, (cfg["n_embd"], d_inner))
        elif c == "*":
            for w, hf in (("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj")):
                add(f"{b}.{w}.weight", f"{p}.mixer.{hf}.weight", f16,
                    (cfg["n_head"] * cfg["head_dim"], cfg["n_embd"]) if hf == "q_proj"
                    else (cfg["n_kv_head"] * cfg["head_dim"], cfg["n_embd"]))
            add(f"{b}.attn_o.weight", f"{p}.mixer.o_proj.weight", f16,
                (cfg["n_embd"], cfg["n_head"] * cfg["head_dim"]))
        elif c == "E":
            add(f"{b}.ffn_gate_inp.weight", f"{p}.mixer.gate.weight", f32,
                (cfg["n_expert"], cfg["n_embd"]))
            add(f"{b}.ffn_exp_probs_b", f"{p}.mixer.gate.e_score_correction_bias", f32,
                (cfg["n_expert"],))
            add(f"{b}.ffn_up_exps.weight", f"{p}.mixer.experts", "exps_up",
                (cfg["n_expert"], cfg["ff_exp"], cfg["n_embd"]))
            add(f"{b}.ffn_down_exps.weight", f"{p}.mixer.experts", "exps_down",
                (cfg["n_expert"], cfg["n_embd"], cfg["ff_exp"]))
            add(f"{b}.ffn_up_shexp.weight", f"{p}.mixer.shared_experts.up_proj.weight", f16,
                (cfg["ff_shexp"], cfg["n_embd"]))
            add(f"{b}.ffn_down_shexp.weight", f"{p}.mixer.shared_experts.down_proj.weight", f16,
                (cfg["n_embd"], cfg["ff_shexp"]))
        else:
            fail(f"unknown hybrid pattern char {c!r} at layer {i}")

    add("lm_head.weight", "language_model.lm_head.weight", f16, (cfg["vocab"], cfg["n_embd"]))

    # ---------------------------------------------------------- vision tower
    add("v.patch_embd.weight", f"{VP}.patch_generator.embedder.weight", f16, (1280, 768))
    add("v.video_embd.weight", f"{VP}.patch_generator.video_embedder.weight", f16, (1280, 1536))
    add("v.cls_token.weight", f"{VP}.patch_generator.cls_token.token", f32, (10, 1280))
    add("v.pos_embd.weight", f"{VP}.patch_generator.pos_embed", f32, (16384, 1280), t="sq1")
    for i in range(cfg["v_blocks"]):
        h, b = f"{VP}.blocks.{i}", f"v.blk.{i}"
        for ln, hf in (("ln1", "norm1"), ("ln2", "norm2")):
            add(f"{b}.{ln}.weight", f"{h}.{hf}.weight", f32, (1280,))
            add(f"{b}.{ln}.bias", f"{h}.{hf}.bias", f32, (1280,))
        add(f"{b}.wqkv.weight", f"{h}.attn.qkv.weight", f16, (3840, 1280))
        add(f"{b}.bqkv", f"{h}.attn.qkv.bias", f32, (3840,))
        add(f"{b}.wo.weight", f"{h}.attn.proj.weight", f16, (1280, 1280))
        add(f"{b}.bo", f"{h}.attn.proj.bias", f32, (1280,))
        add(f"{b}.fc0.weight", f"{h}.mlp.fc1.weight", f16, (5120, 1280))
        add(f"{b}.fc0.bias", f"{h}.mlp.fc1.bias", f32, (5120,))
        add(f"{b}.fc1.weight", f"{h}.mlp.fc2.weight", f16, (1280, 5120))
        add(f"{b}.fc1.bias", f"{h}.mlp.fc2.bias", f32, (1280,))
    add("mm.ln", "mlp1.0.weight", f32, (5120,))
    add("mm.fc0.weight", "mlp1.1.weight", f16, (20480, 5120))
    add("mm.fc1.weight", "mlp1.3.weight", f16, (2688, 20480))

    # ------------------------------------------------------------ audio tower
    add("a.mel.fb", f"{AP}.feature_extractor.featurizer.fb", f32, (128, 257), t="sq1")
    add("a.mel.window", f"{AP}.feature_extractor.featurizer.window", f32, (400,))
    # HF subsampling ModuleList interleaves ReLU modules: 0=conv3x3 (1->256),
    # 2=dw3x3, 3=pw1x1, 5=dw3x3, 6=pw1x1 (stride 2 each -> 8x subsample);
    # every 3x3 stage has in_channels 1 (layer 0: 1 input ch, 2/5: depthwise)
    for k, hf_i in enumerate((0, 2, 3, 5, 6)):
        add(f"a.sub.{k}.weight", f"{AP}.subsampling.layers.{hf_i}.weight", f16,
            (256, 1, 3, 3) if hf_i in (0, 2, 5) else (256, 256, 1, 1))
        add(f"a.sub.{k}.bias", f"{AP}.subsampling.layers.{hf_i}.bias", f32, (256,))
    add("a.sub.proj.weight", f"{AP}.subsampling.linear.weight", f16, (1024, 4096))
    add("a.sub.proj.bias", f"{AP}.subsampling.linear.bias", f32, (1024,))
    for i in range(cfg["a_blocks"]):
        p, b = f"{AP}.layers.{i}", f"a.blk.{i}"
        for ln, hf in (("ln_ff1", "norm_feed_forward1"), ("ln_att", "norm_self_att"),
                       ("ln_conv", "norm_conv"), ("ln_ff2", "norm_feed_forward2"),
                       ("ln_out", "norm_out")):
            add(f"{b}.{ln}.weight", f"{p}.{hf}.weight", f32, (1024,))
            add(f"{b}.{ln}.bias", f"{p}.{hf}.bias", f32, (1024,))
        add(f"{b}.ff1.w1", f"{p}.feed_forward1.linear1.weight", f16, (4096, 1024))
        add(f"{b}.ff1.w2", f"{p}.feed_forward1.linear2.weight", f16, (1024, 4096))
        add(f"{b}.ff2.w1", f"{p}.feed_forward2.linear1.weight", f16, (4096, 1024))
        add(f"{b}.ff2.w2", f"{p}.feed_forward2.linear2.weight", f16, (1024, 4096))
        for w, hf in (("att.q", "q_proj"), ("att.k", "k_proj"),
                      ("att.v", "v_proj"), ("att.o", "o_proj")):
            add(f"{b}.{w}", f"{p}.self_attn.{hf}.weight", f16, (1024, 1024))
        add(f"{b}.att.rel_k", f"{p}.self_attn.relative_k_proj.weight", f16, (1024, 1024))
        add(f"{b}.att.bias_u", f"{p}.self_attn.bias_u", f32, (8, 128))
        add(f"{b}.att.bias_v", f"{p}.self_attn.bias_v", f32, (8, 128))
        add(f"{b}.conv.pw1", f"{p}.conv.pointwise_conv1.weight", f16, (2048, 1024), t="pw")
        add(f"{b}.conv.dw", f"{p}.conv.depthwise_conv.weight", f16, (1024, 9), t="dw9")
        add(f"{b}.conv.bn.weight", f"{p}.conv.norm.weight", f32, (1024,))
        add(f"{b}.conv.bn.bias", f"{p}.conv.norm.bias", f32, (1024,))
        add(f"{b}.conv.bn.mean", f"{p}.conv.norm.running_mean", f32, (1024,))
        add(f"{b}.conv.bn.var", f"{p}.conv.norm.running_var", f32, (1024,))
        add(f"{b}.conv.pw2", f"{p}.conv.pointwise_conv2.weight", f16, (1024, 1024), t="pw")
    add("a.proj.ln", "sound_projection.norm.weight", f32, (1024,))
    add("a.proj.fc1", "sound_projection.linear1.weight", f16, (4096, 1024))
    add("a.proj.fc2", "sound_projection.linear2.weight", f16, (2688, 4096))
    return ops


def materialize(op, sh):
    """Build the exact final tensor for one op (mmap -> transform -> dtype)."""
    if op["k"] in ("exps_up", "exps_down"):
        leaf = "up_proj.weight" if op["k"] == "exps_up" else "down_proj.weight"
        n_exp = op["s"][0]
        qt = op.get("q")
        if qt is not None:
            buf = bytearray()
            for e in range(n_exp):
                raw, dt, shp = sh.raw(f"{op['h']}.{e}.{leaf}")
                a32 = _expand_bf16(raw, np.empty(raw.size, np.float32)).reshape(shp) \
                    if dt == "BF16" else raw.astype(np.float32)
                buf += _lib.quantize(np.ascontiguousarray(a32), qt).tobytes()
                del a32
            return np.frombuffer(bytes(buf), dtype=np.uint8)
        out = np.empty(op["s"], dtype=np.float16)
        for e in range(n_exp):
            raw, dt, shp = sh.raw(f"{op['h']}.{e}.{leaf}")
            dst = out.reshape(-1)[e * out[0].size:(e + 1) * out[0].size]
            if dt == "BF16":
                _expand_bf16(raw, dst)   # f32 intermediate per chunk, cast on store
            else:
                dst[:] = raw.reshape(-1)
        return out

    raw, dt, hf_shape = sh.raw(op["h"])
    if op["t"] == "c14":                       # [6144, 1, 4] -> [6144, 4]
        raw, hf_shape = raw.reshape(hf_shape[0], hf_shape[2]), (hf_shape[0], hf_shape[2])
    elif op["t"] == "sq1":                     # [1, N, D] -> [N, D]
        raw, hf_shape = raw.reshape(hf_shape[1], hf_shape[2]), (hf_shape[1], hf_shape[2])
    elif op["t"] == "pw":                      # [out, in, 1] -> [out, in]
        raw, hf_shape = raw.reshape(hf_shape[0], hf_shape[1]), (hf_shape[0], hf_shape[1])
    elif op["t"] == "dw9":                     # [1024, 1, 9] -> [1024, 9]
        raw, hf_shape = raw.reshape(hf_shape[0], hf_shape[2]), (hf_shape[0], hf_shape[2])
    if op["k"] in ("a", "d"):                  # ssm_a / ssm_d: HF [heads] -> [heads, 1]
        if raw.size != int(np.prod(op["s"])):
            fail(f"{op['h']}: {raw.size} elements != planned {op['s']}")
    elif op["t"] == "gn":                      # flat [4096] -> group-major (8, 512)
        raw, hf_shape = raw.reshape(op["s"]), op["s"]
    elif tuple(hf_shape) != op["s"]:
        fail(f"{op['h']}: shape {hf_shape} != planned {op['s']}")

    if op["k"] in ("f16", "f32"):
        out = np.empty(raw.size, dtype=np.float16 if op["k"] == "f16" else np.float32)
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
    if op["k"] in ("a", "d"):                  # ssm_a / ssm_d, F32, [heads, 1]
        out = np.empty(raw.size, dtype=np.float32)
        if dt == "BF16":
            _expand_bf16(raw, out)
        else:
            out[:] = raw.reshape(-1)
        if op["k"] == "a":                     # A = -exp(A_log)
            np.negative(np.exp(out), out=out)
        return out.reshape(op["s"])
    fail(f"unknown op kind {op['k']}")


# ---------------------------------------------------------------- metadata
def write_metadata(gw, cfg, outtype, rope_mode):
    ftype = OUTTYPES[outtype][1]
    gw.add_name("Nemotron-3-Nano-Omni-30B-A3B")
    gw.add_file_type(ftype)
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32(f"{ARCH}.block_count", cfg["n_layer"])
    gw.add_uint32(f"{ARCH}.embedding_length", cfg["n_embd"])
    gw.add_uint32(f"{ARCH}.vocab_size", cfg["vocab"])
    gw.add_uint32(f"{ARCH}.context_length", cfg["ctx"])
    gw.add_uint32(f"{ARCH}.attention.head_count", cfg["n_head"])
    # per-layer kv heads (llama.cpp hybrid discriminator): attn layers = 2, else 0
    gw.add_key_value(f"{ARCH}.attention.head_count_kv",
                     [cfg["n_kv_head"] if c == "*" else 0 for c in cfg["pattern"]],
                     gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.UINT32)
    gw.add_uint32(f"{ARCH}.attention.key_length", cfg["head_dim"])
    gw.add_uint32(f"{ARCH}.attention.value_length", cfg["head_dim"])
    gw.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", float(cfg["eps"]))
    gw.add_uint32(f"{ARCH}.expert_count", cfg["n_expert"])
    gw.add_uint32(f"{ARCH}.expert_used_count", cfg["n_used"])
    gw.add_uint32(f"{ARCH}.expert_shared_count", cfg["n_shared"])
    gw.add_uint32(f"{ARCH}.expert_feed_forward_length", cfg["ff_exp"])
    gw.add_uint32(f"{ARCH}.expert_shared_feed_forward_length", cfg["ff_shexp"])
    gw.add_bool(f"{ARCH}.expert_weights_norm", True)          # norm_topk_prob
    gw.add_float32(f"{ARCH}.expert_scale", 2.5)               # routed_scaling_factor
    gw.add_uint32(f"{ARCH}.ssm.conv_kernel", cfg["conv_kernel"])
    gw.add_uint32(f"{ARCH}.ssm.inner_size", cfg["mamba_heads"] * cfg["mamba_hd"])
    gw.add_uint32(f"{ARCH}.ssm.state_size", cfg["d_state"])
    gw.add_uint32(f"{ARCH}.ssm.group_count", cfg["n_groups"])
    gw.add_uint32(f"{ARCH}.ssm.time_step_rank", cfg["mamba_heads"])
    gw.add_string(f"{ARCH}.hybrid_override_pattern", cfg["pattern"])
    gw.add_uint32(f"{ARCH}.bos_token_id", cfg["bos"])
    gw.add_uint32(f"{ARCH}.eos_token_id", cfg["eos"])         # 11 <|im_end|>; gen cfg [2, 11]
    gw.add_uint32(f"{ARCH}.pad_token_id", cfg["pad"])
    if rope_mode == "partial84":    # §5 experiment only; default NoPE writes no rope.*
        gw.add_uint32(f"{ARCH}.rope.dimension_count", 84)
        gw.add_float32(f"{ARCH}.rope.freq_base", 10000.0)

    gw.add_uint32(f"{ARCH}.vision.block_count", cfg["v_blocks"])
    gw.add_uint32(f"{ARCH}.vision.embedding_length", cfg["v_embd"])
    gw.add_uint32(f"{ARCH}.vision.feed_forward_length", cfg["v_ff"])
    gw.add_uint32(f"{ARCH}.vision.attention.head_count", cfg["v_heads"])
    gw.add_uint32(f"{ARCH}.vision.patch_size", cfg["v_patch"])
    gw.add_float32(f"{ARCH}.vision.layer_norm_epsilon", 1e-6)  # timm ViT default
    gw.add_array(f"{ARCH}.vision.image_mean", [float(x) for x in cfg["mean"]])
    gw.add_array(f"{ARCH}.vision.image_std", [float(x) for x in cfg["std"]])

    gw.add_uint32(f"{ARCH}.audio.block_count", cfg["a_blocks"])
    gw.add_uint32(f"{ARCH}.audio.embedding_length", cfg["a_embd"])
    gw.add_uint32(f"{ARCH}.audio.feed_forward_length", cfg["a_ff"])
    gw.add_uint32(f"{ARCH}.audio.attention.head_count", cfg["a_heads"])
    gw.add_uint32(f"{ARCH}.audio.num_mel_bins", cfg["a_mel"])


def op_decl(op):
    """(shape_for_info, np_dtype, nbytes, raw_dtype) of the final tensor."""
    # quantizable kinds: the expert stacks keep their kind once the tier
    # assigns Q5_0; the float16-dt kinds are the f16 matmul operands and the
    # non-quantized expert stacks
    return _plan.op_decl(op, quant_kinds=("exps_up", "exps_down"),
                         f16_kinds=("f16", "exps_up", "exps_down"))


def write_gguf(out, cfg, ops, sh, outtype, rope_mode, quiet=True):
    """Stream the GGUF through the shared skeleton (pre-declared header, then
    the data section one op at a time); metadata + materialize stay here."""
    return _plan.write_gguf(out, ARCH, ops, sh,
                            lambda gw: write_metadata(gw, cfg, outtype, rope_mode),
                            op_decl, materialize, quiet=quiet)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-type", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the matmul operands (default f16; norms/"
                         "biases/router stay F32 regardless)")
    ap.add_argument("--rope", default="off", choices=("off", "partial84"),
                    help="NoPE (default, per port-plan §5 prior) or write "
                         "rope.dimension_count=84 metadata for the D4 experiment")
    ap.add_argument("-o", "--out", default=None,
                    help=f"output file (default {DEFAULT_OUT_DIR}\\{ARCH}-<out-type>.gguf)")
    ap.add_argument("--src", default=os.environ.get("NEMOTRONOMNI_SRC", DEFAULT_SRC),
                    help="checkpoint dir (17 model-*.safetensors + index json)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    src = os.path.abspath(args.src)
    index = os.path.join(src, "model.safetensors.index.json")
    if not os.path.isfile(index):
        fail(f"no model.safetensors.index.json in {src}")
    with open(index, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    files = sorted(set(weight_map.values()))
    for fname in files:   # download-integrity guard: refuse partial shards
        if not os.path.exists(os.path.join(src, fname + ".done")):
            fail(f"{fname} has no .done marker (download incomplete?)")
    log(f"source: {src} ({len(files)} shards, {len(weight_map)} tensors)")

    cfg = read_config(src)
    ops = build_ops(cfg)
    quant_policy(ops, args.out_type)
    _plan.log_quant_policy(ops, args.out_type)

    sh = Shards(src, weight_map)

    def exps_names(op):
        if op["k"] in ("exps_up", "exps_down"):
            leaf = "up_proj.weight" if op["k"] == "exps_up" else "down_proj.weight"
            return [f"{op['h']}.{e}.{leaf}" for e in range(op["s"][0])]
        return [op["h"]]

    consumed, skipped = _plan.check_coverage(ops, weight_map, sh, exps_names,
                                             SKIP_EXACT, SKIP_SUFFIX)
    log(f"plan: {len(ops)} GGUF tensors <- {len(consumed)} HF tensors "
        f"({len(skipped)} skipped: 2 input_conditioner + 24 num_batches_tracked)")

    out = args.out or os.path.join(DEFAULT_OUT_DIR, f"{ARCH}-{args.out_type}.gguf")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    elapsed = write_gguf(out, cfg, ops, sh, args.out_type, args.rope, args.quiet)
    sh.close()

    size = os.path.getsize(out)
    log(f"done: {out} ({size / 1e9:.2f} GB, {len(ops)} tensors, "
        f"{elapsed:.0f}s, peak RSS {peak_rss_mib():.0f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
