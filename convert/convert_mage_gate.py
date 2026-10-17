"""Convert the Mage-VL StreamMind gate (System 1) to GGUF.

Usage::

  python convert/convert_mage_gate.py [--src DIR] [-o OUT.gguf] [--outtype f16|q8_0]

Source: ``<checkpoint>/streammind_gate.safetensors`` from microsoft/Mage-VL.

The gate is the proactive-streaming cognition head:
  pre_net (fc + leaky_relu) -> 1x Mamba-1 (d_model 2560, d_inner 5120,
  dt_rank 160, n_state 16, conv kernel 4) -> post_net -> a 4-layer vocab-2
  Qwen3 classifier over a [perception, target-embedding] token pair.

Produced layout (arch "mage_gate"):
  gm.pre.fc3.{weight,bias}            pre_net
  gm.mamba.in_proj.weight             [10240, 2560] (x|z)
  gm.mamba.conv1d.{weight,bias}       [5120, 1, 4] -> flattened [5120, 4]
  gm.mamba.x_proj.weight              [192, 5120]  (dt 160 | B 16 | C 16)
  gm.mamba.dt_proj.{weight,bias}      [5120, 160]
  gm.mamba.A_log.weight               [5120, 16]   F32 (exp() at load)
  gm.mamba.D.weight                   [5120]
  gm.mamba.norm.{weight,bias}         RMSNorm (the mixer's pre-norm)
  gm.mamba.out_proj.weight            [2560, 5120]
  gm.norm_fn.{weight,bias}            LayerNorm after the block
  gm.post.fc3.{weight,bias}           post_net
  cls.*                               4-layer Qwen3 vocab-2 classifier
                                      (token_embd/output/blk.* like the main
                                      text stack, head 32/8, hd 128, ffn 12288)
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

ENV_VAR = "MAGE_VL_SRC"
SOURCE = "streammind_gate.safetensors"

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0}

CLS_CFG = dict(n_layer=4, n_embd=2560, n_head=32, n_kv_head=8, head_dim=128,
               n_ff=12288, n_vocab=2, rms_eps=1e-6, rope_theta=1000000.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="f16", choices=sorted(OUTTYPES),
                    help="quantization for the cls_net linears (default f16; the "
                         "gate runs on the CPU, memory is cheap)")
    ap.add_argument("-o", "--out", default="mage-gate-f16.gguf")
    ap.add_argument("-q", "--quiet", action="store_true")
    hf_source.add_src_argument(ap, model_id="microsoft/Mage-VL", env_var=ENV_VAR)
    args = ap.parse_args()

    src = hf_source.find_checkpoint_dir(args.src, model_id="microsoft/Mage-VL",
                                        env_var=ENV_VAR,
                                        patterns=("streammind_gate.safetensors",))
    path = os.path.join(src, SOURCE)
    if not os.path.isfile(path):
        fail(f"no {SOURCE} in {src}")
    log(f"source: {path}")
    log(f"output: {out if (out := args.out) else ''} ({args.outtype})")

    qtype = OUTTYPES[args.outtype]
    from _lib import _RawSafetensors
    w = _RawSafetensors(path)

    gw = gguf.GGUFWriter(args.out, "mage_gate")
    gw.add_architecture()
    gw.add_name("Mage-VL StreamMind gate")
    gw.add_uint32("general.quantization_version", 2)
    gw.add_uint32("mage_gate.cls_block_count", CLS_CFG["n_layer"])
    gw.add_uint32("mage_gate.cls_embedding_length", CLS_CFG["n_embd"])
    gw.add_uint32("mage_gate.cls_feed_forward_length", CLS_CFG["n_ff"])
    gw.add_uint32("mage_gate.cls_attention.head_count", CLS_CFG["n_head"])
    gw.add_uint32("mage_gate.cls_attention.head_count_kv", CLS_CFG["n_kv_head"])
    gw.add_uint32("mage_gate.cls_attention.key_length", CLS_CFG["head_dim"])
    gw.add_float32("mage_gate.cls_attention.layer_norm_rms_epsilon", CLS_CFG["rms_eps"])
    gw.add_float32("mage_gate.cls_rope.freq_base", CLS_CFG["rope_theta"])

    stats = {"quantized": 0, "f16": 0}

    def add(name, hf, quantize=False):
        arr = w.get(hf)
        blocks = None
        if quantize and qtype is not None:
            if arr.shape[-1] % 256 == 0:
                blocks = _lib.quantize(arr, qtype)
            else:
                warn(f"{hf}: row length {arr.shape[-1]} not quantizable, keeping F16")
        if blocks is None:
            _lib.add_f16(gw, name, arr)
            stats["f16"] += 1
        else:
            _lib.add_quantized(gw, name, blocks, qtype)
            stats["quantized"] += 1

    # --- perception heads ---
    add("gm.pre.fc3.weight", "pre_net.fc3.weight")
    add("gm.pre.fc3.bias", "pre_net.fc3.bias")
    add("gm.mamba.in_proj.weight", "mamba_model.ssms.0.mixer.in_proj.weight")
    conv = w.get("mamba_model.ssms.0.mixer.conv1d.weight")     # (5120, 1, 4)
    _lib.add_f16(gw, "gm.mamba.conv1d.weight", conv.reshape(conv.shape[0], conv.shape[2]))
    add("gm.mamba.conv1d.bias", "mamba_model.ssms.0.mixer.conv1d.bias")
    add("gm.mamba.x_proj.weight", "mamba_model.ssms.0.mixer.x_proj.weight")
    add("gm.mamba.dt_proj.weight", "mamba_model.ssms.0.mixer.dt_proj.weight")
    add("gm.mamba.dt_proj.bias", "mamba_model.ssms.0.mixer.dt_proj.bias")
    # A_log / D stay F32: exponentiated / applied elementwise on the host
    gw.add_tensor("gm.mamba.A_log.weight",
                  np.ascontiguousarray(w.get("mamba_model.ssms.0.mixer.A_log"), dtype=np.float32))
    gw.add_tensor("gm.mamba.D.weight",
                  np.ascontiguousarray(w.get("mamba_model.ssms.0.mixer.D"), dtype=np.float32))
    add("gm.mamba.norm.weight", "mamba_model.ssms.0.norm.weight")
    add("gm.mamba.norm.bias", "mamba_model.ssms.0.norm.bias")
    add("gm.mamba.out_proj.weight", "mamba_model.ssms.0.mixer.out_proj.weight")
    add("gm.norm_fn.weight", "mamba_model.norm_fn.weight")
    add("gm.norm_fn.bias", "mamba_model.norm_fn.bias")
    add("gm.post.fc3.weight", "post_net.fc3.weight")
    add("gm.post.fc3.bias", "post_net.fc3.bias")

    # --- cls_net: 4-layer vocab-2 Qwen3 ---
    lm = "cls_net.cls_model"
    add("cls.token_embd.weight", f"{lm}.model.embed_tokens.weight")
    add("cls.output.weight", f"{lm}.lm_head.weight")
    add("cls.output_norm.weight", f"{lm}.model.norm.weight")
    for i in range(CLS_CFG["n_layer"]):
        p, b = f"{lm}.model.layers.{i}", f"cls.blk.{i}"
        add(f"{b}.attn_norm.weight", f"{p}.input_layernorm.weight")
        add(f"{b}.attn_q.weight", f"{p}.self_attn.q_proj.weight", quantize=True)
        add(f"{b}.attn_k.weight", f"{p}.self_attn.k_proj.weight", quantize=True)
        add(f"{b}.attn_v.weight", f"{p}.self_attn.v_proj.weight", quantize=True)
        add(f"{b}.attn_output.weight", f"{p}.self_attn.o_proj.weight", quantize=True)
        add(f"{b}.attn_q_norm.weight", f"{p}.self_attn.q_norm.weight")
        add(f"{b}.attn_k_norm.weight", f"{p}.self_attn.k_norm.weight")
        add(f"{b}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight")
        add(f"{b}.ffn_gate.weight", f"{p}.mlp.gate_proj.weight", quantize=True)
        add(f"{b}.ffn_up.weight", f"{p}.mlp.up_proj.weight", quantize=True)
        add(f"{b}.ffn_down.weight", f"{p}.mlp.down_proj.weight", quantize=True)
        if not args.quiet:
            progress(i, CLS_CFG["n_layer"], "cls layer")

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    log(f"done: {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB, "
        f"{stats['quantized']} quantized / {stats['f16']} f16 tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
