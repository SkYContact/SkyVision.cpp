"""Convert moondream3.1-9B-A2B HF safetensors (FP8 MoE) to GGUF for SkyVision.cpp.

Usage:
  python convert/convert_hf_to_gguf.py [--q6 | --q4] [--q8-dense] [--src DIR] [-o OUT.gguf]

-o is a format string expanded with the quantization tag, so the default writes
moondream3.1-q8_0.gguf / -q6k.gguf / -q4k.gguf into the current directory.

- BF16 tensors -> F16 (dropped: kv_quantizer, region_model)
- FP8 experts  -> Q8_0 (default), Q6_K with --q6 or Q4_K with --q4, with the
  per-channel scales folded in
"""
import argparse
import os

import numpy as np
import torch
from safetensors import safe_open
import gguf
from gguf import quants

import hf_source

N_LAYERS = 24
N_DENSE = 4          # layers 0..3 dense MLP
N_HEADS = 32
N_KV_HEADS = 32
DIM = 2048
FF = 8192
VOCAB = 51200
N_EXPERTS = 64
N_USED = 8
INTER = 1024         # expert inner dim (up is 2*inter=2048)
MOE_START = 4
ROPE_THETA = 1.5e6


def dequant_fp8(raw: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """FP8 e4m3 bytes -> float32, times per-(expert, out-channel) scale."""
    uint8 = raw.view(np.uint8)
    # e4m3fn decode: sign, exp bias 7, no inf; mantissa
    exp = (uint8 >> 3) & 0x0F
    man = uint8 & 0x07
    val = np.where(
        exp == 0,
        (man.astype(np.float32) * (2.0 ** -3)) * (2.0 ** -6),      # subnormal
        ((1.0 + man.astype(np.float32) / 8.0) * (2.0 ** (exp.astype(np.int32) - 7))),
    )
    val = np.where(uint8 & 0x80, -val, val)                        # sign bit
    return val.astype(np.float32) * scale[:, :, None].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    eq = ap.add_mutually_exclusive_group()
    eq.add_argument("--q4", action="store_true", help="quantize experts to Q4_K instead of Q8_0")
    eq.add_argument("--q6", action="store_true", help="quantize experts to Q6_K instead of Q8_0")
    ap.add_argument("--q8-dense", action="store_true",
                    help="also quantize big dense linears (Wqkv/out_proj/fc1/fc2/lm_head) to Q8_0 "
                         "(halves their decode-time read bandwidth; norms/biases/tau/router stay F16)")
    ap.add_argument("-o", "--out", default="moondream3.1-%s.gguf",
                    help="output file; %%s is replaced by the quantization tag")
    hf_source.add_src_argument(ap)
    args = ap.parse_args()
    tag = "q6k" if args.q6 else ("q4k" if args.q4 else "q8_0")
    if args.q8_dense:
        tag = ("q6k_q8d" if args.q6 else "q4k_q8d" if args.q4 else "q8full")
    if "%s" in args.out:
        out_path = args.out % tag
    else:
        # a plain path is fine too: splice the quantization tag in front of the suffix
        root, ext = os.path.splitext(args.out)
        out_path = f"{root}-{tag}{ext}"
    eqtype = (gguf.GGMLQuantizationType.Q6_K if args.q6 else
              gguf.GGMLQuantizationType.Q4_K if args.q4 else
              gguf.GGMLQuantizationType.Q8_0)

    def maybe_q8(arr_f32):
        return quants.quantize(arr_f32, gguf.GGMLQuantizationType.Q8_0) if args.q8_dense else arr_f32.astype(np.float16)

    src = hf_source.find_model_safetensors(args.src)
    print(f"source: {src}")
    print(f"output: {out_path}")
    w = safe_open(src, framework="pt")

    def get(name):
        t = w.get_tensor(name)
        if t.dtype == torch.bfloat16:
            t = t.float()
        elif t.dtype == torch.float8_e4m3fn:
            t = t.view(torch.uint8)
        return t.contiguous().numpy()

    gw = gguf.GGUFWriter(out_path, "moondream3")   # GGUFWriter writes general.architecture
    gw.add_file_type(gguf.LlamaFileType.MOSTLY_Q6_K if args.q6 else
                     gguf.LlamaFileType.MOSTLY_Q4_K_M if args.q4 else
                     gguf.LlamaFileType.MOSTLY_Q8_0)
    gw.add_name("moondream3.1-9B-A2B")

    k = gguf.Keys
    gw.add_uint32(k.LLM.BLOCK_COUNT, N_LAYERS)
    gw.add_uint32(k.LLM.EMBEDDING_LENGTH, DIM)
    gw.add_uint32(k.LLM.FEED_FORWARD_LENGTH, FF)
    gw.add_uint32(k.Attention.HEAD_COUNT, N_HEADS)
    gw.add_uint32(k.Attention.HEAD_COUNT_KV, N_KV_HEADS)
    gw.add_uint32(k.LLM.CONTEXT_LENGTH, 32768)
    gw.add_uint32(k.LLM.VOCAB_SIZE, VOCAB)
    gw.add_float32(k.Rope.FREQ_BASE, ROPE_THETA)
    gw.add_uint32(k.LLM.EXPERT_COUNT, N_EXPERTS)
    gw.add_uint32(k.LLM.EXPERT_USED_COUNT, N_USED)
    gw.add_uint32(k.LLM.EXPERT_FEED_FORWARD_LENGTH, INTER)
    gw.add_uint32(k.LLM.LEADING_DENSE_BLOCK_COUNT, N_DENSE)
    # moondream3 specifics (custom keys)
    gw.add_uint32("moondream3.moe_start_layer", MOE_START)
    gw.add_bool("moondream3.tau_attention", True)

    def add_tensor(gguf_name, arr, ftype=None):
        if ftype is None:
            gw.add_tensor(gguf_name, arr)
        else:
            gw.add_tensor(gguf_name, arr, raw_shape=arr.shape, raw_dtype=ftype)

    def quant_experts(name_up, name_down, scale_up, scale_down):
        # torch-native fp8 decode, then numpy for ggml quantization
        up = w.get_tensor(name_up).view(torch.uint8).view(torch.float8_e4m3fn).float()
        up = (up * torch.from_numpy(get(scale_up))[:, :, None].float()).numpy()
        down = w.get_tensor(name_down).view(torch.uint8).view(torch.float8_e4m3fn).float()
        down = (down * torch.from_numpy(get(scale_down))[:, :, None].float()).numpy()
        if args.q6:
            import q6k
            return q6k.quantize_q6_k(up), q6k.quantize_q6_k(down)
        return quants.quantize(up, eqtype), quants.quantize(down, eqtype)

    print("tokens...")
    tok_emb = get("text_model.transformer.embd.wte.weight")
    gw.add_tensor("token_embd.weight", tok_emb.astype(np.float16))
    gw.add_tensor("output_norm.weight", get("text_model.lm_head.ln.weight").astype(np.float16))
    gw.add_tensor("output_norm.bias", get("text_model.lm_head.ln.bias").astype(np.float16))
    add_tensor("output.weight", maybe_q8(get("text_model.lm_head.linear.weight")),
               gguf.GGMLQuantizationType.Q8_0 if args.q8_dense else None)
    gw.add_tensor("output.bias", get("text_model.lm_head.linear.bias").astype(np.float16))

    for i in range(N_LAYERS):
        p = f"text_model.transformer.h.{i}"
        print(f"layer {i}...")
        gw.add_tensor(f"blk.{i}.attn_norm.weight", get(f"{p}.ln.weight").astype(np.float16))
        gw.add_tensor(f"blk.{i}.attn_norm.bias", get(f"{p}.ln.bias").astype(np.float16))
        # moondream uses LayerNorm (bias everywhere), recorded via custom key
        add_tensor(f"blk.{i}.attn_qkv.weight", maybe_q8(get(f"{p}.mixer.Wqkv.weight")),
                   gguf.GGMLQuantizationType.Q8_0 if args.q8_dense else None)
        gw.add_tensor(f"blk.{i}.attn_qkv.bias", get(f"{p}.mixer.Wqkv.bias").astype(np.float16))
        add_tensor(f"blk.{i}.attn_out.weight", maybe_q8(get(f"{p}.mixer.out_proj.weight")),
                   gguf.GGMLQuantizationType.Q8_0 if args.q8_dense else None)
        gw.add_tensor(f"blk.{i}.attn_out.bias", get(f"{p}.mixer.out_proj.bias").astype(np.float16))
        gw.add_tensor(f"blk.{i}.tau_wq.weight", get(f"{p}.tau_wq").astype(np.float16))
        gw.add_tensor(f"blk.{i}.tau_wv.weight", get(f"{p}.tau_wv").astype(np.float16))
        gw.add_tensor(f"blk.{i}.tau_alpha.weight", get(f"{p}.tau_alpha").astype(np.float16))

        if i < MOE_START:
            add_tensor(f"blk.{i}.ffn_up.weight", maybe_q8(get(f"{p}.mlp.fc1.weight")),
                       gguf.GGMLQuantizationType.Q8_0 if args.q8_dense else None)
            gw.add_tensor(f"blk.{i}.ffn_up.bias", get(f"{p}.mlp.fc1.bias").astype(np.float16))
            add_tensor(f"blk.{i}.ffn_down.weight", maybe_q8(get(f"{p}.mlp.fc2.weight")),
                       gguf.GGMLQuantizationType.Q8_0 if args.q8_dense else None)
            gw.add_tensor(f"blk.{i}.ffn_down.bias", get(f"{p}.mlp.fc2.bias").astype(np.float16))
        else:
            gw.add_tensor(f"blk.{i}.ffn_gate_inp.weight", get(f"{p}.gate.weight").astype(np.float16))
            gw.add_tensor(f"blk.{i}.ffn_gate_inp.bias", get(f"{p}.gate.bias").astype(np.float16))
            qu, qd = quant_experts(
                f"{p}.mlp.experts.weight", f"{p}.mlp.output_experts.weight",
                f"{p}.moe_quant.up_scale", f"{p}.moe_quant.down_scale")
            add_tensor(f"blk.{i}.ffn_up_exps.weight", qu, eqtype)
            add_tensor(f"blk.{i}.ffn_down_exps.weight", qd, eqtype)

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print("done:", out_path)


if __name__ == "__main__":
    main()
