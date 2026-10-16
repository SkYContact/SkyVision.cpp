"""Compare the hand-rolled NF4 dequantization against bitsandbytes itself.

Usage::

  python convert/check_moss_vl_nf4.py [--src DIR]

Acceptance gate for convert_moss_vl.py: for each exemplar tensor the manual
dequant (CPU, float32) must match ``bitsandbytes.functional.dequantize_4bit``
(GPU, bfloat16 output) bitwise, or within max rel diff < 1e-6. The reference
feeds the kernel the absmax dequantized by ``dequantize_blockwise`` (the same
op bnb's nested path runs internally), so the kernel input is identical.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from bitsandbytes import functional as bnb_f

import convert_moss_vl as cm
from _lib import log

EXEMPLARS = (
    "model.language_model.layers.4.self_attn.q_proj",
    "model.language_model.layers.5.mlp.down_proj",
    "model.language_model.layers.6.mlp.gate_proj",
)


def bnf4_reference(reader, module, device):
    """bitsandbytes reference: packed uint8 -> dequantized float (on device)."""
    def raw(suffix, dtype):
        return torch.from_numpy(np.ascontiguousarray(reader.raw(f"{module}.{suffix}", dtype)))

    qs = json.loads(raw(cm.NF4_QUANT_STATE, np.uint8).numpy().tobytes().decode("utf-8"))
    absmax_u8 = raw("weight.absmax", np.uint8).to(device)
    state2 = bnb_f.QuantState(
        absmax=raw("weight.nested_absmax", np.float32).to(device),
        shape=(absmax_u8.numel(),),
        code=raw("weight.nested_quant_map", np.float32).to(device),
        blocksize=qs["nested_blocksize"], quant_type="fp8",
        dtype=torch.float32, offset=0.0)
    absmax = bnb_f.dequantize_blockwise(absmax_u8, state2) + qs["nested_offset"]
    state = bnb_f.QuantState(
        absmax=absmax, shape=tuple(qs["shape"]),
        code=raw("weight.quant_map", np.float32).to(device),
        blocksize=qs["blocksize"], quant_type="nf4",
        dtype=torch.bfloat16, offset=0.0, state2=None)
    packed = raw("weight", np.uint8).to(device).view(-1)
    return absmax, bnb_f.dequantize_4bit(packed, state)   # kernel output: bfloat16


def compare(module, mine_f32, ref_bf16, absmax_ref, scale_mine):
    name = module[len("model.language_model.layers."):]
    abs_ok = torch.equal(scale_mine, absmax_ref)
    a = mine_f32.view(-1).to(torch.bfloat16)              # same rounding as the kernel
    b = ref_bf16.view(-1)
    bitwise = (a.view(torch.uint16) == b.view(torch.uint16)).float().mean().item()
    a, b = a.float(), b.float()
    abs_diff = (a - b).abs().max().item()
    rel_diff = ((a - b).abs() / b.abs().clamp(min=1e-8)).max().item()
    log(f"{name}: absmax_bitwise_equal={str(abs_ok).lower()} bitwise_equal={bitwise:.6f} "
        f"max_abs_diff={abs_diff:.3g} max_rel_diff={rel_diff:.3g}")
    return abs_ok and (bitwise == 1.0 or rel_diff < 1e-6)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=os.environ.get("MOSS_VL_SRC", "D:/MOSS-VL-Realtime-NF4"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"bitsandbytes reference on {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    reader = cm.CheckpointReader(args.src)
    ok = True
    for module in EXEMPLARS:
        qs = json.loads(reader.raw(f"{module}.{cm.NF4_QUANT_STATE}", np.uint8)
                        .tobytes().decode("utf-8"))
        nbs, offset = qs["nested_blocksize"], qs["nested_offset"]
        absmax_u8 = torch.from_numpy(reader.raw(f"{module}.weight.absmax", np.uint8))
        nested_map = torch.from_numpy(reader.raw(f"{module}.weight.nested_quant_map"))
        nested_absmax = torch.from_numpy(reader.raw(f"{module}.weight.nested_absmax"))
        scale_mine = nested_map[absmax_u8.long()] * torch.repeat_interleave(
            nested_absmax, nbs)[:absmax_u8.numel()] + offset

        absmax_ref, ref = bnf4_reference(reader, module, device)
        ref = ref.cpu()
        absmax_ref = absmax_ref.cpu()
        mine = torch.from_numpy(cm.dequantize_nf4(reader, module))
        ok &= compare(module, mine, ref, absmax_ref, scale_mine)
    log("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
