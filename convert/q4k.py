# -*- coding: utf-8 -*-
"""Vectorized port of ggml's quantize_row_q4_K_ref (ggml-quants.c).

Same story as q6k.py: gguf-py never implemented Q4_K quantization (only
dequant), so we port the reference C algorithm directly. Torch only (CUDA when
available, CPU otherwise) -- the NF4 converters need torch anyway. The result
is validated against gguf-py's Q4_K *decoder* by the converter's round-trip.

Block layout (144 bytes / 256 elems, ggml-common.h block_q4_K):
    d fp16 (2B) | dmin fp16 (2B) | scales[12] (6-bit pairs) | qs[128] (4-bit)
"""
import numpy as np

QK_K = 256

try:
    import torch
except ImportError:
    torch = None


def _nearest_int(f):
    # C: val = f + 12582912.0f; reinterpret low 23 bits minus 0x400000
    return ((f + 12582912.0).view(torch.int32) & 0x007FFFFF) - 0x00400000


def _make_qkx2_quants(x, w, nmax=15.0, rmin=-1.0, rdelta=0.1, nstep=20):
    """Port of make_qkx2_quants(n=32, use_mad=false) for [B, 32] blocks.

    Returns (scale [B], the_min [B], L [B, 32] int32 0..nmax).
    """
    mn = x.min(dim=-1).values.clamp(max=0.0)
    mx = x.max(dim=-1).values
    degenerate = mx == mn
    span = torch.where(degenerate, torch.ones_like(mx), mx - mn)

    iscale = nmax / span
    scale = 1.0 / iscale
    L = _nearest_int(iscale.unsqueeze(-1) * (x - mn.unsqueeze(-1))).clamp(0, int(nmax))
    best = (w * (scale.unsqueeze(-1) * L.float() + mn.unsqueeze(-1) - x) ** 2).sum(-1)

    sum_w = w.sum(-1)
    sum_x = (w * x).sum(-1)
    for is_ in range(nstep + 1):
        isc = (rmin + rdelta * is_ + nmax) / span
        l = _nearest_int(isc.unsqueeze(-1) * (x - mn.unsqueeze(-1))).clamp(0, int(nmax))
        lf = l.float()
        sum_l = (w * lf).sum(-1)
        sum_l2 = (w * lf * lf).sum(-1)
        sum_xl = (w * lf * x).sum(-1)
        D = sum_w * sum_l2 - sum_l * sum_l
        ok = D > 0
        safe = torch.where(ok, D, torch.ones_like(D))
        this_scale = torch.where(ok, (sum_w * sum_xl - sum_x * sum_l) / safe, scale)
        this_min = torch.where(ok, (sum_l2 * sum_x - sum_l * sum_xl) / safe, mn)
        pos = this_min > 0
        this_min = torch.where(pos, torch.zeros_like(this_min), this_min)
        this_scale = torch.where(pos, sum_xl / torch.where(
            sum_l2 != 0, sum_l2, torch.ones_like(sum_l2)), this_scale)
        cur = (w * (this_scale.unsqueeze(-1) * lf + this_min.unsqueeze(-1) - x) ** 2).sum(-1)
        upd = ok & (cur < best)
        L = torch.where(upd.unsqueeze(-1), l, L)
        best = torch.where(upd, cur, best)
        scale = torch.where(upd, this_scale, scale)
        mn = torch.where(upd, this_min, mn)
    scale = torch.where(degenerate, torch.zeros_like(scale), scale)
    return scale, -mn, L


def _quantize_q4_k_torch(rows_f32, device):
    """rows_f32: torch float32 [nb, 256] -> packed uint8 cpu [nb, 144]."""
    dev = rows_f32.device
    nb = rows_f32.shape[0]

    x = rows_f32.view(nb, 8, 32)
    av_x = (x * x).sum(-1).div_(32).sqrt_().unsqueeze(-1)
    w = av_x + x.abs()

    scale, the_min, _ = _make_qkx2_quants(x, w)          # [nb, 8] each

    max_scale = scale.max(-1).values                     # [nb]
    max_min = the_min.max(-1).values
    max_scale = torch.where(max_scale > 0, max_scale, torch.zeros_like(max_scale))
    max_min = torch.where(max_min > 0, max_min, torch.zeros_like(max_min))
    inv_scale = torch.where(max_scale > 0, 63.0 / torch.where(
        max_scale > 0, max_scale, torch.ones_like(max_scale)), torch.zeros_like(max_scale))
    inv_min = torch.where(max_min > 0, 63.0 / torch.where(
        max_min > 0, max_min, torch.ones_like(max_min)), torch.zeros_like(max_min))
    ls = _nearest_int(inv_scale.unsqueeze(-1) * scale).clamp(0, 63)
    lm = _nearest_int(inv_min.unsqueeze(-1) * the_min).clamp(0, 63)
    ls, lm = ls.byte(), lm.byte()

    sc = torch.zeros(nb, 12, dtype=torch.uint8, device=dev)
    sc[:, 0:4] = ls[:, 0:4]
    sc[:, 4:8] = lm[:, 0:4]
    sc[:, 8:12] = (ls[:, 4:8] & 0xF) | (lm[:, 4:8] & 0xF) << 4
    sc[:, 0:4] |= (ls[:, 4:8] >> 4) << 6
    sc[:, 4:8] |= (lm[:, 4:8] >> 4) << 6

    d16 = (max_scale / 63.0).half()                      # [nb]
    dmin16 = (max_min / 63.0).half()

    # requantize against the 6-bit-quantized scales (exact round trip values)
    scq = torch.empty(nb, 8, dtype=torch.int32, device=dev)
    mq = torch.empty(nb, 8, dtype=torch.int32, device=dev)
    sc32 = sc.int()
    scq[:, 0:4] = sc32[:, 0:4] & 63
    mq[:, 0:4] = sc32[:, 4:8] & 63
    scq[:, 4:8] = (sc32[:, 8:12] & 0xF) | ((sc32[:, 0:4] >> 6) << 4)
    mq[:, 4:8] = (sc32[:, 8:12] >> 4) | ((sc32[:, 4:8] >> 6) << 4)
    d = d16.float().unsqueeze(-1) * scq.float()          # [nb, 8]
    dm = dmin16.float().unsqueeze(-1) * mq.float()
    zero = d == 0
    L = _nearest_int((x + dm.unsqueeze(-1)) / torch.where(
        zero, torch.ones_like(d), d).unsqueeze(-1)).clamp(0, 15).byte()
    L = torch.where(zero.unsqueeze(-1), torch.zeros_like(L), L).view(nb, 256)

    qs = torch.zeros(nb, 128, dtype=torch.uint8, device=dev)
    for j in (0, 64, 128, 192):
        qs[:, j // 2:j // 2 + 32] = L[:, j:j + 32] | (L[:, j + 32:j + 64] << 4)

    out = torch.zeros(nb, 144, dtype=torch.uint8, device=dev)
    out[:, 0:2] = d16.view(torch.uint8).view(-1, 2)
    out[:, 2:4] = dmin16.view(torch.uint8).view(-1, 2)
    out[:, 4:16] = sc
    out[:, 16:144] = qs
    return out.cpu().numpy()


def quantize_q4_k(arr):
    """arr: float32, last dim multiple of 256 -> packed uint8 (gguf-py byte convention)."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    assert a.shape[-1] % QK_K == 0, f"last dim {a.shape[-1]} not multiple of {QK_K}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    flat = torch.from_numpy(a.reshape(-1, QK_K))
    chunks, chunk = [], 8192                             # bound the search temporaries
    for i in range(0, flat.shape[0], chunk):
        chunks.append(_quantize_q4_k_torch(flat[i:i + chunk].to(dev), dev))
    out = np.concatenate(chunks)
    return out.reshape(*a.shape[:-1], -1)
