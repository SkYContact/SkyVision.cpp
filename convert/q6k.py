# -*- coding: utf-8 -*-
"""Vectorized port of ggml's quantize_row_q6_K_ref (ggml-quants.c).

gguf-py never implemented Q6_K quantization (only dequant), so we port the
reference C algorithm directly. Bit-identical to llama.cpp's quantizer:
same make_qx_quants(rmse_type=1) scale search, same packing. The 19-pass
scale search is the hot loop -> runs on CUDA via torch when available,
numpy otherwise.

Block layout (210 bytes / 256 elems, ggml-common.h block_q6_K):
    ql[128] | qh[64] | scales[16] (int8) | d fp16 (2B)
"""
import numpy as np

QK_K = 256
GROUP_MAX_EPS = 1e-15

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False


def _quantize_q6_k_torch(rows_f32):
    """rows_f32: torch float32 CUDA [nb, 256] -> packed uint8 cpu [nb, 210]."""
    dev = rows_f32.device

    def nearest_int_t(f):
        return ((f + 12582912.0).view(torch.int32) & 0x007FFFFF) - 0x00400000

    x = rows_f32.view(-1, 16)                                   # [nb*16, 16]
    nmax = 32.0

    amax, idx = x.abs().max(dim=1)
    mx = x.gather(1, idx.unsqueeze(1)).squeeze(1)
    nz = amax >= GROUP_MAX_EPS
    xz = x[nz]
    mxz = mx[nz]                                                # nonzero by construction
    w = xz * xz                                                  # rmse_type == 1

    iscale = -nmax / mxz
    l = nearest_int_t(iscale.unsqueeze(1) * xz).clamp(-nmax, nmax - 1)
    sumlx = (w * xz * l).sum(dim=1)
    suml2 = (w * l * l).sum(dim=1)
    scale = torch.where(suml2 > 0, sumlx / suml2.clamp(min=1e-30), torch.zeros_like(suml2))
    best = scale * sumlx

    for is_ in range(-9, 10):
        if is_ == 0:
            continue
        isc = -(nmax + 0.1 * is_) / mxz
        l = nearest_int_t(isc.unsqueeze(1) * xz).clamp(-nmax, nmax - 1)
        sumlx = (w * xz * l).sum(dim=1)
        suml2 = (w * l * l).sum(dim=1)
        better = (suml2 > 0) & (sumlx * sumlx > best * suml2)
        scale = torch.where(better, sumlx / suml2.clamp(min=1e-30), scale)
        best = torch.where(better, scale * sumlx, best)

    scales16 = torch.zeros(x.shape[0], dtype=torch.float32, device=dev)
    scales16[nz.nonzero().squeeze(1)] = scale
    scales16 = scales16.view(-1, 16)                            # [nb, 16]

    _, k = scales16.abs().max(dim=1)
    max_scale = scales16.gather(1, k.unsqueeze(1)).squeeze(1)
    snz = scales16.abs().gather(1, k.unsqueeze(1)).squeeze(1) >= GROUP_MAX_EPS
    isc2 = torch.where(snz, -128.0 / torch.where(snz, max_scale, torch.ones_like(max_scale)),
                       torch.zeros_like(max_scale))
    d = torch.where(snz, 1.0 / torch.where(snz, isc2, torch.ones_like(isc2)),
                    torch.zeros_like(isc2))
    sc = nearest_int_t(isc2.unsqueeze(1) * scales16).clamp(max=127)
    sc = torch.where(snz.unsqueeze(1), sc, torch.zeros_like(sc)).to(torch.int8)

    ds = d.unsqueeze(1) * sc.float()                            # [nb,16]
    dsb = ds.repeat_interleave(16, dim=1)                       # [nb,256]
    L = nearest_int_t(rows_f32 / torch.where(dsb == 0, torch.ones_like(dsb), dsb))
    L = (L.clamp(-32, 31) + 32)
    L = torch.where(dsb == 0, torch.zeros_like(L), L).to(torch.uint8)  # 0..63

    nb = rows_f32.shape[0]
    ql = torch.zeros(nb, 128, dtype=torch.uint8, device=dev)
    qh = torch.zeros(nb, 64, dtype=torch.uint8, device=dev)
    for j in (0, 128):
        qo = 0 if j == 0 else 64
        ho = 0 if j == 0 else 32
        ql[:, qo:qo + 32] = (L[:, j:j + 32] & 0xF) | ((L[:, j + 64:j + 96] & 0xF) << 4)
        ql[:, qo + 32:qo + 64] = (L[:, j + 32:j + 64] & 0xF) | ((L[:, j + 96:j + 128] & 0xF) << 4)
        qh[:, ho:ho + 32] = ((L[:, j:j + 32] >> 4)
                             | ((L[:, j + 32:j + 64] >> 4) << 2)
                             | ((L[:, j + 64:j + 96] >> 4) << 4)
                             | ((L[:, j + 96:j + 128] >> 4) << 6))
    out = torch.zeros(nb, 210, dtype=torch.uint8, device=dev)
    out[:, 0:128] = ql
    out[:, 128:192] = qh
    out[:, 192:208] = sc.view(torch.uint8)
    # fp16 d bytes: easiest via cpu numpy (little-endian)
    d_bytes = d.half().cpu().numpy().view(np.uint8).reshape(nb, 2)
    out_cpu = out.cpu().numpy()
    out_cpu[:, 208:210] = d_bytes
    return out_cpu


def _nearest_int(f):
    # C: val = f + 12582912.0f; reinterpret low 23 bits minus 0x400000
    val = (f.astype(np.float32) + np.float32(12582912.0)).astype(np.uint32)
    return (val & np.uint32(0x007FFFFF)).astype(np.int32) - 0x400000


def _make_qx_scales(x):
    """make_qx_quants(n=16, nmax=32, rmse_type=1) -> per-group scale.
    x: [n_blocks*16, 16] float32. Only the scale is consumed by q6_K."""
    nmax = 32
    ax = np.abs(x)
    idx = ax.argmax(axis=1, keepdims=True)
    amax = np.take_along_axis(ax, idx, 1)[:, 0]
    mx = np.take_along_axis(x, idx, 1)[:, 0]
    out = np.zeros(x.shape[0], dtype=np.float32)
    nz = amax >= GROUP_MAX_EPS
    if not nz.any():
        return out
    xz = x[nz]
    mxz = mx[nz]
    w = xz * xz  # rmse_type == 1

    iscale = np.float32(-nmax) / mxz
    l = np.clip(_nearest_int(iscale[:, None] * xz), -nmax, nmax - 1)
    sumlx = (w * xz * l).sum(axis=1)
    suml2 = (w * l * l).sum(axis=1)
    scale = np.where(suml2 > 0, sumlx / np.where(suml2 > 0, suml2, 1), np.float32(0))
    best = scale * sumlx

    for is_ in range(-9, 10):
        if is_ == 0:
            continue
        isc = -(np.float32(nmax) + np.float32(0.1) * is_) / mxz
        l = np.clip(_nearest_int(isc[:, None] * xz), -nmax, nmax - 1)
        sumlx = (w * xz * l).sum(axis=1)
        suml2 = (w * l * l).sum(axis=1)
        better = (suml2 > 0) & (sumlx * sumlx > best * suml2)
        if better.any():
            scale = np.where(better, sumlx / np.where(suml2 > 0, suml2, 1), scale)
            best = np.where(better, scale * sumlx, best)
    out[nz] = scale
    return out


def _quantize_q6_k_numpy(rows):
    """rows: [nb, 256] float32 -> packed [nb, 210] uint8 (numpy reference path)."""
    nb = rows.shape[0]
    out = np.zeros((nb, 210), dtype=np.uint8)
    scales16 = _make_qx_scales(rows.reshape(nb * 16, 16)).reshape(nb, 16)

    aabs = np.abs(scales16)
    idx = aabs.argmax(axis=1, keepdims=True)
    max_abs = np.take_along_axis(aabs, idx, 1)[:, 0]
    max_scale = np.take_along_axis(scales16, idx, 1)[:, 0]
    nz = max_abs >= GROUP_MAX_EPS

    # block layout (ggml-common.h block_q6_K): ql[128] | qh[64] | scales[16] | d fp16
    iscale = np.zeros(nb, dtype=np.float32)
    iscale[nz] = np.float32(-128.0) / max_scale[nz]
    d = np.zeros(nb, dtype=np.float32)
    d[nz] = 1.0 / iscale[nz]
    out[:, 208:210] = d.astype(np.float16).view(np.uint8).reshape(nb, 2)

    sc = np.zeros((nb, 16), dtype=np.int32)
    sc[nz] = np.minimum(127, _nearest_int(iscale[nz, None] * scales16[nz]))
    out[:, 192:208] = sc.astype(np.int8).view(np.uint8)

    # requantize with final d*scale and pack (vectorized; zero scales -> L=0)
    ds = (d[:, None] * sc.astype(np.float32)).astype(np.float32)   # [nb,16]
    dsb = np.repeat(ds, 16, axis=1)
    zero = dsb == 0
    dsb = np.where(zero, np.float32(1.0), dsb)   # keep signs; only exact zeros replaced
    L = np.clip(_nearest_int(rows / dsb), -32, 31) + 32
    L[zero] = 0
    L64 = L.astype(np.uint8)  # values 0..63
    for j in (0, 128):
        q1 = L64[:, j:j + 32] & 0xF
        q2 = L64[:, j + 32:j + 64] & 0xF
        q3 = L64[:, j + 64:j + 96] & 0xF
        q4 = L64[:, j + 96:j + 128] & 0xF
        ql_off = 0 if j == 0 else 64
        qh_off = 128 + (0 if j == 0 else 32)
        out[:, ql_off:ql_off + 32] = q1 | (q3 << 4)
        out[:, ql_off + 32:ql_off + 64] = q2 | (q4 << 4)
        out[:, qh_off:qh_off + 32] = (
            (L64[:, j:j + 32] >> 4)
            | ((L64[:, j + 32:j + 64] >> 4) << 2)
            | ((L64[:, j + 64:j + 96] >> 4) << 4)
            | ((L64[:, j + 96:j + 128] >> 4) << 6)
        )
    return out


def quantize_q6_k(arr):
    """arr: float32, last dim multiple of 256 -> packed uint8 (gguf-py byte convention)."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    assert a.shape[-1] % QK_K == 0, f"last dim {a.shape[-1]} not multiple of {QK_K}"
    if _HAS_CUDA:
        out = _quantize_q6_k_torch(torch.from_numpy(a.reshape(-1, QK_K)).cuda())
    else:
        out = _quantize_q6_k_numpy(a.reshape(-1, QK_K))
    return out.reshape(*a.shape[:-1], -1)
