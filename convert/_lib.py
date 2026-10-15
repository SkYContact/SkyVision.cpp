"""Shared helpers for the HF -> GGUF converters in this directory.

Deliberately dependency-light (numpy + gguf + safetensors). Model specific maths
stays where it belongs: FP8 e4m3 decoding in convert_hf_to_gguf.py, the Q6_K
quantizer in q6k.py.

What lives here:

* :func:`log`, :func:`warn`, :func:`fail` -- one logging style for every script
* :func:`quantizer_available` -- gguf-py can *decode* many types but only
  *encode* a few; this probes it once per type instead of dying with a
  ``NotImplementedError`` halfway through a conversion
* :func:`quantize`, :func:`add_quantized` -- the raw_shape/raw_dtype dance that
  the gguf writer needs for pre-quantized blocks, in one place
* :class:`TensorSource` -- sharded safetensors reader with a name -> float32 API
* :func:`progress` -- compact in-place progress for the per-layer loops
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import gguf
from gguf import quants
from safetensors import safe_open

LOG_PREFIX = "[convert]"


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{LOG_PREFIX} warning: {msg}", file=sys.stderr, flush=True)


def fail(msg: str, code: int = 1):
    """Print a diagnostic and exit -- no traceback for user errors."""
    print(f"{LOG_PREFIX} error: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def progress(i: int, n: int, label: str, every: int = 1) -> None:
    """In-place progress: only the last update keeps a newline."""
    if n <= 1 or (i + 1) % every == 0 or i + 1 == n:
        end = "\n" if i + 1 == n else "\r"
        print(f"{LOG_PREFIX} {label} {i + 1}/{n}", end=end, flush=True)


# ---------------------------------------------------------------- quantization
_probe_cache: dict[gguf.GGMLQuantizationType, bool] = {}


def quantizer_available(qtype: gguf.GGMLQuantizationType) -> bool:
    """True when gguf-py can *encode* this type (probing a tiny block)."""
    if qtype not in _probe_cache:
        block = 256
        probe = np.zeros((block,), dtype=np.float32)
        try:
            quants.quantize(probe, qtype)
            _probe_cache[qtype] = True
        except Exception:
            _probe_cache[qtype] = False
    return _probe_cache[qtype]


def quantize(arr_f32: np.ndarray, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
    """Quantize a 2D float32 array, failing with a readable message."""
    if not quantizer_available(qtype):
        fail(f"gguf-py in this environment cannot *quantize* {qtype.name} "
             f"(it implements decode only); pick another --outtype")
    return quants.quantize(np.ascontiguousarray(arr_f32, dtype=np.float32), qtype)


def add_quantized(gw: gguf.GGUFWriter, name: str, blocks: np.ndarray,
                  qtype: gguf.GGMLQuantizationType) -> None:
    """Write pre-quantized blocks.

    ``blocks`` is the byte array returned by the quantizers: gguf stores the
    logical shape separately, hence raw_shape/raw_dtype (the writer turns the
    byte shape back into [out, in] via GGML_QUANT_SIZES).
    """
    gw.add_tensor(name, blocks, raw_shape=blocks.shape, raw_dtype=qtype)


def add_f16(gw: gguf.GGUFWriter, name: str, arr) -> None:
    gw.add_tensor(name, np.ascontiguousarray(arr.astype(np.float16)))


# ---------------------------------------------------------------- checkpoints
class TensorSource:
    """Sharded (or single file) safetensors reader returning float32 numpy."""

    def __init__(self, src: str):
        self.dir = src if os.path.isdir(src) else os.path.dirname(src) or "."
        index = os.path.join(self.dir, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index, encoding="utf-8") as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            self.weight_map = {}
        files = sorted(set(self.weight_map.values())) or sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(self.dir, "*.safetensors")))
        if not files:
            fail(f"no *.safetensors found in {self.dir}")
        self.handles = {f: safe_open(os.path.join(self.dir, f), framework="pt") for f in files}
        self.files = files

    def __len__(self) -> int:
        return len(self.weight_map) if self.weight_map else sum(
            len(h.keys()) for h in self.handles.values())

    def names(self):
        if self.weight_map:
            return list(self.weight_map)
        out = []
        for h in self.handles.values():
            out += list(h.keys())
        return out

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map if self.weight_map else any(
            name in h.keys() for h in self.handles.values())

    def get(self, name: str, dtype=np.float32) -> np.ndarray:
        """Fetch a tensor as ``dtype`` (float32 by default)."""
        if name not in self:
            fail(f"checkpoint has no tensor '{name}' "
                 f"(wrong --src? expected the HF release of this model)")
        if self.weight_map:
            handle = self.handles[self.weight_map[name]]
        else:
            handle = next(h for h in self.handles.values() if name in h.keys())
        # framework="pt" always hands back a torch tensor (bf16/fp16 included)
        return handle.get_tensor(name).to("cpu").float().numpy().astype(dtype, copy=False)
