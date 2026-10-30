"""The streaming-converter skeleton above _lib, shared by the big omni
converters (convert_minicpmo45.py / convert_nemotronomni.py).

_lib.py stays the dependency-light toolbox; this module holds the pieces the
big BF16-shard converters had duplicated wholesale (pulled out as pure code
motion - the outputs are gated byte-for-byte via SHA256, so no behavior may
change in transit):

* :class:`Shards` -- lazy mmap'd safetensors shard set (raw dtype views,
  never a full copy: the 18.7/66 GB checkpoints must never be resident)
* :func:`expand_bf16` -- chunked BF16 uint16 -> f32/f16 expansion
* ``QUANT_BLOCK`` -- (GGML_QUANT_SIZES) block size and bytes per block
* :func:`op_decl` -- per-op (shape_for_info, dtype, nbytes, raw_dtype); the
  per-model policy (which op kinds are quantizable / F16-contract) is passed
  in as ``quant_kinds`` / ``f16_kinds``
* :func:`log_quant_policy` -- the per-tier tensor-count summary line
* :func:`check_coverage` -- plan-vs-checkpoint contract guard (refuses to run
  on stale plans or partial shards)
* :func:`write_gguf` -- pre-declared header, then the data section streamed
  one op at a time (the caller owns metadata/materialize; those differ per
  model on purpose)

Model-specific maths stays in the converters: quant_policy tiers, materialize
transforms (weight_norm recombine, expert stacks, ...), and write_metadata.
"""
import json
import os
import time

import numpy as np
import gguf

from _lib import fail, log


# ------------------------------------------------------------- shard access
class Shards:
    """Lazy mmap'd safetensors shard set; raw dtype views without full copies."""

    _DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}

    def __init__(self, src, weight_map):
        self.dir = src
        self.map = weight_map
        self.handles = {}

    def handle(self, fname):
        if fname not in self.handles:
            import mmap, struct
            f = open(os.path.join(self.dir, fname), "rb")
            m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            n = struct.unpack("<Q", m[:8])[0]
            meta = json.loads(m[8:8 + n])
            base = 8 + n
            self.handles[fname] = (f, m, meta, base)
        return self.handles[fname]

    def meta(self, hf_name):
        e = self.handle(self.map[hf_name])[2][hf_name]
        return e["dtype"], e["shape"]

    def raw(self, hf_name):
        """Raw numpy view over the mmap: uint16 for BF16 (decode at use)."""
        fname = self.map[hf_name]
        f, m, meta, base = self.handle(fname)
        e = meta[hf_name]
        dt, (s, t) = e["dtype"], e["data_offsets"]
        arr = np.frombuffer(m[base + s:base + t], dtype=self._DT[dt])
        return arr.reshape(e["shape"]) if dt != "BF16" else arr.reshape(-1), dt, e["shape"]

    def close(self):
        for f, m, _, _ in self.handles.values():
            m.close()
            f.close()


def expand_bf16(raw, out):
    """BF16 uint16 -> float32/float16, chunked so the f32 copy stays bounded."""
    raw = raw.reshape(-1)
    step = 64 << 20
    flat = out.reshape(-1)
    for i in range(0, raw.size, step):
        c = raw[i:i + step].astype(np.uint32) << np.uint32(16)
        flat[i:i + step] = c.view(np.float32)
    return out


# (GGML_QUANT_SIZES) block size and bytes per block
QUANT_BLOCK = {"Q8_0": (32, 34), "Q5_0": (32, 22), "Q4_0": (32, 18),
               "Q6_K": (256, 210), "Q4_K": (256, 144)}


def peak_rss_mib():
    """Windows peak working set, MiB (psutil, falling back to raw ctypes)."""
    import ctypes
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / (1 << 20)
    except Exception:
        pass
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + \
                   [(n, ctypes.c_size_t) for n in
                    ("PeakWorkingSetSize", "WorkingSetSize",
                     "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                     "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                     "PagefileUsage", "PeakPagefileUsage")]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if ctypes.windll.kernel32.K32GetProcessMemoryInfo(
            ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess()),
            ctypes.byref(pmc), pmc.cb):
        return pmc.PeakWorkingSetSize / (1 << 20)
    return -1.0


def op_decl(op, quant_kinds, f16_kinds):
    """(shape_for_info, np_dtype, nbytes, raw_dtype) of the final tensor.

    ``quant_kinds``: op kinds that carry a quantized type once the tier policy
    assigned one (kind "q" always qualifies); ``f16_kinds``: the kinds that
    stay float16 when not quantized (everything else is F32).
    """
    qt = op.get("q")
    if op["k"] == "q" or (op["k"] in quant_kinds and qt is not None):
        # info takes the BYTE shape and stores the logical shape after
        # quant_shape_from_byte_shape; nbytes is the byte count
        blk, tsz = QUANT_BLOCK[qt.name]
        byte_shape = (*op["s"][:-1], op["s"][-1] // blk * tsz)
        return byte_shape, np.dtype("uint8"), int(np.prod(byte_shape)), qt
    dt = np.dtype("float16") if op["k"] in f16_kinds else np.dtype("float32")
    return op["s"], dt, int(np.prod(op["s"])) * dt.itemsize, None


def log_quant_policy(ops, out_type):
    """The per-tier tensor-count summary both converters log after quant_policy."""
    n_q = sum(1 for op in ops if op.get("q"))
    if n_q:
        by_type = {}
        for op in ops:
            if op.get("q"):
                by_type[op["q"].name] = by_type.get(op["q"].name, 0) + 1
        log("quant policy [" + out_type + "]: " +
            ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())) +
            f", f16/f32={len(ops) - n_q}")


def check_coverage(ops, weight_map, sh, names_of, skip_exact, skip_suffix):
    """Plan-vs-checkpoint contract guard: every op's HF tensors must exist in
    the weight map with a supported dtype, and every checkpoint tensor must be
    consumed or on the documented skip lists (nothing extra). ``names_of(op)``
    yields the HF tensor names an op consumes (multi-tensor ops like the
    weight_norm recombine or expert stacks merge several)."""
    consumed = set()
    for op in ops:
        for n in names_of(op):
            if n not in weight_map:
                fail(f"checkpoint has no tensor '{n}' (stale plan?)")
            dt, shape = sh.meta(n)
            if dt not in Shards._DT:
                fail(f"{n}: unsupported dtype {dt}")
            consumed.add(n)
    skipped = [n for n in weight_map
               if n not in consumed and (n in skip_exact
                                         or n.endswith(skip_suffix))]
    extra = sorted(set(weight_map) - consumed - set(skipped))
    if extra:
        fail(f"{len(extra)} checkpoint tensors are not covered by the mapping "
             f"(contract violation), e.g. {extra[:5]}")
    return consumed, skipped


def write_gguf(out, arch, ops, sh, write_metadata, op_decl_fn, materialize_fn,
               quiet=True):
    """Pre-declare every tensor (offsets are assigned sequentially), then
    stream the data one op at a time -- nothing big is ever held twice.
    ``write_metadata(gw)`` / ``op_decl_fn(op)`` / ``materialize_fn(op, sh)``
    are the model-owned policies."""
    t0 = time.time()
    gw = gguf.GGUFWriter(out, arch)
    write_metadata(gw)
    for op in ops:
        shape, dt, nbytes, raw = op_decl_fn(op)
        gw.add_tensor_info(op["g"], shape, dt, nbytes, raw_dtype=raw)
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_ti_data_to_file()

    done = 0
    for op in ops:
        arr = materialize_fn(op, sh)
        try:
            gw.write_tensor_data(np.ascontiguousarray(arr))
        except AssertionError:
            fail(f"{op['g']}: produced {arr.nbytes} bytes, header declared "
                 f"{op_decl_fn(op)[2]} (kind {op['k']}, shape {op['s']})")
        del arr
        done += 1
        if not quiet and (done % 128 == 0 or done == len(ops)):
            log(f"  {done}/{len(ops)} tensors, peak RSS {peak_rss_mib():.0f} MiB, "
                f"{time.time() - t0:.0f}s")
    gw.close()
    return time.time() - t0
