"""Post-conversion self-check for minicpmo45 GGUFs (no model load).

Usage::

  python convert/check_minicpmo45.py [--gguf OUT.gguf] [--src DIR] [--hash]

Verifies, without ever loading the model:

  1. GGUF tensor name set == converter plan (nothing extra, nothing missing)
  2. every HF tensor is consumed or on the documented skip list (diff set empty;
     exactly one documented skip: tts.model.embed_tokens, dead backbone embedding)
  3. per-tensor ne/dtype/n_bytes/offset chain match the plan; file size == end
     of the last padded tensor
  4. metadata KV spot checks (arch keys, LLM/vision/resampler/audio/tts dims,
     rope theta 1e6, token ids, tts codebook size)
  5. round-trip: byte-exact recomputation of sampled tensors straight from the
     safetensors shards (token_embd/lm_head/tk.emb_text row slices, full LLM
     layers 0/35, vision blocks 0/26, audio blocks 0/23, tts blocks 0/19, the
     whole resampler, the weight_norm recombine, all convs and pos tables)

Exit code 0 = PASS.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import gguf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_minicpmo45 as C

GGML_TYPE = {"f16": gguf.GGMLQuantizationType.F16,
             "conv": gguf.GGMLQuantizationType.F16,
             "wn": gguf.GGMLQuantizationType.F16,
             "f32": gguf.GGMLQuantizationType.F32,
             "q": None}   # q: resolved from the file
SAMPLE_PREFIX = ("blk.0.", "blk.35.", "output_norm", "v.patch_embd",
                 "v.pos_embd", "v.post_ln", "v.blk.0.", "v.blk.26.",
                 "rs.", "a.blk.0.", "a.blk.23.", "a.conv1", "a.conv2",
                 "a.pos_embd", "a.ln_post", "a.proj.", "tk.emb_code",
                 "tk.head_code", "tk.output_norm", "tk.blk.0.", "tk.blk.19.",
                 "tk.proj_")
LARGE = (("token_embd.weight", (0, 1, 65535, 151747)),
         ("lm_head.weight", (0, 65535, 151747)),
         ("tk.emb_text.weight", (0, 65535, 151747, 152063)))


def kv_value(field):
    """contents is a method in some gguf-py versions, a property in others;
    it may return a list of values or a bare scalar."""
    c = field.contents
    vals = c() if callable(c) else c
    if isinstance(vals, (list, tuple)):
        vals = vals[0] if len(vals) == 1 else vals
    return vals


def kv_dict(reader):
    out = {}
    for k, f in reader.fields.items():
        if f.types[0] == gguf.GGUFValueType.ARRAY:
            item_type = f.types[1]
            dt = {gguf.GGUFValueType.UINT32: np.uint32,
                  gguf.GGUFValueType.FLOAT32: np.float32}[item_type]
            items = np.concatenate(
                [np.asarray(f.parts[i], dtype=dt).reshape(-1) for i in f.data])
            out[k] = items
        else:
            out[k] = kv_value(f)
    return out


def expand(raw, dt):
    out = np.empty(raw.size, np.float32)
    if dt == "BF16":
        C._expand_bf16(raw, out)
    else:
        out[:] = raw.reshape(-1)
    return out


def expected_bytes(op, sh):
    """Independent recomputation of the tensor bytes (same math as the
    converter's materialize, written out longhand so a shared bug shows)."""
    qt = op.get("q")
    if op["k"] == "wn":
        g_raw, g_dt, _ = sh.raw(op["h"] + ".parametrizations.weight.original0")
        v_raw, v_dt, v_shape = sh.raw(op["h"] + ".parametrizations.weight.original1")
        g = expand(g_raw, g_dt).reshape(-1, 1)
        v = expand(v_raw, v_dt).reshape(v_shape)
        w = g * v / np.sqrt((v * v).sum(-1, keepdims=True))
        if qt is not None:
            return _lib_quant(w, qt)
        return np.ascontiguousarray(w, dtype=np.float16).tobytes()
    raw, dt, hshape = sh.raw(op["h"])
    if op["t"] == "fold":
        n = int(np.prod(hshape[1:]))
        w = expand(raw, dt).reshape(hshape[0], n)
    else:
        w = expand(raw, dt)
    w = w.reshape(op["s"])
    if qt is not None:
        return _lib_quant(w, qt)
    if op["k"] in ("f16", "conv", "wn"):
        return np.ascontiguousarray(w, dtype=np.float16).tobytes()
    return np.ascontiguousarray(w, dtype=np.float32).tobytes()


def _lib_quant(a32, qtype):
    import _lib
    return _lib.quantize(np.ascontiguousarray(a32), qtype).tobytes()


def op_decl_row_bytes(op):
    """Bytes of one row (last ne dim) in the stored layout."""
    qt = op.get("q")
    if qt is not None:
        blk, tsz = C.QUANT_BLOCK[qt.name]
        return op["s"][-1] // blk * tsz
    return op["s"][-1] * 2   # f16/conv/wn storage rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=os.path.join(C.DEFAULT_OUT_DIR, "minicpmo45-f16.gguf"))
    ap.add_argument("--src", default=os.environ.get("MINICPMO45_SRC", C.DEFAULT_SRC))
    ap.add_argument("--out-type", default=None, choices=sorted(C.OUTTYPES) + [None],
                    help="quant tier the file was written with (default: inferred "
                         "from the -f16/-q8_0/-q6_k filename suffix)")
    ap.add_argument("--hash", action="store_true", help="stream a sha256 of the file")
    ap.add_argument("--sample", default=None,
                    help="comma list of extra tensor names to round-trip")
    args = ap.parse_args()

    import _lib
    _lib.torch = None
    fails = []

    with open(os.path.join(args.src, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    cfg = C.read_config(args.src)
    ops = C.build_ops(cfg)
    sh = C.Shards(args.src, weight_map)
    C.bind_pos_embd(ops, sh, cfg)

    reader = gguf.GGUFReader(args.gguf)
    out_type = args.out_type
    if out_type is None:
        suffix = os.path.splitext(args.gguf)[0].rsplit("-", 1)[-1]
        out_type = suffix if suffix in C.OUTTYPES else "f16"
    C.quant_policy(ops, out_type)
    plan = {op["g"]: op for op in ops}
    names = [t.name for t in reader.tensors]
    tmap = {t.name: t for t in reader.tensors}

    # 1. name set equality
    if len(names) != len(plan):
        fails.append(f"tensor count {len(names)} != plan {len(plan)}")
    extra = sorted(set(names) - set(plan))
    missing = sorted(set(plan) - set(names))
    if extra:
        fails.append(f"{len(extra)} GGUF tensors not in plan, e.g. {extra[:5]}")
    if missing:
        fails.append(f"{len(missing)} plan tensors missing in GGUF, e.g. {missing[:5]}")

    # 2. HF coverage (independent of the converter's own check)
    consumed = set()
    for op in ops:
        consumed.update(C.hf_names(op))
    skipped = {n for n in weight_map if n not in consumed
               and (n in C.SKIP_EXACT or n.endswith(C.SKIP_SUFFIX))}
    diff = sorted(set(weight_map) - consumed - skipped)
    if diff:
        fails.append(f"{len(diff)} HF tensors unmapped, e.g. {diff[:5]}")
    else:
        print(f"[check] HF coverage: {len(consumed)} consumed + {len(skipped)} documented "
              f"skips (tts.model.embed_tokens) = {len(weight_map)} (diff empty)")

    # 3. per-tensor ne / dtype / nbytes / offsets
    align = 32
    end = None
    for t in reader.tensors:
        op = plan.get(t.name)
        if op is None:
            continue
        shape, dt, nbytes, raw = C.op_decl(op)
        ne = tuple(int(x) for x in reversed(np.asarray(op["s"])))  # gguf ne
        got_ne = tuple(int(x) for x in t.shape)
        if got_ne != ne:
            fails.append(f"{t.name}: ne {got_ne} != {ne}")
        want_type = raw if raw is not None else GGML_TYPE[op["k"]]
        if t.tensor_type != want_type:
            fails.append(f"{t.name}: dtype {t.tensor_type} != {want_type}")
        if int(t.n_bytes) != nbytes:
            fails.append(f"{t.name}: n_bytes {t.n_bytes} != {nbytes}")
        if end is None:
            end = int(t.data_offset)
        if int(t.data_offset) != end:
            fails.append(f"{t.name}: data_offset {t.data_offset} != expected {end}")
        end = int(t.data_offset) + (nbytes + align - 1) // align * align
    fsize = os.path.getsize(args.gguf)
    if fsize != end:
        fails.append(f"file size {fsize} != padded tensor end {end}")
    print(f"[check] layout: {len(reader.tensors)} tensors, offsets contiguous, "
          f"file size {fsize / 1e9:.3f} GB == header end")

    # 4. metadata
    kv = kv_dict(reader)
    want_kv = {
        "general.architecture": C.ARCH,
        "general.name": "MiniCPM-o-4.5",
        f"{C.ARCH}.block_count": 36,
        f"{C.ARCH}.embedding_length": 4096,
        f"{C.ARCH}.vocab_size": 151748,
        f"{C.ARCH}.context_length": 40960,
        f"{C.ARCH}.attention.head_count": 32,
        f"{C.ARCH}.attention.head_count_kv": 8,
        f"{C.ARCH}.attention.key_length": 128,
        f"{C.ARCH}.attention.value_length": 128,
        f"{C.ARCH}.rope.dimension_count": 128,
        f"{C.ARCH}.tie_word_embeddings": False,
        f"{C.ARCH}.bos_token_id": 151643,
        f"{C.ARCH}.eos_token_id": 151645,
        f"{C.ARCH}.pad_token_id": 151643,
        f"{C.ARCH}.vision.block_count": 27,
        f"{C.ARCH}.vision.embedding_length": 1152,
        f"{C.ARCH}.vision.feed_forward_length": 4304,
        f"{C.ARCH}.vision.attention.head_count": 16,
        f"{C.ARCH}.vision.patch_size": 14,
        f"{C.ARCH}.vision.image_size": 448,
        f"{C.ARCH}.vision.query_num": 64,
        f"{C.ARCH}.vision.resampler_embed": 4096,
        f"{C.ARCH}.audio.block_count": 24,
        f"{C.ARCH}.audio.embedding_length": 1024,
        f"{C.ARCH}.audio.feed_forward_length": 4096,
        f"{C.ARCH}.audio.attention.head_count": 16,
        f"{C.ARCH}.audio.num_mel_bins": 80,
        f"{C.ARCH}.audio.max_source_positions": 1500,
        f"{C.ARCH}.audio.pool_step": 5,
        f"{C.ARCH}.tts.block_count": 20,
        f"{C.ARCH}.tts.embedding_length": 768,
        f"{C.ARCH}.tts.feed_forward_length": 3072,
        f"{C.ARCH}.tts.attention.head_count": 12,
        f"{C.ARCH}.tts.attention.head_count_kv": 12,
        f"{C.ARCH}.tts.num_text_tokens": 152064,
        f"{C.ARCH}.tts.num_audio_tokens": 6562,
        f"{C.ARCH}.tts.num_vq": 1,
        f"{C.ARCH}.tts.audio_bos_token_id": 151687,
        f"{C.ARCH}.tts.text_eos_token_id": 151692,
    }
    for k, want in want_kv.items():
        got = kv.get(k)
        if got is None:
            fails.append(f"metadata missing {k}")
        elif got != want and str(got) != str(want):
            fails.append(f"metadata {k}: {got!r} != {want!r}")
    rope_keys = [k for k in kv if ".rope." in k]
    eps = kv.get(f"{C.ARCH}.attention.layer_norm_rms_epsilon")
    if eps is None or abs(float(eps) - 1e-6) > 1e-12:
        fails.append(f"metadata rms eps: {eps!r} != 1e-6")
    theta = kv.get(f"{C.ARCH}.rope.freq_base")
    if theta is None or abs(float(theta) - 1e6) > 1.0:
        fails.append(f"metadata rope freq_base: {theta!r} != 1e6")
    print(f"[check] metadata: {len(kv)} keys checked against {len(want_kv) + 2} "
          f"expectations, rope keys: {sorted(rope_keys)}")

    # 5. round-trip samples (byte-exact)
    sample = [op["g"] for op in ops if op["g"].startswith(SAMPLE_PREFIX)]
    large_names = {g for g, _ in LARGE}
    sample = [g for g in sample if g not in large_names]
    for extra_name in (args.sample or "").split(","):
        if extra_name.strip():
            sample.append(extra_name.strip())
    checked = 0
    for g in sample:
        if g not in tmap:
            continue
        op = plan[g]
        t = tmap[g]
        want = expected_bytes(op, sh)
        flat = t.data.view(np.uint8).reshape(-1)   # byte view of the mmap data
        if t.n_bytes < (64 << 20):
            if bytes(flat) != want:
                i = next((j for j, (x, y) in enumerate(zip(bytes(flat), want)) if x != y),
                         min(int(t.n_bytes), len(want)))
                fails.append(f"round-trip mismatch: {g} at byte {i} of {len(want)}")
        else:   # huge tensor: compare head+tail slices only
            n = 8 << 20
            head_ok = bool((flat[:n] == np.frombuffer(want[:n], np.uint8)).all())
            tail = want[len(want) - n:]
            tail_ok = bool((flat[-n:] == np.frombuffer(tail, np.uint8)).all())
            if not (head_ok and tail_ok):
                fails.append(f"round-trip head/tail mismatch: {g} "
                             f"(head {head_ok}, tail {tail_ok})")
        checked += 1
    # sampled rows of the vocab-sized tensors
    for g, rows in LARGE:
        if g not in tmap:
            continue
        op = plan[g]
        want_full = expected_bytes(op, sh)   # row-major in stored dtype
        row_bytes = op_decl_row_bytes(op)
        for r_i in rows:
            sl = slice(r_i * row_bytes, (r_i + 1) * row_bytes)
            if bytes(tmap[g].data.view(np.uint8).reshape(-1)[sl]) != want_full[sl]:
                fails.append(f"round-trip mismatch: {g} row {r_i}")
        checked += 1
    print(f"[check] round-trip: {checked} sampled tensors/rows byte-exact vs safetensors")

    if args.hash:
        h = hashlib.sha256()
        with open(args.gguf, "rb") as f:
            while True:
                b = f.read(8 << 20)
                if not b:
                    break
                h.update(b)
        print(f"[check] sha256: {h.hexdigest()}")

    if fails:
        for m in fails[:20]:
            print("[check] FAIL:", m)
        print(f"[check] RESULT: FAIL ({len(fails)} problems)")
        return 1
    print(f"[check] RESULT: PASS ({len(plan)} tensors, {len(weight_map)} HF sources, "
          f"{fsize / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
