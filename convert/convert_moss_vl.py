"""Convert a MOSS-VL-Realtime NF4 release (bitsandbytes W4A16) to GGUF.

Usage::

  python convert/convert_moss_vl.py [--src DIR] [-o OUT.gguf]
                                   [--nf4-out q4k|q6k|q8_0|f16]
                                   [--bf16-text-out q6k|q8_0|f16]
                                   [--vision-out f16|q6k] [--f16-all]

The checkpoint is located like the other converters (``--src``, then
``$MOSS_VL_SRC``); point it at the local release::

  --src D:/MOSS-VL-Realtime-NF4

Produced layout (one self-contained file; the engine reads it plus the
checkpoint's ``tokenizer.json``)::

  --- text decoder, 48 layers, 12 of them cross-attention ---
  token_embd.weight / output.weight    BF16 linears -> --bf16-text-out
  blk.{i}.attn_norm / ffn_norm.weight  RMSNorm (F32)
  blk.{i}.attn_q/k/v/output.weight     self layers: NF4-dequant -> --nf4-out
  blk.{i}.cross_attn_q/k/v/output      cross layers: BF16 -> --bf16-text-out
  blk.{i}.cross_gate_attn/mlp.weight   tanh gates, [1] (F32)
  blk.{i}.ffn_gate/up/down.weight      NF4 in layers 4..43, else --bf16-text-out

  --- vision tower (27 layers) + merger, --vision-out by default ---
  v.patch_embd.weight [out, c, p, p]   conv3d temporal dim squeezed
  v.pos_emb.weight    [2304, 1152]
  v.blk.{i}.wqkv/wo   q/k rows permuted + zero-padded 72->80 (rope_layout=2)
  v.blk.{i}.ln1/ln2, ffn_up/ffn_down
  mm.ln0..3, mm.fc1/fc2, mm.separator.weight

Tensors are quantized and streamed to disk one at a time (all tensor infos are
registered up front, then the data is produced tensor by tensor), so neither
the 11 GB checkpoint nor the dequantized copies ever sit in RAM at once.
"""
import argparse
import json
import mmap
import os
import struct
import sys

import numpy as np
import gguf

import _lib
import hf_source
import q4k
from _lib import log, warn, fail, progress

MODEL_ID = "MOSS-VL-Realtime-NF4"
ENV_VAR = "MOSS_VL_SRC"
PATTERNS = ("model.safetensors.index.json", "model.safetensors")
ARCH = "moss_vl"
LM = "model.language_model"
VT = "model.visual"
WQKV_PAD = 80                                     # fused vision rope rows per head
NF4_QUANT_STATE = "weight.quant_state.bitsandbytes__nf4"
NF4_AUX = ("weight.absmax", NF4_QUANT_STATE, "weight.quant_map",
           "weight.nested_absmax", "weight.nested_quant_map")

NF4_OUTTYPES = {"q4k": "q4_k", "q6k": "q6_k", "q8_0": "q8_0", "f16": "f16"}
TEXT_OUTTYPES = {"q6k": "q6_k", "q8_0": "q8_0", "f16": "f16"}
VISION_OUTTYPES = {"f16": "f16", "q6k": "q6_k"}
QTYPES = {"q4_k": gguf.GGMLQuantizationType.Q4_K,
          "q6_k": gguf.GGMLQuantizationType.Q6_K,
          "q8_0": gguf.GGMLQuantizationType.Q8_0}
BYTES_PER_BLOCK = {"q4_k": (256, 144), "q6_k": (256, 210), "q8_0": (32, 34)}

# defaults straight from the released config.json (overridden by it when present)
TEXT_CFG = dict(n_layer=48, n_embd=4096, n_head=32, n_kv_head=8, head_dim=128,
                n_ff=12288, n_vocab=151936, rms_eps=1e-6, rope_theta=5e6,
                ctx=262144, mrope=(24, 20, 20), mrope_interleaved=True,
                cross_layers=(2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46))
VIS_CFG = dict(n_layer=27, n_embd=1152, n_head=16, n_ff=4304, patch=16, merge=2,
               temporal=1, out_embd=4096, n_pos=2304, deepstack=(8, 16, 24))


def read_config(src_dir):
    """Merge the checkpoint's config.json into the built-in defaults."""
    path = os.path.join(src_dir, "config.json")
    if not os.path.isfile(path):
        warn("no config.json next to the weights, using built-in defaults")
        return TEXT_CFG.copy(), VIS_CFG.copy()
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    tc, vc = cfg.get("text_config", {}), cfg.get("vision_config", {})
    scaling = tc.get("rope_scaling") or {}
    text = TEXT_CFG.copy()
    text.update({k: v for k, v in dict(
        n_layer=tc.get("num_hidden_layers"), n_embd=tc.get("hidden_size"),
        n_head=tc.get("num_attention_heads"), n_kv_head=tc.get("num_key_value_heads"),
        head_dim=tc.get("head_dim"), n_ff=tc.get("intermediate_size"),
        n_vocab=tc.get("vocab_size"), rms_eps=tc.get("rms_norm_eps"),
        rope_theta=tc.get("rope_theta"), ctx=tc.get("max_position_embeddings"),
        mrope=tuple(scaling.get("mrope_section") or text["mrope"]),
        mrope_interleaved=scaling.get("mrope_interleaved", True),
        cross_layers=tuple(tc.get("cross_attention_layers") or text["cross_layers"]),
    ).items() if v is not None})
    vis = VIS_CFG.copy()
    vis.update({k: v for k, v in dict(
        n_layer=vc.get("depth"), n_embd=vc.get("hidden_size"),
        n_head=vc.get("num_heads"), n_ff=vc.get("intermediate_size"),
        patch=vc.get("patch_size"), merge=vc.get("spatial_merge_size"),
        temporal=vc.get("temporal_patch_size"), out_embd=vc.get("out_hidden_size"),
        n_pos=vc.get("num_position_embeddings"),
        deepstack=tuple(vc.get("deepstack_visual_indexes") or vis["deepstack"]),
    ).items() if v is not None})
    return text, vis


# ------------------------------------------------------------------ bnb NF4
def dequantize_nf4(reader, module):
    """One bitsandbytes NF4 linear -> float32 (out, in) numpy.

    Layout (pinned by convert/inspect_moss_vl_nf4.py): uint8 weights, two
    4-bit codes per byte (HIGH nibble = even element, as bnb packs them),
    64-element blocks scaled by absmax; absmax is itself quantized (256-entry
    codebook indexed by the absmax bytes, per-256 scales in nested_absmax,
    plus nested_offset). Done in numpy: freed torch-CPU blocks keep their
    commit charge on Windows, which fragments across tensor sizes.
    """
    packed = reader.raw(f"{module}.weight", np.uint8).ravel()
    blob = reader.raw(f"{module}.{NF4_QUANT_STATE}", np.uint8).tobytes()
    qs = json.loads(blob.decode("utf-8"))
    if qs.get("quant_type") != "nf4":
        fail(f"{module}: unsupported bnb quant_type {qs.get('quant_type')!r}")
    out_dim, in_dim = qs["shape"]
    bs, nbs, offset = qs["blocksize"], qs["nested_blocksize"], qs["nested_offset"]
    qmap = reader.raw(f"{module}.weight.quant_map").ravel()
    absmax = reader.raw(f"{module}.weight.absmax", np.uint8).ravel()
    nested_map = reader.raw(f"{module}.weight.nested_quant_map").ravel()
    nested_absmax = reader.raw(f"{module}.weight.nested_absmax").ravel()

    n = out_dim * in_dim
    if qmap.size != 16 or 2 * packed.size != n or absmax.size * bs != n:
        fail(f"{module}: inconsistent NF4 tensors "
             f"(packed {packed.size}, absmax {absmax.size}, shape {out_dim}x{in_dim})")
    groups = (absmax.size + nbs - 1) // nbs
    if nested_absmax.size != groups or nested_map.size != 256:
        fail(f"{module}: inconsistent nested absmax "
             f"({absmax.size} codes, {nested_absmax.size} scales)")

    codes = np.empty(n, dtype=np.uint8)
    codes[0::2] = packed >> 4
    codes[1::2] = packed & 0xF
    vals = qmap[codes].reshape(-1, bs)                       # float32
    scale = nested_map[absmax]
    scale *= np.repeat(nested_absmax, nbs)[:absmax.size]
    scale += offset
    vals *= scale[:, None]
    return vals.reshape(out_dim, in_dim)


# ------------------------------------------------------------------ wqkv perm
def permute_wqkv(arr, n_head, head_dim):
    """Permute + zero-pad q/k rows for ggml_rope_multi(GGML_ROPE_TYPE_VISION),
    rope_layout=2: [x-re | y-re | pad | x-im | y-im | pad]. v rows only padded.
    The HF source is [row_re | col_re | row_im | col_im] (freqs = [row | col],
    emb = cat(freqs, freqs), rotate_half pairs (i, i + head_dim/2)).
    """
    npad = WQKV_PAD - head_dim
    half = head_dim // 2
    perm = list(range(half)) + [-1] * (npad // 2) + list(range(half, head_dim)) + [-1] * (npad // 2)
    rows = arr.shape[0]
    heads = rows // head_dim
    if rows != heads * head_dim:
        fail(f"wqkv rows {rows} not a multiple of head_dim {head_dim}")
    out = np.zeros((heads * WQKV_PAD,) + arr.shape[1:], dtype=np.float32)
    for h in range(heads):
        src = arr[h * head_dim:(h + 1) * head_dim]
        dst = out[h * WQKV_PAD:(h + 1) * WQKV_PAD]
        if h >= 2 * n_head:                      # v: plain pad at the end
            dst[:head_dim] = src
        else:
            for j, s in enumerate(perm):
                if s >= 0:
                    dst[j] = src[s]
    return out


# ------------------------------------------------------------------ metadata
def write_metadata(gw, text, vis, nf4_out):
    def add_u32_array(key, values):
        gw.add_key_value(key, list(np.array(values, dtype=np.uint32)),
                         gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.UINT32)

    def add_f32_array(key, values):
        gw.add_key_value(key, [float(v) for v in values],
                         gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.FLOAT32)

    gw.add_architecture()
    gw.add_name("Moss-VL-Realtime")
    gw.add_file_type({
        "q4k": gguf.LlamaFileType.MOSTLY_Q4_K_M,
        "q6k": gguf.LlamaFileType.MOSTLY_Q6_K,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
    }[nf4_out])
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32("moss_vl.block_count", text["n_layer"])
    gw.add_uint32("moss_vl.embedding_length", text["n_embd"])
    gw.add_uint32("moss_vl.feed_forward_length", text["n_ff"])
    gw.add_uint32("moss_vl.attention.head_count", text["n_head"])
    gw.add_uint32("moss_vl.attention.head_count_kv", text["n_kv_head"])
    gw.add_uint32("moss_vl.attention.key_length", text["head_dim"])
    gw.add_float32("moss_vl.attention.layer_norm_rms_epsilon", text["rms_eps"])
    gw.add_float32("moss_vl.rope.freq_base", float(text["rope_theta"]))
    gw.add_uint32("moss_vl.rope.dimension_count", text["head_dim"])
    gw.add_uint32("moss_vl.context_length", text["ctx"])
    gw.add_uint32("moss_vl.vocab_size", text["n_vocab"])
    sec_t, sec_h, sec_w = text["mrope"]
    gw.add_uint32("moss_vl.rope.mrope_section_t", sec_t)
    gw.add_uint32("moss_vl.rope.mrope_section_h", sec_h)
    gw.add_uint32("moss_vl.rope.mrope_section_w", sec_w)
    gw.add_bool("moss_vl.rope.mrope_interleaved", bool(text["mrope_interleaved"]))
    add_u32_array("moss_vl.cross_layer_indexes", text["cross_layers"])

    gw.add_uint32("moss_vl.vision.depth", vis["n_layer"])
    gw.add_uint32("moss_vl.vision.embedding_length", vis["n_embd"])
    gw.add_uint32("moss_vl.vision.num_heads", vis["n_head"])
    gw.add_uint32("moss_vl.vision.patch_size", vis["patch"])
    gw.add_uint32("moss_vl.vision.spatial_merge_size", vis["merge"])
    gw.add_uint32("moss_vl.vision.temporal_patch_size", vis["temporal"])
    gw.add_uint32("moss_vl.vision.feed_forward_length", vis["n_ff"])
    gw.add_uint32("moss_vl.vision.out_embedding_length", vis["out_embd"])
    gw.add_uint32("moss_vl.vision.num_position_embeddings", vis["n_pos"])
    add_u32_array("moss_vl.vision.deepstack_indexes", vis["deepstack"])
    gw.add_float32("moss_vl.vision.rope_freq_base", 10000.0)
    add_f32_array("moss_vl.vision.image_mean", (0.5, 0.5, 0.5))
    add_f32_array("moss_vl.vision.image_std", (0.5, 0.5, 0.5))
    gw.add_uint32("moss_vl.vision.rope_layout", 2)


# ------------------------------------------------------------------ plan
class OutTensor:
    def __init__(self, name, hf_name, group, shape, kind):
        self.name, self.hf_name, self.group, self.kind = name, hf_name, group, kind
        self.qtype = QTYPES.get(kind)
        self.set_shape(shape)

    def set_shape(self, shape):
        self.shape = tuple(shape)
        if self.qtype is not None:
            block, bytes_per = BYTES_PER_BLOCK[self.kind]
            if self.shape[-1] % block:
                fail(f"{self.hf_name}: row length {self.shape[-1]} not a multiple of {block}")
            self.bytes_shape = self.shape[:-1] + (self.shape[-1] // block * bytes_per,)
        else:
            self.bytes_shape = self.shape
        self.nbytes = int(np.prod(self.bytes_shape)) * (1 if self.qtype is not None else
                                                       (2 if self.kind == "f16" else 4))


def build_plan(text, vis, nf4_modules, f16_all, args, merger_shapes):
    """Every output tensor in write order, with its quantization group."""
    cross = set(text["cross_layers"])
    plan: list[OutTensor] = []

    def add(name, hf_name, group, shape):
        if f16_all:                              # parity switch: keep F16/F32
            kind = "f32" if len(shape) < 2 else "f16"
        elif group == "nf4":
            kind = NF4_OUTTYPES[args.nf4_out]
        elif group == "text":
            kind = TEXT_OUTTYPES[args.bf16_text_out]
        elif group in ("vision", "vision_wqkv"):
            kind = VISION_OUTTYPES[args.vision_out]
        elif group == "separator":
            kind = "f16"
        else:
            kind = "f32"                          # norms, biases, gates
        plan.append(OutTensor(name, hf_name, group, shape, kind))

    def linear(name, module, shape, group=None):
        add(name, f"{module}.weight", group or ("nf4" if module in nf4_modules else "text"),
            shape)

    add("token_embd.weight", f"{LM}.embed_tokens.weight", "text",
        (text["n_vocab"], text["n_embd"]))
    add("output.weight", "lm_head.weight", "text", (text["n_vocab"], text["n_embd"]))
    add("output_norm.weight", f"{LM}.norm.weight", "1d", (text["n_embd"],))

    for i in range(text["n_layer"]):
        p, b = f"{LM}.layers.{i}", f"blk.{i}"
        add(f"{b}.attn_norm.weight", f"{p}.input_layernorm.weight", "1d", (text["n_embd"],))
        add(f"{b}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight", "1d",
            (text["n_embd"],))
        if i in cross:
            attn = f"{p}.cross_attn"
            linear(f"{b}.cross_attn_q.weight", f"{attn}.q_proj",
                   (text["n_head"] * text["head_dim"], text["n_embd"]), "text")
            linear(f"{b}.cross_attn_k.weight", f"{attn}.k_proj",
                   (text["n_kv_head"] * text["head_dim"], text["n_embd"]), "text")
            linear(f"{b}.cross_attn_v.weight", f"{attn}.v_proj",
                   (text["n_kv_head"] * text["head_dim"], text["n_embd"]), "text")
            linear(f"{b}.cross_attn_output.weight", f"{attn}.o_proj",
                   (text["n_embd"], text["n_embd"]), "text")
            add(f"{b}.cross_attn_q_norm.weight", f"{attn}.q_norm.weight", "1d",
                (text["head_dim"],))
            add(f"{b}.cross_attn_k_norm.weight", f"{attn}.k_norm.weight", "1d",
                (text["head_dim"],))
            add(f"{b}.cross_gate_attn.weight", f"{p}.cross_attn_attn_gate", "1d", (1,))
            add(f"{b}.cross_gate_mlp.weight", f"{p}.cross_attn_mlp_gate", "1d", (1,))
        else:
            linear(f"{b}.attn_q.weight", f"{p}.self_attn.q_proj",
                   (text["n_head"] * text["head_dim"], text["n_embd"]))
            linear(f"{b}.attn_k.weight", f"{p}.self_attn.k_proj",
                   (text["n_kv_head"] * text["head_dim"], text["n_embd"]))
            linear(f"{b}.attn_v.weight", f"{p}.self_attn.v_proj",
                   (text["n_kv_head"] * text["head_dim"], text["n_embd"]))
            linear(f"{b}.attn_output.weight", f"{p}.self_attn.o_proj",
                   (text["n_embd"], text["n_embd"]))
            add(f"{b}.attn_q_norm.weight", f"{p}.self_attn.q_norm.weight", "1d",
                (text["head_dim"],))
            add(f"{b}.attn_k_norm.weight", f"{p}.self_attn.k_norm.weight", "1d",
                (text["head_dim"],))
        linear(f"{b}.ffn_gate.weight", f"{p}.mlp.gate_proj", (text["n_ff"], text["n_embd"]))
        linear(f"{b}.ffn_up.weight", f"{p}.mlp.up_proj", (text["n_ff"], text["n_embd"]))
        linear(f"{b}.ffn_down.weight", f"{p}.mlp.down_proj", (text["n_embd"], text["n_ff"]))
        if not args.quiet:
            progress(i, text["n_layer"], "text layer", every=6)

    nh = vis["n_head"]
    hd = vis["n_embd"] // vis["n_head"]
    add("v.patch_embd.weight", f"{VT}.patch_embed.proj.weight", "vision",
        (vis["n_embd"], 3, vis["patch"], vis["patch"]))
    add("v.patch_embd.bias", f"{VT}.patch_embed.proj.bias", "1d", (vis["n_embd"],))
    add("v.pos_emb.weight", f"{VT}.pos_embed.weight", "vision", (vis["n_pos"], vis["n_embd"]))
    for i in range(vis["n_layer"]):
        p, b = f"{VT}.blocks.{i}", f"v.blk.{i}"
        add(f"{b}.ln1.weight", f"{p}.norm1.weight", "1d", (vis["n_embd"],))
        add(f"{b}.ln1.bias", f"{p}.norm1.bias", "1d", (vis["n_embd"],))
        add(f"{b}.ln2.weight", f"{p}.norm2.weight", "1d", (vis["n_embd"],))
        add(f"{b}.ln2.bias", f"{p}.norm2.bias", "1d", (vis["n_embd"],))
        linear(f"{b}.wqkv.weight", f"{p}.attn.qkv", (3 * nh * WQKV_PAD, vis["n_embd"]),
               "vision_wqkv")
        add(f"{b}.wqkv.bias", f"{p}.attn.qkv.bias", "vision_wqkv", (3 * nh * WQKV_PAD,))
        linear(f"{b}.wo.weight", f"{p}.attn.proj", (vis["n_embd"], vis["n_embd"]), "vision")
        add(f"{b}.wo.bias", f"{p}.attn.proj.bias", "1d", (vis["n_embd"],))
        linear(f"{b}.ffn_up.weight", f"{p}.mlp.linear_fc1", (vis["n_ff"], vis["n_embd"]),
               "vision")
        add(f"{b}.ffn_up.bias", f"{p}.mlp.linear_fc1.bias", "1d", (vis["n_ff"],))
        linear(f"{b}.ffn_down.weight", f"{p}.mlp.linear_fc2", (vis["n_embd"], vis["n_ff"]),
               "vision")
        add(f"{b}.ffn_down.bias", f"{p}.mlp.linear_fc2.bias", "1d", (vis["n_embd"],))
        if not args.quiet:
            progress(i, vis["n_layer"], "vision layer", every=6)

    merger = f"{VT}.merger"
    for j in range(len(vis["deepstack"]) + 1):
        add(f"mm.ln{j}.weight", f"{merger}.norms.{j}.weight", "1d", (vis["n_embd"],))
        add(f"mm.ln{j}.bias", f"{merger}.norms.{j}.bias", "1d", (vis["n_embd"],))
    add("mm.fc1.weight", f"{merger}.linear_fc1.weight", "vision", merger_shapes["mm.fc1.weight"])
    add("mm.fc1.bias", f"{merger}.linear_fc1.bias", "1d", merger_shapes["mm.fc1.bias"])
    add("mm.fc2.weight", f"{merger}.linear_fc2.weight", "vision", merger_shapes["mm.fc2.weight"])
    add("mm.fc2.bias", f"{merger}.linear_fc2.bias", "1d", merger_shapes["mm.fc2.bias"])
    add("mm.separator.weight", "model.separator_token", "separator", (text["n_embd"],))
    return plan


def quantize(arr, kind):
    """(out, in) float32 -> packed uint8 blocks, or None to keep F16."""
    if kind == "q4_k":
        return q4k.quantize_q4_k(arr)
    if kind == "q6_k":
        import q6k                              # gguf-py decodes Q6_K only
        out = np.empty((arr.shape[0], arr.shape[1] // 256 * 210), dtype=np.uint8)
        step = 4096                             # bound the GPU working set
        for i in range(0, arr.shape[0], step):
            out[i:i + step] = q6k.quantize_q6_k(
                np.ascontiguousarray(arr[i:i + step], dtype=np.float32))
        return out
    if kind == "q8_0":
        return _lib.quantize(arr, QTYPES[kind])
    return None


def squeeze_temporal(arr, t):
    if arr.ndim != 5 or arr.shape[2] != 1 or arr.shape[0:2] + arr.shape[3:5] != t.shape:
        fail(f"{t.hf_name}: unexpected conv shape {arr.shape}, expected (out, 3, 1, p, p)")
    return np.ascontiguousarray(arr.reshape(t.shape), dtype=np.float32)


class CheckpointReader:
    """Streaming safetensors reader: one shard mmap at a time.

    Windows charges a whole-file commit for every mapped shard, so the plan is
    processed shard by shard with a single mapping live. BF16 weights are
    up-converted to float32 in row chunks (bit shift into the high half -- an
    exact no-op), which means a big linear never exists as a full FP32 copy
    next to its mmap.
    """
    DTYPES = {"U8": np.uint8, "F32": np.float32, "F16": np.float16, "I64": np.int64}

    def __init__(self, src):
        self.dir = src
        index = os.path.join(src, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index, encoding="utf-8") as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            self.weight_map = {n: "model.safetensors"
                               for n in self._single_file_names(src)}
        self.files = sorted(set(self.weight_map.values()))
        self._file = self._mm = None
        self._shard = self._header = None
        self._data_start = 0

    @staticmethod
    def _single_file_names(src):
        import glob
        names = {}
        for path in glob.glob(os.path.join(src, "*.safetensors")):
            with open(path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n))
            header.pop("__metadata__", None)
            for name in header:
                names[name] = os.path.basename(path)
        return names

    def _open(self, shard):
        if self._shard == shard:
            return
        self.close()
        self._file = open(os.path.join(self.dir, shard), "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        (n,) = struct.unpack("<Q", self._mm[:8])
        self._header = json.loads(self._mm[8:8 + n])
        self._header.pop("__metadata__", None)
        self._data_start = 8 + n
        self._shard = shard

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = self._header = None
        if self._file is not None:
            self._file.close()
            self._file = None
        self._shard = None

    def shape_of(self, hf_name):
        self._open(self.weight_map[hf_name])
        return tuple(self._header[hf_name]["shape"])

    def _slice(self, hf_name, dtype, row_offset, rows):
        """float/uint view of `rows` rows (all rows if None) of one tensor."""
        self._open(self.weight_map[hf_name])
        info = self._header[hf_name]
        shape = tuple(info["shape"])
        off = self._data_start + info["data_offsets"][0]
        per_row = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        if rows is None:
            count = per_row * (shape[0] if shape else 1)
            arr = np.frombuffer(self._mm, dtype=dtype, count=count, offset=off)
            return arr.reshape(shape)
        start = off + row_offset * per_row * np.dtype(dtype).itemsize
        arr = np.frombuffer(self._mm, dtype=dtype, count=rows * per_row, offset=start)
        return arr.reshape((rows,) + shape[1:])

    def raw(self, hf_name, dtype=None):
        """Small tensors verbatim (uint8 codes, float32 bookkeeping)."""
        self._open(self.weight_map[hf_name])
        if dtype is None:
            dtype = self.DTYPES[self._header[hf_name]["dtype"]]
        return np.array(self._slice(hf_name, dtype, 0, None))

    def f32_rows(self, hf_name, r0, r1):
        """Rows [r0, r1) of a BF16/F32 tensor as float32 (exact conversion)."""
        self._open(self.weight_map[hf_name])
        if self._header[hf_name]["dtype"] == "BF16":
            u = self._slice(hf_name, np.uint16, r0, r1 - r0)
            return (u.astype(np.uint32) << 16).view(np.float32)
        return np.array(self._slice(hf_name, np.float32, r0, r1 - r0), dtype=np.float32)

    def f32(self, hf_name):
        """Whole tensor as float32 -- only for tensors that stay small."""
        self._open(self.weight_map[hf_name])
        if self._header[hf_name]["dtype"] == "BF16":
            u = self._slice(hf_name, np.uint16, 0, None)
            return (u.astype(np.uint32) << 16).view(np.float32).reshape(u.shape)
        return np.array(self._slice(hf_name, np.float32, 0, None), dtype=np.float32)


def produce(reader, t, vis_head, chunk_rows=4096):
    """Materialize one planned tensor's stored bytes/array."""
    if t.group == "nf4":
        arr = dequantize_nf4(reader, t.hf_name[:-len(".weight")])
        blocks = quantize(arr, t.kind)
        if blocks is None:
            return np.ascontiguousarray(arr, dtype=np.float16)
        return blocks.reshape(t.bytes_shape)

    if t.group == "vision_wqkv":
        arr = permute_wqkv(reader.f32(t.hf_name), *vis_head)
        if t.kind == "f16":
            return np.ascontiguousarray(arr, dtype=np.float16)
        return quantize(arr, t.kind).reshape(t.bytes_shape)

    if t.name == "v.patch_embd.weight":
        arr = squeeze_temporal(reader.f32(t.hf_name), t)
        if t.kind == "f32":
            return arr
        return np.ascontiguousarray(arr, dtype=np.float16)

    if t.kind == "f32":
        arr = reader.f32(t.hf_name)
        if t.shape != arr.shape:
            fail(f"{t.hf_name}: shape {arr.shape} != expected {t.shape}")
        return np.ascontiguousarray(arr, dtype=np.float32)

    f32_bytes = int(np.prod(t.shape)) * 4
    if t.kind == "q6_k" and f32_bytes > 512 * 2**20:
        # embed/lm_head: quantize from row chunks, never a full FP32 copy
        import q6k                              # gguf-py decodes Q6_K only
        out = np.empty(t.bytes_shape, dtype=np.uint8)
        for r0 in range(0, t.shape[0], chunk_rows):
            r1 = min(r0 + chunk_rows, t.shape[0])
            out[r0:r1] = q6k.quantize_q6_k(reader.f32_rows(t.hf_name, r0, r1))
        return out
    if t.kind == "f16" and f32_bytes > 512 * 2**20:
        out = np.empty(t.bytes_shape, dtype=np.float16)
        for r0 in range(0, t.shape[0], 4 * chunk_rows):
            r1 = min(r0 + 4 * chunk_rows, t.shape[0])
            out[r0:r1] = reader.f32_rows(t.hf_name, r0, r1)
        return out

    arr = reader.f32(t.hf_name)
    if t.shape != arr.shape:
        fail(f"{t.hf_name}: shape {arr.shape} != expected {t.shape}")
    if t.qtype is not None:
        return quantize(arr, t.kind).reshape(t.bytes_shape)
    return np.ascontiguousarray(arr, dtype=np.float16)


# ------------------------------------------------------------------ main
def verify(path, plan):
    """Reopen the file and check the tensor table (headers only, cheap)."""
    reader = gguf.GGUFReader(path)
    names = {t.name for t in reader.tensors}
    expected = {t.name for t in plan}
    missing, extra = expected - names, names - expected
    if missing:
        fail(f"written GGUF is missing {sorted(missing)[:8]}")
    if extra:
        fail(f"written GGUF has extra tensors {sorted(extra)[:8]}")
    by_type: dict[str, int] = {}
    for t in reader.tensors:
        by_type[str(t.tensor_type.name)] = by_type.get(str(t.tensor_type.name), 0) + 1
    log(f"verified {len(names)} tensors in {os.path.basename(path)}: "
        + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    field = reader.fields.get("general.architecture")
    if field is not None:
        value = field.contents()
        log(f"general.architecture = {value.decode() if isinstance(value, bytes) else value}")
    reader.close() if hasattr(reader, "close") else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nf4-out", default="q4k", choices=sorted(NF4_OUTTYPES),
                    help="quantization for the dequantized NF4 linears (default q4k)")
    ap.add_argument("--bf16-text-out", default="q6k", choices=sorted(TEXT_OUTTYPES),
                    help="quantization for the BF16 text linears (default q6k)")
    ap.add_argument("--vision-out", default="f16", choices=sorted(VISION_OUTTYPES),
                    help="quantization for the vision tower and merger (default f16)")
    ap.add_argument("--f16-all", action="store_true",
                    help="store every >=2D tensor as F16 (parity runs); 1-D stays F32")
    ap.add_argument("-o", "--out", default=None,
                    help="output file (default Moss-VL-Realtime-<nf4-out>.gguf)")
    ap.add_argument("-q", "--quiet", action="store_true", help="less progress output")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"Moss-VL-Realtime-{args.nf4_out}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} (nf4={args.nf4_out} bf16-text={args.bf16_text_out} "
        f"vision={args.vision_out}{' f16-all' if args.f16_all else ''})")

    text, vis = read_config(src)
    log(f"text: {text['n_layer']}L d={text['n_embd']} h={text['n_head']}/{text['n_kv_head']} "
        f"hd={text['head_dim']} ff={text['n_ff']} vocab={text['n_vocab']} "
        f"cross={list(text['cross_layers'])}")
    log(f"vision: {vis['n_layer']}L d={vis['n_embd']} h={vis['n_head']} ff={vis['n_ff']} "
        f"patch={vis['patch']} merge={vis['merge']} deepstack={list(vis['deepstack'])}")

    reader = CheckpointReader(src)
    weight_map = reader.weight_map
    nf4_modules = {n[:-len(".weight.absmax")] for n in weight_map
                   if n.endswith(".weight.absmax")}
    aux = {f"{m}.{s}" for m in nf4_modules for s in NF4_AUX}
    real = set(weight_map) - aux
    log(f"{len(weight_map)} tensors in the checkpoint: {len(nf4_modules)} NF4 linears, "
        f"{len(real)} real weights")

    merger_shapes = {
        "mm.fc1.weight": reader.shape_of(f"{VT}.merger.linear_fc1.weight"),
        "mm.fc1.bias": reader.shape_of(f"{VT}.merger.linear_fc1.bias"),
        "mm.fc2.weight": reader.shape_of(f"{VT}.merger.linear_fc2.weight"),
        "mm.fc2.bias": reader.shape_of(f"{VT}.merger.linear_fc2.bias"),
    }
    plan = build_plan(text, vis, nf4_modules, args.f16_all, args, merger_shapes)

    planned_hf = {t.hf_name for t in plan}
    unused, ghosts = real - planned_hf, planned_hf - real
    if unused:
        fail(f"checkpoint tensors not covered by the mapping: {sorted(unused)[:8]}")
    if ghosts:
        fail(f"mapping wants tensors the checkpoint lacks: {sorted(ghosts)[:8]}")
    plan.sort(key=lambda t: (weight_map[t.hf_name],))      # one shard mmap at a time
    log(f"{len(plan)} output tensors")

    gw = gguf.GGUFWriter(out, ARCH)
    write_metadata(gw, text, vis, args.nf4_out)
    for t in plan:
        if t.qtype is not None:
            gw.add_tensor_info(t.name, t.bytes_shape, np.dtype(np.uint8), t.nbytes,
                               raw_dtype=t.qtype)
        else:
            gw.add_tensor_info(t.name, t.shape,
                               np.dtype(np.float16 if t.kind == "f16" else np.float32),
                               t.nbytes)
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_ti_data_to_file()

    stats: dict[str, int] = {}
    vis_head = (vis["n_head"], vis["n_embd"] // vis["n_head"])
    for i, t in enumerate(plan):
        gw.write_tensor_data(produce(reader, t, vis_head))
        stats[t.kind] = stats.get(t.kind, 0) + 1
        if not args.quiet:
            progress(i, len(plan), "tensors", every=64)
    reader.close()
    gw.close()

    log(f"done: {out} ({os.path.getsize(out) / 1e9:.2f} GB, "
        + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())) + ")")
    verify(out, plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
