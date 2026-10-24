"""Convert a DeepSeek Janus-Pro (understanding path) checkpoint to GGUF.

Usage::

  python convert/convert_janus.py [--src DIR] [-o OUT.gguf] [--outtype f16|q8_0|q4_0]

Source: deepseek-ai/Janus-Pro-1B (or 7B) - pytorch_model.bin, bf16.

Layout (``general.architecture = janus``):

  t.embed.weight            [102400, 2048]   untied lm_head, plain llama trunk
  t.blk.{i}.attn_norm.weight                 RMSNorm eps 1e-5, no qk norms, no bias
  t.blk.{i}.wq/wk/wv/wo.weight               16:16 heads, head_dim 128, rope theta 1e4
  t.blk.{i}.ffn_norm.weight  + w_gate/w_up/w_down
  t.norm.weight / t.lm_head.weight

  v.patch_embd.weight       [1024, 3*16*16]  timm SigLIP-large conv folded to matmul
  v.patch_embd.bias
  v.pos_embd.weight         [1024, 576]      absolute positions, no cls token
  v.blk.{i}.ln1/ln2.weight+bias              LayerNorm with bias
  v.blk.{i}.qkv.weight+bias                  timm fused qkv (exact-erf tower GELU)
  v.blk.{i}.wo.weight+bias
  v.blk.{i}.fc1/fc2.weight+bias              hidden 4096
  v.post_ln.weight+bias                      forward_features ends here
  v.mm_proj0.weight+bias    [2048, 1024]     aligner: Linear -> GELU -> Linear
  v.mm_proj2.weight+bias    [2048, 2048]

The generation path (VQ tokenizer / gen_aligner / gen_head) is intentionally
NOT exported: this engine ships the understanding (VQA) side only.

Preprocessing (official VLMImageProcessor): bicubic resize longest edge to
image_size with antialias, expand2square pad with (127,127,127), /255,
(x-0.5)/0.5. One image = 576 "<image_placeholder>" tokens after the question.
"""
import argparse
import json
import os

import numpy as np
import torch
import gguf

import _lib
from _lib import log, fail, progress

MODEL_ID = "deepseek-ai/Janus-Pro-1B"
ENV_VAR = "JANUS_SRC"
PATTERN = "pytorch_model.bin"

CFG = dict(n_layer=24, n_embd=2048, n_head=16, n_kv_head=16, head_dim=128,
           n_ff=5632, vocab=102400, eps=1e-5, rope_theta=1e4,
           v_layer=24, v_embd=1024, v_head=16, v_ff=4096, v_patch=16,
           v_image=384, v_eps=1e-6, image_seq_len=576,
           id_image=100581, id_eos=100001)

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q4_0": gguf.GGMLQuantizationType.Q4_0}


def read_config(src_dir):
    cfg = dict(CFG)
    raw = json.load(open(os.path.join(src_dir, "config.json"), encoding="utf-8"))
    lc = raw.get("language_config", {}) or {}
    ac = (raw.get("aligner_config", {}) or {}).get("params", {}) or {}
    vp = (raw.get("vision_config", {}) or {}).get("params", {}) or {}
    cfg.update(
        n_layer=int(lc.get("num_hidden_layers", cfg["n_layer"])),
        n_embd=int(lc.get("hidden_size", cfg["n_embd"])),
        n_head=int(lc.get("num_attention_heads", cfg["n_head"])),
        n_kv_head=int(lc.get("num_key_value_heads", cfg["n_kv_head"])),
        head_dim=int(lc.get("hidden_size", cfg["n_embd"]) //
                     lc.get("num_attention_heads", cfg["n_head"])),
        n_ff=int(lc.get("intermediate_size", cfg["n_ff"])),
        vocab=int(lc.get("vocab_size", cfg["vocab"])),
        eps=float(lc.get("rms_norm_eps", cfg["eps"])),
        rope_theta=float(lc.get("rope_theta", cfg["rope_theta"])),
        v_image=int(vp.get("image_size", cfg["v_image"])),
        aligner_depth=int(ac.get("depth", 2)),
    )
    # timm siglip_large_patch16_384 tower geometry; select_layer=-1 keeps all
    cfg["v_layer"] = cfg["n_align_layers"] if False else cfg["v_layer"]
    cfg["image_seq_len"] = (cfg["v_image"] // cfg["v_patch"]) ** 2
    cfg["v_ff"] = cfg["v_embd"] * 4
    pp = os.path.join(src_dir, "preprocessor_config.json")
    if os.path.isfile(pp):
        p = json.load(open(pp, encoding="utf-8"))
        cfg["v_mean"] = [float(x) for x in p.get("image_mean", [0.5] * 3)]
        cfg["v_std"] = [float(x) for x in p.get("image_std", [0.5] * 3)]
        cfg["image_seq_len"] = int(p.get("num_image_tokens", cfg["image_seq_len"]))
    else:
        cfg.setdefault("v_mean", [0.5] * 3)
        cfg.setdefault("v_std", [0.5] * 3)
    return cfg


def write_metadata(gw, cfg, outtype):
    gw.add_name("janus")
    gw.add_file_type({
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }[outtype])
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32("janus.block_count", cfg["n_layer"])
    gw.add_uint32("janus.embedding_length", cfg["n_embd"])
    gw.add_uint32("janus.feed_forward_length", cfg["n_ff"])
    gw.add_uint32("janus.attention.head_count", cfg["n_head"])
    gw.add_uint32("janus.attention.head_count_kv", cfg["n_kv_head"])
    gw.add_uint32("janus.attention.key_length", cfg["head_dim"])
    gw.add_float32("janus.attention.layer_norm_rms_epsilon", float(cfg["eps"]))
    gw.add_float32("janus.rope.freq_base", float(cfg["rope_theta"]))
    gw.add_uint32("janus.vocab_size", cfg["vocab"])

    gw.add_uint32("janus.vision.block_count", cfg["v_layer"])
    gw.add_uint32("janus.vision.embedding_length", cfg["v_embd"])
    gw.add_uint32("janus.vision.feed_forward_length", cfg["v_ff"])
    gw.add_uint32("janus.vision.attention.head_count", cfg["v_head"])
    gw.add_uint32("janus.vision.patch_size", cfg["v_patch"])
    gw.add_uint32("janus.vision.image_size", cfg["v_image"])
    gw.add_uint32("janus.vision.image_seq_len", cfg["image_seq_len"])
    gw.add_float32("janus.vision.layer_norm_epsilon", float(cfg["v_eps"]))
    gw.add_array("janus.vision.image_mean", [float(x) for x in cfg["v_mean"]])
    gw.add_array("janus.vision.image_std", [float(x) for x in cfg["v_std"]])

    gw.add_uint32("janus.image_token_id", int(cfg["id_image"]))
    gw.add_uint32("janus.eos_token_id", int(cfg["id_eos"]))


class BinSource:
    """pytorch_model.bin reader returning float32 numpy (bf16 -> f32)."""

    def __init__(self, src_dir):
        path = os.path.join(src_dir, PATTERN)
        if not os.path.isfile(path):
            fail(f"no {PATTERN} in {src_dir}")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        self.sd = {k: v.float().numpy() for k, v in sd.items()}

    def __len__(self):
        return len(self.sd)

    def get(self, name):
        if name not in self.sd:
            fail(f"tensor not in checkpoint: {name}")
        return self.sd[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the 2D weights (default f16)")
    ap.add_argument("-o", "--out", default=None, help="output file (default janus-<outtype>.gguf)")
    ap.add_argument("--src", default=os.environ.get(ENV_VAR) or
                    os.path.join("models", "Janus-Pro-1B"))
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    out = args.out or f"janus-{args.outtype}.gguf"
    log(f"source: {args.src}")
    log(f"output: {out} ({args.outtype})")

    cfg = read_config(args.src)
    qtype = OUTTYPES[args.outtype]
    log(f"text: {cfg['n_layer']}L d={cfg['n_embd']} nh={cfg['n_head']}/{cfg['n_kv_head']} "
        f"hd={cfg['head_dim']} ff={cfg['n_ff']} vocab={cfg['vocab']} theta={cfg['rope_theta']:g}")
    log(f"vision: {cfg['v_layer']}L d={cfg['v_embd']} heads={cfg['v_head']} "
        f"patch={cfg['v_patch']} size={cfg['v_image']} -> {cfg['image_seq_len']} tokens")

    weights = BinSource(args.src)
    log(f"{len(weights)} tensors in the checkpoint")

    gw = gguf.GGUFWriter(out, "janus")
    write_metadata(gw, cfg, args.outtype)

    def linear(name, hf_name):
        arr = weights.get(hf_name)
        if qtype is not None and arr.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, name, _lib.quantize(arr, qtype), qtype)
        else:
            _lib.add_f16(gw, name, arr)

    def f32(name, hf_name):
        gw.add_tensor(name, np.ascontiguousarray(weights.get(hf_name), dtype=np.float32))

    # ------------------------------------------------- vision (timm SigLIP)
    conv = weights.get("vision_model.vision_tower.patch_embed.proj.weight")
    dim, cin, kh, kw = conv.shape
    if (cin, kh, kw) != (3, cfg["v_patch"], cfg["v_patch"]):
        fail(f"unexpected patch conv shape {conv.shape}")
    w2d = conv.reshape(dim, cin * kh * kw)
    if qtype is not None and w2d.shape[-1] % 256 == 0:
        _lib.add_quantized(gw, "v.patch_embd.weight", _lib.quantize(w2d, qtype), qtype)
    else:
        _lib.add_f16(gw, "v.patch_embd.weight", w2d)
    f32("v.patch_embd.bias", "vision_model.vision_tower.patch_embed.proj.bias")
    pos = weights.get("vision_model.vision_tower.pos_embed")     # [1, 576, 1024]
    gw.add_tensor("v.pos_embd.weight", np.ascontiguousarray(pos[0], dtype=np.float32))

    for i in range(cfg["v_layer"]):
        h, b = f"vision_model.vision_tower.blocks.{i}", f"v.blk.{i}"
        f32(f"{b}.ln1.weight", f"{h}.norm1.weight")
        f32(f"{b}.ln1.bias", f"{h}.norm1.bias")
        f32(f"{b}.ln2.weight", f"{h}.norm2.weight")
        f32(f"{b}.ln2.bias", f"{h}.norm2.bias")
        linear(f"{b}.qkv.weight", f"{h}.attn.qkv.weight")
        f32(f"{b}.qkv.bias", f"{h}.attn.qkv.bias")
        linear(f"{b}.wo.weight", f"{h}.attn.proj.weight")
        f32(f"{b}.wo.bias", f"{h}.attn.proj.bias")
        linear(f"{b}.fc1.weight", f"{h}.mlp.fc1.weight")
        f32(f"{b}.fc1.bias", f"{h}.mlp.fc1.bias")
        linear(f"{b}.fc2.weight", f"{h}.mlp.fc2.weight")
        f32(f"{b}.fc2.bias", f"{h}.mlp.fc2.bias")
        progress(i + 1, cfg["v_layer"], "vision blocks")

    f32("v.post_ln.weight", "vision_model.vision_tower.norm.weight")
    f32("v.post_ln.bias", "vision_model.vision_tower.norm.bias")

    # -------------------------------------------------- aligner (mlp_gelu x2)
    linear("v.mm_proj0.weight", "aligner.layers.0.weight")
    f32("v.mm_proj0.bias", "aligner.layers.0.bias")
    linear("v.mm_proj2.weight", "aligner.layers.2.weight")
    f32("v.mm_proj2.bias", "aligner.layers.2.bias")

    # ----------------------------------------------------------- text trunk
    linear("t.embed.weight", "language_model.model.embed_tokens.weight")
    for i in range(cfg["n_layer"]):
        h, b = f"language_model.model.layers.{i}", f"t.blk.{i}"
        f32(f"{b}.attn_norm.weight", f"{h}.input_layernorm.weight")
        f32(f"{b}.ffn_norm.weight", f"{h}.post_attention_layernorm.weight")
        linear(f"{b}.wq.weight", f"{h}.self_attn.q_proj.weight")
        linear(f"{b}.wk.weight", f"{h}.self_attn.k_proj.weight")
        linear(f"{b}.wv.weight", f"{h}.self_attn.v_proj.weight")
        linear(f"{b}.wo.weight", f"{h}.self_attn.o_proj.weight")
        linear(f"{b}.w_gate.weight", f"{h}.mlp.gate_proj.weight")
        linear(f"{b}.w_up.weight", f"{h}.mlp.up_proj.weight")
        linear(f"{b}.w_down.weight", f"{h}.mlp.down_proj.weight")
        progress(i + 1, cfg["n_layer"], "text blocks")
    f32("t.norm.weight", "language_model.model.norm.weight")
    linear("t.lm_head.weight", "language_model.lm_head.weight")

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    names = {t.name for t in gguf.GGUFReader(out).tensors}
    required = ["t.embed.weight", "t.norm.weight", "t.lm_head.weight",
                f"t.blk.{cfg['n_layer'] - 1}.w_down.weight",
                "v.patch_embd.weight", "v.pos_embd.weight",
                "v.mm_proj0.weight", "v.mm_proj2.weight",
                f"v.blk.{cfg['v_layer'] - 1}.fc2.weight", "v.post_ln.weight"]
    missing = [n for n in required if n not in names]
    if missing:
        fail(f"written GGUF is missing {missing}")
    log(f"verified {len(names)} tensors, {os.path.getsize(out) / 1e6:.0f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
