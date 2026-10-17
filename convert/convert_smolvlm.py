"""Convert a SmolVLM (Idefics3) checkpoint to GGUF.

Usage::

  python convert/convert_smolvlm.py [--src DIR] [-o OUT.gguf] [--outtype f16|q8_0|q4_0]

Checkpoint discovery follows the other converters (``--src``, then ``$SMOLVLM_SRC``,
the HF cache, then ``models/smolvlm-256m``); fetch one with::

  hf download HuggingFaceTB/SmolVLM-256M-Instruct --local-dir models/smolvlm-256m

Layout (``general.architecture = smolvlm``):

  t.embed.weight            [vocab, 576]     untied, lm_head is a separate tensor
  t.blk.{i}.attn_norm.weight                 RMSNorm (eps 1e-5), no biases anywhere
  t.blk.{i}.wq/wk/wv/wo.weight               GQA 9:3, head_dim 64, rope theta 1e5
  t.blk.{i}.ffn_norm.weight  + w_gate/w_up/w_down
  t.norm.weight / t.lm_head.weight

  v.patch_embd.weight       [768, 3*16*16]   Conv2d(3,768,16,16) folded to one matmul
  v.patch_embd.bias
  v.pos_embd.weight         [768, 1024]      learned absolute positions (32x32, no cls)
  v.blk.{i}.ln1/ln2.weight+bias              LayerNorm **with** bias (SigLIP)
  v.blk.{i}.wq/bq wk/bk wv/bv wo/bo          12 heads, head_dim 64, no rope
  v.blk.{i}.fc1.w/b fc2.w/b                  plain MLP + gelu (tanh form)
  v.post_ln.weight+bias
  v.mm_proj.weight          [12288, 576]     pixel-shuffle x4 then this linear (no bias)

Idefics3 preprocessing (mirrors the official processor):

  * resize so the longest edge is ``smolvlm.image_longest_edge`` (2048), then resize
    again so both sides are multiples of the 512 tile (this second resize does not
    preserve the aspect ratio, exactly like the HF processor);
  * the refined image is cut into 512x512 tiles (row-major) and each tile becomes
    ``image_seq_len`` = (512/16)^2 / 4^2 = 64 tokens;
  * one extra **global** 512x512 view of the whole refined image is appended last;
  * the prompt expands ``<image>`` into
    ``<fake><row_R_col_C><image>*64`` per tile and ``<fake><global-img><image>*64<fake>``
    for the global view.
"""
import argparse
import json
import os

import numpy as np
import gguf

import _lib
from _lib import log, fail, progress
import hf_source

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
ENV_VAR = "SMOLVLM_SRC"
PATTERNS = ("model.safetensors", "model.safetensors.index.json")

CFG = dict(n_layer=30, n_embd=576, n_head=9, n_kv_head=3, head_dim=64, n_ff=1536,
           vocab=49280, eps=1e-5, rope_theta=1e5,
           v_layer=12, v_embd=768, v_head=12, v_ff=3072, v_patch=16, v_image=512,
           v_eps=1e-6, v_merge=4, image_seq_len=64, tile=512, longest_edge=2048,
           resample=1,
           id_im_start=1, id_tile=49152, id_row_col=49153, id_fake=49189,
           id_image=49190, id_eou=49279)

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q4_0": gguf.GGMLQuantizationType.Q4_0}


def read_config(src_dir):
    cfg = dict(CFG)
    path = os.path.join(src_dir, "config.json")
    if not os.path.isfile(path):
        fail(f"no config.json in {src_dir}")
    raw = json.load(open(path, encoding="utf-8"))
    tc = raw.get("text_config", {}) or {}
    vc = raw.get("vision_config", {}) or {}
    n_head = int(tc.get("num_attention_heads", cfg["n_head"]))
    cfg.update(
        n_layer=int(tc.get("num_hidden_layers", cfg["n_layer"])),
        n_embd=int(tc.get("hidden_size", cfg["n_embd"])),
        n_head=n_head,
        n_kv_head=int(tc.get("num_key_value_heads", cfg["n_kv_head"])),
        head_dim=int(tc.get("head_dim") or cfg["head_dim"]),
        n_ff=int(tc.get("intermediate_size", cfg["n_ff"])),
        vocab=int(tc.get("vocab_size", cfg["vocab"])),
        eps=float(tc.get("rms_norm_eps", cfg["eps"])),
        rope_theta=float(tc.get("rope_theta", cfg["rope_theta"])),
        v_layer=int(vc.get("num_hidden_layers", cfg["v_layer"])),
        v_embd=int(vc.get("hidden_size", cfg["v_embd"])),
        v_head=int(vc.get("num_attention_heads", cfg["v_head"])),
        v_ff=int(vc.get("intermediate_size", cfg["v_ff"])),
        v_patch=int(vc.get("patch_size", cfg["v_patch"])),
        v_image=int(vc.get("image_size", cfg["v_image"])),
        v_eps=float(vc.get("layer_norm_eps", cfg["v_eps"])),
    )
    cfg["tie_word_embeddings"] = bool(tc.get("tie_word_embeddings", False))
    pp = os.path.join(src_dir, "preprocessor_config.json")
    if os.path.isfile(pp):
        p = json.load(open(pp, encoding="utf-8"))
        msis = p.get("max_image_size") or {}
        size = p.get("size") or {}
        if msis.get("longest_edge"):
            cfg["tile"] = int(msis["longest_edge"])
        if size.get("longest_edge"):
            cfg["longest_edge"] = int(size["longest_edge"])
        if p.get("resample") is not None:
            cfg["resample"] = int(p["resample"])       # PILImageResampling: 1 = LANCZOS
        cfg["v_mean"] = [float(x) for x in (p.get("image_mean") or [0.5, 0.5, 0.5])]
        cfg["v_std"] = [float(x) for x in (p.get("image_std") or [0.5, 0.5, 0.5])]
    else:
        cfg.setdefault("v_mean", [0.5, 0.5, 0.5])
        cfg.setdefault("v_std", [0.5, 0.5, 0.5])
    # tiles are the vision-tower input size
    cfg["tile"] = cfg["v_image"]
    cfg["image_seq_len"] = (cfg["v_image"] // cfg["v_patch"]) ** 2 // (cfg["v_merge"] ** 2)
    return cfg


def write_metadata(gw, cfg, outtype):
    gw.add_name("smolvlm")
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32("smolvlm.block_count", cfg["n_layer"])
    gw.add_uint32("smolvlm.embedding_length", cfg["n_embd"])
    gw.add_uint32("smolvlm.feed_forward_length", cfg["n_ff"])
    gw.add_uint32("smolvlm.attention.head_count", cfg["n_head"])
    gw.add_uint32("smolvlm.attention.head_count_kv", cfg["n_kv_head"])
    gw.add_uint32("smolvlm.attention.key_length", cfg["head_dim"])
    gw.add_float32("smolvlm.attention.layer_norm_rms_epsilon", float(cfg["eps"]))
    gw.add_float32("smolvlm.rope.freq_base", float(cfg["rope_theta"]))
    gw.add_uint32("smolvlm.vocab_size", cfg["vocab"])
    gw.add_bool("smolvlm.tie_word_embeddings", cfg["tie_word_embeddings"])

    gw.add_uint32("smolvlm.vision.block_count", cfg["v_layer"])
    gw.add_uint32("smolvlm.vision.embedding_length", cfg["v_embd"])
    gw.add_uint32("smolvlm.vision.feed_forward_length", cfg["v_ff"])
    gw.add_uint32("smolvlm.vision.attention.head_count", cfg["v_head"])
    gw.add_uint32("smolvlm.vision.patch_size", cfg["v_patch"])
    gw.add_uint32("smolvlm.vision.image_size", cfg["v_image"])
    gw.add_uint32("smolvlm.vision.tile_size", cfg["tile"])
    gw.add_uint32("smolvlm.vision.longest_edge", cfg["longest_edge"])
    gw.add_uint32("smolvlm.vision.resize_resample", cfg["resample"])
    gw.add_uint32("smolvlm.vision.merge_size", cfg["v_merge"])
    gw.add_uint32("smolvlm.vision.image_seq_len", cfg["image_seq_len"])
    gw.add_float32("smolvlm.vision.layer_norm_epsilon", float(cfg["v_eps"]))
    gw.add_array("smolvlm.vision.image_mean", [float(x) for x in cfg["v_mean"]])
    gw.add_array("smolvlm.vision.image_std", [float(x) for x in cfg["v_std"]])

    for key in ("im_start", "tile", "row_col", "fake", "image", "eou"):
        gw.add_uint32(f"smolvlm.{key}_token_id", int(cfg[f"id_{key}"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the 2D weights (default f16)")
    ap.add_argument("-o", "--out", default=None, help="output file (default smolvlm-<outtype>.gguf)")
    ap.add_argument("-q", "--quiet", action="store_true")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"smolvlm-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    cfg = read_config(src)
    qtype = OUTTYPES[args.outtype]
    log(f"text: {cfg['n_layer']}L d={cfg['n_embd']} nh={cfg['n_head']} nkv={cfg['n_kv_head']} "
        f"hd={cfg['head_dim']} ff={cfg['n_ff']} vocab={cfg['vocab']} eps={cfg['eps']:g} "
        f"theta={cfg['rope_theta']:g}")
    log(f"vision: {cfg['v_layer']}L d={cfg['v_embd']} heads={cfg['v_head']} ff={cfg['v_ff']} "
        f"patch={cfg['v_patch']} tile={cfg['tile']} longest_edge={cfg['longest_edge']} "
        f"merge={cfg['v_merge']} -> image_seq_len={cfg['image_seq_len']} "
        f"resample={cfg['resample']}")

    weights = _lib.TensorSource(src)
    log(f"{len(weights)} tensors in the checkpoint")

    gw = gguf.GGUFWriter(out, "smolvlm")
    write_metadata(gw, cfg, args.outtype)

    def linear(name, hf_name):
        arr = weights.get(hf_name)
        if qtype is not None and arr.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, name, _lib.quantize(arr, qtype), qtype)
        else:
            _lib.add_f16(gw, name, arr)

    def f32(name, hf_name):
        gw.add_tensor(name, np.ascontiguousarray(weights.get(hf_name), dtype=np.float32))

    # --------------------------------------------------------------- vision
    conv = weights.get("model.vision_model.embeddings.patch_embedding.weight")
    dim, cin, kh, kw = conv.shape
    if (cin, kh, kw) != (3, cfg["v_patch"], cfg["v_patch"]):
        fail(f"unexpected patch_embedding shape {conv.shape}")
    w2d = conv.reshape(dim, cin * kh * kw)
    if qtype is not None and w2d.shape[-1] % 256 == 0:
        _lib.add_quantized(gw, "v.patch_embd.weight", _lib.quantize(w2d, qtype), qtype)
    else:
        _lib.add_f16(gw, "v.patch_embd.weight", w2d)
    f32("v.patch_embd.bias", "model.vision_model.embeddings.patch_embedding.bias")
    f32("v.pos_embd.weight", "model.vision_model.embeddings.position_embedding.weight")

    for i in range(cfg["v_layer"]):
        h, b = f"model.vision_model.encoder.layers.{i}", f"v.blk.{i}"
        f32(f"{b}.ln1.weight", f"{h}.layer_norm1.weight")
        f32(f"{b}.ln1.bias", f"{h}.layer_norm1.bias")
        f32(f"{b}.ln2.weight", f"{h}.layer_norm2.weight")
        f32(f"{b}.ln2.bias", f"{h}.layer_norm2.bias")
        linear(f"{b}.wq.weight", f"{h}.self_attn.q_proj.weight")
        f32(f"{b}.bq", f"{h}.self_attn.q_proj.bias")
        linear(f"{b}.wk.weight", f"{h}.self_attn.k_proj.weight")
        f32(f"{b}.bk", f"{h}.self_attn.k_proj.bias")
        linear(f"{b}.wv.weight", f"{h}.self_attn.v_proj.weight")
        f32(f"{b}.bv", f"{h}.self_attn.v_proj.bias")
        linear(f"{b}.wo.weight", f"{h}.self_attn.out_proj.weight")
        f32(f"{b}.bo", f"{h}.self_attn.out_proj.bias")
        linear(f"{b}.fc1.weight", f"{h}.mlp.fc1.weight")
        f32(f"{b}.fc1.bias", f"{h}.mlp.fc1.bias")
        linear(f"{b}.fc2.weight", f"{h}.mlp.fc2.weight")
        f32(f"{b}.fc2.bias", f"{h}.mlp.fc2.bias")
        progress(i + 1, cfg["v_layer"], "vision blocks")

    f32("v.post_ln.weight", "model.vision_model.post_layernorm.weight")
    f32("v.post_ln.bias", "model.vision_model.post_layernorm.bias")
    # the connector is one matmul after the 4x4 pixel shuffle: [576, 12288]
    linear("v.mm_proj.weight", "model.connector.modality_projection.proj.weight")

    # ----------------------------------------------------------- text trunk
    linear("t.embed.weight", "model.text_model.embed_tokens.weight")
    for i in range(cfg["n_layer"]):
        h, b = f"model.text_model.layers.{i}", f"t.blk.{i}"
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
    f32("t.norm.weight", "model.text_model.norm.weight")
    if not cfg["tie_word_embeddings"]:
        linear("t.lm_head.weight", "lm_head.weight")

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    names = {t.name for t in gguf.GGUFReader(out).tensors}
    required = ["t.embed.weight", "t.norm.weight", "t.lm_head.weight",
                f"t.blk.{cfg['n_layer'] - 1}.w_down.weight",
                "v.patch_embd.weight", "v.pos_embd.weight", "v.mm_proj.weight",
                f"v.blk.{cfg['v_layer'] - 1}.fc2.weight", "v.post_ln.weight"]
    missing = [n for n in required if n not in names]
    if missing:
        fail(f"written GGUF is missing {missing}")
    log(f"verified {len(names)} tensors, {os.path.getsize(out) / 1e6:.0f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
