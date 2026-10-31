#!/usr/bin/env python3
"""DiffusionGemma 26B-A4B (google/diffusiongemma-26B-A4B-it) -> GGUF.

The checkpoint is an encoder/decoder text-diffusion model (not autoregressive):

  * ONE shared 30-layer sparse-MoE text trunk (hidden 2816, 16 heads,
    sliding layers head_dim 256 / 8 KV heads / theta 1e4 / window 1024,
    full-attn layers head_dim 512 / 2 KV heads / theta 1e6 proportional
    rope 0.25, 128 experts top-8 + 1 dense MLP per layer, GeGLU(tanh),
    Gemma sandwich post-norms, final logit softcap 30, tied embeddings).
  * The encoder runs the trunk causally to fill a KV cache; the decoder runs
    the SAME weights bidirectionally over a 256-token "canvas" that attends to
    the cached prefix and to itself. All non-shared weights in the checkpoint
    are exactly the 30 per-layer `layer_scalar` buffers of each side
    (`model.encoder.language_model.layers.i.layer_scalar` vs
    `model.decoder.layers.i.layer_scalar`), so this converter writes the trunk
    ONCE plus both scalar vectors.
  * A Gemma4 vision tower (27 layers, hidden 1152, patch 16, pooling 3x3,
    learned 2-D position table, axial 2-D rope) feeds 280 soft tokens per
    image through `embed_vision.embedding_projection`.

GGUF layout written here (all names are what src/models/diffusiongemma reads):

  t.embed.weight                      [D, vocab]        (tied lm head)
  t.norm.weight                       decoder final RMSNorm
  t.sc.{pre_norm,post_norm,gate,up,down}.weight          self-conditioning MLP
  t.blk.{i}.attn_norm / attn_post_norm / ffn_norm / ffn_post_norm
  t.blk.{i}.ffn_norm2 / ffn_post_norm_1 / ffn_post_norm_2   (MoE branch norms)
  t.blk.{i}.wq / wk / wv(optional) / wo / q_norm / k_norm
  t.blk.{i}.w_gate / w_up / w_down                       dense MLP (parallel FFN)
  t.blk.{i}.router_proj.weight  [D, E]                   (no bias)
  t.blk.{i}.router_scale.weight [D]                      router RMS-norm scale
  t.blk.{i}.router_per_expert_scale.weight [E]
  t.blk.{i}.gate_exps.weight [D, Ie, E]  up_exps.weight [D, Ie, E]
  t.blk.{i}.down_exps.weight [Ie, D, E]
  t.blk.{i}.out_scale                 decoder layer_scalar [1]
  t.enc.{i}.out_scale                 encoder layer_scalar [1]
  v.patch_dense.weight   [768, 1152]                     (HF [1152,768], no bias)
  v.pos_table.weight     [1152, 10240, 2]                (x table = [...,0])
  v.std_bias.weight / v.std_scale.weight [1152]
  v.blk.{i}.{attn_norm,attn_post_norm,wq,wk,wv,wo,q_norm,k_norm,
            ffn_norm,ffn_post_norm,w_gate,w_up,w_down}.weight
  v.mm_proj.weight       [1152, 2816]                    (embed_vision projection)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lib
from _lib import add_f16, add_quantized, fail, log, quantize
import gguf
import hf_source

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
ENV_VAR = "DIFFUSIONGEMMA_SRC"
PATTERNS = ("model.safetensors", "model.safetensors.index.json")

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q6_K": gguf.GGMLQuantizationType.Q6_K, "q4_0": gguf.GGMLQuantizationType.Q4_0}


def read_config(src_dir):
    raw = json.load(open(os.path.join(src_dir, "config.json"), encoding="utf-8"))
    tc = raw.get("text_config") or raw
    vc = raw.get("vision_config") or {}
    rope = tc.get("rope_parameters") or {}
    rope_full = rope.get("full_attention") or {}
    rope_swa = rope.get("sliding_attention") or {}
    layer_types = list(tc["layer_types"])
    cfg = dict(
        n_layer=int(tc["num_hidden_layers"]),
        n_embd=int(tc["hidden_size"]),
        n_ff=int(tc["intermediate_size"]),                 # dense MLP width
        n_ff_exp=int(tc["moe_intermediate_size"]),         # per-expert width
        n_expert=int(tc["num_experts"]),
        n_expert_used=int(tc["top_k_experts"]),
        n_head=int(tc["num_attention_heads"]),
        n_kv_head=int(tc["num_key_value_heads"]),
        n_kv_head_full=int(tc.get("num_global_key_value_heads", 1)),
        head_dim_swa=int(tc["head_dim"]),
        head_dim_full=int(tc.get("global_head_dim", tc["head_dim"])),
        eps=float(tc["rms_norm_eps"]),
        window=int(tc["sliding_window"]),
        layer_types=layer_types,
        rope_theta_full=float(rope_full.get("rope_theta", 1e6)),
        rope_theta_swa=float(rope_swa.get("rope_theta", 1e4)),
        partial_rotary_full=float(rope_full.get("partial_rotary_factor", 0.25)),
        softcap=float(tc.get("final_logit_softcapping", 30.0) or 30.0),
        max_pos=int(tc.get("max_position_embeddings", 262144)),
        vocab=int(tc["vocab_size"]),
        canvas=int(raw.get("canvas_length", 256)),
        id_bos=int(tc.get("bos_token_id", 2)),
        id_eos=int(tc.get("eos_token_id", 1)),
        id_boi=int(raw.get("boi_token_id", 255999)),
        id_eoi=int(raw.get("eoi_token_id", 258882)),
        id_image=int(raw.get("image_token_id", 258880)),
        # vision tower
        v_layer=int(vc.get("num_hidden_layers", 27)),
        v_embd=int(vc.get("hidden_size", 1152)),
        v_ff=int(vc.get("intermediate_size", 4304)),
        v_head=int(vc.get("num_attention_heads", 16)),
        v_kv=int(vc.get("num_key_value_heads", vc.get("num_attention_heads", 16))),
        v_hd=int(vc.get("head_dim", vc.get("global_head_dim", 72))),
        v_patch=int(vc.get("patch_size", 16)),
        v_pool=int(vc.get("pooling_kernel_size", 3)),
        v_pos=int(vc.get("position_embedding_size", 10240)),
        v_theta=float((vc.get("rope_parameters") or {}).get("rope_theta", 100.0)),
        v_eps=float(vc.get("rms_norm_eps", 1e-6)),
        v_soft=int(raw.get("vision_soft_tokens_per_image", 280)),
        v_global_head_dim=int(vc.get("global_head_dim", vc.get("head_dim", 72))),
    )
    gen = os.path.join(src_dir, "generation_config.json")
    eos = tc.get("eos_token_id")
    if os.path.exists(gen):
        g = json.load(open(gen, encoding="utf-8"))
        cfg["diff_steps"] = int(g.get("max_denoising_steps", 48))
        cfg["t_max"] = float(g.get("t_max", 0.8))
        cfg["t_min"] = float(g.get("t_min", 0.4))
        cfg["conf_thr"] = float(g.get("confidence_threshold", 0.005))
        cfg["stab_thr"] = int(g.get("stability_threshold", 1))
        sc = g.get("sampler_config") or {}
        cfg["entropy_bound"] = float(sc.get("entropy_bound", 0.1))
        geos = g.get("eos_token_id")
        if geos is not None:
            eos = geos[0] if isinstance(geos, list) else geos
        cfg["id_eot"] = int(g.get("eos_token_id")[1]) if isinstance(g.get("eos_token_id"), list) and len(g.get("eos_token_id")) > 1 else 106
    cfg.setdefault("diff_steps", 48)
    cfg.setdefault("t_max", 0.8)
    cfg.setdefault("t_min", 0.4)
    cfg.setdefault("conf_thr", 0.005)
    cfg.setdefault("stab_thr", 1)
    cfg.setdefault("entropy_bound", 0.1)
    cfg.setdefault("id_eot", 106)
    cfg["id_eos"] = int(eos[0]) if isinstance(eos, list) else int(eos)
    return cfg


def write_metadata(gw, cfg, outtype):
    A = "diffusiongemma"
    gw.add_name("diffusiongemma-26b-a4b-it")
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q6_K": gguf.LlamaFileType.MOSTLY_Q6_K,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32(f"{A}.block_count", cfg["n_layer"])
    gw.add_uint32(f"{A}.embedding_length", cfg["n_embd"])
    gw.add_uint32(f"{A}.feed_forward_length", cfg["n_ff"])
    gw.add_uint32(f"{A}.expert_feed_forward_length", cfg["n_ff_exp"])
    gw.add_uint32(f"{A}.expert_count", cfg["n_expert"])
    gw.add_uint32(f"{A}.expert_used_count", cfg["n_expert_used"])
    gw.add_uint32(f"{A}.attention.head_count", cfg["n_head"])
    gw.add_uint32(f"{A}.attention.head_count_kv", cfg["n_kv_head"])
    gw.add_uint32(f"{A}.attention.head_count_kv_full", cfg["n_kv_head_full"])
    gw.add_uint32(f"{A}.attention.key_length", cfg["head_dim_full"])
    gw.add_uint32(f"{A}.attention.key_length_swa", cfg["head_dim_swa"])
    gw.add_uint32(f"{A}.attention.sliding_window", cfg["window"])
    gw.add_float32(f"{A}.attention.layer_norm_rms_epsilon", cfg["eps"])
    gw.add_float32(f"{A}.attention.final_logit_softcapping", cfg["softcap"])
    gw.add_float32(f"{A}.rope.freq_base", cfg["rope_theta_full"])
    gw.add_float32(f"{A}.rope.freq_base_swa", cfg["rope_theta_swa"])
    gw.add_uint32(f"{A}.rope.dimension_count", cfg["head_dim_full"])
    gw.add_uint32(f"{A}.rope.dimension_count_swa", cfg["head_dim_swa"])
    gw.add_float32(f"{A}.rope.partial_rotary_factor_full", cfg["partial_rotary_full"])
    gw.add_uint32(f"{A}.rope.context_length", cfg["max_pos"])
    gw.add_uint32(f"{A}.vocab_size", cfg["vocab"])
    gw.add_array(f"{A}.attention.sliding_window_pattern",
                 [1 if t == "sliding_attention" else 0 for t in cfg["layer_types"]])
    gw.add_uint32(f"{A}.canvas_length", cfg["canvas"])
    # diffusion sampler defaults (generation_config.json)
    gw.add_uint32(f"{A}.diffusion.max_denoising_steps", cfg["diff_steps"])
    gw.add_float32(f"{A}.diffusion.t_max", cfg["t_max"])
    gw.add_float32(f"{A}.diffusion.t_min", cfg["t_min"])
    gw.add_float32(f"{A}.diffusion.entropy_bound", cfg["entropy_bound"])
    gw.add_float32(f"{A}.diffusion.confidence_threshold", cfg["conf_thr"])
    gw.add_uint32(f"{A}.diffusion.stability_threshold", cfg["stab_thr"])
    gw.add_uint32(f"{A}.bos_token_id", cfg["id_bos"])
    gw.add_uint32(f"{A}.eos_token_id", cfg["id_eos"])
    gw.add_uint32(f"{A}.end_of_turn_token_id", cfg["id_eot"])
    gw.add_uint32(f"{A}.boi_token_id", cfg["id_boi"])
    gw.add_uint32(f"{A}.eoi_token_id", cfg["id_eoi"])
    gw.add_uint32(f"{A}.image_token_id", cfg["id_image"])
    # vision tower
    gw.add_uint32(f"{A}.vision.block_count", cfg["v_layer"])
    gw.add_uint32(f"{A}.vision.embedding_length", cfg["v_embd"])
    gw.add_uint32(f"{A}.vision.feed_forward_length", cfg["v_ff"])
    gw.add_uint32(f"{A}.vision.attention.head_count", cfg["v_head"])
    gw.add_uint32(f"{A}.vision.attention.head_count_kv", cfg["v_kv"])
    gw.add_uint32(f"{A}.vision.attention.key_length", cfg["v_hd"])
    gw.add_float32(f"{A}.vision.rope.freq_base", cfg["v_theta"])
    gw.add_float32(f"{A}.vision.attention.layer_norm_rms_epsilon", cfg["v_eps"])
    gw.add_uint32(f"{A}.vision.patch_size", cfg["v_patch"])
    gw.add_uint32(f"{A}.vision.pooling_kernel_size", cfg["v_pool"])
    gw.add_uint32(f"{A}.vision.position_embedding_size", cfg["v_pos"])
    gw.add_uint32(f"{A}.vision.max_soft_tokens", cfg["v_soft"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the dense 2-D weights (default f16)")
    ap.add_argument("-o", "--out", default=None,
                    help="output file (default diffusiongemma-26b-a4b-it-<outtype>.gguf)")
    ap.add_argument("-q", "--quiet", action="store_true")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"diffusiongemma-26b-a4b-it-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    cfg = read_config(src)
    log(f"text: {cfg['n_layer']}L d={cfg['n_embd']} dense_ff={cfg['n_ff']} "
        f"moe={cfg['n_expert']}x{cfg['n_ff_exp']} top{cfg['n_expert_used']} "
        f"heads={cfg['n_head']} kv={cfg['n_kv_head']}(swa)/{cfg['n_kv_head_full']}(full) "
        f"hd={cfg['head_dim_swa']}/{cfg['head_dim_full']} window={cfg['window']} "
        f"canvas={cfg['canvas']} steps={cfg['diff_steps']}")
    log(f"vision: {cfg['v_layer']}L d={cfg['v_embd']} ff={cfg['v_ff']} "
        f"heads={cfg['v_head']} hd={cfg['v_hd']} patch={cfg['v_patch']} "
        f"pool={cfg['v_pool']} pos={cfg['v_pos']} soft={cfg['v_soft']}")

    weights = _lib.TensorSource(src)
    names = set(weights.names())
    DEC = "model.decoder."
    ENC = "model.encoder."
    VIS = "model.encoder.vision_tower."
    log(f"{len(weights)} tensors")

    qtype = OUTTYPES[args.outtype]

    gw = gguf.GGUFWriter(out, "diffusiongemma")
    write_metadata(gw, cfg, args.outtype)

    def need(name):
        if name not in names:
            fail(f"missing tensor {name} (fix the HF NAME MAP in this script)")
        return weights.get(name)

    def lin(gname, hf_name, optional=False, quant=True):
        if hf_name not in names:
            if optional:
                return None
            fail(f"missing tensor {hf_name} (fix the HF NAME MAP in this script)")
        arr = np.ascontiguousarray(weights.get(hf_name))
        if quant and qtype is not None and arr.shape[-1] % 256 == 0:
            add_quantized(gw, gname, quantize(arr, qtype), qtype)
        else:
            add_f16(gw, gname, arr)
        return gname

    def f32(gname, hf_name, optional=False):
        if hf_name not in names:
            if optional:
                return None
        gw.add_tensor(gname, np.ascontiguousarray(weights.get(hf_name), dtype=np.float32))
        return gname

    def f16(gname, hf_name, optional=False):
        if hf_name not in names:
            if optional:
                return None
            fail(f"missing tensor {hf_name}")
        add_f16(gw, gname, np.ascontiguousarray(weights.get(hf_name)))
        return gname

    def quant_type_for(n0):
        """qtype usable for a table whose ggml ne0 is `n0`, or None for F16.

        ggml quantizes along ne0, so ne0 must be a multiple of the block size:
        2816 is fine for Q6_K (block 256), but 704 (= moe_intermediate_size)
        only lines up for the 32-element blocks (Q8_0/Q4_0).
        """
        if qtype is None or n0 % 32 != 0:
            return None
        block, _ = gguf.GGML_QUANT_SIZES[qtype]
        if n0 % block == 0:
            return qtype
        if n0 % 32 == 0:
            return gguf.GGMLQuantizationType.Q8_0
        return None

    def add_table(gname, arr):
        """Write a stacked expert table.

        `arr` is the HF tensor as-is; its LAST axis is the ggml ne0 (that is
        where the quantization blocks run), and the writer flips the numpy
        shape into GGUF dims, so [E, Ie, D] becomes ne [D, Ie, E] with no
        transpose anywhere.
        """
        n0 = int(arr.shape[-1])
        qt = quant_type_for(n0)
        if qt is None:
            add_f16(gw, gname, arr)
            return
        blocks = quantize(arr.reshape(-1, n0), qt)          # [E*Ie, bytes]
        raw = blocks.reshape(*arr.shape[:-1], blocks.shape[-1])
        gw.add_tensor(gname, raw, raw_shape=raw.shape, raw_dtype=qt)

    # ---------------------------------------------------------------- text
    # the tied lm_head lives here: keeping it F16 costs ~0.75 GB and removes the
    # quantization noise from every logit
    f16("t.embed.weight", f"{DEC}embed_tokens.weight")
    f32("t.norm.weight", f"{DEC}norm.weight")
    f32("t.sc.pre_norm.weight", f"{DEC}self_conditioning.pre_norm.weight")
    # NOTE: self_conditioning.post_norm is a with_scale=False RMSNorm - it ships
    # no weight in the checkpoint, the engine applies a bare rms_norm there.
    lin("t.sc.gate.weight", f"{DEC}self_conditioning.gate_proj.weight")
    lin("t.sc.up.weight", f"{DEC}self_conditioning.up_proj.weight")
    lin("t.sc.down.weight", f"{DEC}self_conditioning.down_proj.weight")

    for i in range(cfg["n_layer"]):
        b = f"{DEC}layers.{i}."
        g = f"t.blk.{i}."
        # RMSNorm weights stay F32: quantizing them (the 2816-wide ones hit the
        # %256 quantization branch) wrecks the per-channel scale and the whole
        # trunk blows up - the logits come out nearly flat.
        f32(g + "attn_norm.weight", b + "input_layernorm.weight")
        f32(g + "attn_post_norm.weight", b + "post_attention_layernorm.weight")
        lin(g + "wq.weight", b + "self_attn.q_proj.weight")
        lin(g + "wk.weight", b + "self_attn.k_proj.weight")
        # attention_k_eq_v: full-attention layers ship no v_proj (V = K)
        lin(g + "wv.weight", b + "self_attn.v_proj.weight", optional=True)
        lin(g + "wo.weight", b + "self_attn.o_proj.weight")
        f32(g + "q_norm.weight", b + "self_attn.q_norm.weight")
        f32(g + "k_norm.weight", b + "self_attn.k_norm.weight")
        f32(g + "ffn_norm.weight", b + "pre_feedforward_layernorm.weight")
        f32(g + "ffn_post_norm.weight", b + "post_feedforward_layernorm.weight")
        f32(g + "ffn_norm2.weight", b + "pre_feedforward_layernorm_2.weight")
        f32(g + "ffn_post_norm_1.weight", b + "post_feedforward_layernorm_1.weight")
        f32(g + "ffn_post_norm_2.weight", b + "post_feedforward_layernorm_2.weight")
        lin(g + "w_gate.weight", b + "mlp.gate_proj.weight")
        lin(g + "w_up.weight", b + "mlp.up_proj.weight")
        lin(g + "w_down.weight", b + "mlp.down_proj.weight")
        # router: rms_norm(no scale) * scale / sqrt(D) -> proj -> softmax -> top-k
        lin(g + "router_proj.weight", b + "router.proj.weight")
        f32(g + "router_scale.weight", b + "router.scale")
        f32(g + "router_per_expert_scale.weight", b + "router.per_expert_scale")
        # expert tables: HF gate_up_proj [E, 2*Ie, D] (gate half first, then up)
        # and down_proj [E, D, Ie]. Their LAST axis is already the ggml ne0, so
        # no transpose: gate/up -> ne [D, Ie, E], down -> ne [Ie, D, E].
        gu = need(b + "experts.gate_up_proj")            # [E, 2Ie, D]
        n_e, two_i, n_d = gu.shape
        if two_i != 2 * cfg["n_ff_exp"] or n_d != cfg["n_embd"]:
            fail(f"unexpected gate_up_proj shape {gu.shape} at layer {i}")
        gate = np.ascontiguousarray(gu[:, :cfg["n_ff_exp"], :])     # [E, Ie, D]
        up = np.ascontiguousarray(gu[:, cfg["n_ff_exp"]:, :])       # [E, Ie, D]
        down = np.ascontiguousarray(need(b + "experts.down_proj"))  # [E, D, Ie]
        add_table(g + "gate_exps.weight", gate)
        add_table(g + "up_exps.weight", up)
        add_table(g + "down_exps.weight", down)
        f32(g + "out_scale", b + "layer_scalar")          # decoder layer scalar
        f32(f"t.enc.{i}.out_scale", f"{ENC}language_model.layers.{i}.layer_scalar")
        if not args.quiet:
            _lib.progress(i + 1, cfg["n_layer"], "text layers")

    # -------------------------------------------------------------- vision
    # The tower stays F16 on purpose: its inputs are 2000+ patch columns at once,
    # and this ggml fork's CPU mul_mat mis-indexes quantized weights when the
    # batch (b->ne1) is that large - verified by feeding 2394 identical columns
    # and getting 2394 different outputs. Text activations are short, so Q8_0 is
    # fine there.
    f16("v.patch_dense.weight", f"{VIS}patch_embedder.input_proj.weight")
    # learned 2-D position table [2, pos, D]: written as-is, so the last axis D
    # becomes ggml ne0 and the dims read ne = [D, pos, 2] (x table at ne2 == 0)
    pt = need(f"{VIS}patch_embedder.position_embedding_table")
    gw.add_tensor("v.pos_table.weight", np.ascontiguousarray(pt, dtype=np.float32))
    f32("v.std_bias.weight", f"{VIS}std_bias")
    f32("v.std_scale.weight", f"{VIS}std_scale")
    for i in range(cfg["v_layer"]):
        b = f"{VIS}encoder.layers.{i}."
        g = f"v.blk.{i}."
        f32(g + "attn_norm.weight", b + "input_layernorm.weight")
        f32(g + "attn_post_norm.weight", b + "post_attention_layernorm.weight")
        f16(g + "wq.weight", b + "self_attn.q_proj.linear.weight")
        f16(g + "wk.weight", b + "self_attn.k_proj.linear.weight")
        f16(g + "wv.weight", b + "self_attn.v_proj.linear.weight")
        f16(g + "wo.weight", b + "self_attn.o_proj.linear.weight")
        f32(g + "q_norm.weight", b + "self_attn.q_norm.weight")
        f32(g + "k_norm.weight", b + "self_attn.k_norm.weight")
        f32(g + "ffn_norm.weight", b + "pre_feedforward_layernorm.weight")
        f32(g + "ffn_post_norm.weight", b + "post_feedforward_layernorm.weight")
        f16(g + "w_gate.weight", b + "mlp.gate_proj.linear.weight")
        f16(g + "w_up.weight", b + "mlp.up_proj.linear.weight")
        f16(g + "w_down.weight", b + "mlp.down_proj.linear.weight")
        if not args.quiet:
            _lib.progress(i + 1, cfg["v_layer"], "vision layers")
    # embed_vision: RMSNorm(no scale) -> projection
    f16("v.mm_proj.weight", f"{ENC}embed_vision.embedding_projection.weight")

    log("writing GGUF ...")
    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    # ------------------------------------------------------------ self check
    import struct
    size = os.path.getsize(out)
    log(f"wrote {out} ({size / 1e9:.2f} GB)")

    def peek(f, want):
        f.seek(0)
        magic = f.read(4)
        if magic != b"GGUF":
            fail("not a GGUF file")
        ver = struct.unpack("<I", f.read(4))[0]
        n_t = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        log(f"GGUF v{ver}: {n_t} tensors, {n_kv} metadata keys")
        if n_t < want:
            fail(f"too few tensors ({n_t} < {want})")

    # 12 = embed + final norm + 5 self-conditioning + 5 vision globals; the
    # per-layer floor counts 23 (attention_k_eq_v drops v_proj on full layers)
    need_tensors = 12 + cfg["n_layer"] * 23 + cfg["v_layer"] * 13
    with open(out, "rb") as f:
        peek(f, need_tensors)


if __name__ == "__main__":
    main()
