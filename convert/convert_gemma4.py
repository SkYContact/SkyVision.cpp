"""Convert a Gemma 4 12B (Gemma4Unified) checkpoint to GGUF - text trunk.

Usage::

  python convert/convert_gemma4.py [--src DIR] [-o OUT.gguf] [--outtype f16|q8_0|q6_K|q4_0]

Source (google/gemma-4-12B)::

  hf download google/gemma-4-12B --local-dir models/gemma-4-12B

Gemma 4 12B is ENCODER-FREE: no vision/audio towers, just linear projections
fed by the trunk's shared embeddings (verified against
model.safetensors, 677 tensors; every name below is exact).

Layout (``general.architecture = gemma4``)::

  t.embed.weight            [262144, 3840]   tied lm_head, scaled sqrt(3840) in-graph
  t.norm.weight             [3840]           final norm
  t.blk.{i}.attn_norm / attn_post_norm / ffn_norm / ffn_post_norm  (sandwich norms)
  t.blk.{i}.wq/wk/wo                          wk doubles as wv (attention_k_eq_v)
  t.blk.{i}.q_norm/k_norm                     per-head RMS (v weightless)
  t.blk.{i}.rope_freqs.weight [256]           FULL-ATTN layers only: the
                                              "proportional" rope freq factors
                                              [1.0]*64 + [1e30]*192 (llama.cpp
                                              conversion/gemma.py,
                                              Gemma4Model::generate_extra_tensors)
  t.blk.{i}.w_gate/w_up/w_down                gelu(tanh) SwiGLU
  t.blk.{i}.out_scale        [1] optional     layer_scalar
  t.blk.{i}.ple_* / t.ple_*                    only if hidden_size_per_layer_input > 0
                                              (12B ships 0 - absent)

  v.patch_dense.weight      [3840, 6912]      HF layout kept (HWC columns);
  v.patch_dense.bias        [3840]            backend extracts host patches HWC
  v.ln1.weight/.bias        [6912]            LayerNorm eps 1e-5 (pre-dense)
  v.ln2.weight/.bias        [3840]            LayerNorm eps 1e-5 (post-dense)
  v.pos_norm.weight/.bias   [3840]            LayerNorm eps 1e-5 (post pos add)
  v.pos_table.weight        [2, 1120, 3840]   HF [1120, 2, 3840] permuted (1,0,2):
                                              row blocks = [x table, y table]
  v.mm_proj.weight          [3840, 3840]      embed_vision.embedding_projection
  a.mm_proj.weight          [3840, 640]       embed_audio.embedding_projection
                                              (raw 640-sample frames, RMS 1e-6)

Hyperparameters (gemma4.*) mirror what the backend's load() reads; layer types
follow config.json layer_types (5x sliding_attention + full_attention).
"""
import argparse
import json
import os

import numpy as np
import gguf

import _lib
from _lib import log, fail, progress
import hf_source

MODEL_ID = "google/gemma-4-12B"
ENV_VAR = "GEMMA4_SRC"
PATTERNS = ("model.safetensors", "model.safetensors.index.json")

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q6_K": gguf.GGMLQuantizationType.Q6_K,
            "q4_0": gguf.GGMLQuantizationType.Q4_0}

# ---------------------------------------------------------------------------
# HF NAME MAPPING - VERIFIED against google/gemma-4-12B model.safetensors:
#   model.language_model.layers.{i}.{input,post_attention,pre_feedforward,
#     post_feedforward}_layernorm.weight
#   model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight  (v_proj
#     [2048,3840] present on sliding layers, ABSENT on full-attn layers where
#     V = K per attention_k_eq_v; separate q/k/v, never fused)
#   model.language_model.layers.{i}.self_attn.{q,k}_norm.weight
#   model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight
#   model.language_model.layers.{i}.layer_scalar [1]
#   model.language_model.{embed_tokens,norm}.weight
#   model.vision_embedder.{patch_dense.{weight,bias}, patch_ln1/patch_ln2/
#     pos_norm.{weight,bias}, pos_embedding [1120,2,3840]}
#   model.{embed_vision,embed_audio}.embedding_projection.weight
# ---------------------------------------------------------------------------
HF_PREFIX_CANDIDATES = ("model.language_model.", "language_model.", "", "model.", "text_model.")
QKV_FUSED_NAMES = ("self_attn.qkv_proj.weight", "self_attn.attn_qkv.weight")


def resolve_prefix(weights, names):
    for p in HF_PREFIX_CANDIDATES:
        if f"{p}layers.0.self_attn.q_proj.weight" in names or \
           any(f"{p}layers.0.{q}" in names for q in QKV_FUSED_NAMES):
            return p
    fail(f"no Gemma4 text trunk found under prefixes {HF_PREFIX_CANDIDATES} "
         f"(first keys: {sorted(names)[:8]})")


def read_config(src_dir):
    raw = json.load(open(os.path.join(src_dir, "config.json"), encoding="utf-8"))
    # the unified config nests the trunk under text_config
    tc = raw.get("text_config") or raw
    rope = tc.get("rope_parameters") or {}
    rope_full = rope.get("full_attention") or {}
    rope_swa = rope.get("sliding_attention") or {}
    per_layer = {int(k): v for k, v in (tc.get("per_layer_config") or {}).items()}
    layer_types = list(tc["layer_types"])
    head_dim_full = tc.get("global_head_dim")
    if head_dim_full is None:
        for i, t in enumerate(layer_types):
            if t == "full_attention" and i in per_layer and "head_dim" in per_layer[i]:
                head_dim_full = per_layer[i]["head_dim"]
                break
    if head_dim_full is None:
        fail("cannot determine global (full-attention) head_dim from config.json")
    n_kv_full = tc.get("num_global_key_value_heads")
    if n_kv_full is None:
        for i, t in enumerate(layer_types):
            if t == "full_attention" and i in per_layer and "num_key_value_heads" in per_layer[i]:
                n_kv_full = per_layer[i]["num_key_value_heads"]
                break
    cfg = dict(
        n_layer=int(tc["num_hidden_layers"]), n_embd=int(tc["hidden_size"]),
        n_head=int(tc["num_attention_heads"]), n_kv_head=int(tc["num_key_value_heads"]),
        n_kv_head_full=int(n_kv_full if n_kv_full is not None else 1),
        head_dim_full=int(head_dim_full), head_dim_swa=int(tc["head_dim"]),
        n_ff=int(tc["intermediate_size"]), vocab=int(tc["vocab_size"]),
        eps=float(tc["rms_norm_eps"]), window=int(tc["sliding_window"]),
        layer_types=layer_types,
        ple_dim=int(tc.get("hidden_size_per_layer_input") or 0),
        rope_theta_full=float(rope_full.get("rope_theta", 1e6)),
        rope_theta_swa=float(rope_swa.get("rope_theta", 1e4)),
        partial_rotary_full=float(rope_full.get("partial_rotary_factor", 0.25)),
        partial_rotary_swa=float(rope_swa.get("partial_rotary_factor", 1.0)),
        softcap=float(tc.get("final_logit_softcapping", 30.0)),
        max_pos=int(tc.get("max_position_embeddings", 262144)),
        # token ids (fall back to Gemma conventions; see backend TODO note)
        id_bos=int(tc.get("bos_token_id", 2)),
        id_eos=int(tc.get("eos_token_id", 1)),
        id_eot=int(tc.get("end_of_turn_token_id", 106)),
    )
    # unified (multimodal) extras - top level of the Gemma4Unified config
    vc = raw.get("vision_config") or {}
    ac = raw.get("audio_config") or {}
    cfg.update(
        v_patch=int(vc.get("model_patch_size") or vc.get("patch_size", 16)),
        v_soft=int(vc.get("num_soft_tokens", 280)),
        v_pos_table=int(vc.get("mm_posemb_size", 1120)),
        v_eps=float(vc.get("rms_norm_eps", 1e-6)),
        a_frame=int(ac.get("audio_samples_per_token", 640)),
        a_sr=16000,
        id_boi=int(raw.get("boi_token_id", 255999)),
        id_eoi=int(raw.get("eoi_token_id", 258882)),
        id_image=int(raw.get("image_token_id", 258880)),
        id_boa=int(raw.get("boa_token_id", 256000)),
        id_eoa=int(raw.get("eoa_token_index", 258883)),
        id_audio=int(raw.get("audio_token_id", 258881)),
        id_video=int(raw.get("video_token_id", 258884)),
    )
    # generation_config.json overrides the chat ids when present
    gen = os.path.join(src_dir, "generation_config.json")
    if os.path.exists(gen):
        g = json.load(open(gen, encoding="utf-8"))
        cfg["id_bos"] = int(g.get("bos_token_id", cfg["id_bos"]))
        cfg["id_eos"] = int(g.get("eos_token_id", cfg["id_eos"]))
    return cfg


def write_metadata(gw, cfg, outtype):
    gw.add_name("gemma4")
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q6_K": gguf.LlamaFileType.MOSTLY_Q6_K,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32("gemma4.block_count", cfg["n_layer"])
    gw.add_uint32("gemma4.embedding_length", cfg["n_embd"])
    gw.add_uint32("gemma4.feed_forward_length", cfg["n_ff"])
    gw.add_uint32("gemma4.attention.head_count", cfg["n_head"])
    gw.add_uint32("gemma4.attention.head_count_kv", cfg["n_kv_head"])          # sliding
    gw.add_uint32("gemma4.attention.head_count_kv_full", cfg["n_kv_head_full"])  # full-attn
    gw.add_uint32("gemma4.attention.key_length", cfg["head_dim_full"])          # full-attn hd
    gw.add_uint32("gemma4.attention.key_length_swa", cfg["head_dim_swa"])
    gw.add_uint32("gemma4.attention.sliding_window", cfg["window"])
    gw.add_float32("gemma4.attention.layer_norm_rms_epsilon", cfg["eps"])
    gw.add_float32("gemma4.attention.final_logit_softcapping", cfg["softcap"])
    gw.add_float32("gemma4.rope.freq_base", cfg["rope_theta_full"])
    gw.add_float32("gemma4.rope.freq_base_swa", cfg["rope_theta_swa"])
    # rope n_dims as llama.cpp writes them: full layers rotate the WHOLE head
    # (the freq factors pin the unrotated dims); sliding = head_dim * partial
    gw.add_uint32("gemma4.rope.dimension_count", cfg["head_dim_full"])
    gw.add_uint32("gemma4.rope.dimension_count_swa",
                  int(cfg["head_dim_swa"] * cfg["partial_rotary_swa"]))
    gw.add_float32("gemma4.rope.partial_rotary_factor_full", cfg["partial_rotary_full"])
    gw.add_uint32("gemma4.rope.context_length", cfg["max_pos"])
    gw.add_uint32("gemma4.embedding_length_per_layer_input", cfg["ple_dim"])
    gw.add_uint32("gemma4.vocab_size", cfg["vocab"])
    # 1 = sliding_attention (matches llama.cpp LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN
    # semantics: is_swa_impl per layer)
    gw.add_array("gemma4.attention.sliding_window_pattern",
                 [1 if t == "sliding_attention" else 0 for t in cfg["layer_types"]])
    gw.add_uint32("gemma4.bos_token_id", cfg["id_bos"])
    gw.add_uint32("gemma4.eos_token_id", cfg["id_eos"])
    gw.add_uint32("gemma4.end_of_turn_token_id", cfg["id_eot"])

    # unified multimodal projections (encoder-free: no towers)
    gw.add_uint32("gemma4.vision.patch_size", cfg["v_patch"])          # 48
    gw.add_uint32("gemma4.vision.max_soft_tokens", cfg["v_soft"])     # 280
    gw.add_uint32("gemma4.vision.min_soft_tokens", 70)                # llama.cpp set_limit_image_tokens(70, 1120)
    gw.add_uint32("gemma4.vision.pos_table_size", cfg["v_pos_table"])  # 1120
    gw.add_float32("gemma4.vision.rms_norm_eps", cfg["v_eps"])
    gw.add_uint32("gemma4.audio.frame_samples", cfg["a_frame"])       # 640 @ 16 kHz
    gw.add_uint32("gemma4.audio.sample_rate", cfg["a_sr"])
    gw.add_uint32("gemma4.boi_token_id", cfg["id_boi"])
    gw.add_uint32("gemma4.eoi_token_id", cfg["id_eoi"])
    gw.add_uint32("gemma4.image_token_id", cfg["id_image"])
    gw.add_uint32("gemma4.boa_token_id", cfg["id_boa"])
    gw.add_uint32("gemma4.eoa_token_id", cfg["id_eoa"])
    gw.add_uint32("gemma4.audio_token_id", cfg["id_audio"])
    gw.add_uint32("gemma4.video_token_id", cfg["id_video"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the 2D weights (default f16)")
    ap.add_argument("-o", "--out", default=None, help="output file (default gemma4-12b-<outtype>.gguf)")
    ap.add_argument("-q", "--quiet", action="store_true")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"gemma4-12b-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    cfg = read_config(src)
    qtype = OUTTYPES[args.outtype]
    n_full = sum(1 for t in cfg["layer_types"] if t == "full_attention")
    log(f"text: {cfg['n_layer']}L d={cfg['n_embd']} ff={cfg['n_ff']} "
        f"heads={cfg['n_head']} kv={cfg['n_kv_head']}(swa)/{cfg['n_kv_head_full']}(full) "
        f"hd={cfg['head_dim_swa']}/{cfg['head_dim_full']} window={cfg['window']} "
        f"theta={cfg['rope_theta_swa']:g}/{cfg['rope_theta_full']:g} "
        f"full-attn layers={n_full} ple={cfg['ple_dim']} softcap={cfg['softcap']:g}")

    weights = _lib.TensorSource(src)
    names = set(weights.names())
    prefix = resolve_prefix(weights, names)
    log(f"{len(weights)} tensors; text prefix: {prefix!r}")
    qtype_eff = qtype

    gw = gguf.GGUFWriter(out, "gemma4")
    write_metadata(gw, cfg, args.outtype)

    def linear(name, hf_name, optional=False):
        if hf_name not in names:
            if optional:
                return None
            fail(f"missing tensor {hf_name} (fix the HF NAME MAP in this script)")
        arr = weights.get(hf_name)
        if qtype_eff is not None and arr.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, name, _lib.quantize(arr, qtype_eff), qtype_eff)
        else:
            _lib.add_f16(gw, name, arr)
        return name

    def f32(name, hf_name, optional=False):
        if hf_name not in names:
            if optional:
                return None
            fail(f"missing tensor {hf_name} (fix the HF NAME MAP in this script)")
        gw.add_tensor(name, np.ascontiguousarray(weights.get(hf_name), dtype=np.float32))
        return name

    # ------------------------------------------------------------- text
    # tie_word_embeddings: the output head IS the token embedding table
    linear("t.embed.weight", f"{prefix}embed_tokens.weight")
    f32("t.norm.weight", f"{prefix}norm.weight")

    if cfg["ple_dim"] > 0:
        # PLE table rows are per-layer blocks of ple_dim (HF
        # per_layer_embeddings layout is UNVERIFIED - check the checkpoint)
        f32("t.ple_table.weight", f"{prefix}per_layer_embeddings.weight")
        linear("t.ple_model_proj.weight", f"{prefix}ple.per_layer_model_projection.weight")
        f32("t.ple_norm.weight", f"{prefix}ple.per_layer_projection_norm.weight")

    # proportional-rope freq factors for the full-attn layers, exactly as
    # llama.cpp writes them: [1.0]*(hd*partial/2) + [1e30]*(hd/2 - that)
    hd_full, partial = cfg["head_dim_full"], cfg["partial_rotary_full"]
    n_rot_pairs = int(hd_full * partial) // 2
    rope_freqs = np.array([1.0] * n_rot_pairs + [1e30] * (hd_full // 2 - n_rot_pairs),
                          dtype=np.float32)

    for i in range(cfg["n_layer"]):
        h, b = f"{prefix}layers.{i}", f"t.blk.{i}"
        is_full = cfg["layer_types"][i] == "full_attention"
        f32(f"{b}.attn_norm.weight", f"{h}.input_layernorm.weight")
        f32(f"{b}.attn_post_norm.weight", f"{h}.post_attention_layernorm.weight")
        f32(f"{b}.ffn_norm.weight", f"{h}.pre_feedforward_layernorm.weight")
        f32(f"{b}.ffn_post_norm.weight", f"{h}.post_feedforward_layernorm.weight")
        # QKV may be fused (use_alternative_attention / attn_qkv); if so, split
        # it Q|K|V here so the backend's separate wq/wk(/wv) layout holds
        fused = next((q for q in QKV_FUSED_NAMES if f"{h}.{q}" in names), None)
        if fused is not None:
            hd = cfg["head_dim_full"] if is_full else cfg["head_dim_swa"]
            nkv = cfg["n_kv_head_full"] if is_full else cfg["n_kv_head"]
            qkv = weights.get(f"{h}.{fused}")
            dims = cfg["n_head"] * hd + 2 * nkv * hd
            if qkv.shape[0] != dims:
                fail(f"layer {i}: fused qkv out-dim {qkv.shape[0]} != {dims}")
            q, k, v = qkv[: cfg["n_head"] * hd], \
                qkv[cfg["n_head"] * hd : cfg["n_head"] * hd + nkv * hd], \
                qkv[cfg["n_head"] * hd + nkv * hd :]
            _lib.add_f16(gw, f"{b}.wq.weight", q)
            _lib.add_f16(gw, f"{b}.wk.weight", k)
            _lib.add_f16(gw, f"{b}.wv.weight", v)   # kept even when k_eq_v
        else:
            linear(f"{b}.wq.weight", f"{h}.self_attn.q_proj.weight")
            linear(f"{b}.wk.weight", f"{h}.self_attn.k_proj.weight")
            # attention_k_eq_v: v_proj usually absent - V = K output (optional)
            linear(f"{b}.wv.weight", f"{h}.self_attn.v_proj.weight", optional=True)
        linear(f"{b}.wo.weight", f"{h}.self_attn.o_proj.weight")
        f32(f"{b}.q_norm.weight", f"{h}.self_attn.q_norm.weight")
        f32(f"{b}.k_norm.weight", f"{h}.self_attn.k_norm.weight")
        linear(f"{b}.w_gate.weight", f"{h}.mlp.gate_proj.weight")
        linear(f"{b}.w_up.weight", f"{h}.mlp.up_proj.weight")
        linear(f"{b}.w_down.weight", f"{h}.mlp.down_proj.weight")
        f32(f"{b}.out_scale", f"{h}.layer_scalar", optional=True)
        if cfg["ple_dim"] > 0:
            linear(f"{b}.ple_gate.weight", f"{h}.ple_block.per_layer_input_gate.weight")
            linear(f"{b}.ple_proj.weight", f"{h}.ple_block.per_layer_projection.weight")
            f32(f"{b}.ple_post_norm.weight", f"{h}.ple_block.post_per_layer_input_norm.weight")
        if is_full:
            gw.add_tensor(f"{b}.rope_freqs.weight", rope_freqs)
        progress(i + 1, cfg["n_layer"], "text blocks")

    # ------------------------------------------- unified vision projection
    # HF patch_dense columns are HWC (the HF processor flattens each 48x48x3
    # patch row-major over (dy, dx, c)); the backend extracts host patches in
    # exactly that order, so the weight ships UNPERMUTED (unlike llama.cpp,
    # which permutes it to CHW to match ggml im2col).
    ve = "model.vision_embedder."
    linear("v.patch_dense.weight", f"{ve}patch_dense.weight")
    f32("v.patch_dense.bias", f"{ve}patch_dense.bias")
    f32("v.ln1.weight", f"{ve}patch_ln1.weight")
    f32("v.ln1.bias", f"{ve}patch_ln1.bias")
    f32("v.ln2.weight", f"{ve}patch_ln2.weight")
    f32("v.ln2.bias", f"{ve}patch_ln2.bias")
    f32("v.pos_norm.weight", f"{ve}pos_norm.weight")
    f32("v.pos_norm.bias", f"{ve}pos_norm.bias")
    # pos_embedding [1120, 2, 3840] -> [2, 1120, 3840]: row block 0 = x table,
    # row block 1 = y table (the backend views the two halves for get_rows)
    pos = weights.get(f"{ve}pos_embedding").transpose(1, 0, 2)
    gw.add_tensor("v.pos_table.weight", np.ascontiguousarray(pos, dtype=np.float32))
    linear("v.mm_proj.weight", "model.embed_vision.embedding_projection.weight")
    log("vision projection: patch_dense + ln1/ln2/pos_norm + pos table "
        f"[{pos.shape[0]}, {pos.shape[1]}, {pos.shape[2]}] + mm_proj")

    # -------------------------------------------- unified audio projection
    linear("a.mm_proj.weight", "model.embed_audio.embedding_projection.weight")
    log(f"audio projection: mm_proj [3840, {cfg['a_frame']}] "
        f"({cfg['a_frame']} raw samples @ 16 kHz per token)")

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    names = {t.name for t in gguf.GGUFReader(out).tensors}
    required = ["t.embed.weight", "t.norm.weight",
                f"t.blk.{cfg['n_layer'] - 1}.w_down.weight",
                f"t.blk.{cfg['n_layer'] - 1}.rope_freqs.weight",
                "v.patch_dense.weight", "v.pos_table.weight",
                "v.mm_proj.weight", "a.mm_proj.weight"]
    missing = [n for n in required if n not in names]
    if missing:
        fail(f"written GGUF is missing {missing}")
    log(f"verified {len(names)} tensors, {os.path.getsize(out) / 1e6:.0f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
