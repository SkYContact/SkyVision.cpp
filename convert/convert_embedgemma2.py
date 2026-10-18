"""Convert an EmbeddingGemma-2 checkpoint to GGUF.

Usage::

  python convert/convert_embedgemma2.py [--src DIR] [-o OUT.gguf] [--outtype f16|q8_0|q4_0]

Source (google/embeddinggemma-2, 740M: text 270M + vision 170M + audio 300M)::

  hf download google/embeddinggemma-2 --local-dir models/embeddinggemma-2
  # (this machine: modelscope.cn mirror works without a token)

Layout (``general.architecture = embgemma2``):

  text (bidirectional encoder, no KV cache)
    t.embed.weight            [262144, 512]    scaled by sqrt(512) in-graph
    t.blk.{i}.attn_norm / attn_post_norm / ffn_norm / ffn_post_norm
    t.blk.{i}.wq/wk/wv/wo                      sliding layers hd=256, 4:2 GQA;
                                               full layers (5,11,17,23) hd=512, 4:1
    t.blk.{i}.q_norm/k_norm                    per-head RMS (k/v_norm weightless, v raw)
    t.blk.{i}.ffn_norm + w_gate/w_up/w_down    gelu (tanh) SwiGLU
    t.blk.{i}.ple_gate/ple_proj/ple_post_norm  per-layer-embedding block
    t.blk.{i}.out_scale                        per-layer scalar
    t.ple_model_proj / t.ple_norm              projection-only PLE from inputs_embeds
    t.norm / t.out_proj                        final norm + 512 -> 768 projection
  vision (Gemma4 tower, SigLIP-style)
    v.patch_proj.weight       [768, 3*16*16]   linear patchify (no conv), x = 2*(px-0.5)
    v.pos_table               [2, 10240, 768]  learned (x, y) position embeddings
    v.blk.{i}.*                                pre/post-norm attention + MLP, axial 2D
                                               rope (theta 100, H-H-W-W layout)
    v.mm_proj.weight          [512, 768]       soft-token projection (after 3x3 avg
                                               pool * sqrt(768), pooling done in-graph)
  audio (Gemma4 tower, USM-style conformer)
    a.sub.l0.conv/l1.conv     [128, 9] [32, 1152]  stride-2 3x3 convs over the mel axis
    a.sub.l0.norm/l1.norm     LayerNorm (weight only) over channels, then ReLU
    a.sub.in_proj             [1024, 1024]
    a.blk.{i}.ff1/ff2.l1/l2   clipped linears (+ per-linear input/output clamp scalars)
    a.blk.{i}.attn.wq/wk/wv/post + rel_k + per_dim_scale   chunked (12) local attention,
                                               past 12 / future 0, softcap 50
    a.blk.{i}.lconv.start/end/dw               GLU + causal depthwise conv (k=5)
    a.blk.{i}.norm_*                           RMS norms; layer out = norm_out (no add)
    a.out_proj.weight+bias     [1536, 1024]
    a.mm_proj.weight           [512, 1536]      soft-token projection

Fusion: image/video/audio soft tokens replace their placeholder ids in the token
sequence (image <|image|> x N wrapped in <|image>..<image|>, audio <|audio|> x N
in <|audio>..<audio|>), the text encoder runs over the whole sequence, and the
embedding is mean-pooled over all tokens then L2-normalized (MRL: slice to
512/256/128/64 and re-normalize for smaller dims).
"""
import argparse
import json
import os

import numpy as np
import gguf

import _lib
from _lib import log, fail, progress
import hf_source

MODEL_ID = "google/embeddinggemma-2"
ENV_VAR = "EMBEDDINGGEMMA2_SRC"
PATTERNS = ("model.safetensors", "model.safetensors.index.json")

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q4_0": gguf.GGMLQuantizationType.Q4_0}


def read_config(src_dir):
    raw = json.load(open(os.path.join(src_dir, "config.json"), encoding="utf-8"))
    tc = raw["text_config"]
    vc = raw["vision_config"]
    ac = raw["audio_config"]
    pp = json.load(open(os.path.join(src_dir, "preprocessor_config.json"), encoding="utf-8"))
    pconf = json.load(open(os.path.join(src_dir, "processor_config.json"), encoding="utf-8"))
    cfg = dict(
        # text
        n_layer=int(tc["num_hidden_layers"]), n_embd=int(tc["hidden_size"]),
        n_head=int(tc["num_attention_heads"]), n_kv_head=int(tc["num_key_value_heads"]),
        head_dim=int(tc["head_dim"]), n_ff=int(tc["intermediate_size"]),
        vocab=int(tc["vocab_size"]), eps=float(tc["rms_norm_eps"]),
        ple_dim=int(tc["hidden_size_per_layer_input"]), out_dim=int(tc["embedding_dim"]),
        window=int(tc["sliding_window"]),
        rope_full=float(tc["rope_parameters"]["full_attention"]["rope_theta"]),
        rope_swa=float(tc["rope_parameters"]["sliding_attention"]["rope_theta"]),
        layer_types=list(tc["layer_types"]),
        per_layer_config={int(k): v for k, v in (tc.get("per_layer_config") or {}).items()},
        # vision
        v_layer=int(vc["num_hidden_layers"]), v_embd=int(vc["hidden_size"]),
        v_head=int(vc["num_attention_heads"]), v_head_kv=int(vc["num_key_value_heads"]),
        v_head_dim=int(vc["head_dim"]), v_ff=int(vc["intermediate_size"]),
        v_patch=int(vc["patch_size"]), v_pool=int(vc["pooling_kernel_size"]),
        v_pos_table=int(vc["position_embedding_size"]),
        v_rope=float(vc["rope_parameters"]["rope_theta"]), v_eps=float(vc["rms_norm_eps"]),
        max_soft_tokens=int(pconf["image_processor"]["max_soft_tokens"]),
        video_max_soft_tokens=int(pconf["video_processor"].get("max_soft_tokens", 140)),
        v_resample=int(pconf["image_processor"]["resample"]),
        v_fps=float(pconf["video_processor"]["fps"]),
        # audio
        a_layer=int(ac["num_hidden_layers"]), a_embd=int(ac["hidden_size"]),
        a_head=int(ac["num_attention_heads"]), a_ff=int(ac["hidden_size"]) * 4,
        a_conv_kernel=int(ac["conv_kernel_size"]), a_chunk=int(ac["attention_chunk_size"]),
        a_ctx_left=int(ac["attention_context_left"]), a_ctx_right=int(ac["attention_context_right"]),
        a_logit_cap=float(ac["attention_logit_cap"]),
        a_sub_channels=[int(x) for x in ac["subsampling_conv_channels"]],
        a_out_dim=int(ac["output_proj_dims"]), a_eps=float(ac["rms_norm_eps"]),
        a_residual=float(ac["residual_weight"]),
        # mel (Gemma4AudioFeatureExtractor)
        mel_bins=int(pp["feature_size"]), mel_fft=int(pp["fft_length"]),
        mel_frame=int(pp["frame_length"]), mel_hop=int(pp["hop_length"]),
        mel_sr=int(pp["sampling_rate"]), mel_floor=float(pp["mel_floor"]),
        mel_max_freq=float(pp["max_frequency"]), mel_min_freq=float(pp["min_frequency"]),
        # tokens
        id_bos=int(tc.get("bos_token_id", 2)), id_eos=int(tc.get("eos_token_id", 1)),
        id_pad=int(tc.get("pad_token_id", 0)),
        id_boi=int(raw["boi_token_id"]), id_eoi=int(raw["eoi_token_id"]),
        id_image=int(raw["image_token_id"]), id_video=int(raw["video_token_id"]),
        id_boa=int(raw["boa_token_id"]), id_eoa=int(raw["eoa_token_index"]),
        id_audio=int(raw["audio_token_id"]),
        audio_ms_per_token=int(pconf.get("audio_ms_per_token", 40)),
    )
    cfg["max_patches"] = cfg["max_soft_tokens"] * cfg["v_pool"] ** 2
    return cfg


def write_metadata(gw, cfg, outtype):
    gw.add_name("embgemma2")
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32("embgemma2.block_count", cfg["n_layer"])
    gw.add_uint32("embgemma2.embedding_length", cfg["n_embd"])
    gw.add_uint32("embgemma2.feed_forward_length", cfg["n_ff"])
    gw.add_uint32("embgemma2.attention.head_count", cfg["n_head"])
    gw.add_uint32("embgemma2.attention.head_count_kv", cfg["n_kv_head"])
    gw.add_uint32("embgemma2.attention.key_length", cfg["head_dim"])
    gw.add_uint32("embgemma2.attention.sliding_window", cfg["window"])
    gw.add_float32("embgemma2.attention.layer_norm_rms_epsilon", float(cfg["eps"]))
    gw.add_float32("embgemma2.rope.freq_base", float(cfg["rope_full"]))
    gw.add_float32("embgemma2.rope.freq_base_swa", float(cfg["rope_swa"]))
    gw.add_uint32("embgemma2.embedding_length_per_layer_input", cfg["ple_dim"])
    gw.add_uint32("embgemma2.embedding_out_length", cfg["out_dim"])
    gw.add_uint32("embgemma2.vocab_size", cfg["vocab"])

    gw.add_uint32("embgemma2.vision.block_count", cfg["v_layer"])
    gw.add_uint32("embgemma2.vision.embedding_length", cfg["v_embd"])
    gw.add_uint32("embgemma2.vision.feed_forward_length", cfg["v_ff"])
    gw.add_uint32("embgemma2.vision.attention.head_count", cfg["v_head"])
    gw.add_uint32("embgemma2.vision.attention.head_count_kv", cfg["v_head_kv"])
    gw.add_uint32("embgemma2.vision.attention.key_length", cfg["v_head_dim"])
    gw.add_uint32("embgemma2.vision.patch_size", cfg["v_patch"])
    gw.add_uint32("embgemma2.vision.pooling_kernel_size", cfg["v_pool"])
    gw.add_uint32("embgemma2.vision.position_table_size", cfg["v_pos_table"])
    gw.add_float32("embgemma2.vision.rope.freq_base", float(cfg["v_rope"]))
    gw.add_float32("embgemma2.vision.layer_norm_rms_epsilon", float(cfg["v_eps"]))
    gw.add_uint32("embgemma2.vision.max_soft_tokens", cfg["max_soft_tokens"])
    gw.add_uint32("embgemma2.vision.video_max_soft_tokens",
                  cfg["video_max_soft_tokens"])
    gw.add_uint32("embgemma2.vision.resize_resample", cfg["v_resample"])
    gw.add_float32("embgemma2.vision.video_fps", float(cfg["v_fps"]))

    gw.add_uint32("embgemma2.audio.block_count", cfg["a_layer"])
    gw.add_uint32("embgemma2.audio.embedding_length", cfg["a_embd"])
    gw.add_uint32("embgemma2.audio.feed_forward_length", cfg["a_ff"])
    gw.add_uint32("embgemma2.audio.attention.head_count", cfg["a_head"])
    gw.add_uint32("embgemma2.audio.conv_kernel_size", cfg["a_conv_kernel"])
    gw.add_uint32("embgemma2.audio.attention_chunk_size", cfg["a_chunk"])
    gw.add_uint32("embgemma2.audio.attention_context_left", cfg["a_ctx_left"])
    gw.add_uint32("embgemma2.audio.attention_context_right", cfg["a_ctx_right"])
    gw.add_float32("embgemma2.audio.attention_logit_cap", float(cfg["a_logit_cap"]))
    gw.add_array("embgemma2.audio.subsampling_conv_channels",
                 [float(x) for x in cfg["a_sub_channels"]])
    gw.add_uint32("embgemma2.audio.out_proj_length", cfg["a_out_dim"])
    gw.add_float32("embgemma2.audio.layer_norm_rms_epsilon", float(cfg["a_eps"]))
    gw.add_float32("embgemma2.audio.residual_weight", float(cfg["a_residual"]))

    gw.add_uint32("embgemma2.mel.bins", cfg["mel_bins"])
    gw.add_uint32("embgemma2.mel.fft_length", cfg["mel_fft"])
    gw.add_uint32("embgemma2.mel.frame_length", cfg["mel_frame"])
    gw.add_uint32("embgemma2.mel.hop_length", cfg["mel_hop"])
    gw.add_uint32("embgemma2.mel.sample_rate", cfg["mel_sr"])
    gw.add_float32("embgemma2.mel.floor", float(cfg["mel_floor"]))
    gw.add_float32("embgemma2.mel.max_frequency", float(cfg["mel_max_freq"]))
    gw.add_float32("embgemma2.mel.min_frequency", float(cfg["mel_min_freq"]))
    gw.add_uint32("embgemma2.audio_ms_per_token", cfg["audio_ms_per_token"])

    for key in ("bos", "eos", "pad", "boi", "eoi", "image", "video", "boa", "eoa", "audio"):
        gw.add_uint32(f"embgemma2.{key}_token_id", int(cfg[f"id_{key}"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the 2D weights (default f16)")
    ap.add_argument("-o", "--out", default=None, help="output file (default embeddinggemma2-<outtype>.gguf)")
    ap.add_argument("-q", "--quiet", action="store_true")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or f"embeddinggemma2-{args.outtype}.gguf"
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    cfg = read_config(src)
    qtype = OUTTYPES[args.outtype]
    log(f"text: {cfg['n_layer']}L d={cfg['n_embd']} ff={cfg['n_ff']} ple={cfg['ple_dim']} "
        f"out={cfg['out_dim']} window={cfg['window']} theta={cfg['rope_full']:g}/{cfg['rope_swa']:g}")
    log(f"vision: {cfg['v_layer']}L d={cfg['v_embd']} hd={cfg['v_head_dim']} "
        f"patch={cfg['v_patch']} pool={cfg['v_pool']} soft<={cfg['max_soft_tokens']} "
        f"theta={cfg['v_rope']:g} resample={cfg['v_resample']}")
    log(f"audio: {cfg['a_layer']}L d={cfg['a_embd']} heads={cfg['a_head']} "
        f"chunk={cfg['a_chunk']} ctx=({cfg['a_ctx_left']},{cfg['a_ctx_right']}) "
        f"out={cfg['a_out_dim']} mel={cfg['mel_bins']}x{cfg['mel_fft']}")

    weights = _lib.TensorSource(src)
    log(f"{len(weights)} tensors in the checkpoint")

    gw = gguf.GGUFWriter(out, "embgemma2")
    write_metadata(gw, cfg, args.outtype)

    def linear(name, hf_name, min_k=None, max_last=256):
        arr = weights.get(hf_name)
        if qtype is not None and arr.shape[-1] % max_last == 0:
            _lib.add_quantized(gw, name, _lib.quantize(arr, qtype), qtype)
        else:
            _lib.add_f16(gw, name, arr)

    def f32(name, hf_name):
        gw.add_tensor(name, np.ascontiguousarray(weights.get(hf_name), dtype=np.float32))

    def f16(name, hf_name):
        _lib.add_f16(gw, name, weights.get(hf_name))

    def clip(name, hf_base):
        for s in ("input_min", "input_max", "output_min", "output_max"):
            f32(f"{name}.{s}", f"{hf_base}.{s}")

    full_layers = {i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"}

    # --------------------------------------------------------------- text
    f16("t.embed.weight", "language_model.embed_tokens.weight")
    f16("t.ple_model_proj.weight", "language_model.ple.per_layer_model_projection.weight")
    f32("t.ple_norm.weight", "language_model.ple.per_layer_projection_norm.weight")
    for i in range(cfg["n_layer"]):
        h, b = f"language_model.layers.{i}", f"t.blk.{i}"
        f32(f"{b}.attn_norm.weight", f"{h}.input_layernorm.weight")
        f32(f"{b}.attn_post_norm.weight", f"{h}.post_attention_layernorm.weight")
        f32(f"{b}.ffn_norm.weight", f"{h}.pre_feedforward_layernorm.weight")
        f32(f"{b}.ffn_post_norm.weight", f"{h}.post_feedforward_layernorm.weight")
        linear(f"{b}.wq.weight", f"{h}.self_attn.q_proj.weight")
        linear(f"{b}.wk.weight", f"{h}.self_attn.k_proj.weight")
        linear(f"{b}.wv.weight", f"{h}.self_attn.v_proj.weight")
        linear(f"{b}.wo.weight", f"{h}.self_attn.o_proj.weight")
        f32(f"{b}.q_norm.weight", f"{h}.self_attn.q_norm.weight")
        f32(f"{b}.k_norm.weight", f"{h}.self_attn.k_norm.weight")
        linear(f"{b}.w_gate.weight", f"{h}.mlp.gate_proj.weight")
        linear(f"{b}.w_up.weight", f"{h}.mlp.up_proj.weight")
        linear(f"{b}.w_down.weight", f"{h}.mlp.down_proj.weight")
        linear(f"{b}.ple_gate.weight", f"{h}.ple_block.per_layer_input_gate.weight")
        linear(f"{b}.ple_proj.weight", f"{h}.ple_block.per_layer_projection.weight")
        f32(f"{b}.ple_post_norm.weight", f"{h}.ple_block.post_per_layer_input_norm.weight")
        f32(f"{b}.out_scale", f"{h}.layer_scalar")
        progress(i + 1, cfg["n_layer"], "text blocks")
    f32("t.norm.weight", "language_model.norm.weight")
    linear("t.out_proj.weight", "language_model.embedding_projection.weight")

    # --------------------------------------------------------------- vision
    f16("v.patch_proj.weight", "vision_tower.patch_embedder.input_proj.weight")
    f16("v.pos_table.weight", "vision_tower.patch_embedder.position_embedding_table")
    for i in range(cfg["v_layer"]):
        h, b = f"vision_tower.encoder.layers.{i}", f"v.blk.{i}"
        f32(f"{b}.attn_norm.weight", f"{h}.input_layernorm.weight")
        f32(f"{b}.attn_post_norm.weight", f"{h}.post_attention_layernorm.weight")
        f32(f"{b}.ffn_norm.weight", f"{h}.pre_feedforward_layernorm.weight")
        f32(f"{b}.ffn_post_norm.weight", f"{h}.post_feedforward_layernorm.weight")
        linear(f"{b}.wq.weight", f"{h}.self_attn.q_proj.linear.weight")
        linear(f"{b}.wk.weight", f"{h}.self_attn.k_proj.linear.weight")
        linear(f"{b}.wv.weight", f"{h}.self_attn.v_proj.linear.weight")
        linear(f"{b}.wo.weight", f"{h}.self_attn.o_proj.linear.weight")
        f32(f"{b}.q_norm.weight", f"{h}.self_attn.q_norm.weight")
        f32(f"{b}.k_norm.weight", f"{h}.self_attn.k_norm.weight")
        linear(f"{b}.w_gate.weight", f"{h}.mlp.gate_proj.linear.weight")
        linear(f"{b}.w_up.weight", f"{h}.mlp.up_proj.linear.weight")
        linear(f"{b}.w_down.weight", f"{h}.mlp.down_proj.linear.weight")
        progress(i + 1, cfg["v_layer"], "vision blocks")
    linear("v.mm_proj.weight", "embed_vision.embedding_projection.weight")

    # --------------------------------------------------------------- audio
    f16("a.sub.l0.conv.weight", "audio_tower.subsample_conv_projection.layer0.conv.weight")
    f32("a.sub.l0.norm.weight", "audio_tower.subsample_conv_projection.layer0.norm.weight")
    f16("a.sub.l1.conv.weight", "audio_tower.subsample_conv_projection.layer1.conv.weight")
    f32("a.sub.l1.norm.weight", "audio_tower.subsample_conv_projection.layer1.norm.weight")
    f16("a.sub.in_proj.weight", "audio_tower.subsample_conv_projection.input_proj_linear.weight")
    for i in range(cfg["a_layer"]):
        h, b = f"audio_tower.layers.{i}", f"a.blk.{i}"
        for ff in ("ff1", "ff2"):
            f32(f"{b}.{ff}.pre_norm", f"{h}.feed_forward{ff[-1]}.pre_layer_norm.weight")
            f32(f"{b}.{ff}.post_norm", f"{h}.feed_forward{ff[-1]}.post_layer_norm.weight")
            linear(f"{b}.{ff}.l1.weight", f"{h}.feed_forward{ff[-1]}.ffw_layer_1.linear.weight")
            clip(f"{b}.{ff}.l1", f"{h}.feed_forward{ff[-1]}.ffw_layer_1")
            linear(f"{b}.{ff}.l2.weight", f"{h}.feed_forward{ff[-1]}.ffw_layer_2.linear.weight")
            clip(f"{b}.{ff}.l2", f"{h}.feed_forward{ff[-1]}.ffw_layer_2")
        f32(f"{b}.norm_pre_attn.weight", f"{h}.norm_pre_attn.weight")
        f32(f"{b}.norm_post_attn.weight", f"{h}.norm_post_attn.weight")
        f32(f"{b}.norm_out.weight", f"{h}.norm_out.weight")
        for proj in ("wq", "wk", "wv", "post"):
            hf = {"wq": "q_proj", "wk": "k_proj", "wv": "v_proj", "post": "post"}[proj]
            linear(f"{b}.attn.{proj}.weight", f"{h}.self_attn.{hf}.linear.weight")
            clip(f"{b}.attn.{proj}", f"{h}.self_attn.{hf}")
        f16(f"{b}.attn.rel_k.weight", f"{h}.self_attn.relative_k_proj.weight")
        f32(f"{b}.attn.per_dim_scale.weight", f"{h}.self_attn.per_dim_scale")
        f32(f"{b}.lconv.pre_norm.weight", f"{h}.lconv1d.pre_layer_norm.weight")
        f32(f"{b}.lconv.conv_norm.weight", f"{h}.lconv1d.conv_norm.weight")
        linear(f"{b}.lconv.start.weight", f"{h}.lconv1d.linear_start.linear.weight")
        clip(f"{b}.lconv.start", f"{h}.lconv1d.linear_start")
        linear(f"{b}.lconv.end.weight", f"{h}.lconv1d.linear_end.linear.weight")
        clip(f"{b}.lconv.end", f"{h}.lconv1d.linear_end")
        f32(f"{b}.lconv.dw.weight", f"{h}.lconv1d.depthwise_conv1d.weight")
        progress(i + 1, cfg["a_layer"], "audio blocks")
    linear("a.out_proj.weight", "audio_tower.output_proj.weight")
    f32("a.out_proj.bias", "audio_tower.output_proj.bias")
    linear("a.mm_proj.weight", "embed_audio.embedding_projection.weight")

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    names = {t.name for t in gguf.GGUFReader(out).tensors}
    required = ["t.embed.weight", "t.norm.weight", "t.out_proj.weight",
                f"t.blk.{cfg['n_layer'] - 1}.w_down.weight",
                "t.ple_model_proj.weight", "v.patch_proj.weight", "v.pos_table.weight", "v.mm_proj.weight",
                f"v.blk.{cfg['v_layer'] - 1}.w_down.weight",
                "a.sub.in_proj.weight", f"a.blk.{cfg['a_layer'] - 1}.lconv.dw.weight",
                "a.out_proj.weight", "a.out_proj.bias", "a.mm_proj.weight"]
    missing = [n for n in required if n not in names]
    if missing:
        fail(f"written GGUF is missing {missing}")
    log(f"verified {len(names)} tensors, {os.path.getsize(out) / 1e6:.0f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
