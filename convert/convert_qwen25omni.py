"""Convert a Qwen2.5-Omni checkpoint (transformers `qwen2_5_omni`) to GGUF.

Usage::

  python convert/convert_qwen25omni.py [--src DIR] [--out-dir DIR] [--outtype f16|q8_0|q4_0]
                                       [--no-talker] [--skip-thinker]

Emits two files next to each other (same convention as the Mage-VL gate gguf):

  qwen25omni-<outtype>.gguf         thinker: text trunk + vision tower + audio tower
  qwen25omni-talker-<outtype>.gguf  talker LM + token2wav (DiT + BigVGAN + ECAPA)
                                    + the baked speaker dict (Chelsie/Ethan)

Layout (``general.architecture = qwen25omni`` / ``qwen25omni-talker``):

  t.*                    Qwen2.5 text trunk, UNtied (t.output.weight carries lm_head)
  v.*                    Qwen2.5-VL vision tower - q/k/v arrive separate in Omni and are
                         fused into .wqkv here so the qwen2vl graph runs unchanged
  a.*                    Whisper-style audio tower (conv1/conv2 1d, sinusoids position
                         table generated here, 128-bin mel filterbank dumped from the
                         HF feature extractor - the checkpoint does not carry either)
  tk.*                   talker: Qwen2-ish 24L LM at width 896, embed width 2048,
                         thinker_to_talker_proj, codec_head
  d.*                    token2wav DiT (flow matching, RK4 in the runtime)
  bg.*                   BigVGAN vocoder
  spk.*                  baked speakers: spk.<name>.bos_token / .cond / .ref_mel
"""
import argparse
import json
import os

import numpy as np
import gguf

import _lib
from _lib import log, warn, fail, progress
import hf_source

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
ENV_VAR = "QWEN25OMNI_SRC"
PATTERNS = ("model.safetensors", "model.safetensors.index.json")

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0,
            "q4_0": gguf.GGMLQuantizationType.Q4_0}

SPEAKERS = ("Chelsie", "Ethan")


def read_config(src_dir):
    path = os.path.join(src_dir, "config.json")
    if not os.path.isfile(path):
        fail(f"no config.json in {src_dir}")
    raw = json.load(open(path, encoding="utf-8"))
    th = raw["thinker_config"]
    txt, vis, aud = th["text_config"], th["vision_config"], th["audio_config"]
    tk, t2w = raw["talker_config"], raw["token2wav_config"]
    dit, bvg = t2w["dit_config"], t2w["bigvgan_config"]
    n_head = int(txt["num_attention_heads"])
    cfg = dict(
        # text trunk
        n_layer=int(txt["num_hidden_layers"]), n_embd=int(txt["hidden_size"]),
        n_head=n_head, n_kv_head=int(txt["num_key_value_heads"]),
        head_dim=int(txt["head_dim"]) if "head_dim" in txt else int(txt["hidden_size"]) // n_head,
        n_ff=int(txt["intermediate_size"]), vocab=int(txt["vocab_size"]),
        eps=float(txt["rms_norm_eps"]), rope_theta=float(txt["rope_theta"]),
        mrope_section=tuple(txt.get("rope_scaling", {}).get("mrope_section", (16, 24, 24))),
        tie_word_embeddings=bool(txt.get("tie_word_embeddings", False)),
        # token ids (thinker_config level, with talker as fallback)
        audio_token_id=int(th.get("audio_token_index", 151646)),
        audio_start_id=int(th.get("audio_start_token_id", 151647)),
        audio_end_id=int(th.get("audio_end_token_id", 151648)),
        image_token_id=int(th.get("image_token_index", 151655)),
        video_token_id=int(th.get("video_token_index", 151656)),
        vision_start_id=int(th.get("vision_start_token_id", 151652)),
        vision_end_id=int(th.get("vision_end_token_id", 151653)),
        # vision tower
        v_depth=int(vis["depth"]), v_embd=int(vis["embed_dim"]),
        v_head=int(vis["num_heads"]), v_ff=int(vis["intermediate_size"]),
        v_patch=int(vis["patch_size"]), v_temporal=int(vis["temporal_patch_size"]),
        v_merge=int(vis["spatial_merge_size"]), v_out=int(vis["out_hidden_size"]),
        v_eps=float(vis.get("rms_norm_eps", 1e-6)),
        v_full_blocks=tuple(vis["fullatt_block_indexes"]),
        v_window=int(vis["window_size"]),
        # audio tower
        a_layer=int(aud["encoder_layers"]), a_embd=int(aud["d_model"]),
        a_head=int(aud["encoder_attention_heads"]), a_ff=int(aud["encoder_ffn_dim"]),
        a_mel=int(aud["num_mel_bins"]), a_pos=int(aud["max_source_positions"]),
        a_window=int(aud["n_window"]), a_out=int(aud["output_dim"]),
        a_eps=float(aud.get("layer_norm_eps", 1e-5)),
        # talker
        tk_layer=int(tk["num_hidden_layers"]), tk_embd=int(tk["hidden_size"]),
        tk_embw=int(tk["embedding_size"]), tk_head=int(tk["num_attention_heads"]),
        tk_kv_head=int(tk["num_key_value_heads"]),
        tk_hd=int(tk["head_dim"]) if "head_dim" in tk else int(tk["hidden_size"]) // int(tk["num_attention_heads"]),
        tk_ff=int(tk["intermediate_size"]), tk_vocab=int(tk["vocab_size"]),
        tk_eps=float(tk["rms_norm_eps"]), tk_theta=float(tk["rope_theta"]),
        tk_section=tuple(tk.get("rope_scaling", {}).get("mrope_section", (16, 16, 0))),
        codec_start=int(tk["tts_codec_start_token_id"]),
        codec_end=int(tk["tts_codec_end_token_id"]),
        codec_pad=int(tk["tts_codec_pad_token_id"]),
        codec_mask=int(tk["tts_codec_mask_token_id"]),
        text_start=int(tk["tts_text_start_token_id"]),
        text_end=int(tk["tts_text_end_token_id"]),
        text_pad=int(tk["tts_text_pad_token_id"]),
        # token2wav DiT
        d_depth=int(dit["depth"]), d_dim=int(dit["dim"]), d_emb=int(dit["emb_dim"]),
        d_head=int(dit["heads"]), d_hd=int(dit["head_dim"]), d_mult=int(dit["ff_mult"]),
        d_mel=int(dit["mel_dim"]), d_embeds=int(dit["num_embeds"]),
        d_repeats=int(dit["repeats"]),
        d_block_size=int(dit.get("block_size", 24)),
        d_look_ahead=tuple(dit.get("look_ahead_layers", (10,))),
        d_look_backward=tuple(dit.get("look_backward_layers", (0, 20))),
        d_theta=float((dit.get("rope_parameters") or {}).get("rope_theta", 10000.0)),
        d_max_pos=int(dit.get("max_position_embeddings", 32768)),
        enc_channels=tuple(int(x) for x in dit["enc_channels"]),
        enc_kernel_sizes=tuple(int(x) for x in dit["enc_kernel_sizes"]),
        enc_dilations=tuple(int(x) for x in dit["enc_dilations"]),
        enc_dim=int(dit["enc_dim"]), enc_emb_dim=int(dit["enc_emb_dim"]),
        enc_scale=int(dit["enc_res2net_scale"]), enc_se=int(dit["enc_se_channels"]),
        enc_attn=int(dit["enc_attention_channels"]),
        # BigVGAN
        b_up_rates=tuple(int(x) for x in bvg["upsample_rates"]),
        b_up_kernels=tuple(int(x) for x in bvg["upsample_kernel_sizes"]),
        b_ch=int(bvg["upsample_initial_channel"]),
        b_rb_kernels=tuple(int(x) for x in bvg["resblock_kernel_sizes"]),
        b_rb_dilations=tuple(tuple(int(x) for x in d) for d in bvg["resblock_dilation_sizes"]),
        b_mel=int(bvg["mel_dim"]),
    )
    # image normalization lives in the processor config (CLIP statistics)
    pp = os.path.join(src_dir, "preprocessor_config.json")
    if os.path.isfile(pp):
        p = json.load(open(pp, encoding="utf-8"))
        cfg["v_mean"] = [float(x) for x in p.get("image_mean", (0.48145466, 0.4578275, 0.40821073))]
        cfg["v_std"] = [float(x) for x in p.get("image_std", (0.26862954, 0.26130258, 0.27577711))]
        cfg["a_hop"] = int(p.get("hop_length", 160))
        cfg["a_nfft"] = int(p.get("n_fft", 400))
        cfg["a_sr"] = int(p.get("sampling_rate", 16000))
    else:
        cfg["v_mean"] = [0.48145466, 0.4578275, 0.40821073]
        cfg["v_std"] = [0.26862954, 0.26130258, 0.27577711]
        cfg["a_hop"], cfg["a_nfft"], cfg["a_sr"] = 160, 400, 16000
    return cfg


def sinusoids(length, channels, max_timescale=10000.0):
    """Same formula as the PR converter / HF SinusoidsPositionEmbedding."""
    log_inc = np.log(max_timescale) / (channels // 2 - 1)
    inv = np.exp(-log_inc * np.arange(channels // 2, dtype=np.float32))
    scaled = np.arange(length, dtype=np.float32)[:, None] * inv[None, :]
    return np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1).astype(np.float32)


def mel_filterbank(src_dir, n_mel, n_fft, sr):
    """Dump the exact mel filterbank the HF feature extractor uses (Slaney)."""
    try:
        from transformers import WhisperFeatureExtractor
        fe = WhisperFeatureExtractor.from_pretrained(src_dir)
        filt = np.asarray(fe.mel_filters, dtype=np.float32)
        if filt.shape == (n_fft // 2 + 1, n_mel):   # transformers 5.x stores it transposed
            filt = filt.T
        if filt.shape != (n_mel, n_fft // 2 + 1):
            fail(f"feature extractor mel bank {filt.shape} != ({n_mel}, {n_fft // 2 + 1})")
        return filt
    except Exception as e:
        warn(f"WhisperFeatureExtractor unavailable ({e}); computing Slaney bank manually")
        # librosa.filters.mel(sr, n_fft, n_mel, fmin=0, fmax=8000, norm="slaney")
        def hz_to_mel(f):
            return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

        m_min, m_max = 0.0, hz_to_mel(sr / 2.0)
        pts = np.linspace(m_min, m_max, n_mel + 2)
        freqs = mel_to_hz(pts)
        ffty = np.fft.rfftfreq(n_fft, 1.0 / sr)
        bank = np.zeros((n_mel, len(ffty)), dtype=np.float32)
        for i in range(n_mel):
            lo, hi = freqs[i], freqs[i + 2]
            for k, f in enumerate(ffty):
                if lo <= f <= hi:
                    bank[i, k] = (f - lo) / (freqs[i + 1] - lo) if f <= freqs[i + 1] \
                        else (hi - f) / (hi - freqs[i + 1])
        enorm = 2.0 / (freqs[2:n_mel + 2] - freqs[:n_mel])
        return bank * enorm[:, None]


def write_thinker_metadata(gw, cfg, outtype):
    A = "qwen25omni"
    gw.add_name(A)
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32(f"{A}.block_count", cfg["n_layer"])
    gw.add_uint32(f"{A}.embedding_length", cfg["n_embd"])
    gw.add_uint32(f"{A}.feed_forward_length", cfg["n_ff"])
    gw.add_uint32(f"{A}.attention.head_count", cfg["n_head"])
    gw.add_uint32(f"{A}.attention.head_count_kv", cfg["n_kv_head"])
    gw.add_uint32(f"{A}.attention.key_length", cfg["head_dim"])
    gw.add_float32(f"{A}.attention.layer_norm_rms_epsilon", float(cfg["eps"]))
    gw.add_float32(f"{A}.rope.freq_base", float(cfg["rope_theta"]))
    gw.add_array(f"{A}.rope.mrope_section", [int(x) for x in cfg["mrope_section"]])
    gw.add_uint32(f"{A}.vocab_size", cfg["vocab"])
    gw.add_bool(f"{A}.tie_word_embeddings", cfg["tie_word_embeddings"])
    for key in ("audio_token_id", "audio_start_id", "audio_end_id", "image_token_id",
                "video_token_id", "vision_start_id", "vision_end_id"):
        gw.add_uint32(f"{A}.{key}", int(cfg[key]))

    gw.add_uint32(f"{A}.vision.block_count", cfg["v_depth"])
    gw.add_uint32(f"{A}.vision.embedding_length", cfg["v_embd"])
    gw.add_uint32(f"{A}.vision.feed_forward_length", cfg["v_ff"])
    gw.add_uint32(f"{A}.vision.attention.head_count", cfg["v_head"])
    gw.add_uint32(f"{A}.vision.patch_size", cfg["v_patch"])
    gw.add_uint32(f"{A}.vision.temporal_patch_size", cfg["v_temporal"])
    gw.add_uint32(f"{A}.vision.spatial_merge_size", cfg["v_merge"])
    gw.add_uint32(f"{A}.vision.out_hidden_size", cfg["v_out"])
    gw.add_uint32(f"{A}.vision.window_size", cfg["v_window"])
    gw.add_float32(f"{A}.vision.layer_norm_epsilon", float(cfg["v_eps"]))
    gw.add_array(f"{A}.vision.image_mean", [float(x) for x in cfg["v_mean"]])
    gw.add_array(f"{A}.vision.image_std", [float(x) for x in cfg["v_std"]])
    gw.add_array(f"{A}.vision.fullatt_blocks", [int(x) for x in cfg["v_full_blocks"]])

    gw.add_uint32(f"{A}.audio.block_count", cfg["a_layer"])
    gw.add_uint32(f"{A}.audio.embedding_length", cfg["a_embd"])
    gw.add_uint32(f"{A}.audio.feed_forward_length", cfg["a_ff"])
    gw.add_uint32(f"{A}.audio.attention.head_count", cfg["a_head"])
    gw.add_uint32(f"{A}.audio.num_mel_bins", cfg["a_mel"])
    gw.add_uint32(f"{A}.audio.max_source_positions", cfg["a_pos"])
    gw.add_uint32(f"{A}.audio.n_window", cfg["a_window"])
    gw.add_uint32(f"{A}.audio.output_dim", cfg["a_out"])
    gw.add_float32(f"{A}.audio.layer_norm_epsilon", float(cfg["a_eps"]))
    gw.add_uint32(f"{A}.audio.hop_length", cfg["a_hop"])
    gw.add_uint32(f"{A}.audio.n_fft", cfg["a_nfft"])
    gw.add_uint32(f"{A}.audio.sample_rate", cfg["a_sr"])


def write_talker_metadata(gw, cfg, outtype):
    A = "qwen25omni-talker"
    gw.add_name(A)
    gw.add_file_type({
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    }.get(outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    gw.add_uint32(f"{A}.block_count", cfg["tk_layer"])
    gw.add_uint32(f"{A}.embedding_length", cfg["tk_embd"])
    gw.add_uint32(f"{A}.embedding_width", cfg["tk_embw"])
    gw.add_uint32(f"{A}.feed_forward_length", cfg["tk_ff"])
    gw.add_uint32(f"{A}.attention.head_count", cfg["tk_head"])
    gw.add_uint32(f"{A}.attention.head_count_kv", cfg["tk_kv_head"])
    gw.add_uint32(f"{A}.attention.key_length", cfg["tk_hd"])
    gw.add_float32(f"{A}.attention.layer_norm_rms_epsilon", float(cfg["tk_eps"]))
    gw.add_float32(f"{A}.rope.freq_base", float(cfg["tk_theta"]))
    gw.add_array(f"{A}.rope.mrope_section", [int(x) for x in cfg["tk_section"]])
    gw.add_uint32(f"{A}.codec_vocab_size", cfg["tk_vocab"])
    for key in ("codec_start", "codec_end", "codec_pad", "codec_mask",
                "text_start", "text_end", "text_pad"):
        gw.add_uint32(f"{A}.{key}", int(cfg[key]))

    gw.add_uint32(f"{A}.dit.depth", cfg["d_depth"])
    gw.add_uint32(f"{A}.dit.dim", cfg["d_dim"])
    gw.add_uint32(f"{A}.dit.emb_dim", cfg["d_emb"])
    gw.add_uint32(f"{A}.dit.head_count", cfg["d_head"])
    gw.add_uint32(f"{A}.dit.key_length", cfg["d_hd"])
    gw.add_uint32(f"{A}.dit.mel_dim", cfg["d_mel"])
    gw.add_uint32(f"{A}.dit.num_embeds", cfg["d_embeds"])
    gw.add_uint32(f"{A}.dit.repeats", cfg["d_repeats"])
    gw.add_uint32(f"{A}.dit.block_size", cfg["d_block_size"])
    gw.add_array(f"{A}.dit.look_ahead_layers", [int(x) for x in cfg["d_look_ahead"]])
    gw.add_array(f"{A}.dit.look_backward_layers", [int(x) for x in cfg["d_look_backward"]])
    gw.add_float32(f"{A}.dit.rope_freq_base", float(cfg["d_theta"]))
    gw.add_uint32(f"{A}.dit.max_position_embeddings", cfg["d_max_pos"])
    gw.add_array(f"{A}.dit.enc_channels", [int(x) for x in cfg["enc_channels"]])
    gw.add_array(f"{A}.dit.enc_kernel_sizes", [int(x) for x in cfg["enc_kernel_sizes"]])
    gw.add_array(f"{A}.dit.enc_dilations", [int(x) for x in cfg["enc_dilations"]])
    gw.add_uint32(f"{A}.dit.enc_dim", cfg["enc_dim"])
    gw.add_uint32(f"{A}.dit.enc_emb_dim", cfg["enc_emb_dim"])
    gw.add_uint32(f"{A}.dit.enc_emb_dim", cfg["enc_emb_dim"])
    gw.add_uint32(f"{A}.dit.enc_res2net_scale", cfg["enc_scale"])
    gw.add_uint32(f"{A}.dit.enc_se_channels", cfg["enc_se"])
    gw.add_uint32(f"{A}.dit.enc_attention_channels", cfg["enc_attn"])

    gw.add_array(f"{A}.bigvgan.upsample_rates", [int(x) for x in cfg["b_up_rates"]])
    gw.add_array(f"{A}.bigvgan.upsample_kernel_sizes", [int(x) for x in cfg["b_up_kernels"]])
    gw.add_uint32(f"{A}.bigvgan.channels", cfg["b_ch"])
    gw.add_array(f"{A}.bigvgan.resblock_kernel_sizes", [int(x) for x in cfg["b_rb_kernels"]])
    gw.add_array(f"{A}.bigvgan.resblock_dilation_sizes",
                 [int(x) for d in cfg["b_rb_dilations"] for x in d])
    gw.add_uint32(f"{A}.bigvgan.mel_dim", cfg["b_mel"])

    gw.add_array(f"{A}.speakers", [str(s) for s in SPEAKERS])


def convert_thinker(weights, gw, cfg, qtype, args):
    def linear(name, arr):
        arr = np.ascontiguousarray(arr)
        if qtype is not None and arr.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, name, _lib.quantize(arr, qtype), qtype)
        else:
            _lib.add_f16(gw, name, arr)

    def f32(name, arr):
        gw.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))

    # ------------------------------------------------------------- text trunk
    linear("t.embed.weight", weights.get("thinker.model.embed_tokens.weight"))
    if not cfg["tie_word_embeddings"]:
        linear("t.output.weight", weights.get("thinker.lm_head.weight"))
    for i in range(cfg["n_layer"]):
        h, b = f"thinker.model.layers.{i}", f"t.blk.{i}"
        f32(f"{b}.attn_norm.weight", weights.get(f"{h}.input_layernorm.weight"))
        f32(f"{b}.ffn_norm.weight", weights.get(f"{h}.post_attention_layernorm.weight"))
        linear(f"{b}.wq.weight", weights.get(f"{h}.self_attn.q_proj.weight"))
        f32(f"{b}.bq", weights.get(f"{h}.self_attn.q_proj.bias"))
        linear(f"{b}.wk.weight", weights.get(f"{h}.self_attn.k_proj.weight"))
        f32(f"{b}.bk", weights.get(f"{h}.self_attn.k_proj.bias"))
        linear(f"{b}.wv.weight", weights.get(f"{h}.self_attn.v_proj.weight"))
        f32(f"{b}.bv", weights.get(f"{h}.self_attn.v_proj.bias"))
        linear(f"{b}.wo.weight", weights.get(f"{h}.self_attn.o_proj.weight"))
        linear(f"{b}.w_gate.weight", weights.get(f"{h}.mlp.gate_proj.weight"))
        linear(f"{b}.w_up.weight", weights.get(f"{h}.mlp.up_proj.weight"))
        linear(f"{b}.w_down.weight", weights.get(f"{h}.mlp.down_proj.weight"))
        progress(i + 1, cfg["n_layer"], "text blocks")
    f32("t.norm.weight", weights.get("thinker.model.norm.weight"))

    # ---------------------------------------------------------- vision tower
    # identical layout to convert_qwen2vl.py; Omni arrives with separate
    # q/k/v projections which are fused so the proven graph runs unchanged
    conv = weights.get("thinker.visual.patch_embed.proj.weight")
    dim, cin, kt, kh, kw = conv.shape
    if (cin, kt, kh, kw) != (3, cfg["v_temporal"], cfg["v_patch"], cfg["v_patch"]):
        fail(f"unexpected patch_embed shape {conv.shape}")
    w2d = conv.reshape(dim, cin * kt * kh * kw)
    if args.vision_f32:
        gw.add_tensor("v.patch_embed.weight", np.ascontiguousarray(w2d, dtype=np.float32))
    elif qtype is not None and w2d.shape[-1] % 256 == 0:
        _lib.add_quantized(gw, "v.patch_embed.weight", _lib.quantize(w2d, qtype), qtype)
    else:
        _lib.add_f16(gw, "v.patch_embed.weight", w2d)

    for i in range(cfg["v_depth"]):
        h, b = f"thinker.visual.blocks.{i}", f"v.blk.{i}"
        f32(f"{b}.norm1.weight", weights.get(f"{h}.norm1.weight"))
        f32(f"{b}.norm2.weight", weights.get(f"{h}.norm2.weight"))
        q = weights.get(f"{h}.attn.q.weight")
        k = weights.get(f"{h}.attn.k.weight")
        v = weights.get(f"{h}.attn.v.weight")
        qkv = np.concatenate([q, k, v], axis=0)
        if args.vision_f32:
            gw.add_tensor(f"{b}.wqkv.weight", np.ascontiguousarray(qkv, dtype=np.float32))
        else:
            _lib.add_f16(gw, f"{b}.wqkv.weight", qkv)
        bq = weights.get(f"{h}.attn.q.bias")
        bk = weights.get(f"{h}.attn.k.bias")
        bv = weights.get(f"{h}.attn.v.bias")
        f32(f"{b}.wqkv.bias", np.concatenate([bq, bk, bv], axis=0))
        if args.vision_f32:
            gw.add_tensor(f"{b}.wo.weight", np.ascontiguousarray(weights.get(f"{h}.attn.proj.weight"), dtype=np.float32))
        else:
            _lib.add_f16(gw, f"{b}.wo.weight", weights.get(f"{h}.attn.proj.weight"))
        f32(f"{b}.wo.bias", weights.get(f"{h}.attn.proj.bias"))
        if args.vision_f32:
            for pre, hf in (("w_gate", "gate_proj"), ("w_up", "up_proj"), ("w_down", "down_proj")):
                gw.add_tensor(f"{b}.{pre}.weight",
                              np.ascontiguousarray(weights.get(f"{h}.mlp.{hf}.weight"), dtype=np.float32))
        else:
            _lib.add_f16(gw, f"{b}.w_gate.weight", weights.get(f"{h}.mlp.gate_proj.weight"))
            _lib.add_f16(gw, f"{b}.w_up.weight", weights.get(f"{h}.mlp.up_proj.weight"))
            _lib.add_f16(gw, f"{b}.w_down.weight", weights.get(f"{h}.mlp.down_proj.weight"))
        f32(f"{b}.w_gate.bias", weights.get(f"{h}.mlp.gate_proj.bias"))
        f32(f"{b}.w_up.bias", weights.get(f"{h}.mlp.up_proj.bias"))
        f32(f"{b}.w_down.bias", weights.get(f"{h}.mlp.down_proj.bias"))
        progress(i + 1, cfg["v_depth"], "vision blocks")

    f32("v.merger.ln.weight", weights.get("thinker.visual.merger.ln_q.weight"))
    if args.vision_f32:
        gw.add_tensor("v.merger.fc1.weight",
                      np.ascontiguousarray(weights.get("thinker.visual.merger.mlp.0.weight"), dtype=np.float32))
        gw.add_tensor("v.merger.fc2.weight",
                      np.ascontiguousarray(weights.get("thinker.visual.merger.mlp.2.weight"), dtype=np.float32))
    else:
        _lib.add_f16(gw, "v.merger.fc1.weight", weights.get("thinker.visual.merger.mlp.0.weight"))
        _lib.add_f16(gw, "v.merger.fc2.weight", weights.get("thinker.visual.merger.mlp.2.weight"))
    f32("v.merger.fc1.bias", weights.get("thinker.visual.merger.mlp.0.bias"))
    f32("v.merger.fc2.bias", weights.get("thinker.visual.merger.mlp.2.bias"))

    # ----------------------------------------------------------- audio tower
    pos = sinusoids(cfg["a_pos"], cfg["a_embd"])
    gw.add_tensor("a.position_embd.weight", pos)
    filt = mel_filterbank(args.src, cfg["a_mel"], cfg["a_nfft"], cfg["a_sr"])
    gw.add_tensor("a.mel_filters", filt)

    def conv1d(name, arr):
        # torch Conv1d weight [out, in, k]: gguf-py stores numpy C-order, which
        # ggml reads as ne=[k, in, out] - exactly the conv_1d kernel layout it
        # wants (torch bytes, ggml shape), so the array goes in unchanged
        _lib.add_f16(gw, name, arr)

    conv1d("a.conv1.weight", weights.get("thinker.audio_tower.conv1.weight"))
    f32("a.conv1.bias", weights.get("thinker.audio_tower.conv1.bias"))
    conv1d("a.conv2.weight", weights.get("thinker.audio_tower.conv2.weight"))
    f32("a.conv2.bias", weights.get("thinker.audio_tower.conv2.bias"))
    for i in range(cfg["a_layer"]):
        h, b = f"thinker.audio_tower.layers.{i}", f"a.blk.{i}"
        f32(f"{b}.ln_1.weight", weights.get(f"{h}.self_attn_layer_norm.weight"))
        f32(f"{b}.ln_1.bias", weights.get(f"{h}.self_attn_layer_norm.bias"))
        if qtype is not None:
            for pre, hf in (("wq", f"{h}.self_attn.q_proj.weight"),
                            ("wk", f"{h}.self_attn.k_proj.weight"),
                            ("wv", f"{h}.self_attn.v_proj.weight"),
                            ("wo", f"{h}.self_attn.out_proj.weight"),
                            ("w_gate", f"{h}.fc1.weight"),
                            ("w_down", f"{h}.fc2.weight")):
                linear(f"{b}.{pre}.weight", weights.get(hf))
        else:
            _lib.add_f16(gw, f"{b}.wq.weight", weights.get(f"{h}.self_attn.q_proj.weight"))
            _lib.add_f16(gw, f"{b}.wk.weight", weights.get(f"{h}.self_attn.k_proj.weight"))
            _lib.add_f16(gw, f"{b}.wv.weight", weights.get(f"{h}.self_attn.v_proj.weight"))
            _lib.add_f16(gw, f"{b}.wo.weight", weights.get(f"{h}.self_attn.out_proj.weight"))
            _lib.add_f16(gw, f"{b}.w_gate.weight", weights.get(f"{h}.fc1.weight"))
            _lib.add_f16(gw, f"{b}.w_down.weight", weights.get(f"{h}.fc2.weight"))
        f32(f"{b}.bq", weights.get(f"{h}.self_attn.q_proj.bias"))
        f32(f"{b}.bv", weights.get(f"{h}.self_attn.v_proj.bias"))
        f32(f"{b}.bo", weights.get(f"{h}.self_attn.out_proj.bias"))
        f32(f"{b}.b_gate", weights.get(f"{h}.fc1.bias"))
        f32(f"{b}.b_down", weights.get(f"{h}.fc2.bias"))
        f32(f"{b}.ln_2.weight", weights.get(f"{h}.final_layer_norm.weight"))
        f32(f"{b}.ln_2.bias", weights.get(f"{h}.final_layer_norm.bias"))
        progress(i + 1, cfg["a_layer"], "audio blocks")
    f32("a.ln_post.weight", weights.get("thinker.audio_tower.ln_post.weight"))
    f32("a.ln_post.bias", weights.get("thinker.audio_tower.ln_post.bias"))
    linear("a.proj.weight", weights.get("thinker.audio_tower.proj.weight"))
    f32("a.proj.bias", weights.get("thinker.audio_tower.proj.bias"))


def convert_talker(weights, gw, cfg, qtype, args):
    """talker LM + token2wav. All modules keep their HF name shape under short
    prefixes; the runtime graphs mirror the module trees one to one."""

    def lin(name, arr, prefix):
        # prefix: "q8" allows quantization, "f16" forces f16 (t2w runs fp32 at
        # runtime but f16 storage is fine for mul_mat), "q" honours the outtype
        arr = np.ascontiguousarray(arr)
        if qtype is not None and arr.shape[-1] % 256 == 0:
            _lib.add_quantized(gw, name, _lib.quantize(arr, qtype), qtype)
        else:
            _lib.add_f16(gw, name, arr)

    def f32(name, arr):
        gw.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))

    # ------------------------------------------------------------- talker LM
    lin("tk.embed.weight", weights.get("talker.model.embed_tokens.weight"), "q")
    for i in range(cfg["tk_layer"]):
        h, b = f"talker.model.layers.{i}", f"tk.blk.{i}"
        f32(f"{b}.attn_norm.weight", weights.get(f"{h}.input_layernorm.weight"))
        f32(f"{b}.ffn_norm.weight", weights.get(f"{h}.post_attention_layernorm.weight"))
        lin(f"{b}.wq.weight", weights.get(f"{h}.self_attn.q_proj.weight"), "q")
        f32(f"{b}.bq", weights.get(f"{h}.self_attn.q_proj.bias"))
        lin(f"{b}.wk.weight", weights.get(f"{h}.self_attn.k_proj.weight"), "q")
        f32(f"{b}.bk", weights.get(f"{h}.self_attn.k_proj.bias"))
        lin(f"{b}.wv.weight", weights.get(f"{h}.self_attn.v_proj.weight"), "q")
        f32(f"{b}.bv", weights.get(f"{h}.self_attn.v_proj.bias"))
        lin(f"{b}.wo.weight", weights.get(f"{h}.self_attn.o_proj.weight"), "q")
        lin(f"{b}.w_gate.weight", weights.get(f"{h}.mlp.gate_proj.weight"), "q")
        lin(f"{b}.w_up.weight", weights.get(f"{h}.mlp.up_proj.weight"), "q")
        lin(f"{b}.w_down.weight", weights.get(f"{h}.mlp.down_proj.weight"), "q")
        progress(i + 1, cfg["tk_layer"], "talker blocks")
    f32("tk.norm.weight", weights.get("talker.model.norm.weight"))
    lin("tk.to_proj.weight", weights.get("talker.thinker_to_talker_proj.weight"), "q")
    f32("tk.to_proj.bias", weights.get("talker.thinker_to_talker_proj.bias"))
    lin("tk.codec_head.weight", weights.get("talker.codec_head.weight"), "q")

    D = "token2wav.code2wav_dit_model"
    B = "token2wav.code2wav_bigvgan_model"

    # ------------------------------------------------------------- DiT
    f32("d.time_mlp.0.weight", weights.get(f"{D}.time_embed.time_mlp.0.weight"))
    f32("d.time_mlp.0.bias", weights.get(f"{D}.time_embed.time_mlp.0.bias"))
    f32("d.time_mlp.1.weight", weights.get(f"{D}.time_embed.time_mlp.2.weight"))
    f32("d.time_mlp.1.bias", weights.get(f"{D}.time_embed.time_mlp.2.bias"))
    lin("d.codec_embed.weight", weights.get(f"{D}.text_embed.codec_embed.weight"), "f16")
    lin("d.in_proj.weight", weights.get(f"{D}.input_embed.proj.weight"), "f16")
    f32("d.in_proj.bias", weights.get(f"{D}.input_embed.proj.bias"))
    f32("d.inv_freq", weights.get(f"{D}.rotary_embed.inv_freq"))
    for i in range(cfg["d_depth"]):
        h, b = f"{D}.transformer_blocks.{i}", f"d.blk.{i}"
        f32(f"{b}.attn_norm.weight", weights.get(f"{h}.attn_norm.linear.weight"))
        f32(f"{b}.attn_norm.bias", weights.get(f"{h}.attn_norm.linear.bias"))
        lin(f"{b}.wq.weight", weights.get(f"{h}.attn.to_q.weight"), "f16")
        f32(f"{b}.bq", weights.get(f"{h}.attn.to_q.bias"))
        lin(f"{b}.wk.weight", weights.get(f"{h}.attn.to_k.weight"), "f16")
        f32(f"{b}.bk", weights.get(f"{h}.attn.to_k.bias"))
        lin(f"{b}.wv.weight", weights.get(f"{h}.attn.to_v.weight"), "f16")
        f32(f"{b}.bv", weights.get(f"{h}.attn.to_v.bias"))
        lin(f"{b}.wo.weight", weights.get(f"{h}.attn.to_out.0.weight"), "f16")
        f32(f"{b}.bo", weights.get(f"{h}.attn.to_out.0.bias"))
        lin(f"{b}.w_gate.weight", weights.get(f"{h}.ff.ff.0.weight"), "f16")
        f32(f"{b}.b_gate", weights.get(f"{h}.ff.ff.0.bias"))
        lin(f"{b}.w_down.weight", weights.get(f"{h}.ff.ff.3.weight"), "f16")
        f32(f"{b}.b_down", weights.get(f"{h}.ff.ff.3.bias"))
        progress(i + 1, cfg["d_depth"], "dit blocks")
    f32("d.norm_out.weight", weights.get(f"{D}.norm_out.linear.weight"))
    f32("d.norm_out.bias", weights.get(f"{D}.norm_out.linear.bias"))
    lin("d.proj_out.weight", weights.get(f"{D}.proj_out.weight"), "f16")
    f32("d.proj_out.bias", weights.get(f"{D}.proj_out.bias"))

    # ------------------------------------------------------- ECAPA (CampPlus)
    S = f"{D}.input_embed.spk_encoder"
    lin("d.spk.tdnn0.weight", weights.get(f"{S}.blocks.0.conv.weight"), "f16")
    f32("d.spk.tdnn0.bias", weights.get(f"{S}.blocks.0.conv.bias"))
    for bi in range(1, len(cfg["enc_channels"]) - 1):
        h, b = f"{S}.blocks.{bi}", f"d.spk.blk.{bi - 1}"
        lin(f"{b}.tdnn1.weight", weights.get(f"{h}.tdnn1.conv.weight"), "f16")
        f32(f"{b}.tdnn1.bias", weights.get(f"{h}.tdnn1.conv.bias"))
        # Res2NetBlock: branch 0 is the identity path, so only scale-1 convs exist
        for si in range(cfg["enc_scale"] - 1):
            lin(f"{b}.res2net.{si}.weight", weights.get(f"{h}.res2net_block.blocks.{si}.conv.weight"), "f16")
            f32(f"{b}.res2net.{si}.bias", weights.get(f"{h}.res2net_block.blocks.{si}.conv.bias"))
        lin(f"{b}.tdnn2.weight", weights.get(f"{h}.tdnn2.conv.weight"), "f16")
        f32(f"{b}.tdnn2.bias", weights.get(f"{h}.tdnn2.conv.bias"))
        f32(f"{b}.se1.weight", weights.get(f"{h}.se_block.conv1.weight"))
        f32(f"{b}.se1.bias", weights.get(f"{h}.se_block.conv1.bias"))
        f32(f"{b}.se2.weight", weights.get(f"{h}.se_block.conv2.weight"))
        f32(f"{b}.se2.bias", weights.get(f"{h}.se_block.conv2.bias"))
    last = len(cfg["enc_channels"]) - 1
    lin("d.spk.mfa.weight", weights.get(f"{S}.mfa.conv.weight"), "f16")
    f32("d.spk.mfa.bias", weights.get(f"{S}.mfa.conv.bias"))
    f32("d.spk.asp.tdnn.weight", weights.get(f"{S}.asp.tdnn.conv.weight"))
    f32("d.spk.asp.tdnn.bias", weights.get(f"{S}.asp.tdnn.conv.bias"))
    f32("d.spk.asp.conv.weight", weights.get(f"{S}.asp.conv.weight"))
    f32("d.spk.asp.conv.bias", weights.get(f"{S}.asp.conv.bias"))
    lin("d.spk.fc.weight", weights.get(f"{S}.fc.weight"), "f16")
    f32("d.spk.fc.bias", weights.get(f"{S}.fc.bias"))

    # -------------------------------------------------------------- BigVGAN
    lin("bg.conv_pre.weight", weights.get(f"{B}.conv_pre.weight"), "f16")
    f32("bg.conv_pre.bias", weights.get(f"{B}.conv_pre.bias"))
    n_up = len(cfg["b_up_rates"])
    n_rb_per_up = len(cfg["b_rb_kernels"])
    rb = 0
    for u in range(n_up):
        lin(f"bg.up.{u}.weight", weights.get(f"{B}.ups.{u}.0.weight"), "f16")
        f32(f"bg.up.{u}.bias", weights.get(f"{B}.ups.{u}.0.bias"))
        for r in range(n_rb_per_up):
            for k in range(len(cfg["b_rb_dilations"])):
                lin(f"bg.rb.{rb}.c1.{k}.weight",
                    weights.get(f"{B}.resblocks.{rb}.convs1.{k}.weight"), "f16")
                f32(f"bg.rb.{rb}.c1.{k}.bias", weights.get(f"{B}.resblocks.{rb}.convs1.{k}.bias"))
                lin(f"bg.rb.{rb}.c2.{k}.weight",
                    weights.get(f"{B}.resblocks.{rb}.convs2.{k}.weight"), "f16")
                f32(f"bg.rb.{rb}.c2.{k}.bias", weights.get(f"{B}.resblocks.{rb}.convs2.{k}.bias"))
            rb += 1
        progress(u + 1, n_up, "bigvgan ups")
    lin("bg.conv_post.weight", weights.get(f"{B}.conv_post.weight"), "f16")
    if f"{B}.conv_post.bias" in weights:      # bias=False in the official config
        f32("bg.conv_post.bias", weights.get(f"{B}.conv_post.bias"))
    # snake activations: activation_post once after the ups stack, plus 6 per
    # resblock (one before each of convs1.{0,1,2} and convs2.{0,1,2})
    f32("bg.snake_pre.alpha", weights.get(f"{B}.activation_post.act.alpha"))
    f32("bg.snake_pre.beta", weights.get(f"{B}.activation_post.act.beta"))
    rb = 0
    for u in range(n_up):
        for r in range(n_rb_per_up):
            for k in range(len(cfg["b_rb_dilations"])):
                # HF pairs iteration k with activations[2k] (before convs1.k)
                # and activations[2k+1] (before convs2.k): acts1=[::2], acts2=[1::2]
                f32(f"bg.rb.{rb}.s1.{k}.alpha",
                    weights.get(f"{B}.resblocks.{rb}.activations.{2 * k}.act.alpha"))
                f32(f"bg.rb.{rb}.s1.{k}.beta",
                    weights.get(f"{B}.resblocks.{rb}.activations.{2 * k}.act.beta"))
                f32(f"bg.rb.{rb}.s2.{k}.alpha",
                    weights.get(f"{B}.resblocks.{rb}.activations.{2 * k + 1}.act.alpha"))
                f32(f"bg.rb.{rb}.s2.{k}.beta",
                    weights.get(f"{B}.resblocks.{rb}.activations.{2 * k + 1}.act.beta"))
            rb += 1

    # ------------------------------------------------------------ speakers
    import torch
    spk_path = os.path.join(args.src, "spk_dict.pt")
    if os.path.isfile(spk_path):
        d = torch.load(spk_path, weights_only=True)
        for name in SPEAKERS:
            if name not in d:
                warn(f"speaker {name} not in spk_dict.pt (has {list(d)})")
                continue
            e = d[name]
            gw.add_uint32(f"spk.{name}.bos_token", int(e["bos_token"]))
            f32(f"spk.{name}.cond", e["cond"].reshape(-1).numpy())
            f32(f"spk.{name}.ref_mel", e["ref_mel"].reshape(-1).numpy())
    else:
        warn(f"no spk_dict.pt in {args.src}; talker GGUF carries no speakers")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="f16", choices=sorted(OUTTYPES),
                    help="precision of the 2D weights (default f16)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: the source checkpoint dir)")
    ap.add_argument("--vision-f32", action="store_true",
                    help="keep the vision tower in F32 (same rationale as qwen2vl)")
    ap.add_argument("--no-talker", action="store_true", help="skip the talker GGUF")
    ap.add_argument("--skip-thinker", action="store_true", help="skip the thinker GGUF")
    ap.add_argument("-q", "--quiet", action="store_true")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR,
                                        patterns=PATTERNS)
    log(f"source: {src}")
    out_dir = args.out_dir or src
    cfg = read_config(src)
    qtype = OUTTYPES[args.outtype]
    log(f"text: {cfg['n_layer']}L d={cfg['n_embd']} nh={cfg['n_head']} nkv={cfg['n_kv_head']} "
        f"hd={cfg['head_dim']} ff={cfg['n_ff']} vocab={cfg['vocab']} theta={cfg['rope_theta']:g}")
    log(f"vision: {cfg['v_depth']}L d={cfg['v_embd']} window={cfg['v_window']} full={cfg['v_full_blocks']}")
    log(f"audio: {cfg['a_layer']}L d={cfg['a_embd']} nh={cfg['a_head']} ff={cfg['a_ff']} "
        f"mel={cfg['a_mel']} win={cfg['a_window']} out={cfg['a_out']}")
    log(f"talker: {cfg['tk_layer']}L d={cfg['tk_embd']} embedw={cfg['tk_embw']} "
        f"vocab={cfg['tk_vocab']}; dit {cfg['d_depth']}L d={cfg['d_dim']}")

    weights = _lib.TensorSource(src)
    log(f"{len(weights)} tensors in the checkpoint")

    if not args.skip_thinker:
        out = os.path.join(out_dir, f"qwen25omni-{args.outtype}.gguf")
        log(f"thinker -> {out}")
        gw = gguf.GGUFWriter(out, "qwen25omni")
        write_thinker_metadata(gw, cfg, args.outtype)
        convert_thinker(weights, gw, cfg, qtype, args)
        gw.write_header_to_file()
        gw.write_kv_data_to_file()
        gw.write_tensors_to_file()
        gw.close()
        names = {t.name for t in gguf.GGUFReader(out).tensors}
        required = ["t.embed.weight", "t.output.weight", "t.norm.weight",
                    "v.patch_embed.weight", "v.merger.fc2.weight",
                    "a.conv1.weight", "a.proj.weight", "a.mel_filters",
                    f"t.blk.{cfg['n_layer'] - 1}.w_down.weight",
                    f"v.blk.{cfg['v_depth'] - 1}.w_down.weight",
                    f"a.blk.{cfg['a_layer'] - 1}.w_down.weight"]
        missing = [n for n in required if n not in names]
        if missing:
            fail(f"written GGUF is missing {missing}")
        log(f"verified {len(names)} tensors, {os.path.getsize(out) / 1e9:.2f} GB -> {out}")

    if not args.no_talker:
        out = os.path.join(out_dir, f"qwen25omni-talker-{args.outtype}.gguf")
        log(f"talker -> {out}")
        gw = gguf.GGUFWriter(out, "qwen25omni-talker")
        write_talker_metadata(gw, cfg, args.outtype)
        convert_talker(weights, gw, cfg, qtype, args)
        gw.write_header_to_file()
        gw.write_kv_data_to_file()
        gw.write_tensors_to_file()
        gw.close()
        names = {t.name for t in gguf.GGUFReader(out).tensors}
        required = ["tk.embed.weight", "tk.codec_head.weight", "d.proj_out.weight",
                    "bg.conv_post.weight", f"tk.blk.{cfg['tk_layer'] - 1}.w_down.weight",
                    f"d.blk.{cfg['d_depth'] - 1}.w_down.weight",
                    f"bg.rb.{(len(cfg['b_up_rates']) * len(cfg['b_rb_kernels'])) - 1}.c2.0.weight"]
        missing = [n for n in required if n not in names]
        if missing:
            fail(f"written GGUF is missing {missing}")
        log(f"verified {len(names)} tensors, {os.path.getsize(out) / 1e9:.2f} GB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
