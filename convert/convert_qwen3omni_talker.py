"""Convert Qwen3-Omni-30B-A3B-Instruct talker (speech output stack) to GGUF.

Usage::

  python convert/convert_qwen3omni_talker.py [--src DIR] [-o OUT.gguf]
                                            [--outtype q8_0|f16]

Emits a standalone file separate from the thinker GGUF (same convention as
convert_qwen25omni.py: qwen25omni-talker-q8_0.gguf next to the main file)::

  qwen3omni-talker-q8_0.gguf

Covers talker.model (20L MoE codec LM), talker.text_projection /
talker.hidden_projection, talker.code_predictor (5L dense LM, 15 input
embeddings + 15 untied lm heads) and code2wav (8L pre-transformer with
LayerScale, ConvNeXt/TransConv upsample, causal conv decoder).

Layout (naming verified against model.safetensors.index.json, see the
"Checkpoint facts" block below; all tensor names are the index truth):

* tk.*   talker codec LM.  token_embd = talker.model.codec_embedding
         (talker.codec_head is NOT tied to it - maxdiff 8.1 - so both are
         written).  Expert stacks follow the thinker convention: per-expert
         HF [out, in] arrays concatenated on the slow axis
         (ne = [in, out, n_exp]).  down_exps (in=384) and the shared-expert
         down (in=768) are not multiples of 256 so they stay F16 even for
         --outtype q8_0; gate/up stacks (in=1024) quantize.
         Name collision note: the scalar shared-expert gate keeps the
         .ffn_shared_gate.weight name, the shared expert MLP is
         .ffn_shared_{gate,up,down}_proj.weight.
* tk.text_projection.* / tk.hidden_projection.*   ResizeMLP (fc1/fc2)
* cp.*   code predictor: cp.token_embd.{k} (= model.codec_embedding.{k}) and
         cp.head.{k} (= lm_head.{k}); the 15 heads are untied (config
         tie_word_embeddings=false, and head k != embedding k, maxdiff 0.15).
* c2w.*  code2wav: code_embedding [32768=2048*16, 1024] F16; pre_transformer
         layers renamed to c2w.blk.{i} (attn_norm/attn_q/.../ffn_down plus
         attn_layer_scale/ffn_layer_scale F32 = LayerScale); upsample and
         decoder conv trees keep their HF module names verbatim under c2w.
"""
import argparse
import json
import os
import sys

import numpy as np
import gguf

import _lib
import hf_source
from _lib import log, warn, fail, progress

_lib.torch = None   # torch 2.11 win: storage __getitem__ access-violates on this
                    # checkpoint's mmap; the numpy bf16 decode path is exact and
                    # fast enough

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
ENV_VAR = "QWEN3OMNI_SRC"
PATTERNS = ("model.safetensors.index.json",)

OUTTYPES = {"f16": None, "q8_0": gguf.GGMLQuantizationType.Q8_0}

DEFAULT_OUT_DIR = "E:/SkyVision-weights/qwen3-omni"

N_CODE_GROUPS = 15          # cp.token_embd.0 .. 14 / cp.head.0 .. 14


def read_config(src_dir):
    with open(os.path.join(src_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["talker_config"], cfg["code2wav_config"], cfg


def write_metadata(gw, tk, c2w, root, outtype):
    gw.add_architecture()
    gw.add_name("Qwen3-Omni-30B-A3B-Instruct-talker")
    gw.add_file_type({"q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}.get(
        outtype, gguf.LlamaFileType.ALL_F32))
    gw.add_uint32("general.quantization_version", 2)

    text = tk["text_config"]
    cp = tk["code_predictor_config"]
    A = "qwen3omni.talker"
    gw.add_uint32(f"{A}.block_count", text["num_hidden_layers"])
    gw.add_uint32(f"{A}.embedding_length", text["hidden_size"])
    gw.add_uint32(f"{A}.attention.head_count", text["num_attention_heads"])
    gw.add_uint32(f"{A}.attention.head_count_kv", text["num_key_value_heads"])
    gw.add_uint32(f"{A}.attention.key_length", text["head_dim"])
    gw.add_uint32(f"{A}.attention.value_length", text["head_dim"])
    gw.add_uint32(f"{A}.vocab_size", text["vocab_size"])
    gw.add_uint32(f"{A}.context_length", text["max_position_embeddings"])
    gw.add_float32(f"{A}.attention.layer_norm_rms_epsilon", text["rms_norm_eps"])
    gw.add_float32(f"{A}.rope.freq_base", float(text["rope_theta"]))
    for i, s in enumerate(text["rope_scaling"]["mrope_section"]):
        gw.add_uint32(f"{A}.rope.mrope_section_{i}", s)
    gw.add_uint32(f"{A}.expert_count", text["num_experts"])
    gw.add_uint32(f"{A}.expert_used_count", text["num_experts_per_tok"])
    gw.add_uint32(f"{A}.expert_feed_forward_length", text["moe_intermediate_size"])
    gw.add_uint32(f"{A}.shared_feed_forward_length", text["shared_expert_intermediate_size"])
    gw.add_bool(f"{A}.expert_norm_topk_prob", text.get("norm_topk_prob", True))
    gw.add_uint32(f"{A}.accept_hidden_layer", tk["accept_hidden_layer"])
    gw.add_uint32(f"{A}.num_code_groups", tk["num_code_groups"])
    gw.add_uint32(f"{A}.thinker_hidden_size", tk["thinker_hidden_size"])

    gw.add_uint32("qwen3omni.codec.eos_token_id", tk["codec_eos_token_id"])
    gw.add_uint32("qwen3omni.codec.bos_token_id", tk["codec_bos_id"])
    gw.add_uint32("qwen3omni.codec.pad_token_id", tk["codec_pad_id"])
    gw.add_uint32("qwen3omni.codec.nothink_token_id", tk["codec_nothink_id"])
    gw.add_uint32("qwen3omni.codec.think_bos_token_id", tk["codec_think_bos_id"])
    gw.add_uint32("qwen3omni.codec.think_eos_token_id", tk["codec_think_eos_id"])

    gw.add_uint32("qwen3omni.tts.bos_token_id", root["tts_bos_token_id"])
    gw.add_uint32("qwen3omni.tts.eos_token_id", root["tts_eos_token_id"])
    gw.add_uint32("qwen3omni.tts.pad_token_id", root["tts_pad_token_id"])

    for name, sid in tk["speaker_id"].items():
        gw.add_uint32(f"qwen3omni.speaker.{name}", sid)

    C = "qwen3omni.cp"
    gw.add_uint32(f"{C}.block_count", cp["num_hidden_layers"])
    gw.add_uint32(f"{C}.embedding_length", cp["hidden_size"])
    gw.add_uint32(f"{C}.feed_forward_length", cp["intermediate_size"])
    gw.add_uint32(f"{C}.attention.head_count", cp["num_attention_heads"])
    gw.add_uint32(f"{C}.attention.head_count_kv", cp["num_key_value_heads"])
    gw.add_uint32(f"{C}.attention.key_length", cp["head_dim"])
    gw.add_uint32(f"{C}.attention.value_length", cp["head_dim"])
    gw.add_uint32(f"{C}.vocab_size", cp["vocab_size"])
    gw.add_uint32(f"{C}.num_code_groups", N_CODE_GROUPS + 1)   # 15 heads
    gw.add_float32(f"{C}.attention.layer_norm_rms_epsilon", cp["rms_norm_eps"])
    gw.add_float32(f"{C}.rope.freq_base", float(cp["rope_theta"]))

    W = "qwen3omni.c2w"
    gw.add_uint32(f"{W}.codebook_size", c2w["codebook_size"])
    gw.add_uint32(f"{W}.num_quantizers", c2w["num_quantizers"])
    gw.add_uint32(f"{W}.hidden_size", c2w["hidden_size"])
    gw.add_uint32(f"{W}.intermediate_size", c2w["intermediate_size"])
    gw.add_uint32(f"{W}.num_hidden_layers", c2w["num_hidden_layers"])
    gw.add_uint32(f"{W}.attention.head_count", c2w["num_attention_heads"])
    gw.add_uint32(f"{W}.attention.head_count_kv", c2w["num_key_value_heads"])
    gw.add_uint32(f"{W}.attention.key_length", c2w["hidden_size"] // c2w["num_attention_heads"])
    gw.add_uint32(f"{W}.sliding_window", c2w["sliding_window"])
    gw.add_uint32(f"{W}.decoder_dim", c2w["decoder_dim"])
    gw.add_uint32(f"{W}.max_position_embeddings", c2w["max_position_embeddings"])
    gw.add_float32(f"{W}.attention.layer_norm_rms_epsilon", c2w["rms_norm_eps"])
    gw.add_float32(f"{W}.rope.freq_base", float(c2w["rope_theta"]))
    gw.add_float32(f"{W}.layer_scale_initial_scale", c2w["layer_scale_initial_scale"])
    for i, r in enumerate(c2w["upsample_rates"]):
        gw.add_uint32(f"{W}.upsample_rate_{i}", r)
    for i, r in enumerate(c2w["upsampling_ratios"]):
        gw.add_uint32(f"{W}.upsampling_ratio_{i}", r)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outtype", default="q8_0", choices=sorted(OUTTYPES))
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("-q", "--quiet", action="store_true")
    hf_source.add_src_argument(ap, model_id=MODEL_ID, env_var=ENV_VAR)
    args = ap.parse_args()

    out = args.out or os.path.join(DEFAULT_OUT_DIR, f"qwen3omni-talker-{args.outtype}.gguf")
    src = hf_source.find_checkpoint_dir(args.src, model_id=MODEL_ID, env_var=ENV_VAR, patterns=PATTERNS)
    log(f"source: {src}")
    log(f"output: {out} ({args.outtype})")

    qtype = OUTTYPES[args.outtype]
    tk, c2w, root = read_config(src)
    text, cp = tk["text_config"], tk["code_predictor_config"]
    n_l, n_exp = text["num_hidden_layers"], text["num_experts"]
    moe_i = text["moe_intermediate_size"]
    log(f"talker: {n_l}L d={text['hidden_size']} moe={n_exp}x{text['num_experts_per_tok']} "
        f"I={moe_i} vocab={text['vocab_size']}")
    log(f"code predictor: {cp['num_hidden_layers']}L d={cp['hidden_size']} "
        f"vocab={cp['vocab_size']} heads={cp['num_attention_heads']}/{cp['num_key_value_heads']}kv")
    log(f"code2wav: {c2w['num_hidden_layers']}L d={c2w['hidden_size']} "
        f"cb={c2w['codebook_size']}x{c2w['num_quantizers']} win={c2w['sliding_window']}")

    def quantize_text(arr):
        if qtype is None:
            return None
        if arr.shape[-1] % 256:
            warn(f"row length {arr.shape[-1]} not multiple of 256, keeping F16: {arr.shape}")
            return None
        return _lib.quantize(arr, qtype)

    weights = _lib.TensorSource(src)
    log(f"{len(weights)} tensors in the checkpoint (talker subset will be used)")

    gw = gguf.GGUFWriter(out, "qwen3omni-talker")
    write_metadata(gw, tk, c2w, root, args.outtype)

    stats = {"quantized": 0, "f16": 0, "f32": 0}

    def add_tensor(name, hf_name, quantize=False, f32=False):
        arr = weights.get(hf_name)
        if f32:
            gw.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
            stats["f32"] += 1
            return
        blocks = quantize_text(arr) if quantize else None
        if blocks is None:
            _lib.add_f16(gw, name, arr)
            stats["f16"] += 1
        else:
            _lib.add_quantized(gw, name, blocks, qtype)
            stats["quantized"] += 1

    def add_exps(name_fmt, p, leaf, n_in):
        """Stack n_experts per-expert HF [out, in] arrays and write one
        tensor. Byte order: expert-major concatenation of each expert's flat
        ne=[in,out] block (= np.stack axis 0, C order), thinker convention.
        Rows not a multiple of 256 (down_proj, in=384) stay F16."""
        parts = [weights.get(f"{p}.experts.{e}.{leaf}.weight") for e in range(n_exp)]
        arr = np.stack(parts, axis=0).astype(np.float32)          # [E, out, in]
        assert arr.shape[-1] == n_in
        if qtype is not None and n_in % 256 == 0:
            flat = arr.reshape(-1, n_in)
            q = _lib.quantize(flat, qtype)
            gw.add_tensor(name_fmt, q, raw_shape=(n_exp, arr.shape[1], q.shape[-1]), raw_dtype=qtype)
            stats["quantized"] += 1
        else:
            gw.add_tensor(name_fmt, np.ascontiguousarray(arr.astype(np.float16)),
                          raw_shape=arr.shape)
            stats["f16"] += 1

    # ------------------------------------------------------------ talker LM
    tm = "talker.model"
    log("talker codec LM")
    # embeddings stay F16 per the talker policy (norms/router are F32)
    add_tensor("tk.token_embd.weight", f"{tm}.codec_embedding.weight")
    add_tensor("tk.codec_head.weight", "talker.codec_head.weight")
    add_tensor("tk.output_norm.weight", f"{tm}.norm.weight")
    for i in range(n_l):
        p, b = f"{tm}.layers.{i}", f"tk.blk.{i}"
        add_tensor(f"{b}.attn_norm.weight", f"{p}.input_layernorm.weight", f32=True)
        add_tensor(f"{b}.attn_q.weight", f"{p}.self_attn.q_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_k.weight", f"{p}.self_attn.k_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_v.weight", f"{p}.self_attn.v_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_output.weight", f"{p}.self_attn.o_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_q_norm.weight", f"{p}.self_attn.q_norm.weight", f32=True)
        add_tensor(f"{b}.attn_k_norm.weight", f"{p}.self_attn.k_norm.weight", f32=True)
        add_tensor(f"{b}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight", f32=True)
        add_tensor(f"{b}.ffn_router.weight", f"{p}.mlp.gate.weight", f32=True)
        add_exps(f"{b}.ffn_gate_exps.weight", f"{p}.mlp", "gate_proj", text["hidden_size"])
        add_exps(f"{b}.ffn_up_exps.weight", f"{p}.mlp", "up_proj", text["hidden_size"])
        add_exps(f"{b}.ffn_down_exps.weight", f"{p}.mlp", "down_proj", moe_i)
        add_tensor(f"{b}.ffn_shared_gate_proj.weight", f"{p}.mlp.shared_expert.gate_proj.weight", quantize=True)
        add_tensor(f"{b}.ffn_shared_up_proj.weight", f"{p}.mlp.shared_expert.up_proj.weight", quantize=True)
        add_tensor(f"{b}.ffn_shared_down_proj.weight", f"{p}.mlp.shared_expert.down_proj.weight")
        add_tensor(f"{b}.ffn_shared_gate.weight", f"{p}.mlp.shared_expert_gate.weight", f32=True)
        if not args.quiet:
            progress(i, n_l, "talker layer", every=4)

    # ------------------------------------------------- thinker -> talker MLPs
    log("projections")
    for pre, hf in (("tk.text_projection", "talker.text_projection"),
                    ("tk.hidden_projection", "talker.hidden_projection")):
        add_tensor(f"{pre}.fc1.weight", f"{hf}.linear_fc1.weight", quantize=True)
        add_tensor(f"{pre}.fc1.bias", f"{hf}.linear_fc1.bias", f32=True)
        add_tensor(f"{pre}.fc2.weight", f"{hf}.linear_fc2.weight", quantize=True)
        add_tensor(f"{pre}.fc2.bias", f"{hf}.linear_fc2.bias", f32=True)

    # -------------------------------------------------------- code predictor
    pcm = "talker.code_predictor.model"
    log("code predictor")
    for k in range(N_CODE_GROUPS):
        add_tensor(f"cp.token_embd.{k}.weight", f"{pcm}.codec_embedding.{k}.weight")
        add_tensor(f"cp.head.{k}.weight", f"talker.code_predictor.lm_head.{k}.weight")
    add_tensor("cp.output_norm.weight", f"{pcm}.norm.weight", f32=True)
    for i in range(cp["num_hidden_layers"]):
        p, b = f"{pcm}.layers.{i}", f"cp.blk.{i}"
        add_tensor(f"{b}.attn_norm.weight", f"{p}.input_layernorm.weight", f32=True)
        add_tensor(f"{b}.attn_q.weight", f"{p}.self_attn.q_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_k.weight", f"{p}.self_attn.k_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_v.weight", f"{p}.self_attn.v_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_output.weight", f"{p}.self_attn.o_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_q_norm.weight", f"{p}.self_attn.q_norm.weight", f32=True)
        add_tensor(f"{b}.attn_k_norm.weight", f"{p}.self_attn.k_norm.weight", f32=True)
        add_tensor(f"{b}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight", f32=True)
        add_tensor(f"{b}.ffn_gate.weight", f"{p}.mlp.gate_proj.weight", quantize=True)
        add_tensor(f"{b}.ffn_up.weight", f"{p}.mlp.up_proj.weight", quantize=True)
        add_tensor(f"{b}.ffn_down.weight", f"{p}.mlp.down_proj.weight", quantize=True)
        if not args.quiet:
            progress(i, cp["num_hidden_layers"], "code predictor layer")

    # ------------------------------------------------------------- code2wav
    log("code2wav")
    add_tensor("c2w.code_embedding.weight", "code2wav.code_embedding.weight")
    pt = "code2wav.pre_transformer.layers"
    for i in range(c2w["num_hidden_layers"]):
        p, b = f"{pt}.{i}", f"c2w.blk.{i}"
        add_tensor(f"{b}.attn_norm.weight", f"{p}.input_layernorm.weight", f32=True)
        add_tensor(f"{b}.attn_q.weight", f"{p}.self_attn.q_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_k.weight", f"{p}.self_attn.k_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_v.weight", f"{p}.self_attn.v_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_output.weight", f"{p}.self_attn.o_proj.weight", quantize=True)
        add_tensor(f"{b}.ffn_norm.weight", f"{p}.post_attention_layernorm.weight", f32=True)
        add_tensor(f"{b}.ffn_gate.weight", f"{p}.mlp.gate_proj.weight", quantize=True)
        add_tensor(f"{b}.ffn_up.weight", f"{p}.mlp.up_proj.weight", quantize=True)
        add_tensor(f"{b}.ffn_down.weight", f"{p}.mlp.down_proj.weight", quantize=True)
        add_tensor(f"{b}.attn_layer_scale.weight", f"{p}.self_attn_layer_scale.scale", f32=True)
        add_tensor(f"{b}.ffn_layer_scale.weight", f"{p}.mlp_layer_scale.scale", f32=True)
        if not args.quiet:
            progress(i, c2w["num_hidden_layers"], "code2wav layer")
    add_tensor("c2w.output_norm.weight", "code2wav.pre_transformer.norm.weight", f32=True)

    # upsample (TransConvNet + ConvNeXt) and decoder: HF module tree verbatim
    # under c2w.; convs/snake/layer params F16/F32, point-wise linears quantize
    c2w_names = [n for n in weights.names()
                 if n.startswith("code2wav.upsample.") or n.startswith("code2wav.decoder.")]
    c2w_names.sort(key=lambda s: [int(x) if x.isdigit() else x for x in s.split(".")])
    for hf_name in c2w_names:
        leaf = hf_name.rsplit(".", 1)[-1]
        name = "c2w." + hf_name[len("code2wav."):]
        is_linear = (".pwconv1." in name or ".pwconv2." in name) and leaf == "weight"
        if leaf == "weight" and is_linear:
            add_tensor(name, hf_name, quantize=True)
        elif leaf in ("alpha", "beta", "gamma", "bias", "scale"):
            add_tensor(name, hf_name, f32=True)   # snake/layer-scale/bias
        else:
            add_tensor(name, hf_name)             # conv weights F16
    log(f"code2wav conv tree: {len(c2w_names)} tensors")

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()

    log(f"done: {out} ({os.path.getsize(out) / 1e9:.2f} GB, "
        f"{stats['quantized']} quantized / {stats['f16']} f16 / {stats['f32']} f32 tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
