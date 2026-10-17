"""Write a tiny random-weight VideoJev checkpoint (HF layout) for smoke tests.

The real checkpoint is 9 GB; this one is a few MB and exercises every code path
of convert/convert_videojev.py and the engine's --score entry point: hybrid
text layers (linear + full attention), a GQA linear-attention layer (2 key / 4
value heads), the attention-only last layer, the vision tower + merger and the
cosine decision head.

  python scripts/gen_fake_videojev.py --out models/VideoJev-4B-fake

Nothing it writes is numerically meaningful; use it to prove that a change
still converts, loads and runs (then test quality against the real weights).
"""
import argparse
import json
import os
import struct

import numpy as np

# ---- shape of the tiny model -------------------------------------------------
H = 64            # text hidden
NH, NKV, HD = 4, 2, 16
FF = 128
NL = 2            # layer 0 = linear attention, layer 1 = full attention (last, attn-only)
ROPE = HD // 4    # partial_rotary_factor 0.25
MR = (ROPE // 2 // 2, ROPE // 2 - ROPE // 2 // 2, 0)   # mrope_section sums to ROPE/2

KHEADS, VHEADS, KDIM = 2, 4, 8
CONV_K = 4
QKV_ROWS = 2 * KHEADS * KDIM + VHEADS * KDIM
Z_ROWS = VHEADS * KDIM

VHID, VHEADS_V, VFF = 32, 2, 64
POS = 2304
HEAD_DIM = 32
VOCAB = 248320

IMAGE_ID, VIDEO_ID, VSTART, VEND, EOS = 248056, 248057, 248053, 248054, 248044
DECIDE, OPTION_END = 248077, 248078


def w(shape, scale=0.02):
    return (np.random.default_rng(0).standard_normal(shape) * scale).astype(np.float32)


def build():
    t = {}
    lm = "backbone.language_model"
    t[f"{lm}.embed_tokens.weight"] = w((VOCAB, H), 0.05)
    t[f"{lm}.norm.weight"] = np.zeros(H, dtype=np.float32)          # RMSNorm stores w+1

    for i in range(NL):
        p = f"{lm}.layers.{i}"
        t[f"{p}.input_layernorm.weight"] = np.zeros(H, dtype=np.float32)
        last = i == NL - 1
        if i == 0:
            la = f"{p}.linear_attn"
            t[f"{la}.in_proj_qkv.weight"] = w((QKV_ROWS, H))
            t[f"{la}.in_proj_z.weight"] = w((Z_ROWS, H))
            t[f"{la}.in_proj_b.weight"] = w((VHEADS, H))
            t[f"{la}.in_proj_a.weight"] = w((VHEADS, H))
            t[f"{la}.dt_bias"] = w((VHEADS,))
            t[f"{la}.A_log"] = w((VHEADS,))
            t[f"{la}.conv1d.weight"] = w((QKV_ROWS, 1, CONV_K))
            t[f"{la}.norm.weight"] = np.ones(KDIM, dtype=np.float32)  # gated norm: no +1
            t[f"{la}.out_proj.weight"] = w((H, Z_ROWS))
        else:
            sa = f"{p}.self_attn"
            t[f"{sa}.q_proj.weight"] = w((2 * NH * HD, H))     # per-head [q | gate]
            t[f"{sa}.k_proj.weight"] = w((NKV * HD, H))
            t[f"{sa}.v_proj.weight"] = w((NKV * HD, H))
            t[f"{sa}.o_proj.weight"] = w((H, NH * HD))
            t[f"{sa}.q_norm.weight"] = np.zeros(HD, dtype=np.float32)
            t[f"{sa}.k_norm.weight"] = np.zeros(HD, dtype=np.float32)
        if not last:
            t[f"{p}.post_attention_layernorm.weight"] = np.zeros(H, dtype=np.float32)
            t[f"{p}.mlp.gate_proj.weight"] = w((FF, H))
            t[f"{p}.mlp.up_proj.weight"] = w((FF, H))
            t[f"{p}.mlp.down_proj.weight"] = w((H, FF))

    vt = "backbone.visual"
    t[f"{vt}.patch_embed.proj.weight"] = w((VHID, 3, 2, 16, 16))
    t[f"{vt}.patch_embed.proj.bias"] = w((VHID,))
    t[f"{vt}.pos_embed.weight"] = w((POS, VHID))
    for i in range(1):
        p = f"{vt}.blocks.{i}"
        for name, shape in (("norm1.weight", (VHID,)), ("norm1.bias", (VHID,)),
                            ("norm2.weight", (VHID,)), ("norm2.bias", (VHID,)),
                            ("attn.qkv.weight", (3 * VHID, VHID)), ("attn.qkv.bias", (3 * VHID,)),
                            ("attn.proj.weight", (VHID, VHID)), ("attn.proj.bias", (VHID,)),
                            ("mlp.linear_fc1.weight", (VFF, VHID)), ("mlp.linear_fc1.bias", (VFF,)),
                            ("mlp.linear_fc2.weight", (VHID, VFF)), ("mlp.linear_fc2.bias", (VHID,))):
            t[f"{p}.{name}"] = w(shape)
    for name, shape in (("merger.norm.weight", (VHID,)), ("merger.norm.bias", (VHID,)),
                        ("merger.linear_fc1.weight", (VHID, VHID * 4)),
                        ("merger.linear_fc1.bias", (VHID,)),
                        ("merger.linear_fc2.weight", (H, VHID)), ("merger.linear_fc2.bias", (H,))):
        t[f"{vt}.{name}"] = w(shape)

    t["head.q.weight"] = w((HEAD_DIM, H))
    t["head.q.bias"] = w((HEAD_DIM,))
    t["head.k.weight"] = w((HEAD_DIM, H))
    t["head.k.bias"] = w((HEAD_DIM,))
    t["head.bias"] = np.zeros((), dtype=np.float32)
    return t


def config():
    return {
        "architectures": ["VideoJevForVideoUnderstanding"],
        "model_type": "videojev",
        "head_dim": HEAD_DIM,
        "head_logit_scale": 10.0,
        "decide_token_id": DECIDE,
        "option_end_token_id": OPTION_END,
        "tie_word_embeddings": False,
        "decoder": {"postprocess": "round_repair_clamp_v1", "smooth": 1, "threshold": 0.25},
        "backbone_config": {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "image_token_id": IMAGE_ID, "video_token_id": VIDEO_ID,
            "vision_start_token_id": VSTART, "vision_end_token_id": VEND,
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": NL, "hidden_size": H,
                "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
                "intermediate_size": FF, "vocab_size": VOCAB, "rms_norm_eps": 1e-6,
                "max_position_embeddings": 4096, "eos_token_id": EOS,
                "attn_output_gate": True, "full_attention_interval": 2,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_conv_kernel_dim": CONV_K, "linear_key_head_dim": KDIM,
                "linear_num_key_heads": KHEADS, "linear_num_value_heads": VHEADS,
                "linear_value_head_dim": KDIM,
                "rope_parameters": {"mrope_interleaved": True, "mrope_section": list(MR),
                                    "partial_rotary_factor": 0.25, "rope_theta": 10000.0},
            },
            "vision_config": {
                "model_type": "qwen3_5_vision", "depth": 1, "hidden_size": VHID,
                "num_heads": VHEADS_V, "intermediate_size": VFF, "patch_size": 16,
                "temporal_patch_size": 2, "spatial_merge_size": 2,
                "num_position_embeddings": POS, "out_hidden_size": H,
                "deepstack_visual_indexes": [],
            },
        },
    }


def write_safetensors(path, tensors):
    """Minimal writer (no torch/safetensors dependency)."""
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        header[name] = {"dtype": "F32", "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes]}
        offset += arr.nbytes
        blobs.append(arr.tobytes())
    meta = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(meta)))
        f.write(meta)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="models/VideoJev-4B-fake")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config(), f, indent=1)
    tensors = build()
    write_safetensors(os.path.join(args.out, "model.safetensors"), tensors)
    print(f"{args.out}: {len(tensors)} tensors, "
          f"{sum(a.nbytes for a in tensors.values()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
