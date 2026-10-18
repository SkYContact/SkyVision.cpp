"""Convert moondream3.1-9B-A2B vision tower (SigLIP + projection) to a standalone GGUF.

Usage:
  python convert/convert_vision.py [--src DIR] [-o moondream3.1-vision.gguf]

Source keys (HF model.safetensors)                    -> GGUF keys
--------------------------------------------------------+----------------------------------------
vision_encoder.encoder.model.visual.patch_embed.linear.* -> patch_emb.{weight,bias}
vision_encoder.encoder.model.visual.pos_embed [1,729,1152] -> pos_emb.weight (kept as-is; ggml ne becomes [1152,729])
vision_encoder.encoder.model.visual.blocks.N.norm1.*     -> blk.N.ln1.{weight,bias}
vision_encoder.encoder.model.visual.blocks.N.attn.qkv.*  -> blk.N.attn_qkv.{weight,bias}
vision_encoder.encoder.model.visual.blocks.N.attn.proj.* -> blk.N.attn_out.{weight,bias}
vision_encoder.encoder.model.visual.blocks.N.norm2.*     -> blk.N.ln2.{weight,bias}
vision_encoder.encoder.model.visual.blocks.N.mlp.fc1.*   -> blk.N.ffn_up.{weight,bias}
vision_encoder.encoder.model.visual.blocks.N.mlp.fc2.*   -> blk.N.ffn_down.{weight,bias}
vision_encoder.encoder.model.visual.norm.*               -> post_ln.{weight,bias}
vision_encoder.projection.mlp.fc1.*                      -> proj_mlp.fc1.{weight,bias}
vision_encoder.projection.mlp.fc2.*                      -> proj_mlp.fc2.{weight,bias}

All tensors are BF16 in the checkpoint -> F16 here (same policy as the text converter).
pos_emb is stored pre-transposed [1152,729] so the C++ graph can add it to the
[enc_dim, n_patch] activation without a runtime permute (ggml permute+add traps).
"""
import argparse
import json
import struct

import numpy as np
import gguf

import hf_source


class SfReader:
    """Minimal torch-free BF16 safetensors reader (the torch lazy-loader segfaults
    under Windows commit pressure when the big text model is also resident)."""

    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.hdr = json.loads(self.f.read(n))
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def get_f32(self, name):
        info = self.hdr[name]
        assert info["dtype"] == "BF16", (name, info["dtype"])
        s, e = info["data_offsets"]
        bits = self.mm[self.base + s:self.base + e].view(np.uint16)
        f32 = (bits.astype(np.uint32) << 16).view(np.float32)
        # keep torch order (out, in): gguf-py stores ne = numpy.shape[::-1],
        # so weights end up ne=[in,out] — exactly what ggml_mul_mat wants.
        return f32.reshape(info["shape"])

    def get_f16(self, name):
        return self.get_f32(name).astype(np.float16)

# VisionConfig defaults (moondream/torch/config.py)
ENC_DIM = 1152
PATCH = 14
N_LAYERS = 27
FF_DIM = 4304
N_HEADS = 16
PROJ_OUT = 2048
CROP = 378
IN_CH = 3
MAX_CROPS = 12
OVERLAP = 4
PROJ_INNER = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="moondream3.1-vision.gguf")
    hf_source.add_src_argument(ap)
    args = ap.parse_args()

    src = hf_source.find_model_safetensors(args.src)
    print(f"source: {src}")
    print(f"output: {args.out}")
    w = SfReader(src)

    gw = gguf.GGUFWriter(args.out, "moondream3-vision")   # GGUFWriter writes general.architecture
    gw.add_name("moondream3.1-9B-A2B-vision")
    gw.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    # Same gguf-py Keys trap as convert_hf_to_gguf.py: the constants are
    # "{arch}.*" format templates; without .format() they are written as a
    # literal "{arch}.block_count" key (the in-tree vision gguf carries them).
    def key(t):
        return t.format(arch="moondream3-vision")

    gw.add_uint32(key(gguf.Keys.LLM.BLOCK_COUNT), N_LAYERS)
    gw.add_uint32(key(gguf.Keys.LLM.EMBEDDING_LENGTH), ENC_DIM)
    gw.add_uint32(key(gguf.Keys.LLM.FEED_FORWARD_LENGTH), FF_DIM)
    gw.add_uint32(key(gguf.Keys.Attention.HEAD_COUNT), N_HEADS)
    # vision-specific keys (custom)
    v = "moondream3.vision."
    gw.add_uint32(v + "enc_dim", ENC_DIM)
    gw.add_uint32(v + "patch_size", PATCH)
    gw.add_uint32(v + "enc_n_layers", N_LAYERS)
    gw.add_uint32(v + "enc_ff_dim", FF_DIM)
    gw.add_uint32(v + "enc_n_heads", N_HEADS)
    gw.add_uint32(v + "proj_out_dim", PROJ_OUT)
    gw.add_uint32(v + "crop_size", CROP)
    gw.add_uint32(v + "in_channels", IN_CH)
    gw.add_uint32(v + "max_crops", MAX_CROPS)
    gw.add_uint32(v + "overlap_margin", OVERLAP)
    gw.add_uint32(v + "proj_inner_dim", PROJ_INNER)

    P = "vision_encoder.encoder.model.visual."

    def get_f32(name):
        return w.get_f32(name)

    print("patch_emb / pos_emb / post_ln ...")
    # patch_emb.weight is stored as F32: its F16 row stride would be 588*2=1176
    # bytes, 16-byte UNALIGNED, which corrupts ggml's vectorized F16 matmul.
    gw.add_tensor("patch_emb.weight", get_f32(P + "patch_embed.linear.weight"))
    gw.add_tensor("patch_emb.bias", w.get_f16(P + "patch_embed.linear.bias"))
    # gguf-py stores ne = numpy.shape[::-1]; to get ggml ne=[1152,729] (channel
    # rows, matching the [enc_dim, n_patch] activation) the numpy shape must be
    # [729,1152] -- i.e. the checkpoint layout, NO transpose needed.
    # safetensors [1,729,1152]; squeeze -> numpy (729,1152) which gguf-py stores
    # as ne=[1152,729]: channel rows, directly addable to the [enc_dim, 729] activation.
    pos = w.get_f16(P + "pos_embed").squeeze()
    gw.add_tensor("pos_emb.weight", np.ascontiguousarray(pos))
    gw.add_tensor("post_ln.weight", w.get_f16(P + "norm.weight"))
    gw.add_tensor("post_ln.bias", w.get_f16(P + "norm.bias"))

    for i in range(N_LAYERS):
        b = f"{P}blocks.{i}."
        print(f"layer {i}...")
        gw.add_tensor(f"blk.{i}.ln1.weight", w.get_f16(b + "norm1.weight"))
        gw.add_tensor(f"blk.{i}.ln1.bias", w.get_f16(b + "norm1.bias"))
        gw.add_tensor(f"blk.{i}.attn_qkv.weight", w.get_f16(b + "attn.qkv.weight"))
        gw.add_tensor(f"blk.{i}.attn_qkv.bias", w.get_f16(b + "attn.qkv.bias"))
        gw.add_tensor(f"blk.{i}.attn_out.weight", w.get_f16(b + "attn.proj.weight"))
        gw.add_tensor(f"blk.{i}.attn_out.bias", w.get_f16(b + "attn.proj.bias"))
        gw.add_tensor(f"blk.{i}.ln2.weight", w.get_f16(b + "norm2.weight"))
        gw.add_tensor(f"blk.{i}.ln2.bias", w.get_f16(b + "norm2.bias"))
        gw.add_tensor(f"blk.{i}.ffn_up.weight", w.get_f16(b + "mlp.fc1.weight"))
        gw.add_tensor(f"blk.{i}.ffn_up.bias", w.get_f16(b + "mlp.fc1.bias"))
        gw.add_tensor(f"blk.{i}.ffn_down.weight", w.get_f16(b + "mlp.fc2.weight"))
        gw.add_tensor(f"blk.{i}.ffn_down.bias", w.get_f16(b + "mlp.fc2.bias"))

    print("projection ...")
    pr = "vision_encoder.projection.mlp."
    gw.add_tensor("proj_mlp.fc1.weight", w.get_f16(pr + "fc1.weight"))
    gw.add_tensor("proj_mlp.fc1.bias", w.get_f16(pr + "fc1.bias"))
    gw.add_tensor("proj_mlp.fc2.weight", w.get_f16(pr + "fc2.weight"))
    gw.add_tensor("proj_mlp.fc2.bias", w.get_f16(pr + "fc2.bias"))

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print("done:", args.out)


if __name__ == "__main__":
    main()
