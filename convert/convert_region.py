"""Convert the moondream3.1 region head (detect/point) to a standalone GGUF.

Usage:
  python convert/convert_region.py [--src DIR] [-o moondream3.1-region.gguf]

Source keys (HF model.safetensors)         -> GGUF keys
-------------------------------------------+--------------------------------
region_model.coordinate_features.weight [128,1]  -> coord_features.weight (F32, flat 128)
region_model.coordinate_encoder.weight [2048,256] -> coord_encoder.weight  (F32, torch (out,in) row-major)
region_model.coordinate_encoder.bias [2048]      -> coord_encoder.bias
region_model.coordinate_head.weight [1024,2048]  -> coord_decoder.weight
region_model.coordinate_head.bias [1024]         -> coord_decoder.bias
region_model.size_features.weight [256,2]        -> size_features.weight (F32, [2,256] = transpose; row c = channel c freqs)
region_model.size_encoder.weight [2048,512]      -> size_encoder.weight
region_model.size_encoder.bias [2048]            -> size_encoder.bias
region_model.size_head.weight [2048,2048]        -> size_decoder.weight
region_model.size_head.bias [2048]               -> size_decoder.bias
region_model.ln.weight / .bias [2048]            -> ln.weight / ln.bias

All tensors F32 (they are tiny; F32 keeps the C++ parity exact against the
BF16 checkpoint upcast to F32). Features matrices are transposed to the
math convention used by ref_region.py: x @ W with W [in, out_per_freq]:
  coord_features: checkpoint [128,1] -> flat [128] (already row-major in file)
  size_features:  checkpoint [256,2] -> transposed [2,256]
"""
import argparse
import json
import struct

import numpy as np
import gguf

import hf_source


class SfReader:
    """torch-free BF16 safetensors reader (same as convert_vision.py)."""

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
        return f32.reshape(info["shape"])


# RegionConfig defaults (moondream/torch/config.py)
DIM = 2048
COORD_FEAT_DIM = 256
COORD_OUT_DIM = 1024
SIZE_FEAT_DIM = 512
SIZE_OUT_DIM = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="moondream3.1-region.gguf")
    hf_source.add_src_argument(ap)
    args = ap.parse_args()

    src = hf_source.find_model_safetensors(args.src)
    print(f"source: {src}")
    print(f"output: {args.out}")
    w = SfReader(src)

    gw = gguf.GGUFWriter(args.out, "moondream3-region")   # GGUFWriter writes general.architecture
    gw.add_name("moondream3.1-9B-A2B-region")
    r = "moondream3.region."
    gw.add_uint32(r + "dim", DIM)
    gw.add_uint32(r + "coord_feat_dim", COORD_FEAT_DIM)
    gw.add_uint32(r + "coord_out_dim", COORD_OUT_DIM)
    gw.add_uint32(r + "size_feat_dim", SIZE_FEAT_DIM)
    gw.add_uint32(r + "size_out_dim", SIZE_OUT_DIM)

    P = "region_model."
    # linear weights/biases: keep torch (out,in) row-major; add 1-D for biases.
    # gguf-py stores ne = numpy.shape[::-1], which does not matter here: the
    # C++ loader treats every region tensor as a flat F32 blob in file order.
    gw.add_tensor("coord_features.weight",
                  np.ascontiguousarray(w.get_f32(P + "coordinate_features.weight").reshape(-1)))
    gw.add_tensor("coord_encoder.weight", w.get_f32(P + "coordinate_encoder.weight").reshape(-1))
    gw.add_tensor("coord_encoder.bias", w.get_f32(P + "coordinate_encoder.bias"))
    gw.add_tensor("coord_decoder.weight", w.get_f32(P + "coordinate_head.weight").reshape(-1))
    gw.add_tensor("coord_decoder.bias", w.get_f32(P + "coordinate_head.bias"))
    # checkpoint [256,2] -> math [2,256] (row c = freqs for channel c)
    gw.add_tensor("size_features.weight",
                  np.ascontiguousarray(w.get_f32(P + "size_features.weight").T.reshape(-1)))
    gw.add_tensor("size_encoder.weight", w.get_f32(P + "size_encoder.weight").reshape(-1))
    gw.add_tensor("size_encoder.bias", w.get_f32(P + "size_encoder.bias"))
    gw.add_tensor("size_decoder.weight", w.get_f32(P + "size_head.weight").reshape(-1))
    gw.add_tensor("size_decoder.bias", w.get_f32(P + "size_head.bias"))
    gw.add_tensor("ln.weight", w.get_f32(P + "ln.weight"))
    gw.add_tensor("ln.bias", w.get_f32(P + "ln.bias"))

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print("done:", args.out)


if __name__ == "__main__":
    main()
