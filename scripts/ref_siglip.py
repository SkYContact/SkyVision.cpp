"""Reference activations for a SigLIP checkpoint (PyTorch / transformers).

Runs the official implementation on the same image the engine sees and dumps
the tensors the C++ backend names, so the two can be compared stage by stage:

  python scripts/ref_siglip.py --model models/siglip-base --image test_image.png \
      --out parity/siglip
  # -> ref_pixel_values.npy [3,224,224], ref_v_patch_out.npy [768,196],
  #    ref_v_final.npy [768,196], ref_v_pooled.npy [768]

The engine writes the matching cpp_<name>.bin / .shape pairs when run with
--dump-dir parity/siglip (plus cpp_pixels.bin = its own resize/normalize
output), so preprocessing, patch embedding and the pooling head can each be
blamed separately.

transformers 4.57 detail that matters: SiglipModel.get_image_features is the
vision tower's pooler_output VERBATIM - no projection linear in this release
lineage. pooler_output = MultiheadAttentionPoolingHead applied on top of
post_layernorm(last_hidden_state).
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF checkpoint dir (or hub id)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    proc = AutoImageProcessor.from_pretrained(args.model)
    model = SiglipModel.from_pretrained(args.model).eval()
    print(f"vision cfg: hidden {model.config.vision_config.hidden_size} "
          f"layers {model.config.vision_config.num_hidden_layers} "
          f"heads {model.config.vision_config.num_attention_heads} "
          f"patch {model.config.vision_config.patch_size} "
          f"eps {model.config.vision_config.layer_norm_eps} "
          f"act {model.config.vision_config.hidden_act}")

    img = Image.open(args.image).convert("RGB")
    print(f"image: {args.image} {img.size[0]}x{img.size[1]}")
    pv = proc([img], return_tensors="pt")["pixel_values"]   # [1, 3, 224, 224]

    seen = {}

    def hook(name):
        def fn(_m, _i, out):
            seen[name] = out.detach().float()
        return fn

    vm = model.vision_model
    vm.embeddings.register_forward_hook(hook("v_patch_out"))
    vm.post_layernorm.register_forward_hook(hook("v_final"))

    with torch.no_grad():
        feats = model.get_image_features(pixel_values=pv)

    def save(name, arr):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        np.save(os.path.join(args.out, f"ref_{name}.npy"), a)
        print(f"{name:16s} shape {a.shape} mean {a.mean():+.6f} "
              f"absmax {np.abs(a).max():.6f}")

    save("pixel_values", pv[0].numpy())                       # [3, 224, 224]
    save("v_patch_out", seen["v_patch_out"][0].numpy().T)     # [dim, tokens]
    save("v_final", seen["v_final"][0].numpy().T)             # [dim, tokens]
    pooled = feats[0].float().numpy()
    save("v_pooled", pooled)
    print(f"pooled: dim {pooled.shape[0]} l2 {np.linalg.norm(pooled):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
