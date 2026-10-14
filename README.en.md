# moondream.cpp

Run **moondream3.1-9B-A2B** locally with a standalone [ggml](https://github.com/ggml-org/ggml)-based
inference engine — no Python, no CUDA toolkit, no vendor runtime required.

[中文](README.md)

moondream3.1 is a 9B-parameter mixture-of-experts vision-language model (2B active per token)
from the Moondream team. Until now the only way to run it locally was Moondream's own
[Photon](https://moondream.ai/p/photon) engine (Python + proprietary CUDA kernels, 24 GB+ VRAM).
llama.cpp support has been an
[open feature request](https://github.com/ggml-org/llama.cpp/issues/13332) since 2025 — so we
built our own engine, following the same standalone pattern as
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) and [audio.cpp](https://github.com/amusecule/audio.cpp).

## Status

> Feature-complete: text + vision + all five official skills (caption / query / detect /
> point / gaze) + embedded web UI + batch inference + streaming, on both CPU and CUDA.

| | |
|---|---|
| Text decode (24 layers) | ✅ quality parity with the official implementation (per-tensor diff ~1e-3), no reserved tokens |
| GPU backend (CUDA via ggml) | ✅ ~106 tok/s single stream, 231 tok/s 8-way batch (5070 Ti Laptop, Q6_K; flash-attn + F16 KV + CUDA graph replay) |
| CPU backend | ✅ 12.8 tok/s (16 threads, numerically identical to CUDA) |
| SigLIP vision encoder | ✅ official multi-crop Lanczos preprocessing + batched encode, ~0.56 s/image (10 crops), parity < 0.04 |
| All five official skills | ✅ caption (short/normal/long) / query / detect / point / gaze, token-for-token official templates |
| Web UI (`--serve`) | ✅ image-first UI: drag/paste, 3 caption lengths, VQA, detect boxes, point crosshairs, gaze prediction; same-image cache hits in ~66 ms |
| FP8 expert weights → Q8_0 / Q6_K / Q4_K | ✅ converter included (Q6_K ≈ 7.6 GB — the pick for 12 GB GPUs) |
| Batch inference (`--batch`) / streaming | ✅ |

## Why a standalone engine?

moondream3.1 is architecturally far from anything llama.cpp supports:

- **tau attention** — a per-head, per-token learned scaling of Q/V:
  `q *= tanh(Wq·gelu(qkv)) + 1 + (sigmoid(alpha·log(pos+1)) − 0.5)`
- **parallel-residual blocks** — `x = x + attn(LN(x)) + mlp(LN(x))`, GPT-J style
- **LayerNorm with bias everywhere** (attention, pre-FFN, and the final norm), biased lm_head
- **GeGLU variant** `gelu(h) · (g + 1)` inside the MoE experts
- **dense prefix + MoE tail** — layers 0–3 are dense MLPs, layers 4–23 are MoE
- **official checkpoints ship FP8 (e4m3) expert weights** with per-channel scales
- a SigLIP vision encoder with 588-column patch embeddings and multi-crop tiling

Instead of wedging this into llama.cpp's model registry, moondream.cpp owns the whole stack:
weight conversion, graph construction, KV cache, and sampling — vendoring ggml for the kernels,
exactly like whisper.cpp does for audio.

## Weights

The converter (`convert/`) reads the official FP8 checkpoint
[`moondream/moondream3.1-9B-A2B`](https://huggingface.co/moondream/moondream3.1-9B-A2B)
and produces standard GGUF:

- FP8 expert tensors are decoded exactly (bit-identical to PyTorch's e4m3 decoding) and
  requantized to Q8_0 — `--q6` gives Q6_K (~7.6 GB, recommended for 12 GB GPUs),
  `--q4` gives Q4_K
- BF16 tensors become F16
- the per-channel FP8 scales are folded into the quantized blocks

```bash
pip install gguf safetensors torch
huggingface-cli download moondream/moondream3.1-9B-A2B
python convert/convert_hf_to_gguf.py            # → moondream3.1-q8_0.gguf (~10 GB)
python convert/convert_hf_to_gguf.py --q6      # → moondream3.1-q6k.gguf (~7.6 GB)
python convert/convert_vision.py               # → moondream3.1-vision.gguf (SigLIP tower)
python convert/convert_region.py               # → moondream3.1-region.gguf (detect/point/gaze heads)
```

Model weights are licensed separately under the
[Moondream Model License 1.0](https://moondream.ai/licenses/model/1.0) and are **not** distributed
from this repository. The license explicitly allows quantization, conversion, redistribution,
self-hosting and commercial integration (see the official
[license matrix](https://moondream.ai/licenses/model/1.0)); only reselling a public
general-purpose Moondream inference API requires a separate license — that obligation falls on
the operator, not on this engine.

## Building

Windows (MSVC) today; Linux via CMake should work with minor script tweaks.

```bash
git clone https://github.com/<you>/moondream.cpp
cd moondream.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp   # provides ggml

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build --config Release --target moondream-cli
```

## Running

```bash
# tokenizer (byte-level BPE, from moondream/starmie-v1)
curl -L https://huggingface.co/moondream/starmie-v1/resolve/main/tokenizer.json -o tokenizer.json

# text only
moondream-cli -m moondream3.1-q6k.gguf -t tokenizer.json -p "hi" -n 16

# image VQA; --detect / --point return boxes/coordinates
moondream-cli -m moondream3.1-q6k.gguf -t tokenizer.json \
    -v moondream3.1-vision.gguf --region moondream3.1-region.gguf \
    -i image.png -p "What is in this image?"

# web UI: http://127.0.0.1:8080 (--host 0.0.0.0 binds the LAN)
moondream-cli -m moondream3.1-q6k.gguf -t tokenizer.json \
    -v moondream3.1-vision.gguf --region moondream3.1-region.gguf --serve

# batch inference (list of text prompts, partitioned KV slots)
moondream-cli -m moondream3.1-q6k.gguf -t tokenizer.json --batch prompts.txt
```

The web UI covers all five official skills: caption (short/normal/long), query (VQA),
detect (green boxes), point (yellow crosshairs), and gaze prediction (click a source
point, predict the fixation point). Repeated queries on the same image reuse the
vision encode and KV prefix (~1.2 s → ~66 ms).

HTTP API: `POST /api/vision` with `mode=ask|caption|detect|point|gaze`
(`caption` takes `len=short|normal|long`, `gaze` takes source `x=&y=`), image bytes as body.

## Architecture summary

(Reverse-engineered from the official checkpoint and the open-source
[vikhyat/moondream](https://github.com/vikhyat/moondream) implementation.)


| | |
|---|---|
| layers | 24 (0–3 dense, 4–23 MoE) |
| dim / heads | 2048 / 32 heads, no GQA, head_dim 64 |
| RoPE | half-rotation on first 32 dims, theta 1.5e6 |
| MoE | 64 experts, 8 active, inter dim 1024, router with bias |
| vocab | 51200 (`moondream/starmie-v1`) |
| context | 32768 |

## License

MIT — see [LICENSE](LICENSE). ggml is MIT (ggml-org). Moondream model weights and the
tokenizer are subject to their own terms and are downloaded separately.
