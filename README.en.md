# moondream.cpp

Local **image/video VQA** on top of [ggml](https://github.com/ggml-org/ggml) — a multi-model
inference engine: every vision-language model is a backend implementing `vqa::Model`, and one
front end picks the backend from the GGUF `general.architecture` field. Images and video frame
sequences share the same input abstraction. No Python, no CUDA toolkit and no vendor runtime is
needed to run it.

[中文](README.md)

## Supported backends

| Backend | Architecture tag | Input | Notes |
|---|---|---|---|
| **moondream3.1-9B-A2B** | `moondream3` | image | MoE text model + SigLIP tower; all five official skills (caption / query / detect / point / gaze), web UI, batch inference |
| **VideoChat3-4B** | `videochat3` | image | Qwen3 text trunk (36 layers, GQA, per-head qk-norm) + VideoChat3's own 27-layer ViT (2D RoPE, 2×2 merge), built-in merger and multimodal projector |

## Status

> moondream3.1 is feature-complete: text + vision + all five official skills + embedded web UI
> + batch inference + streaming, on both CPU and CUDA. VideoChat3-4B coverage is noted below.

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
| VideoChat3-4B text / image VQA | ✅ works end to end (`vqa-cli`, verified on CPU and CUDA); **stage-by-stage parity against the PyTorch reference is still outstanding** |
| VideoChat3-4B video input | 🔜 frame decoding and timestamped prompts are designed but not wired up yet |

## Quick start

### 1. Build

The only third-party dependency is ggml itself (it ships gguf too):

```bash
git clone https://github.com/<you>/moondream.cpp
cd moondream.cpp
git clone --depth 1 https://github.com/ggml-org/ggml

# Windows (MSVC + Ninja): builds vqa-cli and moondream-cli in one go
scripts\build-cuda.bat        # CUDA backend (defaults to sm_89 = Ada, override with CUDA_ARCH)
scripts\build-vqa.bat         # CPU only
scripts\build.bat             # CPU only, Visual Studio generator (artifacts in build\Release\)
# or plain CMake:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build --config Release --target vqa-cli moondream-cli
```

`vqa-cli` is the architecture-dispatching front end; `moondream-cli` is the standalone
moondream3.1 binary (it owns the web UI, batch mode and the skill flags). Both share the same
engine code. In a CUDA build the engine prefers the GPU; `--cpu` (or `MD_CPU=1` for the
moondream binary) forces the CPU backend.

### 2. Get the weights

```bash
pip install -r convert/requirements.txt

# moondream3.1-9B-A2B (official FP8 checkpoint)
hf download moondream/moondream3.1-9B-A2B --local-dir models/moondream3.1-9B-A2B

# VideoChat3-4B
hf download qqceqqq/VideoChat3-4B --local-dir models/VideoChat3-4B
```

The converters find the checkpoint themselves: `--src <dir|file>` → the model's
`$MOONDREAM_SRC` / `$VIDEOCHAT3_SRC` → the HF cache (`$HF_HOME` or `~/.cache/huggingface/hub`)
→ `models/` inside the repo. See [Model conversion](#model-conversion).

### 3. Run

```bash
# unified front end: the backend follows the GGUF's general.architecture
vqa-cli --model models/VideoChat3-4B/videochat3-4b-q8_0.gguf \
        --tokenizer models/VideoChat3-4B \
        --image photo.jpg --ask "What is in this image?" --max-tokens 256

vqa-cli --model moondream3.1-q6k.gguf --tokenizer tokenizer.json \
        --vision moondream3.1-vision.gguf --region moondream3.1-region.gguf \
        --image photo.jpg --ask "What is in this image?" --n-ctx 1024

# standalone moondream3.1 binary: skills, web UI, batch
moondream-cli -m moondream3.1-q6k.gguf -t tokenizer.json \
    -v moondream3.1-vision.gguf --region moondream3.1-region.gguf \
    -i photo.jpg -p "What is in this image?"
moondream-cli -m moondream3.1-q6k.gguf -t tokenizer.json \
    -v moondream3.1-vision.gguf --region moondream3.1-region.gguf --serve   # http://127.0.0.1:8080
moondream-cli -m moondream3.1-q6k.gguf -t tokenizer.json --batch prompts.txt
```

Common `vqa-cli` flags: `--frames <dir> --fps 2` (video frames), `--temp/--top-k/--top-p`
(sampling, greedy by default), `--max-tokens`, `--n-ctx`, `--threads`, `--cpu`,
`--dump-ids` (tokenizer self test) — see `--help` for the rest.

The web UI covers all five official skills: caption (short/normal/long), query (VQA),
detect (green boxes), point (yellow crosshairs) and gaze prediction (click a source point,
predict the fixation point). Repeated queries on the same image reuse the vision encode and
KV prefix (~1.2 s → ~66 ms). HTTP API: `POST /api/vision` with `mode=ask|caption|detect|point|gaze`
(`caption` takes `len=short|normal|long`, `gaze` takes source `x=&y=`), image bytes as body.

## Model conversion

Every converter lives in `convert/` and shares `hf_source.py` (checkpoint discovery) and
`_lib.py` (GGUF/quantization helpers). Logs use a `[convert]` prefix and the written file is
re-opened and its tensor table verified.

| Script | Output |
|---|---|
| `convert_hf_to_gguf.py` | `moondream3.1-q8_0.gguf` / `-q6k` / `-q4k` (text trunk; `--q6` / `--q4` / `--q8-dense`) |
| `convert_vision.py` | `moondream3.1-vision.gguf` (SigLIP tower, ~0.84 GB) |
| `convert_region.py` | `moondream3.1-region.gguf` (detect/point/gaze head, ~30 MB) |
| `convert_videochat3.py` | `videochat3-4b-<outtype>.gguf` (Qwen3 + VideoChat3 ViT, single file, `--outtype f16\|q8_0\|q6_k\|q4_0`) |

### moondream3.1

Reads the official FP8 checkpoint
[`moondream/moondream3.1-9B-A2B`](https://huggingface.co/moondream/moondream3.1-9B-A2B):

- FP8 expert tensors are decoded exactly (bit-identical to PyTorch's e4m3 decoding) and
  requantized to Q8_0 — `--q6` gives Q6_K (~7.6 GB, recommended for 12 GB GPUs),
  `--q4` gives Q4_K
- BF16 tensors become F16
- the per-channel FP8 scales are folded into the quantized blocks

```bash
python convert/convert_hf_to_gguf.py          # → moondream3.1-q8_0.gguf (~10 GB)
python convert/convert_hf_to_gguf.py --q6     # → moondream3.1-q6k.gguf (~7.6 GB)
python convert/convert_vision.py              # → moondream3.1-vision.gguf
python convert/convert_region.py              # → moondream3.1-region.gguf
```

Output goes to the current directory (`-o` overrides it; on the moondream side `%s` in the
output name expands to the quantization tag). The Q6_K quantizer is bundled (gguf-py still
ships no Q6_K quantization — `quants.quantize` raises `NotImplementedError`);
`python scripts/test_q6k.py` checks the block layout and quantization error.

### VideoChat3-4B

One GGUF holds everything: the text decoder (`--outtype`, default `q8_0`), the 27-layer ViT and
the multimodal projector (kept F16), and the learned position tables (F32, the host interpolates
them per image). The tokenizer is not embedded — the engine reads `tokenizer.json` from the
checkpoint directory at run time (`--tokenizer <dir>`).

```bash
python convert/convert_videochat3.py --outtype q8_0   # → videochat3-4b-q8_0.gguf (4.83 GB)
python convert/convert_videochat3.py --outtype q6_k   # → 3.92 GB
python convert/convert_videochat3.py --outtype q4_0   # → 2.95 GB
```

`q4_k` / `q5_k` / `q2_k` are not available yet (upstream gguf-py implements their decode only);
`q6_k` uses this repository's `convert/q6k.py`. Q8_0 and above hold more precision, Q6_K/Q4_0
use less memory and bandwidth.

Model weights follow their own licenses (moondream: the
[Moondream Model License 1.0](https://moondream.ai/licenses/model/1.0)) and are **not**
distributed from this repository. The license explicitly allows quantization, conversion,
redistribution, self-hosting and commercial integration; only reselling a public
general-purpose Moondream inference API requires a separate license — that obligation falls on
the operator, not on this engine.

## Why a standalone engine?

moondream3.1 is a 9B-parameter mixture-of-experts vision-language model (2B active per token)
from the Moondream team. Until now the only way to run it locally was Moondream's own
[Photon](https://moondream.ai/p/photon) engine (Python + proprietary CUDA kernels, 24 GB+ VRAM).
llama.cpp support has been an
[open feature request](https://github.com/ggml-org/llama.cpp/issues/13332) since 2025 — its
architecture is far from anything llama.cpp supports:

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
exactly like [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and
[audio.cpp](https://github.com/0xShug0/audio.cpp). VideoChat3-4B is the second backend in the
same framework: its vision tower, positional encoding and multimodal projector are equally
absent from ggml's existing models.

## Architecture summary

### moondream3.1 (reverse-engineered from the official checkpoint and [vikhyat/moondream](https://github.com/vikhyat/moondream))

| | |
|---|---|
| layers | 24 (0–3 dense, 4–23 MoE) |
| dim / heads | 2048 / 32 heads, no GQA, head_dim 64 |
| RoPE | half-rotation on first 32 dims, theta 1.5e6 |
| MoE | 64 experts, 8 active, inter dim 1024, router with bias |
| vocab | 51200 (`moondream/starmie-v1`) |
| context | 32768 |

### VideoChat3-4B

| | |
|---|---|
| text trunk | Qwen3: 36 layers, dim 2560, 32 heads / 8 KV heads, head_dim 128, per-head qk-norm, RoPE theta 5e6, tied lm_head |
| vision tower | 27 layers, dim 1152, 16 heads, patch 14, 2D RoPE, 2×2 merge, GELU(tanh) MLP, learned 64×64 position table |
| fusion | merger concatenates 4 patches into 4608 dims → LayerNorm → 4608→4608 GELU(erf) → 2560 |
| vocab / context | 151936 (Qwen2 BPE); 262144 text context, engine KV default 8192 |
| prompt | `<|im_start|>user\n<|vision_start|>` + N×`<|image_pad|>` + `<|vision_end|>` + question |

## Performance

VideoChat3-4B at Q8_0 is memory-bandwidth bound on CPU; smaller quantizations or a CUDA build
help a lot (measured here, 37-token prompt, greedy decoding — numbers move with hardware and load):

| Model / quantization | Backend | Hardware | prefill | decode |
|---|---|---|---|---|
| VideoChat3-4B Q8_0 (4.83 GB) | CPU, 14 threads | i7-14700K | ~71 tok/s | ~10.4 tok/s |
| VideoChat3-4B Q8_0 (4.83 GB) | CUDA | RTX 4070 SUPER | ~401 tok/s | ~49.8 tok/s |

Every run prints a `[perf] prefill ... | decode ...` line on stderr so you can check your own
setup. moondream3.1 numbers are in [Status](#status).

## Repository layout

```
src/
  core/                shared layer: vqa::Model interface, mmap/UTF-8/JSON helpers, byte-level
                       BPE tokenizer, image & video-frame decoding (the single stb_image
                       implementation lives here), architecture discovery
  models/moondream/    moondream3.1 backend (engine + vision tower + multi-crop preprocessing
                       + region head + server/batch)
  models/videochat3/   VideoChat3-4B backend (vision preprocessing + graphs)
  cli/                 unified front end (vqa-cli)
convert/               HF checkpoint → GGUF converters
scripts/               build scripts, parity and self-test tooling
vendor/stb/            vendored single-header image decoder
ggml/                  the only third-party dependency (clone it, not tracked)
```

## Development

- **Parity tooling**: `scripts/ref_videochat3.py` runs the official transformers implementation
  and dumps intermediate tensors; `scripts/compare_vc3.py` compares them stage by stage against
  the C++ dumps (preprocessing / patch embedding / layer outputs / logits).
- **Debug switches**: `VC3_DUMP_DIR=<dir>` dumps intermediate tensors, `VC3_MAX_LAYERS=N`
  truncates the stack for layer-by-layer comparison, `VC3_NO_FUSION=1` falls back to the
  unfused SwiGLU, `--dump-ids` tokenizes and exits.
- **Tokenizer data**: `scripts/gen_unicode_ranges.py` regenerates `src/core/unicode_ranges.inc`.

## License

MIT — see [LICENSE](LICENSE). ggml is MIT (ggml-org). Model weights and tokenizers are subject
to their own terms and are downloaded separately.
