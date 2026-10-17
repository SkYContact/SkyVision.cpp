# SkyVision.cpp

[![docker-mcp](https://github.com/SkYContact/SkyVision.cpp/actions/workflows/docker-mcp.yml/badge.svg)](https://github.com/SkYContact/SkyVision.cpp/actions/workflows/docker-mcp.yml)

Local **image/video VQA** on top of [ggml](https://github.com/ggml-org/ggml) — a multi-model
inference engine: every vision-language model is a backend implementing `vqa::Model`, and one
front end picks the backend from the GGUF `general.architecture` field. Images and video frame
sequences share the same input abstraction. No Python, no CUDA toolkit and no vendor runtime is
needed to run it.

[中文](README.md)

It also runs as an **MCP (Model Context Protocol) server**: `vqa-cli --mcp` exposes the same
engine over stdio JSON-RPC with 9 tools (image Q&A, captioning, object detection / pointing,
gaze, video Q&A, video classification, model management). The repository ships a `Dockerfile`
whose image bundles a 403 MB SmolVLM, so it answers "what is in this image" out of the box
with nothing mounted - see [the MCP section](#4-mcp-server-for-agents).

## Supported backends

| Backend | Architecture tag | Input | Notes |
|---|---|---|---|
| **moondream3.1-9B-A2B** | `moondream3` | image | MoE text model + SigLIP tower; all five official skills (caption / query / detect / point / gaze), web UI, batch inference |
| **VideoChat3-4B** | `videochat3` | image | Qwen3 text trunk (36 layers, GQA, per-head qk-norm) + VideoChat3's own 27-layer ViT (2D RoPE, 2×2 merge), built-in merger and multimodal projector |
| **Marlin-2B** | `marlin` | image + video | Qwen3.5 hybrid trunk (18 gated-delta linear-attention + 6 full-attention layers, interleaved MRoPE) + Qwen3-VL style vision tower (learned absolute pos embed, 2D RoPE, per-temporal-step attention); the two trained skills: caption (timestamped event list) and find (temporal grounding) |
| **VideoJev-4B (scoring)** | `videojev` | video | Qwen3.5-4B trunk (32 layers: 24 linear + 8 full, **16 key / 32 value heads in the linear layers**) + the same vision tower, but the last layer is **attention-only** (MLP and post-norm trimmed) and a trained cosine decision head replaces the lm head: one forward scores a fixed candidate list (temporal grounding = per-patch sigmoid, single-choice VQA = option softmax). No text generation at all — drive it with `--score <qs.json>` |
| **VideoMAE (base)** | `videomae` | video (image = still clip) | vision-only tower, no text side: Conv3d tubelet patch embedding + fixed sin-cos position table (**no cls token**) + 12 full space-time attention layers; one forward pass returns a label distribution (fine-tunes such as Kinetics-400) or the pooled embedding (`MCG-NJU/videomae-base` is the pretraining release, no head) |

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
| Marlin-2B image/video VQA | ✅ end to end with **stage-by-stage parity verified** against the official transformers 5.20 reference (pixel values bitwise-identical, tower drift beats official bf16, logits 1.5e-2 vs f32); `--mode caption` / `--mode find` wired to the canonical trained prompts |
| VideoJev-4B candidate scoring | ✅ whole path wired + **verified against the official transformers 5.14 code** (`scripts/parity_videojev.py`; prompt token ids are byte-identical). Taking a CPU float32 reference as truth: the engine (f16) is within **6.8e-3** max probability error (q8_0: 5.9e-3), while the official repo's own bf16 run is off by 5.7e-2 — the engine tracks fp32 closer than the reference default. Choice head: 6.6e-6 vs 5.1e-5 |
| VideoChat3-4B video input | 🔜 frame decoding and timestamped prompts are designed but not wired up yet |
| VideoMAE classification / features | ✅ single-GGUF tower (94M params, ~180 MB at f16): 16 uniform frames → center crop 224 → one full space-time attention pass → top-k labels; no tokenizer, no KV cache. **Parity verified** against an independent numpy recomputation from the GGUF weights: ~1e-3 relative over 12 layers, patch embedding 3e-3, block-0 attention 3e-4, all 400 Kinetics probabilities within 1e-4. Headless pretraining checkpoints return the pooled embedding (`--mode embed` dumps 768 floats as JSON). ~0.85 s per clip on CPU (q8_0 weights: 0.59 s) |
| ViViT-b/16x2 classification | ✅ same video tower (the `videomae` backend, arch `vivit`): the only differences are a learned **cls token**, a learned position table, `gelu_fast` (tanh form), LN eps 1e-6, bias on all of q/k/v and a bilinear 256→224 preprocessing with zero centering. 32 frames → 3136+1 tokens, Kinetics-400 labels. Parity vs `VivitForVideoClassification`: probabilities within 6e-3, identical top-5 order, ~2.2 s per clip on CPU |
| Qwen2.5-VL-3B | ✅ arch `qwen2vl`: Qwen2 text trunk (36L GQA 16:2, qkv bias, section MRoPE 16/24/24, SwiGLU) + Qwen2.5 ViT (32L, patch 14, temporal 2, 2×2 merge, **window attention** with a global layer every 8th) + merger, dynamic resolution through `smart_resize`. Preprocessing is CLIP mean/std, bicubic, patches in **2×2 group order**. Merged features corr 0.9935 vs the official tower, same answers ("There are five people in the image.") |
| SmolVLM-256M (Idefics3) | ✅ arch `smolvlm`, the smallest backend here: SigLIP-base tower (12L, patch 16, 512², learned positions, LayerNorm/attention/MLP all with bias, `gelu_tanh`) + **4×4 pixel shuffle** connector (12288→576) + SmolLM2-135M trunk (30L, no biases, GQA 9:3, rope 1e5). Idefics3 preprocessing: longest edge 2048 → pad to multiples of 512 → 512² tiles + one global view, 64 tokens per tile. 515 MB at f16 / ~270 MB at q8. Logits corr 0.99991 and byte-identical answers |

## Quick start

### 1. Build

The only third-party dependency is ggml itself (it ships gguf too):

```bash
git clone https://github.com/SkYContact/SkyVision.cpp
cd SkyVision.cpp
git clone --depth 1 https://github.com/ggml-org/ggml

# Windows (MSVC + Ninja)
scripts\build-cuda.bat        # CUDA backend (defaults to sm_89 = Ada, override with CUDA_ARCH)
scripts\build-vqa.bat         # CPU only
scripts\build.bat             # CPU only, Visual Studio generator (artifacts in build\Release\)
# or plain CMake:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build --config Release --target vqa-cli
```

`vqa-cli` is the architecture-dispatching front end - **moondream3.1 included**: it has no
standalone binary and no HTTP server of its own. Every capability (VQA, caption/detect/point/
gaze/ask-think, batch, web UI) goes through `vqa::Model` like the other backends. In a CUDA
build the engine prefers the GPU; `--cpu` (or `MD_CPU=1`) forces the CPU backend.

### 2. Get the weights

Prebuilt GGUF files, no conversion needed: [ModelScope](https://modelscope.cn/collections/SkyContact/SkyVisioncpp) · [Hugging Face](https://huggingface.co/collections/SkyContact/skyvisioncpp)

Or convert them yourself, or for other quantization levels:

```bash
pip install -r convert/requirements.txt

# moondream3.1-9B-A2B (official FP8 checkpoint)
hf download moondream/moondream3.1-9B-A2B --local-dir models/moondream3.1-9B-A2B

# VideoChat3-4B
hf download MCG-NJU/VideoChat3-4B --local-dir models/VideoChat3-4B
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

# moondream3.1 is a normal backend: skills, batch and the web UI live in vqa-cli
vqa-cli --model models/moondream3.1-q6k.gguf --image photo.jpg --ask "What is in this image?"
vqa-cli --model models/moondream3.1-q6k.gguf --image photo.jpg --ask person --mode detect
vqa-cli --model models/moondream3.1-q6k.gguf --image photo.jpg --ask normal --mode caption
vqa-cli --model models/moondream3.1-q6k.gguf --batch prompts.txt
vqa-cli --serve 8080
# --vision/--region default to the moondream3.1-*.gguf siblings of the model file
```

`--vision/--region` default to the `moondream3.1-*.gguf` siblings of the model file, so
`--model models/moondream3.1-q6k.gguf` is enough for the skills too.

Common `vqa-cli` flags: `--frames <dir> --fps 2` (video frames), `--temp/--top-k/--top-p`
(sampling, greedy by default), `--max-tokens`, `--n-ctx`, `--threads`, `--cpu`,
`--dump-ids` (tokenizer self test) — see `--help` for the rest.

The web UI covers all five official skills: caption (short/normal/long), query (VQA),
detect (green boxes), point (yellow crosshairs) and gaze prediction (click a source point,
predict the fixation point). Repeated queries on the same image reuse the vision encode and
KV prefix (~1.2 s → ~66 ms). HTTP API: `POST /api/vision` with `mode=ask|caption|detect|point|gaze`
(`caption` takes `len=short|normal|long`, `gaze` takes source `x=&y=`), image bytes as body.

### 4. MCP server (for agents)

`vqa-cli --mcp` wraps the same engine as a **Model Context Protocol** server over stdio
(newline-delimited JSON-RPC), so any MCP client (CodeBuddy, Claude Desktop, a custom agent)
can point the local models at images and videos without glue code:

```bash
# from the repo root; --models-dir defaults to ./models when it exists
vqa-cli --mcp --models-dir models
vqa-cli --mcp --models-dir models --model moondream3.1-q6k.gguf   # optional default model
```

Client entry (`mcpServers`) — the bundled launcher is the easy path: it cd's to the repo root,
picks a build whose `vqa-cli` really knows `--mcp`, puts the `ggml*.dll` directory on `PATH`,
then hands stdio to the server:

```json
{
  "mcpServers": {
    "skyvision": {
      "command": "cmd",
      "args": ["/c", "C:\\moondream\\scripts\\mcp-server.bat"],
      "env": { "FFMPEG": "ffmpeg" }
    }
  }
}
```

Pointing straight at the binary works too (then the DLLs are your responsibility):

```json
{
  "mcpServers": {
    "skyvision": {
      "command": "C:/moondream/build-ninja/vqa-cli.exe",
      "args": ["--mcp", "--models-dir", "C:/moondream/models"],
      "env": { "PATH": "C:/moondream/build-ninja/bin;%PATH%" }
    }
  }
}
```

> Shared-library builds keep `ggml*.dll` in `<build>/bin/`, which is not on a client's `PATH`
> and not next to the exe, so use the launcher or fix `PATH` / copy the DLLs. Both
> `scripts/mcp-server.bat` (Windows) and `scripts/mcp-server.sh` (Linux/macOS/WSL) honour
> `SKYVISION_BUILD` and `SKYVISION_MODELS`. Video tools need `ffmpeg` (`FFMPEG` overrides the
> path); image tools do not.

#### Docker (what an MCP directory runs)

The repository ships a `Dockerfile` that builds the CPU binary and **bundles the 403 MB
SmolVLM-256M q8 checkpoint**, so the image is a working MCP server with nothing mounted:

```bash
docker build -t skyvision-mcp .
docker run --rm -i skyvision-mcp                       # smolvlm is ready out of the box
docker run --rm -i -v ~/models:/models skyvision-mcp    # or point it at your own GGUFs
```

Entry point is `vqa-cli --mcp --models-dir /models`; the bundled model is what
`describe_image` / `ask_image` fall back to, and every `*.gguf` you mount appears in
`list_models`. `SMOLVLM_MAX_EDGE=1024` is preset in the image (see the SmolVLM section:
the official processor upscales to 2048, which costs 2.3x more for the same answers).

An image with **no** weights still answers `initialize`, `tools/list` and `list_models`
(empty list) instead of failing — that is exactly what directory crawlers, CI smoke tests
and the review step of an MCP registry exercise.

The bundled checkpoint is pulled from `SkyContact/SmolVLM-256M-GGUF` on Hugging Face
(mirrored on ModelScope); pin or switch sources with
`--build-arg MODEL_BASE=... --build-arg MODEL_REV=...`.

[`.github/workflows/docker-mcp.yml`](.github/workflows/docker-mcp.yml) builds this image on
every push and runs both smoke tests above (handshake with no models, then `describe_image`
against the bundled checkpoint), so the badge reflects a real Linux build rather than a
claim. Layers are ordered so that apt, the ggml checkout and the 403 MB checkpoint are
cached: a source change re-runs only the compile (through ccache).

| Tool | What it does |
|---|---|
| `list_models` | Runnable GGUFs (arch / quant / size / resident) plus the loaded model's skill bits |
| `load_model` | Load explicitly (one resident model; loading another unloads it), `n_ctx` / `threads` / `gpu` |
| `describe_image` | Caption, `length=short\|normal\|long` |
| `ask_image` | Free-form VQA (`mode=ask-think` for moondream grounded reasoning), `session` for multi-turn |
| `detect_objects` | Bounding boxes as JSON (normalised 0..1 + pixels); needs moondream3 + `*-region.gguf` |
| `point_objects` | Target centre point, same requirement |
| `gaze_point` | Fixation point from a start point, same requirement |
| `analyze_video` | Video VQA (ffmpeg sampling, `fps` / `max_frames`); Marlin adds `mode=caption\|find` |
| `classify_video` | Pure video tower (VideoMAE / ViViT) → top-k labels or the 768-d embedding |

Conventions: media arguments accept a local path or inline bytes (`data:image/...;base64,<data>`,
`base64:<data>`); a still image works for `classify_video` / `analyze_video` as a one-frame clip
(no ffmpeg needed); coordinate tools return both normalised and pixel values. Tool failures come
back as `isError: true` with a readable reason (not a JSON-RPC error code) — only protocol
violations use JSON-RPC errors.

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
| `convert_videomae.py` | `videomae-<outtype>.gguf` (VideoMAE video tower, single file, `--outtype f16\|q8_0\|q4_0`; the classifier and its `id2label` table travel inside the GGUF) |

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

### VideoMAE

Reads any transformers `videomae` checkpoint (`MCG-NJU/videomae-base`,
`MCG-NJU/videomae-base-finetuned-kinetics`, ...):

```bash
python convert/convert_videomae.py --src models/videomae-base-ft   # → videomae-f16.gguf (~190 MB)
vqa-cli --model videomae-f16.gguf --video clip.mp4 --mode 5        # top-5 labels
vqa-cli --model videomae-f16.gguf --video clip.mp4 --mode embed    # 768-d pooled embedding
```

- the Conv3d patch embedding is folded into one matmul (`[dim, C*tubelet*patch*patch]`),
  tokens ordered `(t', h', w')` with elements `(c, dt, dh, dw)` — the Conv3d weight order
- the position table is **not** a learned weight: it is the fixed sin-cos table from the
  original implementation (1-D indices 0..1567), generated by the converter and stored as F32
- there is **no cls token**: `pooled` is token 0 (with a final encoder LayerNorm) for the
  pretraining release, or the mean over tokens + `fc_norm` (eps 1e-5, and *no* encoder
  LayerNorm) for `use_mean_pooling` fine-tunes; the converter writes both facts as metadata
- q/k/v are fused at conversion time (`v.blk.{i}.wqkv.weight`), with bias `q_bias | 0 | v_bias` —
  VideoMAE gives keys no bias
- `id2label` is stored as the `videomae.labels` string array, so labels print by name;
  a pretraining checkpoint (no `classifier.*`) only exposes the embedding
- **the graph is built once**: the clip shape is fixed (16 frames → 1568 tokens), so `load()`
  builds a static graph and each clip only re-fills the input tensor. That saves a
  per-clip rebuild (<0.5 ms — negligible) but more importantly it keeps one `ggml_cgraph`
  alive across calls, which is what CUDA graph capture keys on (`nodes[0]`)
- on CPU, quantized weights are the real speed lever: `--outtype q8_0` (~100 MB,
  850 ms → 590 ms on an i7-14700K, top-1 probability drift 4e-4). f16 stays the default
  (smaller change, faster on GPU)

Parity: `python scripts/ref_videomae.py --model <hf_dir> --video clip.mp4 --out ref_videomae`
dumps the official `v_patch_out / v_blk0_out / v_final / v_pooled / v_logits / v_probs`; run the
engine with `--dump-dir ref_videomae` on the same clip and diff the `cpp_*.bin` files stage by stage
(the engine also exports `v_blk0_qkv` and `v_blk0_attn_out`, plus `VIDEOMAE_MAX_LAYERS=N` for
per-layer drift).

### ViViT

`google/vivit-b-16x2` is, in the transformers implementation, a ViT over tubelet tokens with a
cls token - so it reuses the videomae backend and everything else is GGUF metadata:

```bash
python convert/convert_vivit.py --src models/vivit-b-16x2      # -> vivit-f16.gguf (~183MB)
```

- learned **cls token** (`v.cls_token.weight`), concatenated after the patch embedding and
  before the position add; the position table has 3137 rows (cls included);
- the position table is a **learned parameter**, unlike VideoMAE's generated sin-cos table;
- `hidden_act = gelu_fast` is transformers' `FastGELUActivation`, i.e. the **tanh** form, not
  `x*sigmoid(1.702x)` (that one is `quick_gelu`) - assuming the sigmoid form is what made block 0
  diverge at first;
- LN eps **1e-6** and a bias on all of q/k/v;
- preprocessing: shortest edge 256 -> center crop 224 with **bilinear** resampling
  (`resample: 2` is PIL's BILINEAR) and zero centering (mean = std = 0.5).

Parity: `python scripts/ref_vivit.py --model models/vivit-b-16x2 --image photo.jpg --out ref_vivit`,
engine with `--dump-dir ref_vivit` - patch tokens mean|d| 3.5e-3, block 0 1.0e-2, 12 layers
1.4e-2 (values up to 22, corr 0.99977), all 400 Kinetics probabilities within **6e-3** with the
same top-5 order. The residual comes from resampling rounding (~1.8 grey levels per pixel).

### Qwen2.5-VL

```bash
python convert/convert_qwen2vl.py --src models/qwen2.5-vl-3b     # -> qwen2vl-f16.gguf (7.5GB)
vqa-cli --model qwen2vl-f16.gguf --tokenizer models/qwen2.5-vl-3b \
        --image photo.jpg --ask "How many people are in the image?"
```

Three bugs cost most of the debugging time here; all of them are the "numbers look plausible,
answer is wrong" kind:

1. **patches arrive in 2x2 group order**. `Qwen2VLImageProcessor` reshapes to
   `(grid_t, gh/2, 2, gw/2, 2, C, T, P, P)` and transposes to
   `(t, gh/2, gw/2, 2, 2)`, so packing them raster-order silently ruins the features.
2. **the merger RMS-normalizes per token over 1280** and only then folds the 2x2 group into a
   5120 row; normalizing after the fold destroys the features (this one hid behind "tower corr
   0.98 but the answer is wrong").
3. **the KV cache slot is not the MRoPE `t` axis**: every image token shares `t = st`, so using
   that array as the `ggml_set_rows` index writes all 612 tokens into one slot.

Also: preprocessing uses **CLIP mean/std** (not 0.5) with bicubic resizing and `smart_resize` to
multiples of 28; text is section MRoPE (16/24/24), image tokens get `(st, st+h, st+w)` and `st`
advances by `max(gh, gw)` afterwards; the next decode position is `max(pos)+1`, not `T`.

Parity: `python scripts/ref_qwen2vl.py --model models/qwen2.5-vl-3b --image photo.jpg --out ref_q2v`
plus the engine with `--dump-dir ref_q2v`. Measured: patches identical, block 0 maxdiff 0.16
(values up to 15), tower output corr 0.982 (values up to 2.3e4), merged features corr **0.9935**,
first token matches the official argmax and the answer matches too
("There are five people in the image."). Debug knobs: `Q2V_MAX_VLAYERS=N` truncates the tower,
`Q2V_FEATS=<file>` feeds features from disk (isolates tower vs text), `Q2V_DEBUG=1` prints tokens
and the first-token top-5.

### SmolVLM (the smallest backend)

A 256M Idefics3 model, ~270 MB quantized - small enough to ship inside a container
image so "it always runs" (CI, demos, an MCP server that must answer without weights
mounted):

```bash
python convert/convert_smolvlm.py --src models/smolvlm-256m                  # -> f16, 515 MB
python convert/convert_smolvlm.py --src models/smolvlm-256m --outtype q8_0   # -> q8, ~270 MB
vqa-cli --model smolvlm-256m-f16.gguf --tokenizer models/smolvlm-256m \
        --image photo.jpg --ask "How many people are in the image?"
```

What it actually is (measured from the checkpoint, not assumed):

- the tower is SigLIP-base: `Conv2d(3,768,16,16)` folded into one matmul (with bias), a
  **learned 32×32 position table** (no cls), 12 layers of **LayerNorm with bias** +
  attention with bias on qkv/out (**no rope**) + a plain MLP (`gelu_pytorch_tanh`, not
  SwiGLU);
- the connector is a 4×4 pixel shuffle followed by one `Linear(12288→576)`, so a
  512×512 tile becomes exactly 64 text tokens;
- the trunk is SmolLM2-135M: 30 layers, d=576, 9/3 heads, **no biases anywhere**,
  RMSNorm 1e-5, rope theta 1e5 and an **untied** lm_head.

Idefics3 preprocessing is the easy thing to get wrong: resize the longest edge to
**2048** (not 512), then resize again so both sides are multiples of 512 (this second
pass does not preserve the aspect ratio), cut 512² tiles row-major and **append one
global view** of the whole image last; `<image>` expands into
`<fake><row_R_col_C><image>×64` per tile plus `<fake><global-img><image>×64<fake>` at
the end. The 960×504 sample becomes 4×3 tiles + global = **13 crops / 832 tokens**,
matching the official processor item by item.

The tokenizer is SmolLM2's `Digits + ByteLevel/GPT-2 regex`, which is why
`core/tokenizer` grew a third mode (`Gpt2Regex`); it differs from the Qwen pattern in
whether punctuation may prefix a word run and whether CR/LF glues to punctuation.
Three test strings (specials, contractions, digits, tabs, repeated spaces) tokenize
**identically** to the official tokenizer.

Parity: `python scripts/ref_smolvlm.py --model models/smolvlm-256m --image photo.jpg
--out ref_smolvlm` plus the engine with `--dump-dir ref_smolvlm`. Measured: pixels corr
**0.99997** (maxdiff 0.046, from rounding the intermediate back to u8 before the second
resize), last-position logits corr **0.99991** (maxdiff 0.23) and a **byte-identical**
answer (" There are five persons in the image.").

Speed (CPU, 19 threads, f16 weights, the 960x504 sample):

| setting | crops | end to end |
|---|---|---|
| `SMOLVLM_MAX_EDGE=2048` (the official preprocessing) | 4x3 tiles + global = 13 | 8.6 s |
| **default (longest edge capped at 1024)** | 2x2 + global = 5 | **3.8 s** |
| `SMOLVLM_MAX_EDGE=512` | 1x1 + global = 2 | 1.6 s |

All three give the same answer. The cost is entirely the tower's **full attention**:
one 1024-token tile is 2*1024^2*64*12 ~ 1.6 GFLOP per layer, ~19 GFLOP over 12 layers,
so time scales with the number of tiles.

One counter-intuitive result worth recording: batching all tiles into a single forward
(batch = crops) buys **nothing** on CPU (8.7 s -> 8.7 s) - it is FLOP bound, not launch
bound. The batched form is kept anyway (it is what a GPU benefits from) and `--dump-dir`
falls back to one crop at a time so the stage-by-stage comparison with the reference
stays valid.

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

Instead of wedging this into llama.cpp's model registry, SkyVision.cpp owns the whole stack:
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
                       implementation lives here), architecture discovery, model discovery +
                       the single-resident slot (model_registry, shared by Web UI and MCP)
  models/moondream/    moondream3.1 backend (engine + vision tower + multi-crop preprocessing
                       + region head + server/batch)
  models/videochat3/   VideoChat3-4B backend (vision preprocessing + graphs)
  models/videomae/     VideoMAE video tower backend (clip preprocessing + space-time attention
                       + classifier / embedding)
  cli/                 unified front end (vqa-cli: inference / --serve Web UI / --mcp server)
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
- **Shared layer (`src/core`)**: no duplicated implementations across backends — `vqa.hpp`
  (interface), `backend.hpp` (CUDA-first bring-up + the thread default, which counts **physical**
  cores because SMT hurts these GEMV/small-GEMM shapes), `gguf_util.hpp` (metadata readers +
  tensor→f32), `weight_load.hpp` (mmap binding, 1D→host-F32 promotion, and the CPU zero-copy /
  GPU upload buffer assignment), `load_util.hpp` (LoadOptions plumbing, the per-backend
  `<PREFIX>_DUMP_DIR` override, and chat-template token id resolution), `vit_block.hpp` (the one ViT block body, shared by moondream / VideoChat3 /
  VideoMAE, differences via hooks), `resample.hpp` (smart_resize, PIL-semantics antialias
  bicubic resize, bilinear position-table interpolation — these used to live inside the
  videochat3 directory and get include'd back out of it by three other backends),
  `attention.hpp` (KV cache + masks), `stage_buf.hpp` (device-side staging), `ggml_dump.hpp`
  (parity dumps), `sampler.hpp`, `tokenizer.cpp`, `media_io.cpp`.
- **Tokenizer data**: `scripts/gen_unicode_ranges.py` regenerates `src/core/unicode_ranges.inc`.

## License

MIT — see [LICENSE](LICENSE). ggml is MIT (ggml-org). Model weights and tokenizers are subject
to their own terms and are downloaded separately.
