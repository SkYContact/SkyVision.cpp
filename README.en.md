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
| **VideoChat3-4B** | `videochat3` | image + video | Qwen3 text trunk (36 layers, GQA, per-head qk-norm) + VideoChat3's own 27-layer ViT (2D RoPE, 2×2 merge), built-in merger and multimodal projector |
| **Qwen3.5 family (incl. Marlin-2B)** | `qwen3_5` (legacy GGUF tag `marlin`) | image + video | Qwen3.5 hybrid trunk (18 gated-delta linear-attention + 6 full-attention layers, interleaved MRoPE) + Qwen3-VL style vision tower (learned absolute pos embed, 2D RoPE, per-temporal-step attention). Official `Qwen/Qwen3.5-*` checkpoints convert as-is (same structure, parity verified stage by stage); Marlin-2B is the video-captioning specialisation with two trained skills: caption (timestamped event list) and find (temporal grounding) |
| **VideoJev-4B (scoring)** | `videojev` | video | Qwen3.5-4B trunk (32 layers: 24 linear + 8 full, **16 key / 32 value heads in the linear layers**) + the same vision tower, but the last layer is **attention-only** (MLP and post-norm trimmed) and a trained cosine decision head replaces the lm head: one forward scores a fixed candidate list (temporal grounding = per-patch sigmoid, single-choice VQA = option softmax). No text generation at all — drive it with `--score <qs.json>` |
| **Mage-VL-4B** | `mage_vl` | image + video | Microsoft Research Asia's codec-native VLM: Mage-ViT Codec-ViT (24 layers, 16×16 patches, per-axis 4:6:6 T/H/W RoPE, per-frame block attention, blocks folded straight into the merger) + Qwen3-4B-Instruct-2507 text trunk (36 layers, GQA + qk-norm, same shape as VideoChat3). Official `microsoft/Mage-VL` checkpoints convert as-is; single-block images, per-frame `<X.X seconds>` timestamp labels on video (frame-sampling backend). First ggml implementation of this model |
| **MOSS-VL-Realtime (11B)** | `moss_vl` | image + video | OpenMOSS cross-attention architecture: Qwen3 text trunk (36 self layers + 12 pure cross-attn layers, tanh dual gating) + SigLIP tower (deepstack 8/16/24 + merger) + a learnable per-frame separator. Vision KV goes through a separate cross attention; interleaved MRoPE (planar 3-axis vision positions). Multi-turn sessions supported (`?session=`) |
| **VideoMAE (base)** | `videomae` | video (image = still clip) | vision-only tower, no text side: Conv3d tubelet patch embedding + fixed sin-cos position table (**no cls token**) + 12 full space-time attention layers; one forward pass returns a label distribution (fine-tunes such as Kinetics-400) or the pooled embedding (`MCG-NJU/videomae-base` is the pretraining release, no head) |
| **ViViT (b-16x2, video tower)** | `vivit` | video (image = still clip) | the same video tower (reuses the `videomae` backend): the only differences are a **cls token**, a **learned** position table, `gelu_fast` (tanh form), LN eps 1e-6, bias on all of q/k/v, and a shortest-edge-256 → crop-224 **bilinear** preprocessing with zero centering. 32 frames → 3136+1 tokens, one forward returns Kinetics-400 labels |
| **Qwen2.5-VL (3B)** | `qwen2vl` | image (video pending) | Qwen2 text trunk (36 layers, GQA 16:2, qkv bias, section MRoPE 16/24/24, SwiGLU) + Qwen2.5 ViT (32 layers, patch 14, temporal 2, 2×2 merge, **window attention** with a global layer every 8th) + merger. Dynamic resolution via `smart_resize` (multiples of 28); preprocessing is **CLIP mean/std**, bicubic, patches packed in **2×2 group order** |
| **Qwen2.5-Omni (3B)** | `qwen25omni` | image + audio in, text + speech out | full thinker-talker: Qwen2.5 text trunk (untied lm_head) + the Qwen2.5-VL tower (same window-attention graph as `qwen2vl`) + a Whisper-style audio tower (128-bin log-mel, **n_window=100 chunked convolution + chunked attention**, 25 tok/s) → **Talker** (24-layer dual-track LM: the text track consumes thinker hidden states, emits 25 Hz codec tokens) → **Token2Wav** (flow-matching DiT, RK4×10 steps + CFG + sway sampling; BigVGAN v2 anti-aliased activations, pure C++ host implementation). Audio input via `--audio` (ffmpeg decodes to 16 kHz mono). References: llama.cpp PR #13784 (thinker) + the HF transformers 5.20 sources (all semantics); mel/audio tower bit-aligned with the official implementation |
| **SmolVLM (256M, Idefics3)** | `smolvlm` | image | the smallest usable backend here: SigLIP-base tower (12 layers, patch 16, 512², learned positions, LN/attention/MLP all with bias, `gelu_tanh`) + **4×4 pixel shuffle** connector (12288→576) + SmolLM2-135M trunk (30 layers, no bias, GQA 9:3, rope 1e5). Official Idefics3 preprocessing: longest edge **2048** → pad to multiples of 512 → **512² tiles** (row-major) + one global view, 64 tokens per tile. 515 MB at f16 / ~270 MB at q8 |
| **EmbeddingGemma-2 (740M)** | `embgemma2` | image + video + audio + text (embeddings) | multimodal embedding model (no text generation): Gemma3 text tower (24 layers, d=512, GQA 4:1, interleaved SWA/full, **per-layer PLE**) + Gemma4 ViT (16 layers, patch 16, axial 2D RoPE, 3×3 position pooling) + Gemma4 audio tower (USM conformer: dual 3×3 stride-2 conv subsampling + 12 layers of macaron FFN×2 / LightConv1d / **chunked local attention with Transformer-XL relative position bias**, ClippableLinear clamps throughout). `--embed` emits 768-d vectors (Matryoshka truncation to 512/256/128/64), `--embed-preset` official task prefixes, `--embed-file` batch JSONL. All four modalities vs the fp32 HF reference: **0.99999+** |

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
| VideoChat3-4B text / image VQA | ✅ end to end + **stage-by-stage parity verified**: patch embedding rel 4e-4, the 27-layer ViT + merger + projector rel 0.026 (F16 weight-noise level), greedy output byte-identical to the official bf16 |
| Marlin-2B image/video VQA | ✅ end to end with **stage-by-stage parity verified** against the official transformers 5.20 reference (pixel values bitwise-identical, tower drift beats official bf16, logits 1.5e-2 vs f32); `--mode caption` / `--mode find` wired to the canonical trained prompts |
| VideoJev-4B candidate scoring | ✅ whole path wired + **verified against the official transformers 5.14 code** (`scripts/parity_videojev.py`; prompt token ids are byte-identical). Taking a CPU float32 reference as truth: the engine (f16) is within **6.8e-3** max probability error (q8_0: 5.9e-3), while the official repo's own bf16 run is off by 5.7e-2 — the engine tracks fp32 closer than the reference default. Choice head: 6.6e-6 vs 5.1e-5 |
| MOSS-VL-Realtime image/video QA | ✅ end to end + parity against the official stack: image prefill token stream 24/24 and video 78/78 bitwise-identical, layer-2 cross-attn intermediates corr ≥ 0.999, KV readback vs the official roped K corr 0.9997. CUDA (5070 Ti Laptop, Q4_K): 213 ms/image vision, 67 tok/s decode; CPU 16 threads 8.9 tok/s. Multi-turn via `/ask?session=<id>` (`reset=1` clears), realtime stream via `rt/start?session=&src=&interval=` (`rt/say`, `rt/events`, `rt/stop`) |
| Mage-VL image/video QA | ✅ end to end + per-stage vision-tower parity; greedy answers match the official transformers oracle. CUDA (3060 6 GB, Q6_K): 162 ms/image vision, 63 tok/s decode, weights + KV fit a 6 GB card. `--video` with per-frame `<X.X seconds>` labels; the streammind gate ships as `--mode gate` / `--mode gated[:τ]` (the `mage-gate-f16.gguf` from `convert/convert_mage_gate.py` auto-loads next to the model) |
| Qwen2.5-Omni image+audio → text+speech | ✅ thinker + talker: q8_0 is bit-identical to ggml-org's official Q8_0 tensor for tensor, logits corr 0.9928, audio tower corr 0.9992, answers match f16; `--audio` in, `--speak reply.wav` out (24 kHz wav). Quantization table in the Qwen2.5-Omni section |
| VideoChat3-4B video input | ✅ built-in `--video` (ffmpeg pipe; `FFMPEG=<path>` overrides the binary): frame sampling + timestamped prompts + joint attention over 4-frame temporal clips; the vision tower runs clip by clip (peak activations stay at one clip), and oversized images/videos are clamped to the same context budget |
| VideoMAE classification / features | ✅ single-GGUF tower (94M params, ~180 MB at f16): 16 uniform frames → center crop 224 → one full space-time attention pass → top-k labels; no tokenizer, no KV cache. **Parity verified** against an independent numpy recomputation from the GGUF weights: ~1e-3 relative over 12 layers, patch embedding 3e-3, block-0 attention 3e-4, all 400 Kinetics probabilities within 1e-4. Headless pretraining checkpoints return the pooled embedding (`--mode embed` dumps 768 floats as JSON). ~0.85 s per clip on CPU (q8_0 weights: 0.59 s) |
| ViViT-b/16x2 classification | ✅ same video tower (the `videomae` backend, arch `vivit`): the only differences are a learned **cls token**, a learned position table, `gelu_fast` (tanh form), LN eps 1e-6, bias on all of q/k/v and a bilinear 256→224 preprocessing with zero centering. 32 frames → 3136+1 tokens, Kinetics-400 labels. Parity vs `VivitForVideoClassification`: probabilities within 6e-3, identical top-5 order, ~2.2 s per clip on CPU |
| Qwen2.5-VL-3B | ✅ arch `qwen2vl`: Qwen2 text trunk (36L GQA 16:2, qkv bias, section MRoPE 16/24/24, SwiGLU) + Qwen2.5 ViT (32L, patch 14, temporal 2, 2×2 merge, **window attention** with a global layer every 8th) + merger, dynamic resolution through `smart_resize`. Preprocessing is CLIP mean/std, bicubic, patches in **2×2 group order**. Merged features corr 0.9935 vs the official tower, same answers ("There are five people in the image.") |
| SmolVLM-256M (Idefics3) | ✅ arch `smolvlm`, the smallest backend here: SigLIP-base tower (12L, patch 16, 512², learned positions, LayerNorm/attention/MLP all with bias, `gelu_tanh`) + **4×4 pixel shuffle** connector (12288→576) + SmolLM2-135M trunk (30L, no biases, GQA 9:3, rope 1e5). Idefics3 preprocessing: longest edge 2048 → pad to multiples of 512 → 512² tiles + one global view, 64 tokens per tile. 515 MB at f16 / ~270 MB at q8. Logits corr 0.99991 and byte-identical answers |
| EmbeddingGemma-2 (740M) | ✅ arch `embgemma2`, multimodal **embedding** (no text generation): Gemma3 text tower (24L d=512, GQA 4:1, interleaved SWA/full, per-layer PLE) + Gemma4 ViT (16L patch 16, axial 2D RoPE, 3×3 position pooling) + Gemma4 audio tower (USM conformer: dual 3×3 stride-2 conv subsampling + 12 layers of macaron FFN×2 / LightConv1d / chunk-12 local attention with Transformer-XL rel-shift bias; every linear is a ClippableLinear with finite clamps from the checkpoint). `--embed` emits 768-d vectors (Matryoshka truncation `--embed-dim`, official task prefixes `--embed-preset`, batch `--embed-file`). All four modalities vs the fp32 HF reference: text 0.9999999 / image 0.99999 / audio 0.99999 / video 0.9999998 |

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

# MOSS-VL-Realtime (image/video + multi-turn sessions)
vqa-cli --model Moss-VL-Realtime-q4k.gguf --image photo.jpg --ask "What is in this image?"
vqa-cli --model Moss-VL-Realtime-q4k.gguf --video clip.mp4 --max-frames 8 --ask "What happens?"   # video: per-frame timestamped prompts
# server-side multi-turn: POST /api/model/Moss-VL-Realtime-q4k/ask?q=...&session=abc (reset=1 clears)

# Marlin-2B (ModelScope mirror, no login)
modelscope download NemoStation/Marlin-2B --local_dir models/Marlin-2B

# Qwen3.5-2B (official, general image/video VQA)
modelscope download Qwen/Qwen3.5-2B --local_dir models/Qwen3.5-2B

# VideoJev-4B (scoring, no text generation; driven with --score)
hf download ZhuYuhan/VideoJev-4B --local-dir models/VideoJev-4B
python convert/convert_videojev.py --src models/VideoJev-4B -o videojev-4b-q8_0.gguf

# VideoMAE: the pretraining release (embedding only) or the Kinetics-400 fine-tune
hf download MCG-NJU/videomae-base --local-dir models/videomae-base
hf download MCG-NJU/videomae-base-finetuned-kinetics --local-dir models/videomae-base-ft
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
docker build -t skyvision-mcp .                        # bundles the 403 MB smolvlm
docker build --build-arg MODEL_REPO= -t skyvision-mcp:bare .   # no weights (what CI does)
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
`--build-arg MODEL_BASE=... --build-arg MODEL_REV=...`, or pass
`--build-arg MODEL_REPO=` to skip it entirely (a binary-only image that expects `-v`).

[`.github/workflows/docker-mcp.yml`](.github/workflows/docker-mcp.yml) builds this image on
a Linux runner and runs the smoke tests, so the badge reflects a real build rather than a
claim. Cost is split across three cacheable stages:

| Stage | Invalidated by | Note |
|---|---|---|
| `toolchain` | apt package list | apt lists come from a cache mount |
| `ggml` | `GGML_REF` only | built as its own stage and consumed through `-DGGML_PREBUILT_DIR`, so editing `src/` never rebuilds it |
| `build` | our sources | `COPY .` + ccache (helps repeated local builds; it cannot cross runners — a cache mount never leaves the builder and buildx has no host-bind flag) |

Push builds deliberately ship **without** weights (`MODEL_REPO=`): `--mount=type=cache` is
empty on a fresh runner, so bundling them would move 400 MB per run. The full image plus the
`describe_image` smoke test builds weekly and on `workflow_dispatch`, with a ghcr registry
cache (`mode=max`, not subject to the 10 GB Actions cache quota).

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
| `convert_qwen3_5.py` | shared by Qwen3.5 and Marlin-2B: `<checkpoint>-<outtype>.gguf` (hybrid trunk + Qwen3-VL tower, single file, `--outtype f16\|q8_0\|q6_k\|q4_0`, default q8_0; Marlin checkpoints get the `marlin.skills` bit automatically) |
| `convert_videojev.py` | `videojev-4b-<outtype>.gguf` (the cosine scoring head and special tokens travel inside the GGUF; `--outtype` defaults to q8_0) |
| `convert_mage_vl.py` | `mage-vl-4b-<outtype>.gguf` (Codec-ViT + Qwen3 trunk, single file, `--outtype f16\|q8_0\|q6_k\|q4_0`) |
| `convert_moss_vl.py` | `Moss-VL-Realtime-<nf4-out>.gguf` (the official NF4 self-layer weights are dequantized and requantized, default q4_k; `--nf4-out` / `--bf16-text-out` / `--vision-out` pick per-part precision) |
| `convert_videomae.py` | `videomae-<outtype>.gguf` (VideoMAE video tower, single file, `--outtype f16\|q8_0\|q4_0`; the classifier and its `id2label` table travel inside the GGUF) |
| `convert_qwen2vl.py` | `qwen2vl-<outtype>.gguf` (single file, `--outtype f16\|q8_0\|q4_0`, default f16) |
| `convert_qwen25omni.py` | `qwen25omni-<outtype>.gguf` (thinker) + `qwen25omni-talker-<outtype>.gguf` (talker + token2wav); `--outtype f16\|q8_0\|q4_0`, `--vision-f32`, `--no-talker` |
| `convert_smolvlm.py` | `smolvlm-<outtype>.gguf` (single file, `--outtype f16\|q8_0\|q4_0`, default f16) |
| `convert_embedgemma2.py` | `embeddinggemma2-<outtype>.gguf` (four-modality embedder, single file, `--outtype f16\|q8_0\|q4_0`; ~1.5 GB at f16) |

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
python convert/convert_qwen3_5.py --src models/Qwen3.5-2B   # → qwen3.5-2b-q8_0.gguf (2.67 GB, shared by Qwen3.5/Marlin)
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

### Qwen2.5-Omni

```bash
python convert/convert_qwen25omni.py --src models/qwen25omni/hf --outtype f16 --vision-f32
# → qwen25omni-f16.gguf (thinker) + qwen25omni-talker-f16.gguf (talker + token2wav + speaker embedding)
vqa-cli --model qwen25omni-f16.gguf --tokenizer models/qwen25omni/hf \
        --image photo.jpg --audio clip.wav --ask "What is shown?" \
        --speak reply.wav          # speech output (24 kHz wav, default voice Chelsie)
```

Thinker side: the text trunk shares the QWEN2VL graph (note the lm_head is **not tied** — the
converter writes `t.output.weight`); the audio tower differs from vanilla whisper in three ways,
all replicated from the official HF 5.20 implementation — the mel spectrogram is chunked into
n_window×2 = 200-frame blocks with **per-block convolution** (on the ggml side, 4 zero columns plus
two zeroed border columns after conv1 reproduce it exactly), intra-block attention, and frame-pair
(2i, 2i+1) average pooling (floor semantics, odd tail frame dropped). The three ingredients of
`ggml_conv_1d` (kernel bytes = torch `[OC,IC,K]` as-is, channel-contiguous data, output `[T,C]` with
the bias added after the transpose) are documented at the top of `qwen25omni_audio.cpp`.

Talker side: prefill is only 3 positions (`[h0+e0, bos+codec_pad, h1+e1+codec_bos]`); the prompt
acts only through the absolute rope offset; codec sampling temp 0.9 / top-k 40 / top-p 0.8 /
repetition penalty 1.05. The Token2Wav DiT runs as a ggml graph (AdaLN-zero, block-24 chunked
attention mask, RoPE); BigVGAN and the ECAPA speaker tower are host C++ (the grouped FIR of the
anti-aliased activations has no ggml operator; kaiser windows are computed from the HF formulas).
Known gap: speech is not yet aligned against HF (HF's token2wav hits a dtype problem on CPU bf16,
reference output pending), volume runs low, and BigVGAN's snake/fir semantics deserve a second pass
with the official fp32 weights.

Quantized builds:

| Precision | thinker / talker | Verification (multimodal ablation: test-1.jpeg + 4 s wav vs the HF bf16 ref) | Use |
|---|---|---|---|
| f16 | 11.2 GB / 1.66 GB (q8) | logits0 corr 0.9946; audio tower corr 0.9997; greedy answer matches HF | runs on 12 GB cards (with the q8 talker) |
| q8_0 | 5.64 GB / 1.66 GB | bit-identical to ggml-org's official Q8_0 **tensor for tensor**; logits corr 0.9928 (0.993 on both CPU and GPU); audio tower corr 0.9992; same answers as f16 | full feature set on CPU (incl. `--speak`); full feature set on 12 GB cards |
| q4_0 | 3.63 GB / 1.50 GB | logits corr 0.9475; audio tower corr 0.9714; a shorter phrasing of the same facts | full text→speech chain on 6 GB cards; full feature set on 12 GB |

Speech output (every quantization): after the token2wav fixes the wav rms is 0.16-0.31 (HF lands at
0.08-0.10); greedy decoding is a degenerate mode for both talkers (HF and this engine), and under
sampling (temp 0.9 / top-k 40 / top-p 0.8) EOS fires normally. End-to-end multimodal QA plus 7.9 s
of speech takes 54 s on a 12 GB card (q8, host vocoder parallelized).

On the GPU, every spot where the host reads weights directly (mel filters, BigVGAN/ECAPA/spk) must
go through `ggml_backend_tensor_get` — on the GPU `t->data` is a null pointer, and reading it
through `tensor_to_f32` will crash.

### EmbeddingGemma-2 (four-modality embeddings)

```bash
hf download google/embeddinggemma-2 --local-dir models/embeddinggemma-2
python convert/convert_embedgemma2.py --src models/embeddinggemma-2 -o embeddinggemma2-f16.gguf
vqa-cli --model embeddinggemma2-f16.gguf --tokenizer models/embeddinggemma-2 \
        --image photo.jpg --embed "describe" --embed-preset query     # JSON embedding
```

740M (text 270M + vision 170M + audio 300M); embeddings only, no generation. Highlights
(all measured against the reference, not inferred from docs):

- **Text tower**: Gemma3-style encoder — 24L d=512, GQA 4:1, interleaved sliding/full attention,
  **per-layer embeddings (PLE)** (12288→512 projection + RMSNorm per layer), mean pooling +
  projection to 768
- **Vision tower**: Gemma4 ViT, 16L patch 16, axial 2D RoPE (32 dims per axis, theta 100),
  3×3 position pooling. Soft budget: 280 for images, **140 per frame for video**; attention
  never crosses frames (frames run through the single-frame graph one at a time)
- **Audio tower**: USM conformer — 128-bin log-mel → dual 3×3 stride-2 conv subsampling →
  12 layers (macaron FFN ×2, LightConv1d with GLU + causal depthwise conv, chunk-12 local
  attention, softcap 50). Relative bias is the Transformer-XL rel-shift with a **per-head**
  rel-k projection; every linear is a **ClippableLinear** (finite input/output clamps travel
  in the GGUF)
- **Parity** (fp32 HF reference, f16 weights): text 0.9999999 / image 0.99999 /
  audio 0.99999 / video 0.9999998. The audio tower sat at 0.84 for a long time — the math
  (full-matrix + 13 diagonal masks, provably equivalent to rel-shift) was right; the real
  culprits were five ggml graph bugs found via per-layer hook dumps (commit 7af56f7), and the
  video path needed the rope position axis, the video preprocessing budget, the placeholder
  expansion and per-frame attention fixed (commit 1b8680a)
- Matryoshka truncation via `--embed-dim 512|256|128|64`; sentence-transformers task
  prefixes via `--embed-preset query|document|sts|classification|clustering`; batch mode
  `--embed-file` (one input per line → JSONL)
- **Performance** (f16, RTX 3060 Laptop 6GB / i7 8 threads, incl. ~1.2s weight upload):
  GPU text 1.2s, image 1.4s, 30s audio 1.5s, 32-frame video 3.9s; CPU image 4.0s /
  32-frame video 39s — roughly 10× faster on GPU. Flash attention end to end: CUDA fattn
  requires the KV length to be a multiple of 256 for head_dim 256/512, so the sequence is
  padded to a 256 multiple the standard way with the pad keys masked out (mean pooling
  counts real tokens only); GPU matches CPU to 0.999996+. Text n_ctx defaults to 8192
  (32 frames ≈ 3.9k tokens). On CPU, q8_0 weights are only ~8% faster — the bottleneck is
  f32 attention, not weight bandwidth

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
  models/qwen3_5/      Qwen3.5 / Marlin / VideoJev backend (gated-delta + full-attention hybrid
                       graph, caption/find skills, the scoring head)
  models/moss_vl/      MOSS-VL-Realtime backend (self+cross attention graphs, multi-turn and
                       realtime-stream sessions)
  models/mage_vl/      Mage-VL backend (Codec-ViT tower + System 1 gating)
  models/qwen2vl/      Qwen2.5-VL backend (window-attention ViT + 2×2 group-order patches)
  models/qwen25omni/   Qwen2.5-Omni backend (thinker/talker GGUFs + host Token2Wav)
  models/smolvlm/      SmolVLM-256M backend (Idefics3 tile preprocessing)
  models/embedgemma2/  EmbeddingGemma-2 backend (four-modality embeddings)
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
  VideoMAE, differences via hooks), `llm_block.hpp` (the Qwen-style causal decoder layer body,
  decoder-side sibling of `vit_block.hpp`: the hand-rolled per-layer decoder loops collapse into
  one graph builder — VideoChat3 / SmolVLM / MOSS-VL (its cross-attention layers ride the
  hooks.attention override) live here; null weight fields skip the corresponding op, and
  backend-only behavior hangs off BlockHooks), `qwen2vl_tower.hpp` (the whole Qwen2.5-VL vision
  tower shared op-for-op by the qwen2vl and qwen25omni engines: Conv3d patch embed, window
  attention with periodic full-attention layers, 4-section 2D M-RoPE, the merger; call-site
  differences are only cfg fields and nullable debug hooks), `resample.hpp` (smart_resize, PIL-semantics antialias
  bicubic resize, bilinear position-table interpolation — these used to live inside the
  videochat3 directory and get include'd back out of it by three other backends),
  `audio_feats.hpp` (audio front-end primitives — FFT / Hann windows / mel filterbanks /
  log-mel — where the Qwen2.5-Omni whisper-family and EmbedGemma-2 Gemma4-family implementations
  sit side by side, numerically not interchangeable by design and not to be "unified"),
  `attention.hpp` (KV cache + masks), `stage_buf.hpp` (device-side staging), `ggml_dump.hpp`
  (parity dumps), `sampler.hpp`, `tokenizer.cpp`, `media_io.cpp`.
- **Tokenizer data**: `scripts/gen_unicode_ranges.py` regenerates `src/core/unicode_ranges.inc`.

## License

MIT — see [LICENSE](LICENSE). ggml is MIT (ggml-org). Model weights and tokenizers are subject
to their own terms and are downloaded separately.
