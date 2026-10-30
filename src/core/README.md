# src/core - the shared engine layer

Everything here exists for one reason: **two or more backends needed the same
code and the same numbers**. Anything that only one architecture needs lives in
`src/models/<arch>/` instead. When you are tempted to copy a loop into a second
backend, extract it here (the "pull it in" checklist is at the bottom).

```
src/core/
  vqa.hpp                 the public Model interface + capability table (the only
                          header at the root: every front end includes this)
  engine/                 ggml plumbing
    backend.hpp/.cpp        device + thread bring-up (CUDA if built, else CPU)
    weight_load.hpp         GGUF tensor binding: mmap zero-copy, 1D -> host F32,
                            CPU/GPU buffer ownership
    gguf_util.hpp           GGUF metadata readers (typed, with defaults)
    ggml_dump.hpp           parity dumps for named tensors (sink protocol)
    attention.hpp           KV-cache allocation + causal/windowed masks
    sampler.hpp             greedy / top-k / nucleus with a caller-held seed
    stage_buf.hpp           persistent on-device staging tensor shared by graphs
  blocks/                 reusable graph bodies
    vit_block.hpp           pre-norm ViT block (LN -> FA -> wo -> MLP) + the
                            generic pair-rotation rope the vision towers share
    llm_block.hpp           Qwen-style causal decoder layer
    qwen2vl_tower.hpp       whole Qwen2.5-VL vision tower (qwen2vl + qwen25omni)
  media/                  pixels, audio, video frames
    media_io.hpp/.cpp       image/video decode -> vqa::MediaInput
    jpeg_decode.hpp/.cpp    libjpeg-turbo path (bit-exact vs PIL) when built
    resample.hpp/.cpp       PIL-semantics resize / crop / position tables
    audio_feats.hpp         FFT, windows, mel filterbanks, log-mel
    video_pipe.cpp          ffmpeg pipe frame sampling
  text/                   text-side data
    tokenizer.hpp/.cpp      byte-level BPE (Qwen2/Qwen3 regex pre-tokenizer)
    utf8.hpp/.cpp           UTF-8 codec + the GPT-2 byte<->unicode alphabet
    json.hpp                small recursive-descent JSON value/parser
    json_scan.hpp/.cpp      key scans over a JSON blob (no DOM) for HF configs
    unicode_ranges.inc      generated tables - scripts/gen_unicode_ranges.py
  runtime/                model lifecycle + front-end glue
    model_registry.hpp/.cpp which GGUFs this build can run, tokenizer resolution,
                            single resident model slot
    vqa_registry.cpp        arch name -> factory table (the only file that knows
                            the backend list)
    backend_factories.hpp   factory declarations (kept apart so vqa.hpp stays
                            free of the backend list)
    load_util.hpp           LoadOptions -> members, <PREFIX>_DUMP_DIR, special ids
    vram_guard.cpp          device memory probe + requirement estimate
    http_server.hpp         minimal HTTP/1.1 for the embedded Web UI
  test/                   header-only harnesses for the tools/ test drivers
    seg_test.hpp            seg-test drivers' shared harness (seeded LCG, weight
                            workspace, graph runner, cpp_*.bin dump convention)
  util/                   pure helpers, no ggml
    mmap.hpp/.cpp           read-only file map + whole-file read
    aligned_buf.hpp         64-byte aligned byte buffer
```

## Include convention

Every core header is included **root-relative from `src/`**:

```cpp
#include "core/vqa.hpp"
#include "core/engine/weight_load.hpp"
#include "core/blocks/vit_block.hpp"
#include "core/media/resample.hpp"
```

`CORE_INC` in CMakeLists.txt is `${CMAKE_SOURCE_DIR}/src`, so one include
directory covers the whole shared layer no matter how it is subdivided. A backend
keeps its own directory on the include path (`MD_INC`, `VC3_INC`, ...) and may
cross-reference a sibling backend by basename (e.g. vjepa2 reuses
`videomae_preprocess.hpp`) - that is deliberate: those two are shipped and
tested as a pair.

## Where does new code go?

| the code is ... | goes to |
|---|---|
| an interface the CLI / server / MCP see | `vqa.hpp` |
| ggml call that every backend needs (buffer, mask, dump, sampler) | `engine/` |
| a whole transformer block or tower that 2+ backends instantiate | `blocks/` |
| decoding a file format (image, JPEG, audio, video pipe) | `media/` |
| text: BPE, UTF-8, JSON | `text/` |
| model discovery, load preamble, device guard, HTTP UI | `runtime/` |
| plain C++ helper with no ggml and no domain knowledge | `util/` |
| anything an architecture needs alone | that backend's `src/models/<arch>/` |

## Pulling a shared layer in

The order that works (learned the hard way, see the header comments):

1. **Both copies must already produce the same numbers.** If they differ on
   purpose (mel for Qwen2.5-Omni vs Gemma4, say), keep them side by side in one
   header with the difference documented instead of "unifying" the float math.
2. **Diff the two implementations before merging**, not after: the duplicated
   copies drift in small guards (a missing `bias`, an eps, an off-by-one in a
   mask) and those are exactly what the shared version must not lose.
3. **Make the shared function take the union of flags explicitly** and default
   them to the old behavior, so untouched backends keep their op sequence
   bit-for-bit. `vit_block.hpp`'s hook struct is the reference example.
4. **Verify with the per-backend parity scripts** (`scripts/ref_*.py` +
   `--dump-dir`), then the smoke battery (`scripts/smoke.sh`).

## History

- v1: everything at `src/core/*.hpp` in one flat directory - fine at three
  backends, unreadable at fourteen.
- v2: split by responsibility (`engine/`, `blocks/`, `media/`, `text/`,
  `runtime/`, `util/`) with root-relative includes, and the old `vqa_util.hpp`
  grab-bag dissolved into `util/mmap.hpp`, `util/aligned_buf.hpp`,
  `text/utf8.hpp` and `text/json_scan.hpp`.
