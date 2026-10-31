#!/bin/bash
# Cross-backend smoke battery - the nightly regression golden set.
#
# usage:
#   scripts/smoke.sh [binary]                # run everything, print outputs
#   scripts/smoke.sh [binary] --golden FILE  # byte-compare per section against FILE
#
# The golden file format matches this script's output: "== [n] label ==" header
# lines followed by the expected stdout. Compare with `diff` semantics; sections
# whose models are missing locally are SKIPPED (not failures).
#
# All checks run greedy (--temp 0) on CPU so results are reproducible across
# GPUs; note outputs are thread-count-dependent for some backends (fp16
# reduction order in flash-attn), so pin --threads if you need bit-stability.
set -u
BIN="${1:-build/Release/vqa-cli.exe}"; shift 2>/dev/null || true
GOLDEN=""; [ "${1:-}" = "--golden" ] && GOLDEN="$2"
OUT="$(mktemp -t smoke.XXXXXX)"

run() { # run <label> <golden-cmd...>
  local label="$1"; shift
  if "$@" > "$OUT.part" 2>/dev/null; then
    { echo "== $label =="; cat "$OUT.part"; } >> "$OUT"
  else
    { echo "== $label == (SKIPPED: command failed)"; } >> "$OUT"
  fi
}

cd "$(dirname "$0")/.."

[ -f models/smolvlm-f16.gguf ] && \
  run "[1] smolvlm text" "$BIN" -m models/smolvlm-f16.gguf -t models/smolvlm-256m \
      --ask "What is the capital of France?" -n 48 --cpu --temp 0 --threads 1
[ -f models/smolvlm-f16.gguf ] && \
  run "[2] smolvlm image" "$BIN" -m models/smolvlm-f16.gguf -t models/smolvlm-256m \
      --image test_image.png --ask "What do you see in this image?" -n 48 --cpu --temp 0 --threads 1
[ -f models/videochat3-4b-q8_0.gguf ] && \
  run "[3] vc3 text smoke" "$BIN" -m models/videochat3-4b-q8_0.gguf -t models/VideoChat3-4B \
      --ask "Reply with exactly one word: what color is grass?" -n 24 --cpu --temp 0 --threads 1
[ -f models/Moss-VL-Realtime-q4k.gguf ] && \
  run "[4] moss image" "$BIN" -m models/Moss-VL-Realtime-q4k.gguf -t models/Moss-VL-Realtime-q4k \
      --image test_image.png --ask "Describe this image." -n 96 --cpu --temp 0 --threads 1
[ -f models/qwen3.5-2b-q8_0.gguf ] || [ -f models/Qwen3.5-2B/qwen3.5-2b-q8_0.gguf ] && \
  run "[5] qwen3.5 text" "$BIN" -m models/Qwen3.5-2B/qwen3.5-2b-q8_0.gguf -t models/Qwen3.5-2B \
      --ask "What is the capital of France?" -n 48 --cpu --temp 0 --threads 1
# DiffusionGemma is a block-diffusion decoder, so there is no greedy path; the
# sampler still runs off a fixed built-in seed, which makes the answer byte-stable
# for byte-stable inputs. DG_STEPS caps the denoising passes per canvas (the 26B
# MoE otherwise runs up to 48 of them, and this smoke test runs --threads 1).
[ -f models/diffusiongemma-26b-a4b-it-q8_0.gguf ] && \
  DG_STEPS=8 run "[6] diffusiongemma text" "$BIN" -m models/diffusiongemma-26b-a4b-it-q8_0.gguf \
      -t models/diffusiongemma-26B-A4B-it --ask "What is the capital of France?" -n 64 \
      --cpu --threads 1

if [ -n "$GOLDEN" ]; then
  if diff "$GOLDEN" "$OUT"; then echo "SMOKE: PASS (byte-identical to $GOLDEN)"
  else echo "SMOKE: FAIL (diff above)"; exit 1; fi
else
  cat "$OUT"
  echo "SMOKE: done - eyeball the outputs or capture as a golden file"
fi
rm -f "$OUT" "$OUT.part"
