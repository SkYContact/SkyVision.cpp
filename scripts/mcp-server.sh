#!/bin/sh
# MCP stdio server launcher (POSIX: Linux/macOS/WSL).
#
# MCP clients start this script instead of vqa-cli directly so the build's shared
# libraries resolve no matter what working directory the client uses.
#
#   SKYVISION_BUILD   build dir (default: build-ninja is tried first, then build-cuda)
#   SKYVISION_MODELS  models dir to scan (default: <repo>/models)
#
# Extra arguments are forwarded to vqa-cli (e.g. --model <gguf>).
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

BUILD=${SKYVISION_BUILD:-}
if [ -z "$BUILD" ]; then
    for cand in build-ninja build-cuda build; do
        if [ -x "$cand/vqa-cli" ]; then BUILD=$cand; break; fi
    done
fi
if [ -z "$BUILD" ] || [ ! -x "$BUILD/vqa-cli" ]; then
    echo "[mcp-server] vqa-cli not found - build it first (scripts/build-vqa.bat or cmake --build)" >&2
    exit 1
fi
# shared-library builds keep libggml* under <build>/bin or <build> itself
LD_LIBRARY_PATH="$ROOT/$BUILD/bin:$ROOT/$BUILD:$ROOT/$BUILD/src:${LD_LIBRARY_PATH:-}"
DYLD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH DYLD_LIBRARY_PATH

MODELS=${SKYVISION_MODELS:-$ROOT/models}
exec "./$BUILD/vqa-cli" --mcp --models-dir "$MODELS" "$@"
