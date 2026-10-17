# syntax=docker/dockerfile:1
#
# SkyVision.cpp — MCP server image.
#
#   docker build -t skyvision-mcp .
#   docker run --rm -i skyvision-mcp            # MCP over stdio, smolvlm bundled
#   docker run --rm -i -v ~/models:/models skyvision-mcp   # your own GGUF files instead
#
# The image ships the 403 MB SmolVLM-256M q8 checkpoint so the server can answer
# `describe_image` / `ask_image` without anything mounted: a directory listing needs
# weights, so a weights-less image would only be able to answer `tools/list`.
# Bigger models (moondream3.1, VideoChat3, Qwen2.5-VL, MOSS-VL) go in through the
# volume mount — drop their .gguf next to a tokenizer.json under /models.
#
# Layer order is deliberate: everything that does not depend on our sources (apt,
# the ggml checkout, the model download) comes BEFORE `COPY .`, so editing source
# only re-runs the compile (through ccache, so mostly relinking).

FROM ubuntu:24.04 AS build

# cache mounts keep the apt lists between builds instead of re-downloading them
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build g++ ccache ca-certificates

WORKDIR /src/SkyVision

# CMakeLists does `add_subdirectory(ggml)`, so ggml has to live inside the repo tree
# (in a dev checkout it is a sibling clone and .dockerignore drops it). It does not
# depend on our code, so this layer survives source edits.
ARG GGML_REF=master
RUN --mount=type=cache,target=/root/.cache/ggml-git \
    rm -rf ggml \
 && git clone --depth 1 --branch ${GGML_REF} https://github.com/ggml-org/ggml ggml

COPY . /src/SkyVision

# BUILD_SHARED_LIBS=OFF matters for the runtime stage: ggml defaults to shared
# libraries on Linux, and a dynamically linked vqa-cli dies with "cannot open shared
# object file libggml.so" (exit 127) inside an image that only copies the binary.
# CCACHE_DIR + the cache mount make a rebuild after a source edit take seconds for
# the unchanged ggml objects instead of recompiling all of it.
ENV CCACHE_DIR=/ccache
RUN --mount=type=cache,target=/ccache \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF \
        -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF -DGGML_OPENMP=OFF \
        -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    && cmake --build build --target vqa-cli


FROM ubuntu:24.04

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates curl

# The bundled checkpoint is downloaded BEFORE the binary is copied in: this layer
# only depends on the ARGs below, so it stays cached across source edits instead of
# pulling 403 MB on every build.
ARG MODEL_BASE=https://huggingface.co
ARG MODEL_REPO=SkyContact/SmolVLM-256M-GGUF
ARG MODEL_REV=main
RUN --mount=type=cache,target=/root/.cache/model \
    mkdir -p /models/smolvlm-256m && cd /models/smolvlm-256m \
    && (test -f /root/.cache/model/smolvlm-256m-q8_0.gguf \
        || curl -fsSL -o /root/.cache/model/smolvlm-256m-q8_0.gguf \
             "${MODEL_BASE}/${MODEL_REPO}/resolve/${MODEL_REV}/smolvlm-256m-q8_0.gguf") \
    && (test -f /root/.cache/model/tokenizer.json \
        || curl -fsSL -o /root/.cache/model/tokenizer.json \
             "${MODEL_BASE}/${MODEL_REPO}/resolve/${MODEL_REV}/tokenizer.json") \
    && cp /root/.cache/model/smolvlm-256m-q8_0.gguf /root/.cache/model/tokenizer.json . \
    && ls -l

COPY --from=build /src/SkyVision/build/vqa-cli /usr/local/bin/vqa-cli

# fail the build here, not at the first MCP request, if the binary is not runnable
# (missing shared libs used to surface as an exit-127 from docker run). `grep -q` is
# last in the pipeline, so a non-runnable binary fails the step too.
RUN vqa-cli --help 2>&1 | grep -q "usage:"

# the official processor upscales every image to 2048 (= up to 16 crops); capping it
# at 1024 keeps the same answers for ~2.3x less work per request
ENV SMOLVLM_MAX_EDGE=1024

WORKDIR /models
ENTRYPOINT ["vqa-cli", "--mcp", "--models-dir", "/models"]
