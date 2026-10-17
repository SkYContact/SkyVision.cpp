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

# syntax=docker/dockerfile:1
FROM ubuntu:24.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 https://github.com/ggml-org/ggml ggml
COPY . SkyVision
WORKDIR /src/SkyVision

# MSVC-style GGML_NATIVE does not exist here, but the x86 flags are what make the
# difference: the CPU backend vectorises the fp32 matmuls through them.
RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF -DGGML_OPENMP=OFF \
        -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    && cmake --build build --target vqa-cli


FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/SkyVision/build/vqa-cli /usr/local/bin/vqa-cli

# bundled demo model (override the repo/rev to pin or to use a mirror)
ARG MODEL_REPO=SkyContact/SmolVLM-256M-GGUF
ARG MODEL_REV=main
ARG MODEL_BASE=https://huggingface.co
RUN mkdir -p /models/smolvlm-256m && cd /models/smolvlm-256m \
    && curl -fsSL -o smolvlm-256m-q8_0.gguf \
         "${MODEL_BASE}/${MODEL_REPO}/resolve/${MODEL_REV}/smolvlm-256m-q8_0.gguf" \
    && curl -fsSL -o tokenizer.json \
         "${MODEL_BASE}/${MODEL_REPO}/resolve/${MODEL_REV}/tokenizer.json"

# the official processor upscales every image to 2048 (= up to 16 crops); capping it
# at 1024 keeps the same answers for ~2.3x less work per request
ENV SMOLVLM_MAX_EDGE=1024

WORKDIR /models
ENTRYPOINT ["vqa-cli", "--mcp", "--models-dir", "/models"]
