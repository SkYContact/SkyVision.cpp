# syntax=docker/dockerfile:1
#
# SkyVision.cpp — MCP server image.
#
#   docker build -t skyvision-mcp .                    # bundles the demo model
#   docker build --build-arg MODEL_REPO= -t skyvision-mcp:bare .   # no weights (CI)
#   docker run --rm -i skyvision-mcp                   # MCP over stdio, smolvlm bundled
#   docker run --rm -i -v ~/models:/models skyvision-mcp   # your own GGUF files instead
#
# Stage split, cheapest-to-invalidate last:
#
#   toolchain  apt packages            (cache mount keeps the apt lists)
#   ggml       git clone + ggml build  (depends only on GGML_REF: editing src/ never
#                                       reaches this stage, so 30+ s of C++ survives)
#   build      COPY . + our sources    (ccache mount makes this mostly relinking)
#   runtime    weights + binary
#
# Build weights go in only when MODEL_REPO is non-empty: CI builds without them
# (500 MB per run otherwise, and BuildKit cache mounts do not survive a fresh
# runner), release images build with them.

# ---------------------------------------------------------------- toolchain
FROM ubuntu:24.04 AS toolchain

# cache mounts keep the apt lists between builds instead of re-downloading them
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build g++ ccache ca-certificates

# ---------------------------------------------------------------- ggml
# Built as a standalone project (not add_subdirectory) so that its object files
# live in this stage: a source edit only invalidates the `build` stage below.
FROM toolchain AS ggml

ARG GGML_REF=master
WORKDIR /opt/ggml
RUN git clone --depth 1 --branch ${GGML_REF} https://github.com/ggml-org/ggml src \
 && rm -rf src/.git

# BUILD_SHARED_LIBS=OFF matters for the runtime stage: a dynamically linked
# vqa-cli dies with "cannot open shared object file libggml.so" (exit 127)
# inside an image that only copies the binary.
RUN --mount=type=cache,target=/ccache \
    CCACHE_DIR=/ccache \
    cmake -S src -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF \
        -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF -DGGML_OPENMP=OFF \
        -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
 && cmake --build build --target ggml ggml-base ggml-cpu \
 && for f in libggml.a libggml-base.a libggml-cpu.a; do \
        test -f "build/src/$f" || { echo "missing build/src/$f"; ls -l build/src; exit 1; }; \
    done \
 && mkdir -p /opt/ggml/prebuilt/lib /opt/ggml/prebuilt/include \
 && cp build/src/libggml*.a /opt/ggml/prebuilt/lib/ \
 && cp -r src/include/. /opt/ggml/prebuilt/include/ \
 && ls -l /opt/ggml/prebuilt/lib

# ---------------------------------------------------------------- engine
FROM ggml AS build

WORKDIR /src/SkyVision
COPY . /src/SkyVision

# CCACHE_DIR + the cache mount make a rebuild after a source edit take seconds
# for the unchanged sources instead of recompiling them
RUN --mount=type=cache,target=/ccache \
    CCACHE_DIR=/ccache \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DGGML_PREBUILT_DIR=/opt/ggml/prebuilt \
    && cmake --build build --target vqa-cli


# ---------------------------------------------------------------- runtime
FROM ubuntu:24.04

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates curl

# The bundled checkpoint is downloaded BEFORE the binary is copied in: this layer
# only depends on the ARGs below, so it stays cached across source edits instead
# of pulling 400 MB on every build. `--build-arg MODEL_REPO=` skips it entirely
# (the MCP server still answers list_models with an empty list).
ARG MODEL_BASE=https://huggingface.co
ARG MODEL_REPO=SkyContact/SmolVLM-256M-GGUF
ARG MODEL_REV=main
RUN --mount=type=cache,target=/root/.cache/model \
    if [ -z "${MODEL_REPO}" ]; then \
        echo "MODEL_REPO is empty: building without the bundled checkpoint"; \
        mkdir -p /models/smolvlm-256m; \
    else \
        mkdir -p /models/smolvlm-256m && cd /models/smolvlm-256m \
     && (test -f /root/.cache/model/smolvlm-256m-q8_0.gguf \
         || curl -fsSL -o /root/.cache/model/smolvlm-256m-q8_0.gguf \
              "${MODEL_BASE}/${MODEL_REPO}/resolve/${MODEL_REV}/smolvlm-256m-q8_0.gguf") \
     && (test -f /root/.cache/model/tokenizer.json \
         || curl -fsSL -o /root/.cache/model/tokenizer.json \
              "${MODEL_BASE}/${MODEL_REPO}/resolve/${MODEL_REV}/tokenizer.json") \
     && cp /root/.cache/model/smolvlm-256m-q8_0.gguf /root/.cache/model/tokenizer.json . \
     && ls -l; \
    fi

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
