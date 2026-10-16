cmake_minimum_required(VERSION 3.16)
project(SkyVision.cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ggml only: the standalone https://github.com/ggml-org/ggml checkout (it also
# ships gguf). We do not depend on llama.cpp - only ggml + our own model layers.
# CUDA graph capture for static decode graphs: the standalone ggml checkout
# defaults this OFF ("llama.cpp only"), which silently compiled the feature
# out after the move off llama.cpp - enable it explicitly (a no-op for
# CPU-only builds).
set(GGML_CUDA_GRAPHS ON CACHE BOOL "ggml: use CUDA graphs" FORCE)
add_subdirectory(ggml)

# our sources carry UTF-8 comments/strings; MSVC defaults to the ANSI codepage
# (only our targets, ggml is added above)
if (MSVC)
    add_compile_options(/utf-8)
endif()

# ---------------------------------------------------------------------------
# layout
#   src/core/                shared engine layer: interfaces, utils, media I/O,
#                            tokenizer, GGUF weight loading
#   src/models/moondream/    moondream3.1 backend
#   src/models/videochat3/   VideoChat3-4B backend (Qwen3 text + VideoChat3 ViT)
#   src/cli/                 unified image/video VQA front end
# ---------------------------------------------------------------------------

set(GGML_INC ${CMAKE_SOURCE_DIR}/ggml/include)
set(STB_INC  ${CMAKE_SOURCE_DIR}/vendor/stb)   # vendored single-header image decoder
set(CORE_INC ${CMAKE_SOURCE_DIR}/src/core)
set(MD_INC   ${CMAKE_SOURCE_DIR}/src/models/moondream)
set(VC3_INC  ${CMAKE_SOURCE_DIR}/src/models/videochat3)
set(MV_INC   ${CMAKE_SOURCE_DIR}/src/models/moss_vl)

# shared layer. media_io.cpp is the single translation unit owning the stb_image
# implementation; tokenizer.cpp owns the Unicode tables for the Qwen2
# pre-tokenizer (regenerate them with scripts/gen_unicode_ranges.py)
set(VQA_CORE_SRC
    src/core/vqa_util.cpp
    src/core/vqa_registry.cpp
    src/core/media_io.cpp
    src/core/video_pipe.cpp
    src/core/tokenizer.cpp)

set(MOONDREAM_SRC
    src/models/moondream/moondream.cpp
    src/models/moondream/moondream_vision.cpp
    src/models/moondream/moondream_preprocess.cpp
    src/models/moondream/moondream_region.cpp)

# core pieces the moondream engine itself needs (no registry, no other backend)
set(MOONDREAM_CORE_SRC
    src/core/media_io.cpp
    src/core/vqa_util.cpp
    src/core/tokenizer.cpp)

set(VIDEOCHAT3_SRC
    src/models/videochat3/vc3_preprocess.cpp
    src/models/videochat3/vc3_model.cpp)

#   src/models/qwen3_5/      Qwen3.5 backend (hybrid text + Qwen3-VL ViT; Marlin runs on it)
set(QWEN35_SRC
    src/models/qwen3_5/qwen3_5_preprocess.cpp
    src/models/qwen3_5/qwen3_5_model.cpp)

#   src/models/moss_vl/      MOSS-VL-Realtime backend (cross-attention text decoder)
set(MOSS_VL_SRC
    src/models/moss_vl/moss_vl_preprocess.cpp
    src/models/moss_vl/moss_vl_model.cpp)

# --- unified VQA front end (every backend in one binary) ---
add_executable(vqa-cli
    src/cli/vqa_cli.cpp
    src/cli/vqa_server.cpp
    ${VQA_CORE_SRC}
    ${MOONDREAM_SRC}
    ${VIDEOCHAT3_SRC}
    ${QWEN35_SRC}
    ${MOSS_VL_SRC})
target_link_libraries(vqa-cli PRIVATE ggml ggml-cpu)
target_include_directories(vqa-cli PRIVATE ${GGML_INC} ${STB_INC} ${CORE_INC} ${MD_INC} ${VC3_INC} ${MV_INC})

# --- moondream3.1 standalone binary (same engine, its own argument surface) ---
add_executable(moondream-cli
    src/models/moondream/moondream_cli.cpp
    ${MOONDREAM_SRC}
    ${MOONDREAM_CORE_SRC})
target_link_libraries(moondream-cli PRIVATE ggml ggml-cpu)
target_include_directories(moondream-cli PRIVATE ${GGML_INC} ${STB_INC} ${CORE_INC} ${MD_INC})

# --- vision encoder test target (milestone 6) ---
add_executable(moondream-vision-test src/models/moondream/moondream_vision.cpp src/core/vqa_util.cpp)
target_compile_definitions(moondream-vision-test PRIVATE VISION_TEST_MAIN)
target_link_libraries(moondream-vision-test PRIVATE ggml ggml-cpu)
target_include_directories(moondream-vision-test PRIVATE ${GGML_INC} ${STB_INC} ${CORE_INC} ${MD_INC})

# --- region head test target (milestone 7) ---
add_executable(moondream-region-test src/models/moondream/moondream_region.cpp src/core/vqa_util.cpp)
target_compile_definitions(moondream-region-test PRIVATE REGION_TEST_MAIN)
target_link_libraries(moondream-region-test PRIVATE ggml ggml-cpu)
target_include_directories(moondream-region-test PRIVATE ${GGML_INC} ${STB_INC} ${CORE_INC} ${MD_INC})

# --- multi-crop Lanczos preprocess test target (milestone 6 follow-up) ---
add_executable(moondream-preprocess-test src/models/moondream/moondream_preprocess.cpp)
target_compile_definitions(moondream-preprocess-test PRIVATE PREPROCESS_TEST_MAIN)
target_link_libraries(moondream-preprocess-test PRIVATE ggml ggml-cpu)
target_include_directories(moondream-preprocess-test PRIVATE ${GGML_INC} ${STB_INC} ${MD_INC})
