cmake_minimum_required(VERSION 3.16)
project(SkyVision.cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ggml only: the standalone https://github.com/ggml-org/ggml checkout (it also
# ships gguf). We do not depend on llama.cpp - only ggml + our own model layers.
#
# Two ways to provide it:
#   * in-tree (default): a clone at ./ggml, built here via add_subdirectory
#   * prebuilt: -DGGML_PREBUILT_DIR=<prefix> with lib/libggml*.a + include/,
#     which lets the Docker build compile ggml in an earlier stage so that a
#     source edit never rebuilds it (and CUDA builds keep using the in-tree path)
set(GGML_PREBUILT_DIR "" CACHE PATH
    "Use an already-built ggml (lib/libggml*.a + include/) instead of add_subdirectory(ggml)")
set(GGML_PREBUILT_CUDA OFF CACHE BOOL
    "The prebuilt ggml was compiled with CUDA (adds GGML_USE_CUDA for our sources)")

if (GGML_PREBUILT_DIR)
    get_filename_component(GGML_PREBUILT_DIR "${GGML_PREBUILT_DIR}" ABSOLUTE)
    set(GGML_INC ${GGML_PREBUILT_DIR}/include)
    # libggml.a (Linux) or ggml.lib (MSVC), both spellings are accepted
    set(_ggml_missing "")
    foreach (_lib ggml ggml-base ggml-cpu)
        set(_path "")
        foreach (_cand "${GGML_PREBUILT_DIR}/lib/lib${_lib}.a" "${GGML_PREBUILT_DIR}/lib/${_lib}.lib")
            if (EXISTS "${_cand}")
                set(_path "${_cand}")
                break()
            endif()
        endforeach()
        if (NOT _path)
            list(APPEND _ggml_missing "lib${_lib}.a")
        endif()
        set(_ggml_lib_${_lib} "${_path}")
    endforeach()
    if (_ggml_missing)
        message(FATAL_ERROR "GGML_PREBUILT_DIR=${GGML_PREBUILT_DIR} is missing ${_ggml_missing}")
    endif()
    find_package(Threads REQUIRED)
    set(_ggml_defs)
    if (GGML_PREBUILT_CUDA)
        list(APPEND _ggml_defs GGML_USE_CUDA)
    endif()
    foreach (_lib ggml-base ggml-cpu ggml)
        add_library(${_lib} STATIC IMPORTED GLOBAL)
        set_target_properties(${_lib} PROPERTIES
            IMPORTED_LOCATION "${_ggml_lib_${_lib}}"
            INTERFACE_INCLUDE_DIRECTORIES ${GGML_INC}
            INTERFACE_COMPILE_DEFINITIONS "${_ggml_defs}")
    endforeach()
    # static link order: the backends need ggml-base, and every consumer of either
    # one also needs the C++/thread/dl runtime ggml pulls in
    set_target_properties(ggml-base PROPERTIES
        INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS};m")
    set_target_properties(ggml-cpu PROPERTIES INTERFACE_LINK_LIBRARIES ggml-base)
    set_target_properties(ggml PROPERTIES INTERFACE_LINK_LIBRARIES ggml-base)
    message(STATUS "ggml: prebuilt at ${GGML_PREBUILT_DIR}")
else()
    # CUDA graph capture for static decode graphs: the standalone ggml checkout
    # defaults this OFF ("llama.cpp only"), which silently compiled the feature
    # out after the move off llama.cpp - enable it explicitly (a no-op for
    # CPU-only builds).
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "ggml: use CUDA graphs" FORCE)
    add_subdirectory(ggml)
    set(GGML_INC ${CMAKE_SOURCE_DIR}/ggml/include)
endif()

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

# GGML_INC was set by the ggml block above (in-tree or prebuilt)
set(STB_INC  ${CMAKE_SOURCE_DIR}/vendor/stb)   # vendored single-header image decoder
set(CORE_INC ${CMAKE_SOURCE_DIR}/src/core)
set(MD_INC   ${CMAKE_SOURCE_DIR}/src/models/moondream)
set(VC3_INC  ${CMAKE_SOURCE_DIR}/src/models/videochat3)
set(MV_INC   ${CMAKE_SOURCE_DIR}/src/models/moss_vl)
set(MAGE_INC ${CMAKE_SOURCE_DIR}/src/models/mage_vl)
set(VM_INC   ${CMAKE_SOURCE_DIR}/src/models/videomae)

# shared layer. media_io.cpp is the single translation unit owning the stb_image
# implementation; tokenizer.cpp owns the Unicode tables for the Qwen2
# pre-tokenizer (regenerate them with scripts/gen_unicode_ranges.py)
set(VQA_CORE_SRC
    src/core/vqa_util.cpp
    src/core/vqa_registry.cpp
    src/core/model_registry.cpp
    src/core/media_io.cpp
    src/core/video_pipe.cpp
    src/core/backend.cpp
    src/core/resample.cpp
    src/core/tokenizer.cpp)

set(MOONDREAM_SRC
    src/models/moondream/moondream.cpp
    src/models/moondream/moondream_vision.cpp
    src/models/moondream/moondream_preprocess.cpp
    src/models/moondream/moondream_region.cpp)

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

#   src/models/videomae/     VideoMAE / ViViT video tower (labels or embedding)
set(VIDEOMAE_SRC
    src/models/videomae/videomae_preprocess.cpp
    src/models/videomae/videomae_model.cpp)

#   src/models/qwen2vl/      Qwen2.5-VL backend (Qwen2 trunk + windowed ViT)
set(QWEN2VL_SRC
    src/models/qwen2vl/qwen2vl_model.cpp)

#   src/models/qwen25omni/   Qwen2.5-Omni thinker (Qwen2.5 trunk + VL tower + Whisper audio)
set(QWEN25OMNI_SRC
    src/models/qwen25omni/qwen25omni_model.cpp
    src/models/qwen25omni/qwen25omni_audio.cpp
    src/models/qwen25omni/qwen25omni_talker.cpp
    src/models/qwen25omni/qwen25omni_t2w.cpp)

#   src/models/smolvlm/      SmolVLM / Idefics3 backend (SigLIP + SmolLM2)
set(SMOLVLM_SRC
    src/models/smolvlm/smolvlm_model.cpp)

#   src/models/mage_vl/      Microsoft Mage-VL backend (Qwen3-4B text + Mage-ViT Codec-ViT;
#                            reuses vc3's resize/patchify primitives)
set(MAGE_VL_SRC
    src/models/videochat3/vc3_preprocess.cpp
    src/models/mage_vl/mage_model.cpp)

#   src/models/embedgemma2/  EmbeddingGemma-2 backend (Gemma text encoder + Gemma4
#                            vision/audio towers; embedding output)
set(EMBG2_SRC
    src/models/embedgemma2/embedgemma2_model.cpp)

# --- unified VQA front end (every backend in one binary) ---
add_executable(vqa-cli
    src/cli/vqa_cli.cpp
    src/cli/vqa_server.cpp
    src/cli/mcp_server.cpp
    ${VQA_CORE_SRC}
    ${MOONDREAM_SRC}
    ${VIDEOCHAT3_SRC}
    ${QWEN35_SRC}
    ${MOSS_VL_SRC}
    ${VIDEOMAE_SRC}
    ${QWEN2VL_SRC}
    ${SMOLVLM_SRC}
    ${MAGE_VL_SRC}
    ${QWEN25OMNI_SRC}
    ${EMBG2_SRC})
target_link_libraries(vqa-cli PRIVATE ggml ggml-cpu)
target_include_directories(vqa-cli PRIVATE ${GGML_INC} ${STB_INC} ${CORE_INC} ${MD_INC} ${VC3_INC} ${MV_INC} ${VM_INC} ${MAGE_INC} ${CMAKE_SOURCE_DIR}/src/models/qwen25omni)

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
add_executable(moondream-preprocess-test src/models/moondream/moondream_preprocess.cpp src/core/resample.cpp)
target_compile_definitions(moondream-preprocess-test PRIVATE PREPROCESS_TEST_MAIN)
target_link_libraries(moondream-preprocess-test PRIVATE ggml ggml-cpu)
target_include_directories(moondream-preprocess-test PRIVATE ${GGML_INC} ${STB_INC} ${MD_INC} ${CORE_INC})
