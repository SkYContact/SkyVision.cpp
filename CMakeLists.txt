cmake_minimum_required(VERSION 3.16)
project(moondream.cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(llama.cpp/ggml)

add_executable(moondream-cli src/moondream.cpp src/moondream_vision.cpp src/moondream_preprocess.cpp src/moondream_region.cpp)
target_link_libraries(moondream-cli PRIVATE ggml ggml-cpu)
target_include_directories(moondream-cli PRIVATE ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include
                                                 ${CMAKE_SOURCE_DIR}/llama.cpp/vendor/stb
                                                 ${CMAKE_SOURCE_DIR}/src)

# --- vision encoder test target (milestone 6) ---
add_executable(moondream-vision-test src/moondream_vision.cpp)
target_compile_definitions(moondream-vision-test PRIVATE VISION_TEST_MAIN)
target_link_libraries(moondream-vision-test PRIVATE ggml ggml-cpu)
target_include_directories(moondream-vision-test PRIVATE ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include)

# --- region head test target (milestone 7) ---
add_executable(moondream-region-test src/moondream_region.cpp)
target_compile_definitions(moondream-region-test PRIVATE REGION_TEST_MAIN)
target_link_libraries(moondream-region-test PRIVATE ggml ggml-cpu)
target_include_directories(moondream-region-test PRIVATE ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include)

# --- multi-crop Lanczos preprocess test target (milestone 6 follow-up) ---
add_executable(moondream-preprocess-test src/moondream_preprocess.cpp)
target_compile_definitions(moondream-preprocess-test PRIVATE PREPROCESS_TEST_MAIN)
target_link_libraries(moondream-preprocess-test PRIVATE ggml ggml-cpu)
target_include_directories(moondream-preprocess-test PRIVATE ${CMAKE_SOURCE_DIR}/llama.cpp/ggml/include)
