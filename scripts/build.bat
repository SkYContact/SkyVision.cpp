@echo off
rem Build moondream.cpp (ggml CPU backend) with MSVC
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\moondream.cpp
if not exist build mkdir build
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF %*
cmake --build build --config Release --target moondream-cli %*
