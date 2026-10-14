@echo off
rem Build moondream.cpp with ggml CUDA backend via Ninja.
rem CUDA 13.0 nvcc is incompatible with VS18 MSVC 14.51 (cudafe++ crashes) —
rem use the VS2022 BuildTools environment + MSVC 14.44 cl.exe as CUDA host compiler
rem (same trick as D:\S2TT\audio.cpp\build\configure-cuda-nano.bat).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=C:\Users\41601\AppData\Local\Programs\Python\Python39\Scripts;%PATH%
cd /d D:\moondream.cpp
if not exist build-cuda mkdir build-cuda
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF ^
  -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_GRAPHS=ON ^
  -DCMAKE_CUDA_HOST_COMPILER="C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" ^
  -DCMAKE_CUDA_FLAGS=-Xcompiler=/utf-8 -DCMAKE_CXX_FLAGS=/utf-8 %*
cmake --build build-cuda --target moondream-cli %*
