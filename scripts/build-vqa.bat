@echo off
rem Build the unified image/video VQA front end (vqa-cli) with MSVC + Ninja.
rem Extra cmake args can be appended, e.g.: scripts\build-vqa.bat -DGGML_CUDA=ON
setlocal
cd /d "%~dp0.."

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  echo [build-vqa] vcvars64.bat not found - edit this script for your VS install
  exit /b 1
)
call "%VCVARS%" >nul

set "NINJA=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if not exist "%NINJA%" set "NINJA=ninja"

rem MSVC: GGML_NATIVE does not map to /arch:AVX2, so the x86 flags have to be
rem spelled out - without them ggml is built for the SSE2 baseline (~44 GFLOPS
rem single thread instead of ~48; the bigger win is on the CUDA-less fp16 paths).
cmake -S . -B build-ninja -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF ^
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON ^
  %*
if errorlevel 1 exit /b 1

rem vqa-cli: the single unified front end (inference / --serve Web UI / --mcp server);
rem there is no separate moondream binary any more. Extra targets can be appended,
rem e.g.:  scripts\build-vqa.bat --target all
cmake --build build-ninja --target vqa-cli %*
