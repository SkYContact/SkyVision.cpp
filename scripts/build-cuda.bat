@echo off
rem Build SkyVision.cpp with the ggml CUDA backend via Ninja.
rem
rem CUDA 13.0 nvcc is incompatible with the VS18 MSVC 14.51 host compiler
rem (cudafe++ crashes), so this uses the VS2022 BuildTools environment and an
rem MSVC 14.44 cl.exe as the CUDA host compiler.
rem
rem Everything is overridable:
rem   set VCVARS=<vcvars64.bat>  set CUDA_HOST_CL=<cl.exe>  set CUDA_ARCH=<sm>
setlocal
cd /d "%~dp0.."

if not defined VCVARS set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  echo [build-cuda] vcvars64.bat not found - set VCVARS to your Visual Studio environment script
  exit /b 1
)
call "%VCVARS%" >nul

rem CUDA host compiler: newest MSVC shipped with the VS2022 BuildTools install
if not defined CUDA_HOST_CL (
  for /d %%v in ("C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*") do set "CUDA_HOST_CL=%%~fv\bin\Hostx64\x64\cl.exe"
)
if not exist "%CUDA_HOST_CL%" (
  echo [build-cuda] CUDA host compiler not found - set CUDA_HOST_CL to your cl.exe
  exit /b 1
)

set "NINJA=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if not exist "%NINJA%" set "NINJA=ninja"

rem CUDA_ARCH must match the target GPU. Common values:
rem   75 Turing (RTX 20) | 86 Ampere (RTX 30) | 89 Ada (RTX 40) | 120 Blackwell (RTX 50)
if not defined CUDA_ARCH set "CUDA_ARCH=89"

if not exist build-cuda mkdir build-cuda
cmake -S . -B build-cuda -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_OPENMP=OFF -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF ^
  -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH% ^
  -DGGML_CUDA_GRAPHS=ON -DCMAKE_CUDA_HOST_COMPILER="%CUDA_HOST_CL:\=/%" ^
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON ^
  -DCMAKE_CUDA_FLAGS=-Xcompiler=/utf-8 -DCMAKE_CXX_FLAGS=/utf-8 %*
rem vqa-cli is the single front end (inference / --serve / --mcp); the CPU side
rem still needs the explicit x86 flags because MSVC ignores GGML_NATIVE
cmake --build build-cuda --target vqa-cli %*
