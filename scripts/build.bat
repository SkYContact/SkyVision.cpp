@echo off
rem Build SkyVision.cpp (ggml CPU backend) with MSVC.
rem Override the toolchain with:  set VCVARS=<path to vcvars64.bat>
setlocal
cd /d "%~dp0.."

if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  echo [build] vcvars64.bat not found - set VCVARS to your Visual Studio environment script
  exit /b 1
)
call "%VCVARS%" >nul

if not exist build mkdir build
cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF %*
rem vqa-cli: unified front end   moondream-cli: moondream3.1 standalone binary
cmake --build build --config Release --target vqa-cli moondream-cli %*
