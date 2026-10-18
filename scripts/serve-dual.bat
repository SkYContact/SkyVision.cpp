@echo off
rem Dual-process, dual-resident serving (roadmap §6.3-A).
rem
rem Two 8GB-class models do not fit one 12GB GPU at the same time, and the
rem single-slot registry would thrash a 60s+ swap on every switch. Instead run
rem TWO vqa-cli --serve processes on two ports: the big quantised model on the
rem GPU instance, the second one on a --cpu instance (system RAM):
rem
rem   instance 1  http://<host>:%GPU_PORT%   GPU    ask Moss-VL-Realtime-q4k (8.4G; a
rem                                          12G card holds it + the VRAM guard
rem                                          refuses anything that does not fit)
rem   instance 2  http://127.0.0.1:%CPU_PORT%  --cpu  ask qwen25omni-q8_0 (7.3G incl.
rem                                          talker; full --speak works on CPU)
rem
rem Prerequisites: ~10GB free VRAM for instance 1 (weights + KV headroom) and
rem ~16GB free system RAM for instance 2. Same models dir is fine - each
rem instance loads lazily on first ask, so it pins whichever model it serves.
rem
rem   SKYVISION_GPU_PORT  GPU instance port   (default 8080)
rem   SKYVISION_CPU_PORT  CPU instance port   (default 8081)
rem   SKYVISION_GPU_HOST  GPU instance bind   (default 0.0.0.0 = LAN visible)
rem   SKYVISION_CPU_HOST  CPU instance bind   (default 127.0.0.1)
rem   SKYVISION_BUILD     build dir to use (default: first of build-cuda / build
rem                       whose vqa-cli.exe exists; a CPU-only build makes both
rem                       instances CPU - install the CUDA build for a real GPU)
rem   SKYVISION_MODELS    models dir to scan  (default: <repo>\models)
setlocal EnableDelayedExpansion
pushd "%~dp0.." || exit /b 1

if not defined SKYVISION_GPU_PORT set "SKYVISION_GPU_PORT=8080"
if not defined SKYVISION_CPU_PORT set "SKYVISION_CPU_PORT=8081"
if not defined SKYVISION_GPU_HOST set "SKYVISION_GPU_HOST=0.0.0.0"
if not defined SKYVISION_CPU_HOST set "SKYVISION_CPU_HOST=127.0.0.1"
if not defined SKYVISION_MODELS set "SKYVISION_MODELS=%CD%\models"

rem shared-library builds keep ggml*.dll in <build>\bin; add every candidate
for %%B in (build-cuda build-ninja build) do (
  if exist "%%B\bin" set "PATH=%CD%\%%B\bin;!PATH!"
)

set "EXE="
if defined SKYVISION_BUILD (
  if exist "%SKYVISION_BUILD%\vqa-cli.exe" set "EXE=%CD%\%SKYVISION_BUILD%\vqa-cli.exe"
) else (
  for %%B in (build-cuda build-ninja build) do (
    if not defined EXE if exist "%%B\vqa-cli.exe" set "EXE=%CD%\%%B\vqa-cli.exe"
  )
)
if not defined EXE (
  echo [serve-dual] no vqa-cli.exe found - rebuild first ^(scripts\build-vqa.bat^)
  popd
  exit /b 1
)
for %%P in ("%EXE%") do set "PATH=%%~dpPbin;!PATH!"

echo [serve-dual] GPU instance: http://%SKYVISION_GPU_HOST%:%SKYVISION_GPU_PORT%  (%EXE%)
echo [serve-dual] CPU instance: http://%SKYVISION_CPU_HOST%:%SKYVISION_CPU_PORT%  (--cpu)
start "SkyVision GPU :%SKYVISION_GPU_PORT%" "%EXE%" --serve %SKYVISION_GPU_PORT% --host %SKYVISION_GPU_HOST% --models-dir "%SKYVISION_MODELS%"
timeout /t 2 /nobreak >nul
start "SkyVision CPU :%SKYVISION_CPU_PORT%" "%EXE%" --serve %SKYVISION_CPU_PORT% --host %SKYVISION_CPU_HOST% --cpu --models-dir "%SKYVISION_MODELS%"
echo [serve-dual] two server windows spawned; close them (or Ctrl+C inside) to stop.
popd
exit /b 0
