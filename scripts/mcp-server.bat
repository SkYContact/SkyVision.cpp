@echo off
rem MCP stdio server launcher.
rem
rem MCP clients start this script instead of vqa-cli directly: a shared-library
rem build keeps ggml*.dll in <build>\bin, which is not on a client's PATH, and
rem the client also starts the process from an arbitrary working directory.
rem
rem   SKYVISION_BUILD   build dir to use (default: first of build-cuda / build-ninja
rem                     / build whose vqa-cli.exe actually knows --mcp)
rem   SKYVISION_MODELS  models dir to scan (default: <repo>\models)
rem
rem Any extra arguments are forwarded to vqa-cli (e.g. --model <gguf>).
setlocal EnableDelayedExpansion
pushd "%~dp0.." || exit /b 1

rem the DLLs must resolve before the --mcp probe below can even start the exe
for %%B in (build-cuda build-ninja build) do (
  if exist "%%B\bin" set "PATH=%CD%\%%B\bin;!PATH!"
)

set "EXE="
if defined SKYVISION_BUILD (
  if exist "%SKYVISION_BUILD%\vqa-cli.exe" set "EXE=%CD%\%SKYVISION_BUILD%\vqa-cli.exe"
) else (
  for %%B in (build-cuda build-ninja build) do (
    if not defined EXE if exist "%%B\vqa-cli.exe" (
      "%%B\vqa-cli.exe" --help 2>&1 | findstr /C:"--mcp" >nul && set "EXE=%CD%\%%B\vqa-cli.exe"
    )
  )
)
if not defined EXE (
  echo [mcp-server] no vqa-cli.exe with --mcp support found - rebuild it ^(scripts\build-vqa.bat^) 1>&2
  popd
  exit /b 1
)
for %%P in ("%EXE%") do set "PATH=%%~dpPbin;!PATH!"
if not defined SKYVISION_MODELS set "SKYVISION_MODELS=%CD%\models"

rem stdout belongs to JSON-RPC: nothing in this script may write to it
"%EXE%" --mcp --models-dir "%SKYVISION_MODELS%" %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
