@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\moondream.cpp
cmake --build build-cuda --config Release --target vqa-cli -j 8
