@echo off
@setlocal EnableDelayedExpansion

if defined CUDA_PATH (
    set CUDA_PATH ""
)
set CUDA_PATH="%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\")
echo set CUDA_PATH=%CUDA_PATH%

for /r "%CUDA_PATH%" %%f in (*) do (
    set "p="
    if "%%~nxf"=="cl.h"       set p=%%~dpnxf
    if "%%~nxf"=="OpenCL.lib" set p=%%~dpnxf

    if defined p (
        echo !p!
    )
)
endlocal
