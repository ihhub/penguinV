@echo off
if "%CUDA_PATH%"=="" (
    echo CUDA_PATH is not defined
    set CUDA_PATH="%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\")

echo %CUDA_PATH%

for /r %CUDA_PATH% %%a in (*) do (
    set "p="
    if "%%~nxa"=="cl.h" set p=%%~dpnxa
    if "%%~nxa"=="OpenCL.lib" set p=%%~dpnxa

    if defined p (
        echo %p%
    ) else (
        echo %p% not found.
    )
)
