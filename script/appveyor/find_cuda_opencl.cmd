@echo off
@setlocal EnableDelayedExpansion

if defined CUDA_PATH (
    set CUDA_PATH ""
)
set CUDA_PATH="%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\"
echo set CUDA_PATH=%CUDA_PATH%

rem dir %CUDA_PATH% /b /s /o:gn
endlocal
