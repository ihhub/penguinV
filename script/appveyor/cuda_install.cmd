@echo off
echo Downloading CUDA toolkit 8
appveyor DownloadFile  https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_windows-exe -FileName cuda_8.0.44_windows.exe
echo Installing CUDA toolkit 8
cuda_8.0.44_windows.exe -s compiler_8.0 ^
                           cublas_8.0 ^
                           cublas_dev_8.0 ^
                           cudart_8.0 ^
                           curand_8.0 ^
                           curand_dev_8.0

if NOT EXIST "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\cudart64_80.dll" ( 
echo "Failed to install CUDA"
exit /B 1
)

set CUDA_TOOLKIT_ROOT_DIR=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0
set PATH=%CUDA_TOOLKIT_ROOT_DIR%\bin;%CUDA_TOOLKIT_ROOT_DIR%\libnvvp;C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin;%PATH%
echo "CUDA_ROOT=%CUDA_ROOT%"
nvcc -V
