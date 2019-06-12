@echo off

echo "Downloading CUDA toolkit 8"
set CUDA_URL='https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_windows-exe'
powershell -Command "Invoke-WebRequest %CUDA_URL% -OutFile cuda_8.0.44_windows.exe"
certutil -hashfile cuda_8.0.44_windows.exe e7d0b4da6d01c32a6f1997516820fdac
echo "Installing CUDA toolkit 8"
if EXIST cuda_8.0.44_windows.exe (
    cuda_8.0.44_windows.exe -s compiler_8.0 cublas_8.0 cublas_dev_8.0 cudart_8.0 curand_8.0 curand_dev_8.0
) else (
    echo "cuda_8.0.44_windows.exe does not exist"
)
echo "%ProgramFiles%"
if NOT EXIST "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\cudart64_80.dll" ( 
    echo "Failed to install CUDA"
    exit /B 1
)
