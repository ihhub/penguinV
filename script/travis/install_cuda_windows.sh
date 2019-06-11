#!/bin/bash
set -eu -o pipefail

echo "Downloading CUDA toolkit 8"
CUDA_URL='https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_windows-exe'
wget -q $CUDA_URL -O ~/cuda_8.0.44_windows.exe
echo "Installing CUDA toolkit 8"
# ~/cuda_8.0.44_windows.exe -s compiler_8.0 cublas_8.0 cublas_dev_8.0 cudart_8.0 curand_8.0 curand_dev_8.0
cmd /C "~/cuda_8.0.44_windows.exe --help"

# export CUDA_DIR="%ProgramFiles/NVIDIA GPU Computing Toolkit/CUDA/v8.0";
# export PATH="$CUDA_DIR/bin:$CUDA_DIR/libnvvp:$PATH";
# nvcc -V