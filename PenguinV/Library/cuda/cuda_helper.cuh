#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace Cuda
{
    bool isCudaSupported(); // returns true if there is any CUDA device in system

    void validateKernel(); // validates of last occured error in kernel function on device side
    void cudaCheck( cudaError error ); // validates cudaError value and throws an except if the value is not cudaSuccess
    bool cudaSafeCheck( cudaError error ); // validates cudaError and returns true if the error is cudaSuccess

    struct KernelParameters
    {
        KernelParameters( uint32_t threadsPerBlock_, uint32_t blocksPerGrid_ );

        uint32_t threadsPerBlock;
        uint32_t blocksPerGrid;
    };

    // Helper function which returns proper arguments for CUDA device kernel functions
    KernelParameters getKernelParameters( uint32_t size );

    cudaStream_t getCudaStream();
}
