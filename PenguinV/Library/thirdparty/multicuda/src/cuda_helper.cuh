#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace multiCuda
{
    bool isCudaSupported(); // returns true if there is any CUDA device in system

    void validateKernel(); // validates of last occured error in kernel function on device side
    void cudaCheck( cudaError error ); // validates cudaError value and throws an except if the value is not cudaSuccess
    bool cudaSafeCheck( cudaError error ); // validates cudaError and returns true if the error is cudaSuccess

    // Simple structure which holds minimum list parameters required for kernel launch
    struct GridParameters
    {
        GridParameters();
        GridParameters( const dim3 & blocksPerGrid_, const dim3 & threadsPerBlock_ );

        dim3 blocksPerGrid;   // [1st parameter] number of blocks per grid (grid size in 3D)
        dim3 threadsPerBlock; // [2nd parameter] number of threads per block (block size in 3D)
    };

    // Structure which hold full list of parameters required for kernel launch
    struct KernelParameters
    {
        KernelParameters();
        KernelParameters( const GridParameters & gridParameters, size_t sharedMemorySize_ = 0u, cudaStream_t stream_ = 0 );
        KernelParameters( const dim3 & blocksPerGrid_, const dim3 & threadsPerBlock_, size_t sharedMemorySize_ = 0u, cudaStream_t stream_ = 0 );

        dim3 blocksPerGrid;      // [1st parameter] number of blocks per grid (grid size in 3D)
        dim3 threadsPerBlock;    // [2nd parameter] number of threads per block (block size in 3D)
        size_t sharedMemorySize; // [3rd parameter] shared memory size per block
        cudaStream_t stream;     // [4th parameter] stream in which kernel will be executed
    };

    // Helper function which returns calculated GridParameters structure for kernel to be executed on current CUDA device
    GridParameters getGridParameters( uint32_t sizeX ); // 1D
    GridParameters getGridParameters( uint32_t sizeX, uint32_t sizeY ); // 2D
    GridParameters getGridParameters( uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ ); // 3D

    // Helper function which returns calculated KernelParameters structure for kernel to be executed on current CUDA device
    KernelParameters getKernelParameters( uint32_t sizeX, size_t requiredSharedSize ); // 1D
    KernelParameters getKernelParameters( uint32_t sizeX, uint32_t sizeY, size_t requiredSharedSize ); // 2D
    KernelParameters getKernelParameters( uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, size_t requiredSharedSize ); // 3D

    size_t getSharedMemorySize( size_t requiredSize = 0u ); // validates required shared memory size with supported size on current active device
    cudaStream_t getCudaStream(); // returns current CUDA stream for active device within thread
}

// Macroses for easy kernel launch
#define launchKernel1D(function, sizeX, ...)                                                                                               \
    const multiCuda::KernelParameters & parameters = multiCuda::getKernelParameters(sizeX, 0u);                                            \
    function <<< parameters.blocksPerGrid, parameters.threadsPerBlock, parameters.sharedMemorySize, parameters.stream >>> ( __VA_ARGS__ ); \
    multiCuda::validateKernel();

#define launchKernel2D(function, sizeX, sizeY, ...)                                                                                        \
    const multiCuda::KernelParameters & parameters = multiCuda::getKernelParameters(sizeX, sizeY, 0u);                                     \
    function <<< parameters.blocksPerGrid, parameters.threadsPerBlock, parameters.sharedMemorySize, parameters.stream >>> ( __VA_ARGS__ ); \
    multiCuda::validateKernel();

#define launchKernel3D(function, sizeX, sizeY, sizeZ, ...)                                                                                 \
    const multiCuda::KernelParameters & parameters = multiCuda::getKernelParameters(sizeX, sizeY, sizeZ, 0u);                              \
    function <<< parameters.blocksPerGrid, parameters.threadsPerBlock, parameters.sharedMemorySize, parameters.stream >>> ( __VA_ARGS__ ); \
    multiCuda::validateKernel();
