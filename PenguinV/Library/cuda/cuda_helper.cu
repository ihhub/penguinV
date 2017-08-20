#include <cuda_runtime.h>
#include "cuda_helper.cuh"
#include "../image_exception.h"

KernelParameters::KernelParameters( uint32_t threadsPerBlock_, uint32_t blocksPerGrid_ )
    : threadsPerBlock( threadsPerBlock_ )
    , blocksPerGrid  ( blocksPerGrid_ )
{
}

KernelParameters getKernelParameters( uint32_t size )
{
    static const uint32_t threadsPerBlock = 256;
    return KernelParameters( threadsPerBlock, (size + threadsPerBlock - 1) / threadsPerBlock );
}

void ValidateLastError()
{
    cudaError_t error = cudaGetLastError();
    if( error != cudaSuccess )
        throw imageException( "Failed to launch CUDA kernel" );
}

void cudaCheck( cudaError error )
{
    if( error != cudaSuccess ) {
        cudaGetLastError();

        char errorMessage[64];
        sprintf( errorMessage, "Failed to run CUDA function with error %d", error );

        throw imageException( errorMessage );
    }
}

bool cudaSafeCheck( cudaError error )
{
    const bool sucess = (error == cudaSuccess);
    if( !sucess )
        cudaGetLastError();

    return sucess;
}

bool IsCudaSupported()
{
    int deviceCount = 0;
    if( !cudaSafeCheck( cudaGetDeviceCount( &deviceCount ) ) )
        return false;

    return (deviceCount > 0);
}
