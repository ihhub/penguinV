#include "cuda_helper.cuh"
#include "../image_exception.h"

namespace Cuda
{
    bool isCudaSupported()
    {
        int deviceCount = 0;
        if( !cudaSafeCheck( cudaGetDeviceCount( &deviceCount ) ) )
            return false;

        return (deviceCount > 0);
    }

    void validateKernel()
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

    KernelParameters::KernelParameters( dim3 threadsPerBlock_, dim3 blocksPerGrid_ )
        : threadsPerBlock( threadsPerBlock_ )
        , blocksPerGrid  ( blocksPerGrid_ )
    {
    }

    KernelParameters getKernelParameters( uint32_t size )
    {
        static const uint32_t threadsPerBlock = 256;
        return KernelParameters( threadsPerBlock, (size + threadsPerBlock - 1) / threadsPerBlock );
    }

    KernelParameters getKernelParameters( uint32_t width, uint32_t height )
    {
        dim3 threadsPerBlock( 16, 16 );
        
        if( width > height ) {
            uint32_t increasedHeight = height * 2;

            while( (threadsPerBlock.y > 1) && (width >= increasedHeight) ) {
                increasedHeight <<= 1;
                threadsPerBlock.x <<= 1;
                threadsPerBlock.y >>= 1;
            }
        }
        else if( width < height ) {
            uint32_t increasedWidth = width * 2;

            while( (threadsPerBlock.x > 1) && (height >= increasedWidth) ) {
                increasedWidth <<= 1;
                threadsPerBlock.x >>= 1;
                threadsPerBlock.y <<= 1;
            }
        }

        const dim3 blocksPerGrid( (width  + threadsPerBlock.x - 1) / threadsPerBlock.x,
                                  (height + threadsPerBlock.y - 1) / threadsPerBlock.y );

        return KernelParameters( threadsPerBlock, blocksPerGrid );
    }

    cudaStream_t getCudaStream()
    {
        return 0;
    }
}
