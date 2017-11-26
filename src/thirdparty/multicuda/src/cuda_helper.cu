#include "cuda_device.cuh"
#include "cuda_helper.cuh"
#include "multicuda_exception.h"

namespace
{
    // Helper functions for internal calculations
    dim3 get2DThreadsPerBlock( uint32_t threadTotalCount )
    {
        uint32_t threadCount = 1;
        uint32_t overallThreadCount = (1 << 2);
        while( overallThreadCount <= threadTotalCount ) {
            threadCount <<= 1;
            overallThreadCount <<= 2;
        }

        return dim3( threadTotalCount / threadCount, threadCount );
    }

    dim3 get3DThreadsPerBlock( uint32_t threadTotalCount )
    {
        uint32_t threadCount = 1;
        uint32_t overallThreadCount = (1 << 3);
        while( overallThreadCount <= threadTotalCount ) {
            threadCount <<= 1;
            overallThreadCount <<= 3;
        }

        dim3 threadsPerBlock = get2DThreadsPerBlock( threadTotalCount / threadCount );
        threadsPerBlock.z = threadCount;

        return threadsPerBlock;
    }

    void adjust2DThreadsPerBlock( uint32_t threadTotalCount, const dim3 & blockDimension, dim3 & threadsPerBlock )
    {
        if( threadsPerBlock.y > blockDimension.y ) {
            threadsPerBlock.y = blockDimension.y;
            threadsPerBlock.x = threadTotalCount / threadsPerBlock.y;
        }

        if( threadsPerBlock.x > blockDimension.x ) {
            threadsPerBlock.x = blockDimension.x;
            if( blockDimension.y >= threadTotalCount / threadsPerBlock.x )
                threadsPerBlock.y = threadTotalCount / threadsPerBlock.x;
        }
    }

    void adjust3DThreadsPerBlock( uint32_t threadTotalCount, const dim3 & blockDimension, dim3 & threadsPerBlock )
    {
        if( threadsPerBlock.z > blockDimension.z ) {
            threadsPerBlock.z = blockDimension.z;
            threadsPerBlock.y = threadTotalCount / (threadsPerBlock.x * threadsPerBlock.z);
        }

        adjust2DThreadsPerBlock( threadTotalCount, blockDimension, threadsPerBlock );
    }
}

namespace multiCuda
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
        if( error != cudaSuccess ) {
            char errorMessage[64];
            sprintf( errorMessage, "Failed to launch CUDA kernel with error %d", error );
            throw multiCudaException( errorMessage );
        }
    }

    void cudaCheck( cudaError error )
    {
        if( error != cudaSuccess ) {
            cudaGetLastError();

            char errorMessage[64];
            sprintf( errorMessage, "Failed to run CUDA function with error %d", error );

            throw multiCudaException( errorMessage );
        }
    }

    bool cudaSafeCheck( cudaError error )
    {
        const bool sucess = (error == cudaSuccess);
        if( !sucess )
            cudaGetLastError();

        return sucess;
    }

    GridParameters::GridParameters()
        : blocksPerGrid  ( 0u )
        , threadsPerBlock( 0u )
    {
    }

    GridParameters::GridParameters( const dim3 & blocksPerGrid_, const dim3 & threadsPerBlock_ )
        : blocksPerGrid  ( blocksPerGrid_ )
        , threadsPerBlock( threadsPerBlock_ )
    {
    }

    KernelParameters::KernelParameters()
        : blocksPerGrid   ( 0u )
        , threadsPerBlock ( 0u )
        , sharedMemorySize( 0u )
        , stream          ( 0 )
    {
    }

    KernelParameters::KernelParameters( const GridParameters & gridParameters, size_t sharedMemorySize_, cudaStream_t stream_ )
        : blocksPerGrid   ( gridParameters.blocksPerGrid )
        , threadsPerBlock ( gridParameters.threadsPerBlock )
        , sharedMemorySize( sharedMemorySize_ )
        , stream          ( stream_ )
    {
    }

    KernelParameters::KernelParameters( const dim3 & blocksPerGrid_, const dim3 & threadsPerBlock_, size_t sharedMemorySize_, cudaStream_t stream_ )
        : blocksPerGrid   ( blocksPerGrid_ )
        , threadsPerBlock ( threadsPerBlock_ )
        , sharedMemorySize( sharedMemorySize_ )
        , stream          ( stream_ )
    {
    }

    GridParameters getGridParameters( uint32_t sizeX )
    {
        const CudaDevice & device = CudaDeviceManager::instance().device();
        const uint32_t threadTotalCount = device.threadsPerBlock();
        const dim3 & blockDimension = device.blockDimension();
        const dim3 & dimensionSize = device.dimensionSize();

        const uint32_t threadsPerBlockX = (threadTotalCount <= blockDimension.x) ? threadTotalCount : blockDimension.x;

        if( threadsPerBlockX * dimensionSize.x < sizeX )
            throw multiCudaException( "Input parameters for kernel execution are out of limits" );

        return GridParameters( (sizeX + threadsPerBlockX - 1) / threadsPerBlockX, threadsPerBlockX );
    }

    GridParameters getGridParameters( uint32_t sizeX, uint32_t sizeY )
    {
        const CudaDevice & device = CudaDeviceManager::instance().device();
        const uint32_t threadTotalCount = device.threadsPerBlock();
        const dim3 & blockDimension = device.blockDimension();
        const dim3 & dimensionSize = device.dimensionSize();

        dim3 threadsPerBlock = get2DThreadsPerBlock( threadTotalCount );
        adjust2DThreadsPerBlock( threadTotalCount, blockDimension, threadsPerBlock );

        if( (threadsPerBlock.x * dimensionSize.x < sizeX) || (threadsPerBlock.y * dimensionSize.y < sizeY) )
            throw multiCudaException( "Input parameters for kernel execution are out of limits" );

        const dim3 blocksPerGrid( (sizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                                  (sizeY + threadsPerBlock.y - 1) / threadsPerBlock.y );

        return GridParameters( blocksPerGrid, threadsPerBlock );
    }

    GridParameters getGridParameters( uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ )
    {
        const CudaDevice & device = CudaDeviceManager::instance().device();
        const uint32_t threadTotalCount = device.threadsPerBlock();
        const dim3 blockDimension = device.blockDimension();
        const dim3 & dimensionSize = device.dimensionSize();

        dim3 threadsPerBlock = get3DThreadsPerBlock( threadTotalCount );
        adjust3DThreadsPerBlock( threadTotalCount, blockDimension, threadsPerBlock );

        if( (threadsPerBlock.x * dimensionSize.x < sizeX) || (threadsPerBlock.y * dimensionSize.y < sizeY) || (threadsPerBlock.z * dimensionSize.z < sizeZ) )
            throw multiCudaException( "Input parameters for kernel execution are out of limits" );

        const dim3 blocksPerGrid( (sizeX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                                  (sizeY + threadsPerBlock.y - 1) / threadsPerBlock.y,
                                  (sizeZ + threadsPerBlock.z - 1) / threadsPerBlock.z );

        return GridParameters( blocksPerGrid, threadsPerBlock );
    }

    KernelParameters getKernelParameters( uint32_t sizeX, size_t requiredSharedSize )
    {
        return KernelParameters( getGridParameters( sizeX ), requiredSharedSize, getCudaStream() );
    }

    KernelParameters getKernelParameters( uint32_t sizeX, uint32_t sizeY, size_t requiredSharedSize )
    {
        return KernelParameters( getGridParameters( sizeX, sizeY ), requiredSharedSize, getCudaStream() );
    }

    KernelParameters getKernelParameters( uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, size_t requiredSharedSize )
    {
        return KernelParameters( getGridParameters( sizeX, sizeY, sizeZ ), requiredSharedSize, getCudaStream() );
    }

    size_t getSharedMemorySize( size_t requiredSize )
    {
        if( requiredSize == 0u )
            return 0u;

        if( CudaDeviceManager::instance().device().sharedMemoryPerBlock() < requiredSize )
            throw multiCudaException( "Requested shared memory size per block is bigger than supported memory size" );

        return requiredSize;
    }

    cudaStream_t getCudaStream()
    {
        const CudaDevice & device = CudaDeviceManager::instance().device();

        if( device.streamCount() > 0 )
            return device.stream().id();
        else
            return 0;
    }
}
