#include <cuda_runtime.h>
#include "../../Library/image_buffer.h"
#include "../../Library/image_exception.h"
#include "../../Library/image_function.h"
#include "../../Library/cuda/cuda_types.cuh"
#include "../../Library/cuda/cuda_helper.cuh"
#include "../../Library/cuda/image_function_cuda.cuh"
#include "unit_test_helper_cuda.cuh"

namespace
{
    // This function must run with thread count as 1
    __global__ void isEqualCuda( const uint8_t * image, uint8_t value, size_t width, size_t height, size_t * differenceCount )
    {
        const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const size_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if( x < width && y < height )
        {
            const size_t id = y * width + x;

            if( image[id] == value )
                atomicAdd( differenceCount, 1 );
        }
    };

    __global__ void isAnyEqualCuda( const uint8_t * image, uint8_t * value, size_t valueCount, size_t width, size_t height,
                                    size_t * differenceCount )
    {
        const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const size_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if( x < width && y < height )
        {
            const size_t id = y * width + x;

            bool equal = false;

            for( size_t i = 0; i < valueCount; ++i )
            {
                if( image[id] == value[i] )
                {
                    equal = true;
                    break;
                }
            }

            if( equal )
                atomicAdd( differenceCount, 1 );
        }
    };
};

namespace Unit_Test
{
    namespace Cuda
    {
        Bitmap_Image_Cuda::Image uniformImage( uint8_t value )
        {
            Bitmap_Image_Cuda::Image image( randomValue<size_t>( 1, 2048 ), randomValue<size_t>( 1, 2048 ) );

            image.fill( value );

            return image;
        }

        Bitmap_Image_Cuda::Image uniformImage()
        {
            return uniformImage( randomValue<uint8_t>( 256 ) );
        }

        Bitmap_Image_Cuda::Image uniformColorImage()
        {
            return uniformColorImage( randomValue<uint8_t>( 256 ) );
        }

        Bitmap_Image_Cuda::Image uniformColorImage( uint8_t value )
        {
            Bitmap_Image_Cuda::Image image( randomValue<size_t>( 1, 2048 ), randomValue<size_t>( 1, 2048 ), Bitmap_Image_Cuda::RGB );

            image.fill( value );

            return image;
        }

        Bitmap_Image_Cuda::Image blackImage()
        {
            return uniformImage( 0u );
        }

        Bitmap_Image_Cuda::Image whiteImage()
        {
            return uniformImage( 255u );
        }

        std::vector < Bitmap_Image_Cuda::Image > uniformImages( size_t images )
        {
            if( images == 0 )
                throw imageException( "Invalid parameter" );

            std::vector < Bitmap_Image_Cuda::Image > image;

            image.push_back( uniformImage() );

            image.resize( images );

            for( size_t i = 1; i < image.size(); ++i ) {
                image[i].resize( image[0].width(), image[0].height() );
                image[i].fill( randomValue<uint8_t>( 256 ) );
            }

            return image;
        }

        std::vector < Bitmap_Image_Cuda::Image > uniformImages( std::vector < uint8_t > intensityValue )
        {
            if( intensityValue.size() == 0 )
                throw imageException( "Invalid parameter" );

            std::vector < Bitmap_Image_Cuda::Image > image;

            image.push_back( uniformImage( intensityValue[0] ) );

            image.resize( intensityValue.size() );

            for( size_t i = 1; i < image.size(); ++i ) {
                image[i].resize( image[0].width(), image[0].height() );
                image[i].fill( intensityValue[i] );
            }

            return image;
        }

        bool verifyImage( const Bitmap_Image_Cuda::Image & image, uint8_t value )
        {
            Cuda_Types::_csize_t differenceCount( 0 );

            const size_t rowSize = image.rowSize();
            const size_t height = image.height();

            const ::Cuda::KernelParameters kernel = ::Cuda::getKernelParameters( rowSize, height );

            isEqualCuda<<< kernel.blocksPerGrid, kernel.threadsPerBlock >>>( image.data(), value, rowSize, height, differenceCount.data() );
            cudaError_t error = cudaGetLastError();
            if( error != cudaSuccess )
                throw imageException( "Failed to launch CUDA kernel" );

            return differenceCount.get() == rowSize * height;
        }

        bool verifyImage( const Bitmap_Image_Cuda::Image & image, const std::vector < uint8_t > & value )
        {
            Cuda_Types::_csize_t differenceCount( 0 );
            Cuda_Types::Array<uint8_t> valueCuda( value );

            const size_t rowSize = image.rowSize();
            const size_t height = image.height();

            const ::Cuda::KernelParameters kernel = ::Cuda::getKernelParameters( rowSize, height );

            isAnyEqualCuda<<< kernel.blocksPerGrid, kernel.threadsPerBlock >>>( image.data(), valueCuda.data(), valueCuda.size(), rowSize,
                                                                                height, differenceCount.data() );
            cudaError_t error = cudaGetLastError();
            if( error != cudaSuccess )
                throw imageException( "Failed to launch CUDA kernel" );

            return differenceCount.get() == rowSize * height;
        }
    };
};
