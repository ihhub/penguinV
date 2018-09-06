#include <cuda_runtime.h>
#include "../../../src/image_buffer.h"
#include "../../../src/image_exception.h"
#include "../../../src/image_function.h"
#include "../../../src/cuda/cuda_types.cuh"
#include "../../../src/cuda/cuda_helper.cuh"
#include "../../../src/cuda/image_function_cuda.cuh"
#include "../unit_test_helper.h"
#include "unit_test_helper_cuda.cuh"

namespace
{
    // This function must run with thread count as 1
    __global__ void isEqualCuda( const uint8_t * image, uint8_t value, uint32_t width, uint32_t height, uint32_t * differenceCount )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if( x < width && y < height )
        {
            const uint32_t id = y * width + x;

            if( image[id] == value )
                atomicAdd( differenceCount, 1 );
        }
    }

    __global__ void isAnyEqualCuda( const uint8_t * image, uint8_t * value, size_t valueCount, uint32_t width, uint32_t height,
                                    uint32_t * differenceCount )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if( x < width && y < height )
        {
            const uint32_t id = y * width + x;

            bool equal = false;

            for( uint32_t i = 0; i < valueCount; ++i )
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
    }

    PenguinV_Image::Image generateImage( uint32_t width, uint32_t height, uint8_t colorCount, uint8_t value )
    {
        PenguinV_Image::ImageCuda image( width, height, colorCount );

        image.fill( value );

        PenguinV_Image::Image imageOut;
        imageOut.swap( image );

        return imageOut;
    }
}

namespace Unit_Test
{
    namespace Cuda
    {
        PenguinV_Image::Image uniformImage( uint8_t value )
        {
            return generateImage( randomValue<uint32_t>( 1, 2048 ), randomValue<uint32_t>( 1, 2048 ), PenguinV_Image::GRAY_SCALE, value );
        }

        PenguinV_Image::Image uniformImage()
        {
            return uniformImage( randomValue<uint8_t>( 256 ) );
        }

        PenguinV_Image::Image uniformRGBImage()
        {
            return uniformRGBImage( randomValue<uint8_t>( 256 ) );
        }

        PenguinV_Image::Image uniformRGBImage( uint8_t value )
        {
            return generateImage( randomValue<uint32_t>( 1, 2048 ), randomValue<uint32_t>( 1, 2048 ), PenguinV_Image::RGB, value );
        }

        PenguinV_Image::Image blackImage()
        {
            return uniformImage( 0u );
        }

        PenguinV_Image::Image whiteImage()
        {
            return uniformImage( 255u );
        }

        std::vector < PenguinV_Image::Image > uniformImages( uint32_t images )
        {
            if( images == 0 )
                throw imageException( "Invalid parameter" );

            std::vector < PenguinV_Image::Image > image;

            image.push_back( uniformImage() );

            image.resize( images );

            for( size_t i = 1; i < image.size(); ++i ) {
                image[i] = image.front().generate( image[0].width(), image[0].height() );
                image[i].fill( randomValue<uint8_t>( 256 ) );
            }

            return image;
        }

        std::vector < PenguinV_Image::Image > uniformImages( std::vector < uint8_t > intensityValue )
        {
            if( intensityValue.size() == 0 )
                throw imageException( "Invalid parameter" );

            std::vector < PenguinV_Image::Image > image;

            image.push_back( uniformImage( intensityValue[0] ) );

            image.resize( intensityValue.size() );

            for( size_t i = 1; i < image.size(); ++i ) {
                image[i] = image.front().generate( image[0].width(), image[0].height() );
                image[i].fill( intensityValue[i] );
            }

            return image;
        }

        bool verifyImage( const PenguinV_Image::Image & image, uint8_t value )
        {
            multiCuda::Type<uint32_t> differenceCount( 0 );

            const uint32_t rowSize = image.rowSize();
            const uint32_t height = image.height();

            launchKernel2D( isEqualCuda, rowSize, height,
                            image.data(), value, rowSize, height, differenceCount.data() );

            return differenceCount.get() == rowSize * height;
        }

        bool verifyImage( const PenguinV_Image::Image & image, const std::vector < uint8_t > & value )
        {
            multiCuda::Type<uint32_t> differenceCount( 0 );
            multiCuda::Array<uint8_t> valueCuda( value );

            const uint32_t rowSize = image.rowSize();
            const uint32_t height = image.height();

            launchKernel2D( isAnyEqualCuda, rowSize, height,
                            image.data(), valueCuda.data(), valueCuda.size(), rowSize, height, differenceCount.data() );

            return differenceCount.get() == rowSize * height;
        }
    }
}
