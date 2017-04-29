#include "../../Library/image_buffer.h"
#include "../../Library/image_function.h"
#include "../../Library/cuda/cuda_types.cuh"
#include "../../Library/cuda/image_function_cuda.cuh"
#include "unit_test_helper_cuda.cuh"

namespace
{
    // This function must run with thread count as 1
    __global__ void isEqual( const uint8_t * image, uint8_t value, uint32_t size, uint8_t * equal )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( image[id] != value )
            *equal = false;
    };
};

namespace Unit_Test
{
    namespace Cuda
    {
        Bitmap_Image_Cuda::Image uniformImage( uint8_t value )
        {
            Bitmap_Image_Cuda::Image image( randomValue<uint32_t>( 1, 2048 ), randomValue<uint32_t>( 1, 2048 ) );

            image.fill( value );

            return image;
        }

        Bitmap_Image_Cuda::Image uniformImage()
        {
            return uniformImage( randomValue<uint8_t>( 256 ) );
        }

        Bitmap_Image_Cuda::Image blackImage()
        {
            return uniformImage( 0u );
        }

        Bitmap_Image_Cuda::Image whiteImage()
        {
            return uniformImage( 255u );
        }

        std::vector < Bitmap_Image_Cuda::Image > uniformImages( uint32_t images )
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
            Cuda_Types::_cbool equal( true );

            isEqual<<< max( image.width(), image.height() ), 1 >>>(image.data(), value, image.width() * image.height(), &equal);
            cudaError_t error = cudaGetLastError();
            if( error != cudaSuccess )
                throw imageException( "Failed to launch CUDA kernel" );

            return equal.get() != 0;
        }
    };
};
