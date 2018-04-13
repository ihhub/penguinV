#pragma once

#include <algorithm>
#include <cstdlib>
#include <vector>
#include "../../../src/cuda/image_buffer_cuda.cuh"

// A bunch of functions to help writing unit tests
namespace Unit_Test
{
    namespace Cuda
    {
        // Generate images
        Bitmap_Image_Cuda::Image uniformImage();
        Bitmap_Image_Cuda::Image uniformImage( uint8_t value );
        Bitmap_Image_Cuda::Image uniformColorImage();
        Bitmap_Image_Cuda::Image uniformColorImage( uint8_t value );
        Bitmap_Image_Cuda::Image blackImage();
        Bitmap_Image_Cuda::Image whiteImage();
        std::vector < Bitmap_Image_Cuda::Image > uniformImages( uint32_t images );
        std::vector < Bitmap_Image_Cuda::Image > uniformImages( std::vector < uint8_t > intensityValue );

        // Image size and ROI verification
        bool verifyImage( const Bitmap_Image_Cuda::Image & image, uint8_t value );
        bool verifyImage( const Bitmap_Image_Cuda::Image & image, const std::vector < uint8_t > & value );
    }
}
