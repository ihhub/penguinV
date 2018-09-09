#pragma once

#include <algorithm>
#include <cstdlib>
#include <vector>
#include "../../../src/opencl/image_buffer_opencl.h"

// A bunch of functions to help writing unit tests
namespace Unit_Test
{
    namespace OpenCL
    {
        // Generate images
        PenguinV_Image::Image uniformImage();
        PenguinV_Image::Image uniformImage( uint8_t value );
        PenguinV_Image::Image uniformRGBImage();
        PenguinV_Image::Image uniformRGBImage( uint8_t value );
        PenguinV_Image::Image blackImage();
        PenguinV_Image::Image whiteImage();
        std::vector < PenguinV_Image::Image > uniformImages( uint32_t images );
        std::vector < PenguinV_Image::Image > uniformImages( std::vector < uint8_t > intensityValue );

        // Image size and ROI verification
        bool verifyImage( const PenguinV_Image::Image & image, uint8_t value );
        bool verifyImage( const PenguinV_Image::Image & image, const std::vector < uint8_t > & value );
    }
}
