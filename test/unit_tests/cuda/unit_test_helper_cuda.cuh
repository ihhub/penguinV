#pragma once

#include <cstdlib>
#include <vector>
#include "../../../src/image_buffer.h"

// A bunch of functions to help writing unit tests
namespace Unit_Test
{
    namespace Cuda
    {
        // Image size and ROI verification
        bool verifyImage( const PenguinV_Image::Image & image, uint8_t value );
        bool verifyImage( const PenguinV_Image::Image & image, const std::vector < uint8_t > & value );
    }
}
