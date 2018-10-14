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
        // Image size and ROI verification
        bool verifyImage( const PenguinV_Image::Image & image, uint8_t value );
        bool verifyImage( const PenguinV_Image::Image & image, const std::vector < uint8_t > & value );
    }
}
