#pragma once

#include <algorithm>
#include <cstdlib>
#include <vector>
#include "../../../src/image_buffer.h"

// A bunch of functions to help writing unit tests
namespace Unit_Test
{
    namespace OpenCL
    {
        // Image size and ROI verification
        bool verifyImage( const penguinV::Image & image, uint8_t value );
        bool verifyImage( const penguinV::Image & image, const std::vector <uint8_t> & value );
        bool verifyImage( const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value );
        bool verifyImage( const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const std::vector <uint8_t> & value );
    }
}
