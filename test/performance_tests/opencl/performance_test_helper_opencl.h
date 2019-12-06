#pragma once

#include <vector>
#include "../../../src/opencl/image_buffer_opencl.h"
#include "../performance_test_helper.h"

namespace Performance_Test
{
    namespace OpenCL_Helper
    {
        // Functions to generate images
        penguinV::Image uniformImage( uint32_t width, uint32_t height );
        penguinV::Image uniformImage( uint32_t width, uint32_t height, uint8_t value );
        std::vector< penguinV::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height );
    }
}
