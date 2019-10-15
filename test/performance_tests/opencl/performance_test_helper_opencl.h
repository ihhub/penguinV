#pragma once

#include <vector>
#include "../../../src/opencl/image_buffer_opencl.h"
#include "../performance_test_helper.h"

namespace Performance_Test
{
    namespace OpenCL_Helper
    {
        // Functions to generate images
        PenguinV::Image uniformImage( uint32_t width, uint32_t height );
        PenguinV::Image uniformImage( uint32_t width, uint32_t height, uint8_t value );
        std::vector< PenguinV::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height );
    }
}
