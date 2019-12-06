#include "performance_test_helper_opencl.h"
#include "../../../src/opencl/opencl_helper.h"

namespace Performance_Test
{
    namespace OpenCL_Helper
    {
        penguinV::Image uniformImage( uint32_t width, uint32_t height )
        {
            return uniformImage( width, height, randomValue<uint8_t>( 256 ) );
        }

        penguinV::Image uniformImage( uint32_t width, uint32_t height, uint8_t value )
        {
            penguinV::ImageOpenCL image( width, height );

            image.fill( value );

            penguinV::Image imageOut;
            imageOut.swap( image );

            return imageOut;
        }

        std::vector<penguinV::Image> uniformImages( uint32_t count, uint32_t width, uint32_t height )
        {
            std::vector<penguinV::Image> image( count );

            for ( std::vector<penguinV::Image>::iterator im = image.begin(); im != image.end(); ++im )
                *im = uniformImage( width, height );

            return image;
        }
    }
}
