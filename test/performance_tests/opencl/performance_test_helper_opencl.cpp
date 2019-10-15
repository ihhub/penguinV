#include "performance_test_helper_opencl.h"
#include "../../../src/opencl/opencl_helper.h"

namespace Performance_Test
{
    namespace OpenCL_Helper
    {
        PenguinV::Image uniformImage( uint32_t width, uint32_t height )
        {
            return uniformImage( width, height, randomValue<uint8_t>( 256 ) );
        }

        PenguinV::Image uniformImage( uint32_t width, uint32_t height, uint8_t value )
        {
            PenguinV::ImageOpenCL image( width, height );

            image.fill( value );

            PenguinV::Image imageOut;
            imageOut.swap( image );

            return imageOut;
        }

        std::vector< PenguinV::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height )
        {
            std::vector < PenguinV::Image > image( count );

            for( std::vector< PenguinV::Image >::iterator im = image.begin(); im != image.end(); ++im )
                *im = uniformImage( width, height );

            return image;
        }
    }
}
