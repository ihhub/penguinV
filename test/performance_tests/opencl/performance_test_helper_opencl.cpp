#include "performance_test_helper_opencl.h"
#include "../../../src/opencl/opencl_helper.h"

namespace Performance_Test
{
    namespace OpenCL_Helper
    {
        PenguinV_Image::Image uniformImage( uint32_t width, uint32_t height )
        {
            return uniformImage( width, height, randomValue<uint8_t>( 256 ) );
        }

        PenguinV_Image::Image uniformImage( uint32_t width, uint32_t height, uint8_t value )
        {
            PenguinV_Image::ImageOpenCL image( width, height );

            image.fill( value );

            PenguinV_Image::Image imageOut;
            imageOut.swap( image );

            return imageOut;
        }

        std::vector< PenguinV_Image::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height )
        {
            std::vector < PenguinV_Image::Image > image( count );

            for( std::vector< PenguinV_Image::Image >::iterator im = image.begin(); im != image.end(); ++im )
                *im = uniformImage( width, height );

            return image;
        }
    }
}
