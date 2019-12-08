#include "jpeg_image.h"
#include "../image_exception.h"

#ifndef PENGUINV_ENABLED_JPEG_SUPPORT

namespace Jpeg_Operation
{
    penguinV::Image Load( const std::string & path )
    {
        penguinV::Image image;

        Load( path, image );
        return image;
    }

    void Load( const std::string &, penguinV::Image & )
    {
        throw imageException( "JPEG is not supported" );
    }

    void Save( const std::string &, const penguinV::Image & )
    {
        throw imageException( "JPEG is not supported" );
    }

    void Save( const std::string &, const penguinV::Image &, uint32_t, uint32_t, uint32_t, uint32_t )
    {
        throw imageException( "JPEG is not supported" );
    }
}

#else

#error "No implementation for JPEG exists. Please do not set PENGUINV_ENABLED_JPEG_SUPPORT macro"

#endif
