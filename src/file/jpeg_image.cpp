#include "jpeg_image.h"
#include "../image_exception.h"

#ifndef PENGUINV_ENABLED_JPEG_SUPPORT

namespace Jpeg_Operation
{
    PenguinV_Image::Image Load( const std::string & path )
    {
        PenguinV_Image::Image image;

        Load( path, image );
        return image;
    }

    void Load( const std::string &, PenguinV_Image::Image & )
    {
        throw imageException( "JPEG is not supported" );
    }

    void Save( const std::string &, const PenguinV_Image::Image & )
    {
        throw imageException( "JPEG is not supported" );
    }

    void Save( const std::string &, const PenguinV_Image::Image &, uint32_t, uint32_t, uint32_t, uint32_t )
    {
        throw imageException( "JPEG is not supported" );
    }
}

#else

#error "No implementation for JPEG exists. Please do not set PENGUINV_ENABLED_JPEG_SUPPORT macro"

#endif
