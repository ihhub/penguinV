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

#include <jpeglib.h>
#include <jerror.h>

namespace Jpeg_Operation
{
    PenguinV_Image::Image Load( const std::string & path )
    {
        PenguinV_Image::Image image;

        Load( path, image );
        return image;
    }

    void Load( const std::string & path, PenguinV_Image::Image & image )
    {
        if( path.empty() )
            throw imageException( "Incorrect parameters for Jpeg image loading" );

        FILE * file = fopen(path.data(), "rb");
        if ( file == nullptr )
            throw imageException( "Cannot open Jpeg image" );

        struct jpeg_decompress_struct info;
        struct jpeg_error_mgr error;
        info.err = jpeg_std_error( &error );
        jpeg_create_decompress( &info );

        jpeg_stdio_src( &info, file );
        jpeg_read_header( &info, TRUE );

        jpeg_start_decompress( &info );

        const uint8_t colorCount = ( info.num_components > 0 && info.num_components < 256 ) ? static_cast<uint8_t>( info.num_components ) : 1;
        image.setColorCount( colorCount );
        image.resize( info.output_width, info.output_height );

        uint8_t * line[1] = { nullptr };
        while (info.output_scanline < info.output_height) {
            line[0] = image.data() + image.rowSize() * info.output_scanline;
            jpeg_read_scanlines( &info, line, 1) ;
        }

        jpeg_finish_decompress(&info);
        jpeg_destroy_decompress(&info);
        fclose( file );
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
#endif
