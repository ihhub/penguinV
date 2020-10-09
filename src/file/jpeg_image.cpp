#include "jpeg_image.h"
#include "../penguin_v_exception.h"
#include "../parameter_validation.h"

namespace
{
    int jpegQuality = 100;
}

namespace Jpeg_Operation
{
    penguinV::Image Load( const std::string & path )
    {
        penguinV::Image image;

        Load( path, image );
        return image;
    }

    void Save( const std::string & path, const penguinV::Image & image )
    {
        Save( path, image, 0, 0, image.width(), image.height() );
    }

    void SetImageQuality( int quality )
    {
        if ( quality < 1 || quality > 100 )
            throw penguinVException( "JPEG quality value must be between 1 and 100" );

        jpegQuality = quality;
    }

    int GetImageQuality()
    {
        return jpegQuality;
    }
}

#ifndef PENGUINV_ENABLED_JPEG_SUPPORT

namespace Jpeg_Operation
{
    void Load( const std::string &, penguinV::Image & )
    {
        throw penguinVException( "JPEG is not supported" );
    }

    void Save( const std::string &, const penguinV::Image &, uint32_t, uint32_t, uint32_t, uint32_t )
    {
        throw penguinVException( "JPEG is not supported" );
    }
}

#else

#include <jerror.h>
#include <jpeglib.h>

namespace Jpeg_Operation
{
    void Load( const std::string & path, penguinV::Image & image )
    {
        if ( path.empty() )
            throw penguinVException( "Incorrect parameters for Jpeg image loading" );

        FILE * file = fopen( path.data(), "rb" );
        if ( file == nullptr )
            throw penguinVException( "Cannot open Jpeg image" );

        struct jpeg_decompress_struct info;
        struct jpeg_error_mgr error;
        info.err = jpeg_std_error( &error );
        jpeg_create_decompress( &info );

        jpeg_stdio_src( &info, file );
        jpeg_read_header( &info, TRUE );

        jpeg_start_decompress( &info );

        const uint8_t colorCount = ( info.num_components > 0 && info.num_components < 256 ) ? static_cast<uint8_t>( info.num_components ) : 1u;
        image.setColorCount( colorCount );
        image.resize( info.output_width, info.output_height );

        uint8_t * line[1] = {nullptr};
        while ( info.output_scanline < info.output_height ) {
            line[0] = image.data() + image.rowSize() * info.output_scanline;
            jpeg_read_scanlines( &info, line, 1 );
        }

        jpeg_finish_decompress( &info );
        jpeg_destroy_decompress( &info );
        fclose( file );
    }

    void Save( const std::string & path, const penguinV::Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height )
    {
        Image_Function::ValidateImageParameters( image, startX, startY, width, height );

        FILE * file = fopen( path.data(), "wb" );
        if ( !file )
            throw penguinVException( "Cannot create file for saving" );

        struct jpeg_compress_struct info;
        struct jpeg_error_mgr jerr;
        JSAMPROW row_pointer[1];

        info.err = jpeg_std_error( &jerr );
        jpeg_create_compress( &info );

        jpeg_stdio_dest( &info, file );

        info.image_width = width;
        info.image_height = height;
        info.input_components = image.colorCount();
        info.in_color_space = ( ( image.colorCount() == 1 ) ? JCS_GRAYSCALE : JCS_RGB );

        jpeg_set_defaults( &info );
        jpeg_set_quality( &info, jpegQuality, TRUE ); // limit to baseline-JPEG values

        jpeg_start_compress( &info, TRUE );

        while ( info.next_scanline < info.image_height ) {
            row_pointer[0] = const_cast<uint8_t *>( image.data() + image.rowSize() * info.next_scanline ); // libjpeg is C based library so no choice
            jpeg_write_scanlines( &info, row_pointer, 1 );
        }

        jpeg_finish_compress( &info );

        fclose( file );

        jpeg_destroy_compress( &info );
    }
}
#endif
