#include "png_image.h"
#include "../image_exception.h"

#ifndef PENGUINV_ENABLED_PNG_SUPPORT

namespace Png_Operation
{
    PenguinV_Image::Image Load( const std::string & path )
    {
        PenguinV_Image::Image image;

        Load( path, image );
        return image;
    }

    void Load( const std::string &, PenguinV_Image::Image & )
    {
        throw imageException( "PNG is not supported" );
    }

    void Save( const std::string &, const PenguinV_Image::Image & )
    {
        throw imageException( "PNG is not supported" );
    }

    void Save( const std::string &, const PenguinV_Image::Image &, uint32_t, uint32_t, uint32_t, uint32_t )
    {
        throw imageException( "PNG is not supported" );
    }
}

#else

#include <stdlib.h>
#include <png.h>

#include "../parameter_validation.h"

namespace Png_Operation
{
    PenguinV_Image::Image Load( const std::string & path )
    {
        if( path.empty() )
            throw imageException( "Incorrect file path for image file loading" );

        FILE * file = fopen( path.data(), "rb" );
        if( !file )
            return PenguinV_Image::Image();

        png_structp png = png_create_read_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
        if ( !png ) {
            fclose( file );
            return PenguinV_Image::Image();
        }

        png_infop info = png_create_info_struct( png );
        if ( !info ) {
            fclose( file );
            return PenguinV_Image::Image();
        }

        png_init_io( png, file );
        png_read_info( png, info );

        const uint32_t width    = static_cast<uint32_t>( png_get_image_width ( png, info ) );
        const uint32_t height   = static_cast<uint32_t>( png_get_image_height( png, info ) );
        const uint8_t colorType = png_get_color_type( png, info );
        const uint8_t bitDepth  = png_get_bit_depth ( png, info );

        if( bitDepth == 16u )
            png_set_strip_16( png );

        if( colorType == PNG_COLOR_TYPE_PALETTE )
            png_set_palette_to_rgb( png );

        // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth
        if( colorType == PNG_COLOR_TYPE_GRAY && bitDepth < 8u )
            png_set_expand_gray_1_2_4_to_8( png );

        if( png_get_valid( png, info, PNG_INFO_tRNS ) )
            png_set_tRNS_to_alpha( png );

        // These color types don't have an alpha channel then fill it with 0xff
        if( colorType == PNG_COLOR_TYPE_RGB || colorType == PNG_COLOR_TYPE_GRAY || colorType == PNG_COLOR_TYPE_PALETTE )
            png_set_filler( png, 0xFF, PNG_FILLER_AFTER );

        if( colorType == PNG_COLOR_TYPE_GRAY || colorType == PNG_COLOR_TYPE_GRAY_ALPHA )
            png_set_gray_to_rgb( png );

        png_read_update_info( png, info );

        uint8_t ** row_pointers = reinterpret_cast<uint8_t**>( malloc( sizeof( uint8_t* ) * height ) );

        const size_t rowByteCount = png_get_rowbytes( png, info );

        for( uint32_t y = 0; y < height; ++y )
            row_pointers[y] = reinterpret_cast<uint8_t*>( malloc( rowByteCount ) );

        png_read_image( png, row_pointers );

        const bool isGrayScale = (colorType == PNG_COLOR_TYPE_GRAY || colorType == PNG_COLOR_TYPE_GRAY_ALPHA);

        PenguinV_Image::Image image( width, height, isGrayScale ? PenguinV_Image::GRAY_SCALE : PenguinV_Image::RGB );

        uint8_t * outY = image.data();
        for( uint32_t y = 0; y < height; ++y, outY += image.rowSize() ) {
            const uint8_t * column = row_pointers[y];
            uint8_t * outX = outY;

            if ( isGrayScale ) {
                for( uint32_t x = 0; x < width; ++x, column += 4 )
                    *(outX++) = column[0];
            }
            else {
                for( uint32_t x = 0; x < width; ++x, column += 4 ) {
                    *(outX++) = column[2];
                    *(outX++) = column[1];
                    *(outX++) = column[0];
                }
            }
        }

        fclose( file );

        for( uint32_t y = 0; y < height; ++y )
            free( row_pointers[y] );
        free( row_pointers );

        png_destroy_read_struct( &png, &info, NULL );

        return image;
    }

    void Load( const std::string & path, PenguinV_Image::Image & raw )
    {
        raw = Load( path );
    }

    void Save( const std::string & path, const PenguinV_Image::Image & image )
    {
        Save( path, image, 0, 0, image.width(), image.height() );
    }

    void Save( const std::string & path, const PenguinV_Image::Image & image, uint32_t startX, uint32_t startY,
               uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( image, startX, startY, width, height );

        FILE * file = fopen( path.data(), "wb" );
        if( !file )
            throw imageException( "Cannot create file for saving" );

        png_structp png = png_create_write_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
        if( !png )
            throw imageException( "Cannot create file for saving" );

        png_infop info = png_create_info_struct( png );
        if( !info )
            throw imageException( "Cannot create file for saving" );

        png_init_io( png, file );

        const bool grayScaleImage = image.colorCount() == PenguinV_Image::GRAY_SCALE;

        // Output is 8 bit depth, Gray-Scale or RGB
        png_set_IHDR( png, info, width, height, 8, (grayScaleImage ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB), PNG_INTERLACE_NONE,
                      PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT );

        png_write_info( png, info );

        uint8_t ** row_pointers = reinterpret_cast<uint8_t**>( malloc( sizeof( uint8_t* ) * height ) );

        const size_t rowByteCount = png_get_rowbytes( png, info );

        for( uint32_t y = 0; y < height; ++y )
            row_pointers[y] = reinterpret_cast<uint8_t*>( malloc( rowByteCount ) );

        const uint8_t * outY = image.data() + startY * image.rowSize() + startX * image.colorCount();
        for( uint32_t y = 0; y < height; ++y, outY += image.rowSize() ) {
            uint8_t * column = row_pointers[y];
            const uint8_t * outX = outY;

            if ( grayScaleImage ) {
                for( uint32_t x = 0; x < width; ++x, ++column ) {
                    *column = *(outX++);
                }
            }
            else {
                for( uint32_t x = 0; x < width; ++x, column += 3 ) {
                    column[2] = *(outX++);
                    column[1] = *(outX++);
                    column[0] = *(outX++);
                }
            }
        }

        png_write_image( png, row_pointers );
        png_write_end( png, NULL );

        fclose( file );

        for( uint32_t y = 0; y < height; ++y )
            free( row_pointers[y] );
        free( row_pointers );

        png_destroy_write_struct( &png, &info );
    }
}

#endif
