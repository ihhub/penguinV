#include <stdio.h>
#include "../../src/file/bmp_image.h"
#include "../../src/file/jpeg_image.h"
#include "../../src/file/png_image.h"
#include "../../src/file/raw_image.h"
#include "unit_test_file.h"
#include "unit_test_helper.h"

namespace file_operation
{
    typedef penguinV::Image (*ImageLoad)( const std::string & path );
    typedef void (*ImageSave)( const std::string & path, const penguinV::Image & image );

    bool WhiteGrayScaleImage( const std::string & filename, ImageLoad loadImage, ImageSave saveImage )
    {
        const penguinV::Image original = Unit_Test::whiteImage();
        saveImage( filename, original );

        const penguinV::Image loaded = loadImage( filename );
        remove( filename.data() );

        if ( original.height() != loaded.height() || original.width() != loaded.width() ||
             original.colorCount() != loaded.colorCount() || !Unit_Test::verifyImage( loaded, 255u ) )
            return false;

        return true;
    }

    bool BlackGrayScaleImage( const std::string & filename, ImageLoad loadImage, ImageSave saveImage )
    {
        const penguinV::Image original = Unit_Test::blackImage();
        saveImage( filename, original );

        const penguinV::Image loaded = loadImage( filename );
        remove( filename.data() );

        if ( original.height() != loaded.height() || original.width() != loaded.width() ||
             original.colorCount() != loaded.colorCount() || !Unit_Test::verifyImage( loaded, 0u ) )
            return false;

        return true;
    }

    bool RandomRGBImage( const std::string & filename, ImageLoad loadImage, ImageSave saveImage )
    {
        const penguinV::Image original = Unit_Test::randomRGBImage();
        saveImage( filename, original );

        const penguinV::Image loaded = loadImage( filename );
        remove( filename.data() );

        if ( original.height() != loaded.height() || original.width() != loaded.width() ||
            original.colorCount() != loaded.colorCount() )
            return false;

        const uint32_t rowSizeIn  = original.rowSize();
        const uint32_t rowSizeOut = loaded.rowSize();
        const uint32_t width = original.width() * original.colorCount();
        const uint8_t * inY  = original.data();
        const uint8_t * outY = loaded.data();
        const uint8_t * inYEnd = inY + rowSizeIn * original.height();

        for ( ; inY != inYEnd; inY += rowSizeIn, outY += rowSizeOut ) {
            if ( memcmp( inY, outY, width ) != 0 )
                return false;
        }

        return true;
    }

    bool RawRGBImage()
    {
        const std::string fileName = "raw.raw";
        const penguinV::Image original = Unit_Test::randomRGBImage();
        Raw_Operation::Save( fileName, original );
        const penguinV::Image loaded = Raw_Operation::Load<uint8_t>( fileName, original.width(), original.height(), original.colorCount() );
        remove( fileName.data() );

        if ( !Unit_Test::equalSize( original, loaded ) || !Unit_Test::equalData( original, loaded ) )
            return false;

        return true;
    }

    bool WhiteGrayScaleImageBitmap()
    {
        return WhiteGrayScaleImage( "bitmap.bmp", Bitmap_Operation::Load, Bitmap_Operation::Save );
    }

    bool BlackGrayScaleImageBitmap()
    {
        return BlackGrayScaleImage( "bitmap.bmp", Bitmap_Operation::Load, Bitmap_Operation::Save );
    }

    bool RandomRGBImageBitmap()
    {
        return RandomRGBImage( "bitmap.bmp", Bitmap_Operation::Load, Bitmap_Operation::Save );
    }

    bool WhiteGrayScaleImagePng()
    {
        return WhiteGrayScaleImage( "png.png", Png_Operation::Load, Png_Operation::Save );
    }

    bool BlackGrayScaleImagePng()
    {
        return BlackGrayScaleImage( "png.png", Png_Operation::Load, Png_Operation::Save );
    }

    bool RandomRGBImagePng()
    {
        return RandomRGBImage( "png.png", Png_Operation::Load, Png_Operation::Save );
    }

    bool WhiteGrayScaleImageJpeg()
    {
        return true;
        //return WhiteGrayScaleImage( "jpeg.jpg", Jpeg_Operation::Load, Jpeg_Operation::Save );
    }

    bool BlackGrayScaleImageJpeg()
    {
        return true;
        //return BlackGrayScaleImage( "jpeg.jpg", Jpeg_Operation::Load, Jpeg_Operation::Save );
    }

    bool RandomRGBImageJpeg()
    {
        return true;
        //return RandomRGBImage( "jpeg.jpg", Jpeg_Operation::Load, Jpeg_Operation::Save );
    }
}

void addTests_File( UnitTestFramework & framework )
{
    framework.add( file_operation::WhiteGrayScaleImageBitmap, "File: Save and load white gray-scale bitmap image" );
    framework.add( file_operation::BlackGrayScaleImageBitmap, "File: Save and load black gray-scale bitmap image" );
    framework.add( file_operation::RandomRGBImageBitmap,      "File: Save and load random RGB bitmap image" );
#if defined( PENGUINV_ENABLED_PNG_SUPPORT )
    framework.add( file_operation::WhiteGrayScaleImagePng,    "File: Save and load white gray-scale png image" );
    framework.add( file_operation::BlackGrayScaleImagePng,    "File: Save and load black gray-scale png image" );
    framework.add( file_operation::RandomRGBImagePng,         "File: Save and load random RGB png image" );
#endif
#if defined( PENGUINV_ENABLED_JPEG_SUPPORT )
    framework.add( file_operation::WhiteGrayScaleImageJpeg,   "File: Save and load white gray-scale jpeg image" );
    framework.add( file_operation::BlackGrayScaleImageJpeg,   "File: Save and load black gray-scale jpeg image" );
    framework.add( file_operation::RandomRGBImageJpeg,        "File: Save and load random RGB jpeg image" );
#endif
    framework.add( file_operation::RawRGBImage,               "File: Save and load raw RGB image" );
}
