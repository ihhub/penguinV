#include <stdio.h>
#include "../../src/file/file_image.h"
#include "unit_test_file.h"
#include "unit_test_helper.h"

namespace file_operation
{
    bool WhiteGrayScaleImage()
    {
        std::vector<std::string> filenames;
        filenames.push_back( "bitmap.bmp" );
#ifdef PENGUINV_ENABLED_PNG_SUPPORT
        filenames.push_back( "png.png" );
#endif
        const PenguinV_Image::Image original = Unit_Test::whiteImage();

        for ( auto filename = filenames.begin(); filename != filenames.end(); ++filename ) {
            File_Operation::Save( filename->c_str(), original );

            const PenguinV_Image::Image loaded = File_Operation::Load( filename->c_str() );
            remove( filename->c_str() );

            if ( original.height() != loaded.height() || original.width() != loaded.width() || original.colorCount() != loaded.colorCount()
                 || !Unit_Test::verifyImage( loaded, 255u ) )
                return false;
        }
        return true;
    }

    bool BlackGrayScaleImage()
    {
        std::vector<std::string> filenames;
        filenames.push_back( "bitmap.bmp" );
#ifdef PENGUINV_ENABLED_PNG_SUPPORT
        filenames.push_back( "png.png" );
#endif
        const PenguinV_Image::Image original = Unit_Test::blackImage();

        for ( auto filename = filenames.begin(); filename != filenames.end(); ++filename ) {
            File_Operation::Save( filename->c_str(), original );

            const PenguinV_Image::Image loaded = File_Operation::Load( filename->c_str() );
            remove( filename->c_str() );

            if ( original.height() != loaded.height() || original.width() != loaded.width() || original.colorCount() != loaded.colorCount()
                 || !Unit_Test::verifyImage( loaded, 0u ) )
                return false;
        }

        return true;
    }

    bool RandomRGBImage()
    {
        const PenguinV_Image::Image original = Unit_Test::randomRGBImage();
        std::vector<std::string> filenames;
        filenames.push_back( "bitmap.bmp" );

#ifdef PENGUINV_ENABLED_PNG_SUPPORT
        filenames.push_back( "png.png" );
#endif
        for ( auto filename = filenames.begin(); filename != filenames.end(); ++filename ) {
            File_Operation::Save( filename->c_str(), original );

            const PenguinV_Image::Image loaded = File_Operation::Load( filename->c_str() );
            remove( filename->c_str() );

            if ( original.height() != loaded.height() || original.width() != loaded.width() || original.colorCount() != loaded.colorCount() )
                return false;

            const uint32_t rowSizeIn = original.rowSize();
            const uint32_t rowSizeOut = loaded.rowSize();
            const uint32_t width = original.width() * original.colorCount();
            const uint8_t * inY = original.data();
            const uint8_t * outY = loaded.data();
            const uint8_t * inYEnd = inY + rowSizeIn * original.height();

            for ( ; inY != inYEnd; inY += rowSizeIn, outY += rowSizeOut ) {
                if ( memcmp( inY, outY, width ) != 0 )
                    return false;
            }
        }
        return true;
    }
}

void addTests_File( UnitTestFramework & framework )
{
    framework.add( file_operation::WhiteGrayScaleImage, "File: Save and load white gray-scale image" );
    framework.add( file_operation::BlackGrayScaleImage, "File: Save and load black gray-scale image" );
    framework.add( file_operation::RandomRGBImage,      "File: Save and load random RGB image" );
}
