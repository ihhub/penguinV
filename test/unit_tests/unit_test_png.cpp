#include <stdio.h>

#include "unit_test_bitmap.h"
#include "unit_test_helper.h"
#include "../../src/file/png_image.h"

namespace png_operation
{
    bool WhiteGrayScaleImage()
    {
        const PenguinV_Image::Image original = Unit_Test::whiteImage();
        Png_Operation::Save( "png.png", original );

        const PenguinV_Image::Image loaded = Png_Operation::Load( "png.png" );
        remove("png.png");

        if( original.height() != loaded.height() || original.width() != loaded.width() ||
            original.colorCount() != loaded.colorCount() || !Unit_Test::verifyImage( loaded, 255u ) )
            return false;

        return true;
    }

    bool BlackGrayScaleImage()
    {
        const PenguinV_Image::Image original = Unit_Test::blackImage();
        Png_Operation::Save( "png.png", original );

        const PenguinV_Image::Image loaded = Png_Operation::Load( "png.png" );
        remove("png.png");

        if( original.height() != loaded.height() || original.width() != loaded.width() ||
            original.colorCount() != loaded.colorCount() || !Unit_Test::verifyImage( loaded, 0u ) )
            return false;

        return true;
    }

    bool RandomRGBImage()
    {
        const PenguinV_Image::Image original = Unit_Test::randomRGBImage();
        Png_Operation::Save("png.png", original);

        const PenguinV_Image::Image loaded = Png_Operation::Load("png.png");
        remove("png.png");

        if( original.height() != loaded.height() || original.width() != loaded.width() ||
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
}

#ifdef PENGUINV_ENABLED_PNG_SUPPORT
void addTests_Png( UnitTestFramework & framework )
{
    framework.add(png_operation::WhiteGrayScaleImage, "PNG: Save and load white gray-scale image");
    framework.add(png_operation::BlackGrayScaleImage, "PNG: Save and load black gray-scale image");
    framework.add(png_operation::RandomRGBImage,      "PNG: Save and load random RGB image");
}
#else
void addTests_Png( UnitTestFramework & )
{
}
#endif

