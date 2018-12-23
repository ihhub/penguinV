#include <stdio.h>

#include "unit_test_bitmap.h"
#include "unit_test_helper.h"
#include "../../src/file/bmp_image.h"

namespace bitmap_operation
{
    bool WhiteGrayScaleImage()
    {
        const PenguinV_Image::Image original = Unit_Test::whiteImage();
        Bitmap_Operation::Save( "bitmap.bmp", original );

        const PenguinV_Image::Image loaded = Bitmap_Operation::Load( "bitmap.bmp" );
        remove("bitmap.bmp");

        if( original.height() != loaded.height() || original.width() != loaded.width() ||
            original.colorCount() != loaded.colorCount() || !Unit_Test::verifyImage( loaded, 255u ) )
            return false;

        return true;
    }

    bool BlackGrayScaleImage()
    {
        const PenguinV_Image::Image original = Unit_Test::blackImage();
        Bitmap_Operation::Save( "bitmap.bmp", original );

        const PenguinV_Image::Image loaded = Bitmap_Operation::Load( "bitmap.bmp" );
        remove("bitmap.bmp");

        if( original.height() != loaded.height() || original.width() != loaded.width() ||
            original.colorCount() != loaded.colorCount() || !Unit_Test::verifyImage( loaded, 0u ) )
            return false;

        return true;
    }

    bool RandomRGBImage()
    {
        const PenguinV_Image::Image original = Unit_Test::randomRGBImage();
        Bitmap_Operation::Save("bitmap.bmp", original);

        const PenguinV_Image::Image loaded = Bitmap_Operation::Load("bitmap.bmp");
        remove("bitmap.bmp");

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


void addTests_Bitmap( UnitTestFramework & framework )
{
    framework.add(bitmap_operation::WhiteGrayScaleImage, "Save and load white gray-scale image");
    framework.add(bitmap_operation::BlackGrayScaleImage, "Save and load black gray-scale image");
    framework.add(bitmap_operation::RandomRGBImage,      "Save and load random RGB image");
}
