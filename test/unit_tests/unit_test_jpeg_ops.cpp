#include <stdio.h>
#include "unit_test_jpeg_ops.h"
#include "unit_test_helper.h"
#include "../../src/FileOperation/jpeg_file.h"

namespace jpeg_operation
{
    bool WhiteGrayScaleImage()
    {
        const PenguinV_Image::Image original = Unit_Test::whiteImage();
        Jpeg_Operation::Save( "jpeg.jpg", original );

        const PenguinV_Image::Image loaded = Jpeg_Operation::Load( "jpeg.jpg" );
        remove("jpeg.jpg");

        if( original.height() != loaded.height() || original.width() != loaded.width() ||
            original.colorCount() != loaded.colorCount() || !Unit_Test::verifyImage( loaded, 255u ) )
            return false;

        return true;
    }

    bool BlackGrayScaleImage()
    {
        const PenguinV_Image::Image original = Unit_Test::blackImage();
        Jpeg_Operation::Save( "jpeg.jpg", original );

        const PenguinV_Image::Image loaded = Jpeg_Operation::Load( "jpeg.jpg" );
        remove("jpeg.jpg");

        if( original.height() != loaded.height() || original.width() != loaded.width() ||
            original.colorCount() != loaded.colorCount() || !Unit_Test::verifyImage( loaded, 0u ) )
            return false;

        return true;
    }

    bool RandomRGBImage()
    {
        const PenguinV_Image::Image original = Unit_Test::randomRGBImage();
        Jpeg_Operation::Save("rgb.jpg", original);

        const PenguinV_Image::Image loaded = Jpeg_Operation::Load("jpeg.jpg");
        remove("jpeg.jpg");

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

void addTests_Jpeg( UnitTestFramework & framework )
{
    framework.add(jpeg_operation::WhiteGrayScaleImage, "Load and save white gray-scale image");
    framework.add(jpeg_operation::BlackGrayScaleImage, "Load and save black gray-scale image");
    framework.add(jpeg_operation::RandomRGBImage,      "Load and save random RGB image");
}
