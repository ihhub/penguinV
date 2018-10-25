#include "unit_test_bitmap.h"
#include "unit_test_helper.h"
#include "../../src/FileOperation/bitmap.h"

namespace bitmap_operation
{
    bool WhiteGrayScaleImage()
    {
        PenguinV_Image::Image original = Unit_Test::whiteImage();

        Bitmap_Operation::Save( "bitmap.bmp", original );

        PenguinV_Image::Image loaded = Bitmap_Operation::Load( "bitmap.bmp" );

        if( original.height() != loaded.height() || original.width() != loaded.width() ||
            original.colorCount() != loaded.colorCount() || !Unit_Test::verifyImage( loaded, 255u ) )
            return false;

        return true;
    }

    bool BlackGrayScaleImage()
    {
        PenguinV_Image::Image original = Unit_Test::blackImage();
        Bitmap_Operation::Save( "bitmap.bmp", original );
        PenguinV_Image::Image loaded = Bitmap_Operation::Load( "bitmap.bmp" );

        if (original.height() != loaded.height() || 
            original.width() != loaded.width() ||
            original.colorCount() != loaded.colorCount() || 
            !Unit_Test::verifyImage( loaded, 0u ))
            return false;
        return true;
    }

    bool RandomRGBImage()
    {
        PenguinV_Image::Image original = Unit_Test::randomImage();
        Bitmap_Operation::Save("bitmap.bmp", original);
        PenguinV_Image::Image loaded = Bitmap_Operation::Load("bitmap.bmp");

        if ((original.height() != loaded.height()) ||
            (original.width() != loaded.width()) ||
            (original.colorCount() != loaded.colorCount()))
            return false;
        else
        {
            uint32_t rowsizeDiff = loaded.rowSize() - original.rowSize();
            for (size_t row = 0; row < original.height(); row++)
            {
                if (memcmp(&original.data()[row * original.width()],
                           &loaded.data()[row * (original.width() + 
                                                 rowsizeDiff)],
                           original.width()))
                    return false;
            }
        }
        return true;
    }
}


void addTests_Bitmap( UnitTestFramework & framework )
{
    framework.add(bitmap_operation::WhiteGrayScaleImage, 
                  "Load and save white gray-scale image");
    framework.add(bitmap_operation::BlackGrayScaleImage, 
                  "Load and save black gray-scale image");
    framework.add(bitmap_operation::RandomRGBImage, 
                  "Load and save random RGB image");
}
