#include "unit_test_bitmap.h"
#include "unit_test_helper.h"
#include "../../src/FileOperation/bitmap.h"

namespace bitmap_operation
{
    bool LoadSaveGrayScaleImage()
    {
        PenguinV_Image::Image original = Unit_Test::whiteImage();

        Bitmap_Operation::Save( "white.bmp", original );

        PenguinV_Image::Image loaded = Bitmap_Operation::Load( "white.bmp" );

        if( original.height() != loaded.height() || original.width() != loaded.width() ||
            original.colorCount() != loaded.colorCount() || !Unit_Test::verifyImage( loaded, 255u ) )
            return false;

        return true;
    }
}


void addTests_Bitmap( UnitTestFramework & framework )
{
    ADD_TEST( framework, bitmap_operation::LoadSaveGrayScaleImage );
}
