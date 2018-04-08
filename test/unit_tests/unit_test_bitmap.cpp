#include "unit_test_bitmap.h"
#include "unit_test_helper.h"
#include "../../src/FileOperation/bitmap.h"

namespace Unit_Test
{
    namespace bitmap_operation
    {
        bool LoadSaveGrayScaleImage()
        {
            PenguinV_Image::Image original = whiteImage();

            Bitmap_Operation::Save( "white.bmp", original );

            PenguinV_Image::Image loaded = Bitmap_Operation::Load( "white.bmp" );

            if( original.height() != loaded.height() && original.width() != loaded.width() && original.alignment() != loaded.alignment() &&
                original.colorCount() == loaded.colorCount() || !verifyImage( loaded, 255u ) )
                return false;

            return true;
        }
    }

    void addTests_Bitmap( UnitTestFramework & framework )
    {
        ADD_TEST( framework, bitmap_operation::LoadSaveGrayScaleImage );
    }
}
