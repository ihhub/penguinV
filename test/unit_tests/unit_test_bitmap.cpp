#include "unit_test_bitmap.h"
#include "unit_test_helper.h"
#include "../../src/FileOperation/bitmap.h"

namespace Unit_Test
{
    namespace bitmap_operation
    {
        bool LoadSaveGrayScaleImage()
        {
            Bitmap_Image::Image original = whiteImage();

            Bitmap_Operation::Save( "white.bmp", original );

            Bitmap_Image::Image loaded = Bitmap_Operation::Load( "white.bmp" );

            if( !equalSize( original, loaded ) || !verifyImage( loaded, 255u ) )
                return false;

            return true;
        }
    }

    void addTests_Bitmap( UnitTestFramework & framework )
    {
        ADD_TEST( framework, bitmap_operation::LoadSaveGrayScaleImage );
    }
}
