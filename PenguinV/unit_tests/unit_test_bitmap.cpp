#include "unit_test_bitmap.h"
#include "unit_test_helper.h"
#include "../Library/FileOperation/bitmap.h"

namespace Unit_Test
{
    void addTests_Bitmap( UnitTestFramework & framework )
    {
        ADD_TEST( framework, Bitmap_Operation_Test::LoadSaveGrayScaleImage );
    }

    namespace Bitmap_Operation_Test
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
    };
};
