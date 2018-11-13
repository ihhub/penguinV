#include "unit_test_jpeg_ops.h"
#include "unit_test_helper.h"
#include "../../src/FileOperation/jpeg_file.h"

namespace jpeg_operation
{
    bool LoadSaveRGBImage()
    {
        const PenguinV_Image::Image original = Unit_Test::uniformRGBImage();
        Jpeg_Operation::Save("rgb.jpg", original);

        const PenguinV_Image::Image loaded = Jpeg_Operation::Load("rgb.bmp");

        if (original.height() != loaded.height() || original.width() != loaded.width() ||
            original.colorCount() != loaded.colorCount() || !Unit_Test::verifyImage(loaded, 255u))
            return false;
    
        return true;
    }
}

void addTests_Jpeg( UnitTestFramework & framework )
{
    ADD_TEST(framework, jpeg_operation::LoadSaveRGBImage);
}
