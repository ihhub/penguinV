#include "unit_test_jpeg_ops.h"
#include "unit_test_helper.h"
#include "../../src/FileOperation/jpeg_file.h"

namespace jpeg_operation
{
bool LoadSaveRGBImage()
{
    PenguinV_Image::Image jpegImg = Unit_Test::uniformRGBImage();

    Jpeg_Operation::Save("rgb.jpg", jpegImg);

    PenguinV_Image::Image loaded = Jpeg_Operation::Load("rgb.bmp");

    if (jpegImg.height() != loaded.height() || jpegImg.width() != loaded.width() ||
        jpegImg.colorCount() != loaded.colorCount() || !Unit_Test::verifyImage(loaded, 255u))
        return false;

    return true;
}
} // namespace jpeg_operation

void addTests_Jpeg(UnitTestFramework &framework)
{
    ADD_TEST(framework, jpeg_operation::LoadSaveRGBImage);
}