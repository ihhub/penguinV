#include "../image_buffer.h"

namespace Jpeg_Operation
{
// load a jpeg image
PenguinV_Image::Image Load(const std::string &path);
void Load(const std::string &path, PenguinV_Image::Image &image);

// save a jpeg image
void Save(const std::string &path, PenguinV_Image::Image &image);
} // namespace Jpeg_Operation
