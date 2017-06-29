#include "../image_buffer.h"

namespace Bitmap_Operation
{
    // Below functions support only Bitmap_Image::Image and Bitmap_Image::ColorImage classes
    Bitmap_Image::Image Load( const std::string & path );
    void                Load( const std::string & path, Bitmap_Image::Image & image );

    void Save( const std::string & path, const Bitmap_Image::Image & image );
    void Save( const std::string & path, const Bitmap_Image::Image & image, uint32_t startX, uint32_t startY,
               uint32_t width, uint32_t height );
};
