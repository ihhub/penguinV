#pragma once
#include "../image_buffer.h"

namespace Png_Operation
{
    // Below functions support only PenguinV_Image::Image and PenguinV_Image::ColorImage classes
    PenguinV_Image::Image Load( const std::string & path );
    void                  Load( const std::string & path, PenguinV_Image::Image & image );

    void Save( const std::string & path, const PenguinV_Image::Image & image );
    void Save( const std::string & path, const PenguinV_Image::Image & image, uint32_t startX, uint32_t startY,
               uint32_t width, uint32_t height );
}
