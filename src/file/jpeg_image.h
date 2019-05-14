#pragma once
#include "../image_buffer.h"

namespace Jpeg_Operation
{
    PenguinV_Image::Image Load( const std::string & path );
    void                  Load( const std::string & path, PenguinV_Image::Image & image );

    void Save( const std::string & path, const PenguinV_Image::Image & image );
    void Save( const std::string & path, const PenguinV_Image::Image & image, uint32_t startX, uint32_t startY,
               uint32_t width, uint32_t height );
}
