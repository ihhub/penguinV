#pragma once
#include "../image_buffer.h"

namespace Bitmap_Operation
{
    penguinV::Image Load( const std::string & path );
    void            Load( const std::string & path, penguinV::Image & image );

    void Save( const std::string & path, const penguinV::Image & image );
    void Save( const std::string & path, const penguinV::Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height );
}
