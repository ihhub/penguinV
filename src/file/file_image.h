#pragma once
#include "../image_buffer.h"

namespace File_Operation
{
    PenguinV::Image Load( const std::string & path );
    void                  Load( const std::string & path, PenguinV::Image & image );

    void Save( const std::string & path, const PenguinV::Image & image );
    void Save( const std::string & path, const PenguinV::Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height );
}
