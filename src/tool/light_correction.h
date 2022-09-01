#pragma once

#include "../image_buffer.h"
#include "../math/math_base.h"
#include <vector>

// Input image for analysis should not contain pixel intensity values 0 or 255
// If an image contains pixels with such values they would be ignored during correction
class LightCorrection
{
public:
    void analyze( const penguinV::Image & image );
    void correct( penguinV::Image & image ) const;
    void correct( penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height ) const;

    // Returns an array of pixel coordinates on which pixel intensity equal to 0 or 255
    std::vector<PointBase2D<uint32_t>> findIncorrectPixels( const penguinV::Image & image ) const;

private:
    std::vector<uint32_t> _data;
    uint32_t _width;
    uint32_t _height;
};
