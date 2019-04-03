#pragma once

#include "image_buffer.h"

template<typename _Type>
class EdgeDetection;

struct EdgeParameter;

class EdgeDetectionHelper
{
public:
    static void find(EdgeDetection<double> & edgeDetection, const PenguinV_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const EdgeParameter & edgeParameter);
    
    static void find(EdgeDetection<float> & edgeDetection, const PenguinV_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const EdgeParameter & edgeParameter);
private:
    EdgeDetectionHelper() {} // No need for creating an instance of the class
};
