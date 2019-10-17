#pragma once

#include <vector>
#include "image_buffer.h"

namespace Image_Function
{
    using namespace PenguinV_Image;

    Image Median( const Image & in, uint32_t kernelSize );
    void  Median( const Image & in, Image & out, uint32_t kernelSize );
    Image Median( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint32_t kernelSize );
    void  Median( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                  uint32_t width, uint32_t height, uint32_t kernelSize );

    // This filter returns image based on gradient magnitude in both X and Y directions
    Image Prewitt( const Image & in );
    void  Prewitt( const Image & in, Image & out );
    Image Prewitt( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
    void  Prewitt( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                   uint32_t width, uint32_t height );

    // This filter returns image based on gradient magnitude in both X and Y directions
    Image Sobel( const Image & in );
    void  Sobel( const Image & in, Image & out );
    Image Sobel( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
    void  Sobel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                 uint32_t width, uint32_t height );

    void GetGaussianKernel( std::vector<float> & filter, uint32_t width, uint32_t height, uint32_t kernelSize, float sigma );
}
