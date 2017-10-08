#pragma once

#include <vector>
#include "image_buffer.h"

namespace Image_Function
{
    namespace Filtering
    {
        using namespace Bitmap_Image;

        Image Median( const Image & in, size_t kernelSize );
        void  Median( const Image & in, Image & out, size_t kernelSize );
        Image Median( const Image & in, size_t startXIn, size_t startYIn, size_t width, size_t height, size_t kernelSize );
        void  Median( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                      size_t width, size_t height, size_t kernelSize );

        // This filter returns image based on gradient magnitude in both X and Y directions
        Image Prewitt( const Image & in );
        void  Prewitt( const Image & in, Image & out );
        Image Prewitt( const Image & in, size_t startXIn, size_t startYIn, size_t width, size_t height );
        void  Prewitt( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                       size_t width, size_t height );

        // This filter returns image based on gradient magnitude in both X and Y directions
        Image Sobel( const Image & in );
        void  Sobel( const Image & in, Image & out );
        Image Sobel( const Image & in, size_t startXIn, size_t startYIn, size_t width, size_t height );
        void  Sobel( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                     size_t width, size_t height );

        void GetGaussianKernel( std::vector<float> & filter, size_t width, size_t height, size_t kernelSize, float sigma );
    };
};
