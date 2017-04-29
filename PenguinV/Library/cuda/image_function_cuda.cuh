#pragma once

#include <stdint.h>
#include "../image_buffer.h"
#include "image_buffer_cuda.cuh"

namespace Image_Function_Cuda
{
    using namespace Bitmap_Image_Cuda;

    Image AbsoluteDifference( const Image & in1, const Image & in2 );
    void  AbsoluteDifference( const Image & in1, const Image & in2, Image & out );

    Image BitwiseAnd( const Image & in1, const Image & in2 );
    void  BitwiseAnd( const Image & in1, const Image & in2, Image & out );

    Image BitwiseOr( const Image & in1, const Image & in2 );
    void  BitwiseOr( const Image & in1, const Image & in2, Image & out );

    Image BitwiseXor( const Image & in1, const Image & in2 );
    void  BitwiseXor( const Image & in1, const Image & in2, Image & out );

    void  Convert( const Bitmap_Image::Image & in, Image & out );
    void  Convert( const Image & in, Bitmap_Image::Image & out );

    void  Copy( const Image & in, Image & out );

    void  Fill( Image & image, uint8_t value );

    // Gamma correction works by formula:
    // output = A * (input ^ gamma), where A - multiplication, gamma - power base. Both values must be greater than 0
    // Usually people set A as 1
    Image GammaCorrection( const Image & in, double a, double gamma );
    void  GammaCorrection( const Image & in, Image & out, double a, double gamma );

    // Invert function is Bitwise NOT operation. But to make function name more user-friendly we named it like this
    Image Invert( const Image & in );
    void  Invert( const Image & in, Image & out );

    Image Maximum( const Image & in1, const Image & in2 );
    void  Maximum( const Image & in1, const Image & in2, Image & out );

    Image Minimum( const Image & in1, const Image & in2 );
    void  Minimum( const Image & in1, const Image & in2, Image & out );

    Image Subtract( const Image & in1, const Image & in2 );
    void  Subtract( const Image & in1, const Image & in2, Image & out );
};
