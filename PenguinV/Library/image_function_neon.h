#pragma once

#include "penguinv/cpu_identification.h"

#ifdef PENGUINV_NEON_SET

#include <vector>
#include "image_buffer.h"

// Utilize these image functions only if your CPU is ARM with NEON intructions support!!!
// These functions contain NEON code
// Functions have totally same results like normal functions but they are faster!
// You will have speed up compare to normal functions if the width of inspection area is bigger than 16 pixels
// because 16 pixels is minimum width what NEON function can process
// Anyway you do not need to care about this. Everything is done inside. Just use it if you want to have faster code

namespace Image_Function_Neon
{
    using namespace Bitmap_Image;

    Image AbsoluteDifference( const Image & in1, const Image & in2 );
    void  AbsoluteDifference( const Image & in1, const Image & in2, Image & out );
    Image AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                              uint32_t width, uint32_t height );
    void  AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                              Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    Image BitwiseAnd( const Image & in1, const Image & in2 );
    void  BitwiseAnd( const Image & in1, const Image & in2, Image & out );
    Image BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height );
    void  BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    Image BitwiseOr( const Image & in1, const Image & in2 );
    void  BitwiseOr( const Image & in1, const Image & in2, Image & out );
    Image BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     uint32_t width, uint32_t height );
    void  BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    Image BitwiseXor( const Image & in1, const Image & in2 );
    void  BitwiseXor( const Image & in1, const Image & in2, Image & out );
    Image BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height );
    void  BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    // Invert function is Bitwise NOT operation. But to make function name more user-friendly we named it like this
    Image Invert( const Image & in );
    void  Invert( const Image & in, Image & out );
    Image Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
    void  Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                  uint32_t width, uint32_t height );

    Image Maximum( const Image & in1, const Image & in2 );
    void  Maximum( const Image & in1, const Image & in2, Image & out );
    Image Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height );
    void  Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    Image Minimum( const Image & in1, const Image & in2 );
    void  Minimum( const Image & in1, const Image & in2, Image & out );
    Image Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height );
    void  Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    Image Subtract( const Image & in1, const Image & in2 );
    void  Subtract( const Image & in1, const Image & in2, Image & out );
    Image Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    uint32_t width, uint32_t height );
    void  Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    // Thresholding works in such way:
        // if pixel intensity on input image is          less (  < ) than threshold then set pixel intensity on output image as 0
        // if pixel intensity on input image is equal or more ( >= ) than threshold then set pixel intensity on output image as 255
    Image Threshold( const Image & in, uint8_t threshold );
    void  Threshold( const Image & in, Image & out, uint8_t threshold );
    Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t threshold );
    void  Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                     uint32_t width, uint32_t height, uint8_t threshold );

    // Thresholding works in such way:
        // if pixel intensity on input image is less ( < ) than minimum threshold or more ( > ) than maximum threshold
        // then      set pixel intensity on output image as 0
        // otherwise set pixel intensity on output image as 255
    Image Threshold( const Image & in, uint8_t minThreshold, uint8_t maxThreshold );
    void  Threshold( const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold );
    Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t minThreshold,
                     uint8_t maxThreshold );
    void  Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                     uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold );
}
#endif
