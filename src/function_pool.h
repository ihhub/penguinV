#pragma once
#include <vector>
#include "image_buffer.h"

namespace Function_Pool
{
    // Use this namespace if your compiler supports C++11 threads

    // This namespace's functions support thread pool utilization through Thread_Pool::ThreadPoolMonoid class
    // Please make sure before calling of any of these functions that global (singleton) thread pool has at least 1 thread!
    using namespace PenguinV_Image;

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

    Image ConvertToGrayScale( const Image & in );
    void  ConvertToGrayScale( const Image & in, Image & out );
    Image ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
    void  ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                              uint32_t width, uint32_t height );

    Image ConvertToRgb( const Image & in );
    void  ConvertToRgb( const Image & in, Image & out );
    Image ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
    void  ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                        uint32_t width, uint32_t height );

    Image ExtractChannel( const Image & in, uint8_t channelId );
    void  ExtractChannel( const Image & in, Image & out, uint8_t channelId );
    Image ExtractChannel( const Image & in, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t channelId );
    void  ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
                          uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId );

    // Gamma correction works by formula:
    // output = A * ((input / 255) ^ gamma) * 255, where A - multiplication, gamma - power base. Both values must be greater than 0
    // Usually people set A as 1
    Image GammaCorrection( const Image & in, double a, double gamma );
    void  GammaCorrection( const Image & in, Image & out, double a, double gamma );
    Image GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double a, double gamma );
    void  GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                           uint32_t width, uint32_t height, double a, double gamma );

    std::vector < uint32_t > Histogram( const Image & image );
    void                     Histogram( const Image & image, std::vector < uint32_t > & histogram );
    std::vector < uint32_t > Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height );
    void                     Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                        std::vector < uint32_t > & histogram );

    // Invert function is Bitwise NOT operation. But to make function name more user-friendly we named it like this
    Image Invert( const Image & in );
    void  Invert( const Image & in, Image & out );
    Image Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
    void  Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                  uint32_t width, uint32_t height );

    bool IsEqual( const Image & in1, const Image & in2 );
    bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  uint32_t width, uint32_t height );

    Image LookupTable( const Image & in, const std::vector < uint8_t > & table );
    void  LookupTable( const Image & in, Image & out, const std::vector < uint8_t > & table );
    Image LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                       const std::vector < uint8_t > & table );
    void  LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height, const std::vector < uint8_t > & table );

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

    Image Normalize( const Image & in );
    void  Normalize( const Image & in, Image & out );
    Image Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
    void  Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                     uint32_t width, uint32_t height );

    std::vector < uint32_t > ProjectionProfile( const Image & image, bool horizontal );
    void                     ProjectionProfile( const Image & image, bool horizontal, std::vector < uint32_t > & projection );
    std::vector < uint32_t > ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal );
    void                     ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                                                std::vector < uint32_t > & projection );

    // Image resizing (scaling) is based on nearest-neighbour interpolation method
    Image Resize( const Image & in, uint32_t widthOut, uint32_t heightOut );
    void  Resize( const Image & in, Image & out );
    Image Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                  uint32_t widthOut, uint32_t heightOut );
    void  Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut );

    Image RgbToBgr( const Image & in );
    void  RgbToBgr( const Image & in, Image & out );
    Image RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
    void  RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height );

    Image Subtract( const Image & in1, const Image & in2 );
    void  Subtract( const Image & in1, const Image & in2, Image & out );
    Image Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    uint32_t width, uint32_t height );
    void  Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    // Make sure that your image is not so big to do not have overloaded uint32_t value
    // For example not bigger than [4096 * 4096] for 32-bit application
    uint32_t Sum( const Image & image );
    uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height );

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
