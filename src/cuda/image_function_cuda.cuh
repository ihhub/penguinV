#pragma once

#include <stdint.h>
#include "image_buffer_cuda.cuh"

namespace Image_Function_Cuda
{
    using namespace penguinV;

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

    Image ConvertToCuda( const Image & in );
    void  ConvertToCuda( const Image & in, Image & out );
    Image ConvertFromCuda( const Image & in );
    void  ConvertFromCuda( const Image & in, Image & out );

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

    void  Copy( const Image & in, Image & out );
    Image Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
    void  Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                uint32_t width, uint32_t height );

    Image ExtractChannel( const Image & in, uint8_t channelId );
    void  ExtractChannel( const Image & in, Image & out, uint8_t channelId );
    Image ExtractChannel( const Image & in, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t channelId );
    void  ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
                          uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId );

    void Fill( Image & image, uint8_t value );
    void Fill( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value );

    // Make sure that input parameters such as input and output images are not same image!
    // horizontal flip: left-right --> right-left
    // vertical flip: top-bottom --> bottom-top
    Image Flip( const Image & in, bool horizontal, bool vertical );
    void  Flip( const Image & in, Image & out, bool horizontal, bool vertical );
    Image Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                bool horizontal, bool vertical );
    void  Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                uint32_t width, uint32_t height, bool horizontal, bool vertical );

    // Gamma correction works by formula:
    // output = A * ((input / 255) ^ gamma) * 255, where A - multiplication, gamma - power base. Both values must be greater than 0
    // Usually people set A as 1
    Image GammaCorrection( const Image & in, double a, double gamma );
    void  GammaCorrection( const Image & in, Image & out, double a, double gamma );
    Image GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double a, double gamma );
    void  GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                           uint32_t width, uint32_t height, double a, double gamma );

    uint8_t GetThreshold( const std::vector < uint32_t > & histogram );

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

    std::vector < uint32_t > ProjectionProfile( const Image & image, bool horizontal );
    void                     ProjectionProfile( const Image & image, bool horizontal, std::vector < uint32_t > & projection );
    std::vector < uint32_t > ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal );
    void                     ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                                                std::vector < uint32_t > & projection );

    void Rotate( const Image & in, float centerXIn, float centerYIn, Image & out, float centerXOut, float centerYOut, float angle );
    
    void SetPixel( Image & image, uint32_t x, uint32_t y, uint8_t value );
    void SetPixel( Image & image, const std::vector<uint32_t> & X, const std::vector<uint32_t> & Y, uint8_t value );

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
