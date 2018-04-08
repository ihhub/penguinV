#pragma once

#include <vector>
#include "../image_buffer.h"
#include "../image_function_helper.h"

namespace penguinV
{
    using namespace PenguinV_Image;

    // A table which contains pointers to basic functions
    struct FunctionTable
    {
        Image_Function_Helper::FunctionTable::AbsoluteDifference AbsoluteDifference;
        Image_Function_Helper::FunctionTable::Accumulate Accumulate;
        Image_Function_Helper::FunctionTable::BitwiseAnd BitwiseAnd;
        Image_Function_Helper::FunctionTable::BitwiseOr BitwiseOr;
        Image_Function_Helper::FunctionTable::BitwiseXor BitwiseXor;
        Image_Function_Helper::FunctionTable::ConvertToGrayScale ConvertToGrayScale;
        Image_Function_Helper::FunctionTable::ConvertToRgb ConvertToRgb;
        Image_Function_Helper::FunctionTable::Copy Copy;
        Image_Function_Helper::FunctionTable::ExtractChannel ExtractChannel;
        Image_Function_Helper::FunctionTable::Fill Fill;
        Image_Function_Helper::FunctionTable::Flip Flip;
        Image_Function_Helper::FunctionTable::GammaCorrection GammaCorrection;
        Image_Function_Helper::FunctionTable::GetPixel GetPixel;
        Image_Function_Helper::FunctionTable::GetThreshold GetThreshold;
        Image_Function_Helper::FunctionTable::Histogram Histogram;
        Image_Function_Helper::FunctionTable::Invert Invert;
        Image_Function_Helper::FunctionTable::IsEqual IsEqual;
        Image_Function_Helper::FunctionTable::LookupTable LookupTable;
        Image_Function_Helper::FunctionTable::Maximum Maximum;
        Image_Function_Helper::FunctionTable::Merge Merge;
        Image_Function_Helper::FunctionTable::Minimum Minimum;
        Image_Function_Helper::FunctionTable::Normalize Normalize;
        Image_Function_Helper::FunctionTable::ProjectionProfile ProjectionProfile;
        Image_Function_Helper::FunctionTable::Resize Resize;
        Image_Function_Helper::FunctionTable::RgbToBgr RgbToBgr;
        Image_Function_Helper::FunctionTable::SetPixel SetPixel;
        Image_Function_Helper::FunctionTable::SetPixel2 SetPixel2;
        Image_Function_Helper::FunctionTable::Split Split;
        Image_Function_Helper::FunctionTable::Subtract Subtract;
        Image_Function_Helper::FunctionTable::Sum Sum;
        Image_Function_Helper::FunctionTable::Threshold Threshold;
        Image_Function_Helper::FunctionTable::Threshold2 Threshold2;
        Image_Function_Helper::FunctionTable::Transpose Transpose;
    };

    // A function which returns reference to a function table
    const FunctionTable & functionTable();

    // A list of basic functions
    inline void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                                    Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        functionTable().AbsoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline void Accumulate( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & result )
    {
        functionTable().Accumulate( image, x, y, width, height, result );
    }

    inline void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                            Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        functionTable().BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                           Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        functionTable().BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                            Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        functionTable().BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                    uint32_t width, uint32_t height )
    {
        functionTable().ConvertToGrayScale( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    inline void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                              uint32_t width, uint32_t height )
    {
        functionTable().ConvertToRgb( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    inline void Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                      uint32_t width, uint32_t height )
    {
        functionTable().Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    inline void ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
                                uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId )
    {
        functionTable().ExtractChannel( in, startXIn, startYIn, out, startXOut, startYOut, width, height, channelId );
    }

    inline void Fill( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value )
    {
        functionTable().Fill( image, x, y, width, height, value );
    }

    inline void Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                      uint32_t width, uint32_t height, bool horizontal, bool vertical )
    {
        functionTable().Flip( in, startXIn, startYIn, out, startXOut, startYOut, width, height, horizontal, vertical );
    }

    inline void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                 uint32_t width, uint32_t height, double a, double gamma )
    {
        functionTable().GammaCorrection( in, startXIn, startYIn, out, startXOut, startYOut, width, height, a, gamma );
    }

    inline uint8_t GetPixel( const Image & image, uint32_t x, uint32_t y )
    {
        return functionTable().GetPixel( image, x, y );
    }

    inline uint8_t GetThreshold( const std::vector < uint32_t > & histogram )
    {
        return functionTable().GetThreshold( histogram );
    }

    inline void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                           std::vector < uint32_t > & histogram )
    {
        functionTable().Histogram( image, x, y, width, height, histogram );
    }

    inline void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                        uint32_t width, uint32_t height )
    {
        functionTable().Invert( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    inline bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                         uint32_t width, uint32_t height )
    {
        return functionTable().IsEqual( in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    inline void LookupTable ( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                              uint32_t width, uint32_t height, const std::vector < uint8_t > & table )
    {
        functionTable().LookupTable( in, startXIn, startYIn, out, startXOut, startYOut, width, height, table );
    }

    inline void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                         Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        functionTable().Maximum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline void Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                       const Image & in3, uint32_t startXIn3, uint32_t startYIn3, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height )
    {
        functionTable().Merge( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3,
                               out, startXOut, startYOut, width, height );
    }

    inline void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                         Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        functionTable().Minimum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline void Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                           uint32_t width, uint32_t height )
    {
        functionTable().Normalize( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    inline void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                                   std::vector < uint32_t > & projection )
    {
        functionTable().ProjectionProfile( image, x, y, width, height, horizontal, projection );
    }

    inline void Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                        Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut )
    {
        functionTable().Resize( in, startXIn, startYIn, widthIn, heightIn, out, startXOut, startYOut, widthOut, heightOut );
    }

    inline void RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                          uint32_t width, uint32_t height )
    {
        functionTable().RgbToBgr( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    inline void SetPixel( Image & image, uint32_t x, uint32_t y, uint8_t value )
    {
        functionTable().SetPixel( image, x, y, value );
    }

    inline void SetPixel( Image & image, const std::vector < uint32_t > & X, const std::vector < uint32_t > & Y, uint8_t value )
    {
        functionTable().SetPixel2( image, X, Y, value );
    }

    inline void Split( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out1, uint32_t startXOut1, uint32_t startYOut1,
                       Image & out2, uint32_t startXOut2, uint32_t startYOut2, Image & out3, uint32_t startXOut3, uint32_t startYOut3,
                       uint32_t width, uint32_t height )
    {
        functionTable().Split( in, startXIn, startYIn, out1, startXOut1, startYOut1, out2, startXOut2, startYOut2,
                               out3, startXOut3, startYOut3, width, height );
    }

    inline void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                          Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        functionTable().Subtract( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        return functionTable().Sum( image, x, y, width, height );
    }

    inline void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                           uint32_t width, uint32_t height, uint8_t threshold )
    {
        functionTable().Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, threshold );
    }

    inline void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                           uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold )
    {
        functionTable().Threshold2( in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold );
    }

    inline void Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                           uint32_t width, uint32_t height )
    {
        functionTable().Transpose( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }
}
