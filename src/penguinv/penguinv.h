#pragma once

#include <vector>
#include "../image_buffer.h"
#include "../image_function_helper.h"

namespace penguinV
{
    using namespace PenguinV_Image;

    void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                             Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    void Accumulate( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & result );

    void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                             uint32_t width, uint32_t height );

    void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height );

    void Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
               uint32_t width, uint32_t height );

    void ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
                         uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId );

    void Fill( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value );

    void Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
               uint32_t width, uint32_t height, bool horizontal, bool vertical );

    void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                          uint32_t width, uint32_t height, double a, double gamma );

    uint8_t GetPixel( const Image & image, uint32_t x, uint32_t y );

    uint8_t GetThreshold( const std::vector < uint32_t > & histogram );

    void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                    std::vector < uint32_t > & histogram );

    void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                 uint32_t width, uint32_t height );

    bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  uint32_t width, uint32_t height );

    void LookupTable ( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height, const std::vector < uint8_t > & table );

    void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    void Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                const Image & in3, uint32_t startXIn3, uint32_t startYIn3, Image & out, uint32_t startXOut, uint32_t startYOut,
                uint32_t width, uint32_t height );

    void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    void Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height );

    void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                            std::vector < uint32_t > & projection );

    void Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut );

    void RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                   uint32_t width, uint32_t height );

    void SetPixel( Image & image, uint32_t x, uint32_t y, uint8_t value );

    void SetPixel( Image & image, const std::vector < uint32_t > & X, const std::vector < uint32_t > & Y, uint8_t value );

    void Split( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out1, uint32_t startXOut1, uint32_t startYOut1,
                Image & out2, uint32_t startXOut2, uint32_t startYOut2, Image & out3, uint32_t startXOut3, uint32_t startYOut3,
                uint32_t width, uint32_t height );

    void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height );

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t threshold );

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold );

    void Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height );

    // A table which contains pointers to basic functions
    struct FunctionTable
    {
        Image_Function_Helper::FunctionTable::AbsoluteDifference AbsoluteDifference = nullptr;
        Image_Function_Helper::FunctionTable::Accumulate         Accumulate         = nullptr;
        Image_Function_Helper::FunctionTable::BitwiseAnd         BitwiseAnd         = nullptr;
        Image_Function_Helper::FunctionTable::BitwiseOr          BitwiseOr          = nullptr;
        Image_Function_Helper::FunctionTable::BitwiseXor         BitwiseXor         = nullptr;
        Image_Function_Helper::FunctionTable::ConvertToGrayScale ConvertToGrayScale = nullptr;
        Image_Function_Helper::FunctionTable::ConvertToRgb       ConvertToRgb       = nullptr;
        Image_Function_Helper::FunctionTable::Copy               Copy               = nullptr;
        Image_Function_Helper::FunctionTable::ExtractChannel     ExtractChannel     = nullptr;
        Image_Function_Helper::FunctionTable::Fill               Fill               = nullptr;
        Image_Function_Helper::FunctionTable::Flip               Flip               = nullptr;
        Image_Function_Helper::FunctionTable::GammaCorrection    GammaCorrection    = nullptr;
        Image_Function_Helper::FunctionTable::GetPixel           GetPixel           = nullptr;
        Image_Function_Helper::FunctionTable::Histogram          Histogram          = nullptr;
        Image_Function_Helper::FunctionTable::Invert             Invert             = nullptr;
        Image_Function_Helper::FunctionTable::IsEqual            IsEqual            = nullptr;
        Image_Function_Helper::FunctionTable::LookupTable        LookupTable        = nullptr;
        Image_Function_Helper::FunctionTable::Maximum            Maximum            = nullptr;
        Image_Function_Helper::FunctionTable::Merge              Merge              = nullptr;
        Image_Function_Helper::FunctionTable::Minimum            Minimum            = nullptr;
        Image_Function_Helper::FunctionTable::Normalize          Normalize          = nullptr;
        Image_Function_Helper::FunctionTable::ProjectionProfile  ProjectionProfile  = nullptr;
        Image_Function_Helper::FunctionTable::Resize             Resize             = nullptr;
        Image_Function_Helper::FunctionTable::RgbToBgr           RgbToBgr           = nullptr;
        Image_Function_Helper::FunctionTable::SetPixel           SetPixel           = nullptr;
        Image_Function_Helper::FunctionTable::SetPixel2          SetPixel2          = nullptr;
        Image_Function_Helper::FunctionTable::Split              Split              = nullptr;
        Image_Function_Helper::FunctionTable::Subtract           Subtract           = nullptr;
        Image_Function_Helper::FunctionTable::Sum                Sum                = nullptr;
        Image_Function_Helper::FunctionTable::Threshold          Threshold          = nullptr;
        Image_Function_Helper::FunctionTable::Threshold2         Threshold2         = nullptr;
        Image_Function_Helper::FunctionTable::Transpose          Transpose          = nullptr;
    };

    // Register function table for specific image type. This function must be called within source file (*.cpp) during startup of the library
    // forceSetup flag is needed for SIMD function table set as we are not sure in which order normal CPU and SIMD global function code would be called
    void registerFunctionTable( const Image & image, const FunctionTable & table, bool forceSetup = false );
}
