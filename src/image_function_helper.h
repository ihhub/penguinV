#pragma once
#include "image_buffer.h"
#include <map>
#include <vector>

namespace Image_Function_Helper
{
    using namespace penguinV;

    namespace FunctionTable
    {
        // Function pointer definitions
        typedef Image ( *AbsoluteDifferenceForm1 )( const Image & in1, const Image & in2 );
        typedef void ( *AbsoluteDifferenceForm2 )( const Image & in1, const Image & in2, Image & out );
        typedef Image ( *AbsoluteDifferenceForm3 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                                                    uint32_t width, uint32_t height );
        typedef void ( *AbsoluteDifferenceForm4 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                                                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

        typedef void ( *AccumulateForm1 )( const Image & image, std::vector<uint32_t> & result );
        typedef void ( *AccumulateForm2 )( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector<uint32_t> & result );

        typedef void ( *BinaryDilateForm1 )( Image & image, uint32_t dilationX, uint32_t dilationY );
        typedef void ( *BinaryDilateForm2 )( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint32_t dilationX, uint32_t dilationY );

        typedef void ( *BinaryErodeForm1 )( Image & image, uint32_t erosionX, uint32_t erosionY );
        typedef void ( *BinaryErodeForm2 )( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint32_t erosionX, uint32_t erosionY );

        typedef Image ( *BitwiseAndForm1 )( const Image & in1, const Image & in2 );
        typedef void ( *BitwiseAndForm2 )( const Image & in1, const Image & in2, Image & out );
        typedef Image ( *BitwiseAndForm3 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width,
                                            uint32_t height );
        typedef void ( *BitwiseAndForm4 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out,
                                           uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

        typedef Image ( *BitwiseOrForm1 )( const Image & in1, const Image & in2 );
        typedef void ( *BitwiseOrForm2 )( const Image & in1, const Image & in2, Image & out );
        typedef Image ( *BitwiseOrForm3 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width,
                                           uint32_t height );
        typedef void ( *BitwiseOrForm4 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out,
                                          uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

        typedef Image ( *BitwiseXorForm1 )( const Image & in1, const Image & in2 );
        typedef void ( *BitwiseXorForm2 )( const Image & in1, const Image & in2, Image & out );
        typedef Image ( *BitwiseXorForm3 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width,
                                            uint32_t height );
        typedef void ( *BitwiseXorForm4 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out,
                                           uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

        typedef Image16Bit ( *ConvertTo16BitForm1 )( const Image & in );
        typedef void ( *ConvertTo16BitForm2 )( const Image & in, Image16Bit & out );
        typedef Image16Bit ( *ConvertTo16BitForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
        typedef void ( *ConvertTo16BitForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image16Bit & out, uint32_t startXOut, uint32_t startYOut,
                                               uint32_t width, uint32_t height );

        typedef Image ( *ConvertTo8BitForm1 )( const Image16Bit & in );
        typedef void ( *ConvertTo8BitForm2 )( const Image16Bit & in, Image & out );
        typedef Image ( *ConvertTo8BitForm3 )( const Image16Bit & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
        typedef void ( *ConvertTo8BitForm4 )( const Image16Bit & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                              uint32_t width, uint32_t height );

        typedef Image ( *ConvertToGrayScaleForm1 )( const Image & in );
        typedef void ( *ConvertToGrayScaleForm2 )( const Image & in, Image & out );
        typedef Image ( *ConvertToGrayScaleForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
        typedef void ( *ConvertToGrayScaleForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                                   uint32_t width, uint32_t height );

        typedef Image ( *ConvertToRgbForm1 )( const Image & in );
        typedef void ( *ConvertToRgbForm2 )( const Image & in, Image & out );
        typedef Image ( *ConvertToRgbForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
        typedef void ( *ConvertToRgbForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                             uint32_t height );

        typedef void ( *CopyForm1 )( const Image & in, Image & out );
        typedef Image ( *CopyForm2 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
        typedef void ( *CopyForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                     uint32_t height );

        typedef Image ( *ExtractChannelForm1 )( const Image & in, uint8_t channelId );
        typedef void ( *ExtractChannelForm2 )( const Image & in, Image & out, uint8_t channelId );
        typedef Image ( *ExtractChannelForm3 )( const Image & in, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t channelId );
        typedef void ( *ExtractChannelForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                               uint32_t width, uint32_t height, uint8_t channelId );

        typedef void ( *FillForm1 )( Image & image, uint8_t value );
        typedef void ( *FillForm2 )( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value );

        typedef Image ( *FlipForm1 )( const Image & in, bool horizontal, bool vertical );
        typedef void ( *FlipForm2 )( const Image & in, Image & out, bool horizontal, bool vertical );
        typedef Image ( *FlipForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, bool horizontal, bool vertical );
        typedef void ( *FlipForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                     uint32_t height, bool horizontal, bool vertical );

        typedef Image ( *GammaCorrectionForm1 )( const Image & in, double a, double gamma );
        typedef void ( *GammaCorrectionForm2 )( const Image & in, Image & out, double a, double gamma );
        typedef Image ( *GammaCorrectionForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double a, double gamma );
        typedef void ( *GammaCorrectionForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                                uint32_t width, uint32_t height, double a, double gamma );

        typedef uint8_t ( *GetPixelForm1 )( const Image & image, uint32_t x, uint32_t y );

        typedef uint8_t ( *GetThresholdForm1 )( const std::vector<uint32_t> & histogram );

        typedef std::vector<uint32_t> ( *HistogramForm1 )( const Image & image );
        typedef void ( *HistogramForm2 )( const Image & image, std::vector<uint32_t> & histogram );
        typedef std::vector<uint32_t> ( *HistogramForm3 )( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height );
        typedef void ( *HistogramForm4 )( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector<uint32_t> & histogram );

        typedef std::vector<uint32_t> ( *HistogramForm5 )( const Image & image, const Image & mask );
        typedef void ( *HistogramForm6 )( const Image & image, const Image & mask, std::vector<uint32_t> & histogram );
        typedef std::vector<uint32_t> ( *HistogramForm7 )( const Image & image, uint32_t x, uint32_t y, const Image & mask, uint32_t maskX, uint32_t maskY,
                                                           uint32_t width, uint32_t height );
        typedef void ( *HistogramForm8 )( const Image & image, uint32_t x, uint32_t y, const Image & mask, uint32_t maskX, uint32_t maskY, uint32_t width,
                                          uint32_t height, std::vector<uint32_t> & histogram );

        typedef Image ( *InvertForm1 )( const Image & in );
        typedef void ( *InvertForm2 )( const Image & in, Image & out );
        typedef Image ( *InvertForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
        typedef void ( *InvertForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                       uint32_t height );

        typedef bool ( *IsBinaryForm1 )( const Image & image );
        typedef bool ( *IsBinaryForm2 )( const Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height );

        typedef bool ( *IsEqualForm1 )( const Image & in1, const Image & in2 );
        typedef bool ( *IsEqualForm2 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width,
                                        uint32_t height );

        typedef Image ( *LookupTableForm1 )( const Image & in, const std::vector<uint8_t> & table );
        typedef void ( *LookupTableForm2 )( const Image & in, Image & out, const std::vector<uint8_t> & table );
        typedef Image ( *LookupTableForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                                             const std::vector<uint8_t> & table );
        typedef void ( *LookupTableForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                            uint32_t height, const std::vector<uint8_t> & table );

        typedef Image ( *MaximumForm1 )( const Image & in1, const Image & in2 );
        typedef void ( *MaximumForm2 )( const Image & in1, const Image & in2, Image & out );
        typedef Image ( *MaximumForm3 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width,
                                         uint32_t height );
        typedef void ( *MaximumForm4 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out,
                                        uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

        typedef Image ( *MergeForm1 )( const Image & in1, const Image & in2, const Image & in3 );
        typedef void ( *MergeForm2 )( const Image & in1, const Image & in2, const Image & in3, Image & out );
        typedef Image ( *MergeForm3 )( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                                       const Image & in3, uint32_t startXIn3, uint32_t startYIn3, uint32_t width, uint32_t height );
        typedef void ( *MergeForm4 )( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                                      const Image & in3, uint32_t startXIn3, uint32_t startYIn3, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                      uint32_t height );

        typedef Image ( *MinimumForm1 )( const Image & in1, const Image & in2 );
        typedef void ( *MinimumForm2 )( const Image & in1, const Image & in2, Image & out );
        typedef Image ( *MinimumForm3 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width,
                                         uint32_t height );
        typedef void ( *MinimumForm4 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out,
                                        uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

        typedef Image ( *NormalizeForm1 )( const Image & in );
        typedef void ( *NormalizeForm2 )( const Image & in, Image & out );
        typedef Image ( *NormalizeForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
        typedef void ( *NormalizeForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                          uint32_t height );

        typedef std::vector<uint32_t> ( *ProjectionProfileForm1 )( const Image & image, bool horizontal );
        typedef void ( *ProjectionProfileForm2 )( const Image & image, bool horizontal, std::vector<uint32_t> & projection );
        typedef std::vector<uint32_t> ( *ProjectionProfileForm3 )( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal );
        typedef void ( *ProjectionProfileForm4 )( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                                                  std::vector<uint32_t> & projection );

        typedef void ( *ReplaceChannelForm1 )( const Image & channel, Image & rgb, uint8_t channelId );
        typedef void ( *ReplaceChannelForm2 )( const Image & channel, uint32_t startXChannel, uint32_t startYChannel, Image & rgb, uint32_t startXRgb, uint32_t startYRgb,
                                               uint32_t width, uint32_t height, uint8_t channelId );

        typedef Image ( *ResizeForm1 )( const Image & in, uint32_t widthOut, uint32_t heightOut );
        typedef void ( *ResizeForm2 )( const Image & in, Image & out );
        typedef Image ( *ResizeForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn, uint32_t widthOut,
                                        uint32_t heightOut );
        typedef void ( *ResizeForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn, Image & out, uint32_t startXOut,
                                       uint32_t startYOut, uint32_t widthOut, uint32_t heightOut );

        typedef Image ( *RgbToBgrForm1 )( const Image & in );
        typedef void ( *RgbToBgrForm2 )( const Image & in, Image & out );
        typedef Image ( *RgbToBgrForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
        typedef void ( *RgbToBgrForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                         uint32_t height );

        typedef Image ( *RgbToRgbaForm1 )( const Image & in );
        typedef void ( *RgbToRgbaForm2 )( const Image & in, Image & out );
        typedef Image ( *RgbToRgbaForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
        typedef void ( *RgbToRgbaForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                          uint32_t height );

        typedef Image ( *RgbaToRgbForm1 )( const Image & in );
        typedef void ( *RgbaToRgbForm2 )( const Image & in, Image & out );
        typedef Image ( *RgbaToRgbForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
        typedef void ( *RgbaToRgbForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                          uint32_t height );

        typedef Image ( *RotateForm1 )( const Image & in, double centerX, double centerY, double angle );
        typedef void ( *RotateForm2 )( const Image & in, double centerXIn, double centerYIn, Image & out, double centerXOut, double centerYOut, double angle );
        typedef Image ( *RotateForm3 )( const Image & in, uint32_t x, uint32_t y, double centerX, double centerY, uint32_t width, uint32_t height, double angle );
        typedef void ( *RotateForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, double centerXIn, double centerYIn, Image & out, uint32_t startXOut,
                                       uint32_t startYOut, double centerXOut, double centerYOut, uint32_t width, uint32_t height, double angle );

        typedef void ( *SetPixelForm1 )( Image & image, uint32_t x, uint32_t y, uint8_t value );
        typedef void ( *SetPixelForm2 )( Image & image, const std::vector<uint32_t> & X, const std::vector<uint32_t> & Y, uint8_t value );

        typedef Image ( *ShiftForm1 )( const Image & in, double shiftX, double shiftY );
        typedef void ( *ShiftForm2 )( const Image & in, Image & out, double shiftX, double shiftY );
        typedef Image ( *ShiftForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double shiftX, double shiftY );
        typedef void ( *ShiftForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                      uint32_t height, double shiftX, double shiftY );

        typedef void ( *SplitForm1 )( const Image & in, Image & out1, Image & out2, Image & out3 );
        typedef void ( *SplitForm2 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out1, uint32_t startXOut1, uint32_t startYOut1, Image & out2,
                                      uint32_t startXOut2, uint32_t startYOut2, Image & out3, uint32_t startXOut3, uint32_t startYOut3, uint32_t width, uint32_t height );

        typedef Image ( *SubtractForm1 )( const Image & in1, const Image & in2 );
        typedef void ( *SubtractForm2 )( const Image & in1, const Image & in2, Image & out );
        typedef Image ( *SubtractForm3 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, uint32_t width,
                                          uint32_t height );
        typedef void ( *SubtractForm4 )( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2, Image & out,
                                         uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

        typedef uint32_t ( *SumForm1 )( const Image & image );
        typedef uint32_t ( *SumForm2 )( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height );

        typedef Image ( *ThresholdForm1 )( const Image & in, uint8_t threshold );
        typedef void ( *ThresholdForm2 )( const Image & in, Image & out, uint8_t threshold );
        typedef Image ( *ThresholdForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t threshold );
        typedef void ( *ThresholdForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                          uint32_t height, uint8_t threshold );

        typedef Image ( *ThresholdDoubleForm1 )( const Image & in, uint8_t minThreshold, uint8_t maxThreshold );
        typedef void ( *ThresholdDoubleForm2 )( const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold );
        typedef Image ( *ThresholdDoubleForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t minThreshold,
                                                 uint8_t maxThreshold );
        typedef void ( *ThresholdDoubleForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                                uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold );

        typedef Image ( *TransposeForm1 )( const Image & in );
        typedef void ( *TransposeForm2 )( const Image & in, Image & out );
        typedef Image ( *TransposeForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
        typedef void ( *TransposeForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                          uint32_t height );

        // Filters
        typedef Image ( *PrewittForm1 )( const Image & in );
        typedef void ( *PrewittForm2 )( const Image & in, Image & out );
        typedef Image ( *PrewittForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
        typedef void ( *PrewittForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                        uint32_t height );

        typedef Image ( *SobelForm1 )( const Image & in );
        typedef void ( *SobelForm2 )( const Image & in, Image & out );
        typedef Image ( *SobelForm3 )( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
        typedef void ( *SobelForm4 )( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width,
                                      uint32_t height );
    }

    Image AbsoluteDifference( FunctionTable::AbsoluteDifferenceForm4 absoluteDifference, const Image & in1, const Image & in2 );

    void AbsoluteDifference( FunctionTable::AbsoluteDifferenceForm4 absoluteDifference, const Image & in1, const Image & in2, Image & out );

    Image AbsoluteDifference( FunctionTable::AbsoluteDifferenceForm4 absoluteDifference, const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2,
                              uint32_t startX2, uint32_t startY2, uint32_t width, uint32_t height );

    void Accumulate( FunctionTable::AccumulateForm2 accumulate, const Image & image, std::vector<uint32_t> & result );

    Image BitwiseAnd( FunctionTable::BitwiseAndForm4 bitwiseAnd, const Image & in1, const Image & in2 );

    void BitwiseAnd( FunctionTable::BitwiseAndForm4 bitwiseAnd, const Image & in1, const Image & in2, Image & out );

    Image BitwiseAnd( FunctionTable::BitwiseAndForm4 bitwiseAnd, const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2,
                      uint32_t startY2, uint32_t width, uint32_t height );

    Image BitwiseOr( FunctionTable::BitwiseOrForm4 bitwiseOr, const Image & in1, const Image & in2 );

    void BitwiseOr( FunctionTable::BitwiseOrForm4 bitwiseOr, const Image & in1, const Image & in2, Image & out );

    Image BitwiseOr( FunctionTable::BitwiseOrForm4 bitwiseOr, const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2,
                     uint32_t startY2, uint32_t width, uint32_t height );

    Image BitwiseXor( FunctionTable::BitwiseXorForm4 bitwiseXor, const Image & in1, const Image & in2 );

    void BitwiseXor( FunctionTable::BitwiseXorForm4 bitwiseXor, const Image & in1, const Image & in2, Image & out );

    Image BitwiseXor( FunctionTable::BitwiseXorForm4 bitwiseXor, const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2,
                      uint32_t startY2, uint32_t width, uint32_t height );

    void ConvertTo16Bit( FunctionTable::ConvertTo16BitForm4 convertTo16Bit, const Image & in, Image16Bit & out );

    void ConvertTo8Bit( FunctionTable::ConvertTo8BitForm4 convertTo8Bit, const Image16Bit & in, Image & out );

    Image ConvertToGrayScale( FunctionTable::ConvertToGrayScaleForm4 convertToGrayScale, const Image & in );

    void ConvertToGrayScale( FunctionTable::ConvertToGrayScaleForm4 convertToGrayScale, const Image & in, Image & out );

    Image ConvertToGrayScale( FunctionTable::ConvertToGrayScaleForm4 convertToGrayScale, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width,
                              uint32_t height );

    Image ConvertToRgb( FunctionTable::ConvertToRgbForm4 convertToRgb, const Image & in );

    void ConvertToRgb( FunctionTable::ConvertToRgbForm4 convertToRgb, const Image & in, Image & out );

    Image ConvertToRgb( FunctionTable::ConvertToRgbForm4 convertToRgb, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    Image Copy( FunctionTable::CopyForm3 copy, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    Image ExtractChannel( FunctionTable::ExtractChannelForm4 extractChannel, const Image & in, uint8_t channelId );

    void ExtractChannel( FunctionTable::ExtractChannelForm4 extractChannel, const Image & in, Image & out, uint8_t channelId );

    Image ExtractChannel( FunctionTable::ExtractChannelForm4 extractChannel, const Image & in, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                          uint8_t channelId );

    Image Flip( FunctionTable::FlipForm4 flip, const Image & in, bool horizontal, bool vertical );

    void Flip( FunctionTable::FlipForm4 flip, const Image & in, Image & out, bool horizontal, bool vertical );

    Image Flip( FunctionTable::FlipForm4 flip, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, bool horizontal, bool vertical );

    Image GammaCorrection( FunctionTable::GammaCorrectionForm4 gammaCorrection, const Image & in, double a, double gamma );

    void GammaCorrection( FunctionTable::GammaCorrectionForm4 gammaCorrection, const Image & in, Image & out, double a, double gamma );

    Image GammaCorrection( FunctionTable::GammaCorrectionForm4 gammaCorrection, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                           double a, double gamma );

    std::vector<uint8_t> GetGammaCorrectionLookupTable( double a, double gamma );

    uint8_t GetThreshold( const std::vector<uint32_t> & histogram );

    std::vector<uint32_t> Histogram( FunctionTable::HistogramForm4 histogram, const Image & image );

    void Histogram( FunctionTable::HistogramForm4 histogram, const Image & image, std::vector<uint32_t> & histogramTable );

    std::vector<uint32_t> Histogram( FunctionTable::HistogramForm4 histogram, const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height );

    std::vector<uint32_t> Histogram( FunctionTable::HistogramForm8 histogram, const Image & image, const Image & mask );

    void Histogram( FunctionTable::HistogramForm8 histogram, const Image & image, const Image & mask, std::vector<uint32_t> & histogramTable );

    std::vector<uint32_t> Histogram( FunctionTable::HistogramForm8 histogram, const Image & image, uint32_t x, uint32_t y, const Image & mask, uint32_t maskX,
                                     uint32_t maskY, uint32_t width, uint32_t height );

    Image Invert( FunctionTable::InvertForm4 invert, const Image & in );

    void Invert( FunctionTable::InvertForm4 invert, const Image & in, Image & out );

    Image Invert( FunctionTable::InvertForm4 invert, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    bool IsEqual( FunctionTable::IsEqualForm2 isEqual, const Image & in1, const Image & in2 );

    Image LookupTable( FunctionTable::LookupTableForm4 lookupTable, const Image & in, const std::vector<uint8_t> & table );

    void LookupTable( FunctionTable::LookupTableForm4 lookupTable, const Image & in, Image & out, const std::vector<uint8_t> & table );

    Image LookupTable( FunctionTable::LookupTableForm4 lookupTable, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                       const std::vector<uint8_t> & table );

    Image Maximum( FunctionTable::MaximumForm4 maximum, const Image & in1, const Image & in2 );

    void Maximum( FunctionTable::MaximumForm4 maximum, const Image & in1, const Image & in2, Image & out );

    Image Maximum( FunctionTable::MaximumForm4 maximum, const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height );

    Image Merge( FunctionTable::MergeForm4 merge, const Image & in1, const Image & in2, const Image & in3 );

    void Merge( FunctionTable::MergeForm4 merge, const Image & in1, const Image & in2, const Image & in3, Image & out );

    Image Merge( FunctionTable::MergeForm4 merge, const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                 const Image & in3, uint32_t startXIn3, uint32_t startYIn3, uint32_t width, uint32_t height );

    Image Minimum( FunctionTable::MinimumForm4 minimum, const Image & in1, const Image & in2 );

    void Minimum( FunctionTable::MinimumForm4 minimum, const Image & in1, const Image & in2, Image & out );

    Image Minimum( FunctionTable::MinimumForm4 minimum, const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height );

    Image Normalize( FunctionTable::NormalizeForm4 normalize, const Image & in );

    void Normalize( FunctionTable::NormalizeForm4 normalize, const Image & in, Image & out );

    Image Normalize( FunctionTable::NormalizeForm4 normalize, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    std::vector<uint32_t> ProjectionProfile( FunctionTable::ProjectionProfileForm4 projectionProfile, const Image & image, bool horizontal );

    void ProjectionProfile( FunctionTable::ProjectionProfileForm4 projectionProfile, const Image & image, bool horizontal, std::vector<uint32_t> & projection );

    std::vector<uint32_t> ProjectionProfile( FunctionTable::ProjectionProfileForm4 projectionProfile, const Image & image, uint32_t x, uint32_t y, uint32_t width,
                                             uint32_t height, bool horizontal );

    Image Resize( FunctionTable::ResizeForm4 resize, const Image & in, uint32_t widthOut, uint32_t heightOut );

    void Resize( FunctionTable::ResizeForm4 resize, const Image & in, Image & out );

    Image Resize( FunctionTable::ResizeForm4 resize, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn, uint32_t widthOut,
                  uint32_t heightOut );

    Image RgbToBgr( FunctionTable::RgbToBgrForm4 rgbToBgr, const Image & in );

    void RgbToBgr( FunctionTable::RgbToBgrForm4 rgbToBgr, const Image & in, Image & out );

    Image RgbToBgr( FunctionTable::RgbToBgrForm4 rgbToBgr, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    Image RgbToRgba( FunctionTable::RgbToRgbaForm4 rgbToRgba, const Image & in );

    void RgbToRgba( FunctionTable::RgbToRgbaForm4 rgbToRgba, const Image & in, Image & out );

    Image RgbToRgba( FunctionTable::RgbToRgbaForm4 rgbToRgba, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    Image RgbaToRgb( FunctionTable::RgbaToRgbForm4 rgbaToRgb, const Image & in );

    void RgbaToRgb( FunctionTable::RgbaToRgbForm4 rgbaToRgb, const Image & in, Image & out );

    Image RgbaToRgb( FunctionTable::RgbaToRgbForm4 rgbaToRgb, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    Image Rotate( FunctionTable::RotateForm4 rotate, const Image & in, double centerX, double centerY, double angle );

    void Rotate( FunctionTable::RotateForm4 rotate, const Image & in, double centerXIn, double centerYIn, Image & out, double centerXOut, double centerYOut,
                 double angle );

    Image Rotate( FunctionTable::RotateForm4 rotate, const Image & in, uint32_t x, uint32_t y, double centerX, double centerY, uint32_t width, uint32_t height,
                  double angle );

    Image Shift( FunctionTable::ShiftForm4 shift, const Image & in, double shiftX, double shiftY );

    void Shift( FunctionTable::ShiftForm4 shift, const Image & in, Image & out, double shiftX, double shiftY );

    Image Shift( FunctionTable::ShiftForm4 shift, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double shiftX, double shiftY );

    void Split( FunctionTable::SplitForm2 split, const Image & in, Image & out1, Image & out2, Image & out3 );

    Image Subtract( FunctionTable::SubtractForm4 subtract, const Image & in1, const Image & in2 );

    void Subtract( FunctionTable::SubtractForm4 subtract, const Image & in1, const Image & in2, Image & out );

    Image Subtract( FunctionTable::SubtractForm4 subtract, const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    uint32_t width, uint32_t height );

    Image Threshold( FunctionTable::ThresholdForm4 threshold, const Image & in, uint8_t thresholdValue );

    void Threshold( FunctionTable::ThresholdForm4 threshold, const Image & in, Image & out, uint8_t thresholdValue );

    Image Threshold( FunctionTable::ThresholdForm4 threshold, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                     uint8_t thresholdValue );

    Image Threshold( FunctionTable::ThresholdDoubleForm4 threshold, const Image & in, uint8_t minThreshold, uint8_t maxThreshold );

    void Threshold( FunctionTable::ThresholdDoubleForm4 threshold, const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold );

    Image Threshold( FunctionTable::ThresholdDoubleForm4 threshold, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                     uint8_t minThreshold, uint8_t maxThreshold );

    Image Transpose( FunctionTable::TransposeForm4 transpose, const Image & in );

    void Transpose( FunctionTable::TransposeForm4 transpose, const Image & in, Image & out );

    Image Transpose( FunctionTable::TransposeForm4 transpose, const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    struct FunctionTableHolder
    {
        FunctionTable::AbsoluteDifferenceForm4 AbsoluteDifference = nullptr;
        FunctionTable::AccumulateForm2 Accumulate = nullptr;
        FunctionTable::BitwiseAndForm4 BitwiseAnd = nullptr;
        FunctionTable::BitwiseOrForm4 BitwiseOr = nullptr;
        FunctionTable::BitwiseXorForm4 BitwiseXor = nullptr;
        FunctionTable::ConvertTo16BitForm4 ConvertTo16Bit = nullptr;
        FunctionTable::ConvertTo8BitForm4 ConvertTo8Bit = nullptr;
        FunctionTable::ConvertToGrayScaleForm4 ConvertToGrayScale = nullptr;
        FunctionTable::ConvertToRgbForm4 ConvertToRgb = nullptr;
        FunctionTable::CopyForm3 Copy = nullptr;
        FunctionTable::ExtractChannelForm4 ExtractChannel = nullptr;
        FunctionTable::FillForm2 Fill = nullptr;
        FunctionTable::FlipForm4 Flip = nullptr;
        FunctionTable::GammaCorrectionForm4 GammaCorrection = nullptr;
        FunctionTable::GetPixelForm1 GetPixel = nullptr;
        FunctionTable::HistogramForm4 Histogram = nullptr;
        FunctionTable::InvertForm4 Invert = nullptr;
        FunctionTable::IsEqualForm2 IsEqual = nullptr;
        FunctionTable::LookupTableForm4 LookupTable = nullptr;
        FunctionTable::MaximumForm4 Maximum = nullptr;
        FunctionTable::MergeForm4 Merge = nullptr;
        FunctionTable::MinimumForm4 Minimum = nullptr;
        FunctionTable::NormalizeForm4 Normalize = nullptr;
        FunctionTable::ProjectionProfileForm4 ProjectionProfile = nullptr;
        FunctionTable::ResizeForm4 Resize = nullptr;
        FunctionTable::RgbToBgrForm4 RgbToBgr = nullptr;
        FunctionTable::RgbToRgbaForm4 RgbToRgba = nullptr;
        FunctionTable::RgbaToRgbForm4 RgbaToRgb = nullptr;
        FunctionTable::SetPixelForm1 SetPixel = nullptr;
        FunctionTable::SetPixelForm2 SetPixel2 = nullptr;
        FunctionTable::ShiftForm4 Shift = nullptr;
        FunctionTable::SplitForm2 Split = nullptr;
        FunctionTable::SubtractForm4 Subtract = nullptr;
        FunctionTable::SumForm2 Sum = nullptr;
        FunctionTable::ThresholdForm4 Threshold = nullptr;
        FunctionTable::ThresholdDoubleForm4 Threshold2 = nullptr;
        FunctionTable::TransposeForm4 Transpose = nullptr;
    };
}

class ImageTypeManager
{
public:
    static ImageTypeManager & instance();

    // Register function table for specific image type. This function must be called within source file (*.cpp) during startup of the library
    // forceSetup flag is needed for SIMD function table set as we are not sure in which order normal CPU and SIMD global function code would be called
    void setFunctionTable( uint8_t type, const Image_Function_Helper::FunctionTableHolder & table, bool forceSetup = false );
    const Image_Function_Helper::FunctionTableHolder & functionTable( uint8_t type ) const;

    void setConvertFunction( Image_Function_Helper::FunctionTable::CopyForm1 Copy, const penguinV::Image & in, const penguinV::Image & out );
    void convert( const penguinV::Image & in, penguinV::Image & out ) const;

    penguinV::Image image( uint8_t type ) const;
    std::vector<uint8_t> imageTypes() const;

    void enableIntertypeConversion( bool enable );
    bool isIntertypeConversionEnabled() const;

private:
    std::map<uint8_t, Image_Function_Helper::FunctionTableHolder> _functionTableMap;
    std::map<std::pair<uint8_t, uint8_t>, Image_Function_Helper::FunctionTable::CopyForm1> _intertypeConvertMap;
    std::map<uint8_t, penguinV::Image> _image;
    bool _enabledIntertypeConversion;

    ImageTypeManager();
};

// This namespace is a helper namespace for SIMD instruction based code
namespace simd
{
    // These functions are designed only for testing simd technique functions individually
    void EnableSimd( bool enable );
    void EnableAvx512( bool enable );
    void EnableAvx( bool enable );
    void EnableSse( bool enable );
    void EnableNeon( bool enable );

    enum SIMDType
    {
        avx512_function,
        avx_function,
        sse_function,
        neon_function,
        cpu_function
    };

    SIMDType actualSimdType();
}
