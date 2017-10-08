#pragma once

#include <vector>
#include "../image_buffer.h"

namespace penguinV
{
    using namespace Bitmap_Image;

    // A table which contains pointers to basic functions
    struct FunctionTable
    {
        void ( *AbsoluteDifference )(const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                                     Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height);
        void ( *Accumulate )       (const Image & image, size_t x, size_t y, size_t width, size_t height, std::vector < size_t > & result);
        void ( *BitwiseAnd )       (const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                                    Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height);
        void ( *BitwiseOr )        (const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                                    Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height);
        void ( *BitwiseXor )       (const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                                    Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height);
        void ( *ConvertToGrayScale )(const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                     size_t width, size_t height);
        void ( *ConvertToRgb )     (const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                    size_t width, size_t height);
        void ( *Copy )             (const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                    size_t width, size_t height);
        void ( *ExtractChannel )   (const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut,
                                    size_t startYOut, size_t width, size_t height, uint8_t channelId);
        void ( *Fill )             (Image & image, size_t x, size_t y, size_t width, size_t height, uint8_t value);
        void ( *Flip )             (const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                    size_t width, size_t height, bool horizontal, bool vertical);
        void ( *GammaCorrection )  (const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                    size_t width, size_t height, double a, double gamma);
        uint8_t ( *GetPixel )      (const Image & image, size_t x, size_t y);
        uint8_t ( *GetThreshold )  (const std::vector < size_t > & histogram);
        void ( *Histogram )        (const Image & image, size_t x, size_t y, size_t width, size_t height,
                                    std::vector < size_t > & histogram);
        void ( *Invert )           (const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                    size_t width, size_t height);
        bool ( *IsEqual )          (const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                                    size_t width, size_t height);
        void ( *LookupTable )      (const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                    size_t width, size_t height, const std::vector < uint8_t > & table);
        void ( *Maximum )          (const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                                    Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height);
        void ( *Merge )            (const Image & in1, size_t startXIn1, size_t startYIn1, const Image & in2, size_t startXIn2, size_t startYIn2,
                                    const Image & in3, size_t startXIn3, size_t startYIn3, Image & out, size_t startXOut, size_t startYOut,
                                    size_t width, size_t height);
        void ( *Minimum )          (const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                                    Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height);
        void ( *Normalize )        (const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                    size_t width, size_t height);
        void ( *ProjectionProfile )(const Image & image, size_t x, size_t y, size_t width, size_t height, bool horizontal,
                                    std::vector < size_t > & projection);
        void ( *Resize )           (const Image & in, size_t startXIn, size_t startYIn, size_t widthIn, size_t heightIn,
                                    Image & out, size_t startXOut, size_t startYOut, size_t widthOut, size_t heightOut);
        void ( *RgbToBgr )         (const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                    size_t width, size_t height);
        void ( *SetPixel )         (Image & image, size_t x, size_t y, uint8_t value);
        void ( *SetPixel2 )        (Image & image, const std::vector < size_t > & X, const std::vector < size_t > & Y, uint8_t value);
        void ( *Split )            (const Image & in, size_t startXIn, size_t startYIn, Image & out1, size_t startXOut1, size_t startYOut1,
                                    Image & out2, size_t startXOut2, size_t startYOut2, Image & out3, size_t startXOut3, size_t startYOut3,
                                    size_t width, size_t height);
        void ( *Subtract )         (const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                                    Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height);
        size_t ( *Sum )          (const Image & image, size_t x, size_t y, size_t width, size_t height);
        void ( *Threshold )        (const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                    size_t width, size_t height, uint8_t threshold);
        void ( *Threshold2 )       (const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                    size_t width, size_t height, uint8_t minThreshold, uint8_t maxThreshold);
        void ( *Transpose )        (const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                    size_t width, size_t height);
    };

    // A function which returns reference to a function table
    const FunctionTable & functionTable();

    // A list of basic functions
    inline void AbsoluteDifference( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                                    Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        functionTable().AbsoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline void Accumulate( const Image & image, size_t x, size_t y, size_t width, size_t height, std::vector < size_t > & result )
    {
        functionTable().Accumulate( image, x, y, width, height, result );
    }

    inline void BitwiseAnd( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                            Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        functionTable().BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline void BitwiseOr( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                           Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        functionTable().BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline void BitwiseXor( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                            Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        functionTable().BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline void ConvertToGrayScale( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                    size_t width, size_t height )
    {
        functionTable().ConvertToGrayScale( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    inline void ConvertToRgb( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                              size_t width, size_t height )
    {
        functionTable().ConvertToRgb( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    inline void Copy( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                      size_t width, size_t height )
    {
        functionTable().Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    inline void ExtractChannel( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut,
                                size_t startYOut, size_t width, size_t height, uint8_t channelId )
    {
        functionTable().ExtractChannel( in, startXIn, startYIn, out, startXOut, startYOut, width, height, channelId );
    }

    inline void Fill( Image & image, size_t x, size_t y, size_t width, size_t height, uint8_t value )
    {
        functionTable().Fill( image, x, y, width, height, value );
    }

    inline void Flip( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                      size_t width, size_t height, bool horizontal, bool vertical )
    {
        functionTable().Flip( in, startXIn, startYIn, out, startXOut, startYOut, width, height, horizontal, vertical );
    }

    inline void GammaCorrection( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                                 size_t width, size_t height, double a, double gamma )
    {
        functionTable().GammaCorrection( in, startXIn, startYIn, out, startXOut, startYOut, width, height, a, gamma );
    }

    inline uint8_t GetPixel( const Image & image, size_t x, size_t y )
    {
        return functionTable().GetPixel( image, x, y );
    }

    inline uint8_t GetThreshold( const std::vector < size_t > & histogram )
    {
        return functionTable().GetThreshold( histogram );
    }

    inline void Histogram( const Image & image, size_t x, size_t y, size_t width, size_t height,
                           std::vector < size_t > & histogram )
    {
        functionTable().Histogram( image, x, y, width, height, histogram );
    }

    inline void Invert( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                        size_t width, size_t height )
    {
        functionTable().Invert( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    inline bool IsEqual( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                         size_t width, size_t height )
    {
        return functionTable().IsEqual( in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    inline void LookupTable ( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                              size_t width, size_t height, const std::vector < uint8_t > & table )
    {
        functionTable().LookupTable( in, startXIn, startYIn, out, startXOut, startYOut, width, height, table );
    }

    inline void Maximum( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                         Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        functionTable().Maximum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline void Merge( const Image & in1, size_t startXIn1, size_t startYIn1, const Image & in2, size_t startXIn2, size_t startYIn2,
                       const Image & in3, size_t startXIn3, size_t startYIn3, Image & out, size_t startXOut, size_t startYOut,
                       size_t width, size_t height )
    {
        functionTable().Merge( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3,
                               out, startXOut, startYOut, width, height );
    }

    inline void Minimum( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                         Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        functionTable().Minimum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline void Normalize( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                           size_t width, size_t height )
    {
        functionTable().Normalize( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    inline void ProjectionProfile( const Image & image, size_t x, size_t y, size_t width, size_t height, bool horizontal,
                                   std::vector < size_t > & projection )
    {
        functionTable().ProjectionProfile( image, x, y, width, height, horizontal, projection );
    }

    inline void Resize( const Image & in, size_t startXIn, size_t startYIn, size_t widthIn, size_t heightIn,
                        Image & out, size_t startXOut, size_t startYOut, size_t widthOut, size_t heightOut )
    {
        functionTable().Resize( in, startXIn, startYIn, widthIn, heightIn, out, startXOut, startYOut, widthOut, heightOut );
    }

    inline void RgbToBgr( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                          size_t width, size_t height )
    {
        functionTable().RgbToBgr( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    inline void SetPixel( Image & image, size_t x, size_t y, uint8_t value )
    {
        functionTable().SetPixel( image, x, y, value );
    }

    inline void SetPixel( Image & image, const std::vector < size_t > & X, const std::vector < size_t > & Y, uint8_t value )
    {
        functionTable().SetPixel2( image, X, Y, value );
    }

    inline void Split( const Image & in, size_t startXIn, size_t startYIn, Image & out1, size_t startXOut1, size_t startYOut1,
                       Image & out2, size_t startXOut2, size_t startYOut2, Image & out3, size_t startXOut3, size_t startYOut3,
                       size_t width, size_t height )
    {
        functionTable().Split( in, startXIn, startYIn, out1, startXOut1, startYOut1, out2, startXOut2, startYOut2,
                               out3, startXOut3, startYOut3, width, height );
    }

    inline void Subtract( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                          Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        functionTable().Subtract( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    inline size_t Sum( const Image & image, size_t x, size_t y, size_t width, size_t height )
    {
        return functionTable().Sum( image, x, y, width, height );
    }

    inline void Threshold( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                           size_t width, size_t height, uint8_t threshold )
    {
        functionTable().Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, threshold );
    }

    inline void Threshold( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                           size_t width, size_t height, uint8_t minThreshold, uint8_t maxThreshold )
    {
        functionTable().Threshold2( in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold );
    }

    inline void Transpose( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                           size_t width, size_t height )
    {
        functionTable().Transpose( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }
};
