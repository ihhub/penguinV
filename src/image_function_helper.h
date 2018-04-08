#include <vector>
#include "image_buffer.h"

namespace Image_Function_Helper
{
    using namespace PenguinV_Image;

    namespace FunctionTable
    {
        typedef void ( *AbsoluteDifference )(const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                                            Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height);
        typedef void ( *Accumulate )       (const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                            std::vector < uint32_t > & result);
        typedef void ( *BitwiseAnd )       (const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                                            Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height);
        typedef void ( *BitwiseOr )        (const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                                            Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height);
        typedef void ( *BitwiseXor )       (const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                                            Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height);
        typedef void ( *ConvertToGrayScale )(const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                            uint32_t width, uint32_t height);
        typedef void ( *ConvertToRgb )     (const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                            uint32_t width, uint32_t height);
        typedef void ( *Copy )             (const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                            uint32_t width, uint32_t height);
        typedef void ( *ExtractChannel )   (const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
                                            uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId);
        typedef void ( *Fill )             (Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value);
        typedef void ( *Flip )             (const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                            uint32_t width, uint32_t height, bool horizontal, bool vertical);
        typedef void ( *GammaCorrection )  (const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                            uint32_t width, uint32_t height, double a, double gamma);
        typedef uint8_t ( *GetPixel )      (const Image & image, uint32_t x, uint32_t y);
        typedef uint8_t ( *GetThreshold )  (const std::vector < uint32_t > & histogram);
        typedef void ( *Histogram )        (const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                            std::vector < uint32_t > & histogram);
        typedef void ( *Invert )           (const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                            uint32_t width, uint32_t height);
        typedef bool ( *IsEqual )          (const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                                            uint32_t width, uint32_t height);
        typedef void ( *LookupTable )      (const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                            uint32_t width, uint32_t height, const std::vector < uint8_t > & table);
        typedef void ( *Maximum )          (const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                                            Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height);
        typedef void ( *Merge )            (const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2,
                                            uint32_t startXIn2, uint32_t startYIn2, const Image & in3, uint32_t startXIn3, uint32_t startYIn3,
                                            Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height);
        typedef void ( *Minimum )          (const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                                            Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height);
        typedef void ( *Normalize )        (const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                            uint32_t width, uint32_t height);
        typedef void ( *ProjectionProfile )(const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                                            std::vector < uint32_t > & projection);
        typedef void ( *Resize )           (const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                                            Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut);
        typedef void ( *RgbToBgr )         (const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                            uint32_t width, uint32_t height);
        typedef void ( *SetPixel )         (Image & image, uint32_t x, uint32_t y, uint8_t value);
        typedef void ( *SetPixel2 )        (Image & image, const std::vector < uint32_t > & X, const std::vector < uint32_t > & Y, uint8_t value);
        typedef void ( *Split )            (const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out1, uint32_t startXOut1, uint32_t startYOut1,
                                            Image & out2, uint32_t startXOut2, uint32_t startYOut2, Image & out3, uint32_t startXOut3, uint32_t startYOut3,
                                            uint32_t width, uint32_t height);
        typedef void ( *Subtract )         (const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                                            Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height);
        typedef uint32_t ( *Sum )          (const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height);
        typedef void ( *Threshold )        (const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                            uint32_t width, uint32_t height, uint8_t threshold);
        typedef void ( *Threshold2 )       (const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                            uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold);
        typedef void ( *Transpose )        (const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                                            uint32_t width, uint32_t height);
    }

    Image AbsoluteDifference( FunctionTable::AbsoluteDifference absoluteDifference,
                              const Image & in1, const Image & in2 );

    void AbsoluteDifference( FunctionTable::AbsoluteDifference absoluteDifference,
                             const Image & in1, const Image & in2, Image & out );

    Image AbsoluteDifference( FunctionTable::AbsoluteDifference absoluteDifference,
                              const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                              uint32_t width, uint32_t height );

    void Accumulate( FunctionTable::Accumulate accumulate,
                     const Image & image, std::vector < uint32_t > & result );

    Image BitwiseAnd( FunctionTable::BitwiseAnd bitwiseAnd,
                      const Image & in1, const Image & in2 );

    void BitwiseAnd( FunctionTable::BitwiseAnd bitwiseAnd,
                     const Image & in1, const Image & in2, Image & out );

    Image BitwiseAnd( FunctionTable::BitwiseAnd bitwiseAnd,
                      const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height );

    Image BitwiseOr( FunctionTable::BitwiseOr bitwiseOr,
                     const Image & in1, const Image & in2 );

    void BitwiseOr( FunctionTable::BitwiseOr bitwiseOr,
                    const Image & in1, const Image & in2, Image & out );

    Image BitwiseOr( FunctionTable::BitwiseOr bitwiseOr,
                     const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     uint32_t width, uint32_t height );

    Image BitwiseXor( FunctionTable::BitwiseXor bitwiseXor,
                      const Image & in1, const Image & in2 );

    void BitwiseXor( FunctionTable::BitwiseXor bitwiseXor,
                     const Image & in1, const Image & in2, Image & out );

    Image BitwiseXor( FunctionTable::BitwiseXor bitwiseXor,
                      const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height );

    Image ConvertToGrayScale( FunctionTable::ConvertToGrayScale convertToGrayScale,
                              const Image & in );

    void ConvertToGrayScale( FunctionTable::ConvertToGrayScale convertToGrayScale,
                             const Image & in, Image & out );

    Image ConvertToGrayScale( FunctionTable::ConvertToGrayScale convertToGrayScale,
                              const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    Image ConvertToRgb( FunctionTable::ConvertToRgb convertToRgb,
                        const Image & in );

    void ConvertToRgb( FunctionTable::ConvertToRgb convertToRgb,
                       const Image & in, Image & out );

    Image ConvertToRgb( FunctionTable::ConvertToRgb convertToRgb,
                        const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    Image Copy( FunctionTable::Copy copy,
                const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    Image ExtractChannel( FunctionTable::ExtractChannel extractChannel,
                          const Image & in, uint8_t channelId );

    void ExtractChannel( FunctionTable::ExtractChannel extractChannel,
                         const Image & in, Image & out, uint8_t channelId );

    Image ExtractChannel( FunctionTable::ExtractChannel extractChannel,
                          const Image & in, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t channelId );

    Image Flip( FunctionTable::Flip flip,
                const Image & in, bool horizontal, bool vertical );

    void Flip( FunctionTable::Flip flip,
               const Image & in, Image & out, bool horizontal, bool vertical );

    Image Flip( FunctionTable::Flip flip,
                const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                bool horizontal, bool vertical );

    Image GammaCorrection( FunctionTable::GammaCorrection gammaCorrection,
                           const Image & in, double a, double gamma );

    void GammaCorrection( FunctionTable::GammaCorrection gammaCorrection,
                          const Image & in, Image & out, double a, double gamma );

    Image GammaCorrection( FunctionTable::GammaCorrection gammaCorrection,
                           const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double a, double gamma );

    std::vector < uint32_t > Histogram( FunctionTable::Histogram histogram,
                                        const Image & image );

    void Histogram( FunctionTable::Histogram histogram,
                    const Image & image, std::vector < uint32_t > & histogramTable );

    std::vector < uint32_t > Histogram( FunctionTable::Histogram histogram,
                                        const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height );

    Image Invert( FunctionTable::Invert invert,
                  const Image & in );

    void Invert( FunctionTable::Invert invert,
                 const Image & in, Image & out );

    Image Invert( FunctionTable::Invert invert,
                  const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    Image LookupTable( FunctionTable::LookupTable lookupTable,
                       const Image & in, const std::vector < uint8_t > & table );

    void LookupTable( FunctionTable::LookupTable lookupTable,
                      const Image & in, Image & out, const std::vector < uint8_t > & table );

    Image LookupTable( FunctionTable::LookupTable lookupTable,
                       const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                       const std::vector < uint8_t > & table );

    Image Maximum( FunctionTable::Maximum maximum,
                   const Image & in1, const Image & in2 );

    void Maximum( FunctionTable::Maximum maximum,
                  const Image & in1, const Image & in2, Image & out );

    Image Maximum( FunctionTable::Maximum maximum,
                   const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height );

    Image Merge( FunctionTable::Merge merge,
                 const Image & in1, const Image & in2, const Image & in3 );

    void Merge( FunctionTable::Merge merge,
                const Image & in1, const Image & in2, const Image & in3, Image & out );

    Image Merge( FunctionTable::Merge merge,
                 const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                 const Image & in3, uint32_t startXIn3, uint32_t startYIn3, uint32_t width, uint32_t height );

    Image Minimum( FunctionTable::Minimum minimum,
                   const Image & in1, const Image & in2 );

    void Minimum( FunctionTable::Minimum minimum,
                  const Image & in1, const Image & in2, Image & out );

    Image Minimum( FunctionTable::Minimum minimum,
                   const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height );

    Image Normalize( FunctionTable::Normalize normalize,
                     const Image & in );

    void Normalize( FunctionTable::Normalize normalize,
                    const Image & in, Image & out );

    Image Normalize( FunctionTable::Normalize normalize,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    std::vector < uint32_t > ProjectionProfile( FunctionTable::ProjectionProfile projectionProfile,
                                                const Image & image, bool horizontal );

    void ProjectionProfile( FunctionTable::ProjectionProfile projectionProfile,
                            const Image & image, bool horizontal, std::vector < uint32_t > & projection );

    std::vector < uint32_t > ProjectionProfile( FunctionTable::ProjectionProfile projectionProfile,
                                                const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal );

    Image Resize( FunctionTable::Resize resize,
                  const Image & in, uint32_t widthOut, uint32_t heightOut );

    void Resize( FunctionTable::Resize resize,
                 const Image & in, Image & out );

    Image Resize( FunctionTable::Resize resize,
                  const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                  uint32_t widthOut, uint32_t heightOut );

    Image RgbToBgr( FunctionTable::RgbToBgr rgbToBgr,
                    const Image & in );

    void RgbToBgr( FunctionTable::RgbToBgr rgbToBgr,
                  const Image & in, Image & out );

    Image RgbToBgr( FunctionTable::RgbToBgr rgbToBgr,
                    const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );

    Image Subtract( FunctionTable::Subtract subtract,
                    const Image & in1, const Image & in2 );

    void Subtract( FunctionTable::Subtract subtract,
                   const Image & in1, const Image & in2, Image & out );

    Image Subtract( FunctionTable::Subtract subtract,
                    const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    uint32_t width, uint32_t height );

    Image Threshold( FunctionTable::Threshold threshold,
                     const Image & in, uint8_t thresholdValue );

    void Threshold( FunctionTable::Threshold threshold,
                    const Image & in, Image & out, uint8_t thresholdValue );

    Image Threshold( FunctionTable::Threshold threshold,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t thresholdValue );

    Image Threshold( FunctionTable::Threshold2 threshold,
                     const Image & in, uint8_t minThreshold, uint8_t maxThreshold );

    void Threshold( FunctionTable::Threshold2 threshold,
                    const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold );

    Image Threshold( FunctionTable::Threshold2 threshold,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t minThreshold,
                     uint8_t maxThreshold );

    Image Transpose( FunctionTable::Transpose transpose,
                     const Image & in );

    void Transpose( FunctionTable::Transpose transpose,
                    const Image & in, Image & out );

    Image Transpose( FunctionTable::Transpose transpose,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
}
