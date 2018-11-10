#include "image_function_helper.h"

#include <map>
#include <mutex>

#include "parameter_validation.h"

namespace
{
    std::map< uint8_t, Image_Function_Helper::FunctionTableHolder > & functionTableMap()
    {
        static std::map< uint8_t, Image_Function_Helper::FunctionTableHolder > map;
        return map;
    }

    template <typename _Function>
    void setFunction( _Function & F1, const _Function & F2, bool forceSetup )
    {
        if ( (F1 == nullptr) || (forceSetup && (F2 != nullptr)) )
        {
            F1 = F2;
        }
    }

#define SET_FUNCTION( functionName ) \
    setFunction( oldTable.functionName, newTable.functionName, forceSetup );

    void setupTable( Image_Function_Helper::FunctionTableHolder & oldTable, const Image_Function_Helper::FunctionTableHolder & newTable, bool forceSetup )
    {
        SET_FUNCTION(AbsoluteDifference)
        SET_FUNCTION(Accumulate)
        SET_FUNCTION(BitwiseAnd)
        SET_FUNCTION(BitwiseOr)
        SET_FUNCTION(BitwiseXor)
        SET_FUNCTION(ConvertToGrayScale)
        SET_FUNCTION(ConvertToRgb)
        SET_FUNCTION(Copy)
        SET_FUNCTION(ExtractChannel)
        SET_FUNCTION(Fill)
        SET_FUNCTION(Flip)
        SET_FUNCTION(GammaCorrection)
        SET_FUNCTION(GetPixel)
        SET_FUNCTION(Histogram)
        SET_FUNCTION(Invert)
        SET_FUNCTION(IsEqual)
        SET_FUNCTION(LookupTable)
        SET_FUNCTION(Maximum)
        SET_FUNCTION(Merge)
        SET_FUNCTION(Minimum)
        SET_FUNCTION(Normalize)
        SET_FUNCTION(ProjectionProfile)
        SET_FUNCTION(Resize)
        SET_FUNCTION(RgbToBgr)
        SET_FUNCTION(SetPixel)
        SET_FUNCTION(SetPixel2)
        SET_FUNCTION(Split)
        SET_FUNCTION(Subtract)
        SET_FUNCTION(Sum)
        SET_FUNCTION(Threshold)
        SET_FUNCTION(Threshold2)
        SET_FUNCTION(Transpose)
    }
}

namespace Image_Function_Helper
{
    Image AbsoluteDifference( FunctionTable::AbsoluteDifference absoluteDifference,
                              const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height() );

        absoluteDifference( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void AbsoluteDifference( FunctionTable::AbsoluteDifference absoluteDifference,
                             const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        absoluteDifference( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image AbsoluteDifference( FunctionTable::AbsoluteDifference absoluteDifference,
                              const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                              uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height );

        absoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Accumulate( FunctionTable::Accumulate accumulate,
                     const Image & image, std::vector < uint32_t > & result )
    {
        Image_Function::ParameterValidation( image );

        accumulate( image, 0, 0, image.width(), image.height(), result );
    }

    Image BitwiseAnd( FunctionTable::BitwiseAnd bitwiseAnd,
                      const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height() );

        bitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseAnd( FunctionTable::BitwiseAnd bitwiseAnd,
                     const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        bitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image BitwiseAnd( FunctionTable::BitwiseAnd bitwiseAnd,
                      const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height );

        bitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    Image BitwiseOr( FunctionTable::BitwiseOr bitwiseOr,
                     const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height() );

        bitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseOr( FunctionTable::BitwiseOr bitwiseOr,
                    const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        bitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image BitwiseOr( FunctionTable::BitwiseOr bitwiseOr,
                     const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height );

        bitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    Image BitwiseXor( FunctionTable::BitwiseXor bitwiseXor,
                      const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height() );

        bitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseXor( FunctionTable::BitwiseXor bitwiseXor,
                     const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        bitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image BitwiseXor( FunctionTable::BitwiseXor bitwiseXor,
                      const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height );

        bitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    Image ConvertToGrayScale( FunctionTable::ConvertToGrayScale convertToGrayScale,
                              const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        convertToGrayScale( in, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void ConvertToGrayScale( FunctionTable::ConvertToGrayScale convertToGrayScale,
                             const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        convertToGrayScale( in, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image ConvertToGrayScale( FunctionTable::ConvertToGrayScale convertToGrayScale,
                              const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        convertToGrayScale( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    Image ConvertToRgb( FunctionTable::ConvertToRgb convertToRgb,
                        const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height(), RGB );

        convertToRgb( in, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void ConvertToRgb( FunctionTable::ConvertToRgb convertToRgb,
                       const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        convertToRgb( in, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image ConvertToRgb( FunctionTable::ConvertToRgb convertToRgb,
                        const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height, RGB );

        convertToRgb( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    Image Copy( FunctionTable::Copy copy,
                const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        copy( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    Image ExtractChannel( FunctionTable::ExtractChannel extractChannel,
                          const Image & in, uint8_t channelId )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        extractChannel( in, 0, 0, out, 0, 0, in.width(), in.height(), channelId );

        return out;
    }

    void ExtractChannel( FunctionTable::ExtractChannel extractChannel,
                         const Image & in, Image & out, uint8_t channelId )
    {
        Image_Function::ParameterValidation( in, out );

        extractChannel( in, 0, 0, out, 0, 0, in.width(), in.height(), channelId );
    }

    Image ExtractChannel( FunctionTable::ExtractChannel extractChannel,
                          const Image & in, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t channelId )
    {
        Image_Function::ParameterValidation( in, x, y, width, height );

        Image out = in.generate( width, height );

        extractChannel( in, x, y, out, 0, 0, width, height, channelId );

        return out;
    }

    Image Flip( FunctionTable::Flip flip,
                const Image & in, bool horizontal, bool vertical )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        flip( in, 0, 0, out, 0, 0, out.width(), out.height(), horizontal, vertical );

        return out;
    }

    void Flip( FunctionTable::Flip flip,
               const Image & in, Image & out, bool horizontal, bool vertical )
    {
        Image_Function::ParameterValidation( in, out );

        flip( in, 0, 0, out, 0, 0, out.width(), out.height(), horizontal, vertical );
    }

    Image Flip( FunctionTable::Flip flip,
                const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                bool horizontal, bool vertical )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        flip( in, startXIn, startYIn, out, 0, 0, width, height, horizontal, vertical );

        return out;
    }

    Image GammaCorrection( FunctionTable::GammaCorrection gammaCorrection,
                           const Image & in, double a, double gamma )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        gammaCorrection( in, 0, 0, out, 0, 0, out.width(), out.height(), a, gamma );

        return out;
    }

    void GammaCorrection( FunctionTable::GammaCorrection gammaCorrection,
                          const Image & in, Image & out, double a, double gamma )
    {
        Image_Function::ParameterValidation( in, out );

        gammaCorrection( in, 0, 0, out, 0, 0, out.width(), out.height(), a, gamma );
    }

    Image GammaCorrection( FunctionTable::GammaCorrection gammaCorrection,
                           const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double a, double gamma )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        gammaCorrection( in, startXIn, startYIn, out, 0, 0, width, height, a, gamma );

        return out;
    }

    uint8_t GetThreshold( const std::vector < uint32_t > & histogram )
    {
        if( histogram.size() != 256 )
            throw imageException( "Histogram size is not 256" );

        // It is well-known Otsu's method to find threshold
        uint32_t pixelCount = histogram[0] + histogram[1];
        uint32_t sum = histogram[1];
        for( uint16_t i = 2; i < 256; ++i ) {
            sum = sum + i * histogram[i];
            pixelCount += histogram[i];
        }

        uint32_t sumTemp = 0;
        uint32_t pixelCountTemp = 0;

        double maximumSigma = -1;

        uint8_t threshold = 0;

        for( uint16_t i = 0; i < 256; ++i ) {
            pixelCountTemp += histogram[i];

            if( pixelCountTemp == pixelCount )
                break;

            if( pixelCountTemp > 0 ) {
                sumTemp += i * histogram[i];

                const double w1 = static_cast<double>(pixelCountTemp) / pixelCount;
                const double a  = static_cast<double>(sumTemp       ) / pixelCountTemp -
                                  static_cast<double>(sum - sumTemp ) / (pixelCount - pixelCountTemp);
                const double sigma = w1 * (1 - w1) * a * a;

                if( sigma > maximumSigma ) {
                    maximumSigma = sigma;
                    threshold = static_cast <uint8_t>(i);
                }
            }
        }

        return threshold;
    }

    std::vector < uint32_t > Histogram( FunctionTable::Histogram histogram,
                                        const Image & image )
    {
        std::vector < uint32_t > histogramTable;

        histogram( image, 0, 0, image.width(), image.height(), histogramTable );

        return histogramTable;
    }

    void Histogram( FunctionTable::Histogram histogram,
                    const Image & image, std::vector < uint32_t > & histogramTable )
    {
        Image_Function::ParameterValidation( image );

        histogram( image, 0, 0, image.width(), image.height(), histogramTable );
    }

    std::vector < uint32_t > Histogram( FunctionTable::Histogram histogram,
                                        const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( image, x, y, width, height );

        std::vector < uint32_t > histogramTable;

        histogram( image, x, y, width, height, histogramTable );

        return histogramTable;
    }

    Image Invert( FunctionTable::Invert invert,
                  const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        invert( in, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Invert( FunctionTable::Invert invert,
                 const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        invert( in, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Invert( FunctionTable::Invert invert,
                  const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        invert( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    Image LookupTable( FunctionTable::LookupTable lookupTable,
                       const Image & in, const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        lookupTable( in, 0, 0, out, 0, 0, out.width(), out.height(), table );

        return out;
    }

    void LookupTable( FunctionTable::LookupTable lookupTable,
                      const Image & in, Image & out, const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in, out );

        lookupTable( in, 0, 0, out, 0, 0, out.width(), out.height(), table );
    }

    Image LookupTable( FunctionTable::LookupTable lookupTable,
                       const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                       const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        lookupTable( in, startXIn, startYIn, out, 0, 0, width, height, table );

        return out;
    }

    Image Maximum( FunctionTable::Maximum maximum,
                   const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height() );

        maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Maximum( FunctionTable::Maximum maximum,
                  const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Maximum( FunctionTable::Maximum maximum,
                   const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height );

        maximum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    Image Merge( FunctionTable::Merge merge,
                 const Image & in1, const Image & in2, const Image & in3 )
    {
        Image_Function::ParameterValidation( in1, in2, in3 );

        Image out = in1.generate( in1.width(), in1.height(), RGB );

        merge( in1, 0, 0, in2, 0, 0, in3, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Merge( FunctionTable::Merge merge,
                const Image & in1, const Image & in2, const Image & in3, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, in3 );
        Image_Function::ParameterValidation( in1, out );

        merge( in1, 0, 0, in2, 0, 0, in3, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Merge( FunctionTable::Merge merge,
                 const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                 const Image & in3, uint32_t startXIn3, uint32_t startYIn3, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3, width, height );

        Image out = in1.generate( width, height, RGB );

        merge( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3, out, 0, 0, width, height );

        return out;
    }

    Image Minimum( FunctionTable::Minimum minimum,
                   const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height() );

        minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Minimum( FunctionTable::Minimum minimum,
                  const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Minimum( FunctionTable::Minimum minimum,
                   const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height );

        minimum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    Image Normalize( FunctionTable::Normalize normalize,
                     const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        normalize( in, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Normalize( FunctionTable::Normalize normalize,
                    const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        normalize( in, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Normalize( FunctionTable::Normalize normalize,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        normalize( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    std::vector < uint32_t > ProjectionProfile( FunctionTable::ProjectionProfile projectionProfile,
                                                const Image & image, bool horizontal )
    {
        std::vector < uint32_t > projection;

        projectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );

        return projection;
    }

    void ProjectionProfile( FunctionTable::ProjectionProfile projectionProfile,
                            const Image & image, bool horizontal, std::vector < uint32_t > & projection )
    {
        projectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );
    }

    std::vector < uint32_t > ProjectionProfile( FunctionTable::ProjectionProfile projectionProfile,
                                                const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal )
    {
        std::vector < uint32_t > projection;

        projectionProfile( image, x, y, width, height, horizontal, projection );

        return projection;
    }

    Image Resize( FunctionTable::Resize resize,
                  const Image & in, uint32_t widthOut, uint32_t heightOut )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( widthOut, heightOut );

        resize( in, 0, 0, in.width(), in.height(), out, 0, 0, widthOut, heightOut );

        return out;
    }

    void Resize( FunctionTable::Resize resize,
                 const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        resize( in, 0, 0, in.width(), in.height(), out, 0, 0, out.width(), out.height() );
    }

    Image Resize( FunctionTable::Resize resize,
                  const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                  uint32_t widthOut, uint32_t heightOut )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, widthIn, heightIn );

        Image out = in.generate( widthOut, heightOut );

        resize( in, startXIn, startYIn, widthIn, heightIn, out, 0, 0, widthOut, heightOut );

        return out;
    }

    Image RgbToBgr( FunctionTable::RgbToBgr rgbToBgr,
                    const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height(), PenguinV_Image::RGB );

        rgbToBgr( in, 0, 0, out, 0, 0, in.width(), in.height() );

        return out;
    }

    void RgbToBgr( FunctionTable::RgbToBgr rgbToBgr,
                  const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        rgbToBgr( in, 0, 0, out, 0, 0, in.width(), in.height() );
    }

    Image RgbToBgr( FunctionTable::RgbToBgr rgbToBgr,
                    const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height, PenguinV_Image::RGB );

        rgbToBgr( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    Image Subtract( FunctionTable::Subtract subtract,
                    const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height() );

        subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Subtract( FunctionTable::Subtract subtract,
                   const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Subtract( FunctionTable::Subtract subtract,
                    const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height );

        subtract( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    Image Threshold( FunctionTable::Threshold threshold,
                     const Image & in, uint8_t thresholdValue )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), thresholdValue );

        return out;
    }

    void Threshold( FunctionTable::Threshold threshold,
                    const Image & in, Image & out, uint8_t thresholdValue )
    {
        Image_Function::ParameterValidation( in, out );

        threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), thresholdValue );
    }

    Image Threshold( FunctionTable::Threshold threshold,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t thresholdValue )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        threshold( in, startXIn, startYIn, out, 0, 0, width, height, thresholdValue );

        return out;
    }

    Image Threshold( FunctionTable::Threshold2 threshold,
                     const Image & in, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), minThreshold, maxThreshold );

        return out;
    }

    void Threshold( FunctionTable::Threshold2 threshold,
                    const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in, out );

        threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), minThreshold, maxThreshold );
    }

    Image Threshold( FunctionTable::Threshold2 threshold,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t minThreshold,
                     uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        threshold( in, startXIn, startYIn, out, 0, 0, width, height, minThreshold, maxThreshold );

        return out;
    }

    Image Transpose( FunctionTable::Transpose transpose,
                     const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.height(), in.width() );

        transpose( in, 0, 0, out, 0, 0, in.width(), in.height() );

        return out;
    }

    void Transpose( FunctionTable::Transpose transpose,
                    const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        transpose( in, 0, 0, out, 0, 0, in.width(), in.height() );
    }

    Image Transpose( FunctionTable::Transpose transpose,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( height, width );

        transpose( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    void registerFunctionTable( const Image & image, const FunctionTableHolder & table, bool forceSetup )
    {
        static std::mutex mapGuard;

        mapGuard.lock();
        std::map< uint8_t, FunctionTableHolder >::iterator oldTable = functionTableMap().find( image.type() );
        if (oldTable != functionTableMap().end())
            setupTable( oldTable->second, table, forceSetup );
        else
            functionTableMap()[image.type()] = table;
        mapGuard.unlock();
    }

    const FunctionTableHolder & getFunctionTableHolder( const Image & image )
    {
        std::map< uint8_t, FunctionTableHolder >::const_iterator table = functionTableMap().find( image.type() );
        if (table == functionTableMap().end())
            throw imageException( "Function table is not initialised" );

        return table->second;
    }
}
