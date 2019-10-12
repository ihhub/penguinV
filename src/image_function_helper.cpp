#include "image_function_helper.h"
#include <cmath>
#include "parameter_validation.h"
#include "penguinv/cpu_identification.h"

namespace
{
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
        SET_FUNCTION(ConvertTo16Bit)
        SET_FUNCTION(ConvertTo8Bit)
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
    Image AbsoluteDifference( FunctionTable::AbsoluteDifferenceForm4 absoluteDifference,
                              const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height(), in1.colorCount() );

        absoluteDifference( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void AbsoluteDifference( FunctionTable::AbsoluteDifferenceForm4 absoluteDifference,
                             const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        absoluteDifference( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image AbsoluteDifference( FunctionTable::AbsoluteDifferenceForm4 absoluteDifference,
                              const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                              uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height, in1.colorCount() );

        absoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Accumulate( FunctionTable::AccumulateForm2 accumulate,
                     const Image & image, std::vector < uint32_t > & result )
    {
        Image_Function::ParameterValidation( image );

        accumulate( image, 0, 0, image.width(), image.height(), result );
    }

    Image BitwiseAnd( FunctionTable::BitwiseAndForm4 bitwiseAnd,
                      const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height(), in1.colorCount() );

        bitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseAnd( FunctionTable::BitwiseAndForm4 bitwiseAnd,
                     const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        bitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image BitwiseAnd( FunctionTable::BitwiseAndForm4 bitwiseAnd,
                      const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height, in1.colorCount() );

        bitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    Image BitwiseOr( FunctionTable::BitwiseOrForm4 bitwiseOr,
                     const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height(), in1.colorCount() );

        bitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseOr( FunctionTable::BitwiseOrForm4 bitwiseOr,
                    const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        bitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image BitwiseOr( FunctionTable::BitwiseOrForm4 bitwiseOr,
                     const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height, in1.colorCount() );

        bitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    Image BitwiseXor( FunctionTable::BitwiseXorForm4 bitwiseXor,
                      const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height(), in1.colorCount() );

        bitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseXor( FunctionTable::BitwiseXorForm4 bitwiseXor,
                     const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        bitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image BitwiseXor( FunctionTable::BitwiseXorForm4 bitwiseXor,
                      const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height, in1.colorCount() );

        bitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void ConvertTo16Bit( FunctionTable::ConvertTo16BitForm4 convertTo16Bit,
                         const Image & in, Image16Bit & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        convertTo16Bit( in, 0, 0, out, 0, 0, in.width(), in.height() );
    }

    void ConvertTo8Bit( FunctionTable::ConvertTo8BitForm4 convertTo8Bit,
                        const Image16Bit & in, Image & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        convertTo8Bit( in, 0, 0, out, 0, 0, in.width(), in.height() );
    }

    Image ConvertToGrayScale( FunctionTable::ConvertToGrayScaleForm4 convertToGrayScale,
                              const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        convertToGrayScale( in, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void ConvertToGrayScale( FunctionTable::ConvertToGrayScaleForm4 convertToGrayScale,
                             const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        convertToGrayScale( in, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image ConvertToGrayScale( FunctionTable::ConvertToGrayScaleForm4 convertToGrayScale,
                              const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        convertToGrayScale( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    Image ConvertToRgb( FunctionTable::ConvertToRgbForm4 convertToRgb,
                        const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height(), RGB );

        convertToRgb( in, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void ConvertToRgb( FunctionTable::ConvertToRgbForm4 convertToRgb,
                       const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        convertToRgb( in, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image ConvertToRgb( FunctionTable::ConvertToRgbForm4 convertToRgb,
                        const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height, RGB );

        convertToRgb( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    Image Copy( FunctionTable::CopyForm3 copy,
                const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height, in.colorCount() );

        copy( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    Image ExtractChannel( FunctionTable::ExtractChannelForm4 extractChannel,
                          const Image & in, uint8_t channelId )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        extractChannel( in, 0, 0, out, 0, 0, in.width(), in.height(), channelId );

        return out;
    }

    void ExtractChannel( FunctionTable::ExtractChannelForm4 extractChannel,
                         const Image & in, Image & out, uint8_t channelId )
    {
        Image_Function::ParameterValidation( in, out );

        extractChannel( in, 0, 0, out, 0, 0, in.width(), in.height(), channelId );
    }

    Image ExtractChannel( FunctionTable::ExtractChannelForm4 extractChannel,
                          const Image & in, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t channelId )
    {
        Image_Function::ParameterValidation( in, x, y, width, height );

        Image out = in.generate( width, height );

        extractChannel( in, x, y, out, 0, 0, width, height, channelId );

        return out;
    }

    Image Flip( FunctionTable::FlipForm4 flip,
                const Image & in, bool horizontal, bool vertical )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        flip( in, 0, 0, out, 0, 0, out.width(), out.height(), horizontal, vertical );

        return out;
    }

    void Flip( FunctionTable::FlipForm4 flip,
               const Image & in, Image & out, bool horizontal, bool vertical )
    {
        Image_Function::ParameterValidation( in, out );

        flip( in, 0, 0, out, 0, 0, out.width(), out.height(), horizontal, vertical );
    }

    Image Flip( FunctionTable::FlipForm4 flip,
                const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                bool horizontal, bool vertical )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        flip( in, startXIn, startYIn, out, 0, 0, width, height, horizontal, vertical );

        return out;
    }

    Image GammaCorrection( FunctionTable::GammaCorrectionForm4 gammaCorrection,
                           const Image & in, double a, double gamma )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        gammaCorrection( in, 0, 0, out, 0, 0, out.width(), out.height(), a, gamma );

        return out;
    }

    void GammaCorrection( FunctionTable::GammaCorrectionForm4 gammaCorrection,
                          const Image & in, Image & out, double a, double gamma )
    {
        Image_Function::ParameterValidation( in, out );

        gammaCorrection( in, 0, 0, out, 0, 0, out.width(), out.height(), a, gamma );
    }

    Image GammaCorrection( FunctionTable::GammaCorrectionForm4 gammaCorrection,
                           const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double a, double gamma )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        gammaCorrection( in, startXIn, startYIn, out, 0, 0, width, height, a, gamma );

        return out;
    }

    std::vector<uint8_t> GetGammaCorrectionLookupTable( double a, double gamma )
    {
        if ( a < 0 || gamma < 0 )
            throw imageException( "Gamma correction parameters are invalid" );

        // We precalculate all values and store them in lookup table
        std::vector<uint8_t> value( 256, 255u );

        for ( uint16_t i = 0; i < 256; ++i ) {
            double data = a * pow( i / 255.0, gamma ) * 255 + 0.5;

            if ( data < 256 )
                value[i] = static_cast<uint8_t>( data );
        }

        return value;
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

    std::vector < uint32_t > Histogram( FunctionTable::HistogramForm4 histogram,
                                        const Image & image )
    {
        std::vector < uint32_t > histogramTable;

        histogram( image, 0, 0, image.width(), image.height(), histogramTable );

        return histogramTable;
    }

    void Histogram( FunctionTable::HistogramForm4 histogram,
                    const Image & image, std::vector < uint32_t > & histogramTable )
    {
        Image_Function::ParameterValidation( image );

        histogram( image, 0, 0, image.width(), image.height(), histogramTable );
    }

    std::vector < uint32_t > Histogram( FunctionTable::HistogramForm4 histogram,
                                        const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( image, x, y, width, height );

        std::vector < uint32_t > histogramTable;

        histogram( image, x, y, width, height, histogramTable );

        return histogramTable;
    }

    std::vector < uint32_t > Histogram( FunctionTable::HistogramForm8 histogram,
                                        const Image & image, const Image & mask )
    {
        Image_Function::ParameterValidation( image, mask );

        std::vector < uint32_t > histogramTable;

        histogram( image, 0, 0, mask, 0, 0, image.width(), image.height(), histogramTable );

        return histogramTable;
    }

    void Histogram( FunctionTable::HistogramForm8 histogram,
                    const Image & image, const Image & mask, std::vector < uint32_t > & histogramTable )
    {
        Image_Function::ParameterValidation( image, mask );

        histogram( image, 0, 0, mask, 0, 0, image.width(), image.height(), histogramTable );
    }

    std::vector < uint32_t > Histogram( FunctionTable::HistogramForm8 histogram,
                                        const Image & image, uint32_t x, uint32_t y, const Image & mask, uint32_t maskX, uint32_t maskY,
                                        uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( image, x, y, mask, maskX, maskY, width, height );

        std::vector < uint32_t > histogramTable;

        histogram( image, x, y, mask, maskX, maskY, width, height, histogramTable );

        return histogramTable;
    }

    Image Invert( FunctionTable::InvertForm4 invert,
                  const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height(), in.colorCount() );

        invert( in, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Invert( FunctionTable::InvertForm4 invert,
                 const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        invert( in, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Invert( FunctionTable::InvertForm4 invert,
                  const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height, in.colorCount() );

        invert( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    bool IsEqual( FunctionTable::IsEqualForm2 isEqual,
                  const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        return isEqual( in1, 0, 0, in2, 0, 0, in1.width(), in1.height() );
    }

    Image LookupTable( FunctionTable::LookupTableForm4 lookupTable,
                       const Image & in, const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        lookupTable( in, 0, 0, out, 0, 0, out.width(), out.height(), table );

        return out;
    }

    void LookupTable( FunctionTable::LookupTableForm4 lookupTable,
                      const Image & in, Image & out, const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in, out );

        lookupTable( in, 0, 0, out, 0, 0, out.width(), out.height(), table );
    }

    Image LookupTable( FunctionTable::LookupTableForm4 lookupTable,
                       const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                       const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        lookupTable( in, startXIn, startYIn, out, 0, 0, width, height, table );

        return out;
    }

    Image Maximum( FunctionTable::MaximumForm4 maximum,
                   const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height(), in1.colorCount() );

        maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Maximum( FunctionTable::MaximumForm4 maximum,
                  const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Maximum( FunctionTable::MaximumForm4 maximum,
                   const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height, in1.colorCount() );

        maximum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    Image Merge( FunctionTable::MergeForm4 merge,
                 const Image & in1, const Image & in2, const Image & in3 )
    {
        Image_Function::ParameterValidation( in1, in2, in3 );

        Image out = in1.generate( in1.width(), in1.height(), RGB );

        merge( in1, 0, 0, in2, 0, 0, in3, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Merge( FunctionTable::MergeForm4 merge,
                const Image & in1, const Image & in2, const Image & in3, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, in3 );
        Image_Function::ParameterValidation( in1, out );

        merge( in1, 0, 0, in2, 0, 0, in3, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Merge( FunctionTable::MergeForm4 merge,
                 const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                 const Image & in3, uint32_t startXIn3, uint32_t startYIn3, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3, width, height );

        Image out = in1.generate( width, height, RGB );

        merge( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3, out, 0, 0, width, height );

        return out;
    }

    Image Minimum( FunctionTable::MinimumForm4 minimum,
                   const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height(), in1.colorCount() );

        minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Minimum( FunctionTable::MinimumForm4 minimum,
                  const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Minimum( FunctionTable::MinimumForm4 minimum,
                   const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height, in1.colorCount() );

        minimum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    Image Normalize( FunctionTable::NormalizeForm4 normalize,
                     const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        normalize( in, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Normalize( FunctionTable::NormalizeForm4 normalize,
                    const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        normalize( in, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Normalize( FunctionTable::NormalizeForm4 normalize,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        normalize( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    std::vector < uint32_t > ProjectionProfile( FunctionTable::ProjectionProfileForm4 projectionProfile,
                                                const Image & image, bool horizontal )
    {
        std::vector < uint32_t > projection;

        projectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );

        return projection;
    }

    void ProjectionProfile( FunctionTable::ProjectionProfileForm4 projectionProfile,
                            const Image & image, bool horizontal, std::vector < uint32_t > & projection )
    {
        projectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );
    }

    std::vector < uint32_t > ProjectionProfile( FunctionTable::ProjectionProfileForm4 projectionProfile,
                                                const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal )
    {
        std::vector < uint32_t > projection;

        projectionProfile( image, x, y, width, height, horizontal, projection );

        return projection;
    }

    Image Resize( FunctionTable::ResizeForm4 resize,
                  const Image & in, uint32_t widthOut, uint32_t heightOut )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( widthOut, heightOut );

        resize( in, 0, 0, in.width(), in.height(), out, 0, 0, widthOut, heightOut );

        return out;
    }

    void Resize( FunctionTable::ResizeForm4 resize,
                 const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        resize( in, 0, 0, in.width(), in.height(), out, 0, 0, out.width(), out.height() );
    }

    Image Resize( FunctionTable::ResizeForm4 resize,
                  const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                  uint32_t widthOut, uint32_t heightOut )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, widthIn, heightIn );

        Image out = in.generate( widthOut, heightOut );

        resize( in, startXIn, startYIn, widthIn, heightIn, out, 0, 0, widthOut, heightOut );

        return out;
    }

    Image RgbToBgr( FunctionTable::RgbToBgrForm4 rgbToBgr,
                    const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height(), RGB );

        rgbToBgr( in, 0, 0, out, 0, 0, in.width(), in.height() );

        return out;
    }

    void RgbToBgr( FunctionTable::RgbToBgrForm4 rgbToBgr,
                  const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        rgbToBgr( in, 0, 0, out, 0, 0, in.width(), in.height() );
    }

    Image RgbToBgr( FunctionTable::RgbToBgrForm4 rgbToBgr,
                    const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height, RGB );

        rgbToBgr( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    Image RgbToRgba( FunctionTable::RgbToRgbaForm4 rgbToRgba,
                     const Image & in )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::VerifyRGBImage( in );

        Image out = in.generate( in.width(), in.height(), RGBA );

        rgbToRgba( in, 0, 0, out, 0, 0, in.width(), in.height() );

        return out;
    }

    void RgbToRgba( FunctionTable::RgbToRgbaForm4 rgbToRgba,
                    const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );
        Image_Function::VerifyRGBImage( in );
        Image_Function::VerifyRGBAImage( out );

        rgbToRgba( in, 0, 0, out, 0, 0, in.width(), in.height() );
    }

    Image RgbToRgba( FunctionTable::RgbToRgbaForm4 rgbToRgba,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );
        Image_Function::VerifyRGBImage( in );

        Image out = in.generate( width, height, RGBA );

        rgbToRgba( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    Image RgbaToRgb( FunctionTable::RgbaToRgbForm4 rgbaToRgb,
                     const Image & in )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::VerifyRGBAImage( in );

        Image out = in.generate( in.width(), in.height(), RGB );

        rgbaToRgb( in, 0, 0, out, 0, 0, in.width(), in.height() );

        return out;
    }

    void RgbaToRgb( FunctionTable::RgbaToRgbForm4 rgbaToRgb,
                    const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );
        Image_Function::VerifyRGBAImage( in );
        Image_Function::VerifyRGBImage( out );

        rgbaToRgb( in, 0, 0, out, 0, 0, in.width(), in.height() );
    }

    Image RgbaToRgb( FunctionTable::RgbaToRgbForm4 rgbaToRgb,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );
        Image_Function::VerifyRGBAImage( in );

        Image out = in.generate( width, height, RGB );

        rgbaToRgb( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    Image Rotate( FunctionTable::RotateForm4 rotate,
                  const Image & in, double centerX, double centerY, double angle )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::VerifyGrayScaleImage( in );
        Image out = in.generate( in.width(), in.height() );

        rotate( in, 0, 0, centerX, centerY, out, 0, 0, centerX, centerY, in.width(), in.height(), angle );

        return out;
    }

    void Rotate( FunctionTable::RotateForm4 rotate,
                 const Image & in, double centerXIn, double centerYIn, Image & out, double centerXOut, double centerYOut, double angle )
    {
        Image_Function::ParameterValidation( in, out );
        Image_Function::VerifyGrayScaleImage( in, out );
        rotate( in, 0, 0, centerXIn, centerYIn, out, 0, 0, centerXOut, centerYOut, in.width(), in.height(), angle );
    }

    Image Rotate( FunctionTable::RotateForm4 rotate,
                  const Image & in, uint32_t x, uint32_t y, double centerX, double centerY, uint32_t width, uint32_t height, double angle )
    {
        Image_Function::ParameterValidation( in, x, y, width, height );
        Image_Function::VerifyGrayScaleImage( in );

        Image out = in.generate( width, height, in.colorCount() );

        rotate( in, x, y, centerX, centerY, out, 0, 0, centerX, centerY, width, height, angle );

        return out;
    }

    Image Shift( FunctionTable::ShiftForm4 shift,
                 const Image & in, double shiftX, double shiftY )
    {
        Image_Function::ParameterValidation( in );
        Image out = in.generate( in.width(), in.height() );

        shift( in, 0, 0, out, 0, 0, out.width(), out.height(), shiftX, shiftY );
        return out;
    }

    void  Shift( FunctionTable::ShiftForm4 shift,
                 const Image & in, Image & out, double shiftX, double shiftY )
    {
        Image_Function::ParameterValidation( in, out );

        shift( in, 0, 0, out, 0, 0, out.width(), out.height(), shiftX, shiftY );
    }

    Image Shift( FunctionTable::ShiftForm4 shift,
                 const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double shiftX, double shiftY )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        shift( in, startXIn, startYIn, out, 0, 0, width, height, shiftX, shiftY );

        return out;
    }

    void Split( FunctionTable::SplitForm2 split,
                const Image & in, Image & out1, Image & out2, Image & out3 )
    {
        Image_Function::ParameterValidation( in, out1, out2 );
        Image_Function::ParameterValidation( in, out3 );

        split( in, 0, 0, out1, 0, 0, out2, 0, 0, out3, 0, 0, in.width(), in.height() );
    }

    Image Subtract( FunctionTable::SubtractForm4 subtract,
                    const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out = in1.generate( in1.width(), in1.height(), in1.colorCount() );

        subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Subtract( FunctionTable::SubtractForm4 subtract,
                   const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Subtract( FunctionTable::SubtractForm4 subtract,
                    const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out = in1.generate( width, height, in1.colorCount() );

        subtract( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    Image Threshold( FunctionTable::ThresholdForm4 threshold,
                     const Image & in, uint8_t thresholdValue )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), thresholdValue );

        return out;
    }

    void Threshold( FunctionTable::ThresholdForm4 threshold,
                    const Image & in, Image & out, uint8_t thresholdValue )
    {
        Image_Function::ParameterValidation( in, out );

        threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), thresholdValue );
    }

    Image Threshold( FunctionTable::ThresholdForm4 threshold,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t thresholdValue )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        threshold( in, startXIn, startYIn, out, 0, 0, width, height, thresholdValue );

        return out;
    }

    Image Threshold( FunctionTable::ThresholdDoubleForm4 threshold,
                     const Image & in, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.width(), in.height() );

        threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), minThreshold, maxThreshold );

        return out;
    }

    void Threshold( FunctionTable::ThresholdDoubleForm4 threshold,
                    const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in, out );

        threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), minThreshold, maxThreshold );
    }

    Image Threshold( FunctionTable::ThresholdDoubleForm4 threshold,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t minThreshold,
                     uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( width, height );

        threshold( in, startXIn, startYIn, out, 0, 0, width, height, minThreshold, maxThreshold );

        return out;
    }

    Image Transpose( FunctionTable::TransposeForm4 transpose,
                     const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out = in.generate( in.height(), in.width() );

        transpose( in, 0, 0, out, 0, 0, in.width(), in.height() );

        return out;
    }

    void Transpose( FunctionTable::TransposeForm4 transpose,
                    const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        transpose( in, 0, 0, out, 0, 0, in.width(), in.height() );
    }

    Image Transpose( FunctionTable::TransposeForm4 transpose,
                     const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = in.generate( height, width );

        transpose( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }
}

ImageTypeManager::ImageTypeManager()
    : _enabledIntertypeConversion( false )
{
}

ImageTypeManager & ImageTypeManager::instance()
{
    static ImageTypeManager manager;
    return manager;
}

void ImageTypeManager::setFunctionTable( uint8_t type, const Image_Function_Helper::FunctionTableHolder & table, bool forceSetup )
{
    std::map< uint8_t, Image_Function_Helper::FunctionTableHolder >::iterator oldTable = _functionTableMap.find( type );
    if ( oldTable != _functionTableMap.end() )
        setupTable( oldTable->second, table, forceSetup );
    else
        _functionTableMap[type] = table;
}

const Image_Function_Helper::FunctionTableHolder & ImageTypeManager::functionTable( uint8_t type ) const
{
    std::map< uint8_t, Image_Function_Helper::FunctionTableHolder >::const_iterator table = _functionTableMap.find( type );
    if ( table == _functionTableMap.end() )
        throw imageException( "Function table is not initialised" );

    return table->second;
}

void ImageTypeManager::setConvertFunction( Image_Function_Helper::FunctionTable::CopyForm1 Copy, const PenguinV_Image::Image & in, const PenguinV_Image::Image & out )
{
    if ( in.type() == out.type() )
        throw imageException( "Cannot register same type images for intertype copy" );

    _intertypeConvertMap[std::pair<uint8_t, uint8_t>( in.type(), out.type() )] = Copy;

    _image[in.type()] = in.generate();
    _image[out.type()] = out.generate();
}

void ImageTypeManager::convert( const PenguinV_Image::Image & in, PenguinV_Image::Image & out ) const
{
    std::map< std::pair<uint8_t, uint8_t>, Image_Function_Helper::FunctionTable::CopyForm1 >::const_iterator copy =
        _intertypeConvertMap.find( std::pair<uint8_t, uint8_t>( in.type(), out.type() ) );
    if ( copy == _intertypeConvertMap.cend() )
        throw imageException( "Copy function between different image types is not registered" );

    copy->second( in, out );
}

PenguinV_Image::Image ImageTypeManager::image( uint8_t type ) const
{
    std::map< uint8_t, PenguinV_Image::Image >::const_iterator image = _image.find( type );
    if ( image == _image.cend() )
        throw imageException( "Image is not registered" );

    return image->second;
}

std::vector< uint8_t > ImageTypeManager::imageTypes() const
{
    std::vector< uint8_t > type;

    for ( std::map< uint8_t, Image_Function_Helper::FunctionTableHolder >::const_iterator item = _functionTableMap.cbegin(); item != _functionTableMap.cend(); ++item )
        type.push_back( item->first );

    return type;
}

void ImageTypeManager::enableIntertypeConversion( bool enable )
{
    _enabledIntertypeConversion = enable;
}

bool ImageTypeManager::isIntertypeConversionEnabled() const
{
    return _enabledIntertypeConversion;
}

namespace simd
{
    bool isAvx512Enabled = true;
    bool isAvxEnabled = true;
    bool isSseEnabled = true;
    bool isNeonEnabled = true;

    void EnableSimd( bool enable )
    {
        EnableAvx512(enable);
        EnableAvx( enable );
        EnableSse( enable );
        EnableNeon( enable );
    }

    void EnableAvx512( bool enable )
    {
        isAvx512Enabled = enable;
    }
    
    void EnableAvx( bool enable )
    {
        isAvxEnabled = enable;
    }
    
    void EnableSse( bool enable )
    {
        isSseEnabled = enable;
    }
    
    void EnableNeon( bool enable )
    {
        isNeonEnabled = enable;
    }

    SIMDType actualSimdType()
    {
        #ifdef PENGUINV_AVX512BW_SET
        if ( SimdInfo::isAvxAvailable() && isAvxEnabled )
            return avx_function;
        #endif

        #ifdef PENGUINV_AVX_SET
        if ( SimdInfo::isAVX512BWAvailable() && isAvx512Enabled )
            return avx512_function;
        #endif

        #ifdef PENGUINV_SSE_SET
        if ( SimdInfo::isSseAvailable() && isSseEnabled )
            return sse_function;
        #endif

        #ifdef PENGUINV_NEON_SET
        if ( SimdInfo::isNeonAvailable() && isNeonEnabled )
            return neon_function;
        #endif

        return cpu_function;
    }
}
