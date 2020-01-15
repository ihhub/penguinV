#include "penguinv.h"
#include "../image_function_helper.h"
#include "image_manager.h"

namespace
{
    penguinV::Image generateImage( uint8_t imageType )
    {
        return ImageTypeManager::instance().image( imageType );
    }

    void convertImage( const penguinV::Image & in, penguinV::Image & out )
    {
        ImageTypeManager::instance().convert( in, out );
    }

    template <typename _T>
    void verifyFunction(_T func, const char * functionName)
    {
        if ( func == nullptr ) {
            const std::string error( std::string("Function ") + std::string(functionName) + std::string(" is not defined") );
            throw imageException(error.data());
        }
    }

#define initialize( image, func_ )                                                                           \
    ImageTypeManager & registrator = ImageTypeManager::instance();                                           \
    auto func = registrator.functionTable( image.type() ).func_;                                             \
    uint8_t imageType = image.type();                                                                        \
    if ( func == nullptr && registrator.isIntertypeConversionEnabled() ) {                                   \
        const std::vector< uint8_t > & types = registrator.imageTypes();                                     \
        for ( std::vector< uint8_t >::const_iterator type = types.cbegin(); type != types.cend(); ++type ) { \
            auto funcTemp = registrator.functionTable( *type ).func_;                                        \
            if ( funcTemp != nullptr  ) {                                                                    \
                func = funcTemp;                                                                             \
                imageType = *type;                                                                           \
                break;                                                                                       \
            }                                                                                                \
        }                                                                                                    \
    }                                                                                                        \
    verifyFunction( func, #func_ );                                                                          \
    ImageManager manager( imageType, generateImage, convertImage );
}

namespace penguinV
{
    Image AbsoluteDifference( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::AbsoluteDifference( AbsoluteDifference, in1, in2 );
    }

    void AbsoluteDifference( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::AbsoluteDifference( AbsoluteDifference, in1, in2, out );
    }

    Image AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                              uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::AbsoluteDifference( AbsoluteDifference, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                             Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, AbsoluteDifference )
        func( manager(in1), startX1, startY1, manager(in2), startX2, startY2, manager(out), startXOut, startYOut, width, height );
    }

    void Accumulate( const Image & image, std::vector < uint32_t > & result )
    {
        Image_Function_Helper::Accumulate( Accumulate, image, result );
    }

    void Accumulate( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & result )
    {
        initialize( image, Accumulate )
        func( image, x, y, width, height, result );
    }

    Image BitwiseAnd( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::BitwiseAnd( BitwiseAnd, in1, in2 );
    }

    void BitwiseAnd( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::BitwiseAnd( BitwiseAnd, in1, in2, out );
    }

    Image BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseAnd( BitwiseAnd, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, BitwiseAnd )
        func( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    Image BitwiseOr( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::BitwiseOr( BitwiseOr, in1, in2 );
    }

    void BitwiseOr( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::BitwiseOr( BitwiseOr, in1, in2, out );
    }

    Image BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseOr( BitwiseOr, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, BitwiseOr )
        func( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    Image BitwiseXor( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::BitwiseXor( BitwiseXor, in1, in2 );
    }

    void BitwiseXor( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::BitwiseXor( BitwiseXor, in1, in2, out );
    }

    Image BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseXor( BitwiseXor, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, BitwiseXor )
        func( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    Image ConvertToGrayScale( const Image & in )
    {
        return Image_Function_Helper::ConvertToGrayScale( ConvertToGrayScale, in );
    }

    void ConvertToGrayScale( const Image & in, Image & out )
    {
        Image_Function_Helper::ConvertToGrayScale( ConvertToGrayScale, in, out );
    }

    Image ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::ConvertToGrayScale( ConvertToGrayScale, in, startXIn, startYIn, width, height );
    }

    void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                             uint32_t width, uint32_t height )
    {
        initialize( in, ConvertToGrayScale )
        func( manager(in), startXIn, startYIn, manager(out), startXOut, startYOut, width, height );
    }

    Image ConvertToRgb( const Image & in )
    {
        return Image_Function_Helper::ConvertToRgb( ConvertToRgb, in );
    }

    void ConvertToRgb( const Image & in, Image & out )
    {
        Image_Function_Helper::ConvertToRgb( ConvertToRgb, in, out );
    }

    Image ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::ConvertToRgb( ConvertToRgb, in, startXIn, startYIn, width, height );
    }

    void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height )
    {
        initialize( in, ConvertToRgb )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    void Copy( const Image & in, Image & out )
    {
        out = Image_Function_Helper::Copy( Copy, in, 0, 0, in.width(), in.height() );
    }

    Image Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Copy( Copy, in, startXIn, startYIn, width, height );
    }

    void Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
               uint32_t width, uint32_t height )
    {
        initialize( in, Copy )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    Image ExtractChannel( const Image & in, uint8_t channelId )
    {
        return Image_Function_Helper::ExtractChannel( ExtractChannel, in, channelId );
    }

    void ExtractChannel( const Image & in, Image & out, uint8_t channelId )
    {
        Image_Function_Helper::ExtractChannel( ExtractChannel, in, out, channelId );
    }

    Image ExtractChannel( const Image & in, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t channelId )
    {
        return Image_Function_Helper::ExtractChannel( ExtractChannel, in, x, y, width, height, channelId );
    }

    void ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
                         uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId )
    {
        initialize( in, ExtractChannel )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height, channelId );
    }

    void Fill( Image & image, uint8_t value )
    {
        image.fill( value );
    }

    void Fill( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value )
    {
        initialize( image, Fill )
        func( image, x, y, width, height, value );
    }

    Image Flip( const Image & in, bool horizontal, bool vertical )
    {
        return Image_Function_Helper::Flip( Flip, in, horizontal, vertical );
    }

    void Flip( const Image & in, Image & out, bool horizontal, bool vertical )
    {
        Image_Function_Helper::Flip( Flip, in, out, horizontal, vertical );
    }

    Image Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                bool horizontal, bool vertical )
    {
        return Image_Function_Helper::Flip( Flip, in, startXIn, startYIn, width, height, horizontal, vertical );
    }

    void Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
               uint32_t width, uint32_t height, bool horizontal, bool vertical )
    {
        initialize( in, Flip )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height, horizontal, vertical );
    }

    Image GammaCorrection( const Image & in, double a, double gamma )
    {
        return Image_Function_Helper::GammaCorrection( GammaCorrection, in, a, gamma );
    }

    void GammaCorrection( const Image & in, Image & out, double a, double gamma )
    {
        Image_Function_Helper::GammaCorrection( GammaCorrection, in, out, a, gamma );
    }

    Image GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double a, double gamma )
    {
        return Image_Function_Helper::GammaCorrection( GammaCorrection, in, startXIn, startYIn, width, height, a, gamma );
    }

    void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                          uint32_t width, uint32_t height, double a, double gamma )
    {
        initialize( in, GammaCorrection )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height, a, gamma );
    }

    uint8_t GetPixel( const Image & image, uint32_t x, uint32_t y )
    {
        initialize( image, GetPixel );
        return func( image, x, y );
    }

    uint8_t GetThreshold( const std::vector < uint32_t > & histogram )
    {
        return Image_Function_Helper::GetThreshold( histogram );
    }

    std::vector < uint32_t > Histogram( const Image & image )
    {
        return Image_Function_Helper::Histogram( Histogram, image );
    }

    void Histogram( const Image & image, std::vector < uint32_t > & histogram )
    {
        Image_Function_Helper::Histogram( Histogram, image, histogram );
    }

    std::vector < uint32_t > Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Histogram( Histogram, image, x, y, width, height );
    }

    void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                    std::vector < uint32_t > & histogram )
    {
        initialize( image, Histogram )
        func( manager(image), x, y, width, height, histogram );
    }

    Image Invert( const Image & in )
    {
        return Image_Function_Helper::Invert( Invert, in );
    }

    void Invert( const Image & in, Image & out )
    {
        Image_Function_Helper::Invert( Invert, in, out );
    }

    Image Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Invert( Invert, in, startXIn, startYIn, width, height );
    }

    void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                 uint32_t width, uint32_t height )
    {
        initialize( in, Invert )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    bool IsEqual( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::IsEqual( IsEqual, in1, in2 );
    }

    bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  uint32_t width, uint32_t height )
    {
        initialize( in1, IsEqual );
        return func( in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    Image LookupTable( const Image & in, const std::vector < uint8_t > & table )
    {
        return Image_Function_Helper::LookupTable( LookupTable, in, table );
    }

    void LookupTable( const Image & in, Image & out, const std::vector < uint8_t > & table )
    {
        Image_Function_Helper::LookupTable( LookupTable, in, out, table );
    }

    Image LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                       const std::vector < uint8_t > & table )
    {
        return Image_Function_Helper::LookupTable( LookupTable, in, startXIn, startYIn, width, height, table );
    }

    void LookupTable ( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height, const std::vector < uint8_t > & table )
    {
        initialize( in, LookupTable )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height, table );
    }

    Image Maximum( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::Maximum( Maximum, in1, in2 );
    }

    void Maximum( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::Maximum( Maximum, in1, in2, out );
    }

    Image Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Maximum( Maximum, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, Maximum )
        func( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    Image Merge( const Image & in1, const Image & in2, const Image & in3 )
    {
        return Image_Function_Helper::Merge( Merge, in1, in2, in3 );
    }

    void Merge( const Image & in1, const Image & in2, const Image & in3, Image & out )
    {
        Image_Function_Helper::Merge( Merge, in1, in2, in3, out );
    }

    Image Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                 const Image & in3, uint32_t startXIn3, uint32_t startYIn3, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Merge( Merge, in1, startXIn1, startYIn1, in2, startXIn2, startYIn2,
                                             in3, startXIn3, startYIn3, width, height  );
    }

    void Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                const Image & in3, uint32_t startXIn3, uint32_t startYIn3, Image & out, uint32_t startXOut, uint32_t startYOut,
                uint32_t width, uint32_t height )
    {
        initialize( in1, Merge )
        func( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3,
                           out, startXOut, startYOut, width, height );
    }

    Image Minimum( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::Minimum( Minimum, in1, in2 );
    }

    void Minimum( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::Minimum( Minimum, in1, in2, out );
    }

    Image Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Minimum( Minimum, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, Minimum )
        func( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    Image Normalize( const Image & in )
    {
        return Image_Function_Helper::Normalize( Normalize, in );
    }

    void Normalize( const Image & in, Image & out )
    {
        Image_Function_Helper::Normalize( Normalize, in, out );
    }

    Image Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Normalize( Normalize, in, startXIn, startYIn, width, height );
    }

    void Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height )
    {
        initialize( in, Normalize )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    std::vector < uint32_t > ProjectionProfile( const Image & image, bool horizontal )
    {
        return Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, horizontal );
    }

    void ProjectionProfile( const Image & image, bool horizontal, std::vector < uint32_t > & projection )
    {
        ProjectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );
    }

    std::vector < uint32_t > ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal )
    {
        return Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, x, y, width, height, horizontal );
    }

    void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                            std::vector < uint32_t > & projection )
    {
        initialize( image, ProjectionProfile )
        func( image, x, y, width, height, horizontal, projection );
    }

    Image Resize( const Image & in, uint32_t widthOut, uint32_t heightOut )
    {
        return Image_Function_Helper::Resize( Resize, in, widthOut, heightOut );
    }

    void Resize( const Image & in, Image & out )
    {
        Image_Function_Helper::Resize( Resize, in, out );
    }

    Image Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                  uint32_t widthOut, uint32_t heightOut )
    {
        return Image_Function_Helper::Resize( Resize, in, startXIn, startYIn, widthIn, heightIn, widthOut, heightOut );
    }

    void Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut )
    {
        initialize( in, Resize )
        func( in, startXIn, startYIn, widthIn, heightIn, out, startXOut, startYOut, widthOut, heightOut );
    }

    Image RgbToBgr( const Image & in )
    {
        return Image_Function_Helper::RgbToBgr( RgbToBgr, in );
    }

    void RgbToBgr( const Image & in, Image & out )
    {
        Image_Function_Helper::RgbToBgr( RgbToBgr, in, out );
    }

    Image RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::RgbToBgr( RgbToBgr, in, startXIn, startYIn, width, height );
    }

    void RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                   uint32_t width, uint32_t height )
    {
        initialize( in, RgbToBgr )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    void SetPixel( Image & image, uint32_t x, uint32_t y, uint8_t value )
    {
        initialize( image, SetPixel )
        func( image, x, y, value );
    }

    void SetPixel( Image & image, const std::vector < uint32_t > & X, const std::vector < uint32_t > & Y, uint8_t value )
    {
        initialize( image, SetPixel2 )
        func( image, X, Y, value );
    }

    void Split( const Image & in, Image & out1, Image & out2, Image & out3 )
    {
        Image_Function_Helper::Split( Split, in, out1, out2, out3 );
    }

    void Split( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out1, uint32_t startXOut1, uint32_t startYOut1,
                Image & out2, uint32_t startXOut2, uint32_t startYOut2, Image & out3, uint32_t startXOut3, uint32_t startYOut3,
                uint32_t width, uint32_t height )
    {
        initialize( in, Split )
        func( in, startXIn, startYIn, out1, startXOut1, startYOut1, out2, startXOut2, startYOut2,
                          out3, startXOut3, startYOut3, width, height );
    }

    Image Subtract( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::Subtract( Subtract, in1, in2 );
    }

    void Subtract( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::Subtract( Subtract, in1, in2, out );
    }

    Image Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Subtract( Subtract, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, Subtract )
        func( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    uint32_t Sum( const Image & image )
    {
        return Sum( image, 0, 0, image.width(), image.height() );
    }

    uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        initialize( image, Sum );
        return func( image, x, y, width, height );
    }

    Image Threshold( const Image & in, uint8_t threshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, threshold );
    }

    void Threshold( const Image & in, Image & out, uint8_t threshold )
    {
        Image_Function_Helper::Threshold( Threshold, in, out, threshold );
    }

    Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t threshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, startXIn, startYIn, width, height, threshold );
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t threshold )
    {
        initialize( in, Threshold )
        func( manager(in), startXIn, startYIn, manager(out), startXOut, startYOut, width, height, threshold );
    }

    Image Threshold( const Image & in, uint8_t minThreshold, uint8_t maxThreshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, minThreshold, maxThreshold );
    }

    void Threshold( const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function_Helper::Threshold( Threshold, in, out, minThreshold, maxThreshold );
    }

    Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t minThreshold,
                     uint8_t maxThreshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, startXIn, startYIn, width, height, minThreshold, maxThreshold );
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold )
    {
        initialize( in, Threshold2 )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold );
    }

    Image Transpose( const Image & in )
    {
        return Image_Function_Helper::Transpose( Transpose, in );
    }

    void Transpose( const Image & in, Image & out )
    {
        Image_Function_Helper::Transpose( Transpose, in, out );
    }

    Image Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Transpose( Transpose, in, startXIn, startYIn, width, height );
    }

    void Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height )
    {
        initialize( in, Transpose )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }
}
