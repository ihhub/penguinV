#include "penguinv.h"

#include <map>
#include <mutex>

namespace
{
    std::map< uint8_t, penguinV::FunctionTable > functionTableMap;

    template <typename _Function>
    void setFunction( _Function F1, _Function F2, bool forceSetup )
    {
        if ( (F1 == nullptr) || (forceSetup && (F2 != nullptr)) )
        {
            F1 = F2;
        }
    }

#define SET_FUNCTION( functionName ) \
    setFunction( oldTable.functionName, newTable.functionName, forceSetup );

    void setupTable( penguinV::FunctionTable & oldTable, const penguinV::FunctionTable & newTable, bool forceSetup )
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

    const penguinV::FunctionTable & functionTable( const PenguinV_Image::Image & image )
    {
        std::map< uint8_t, penguinV::FunctionTable >::const_iterator table = functionTableMap.find( image.type() );
        if (table == functionTableMap.end())
            throw imageException( "Function table is not initialised" );

        return table->second;
    }

#define run( image, func_ )                                                                                        \
    auto func = functionTable( image ).##func_;                                                                    \
    if ( func == nullptr ) {                                                                                       \
        const std::string error( std::string("Function ") + std::string(#func_) + std::string("is not defined") ); \
        throw imageException(error.data());                                                                        \
    }                                                                                                              \
    func

}

namespace penguinV
{
    void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                             Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        run( in1, AbsoluteDifference )( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    void Accumulate( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & result )
    {
        run( image, Accumulate )( image, x, y, width, height, result );
    }

    void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        run( in1, BitwiseAnd )( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        run( in1, BitwiseOr )( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        run( in1, BitwiseXor )( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                             uint32_t width, uint32_t height )
    {
        run( in, ConvertToGrayScale )( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height )
    {
        run( in, ConvertToRgb )( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    void Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
               uint32_t width, uint32_t height )
    {
        run( in, Copy )( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    void ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
                         uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId )
    {
        run( in, ExtractChannel )( in, startXIn, startYIn, out, startXOut, startYOut, width, height, channelId );
    }

    void Fill( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value )
    {
        run( image, Fill )( image, x, y, width, height, value );
    }

    void Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
               uint32_t width, uint32_t height, bool horizontal, bool vertical )
    {
        run( in, Flip )( in, startXIn, startYIn, out, startXOut, startYOut, width, height, horizontal, vertical );
    }

    void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                          uint32_t width, uint32_t height, double a, double gamma )
    {
        run( in, GammaCorrection )( in, startXIn, startYIn, out, startXOut, startYOut, width, height, a, gamma );
    }

    uint8_t GetPixel( const Image & image, uint32_t x, uint32_t y )
    {
        run( image, GetPixel );
        return func( image, x, y );
    }

    uint8_t GetThreshold( const std::vector < uint32_t > & histogram )
    {
        return Image_Function_Helper::GetThreshold( histogram );
    }

    void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                    std::vector < uint32_t > & histogram )
    {
        run( image, Histogram )( image, x, y, width, height, histogram );
    }

    void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                 uint32_t width, uint32_t height )
    {
        run( in, Invert )( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  uint32_t width, uint32_t height )
    {
        run( in1, IsEqual );
        return func( in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void LookupTable ( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height, const std::vector < uint8_t > & table )
    {
        run( in, LookupTable )( in, startXIn, startYIn, out, startXOut, startYOut, width, height, table );
    }

    void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        run( in1, Maximum )( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    void Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                const Image & in3, uint32_t startXIn3, uint32_t startYIn3, Image & out, uint32_t startXOut, uint32_t startYOut,
                uint32_t width, uint32_t height )
    {
        run( in1, Merge )( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3,
                                    out, startXOut, startYOut, width, height );
    }

    void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        run( in1, Minimum )( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    void Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height )
    {
        run( in, Normalize )( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                            std::vector < uint32_t > & projection )
    {
        run( image, ProjectionProfile )( image, x, y, width, height, horizontal, projection );
    }

    void Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut )
    {
        run( in, Resize )( in, startXIn, startYIn, widthIn, heightIn, out, startXOut, startYOut, widthOut, heightOut );
    }

    void RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                   uint32_t width, uint32_t height )
    {
        run( in, RgbToBgr )( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    void SetPixel( Image & image, uint32_t x, uint32_t y, uint8_t value )
    {
        run( image, SetPixel )( image, x, y, value );
    }

    void SetPixel( Image & image, const std::vector < uint32_t > & X, const std::vector < uint32_t > & Y, uint8_t value )
    {
        run( image, SetPixel2 )( image, X, Y, value );
    }

    void Split( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out1, uint32_t startXOut1, uint32_t startYOut1,
                Image & out2, uint32_t startXOut2, uint32_t startYOut2, Image & out3, uint32_t startXOut3, uint32_t startYOut3,
                uint32_t width, uint32_t height )
    {
        run( in, Split )( in, startXIn, startYIn, out1, startXOut1, startYOut1, out2, startXOut2, startYOut2,
                                   out3, startXOut3, startYOut3, width, height );
    }

    void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        run( in1, Subtract )( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        run( image, Sum );
        return func( image, x, y, width, height );
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t threshold )
    {
        run( in, Threshold )( in, startXIn, startYIn, out, startXOut, startYOut, width, height, threshold );
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold )
    {
        run( in, Threshold2 )( in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold );
    }

    void Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height )
    {
        run( in, Transpose )( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    void registerFunctionTable( const Image & image, const FunctionTable & table, bool forceSetup )
    {
        static std::mutex mapGuard;

        mapGuard.lock();
        std::map< uint8_t, penguinV::FunctionTable >::iterator oldTable = functionTableMap.find( image.type() );
        if (oldTable != functionTableMap.end())
            setupTable( oldTable->second, table, forceSetup );
        else
            functionTableMap[image.type()] = table;
        mapGuard.unlock();
    }
}
