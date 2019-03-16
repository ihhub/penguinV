#include "penguinv.h"
#include "../image_function_helper.h"

namespace
{
    struct ReferenceOwner
    {
        explicit ReferenceOwner( PenguinV_Image::Image & data_ )
            : data( data_ )
        {
        }
        PenguinV_Image::Image & data;
    };

    struct ConstReferenceOwner
    {
        explicit ConstReferenceOwner( const PenguinV_Image::Image & data_ )
            : data( data_ )
        {
        }
        const PenguinV_Image::Image & data;
    };

    class ImageManager
    {
    public:
        explicit ImageManager( uint8_t requiredType )
            : _registrator( ImageTypeManager::instance() )
            , _type( requiredType )
        {
        }

        ~ImageManager()
        {
            for ( std::vector<ConstReferenceOwner *>::iterator data = _input.begin(); data != _input.end(); ++data )
                delete *data;

            for ( size_t i = 0u; i < _output.size(); ++i ) {
                _restore( _outputClone[i], _output[i]->data );

                delete _output[i];
            }
        }

        const PenguinV_Image::Image & operator ()( const PenguinV_Image::Image & image )
        {
            if ( image.type() != _type ) {
                _inputClone.push_back( _clone( image ) );
                return _inputClone.back();
            }
            else {
                return image;
            }
        }

        PenguinV_Image::Image & operator ()( PenguinV_Image::Image & image )
        {
            if ( image.type() != _type ) {
                _output.push_back( new ReferenceOwner(image) );
                _outputClone.push_back( _clone( image ) );
                return _outputClone.back();
            }
            else {
                return image;
            }
        }
    private:
        ImageTypeManager & _registrator;
        std::vector< ConstReferenceOwner * > _input;
        std::vector< ReferenceOwner * > _output;
        std::vector< PenguinV_Image::Image > _inputClone;
        std::vector< PenguinV_Image::Image > _outputClone;
        uint8_t _type;

        PenguinV_Image::Image _clone( const PenguinV_Image::Image & in )
        {
            PenguinV_Image::Image temp = _registrator.image( _type ).generate( in.width(), in.height(), in.colorCount() );
            _registrator.convert( in.type(), _type )( in, temp );
            return temp;
        }

        void _restore( const PenguinV_Image::Image & in, PenguinV_Image::Image & out )
        {
            _registrator.convert( in.type(), out.type() )( in, out );
        }
    };

    template <typename _T>
    void verifyFunction(_T func, const char * functionName)
    {
        if (func == nullptr) {
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
    ImageManager manager( imageType );
}

namespace penguinV
{
    void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                             Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, AbsoluteDifference )
        func( manager(in1), startX1, startY1, manager(in2), startX2, startY2, manager(out), startXOut, startYOut, width, height );
    }

    void Accumulate( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & result )
    {
        initialize( image, Accumulate )
        func( image, x, y, width, height, result );
    }

    void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, BitwiseAnd )
        func( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, BitwiseOr )
        func( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, BitwiseXor )
        func( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                             uint32_t width, uint32_t height )
    {
        initialize( in, ConvertToGrayScale )
        func( manager(in), startXIn, startYIn, manager(out), startXOut, startYOut, width, height );
    }

    void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height )
    {
        initialize( in, ConvertToRgb )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    void Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
               uint32_t width, uint32_t height )
    {
        initialize( in, Copy )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    void ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
                         uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId )
    {
        initialize( in, ExtractChannel )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height, channelId );
    }

    void Fill( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value )
    {
        initialize( image, Fill )
        func( image, x, y, width, height, value );
    }

    void Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
               uint32_t width, uint32_t height, bool horizontal, bool vertical )
    {
        initialize( in, Flip )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height, horizontal, vertical );
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

    void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                    std::vector < uint32_t > & histogram )
    {
        initialize( image, Histogram )
        func( manager(image), x, y, width, height, histogram );
    }

    void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                 uint32_t width, uint32_t height )
    {
        initialize( in, Invert )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  uint32_t width, uint32_t height )
    {
        initialize( in1, IsEqual );
        return func( in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void LookupTable ( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height, const std::vector < uint8_t > & table )
    {
        initialize( in, LookupTable )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height, table );
    }

    void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, Maximum )
        func( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    void Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                const Image & in3, uint32_t startXIn3, uint32_t startYIn3, Image & out, uint32_t startXOut, uint32_t startYOut,
                uint32_t width, uint32_t height )
    {
        initialize( in1, Merge )
        func( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3,
                           out, startXOut, startYOut, width, height );
    }

    void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, Minimum )
        func( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    void Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height )
    {
        initialize( in, Normalize )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }

    void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                            std::vector < uint32_t > & projection )
    {
        initialize( image, ProjectionProfile )
        func( image, x, y, width, height, horizontal, projection );
    }

    void Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut )
    {
        initialize( in, Resize )
        func( in, startXIn, startYIn, widthIn, heightIn, out, startXOut, startYOut, widthOut, heightOut );
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

    void Split( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out1, uint32_t startXOut1, uint32_t startYOut1,
                Image & out2, uint32_t startXOut2, uint32_t startYOut2, Image & out3, uint32_t startXOut3, uint32_t startYOut3,
                uint32_t width, uint32_t height )
    {
        initialize( in, Split )
        func( in, startXIn, startYIn, out1, startXOut1, startYOut1, out2, startXOut2, startYOut2,
                          out3, startXOut3, startYOut3, width, height );
    }

    void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        initialize( in1, Subtract )
        func( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
    }

    uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        initialize( image, Sum );
        return func( image, x, y, width, height );
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t threshold )
    {
        initialize( in, Threshold )
        func( manager(in), startXIn, startYIn, manager(out), startXOut, startYOut, width, height, threshold );
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold )
    {
        initialize( in, Threshold2 )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold );
    }

    void Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height )
    {
        initialize( in, Transpose )
        func( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
    }
}
