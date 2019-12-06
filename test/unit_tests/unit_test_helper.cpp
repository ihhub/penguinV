#include "unit_test_helper.h"
#include "../../src/image_exception.h"
#include "../../src/image_function.h"
#include "../../src/parameter_validation.h"

namespace
{
    uint32_t randomSize()
    {
        return Unit_Test::randomValue<uint32_t>( 1, 2048 );
    }

    template <typename _Type>
    bool imageVerification( const penguinV::ImageTemplate<_Type> & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, _Type value )
    {
        if( image.empty() || width == 0 || height == 0 || x + width > image.width() || y + height > image.height() )
            throw imageException( "Bad input parameters in image function" );

        width = width * image.colorCount();
        Image_Function::OptimiseRoi( width, height, image );

        const uint32_t rowSize = image.rowSize();
        const _Type * outputY  = image.data() + y * rowSize + x * image.colorCount();
        const _Type * endY     = outputY + rowSize * height;

        for ( ; outputY != endY; outputY += rowSize ) {
            const _Type * outputX = outputY;
            const _Type * endX    = outputX + width;

            for( ; outputX != endX; ++outputX ) {
                if( (*outputX) != value )
                    return false;
            }
        }

        return true;
    }

    template <typename _Type>
    bool imageVerification( const penguinV::ImageTemplate<_Type> & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                            const std::vector < _Type > & value, bool isAnyValue )
    {
        if( image.empty() || width == 0 || height == 0 || x + width > image.width() || y + height > image.height() )
            throw imageException( "Bad input parameters in image function" );

        const uint32_t rowSize = image.rowSize();
        const _Type * outputY  = image.data() + y * rowSize + x * image.colorCount();
        const _Type * endY     = outputY + rowSize * height;

        width = width * image.colorCount();

        for ( ; outputY != endY; outputY += rowSize ) {
            const _Type * outputX = outputY;
            const _Type * endX    = outputX + width;

            if( isAnyValue ) {
                for( ; outputX != endX; ++outputX ) {
                    bool equal = false;

                    for( typename std::vector < _Type >::const_iterator v = value.begin(); v != value.end(); ++v ) {
                        if( (*outputX) == (*v) ) {
                            equal = true;
                            break;
                        }
                    }

                    if( !equal )
                        return false;
                }
            }
            else {
                size_t id = 0;
                for( ; outputX != endX; ++outputX ) {
                    if( (*outputX) != value[id++] )
                        return false;

                    if( id == value.size() )
                        id = 0;
                }
            }
        }

        return true;
    }
}

namespace Unit_Test
{
    penguinV::Image blackImage( const penguinV::Image & reference )
    {
        return uniformImage( 0u, 0, 0, reference );
    }

    penguinV::Image whiteImage( const penguinV::Image & reference )
    {
        return uniformImage( 255u, 0, 0, reference );
    }

    penguinV::Image randomImage( const std::vector <uint8_t> & value )
    {
        if( value.empty() )
            return Test_Helper::randomImage();

        penguinV::Image image( randomSize(), randomSize() );

        uint32_t height = image.height();
        uint32_t width = image.width();

        const size_t valueSize = value.size();

        if ( valueSize <= width && (width % static_cast<uint32_t>(valueSize)) == 0 )
            Image_Function::OptimiseRoi( width, height, image );

        const uint32_t rowSize  = image.rowSize();
        uint8_t * outY          = image.data();
        const uint8_t * outYEnd = outY + height * rowSize;

        size_t id = 0;

        for ( ; outY != outYEnd; outY += rowSize ) {
            uint8_t * outX = outY;
            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX ) {
                (*outX) = value[id++];
                if ( id == valueSize )
                    id = 0u;
            }
        }

        return image;
    }

    template <typename data>
    std::vector < data > generateArray( uint32_t size, uint32_t maximumValue )
    {
        std::vector < data > fillArray( size );

        std::for_each( fillArray.begin(), fillArray.end(), [&]( data & value ) { value = randomValue<data>( maximumValue ); } );

        return fillArray;
    }

    uint8_t intensityValue()
    {
        return randomValue<uint8_t>( 255 );
    }

    std::vector < uint8_t > intensityArray( uint32_t size )
    {
        return generateArray<uint8_t>( size, 256u );
    }

    bool equalSize( const penguinV::Image & image, uint32_t width, uint32_t height )
    {
        return image.width() == width && image.height() == height && !image.empty();
    }

    bool verifyImage( const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value )
    {
        return imageVerification<uint8_t>( image, x, y, width, height, value );
    }

    bool verifyImage( const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                      const std::vector < uint8_t > & value, bool isAnyValue )
    {
        return imageVerification<uint8_t>( image, x, y, width, height, value, isAnyValue );
    }

    bool verifyImage( const penguinV::Image & image, uint8_t value )
    {
        return verifyImage( image, 0, 0, image.width(), image.height(), value );
    }

    bool verifyImage( const penguinV::Image & image, const std::vector < uint8_t > & value, bool isAnyValue )
    {
        return verifyImage( image, 0, 0, image.width(), image.height(), value, isAnyValue );
    }

    bool verifyImage( const penguinV::Image16Bit & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint16_t value )
    {
        return imageVerification<uint16_t>( image, x, y, width, height, value );
    }

    bool verifyImage( const penguinV::Image16Bit & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                      const std::vector < uint16_t > & value, bool isAnyValue )
    {
        return imageVerification<uint16_t>( image, x, y, width, height, value, isAnyValue );
    }

    bool verifyImage( const penguinV::Image16Bit & image, uint16_t value )
    {
        return verifyImage( image, 0, 0, image.width(), image.height(), value );
    }

    bool verifyImage( const penguinV::Image16Bit & image, const std::vector < uint16_t > & value, bool isAnyValue )
    {
        return verifyImage( image, 0, 0, image.width(), image.height(), value, isAnyValue );
    }

    void fillImage( penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value )
    {
        Image_Function::Fill( image, x, y, width, height, value );
    }

    void fillImage( penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                    const std::vector < uint8_t > & value )
    {
        Image_Function::ParameterValidation( image, x, y, width, height );

        const uint32_t rowSize = image.rowSize();
        uint8_t * outputY      = image.data() + y * rowSize + x * image.colorCount();
        const uint8_t * endY   = outputY + rowSize * height;

        width = width * image.colorCount();
        const size_t valueSize = value.size();

        size_t id = 0;

        for ( ; outputY != endY; outputY += rowSize ) {
            uint8_t * outputX    = outputY;
            const uint8_t * endX = outputX + width;

            for( ; outputX != endX; ++outputX ) {
                (*outputX) = value[id++];
                if ( id == valueSize )
                    id = 0;
            }
        }
    }

    void generateRoi( const std::vector < penguinV::Image > & image, std::vector < uint32_t > & x, std::vector < uint32_t > & y,
                      uint32_t & width, uint32_t & height )
    {
        std::vector < std::pair < uint32_t, uint32_t> > imageSize( image.size() );

        for( size_t i = 0; i < image.size(); ++i )
            imageSize[i] = std::pair < uint32_t, uint32_t >( image[i].width(), image[i].height() ) ;

        generateRoi( imageSize, x, y, width, height );
    }

    void generateRoi( const std::vector < std::pair< uint32_t, uint32_t > > & imageSize, std::vector < uint32_t > & x,
                      std::vector < uint32_t > & y, uint32_t & width, uint32_t & height )
    {
        uint32_t maximumWidth  = 0;
        uint32_t maximumHeight = 0;

        for( std::vector < std::pair< uint32_t, uint32_t > >::const_iterator im = imageSize.begin();
             im != imageSize.end(); ++im ) {
            if( maximumWidth == 0 )
                maximumWidth = im->first;
            else if( maximumWidth > im->first )
                maximumWidth = im->first;

            if( maximumHeight == 0 )
                maximumHeight = im->second;
            else if( maximumHeight > im->second )
                maximumHeight = im->second;
        }

        width  = randomValue<uint32_t>( 1, maximumWidth  + 1 );
        height = randomValue<uint32_t>( 1, maximumHeight + 1 );

        x.resize( imageSize.size() );
        y.resize( imageSize.size() );

        for( size_t i = 0; i < imageSize.size(); ++i ) {
            x[i] = randomValue<uint32_t>( imageSize[i].first  - width );
            y[i] = randomValue<uint32_t>( imageSize[i].second - height );
        }
    }

    void generateOffset( const penguinV::Image & image, uint32_t & x, uint32_t & y, uint32_t width, uint32_t height )
    {
        x = randomValue<uint32_t>( image.width()  - width );
        y = randomValue<uint32_t>( image.height() - height );
    }

    uint32_t rowSize( uint32_t width, uint8_t colorCount, uint8_t alignment )
    {
        uint32_t size = width * colorCount;
        if( size % alignment != 0 )
            size = ((size / alignment) + 1) * alignment;

        return size;
    }
}
