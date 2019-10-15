#include "test_helper.h"
#include "../src/parameter_validation.h"

namespace
{
    uint32_t randomSize()
    {
        return Test_Helper::randomValue<uint32_t>( 1, 2048 );
    }

    template <typename _Type>
    PenguinV::ImageTemplate<_Type> generateImage( uint32_t width, uint32_t height, uint8_t colorCount, _Type value,
                                                        const PenguinV::ImageTemplate<_Type> & reference = PenguinV::ImageTemplate<_Type>() )
    {
        PenguinV::ImageTemplate<_Type> image = reference.generate( width, height, colorCount );

        image.fill( value );

        return image;
    }

    void fillRandomData( PenguinV::Image & image )
    {
        uint32_t height = image.height();
        uint32_t width = image.width() * image.colorCount();
        Image_Function::OptimiseRoi( width, height, image );

        const uint32_t rowSize  = image.rowSize();
        uint8_t * outY          = image.data();
        const uint8_t * outYEnd = outY + height * rowSize;

        for ( ; outY != outYEnd; outY += rowSize ) {
            uint8_t * outX = outY;
            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX )
                (*outX) = Test_Helper::randomValue<uint8_t>( 256 );
        }
    }

    uint32_t testRunCount = 1001;  // some magic number for loop. Higher value = higher chance to verify all possible situations
}

namespace Test_Helper
{
    PenguinV::Image uniformImage( uint32_t width, uint32_t height, const PenguinV::Image & reference )
    {
        return uniformImage( randomValue<uint8_t>( 256 ), width, height, reference );
    }

    PenguinV::Image uniformImage( uint8_t value, uint32_t width, uint32_t height, const PenguinV::Image & reference )
    {
        return generateImage<uint8_t>( (width > 0u) ? width : randomSize(), (height > 0u) ? height : randomSize(), PenguinV::GRAY_SCALE, value, reference );
    }

    PenguinV::Image16Bit uniformImage16Bit( uint16_t value, uint32_t width, uint32_t height, const PenguinV::Image16Bit & reference )
    {
        return generateImage<uint16_t>( (width > 0u) ? width : randomSize(), (height > 0u) ? height : randomSize(), PenguinV::GRAY_SCALE, value, reference );
    }

    PenguinV::Image uniformRGBImage( const PenguinV::Image & reference )
    {
        return uniformRGBImage( randomValue<uint8_t>( 256 ), reference );
    }

    PenguinV::Image uniformRGBImage( uint8_t value, const PenguinV::Image & reference )
    {
        return generateImage<uint8_t>( randomSize(), randomSize(), PenguinV::RGB, value, reference );
    }

    PenguinV::Image uniformRGBImage( uint32_t width, uint32_t height )
    {
        return uniformRGBImage( width, height, randomValue<uint8_t>( 256 ) );
    }

    PenguinV::Image uniformRGBImage( uint32_t width, uint32_t height, uint8_t value )
    {
        return generateImage<uint8_t>( width, height, PenguinV::RGB, value);
    }

    PenguinV::Image uniformRGBAImage( uint32_t width, uint32_t height )
    {
        return uniformRGBAImage( width, height, randomValue<uint8_t>( 256 ) );
    }

    PenguinV::Image uniformRGBAImage( uint32_t width, uint32_t height, uint8_t value )
    {
        return generateImage<uint8_t>( width, height, PenguinV::RGBA, value);
    }

    PenguinV::Image uniformRGBAImage( const PenguinV::Image & reference )
    {
        return uniformRGBAImage( randomValue<uint8_t>( 256 ), reference );
    }

    PenguinV::Image uniformRGBAImage( uint8_t value, const PenguinV::Image & reference )
    {
        return generateImage<uint8_t>( randomSize(), randomSize(), PenguinV::RGBA, value, reference );
    }

    std::vector< PenguinV::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height )
    {
        std::vector < PenguinV::Image > image( count );

        for( std::vector< PenguinV::Image >::iterator im = image.begin(); im != image.end(); ++im )
            *im = uniformImage( width, height );

        return image;
    }

    std::vector< PenguinV::Image > uniformRGBImages( uint32_t count, uint32_t width, uint32_t height )
    {
        std::vector < PenguinV::Image > image( count );

        for( std::vector< PenguinV::Image >::iterator im = image.begin(); im != image.end(); ++im )
            *im = uniformRGBImage( width, height );

        return image;
    }
    
    std::vector < PenguinV::Image > uniformImages( uint32_t images, const PenguinV::Image & reference )
    {
        if( images == 0 )
            throw imageException( "Invalid parameter: number of images is 0" );

        std::vector<uint8_t> intesity( images );
        for( size_t i = 0u; i < intesity.size(); ++i )
            intesity[i] = randomValue<uint8_t>( 256 );

        return uniformImages( intesity, reference );
    }

    std::vector < PenguinV::Image > uniformImages( const std::vector<uint8_t> & intensityValue, const PenguinV::Image & reference )
    {
        if( intensityValue.size() == 0 )
            throw imageException( "Invalid parameter" );

        std::vector < PenguinV::Image > image;

        image.push_back( uniformImage( intensityValue[0], 0, 0, reference ) );

        image.resize( intensityValue.size() );

        for( size_t i = 1u; i < image.size(); ++i ) {
            image[i] = reference.generate( image[0].width(), image[0].height() );
            image[i].fill( intensityValue[i] );
        }

        return image;
    }

    PenguinV::Image randomImage( uint32_t width, uint32_t height )
    {
        PenguinV::Image image( (width == 0) ? randomSize() : width, (height == 0) ? randomSize() : height );

        fillRandomData( image );

        return image;
    }

    PenguinV::Image randomRGBImage( const PenguinV::Image & reference )
    {
        PenguinV::Image image = reference.generate(randomSize(), randomSize(), PenguinV::RGB);

        fillRandomData( image );

        return image;
    }

    PenguinV::Image randomImage( const std::vector <uint8_t> & value )
    {
        if( value.empty() )
            return randomImage();

        PenguinV::Image image( randomSize(), randomSize() );

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

    uint32_t runCount()
    {
        return testRunCount;
    }

    void setRunCount( int argc, char* argv[], uint32_t count )
    {
        testRunCount = count;
        if ( argc >= 2 ) {
            const int testCount = std::atoi( argv[1] );
            if ( testCount > 0 )
                testRunCount = static_cast<uint32_t>( testCount );
        }
    }
}

