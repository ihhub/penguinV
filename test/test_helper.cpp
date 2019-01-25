#include "test_helper.h"

namespace
{
    uint32_t randomSize()
    {
        return Test_Helper::randomValue<uint32_t>( 1, 2048 );
    }

    PenguinV_Image::Image generateImage( uint32_t width, uint32_t height, uint8_t colorCount, uint8_t value,
                                         const PenguinV_Image::Image & reference = PenguinV_Image::Image() )
    {
        PenguinV_Image::Image image = reference.generate( width, height, colorCount );

        image.fill( value );

        return image;
    }

    uint32_t testRunCount = 1001;  // some magic number for loop. Higher value = higher chance to verify all possible situations
}

namespace Test_Helper
{
    PenguinV_Image::Image uniformImage( uint32_t width, uint32_t height, const PenguinV_Image::Image & reference )
    {
        return uniformImage( randomValue<uint8_t>( 256 ), width, height, reference );
    }

    PenguinV_Image::Image uniformImage( uint8_t value, uint32_t width, uint32_t height, const PenguinV_Image::Image & reference )
    {
        return generateImage( (width > 0u) ? width : randomSize(), (height > 0u) ? height : randomSize(), PenguinV_Image::GRAY_SCALE, value, reference );
    }

    PenguinV_Image::Image uniformRGBImage( const PenguinV_Image::Image & reference )
    {
        return uniformRGBImage( randomValue<uint8_t>( 256 ), reference );
    }

    PenguinV_Image::Image uniformRGBImage( uint8_t value, const PenguinV_Image::Image & reference )
    {
        return generateImage( randomSize(), randomSize(), PenguinV_Image::RGB, value, reference );
    }

    PenguinV_Image::Image uniformRGBImage( uint32_t width, uint32_t height )
    {
        return uniformRGBImage( width, height, randomValue<uint8_t>( 256 ) );
    }

    PenguinV_Image::Image uniformRGBImage( uint32_t width, uint32_t height, uint8_t value )
    {
        return generateImage( width, height, PenguinV_Image::RGB, value);
    }

    std::vector< PenguinV_Image::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height )
    {
        std::vector < PenguinV_Image::Image > image( count );

        for( std::vector< PenguinV_Image::Image >::iterator im = image.begin(); im != image.end(); ++im )
            *im = uniformImage( width, height );

        return image;
    }

    std::vector< PenguinV_Image::Image > uniformRGBImages( uint32_t count, uint32_t width, uint32_t height )
    {
        std::vector < PenguinV_Image::Image > image( count );

        for( std::vector< PenguinV_Image::Image >::iterator im = image.begin(); im != image.end(); ++im )
            *im = uniformRGBImage( width, height );

        return image;
    }
    
    std::vector < PenguinV_Image::Image > uniformImages( uint32_t images, const PenguinV_Image::Image & reference )
    {
        if( images == 0 )
            throw imageException( "Invalid parameter: number of images is 0" );

        std::vector<uint8_t> intesity( images );
        for( size_t i = 0u; i < intesity.size(); ++i )
            intesity[i] = randomValue<uint8_t>( 256 );

        return uniformImages( intesity, reference );
    }

    std::vector < PenguinV_Image::Image > uniformImages( const std::vector<uint8_t> & intensityValue, const PenguinV_Image::Image & reference )
    {
        if( intensityValue.size() == 0 )
            throw imageException( "Invalid parameter" );

        std::vector < PenguinV_Image::Image > image;

        image.push_back( uniformImage( intensityValue[0], 0, 0, reference ) );

        image.resize( intensityValue.size() );

        for( size_t i = 1u; i < image.size(); ++i ) {
            image[i] = reference.generate( image[0].width(), image[0].height() );
            image[i].fill( intensityValue[i] );
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

