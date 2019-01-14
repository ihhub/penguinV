
#include "test_helper.h"

namespace
{
    uint32_t randomSize()
    {
        return Test_Helper::randomValue<uint32_t>( 1, 2048 );
    }

    PenguinV_Image::Image generateImage( uint32_t width, uint32_t height, uint8_t colorCount, uint8_t value, const PenguinV_Image::Image & reference )
    {
        PenguinV_Image::Image image = reference.generate( width, height, colorCount );

        image.fill( value );

        return image;
    }

    void fillRandomData( PenguinV_Image::Image & image )
    {
        uint8_t * outY = image.data();
        const uint8_t * outYEnd = outY + image.height() * image.rowSize();

        for( ; outY != outYEnd; outY += image.rowSize() ) {
            uint8_t * outX = outY;
            const uint8_t * outXEnd = outX + image.width() * image.colorCount();

            for( ; outX != outXEnd; ++outX )
                (*outX) = Test_Helper::randomValue<uint8_t>( 256 );
        }
    }

    uint32_t testRunCount = 1001;  // some magic number for loop. Higher value = higher chance to verify all possible situations
}

namespace Test_Helper {

    PenguinV_Image::Image uniformImage( const PenguinV_Image::Image & reference = PenguinV_Image::Image() ) 
    {
      return uniformImage( randomValue<uint8_t>( 256 ), reference );
    }

    PenguinV_Image::Image uniformImage( uint8_t value, const PenguinV_Image::Image & reference = PenguinV_Image::Image() )
    {
        return generateImage( randomSize(), randomSize(), PenguinV_Image::GRAY_SCALE, value, reference );
    }

    PenguinV_Image::Image uniformRGBImage( const PenguinV_Image::Image & reference = PenguinV_Image::Image() )
    {

    }

    PenguinV_Image::Image uniformRGBImage( uint8_t value, const PenguinV_Image::Image & reference = PenguinV_Image::Image() )
    {

    }

    PenguinV_Image::Image blackImage( const PenguinV_Image::Image & reference = PenguinV_Image::Image() )
    {

    }

    PenguinV_Image::Image whiteImage( const PenguinV_Image::Image & reference = PenguinV_Image::Image() )
    {

    }

    PenguinV_Image::Image randomImage()
    {

    }

    PenguinV_Image::Image randomImage( const std::vector <uint8_t> & value )
    {

    }

    std::vector < PenguinV_Image::Image > uniformImages( uint32_t images, const PenguinV_Image::Image & reference = PenguinV_Image::Image() )
    {

    }

    std::vector < PenguinV_Image::Image > uniformImages( const std::vector < uint8_t > & intensityValue, const PenguinV_Image::Image & reference = PenguinV_Image::Image() )
    {

    }

    uint32_t runCount()
    {

    }

    data randomValue( int maximum )
    {

    }

    data randomValue( data minimum, int maximum ) 
    {

    }

}
