#pragma once

#include <cstdlib>
#include "../src/image_buffer.h"

namespace Test_Helper
{
    // Functions to generate images
    PenguinV::Image uniformImage( uint32_t width = 0, uint32_t height = 0,  const PenguinV::Image & reference = PenguinV::Image() );
    PenguinV::Image uniformImage( uint8_t value, uint32_t width = 0, uint32_t height = 0, const PenguinV::Image & reference = PenguinV::Image() );
    PenguinV::Image16Bit uniformImage16Bit( uint16_t value, uint32_t width = 0, uint32_t height = 0,
                                                  const PenguinV::Image16Bit & reference = PenguinV::Image16Bit() );
    PenguinV::Image uniformRGBImage( uint32_t width, uint32_t height );
    PenguinV::Image uniformRGBImage( uint32_t width, uint32_t height, uint8_t value );
    PenguinV::Image uniformRGBImage( const PenguinV::Image & reference = PenguinV::Image() );
    PenguinV::Image uniformRGBImage( uint8_t value, const PenguinV::Image & reference = PenguinV::Image() );
    PenguinV::Image uniformRGBAImage( uint32_t width, uint32_t height );
    PenguinV::Image uniformRGBAImage( uint32_t width, uint32_t height, uint8_t value );
    PenguinV::Image uniformRGBAImage( const PenguinV::Image & reference = PenguinV::Image() );
    PenguinV::Image uniformRGBAImage( uint8_t value, const PenguinV::Image & reference = PenguinV::Image() );
    std::vector < PenguinV::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height );
    std::vector < PenguinV::Image > uniformRGBImages( uint32_t count, uint32_t width, uint32_t height );
    std::vector < PenguinV::Image > uniformImages( uint32_t images, const PenguinV::Image & reference = PenguinV::Image() );
    std::vector < PenguinV::Image > uniformImages( const std::vector < uint8_t > & intensityValue, const PenguinV::Image & reference = PenguinV::Image() );
    PenguinV::Image randomImage( uint32_t width = 0, uint32_t height = 0 );
    PenguinV::Image randomRGBImage(const PenguinV::Image & reference = PenguinV::Image());
    PenguinV::Image randomImage( const std::vector <uint8_t> & value );

    // Iteration count for tests
    uint32_t runCount(); // fixed value for all test loops
    void setRunCount( int argc, char* argv[], uint32_t count );

    // Return random value for specific range or variable type
    template <typename data>
    data randomValue( uint32_t maximum )
    {
        if( maximum == 0 )
            return 0;
        else
            return static_cast<data>(static_cast<uint32_t>(rand()) % maximum);
    }

    template <typename data>
    data randomValue( data minimum, uint32_t maximum )
    {
        if( maximum == 0 ) {
            return 0;
        }
        else {
            data value = static_cast<data>(static_cast<uint32_t>(rand()) % maximum);

            if( value < minimum )
                value = minimum;

            return value;
        }
    }
}
