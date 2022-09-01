#pragma once

#include "../src/image_buffer.h"
#include <cstdlib>

namespace Test_Helper
{
    // Functions to generate images
    penguinV::Image uniformImage( uint32_t width = 0, uint32_t height = 0, const penguinV::Image & reference = penguinV::Image() );
    penguinV::Image uniformImage( uint8_t value, uint32_t width = 0, uint32_t height = 0, const penguinV::Image & reference = penguinV::Image() );
    penguinV::Image16Bit uniformImage16Bit( uint16_t value, uint32_t width = 0, uint32_t height = 0, const penguinV::Image16Bit & reference = penguinV::Image16Bit() );
    penguinV::Image uniformRGBImage( uint32_t width, uint32_t height );
    penguinV::Image uniformRGBImage( uint32_t width, uint32_t height, uint8_t value );
    penguinV::Image uniformRGBImage( const penguinV::Image & reference = penguinV::Image() );
    penguinV::Image uniformRGBImage( uint8_t value, const penguinV::Image & reference = penguinV::Image() );
    penguinV::Image uniformRGBAImage( uint32_t width, uint32_t height );
    penguinV::Image uniformRGBAImage( uint32_t width, uint32_t height, uint8_t value );
    penguinV::Image uniformRGBAImage( const penguinV::Image & reference = penguinV::Image() );
    penguinV::Image uniformRGBAImage( uint8_t value, const penguinV::Image & reference = penguinV::Image() );
    std::vector<penguinV::Image> uniformImages( uint32_t count, uint32_t width, uint32_t height );
    std::vector<penguinV::Image> uniformRGBImages( uint32_t count, uint32_t width, uint32_t height );
    std::vector<penguinV::Image> uniformImages( uint32_t images, const penguinV::Image & reference = penguinV::Image() );
    std::vector<penguinV::Image> uniformImages( const std::vector<uint8_t> & intensityValue, const penguinV::Image & reference = penguinV::Image() );
    penguinV::Image randomImage( uint32_t width = 0, uint32_t height = 0 );
    penguinV::Image randomRGBImage( const penguinV::Image & reference = penguinV::Image() );
    penguinV::Image randomImage( const std::vector<uint8_t> & value );

    // Iteration count for tests
    uint32_t runCount(); // fixed value for all test loops
    void setRunCount( int argc, char * argv[], uint32_t count );

    // Return random value for specific range or variable type
    template <typename data>
    data randomValue( uint32_t maximum )
    {
        if ( maximum == 0 )
            return 0;
        else
            return static_cast<data>( static_cast<uint32_t>( rand() ) % maximum );
    }

    template <typename data>
    data randomValue( data minimum, uint32_t maximum )
    {
        if ( maximum == 0 ) {
            return 0;
        }
        else {
            data value = static_cast<data>( static_cast<uint32_t>( rand() ) % maximum );

            if ( value < minimum )
                value = minimum;

            return value;
        }
    }
}
