#pragma once

#include <cstdlib>
#include <vector>
#include "../src/image_buffer.h"

namespace Test_Helper
{
    // Functions to generate images
    PenguinV_Image::Image uniformImage( uint32_t width = 0, uint32_t height = 0,  const PenguinV_Image::Image & reference = PenguinV_Image::Image() );
    PenguinV_Image::Image uniformImage( uint8_t value, uint32_t width = 0, uint32_t height = 0, const PenguinV_Image::Image & reference = PenguinV_Image::Image() );
    PenguinV_Image::Image uniformRGBImage( uint32_t width, uint32_t height );
    PenguinV_Image::Image uniformRGBImage( uint32_t width, uint32_t height, uint8_t value );
    PenguinV_Image::Image uniformRGBImage( const PenguinV_Image::Image & reference = PenguinV_Image::Image() );
    PenguinV_Image::Image uniformRGBImage( uint8_t value, const PenguinV_Image::Image & reference = PenguinV_Image::Image() );
    std::vector < PenguinV_Image::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height );
    std::vector < PenguinV_Image::Image > uniformRGBImages( uint32_t count, uint32_t width, uint32_t height );
    std::vector < PenguinV_Image::Image > uniformImages( uint32_t images, const PenguinV_Image::Image & reference = PenguinV_Image::Image() );
    std::vector < PenguinV_Image::Image > uniformImages( const std::vector < uint8_t > & intensityValue, const PenguinV_Image::Image & reference = PenguinV_Image::Image() );

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
     
  



