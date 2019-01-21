#pragma once

#include <cstdlib>
#include <vector>
#include "../../src/image_buffer.h"

// A bunch of functions to help writing unit tests
namespace Unit_Test
{
    // Generate images
    // reference is used to generate a specific type of images aka CPU, CUDA, OpenCL.
    // For CPU memory based image you could skip reference parameter
    PenguinV_Image::Image blackImage( const PenguinV_Image::Image & reference = PenguinV_Image::Image() );
    PenguinV_Image::Image whiteImage( const PenguinV_Image::Image & reference = PenguinV_Image::Image() );
    PenguinV_Image::Image randomImage( uint32_t width = 0, uint32_t height = 0 );
    PenguinV_Image::Image randomImage( const std::vector <uint8_t> & value );
    PenguinV_Image::Image randomRGBImage(const PenguinV_Image::Image & reference = PenguinV_Image::Image());

    // Generate pixel intensity values
    uint8_t intensityValue();
    std::vector < uint8_t > intensityArray( uint32_t size );

    // Image size and ROI verification
    template <typename data>
    bool equalSize( const data & image1, const data & image2 )
    {
        return image1.height() == image2.height() && image1.width() == image2.width() &&
            image1.alignment() == image2.alignment() && image1.colorCount() == image2.colorCount()
            && image1.rowSize() == image2.rowSize();
    }

    template <typename data>
    bool equalSize( const data & image, uint32_t width, uint32_t height, uint32_t rowSize, uint8_t colorCount,
                    uint8_t alignment )
    {
        return ((width == 0 || height == 0) && image.data() == nullptr && image.width() == 0 && image.height() == 0 &&
                 image.colorCount() == colorCount && image.alignment() == alignment && image.rowSize() == 0) ||
                 (width == image.width() && height == image.height() && colorCount == image.colorCount() &&
                   alignment == image.alignment() && rowSize == image.rowSize());
    }

    bool equalSize( const PenguinV_Image::Image & image, uint32_t width, uint32_t height );

    template <typename data>
    bool equalData( const PenguinV_Image::ImageTemplate < data > & image1, const PenguinV_Image::ImageTemplate < data > & image2 )
    {
        return memcmp( image1.data(), image2.data(), sizeof( data ) * image1.height() * image1.rowSize() ) == 0;
    }

    template <typename data>
    bool isEmpty( const PenguinV_Image::ImageTemplate < data > & image )
    {
        return image.data() == nullptr && image.width() == 0 && image.height() == 0 &&
               image.colorCount() == 1 && image.alignment() == 1 && image.rowSize() == 0;
    }

    bool verifyImage( const PenguinV_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value );
    bool verifyImage( const PenguinV_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                      const std::vector < uint8_t > & value, bool isAnyValue = true );
    bool verifyImage( const PenguinV_Image::Image & image, uint8_t value );
    bool verifyImage( const PenguinV_Image::Image & image, const std::vector < uint8_t > & value, bool isAnyValue = true );

    // Fill image ROI with specific intensity
    void fillImage( PenguinV_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value );
    void fillImage( PenguinV_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                    const std::vector < uint8_t > & value );

    // Generate and return ROI based on full image size
    void generateRoi( const PenguinV_Image::Image & image, uint32_t & x, uint32_t & y, uint32_t & width, uint32_t & height );
    void generateRoi( const std::vector < PenguinV_Image::Image > & image, std::vector < uint32_t > & x, std::vector < uint32_t > & y,
                      uint32_t & width, uint32_t & height );
    // first element in pair structure is width, second - height
    void generateRoi( const std::vector < std::pair< uint32_t, uint32_t > > & imageSize, std::vector < uint32_t > & x,
                      std::vector < uint32_t > & y, uint32_t & width, uint32_t & height );

    void generateOffset( const PenguinV_Image::Image & image, uint32_t & x, uint32_t & y, uint32_t width, uint32_t height );

    std::pair <uint32_t, uint32_t> imageSize( const PenguinV_Image::Image & image );

    // Return calculated row size
    uint32_t rowSize( uint32_t width, uint8_t colorCount = 1, uint8_t alignment = 1 );

    uint32_t runCount(); // fixed value for all test loops
    void setRunCount( uint32_t count );

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

    template <typename data>
    data randomFloatValue( data minimum, data maximum, data stepVal )
    {
        if (minimum > maximum || stepVal < 0)
            return minimum;

        int range = static_cast<int>( (maximum - minimum) / stepVal );
        if (range <= 0)
            range = 1;

        return static_cast<data>(rand() % range) * stepVal + minimum;
    }
}
