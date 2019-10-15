#pragma once

#include <cstdlib>
#include <vector>
#include "../../src/image_buffer.h"
#include "../test_helper.h"

// A bunch of functions to help writing unit tests
namespace Unit_Test
{
    using namespace Test_Helper;

    // Generate images
    // reference is used to generate a specific type of images aka CPU, CUDA, OpenCL.
    // For CPU memory based image you could skip reference parameter
    PenguinV_Image::Image blackImage( const PenguinV_Image::Image & reference = PenguinV_Image::Image() );
    PenguinV_Image::Image whiteImage( const PenguinV_Image::Image & reference = PenguinV_Image::Image() );

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

    bool verifyImage( const PenguinV_Image::Image16Bit & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint16_t value );
    bool verifyImage( const PenguinV_Image::Image16Bit & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                      const std::vector < uint16_t > & value, bool isAnyValue = true );
    bool verifyImage( const PenguinV_Image::Image16Bit & image, uint16_t value );
    bool verifyImage( const PenguinV_Image::Image16Bit & image, const std::vector < uint16_t > & value, bool isAnyValue = true );

    // Fill image ROI with specific intensity
    void fillImage( PenguinV_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value );
    void fillImage( PenguinV_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                    const std::vector < uint8_t > & value );

    // Generate and return ROI based on full image size
    template <typename _Type>
    void generateRoi( const PenguinV_Image::ImageTemplate<_Type> & image, uint32_t & x, uint32_t & y, uint32_t & width, uint32_t & height )
    {
        width  = randomValue<uint32_t>( 1, image.width() );
        height = randomValue<uint32_t>( 1, image.height() );

        x = randomValue<uint32_t>( image.width()  - width );
        y = randomValue<uint32_t>( image.height() - height );
    }
    void generateRoi( const std::vector < PenguinV_Image::Image > & image, std::vector < uint32_t > & x, std::vector < uint32_t > & y,
                      uint32_t & width, uint32_t & height );
    // first element in pair structure is width, second - height
    void generateRoi( const std::vector < std::pair< uint32_t, uint32_t > > & imageSize, std::vector < uint32_t > & x,
                      std::vector < uint32_t > & y, uint32_t & width, uint32_t & height );

    void generateOffset( const PenguinV_Image::Image & image, uint32_t & x, uint32_t & y, uint32_t width, uint32_t height );

    template <typename _Type>
    std::pair <uint32_t, uint32_t> imageSize( const PenguinV_Image::ImageTemplate<_Type> & image )
    {
        return std::pair <uint32_t, uint32_t>( image.width(), image.height() );
    }

    // Return calculated row size
    uint32_t rowSize( uint32_t width, uint8_t colorCount = 1, uint8_t alignment = 1 );

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
