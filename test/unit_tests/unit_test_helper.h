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
    penguinV::Image blackImage( const penguinV::Image & reference = penguinV::Image() );
    penguinV::Image whiteImage( const penguinV::Image & reference = penguinV::Image() );

    // Generate pixel intensity values
    uint8_t intensityValue();
    std::vector < uint8_t > intensityArray( uint32_t size );

    // Image size and ROI verification
    template <typename data>
    bool equalSize( const data & image1, const data & image2 )
    {
        return image1.height() == image2.height() && image1.width() == image2.width() &&
            image1.alignment() == image2.alignment() && image1.colorCount() == image2.colorCount()
            && image1.rowSize() == image2.rowSize() && image1.dataSize() == image2.dataSize()
            && image1.dataType() == image2.dataType();
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

    bool equalSize( const penguinV::Image & image, uint32_t width, uint32_t height );

    template <typename TData>
    bool equalData( const penguinV::Image & image1, const penguinV::Image & image2 )
    {
        return memcmp( image1.data<TData>(), image2.data<TData>(), sizeof( TData ) * image1.height() * image1.rowSize() ) == 0;
    }

    bool isEmpty( const penguinV::Image & image );

    bool verifyImage( const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value );
    bool verifyImage( const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const std::vector<uint8_t> & value,
                      bool isAnyValue = true );
    bool verifyImage( const penguinV::Image & image, uint8_t value );
    bool verifyImage( const penguinV::Image & image, const std::vector<uint8_t> & value, bool isAnyValue = true );

    bool verify16BitImage( const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint16_t value );
    bool verify16BitImage( const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const std::vector<uint16_t> & value,
                           bool isAnyValue = true );
    bool verify16BitImage( const penguinV::Image & image, uint16_t value );
    bool verify16BitImage( const penguinV::Image & image, const std::vector<uint16_t> & value, bool isAnyValue = true );

    // Fill image ROI with specific intensity
    void fillImage( penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value );
    void fillImage( penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const std::vector<uint8_t> & value );

    // Generate and return ROI based on full image size
    void generateRoi( const penguinV::Image & image, uint32_t & x, uint32_t & y, uint32_t & width, uint32_t & height );
    void generateRoi( const std::vector<penguinV::Image> & image, std::vector<uint32_t> & x, std::vector<uint32_t> & y, uint32_t & width, uint32_t & height );
    // first element in pair structure is width, second - height
    void generateRoi( const std::vector < std::pair< uint32_t, uint32_t > > & imageSize, std::vector < uint32_t > & x,
                      std::vector < uint32_t > & y, uint32_t & width, uint32_t & height );

    void generateOffset( const penguinV::Image & image, uint32_t & x, uint32_t & y, uint32_t width, uint32_t height );

    std::pair<uint32_t, uint32_t> imageSize( const penguinV::Image & image );

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
