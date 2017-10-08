#pragma once

#include <algorithm>
#include <cstdlib>
#include <vector>
#include "../Library/image_buffer.h"

// A bunch of functions to help writing unit tests
namespace Unit_Test
{
    // Generate images
    Bitmap_Image::Image uniformImage();
    Bitmap_Image::Image uniformImage( uint8_t value );
    Bitmap_Image::Image uniformColorImage();
    Bitmap_Image::Image uniformColorImage( uint8_t value );
    Bitmap_Image::Image blackImage();
    Bitmap_Image::Image whiteImage();
    Bitmap_Image::Image randomImage();
    Bitmap_Image::Image randomImage( const std::vector <uint8_t> & value );
    std::vector < Bitmap_Image::Image > uniformImages( size_t images );
    std::vector < Bitmap_Image::Image > uniformImages( std::vector < uint8_t > intensityValue );

    // Generate pixel intensity values
    uint8_t intensityValue();
    std::vector < uint8_t > intensityArray( size_t size );

    // Image size and ROI verification
    template <typename data>
    bool equalSize( const data & image1, const data & image2 )
    {
        return image1.height() == image2.height() && image1.width() == image2.width() &&
            image1.alignment() == image2.alignment() && image1.colorCount() == image2.colorCount()
            && image1.rowSize() == image2.rowSize();
    };

    template <typename data>
    bool equalSize( const data & image, size_t width, size_t height, size_t rowSize, uint8_t colorCount,
                    uint8_t alignment )
    {
        return ((width == 0 || height == 0) && image.data() == nullptr && image.width() == 0 && image.height() == 0 &&
                 image.colorCount() == colorCount && image.alignment() == alignment && image.rowSize() == 0) ||
                 (width == image.width() && height == image.height() && colorCount == image.colorCount() &&
                   alignment == image.alignment() && rowSize == image.rowSize());
    };

    bool equalSize( const Bitmap_Image::Image & image, size_t width, size_t height );

    template <typename data>
    bool equalData( const Template_Image::ImageTemplate < data > & image1, const Template_Image::ImageTemplate < data > & image2 )
    {
        return memcmp( image1.data(), image2.data(), sizeof( data ) * image1.height() * image1.rowSize() ) == 0;
    };

    template <typename data>
    bool isEmpty( const Template_Image::ImageTemplate < data > & image )
    {
        return image.data() == nullptr && image.width() == 0 && image.height() == 0 &&
            image.colorCount() == 1 && image.alignment() == 1 && image.rowSize() == 0;
    };

    bool verifyImage( const Bitmap_Image::Image & image, size_t x, size_t y, size_t width, size_t height, uint8_t value );
    bool verifyImage( const Bitmap_Image::Image & image, size_t x, size_t y, size_t width, size_t height,
                      const std::vector < uint8_t > & value );
    bool verifyImage( const Bitmap_Image::Image & image, uint8_t value );
    bool verifyImage( const Bitmap_Image::Image & image, const std::vector < uint8_t > & value );

    // Fill image ROI with specific intensity
    void fillImage( Bitmap_Image::Image & image, size_t x, size_t y, size_t width, size_t height, uint8_t value );
    void fillImage( Bitmap_Image::Image & image, size_t x, size_t y, size_t width, size_t height,
                    const std::vector < uint8_t > & value );

    // Generate and return ROI based on full image size
    void generateRoi( const Bitmap_Image::Image & image, size_t & x, size_t & y, size_t & width, size_t & height );
    void generateRoi( const std::vector < Bitmap_Image::Image > & image, std::vector < size_t > & x, std::vector < size_t > & y,
                      size_t & width, size_t & height );
    // first element in pair structure is width, second - height
    void generateRoi( const std::vector < std::pair< size_t, size_t > > & imageSize, std::vector < size_t > & x,
                      std::vector < size_t > & y, size_t & width, size_t & height );

    void generateOffset( const Bitmap_Image::Image & image, size_t & x, size_t & y, size_t width, size_t height );

    std::pair <size_t, size_t> imageSize( const Bitmap_Image::Image & image );

    // Return calculated row size
    size_t rowSize( size_t width, uint8_t colorCount = 1, uint8_t alignment = 1 );

    size_t runCount(); // fixed value for all test loops

    // Return random value for specific range or variable type
    template <typename data>
    data randomValue( size_t maximum )
    {
        if( maximum <= 0 )
            return 0;
        else
            return static_cast<data>(static_cast<size_t>(rand()) % maximum);
    };

    template <typename data>
    data randomValue( data minimum, size_t maximum )
    {
        if( maximum <= 0 ) {
            return 0;
        }
        else {
            data value = static_cast<data>(static_cast<size_t>(rand()) % maximum);

            if( value < minimum )
                value = minimum;

            return value;
        }
    };
};
