#pragma once

#include <algorithm>
#include <cstdlib>
#include <vector>
#include "../../Library/cuda/image_buffer_cuda.cuh"

// A bunch of functions to help writing unit tests
namespace Unit_Test
{
    namespace Cuda
    {
        // Generate images
        Bitmap_Image_Cuda::Image uniformImage();
        Bitmap_Image_Cuda::Image uniformImage( uint8_t value );
        Bitmap_Image_Cuda::Image uniformColorImage();
        Bitmap_Image_Cuda::Image uniformColorImage( uint8_t value );
        Bitmap_Image_Cuda::Image blackImage();
        Bitmap_Image_Cuda::Image whiteImage();
        std::vector < Bitmap_Image_Cuda::Image > uniformImages( size_t images );
        std::vector < Bitmap_Image_Cuda::Image > uniformImages( std::vector < uint8_t > intensityValue );

        // Image size and ROI verification
        template <typename data>
        bool equalSize( const data & image1, const data & image2 )
        {
            return image1.height() == image2.height() && image1.width() == image2.width() &&
                image1.colorCount() == image2.colorCount();
        };

        bool verifyImage( const Bitmap_Image_Cuda::Image & image, uint8_t value );
        bool verifyImage( const Bitmap_Image_Cuda::Image & image, const std::vector < uint8_t > & value );

        // Return random value for specific range or variable type
        template <typename data>
        data randomValue( int maximum )
        {
            if( maximum <= 0 )
                return 0;
            else
                return static_cast<data>(rand()) % maximum;
        };

        template <typename data>
        data randomValue( data minimum, int maximum )
        {
            if( maximum <= 0 ) {
                return 0;
            }
            else {
                data value = static_cast<data>(rand()) % maximum;

                if( value < minimum )
                    value = minimum;

                return value;
            }
        };

    };
};
