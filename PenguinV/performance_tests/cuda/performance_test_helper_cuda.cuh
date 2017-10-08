#pragma once

#include <vector>
#include "../../Library/cuda/image_buffer_cuda.cuh"
#include "../performance_test_helper.h"

namespace Performance_Test
{
    namespace Cuda_Helper
    {
        class TimerContainerCuda : public TimerContainer
        {
        public:
            void stop();  // stop time measurement
        };

        // Functions to generate images
        Bitmap_Image_Cuda::Image uniformImage( size_t width, size_t height );
        Bitmap_Image_Cuda::Image uniformImage( size_t width, size_t height, uint8_t value );
        std::vector< Bitmap_Image_Cuda::Image > uniformImages( size_t count, size_t width, size_t height );

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
