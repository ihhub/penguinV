#pragma once

#include <vector>
#include "../../../src/cuda/image_buffer_cuda.cuh"
#include "../performance_test_helper.h"

namespace Performance_Test
{
    namespace Cuda_Helper
    {
        class TimerContainerCuda : public BaseTimerContainer
        {
        public:
            TimerContainerCuda();
            ~TimerContainerCuda();

            void start(); // start time measurement
            void stop();  // stop time measurement

        private:
            cudaEvent_t _startEvent;
            cudaEvent_t _stopEvent;
        };

        typedef void(*performanceFunctionCuda)( TimerContainerCuda &, uint32_t);
        std::pair < double, double > runPerformanceTestCuda( performanceFunctionCuda function, uint32_t size, uint32_t threadCountDivider );

        // Functions to generate images
        Bitmap_Image_Cuda::Image uniformImage( uint32_t width, uint32_t height );
        Bitmap_Image_Cuda::Image uniformImage( uint32_t width, uint32_t height, uint8_t value );
        std::vector< Bitmap_Image_Cuda::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height );

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
        }
    }
}
