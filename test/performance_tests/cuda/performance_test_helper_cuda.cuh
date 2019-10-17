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
        PenguinV_Image::Image uniformImage( uint32_t width, uint32_t height );
        PenguinV_Image::Image uniformImage( uint32_t width, uint32_t height, uint8_t value );
        std::vector< PenguinV_Image::Image > uniformImages( uint32_t count, uint32_t width, uint32_t height );
    }
}
