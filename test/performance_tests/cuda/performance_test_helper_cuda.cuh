/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2022                                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

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
        penguinV::Image uniformImage( uint32_t width, uint32_t height );
        penguinV::Image uniformImage( uint32_t width, uint32_t height, uint8_t value );
        std::vector<penguinV::Image> uniformImages( uint32_t count, uint32_t width, uint32_t height );
    }
}
