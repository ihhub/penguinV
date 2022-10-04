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

#include <cuda_runtime.h>
#include "../../../src/image_buffer.h"
#include "../../../src/penguinv_exception.h"
#include "../../../src/image_function.h"
#include "../../../src/cuda/cuda_types.cuh"
#include "../../../src/cuda/cuda_helper.cuh"
#include "../../../src/cuda/image_buffer_cuda.cuh"
#include "../unit_test_helper.h"
#include "unit_test_helper_cuda.cuh"

namespace
{
    __global__ void isEqualCuda( const uint8_t * image, uint8_t value, uint32_t rowSize, uint32_t width, uint32_t height, uint32_t * differenceCount )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            const uint32_t id = y * rowSize + x;

            if ( image[id] == value )
                atomicAdd( differenceCount, 1 );
        }
    }

    __global__ void isAnyEqualCuda( const uint8_t * image, uint8_t * value, size_t valueCount, uint32_t width, uint32_t height,
                                    uint32_t * differenceCount )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            const uint32_t id = y * width + x;

            bool equal = false;

            for ( uint32_t i = 0; i < valueCount; ++i ) {
                if ( image[id] == value[i] ) {
                    equal = true;
                    break;
                }
            }

            if ( equal )
                atomicAdd( differenceCount, 1 );
        }
    }
}

namespace Unit_Test
{
    namespace Cuda
    {
        bool verifyImage( const penguinV::Image & image, uint8_t value )
        {
            return verifyImage( image, 0, 0, image.width(), image.height(), value );
        }

        bool verifyImage( const penguinV::Image & image, const std::vector<uint8_t> & value )
        {
            multiCuda::Type<uint32_t> differenceCount( 0 );
            multiCuda::Array<uint8_t> valueCuda( value );

            const uint32_t rowSize = image.rowSize();
            const uint32_t height = image.height();

            launchKernel2D( isAnyEqualCuda, rowSize, height,
                            image.data(), valueCuda.data(), valueCuda.size(), rowSize, height, differenceCount.data() );

            return differenceCount.get() == rowSize * height;
        }

        bool verifyImage( const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value )
        {
            multiCuda::Type<uint32_t> differenceCount( 0 );

            const uint8_t colorCount = image.colorCount();
            width = width * colorCount;
            const uint32_t rowSize = image.rowSize();
            const uint8_t * data = image.data() + y * rowSize + x * colorCount;

            launchKernel2D( isEqualCuda, width, height,
                            data, value, rowSize, width, height, differenceCount.data() );

            return differenceCount.get() == width * height;
        }
    }
}
