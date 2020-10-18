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
