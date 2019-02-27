#include <map>
#include <memory>

#include "../../../src/image_exception.h"
#include "../../../src/opencl/opencl_helper.h"
#include "../../../src/opencl/opencl_types.h"
#include "../../../src/opencl/image_buffer_opencl.h"
#include "../unit_test_helper.h"
#include "unit_test_helper_opencl.h"

namespace
{
    const std::string programCode = R"(
        #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

        __kernel void isEqualOpenCL( __global const uchar * data, uchar value, uint rowSize, uint width, uint height, volatile __global uint * differenceCount )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * rowSize + x;
                if( data[id] == value )
                    atomic_add( differenceCount, 1 );
            }
        }

        __kernel void isAnyEqualOpenCL( __global const uchar * data, __global uchar * value, uint valueCount, uint rowSize, uint width, uint height, volatile __global uint * differenceCount )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * rowSize + x;

                bool equal = false;

                for( uint i = 0; i < valueCount; ++i ) {
                    if( data[id] == value[i] ) {
                        equal = true;
                        break;
                    }
                }

                if( equal )
                    atomic_add( differenceCount, 1 );
            }
        }
    )";

    const multiCL::OpenCLProgram& GetProgram()
    {
        static std::map< cl_device_id, std::shared_ptr< multiCL::OpenCLProgram > > deviceProgram;

        multiCL::OpenCLDevice & device = multiCL::OpenCLDeviceManager::instance().device();
        std::map< cl_device_id, std::shared_ptr< multiCL::OpenCLProgram > >::const_iterator program = deviceProgram.find( device.deviceId() );
        if ( program != deviceProgram.cend() )
            return *(program->second);

        deviceProgram[device.deviceId()] = std::shared_ptr< multiCL::OpenCLProgram >( new multiCL::OpenCLProgram( device.context(), programCode.data() ) );
        return *(deviceProgram[device.deviceId()]);
    }
}

namespace Unit_Test
{
    namespace OpenCL
    {
        bool verifyImage( const PenguinV_Image::Image & image, uint8_t value )
        {
            multiCL::Type< uint32_t > differenceCount( 0u );

            const multiCL::OpenCLProgram & program = GetProgram();
            multiCL::OpenCLKernel kernel( program, "isEqualOpenCL" );

            const uint32_t rowSize = image.rowSize();
            const uint32_t width = image.width() * image.colorCount();
            const uint32_t height = image.height();

            kernel.setArgument( image.data(), value, rowSize, width, height, differenceCount.data() );

            multiCL::launchKernel2D( kernel, width, height );

            return (differenceCount.get() == width * height);
        }

        bool verifyImage( const PenguinV_Image::Image & image, const std::vector < uint8_t > & value )
        {
            multiCL::Type< uint32_t > differenceCount( 0u );

            cl_mem valueOpenCL = multiCL::MemoryManager::memory().allocate<uint8_t>( value.size() );
            multiCL::writeBuffer( valueOpenCL, sizeof( uint8_t ) * value.size(), value.data() );

            const multiCL::OpenCLProgram & program = GetProgram();
            multiCL::OpenCLKernel kernel( program, "isAnyEqualOpenCL" );

            const uint32_t rowSize = image.rowSize();
            const uint32_t width = image.width() * image.colorCount();
            const uint32_t height = image.height();

            kernel.setArgument( image.data(), valueOpenCL, static_cast<uint32_t>( value.size() ), rowSize, width, height, differenceCount.data() );

            multiCL::launchKernel2D( kernel, width, height );

            multiCL::MemoryManager::memory().free( valueOpenCL );

            return (differenceCount.get() == width * height);
        }
    }
}
