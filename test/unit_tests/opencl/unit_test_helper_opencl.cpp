#include <map>
#include <memory>

#include "../../../src/image_buffer.h"
#include "../../../src/image_exception.h"
#include "../../../src/opencl/opencl_helper.h"
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

    PenguinV_Image::Image generateImage( uint32_t width, uint32_t height, uint8_t colorCount, uint8_t value )
    {
        PenguinV_Image::ImageOpenCL image( width, height, colorCount );

        image.fill( value );

        PenguinV_Image::Image imageOut;
        imageOut.swap( image );

        return imageOut;
    }
}

namespace Unit_Test
{
    namespace OpenCL
    {
        PenguinV_Image::Image uniformImage( uint8_t value )
        {
            return generateImage( randomValue<uint32_t>( 1, 2048 ), randomValue<uint32_t>( 1, 2048 ), PenguinV_Image::GRAY_SCALE, value );
        }

        PenguinV_Image::Image uniformImage()
        {
            return uniformImage( randomValue<uint8_t>( 256 ) );
        }

        PenguinV_Image::Image uniformRGBImage()
        {
            return uniformRGBImage( randomValue<uint8_t>( 256 ) );
        }

        PenguinV_Image::Image uniformRGBImage( uint8_t value )
        {
            return generateImage( randomValue<uint32_t>( 1, 2048 ), randomValue<uint32_t>( 1, 2048 ), PenguinV_Image::RGB, value );
        }

        PenguinV_Image::Image blackImage()
        {
            return uniformImage( 0u );
        }

        PenguinV_Image::Image whiteImage()
        {
            return uniformImage( 255u );
        }

        std::vector < PenguinV_Image::Image > uniformImages( uint32_t images )
        {
            if ( images == 0 )
                throw imageException( "Invalid parameter" );

            std::vector < PenguinV_Image::Image > image;

            image.push_back( uniformImage() );

            image.resize( images );

            for ( size_t i = 1; i < image.size(); ++i ) {
                image[i] = image.front().generate( image[0].width(), image[0].height() );
                image[i].fill( randomValue<uint8_t>( 256 ) );
            }

            return image;
        }

        std::vector < PenguinV_Image::Image > uniformImages( std::vector < uint8_t > intensityValue )
        {
            if ( intensityValue.size() == 0 )
                throw imageException( "Invalid parameter" );

            std::vector < PenguinV_Image::Image > image;

            image.push_back( uniformImage( intensityValue[0] ) );

            image.resize( intensityValue.size() );

            for ( size_t i = 1; i < image.size(); ++i ) {
                image[i] = image.front().generate( image[0].width(), image[0].height() );
                image[i].fill( intensityValue[i] );
            }

            return image;
        }

        bool verifyImage( const PenguinV_Image::Image & image, uint8_t value )
        {
            uint32_t count = 0;
            cl_mem differenceCount = multiCL::MemoryManager::memory().allocate<uint32_t>( 1 );
            multiCL::writeBuffer( differenceCount, sizeof( uint32_t ), &count );

            const multiCL::OpenCLProgram & program = GetProgram();
            multiCL::OpenCLKernel kernel( program, "isEqualOpenCL" );

            const uint32_t rowSize = image.rowSize();
            const uint32_t width = image.width() * image.colorCount();
            const uint32_t height = image.height();

            kernel.setArgument( image.data(), value, rowSize, width, height, differenceCount );

            multiCL::launchKernel2D( kernel, width, height );

            multiCL::readBuffer( differenceCount, sizeof( uint32_t ), &count );
            multiCL::MemoryManager::memory().free( differenceCount );

            return (count == width * height);
        }

        bool verifyImage( const PenguinV_Image::Image & image, const std::vector < uint8_t > & value )
        {
            uint32_t count = 0;
            cl_mem differenceCount = multiCL::MemoryManager::memory().allocate<uint32_t>( 1 );
            multiCL::writeBuffer( differenceCount, sizeof( uint32_t ), &count );

            cl_mem valueOpenCL = multiCL::MemoryManager::memory().allocate<uint8_t>( value.size() );
            multiCL::writeBuffer( valueOpenCL, sizeof( uint8_t ) * value.size(), value.data() );

            const multiCL::OpenCLProgram & program = GetProgram();
            multiCL::OpenCLKernel kernel( program, "isAnyEqualOpenCL" );

            const uint32_t rowSize = image.rowSize();
            const uint32_t width = image.width() * image.colorCount();
            const uint32_t height = image.height();

            kernel.setArgument( image.data(), valueOpenCL, static_cast<uint32_t>( value.size() ), rowSize, width, height, differenceCount );

            multiCL::launchKernel2D( kernel, width, height );

            multiCL::MemoryManager::memory().free( valueOpenCL );

            multiCL::readBuffer( differenceCount, sizeof( uint32_t ), &count );
            multiCL::MemoryManager::memory().free( differenceCount );

            return (count == width * height);
        }
    }
}
