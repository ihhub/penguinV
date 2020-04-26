#pragma once

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "../image_buffer.h"
#include "opencl_device.h"

namespace penguinV
{
    class ImageOpenCL : public Image
    {
    public:
        explicit ImageOpenCL( uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u )
        {
            Image::_setType<uint8_t>( 3, _allocateMemory, _deallocateMemory, _copyMemory, _setMemory );
            Image::setColorCount( colorCount_ );
            Image::setAlignment( alignment_ );
            Image::resize( width_, height_ );
        }

        ImageOpenCL( const ImageOpenCL & image )
        {
            Image::operator=( image );
        }

        ImageOpenCL( ImageOpenCL && image )
        {
            Image::swap( image );
        }

        ImageOpenCL & operator=( const ImageOpenCL & image )
        {
            Image::operator=( image );

            return (*this);
        }

        ImageOpenCL & operator=( ImageOpenCL && image )
        {
            Image::swap( image );

            return (*this);
        }
    protected:
        template <typename TColorDepth>
        static TColorDepth * _allocateMemory( size_t size )
        {
            return reinterpret_cast<TColorDepth*>( multiCL::MemoryManager::memory().allocate<TColorDepth>( size ) );
        }

        template <typename TColorDepth>
        static void _deallocateMemory( TColorDepth * data )
        {
            multiCL::MemoryManager::memory().free( reinterpret_cast<cl_mem>( data ) );
        }

        template <typename TColorDepth>
        static void _copyMemory( TColorDepth * out, TColorDepth * in, size_t size )
        {
            cl_mem inMem  = reinterpret_cast<cl_mem>( in );
            cl_mem outMem = reinterpret_cast<cl_mem>( out );

            const cl_int error = clEnqueueCopyBuffer( multiCL::OpenCLDeviceManager::instance().device().queue()(), inMem, outMem, 0, 0, size, 0, NULL, NULL );
            if( error != CL_SUCCESS )
                throw imageException( "Cannot copy a memory in GPU device" );
        }

        template <typename TColorDepth>
        static void _setMemory( TColorDepth * data, TColorDepth value, size_t size )
        {
            cl_mem dataMem = reinterpret_cast<cl_mem>( data );

            multiCL::MemoryManager::memorySet( dataMem, &value, sizeof( TColorDepth ), 0, size );
        }
    };

    template<typename TImageColorType>
    class ImageOpenCLXType : public ImageOpenCL
    {
    public:
        explicit ImageOpenCLXType( uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u )
        {
            Image::_setType<TImageColorType>( 3, _allocateMemory, _deallocateMemory, _copyMemory, _setMemory );
            _setDataType<TImageColorType>();

            setColorCount( colorCount_ );
            setAlignment( alignment_ );
            resize( width_, height_ );
        }
    };

    typedef ImageOpenCLXType <uint16_t> ImageOpenCL16Bit;
}
