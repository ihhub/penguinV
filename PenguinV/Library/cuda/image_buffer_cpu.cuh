#pragma once

#include <cuda_runtime.h>
#include "../image_buffer.h"

namespace Bitmap_Image_Cuda_Cpu
{
    class Image : public Bitmap_Image::Image
    {
    public:
        Image()
        {
        }

        explicit Image( uint8_t colorCount_ )
        {
            setColorCount( colorCount_ );
        }

        Image( uint32_t width_, uint32_t height_ )
        {
            resize( width_, height_ );
        }

        Image( uint32_t width_, uint32_t height_, uint8_t colorCount_ )
        {
            setColorCount( colorCount_ );
            resize( width_, height_ );
        }

        Image( uint32_t width_, uint32_t height_, uint8_t colorCount_, uint8_t alignment_ )
        {
            setColorCount( colorCount_ );
            setAlignment( alignment_ );
            resize( width_, height_ );
        }

        Image( const Image & image )
        {
            copy( image );
        }

        Image( Image && image )
        {
            swap( image );
        }

        Image & operator=( const Image & image )
        {
            Bitmap_Image::Image::operator=( image );

            return (*this);
        }

        Image & operator=( Image && image )
        {
            swap( image );
        
            return (*this);
        }

        virtual ~Image()
        {
            clear();
        }

        virtual uint8_t * _allocate( size_t size ) const
        {
            uint8_t * data = nullptr;

            cudaError_t error = cudaMallocHost( &data, sizeof(uint8_t) * size );
            if( error != cudaSuccess )
                throw imageException( "Cannot allocate pinned memory on HOST" );

            return data;
        }

        virtual void _deallocate( uint8_t * data ) const
        {
            cudaError_t error = cudaFreeHost( data );
            if( error != cudaSuccess )
                throw imageException( "Cannot deallocate pinned memory on HOST" );
        }
    };
}
