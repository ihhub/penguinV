#pragma once

#include <cuda_runtime.h>
#include "../image_buffer.h"

namespace PenguinV_Image
{
    template <typename TColorDepth>
    class ImageTemplateCudaPinned : public ImageTemplate<TColorDepth>
    {
    public:
        explicit ImageTemplateCudaPinned( uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u )
        {
            ImageTemplate<TColorDepth>::_setType( 2, _allocateMemory, _deallocateMemory );
            ImageTemplate<TColorDepth>::setColorCount( colorCount_ );
            ImageTemplate<TColorDepth>::setAlignment( alignment_ );
            ImageTemplate<TColorDepth>::resize( width_, height_ );
        }

        ImageTemplateCudaPinned( const ImageTemplateCudaPinned & image )
        {
            ImageTemplate<TColorDepth>::operator=( image );
        }

        ImageTemplateCudaPinned( ImageTemplateCudaPinned && image )
        {
            ImageTemplate<TColorDepth>::swap( image );
        }

        ImageTemplateCudaPinned & operator=( const ImageTemplateCudaPinned & image )
        {
            ImageTemplate<TColorDepth>::operator=( image );

            return (*this);
        }

        ImageTemplateCudaPinned & operator=( ImageTemplateCudaPinned && image )
        {
            ImageTemplate<TColorDepth>::swap( image );

            return (*this);
        }
    private:
        static TColorDepth * _allocateMemory( size_t size )
        {
            uint8_t * data = nullptr;

            cudaError_t error = cudaMallocHost( &data, sizeof(uint8_t) * size );
            if( error != cudaSuccess )
                throw imageException( "Cannot allocate pinned memory on HOST" );

            return data;
        }

        static void _deallocateMemory( TColorDepth * data )
        {
            cudaError_t error = cudaFreeHost( data );
            if( error != cudaSuccess )
                throw imageException( "Cannot deallocate pinned memory on HOST" );
        }
    };

    typedef PenguinV_Image::ImageTemplateCudaPinned <uint8_t> ImageCudaPinned;
}
