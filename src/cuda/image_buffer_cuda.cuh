#pragma once

#include <cuda_runtime.h>
#include "../image_buffer.h"
#include "../thirdparty/multicuda/src/cuda_device.cuh"

namespace Bitmap_Image_Cuda
{
    template <typename TColorDepth>
    class ImageTemplateCuda : public PenguinV_Image::ImageTemplate<TColorDepth>
    {
    public:
        ImageTemplateCuda( uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u )
        {
            PenguinV_Image::ImageTemplate<TColorDepth>::setColorCount( colorCount_ );
            PenguinV_Image::ImageTemplate<TColorDepth>::setAlignment( alignment_ );
            PenguinV_Image::ImageTemplate<TColorDepth>::resize( width_, height_ );
        }

        ImageTemplateCuda( const ImageTemplateCuda & image )
        {
            PenguinV_Image::ImageTemplate<TColorDepth>::operator=( image );
        }

        ImageTemplateCuda( ImageTemplateCuda && image )
        {
            PenguinV_Image::ImageTemplate<TColorDepth>::swap( image );
        }

        ImageTemplateCuda & operator=( const ImageTemplateCuda & image )
        {
            PenguinV_Image::ImageTemplate<TColorDepth>::operator=( image );

            return (*this);
        }

        ImageTemplateCuda & operator=( ImageTemplateCuda && image )
        {
            PenguinV_Image::ImageTemplate<TColorDepth>::swap( image );

            return (*this);
        }

        virtual ~ImageTemplateCuda()
        {
            PenguinV_Image::ImageTemplate<TColorDepth>::clear();
        }
    protected:
        virtual TColorDepth * _allocate( size_t size ) const
        {
            return multiCuda::MemoryManager::memory().allocate<TColorDepth>( size );
        }

        virtual void _deallocate( TColorDepth * data ) const
        {
            multiCuda::MemoryManager::memory().free( data );
        }

        virtual void _copy( TColorDepth * out, TColorDepth * in, size_t size )
        {
            cudaError error = cudaMemcpy( in, out, size, cudaMemcpyDeviceToDevice );
            if( error != cudaSuccess )
                throw imageException( "Cannot copy a memory in CUDA device" );
        }

        virtual void _set( TColorDepth * data, TColorDepth value, size_t size )
        {
            cudaError_t error = cudaMemset( data, value, size );
            if( error != cudaSuccess )
                throw imageException( "Cannot fill a memory for CUDA device" );
        }
    };

    typedef ImageTemplateCuda <uint8_t> Image;
}
