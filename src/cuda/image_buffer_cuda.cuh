#pragma once

#include "../image_buffer.h"
#include "cuda_device.cuh"

namespace PenguinV_Image
{
    template <typename TColorDepth>
    class ImageTemplateCuda : public ImageTemplate<TColorDepth>
    {
    public:
        ImageTemplateCuda( uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u )
        {
            ImageTemplate<TColorDepth>::_setType( 1, _allocateMemory, _deallocateMemory, _copyMemory, _setMemory );
            ImageTemplate<TColorDepth>::setColorCount( colorCount_ );
            ImageTemplate<TColorDepth>::setAlignment( alignment_ );
            ImageTemplate<TColorDepth>::resize( width_, height_ );
        }

        ImageTemplateCuda( const ImageTemplateCuda & image )
        {
            ImageTemplate<TColorDepth>::operator=( image );
        }

        ImageTemplateCuda( ImageTemplateCuda && image )
        {
            ImageTemplate<TColorDepth>::swap( image );
        }

        ImageTemplateCuda & operator=( const ImageTemplateCuda & image )
        {
            ImageTemplate<TColorDepth>::operator=( image );

            return (*this);
        }

        ImageTemplateCuda & operator=( ImageTemplateCuda && image )
        {
            ImageTemplate<TColorDepth>::swap( image );

            return (*this);
        }
    private:
        static TColorDepth * _allocateMemory( size_t size )
        {
            return multiCuda::MemoryManager::memory().allocate<TColorDepth>( size );
        }

        static void _deallocateMemory( TColorDepth * data )
        {
            multiCuda::MemoryManager::memory().free( data );
        }

        static void _copyMemory( TColorDepth * out, TColorDepth * in, size_t size )
        {
            cudaError error = cudaMemcpy( out, in, size, cudaMemcpyDeviceToDevice );
            if( error != cudaSuccess )
                throw imageException( "Cannot copy a memory in CUDA device" );
        }

        static void _setMemory( TColorDepth * data, TColorDepth value, size_t size )
        {
            cudaError_t error = cudaMemset( data, value, size );
            if( error != cudaSuccess )
                throw imageException( "Cannot fill a memory for CUDA device" );
        }
    };

    typedef PenguinV_Image::ImageTemplateCuda <uint8_t> ImageCuda;
}
