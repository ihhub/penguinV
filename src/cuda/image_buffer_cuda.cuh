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

#include "../image_buffer.h"
#include "cuda_device.cuh"

namespace penguinV
{
    template <typename TColorDepth>
    class ImageTemplateCuda : public ImageTemplate<TColorDepth>
    {
    public:
        explicit ImageTemplateCuda( uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u )
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
                throw penguinVException( "Cannot copy a memory in CUDA device" );
        }

        static void _setMemory( TColorDepth * data, TColorDepth value, size_t size )
        {
            cudaError_t error = cudaMemset( data, value, size );
            if( error != cudaSuccess )
                throw penguinVException( "Cannot fill a memory for CUDA device" );
        }
    };

    typedef penguinV::ImageTemplateCuda <uint8_t> ImageCuda;
}
