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

#include <cuda_runtime.h>
#include "../image_buffer.h"

namespace penguinV
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
                throw penguinVException( "Cannot allocate pinned memory on HOST" );

            return data;
        }

        static void _deallocateMemory( TColorDepth * data )
        {
            cudaError_t error = cudaFreeHost( data );
            if( error != cudaSuccess )
                throw penguinVException( "Cannot deallocate pinned memory on HOST" );
        }
    };

    typedef penguinV::ImageTemplateCudaPinned <uint8_t> ImageCudaPinned;
}
