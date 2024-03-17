/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2024                                             *
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

#define CL_TARGET_OPENCL_VERSION 210

#if defined( __APPLE__ ) || defined( __MACOSX )
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "../image_buffer.h"
#include "opencl_device.h"

namespace penguinV
{
    template <typename TColorDepth>
    class ImageTemplateOpenCL : public ImageTemplate<TColorDepth>
    {
    public:
        explicit ImageTemplateOpenCL( uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u )
        {
            ImageTemplate<TColorDepth>::_setType( 3, _allocateMemory, _deallocateMemory, _copyMemory, _setMemory );
            ImageTemplate<TColorDepth>::setColorCount( colorCount_ );
            ImageTemplate<TColorDepth>::setAlignment( alignment_ );
            ImageTemplate<TColorDepth>::resize( width_, height_ );
        }

        ImageTemplateOpenCL( const ImageTemplateOpenCL & image )
        {
            ImageTemplate<TColorDepth>::operator=( image );
        }

        ImageTemplateOpenCL( ImageTemplateOpenCL && image )
        {
            ImageTemplate<TColorDepth>::swap( image );
        }

        ImageTemplateOpenCL & operator=( const ImageTemplateOpenCL & image )
        {
            ImageTemplate<TColorDepth>::operator=( image );

            return ( *this );
        }

        ImageTemplateOpenCL & operator=( ImageTemplateOpenCL && image )
        {
            ImageTemplate<TColorDepth>::swap( image );

            return ( *this );
        }

    private:
        static TColorDepth * _allocateMemory( size_t size )
        {
            return reinterpret_cast<TColorDepth *>( multiCL::MemoryManager::memory().allocate<TColorDepth>( size ) );
        }

        static void _deallocateMemory( TColorDepth * data )
        {
            multiCL::MemoryManager::memory().free( reinterpret_cast<cl_mem>( data ) );
        }

        static void _copyMemory( TColorDepth * out, TColorDepth * in, size_t size )
        {
            cl_mem inMem = reinterpret_cast<cl_mem>( in );
            cl_mem outMem = reinterpret_cast<cl_mem>( out );

            const cl_int error = clEnqueueCopyBuffer( multiCL::OpenCLDeviceManager::instance().device().queue()(), inMem, outMem, 0, 0, size, 0, NULL, NULL );
            if ( error != CL_SUCCESS )
                throw penguinVException( "Cannot copy a memory in GPU device" );
        }

        static void _setMemory( TColorDepth * data, TColorDepth value, size_t size )
        {
            cl_mem dataMem = reinterpret_cast<cl_mem>( data );

            multiCL::MemoryManager::memorySet( dataMem, &value, sizeof( TColorDepth ), 0, size );
        }
    };

    typedef penguinV::ImageTemplateOpenCL<uint8_t> ImageOpenCL;
}
