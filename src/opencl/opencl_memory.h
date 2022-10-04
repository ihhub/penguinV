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

#if defined( __APPLE__ ) || defined( __MACOSX )
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "../memory/memory_allocator.h"
#include <map>

namespace multiCL
{
    // Class for memory allocation on OpenCL devices
    class MemoryAllocator : public BaseMemoryAllocator
    {
    public:
        MemoryAllocator( cl_context context, cl_device_id deviceId, size_t availableSpace )
            : _context( context )
            , _minimumSizeChunk( 1 )
            , _data( NULL )
            , _availableSize( availableSpace )
        {
            const cl_int error = clGetDeviceInfo( deviceId, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof( cl_uint ), &_minimumSizeChunk, NULL );
            if ( error != CL_SUCCESS )
                throw std::logic_error( "Cannot get an information about minimum allocation size on OpenCL device" );
        }

        virtual ~MemoryAllocator()
        {
            _free();
        }

        // Returns a memory structure pointer to an allocated memory. If memory size of allocated memory chuck is enough for requested size
        // then assign a pointer to preallocated memory, otherwise allocate a new chuck of memory just for this pointer
        template <typename _DataType>
        cl_mem allocate( size_t size = 1 )
        {
            size = size * sizeof( _DataType );

            if ( _data != NULL && size < _size ) {
                const uint8_t level = _getAllocationLevel( size < _minimumSizeChunk ? _minimumSizeChunk : size );

                if ( _split( level ) ) {
                    cl_buffer_region region;
                    region.origin = *( _freeChunk[level].begin() );
                    region.size = size;

                    cl_int error;
                    cl_mem memory = clCreateSubBuffer( _data, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &error );
                    if ( error != CL_SUCCESS )
                        throw std::logic_error( "Cannot allocate a subbuffer memory for OpenCL device" );

                    _allocatedChunk.insert(
                        std::pair<cl_mem, std::pair<size_t, uint8_t>>( memory, std::pair<size_t, uint8_t>( *( _freeChunk[level].begin() ), level ) ) );
                    _freeChunk[level].erase( _freeChunk[level].begin() );
                    return memory;
                }
            }

            // if no space is in preallocated memory just allocate as usual memory
            cl_int error;
            cl_mem memory = clCreateBuffer( _context, CL_MEM_READ_WRITE, size, NULL, &error );
            if ( error != CL_SUCCESS )
                throw std::logic_error( "Cannot allocate a memory for OpenCL device" );

            return memory;
        }

        // deallocates a memory by given memory structure pointer
        // if a pointer points on allocated chuck of memory inside the allocator then
        // the allocator just removes a reference to such area without any cost
        // otherwise OpenCL specific function will be called
        void free( cl_mem memory )
        {
            if ( _data != NULL ) {
                std::map<cl_mem, std::pair<size_t, uint8_t>>::iterator pos = _allocatedChunk.find( memory );

                if ( pos != _allocatedChunk.end() ) {
                    _freeChunk[pos->second.second].insert( pos->second.first );
                    _merge( pos->second.first, pos->second.second );
                    _allocatedChunk.erase( pos );
                }
            }

            if ( clReleaseMemObject( memory ) != CL_SUCCESS )
                throw std::logic_error( "Cannot deallocate a memory for OpenCL device" );
        }

        // returns maximum available space which could be allocated by allocator
        size_t availableSize() const
        {
            return _availableSize;
        }

    private:
        cl_context _context;
        cl_uint _minimumSizeChunk;
        cl_mem _data; // a pointer to memory allocated chunk
        const size_t _availableSize; // maximum available memory size

        // a map which holds an information about allocated memory in preallocated memory chunk
        // first paramter is a pointer to allocated memory in OpenCL terms
        // second parameter is an offset from preallocated memory
        // third parameter is a power of 2 (level)
        std::map<cl_mem, std::pair<size_t, uint8_t>> _allocatedChunk;

        // true memory allocation on OpenCL devices
        virtual void _allocate( size_t size )
        {
            if ( size > _availableSize )
                throw std::logic_error( "Memory size to be allocated is bigger than available size on device" );

            if ( _size != size && size > 0 ) {
                if ( !_allocatedChunk.empty() )
                    throw std::logic_error( "Cannot free a memory on OpenCL device. Not all objects were previously deallocated from allocator." );

                _free();

                cl_int error;
                _data = clCreateBuffer( _context, CL_MEM_READ_WRITE, size, NULL, &error );
                if ( error != CL_SUCCESS )
                    throw std::logic_error( "Cannot allocate a memory for OpenCL device" );

                _size = size;
            }
        }

        // true memory deallocation on OpenCL device
        virtual void _deallocate()
        {
            if ( _data != NULL ) {
                cl_int error = clReleaseMemObject( _data );
                if ( error != CL_SUCCESS )
                    throw std::logic_error( "Cannot deallocate a memory for OpenCL device" );
                _data = NULL;
            }

            _allocatedChunk.clear();
        }

        MemoryAllocator( const MemoryAllocator & allocator )
            : BaseMemoryAllocator( allocator )
            , _availableSize( 0 )
        {}
        MemoryAllocator & operator=( const MemoryAllocator & )
        {
            return ( *this );
        }
    };
}
