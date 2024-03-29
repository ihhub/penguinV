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
#include <map>
#include "../memory/memory_allocator.h"

namespace multiCuda
{
    // Class for memory allocation on devices with CUDA support
    class MemoryAllocator : public BaseMemoryAllocator
    {
    public:
        explicit MemoryAllocator( size_t availableSpace )
            : _data         ( nullptr )
            , _availableSize( availableSpace )
        {
            if ( _availableSize == 0 )
                throw std::logic_error( "Available size cannot be 0" );
        }

        virtual ~MemoryAllocator()
        {
            _free();
        }

        // Returns a pointer to an allocated memory. If memory size of allocated memory chuck of memory is enough for requested size
        // so assign a pointer to preallocated memory, otherwise allocate a new chuck of memory just for the pointer
        template <typename _DataType = uint8_t>
        _DataType* allocate( size_t size = 1 )
        {
            size = size * sizeof( _DataType );

            if ( _data != nullptr && size < _size ) {
                const uint8_t level = _getAllocationLevel( size );

                if ( _split( level ) ) {
                    _DataType* address = reinterpret_cast<_DataType*>(static_cast<uint8_t*>(_data) + *_freeChunk[level].begin());
                    _allocatedChunk.insert( std::pair<size_t, uint8_t >( *_freeChunk[level].begin(), level ) );
                    _freeChunk[level].erase( _freeChunk[level].begin() );
                    return address;
                }
            }

            // if no space in preallocated memory just allocate as usual memory
            _DataType* address = nullptr;
            cudaError_t error = cudaMalloc( &address, size );
            if ( error != cudaSuccess )
                throw std::logic_error( "Cannot allocate a memory for CUDA device" );

            return address;
        }

        // deallocates a memory by input address
        // if a pointer points on allocated chuck of memory inside the allocator then
        // the allocator just removes a reference to such area without any cost
        // otherwise CUDA specific function will be called
        void free( void * address )
        {
            if ( _data != nullptr && address >= _data ) {
                std::map <size_t, uint8_t>::iterator pos =
                    _allocatedChunk.find( static_cast<uint8_t*>(address) - static_cast<uint8_t*>(_data) );

                if ( pos != _allocatedChunk.end() ) {
                    _freeChunk[pos->second].insert( pos->first );
                    _merge( pos->first, pos->second );
                    _allocatedChunk.erase( pos );
                    return;
                }
            }

            cudaError_t error = cudaFree( address );
            if ( error != cudaSuccess )
                throw std::logic_error( "Cannot deallocate memory for CUDA device" );
        }

        // returns maximum available space which could be allocated by allocator
        size_t availableSize() const
        {
            return _availableSize;
        }
    private:
        void * _data; // a pointer to memory allocated chunk
        const size_t _availableSize; // maximum available memory size

        // a map which holds an information about allocated memory in preallocated memory chunk
        // first parameter is an offset from preallocated memory, second parameter is a power of 2 (level)
        std::map <size_t, uint8_t> _allocatedChunk;

        // true memory allocation on devices with CUDA support
        virtual void _allocate( size_t size )
        {
            if ( size > _availableSize )
                throw std::logic_error( "Memory size to be allocated is bigger than available size on device" );

            if ( _size != size && size > 0 ) {
                if ( !_allocatedChunk.empty() )
                    throw std::logic_error( "Cannot free a memory on device with CUDA support. Not all objects were previously deallocated from allocator." );

                _free();

                cudaError_t error = cudaMalloc( &_data, size );
                if ( error != cudaSuccess )
                    throw std::logic_error( "Cannot allocate a memory for CUDA device" );

                _size = size;
            }
        }

        // true memory deallocation on devices with CUDA support
        virtual void _deallocate()
        {
            if ( _data != nullptr ) {
                cudaError_t error = cudaFree( _data );
                if ( error != cudaSuccess )
                    throw std::logic_error( "Cannot deallocate memory for CUDA device" );
                _data = nullptr;
            }

            _allocatedChunk.clear();
        }

        MemoryAllocator( const MemoryAllocator & )
            : _availableSize( 0 )
        {
        }

        MemoryAllocator & operator=( const MemoryAllocator & ) { return (*this); }
    };
}
