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

#include "memory_allocator.h"
#include <map>
#include <mutex>

namespace cpu_Memory
{
    // Class for memory allocation on CPU
    class MemoryAllocator : public BaseMemoryAllocator
    {
    public:
        MemoryAllocator()
            : _data( nullptr )
            , _alignedData( nullptr )
        {}

        static MemoryAllocator & instance()
        {
            static MemoryAllocator allocator;
            return allocator;
        }

        virtual ~MemoryAllocator()
        {
            _lock.lock();
            _free();
            _lock.unlock();
        }

        // Returns a pointer to an allocated memory. If memory size of allocated memory chuck is enough for requested size
        // then return a point from preallocated memory, otherwise allocate heap memory
        template <typename _DataType = uint8_t>
        _DataType * allocate( size_t size = 1 )
        {
            _lock.lock();
            if ( _data != nullptr ) {
                const size_t overallSize = size * sizeof( _DataType );

                if ( overallSize < _size ) {
                    const uint8_t level = _getAllocationLevel( overallSize );

                    if ( _split( level ) ) {
                        std::set<size_t>::iterator chunk = _freeChunk[level].begin();
                        _DataType * address = reinterpret_cast<_DataType *>( _alignedData + *chunk );
                        _allocatedChunk.insert( std::pair<size_t, uint8_t>( *chunk, level ) );
                        _freeChunk[level].erase( chunk );
                        _lock.unlock();
                        return address;
                    }
                }
            }
            _lock.unlock();

            // if no space in preallocated memory, allocate as usual memory
            return new _DataType[size];
        }

        // Deallocates a memory by input address. If a pointer points on allocated chuck of memory inside the allocator then
        // the allocator just removes a reference to such area without any cost, otherwise heap allocation
        template <typename _DataType>
        void free( _DataType * address )
        {
            _lock.lock();
            if ( _data != nullptr && reinterpret_cast<uint8_t *>( address ) >= _alignedData ) {
                std::map<size_t, uint8_t>::iterator pos = _allocatedChunk.find( static_cast<size_t>( reinterpret_cast<uint8_t *>( address ) - _alignedData ) );

                if ( pos != _allocatedChunk.end() ) {
                    _freeChunk[pos->second].insert( pos->first );
                    _merge( pos->first, pos->second );
                    _allocatedChunk.erase( pos );
                    _lock.unlock();
                    return;
                }
            }
            _lock.unlock();

            delete[] address;
        }

    private:
        uint8_t * _data; // a pointer to memory allocated chunk
        uint8_t * _alignedData; // aligned pointer for SIMD access
        std::mutex _lock;

        // a map which holds an information about allocated memory in preallocated memory chunk
        // first parameter is an offset from preallocated memory, second parameter is a power of 2 (level)
        std::map<size_t, uint8_t> _allocatedChunk;

        // true memory allocation on CPU
        virtual void _allocate( size_t size )
        {
            _lock.lock();
            if ( _size != size && size > 0 ) {
                if ( !_allocatedChunk.empty() )
                    throw std::logic_error( "Cannot free a memory on CPU. Not all objects were previously deallocated from allocator." );

                _free();

                const size_t alignment = 32u; // AVX alignment requirement
                _data = new uint8_t[size + alignment];
                const std::uintptr_t dataAddress = reinterpret_cast<std::uintptr_t>( _data );
                _alignedData = ( ( dataAddress % alignment ) == 0 ) ? _data : _data + ( alignment - ( dataAddress % alignment ) );

                _size = size;
            }
            _lock.unlock();
        }

        // true memory deallocation on CPU
        virtual void _deallocate()
        {
            if ( _data != nullptr ) {
                delete[] _data;
                _data = nullptr;
                _alignedData = nullptr;
            }

            _allocatedChunk.clear();
        }

        MemoryAllocator( const MemoryAllocator & ) {}
        MemoryAllocator & operator=( const MemoryAllocator & )
        {
            return ( *this );
        }
    };
}
