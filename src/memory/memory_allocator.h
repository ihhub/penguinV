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

#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

// Base class for memory allocation
class BaseMemoryAllocator
{
public:
    BaseMemoryAllocator() = default;

    virtual ~BaseMemoryAllocator() {}

    // Allocates a chunk of memory. We recommend to call this function only one time at the startup of an application.
    // Do not reallocate memory if some objects in your source code are allocated through this allocator.
    void reserve( size_t size )
    {
        if ( size == 0 ) {
            throw std::logic_error( "Memory size cannot be 0" );
        }

        if ( size == _size ) {
            return;
        }

        _allocate( size );

        size_t usedSize = 0;

        while ( size > 0 ) {
            uint8_t levelCount = _getAllocationLevel( size );
            size_t value = static_cast<size_t>( 1 ) << levelCount;

            if ( value > size ) {
                value >>= 1;
                --levelCount;
            }

            if ( usedSize == 0 ) {
                _freeChunk.resize( levelCount + 1u );
            }

            _freeChunk[levelCount].insert( usedSize );

            usedSize += value;
            size -= value;
        }
    }

protected:
    void _free()
    {
        _deallocate();

        _freeChunk.clear();
        _size = 0;
    }

    // Returns a level (power of 2) needed for a required size.
    static uint8_t _getAllocationLevel( size_t initialSize )
    {
        size_t size = 1;
        uint8_t level = 0;

        while ( size < initialSize ) {
            size <<= 1;
            ++level;
        }

        return level;
    }

    // Splits the preallocated memory by levels.
    bool _split( uint8_t from )
    {
        bool levelFound = false;
        uint8_t startLevel = from;

        for ( uint8_t i = from; i < _freeChunk.size(); ++i ) {
            if ( !_freeChunk[i].empty() ) {
                startLevel = i;
                levelFound = true;
                break;
            }
        }

        if ( !levelFound ) {
            return false;
        }

        if ( startLevel > from ) {
            size_t memorySize = static_cast<size_t>( 1 ) << ( startLevel - 1 );

            for ( ; startLevel > from; --startLevel, memorySize >>= 1 ) {
                const size_t previousLevelValue = *_freeChunk[startLevel].begin();
                _freeChunk[startLevel - 1u].insert( previousLevelValue );
                _freeChunk[startLevel - 1u].insert( previousLevelValue + memorySize );
                _freeChunk[startLevel].erase( _freeChunk[startLevel].begin() );
            }
        }

        return true;
    }

    // merges preallocated memory by levels
    void _merge( size_t offset, uint8_t from )
    {
        size_t memorySize = static_cast<size_t>( 1 ) << from;

        for ( std::vector<std::set<size_t>>::iterator level = _freeChunk.begin() + from; level < _freeChunk.end(); ++level, memorySize <<= 1 ) {
            std::set<size_t>::iterator pos = level->find( offset );
            std::set<size_t>::iterator neighbour = pos;
            ++neighbour;

            if ( neighbour != level->end() ) {
                if ( *( neighbour ) - *( pos ) == memorySize ) {
                    offset = *pos;
                    ( level + 1 )->insert( offset );
                    level->erase( pos, ++neighbour );
                    continue;
                }
            }

            if ( pos != level->begin() ) {
                neighbour = pos;
                --neighbour;

                if ( *( pos ) - *( neighbour ) == memorySize ) {
                    offset = *neighbour;
                    ( level + 1 )->insert( offset );
                    level->erase( neighbour, ++pos );
                    continue;
                }
            }

            return;
        }
    }

    // A size of memory allocated chunk.
    size_t _size{ 0 }; 

    // Free memory in preallocated memory.
    std::vector<std::set<size_t>> _freeChunk;

private:
    // True memory allocation method. Implementation details are in child classes.
    virtual void _allocate( size_t size ) = 0;

    // True memory deallocation method. Implementation details are in child classes.
    virtual void _deallocate() = 0;
};
