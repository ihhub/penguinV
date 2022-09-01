#pragma once

#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

// Base class for memory allocation
class BaseMemoryAllocator
{
public:
    BaseMemoryAllocator()
        : _size( 0 )
    {}

    virtual ~BaseMemoryAllocator() {}

    // Allocates a chunk of memory. We recommend to call this function only one time at the startup of an application.
    // Do not reallocate memory if some objects in your source code are allocated through this allocator.
    void reserve( size_t size )
    {
        if ( size == 0 )
            throw std::logic_error( "Memory size cannot be 0" );

        if ( size == _size )
            return;

        _allocate( size );

        size_t usedSize = 0;

        while ( size > 0 ) {
            uint8_t levelCount = _getAllocationLevel( size );
            size_t value = static_cast<size_t>( 1 ) << levelCount;

            if ( value > size ) {
                value >>= 1;
                --levelCount;
            }

            if ( usedSize == 0 )
                _freeChunk.resize( levelCount + 1u );

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

    // returns a level (power of 2) needed for a required size
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

    // splits the preallocated memory by levels
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

        if ( !levelFound )
            return false;

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

    size_t _size; // a size of memory allocated chunk
    std::vector<std::set<size_t>> _freeChunk; // free memory in preallocated memory

private:
    virtual void _allocate( size_t size ) = 0; // true memory allocation
    virtual void _deallocate() = 0; // true memory deallocation
};
