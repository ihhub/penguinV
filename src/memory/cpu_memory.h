#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include "memory_allocator.h"

namespace cpu_Memory
{
    // Class for memory allocation on CPU
    class MemoryAllocator : public BaseMemoryAllocator
    {
    public:
        MemoryAllocator()
            : _data( nullptr )
        {
        }

        static MemoryAllocator & instance()
        {
            static MemoryAllocator allocator;
            return allocator;
        }

        virtual ~MemoryAllocator()
        {
            _free();
        }

        // Returns a pointer to an allocated memory. If memory size of allocated memory chuck is enough for requested size
        // then return a point from preallocated memory, otherwise allocate heap memory
        template <typename _DataType = uint8_t>
        _DataType* allocate( size_t size = 1 )
        {
            _acquireLock();
            if ( _data != nullptr ) {
                const size_t overallSize = size * sizeof( _DataType );

                if( overallSize < _size ) {
                    const uint8_t level = _getAllocationLevel( overallSize );

                    if( _split( level ) ) {
                        _DataType* address = reinterpret_cast<_DataType*>(static_cast<uint8_t*>(_data) + *_freeChunck[level].begin());
                        _allocatedChunck.insert( std::pair<size_t, uint8_t >( *_freeChunck[level].begin(), level ) );
                        _freeChunck[level].erase( _freeChunck[level].begin() );
                        _releaseLock();
                        return address;
                    }
                }
            }
            _releaseLock();

            // if no space in preallocated memory, allocate as usual memory
            return new _DataType[size];
        }

        // Deallocates a memory by input address. If a pointer points on allocated chuck of memory inside the allocator then
        // the allocator just removes a reference to such area without any cost, otherwise heap allocation
        void free( void * address )
        {
            _acquireLock();
            if( _data != nullptr && address >= _data ) {
                std::map <size_t, uint8_t>::iterator pos = _allocatedChunck.find( static_cast<uint8_t*>(address) - static_cast<uint8_t*>(_data) );

                if( pos != _allocatedChunck.end() ) {
                    _freeChunck[pos->second].insert( pos->first );
                    _merge( pos->first, pos->second );
                    _allocatedChunck.erase( pos );
                    _releaseLock();
                    return;
                }
            }
            _releaseLock();

            delete [] address;
        }
    private:
        uint8_t * _data; // a pointer to memory allocated chunk
        std::recursive_mutex _lock;

        // a map which holds an information about allocated memory in preallocated memory chunck
        // first parameter is an offset from preallocated memory, second parameter is a power of 2 (level)
        std::map <size_t, uint8_t> _allocatedChunck;

        // true memory allocation on CPU
        virtual void _allocate( size_t size )
        {
            _acquireLock();
            if( _size != size && size > 0 ) {
                if( !_allocatedChunck.empty() )
                    throw std::logic_error( "Cannot free a memory on CPU. Not all objects were previously deallocated from allocator." );

                _free();

                _data = new uint8_t [size];
                _size = size;
            }
            _releaseLock();
        }

        // true memory deallocation on CPU
        virtual void _deallocate()
        {
            _acquireLock();
            if( _data != nullptr ) {
                delete [] _data;
                _data = nullptr;
            }

            _allocatedChunck.clear();
            _releaseLock();
        }

        void _acquireLock()
        {
            _lock.lock();
        }

        void _releaseLock()
        {
            _lock.unlock();
        }

        MemoryAllocator(const MemoryAllocator & ) {}
        MemoryAllocator & operator=( const MemoryAllocator & ) { return (*this); }
    };
}
