#pragma once

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <cstdint>
#include <map>
#include <set>
#include <vector>
#include "opencl_exception.h"

namespace multiCL
{
    // Class for memory allocation on GPU devices
    class MemoryAllocator
    {
    public:
        MemoryAllocator( cl_context context, size_t availableSpace )
            : _context      ( context )
            , _data         ( NULL )
            , _size         ( 0 )
            , _availableSize( availableSpace )
        {
            if( _availableSize == 0 )
                throw openCLException( "Available size cannot be 0" );
        }

        ~MemoryAllocator()
        {
            _free();
        }

        // this function allocates a chunk of memory on GPU device
        // we recommend to call this function only one time at the startup of an application
        // do not reallocate memory if some objects in your source code are allocated
        // through this allocator. Future access to such object's memory is unpredictable
        void reserve( size_t size )
        {
            if( size == 0 )
                throw openCLException( "Memory size cannot be 0" );

            if( size > _availableSize )
                throw openCLException( "Memory size to be allocated is bigger than available size on GPU device" );

            if( _data != NULL && size == _size )
                return;

            _allocate( size );

            size_t usedSize = 0;

            while( size > 0 ) {
                uint8_t levelCount = _getAllocationLevel( size );
                size_t value = static_cast<size_t>(1) << levelCount;

                if( value > size ) {
                    value >>= 1;
                    --levelCount;
                }

                if( usedSize == 0 )
                    _freeChunck.resize( levelCount + 1 );

                _freeChunck[levelCount].insert( usedSize );

                usedSize += value;
                size -= value;
            }
        }

        // this function returns a memory structure pointer to an allocated memory
        // if memory size on allocated chuck of memory is enough for requested size
        // so the function just assigns a pointer to preallocated memory
        // otherwise the function will allocate a new chuck of memory just for this pointer
        template <typename _DataType>
        cl_mem allocate( size_t size = 1 )
        {
            size = size * sizeof( _DataType );

            if( _data != NULL && size < _size ) {
                const uint8_t level = _getAllocationLevel( size );

                if( _split( level ) ) {
                    cl_buffer_region region;
                    region.origin = *(_freeChunck[level].begin());
                    region.size = size;

                    cl_int error;
                    cl_mem memory = clCreateSubBuffer( _data, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &error );
                    if( error != CL_SUCCESS )
                        throw openCLException( "Cannot allocate a subbuffer memory for GPU device" );

                    _allocatedChunck.insert( std::pair< cl_mem, std::pair < size_t, uint8_t > >( memory,  std::pair < size_t, uint8_t >(*(_freeChunck[level].begin()), level) ) );
                    _freeChunck[level].erase( _freeChunck[level].begin() );
                    return memory;
                }
            }

            // if no space in preallocated memory just allocate as usual memory
            cl_int error;
            cl_mem memory = clCreateBuffer( _context, CL_MEM_READ_WRITE, size, NULL, &error);
            if( error != CL_SUCCESS )
                throw openCLException( "Cannot allocate a memory for GPU device" );

            return memory;
        }

        // returns true if memory allocator has enough space for specified size in bytes
        bool isSpaceAvailable( size_t size = 1 ) const
        {
            if( _data != NULL && size < _size ) {
                const uint8_t level = _getAllocationLevel( size );

                for( uint8_t i = level; i < _freeChunck.size(); ++i ) {
                    if( !_freeChunck[i].empty() )
                        return true;
                }
            }

            return false;
        }

        // deallocates a memory by given memory structure pointer
        // if a pointer points on allocated chuck of memory inside the allocator then
        // the allocator just removes a reference to such area without any cost
        // otherwise OpenCL specific function will be called
        void free( cl_mem memory )
        {
            if( _data != NULL ) {
                std::map < cl_mem, std::pair < size_t, uint8_t > >::iterator pos = _allocatedChunck.find( memory );

                if( pos != _allocatedChunck.end() ) {
                    _freeChunck[pos->second.second].insert( pos->second.first );
                    _merge( pos->second.first, pos->second.second );
                    _allocatedChunck.erase( pos );
                    return;
                }
            }

            if( clReleaseMemObject( memory ) != CL_SUCCESS )
                throw openCLException( "Cannot deallocate a memory for GPU device" );
        }

        // this function returns maximum availbale space which could be allocated by allocator
        size_t availableSize() const
        {
            return _availableSize;
        }
    private:
        cl_context _context;
        cl_mem _data; // a pointer to memory allocated chunk
        size_t _size; // a size of memory allocated chunk

        // an array which holds an information about free memory in preallocated memory chunck
        std::vector < std::set < size_t > > _freeChunck;
        // a map which holds an information about allocated memory in preallocated memory chunck
        // first paramter is a pointer to allocated memory in OpenCL terms
        // second parameter is an offset from preallocated memory
        // third parameter is a power of 2 (level)
        std::map < cl_mem, std::pair < size_t, uint8_t > > _allocatedChunck;

        size_t _availableSize; // maximum available memory size on GPU device

        // the function for true memory allocation on GPU devices
        void _allocate( size_t size )
        {
            if( _size != size && size > 0 ) {
                if( !_allocatedChunck.empty() )
                    throw openCLException( "Cannot free a memory on GPU device. Not all objects were previously deallocated from allocator." );

                _free();

                cl_int error;
                _data = clCreateBuffer( _context, CL_MEM_READ_WRITE, size, NULL, &error);
                if( error != CL_SUCCESS )
                    throw openCLException( "Cannot allocate a memory for GPU device" );

                _size = size;
            }
        }

        // the function for true memory deallocation on GPU device
        void _free()
        {
            if( _data != NULL ) {
                if( clReleaseMemObject( _data ) != CL_SUCCESS)
                    throw openCLException( "Cannot deallocate a memory for GPU device" );

                _data = NULL;
            }

            _freeChunck.clear();
            _allocatedChunck.clear();

            _size = 0;
        }

        // returns a level (power of 2) needed for a required size
        static uint8_t _getAllocationLevel( size_t initialSize )
        {
            size_t size = 1;
            uint8_t level = 0;

            while( size < initialSize ) {
                size <<= 1;
                ++level;
            }

            return level;
        }

        // split the preallocated memory by levels
        bool _split( uint8_t from )
        {
            bool levelFound = false;
            uint8_t startLevel = from;

            for( uint8_t i = from; i < _freeChunck.size(); ++i ) {
                if( !_freeChunck[i].empty() ) {
                    startLevel = i;
                    levelFound = true;
                    break;
                }
            }

            if( !levelFound )
                return false;

            if( startLevel > from ) {
                size_t memorySize = static_cast<size_t>(1) << (startLevel - 1);

                for( ; startLevel > from; --startLevel, memorySize >>= 1 ) {
                    _freeChunck[startLevel - 1].insert( *_freeChunck[startLevel].begin() );
                    _freeChunck[startLevel - 1].insert( *_freeChunck[startLevel].begin() + memorySize );
                    _freeChunck[startLevel].erase( _freeChunck[startLevel].begin() );
                }
            }

            return true;
        }

        // merge preallocated memory by levels
        void _merge( size_t offset, uint8_t from )
        {
            size_t memorySize = static_cast<size_t>(1) << from;

            for( std::vector < std::set < size_t > >::iterator level = _freeChunck.begin() + from; level < _freeChunck.end();
                 ++level, memorySize <<= 1 ) {
                std::set< size_t >::iterator pos = level->find( offset );
                std::set< size_t >::iterator neighbour = pos;
                ++neighbour;

                if( neighbour != level->end() ) {
                    if( *(neighbour) - *(pos) == memorySize ) {
                        offset = *pos;
                        (level + 1)->insert( offset );
                        level->erase( pos, ++neighbour );
                        continue;
                    }
                }

                if( pos != level->begin() ) {
                    neighbour = pos;
                    --neighbour;

                    if( *(pos) - *(neighbour) == memorySize ) {
                        offset = *neighbour;
                        (level + 1)->insert( offset );
                        level->erase( neighbour, ++pos );
                        continue;
                    }
                }

                return;
            }
        }

        MemoryAllocator(const MemoryAllocator & ) {}
        MemoryAllocator & operator=( const MemoryAllocator & ) { return (*this); }
    };
}
