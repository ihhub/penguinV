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
#include "../memory/memory_allocator.h"
#include "../image_exception.h"

namespace multiCL
{
    // Class for memory allocation on GPU devices
    class MemoryAllocator : public BaseMemoryAllocator
    {
    public:
        MemoryAllocator( cl_context context, size_t availableSpace )
            : BaseMemoryAllocator( availableSpace )
            , _context      ( context )
            , _data         ( NULL )
        {
        }

        virtual ~MemoryAllocator()
        {
            _free();
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
                        throw imageException( "Cannot allocate a subbuffer memory for GPU device" );

                    _allocatedChunck.insert( std::pair< cl_mem, std::pair < size_t, uint8_t > >( memory,  std::pair < size_t, uint8_t >(*(_freeChunck[level].begin()), level) ) );
                    _freeChunck[level].erase( _freeChunck[level].begin() );
                    return memory;
                }
            }

            // if no space in preallocated memory just allocate as usual memory
            cl_int error;
            cl_mem memory = clCreateBuffer( _context, CL_MEM_READ_WRITE, size, NULL, &error);
            if( error != CL_SUCCESS )
                throw imageException( "Cannot allocate a memory for GPU device" );

            return memory;
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
                }
            }

            if( clReleaseMemObject( memory ) != CL_SUCCESS )
                throw imageException( "Cannot deallocate a memory for GPU device" );
        }
    private:
        cl_context _context;
        cl_mem _data; // a pointer to memory allocated chunk

        // a map which holds an information about allocated memory in preallocated memory chunck
        // first paramter is a pointer to allocated memory in OpenCL terms
        // second parameter is an offset from preallocated memory
        // third parameter is a power of 2 (level)
        std::map < cl_mem, std::pair < size_t, uint8_t > > _allocatedChunck;

        // the function for true memory allocation on GPU devices
        virtual void _allocate( size_t size )
        {
            if( _size != size && size > 0 ) {
                if( !_allocatedChunck.empty() )
                    throw imageException( "Cannot free a memory on GPU device. Not all objects were previously deallocated from allocator." );

                _free();

                cl_int error;
                _data = clCreateBuffer( _context, CL_MEM_READ_WRITE, size, NULL, &error);
                if( error != CL_SUCCESS )
                    throw imageException( "Cannot allocate a memory for GPU device" );

                _size = size;
            }
        }

        // the function for true memory deallocation on GPU device
        virtual void _deallocate()
        {
            if( _data != NULL ) {
                cl_int error = clReleaseMemObject( _data );
                if( error != CL_SUCCESS)
                    throw imageException( "Cannot deallocate a memory for GPU device" );
                _data = NULL;
            }

            _allocatedChunck.clear();
        }

        MemoryAllocator(const MemoryAllocator & ) {}
        MemoryAllocator & operator=( const MemoryAllocator & ) { return (*this); }
    };
}
