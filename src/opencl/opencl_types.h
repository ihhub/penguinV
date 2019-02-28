#pragma once

#include <cstdint>
#include <vector>
#include "opencl_device.h"
#include "../image_exception.h"

namespace multiCL
{
    // This namespace contains template classes to simplify development on OpenCL

    // A class which contains a single value of specific type
    template <typename TData>
    class Type
    {
    public:
        Type()
            : _data( NULL )
        {
            _allocate();
        }

        Type( const TData & in )
            : _data( NULL )
        {
            _allocate();
            _copyFrom( in );
        }

        Type( Type && in )
            : _data( NULL )
        {
            _swap( in );
        }

        ~Type()
        {
            _free();
        }

        Type & operator=( const Type & in )
        {
            _copy( in );

            return (*this);
        }

        Type & operator=( Type && in )
        {
            _swap( in );

            return (*this);
        }

        Type & operator=( const TData & in )
        {
            _copyFrom( in );

            return (*this);
        }

        cl_mem data()
        {
            return _data;
        }

        cl_mem data() const
        {
            return _data;
        }

        // Use this function if you want to retrieve a value from device to host
        TData get() const
        {
            return _copyTo();
        }
    private:
        cl_mem _data;

        void _free()
        {
            if( _data != NULL ) {
                MemoryManager::memory().free( _data );
                _data = NULL;
            }
        }

        void _allocate()
        {
            _free();

            _data = MemoryManager::memory().allocate<TData>();
        }

        void _copy( const Type & )
        {
            throw imageException( "Memory copy in OpenCL device is not implemented" );
        }

        void _swap( Type & in )
        {
            std::swap( _data, in._data );
        }

        void _copyFrom( const TData & in )
        {
            if( _data != NULL ) {
                cl_int error = clEnqueueWriteBuffer( OpenCLDeviceManager::instance().device().queue()(), _data, CL_TRUE, 0, sizeof( TData ), &in, 0, NULL, NULL );
                if( error != CL_SUCCESS )
                    throw imageException( "Cannot copy a memory into OpenCL device" );
            }
            else {
                throw imageException( "Memory in OpenCL device is not allocated" );
            }
        }

        TData _copyTo() const
        {
            TData out;

            if( _data != NULL ) {
                cl_int error = clEnqueueReadBuffer( OpenCLDeviceManager::instance().device().queue()(), _data, CL_TRUE, 0, sizeof( TData ), &out, 0, NULL, NULL );
                if( error != CL_SUCCESS )
                    throw imageException( "Cannot copy a memory from OpenCL device" );
            }
            else {
                throw imageException( "Memory in OpenCL device is not allocated" );
            }

            return out;
        }

        Type( const Type & ) // copy constructor is disabled to avoid a situation of assigning this type as a kernel argument
        {}
    };

    // A class which contains an array of values of specific type
    template <typename TData>
    class Array
    {
    public:
        Array()
            : _data( NULL )
            , _size( 0 )
        {}

        Array( const std::vector <TData> & data )
            : _data( NULL )
            , _size( 0 )
        {
            _allocate( data.size() );
            _copyFrom( data );
        }

        Array( size_t size )
            : _data( NULL )
            , _size( 0 )
        {
            _allocate( size );
        }

        Array ( Array && in )
            : _data( NULL )
            , _size( 0 )
        {
            _swap( in );
        }

        ~Array()
        {
            _free();
        }

        Array & operator=( const Array & in )
        {
            _copy( in );

            return (*this);
        }

        Array & operator=( Array && in )
        {
            _swap( in );

            return (*this);
        }

        Array & operator=( const std::vector <TData> & data )
        {
            _allocate( data.size() );
            _copyFrom( data );

            return (*this);
        }

        cl_mem data()
        {
            return _data;
        }

        cl_mem data() const
        {
            return _data;
        }

        // Use this function if you want to retrieve a value from device to host
        std::vector <TData> get() const
        {
            return _copyTo();
        }

        size_t size() const
        {
            return _size;
        }

        bool empty() const
        {
            return _data == NULL;
        }

        void resize( size_t size )
        {
            _allocate( size );
        }
    private:
        cl_mem _data;
        size_t _size;

        void _free()
        {
            if( _data != NULL ) {
                MemoryManager::memory().free( _data );
                _data = NULL;
            }
        }

        void _allocate( size_t size )
        {
            if( _size != size ) {
                _free();

                if( size != 0 )
                    _data = MemoryManager::memory().allocate<TData>( size );

                _size = size;
            }
        }

        void _copy( const Array & )
        {
            throw imageException( "Memory copy in OpenCL device is not implemented" );
        }

        void _swap( Array & in )
        {
            std::swap( _data, in._data );
            std::swap( _size, in._size );
        }

        void _copyFrom( const std::vector <TData> & data )
        {
            if( _data != NULL && _size == data.size() ) {
                cl_int error = clEnqueueWriteBuffer( OpenCLDeviceManager::instance().device().queue()(), _data, CL_TRUE, 0, _size * sizeof( TData ), data.data(), 0, NULL, NULL );
                if( error != CL_SUCCESS )
                    throw imageException( "Cannot copy a memory into OpenCL device" );
            }
        }

        std::vector <TData> _copyTo() const
        {
            std::vector <TData> out( _size );

            if( _data != NULL ) {
                cl_int error = clEnqueueReadBuffer( OpenCLDeviceManager::instance().device().queue()(), _data, CL_TRUE, 0, _size * sizeof( TData ), out.data(), 0, NULL, NULL );
                if( error != CL_SUCCESS )
                    throw imageException( "Cannot copy a memory from OpenCL device" );
            }

            return out;
        }

        Array ( const Array & ) // copy constructor is disabled to avoid a situation of assigning this type as a kernel argument
        {}
    };
}
