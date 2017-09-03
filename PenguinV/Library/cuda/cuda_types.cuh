#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>
#include "../image_exception.h"
#include "cuda_device.cuh"

namespace Cuda_Types
{
    // This namespace contains template classes to simplify development on CUDA

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

        Type( const Type & in )
            : _data( NULL )
        {
            _copy( in );
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

        TData * data()
        {
            return _data;
        }

        const TData * data() const
        {
            return _data;
        }

        // Use this function if you want to retrieve a value from device to host
        TData get() const
        {
            return _copyTo();
        }
    private:
        TData * _data;

        void _free()
        {
            if( _data != NULL ) {
                Cuda::MemoryManager::memory().free( _data );
                _data = NULL;
            }
        }

        void _allocate()
        {
            _free();

            Cuda::MemoryManager::memory().allocate( &_data );
        }

        void _copy( const Type & in )
        {
            if( _data != NULL && in._data != NULL ) {
                cudaError_t error = cudaMemcpy( _data, in._data, sizeof( TData ), cudaMemcpyDeviceToDevice );
                if( error != cudaSuccess )
                    throw imageException( "Cannot copy a memory in CUDA device" );
            }
            else {
                throw imageException( "Memory in CUDA device is not allocated" );
            }
        }

        void _swap( Type & in )
        {
            std::swap( _data, in._data );
        }

        void _copyFrom( const TData & in )
        {
            if( _data != NULL ) {
                cudaError_t error = cudaMemcpy( _data, &in, sizeof( TData ), cudaMemcpyHostToDevice );
                if( error != cudaSuccess )
                    throw imageException( "Cannot copy a memory in CUDA device" );
            }
            else {
                throw imageException( "Memory in CUDA device is not allocated" );
            }
        }

        TData _copyTo() const
        {
            TData out;

            if( _data != NULL ) {
                cudaError_t error = cudaMemcpy( &out, _data, sizeof( TData ), cudaMemcpyDeviceToHost );
                if( error != cudaSuccess )
                    throw imageException( "Cannot copy a memory in CUDA device" );
            }
            else {
                throw imageException( "Memory in CUDA device is not allocated" );
            }

            return out;
        }
    };

    typedef Type <uint8_t > _cbool;
    typedef Type <uint8_t > _cuint8_t;
    typedef Type <uint16_t> _cuint16_t;
    typedef Type <uint32_t> _cuint32_t;
    typedef Type <double  > _cdouble;
    typedef Type <float   > _cfloat;

    // A class which contains an array of values of specific type
    template <typename TData>
    class Array
    {
    public:
        Array()
            : _data( NULL )
            , _size( 0 )
        {
        }

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

        Array ( const Array & in )
            : _data( NULL )
            , _size( 0 )
        {
            _copy( in );
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

        TData * data()
        {
            return _data;
        }

        const TData * data() const
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
        TData * _data;
        size_t _size;

        void _free()
        {
            if( _data != NULL ) {
                Cuda::MemoryManager::memory().free( _data );
                _data = NULL;
            }
        }

        void _allocate( size_t size )
        {
            if( _size != size ) {
                _free();

                if( size != 0 )
                    Cuda::MemoryManager::memory().allocate( &_data, size );

                _size = size;
            }
        }

        void _copy( const Array & in )
        {
            _allocate( in._size );

            if( in._data != NULL ) {
                cudaError_t error = cudaMemcpy( _data, in._data, _size * sizeof( TData ), cudaMemcpyDeviceToDevice );
                if( error != cudaSuccess )
                    throw imageException( "Cannot copy a memory in CUDA device" );
            }
            else {
                throw imageException( "Memory in CUDA device is not allocated" );
            }
        }

        void _swap( Array & in )
        {
            std::swap( _data, in._data );
            std::swap( _size, in._size );
        }

        void _copyFrom( const std::vector <TData> & data )
        {
            if( _data != NULL && _size == data.size() ) {
                cudaError_t error = cudaMemcpy( _data, data.data(), _size * sizeof( TData ), cudaMemcpyHostToDevice );
                if( error != cudaSuccess )
                    throw imageException( "Cannot copy a memory in CUDA device" );
            }
        }

        std::vector <TData> _copyTo() const
        {
            std::vector <TData> out( _size );

            if( _data != NULL ) {
                cudaError_t error = cudaMemcpy( out.data(), _data, _size * sizeof( TData ), cudaMemcpyDeviceToHost );
                if( error != cudaSuccess )
                    throw imageException( "Cannot copy a memory in CUDA device" );
            }

            return out;
        }
    };
};
