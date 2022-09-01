#pragma once

#include "memory/cpu_memory.h"
#include "penguinv_exception.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace penguinV
{
    template <typename TColorDepth>
    class ImageTemplate
    {
    public:
        explicit ImageTemplate( uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u )
            : _width( 0 ) // width of image
            , _height( 0 ) // height of image
            , _colorCount( 1 ) // number of colors per pixel
            , _alignment( 1 ) // some formats require that row size must be a multiple of some value (alignment)
                              // for example for Bitmap it must be a multiple of 4
            , _rowSize( 0 ) // size of single row on image, usually it is equal to width
            , _data( nullptr ) // an array what store image information (pixel data)
            , _type( 0 ) // special attribute to specify different types of images based on technology it is used for
        {
            _setType();

            setColorCount( colorCount_ );
            setAlignment( alignment_ );
            resize( width_, height_ );
        }

        ImageTemplate( const ImageTemplate & image )
            : _data( nullptr )
            , _type( image._type )
        {
            copy( image );
        }

        ImageTemplate( ImageTemplate && image )
            : _width( 0 )
            , _height( 0 )
            , _colorCount( 1 )
            , _alignment( 1 )
            , _rowSize( 0 )
            , _data( nullptr )
            , _type( 0 )
        {
            swap( image );
        }

        ImageTemplate & operator=( const ImageTemplate & image )
        {
            copy( image );

            return ( *this );
        }

        ImageTemplate & operator=( ImageTemplate && image )
        {
            swap( image );

            return ( *this );
        }

        bool operator==( const ImageTemplate & image ) const
        {
            return _data == image._data && _width == image._width && _height == image._height && _colorCount == image._colorCount && _alignment == image._alignment
                   && _type == image._type;
        }

        virtual ~ImageTemplate()
        {
            clear();
        }

        void resize( uint32_t width_, uint32_t height_ )
        {
            if ( width_ > 0 && height_ > 0 && ( width_ != _width || height_ != _height ) ) {
                clear();

                _width = width_;
                _height = height_;

                _rowSize = width() * colorCount();
                if ( _rowSize % alignment() != 0 )
                    _rowSize = ( _rowSize / alignment() + 1 ) * alignment();

                _data = _allocate( static_cast<size_t>( _height ) * static_cast<size_t>( _rowSize ) );
            }
        }

        void clear()
        {
            if ( _data != nullptr ) {
                _deallocate( _data );
                _data = nullptr;
            }

            _width = 0;
            _height = 0;
            _rowSize = 0;
        }

        TColorDepth * data()
        {
            return _data;
        }

        const TColorDepth * data() const
        {
            return _data;
        }

        void assign( TColorDepth * data_, uint32_t width_, uint32_t height_, uint8_t colorCount_, uint8_t alignment_ )
        {
            if ( data_ == nullptr || width_ == 0 || height_ == 0 || colorCount_ == 0 || alignment_ == 0 )
                throw penguinVException( "Invalid image assignment parameters" );

            clear();

            _width = width_;
            _height = height_;

            _colorCount = colorCount_;
            _alignment = alignment_;

            _data = data_;

            _rowSize = width() * colorCount();
            if ( _rowSize % alignment() != 0 )
                _rowSize = ( _rowSize / alignment() + 1 ) * alignment();
        }

        bool empty() const
        {
            return _data == nullptr;
        }

        uint32_t width() const
        {
            return _width;
        }

        uint32_t height() const
        {
            return _height;
        }

        uint32_t rowSize() const
        {
            return _rowSize;
        }

        uint8_t colorCount() const
        {
            return _colorCount;
        }

        void setColorCount( uint8_t colorCount_ )
        {
            if ( colorCount_ > 0 && _colorCount != colorCount_ ) {
                clear();
                _colorCount = colorCount_;
            }
        }

        uint8_t alignment() const
        {
            return _alignment;
        }

        void setAlignment( uint8_t alignment_ )
        {
            if ( alignment_ > 0 && alignment_ != _alignment ) {
                clear();
                _alignment = alignment_;
            }
        }

        void fill( TColorDepth value )
        {
            if ( empty() )
                return;

            _set( data(), value, sizeof( TColorDepth ) * height() * rowSize() );
        }

        void swap( ImageTemplate & image )
        {
            std::swap( _width, image._width );
            std::swap( _height, image._height );

            std::swap( _colorCount, image._colorCount );
            std::swap( _rowSize, image._rowSize );
            std::swap( _alignment, image._alignment );

            std::swap( _data, image._data );
            std::swap( _type, image._type );
        }

        void copy( const ImageTemplate & image )
        {
            if ( _type != image._type )
                throw penguinVException( "Cannot copy image of one type to another type." );

            clear();

            _width = image._width;
            _height = image._height;

            _colorCount = image._colorCount;
            _rowSize = image._rowSize;
            _alignment = image._alignment;

            if ( image._data != nullptr ) {
                _data = _allocate( static_cast<size_t>( _height ) * static_cast<size_t>( _rowSize ) );

                _copy( _data, image._data, sizeof( TColorDepth ) * _height * _rowSize );
            }
        }

        bool mutate( uint32_t width_, uint32_t height_, uint8_t colorCount_, uint8_t alignment_ )
        {
            if ( colorCount_ > 0 && alignment_ > 0 ) {
                uint32_t rowSize_ = width_ * colorCount_;
                if ( rowSize_ % alignment_ != 0 )
                    rowSize_ = ( rowSize_ / alignment_ + 1 ) * alignment_;

                if ( rowSize_ * height_ != _rowSize * _height )
                    return false;

                _width = width_;
                _height = height_;
                _colorCount = colorCount_;
                _alignment = alignment_;
                _rowSize = rowSize_;

                return true;
            }

            return false;
        }

        uint8_t type() const
        {
            return _type;
        }

        ImageTemplate generate( uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u ) const
        {
            ImageTemplate image;
            image._type = _type;

            image.setColorCount( colorCount_ );
            image.setAlignment( alignment_ );
            image.resize( width_, height_ );

            return image;
        }

    protected:
        typedef TColorDepth * ( *AllocateFunction )( size_t size );
        typedef void ( *DeallocateFunction )( TColorDepth * data );
        typedef void ( *CopyFunction )( TColorDepth * out, TColorDepth * in, size_t size );
        typedef void ( *SetFunction )( TColorDepth * data, TColorDepth value, size_t size );

        void _setType( uint8_t type = 0u, AllocateFunction allocateFunction = _allocateMemory, DeallocateFunction deallocateFunction = _deallocateMemory,
                       CopyFunction copyFunction = _copyMemory, SetFunction setFunction = _setMemory )
        {
            _type = type;
            FunctionFacade::instance().initialize( _type, allocateFunction, deallocateFunction, copyFunction, setFunction );
        }

    private:
        TColorDepth * _allocate( size_t size ) const
        {
            return FunctionFacade::instance().allocate( _type )( size );
        }

        void _deallocate( TColorDepth * data ) const
        {
            FunctionFacade::instance().deallocate( _type )( data );
        }

        void _copy( TColorDepth * out, TColorDepth * in, size_t size ) const
        {
            FunctionFacade::instance().copy( _type )( out, in, size );
        }

        void _set( TColorDepth * data, TColorDepth value, size_t size ) const
        {
            FunctionFacade::instance().set( _type )( data, value, size );
        }

        static TColorDepth * _allocateMemory( size_t size )
        {
            return cpu_Memory::MemoryAllocator::instance().allocate<TColorDepth>( size );
        }

        static void _deallocateMemory( TColorDepth * data )
        {
            cpu_Memory::MemoryAllocator::instance().free( data );
        }

        static void _copyMemory( TColorDepth * out, TColorDepth * in, size_t size )
        {
            memcpy( out, in, size );
        }

        static void _setMemory( TColorDepth * data, TColorDepth value, size_t size )
        {
            std::fill( data, data + size / sizeof( TColorDepth ), value );
        }

        uint32_t _width;
        uint32_t _height;

        uint8_t _colorCount;
        uint8_t _alignment;
        uint32_t _rowSize;

        TColorDepth * _data;

        uint8_t _type;

        class FunctionFacade
        {
        public:
            static FunctionFacade & instance()
            {
                static FunctionFacade facade;
                return facade;
            }

            void initialize( uint8_t type, AllocateFunction allocateFunction, DeallocateFunction deallocateFunction, CopyFunction copyFunction, SetFunction setFunction )
            {
                if ( _allocate[type] == nullptr ) {
                    _allocate[type] = allocateFunction;
                    _deallocate[type] = deallocateFunction;
                    _copy[type] = copyFunction;
                    _set[type] = setFunction;
                }
            }

            AllocateFunction allocate( uint8_t type ) const
            {
                return _getFunction( _allocate, type );
            }

            DeallocateFunction deallocate( uint8_t type ) const
            {
                return _getFunction( _deallocate, type );
            }

            CopyFunction copy( uint8_t type ) const
            {
                return _getFunction( _copy, type );
            }

            SetFunction set( uint8_t type )
            {
                return _getFunction( _set, type );
            }

        private:
            FunctionFacade()
            {
                _allocate.resize( 256, nullptr );
                _deallocate.resize( 256, nullptr );
                _copy.resize( 256, nullptr );
                _set.resize( 256, nullptr );
            }

            FunctionFacade & operator=( const FunctionFacade & )
            {
                return ( *this );
            }
            FunctionFacade( const FunctionFacade & ) {}

            template <typename TFunction>
            TFunction _getFunction( const std::vector<TFunction> & data, uint8_t index ) const
            {
                if ( data[index] == nullptr )
                    throw penguinVException( "A function is not defined for this type of image" );

                return data[index];
            }

            std::vector<AllocateFunction> _allocate;
            std::vector<DeallocateFunction> _deallocate;
            std::vector<CopyFunction> _copy;
            std::vector<SetFunction> _set;
        };
    };

    typedef ImageTemplate<uint8_t> Image;
    typedef ImageTemplate<uint16_t> Image16Bit;

    const static uint8_t GRAY_SCALE = 1u;
    const static uint8_t RGB = 3u;
    const static uint8_t RGBA = 4u;
    const static uint8_t RED_CHANNEL = 0u;
    const static uint8_t GREEN_CHANNEL = 1u;
    const static uint8_t BLUE_CHANNEL = 2u;
    const static uint8_t ALPHA_CHANNEL = 3u;
}
