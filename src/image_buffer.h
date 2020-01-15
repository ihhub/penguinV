#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include "image_exception.h"
#include "memory/cpu_memory.h"

namespace penguinV
{
    class Image
    {
    public:
        explicit Image( uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u )
            : _width     ( 0 )       // width of image
            , _height    ( 0 )       // height of image
            , _colorCount( 1 )       // number of colors per pixel
            , _alignment ( 1 )       // some formats require that row size must be a multiple of some value (alignment)
                                     // for example for Bitmap it must be a multiple of 4
            , _rowSize   ( 0 )       // size of single row on image, usually it is equal to width
            , _data      ( nullptr ) // an array what store image information (pixel data)
            , _type      ( 0 )       // special attribute to specify different types of images based on technology it is used for
            , _dataType( typeid( uint8_t ).name() )
            , _dataSize( sizeof( uint8_t ) )
        {
            _setType<uint8_t>();

            setColorCount( colorCount_ );
            setAlignment( alignment_ );
            resize( width_, height_ );
        }

        Image( const Image & image )
            : _data    ( nullptr )
            , _type    ( image._type )
            , _dataType( typeid( uint8_t ).name() )
            , _dataSize( sizeof( uint8_t ) )
        {
            copy( image );
        }

        Image( Image && image )
            : _width     ( 0 )
            , _height    ( 0 )
            , _colorCount( 1 )
            , _alignment ( 1 )
            , _rowSize   ( 0 )
            , _data      ( nullptr )
            , _type      ( 0 )
            , _dataType( typeid( uint8_t ).name() )
            , _dataSize( sizeof( uint8_t ) )
        {
            swap( image );
        }

        Image & operator=( const Image & image )
        {
            copy( image );

            return (*this);
        }

        Image & operator=( Image && image )
        {
            swap( image );

            return (*this);
        }

        bool operator==( const Image & image ) const
        {
            return _data == image._data && _width == image._width && _height == image._height && _colorCount == image._colorCount &&
                   _alignment == image._alignment && _type == image._type && _dataType == image._dataType && _dataSize == image._dataSize;
        }

        virtual ~Image()
        {
            clear();
        }

        void resize( uint32_t width_, uint32_t height_ )
        {
            if( width_ > 0 && height_ > 0 && (width_ != _width || height_ != _height) ) {
                clear();

                _width  = width_;
                _height = height_;

                _rowSize = width() * colorCount();
                if( _rowSize % alignment() != 0 )
                    _rowSize = (_rowSize / alignment() + 1) * alignment();

                _data = _allocate<uint8_t>( _height * _rowSize * _dataSize );
            }
        }

        void clear()
        {
            if( _data != nullptr ) {
                _deallocate( _data );
                _data = nullptr;
            }

            _width   = 0;
            _height  = 0;
            _rowSize = 0;
        }

        template <typename TColorDepth = uint8_t>
        TColorDepth * data()
        {
            return reinterpret_cast<TColorDepth *>( _data );
        }

        template <typename TColorDepth = uint8_t>
        const TColorDepth * data() const
        {
            return reinterpret_cast<TColorDepth *>( _data );
        }

        template <typename TColorDepth>
        void assign( TColorDepth * data_, uint32_t width_, uint32_t height_, uint8_t colorCount_, uint8_t alignment_ )
        {
            if( data_ == nullptr || width_ == 0 || height_ == 0 || colorCount_ == 0 || alignment_ == 0 )
                throw imageException( "Invalid image assignment parameters" );

            clear();

            _width  = width_;
            _height = height_;

            _colorCount = colorCount_;
            _alignment = alignment_;

            _data = data_;

            _dataType = typeid( TColorDepth ).name();
            _dataSize = sizeof( TColorDepth );

            _rowSize = width() * colorCount();
            if( _rowSize % alignment() != 0 )
                _rowSize = (_rowSize / alignment() + 1) * alignment();
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
            if( colorCount_ > 0 && _colorCount != colorCount_ ) {
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
            if( alignment_ > 0 && alignment_ != _alignment ) {
                clear();
                _alignment = alignment_;
            }
        }

        template <typename TColorDepth = uint8_t>
        void fill( TColorDepth value )
        {
            if( empty() )
                return;

            if ( typeid( TColorDepth ).name() != _dataType )
                throw imageException( "Image data type is different compare to fill value type" );

            _setType<TColorDepth>();
            _set<TColorDepth>( reinterpret_cast<TColorDepth *>( data() ), value, sizeof( TColorDepth ) * height() * rowSize() );
        }

        void swap( Image & image )
        {
            std::swap( _width, image._width );
            std::swap( _height, image._height );

            std::swap( _colorCount, image._colorCount );
            std::swap( _rowSize   , image._rowSize );
            std::swap( _alignment , image._alignment );

            std::swap( _data, image._data );
            std::swap( _type, image._type );
            std::swap( _dataType, image._dataType );
            std::swap( _dataSize, image._dataSize );
        }

        void copy( const Image & image )
        {
            if( _type != image._type )
                throw imageException( "Cannot copy image of one type to another type." );

            clear();

            _width  = image._width;
            _height = image._height;

            _colorCount = image._colorCount;
            _rowSize    = image._rowSize;
            _alignment  = image._alignment;
            _dataType   = image._dataType;
            _dataSize   = image._dataSize;

            if( image._data != nullptr ) {
                _data = _allocate<uint8_t>( _height * _rowSize * _dataSize );

                _copy<uint8_t>( _data, image._data, _dataSize * _height * _rowSize );
            }
        }

        bool mutate( uint32_t width_, uint32_t height_, uint8_t colorCount_, uint8_t alignment_ )
        {
            if( colorCount_ > 0 && alignment_ > 0 )
            {
                uint32_t rowSize_ = width_ * colorCount_;
                if( rowSize_ % alignment_ != 0 )
                    rowSize_ = (rowSize_ / alignment_ + 1) * alignment_;

                if( rowSize_ * height_ != _rowSize * _height )
                    return false;

                _width      = width_;
                _height     = height_;
                _colorCount = colorCount_;
                _alignment  = alignment_;
                _rowSize    = rowSize_;

                return true;
            }

            return false;
        }

        uint8_t type() const
        {
            return _type;
        }

        std::string dataType() const
        {
            return _dataType;
        }

        size_t dataSize() const
        {
            return _dataSize;
        }

        template <typename TColorDepth = uint8_t>
        Image generate( uint32_t width_ = 0u, uint32_t height_ = 0u, uint8_t colorCount_ = 1u, uint8_t alignment_ = 1u ) const
        {
            Image image;
            image._type = _type;
            image._dataType = typeid( TColorDepth ).name();
            image._dataSize = sizeof( TColorDepth );

            image.setColorCount( colorCount_ );
            image.setAlignment( alignment_ );
            image.resize( width_, height_ );

            return image;
        }
    private:
        template <typename TColorDepth>
        class FunctionFacade
        {
        public:
            typedef TColorDepth * ( *AllocateFunction   )( size_t size );
            typedef void          ( *DeallocateFunction )( TColorDepth * data );
            typedef void          ( *CopyFunction       )( TColorDepth * out, TColorDepth * in, size_t size );
            typedef void          ( *SetFunction        )( TColorDepth * data, TColorDepth value, size_t size );

            static FunctionFacade & instance()
            {
                static FunctionFacade facade;
                return facade;
            }

            void initialize( uint8_t type, AllocateFunction allocateFunction, DeallocateFunction deallocateFunction, CopyFunction copyFunction,
                             SetFunction setFunction )
            {
                if( _allocate[type] == nullptr )
                {
                    _allocate  [type] = allocateFunction;
                    _deallocate[type] = deallocateFunction;
                    _copy      [type] = copyFunction;
                    _set       [type] = setFunction;
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
                _allocate  .resize( 256, nullptr );
                _deallocate.resize( 256, nullptr );
                _copy      .resize( 256, nullptr );
                _set       .resize( 256, nullptr );
            }

            FunctionFacade & operator=( const FunctionFacade & ) { return (*this); }
            FunctionFacade( const FunctionFacade & ) {}

            template <typename TFunction>
            TFunction _getFunction(const std::vector< TFunction >& data, uint8_t index ) const
            {
                if ( data[index] == nullptr )
                    throw imageException( "A function is not defined for this type of image" );

                return data[index];
            }

            std::vector < AllocateFunction > _allocate;
            std::vector < DeallocateFunction > _deallocate;
            std::vector < CopyFunction > _copy;
            std::vector < SetFunction > _set;
        };
    protected:
        template <typename TColorDepth>
        void _setType( uint8_t type = 0u, typename FunctionFacade<TColorDepth>::AllocateFunction allocateFunction = _allocateMemory,
                       typename FunctionFacade<TColorDepth>::DeallocateFunction deallocateFunction = _deallocateMemory,
                       typename FunctionFacade<TColorDepth>::CopyFunction copyFunction = _copyMemory,
                       typename FunctionFacade<TColorDepth>::SetFunction setFunction = _setMemory )
        {
            _type = type;
            FunctionFacade<TColorDepth>::instance().initialize( _type, allocateFunction, deallocateFunction, copyFunction, setFunction );
        }

        template <typename TColorDepth>
        void setDataType()
        {
            clear();
            _dataType = typeid( TColorDepth ).name();
            _dataSize = sizeof( TColorDepth );
        }
    private:
        template <typename TColorDepth>
        TColorDepth * _allocate( size_t size ) const
        {
            return FunctionFacade<TColorDepth>::instance().allocate( _type )( size );
        }

        template <typename TColorDepth>
        void _deallocate( TColorDepth * data ) const
        {
            FunctionFacade<TColorDepth>::instance().deallocate( _type )( data );
        }

        template <typename TColorDepth>
        void _copy( TColorDepth * out, TColorDepth * in, size_t size ) const
        {
            FunctionFacade<TColorDepth>::instance().copy( _type )( out, in, size );
        }

        template <typename TColorDepth>
        void _set( TColorDepth * data, TColorDepth value, size_t size ) const
        {
            FunctionFacade<TColorDepth>::instance().set( _type )( data, value, size );
        }

        template <typename TColorDepth>
        static TColorDepth * _allocateMemory( size_t size )
        {
            return cpu_Memory::MemoryAllocator::instance().allocate<TColorDepth>( size );
        }

        template <typename TColorDepth>
        static void _deallocateMemory( TColorDepth * data )
        {
            cpu_Memory::MemoryAllocator::instance().free( data );
        }

        template <typename TColorDepth>
        static void _copyMemory( TColorDepth * out, TColorDepth * in, size_t size )
        {
            memcpy( out, in, size );
        }

        template <typename TColorDepth>
        static void _setMemory( TColorDepth * data, TColorDepth value, size_t size )
        {
            std::fill( data, data + size / sizeof( TColorDepth ), value );
        }

        uint32_t _width;
        uint32_t _height;

        uint8_t  _colorCount;
        uint8_t  _alignment;
        uint32_t _rowSize;

        uint8_t * _data;

        uint8_t  _type;
        
        std::string _dataType;
        size_t _dataSize;
    };

    const static uint8_t GRAY_SCALE = 1u;
    const static uint8_t RGB = 3u;
    const static uint8_t RGBA = 4u;
    const static uint8_t RED_CHANNEL = 0u;
    const static uint8_t GREEN_CHANNEL = 1u;
    const static uint8_t BLUE_CHANNEL = 2u;
    const static uint8_t ALPHA_CHANNEL = 3u;
}
