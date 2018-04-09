#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include "image_exception.h"

namespace PenguinV_Image
{
    template <typename TColorDepth>
    class ImageTemplate
    {
    public:
        ImageTemplate()
            : _width     ( 0 )       // width of image
            , _height    ( 0 )       // height of image
            , _colorCount( 1 )       // number of colors per pixel
            , _alignment ( 1 )       // some formats require that row size must be a multiple of some value (alignment)
                                     // for example for Bitmap it must be a multiple of 4
            , _rowSize   ( 0 )       // size of single row on image, usually it is equal to width
            , _data      ( nullptr ) // an array what store image information (pixel data)
        {
        }

        ImageTemplate( uint32_t width_, uint32_t height_ )
            : _width     ( 0 )
            , _height    ( 0 )
            , _colorCount( 1 )
            , _alignment ( 1 )
            , _rowSize   ( 0 )
            , _data      ( nullptr )
        {
            resize( width_, height_ );
        }

        ImageTemplate( uint32_t width_, uint32_t height_, uint8_t colorCount_ )
            : _width     ( 0 )
            , _height    ( 0 )
            , _colorCount( 1 )
            , _alignment ( 1 )
            , _rowSize   ( 0 )
            , _data      ( nullptr )
        {
            setColorCount( colorCount_ );
            resize( width_, height_ );
        }

        ImageTemplate( uint32_t width_, uint32_t height_, uint8_t colorCount_, uint8_t alignment_ )
            : _width     ( 0 )
            , _height    ( 0 )
            , _colorCount( 1 )
            , _alignment ( 1 )
            , _rowSize   ( 0 )
            , _data      ( nullptr )
        {
            setColorCount( colorCount_ );
            setAlignment( alignment_ );
            resize( width_, height_ );
        }

        ImageTemplate( const ImageTemplate & image )
            : _data      ( nullptr )
        {
            copy( image );
        }

        ImageTemplate( ImageTemplate && image )
            : _width     ( 0 )
            , _height    ( 0 )
            , _colorCount( 1 )
            , _alignment ( 1 )
            , _rowSize   ( 0 )
            , _data      ( nullptr )
        {
            swap( image );
        }

        ImageTemplate & operator=( const ImageTemplate & image )
        {
            copy( image );

            return (*this);
        }

        ImageTemplate & operator=( ImageTemplate && image )
        {
            swap( image );

            return (*this);
        }

        virtual ~ImageTemplate()
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

                _data = _allocate( _height * _rowSize );
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
            if( data_ == nullptr || width_ == 0 || height_ == 0 || colorCount_ == 0 || alignment_ == 0 )
                throw imageException( "Invalid image assignment parameters" );

            clear();

            _width  = width_;
            _height = height_;

            _colorCount = colorCount_;
            _alignment = alignment_;

            _data = data_;

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

        void fill( TColorDepth value )
        {
            if( empty() )
                return;

            _set( data(), value, sizeof( TColorDepth ) * height() * rowSize() );
        }

        void swap( ImageTemplate & image )
        {
            std::swap( _width, image._width );
            std::swap( _height, image._height );

            std::swap( _colorCount, image._colorCount );
            std::swap( _rowSize   , image._rowSize );
            std::swap( _alignment , image._alignment );

            std::swap( _data, image._data );
        }

        void copy( const ImageTemplate & image )
        {
            clear();

            _width  = image._width;
            _height = image._height;

            _colorCount = image._colorCount;
            _rowSize    = image._rowSize;
            _alignment  = image._alignment;

            if( image._data != nullptr ) {
                _data = _allocate( _height * _rowSize );

                _copy( _data, image._data, sizeof( TColorDepth ) * _height * _rowSize );
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
    protected:
        virtual TColorDepth * _allocate( size_t size ) const
        {
            return new TColorDepth[size];
        }

        virtual void _deallocate( TColorDepth * data ) const
        {
            delete[] data;
        }

        virtual void _copy( TColorDepth * out, TColorDepth * in, size_t size )
        {
            memcpy( out, in, size );
        }

        virtual void _set( TColorDepth * data, TColorDepth value, size_t size )
        {
            memset( data, static_cast<int>(value), size );
        }

    private:
        uint32_t _width;
        uint32_t _height;

        uint8_t  _colorCount;
        uint8_t  _alignment;
        uint32_t _rowSize;

        TColorDepth * _data;
    };

    typedef ImageTemplate <uint8_t> Image;

    const static uint8_t GRAY_SCALE = 1u;
    const static uint8_t RGB = 3u;
    const static uint8_t RGBA = 4u;
}
