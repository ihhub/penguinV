#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include "image_exception.h"

namespace Template_Image
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

        ImageTemplate( size_t width_, size_t height_ )
            : _width     ( 0 )
            , _height    ( 0 )
            , _colorCount( 1 )
            , _alignment ( 1 )
            , _rowSize   ( 0 )
            , _data      ( nullptr )
        {
            resize( width_, height_ );
        }

        ImageTemplate( size_t width_, size_t height_, uint8_t colorCount_ )
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

        ImageTemplate( size_t width_, size_t height_, uint8_t colorCount_, uint8_t alignment_ )
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

        void resize( size_t width_, size_t height_ )
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

        void assign( TColorDepth * data_, size_t width_, size_t height_, uint8_t colorCount_, uint8_t alignment_ )
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

        size_t width() const
        {
            return _width;
        }

        size_t height() const
        {
            return _height;
        }

        size_t rowSize() const
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

            memset( data(), value, sizeof( TColorDepth ) * height() * rowSize() );
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

                memcpy( _data, image._data, sizeof( TColorDepth ) * _height * _rowSize );
            }
        }

        bool mutate( size_t width_, size_t height_, uint8_t colorCount_, uint8_t alignment_ )
        {
            if( colorCount_ > 0 && alignment_ > 0 )
            {
                size_t rowSize_ = width_ * colorCount_;
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

    private:
        size_t _width;
        size_t _height;

        uint8_t  _colorCount;
        uint8_t  _alignment;
        size_t _rowSize;

        TColorDepth * _data;
    };
};

namespace Bitmap_Image
{
    const static uint8_t BITMAP_ALIGNMENT = 4u; // this is default alignment of Bitmap images
                                                // you can change it for your purposes

    const static uint8_t GRAY_SCALE = 1u;
    const static uint8_t RGB = 3u;
    const static uint8_t RGBA = 4u;

    class Image : public Template_Image::ImageTemplate <uint8_t>
    {
    public:
        Image()
            : ImageTemplate( 0, 0, GRAY_SCALE, BITMAP_ALIGNMENT )
        {
        }

        Image( uint8_t colorCount_ )
            : ImageTemplate( 0, 0, colorCount_, BITMAP_ALIGNMENT )
        {
        }

        Image( size_t width_, size_t height_ )
            : ImageTemplate( width_, height_, GRAY_SCALE, BITMAP_ALIGNMENT )
        {
        }

        Image( size_t width_, size_t height_, uint8_t colorCount_ )
            : ImageTemplate( width_, height_, colorCount_, BITMAP_ALIGNMENT )
        {
        }

        Image( size_t width_, size_t height_, uint8_t colorCount_, uint8_t alignment_ )
            : ImageTemplate( width_, height_, colorCount_, alignment_ )
        {
        }

        Image( const Image & image )
            : ImageTemplate( image )
        {
        }

        Image( Image && image )
            : ImageTemplate( 0, 0, GRAY_SCALE, BITMAP_ALIGNMENT )
        {
            swap( image );
        }

        Image & operator=( const Image & image )
        {
            ImageTemplate::operator=( image );

            return (*this);
        }

        Image & operator=( Image && image )
        {
            swap( image );

            return (*this);
        }

        virtual ~Image()
        {
        }
    };
};

namespace Image_Function
{
    using namespace Bitmap_Image;

    template <typename TImage>
    uint8_t CommonColorCount( const TImage & image1, const TImage & image2 )
    {
        if( image1.colorCount() != image2.colorCount() )
            throw imageException( "Color counts of images are different" );

        return image1.colorCount();
    }

    template <typename TImage>
    uint8_t CommonColorCount( const TImage & image1, const TImage & image2, const TImage & image3 )
    {
        if( image1.colorCount() != image2.colorCount() || image1.colorCount() != image3.colorCount() )
            throw imageException( "Color counts of images are different" );

        return image1.colorCount();
    }

    template <typename TImage>
    bool IsCorrectColorCount( const TImage & image )
    {
        return image.colorCount() == GRAY_SCALE || image.colorCount() == RGB;
    }

    template <typename TImage>
    void VerifyColoredImage( const TImage & image1 )
    {
        if( image1.colorCount() != RGB )
            throw imageException( "Bad input parameters in image function: colored image has different than 3 color channels" );
    }

    template <typename TImage>
    void VerifyColoredImage( const TImage & image1, const TImage & image2 )
    {
        if( image1.colorCount() != RGB || image2.colorCount() != RGB )
            throw imageException( "Bad input parameters in image function: colored image has different than 3 color channels" );
    }

    template <typename TImage>
    void VerifyColoredImage( const TImage & image1, const TImage & image2, const TImage & image3 )
    {
        if( image1.colorCount() != RGB || image2.colorCount() != RGB || image3.colorCount() != RGB )
            throw imageException( "Bad input parameters in image function: colored image has different than 3 color channels" );
    }

    template <typename TImage>
    void VerifyGrayScaleImage( const TImage & image1 )
    {
        if( image1.colorCount() != GRAY_SCALE )
            throw imageException( "Bad input parameters in image function: gray-scaled image has more than 1 color channels" );
    }

    template <typename TImage>
    void VerifyGrayScaleImage( const TImage & image1, const TImage & image2 )
    {
        if( image1.colorCount() != GRAY_SCALE || image2.colorCount() != GRAY_SCALE )
            throw imageException( "Bad input parameters in image function: gray-scaled image has more than 1 color channels" );
    }

    template <typename TImage>
    void VerifyGrayScaleImage( const TImage & image1, const TImage & image2, const TImage & image3 )
    {
        if( image1.colorCount() != GRAY_SCALE || image2.colorCount() != GRAY_SCALE || image3.colorCount() != GRAY_SCALE )
            throw imageException( "Bad input parameters in image function: gray-scaled image has more than 1 color channels" );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image1 )
    {
        if( image1.empty() || !IsCorrectColorCount( image1 ) )
            throw imageException( "Bad input parameters in image function" );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image1, const TImage & image2 )
    {
        if( image1.empty() || image2.empty() || !IsCorrectColorCount( image1 ) || !IsCorrectColorCount( image2 ) ||
            image1.width() != image2.width() || image1.height() != image2.height() )
            throw imageException( "Bad input parameters in image function" );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image1, const TImage & image2, const TImage & image3 )
    {
        if( image1.empty() || image2.empty() || image3.empty() || !IsCorrectColorCount( image1 ) || !IsCorrectColorCount( image2 ) ||
            !IsCorrectColorCount( image3 ) || image1.width() != image2.width() || image1.height() != image2.height() ||
            image1.width() != image3.width() || image1.height() != image3.height() )
            throw imageException( "Bad input parameters in image function" );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image, size_t startX, size_t startY, size_t width, size_t height )
    {
        if( image.empty() || !IsCorrectColorCount( image ) || width == 0 || height == 0 || startX + width > image.width() || startY + height > image.height() )
            throw imageException( "Bad input parameters in image function" );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image1, size_t startX1, size_t startY1,
                              const TImage & image2, size_t startX2, size_t startY2,
                              size_t width, size_t height )
    {
        if( image1.empty() || image2.empty() || !IsCorrectColorCount( image1 ) || !IsCorrectColorCount( image2 ) || width == 0 || height == 0 ||
            startX1 + width > image1.width() || startY1 + height > image1.height() ||
            startX2 + width > image2.width() || startY2 + height > image2.height() )
            throw imageException( "Bad input parameters in image function" );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image1, size_t startX1, size_t startY1,
                              const TImage & image2, size_t startX2, size_t startY2,
                              const TImage & image3, size_t startX3, size_t startY3,
                              size_t width, size_t height )
    {
        if( image1.empty() || image2.empty() || image3.empty() || !IsCorrectColorCount( image1 ) || !IsCorrectColorCount( image2 ) ||
            !IsCorrectColorCount( image3 ) || width == 0 || height == 0 ||
            startX1 + width > image1.width() || startY1 + height > image1.height() ||
            startX2 + width > image2.width() || startY2 + height > image2.height() ||
            startX3 + width > image3.width() || startY3 + height > image3.height() )
            throw imageException( "Bad input parameters in image function" );
    }
};
