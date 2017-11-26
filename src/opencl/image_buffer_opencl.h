#pragma once

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include "../image_exception.h"
#include "../thirdparty/multicl/src/opencl_device.h"

namespace Template_Image_OpenCL
{
    template <typename TColorDepth>
    class ImageTemplateOpenCL
    {
    public:
        ImageTemplateOpenCL()
            : _width     ( 0 )    // width of image
            , _height    ( 0 )    // height of image
            , _colorCount( 1 )    // number of colors per pixel
            , _rowSize   ( 0 )    // size of single row on image which is equal to width * colorCount
            , _data      ( NULL ) // an array what store image information (pixel data)
        {
        }

        ImageTemplateOpenCL( uint32_t width_, uint32_t height_ )
            : _width     ( 0 )
            , _height    ( 0 )
            , _colorCount( 1 )
            , _rowSize   ( 0 )
            , _data      ( NULL )
        {
            resize( width_, height_ );
        }

        ImageTemplateOpenCL( uint32_t width_, uint32_t height_, uint8_t colorCount_ )
            : _width     ( 0 )
            , _height    ( 0 )
            , _colorCount( 1 )
            , _rowSize   ( 0 )
            , _data      ( NULL )
        {
            setColorCount( colorCount_ );
            resize( width_, height_ );
        }

        ImageTemplateOpenCL( const ImageTemplateOpenCL & image )
            : _data      ( NULL )
        {
            copy( image );
        }

        ImageTemplateOpenCL( ImageTemplateOpenCL && image )
            : _width     ( 0 )
            , _height    ( 0 )
            , _colorCount( 1 )
            , _data      ( NULL )
        {
            swap( image );
        }

        ImageTemplateOpenCL & operator=( const ImageTemplateOpenCL & image )
        {
            copy( image );

            return (*this);
        }

        ImageTemplateOpenCL & operator=( ImageTemplateOpenCL && image )
        {
            swap( image );

            return (*this);
        }

        ~ImageTemplateOpenCL()
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

                _data = multiCL::MemoryManager::memory().allocate<TColorDepth>( _rowSize * _height );
            }
        }

        void clear()
        {
            if( _data != NULL ) {
                multiCL::MemoryManager::memory().free( _data );

                _data = NULL;
            }

            _width  = 0;
            _height = 0;
        }

        cl_mem data()
        {
            return _data;
        }

        const cl_mem data() const
        {
            return _data;
        }

        bool empty() const
        {
            return _data == NULL;
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

        void fill( TColorDepth value )
        {
            if( empty() )
                return;

            cl_int error = clEnqueueFillBuffer( multiCL::OpenCLDeviceManager::instance().device().queue()(), _data, &value, sizeof( TColorDepth ),
                                                0, _rowSize * _width, 0, NULL, NULL );
            if( error != CL_SUCCESS )
                throw imageException( "Cannot fill a memory for GPU device" );
        }

        void swap( ImageTemplateOpenCL & image )
        {
            std::swap( _width, image._width );
            std::swap( _height, image._height );

            std::swap( _colorCount, image._colorCount );
            std::swap( _rowSize   , image._rowSize );

            std::swap( _data, image._data );
        }

        void copy( const ImageTemplateOpenCL & image )
        {
            clear();

            _width  = image._width;
            _height = image._height;
            _colorCount = image._colorCount;
            _rowSize    = image._rowSize; 

            if( image._data != NULL ) {
                _data = multiCL::MemoryManager::memory().allocate<TColorDepth>( _rowSize * _width );

                cl_int error = clEnqueueCopyBuffer( multiCL::OpenCLDeviceManager::instance().device().queue()(), image.data(), _data, 0, 0, _rowSize * _width, 0, NULL, NULL );
                if( error != CL_SUCCESS )
                    throw imageException( "Cannot copy a memory in GPU device" );
            }
        }

    private:
        uint32_t _width;
        uint32_t _height;
        uint8_t  _colorCount;
        uint32_t _rowSize;

        cl_mem _data;
    };
};

namespace Bitmap_Image_OpenCL
{
    const static uint8_t GRAY_SCALE = 1u;
    const static uint8_t RGB = 3u;
    const static uint8_t RGBA = 4u;

    class Image : public Template_Image_OpenCL::ImageTemplateOpenCL <uint8_t>
    {
    public:
        Image()
            : ImageTemplateOpenCL( 0, 0, GRAY_SCALE )
        {
        }

        explicit Image( uint8_t colorCount_ )
            : ImageTemplateOpenCL( 0, 0, colorCount_ )
        {
        }

        Image( uint32_t width_, uint32_t height_ )
            : ImageTemplateOpenCL( width_, height_, GRAY_SCALE )
        {
        }

        Image( uint32_t width_, uint32_t height_, uint8_t colorCount_ )
            : ImageTemplateOpenCL( width_, height_, colorCount_ )
        {
        }

        Image( const Image & image )
            : ImageTemplateOpenCL( image )
        {
        }

        Image( Image && image )
            : ImageTemplateOpenCL( 0, 0, GRAY_SCALE )
        {
            swap( image );
        }

        Image & operator=( const Image & image )
        {
            ImageTemplateOpenCL::operator=( image );

            return (*this);
        }

        Image & operator=( Image && image )
        {
            swap( image );

            return (*this);
        }

        ~Image()
        {
        }
    };
};
