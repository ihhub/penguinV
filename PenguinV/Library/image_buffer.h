#pragma once

#include <cstdint>
#include <algorithm>
#include <cstring>
#include "image_exception.h"

namespace Template_Image
{
	template <typename TColorDepth>
	class ImageTemplate
	{
	public:
		ImageTemplate()
			: _width     (0) // width of image
			, _height    (0) // height of image
			, _colorCount(1) // number of colors per pixel
			, _rowSize   (0) // size of single row on image, usually it is equal to width
			, _alignment (1) // some formats require that row size must be a multiple of some value (alignment)
							 // for example for Bitmap it must be a multiple of 4
		{
			_data = nullptr; // an array what store image information (pixel data)
		}

		ImageTemplate(uint32_t width_, uint32_t height_)
			: _width     (0)
			, _height    (0)
			, _colorCount(1)
			, _rowSize   (0)
			, _alignment (1)
		{
			_data = nullptr;
			
			resize( width_, height_ );
		}

		ImageTemplate(uint32_t width_, uint32_t height_, uint8_t colorCount_)
			: _width     (0)
			, _height    (0)
			, _colorCount(1)
			, _rowSize   (0)
			, _alignment (1)
		{
			_data = nullptr;
			
			setColorCount( colorCount_ );
			resize( width_, height_ );
		}

		ImageTemplate(uint32_t width_, uint32_t height_, uint8_t colorCount_, uint8_t alignment_)
			: _width     (0)
			, _height    (0)
			, _colorCount(1)
			, _rowSize   (0)
			, _alignment (1)
		{
			_data = nullptr;

			setColorCount( colorCount_ );
			setAlignment( alignment_ );
			resize( width_, height_ );
		}

		ImageTemplate( const ImageTemplate & image )
		{
			_width  = image._width;
			_height = image._height;

			_colorCount = image._colorCount;
			_rowSize    = image._rowSize;
			_alignment  = image._alignment;

			if( image._data != nullptr ) {

				_data = new TColorDepth [_width * _rowSize];

				memcpy( _data, image._data, sizeof(TColorDepth) * _width * _rowSize );

			}
			else {
				_data = nullptr;
			}
		}

		ImageTemplate( ImageTemplate && image )
		{
			_data = nullptr;

			_swap( image );
		}

		ImageTemplate & operator=(const ImageTemplate & image)
		{
			clear();

			_width  = image._width;
			_height = image._height;

			_colorCount = image._colorCount;
			_rowSize    = image._rowSize;
			_alignment  = image._alignment;

			if( image._data != nullptr ) {

				_data = new TColorDepth [_width * _rowSize];

				memcpy( _data, image._data, sizeof(TColorDepth) * _width * _rowSize );

			}
			else {
				_data = nullptr;
			}

			return (*this);
		}

		ImageTemplate & operator=(ImageTemplate && image)
		{
			_data = nullptr;

			_swap( image );

			return (*this);
		}

		~ImageTemplate()
		{
			clear();
		}

		void resize(uint32_t width_, uint32_t height_)
		{
			if( width_ > 0 && height_ > 0 && (width_ != _width || height_ != _height) ) {
				clear();

				_width  = width_;
				_height = height_;

				_rowSize = ( sizeof(TColorDepth) * width() + colorCount() * sizeof(TColorDepth) - 1) /
						   ( colorCount() * sizeof(TColorDepth) ) * colorCount();

				_data = new TColorDepth [_width * _rowSize];

			}
		}

		void clear()
		{
			if( _data != nullptr ) {
				delete [] _data;
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
			if( width_ == 0 || height_ == 0 || colorCount_ == 0 || alignment_ == 0 )
				throw imageException("Invalid image assignment parameters");

			clear();

			_width  = width_;
			_height = height_;

			_colorCount = colorCount_;
			_alignment = alignment_;

			_data = data_;
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

		void setColorCount(uint8_t colorCount_)
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

		void setAlignment(uint8_t alignment_)
		{
			if( alignment_ > 0 && alignment_ != _alignment ) {
				clear();
				_alignment = alignment_;
			}
		}

	protected:
		void _copy( const ImageTemplate & image )
		{
			if( image.empty() || empty() || image.width() != width() || image.height() != height() || image.colorCount() != colorCount() )
				throw imageException("Invalid image to copy");

			if( image.colorCount() == colorCount() ) {

				TColorDepth       * rowIn = data();
				const TColorDepth * rowOut = image.data();
				const TColorDepth * rowInEnd = rowIn + height() * rowSize();

				for( ; rowIn != rowInEnd; rowIn += _rowSize, rowOut += image._rowSize )
					memcpy( rowIn, rowOut, sizeof(TColorDepth) * width() * colorCount() );

			}
			else {
				memcpy( _data, image._data, sizeof(TColorDepth) * width() * rowSize() );
			}
		}

		void _swap( ImageTemplate & image )
		{
			_width  = image._width;
			_height = image._height;

			_colorCount = image._colorCount;
			_rowSize    = image._rowSize;
			_alignment  = image._alignment;

			std::swap( _data, image._data );
		}

	private:
		TColorDepth * _data;

		uint32_t _width;
		uint32_t _height;

		uint8_t _colorCount;
		uint8_t _alignment;
		uint32_t _rowSize;
	};
};

namespace Bitmap_Image
{
	const static uint8_t BITMAP_ALIGNMENT = 4;

	template <uint8_t bytes = 1>
	class BitmapImage : protected Template_Image::ImageTemplate <uint8_t>
	{
	public:
		BitmapImage()
			: ImageTemplate(0, 0, bytes, BITMAP_ALIGNMENT)
		{
		}

		BitmapImage(uint32_t width_, uint32_t height_)
			: ImageTemplate( width_, height_, bytes, BITMAP_ALIGNMENT )
		{
		}

		BitmapImage(const BitmapImage & image)
		{
			setColorCount( bytes );
			setAlignment( BITMAP_ALIGNMENT );

			resize( image.width(), image.height() );

			_copy( image );
		}

		BitmapImage(BitmapImage && image)
			: ImageTemplate(0, 0, bytes, BITMAP_ALIGNMENT)
		{
			_swap( image );
		}

		BitmapImage & operator=(const BitmapImage & image)
		{
			ImageTemplate::operator=( image );

			return (*this);
		}

		BitmapImage & operator=(BitmapImage && image)
		{
			_swap( image );

			return (*this);
		}

		~BitmapImage()
		{
		}

		void resize(uint32_t width_, uint32_t height_)
		{
			ImageTemplate::resize( width_, height_ );
		}

		void clear()
		{
			ImageTemplate::clear();
		}

		uint8_t * data()
		{
			return ImageTemplate::data();
		}

		const uint8_t * data() const
		{
			return ImageTemplate::data();
		}

		void assign( uint8_t * data_, uint32_t width_, uint32_t height_ )
		{
			ImageTemplate::assign( data_, width_, height_, bytes, BITMAP_ALIGNMENT );
		}

		bool empty() const
		{
			return ImageTemplate::empty();
		}

		uint32_t width() const
		{
			return ImageTemplate::width();
		}

		uint32_t height() const
		{
			return ImageTemplate::height();
		}

		uint32_t rowSize() const
		{
			return ImageTemplate::rowSize();
		}

		uint8_t colorCount() const
		{
			return ImageTemplate::colorCount();
		}

		void fill(uint8_t value)
		{
			if( empty() )
				return;

			memset( data(), value, sizeof(uint8_t) * height() * rowSize() );
		}
	};

	typedef BitmapImage < 1 > Image; // gray-scale image
	typedef BitmapImage < 3 > ColorImage; // RGB image
};