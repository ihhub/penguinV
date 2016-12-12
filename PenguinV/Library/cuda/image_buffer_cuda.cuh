#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include "../image_exception.h"
#include "cuda_memory.cuh"

namespace Template_Image_Cuda
{
	template <typename TColorDepth>
	class ImageTemplateCuda
	{
	public:
		ImageTemplateCuda()
			: _width     (0)    // width of image
			, _height    (0)    // height of image
			, _colorCount(1)    // number of colors per pixel
			, _data      (NULL) // an array what store image information (pixel data)
		{
		}

		ImageTemplateCuda(uint32_t width_, uint32_t height_)
			: _width     (0)
			, _height    (0)
			, _colorCount(1)
			, _data      (NULL)
		{
			resize( width_, height_ );
		}

		ImageTemplateCuda(uint32_t width_, uint32_t height_, uint8_t colorCount_)
			: _width     (0)
			, _height    (0)
			, _colorCount(1)
			, _data      (NULL)
		{
			setColorCount( colorCount_ );
			resize( width_, height_ );
		}

		ImageTemplateCuda( const ImageTemplateCuda & image )
		{
			_width  = image._width;
			_height = image._height;

			_colorCount = image._colorCount;

			if( image._data != NULL ) {
				Cuda_Memory::MemoryAllocator::instance().allocate(&_data, _height * _width);

				cudaError error = cudaMemcpy( _data, image._data, _height * _width * sizeof(TColorDepth), cudaMemcpyDeviceToDevice );
				if( error != cudaSuccess )
					throw imageException("Cannot copy a memory in CUDA device");
			}
			else {
				_data = NULL;
			}
		}

		ImageTemplateCuda( ImageTemplateCuda && image )
			: _width     (0)
			, _height    (0)
			, _colorCount(1)
			, _data      (NULL)
		{
			_swap( image );
		}

		ImageTemplateCuda & operator=(const ImageTemplateCuda & image)
		{
			clear();

			_width  = image._width;
			_height = image._height;

			_colorCount = image._colorCount;

			if( image._data != NULL ) {
				Cuda_Memory::MemoryAllocator::instance().allocate(&_data, _height * _width);

				cudaError error = cudaMemcpy( _data, image._data, _height * _width * sizeof(TColorDepth), cudaMemcpyDeviceToDevice );
				if( error != cudaSuccess )
					throw imageException("Cannot copy a memory in CUDA device");
			}

			return (*this);
		}

		ImageTemplateCuda & operator=(ImageTemplateCuda && image)
		{
			_swap( image );

			return (*this);
		}

		~ImageTemplateCuda()
		{
			clear();
		}

		void resize(uint32_t width_, uint32_t height_)
		{
			if( width_ > 0 && height_ > 0 && (width_ != _width || height_ != _height) ) {
				clear();

				_width  = width_;
				_height = height_;

				Cuda_Memory::MemoryAllocator::instance().allocate(&_data, _height * _width);
			}
		}

		void clear()
		{
			if( _data != NULL ) {
				Cuda_Memory::MemoryAllocator::instance().free(_data);

				_data = NULL;
			}

			_width  = 0;
			_height = 0;
		}

		TColorDepth * data()
		{
			return _data;
		}

		const TColorDepth * data() const
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

		void fill(TColorDepth value)
		{
			if( empty() )
				return;

			cudaError_t error = cudaMemset( data(), value, sizeof(TColorDepth) * height() * width() );
			if (error != cudaSuccess)
					throw imageException("Cannot fill a memory for CUDA device");
		}
	protected:
		void _copy( const ImageTemplateCuda & image )
		{
			if( image.empty() || empty() || image.width() != width() || image.height() != height() || image.colorCount() != colorCount() )
				throw imageException("Invalid image to copy");

			cudaError_t error = cudaMemcpy( _data, image._data, _height * _width * sizeof(TColorDepth), cudaMemcpyDeviceToDevice );
			if( error != cudaSuccess )
				throw imageException("Cannot copy a memory in CUDA device");
		}

		void _swap( ImageTemplateCuda & image )
		{
			_width  = image._width;
			_height = image._height;

			_colorCount = image._colorCount;

			std::swap( _data, image._data );
		}
	private:
		uint32_t _width;
		uint32_t _height;
		uint8_t  _colorCount;

		TColorDepth * _data;
	};
};

namespace Bitmap_Image_Cuda
{
	template <uint8_t bytes = 1>
	class BitmapImageCuda : public Template_Image_Cuda::ImageTemplateCuda <uint8_t>
	{
	public:
		BitmapImageCuda()
			: ImageTemplateCuda(0, 0, bytes)
		{
		}

		BitmapImageCuda(uint32_t width_, uint32_t height_)
			: ImageTemplateCuda( width_, height_, bytes )
		{
		}

		BitmapImageCuda(const BitmapImageCuda & image)
			: ImageTemplateCuda(image)
		{
		}

		BitmapImageCuda(BitmapImageCuda && image)
			: ImageTemplateCuda(0, 0, bytes)
		{
			_swap( image );
		}

		BitmapImageCuda & operator=(const BitmapImageCuda & image)
		{
			ImageTemplateCuda::operator=( image );

			return (*this);
		}

		BitmapImageCuda & operator=(BitmapImageCuda && image)
		{
			_swap( image );

			return (*this);
		}

		~BitmapImageCuda()
		{
		}

		void resize(uint32_t width_, uint32_t height_)
		{
			ImageTemplateCuda::resize( width_, height_ );
		}

		void clear()
		{
			ImageTemplateCuda::clear();
		}

		uint8_t * data()
		{
			return ImageTemplateCuda::data();
		}

		const uint8_t * data() const
		{
			return ImageTemplateCuda::data();
		}

		bool empty() const
		{
			return ImageTemplateCuda::empty();
		}

		uint32_t width() const
		{
			return ImageTemplateCuda::width();
		}

		uint32_t height() const
		{
			return ImageTemplateCuda::height();
		}

		uint8_t colorCount() const
		{
			return ImageTemplateCuda::colorCount();
		}
	private:
		void setColorCount(uint8_t colorCount_)
		{
			ImageTemplateCuda::setColorCount(colorCount_);
		}
	};

	typedef BitmapImageCuda < 1 > ImageCuda;      // gray-scale image (1 color [byte])
	typedef BitmapImageCuda < 3 > ColorImageCuda; // RGB image (usually 3 colors [bytes] but could contain 4)
};
