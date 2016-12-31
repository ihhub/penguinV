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
			swap( image );
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
			swap( image );

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

		void swap( ImageTemplateCuda & image )
		{
			_width  = image._width;
			_height = image._height;

			_colorCount = image._colorCount;

			std::swap( _data, image._data );
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

	private:
		uint32_t _width;
		uint32_t _height;
		uint8_t  _colorCount;

		TColorDepth * _data;
	};
};

namespace Bitmap_Image_Cuda
{
	const static uint8_t GRAY_SCALE = 1u;
	const static uint8_t RGB = 3u;

	class Image : public Template_Image_Cuda::ImageTemplateCuda <uint8_t>
	{
	public:
		Image()
			: ImageTemplateCuda(0, 0, GRAY_SCALE)
		{
		}

		Image(uint8_t colorCount_)
			: ImageTemplateCuda(0, 0, colorCount_)
		{
		}

		Image(uint32_t width_, uint32_t height_)
			: ImageTemplateCuda( width_, height_, GRAY_SCALE )
		{
		}

		Image(uint32_t width_, uint32_t height_, uint8_t colorCount_)
			: ImageTemplateCuda( width_, height_, colorCount_ )
		{
		}

		Image(const Image & image)
			: ImageTemplateCuda(image)
		{
		}

		Image(Image && image)
			: ImageTemplateCuda(0, 0, GRAY_SCALE)
		{
			swap( image );
		}

		Image & operator=(const Image & image)
		{
			ImageTemplateCuda::operator=( image );

			return (*this);
		}

		Image & operator=(Image && image)
		{
			swap( image );

			return (*this);
		}

		~Image()
		{
		}
	};
};
