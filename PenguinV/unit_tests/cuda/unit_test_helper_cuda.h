#pragma once

#include <algorithm>
#include <cstdlib>
#include <vector>
#include "../../Library/image_exception.h"
#include "../../Library/cuda/image_buffer_cuda.cuh"

// A bunch of functions to help writing unit tests
namespace Unit_Test
{
	namespace Cuda
	{
		// Generate images
		Bitmap_Image_Cuda::ImageCuda uniformImage();
		Bitmap_Image_Cuda::ImageCuda uniformImage(uint8_t value);
		Bitmap_Image_Cuda::ImageCuda blackImage();
		Bitmap_Image_Cuda::ImageCuda whiteImage();
		std::vector < Bitmap_Image_Cuda::ImageCuda > uniformImages( uint32_t images );
		std::vector < Bitmap_Image_Cuda::ImageCuda > uniformImages( std::vector < uint8_t > intensityValue );

		// Image size and ROI verification
		template <typename data>
		bool equalSize( const data & image1, const data & image2 )
		{
			return image1.height() == image2.height() && image1.width() == image2.width() &&
				   image1.colorCount() == image2.colorCount();
		};

		bool verifyImage( const Bitmap_Image_Cuda::ImageCuda & image, uint8_t value );

		// Return random value for specific range or variable type
		template <typename data>
		data randomValue(int maximum)
		{
			if( maximum <= 0 )
				return 0;
			else
				return static_cast<data>( rand() ) % maximum;
		};

		template <typename data>
		data randomValue(data minimum, int maximum)
		{
			if( maximum <= 0 ) {
				return 0;
			}
			else {
				data value = static_cast<data>( rand() ) % maximum;

				if( value < minimum )
					value = minimum;

				return value;
			}
		};

	};
};
