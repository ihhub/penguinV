#include "performance_test_helper_cuda.h"

namespace Performance_Test
{
	namespace Cuda
	{
		Bitmap_Image_Cuda::ImageCuda uniformImage(uint32_t width, uint32_t height)
		{
			return uniformImage( width, height, randomValue<uint8_t>(256) );
		}

		Bitmap_Image_Cuda::ImageCuda uniformImage(uint32_t width, uint32_t height, uint8_t value)
		{
			Bitmap_Image_Cuda::ImageCuda image( width, height );

			image.fill( value );

			return image;
		}

		std::vector< Bitmap_Image_Cuda::ImageCuda > uniformImages(uint32_t count, uint32_t width, uint32_t height)
		{
			std::vector < Bitmap_Image_Cuda::ImageCuda > image( count );

			for( std::vector< Bitmap_Image_Cuda::ImageCuda >::iterator im = image.begin(); im != image.end(); ++im )
				*im = uniformImage( width, height );

			return image;
		}
	};
};
