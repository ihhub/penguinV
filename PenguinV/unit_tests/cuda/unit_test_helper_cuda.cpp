#include "../../Library/image_buffer.h"
#include "../../Library/image_function.h"
#include "../../Library/cuda/image_function_cuda.cuh"
#include "unit_test_helper_cuda.h"

namespace Unit_Test
{
	namespace Cuda
	{
		Bitmap_Image_Cuda::ImageCuda uniformImage(uint8_t value)
		{
			Bitmap_Image_Cuda::ImageCuda image( randomValue<uint32_t>( 1, 2048 ), randomValue<uint32_t>( 1, 2048 ) );

			image.fill( value );

			return image;
		}

		Bitmap_Image_Cuda::ImageCuda uniformImage()
		{
			return uniformImage( randomValue<uint8_t>( 256 ) );
		}

		Bitmap_Image_Cuda::ImageCuda blackImage()
		{
			return uniformImage( 0u );
		}

		Bitmap_Image_Cuda::ImageCuda whiteImage()
		{
			return uniformImage( 255u );
		}

		std::vector < Bitmap_Image_Cuda::ImageCuda > uniformImages( uint32_t images )
		{
			if( images == 0 )
				throw imageException( "Invalid parameter" );

			std::vector < Bitmap_Image_Cuda::ImageCuda > image;

			image.push_back( uniformImage() );

			image.resize( images );

			for( size_t i = 1; i < image.size(); ++i ) {
				image[i].resize( image[0].width(), image[0].height() );
				image[i].fill( randomValue<uint8_t>( 256 ) );
			}

			return image;
		}

		std::vector < Bitmap_Image_Cuda::ImageCuda > uniformImages( std::vector < uint8_t > intensityValue )
		{
			if( intensityValue.size() == 0 )
				throw imageException( "Invalid parameter" );

			std::vector < Bitmap_Image_Cuda::ImageCuda > image;

			image.push_back( uniformImage(intensityValue[0]) );

			image.resize( intensityValue.size() );

			for( size_t i = 1; i < image.size(); ++i ) {
				image[i].resize( image[0].width(), image[0].height() );
				image[i].fill( intensityValue[i] );
			}

			return image;
		}

		bool verifyImage( const Bitmap_Image_Cuda::ImageCuda & image, uint8_t value )
		{
			Bitmap_Image::Image imageCpu( image.width(), image.height() );

			Image_Function_Cuda::Convert( image, imageCpu );

			Image_Function::ParameterValidation( imageCpu );

			const uint8_t * outputY = imageCpu.data();
			const uint8_t * endY    = outputY + imageCpu.rowSize() * imageCpu.height();

			for( ; outputY != endY; outputY += imageCpu.rowSize() ) {
				if( std::any_of( outputY, outputY + imageCpu.width(),
					[value](const uint8_t & v){ return v != value; } ) )
					return false;
			}

			return true;
		}
	};
};
