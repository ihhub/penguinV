#include "../../Library/cuda/image_function_cuda.cuh"
#include "../unit_test_helper.h"
#include "unit_test_image_function_cuda.h"
#include "unit_test_helper_cuda.h"

namespace Unit_Test
{
	void addTests_Image_Function_Cuda(UnitTestFramework & framework)
	{
		ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseAnd2ParametersTest );
		ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseAnd3ParametersTest );

		ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseOr2ParametersTest );
		ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseOr3ParametersTest );

		ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseXor2ParametersTest );
		ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseXor3ParametersTest );

		ADD_TEST( framework, Image_Function_Cuda_Test::Invert1ParameterTest );
		ADD_TEST( framework, Image_Function_Cuda_Test::Invert2ParametersTest );
	}

	namespace Image_Function_Cuda_Test
	{
		bool BitwiseAnd2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image_Cuda::ImageCuda > input = Cuda::uniformImages( intensity );

				Bitmap_Image_Cuda::ImageCuda output = Image_Function_Cuda::BitwiseAnd( input[0], input[1] );

				if( !Cuda::equalSize( input[0], output ) || !Cuda::verifyImage( output, intensity[0] & intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseAnd3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages( intensity );

				Image_Function_Cuda::BitwiseAnd( image[0], image[1], image[2] );

				if( !Cuda::verifyImage( image[2], intensity[0] & intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image_Cuda::ImageCuda > input = Cuda::uniformImages( intensity );

				Bitmap_Image_Cuda::ImageCuda output = Image_Function_Cuda::BitwiseOr( input[0], input[1] );

				if( !Cuda::equalSize( input[0], output ) || !Cuda::verifyImage( output, intensity[0] | intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages( intensity );

				Image_Function_Cuda::BitwiseOr( image[0], image[1], image[2] );

				if( !Cuda::verifyImage( image[2], intensity[0] | intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image_Cuda::ImageCuda > input = Cuda::uniformImages( intensity );

				Bitmap_Image_Cuda::ImageCuda output = Image_Function_Cuda::BitwiseXor( input[0], input[1] );

				if( !Cuda::equalSize( input[0], output ) || !Cuda::verifyImage( output, intensity[0] ^ intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages( intensity );

				Image_Function_Cuda::BitwiseXor( image[0], image[1], image[2] );

				if( !Cuda::verifyImage( image[2], intensity[0] ^ intensity[1] ) )
					return false;
			}

			return true;
		}

		bool Invert1ParameterTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image_Cuda::ImageCuda input = Cuda::uniformImage( intensity );

				Bitmap_Image_Cuda::ImageCuda output = Image_Function_Cuda::Invert( input );

				if( !Cuda::equalSize( input, output ) || !Cuda::verifyImage( output, ~intensity ) )
					return false;
			}

			return true;
		}

		bool Invert2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image_Cuda::ImageCuda > input = Cuda::uniformImages( intensity );

				Image_Function_Cuda::Invert( input[0], input[1] );

				if( !Cuda::verifyImage( input[1], ~intensity[0] ) )
					return false;
			}

			return true;
		}
	};
};
