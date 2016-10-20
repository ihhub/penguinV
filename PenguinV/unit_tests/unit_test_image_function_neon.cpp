#include "../Library/image_function_neon.h"
#include "unit_test_image_function_neon.h"
#include "unit_test_helper.h"

namespace Unit_Test
{
	void addTests_Image_Function_Neon(UnitTestFramework & framework)
	{
		ADD_TEST( framework, Image_Function_Neon_Test::BitwiseAnd2ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::BitwiseAnd3ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::BitwiseAnd8ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::BitwiseAnd11ParametersTest );

		ADD_TEST( framework, Image_Function_Neon_Test::BitwiseOr2ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::BitwiseOr3ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::BitwiseOr8ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::BitwiseOr11ParametersTest );

		ADD_TEST( framework, Image_Function_Neon_Test::Maximum2ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::Maximum3ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::Maximum8ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::Maximum11ParametersTest );

		ADD_TEST( framework, Image_Function_Neon_Test::Minimum2ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::Minimum3ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::Minimum8ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::Minimum11ParametersTest );

		ADD_TEST( framework, Image_Function_Neon_Test::Subtract2ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::Subtract3ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::Subtract8ParametersTest );
		ADD_TEST( framework, Image_Function_Neon_Test::Subtract11ParametersTest );
	}

	namespace Image_Function_Neon_Test
	{
		bool BitwiseAnd2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Image_Function_Neon::BitwiseAnd( input[0], input[1] );

				if( !equalSize( input[0], output ) || !verifyImage( output, intensity[0] & intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseAnd3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Image_Function_Neon::BitwiseAnd( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] & intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseAnd8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function_Neon::BitwiseAnd(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity[0] & intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseAnd11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function_Neon::BitwiseAnd( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
												image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] & intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Image_Function_Neon::BitwiseOr( input[0], input[1] );

				if( !equalSize( input[0], output ) || !verifyImage( output, intensity[0] | intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Image_Function_Neon::BitwiseOr( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] | intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function_Neon::BitwiseOr(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity[0] | intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function_Neon::BitwiseOr( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											   image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] | intensity[1] ) )
					return false;
			}

			return true;
		}

		bool Maximum2ParametersTest()
		{
			for (uint32_t i = 0; i < runCount(); ++i) {
				std::vector < uint8_t > intensity = intensityArray(2);
				std::vector < Bitmap_Image::Image > input = uniformImages(intensity);

				Bitmap_Image::Image output = Image_Function_Neon::Maximum(input[0], input[1]);

				if (!equalSize(input[0], output) ||
					!verifyImage(output, intensity[0] > intensity[1] ? intensity[0] : intensity[1]))
					return false;
			}

			return true;
		}

		bool Maximum3ParametersTest()
		{
			for (uint32_t i = 0; i < runCount(); ++i) {
				std::vector < uint8_t > intensity = intensityArray(3);
				std::vector < Bitmap_Image::Image > image = uniformImages(intensity);

				Image_Function_Neon::Maximum(image[0], image[1], image[2]);

				if (!verifyImage(image[2], intensity[0] > intensity[1] ? intensity[0] : intensity[1]))
					return false;
			}

			return true;
		}

		bool Maximum8ParametersTest()
		{
			for (uint32_t i = 0; i < runCount(); ++i) {
				std::vector < uint8_t > intensity = intensityArray(2);
				std::vector < Bitmap_Image::Image > input;

				std::for_each(intensity.begin(), intensity.end(), [&](uint8_t & value)
				{ input.push_back(uniformImage(value)); });

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi(input, roiX, roiY, roiWidth, roiHeight);

				Bitmap_Image::Image output = Image_Function_Neon::Maximum(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight);

				if (!equalSize(output, roiWidth, roiHeight) ||
					!verifyImage(output, intensity[0] > intensity[1] ? intensity[0] : intensity[1]))
					return false;
			}

			return true;
		}

		bool Maximum11ParametersTest()
		{
			for (uint32_t i = 0; i < runCount(); ++i) {
				std::vector < uint8_t > intensity = intensityArray(3);
				std::vector < Bitmap_Image::Image > image;

				std::for_each(intensity.begin(), intensity.end(), [&](uint8_t & value)
				{ image.push_back(uniformImage(value)); });

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi(image, roiX, roiY, roiWidth, roiHeight);

				Image_Function_Neon::Maximum(image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
					image[2], roiX[2], roiY[2], roiWidth, roiHeight);

				if (!verifyImage(image[2], roiX[2], roiY[2], roiWidth, roiHeight,
					intensity[0] > intensity[1] ? intensity[0] : intensity[1]))
					return false;
			}

			return true;
		}

		bool Minimum2ParametersTest()
		{
			for (uint32_t i = 0; i < runCount(); ++i) {
				std::vector < uint8_t > intensity = intensityArray(2);
				std::vector < Bitmap_Image::Image > input = uniformImages(intensity);

				Bitmap_Image::Image output = Image_Function_Neon::Minimum(input[0], input[1]);

				if (!equalSize(input[0], output) ||
					!verifyImage(output, intensity[0] < intensity[1] ? intensity[0] : intensity[1]))
					return false;
			}

			return true;
		}

		bool Minimum3ParametersTest()
		{
			for (uint32_t i = 0; i < runCount(); ++i) {
				std::vector < uint8_t > intensity = intensityArray(3);
				std::vector < Bitmap_Image::Image > image = uniformImages(intensity);

				Image_Function_Neon::Minimum(image[0], image[1], image[2]);

				if (!verifyImage(image[2], intensity[0] < intensity[1] ? intensity[0] : intensity[1]))
					return false;
			}

			return true;
		}

		bool Minimum8ParametersTest()
		{
			for (uint32_t i = 0; i < runCount(); ++i) {
				std::vector < uint8_t > intensity = intensityArray(2);
				std::vector < Bitmap_Image::Image > input;

				std::for_each(intensity.begin(), intensity.end(), [&](uint8_t & value)
				{ input.push_back(uniformImage(value)); });

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi(input, roiX, roiY, roiWidth, roiHeight);

				Bitmap_Image::Image output = Image_Function_Neon::Minimum(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight);

				if (!equalSize(output, roiWidth, roiHeight) ||
					!verifyImage(output, intensity[0] < intensity[1] ? intensity[0] : intensity[1]))
					return false;
			}

			return true;
		}

		bool Minimum11ParametersTest()
		{
			for (uint32_t i = 0; i < runCount(); ++i) {
				std::vector < uint8_t > intensity = intensityArray(3);
				std::vector < Bitmap_Image::Image > image;

				std::for_each(intensity.begin(), intensity.end(), [&](uint8_t & value)
				{ image.push_back(uniformImage(value)); });

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi(image, roiX, roiY, roiWidth, roiHeight);

				Image_Function_Neon::Minimum(image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
					image[2], roiX[2], roiY[2], roiWidth, roiHeight);

				if (!verifyImage(image[2], roiX[2], roiY[2], roiWidth, roiHeight,
					intensity[0] < intensity[1] ? intensity[0] : intensity[1]))
					return false;
			}

			return true;
		}

		bool Subtract2ParametersTest()
		{
			for (uint32_t i = 0; i < runCount(); ++i) {
				std::vector < uint8_t > intensity = intensityArray(2);
				std::vector < Bitmap_Image::Image > input = uniformImages(intensity);

				Bitmap_Image::Image output = Image_Function_Neon::Subtract(input[0], input[1]);

				if (!equalSize(input[0], output) ||
					!verifyImage(output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0))
					return false;
			}

			return true;
		}

		bool Subtract3ParametersTest()
		{
			for (uint32_t i = 0; i < runCount(); ++i) {
				std::vector < uint8_t > intensity = intensityArray(3);
				std::vector < Bitmap_Image::Image > image = uniformImages(intensity);

				Image_Function_Neon::Subtract(image[0], image[1], image[2]);

				if (!verifyImage(image[2], intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0))
					return false;
			}

			return true;
		}

		bool Subtract8ParametersTest()
		{
			for (uint32_t i = 0; i < runCount(); ++i) {
				std::vector < uint8_t > intensity = intensityArray(2);
				std::vector < Bitmap_Image::Image > input;

				std::for_each(intensity.begin(), intensity.end(), [&](uint8_t & value)
				{ input.push_back(uniformImage(value)); });

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi(input, roiX, roiY, roiWidth, roiHeight);

				Bitmap_Image::Image output = Image_Function_Neon::Subtract(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight);

				if (!equalSize(output, roiWidth, roiHeight) ||
					!verifyImage(output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0))
					return false;
			}

			return true;
		}

		bool Subtract11ParametersTest()
		{
			for (uint32_t i = 0; i < runCount(); ++i) {
				std::vector < uint8_t > intensity = intensityArray(3);
				std::vector < Bitmap_Image::Image > image;

				std::for_each(intensity.begin(), intensity.end(), [&](uint8_t & value)
				{ image.push_back(uniformImage(value)); });

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi(image, roiX, roiY, roiWidth, roiHeight);

				Image_Function_Neon::Subtract(image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
					image[2], roiX[2], roiY[2], roiWidth, roiHeight);

				if (!verifyImage(image[2], roiX[2], roiY[2], roiWidth, roiHeight,
					intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0))
					return false;
			}

			return true;
		}
	};
};
