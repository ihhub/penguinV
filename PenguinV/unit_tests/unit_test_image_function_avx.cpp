#include "../Library/image_function_avx.h"
#include "unit_test_image_function_avx.h"
#include "unit_test_helper.h"

namespace Unit_Test
{
	void addTests_Image_Function_Avx(UnitTestFramework & framework)
	{
		ADD_TEST( framework, BitwiseAndAvx2ParametersTest );
		ADD_TEST( framework, BitwiseAndAvx3ParametersTest );
		ADD_TEST( framework, BitwiseAndAvx8ParametersTest );
		ADD_TEST( framework, BitwiseAndAvx11ParametersTest );

		ADD_TEST( framework, BitwiseOrAvx2ParametersTest );
		ADD_TEST( framework, BitwiseOrAvx3ParametersTest );
		ADD_TEST( framework, BitwiseOrAvx8ParametersTest );
		ADD_TEST( framework, BitwiseOrAvx11ParametersTest );

		ADD_TEST( framework, BitwiseXorAvx2ParametersTest );
		ADD_TEST( framework, BitwiseXorAvx3ParametersTest );
		ADD_TEST( framework, BitwiseXorAvx8ParametersTest );
		ADD_TEST( framework, BitwiseXorAvx11ParametersTest );

		ADD_TEST( framework, InvertAvx1ParameterTest );
		ADD_TEST( framework, InvertAvx2ParametersTest );
		ADD_TEST( framework, InvertAvx5ParametersTest );
		ADD_TEST( framework, InvertAvx8ParametersTest );

		ADD_TEST( framework, MaximumAvx2ParametersTest );
		ADD_TEST( framework, MaximumAvx3ParametersTest );
		ADD_TEST( framework, MaximumAvx8ParametersTest );
		ADD_TEST( framework, MaximumAvx11ParametersTest );

		ADD_TEST( framework, MinimumAvx2ParametersTest );
		ADD_TEST( framework, MinimumAvx3ParametersTest );
		ADD_TEST( framework, MinimumAvx8ParametersTest );
		ADD_TEST( framework, MinimumAvx11ParametersTest );

		ADD_TEST( framework, SubtractAvx2ParametersTest );
		ADD_TEST( framework, SubtractAvx3ParametersTest );
		ADD_TEST( framework, SubtractAvx8ParametersTest );
		ADD_TEST( framework, SubtractAvx11ParametersTest );

		ADD_TEST( framework, ThresholdAvx2ParametersTest );
		ADD_TEST( framework, ThresholdAvx3ParametersTest );
		ADD_TEST( framework, ThresholdAvx6ParametersTest );
		ADD_TEST( framework, ThresholdAvx9ParametersTest );
	}

	bool BitwiseAndAvx2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Avx::BitwiseAnd( input[0], input[1] );

			if( !equalSize( input[0], output ) || !verifyImage( output, intensityValue[0] & intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseAndAvx3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

			Image_Function_Avx::BitwiseAnd( image[0], image[1], image[2] );

			if( !verifyImage( image[2], intensityValue[0] & intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseAndAvx8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Avx::BitwiseAnd(
				input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensityValue[0] & intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseAndAvx11ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Avx::BitwiseAnd( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											image[2], roiX[2], roiY[2], roiWidth, roiHeight );

			if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensityValue[0] & intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseOrAvx2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Avx::BitwiseOr( input[0], input[1] );

			if( !equalSize( input[0], output ) || !verifyImage( output, intensityValue[0] | intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseOrAvx3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

			Image_Function_Avx::BitwiseOr( image[0], image[1], image[2] );

			if( !verifyImage( image[2], intensityValue[0] | intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseOrAvx8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Avx::BitwiseOr(
				input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensityValue[0] | intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseOrAvx11ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Avx::BitwiseOr( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
										   image[2], roiX[2], roiY[2], roiWidth, roiHeight );

			if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensityValue[0] | intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseXorAvx2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Avx::BitwiseXor( input[0], input[1] );

			if( !equalSize( input[0], output ) || !verifyImage( output, intensityValue[0] ^ intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseXorAvx3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

			Image_Function_Avx::BitwiseXor( image[0], image[1], image[2] );

			if( !verifyImage( image[2], intensityValue[0] ^ intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseXorAvx8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Avx::BitwiseXor(
				input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensityValue[0] ^ intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseXorAvx11ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Avx::BitwiseXor( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											image[2], roiX[2], roiY[2], roiWidth, roiHeight );

			if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensityValue[0] ^ intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool InvertAvx1ParameterTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 1 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Avx::Invert( input[0] );

			if( !verifyImage( output, ~intensityValue[0] ) )
				return false;
		}

		return true;
	}

	bool InvertAvx2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Image_Function_Avx::Invert( input[0], input[1] );

			if( !verifyImage( input[1], ~intensityValue[0] ) )
				return false;
		}

		return true;
	}

	bool InvertAvx5ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 1 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Avx::Invert( input[0], roiX[0], roiY[0], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, ~intensityValue[0] ) )
				return false;
		}

		return true;
	}

	bool InvertAvx8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Avx::Invert( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, ~intensityValue[0] ) )
				return false;
		}

		return true;
	}

	bool MaximumAvx2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Avx::Maximum( input[0], input[1] );

			if( !equalSize( input[0], output ) ||
				!verifyImage( output, intensityValue[0] > intensityValue[1] ? intensityValue[0] : intensityValue[1] ))
				return false;
		}

		return true;
	}

	bool MaximumAvx3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

			Image_Function_Avx::Maximum( image[0], image[1], image[2] );

			if( !verifyImage( image[2], intensityValue[0] > intensityValue[1] ? intensityValue[0] : intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool MaximumAvx8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Avx::Maximum(
				input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) ||
				!verifyImage( output, intensityValue[0] > intensityValue[1] ? intensityValue[0] : intensityValue[1] ))
				return false;
		}

		return true;
	}

	bool MaximumAvx11ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Avx::Maximum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
										 image[2], roiX[2], roiY[2], roiWidth, roiHeight );

			if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
				intensityValue[0] > intensityValue[1] ? intensityValue[0] : intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool MinimumAvx2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Avx::Minimum( input[0], input[1] );

			if( !equalSize( input[0], output ) ||
				!verifyImage( output, intensityValue[0] < intensityValue[1] ? intensityValue[0] : intensityValue[1] ))
				return false;
		}

		return true;
	}

	bool MinimumAvx3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

			Image_Function_Avx::Minimum( image[0], image[1], image[2] );

			if( !verifyImage( image[2], intensityValue[0] < intensityValue[1] ? intensityValue[0] : intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool MinimumAvx8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Avx::Minimum(
				input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) ||
				!verifyImage( output, intensityValue[0] < intensityValue[1] ? intensityValue[0] : intensityValue[1] ))
				return false;
		}

		return true;
	}

	bool MinimumAvx11ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Avx::Minimum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
										 image[2], roiX[2], roiY[2], roiWidth, roiHeight );

			if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
				intensityValue[0] < intensityValue[1] ? intensityValue[0] : intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool SubtractAvx2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Avx::Subtract( input[0], input[1] );

			if( !equalSize( input[0], output ) ||
				!verifyImage( output, intensityValue[0] > intensityValue[1] ? intensityValue[0] - intensityValue[1] : 0 ))
				return false;
		}

		return true;
	}

	bool SubtractAvx3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

			Image_Function_Avx::Subtract( image[0], image[1], image[2] );

			if( !verifyImage( image[2], intensityValue[0] > intensityValue[1] ? intensityValue[0] - intensityValue[1] : 0 ) )
				return false;
		}

		return true;
	}

	bool SubtractAvx8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Avx::Subtract(
				input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) ||
				!verifyImage( output, intensityValue[0] > intensityValue[1] ? intensityValue[0] - intensityValue[1] : 0 ))
				return false;
		}

		return true;
	}

	bool SubtractAvx11ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Avx::Subtract( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
										  image[2], roiX[2], roiY[2], roiWidth, roiHeight );

			if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
				intensityValue[0] > intensityValue[1] ? intensityValue[0] - intensityValue[1] : 0 ) )
				return false;
		}

		return true;
	}

	bool ThresholdAvx2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 1 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			uint8_t threshold = randomValue <uint8_t>( 255 );

			Bitmap_Image::Image output = Image_Function_Avx::Threshold( input[0], threshold );

			if( !verifyImage( output, intensityValue[0] < threshold ? 0 : 255 ) )
				return false;
		}

		return true;
	}

	bool ThresholdAvx3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			uint8_t threshold = randomValue <uint8_t>( 255 );

			Image_Function_Avx::Threshold( input[0], input[1], threshold );

			if( !verifyImage( input[1], intensityValue[0] < threshold ? 0 : 255 ) )
				return false;
		}

		return true;
	}

	bool ThresholdAvx6ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 1 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			uint8_t threshold = randomValue <uint8_t>( 255 );

			Bitmap_Image::Image output = Image_Function_Avx::Threshold( input[0], roiX[0], roiY[0], roiWidth, roiHeight, threshold );

			if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensityValue[0] < threshold ? 0 : 255 ) )
				return false;
		}

		return true;
	}

	bool ThresholdAvx9ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			uint8_t threshold = randomValue <uint8_t>( 255 );

			Image_Function_Avx::Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, threshold );

			if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, intensityValue[0] < threshold ? 0 : 255 ) )
				return false;
		}

		return true;
	}
};
