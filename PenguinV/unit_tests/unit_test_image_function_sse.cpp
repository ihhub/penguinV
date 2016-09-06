#include "../Library/image_function_sse.h"
#include "unit_test_image_function_sse.h"
#include "unit_test_helper.h"

namespace Unit_Test
{
	void addTests_Image_Function_Sse(UnitTestFramework & framework)
	{
		ADD_TEST( framework, BitwiseAndSse2ParametersTest );
		ADD_TEST( framework, BitwiseAndSse3ParametersTest );
		ADD_TEST( framework, BitwiseAndSse8ParametersTest );
		ADD_TEST( framework, BitwiseAndSse11ParametersTest );

		ADD_TEST( framework, BitwiseOrSse2ParametersTest );
		ADD_TEST( framework, BitwiseOrSse3ParametersTest );
		ADD_TEST( framework, BitwiseOrSse8ParametersTest );
		ADD_TEST( framework, BitwiseOrSse11ParametersTest );

		ADD_TEST( framework, BitwiseXorSse2ParametersTest );
		ADD_TEST( framework, BitwiseXorSse3ParametersTest );
		ADD_TEST( framework, BitwiseXorSse8ParametersTest );
		ADD_TEST( framework, BitwiseXorSse11ParametersTest );

		ADD_TEST( framework, InvertSse1ParameterTest );
		ADD_TEST( framework, InvertSse2ParametersTest );
		ADD_TEST( framework, InvertSse5ParametersTest );
		ADD_TEST( framework, InvertSse8ParametersTest );

		ADD_TEST( framework, MaximumSse2ParametersTest );
		ADD_TEST( framework, MaximumSse3ParametersTest );
		ADD_TEST( framework, MaximumSse8ParametersTest );
		ADD_TEST( framework, MaximumSse11ParametersTest );

		ADD_TEST( framework, MinimumSse2ParametersTest );
		ADD_TEST( framework, MinimumSse3ParametersTest );
		ADD_TEST( framework, MinimumSse8ParametersTest );
		ADD_TEST( framework, MinimumSse11ParametersTest );

		ADD_TEST( framework, SubtractSse2ParametersTest );
		ADD_TEST( framework, SubtractSse3ParametersTest );
		ADD_TEST( framework, SubtractSse8ParametersTest );
		ADD_TEST( framework, SubtractSse11ParametersTest );

		ADD_TEST( framework, ThresholdSse2ParametersTest );
		ADD_TEST( framework, ThresholdSse3ParametersTest );
		ADD_TEST( framework, ThresholdSse6ParametersTest );
		ADD_TEST( framework, ThresholdSse9ParametersTest );
	}

	bool BitwiseAndSse2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Sse::BitwiseAnd( input[0], input[1] );

			if( !equalSize( input[0], output ) || !verifyImage( output, intensityValue[0] & intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseAndSse3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

			Image_Function_Sse::BitwiseAnd( image[0], image[1], image[2] );

			if( !verifyImage( image[2], intensityValue[0] & intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseAndSse8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Sse::BitwiseAnd(
				input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensityValue[0] & intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseAndSse11ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Sse::BitwiseAnd( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											image[2], roiX[2], roiY[2], roiWidth, roiHeight );

			if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensityValue[0] & intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseOrSse2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Sse::BitwiseOr( input[0], input[1] );

			if( !equalSize( input[0], output ) || !verifyImage( output, intensityValue[0] | intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseOrSse3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

			Image_Function_Sse::BitwiseOr( image[0], image[1], image[2] );

			if( !verifyImage( image[2], intensityValue[0] | intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseOrSse8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Sse::BitwiseOr(
				input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensityValue[0] | intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseOrSse11ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Sse::BitwiseOr( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
										   image[2], roiX[2], roiY[2], roiWidth, roiHeight );

			if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensityValue[0] | intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseXorSse2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Sse::BitwiseXor( input[0], input[1] );

			if( !equalSize( input[0], output ) || !verifyImage( output, intensityValue[0] ^ intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseXorSse3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

			Image_Function_Sse::BitwiseXor( image[0], image[1], image[2] );

			if( !verifyImage( image[2], intensityValue[0] ^ intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseXorSse8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Sse::BitwiseXor(
				input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensityValue[0] ^ intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool BitwiseXorSse11ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Sse::BitwiseXor( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											image[2], roiX[2], roiY[2], roiWidth, roiHeight );

			if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensityValue[0] ^ intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool InvertSse1ParameterTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 1 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Sse::Invert( input[0] );

			if( !verifyImage( output, ~intensityValue[0] ) )
				return false;
		}

		return true;
	}

	bool InvertSse2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Image_Function_Sse::Invert( input[0], input[1] );

			if( !verifyImage( input[1], ~intensityValue[0] ) )
				return false;
		}

		return true;
	}

	bool InvertSse5ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 1 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Sse::Invert( input[0], roiX[0], roiY[0], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, ~intensityValue[0] ) )
				return false;
		}

		return true;
	}

	bool InvertSse8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Sse::Invert( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, ~intensityValue[0] ) )
				return false;
		}

		return true;
	}

	bool MaximumSse2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Sse::Maximum( input[0], input[1] );

			if( !equalSize( input[0], output ) ||
				!verifyImage( output, intensityValue[0] > intensityValue[1] ? intensityValue[0] : intensityValue[1] ))
				return false;
		}

		return true;
	}

	bool MaximumSse3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

			Image_Function_Sse::Maximum( image[0], image[1], image[2] );

			if( !verifyImage( image[2], intensityValue[0] > intensityValue[1] ? intensityValue[0] : intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool MaximumSse8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Sse::Maximum(
				input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) ||
				!verifyImage( output, intensityValue[0] > intensityValue[1] ? intensityValue[0] : intensityValue[1] ))
				return false;
		}

		return true;
	}

	bool MaximumSse11ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Sse::Maximum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
										 image[2], roiX[2], roiY[2], roiWidth, roiHeight );

			if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
				intensityValue[0] > intensityValue[1] ? intensityValue[0] : intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool MinimumSse2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Sse::Minimum( input[0], input[1] );

			if( !equalSize( input[0], output ) ||
				!verifyImage( output, intensityValue[0] < intensityValue[1] ? intensityValue[0] : intensityValue[1] ))
				return false;
		}

		return true;
	}

	bool MinimumSse3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

			Image_Function_Sse::Minimum( image[0], image[1], image[2] );

			if( !verifyImage( image[2], intensityValue[0] < intensityValue[1] ? intensityValue[0] : intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool MinimumSse8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Sse::Minimum(
				input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) ||
				!verifyImage( output, intensityValue[0] < intensityValue[1] ? intensityValue[0] : intensityValue[1] ))
				return false;
		}

		return true;
	}

	bool MinimumSse11ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Sse::Minimum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
										 image[2], roiX[2], roiY[2], roiWidth, roiHeight );

			if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
				intensityValue[0] < intensityValue[1] ? intensityValue[0] : intensityValue[1] ) )
				return false;
		}

		return true;
	}

	bool SubtractSse2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			Bitmap_Image::Image output = Image_Function_Sse::Subtract( input[0], input[1] );

			if( !equalSize( input[0], output ) ||
				!verifyImage( output, intensityValue[0] > intensityValue[1] ? intensityValue[0] - intensityValue[1] : 0 ))
				return false;
		}

		return true;
	}

	bool SubtractSse3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

			Image_Function_Sse::Subtract( image[0], image[1], image[2] );

			if( !verifyImage( image[2], intensityValue[0] > intensityValue[1] ? intensityValue[0] - intensityValue[1] : 0 ) )
				return false;
		}

		return true;
	}

	bool SubtractSse8ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ input.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( input, roiX, roiY, roiWidth, roiHeight );

			Bitmap_Image::Image output = Image_Function_Sse::Subtract(
				input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

			if( !equalSize( output, roiWidth, roiHeight ) ||
				!verifyImage( output, intensityValue[0] > intensityValue[1] ? intensityValue[0] - intensityValue[1] : 0 ))
				return false;
		}

		return true;
	}

	bool SubtractSse11ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 3 );
			std::vector < Bitmap_Image::Image > image;

			std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
				{ image.push_back( uniformImage( value ) ); } );

			std::vector < uint32_t > roiX, roiY;
			uint32_t roiWidth, roiHeight;

			generateRoi( image, roiX, roiY, roiWidth, roiHeight );

			Image_Function_Sse::Subtract( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
										  image[2], roiX[2], roiY[2], roiWidth, roiHeight );

			if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
				intensityValue[0] > intensityValue[1] ? intensityValue[0] - intensityValue[1] : 0 ) )
				return false;
		}

		return true;
	}

	bool ThresholdSse2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 1 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			uint8_t threshold = randomValue <uint8_t>( 255 );

			Bitmap_Image::Image output = Image_Function_Sse::Threshold( input[0], threshold );

			if( !verifyImage( output, intensityValue[0] < threshold ? 0 : 255 ) )
				return false;
		}

		return true;
	}

	bool ThresholdSse3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			std::vector < uint8_t > intensityValue = intensityArray( 2 );
			std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

			uint8_t threshold = randomValue <uint8_t>( 255 );

			Image_Function_Sse::Threshold( input[0], input[1], threshold );

			if( !verifyImage( input[1], intensityValue[0] < threshold ? 0 : 255 ) )
				return false;
		}

		return true;
	}

	bool ThresholdSse6ParametersTest()
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

			Bitmap_Image::Image output = Image_Function_Sse::Threshold( input[0], roiX[0], roiY[0], roiWidth, roiHeight, threshold );

			if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensityValue[0] < threshold ? 0 : 255 ) )
				return false;
		}

		return true;
	}

	bool ThresholdSse9ParametersTest()
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

			Image_Function_Sse::Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, threshold );

			if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, intensityValue[0] < threshold ? 0 : 255 ) )
				return false;
		}

		return true;
	}
};
