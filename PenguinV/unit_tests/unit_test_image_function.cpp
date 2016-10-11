#include <numeric>
#include "../Library/image_function.h"
#include "unit_test_image_function.h"
#include "unit_test_helper.h"

namespace Unit_Test
{
	void addTests_Image_Function(UnitTestFramework & framework)
	{
		ADD_TEST( framework, Image_Function_Test::BitwiseAnd2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::BitwiseAnd3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::BitwiseAnd8ParametersTest );
		ADD_TEST( framework, Image_Function_Test::BitwiseAnd11ParametersTest );

		ADD_TEST( framework, Image_Function_Test::BitwiseOr2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::BitwiseOr3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::BitwiseOr8ParametersTest );
		ADD_TEST( framework, Image_Function_Test::BitwiseOr11ParametersTest );

		ADD_TEST( framework, Image_Function_Test::BitwiseXor2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::BitwiseXor3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::BitwiseXor8ParametersTest );
		ADD_TEST( framework, Image_Function_Test::BitwiseXor11ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Copy2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Copy5ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Copy8ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Histogram1ParameterTest );
		ADD_TEST( framework, Image_Function_Test::Histogram2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Histogram4ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Histogram5ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Invert1ParameterTest );
		ADD_TEST( framework, Image_Function_Test::Invert2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Invert5ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Invert8ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Maximum2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Maximum3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Maximum8ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Maximum11ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Minimum2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Minimum3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Minimum8ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Minimum11ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Subtract2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Subtract3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Subtract8ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Subtract11ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Sum1ParameterTest );
		ADD_TEST( framework, Image_Function_Test::Sum5ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Threshold2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Threshold3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Threshold6ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Threshold9ParametersTest );

		ADD_TEST( framework, Image_Function_Test::ThresholdDouble3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::ThresholdDouble4ParametersTest );
		ADD_TEST( framework, Image_Function_Test::ThresholdDouble7ParametersTest );
		ADD_TEST( framework, Image_Function_Test::ThresholdDouble10ParametersTest );
	}

	namespace Image_Function_Test
	{
		bool BitwiseAnd2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				Bitmap_Image::Image output = Image_Function::BitwiseAnd( input[0], input[1] );

				if( !equalSize( input[0], output ) || !verifyImage( output, intensityValue[0] & intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseAnd3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

				Image_Function::BitwiseAnd( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensityValue[0] & intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseAnd8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::BitwiseAnd(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensityValue[0] & intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseAnd11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::BitwiseAnd( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensityValue[0] & intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				Bitmap_Image::Image output = Image_Function::BitwiseOr( input[0], input[1] );

				if( !equalSize( input[0], output ) || !verifyImage( output, intensityValue[0] | intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

				Image_Function::BitwiseOr( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensityValue[0] | intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::BitwiseOr(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensityValue[0] | intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::BitwiseOr( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
										   image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensityValue[0] | intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				Bitmap_Image::Image output = Image_Function::BitwiseXor( input[0], input[1] );

				if( !equalSize( input[0], output ) || !verifyImage( output, intensityValue[0] ^ intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

				Image_Function::BitwiseXor( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensityValue[0] ^ intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::BitwiseXor(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensityValue[0] ^ intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::BitwiseXor( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
										   image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensityValue[0] ^ intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool Copy2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				Image_Function::Copy( input[0], input[1] );

				if( !verifyImage( input[1], intensityValue[0] ) )
					return false;
			}

			return true;
		}

		bool Copy5ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 1 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::Copy( input[0], roiX[0], roiY[0], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensityValue[0] ) )
					return false;
			}

			return true;
		}

		bool Copy8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::Copy( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, intensityValue[0] ) )
					return false;
			}

			return true;
		}

		bool Histogram1ParameterTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 1 );
				Bitmap_Image::Image image = uniformImage( intensityValue[0] );

				std::vector < uint32_t > histogram = Image_Function::Histogram( image );

				if( histogram.size() != 256u || histogram[intensityValue[0]] != image.width() * image.height() ||
					std::accumulate(histogram.begin(), histogram.end(), 0u)  != image.width() * image.height() )
					return false;
			}

			return true;
		}

		bool Histogram2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 1 );
				Bitmap_Image::Image image = uniformImage( intensityValue[0] );

				std::vector < uint32_t > histogram;
				Image_Function::Histogram( image, histogram );

				if( histogram.size() != 256u || histogram[intensityValue[0]] != image.width() * image.height() ||
					std::accumulate(histogram.begin(), histogram.end(), 0u)  != image.width() * image.height() )
					return false;
			}

			return true;
		}

		bool Histogram4ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 1 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				std::vector < uint32_t > histogram = Image_Function::Histogram( input[0], roiX[0], roiY[0], roiWidth, roiHeight );

				if( histogram.size() != 256u || histogram[intensityValue[0]] != roiWidth * roiHeight ||
					std::accumulate(histogram.begin(), histogram.end(), 0u)  != roiWidth * roiHeight )
					return false;
			}

			return true;
		}

		bool Histogram5ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 1 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				std::vector < uint32_t > histogram;
				Image_Function::Histogram( input[0], roiX[0], roiY[0], roiWidth, roiHeight, histogram );

				if( histogram.size() != 256u || histogram[intensityValue[0]] != roiWidth * roiHeight ||
					std::accumulate(histogram.begin(), histogram.end(), 0u)  != roiWidth * roiHeight )
					return false;
			}

			return true;
		}

		bool Invert1ParameterTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 1 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				Bitmap_Image::Image output = Image_Function::Invert( input[0] );

				if( !verifyImage( output, ~intensityValue[0] ) )
					return false;
			}

			return true;
		}

		bool Invert2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				Image_Function::Invert( input[0], input[1] );

				if( !verifyImage( input[1], ~intensityValue[0] ) )
					return false;
			}

			return true;
		}

		bool Invert5ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 1 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::Invert( input[0], roiX[0], roiY[0], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, ~intensityValue[0] ) )
					return false;
			}

			return true;
		}

		bool Invert8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::Invert( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, ~intensityValue[0] ) )
					return false;
			}

			return true;
		}

		bool Maximum2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				Bitmap_Image::Image output = Image_Function::Maximum( input[0], input[1] );

				if( !equalSize( input[0], output ) ||
					!verifyImage( output, intensityValue[0] > intensityValue[1] ? intensityValue[0] : intensityValue[1] ))
					return false;
			}

			return true;
		}

		bool Maximum3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

				Image_Function::Maximum( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensityValue[0] > intensityValue[1] ? intensityValue[0] : intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool Maximum8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::Maximum(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) ||
					!verifyImage( output, intensityValue[0] > intensityValue[1] ? intensityValue[0] : intensityValue[1] ))
					return false;
			}

			return true;
		}

		bool Maximum11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::Maximum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
					intensityValue[0] > intensityValue[1] ? intensityValue[0] : intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool Minimum2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				Bitmap_Image::Image output = Image_Function::Minimum( input[0], input[1] );

				if( !equalSize( input[0], output ) ||
					!verifyImage( output, intensityValue[0] < intensityValue[1] ? intensityValue[0] : intensityValue[1] ))
					return false;
			}

			return true;
		}

		bool Minimum3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

				Image_Function::Minimum( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensityValue[0] < intensityValue[1] ? intensityValue[0] : intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool Minimum8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::Minimum(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) ||
					!verifyImage( output, intensityValue[0] < intensityValue[1] ? intensityValue[0] : intensityValue[1] ))
					return false;
			}

			return true;
		}

		bool Minimum11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::Minimum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
					intensityValue[0] < intensityValue[1] ? intensityValue[0] : intensityValue[1] ) )
					return false;
			}

			return true;
		}

		bool Subtract2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				Bitmap_Image::Image output = Image_Function::Subtract( input[0], input[1] );

				if( !equalSize( input[0], output ) ||
					!verifyImage( output, intensityValue[0] > intensityValue[1] ? intensityValue[0] - intensityValue[1] : 0 ))
					return false;
			}

			return true;
		}

		bool Subtract3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensityValue );

				Image_Function::Subtract( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensityValue[0] > intensityValue[1] ? intensityValue[0] - intensityValue[1] : 0 ) )
					return false;
			}

			return true;
		}

		bool Subtract8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::Subtract(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) ||
					!verifyImage( output, intensityValue[0] > intensityValue[1] ? intensityValue[0] - intensityValue[1] : 0 ))
					return false;
			}

			return true;
		}

		bool Subtract11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::Subtract( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
					intensityValue[0] > intensityValue[1] ? intensityValue[0] - intensityValue[1] : 0 ) )
					return false;
			}

			return true;
		}

		bool Sum1ParameterTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 1 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				if( Image_Function::Sum( input[0] ) != intensityValue[0] * input[0].width() * input[0].height() )
					return false;
			}

			return true;
		}

		bool Sum5ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 1 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				if( Image_Function::Sum( input[0], roiX[0], roiY[0], roiWidth, roiHeight ) != intensityValue[0] * roiWidth * roiHeight )
					return false;
			}

			return true;
		}

		bool Threshold2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 1 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				uint8_t threshold = randomValue <uint8_t>( 255 );

				Bitmap_Image::Image output = Image_Function::Threshold( input[0], threshold );

				if( !verifyImage( output, intensityValue[0] < threshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool Threshold3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				uint8_t threshold = randomValue <uint8_t>( 255 );

				Image_Function::Threshold( input[0], input[1], threshold );

				if( !verifyImage( input[1], intensityValue[0] < threshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool Threshold6ParametersTest()
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

				Bitmap_Image::Image output = Image_Function::Threshold( input[0], roiX[0], roiY[0], roiWidth, roiHeight, threshold );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensityValue[0] < threshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool Threshold9ParametersTest()
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

				Image_Function::Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, threshold );

				if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, intensityValue[0] < threshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool ThresholdDouble3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 1 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				uint8_t minThreshold = randomValue <uint8_t>( 255 );
				uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

				Bitmap_Image::Image output = Image_Function::Threshold( input[0], minThreshold, maxThreshold );

				if( !verifyImage( output, intensityValue[0] < minThreshold || intensityValue[0] > maxThreshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool ThresholdDouble4ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensityValue );

				uint8_t minThreshold = randomValue <uint8_t>( 255 );
				uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

				Image_Function::Threshold( input[0], input[1], minThreshold, maxThreshold );

				if( !verifyImage( input[1], intensityValue[0] < minThreshold || intensityValue[0] > maxThreshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool ThresholdDouble7ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 1 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				uint8_t minThreshold = randomValue <uint8_t>( 255 );
				uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

				Bitmap_Image::Image output = Image_Function::Threshold( input[0], roiX[0], roiY[0], roiWidth, roiHeight, minThreshold,
																		maxThreshold );

				if( !equalSize( output, roiWidth, roiHeight ) ||
					!verifyImage( output, intensityValue[0] < minThreshold || intensityValue[0] > maxThreshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool ThresholdDouble10ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensityValue = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensityValue.begin(), intensityValue.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				uint8_t minThreshold = randomValue <uint8_t>( 255 );
				uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

				Image_Function::Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, minThreshold,
										   maxThreshold);

				if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight,
					intensityValue[0] < minThreshold || intensityValue[0] > maxThreshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}
	};
};
