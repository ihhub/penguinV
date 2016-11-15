#include <numeric>
#include "../Library/image_function.h"
#include "unit_test_image_function.h"
#include "unit_test_helper.h"

namespace Unit_Test
{
	void addTests_Image_Function(UnitTestFramework & framework)
	{
		ADD_TEST( framework, Image_Function_Test::AbsoluteDifference2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::AbsoluteDifference3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::AbsoluteDifference8ParametersTest );
		ADD_TEST( framework, Image_Function_Test::AbsoluteDifference11ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Accumulate2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Accumulate6ParametersTest );

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

		ADD_TEST( framework, Image_Function_Test::ConvertToGray2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::ConvertToGray8ParametersTest );
		ADD_TEST( framework, Image_Function_Test::ConvertToColor2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::ConvertToColor8ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Fill2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Fill6ParametersTest );

		ADD_TEST( framework, Image_Function_Test::GammaCorrection3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::GammaCorrection4ParametersTest );
		ADD_TEST( framework, Image_Function_Test::GammaCorrection7ParametersTest );
		ADD_TEST( framework, Image_Function_Test::GammaCorrection10ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Histogram1ParameterTest );
		ADD_TEST( framework, Image_Function_Test::Histogram2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Histogram4ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Histogram5ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Invert1ParameterTest );
		ADD_TEST( framework, Image_Function_Test::Invert2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Invert5ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Invert8ParametersTest );

		ADD_TEST( framework, Image_Function_Test::IsEqual2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::IsEqual8ParametersTest );

		ADD_TEST( framework, Image_Function_Test::LookupTable2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::LookupTable3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::LookupTable6ParametersTest );
		ADD_TEST( framework, Image_Function_Test::LookupTable9ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Maximum2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Maximum3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Maximum8ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Maximum11ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Minimum2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Minimum3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Minimum8ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Minimum11ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Normalize1ParameterTest );
		ADD_TEST( framework, Image_Function_Test::Normalize2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Normalize5ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Normalize8ParametersTest );

		ADD_TEST( framework, Image_Function_Test::ProjectionProfile2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::ProjectionProfile3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::ProjectionProfile6ParametersTest );
		ADD_TEST( framework, Image_Function_Test::ProjectionProfile7ParametersTest );

		ADD_TEST( framework, Image_Function_Test::Resize2ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Resize3ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Resize7ParametersTest );
		ADD_TEST( framework, Image_Function_Test::Resize9ParametersTest );

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
		bool AbsoluteDifference2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Image_Function::AbsoluteDifference( input[0], input[1] );

				if( !equalSize( input[0], output ) ||
					!verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ))
					return false;
			}

			return true;
		}

		bool AbsoluteDifference3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Image_Function::AbsoluteDifference( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
					return false;
			}

			return true;
		}

		bool AbsoluteDifference8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::AbsoluteDifference(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) ||
					!verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ))
					return false;
			}

			return true;
		}

		bool AbsoluteDifference11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::AbsoluteDifference( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
					intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
					return false;
			}

			return true;
		}

		bool Accumulate2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( randomValue<uint8_t>(1, 16) );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				std::vector < uint32_t > result( input[0].width() * input[0].height(), 0 );

				for( std::vector < Bitmap_Image::Image >::const_iterator image = input.begin(); image != input.end(); ++image ) {
					Image_Function::Accumulate( *image, result );
				}

				uint32_t sum = std::accumulate( intensity.begin(), intensity.end(), 0u );

				if( std::any_of( result.begin(), result.end(), [&sum](uint32_t v){ return v != sum; } ) )
					return false;
			}

			return true;
		}

		bool Accumulate6ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( randomValue<uint8_t>(1, 16) );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				std::vector < uint32_t > result( roiWidth * roiHeight, 0 );

				for( size_t imageId = 0; imageId < input.size(); ++imageId ) {
					Image_Function::Accumulate( input[imageId], roiX[imageId], roiY[imageId], roiWidth, roiHeight, result );
				}

				uint32_t sum = std::accumulate( intensity.begin(), intensity.end(), 0u );

				if( std::any_of( result.begin(), result.end(), [&sum](uint32_t v){ return v != sum; } ) )
					return false;
			}

			return true;
		}

		bool BitwiseAnd2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Image_Function::BitwiseAnd( input[0], input[1] );

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

				Image_Function::BitwiseAnd( image[0], image[1], image[2] );

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

				Bitmap_Image::Image output = Image_Function::BitwiseAnd(
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

				Image_Function::BitwiseAnd( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
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

				Bitmap_Image::Image output = Image_Function::BitwiseOr( input[0], input[1] );

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

				Image_Function::BitwiseOr( image[0], image[1], image[2] );

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

				Bitmap_Image::Image output = Image_Function::BitwiseOr(
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

				Image_Function::BitwiseOr( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
										   image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] | intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Image_Function::BitwiseXor( input[0], input[1] );

				if( !equalSize( input[0], output ) || !verifyImage( output, intensity[0] ^ intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Image_Function::BitwiseXor( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] ^ intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::BitwiseXor(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity[0] ^ intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::BitwiseXor( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
										   image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] ^ intensity[1] ) )
					return false;
			}

			return true;
		}

		bool Copy2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Image_Function::Copy( input[0], input[1] );

				if( !verifyImage( input[1], intensity[0] ) )
					return false;
			}

			return true;
		}

		bool Copy5ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::Copy( input, roiX, roiY, roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity ) )
					return false;
			}

			return true;
		}

		bool Copy8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::Copy( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] ) )
					return false;
			}

			return true;
		}

		bool ConvertToGray2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				Bitmap_Image::ColorImage input = uniformColorImage( intensity[0] );
				Bitmap_Image::Image      output( input.width(), input.height() );
				
				output.fill( intensity[1] );

				Image_Function::Convert( input, output );

				if( !verifyImage( output, intensity[0] ) )
					return false;
			}

			return true;
		}

		bool ConvertToGray8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				Bitmap_Image::ColorImage input  = uniformColorImage( intensity[0] );
				Bitmap_Image::Image      output = uniformImage     ( intensity[1] );

				std::vector < std::pair <uint32_t, uint32_t> > size( 2 );

				size[0] = imageSize( input  );
				size[1] = imageSize( output );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( size, roiX, roiY, roiWidth, roiHeight );

				Image_Function::Convert( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight );

				if( !verifyImage( output, roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] ) )
					return false;
			}

			return true;
		}

		bool ConvertToColor2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				Bitmap_Image::Image      input = uniformImage( intensity[0] );
				Bitmap_Image::ColorImage output( input.width(), input.height() );
				
				output.fill( intensity[1] );

				Image_Function::Convert( input, output );

				if( !verifyImage( output, intensity[0] ) )
					return false;
			}

			return true;
		}

		bool ConvertToColor8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				Bitmap_Image::Image      input  = uniformImage     ( intensity[0] );
				Bitmap_Image::ColorImage output = uniformColorImage( intensity[1] );

				std::vector < std::pair <uint32_t, uint32_t> > size( 2 );

				size[0] = imageSize( input  );
				size[1] = imageSize( output );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( size, roiX, roiY, roiWidth, roiHeight );

				Image_Function::Convert( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight );

				if( !verifyImage( output, roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] ) )
					return false;
			}

			return true;
		}

		bool Fill2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray(2);
				Bitmap_Image::Image image = uniformImage( intensity[0] );

				Image_Function::Fill( image, intensity[1] );

				if( !verifyImage( image, intensity[1] ) )
					return false;
			}

			return true;
		}

		bool Fill6ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray(2);
				Bitmap_Image::Image image = uniformImage( intensity[0] );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::Fill( image, roiX, roiY, roiWidth, roiHeight, intensity[1] );

				if( !verifyImage( image, roiX, roiY, roiWidth, roiHeight, intensity[1] ) )
					return false;
			}

			return true;
		}

		bool GammaCorrection3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				double a     = randomValue <uint32_t>(100) / 100.0;
				double gamma = randomValue <uint32_t>(300) / 100.0;

				Bitmap_Image::Image output = Image_Function::GammaCorrection( input, a, gamma );

				double value = a * pow(intensity / 255.0, gamma) * 255 + 0.5;
				uint8_t corrected = 0;

				if( value < 256 )
					corrected = static_cast<uint8_t>( value );
				else
					corrected = 255;

				if( !verifyImage( output, corrected ) )
					return false;
			}

			return true;
		}

		bool GammaCorrection4ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				double a     = randomValue <uint32_t>(100) / 100.0;
				double gamma = randomValue <uint32_t>(300) / 100.0;

				Image_Function::GammaCorrection( input[0], input[1], a, gamma );

				double value = a * pow(intensity[0] / 255.0, gamma) * 255 + 0.5;
				uint8_t corrected = 0;

				if( value < 256 )
					corrected = static_cast<uint8_t>( value );
				else
					corrected = 255;

				if( !verifyImage( input[1], corrected ) )
					return false;
			}

			return true;
		}

		bool GammaCorrection7ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				double a     = randomValue <uint32_t>(100) / 100.0;
				double gamma = randomValue <uint32_t>(300) / 100.0;

				Bitmap_Image::Image output = Image_Function::GammaCorrection( input, roiX, roiY, roiWidth, roiHeight, a, gamma );

				double value = a * pow(intensity / 255.0, gamma) * 255 + 0.5;
				uint8_t corrected = 0;

				if( value < 256 )
					corrected = static_cast<uint8_t>( value );
				else
					corrected = 255;

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, corrected ) )
					return false;
			}

			return true;
		}

		bool GammaCorrection10ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				double a     = randomValue <uint32_t>(100) / 100.0;
				double gamma = randomValue <uint32_t>(300) / 100.0;

				Image_Function::GammaCorrection( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, a, gamma );

				double value = a * pow(intensity[0] / 255.0, gamma) * 255 + 0.5;
				uint8_t corrected = 0;

				if( value < 256 )
					corrected = static_cast<uint8_t>( value );
				else
					corrected = 255;

				if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, corrected ) )
					return false;
			}

			return true;
		}

		bool Histogram1ParameterTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image image = uniformImage( intensity );

				std::vector < uint32_t > histogram = Image_Function::Histogram( image );

				if( histogram.size() != 256u || histogram[intensity] != image.width() * image.height() ||
					std::accumulate(histogram.begin(), histogram.end(), 0u)  != image.width() * image.height() )
					return false;
			}

			return true;
		}

		bool Histogram2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image image = uniformImage( intensity );

				std::vector < uint32_t > histogram;
				Image_Function::Histogram( image, histogram );

				if( histogram.size() != 256u || histogram[intensity] != image.width() * image.height() ||
					std::accumulate(histogram.begin(), histogram.end(), 0u)  != image.width() * image.height() )
					return false;
			}

			return true;
		}

		bool Histogram4ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				std::vector < uint32_t > histogram = Image_Function::Histogram( input, roiX, roiY, roiWidth, roiHeight );

				if( histogram.size() != 256u || histogram[intensity] != roiWidth * roiHeight ||
					std::accumulate(histogram.begin(), histogram.end(), 0u)  != roiWidth * roiHeight )
					return false;
			}

			return true;
		}

		bool Histogram5ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				std::vector < uint32_t > histogram;
				Image_Function::Histogram( input, roiX, roiY, roiWidth, roiHeight, histogram );

				if( histogram.size() != 256u || histogram[intensity] != roiWidth * roiHeight ||
					std::accumulate(histogram.begin(), histogram.end(), 0u)  != roiWidth * roiHeight )
					return false;
			}

			return true;
		}

		bool Invert1ParameterTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				Bitmap_Image::Image output = Image_Function::Invert( input );

				if( !verifyImage( output, ~intensity ) )
					return false;
			}

			return true;
		}

		bool Invert2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Image_Function::Invert( input[0], input[1] );

				if( !verifyImage( input[1], ~intensity[0] ) )
					return false;
			}

			return true;
		}

		bool Invert5ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::Invert( input, roiX, roiY, roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, ~intensity ) )
					return false;
			}

			return true;
		}

		bool Invert8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::Invert( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, ~intensity[0] ) )
					return false;
			}

			return true;
		}

		bool IsEqual2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				if( (intensity[0] == intensity[1]) != (Image_Function::IsEqual( input[0], input[1] )) )
					return false;
			}

			return true;
		}

		bool IsEqual8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				if( (intensity[0] == intensity[1]) !=
					(Image_Function::IsEqual( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight )) )
					return false;
			}

			return true;
		}

		bool LookupTable2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray(2);
				Bitmap_Image::Image input = randomImage( intensity );

				std::vector < uint8_t > lookupTable( 256, 0 );

				lookupTable[intensity[0]] = intensityValue();
				lookupTable[intensity[1]] = intensityValue();

				Bitmap_Image::Image output = Image_Function::LookupTable( input, lookupTable );

				std::vector < uint8_t > normalized( 2 );

				normalized[0] = lookupTable[intensity[0]];
				normalized[1] = lookupTable[intensity[1]];

				if( !verifyImage( output, normalized ) )
					return false;
			}

			return true;
		}

		bool LookupTable3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				Bitmap_Image::Image input  = randomImage( intensity );
				Bitmap_Image::Image output( input.width(), input.height() );

				output.fill( intensityValue() );

				std::vector < uint8_t > lookupTable( 256, 0 );

				lookupTable[intensity[0]] = intensityValue();
				lookupTable[intensity[1]] = intensityValue();

				Image_Function::LookupTable( input, output, lookupTable );

				std::vector < uint8_t > normalized( 2 );

				normalized[0] = lookupTable[intensity[0]];
				normalized[1] = lookupTable[intensity[1]];

				if( !verifyImage( output, normalized ) )
					return false;
			}

			return true;
		}

		bool LookupTable6ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray(2);
				Bitmap_Image::Image input = uniformImage();

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				fillImage( input, roiX, roiY, roiWidth, roiHeight, intensity );

				std::vector < uint8_t > lookupTable( 256, 0 );

				lookupTable[intensity[0]] = intensityValue();
				lookupTable[intensity[1]] = intensityValue();

				Bitmap_Image::Image output = Image_Function::LookupTable( input, roiX, roiY, roiWidth, roiHeight, lookupTable );

				std::vector < uint8_t > normalized( 2 );

				normalized[0] = lookupTable[intensity[0]];
				normalized[1] = lookupTable[intensity[1]];

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, normalized ) )
					return false;
			}

			return true;
		}

		bool LookupTable9ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				Bitmap_Image::Image input  = uniformImage();
				Bitmap_Image::Image output = uniformImage();

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				std::vector < std::pair < uint32_t, uint32_t > > size(2);

				size[0] = imageSize( input  );
				size[1] = imageSize( output );

				generateRoi( size, roiX, roiY, roiWidth, roiHeight );

				fillImage( input, roiX[0], roiY[0], roiWidth, roiHeight, intensity );

				std::vector < uint8_t > lookupTable( 256, 0 );

				lookupTable[intensity[0]] = intensityValue();
				lookupTable[intensity[1]] = intensityValue();

				Image_Function::LookupTable( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight, lookupTable );

				std::vector < uint8_t > normalized( 2 );

				normalized[0] = lookupTable[intensity[0]];
				normalized[1] = lookupTable[intensity[1]];

				if( !verifyImage( output, roiX[1], roiY[1], roiWidth, roiHeight, normalized ) )
					return false;
			}

			return true;
		}

		bool Maximum2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Image_Function::Maximum( input[0], input[1] );

				if( !equalSize( input[0], output ) ||
					!verifyImage( output, intensity[0] > intensity[1] ? intensity[0] : intensity[1] ))
					return false;
			}

			return true;
		}

		bool Maximum3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Image_Function::Maximum( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
					return false;
			}

			return true;
		}

		bool Maximum8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::Maximum(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) ||
					!verifyImage( output, intensity[0] > intensity[1] ? intensity[0] : intensity[1] ))
					return false;
			}

			return true;
		}

		bool Maximum11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::Maximum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
					intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
					return false;
			}

			return true;
		}

		bool Minimum2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Image_Function::Minimum( input[0], input[1] );

				if( !equalSize( input[0], output ) ||
					!verifyImage( output, intensity[0] < intensity[1] ? intensity[0] : intensity[1] ))
					return false;
			}

			return true;
		}

		bool Minimum3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Image_Function::Minimum( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
					return false;
			}

			return true;
		}

		bool Minimum8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::Minimum(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) ||
					!verifyImage( output, intensity[0] < intensity[1] ? intensity[0] : intensity[1] ))
					return false;
			}

			return true;
		}

		bool Minimum11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::Minimum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
					intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
					return false;
			}

			return true;
		}

		bool Normalize1ParameterTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray(2);
				Bitmap_Image::Image input = randomImage( intensity );

				Bitmap_Image::Image output = Image_Function::Normalize( input );

				std::vector < uint8_t > normalized( 2 );

				if( intensity[0] == intensity[1] || (input.width() == 1 && input.height() == 1) ) {
					normalized[0] = normalized[1] = intensity[0];
				}
				else {
					normalized[0] = 0;
					normalized[1] = 255;
				}

				if( !verifyImage( output, normalized ) )
					return false;
			}

			return true;
		}

		bool Normalize2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				Bitmap_Image::Image input  = randomImage( intensity );
				Bitmap_Image::Image output( input.width(), input.height() );

				output.fill( intensityValue() );

				Image_Function::Normalize( input, output );

				std::vector < uint8_t > normalized( 2 );

				if( intensity[0] == intensity[1] || (input.width() == 1 && input.height() == 1) ) {
					normalized[0] = normalized[1] = intensity[0];
				}
				else {
					normalized[0] = 0;
					normalized[1] = 255;
				}

				if( !verifyImage( output, normalized ) )
					return false;
			}

			return true;
		}

		bool Normalize5ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray(2);
				Bitmap_Image::Image input = uniformImage();

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				fillImage( input, roiX, roiY, roiWidth, roiHeight, intensity );

				Bitmap_Image::Image output = Image_Function::Normalize( input, roiX, roiY, roiWidth, roiHeight );

				std::vector < uint8_t > normalized( 2 );

				if( intensity[0] == intensity[1] || (roiWidth == 1 && roiHeight == 1) ) {
					normalized[0] = normalized[1] = intensity[0];
				}
				else {
					normalized[0] = 0;
					normalized[1] = 255;
				}

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, normalized ) )
					return false;
			}

			return true;
		}

		bool Normalize8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				Bitmap_Image::Image input  = uniformImage();
				Bitmap_Image::Image output = uniformImage();

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				std::vector < std::pair < uint32_t, uint32_t > > size(2);

				size[0] = imageSize( input  );
				size[1] = imageSize( output );

				generateRoi( size, roiX, roiY, roiWidth, roiHeight );

				fillImage( input, roiX[0], roiY[0], roiWidth, roiHeight, intensity );

				Image_Function::Normalize( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight );

				std::vector < uint8_t > normalized( 2 );

				if( intensity[0] == intensity[1] || (roiWidth == 1 && roiHeight == 1) ) {
					normalized[0] = normalized[1] = intensity[0];
				}
				else {
					normalized[0] = 0;
					normalized[1] = 255;
				}

				if( !verifyImage( output, roiX[1], roiY[1], roiWidth, roiHeight, normalized ) )
					return false;
			}

			return true;
		}

		bool ProjectionProfile2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image image = uniformImage( intensity );

				bool horizontal = (i % 2 == 0);

				std::vector < uint32_t > projection = Image_Function::ProjectionProfile( image, horizontal );

				uint32_t value = (horizontal ? image.height() : image.width()) * intensity;

				if( projection.size() != (horizontal ? image.width() : image.height()) ||
					std::any_of( projection.begin(), projection.end(), [&value] ( uint32_t v ) { return value != v; } ) )
					return false;
			}

			return true;
		}

		bool ProjectionProfile3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image image = uniformImage( intensity );

				bool horizontal = (i % 2 == 0);

				std::vector < uint32_t > projection;
				
				Image_Function::ProjectionProfile( image, horizontal, projection );

				uint32_t value = (horizontal ? image.height() : image.width()) * intensity;

				if( projection.size() != (horizontal ? image.width() : image.height()) ||
					std::any_of( projection.begin(), projection.end(), [&value] ( uint32_t v ) { return value != v; } ) )
					return false;
			}

			return true;
		}

		bool ProjectionProfile6ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image image = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				bool horizontal = (i % 2 == 0);

				std::vector < uint32_t > projection = Image_Function::ProjectionProfile( image, roiX, roiY, roiWidth, roiHeight, horizontal );

				uint32_t value = (horizontal ? roiHeight : roiWidth) * intensity;

				if( projection.size() != (horizontal ? roiWidth : roiHeight) ||
					std::any_of( projection.begin(), projection.end(), [&value] ( uint32_t v ) { return value != v; } ) )
					return false;
			}

			return true;
		}

		bool ProjectionProfile7ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image image = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				bool horizontal = (i % 2 == 0);

				std::vector < uint32_t > projection;
				
				Image_Function::ProjectionProfile( image, roiX, roiY, roiWidth, roiHeight, horizontal, projection );

				uint32_t value = (horizontal ? roiHeight : roiWidth) * intensity;

				if( projection.size() != (horizontal ? roiWidth : roiHeight) ||
					std::any_of( projection.begin(), projection.end(), [&value] ( uint32_t v ) { return value != v; } ) )
					return false;
			}

			return true;
		}

		bool Resize2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t outputWidth  = randomValue<uint32_t>(1, 2048);
				uint32_t outputHeight = randomValue<uint32_t>(1, 2048);

				Bitmap_Image::Image output = Image_Function::Resize( input, outputWidth, outputHeight );

				if( !equalSize( output, outputWidth, outputHeight ) || !verifyImage( output, intensity ) )
					return false;
			}

			return true;
		}

		bool Resize3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				Bitmap_Image::Image input  = uniformImage( intensity[0] );
				Bitmap_Image::Image output = uniformImage( intensity[1] );

				Image_Function::Resize( input, output );

				if( !verifyImage( output, intensity[0] ) )
					return false;
			}

			return true;
		}

		bool Resize7ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );
				
				uint32_t outputWidth  = randomValue<uint32_t>(1, 2048);
				uint32_t outputHeight = randomValue<uint32_t>(1, 2048);

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::Resize( input, roiX, roiY, roiWidth, roiHeight, outputWidth, outputHeight );

				if( !equalSize( output, outputWidth, outputHeight ) || !verifyImage( output, intensity ) )
					return false;
			}

			return true;
		}

		bool Resize9ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				Bitmap_Image::Image input  = uniformImage( intensity[0] );
				Bitmap_Image::Image output = uniformImage( intensity[1] );

				std::vector < uint32_t > roiX(2), roiY(2), roiWidth(2), roiHeight(2);

				generateRoi( input , roiX[0], roiY[0], roiWidth[0], roiHeight[0] );
				generateRoi( output, roiX[1], roiY[1], roiWidth[1], roiHeight[1] );

				Image_Function::Resize( input , roiX[0], roiY[0], roiWidth[0], roiHeight[0],
										output, roiX[1], roiY[1], roiWidth[1], roiHeight[1] );

				if( !verifyImage( output, roiX[1], roiY[1], roiWidth[1], roiHeight[1], intensity[0] ) )
					return false;
			}

			return true;
		}

		bool Subtract2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Image_Function::Subtract( input[0], input[1] );

				if( !equalSize( input[0], output ) ||
					!verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ))
					return false;
			}

			return true;
		}

		bool Subtract3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Image_Function::Subtract( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ) )
					return false;
			}

			return true;
		}

		bool Subtract8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Image_Function::Subtract(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) ||
					!verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ))
					return false;
			}

			return true;
		}

		bool Subtract11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Image_Function::Subtract( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
					intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ) )
					return false;
			}

			return true;
		}

		bool Sum1ParameterTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				if( Image_Function::Sum( input ) != intensity * input.width() * input.height() )
					return false;
			}

			return true;
		}

		bool Sum5ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				if( Image_Function::Sum( input, roiX, roiY, roiWidth, roiHeight ) != intensity * roiWidth * roiHeight )
					return false;
			}

			return true;
		}

		bool Threshold2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint8_t threshold = randomValue <uint8_t>( 255 );

				Bitmap_Image::Image output = Image_Function::Threshold( input, threshold );

				if( !verifyImage( output, intensity < threshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool Threshold3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				uint8_t threshold = randomValue <uint8_t>( 255 );

				Image_Function::Threshold( input[0], input[1], threshold );

				if( !verifyImage( input[1], intensity[0] < threshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool Threshold6ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				uint8_t threshold = randomValue <uint8_t>( 255 );

				Bitmap_Image::Image output = Image_Function::Threshold( input, roiX, roiY, roiWidth, roiHeight, threshold );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity < threshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool Threshold9ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				uint8_t threshold = randomValue <uint8_t>( 255 );

				Image_Function::Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, threshold );

				if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] < threshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool ThresholdDouble3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint8_t minThreshold = randomValue <uint8_t>( 255 );
				uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

				Bitmap_Image::Image output = Image_Function::Threshold( input, minThreshold, maxThreshold );

				if( !verifyImage( output, intensity < minThreshold || intensity > maxThreshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool ThresholdDouble4ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				uint8_t minThreshold = randomValue <uint8_t>( 255 );
				uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

				Image_Function::Threshold( input[0], input[1], minThreshold, maxThreshold );

				if( !verifyImage( input[1], intensity[0] < minThreshold || intensity[0] > maxThreshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool ThresholdDouble7ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				uint8_t minThreshold = randomValue <uint8_t>( 255 );
				uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

				Bitmap_Image::Image output = Image_Function::Threshold( input, roiX, roiY, roiWidth, roiHeight, minThreshold,
																		maxThreshold );

				if( !equalSize( output, roiWidth, roiHeight ) ||
					!verifyImage( output, intensity < minThreshold || intensity > maxThreshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool ThresholdDouble10ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				uint8_t minThreshold = randomValue <uint8_t>( 255 );
				uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

				Image_Function::Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, minThreshold,
										   maxThreshold);

				if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight,
					intensity[0] < minThreshold || intensity[0] > maxThreshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}
	};
};
