#include "../Library/function_pool.h"
#include "../Library/thread_pool.h"
#include "unit_test_function_pool.h"
#include "unit_test_helper.h"

namespace Unit_Test
{
	void addTests_Function_Pool(UnitTestFramework & framework)
	{
		ADD_TEST( framework, Function_Pool_Test::AbsoluteDifference2ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::AbsoluteDifference3ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::AbsoluteDifference8ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::AbsoluteDifference11ParametersTest );

		ADD_TEST( framework, Function_Pool_Test::BitwiseAnd2ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::BitwiseAnd3ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::BitwiseAnd8ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::BitwiseAnd11ParametersTest );

		ADD_TEST( framework, Function_Pool_Test::BitwiseOr2ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::BitwiseOr3ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::BitwiseOr8ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::BitwiseOr11ParametersTest );

		ADD_TEST( framework, Function_Pool_Test::BitwiseXor2ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::BitwiseXor3ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::BitwiseXor8ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::BitwiseXor11ParametersTest );

		ADD_TEST( framework, Function_Pool_Test::GammaCorrection3ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::GammaCorrection4ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::GammaCorrection7ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::GammaCorrection10ParametersTest );

		ADD_TEST( framework, Function_Pool_Test::Invert1ParameterTest );
		ADD_TEST( framework, Function_Pool_Test::Invert2ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Invert5ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Invert8ParametersTest );

		ADD_TEST( framework, Function_Pool_Test::IsEqual2ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::IsEqual8ParametersTest );

		ADD_TEST( framework, Function_Pool_Test::Maximum2ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Maximum3ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Maximum8ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Maximum11ParametersTest );

		ADD_TEST( framework, Function_Pool_Test::Minimum2ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Minimum3ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Minimum8ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Minimum11ParametersTest );

		ADD_TEST( framework, Function_Pool_Test::Subtract2ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Subtract3ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Subtract8ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Subtract11ParametersTest );

		ADD_TEST( framework, Function_Pool_Test::Sum1ParameterTest );
		ADD_TEST( framework, Function_Pool_Test::Sum5ParametersTest );

		ADD_TEST( framework, Function_Pool_Test::Threshold2ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Threshold3ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Threshold6ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::Threshold9ParametersTest );

		ADD_TEST( framework, Function_Pool_Test::ThresholdDouble3ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::ThresholdDouble4ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::ThresholdDouble7ParametersTest );
		ADD_TEST( framework, Function_Pool_Test::ThresholdDouble10ParametersTest );
	}

	namespace Function_Pool_Test
	{
		bool AbsoluteDifference2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Function_Pool::AbsoluteDifference( input[0], input[1] );

				if( !equalSize( input[0], output ) ||
					!verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ))
					return false;
			}

			return true;
		}

		bool AbsoluteDifference3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Function_Pool::AbsoluteDifference( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
					return false;
			}

			return true;
		}

		bool AbsoluteDifference8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Function_Pool::AbsoluteDifference(
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
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Function_Pool::AbsoluteDifference( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											  image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
					intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
					return false;
			}

			return true;
		}

		bool BitwiseAnd2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Function_Pool::BitwiseAnd( input[0], input[1] );

				if( !equalSize( input[0], output ) || !verifyImage( output, intensity[0] & intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseAnd3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Function_Pool::BitwiseAnd( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] & intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseAnd8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Function_Pool::BitwiseAnd(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity[0] & intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseAnd11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Function_Pool::BitwiseAnd( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
												image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] & intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Function_Pool::BitwiseOr( input[0], input[1] );

				if( !equalSize( input[0], output ) || !verifyImage( output, intensity[0] | intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Function_Pool::BitwiseOr( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] | intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Function_Pool::BitwiseOr(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity[0] | intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseOr11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Function_Pool::BitwiseOr( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											   image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] | intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Function_Pool::BitwiseXor( input[0], input[1] );

				if( !equalSize( input[0], output ) || !verifyImage( output, intensity[0] ^ intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Function_Pool::BitwiseXor( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] ^ intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Function_Pool::BitwiseXor(
					input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity[0] ^ intensity[1] ) )
					return false;
			}

			return true;
		}

		bool BitwiseXor11ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Function_Pool::BitwiseXor( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
												image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] ^ intensity[1] ) )
					return false;
			}

			return true;
		}

		bool GammaCorrection3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				double a     = randomValue <uint32_t>(100) / 100.0;
				double gamma = randomValue <uint32_t>(300) / 100.0;

				Bitmap_Image::Image output = Function_Pool::GammaCorrection( input, a, gamma );

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
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				double a     = randomValue <uint32_t>(100) / 100.0;
				double gamma = randomValue <uint32_t>(300) / 100.0;

				Function_Pool::GammaCorrection( input[0], input[1], a, gamma );

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
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				double a     = randomValue <uint32_t>(100) / 100.0;
				double gamma = randomValue <uint32_t>(300) / 100.0;

				Bitmap_Image::Image output = Function_Pool::GammaCorrection( input, roiX, roiY, roiWidth, roiHeight, a, gamma );

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
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				double a     = randomValue <uint32_t>(100) / 100.0;
				double gamma = randomValue <uint32_t>(300) / 100.0;

				Function_Pool::GammaCorrection( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, a, gamma );

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

		bool Invert1ParameterTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				Bitmap_Image::Image output = Function_Pool::Invert( input );

				if( !verifyImage( output, ~intensity ) )
					return false;
			}

			return true;
		}

		bool Invert2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Function_Pool::Invert( input[0], input[1] );

				if( !verifyImage( input[1], ~intensity[0] ) )
					return false;
			}

			return true;
		}

		bool Invert5ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Function_Pool::Invert( input, roiX, roiY, roiWidth, roiHeight );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, ~intensity ) )
					return false;
			}

			return true;
		}

		bool Invert8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Function_Pool::Invert( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

				if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, ~intensity[0] ) )
					return false;
			}

			return true;
		}

		bool IsEqual2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				if( (intensity[0] == intensity[1]) != (Function_Pool::IsEqual( input[0], input[1] )) )
					return false;
			}

			return true;
		}

		bool IsEqual8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				if( (intensity[0] == intensity[1]) !=
					(Function_Pool::IsEqual( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight )) )
					return false;
			}

			return true;
		}

		bool Maximum2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Function_Pool::Maximum( input[0], input[1] );

				if( !equalSize( input[0], output ) ||
					!verifyImage( output, intensity[0] > intensity[1] ? intensity[0] : intensity[1] ))
					return false;
			}

			return true;
		}

		bool Maximum3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Function_Pool::Maximum( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
					return false;
			}

			return true;
		}

		bool Maximum8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Function_Pool::Maximum(
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
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Function_Pool::Maximum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
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
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Function_Pool::Minimum( input[0], input[1] );

				if( !equalSize( input[0], output ) ||
					!verifyImage( output, intensity[0] < intensity[1] ? intensity[0] : intensity[1] ))
					return false;
			}

			return true;
		}

		bool Minimum3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Function_Pool::Minimum( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
					return false;
			}

			return true;
		}

		bool Minimum8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Function_Pool::Minimum(
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
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Function_Pool::Minimum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
											 image[2], roiX[2], roiY[2], roiWidth, roiHeight );

				if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
					intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
					return false;
			}

			return true;
		}

		bool Subtract2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				Bitmap_Image::Image output = Function_Pool::Subtract( input[0], input[1] );

				if( !equalSize( input[0], output ) ||
					!verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ))
					return false;
			}

			return true;
		}

		bool Subtract3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

				Function_Pool::Subtract( image[0], image[1], image[2] );

				if( !verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ) )
					return false;
			}

			return true;
		}

		bool Subtract8ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ input.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				Bitmap_Image::Image output = Function_Pool::Subtract(
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
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 3 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				Function_Pool::Subtract( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
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
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				if( Function_Pool::Sum( input ) != input.width() * input.height() * intensity )
					return false;
			}

			return true;
		}

		bool Sum5ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				if( Function_Pool::Sum( input, roiX, roiY, roiWidth, roiHeight ) != roiWidth * roiHeight * intensity )
					return false;
			}

			return true;
		}

		bool Threshold2ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint8_t threshold = randomValue <uint8_t>( 255 );

				Bitmap_Image::Image output = Function_Pool::Threshold( input, threshold );

				if( !verifyImage( output, intensity < threshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool Threshold3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				uint8_t threshold = randomValue <uint8_t>( 255 );

				Function_Pool::Threshold( input[0], input[1], threshold );

				if( !verifyImage( input[1], intensity[0] < threshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool Threshold6ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				uint8_t threshold = randomValue <uint8_t>( 255 );

				Bitmap_Image::Image output = Function_Pool::Threshold( input, roiX, roiY, roiWidth, roiHeight, threshold );

				if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity < threshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool Threshold9ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				uint8_t threshold = randomValue <uint8_t>( 255 );

				Function_Pool::Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, threshold );

				if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] < threshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool ThresholdDouble3ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint8_t minThreshold = randomValue <uint8_t>( 255 );
				uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

				Bitmap_Image::Image output = Function_Pool::Threshold( input, minThreshold, maxThreshold );

				if( !verifyImage( output, intensity < minThreshold || intensity > maxThreshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool ThresholdDouble4ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

				uint8_t minThreshold = randomValue <uint8_t>( 255 );
				uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

				Function_Pool::Threshold( input[0], input[1], minThreshold, maxThreshold );

				if( !verifyImage( input[1], intensity[0] < minThreshold || intensity[0] > maxThreshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}

		bool ThresholdDouble7ParametersTest()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				uint8_t intensity = intensityValue();
				Bitmap_Image::Image input = uniformImage( intensity );

				uint32_t roiX, roiY, roiWidth, roiHeight;

				generateRoi( input, roiX, roiY, roiWidth, roiHeight );

				uint8_t minThreshold = randomValue <uint8_t>( 255 );
				uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

				Bitmap_Image::Image output = Function_Pool::Threshold( input, roiX, roiY, roiWidth, roiHeight, minThreshold,
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
				Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>(1, 8) );

				std::vector < uint8_t > intensity = intensityArray( 2 );
				std::vector < Bitmap_Image::Image > image;

				std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
					{ image.push_back( uniformImage( value ) ); } );

				std::vector < uint32_t > roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				uint8_t minThreshold = randomValue <uint8_t>( 255 );
				uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

				Function_Pool::Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, minThreshold,
										  maxThreshold);

				if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight,
					intensity[0] < minThreshold || intensity[0] > maxThreshold ? 0 : 255 ) )
					return false;
			}

			return true;
		}
	};
};
