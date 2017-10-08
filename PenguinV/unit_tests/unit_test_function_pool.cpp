#include <math.h>
#include "../Library/function_pool.h"
#include "../Library/thread_pool.h"
#include "unit_test_function_pool.h"
#include "unit_test_helper.h"

namespace Unit_Test
{
    void addTests_Function_Pool( UnitTestFramework & framework )
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

        ADD_TEST( framework, Function_Pool_Test::ConvertToGrayScale1ParameterTest );
        ADD_TEST( framework, Function_Pool_Test::ConvertToGrayScale2ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::ConvertToGrayScale5ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::ConvertToGrayScale8ParametersTest );

        ADD_TEST( framework, Function_Pool_Test::ConvertToRgb1ParameterTest );
        ADD_TEST( framework, Function_Pool_Test::ConvertToRgb2ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::ConvertToRgb5ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::ConvertToRgb8ParametersTest );

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

        ADD_TEST( framework, Function_Pool_Test::LookupTable2ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::LookupTable3ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::LookupTable6ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::LookupTable9ParametersTest );

        ADD_TEST( framework, Function_Pool_Test::Maximum2ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::Maximum3ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::Maximum8ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::Maximum11ParametersTest );

        ADD_TEST( framework, Function_Pool_Test::Minimum2ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::Minimum3ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::Minimum8ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::Minimum11ParametersTest );

        ADD_TEST( framework, Function_Pool_Test::Resize2ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::Resize3ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::Resize7ParametersTest );
        ADD_TEST( framework, Function_Pool_Test::Resize9ParametersTest );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                Bitmap_Image::Image output = Function_Pool::AbsoluteDifference( input[0], input[1] );

                if( !equalSize( input[0], output ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool AbsoluteDifference3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Function_Pool::AbsoluteDifference(
                    input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool AbsoluteDifference11ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                Function_Pool::BitwiseXor( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
                                           image[2], roiX[2], roiY[2], roiWidth, roiHeight );

                if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] ^ intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToGrayScale1ParameterTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 1 );
                Bitmap_Image::Image input = uniformColorImage( intensity[0] );

                Bitmap_Image::Image output = Function_Pool::ConvertToGrayScale( input );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToGrayScale2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image::Image input = uniformColorImage( intensity[0] );
                Bitmap_Image::Image output( input.width(), input.height() );

                output.fill( intensity[1] );

                Function_Pool::ConvertToGrayScale( input, output );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToGrayScale5ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 1 );
                Bitmap_Image::Image input  = uniformColorImage( intensity[0] );

                size_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Function_Pool::ConvertToGrayScale( input, roiX, roiY, roiWidth, roiHeight );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToGrayScale8ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image::Image input  = uniformColorImage( intensity[0] );
                Bitmap_Image::Image output = uniformImage     ( intensity[1] );

                std::vector < std::pair <size_t, size_t> > size( 2 );

                size[0] = imageSize( input );
                size[1] = imageSize( output );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( size, roiX, roiY, roiWidth, roiHeight );

                Function_Pool::ConvertToGrayScale( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight );

                if( !verifyImage( output, roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToRgb1ParameterTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 1 );
                Bitmap_Image::Image input = uniformImage( intensity[0] );

                Bitmap_Image::Image output = Function_Pool::ConvertToRgb( input );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToRgb2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image::Image input = uniformImage( intensity[0] );
                Bitmap_Image::Image output( input.width(), input.height(), Bitmap_Image::RGB );

                output.fill( intensity[1] );

                Function_Pool::ConvertToRgb( input, output );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToRgb5ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 1 );
                Bitmap_Image::Image input = uniformImage( intensity[0] );

                size_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Function_Pool::ConvertToRgb( input, roiX, roiY, roiWidth, roiHeight );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToRgb8ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image::Image input  = uniformImage     ( intensity[0] );
                Bitmap_Image::Image output = uniformColorImage( intensity[1] );

                std::vector < std::pair <size_t, size_t> > size( 2 );

                size[0] = imageSize( input );
                size[1] = imageSize( output );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( size, roiX, roiY, roiWidth, roiHeight );

                Function_Pool::ConvertToRgb( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight );

                if( !verifyImage( output, roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool GammaCorrection3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                double a     = randomValue <size_t>( 100 ) / 100.0;
                double gamma = randomValue <size_t>( 300 ) / 100.0;

                Bitmap_Image::Image output = Function_Pool::GammaCorrection( input, a, gamma );

                double value = a * pow( intensity / 255.0, gamma ) * 255 + 0.5;
                uint8_t corrected = 0;

                if( value < 256 )
                    corrected = static_cast<uint8_t>(value);
                else
                    corrected = 255;

                if( !verifyImage( output, corrected ) )
                    return false;
            }

            return true;
        }

        bool GammaCorrection4ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                double a     = randomValue <size_t>( 100 ) / 100.0;
                double gamma = randomValue <size_t>( 300 ) / 100.0;

                Function_Pool::GammaCorrection( input[0], input[1], a, gamma );

                double value = a * pow( intensity[0] / 255.0, gamma ) * 255 + 0.5;
                uint8_t corrected = 0;

                if( value < 256 )
                    corrected = static_cast<uint8_t>(value);
                else
                    corrected = 255;

                if( !verifyImage( input[1], corrected ) )
                    return false;
            }

            return true;
        }

        bool GammaCorrection7ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                size_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                double a     = randomValue <size_t>( 100 ) / 100.0;
                double gamma = randomValue <size_t>( 300 ) / 100.0;

                Bitmap_Image::Image output = Function_Pool::GammaCorrection( input, roiX, roiY, roiWidth, roiHeight, a, gamma );

                double value = a * pow( intensity / 255.0, gamma ) * 255 + 0.5;
                uint8_t corrected = 0;

                if( value < 256 )
                    corrected = static_cast<uint8_t>(value);
                else
                    corrected = 255;

                if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, corrected ) )
                    return false;
            }

            return true;
        }

        bool GammaCorrection10ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                double a     = randomValue <size_t>( 100 ) / 100.0;
                double gamma = randomValue <size_t>( 300 ) / 100.0;

                Function_Pool::GammaCorrection( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, a, gamma );

                double value = a * pow( intensity[0] / 255.0, gamma ) * 255 + 0.5;
                uint8_t corrected = 0;

                if( value < 256 )
                    corrected = static_cast<uint8_t>(value);
                else
                    corrected = 255;

                if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, corrected ) )
                    return false;
            }

            return true;
        }

        bool Invert1ParameterTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                size_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Function_Pool::Invert( input, roiX, roiY, roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, ~intensity ) )
                    return false;
            }

            return true;
        }

        bool Invert8ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                Function_Pool::Invert( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

                if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, ~intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool IsEqual2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                if( (intensity[0] == intensity[1]) != (Function_Pool::IsEqual( input[0], input[1] )) )
                    return false;
            }

            return true;
        }

        bool IsEqual8ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                if( (intensity[0] == intensity[1]) !=
                    (Function_Pool::IsEqual( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight )) )
                    return false;
            }

            return true;
        }

        bool LookupTable2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image::Image input = randomImage( intensity );

                std::vector < uint8_t > lookupTable( 256, 0 );

                lookupTable[intensity[0]] = intensityValue();
                lookupTable[intensity[1]] = intensityValue();

                Bitmap_Image::Image output = Function_Pool::LookupTable( input, lookupTable );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image::Image input  = randomImage( intensity );
                Bitmap_Image::Image output( input.width(), input.height() );

                output.fill( intensityValue() );

                std::vector < uint8_t > lookupTable( 256, 0 );

                lookupTable[intensity[0]] = intensityValue();
                lookupTable[intensity[1]] = intensityValue();

                Function_Pool::LookupTable( input, output, lookupTable );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image::Image input = uniformImage();

                size_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                fillImage( input, roiX, roiY, roiWidth, roiHeight, intensity );

                std::vector < uint8_t > lookupTable( 256, 0 );

                lookupTable[intensity[0]] = intensityValue();
                lookupTable[intensity[1]] = intensityValue();

                Bitmap_Image::Image output = Function_Pool::LookupTable( input, roiX, roiY, roiWidth, roiHeight, lookupTable );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image::Image input  = uniformImage();
                Bitmap_Image::Image output = uniformImage();

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                std::vector < std::pair < size_t, size_t > > size( 2 );

                size[0] = imageSize( input );
                size[1] = imageSize( output );

                generateRoi( size, roiX, roiY, roiWidth, roiHeight );

                fillImage( input, roiX[0], roiY[0], roiWidth, roiHeight, intensity );

                std::vector < uint8_t > lookupTable( 256, 0 );

                lookupTable[intensity[0]] = intensityValue();
                lookupTable[intensity[1]] = intensityValue();

                Function_Pool::LookupTable( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight, lookupTable );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                Bitmap_Image::Image output = Function_Pool::Maximum( input[0], input[1] );

                if( !equalSize( input[0], output ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Maximum3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Function_Pool::Maximum(
                    input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Maximum11ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                Bitmap_Image::Image output = Function_Pool::Minimum( input[0], input[1] );

                if( !equalSize( input[0], output ) ||
                    !verifyImage( output, intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Minimum3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Function_Pool::Minimum(
                    input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) ||
                    !verifyImage( output, intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Minimum11ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                Function_Pool::Minimum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
                                        image[2], roiX[2], roiY[2], roiWidth, roiHeight );

                if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
                                  intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Resize2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                size_t outputWidth  = randomValue<size_t>( 1, 2048 );
                size_t outputHeight = randomValue<size_t>( 1, 2048 );

                Bitmap_Image::Image output = Function_Pool::Resize( input, outputWidth, outputHeight );

                if( !equalSize( output, outputWidth, outputHeight ) || !verifyImage( output, intensity ) )
                    return false;
            }

            return true;
        }

        bool Resize3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image::Image input  = uniformImage( intensity[0] );
                Bitmap_Image::Image output = uniformImage( intensity[1] );

                Function_Pool::Resize( input, output );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool Resize7ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                size_t outputWidth  = randomValue<size_t>( 1, 2048 );
                size_t outputHeight = randomValue<size_t>( 1, 2048 );

                size_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Function_Pool::Resize( input, roiX, roiY, roiWidth, roiHeight, outputWidth, outputHeight );

                if( !equalSize( output, outputWidth, outputHeight ) || !verifyImage( output, intensity ) )
                    return false;
            }

            return true;
        }

        bool Resize9ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image::Image input  = uniformImage( intensity[0] );
                Bitmap_Image::Image output = uniformImage( intensity[1] );

                std::vector < size_t > roiX( 2 ), roiY( 2 ), roiWidth( 2 ), roiHeight( 2 );

                generateRoi( input, roiX[0], roiY[0], roiWidth[0], roiHeight[0] );
                generateRoi( output, roiX[1], roiY[1], roiWidth[1], roiHeight[1] );

                Function_Pool::Resize( input, roiX[0], roiY[0], roiWidth[0], roiHeight[0],
                                       output, roiX[1], roiY[1], roiWidth[1], roiHeight[1] );

                if( !verifyImage( output, roiX[1], roiY[1], roiWidth[1], roiHeight[1], intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool Subtract2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                Bitmap_Image::Image output = Function_Pool::Subtract( input[0], input[1] );

                if( !equalSize( input[0], output ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ) )
                    return false;
            }

            return true;
        }

        bool Subtract3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Function_Pool::Subtract(
                    input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ) )
                    return false;
            }

            return true;
        }

        bool Subtract11ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                if( Function_Pool::Sum( input ) != input.width() * input.height() * intensity )
                    return false;
            }

            return true;
        }

        bool Sum5ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                size_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                if( Function_Pool::Sum( input, roiX, roiY, roiWidth, roiHeight ) != roiWidth * roiHeight * intensity )
                    return false;
            }

            return true;
        }

        bool Threshold2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                size_t roiX, roiY, roiWidth, roiHeight;

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                size_t roiX, roiY, roiWidth, roiHeight;

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
            for( size_t i = 0; i < runCount(); ++i ) {
                Thread_Pool::ThreadPoolMonoid::instance().resize( randomValue<uint8_t>( 1, 8 ) );

                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                uint8_t minThreshold = randomValue <uint8_t>( 255 );
                uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

                Function_Pool::Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, minThreshold,
                                          maxThreshold );

                if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight,
                                  intensity[0] < minThreshold || intensity[0] > maxThreshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }
    };
};
