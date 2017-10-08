#include "../Library/image_function_avx.h"
#include "unit_test_image_function_avx.h"
#include "unit_test_helper.h"

namespace Unit_Test
{
    void addTests_Image_Function_Avx( UnitTestFramework & framework )
    {
        ADD_TEST( framework, Image_Function_Avx_Test::AbsoluteDifference2ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::AbsoluteDifference3ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::AbsoluteDifference8ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::AbsoluteDifference11ParametersTest );

        ADD_TEST( framework, Image_Function_Avx_Test::BitwiseAnd2ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::BitwiseAnd3ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::BitwiseAnd8ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::BitwiseAnd11ParametersTest );

        ADD_TEST( framework, Image_Function_Avx_Test::BitwiseOr2ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::BitwiseOr3ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::BitwiseOr8ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::BitwiseOr11ParametersTest );

        ADD_TEST( framework, Image_Function_Avx_Test::BitwiseXor2ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::BitwiseXor3ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::BitwiseXor8ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::BitwiseXor11ParametersTest );

        ADD_TEST( framework, Image_Function_Avx_Test::Invert1ParameterTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Invert2ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Invert5ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Invert8ParametersTest );

        ADD_TEST( framework, Image_Function_Avx_Test::Maximum2ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Maximum3ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Maximum8ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Maximum11ParametersTest );

        ADD_TEST( framework, Image_Function_Avx_Test::Minimum2ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Minimum3ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Minimum8ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Minimum11ParametersTest );

        ADD_TEST( framework, Image_Function_Avx_Test::Subtract2ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Subtract3ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Subtract8ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Subtract11ParametersTest );

        ADD_TEST( framework, Image_Function_Avx_Test::Sum1ParameterTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Sum5ParametersTest );

        ADD_TEST( framework, Image_Function_Avx_Test::Threshold2ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Threshold3ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Threshold6ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::Threshold9ParametersTest );

        ADD_TEST( framework, Image_Function_Avx_Test::ThresholdDouble3ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::ThresholdDouble4ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::ThresholdDouble7ParametersTest );
        ADD_TEST( framework, Image_Function_Avx_Test::ThresholdDouble10ParametersTest );
    }

    namespace Image_Function_Avx_Test
    {
        bool AbsoluteDifference2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                Bitmap_Image::Image output = Image_Function_Avx::AbsoluteDifference( input[0], input[1] );

                if( !equalSize( input[0], output ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool AbsoluteDifference3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

                Image_Function_Avx::AbsoluteDifference( image[0], image[1], image[2] );

                if( !verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool AbsoluteDifference8ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Image_Function_Avx::AbsoluteDifference(
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
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                Image_Function_Avx::AbsoluteDifference( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
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
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                Bitmap_Image::Image output = Image_Function_Avx::BitwiseAnd( input[0], input[1] );

                if( !equalSize( input[0], output ) || !verifyImage( output, intensity[0] & intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseAnd3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

                Image_Function_Avx::BitwiseAnd( image[0], image[1], image[2] );

                if( !verifyImage( image[2], intensity[0] & intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseAnd8ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Image_Function_Avx::BitwiseAnd(
                    input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity[0] & intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseAnd11ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                Image_Function_Avx::BitwiseAnd( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
                                                image[2], roiX[2], roiY[2], roiWidth, roiHeight );

                if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] & intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseOr2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                Bitmap_Image::Image output = Image_Function_Avx::BitwiseOr( input[0], input[1] );

                if( !equalSize( input[0], output ) || !verifyImage( output, intensity[0] | intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseOr3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

                Image_Function_Avx::BitwiseOr( image[0], image[1], image[2] );

                if( !verifyImage( image[2], intensity[0] | intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseOr8ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Image_Function_Avx::BitwiseOr(
                    input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity[0] | intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseOr11ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                Image_Function_Avx::BitwiseOr( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
                                               image[2], roiX[2], roiY[2], roiWidth, roiHeight );

                if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] | intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseXor2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                Bitmap_Image::Image output = Image_Function_Avx::BitwiseXor( input[0], input[1] );

                if( !equalSize( input[0], output ) || !verifyImage( output, intensity[0] ^ intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseXor3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

                Image_Function_Avx::BitwiseXor( image[0], image[1], image[2] );

                if( !verifyImage( image[2], intensity[0] ^ intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseXor8ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Image_Function_Avx::BitwiseXor(
                    input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity[0] ^ intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseXor11ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                Image_Function_Avx::BitwiseXor( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
                                                image[2], roiX[2], roiY[2], roiWidth, roiHeight );

                if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] ^ intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Invert1ParameterTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                Bitmap_Image::Image output = Image_Function_Avx::Invert( input );

                if( !verifyImage( output, ~intensity ) )
                    return false;
            }

            return true;
        }

        bool Invert2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                Image_Function_Avx::Invert( input[0], input[1] );

                if( !verifyImage( input[1], ~intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool Invert5ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                size_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Image_Function_Avx::Invert( input, roiX, roiY, roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, ~intensity ) )
                    return false;
            }

            return true;
        }

        bool Invert8ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                Image_Function_Avx::Invert( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

                if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, ~intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool Maximum2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                Bitmap_Image::Image output = Image_Function_Avx::Maximum( input[0], input[1] );

                if( !equalSize( input[0], output ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Maximum3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

                Image_Function_Avx::Maximum( image[0], image[1], image[2] );

                if( !verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Maximum8ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Image_Function_Avx::Maximum(
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
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                Image_Function_Avx::Maximum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
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
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                Bitmap_Image::Image output = Image_Function_Avx::Minimum( input[0], input[1] );

                if( !equalSize( input[0], output ) ||
                    !verifyImage( output, intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Minimum3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

                Image_Function_Avx::Minimum( image[0], image[1], image[2] );

                if( !verifyImage( image[2], intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Minimum8ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Image_Function_Avx::Minimum(
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
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                Image_Function_Avx::Minimum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
                                             image[2], roiX[2], roiY[2], roiWidth, roiHeight );

                if( !verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
                                  intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Subtract2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                Bitmap_Image::Image output = Image_Function_Avx::Subtract( input[0], input[1] );

                if( !equalSize( input[0], output ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ) )
                    return false;
            }

            return true;
        }

        bool Subtract3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image = uniformImages( intensity );

                Image_Function_Avx::Subtract( image[0], image[1], image[2] );

                if( !verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ) )
                    return false;
            }

            return true;
        }

        bool Subtract8ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                Bitmap_Image::Image output = Image_Function_Avx::Subtract(
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
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                Image_Function_Avx::Subtract( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1],
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
                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                if( Image_Function_Avx::Sum( input ) != intensity * input.width() * input.height() )
                    return false;
            }

            return true;
        }

        bool Sum5ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                size_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                if( Image_Function_Avx::Sum( input, roiX, roiY, roiWidth, roiHeight ) != intensity * roiWidth * roiHeight )
                    return false;
            }

            return true;
        }

        bool Threshold2ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                uint8_t threshold = randomValue <uint8_t>( 255 );

                Bitmap_Image::Image output = Image_Function_Avx::Threshold( input, threshold );

                if( !verifyImage( output, intensity < threshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool Threshold3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                uint8_t threshold = randomValue <uint8_t>( 255 );

                Image_Function_Avx::Threshold( input[0], input[1], threshold );

                if( !verifyImage( input[1], intensity[0] < threshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool Threshold6ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                size_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                uint8_t threshold = randomValue <uint8_t>( 255 );

                Bitmap_Image::Image output = Image_Function_Avx::Threshold( input, roiX, roiY, roiWidth, roiHeight, threshold );

                if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity < threshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool Threshold9ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                uint8_t threshold = randomValue <uint8_t>( 255 );

                Image_Function_Avx::Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, threshold );

                if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] < threshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool ThresholdDouble3ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                uint8_t minThreshold = randomValue <uint8_t>( 255 );
                uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

                Bitmap_Image::Image output = Image_Function_Avx::Threshold( input, minThreshold, maxThreshold );

                if( !verifyImage( output, intensity < minThreshold || intensity > maxThreshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool ThresholdDouble4ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > input = uniformImages( intensity );

                uint8_t minThreshold = randomValue <uint8_t>( 255 );
                uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

                Image_Function_Avx::Threshold( input[0], input[1], minThreshold, maxThreshold );

                if( !verifyImage( input[1], intensity[0] < minThreshold || intensity[0] > maxThreshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool ThresholdDouble7ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                Bitmap_Image::Image input = uniformImage( intensity );

                size_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                uint8_t minThreshold = randomValue <uint8_t>( 255 );
                uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

                Bitmap_Image::Image output = Image_Function_Avx::Threshold( input, roiX, roiY, roiWidth, roiHeight,
                                                                            minThreshold, maxThreshold );

                if( !equalSize( output, roiWidth, roiHeight ) ||
                    !verifyImage( output, intensity < minThreshold || intensity > maxThreshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool ThresholdDouble10ParametersTest()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < size_t > roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                uint8_t minThreshold = randomValue <uint8_t>( 255 );
                uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

                Image_Function_Avx::Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight,
                                               minThreshold, maxThreshold );

                if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight,
                                  intensity[0] < minThreshold || intensity[0] > maxThreshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }
    };
};
