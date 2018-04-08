#include <math.h>
#include <numeric>
#include "../../src/image_function.h"
#include "unit_test_image_function.h"
#include "unit_test_helper.h"

namespace Unit_Test
{
    namespace image_function
    {
        bool AbsoluteDifference2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

                PenguinV_Image::Image output = Image_Function::AbsoluteDifference( input[0], input[1] );

                if( !equalSize( input[0], output ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool AbsoluteDifference3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

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
                std::vector < PenguinV_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                PenguinV_Image::Image output = Image_Function::AbsoluteDifference(
                    input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool AbsoluteDifference11ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < PenguinV_Image::Image > image;

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
                std::vector < uint8_t > intensity = intensityArray( randomValue<uint8_t>( 1, 16 ) );
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

                std::vector < uint32_t > result( input[0].width() * input[0].height(), 0 );

                for( std::vector < PenguinV_Image::Image >::const_iterator image = input.begin(); image != input.end(); ++image ) {
                    Image_Function::Accumulate( *image, result );
                }

                uint32_t sum = std::accumulate( intensity.begin(), intensity.end(), 0u );

                if( std::any_of( result.begin(), result.end(), [&sum]( uint32_t v ) { return v != sum; } ) )
                    return false;
            }

            return true;
        }

        bool Accumulate6ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( randomValue<uint8_t>( 1, 16 ) );
                std::vector < PenguinV_Image::Image > input;

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

                if( std::any_of( result.begin(), result.end(), [&sum]( uint32_t v ) { return v != sum; } ) )
                    return false;
            }

            return true;
        }

        bool BitwiseAnd2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

                PenguinV_Image::Image output = Image_Function::BitwiseAnd( input[0], input[1] );

                if( !equalSize( input[0], output ) || !verifyImage( output, intensity[0] & intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseAnd3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

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
                std::vector < PenguinV_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                PenguinV_Image::Image output = Image_Function::BitwiseAnd(
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
                std::vector < PenguinV_Image::Image > image;

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
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

                PenguinV_Image::Image output = Image_Function::BitwiseOr( input[0], input[1] );

                if( !equalSize( input[0], output ) || !verifyImage( output, intensity[0] | intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseOr3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

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
                std::vector < PenguinV_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                PenguinV_Image::Image output = Image_Function::BitwiseOr(
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
                std::vector < PenguinV_Image::Image > image;

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
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

                PenguinV_Image::Image output = Image_Function::BitwiseXor( input[0], input[1] );

                if( !equalSize( input[0], output ) || !verifyImage( output, intensity[0] ^ intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseXor3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

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
                std::vector < PenguinV_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                PenguinV_Image::Image output = Image_Function::BitwiseXor(
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
                std::vector < PenguinV_Image::Image > image;

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

        bool ConvertToGrayScale1ParameterTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 1 );
                PenguinV_Image::Image input = uniformColorImage( intensity[0] );

                PenguinV_Image::Image output = Image_Function::ConvertToGrayScale( input );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToGrayScale2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input = uniformColorImage( intensity[0] );
                PenguinV_Image::Image output( input.width(), input.height() );

                output.fill( intensity[1] );

                Image_Function::ConvertToGrayScale( input, output );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToGrayScale5ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 1 );
                PenguinV_Image::Image input  = uniformColorImage( intensity[0] );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                PenguinV_Image::Image output = Image_Function::ConvertToGrayScale( input, roiX, roiY, roiWidth, roiHeight );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToGrayScale8ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input  = uniformColorImage( intensity[0] );
                PenguinV_Image::Image output = uniformImage     ( intensity[1] );

                std::vector < std::pair <uint32_t, uint32_t> > size( 2 );

                size[0] = imageSize( input );
                size[1] = imageSize( output );

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                generateRoi( size, roiX, roiY, roiWidth, roiHeight );

                Image_Function::ConvertToGrayScale( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight );

                if( !verifyImage( output, roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToRgb1ParameterTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 1 );
                PenguinV_Image::Image input = uniformImage( intensity[0] );

                PenguinV_Image::Image output = Image_Function::ConvertToRgb( input );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToRgb2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input = uniformImage( intensity[0] );
                PenguinV_Image::Image output( input.width(), input.height(), PenguinV_Image::RGB );

                output.fill( intensity[1] );

                Image_Function::ConvertToRgb( input, output );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToRgb5ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 1 );
                PenguinV_Image::Image input = uniformImage( intensity[0] );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                PenguinV_Image::Image output = Image_Function::ConvertToRgb( input, roiX, roiY, roiWidth, roiHeight );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToRgb8ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input  = uniformImage     ( intensity[0] );
                PenguinV_Image::Image output = uniformColorImage( intensity[1] );

                std::vector < std::pair <uint32_t, uint32_t> > size( 2 );

                size[0] = imageSize( input );
                size[1] = imageSize( output );

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                generateRoi( size, roiX, roiY, roiWidth, roiHeight );

                Image_Function::ConvertToRgb( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight );

                if( !verifyImage( output, roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool Copy2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

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
                PenguinV_Image::Image input = uniformImage( intensity );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                PenguinV_Image::Image output = Image_Function::Copy( input, roiX, roiY, roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity ) )
                    return false;
            }

            return true;
        }

        bool Copy8ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < PenguinV_Image::Image > image;

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

        bool Fill2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image image = uniformImage( intensity[0] );

                Image_Function::Fill( image, intensity[1] );

                if( !verifyImage( image, intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Fill6ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image image = uniformImage( intensity[0] );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                Image_Function::Fill( image, roiX, roiY, roiWidth, roiHeight, intensity[1] );

                if( !verifyImage( image, roiX, roiY, roiWidth, roiHeight, intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Flip3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input = uniformImage( intensity[0] );

                const bool horizontalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
                const bool verticalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
                const uint32_t xCorrection = input.width() % 2;
                const uint32_t yCorrection = input.height() % 2;

                if (verticalFlip)
                {
                    if (input.height() > 1)
                        Image_Function::Fill(input, 0, 0, input.width(), input.height() / 2, intensity[1]);
                }
                else if (horizontalFlip)
                {
                    if (input.width() > 1)
                        Image_Function::Fill(input, 0, 0, input.width() / 2, input.height(), intensity[1]);
                }

                PenguinV_Image::Image output = Image_Function::Flip( input, horizontalFlip, verticalFlip );

                if( !equalSize( output, input.width(), input.height() ))
                    return false;

                if (verticalFlip) {
                    if( !verifyImage( output, 0, 0, input.width(), input.height() / 2 + yCorrection, intensity[0] ) )
                        return false;
                    if((input.height() > 1) && !verifyImage( output, 0, input.height() / 2 + yCorrection, input.width(), input.height() / 2, intensity[1] ) )
                        return false;
                }
                else {
                    if( !verifyImage( output, 0, 0, input.width() / 2 + xCorrection, input.height(), intensity[0] ) )
                        return false;
                    if((input.width() > 1) && !verifyImage( output, input.width() / 2 + xCorrection, 0, input.width() / 2, input.height(), intensity[horizontalFlip ? 1 : 0] ) )
                        return false;
                }
            }

            return true;
        }

        bool Flip4ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                uint8_t intensityFill = intensityValue();
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

                const bool horizontalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
                const bool verticalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
                const uint32_t xCorrection = input[0].width() % 2;
                const uint32_t yCorrection = input[0].height() % 2;

                if (verticalFlip)
                {
                    if (input[0].height() > 1)
                        Image_Function::Fill(input[0], 0, 0, input[0].width(), input[0].height() / 2, intensityFill);
                }
                else if (horizontalFlip)
                {
                    if (input[0].width() > 1)
                        Image_Function::Fill(input[0], 0, 0, input[0].width() / 2, input[0].height(), intensityFill);
                }

                Image_Function::Flip( input[0], input[1], horizontalFlip, verticalFlip );

                if (verticalFlip) {
                    if( !verifyImage( input[1], 0, 0, input[1].width(), input[1].height() / 2 + yCorrection, intensity[0] ) )
                        return false;
                    if((input[0].height() > 1) && !verifyImage( input[1], 0, input[1].height() / 2 + yCorrection, input[1].width(), input[1].height() / 2, intensityFill ) )
                        return false;
                }
                else {
                    if( !verifyImage( input[1], 0, 0, input[1].width() / 2 + xCorrection, input[1].height(), intensity[0] ) )
                        return false;
                    if((input[0].width() > 1) && !verifyImage( input[1], input[1].width() / 2 + xCorrection, 0, input[1].width() / 2, input[1].height(), horizontalFlip ? intensityFill : intensity[0] ) )
                        return false;
                }
            }

            return true;
        }

        bool Flip7ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input = uniformImage( intensity[0] );

                uint32_t roiX, roiY, roiWidth, roiHeight;
                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                const bool horizontalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
                const bool verticalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
                const uint32_t xCorrection = roiWidth % 2;
                const uint32_t yCorrection = roiHeight % 2;

                if (verticalFlip)
                {
                    if (roiHeight > 1)
                        Image_Function::Fill(input, roiX, roiY, roiWidth, roiHeight / 2, intensity[1]);
                }
                else if (horizontalFlip)
                {
                    if (roiWidth > 1)
                        Image_Function::Fill(input, roiX, roiY, roiWidth / 2, roiHeight, intensity[1]);
                }

                PenguinV_Image::Image output = Image_Function::Flip( input, roiX, roiY, roiWidth, roiHeight, horizontalFlip, verticalFlip );

                if( !equalSize( output, roiWidth, roiHeight ))
                    return false;

                if (verticalFlip) {
                    if( !verifyImage( output, 0, 0, roiWidth, roiHeight / 2 + yCorrection, intensity[0] ) )
                        return false;
                    if((roiHeight > 1) && !verifyImage( output, 0, roiHeight / 2 + yCorrection, roiWidth, roiHeight / 2, intensity[1] ) )
                        return false;
                }
                else {
                    if( !verifyImage( output, 0, 0, roiWidth / 2 + xCorrection, roiHeight, intensity[0] ) )
                        return false;
                    if((roiWidth > 1) && !verifyImage( output, roiWidth / 2 + xCorrection, 0, roiWidth / 2, roiHeight, intensity[horizontalFlip ? 1 : 0] ) )
                        return false;
                }
            }

            return true;
        }

        bool Flip10ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                uint8_t intensityFill = intensityValue();
                std::vector < PenguinV_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                const bool horizontalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
                const bool verticalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
                const uint32_t xCorrection = roiWidth % 2;
                const uint32_t yCorrection = roiHeight % 2;

                if (verticalFlip)
                {
                    if (roiHeight > 1)
                        Image_Function::Fill(image[0], roiX[0], roiY[0], roiWidth, roiHeight / 2, intensityFill);
                }
                else if (horizontalFlip)
                {
                    if (roiWidth > 1)
                        Image_Function::Fill(image[0], roiX[0], roiY[0], roiWidth / 2, roiHeight, intensityFill);
                }

                Image_Function::Flip( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, horizontalFlip, verticalFlip );

                if (verticalFlip) {
                    if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight / 2 + yCorrection, intensity[0] ) )
                        return false;
                    if((roiHeight > 1) && !verifyImage( image[1], roiX[1], roiY[1] + roiHeight / 2 + yCorrection, roiWidth, roiHeight / 2, intensityFill ) )
                        return false;
                }
                else {
                    if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth / 2 + xCorrection, roiHeight, intensity[0] ) )
                        return false;
                    if( (roiWidth > 1) && !verifyImage( image[1], roiX[1] + roiWidth / 2 + xCorrection, roiY[1], roiWidth / 2, roiHeight, horizontalFlip ? intensityFill : intensity[0] ) )
                        return false;
                }
            }

            return true;
        }

        bool GammaCorrection3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image input = uniformImage( intensity );

                double a     = randomValue <uint32_t>( 100 ) / 100.0;
                double gamma = randomValue <uint32_t>( 300 ) / 100.0;

                PenguinV_Image::Image output = Image_Function::GammaCorrection( input, a, gamma );

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
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

                double a     = randomValue <uint32_t>( 100 ) / 100.0;
                double gamma = randomValue <uint32_t>( 300 ) / 100.0;

                Image_Function::GammaCorrection( input[0], input[1], a, gamma );

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
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image input = uniformImage( intensity );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                double a     = randomValue <uint32_t>( 100 ) / 100.0;
                double gamma = randomValue <uint32_t>( 300 ) / 100.0;

                PenguinV_Image::Image output = Image_Function::GammaCorrection( input, roiX, roiY, roiWidth, roiHeight, a, gamma );

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
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < PenguinV_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                double a     = randomValue <uint32_t>( 100 ) / 100.0;
                double gamma = randomValue <uint32_t>( 300 ) / 100.0;

                Image_Function::GammaCorrection( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, a, gamma );

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

        bool GetThreshold1ParameterTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                while (intensity[0] == intensity[1])
                    intensity = intensityArray( 2 );

                std::vector< uint32_t > histogram( 256u, 0);
                ++histogram[intensity[0]];
                ++histogram[intensity[1]];

                if( Image_Function::GetThreshold(histogram) != std::min(intensity[0], intensity[1]) )
                    return false;
            }

            return true;
        }

        bool Histogram1ParameterTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image image = uniformImage( intensity );

                std::vector < uint32_t > histogram = Image_Function::Histogram( image );

                if( histogram.size() != 256u || histogram[intensity] != image.width() * image.height() ||
                    std::accumulate( histogram.begin(), histogram.end(), 0u )  != image.width() * image.height() )
                    return false;
            }

            return true;
        }

        bool Histogram2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image image = uniformImage( intensity );

                std::vector < uint32_t > histogram;
                Image_Function::Histogram( image, histogram );

                if( histogram.size() != 256u || histogram[intensity] != image.width() * image.height() ||
                    std::accumulate( histogram.begin(), histogram.end(), 0u )  != image.width() * image.height() )
                    return false;
            }

            return true;
        }

        bool Histogram4ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image input = uniformImage( intensity );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                std::vector < uint32_t > histogram = Image_Function::Histogram( input, roiX, roiY, roiWidth, roiHeight );

                if( histogram.size() != 256u || histogram[intensity] != roiWidth * roiHeight ||
                    std::accumulate( histogram.begin(), histogram.end(), 0u )  != roiWidth * roiHeight )
                    return false;
            }

            return true;
        }

        bool Histogram5ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image input = uniformImage( intensity );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                std::vector < uint32_t > histogram;
                Image_Function::Histogram( input, roiX, roiY, roiWidth, roiHeight, histogram );

                if( histogram.size() != 256u || histogram[intensity] != roiWidth * roiHeight ||
                    std::accumulate( histogram.begin(), histogram.end(), 0u )  != roiWidth * roiHeight )
                    return false;
            }

            return true;
        }

        bool Invert1ParameterTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image input = uniformImage( intensity );

                PenguinV_Image::Image output = Image_Function::Invert( input );

                if( !verifyImage( output, ~intensity ) )
                    return false;
            }

            return true;
        }

        bool Invert2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

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
                PenguinV_Image::Image input = uniformImage( intensity );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                PenguinV_Image::Image output = Image_Function::Invert( input, roiX, roiY, roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, ~intensity ) )
                    return false;
            }

            return true;
        }

        bool Invert8ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < PenguinV_Image::Image > image;

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
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

                if( (intensity[0] == intensity[1]) != (Image_Function::IsEqual( input[0], input[1] )) )
                    return false;
            }

            return true;
        }

        bool IsEqual8ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < PenguinV_Image::Image > image;

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
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input = randomImage( intensity );

                std::vector < uint8_t > lookupTable( 256, 0 );

                lookupTable[intensity[0]] = intensityValue();
                lookupTable[intensity[1]] = intensityValue();

                PenguinV_Image::Image output = Image_Function::LookupTable( input, lookupTable );

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
                PenguinV_Image::Image input  = randomImage( intensity );
                PenguinV_Image::Image output( input.width(), input.height() );

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
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input = uniformImage();

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                fillImage( input, roiX, roiY, roiWidth, roiHeight, intensity );

                std::vector < uint8_t > lookupTable( 256, 0 );

                lookupTable[intensity[0]] = intensityValue();
                lookupTable[intensity[1]] = intensityValue();

                PenguinV_Image::Image output = Image_Function::LookupTable( input, roiX, roiY, roiWidth, roiHeight, lookupTable );

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
                PenguinV_Image::Image input  = uniformImage();
                PenguinV_Image::Image output = uniformImage();

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                std::vector < std::pair < uint32_t, uint32_t > > size( 2 );

                size[0] = imageSize( input );
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
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

                PenguinV_Image::Image output = Image_Function::Maximum( input[0], input[1] );

                if( !equalSize( input[0], output ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Maximum3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

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
                std::vector < PenguinV_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                PenguinV_Image::Image output = Image_Function::Maximum(
                    input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Maximum11ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < PenguinV_Image::Image > image;

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
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

                PenguinV_Image::Image output = Image_Function::Minimum( input[0], input[1] );

                if( !equalSize( input[0], output ) ||
                    !verifyImage( output, intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Minimum3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

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
                std::vector < PenguinV_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                PenguinV_Image::Image output = Image_Function::Minimum(
                    input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) ||
                    !verifyImage( output, intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Minimum11ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < PenguinV_Image::Image > image;

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
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input = randomImage( intensity );

                PenguinV_Image::Image output = Image_Function::Normalize( input );

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
                PenguinV_Image::Image input  = randomImage( intensity );
                PenguinV_Image::Image output( input.width(), input.height() );

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
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input = uniformImage();

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                fillImage( input, roiX, roiY, roiWidth, roiHeight, intensity );

                PenguinV_Image::Image output = Image_Function::Normalize( input, roiX, roiY, roiWidth, roiHeight );

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
                PenguinV_Image::Image input  = uniformImage();
                PenguinV_Image::Image output = uniformImage();

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                std::vector < std::pair < uint32_t, uint32_t > > size( 2 );

                size[0] = imageSize( input );
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
                PenguinV_Image::Image image = uniformImage( intensity );

                bool horizontal = (i % 2 == 0);

                std::vector < uint32_t > projection = Image_Function::ProjectionProfile( image, horizontal );

                uint32_t value = (horizontal ? image.height() : image.width()) * intensity;

                if( projection.size() != (horizontal ? image.width() : image.height()) ||
                    std::any_of( projection.begin(), projection.end(), [&value]( uint32_t v ) { return value != v; } ) )
                    return false;
            }

            return true;
        }

        bool ProjectionProfile3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image image = uniformImage( intensity );

                bool horizontal = (i % 2 == 0);

                std::vector < uint32_t > projection;

                Image_Function::ProjectionProfile( image, horizontal, projection );

                uint32_t value = (horizontal ? image.height() : image.width()) * intensity;

                if( projection.size() != (horizontal ? image.width() : image.height()) ||
                    std::any_of( projection.begin(), projection.end(), [&value]( uint32_t v ) { return value != v; } ) )
                    return false;
            }

            return true;
        }

        bool ProjectionProfile6ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image image = uniformImage( intensity );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                bool horizontal = (i % 2 == 0);

                std::vector < uint32_t > projection = Image_Function::ProjectionProfile( image, roiX, roiY, roiWidth, roiHeight, horizontal );

                uint32_t value = (horizontal ? roiHeight : roiWidth) * intensity;

                if( projection.size() != (horizontal ? roiWidth : roiHeight) ||
                    std::any_of( projection.begin(), projection.end(), [&value]( uint32_t v ) { return value != v; } ) )
                    return false;
            }

            return true;
        }

        bool ProjectionProfile7ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image image = uniformImage( intensity );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                bool horizontal = (i % 2 == 0);

                std::vector < uint32_t > projection;

                Image_Function::ProjectionProfile( image, roiX, roiY, roiWidth, roiHeight, horizontal, projection );

                uint32_t value = (horizontal ? roiHeight : roiWidth) * intensity;

                if( projection.size() != (horizontal ? roiWidth : roiHeight) ||
                    std::any_of( projection.begin(), projection.end(), [&value]( uint32_t v ) { return value != v; } ) )
                    return false;
            }

            return true;
        }

        bool Resize2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image input = uniformImage( intensity );

                uint32_t outputWidth  = randomValue<uint32_t>( 1, 2048 );
                uint32_t outputHeight = randomValue<uint32_t>( 1, 2048 );

                PenguinV_Image::Image output = Image_Function::Resize( input, outputWidth, outputHeight );

                if( !equalSize( output, outputWidth, outputHeight ) || !verifyImage( output, intensity ) )
                    return false;
            }

            return true;
        }

        bool Resize3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input  = uniformImage( intensity[0] );
                PenguinV_Image::Image output = uniformImage( intensity[1] );

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
                PenguinV_Image::Image input = uniformImage( intensity );

                uint32_t outputWidth  = randomValue<uint32_t>( 1, 2048 );
                uint32_t outputHeight = randomValue<uint32_t>( 1, 2048 );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                PenguinV_Image::Image output = Image_Function::Resize( input, roiX, roiY, roiWidth, roiHeight, outputWidth, outputHeight );

                if( !equalSize( output, outputWidth, outputHeight ) || !verifyImage( output, intensity ) )
                    return false;
            }

            return true;
        }

        bool Resize9ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input  = uniformImage( intensity[0] );
                PenguinV_Image::Image output = uniformImage( intensity[1] );

                std::vector < uint32_t > roiX( 2 ), roiY( 2 ), roiWidth( 2 ), roiHeight( 2 );

                generateRoi( input, roiX[0], roiY[0], roiWidth[0], roiHeight[0] );
                generateRoi( output, roiX[1], roiY[1], roiWidth[1], roiHeight[1] );

                Image_Function::Resize( input, roiX[0], roiY[0], roiWidth[0], roiHeight[0],
                                        output, roiX[1], roiY[1], roiWidth[1], roiHeight[1] );

                if( !verifyImage( output, roiX[1], roiY[1], roiWidth[1], roiHeight[1], intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool SetPixel4ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image image  = uniformImage( intensity[0] );
                const uint32_t x = randomValue<uint32_t>( 0, image.width() );
                const uint32_t y = randomValue<uint32_t>( 0, image.height() );

                Image_Function::SetPixel( image, x, y, intensity[1] );

                if( !verifyImage( image, x, y, 1, 1, intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool SetPixelArray4ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image image  = uniformImage( intensity[0] );
                std::vector< uint32_t > X( randomValue<uint32_t>( 1, 100 ) );
                std::vector< uint32_t > Y( X.size() );
                
                for (size_t j = 0; j < X.size(); j++) {
                    X[j] = randomValue<uint32_t>( 0, image.width() );
                    Y[j] = randomValue<uint32_t>( 0, image.height() );
                }

                Image_Function::SetPixel( image, X, Y, intensity[1] );

                for (size_t j = 0; j < X.size(); j++) {
                    if( !verifyImage( image, X[j], Y[j], 1, 1, intensity[1] ) )
                        return false;
                }
            }

            return true;
        }

        bool Subtract2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

                PenguinV_Image::Image output = Image_Function::Subtract( input[0], input[1] );

                if( !equalSize( input[0], output ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ) )
                    return false;
            }

            return true;
        }

        bool Subtract3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

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
                std::vector < PenguinV_Image::Image > input;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { input.push_back( uniformImage( value ) ); } );

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                PenguinV_Image::Image output = Image_Function::Subtract(
                    input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

                if( !equalSize( output, roiWidth, roiHeight ) ||
                    !verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ) )
                    return false;
            }

            return true;
        }

        bool Subtract11ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < PenguinV_Image::Image > image;

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
                PenguinV_Image::Image input = uniformImage( intensity );

                if( Image_Function::Sum( input ) != intensity * input.width() * input.height() )
                    return false;
            }

            return true;
        }

        bool Sum5ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image input = uniformImage( intensity );

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
                PenguinV_Image::Image input = uniformImage( intensity );

                uint8_t threshold = randomValue <uint8_t>( 255 );

                PenguinV_Image::Image output = Image_Function::Threshold( input, threshold );

                if( !verifyImage( output, intensity < threshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool Threshold3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

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
                PenguinV_Image::Image input = uniformImage( intensity );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                uint8_t threshold = randomValue <uint8_t>( 255 );

                PenguinV_Image::Image output = Image_Function::Threshold( input, roiX, roiY, roiWidth, roiHeight, threshold );

                if( !equalSize( output, roiWidth, roiHeight ) || !verifyImage( output, intensity < threshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool Threshold9ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < PenguinV_Image::Image > image;

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
                PenguinV_Image::Image input = uniformImage( intensity );

                uint8_t minThreshold = randomValue <uint8_t>( 255 );
                uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

                PenguinV_Image::Image output = Image_Function::Threshold( input, minThreshold, maxThreshold );

                if( !verifyImage( output, intensity < minThreshold || intensity > maxThreshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool ThresholdDouble4ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

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
                PenguinV_Image::Image input = uniformImage( intensity );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                uint8_t minThreshold = randomValue <uint8_t>( 255 );
                uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

                PenguinV_Image::Image output = Image_Function::Threshold( input, roiX, roiY, roiWidth, roiHeight, minThreshold,
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
                std::vector < PenguinV_Image::Image > image;

                std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t & value )
                { image.push_back( uniformImage( value ) ); } );

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                uint8_t minThreshold = randomValue <uint8_t>( 255 );
                uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

                Image_Function::Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, minThreshold,
                                           maxThreshold );

                if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight,
                                  intensity[0] < minThreshold || intensity[0] > maxThreshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool Transpose1ParameterTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image input = uniformImage( intensity );

                PenguinV_Image::Image output = Image_Function::Transpose( input );

                if( !equalSize( output, input.height(), input.width() ) || !verifyImage( output, intensity ) )
                    return false;
            }

            return true;
        }

        bool Transpose2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input  = uniformImage( intensity[0] );
                PenguinV_Image::Image output( input.height(), input.width() );

                output.fill( intensity[1] );

                Image_Function::Transpose( input, output );

                if( !verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool Transpose5ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                PenguinV_Image::Image input = uniformImage( intensity );

                uint32_t roiX, roiY, roiWidth, roiHeight;

                generateRoi( input, roiX, roiY, roiWidth, roiHeight );

                PenguinV_Image::Image output = Image_Function::Transpose( input, roiX, roiY, roiWidth, roiHeight );

                if( !equalSize( output, roiHeight, roiWidth ) || !verifyImage( output, intensity ) )
                    return false;
            }

            return true;
        }

        bool Transpose8ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                PenguinV_Image::Image input  = uniformImage( intensity[0] );
                PenguinV_Image::Image output = uniformImage( intensity[1] );

                std::vector < std::pair <uint32_t, uint32_t> > size( 2 );

                size[0] = imageSize( input );
                size[1] = std::pair <uint32_t, uint32_t>( output.height(), output.width() );

                std::vector < uint32_t > roiX, roiY;
                uint32_t roiWidth, roiHeight;

                generateRoi( size, roiX, roiY, roiWidth, roiHeight );

                generateOffset( output, roiX[1], roiY[1], roiHeight, roiWidth );

                Image_Function::Transpose( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight );

                if( !verifyImage( output, roiX[1], roiY[1], roiHeight, roiWidth, intensity[0] ) )
                    return false;
            }

            return true;
        }
    }

    void addTests_Image_Function( UnitTestFramework & framework )
    {
        ADD_TEST( framework, image_function::AbsoluteDifference2ParametersTest );
        ADD_TEST( framework, image_function::AbsoluteDifference3ParametersTest );
        ADD_TEST( framework, image_function::AbsoluteDifference8ParametersTest );
        ADD_TEST( framework, image_function::AbsoluteDifference11ParametersTest );

        ADD_TEST( framework, image_function::Accumulate2ParametersTest );
        ADD_TEST( framework, image_function::Accumulate6ParametersTest );

        ADD_TEST( framework, image_function::BitwiseAnd2ParametersTest );
        ADD_TEST( framework, image_function::BitwiseAnd3ParametersTest );
        ADD_TEST( framework, image_function::BitwiseAnd8ParametersTest );
        ADD_TEST( framework, image_function::BitwiseAnd11ParametersTest );

        ADD_TEST( framework, image_function::BitwiseOr2ParametersTest );
        ADD_TEST( framework, image_function::BitwiseOr3ParametersTest );
        ADD_TEST( framework, image_function::BitwiseOr8ParametersTest );
        ADD_TEST( framework, image_function::BitwiseOr11ParametersTest );

        ADD_TEST( framework, image_function::BitwiseXor2ParametersTest );
        ADD_TEST( framework, image_function::BitwiseXor3ParametersTest );
        ADD_TEST( framework, image_function::BitwiseXor8ParametersTest );
        ADD_TEST( framework, image_function::BitwiseXor11ParametersTest );

        ADD_TEST( framework, image_function::ConvertToGrayScale1ParameterTest );
        ADD_TEST( framework, image_function::ConvertToGrayScale2ParametersTest );
        ADD_TEST( framework, image_function::ConvertToGrayScale5ParametersTest );
        ADD_TEST( framework, image_function::ConvertToGrayScale8ParametersTest );

        ADD_TEST( framework, image_function::ConvertToRgb1ParameterTest );
        ADD_TEST( framework, image_function::ConvertToRgb2ParametersTest );
        ADD_TEST( framework, image_function::ConvertToRgb5ParametersTest );
        ADD_TEST( framework, image_function::ConvertToRgb8ParametersTest );

        ADD_TEST( framework, image_function::Copy2ParametersTest );
        ADD_TEST( framework, image_function::Copy5ParametersTest );
        ADD_TEST( framework, image_function::Copy8ParametersTest );

        ADD_TEST( framework, image_function::Fill2ParametersTest );
        ADD_TEST( framework, image_function::Fill6ParametersTest );

        ADD_TEST( framework, image_function::Flip3ParametersTest );
        ADD_TEST( framework, image_function::Flip4ParametersTest );
        ADD_TEST( framework, image_function::Flip7ParametersTest );
        ADD_TEST( framework, image_function::Flip10ParametersTest );

        ADD_TEST( framework, image_function::GammaCorrection3ParametersTest );
        ADD_TEST( framework, image_function::GammaCorrection4ParametersTest );
        ADD_TEST( framework, image_function::GammaCorrection7ParametersTest );
        ADD_TEST( framework, image_function::GammaCorrection10ParametersTest );

        ADD_TEST( framework, image_function::GetThreshold1ParameterTest );

        ADD_TEST( framework, image_function::Histogram1ParameterTest );
        ADD_TEST( framework, image_function::Histogram2ParametersTest );
        ADD_TEST( framework, image_function::Histogram4ParametersTest );
        ADD_TEST( framework, image_function::Histogram5ParametersTest );

        ADD_TEST( framework, image_function::Invert1ParameterTest );
        ADD_TEST( framework, image_function::Invert2ParametersTest );
        ADD_TEST( framework, image_function::Invert5ParametersTest );
        ADD_TEST( framework, image_function::Invert8ParametersTest );

        ADD_TEST( framework, image_function::IsEqual2ParametersTest );
        ADD_TEST( framework, image_function::IsEqual8ParametersTest );

        ADD_TEST( framework, image_function::LookupTable2ParametersTest );
        ADD_TEST( framework, image_function::LookupTable3ParametersTest );
        ADD_TEST( framework, image_function::LookupTable6ParametersTest );
        ADD_TEST( framework, image_function::LookupTable9ParametersTest );

        ADD_TEST( framework, image_function::Maximum2ParametersTest );
        ADD_TEST( framework, image_function::Maximum3ParametersTest );
        ADD_TEST( framework, image_function::Maximum8ParametersTest );
        ADD_TEST( framework, image_function::Maximum11ParametersTest );

        ADD_TEST( framework, image_function::Minimum2ParametersTest );
        ADD_TEST( framework, image_function::Minimum3ParametersTest );
        ADD_TEST( framework, image_function::Minimum8ParametersTest );
        ADD_TEST( framework, image_function::Minimum11ParametersTest );

        ADD_TEST( framework, image_function::Normalize1ParameterTest );
        ADD_TEST( framework, image_function::Normalize2ParametersTest );
        ADD_TEST( framework, image_function::Normalize5ParametersTest );
        ADD_TEST( framework, image_function::Normalize8ParametersTest );

        ADD_TEST( framework, image_function::ProjectionProfile2ParametersTest );
        ADD_TEST( framework, image_function::ProjectionProfile3ParametersTest );
        ADD_TEST( framework, image_function::ProjectionProfile6ParametersTest );
        ADD_TEST( framework, image_function::ProjectionProfile7ParametersTest );

        ADD_TEST( framework, image_function::Resize2ParametersTest );
        ADD_TEST( framework, image_function::Resize3ParametersTest );
        ADD_TEST( framework, image_function::Resize7ParametersTest );
        ADD_TEST( framework, image_function::Resize9ParametersTest );

        ADD_TEST( framework, image_function::SetPixel4ParametersTest );
        ADD_TEST( framework, image_function::SetPixelArray4ParametersTest );

        ADD_TEST( framework, image_function::Subtract2ParametersTest );
        ADD_TEST( framework, image_function::Subtract3ParametersTest );
        ADD_TEST( framework, image_function::Subtract8ParametersTest );
        ADD_TEST( framework, image_function::Subtract11ParametersTest );

        ADD_TEST( framework, image_function::Sum1ParameterTest );
        ADD_TEST( framework, image_function::Sum5ParametersTest );

        ADD_TEST( framework, image_function::Threshold2ParametersTest );
        ADD_TEST( framework, image_function::Threshold3ParametersTest );
        ADD_TEST( framework, image_function::Threshold6ParametersTest );
        ADD_TEST( framework, image_function::Threshold9ParametersTest );

        ADD_TEST( framework, image_function::ThresholdDouble3ParametersTest );
        ADD_TEST( framework, image_function::ThresholdDouble4ParametersTest );
        ADD_TEST( framework, image_function::ThresholdDouble7ParametersTest );
        ADD_TEST( framework, image_function::ThresholdDouble10ParametersTest );

        ADD_TEST( framework, image_function::Transpose1ParameterTest );
        ADD_TEST( framework, image_function::Transpose2ParametersTest );
        ADD_TEST( framework, image_function::Transpose5ParametersTest );
        ADD_TEST( framework, image_function::Transpose8ParametersTest );
    }
}
