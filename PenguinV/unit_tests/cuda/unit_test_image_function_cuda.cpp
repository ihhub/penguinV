#include <numeric>
#include "../../Library/cuda/image_function_cuda.cuh"
#include "../../Library/cuda/cuda_memory.cuh"
#include "../unit_test_helper.h"
#include "unit_test_helper_cuda.cuh"
#include "unit_test_image_function_cuda.h"

namespace Unit_Test
{
    void addTests_Image_Function_Cuda( UnitTestFramework & framework )
    {
        ADD_TEST( framework, Image_Function_Cuda_Test::AbsoluteDifference2ParametersTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::AbsoluteDifference3ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseAnd2ParametersTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseAnd3ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseOr2ParametersTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseOr3ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseXor2ParametersTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseXor3ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::ConvertToGrayScale1ParameterTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::ConvertToGrayScale2ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::ConvertToRgb1ParameterTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::ConvertToRgb2ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::GammaCorrection3ParametersTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::GammaCorrection4ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::Histogram1ParameterTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::Histogram2ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::Invert1ParameterTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::Invert2ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::LookupTable2ParametersTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::LookupTable3ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::Maximum2ParametersTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::Maximum3ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::Minimum2ParametersTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::Minimum3ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::Subtract2ParametersTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::Subtract3ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::Threshold2ParametersTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::Threshold3ParametersTest );

        ADD_TEST( framework, Image_Function_Cuda_Test::ThresholdDouble3ParametersTest );
        ADD_TEST( framework, Image_Function_Cuda_Test::ThresholdDouble4ParametersTest );
    }

    namespace Image_Function_Cuda_Test
    {
        bool AbsoluteDifference2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image_Cuda::Image > input = Cuda::uniformImages( intensity );

                Bitmap_Image_Cuda::Image output = Image_Function_Cuda::AbsoluteDifference( input[0], input[1] );

                if( !Cuda::equalSize( input[0], output ) ||
                    !Cuda::verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool AbsoluteDifference3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image_Cuda::Image > image = Cuda::uniformImages( intensity );

                Image_Function_Cuda::AbsoluteDifference( image[0], image[1], image[2] );

                if( !Cuda::verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseAnd2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image_Cuda::Image > input = Cuda::uniformImages( intensity );

                Bitmap_Image_Cuda::Image output = Image_Function_Cuda::BitwiseAnd( input[0], input[1] );

                if( !Cuda::equalSize( input[0], output ) || !Cuda::verifyImage( output, intensity[0] & intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseAnd3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image_Cuda::Image > image = Cuda::uniformImages( intensity );

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
                std::vector < Bitmap_Image_Cuda::Image > input = Cuda::uniformImages( intensity );

                Bitmap_Image_Cuda::Image output = Image_Function_Cuda::BitwiseOr( input[0], input[1] );

                if( !Cuda::equalSize( input[0], output ) || !Cuda::verifyImage( output, intensity[0] | intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseOr3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image_Cuda::Image > image = Cuda::uniformImages( intensity );

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
                std::vector < Bitmap_Image_Cuda::Image > input = Cuda::uniformImages( intensity );

                Bitmap_Image_Cuda::Image output = Image_Function_Cuda::BitwiseXor( input[0], input[1] );

                if( !Cuda::equalSize( input[0], output ) || !Cuda::verifyImage( output, intensity[0] ^ intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool BitwiseXor3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image_Cuda::Image > image = Cuda::uniformImages( intensity );

                Image_Function_Cuda::BitwiseXor( image[0], image[1], image[2] );

                if( !Cuda::verifyImage( image[2], intensity[0] ^ intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToGrayScale1ParameterTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 1 );
                Bitmap_Image_Cuda::Image input = Cuda::uniformColorImage( intensity[0] );

                Bitmap_Image_Cuda::Image output = Image_Function_Cuda::ConvertToGrayScale( input );

                if( !Cuda::verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToGrayScale2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image_Cuda::Image input = Cuda::uniformColorImage( intensity[0] );
                Bitmap_Image_Cuda::Image output( input.width(), input.height() );

                output.fill( intensity[1] );

                Image_Function_Cuda::ConvertToGrayScale( input, output );

                if( !Cuda::verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToRgb1ParameterTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 1 );
                Bitmap_Image_Cuda::Image input = Cuda::uniformImage( intensity[0] );

                Bitmap_Image_Cuda::Image output = Image_Function_Cuda::ConvertToRgb( input );

                if( !Cuda::verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool ConvertToRgb2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image_Cuda::Image input = Cuda::uniformImage( intensity[0] );
                Bitmap_Image_Cuda::Image output( input.width(), input.height(), Bitmap_Image_Cuda::RGB );

                output.fill( intensity[1] );

                Image_Function_Cuda::ConvertToRgb( input, output );

                if( !Cuda::verifyImage( output, intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool GammaCorrection3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                Bitmap_Image_Cuda::Image input = Cuda::uniformImage( intensity );

                double a     = randomValue <uint32_t>( 100 ) / 100.0;
                double gamma = randomValue <uint32_t>( 300 ) / 100.0;

                Bitmap_Image_Cuda::Image output = Image_Function_Cuda::GammaCorrection( input, a, gamma );

                double value = a * pow( intensity / 255.0, gamma ) * 255 + 0.5;
                uint8_t corrected = 0;

                if( value < 256 )
                    corrected = static_cast<uint8_t>(value);
                else
                    corrected = 255;

                if( !Cuda::verifyImage( output, corrected ) )
                    return false;
            }

            return true;
        }

        bool GammaCorrection4ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image_Cuda::Image > input = Cuda::uniformImages( intensity );

                double a     = randomValue <uint32_t>( 100 ) / 100.0;
                double gamma = randomValue <uint32_t>( 300 ) / 100.0;

                Image_Function_Cuda::GammaCorrection( input[0], input[1], a, gamma );

                double value = a * pow( intensity[0] / 255.0, gamma ) * 255 + 0.5;
                uint8_t corrected = 0;

                if( value < 256 )
                    corrected = static_cast<uint8_t>(value);
                else
                    corrected = 255;

                if( !Cuda::verifyImage( input[1], corrected ) )
                    return false;
            }

            return true;
        }

        bool Histogram1ParameterTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                Bitmap_Image_Cuda::Image image = Cuda::uniformImage( intensity );

                std::vector < uint32_t > histogram = Image_Function_Cuda::Histogram( image );

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
                Bitmap_Image_Cuda::Image image = Cuda::uniformImage( intensity );

                std::vector < uint32_t > histogram;
                Image_Function_Cuda::Histogram( image, histogram );

                if( histogram.size() != 256u || histogram[intensity] != image.width() * image.height() ||
                    std::accumulate( histogram.begin(), histogram.end(), 0u )  != image.width() * image.height() )
                    return false;
            }

            return true;
        }

        bool Invert1ParameterTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                Bitmap_Image_Cuda::Image input = Cuda::uniformImage( intensity );

                Bitmap_Image_Cuda::Image output = Image_Function_Cuda::Invert( input );

                if( !Cuda::equalSize( input, output ) || !Cuda::verifyImage( output, ~intensity ) )
                    return false;
            }

            return true;
        }

        bool Invert2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image_Cuda::Image > input = Cuda::uniformImages( intensity );

                Image_Function_Cuda::Invert( input[0], input[1] );

                if( !Cuda::verifyImage( input[1], ~intensity[0] ) )
                    return false;
            }

            return true;
        }

        bool LookupTable2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image_Cuda::Image input = Image_Function_Cuda::ConvertToCuda(randomImage( intensity ));

                std::vector < uint8_t > lookupTable( 256, 0 );

                lookupTable[intensity[0]] = intensityValue();
                lookupTable[intensity[1]] = intensityValue();

                Bitmap_Image_Cuda::Image output = Image_Function_Cuda::LookupTable( input, lookupTable );

                std::vector < uint8_t > normalized( 2 );

                normalized[0] = lookupTable[intensity[0]];
                normalized[1] = lookupTable[intensity[1]];

                if( !Cuda::verifyImage( output, normalized ) )
                    return false;
            }

            return true;
        }

        bool LookupTable3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                Bitmap_Image_Cuda::Image input = Image_Function_Cuda::ConvertToCuda(randomImage( intensity ));
                Bitmap_Image_Cuda::Image output( input.width(), input.height() );

                output.fill( intensityValue() );

                std::vector < uint8_t > lookupTable( 256, 0 );

                lookupTable[intensity[0]] = intensityValue();
                lookupTable[intensity[1]] = intensityValue();

                Image_Function_Cuda::LookupTable( input, output, lookupTable );

                std::vector < uint8_t > normalized( 2 );

                normalized[0] = lookupTable[intensity[0]];
                normalized[1] = lookupTable[intensity[1]];

                if( !Cuda::verifyImage( output, normalized ) )
                    return false;
            }

            return true;
        }

        bool Maximum2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image_Cuda::Image > input = Cuda::uniformImages( intensity );

                Bitmap_Image_Cuda::Image output = Image_Function_Cuda::Maximum( input[0], input[1] );

                if( !Cuda::equalSize( input[0], output ) ||
                    !Cuda::verifyImage( output, intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Maximum3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image_Cuda::Image > image = Cuda::uniformImages( intensity );

                Image_Function_Cuda::Maximum( image[0], image[1], image[2] );

                if( !Cuda::verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Minimum2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image_Cuda::Image > input = Cuda::uniformImages( intensity );

                Bitmap_Image_Cuda::Image output = Image_Function_Cuda::Minimum( input[0], input[1] );

                if( !Cuda::equalSize( input[0], output ) ||
                    !Cuda::verifyImage( output, intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Minimum3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image_Cuda::Image > image = Cuda::uniformImages( intensity );

                Image_Function_Cuda::Minimum( image[0], image[1], image[2] );

                if( !Cuda::verifyImage( image[2], intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                    return false;
            }

            return true;
        }

        bool Subtract2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image_Cuda::Image > input = Cuda::uniformImages( intensity );

                Bitmap_Image_Cuda::Image output = Image_Function_Cuda::Subtract( input[0], input[1] );

                if( !Cuda::equalSize( input[0], output ) ||
                    !Cuda::verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ) )
                    return false;
            }

            return true;
        }

        bool Subtract3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 3 );
                std::vector < Bitmap_Image_Cuda::Image > image = Cuda::uniformImages( intensity );

                Image_Function_Cuda::Subtract( image[0], image[1], image[2] );

                if( !Cuda::verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ) )
                    return false;
            }

            return true;
        }

        bool Threshold2ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                Bitmap_Image_Cuda::Image input = Cuda::uniformImage( intensity );

                uint8_t threshold = randomValue <uint8_t>( 255 );

                Image_Function_Cuda::Image output = Image_Function_Cuda::Threshold( input, threshold );

                if( !Cuda::verifyImage( output, intensity < threshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool Threshold3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image_Cuda::Image > input = Cuda::uniformImages( intensity );

                uint8_t threshold = randomValue <uint8_t>( 255 );

                Image_Function_Cuda::Threshold( input[0], input[1], threshold );

                if( !Cuda::verifyImage( input[1], intensity[0] < threshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool ThresholdDouble3ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                uint8_t intensity = intensityValue();
                Bitmap_Image_Cuda::Image input = Cuda::uniformImage( intensity );

                uint8_t minThreshold = randomValue <uint8_t>( 255 );
                uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

                Bitmap_Image_Cuda::Image output = Image_Function_Cuda::Threshold( input, minThreshold, maxThreshold );

                if( !Cuda::verifyImage( output, intensity < minThreshold || intensity > maxThreshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }

        bool ThresholdDouble4ParametersTest()
        {
            for( uint32_t i = 0; i < runCount(); ++i ) {
                std::vector < uint8_t > intensity = intensityArray( 2 );
                std::vector < Bitmap_Image_Cuda::Image > input = Cuda::uniformImages( intensity );

                uint8_t minThreshold = randomValue <uint8_t>( 255 );
                uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

                Image_Function_Cuda::Threshold( input[0], input[1], minThreshold, maxThreshold );

                if( !Cuda::verifyImage( input[1], intensity[0] < minThreshold || intensity[0] > maxThreshold ? 0 : 255 ) )
                    return false;
            }

            return true;
        }
    };
};
