#include <cmath>
#include <numeric>
#include "../../../src/opencl/image_function_opencl.h"
#include "../unit_test_helper.h"
#include "unit_test_helper_opencl.h"
#include "unit_test_image_function_opencl.h"

namespace
{
    const PenguinV_Image::ImageOpenCL reference;
}

namespace image_function_opencl
{
    using namespace Unit_Test;

    bool AbsoluteDifference2ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const PenguinV_Image::Image output = Image_Function_OpenCL::AbsoluteDifference( input[0], input[1] );

            if( !equalSize( input[0], output ) ||
                !OpenCL::verifyImage( output, intensity[0] > intensity[1] ? static_cast<uint8_t>( intensity[0] - intensity[1] ) : static_cast<uint8_t>( intensity[1] - intensity[0] ) ) )
                return false;
        }

        return true;
    }

    bool AbsoluteDifference3ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_OpenCL::AbsoluteDifference( image[0], image[1], image[2] );

            if( !OpenCL::verifyImage( image[2], intensity[0] > intensity[1] ? static_cast<uint8_t>( intensity[0] - intensity[1] ) : static_cast<uint8_t>( intensity[1] - intensity[0] ) ) )
                return false;
        }

        return true;
    }

    bool BitwiseAnd2ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const PenguinV_Image::Image output = Image_Function_OpenCL::BitwiseAnd( input[0], input[1] );

            if( !equalSize( input[0], output ) || !OpenCL::verifyImage( output, intensity[0] & intensity[1] ) )
                return false;
        }

        return true;
    }

    bool BitwiseAnd3ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_OpenCL::BitwiseAnd( image[0], image[1], image[2] );

            if( !OpenCL::verifyImage( image[2], intensity[0] & intensity[1] ) )
                return false;
        }

        return true;
    }

    bool BitwiseOr2ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const PenguinV_Image::Image output = Image_Function_OpenCL::BitwiseOr( input[0], input[1] );

            if( !equalSize( input[0], output ) || !OpenCL::verifyImage( output, intensity[0] | intensity[1] ) )
                return false;
        }

        return true;
    }

    bool BitwiseOr3ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_OpenCL::BitwiseOr( image[0], image[1], image[2] );

            if( !OpenCL::verifyImage( image[2], intensity[0] | intensity[1] ) )
                return false;
        }

        return true;
    }

    bool BitwiseXor2ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            PenguinV_Image::Image output = Image_Function_OpenCL::BitwiseXor( input[0], input[1] );

            if( !equalSize( input[0], output ) || !OpenCL::verifyImage( output, intensity[0] ^ intensity[1] ) )
                return false;
        }

        return true;
    }

    bool BitwiseXor3ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_OpenCL::BitwiseXor( image[0], image[1], image[2] );

            if( !OpenCL::verifyImage( image[2], intensity[0] ^ intensity[1] ) )
                return false;
        }

        return true;
    }

    bool ConvertToGrayScale1ParameterTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 1 );
            const PenguinV_Image::Image input = uniformRGBImage( intensity[0], reference );

            const PenguinV_Image::Image output = Image_Function_OpenCL::ConvertToGrayScale( input );

            if( !OpenCL::verifyImage( output, intensity[0] ) )
                return false;
        }

        return true;
    }

    bool ConvertToGrayScale2ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const PenguinV_Image::Image input = uniformRGBImage( intensity[0], reference );
            PenguinV_Image::ImageOpenCL output( input.width(), input.height() );

            output.fill( intensity[1] );

            Image_Function_OpenCL::ConvertToGrayScale( input, output );

            if( !OpenCL::verifyImage( output, intensity[0] ) )
                return false;
        }

        return true;
    }

    bool ConvertToRgb1ParameterTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 1 );
            const PenguinV_Image::Image input = uniformImage( intensity[0], 0, 0, reference );

            const PenguinV_Image::Image output = Image_Function_OpenCL::ConvertToRgb( input );

            if( !OpenCL::verifyImage( output, intensity[0] ) )
                return false;
        }

        return true;
    }

    bool ConvertToRgb2ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const PenguinV_Image::Image input = uniformImage( intensity[0], 0, 0, reference );
            PenguinV_Image::ImageOpenCL output( input.width(), input.height(), PenguinV_Image::RGB );

            output.fill( intensity[1] );

            Image_Function_OpenCL::ConvertToRgb( input, output );

            if( !OpenCL::verifyImage( output, intensity[0] ) )
                return false;
        }

        return true;
    }

    bool GammaCorrection3ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const uint8_t intensity = intensityValue();
            const PenguinV_Image::Image input = uniformImage( intensity, 0, 0, reference );

            const double a     = randomValue <uint32_t>( 100 ) / 100.0;
            const double gamma = randomValue <uint32_t>( 300 ) / 100.0;

            const PenguinV_Image::Image output = Image_Function_OpenCL::GammaCorrection( input, a, gamma );

            const double value = a * pow( intensity / 255.0, gamma ) * 255 + 0.5;
            const uint8_t corrected = (value < 256) ? static_cast<uint8_t>(value) : 255u;

            if( !OpenCL::verifyImage( output, corrected ) )
                return false;
        }

        return true;
    }

    bool GammaCorrection4ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const double a     = randomValue <uint32_t>( 100 ) / 100.0;
            const double gamma = randomValue <uint32_t>( 300 ) / 100.0;

            Image_Function_OpenCL::GammaCorrection( input[0], input[1], a, gamma );

            double value = a * pow( intensity[0] / 255.0, gamma ) * 255 + 0.5;
            const uint8_t corrected = (value < 256) ? static_cast<uint8_t>(value) : 255u;

            if( !OpenCL::verifyImage( input[1], corrected ) )
                return false;
        }

        return true;
    }

    bool Histogram1ParameterTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const uint8_t intensity = intensityValue();
            const PenguinV_Image::Image image = uniformImage( intensity, 0, 0, reference );

            const std::vector < uint32_t > histogram = Image_Function_OpenCL::Histogram( image );

            if( histogram.size() != 256u || histogram[intensity] != image.width() * image.height() ||
                std::accumulate( histogram.begin(), histogram.end(), 0u )  != image.width() * image.height() )
                return false;
        }

        return true;
    }

    bool Histogram2ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const uint8_t intensity = intensityValue();
            const PenguinV_Image::Image image = uniformImage( intensity, 0, 0, reference );

            std::vector < uint32_t > histogram;
            Image_Function_OpenCL::Histogram( image, histogram );

            if( histogram.size() != 256u || histogram[intensity] != image.width() * image.height() ||
                std::accumulate( histogram.begin(), histogram.end(), 0u )  != image.width() * image.height() )
                return false;
        }

        return true;
    }

    bool Invert1ParameterTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const uint8_t intensity = intensityValue();
            const PenguinV_Image::Image input = uniformImage( intensity, 0, 0, reference );

            const PenguinV_Image::Image output = Image_Function_OpenCL::Invert( input );

            if( !equalSize( input, output ) || !OpenCL::verifyImage( output, static_cast<uint8_t>( ~intensity ) ) )
                return false;
        }

        return true;
    }

    bool Invert2ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            Image_Function_OpenCL::Invert( input[0], input[1] );

            if( !OpenCL::verifyImage( input[1], static_cast<uint8_t>( ~intensity[0] ) ) )
                return false;
        }

        return true;
    }

    bool LookupTable2ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const PenguinV_Image::Image input = Image_Function_OpenCL::ConvertToOpenCL(randomImage( intensity ));

            std::vector < uint8_t > lookupTable( 256, 0 );

            lookupTable[intensity[0]] = intensityValue();
            lookupTable[intensity[1]] = intensityValue();

            const PenguinV_Image::Image output = Image_Function_OpenCL::LookupTable( input, lookupTable );

            std::vector < uint8_t > normalized( 2 );

            normalized[0] = lookupTable[intensity[0]];
            normalized[1] = lookupTable[intensity[1]];

            if( !OpenCL::verifyImage( output, normalized ) )
                return false;
        }

        return true;
    }

    bool LookupTable3ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const PenguinV_Image::Image input = Image_Function_OpenCL::ConvertToOpenCL(randomImage( intensity ));
            PenguinV_Image::ImageOpenCL output( input.width(), input.height() );

            output.fill( intensityValue() );

            std::vector < uint8_t > lookupTable( 256, 0 );

            lookupTable[intensity[0]] = intensityValue();
            lookupTable[intensity[1]] = intensityValue();

            Image_Function_OpenCL::LookupTable( input, output, lookupTable );

            std::vector < uint8_t > normalized( 2 );

            normalized[0] = lookupTable[intensity[0]];
            normalized[1] = lookupTable[intensity[1]];

            if( !OpenCL::verifyImage( output, normalized ) )
                return false;
        }

        return true;
    }

    bool Maximum2ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const PenguinV_Image::Image output = Image_Function_OpenCL::Maximum( input[0], input[1] );

            if( !equalSize( input[0], output ) ||
                !OpenCL::verifyImage( output, intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
                return false;
        }

        return true;
    }

    bool Maximum3ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_OpenCL::Maximum( image[0], image[1], image[2] );

            if( !OpenCL::verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
                return false;
        }

        return true;
    }

    bool Minimum2ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const PenguinV_Image::Image output = Image_Function_OpenCL::Minimum( input[0], input[1] );

            if( !equalSize( input[0], output ) ||
                !OpenCL::verifyImage( output, intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                return false;
        }

        return true;
    }

    bool Minimum3ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_OpenCL::Minimum( image[0], image[1], image[2] );

            if( !OpenCL::verifyImage( image[2], intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                return false;
        }

        return true;
    }

    bool Subtract2ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const PenguinV_Image::Image output = Image_Function_OpenCL::Subtract( input[0], input[1] );

            if( !equalSize( input[0], output ) ||
                !OpenCL::verifyImage( output, intensity[0] > intensity[1] ? static_cast<uint8_t>( intensity[0] - intensity[1] ) : 0 ) )
                return false;
        }

        return true;
    }

    bool Subtract3ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_OpenCL::Subtract( image[0], image[1], image[2] );

            if( !OpenCL::verifyImage( image[2], intensity[0] > intensity[1] ? static_cast<uint8_t>( intensity[0] - intensity[1] ) : 0 ) )
                return false;
        }

        return true;
    }

    bool Threshold2ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const uint8_t intensity = intensityValue();
            const PenguinV_Image::Image input = uniformImage( intensity, 0, 0, reference );

            const uint8_t threshold = randomValue <uint8_t>( 255 );

            const Image_Function_OpenCL::Image output = Image_Function_OpenCL::Threshold( input, threshold );

            if( !OpenCL::verifyImage( output, intensity < threshold ? 0 : 255 ) )
                return false;
        }

        return true;
    }

    bool Threshold3ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const uint8_t threshold = randomValue <uint8_t>( 255 );

            Image_Function_OpenCL::Threshold( input[0], input[1], threshold );

            if( !OpenCL::verifyImage( input[1], intensity[0] < threshold ? 0 : 255 ) )
                return false;
        }

        return true;
    }

    bool ThresholdDouble3ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const uint8_t intensity = intensityValue();
            const PenguinV_Image::Image input = uniformImage( intensity, 0, 0, reference );

            const uint8_t minThreshold = randomValue <uint8_t>( 255 );
            const uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

            const PenguinV_Image::Image output = Image_Function_OpenCL::Threshold( input, minThreshold, maxThreshold );

            if( !OpenCL::verifyImage( output, intensity < minThreshold || intensity > maxThreshold ? 0 : 255 ) )
                return false;
        }

        return true;
    }

    bool ThresholdDouble4ParametersTest()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const uint8_t minThreshold = randomValue <uint8_t>( 255 );
            const uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

            Image_Function_OpenCL::Threshold( input[0], input[1], minThreshold, maxThreshold );

            if( !OpenCL::verifyImage( input[1], intensity[0] < minThreshold || intensity[0] > maxThreshold ? 0 : 255 ) )
                return false;
        }

        return true;
    }
}

void addTests_Image_Function_OpenCL( UnitTestFramework & framework )
{
    ADD_TEST( framework, image_function_opencl::AbsoluteDifference2ParametersTest );
    ADD_TEST( framework, image_function_opencl::AbsoluteDifference3ParametersTest );

    ADD_TEST( framework, image_function_opencl::BitwiseAnd2ParametersTest );
    ADD_TEST( framework, image_function_opencl::BitwiseAnd3ParametersTest );

    ADD_TEST( framework, image_function_opencl::BitwiseOr2ParametersTest );
    ADD_TEST( framework, image_function_opencl::BitwiseOr3ParametersTest );

    ADD_TEST( framework, image_function_opencl::BitwiseXor2ParametersTest );
    ADD_TEST( framework, image_function_opencl::BitwiseXor3ParametersTest );

    ADD_TEST( framework, image_function_opencl::ConvertToGrayScale1ParameterTest );
    ADD_TEST( framework, image_function_opencl::ConvertToGrayScale2ParametersTest );

    ADD_TEST( framework, image_function_opencl::ConvertToRgb1ParameterTest );
    ADD_TEST( framework, image_function_opencl::ConvertToRgb2ParametersTest );

    ADD_TEST( framework, image_function_opencl::GammaCorrection3ParametersTest );
    ADD_TEST( framework, image_function_opencl::GammaCorrection4ParametersTest );

    ADD_TEST( framework, image_function_opencl::Histogram1ParameterTest );
    ADD_TEST( framework, image_function_opencl::Histogram2ParametersTest );

    ADD_TEST( framework, image_function_opencl::Invert1ParameterTest );
    ADD_TEST( framework, image_function_opencl::Invert2ParametersTest );

    ADD_TEST( framework, image_function_opencl::LookupTable2ParametersTest );
    ADD_TEST( framework, image_function_opencl::LookupTable3ParametersTest );

    ADD_TEST( framework, image_function_opencl::Maximum2ParametersTest );
    ADD_TEST( framework, image_function_opencl::Maximum3ParametersTest );

    ADD_TEST( framework, image_function_opencl::Minimum2ParametersTest );
    ADD_TEST( framework, image_function_opencl::Minimum3ParametersTest );

    ADD_TEST( framework, image_function_opencl::Subtract2ParametersTest );
    ADD_TEST( framework, image_function_opencl::Subtract3ParametersTest );

    ADD_TEST( framework, image_function_opencl::Threshold2ParametersTest );
    ADD_TEST( framework, image_function_opencl::Threshold3ParametersTest );

    ADD_TEST( framework, image_function_opencl::ThresholdDouble3ParametersTest );
    ADD_TEST( framework, image_function_opencl::ThresholdDouble4ParametersTest );
}
