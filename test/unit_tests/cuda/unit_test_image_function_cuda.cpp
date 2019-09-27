#include <cmath>
#include <numeric>
#include "../../../src/cuda/image_function_cuda.cuh"
#include "../unit_test_helper.h"
#include "unit_test_helper_cuda.cuh"
#include "unit_test_image_function_cuda.h"

namespace
{
    const PenguinV_Image::ImageCuda reference;
}

namespace image_function_cuda
{
    using namespace Unit_Test;

    bool AbsoluteDifferenceForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const PenguinV_Image::Image output = Image_Function_Cuda::AbsoluteDifference( input[0], input[1] );

            if( !equalSize( input[0], output ) ||
                !Cuda::verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
                return false;
        }

        return true;
    }

    bool AbsoluteDifferenceForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_Cuda::AbsoluteDifference( image[0], image[1], image[2] );

            if( !Cuda::verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) )
                return false;
        }

        return true;
    }

    bool BitwiseAndForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const PenguinV_Image::Image output = Image_Function_Cuda::BitwiseAnd( input[0], input[1] );

            if( !equalSize( input[0], output ) || !Cuda::verifyImage( output, intensity[0] & intensity[1] ) )
                return false;
        }

        return true;
    }

    bool BitwiseAndForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_Cuda::BitwiseAnd( image[0], image[1], image[2] );

            if( !Cuda::verifyImage( image[2], intensity[0] & intensity[1] ) )
                return false;
        }

        return true;
    }

    bool BitwiseOrForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const PenguinV_Image::Image output = Image_Function_Cuda::BitwiseOr( input[0], input[1] );

            if( !equalSize( input[0], output ) || !Cuda::verifyImage( output, intensity[0] | intensity[1] ) )
                return false;
        }

        return true;
    }

    bool BitwiseOrForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_Cuda::BitwiseOr( image[0], image[1], image[2] );

            if( !Cuda::verifyImage( image[2], intensity[0] | intensity[1] ) )
                return false;
        }

        return true;
    }

    bool BitwiseXorForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const PenguinV_Image::Image output = Image_Function_Cuda::BitwiseXor( input[0], input[1] );

            if( !equalSize( input[0], output ) || !Cuda::verifyImage( output, intensity[0] ^ intensity[1] ) )
                return false;
        }

        return true;
    }

    bool BitwiseXorForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_Cuda::BitwiseXor( image[0], image[1], image[2] );

            if( !Cuda::verifyImage( image[2], intensity[0] ^ intensity[1] ) )
                return false;
        }

        return true;
    }

    bool ConvertToGrayScaleForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 1 );
            const PenguinV_Image::Image input = uniformRGBImage( intensity[0], reference );

            const PenguinV_Image::Image output = Image_Function_Cuda::ConvertToGrayScale( input );

            if( !Cuda::verifyImage( output, intensity[0] ) )
                return false;
        }

        return true;
    }

    bool ConvertToGrayScaleForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const PenguinV_Image::Image input = uniformRGBImage( intensity[0], reference );
            PenguinV_Image::ImageCuda output( input.width(), input.height() );

            output.fill( intensity[1] );

            Image_Function_Cuda::ConvertToGrayScale( input, output );

            if( !Cuda::verifyImage( output, intensity[0] ) )
                return false;
        }

        return true;
    }

    bool ConvertToRgbForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 1 );
            const PenguinV_Image::Image input = uniformImage( intensity[0], 0, 0, reference );

            const PenguinV_Image::Image output = Image_Function_Cuda::ConvertToRgb( input );

            if( !Cuda::verifyImage( output, intensity[0] ) )
                return false;
        }

        return true;
    }

    bool ConvertToRgbForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const PenguinV_Image::Image input = uniformImage( intensity[0], 0, 0, reference );
            PenguinV_Image::ImageCuda output( input.width(), input.height(), PenguinV_Image::RGB );

            output.fill( intensity[1] );

            Image_Function_Cuda::ConvertToRgb( input, output );

            if( !Cuda::verifyImage( output, intensity[0] ) )
                return false;
        }

        return true;
    }

    bool GammaCorrectionForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const uint8_t intensity = intensityValue();
            const PenguinV_Image::Image input = uniformImage( intensity, 0, 0, reference );

            const double a     = randomValue <uint32_t>( 100 ) / 100.0;
            const double gamma = randomValue <uint32_t>( 300 ) / 100.0;

            const PenguinV_Image::Image output = Image_Function_Cuda::GammaCorrection( input, a, gamma );

            const double value = a * pow( intensity / 255.0, gamma ) * 255 + 0.5;
            const uint8_t corrected = ( value < 256 ) ? static_cast<uint8_t>(value) : 255u;

            if( !Cuda::verifyImage( output, corrected ) )
                return false;
        }

        return true;
    }

    bool GammaCorrectionForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const double a     = randomValue <uint32_t>( 100 ) / 100.0;
            const double gamma = randomValue <uint32_t>( 300 ) / 100.0;

            Image_Function_Cuda::GammaCorrection( input[0], input[1], a, gamma );

            const double value = a * pow( intensity[0] / 255.0, gamma ) * 255 + 0.5;
            const uint8_t corrected = ( value < 256 ) ? static_cast<uint8_t>(value) : 255u;

            if( !Cuda::verifyImage( input[1], corrected ) )
                return false;
        }

        return true;
    }

    bool HistogramForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const uint8_t intensity = intensityValue();
            const PenguinV_Image::Image image = uniformImage( intensity, 0, 0, reference );

            const std::vector < uint32_t > histogram = Image_Function_Cuda::Histogram( image );

            if( histogram.size() != 256u || histogram[intensity] != image.width() * image.height() ||
                std::accumulate( histogram.begin(), histogram.end(), 0u )  != image.width() * image.height() )
                return false;
        }

        return true;
    }

    bool HistogramForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const uint8_t intensity = intensityValue();
            const PenguinV_Image::Image image = uniformImage( intensity, 0, 0, reference );

            std::vector < uint32_t > histogram;
            Image_Function_Cuda::Histogram( image, histogram );

            if( histogram.size() != 256u || histogram[intensity] != image.width() * image.height() ||
                std::accumulate( histogram.begin(), histogram.end(), 0u )  != image.width() * image.height() )
                return false;
        }

        return true;
    }

    bool InvertForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const uint8_t intensity = intensityValue();
            const PenguinV_Image::Image input = uniformImage( intensity, 0, 0, reference );

            const PenguinV_Image::Image output = Image_Function_Cuda::Invert( input );

            if( !equalSize( input, output ) || !Cuda::verifyImage( output, ~intensity ) )
                return false;
        }

        return true;
    }

    bool InvertForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            Image_Function_Cuda::Invert( input[0], input[1] );

            if( !Cuda::verifyImage( input[1], ~intensity[0] ) )
                return false;
        }

        return true;
    }

    bool LookupTableForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const PenguinV_Image::Image input = Image_Function_Cuda::ConvertToCuda(randomImage( intensity ));

            std::vector < uint8_t > lookupTable( 256, 0 );

            lookupTable[intensity[0]] = intensityValue();
            lookupTable[intensity[1]] = intensityValue();

            const PenguinV_Image::Image output = Image_Function_Cuda::LookupTable( input, lookupTable );

            std::vector < uint8_t > normalized( 2 );

            normalized[0] = lookupTable[intensity[0]];
            normalized[1] = lookupTable[intensity[1]];

            if( !Cuda::verifyImage( output, normalized ) )
                return false;
        }

        return true;
    }

    bool LookupTableForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const PenguinV_Image::Image input = Image_Function_Cuda::ConvertToCuda(randomImage( intensity ));
            PenguinV_Image::ImageCuda output( input.width(), input.height() );

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

    bool MaximumForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const PenguinV_Image::Image output = Image_Function_Cuda::Maximum( input[0], input[1] );

            if( !equalSize( input[0], output ) ||
                !Cuda::verifyImage( output, intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
                return false;
        }

        return true;
    }

    bool MaximumForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_Cuda::Maximum( image[0], image[1], image[2] );

            if( !Cuda::verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] : intensity[1] ) )
                return false;
        }

        return true;
    }

    bool MinimumForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const PenguinV_Image::Image output = Image_Function_Cuda::Minimum( input[0], input[1] );

            if( !equalSize( input[0], output ) ||
                !Cuda::verifyImage( output, intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                return false;
        }

        return true;
    }

    bool MinimumForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_Cuda::Minimum( image[0], image[1], image[2] );

            if( !Cuda::verifyImage( image[2], intensity[0] < intensity[1] ? intensity[0] : intensity[1] ) )
                return false;
        }

        return true;
    }

    bool SubtractForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const PenguinV_Image::Image output = Image_Function_Cuda::Subtract( input[0], input[1] );

            if( !equalSize( input[0], output ) ||
                !Cuda::verifyImage( output, intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ) )
                return false;
        }

        return true;
    }

    bool SubtractForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 3 );
            std::vector < PenguinV_Image::Image > image = uniformImages( intensity, reference );

            Image_Function_Cuda::Subtract( image[0], image[1], image[2] );

            if( !Cuda::verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0 ) )
                return false;
        }

        return true;
    }

    bool ThresholdForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const uint8_t intensity = intensityValue();
            const PenguinV_Image::Image input = uniformImage( intensity, 0, 0, reference );

            const uint8_t threshold = randomValue <uint8_t>( 255 );

            const Image_Function_Cuda::Image output = Image_Function_Cuda::Threshold( input, threshold );

            if( !Cuda::verifyImage( output, intensity < threshold ? 0 : 255 ) )
                return false;
        }

        return true;
    }

    bool ThresholdForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const uint8_t threshold = randomValue <uint8_t>( 255 );

            Image_Function_Cuda::Threshold( input[0], input[1], threshold );

            if( !Cuda::verifyImage( input[1], intensity[0] < threshold ? 0 : 255 ) )
                return false;
        }

        return true;
    }

    bool ThresholdDoubleForm1Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const uint8_t intensity = intensityValue();
            const PenguinV_Image::Image input = uniformImage( intensity, 0, 0, reference );

            const uint8_t minThreshold = randomValue <uint8_t>( 255 );
            const uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

            const PenguinV_Image::Image output = Image_Function_Cuda::Threshold( input, minThreshold, maxThreshold );

            if( !Cuda::verifyImage( output, intensity < minThreshold || intensity > maxThreshold ? 0 : 255 ) )
                return false;
        }

        return true;
    }

    bool ThresholdDoubleForm2Test()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            std::vector < PenguinV_Image::Image > input = uniformImages( intensity, reference );

            const uint8_t minThreshold = randomValue <uint8_t>( 255 );
            const uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

            Image_Function_Cuda::Threshold( input[0], input[1], minThreshold, maxThreshold );

            if( !Cuda::verifyImage( input[1], intensity[0] < minThreshold || intensity[0] > maxThreshold ? 0 : 255 ) )
                return false;
        }

        return true;
    }
}

void addTests_Image_Function_Cuda( UnitTestFramework & framework )
{
    ADD_TEST( framework, image_function_cuda::AbsoluteDifferenceForm1Test );
    ADD_TEST( framework, image_function_cuda::AbsoluteDifferenceForm2Test );

    ADD_TEST( framework, image_function_cuda::BitwiseAndForm1Test );
    ADD_TEST( framework, image_function_cuda::BitwiseAndForm2Test );

    ADD_TEST( framework, image_function_cuda::BitwiseOrForm1Test );
    ADD_TEST( framework, image_function_cuda::BitwiseOrForm2Test );

    ADD_TEST( framework, image_function_cuda::BitwiseXorForm1Test );
    ADD_TEST( framework, image_function_cuda::BitwiseXorForm2Test );

    ADD_TEST( framework, image_function_cuda::ConvertToGrayScaleForm1Test );
    ADD_TEST( framework, image_function_cuda::ConvertToGrayScaleForm2Test );

    ADD_TEST( framework, image_function_cuda::ConvertToRgbForm1Test );
    ADD_TEST( framework, image_function_cuda::ConvertToRgbForm2Test );

    ADD_TEST( framework, image_function_cuda::GammaCorrectionForm1Test );
    ADD_TEST( framework, image_function_cuda::GammaCorrectionForm2Test );

    ADD_TEST( framework, image_function_cuda::HistogramForm1Test );
    ADD_TEST( framework, image_function_cuda::HistogramForm2Test );

    ADD_TEST( framework, image_function_cuda::InvertForm1Test );
    ADD_TEST( framework, image_function_cuda::InvertForm2Test );

    ADD_TEST( framework, image_function_cuda::LookupTableForm1Test );
    ADD_TEST( framework, image_function_cuda::LookupTableForm2Test );

    ADD_TEST( framework, image_function_cuda::MaximumForm1Test );
    ADD_TEST( framework, image_function_cuda::MaximumForm2Test );

    ADD_TEST( framework, image_function_cuda::MinimumForm1Test );
    ADD_TEST( framework, image_function_cuda::MinimumForm2Test );

    ADD_TEST( framework, image_function_cuda::SubtractForm1Test );
    ADD_TEST( framework, image_function_cuda::SubtractForm2Test );

    ADD_TEST( framework, image_function_cuda::ThresholdForm1Test );
    ADD_TEST( framework, image_function_cuda::ThresholdForm2Test );

    ADD_TEST( framework, image_function_cuda::ThresholdDoubleForm1Test );
    ADD_TEST( framework, image_function_cuda::ThresholdDoubleForm2Test );
}
