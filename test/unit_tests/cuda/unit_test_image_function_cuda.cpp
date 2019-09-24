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

    bool AbsoluteDifference2ParametersTest()
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

    bool AbsoluteDifference3ParametersTest()
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

    bool BitwiseAnd2ParametersTest()
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

    bool BitwiseAnd3ParametersTest()
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

    bool BitwiseOr2ParametersTest()
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

    bool BitwiseOr3ParametersTest()
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

    bool BitwiseXor2ParametersTest()
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

    bool BitwiseXor3ParametersTest()
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

    bool ConvertToGrayScale1ParameterTest()
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

    bool ConvertToGrayScale2ParametersTest()
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

    bool ConvertToRgb1ParameterTest()
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

    bool ConvertToRgb2ParametersTest()
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

    bool FlipForm1()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            PenguinV_Image::Image input = uniformImage( intensity[0] );

            const bool horizontalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
            const bool verticalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
            const uint32_t xCorrection = input.width() % 2;
            const uint32_t yCorrection = input.height() % 2;

            if (verticalFlip) {
                if (input.height() > 1)
                    Image_Function::Fill(input, 0, 0, input.width(), input.height() / 2, intensity[1]);
            }
            else if (horizontalFlip) {
                if (input.width() > 1)
                    Image_Function::Fill(input, 0, 0, input.width() / 2, input.height(), intensity[1]);
            }

            const PenguinV_Image::Image output = Image_Function_Cuda::Flip( input, horizontalFlip, verticalFlip );

            if( !equalSize( output, input.width(), input.height() ))
                return false;

            if (verticalFlip) {
                if( !Cuda::verifyImage( output, 0, 0, input.width(), input.height() / 2 + yCorrection, intensity[0] ) )
                    return false;
                if((input.height() > 1) && !Cuda::verifyImage( output, 0, input.height() / 2 + yCorrection, input.width(), input.height() / 2, intensity[1] ) )
                    return false;
            }
            else {
                if( !Cuda::verifyImage( output, 0, 0, input.width() / 2 + xCorrection, input.height(), intensity[0] ) )
                    return false;
                if((input.width() > 1) && !Cuda::verifyImage( output, input.width() / 2 + xCorrection, 0, input.width() / 2, input.height(), intensity[horizontalFlip ? 1 : 0] ) )
                    return false;
            }
        }

        return true;
    }

    bool FlipForm2()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const uint8_t intensityFill = intensityValue();
            std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

            const bool horizontalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
            const bool verticalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
            const uint32_t xCorrection = input[0].width() % 2;
            const uint32_t yCorrection = input[0].height() % 2;

            if (verticalFlip) {
                if (input[0].height() > 1)
                    Image_Function::Fill(input[0], 0, 0, input[0].width(), input[0].height() / 2, intensityFill);
            }
            else if (horizontalFlip) {
                if (input[0].width() > 1)
                    Image_Function::Fill(input[0], 0, 0, input[0].width() / 2, input[0].height(), intensityFill);
            }

            Image_Function_Cuda::Flip( input[0], input[1], horizontalFlip, verticalFlip );

            if (verticalFlip) {
                if( !Cuda::verifyImage( input[1], 0, 0, input[1].width(), input[1].height() / 2 + yCorrection, intensity[0] ) )
                    return false;
                if((input[0].height() > 1) && !Cuda::verifyImage( input[1], 0, input[1].height() / 2 + yCorrection, input[1].width(), input[1].height() / 2, intensityFill ) )
                    return false;
            }
            else {
                if( !Cuda::verifyImage( input[1], 0, 0, input[1].width() / 2 + xCorrection, input[1].height(), intensity[0] ) )
                    return false;
                if((input[0].width() > 1) && !Cuda::verifyImage( input[1], input[1].width() / 2 + xCorrection, 0, input[1].width() / 2, input[1].height(),
                                                                horizontalFlip ? intensityFill : intensity[0] ) )
                    return false;
            }
        }

        return true;
    }

    bool FlipForm3()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            PenguinV_Image::Image input = uniformImage( intensity[0] );

            uint32_t roiX, roiY, roiWidth, roiHeight;
            generateRoi( input, roiX, roiY, roiWidth, roiHeight );

            const bool horizontalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
            const bool verticalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
            const uint32_t xCorrection = roiWidth % 2;
            const uint32_t yCorrection = roiHeight % 2;

            if (verticalFlip) {
                if (roiHeight > 1)
                    Image_Function::Fill(input, roiX, roiY, roiWidth, roiHeight / 2, intensity[1]);
            }
            else if (horizontalFlip) {
                if (roiWidth > 1)
                    Image_Function::Fill(input, roiX, roiY, roiWidth / 2, roiHeight, intensity[1]);
            }

            const PenguinV_Image::Image output = Image_Function_Cuda::Flip( input, roiX, roiY, roiWidth, roiHeight, horizontalFlip, verticalFlip );

            if( !equalSize( output, roiWidth, roiHeight ))
                return false;

            if (verticalFlip) {
                if( !Cuda::verifyImage( output, 0, 0, roiWidth, roiHeight / 2 + yCorrection, intensity[0] ) )
                    return false;
                if((roiHeight > 1) && !Cuda::verifyImage( output, 0, roiHeight / 2 + yCorrection, roiWidth, roiHeight / 2, intensity[1] ) )
                    return false;
            }
            else {
                if( !Cuda::verifyImage( output, 0, 0, roiWidth / 2 + xCorrection, roiHeight, intensity[0] ) )
                    return false;
                if((roiWidth > 1) && !Cuda::verifyImage( output, roiWidth / 2 + xCorrection, 0, roiWidth / 2, roiHeight, intensity[horizontalFlip ? 1 : 0] ) )
                    return false;
            }
        }

        return true;
    }

    bool FlipForm4()
    {
        for( uint32_t i = 0; i < runCount(); ++i ) {
            const std::vector < uint8_t > intensity = intensityArray( 2 );
            const uint8_t intensityFill = intensityValue();
            std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

            std::vector < uint32_t > roiX, roiY;
            uint32_t roiWidth, roiHeight;
            generateRoi( image, roiX, roiY, roiWidth, roiHeight );

            const bool horizontalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
            const bool verticalFlip = (randomValue<uint32_t>( 0, 2 ) == 0);
            const uint32_t xCorrection = roiWidth % 2;
            const uint32_t yCorrection = roiHeight % 2;

            if (verticalFlip) {
                if (roiHeight > 1)
                    Image_Function::Fill(image[0], roiX[0], roiY[0], roiWidth, roiHeight / 2, intensityFill);
            }
            else if (horizontalFlip) {
                if (roiWidth > 1)
                    Image_Function::Fill(image[0], roiX[0], roiY[0], roiWidth / 2, roiHeight, intensityFill);
            }

            Image_Function_Cuda::Flip( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, horizontalFlip, verticalFlip );

            if (verticalFlip) {
                if( !Cuda::verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight / 2 + yCorrection, intensity[0] ) )
                    return false;
                if((roiHeight > 1) && !Cuda::verifyImage( image[1], roiX[1], roiY[1] + roiHeight / 2 + yCorrection, roiWidth, roiHeight / 2, intensityFill ) )
                    return false;
            }
            else {
                if( !Cuda::verifyImage( image[1], roiX[1], roiY[1], roiWidth / 2 + xCorrection, roiHeight, intensity[0] ) )
                    return false;
                if( (roiWidth > 1) && !Cuda::verifyImage( image[1], roiX[1] + roiWidth / 2 + xCorrection, roiY[1], roiWidth / 2, roiHeight,
                                                        horizontalFlip ? intensityFill : intensity[0] ) )
                    return false;
            }
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

            const PenguinV_Image::Image output = Image_Function_Cuda::GammaCorrection( input, a, gamma );

            const double value = a * pow( intensity / 255.0, gamma ) * 255 + 0.5;
            const uint8_t corrected = ( value < 256 ) ? static_cast<uint8_t>(value) : 255u;

            if( !Cuda::verifyImage( output, corrected ) )
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

            Image_Function_Cuda::GammaCorrection( input[0], input[1], a, gamma );

            const double value = a * pow( intensity[0] / 255.0, gamma ) * 255 + 0.5;
            const uint8_t corrected = ( value < 256 ) ? static_cast<uint8_t>(value) : 255u;

            if( !Cuda::verifyImage( input[1], corrected ) )
                return false;
        }

        return true;
    }

    bool Histogram1ParameterTest()
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

    bool Histogram2ParametersTest()
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

    bool Invert1ParameterTest()
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

    bool Invert2ParametersTest()
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

    bool LookupTable2ParametersTest()
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

    bool LookupTable3ParametersTest()
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

    bool Maximum2ParametersTest()
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

    bool Maximum3ParametersTest()
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

    bool Minimum2ParametersTest()
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

    bool Minimum3ParametersTest()
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

    bool Subtract2ParametersTest()
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

    bool Subtract3ParametersTest()
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

    bool Threshold2ParametersTest()
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

    bool Threshold3ParametersTest()
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

    bool ThresholdDouble3ParametersTest()
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

    bool ThresholdDouble4ParametersTest()
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
    ADD_TEST( framework, image_function_cuda::AbsoluteDifference2ParametersTest );
    ADD_TEST( framework, image_function_cuda::AbsoluteDifference3ParametersTest );

    ADD_TEST( framework, image_function_cuda::BitwiseAnd2ParametersTest );
    ADD_TEST( framework, image_function_cuda::BitwiseAnd3ParametersTest );

    ADD_TEST( framework, image_function_cuda::BitwiseOr2ParametersTest );
    ADD_TEST( framework, image_function_cuda::BitwiseOr3ParametersTest );

    ADD_TEST( framework, image_function_cuda::BitwiseXor2ParametersTest );
    ADD_TEST( framework, image_function_cuda::BitwiseXor3ParametersTest );

    ADD_TEST( framework, image_function_cuda::ConvertToGrayScale1ParameterTest );
    ADD_TEST( framework, image_function_cuda::ConvertToGrayScale2ParametersTest );

    ADD_TEST( framework, image_function_cuda::ConvertToRgb1ParameterTest );
    ADD_TEST( framework, image_function_cuda::ConvertToRgb2ParametersTest );

    ADD_TEST( framework, image_function_cuda::FlipForm1 );
    ADD_TEST( framework, image_function_cuda::FlipForm2 );
    ADD_TEST( framework, image_function_cuda::FlipForm3 );
    ADD_TEST( framework, image_function_cuda::FlipForm4 );

    ADD_TEST( framework, image_function_cuda::GammaCorrection3ParametersTest );
    ADD_TEST( framework, image_function_cuda::GammaCorrection4ParametersTest );

    ADD_TEST( framework, image_function_cuda::Histogram1ParameterTest );
    ADD_TEST( framework, image_function_cuda::Histogram2ParametersTest );

    ADD_TEST( framework, image_function_cuda::Invert1ParameterTest );
    ADD_TEST( framework, image_function_cuda::Invert2ParametersTest );

    ADD_TEST( framework, image_function_cuda::LookupTable2ParametersTest );
    ADD_TEST( framework, image_function_cuda::LookupTable3ParametersTest );

    ADD_TEST( framework, image_function_cuda::Maximum2ParametersTest );
    ADD_TEST( framework, image_function_cuda::Maximum3ParametersTest );

    ADD_TEST( framework, image_function_cuda::Minimum2ParametersTest );
    ADD_TEST( framework, image_function_cuda::Minimum3ParametersTest );

    ADD_TEST( framework, image_function_cuda::Subtract2ParametersTest );
    ADD_TEST( framework, image_function_cuda::Subtract3ParametersTest );

    ADD_TEST( framework, image_function_cuda::Threshold2ParametersTest );
    ADD_TEST( framework, image_function_cuda::Threshold3ParametersTest );

    ADD_TEST( framework, image_function_cuda::ThresholdDouble3ParametersTest );
    ADD_TEST( framework, image_function_cuda::ThresholdDouble4ParametersTest );
}
