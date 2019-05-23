#include <math.h>
#include <numeric>
#include "../../src/filtering.h"
#include "../../src/function_pool.h"
#include "../../src/image_function.h"
#include "../../src/image_function_helper.h"
#include "../../src/image_function_simd.h"
#include "../../src/thread_pool.h"
#include "../../src/penguinv/cpu_identification.h"
#include "unit_test_image_function.h"
#include "unit_test_helper.h"

namespace
{
    void PrepareFunction( const std::string& namespaceName )
    {
        if ( namespaceName == "function_pool" ) {
            simd::EnableSimd( true );
            ThreadPoolMonoid::instance().resize( Unit_Test::randomValue<uint8_t>( 1, 8 ) );
        }
        else if ( namespaceName == "image_function_avx" ) {
            simd::EnableSimd( false );
            simd::EnableAvx( true );
        }
        else if ( namespaceName == "image_function_sse" ) {
            simd::EnableSimd( false );
            simd::EnableSse( true );
        }
        else if ( namespaceName == "image_function_neon" ) {
            simd::EnableSimd( false );
            simd::EnableNeon( true );
        }
    }

    void CleanupFunction(const std::string& namespaceName)
    {
        if ( (namespaceName == "image_function_avx") || (namespaceName == "image_function_sse") || (namespaceName == "image_function_neon") )
            simd::EnableSimd( true );
    }

    class FunctionRegistrator
    {
    public:
        static FunctionRegistrator& instance()
        {
            static FunctionRegistrator registrator;
            return registrator;
        }

        void add( const UnitTestFramework::testFunction test, const std::string & name )
        {
            _function[test] = name;
        }

        void set( UnitTestFramework & framework )
        {
            for (std::map < UnitTestFramework::testFunction, std::string >::const_iterator func = _function.begin(); func != _function.end(); ++func)
                framework.add( func->first, func->second );

            _function.clear();
        }

    private:
        std::map < UnitTestFramework::testFunction, std::string > _function; // container with pointer to functions and their names
    };
}

namespace Function_Template
{
    using namespace PenguinV_Image;
    using namespace Unit_Test;
    using namespace Image_Function_Helper::FunctionTable;

    bool form1_AbsoluteDifference(AbsoluteDifferenceForm1 AbsoluteDifference)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        const PenguinV_Image::Image output = AbsoluteDifference( input[0], input[1] );

        return equalSize( input[0], output ) &&
            verifyImage( output, static_cast<uint8_t>( intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) );
    }

    bool form2_AbsoluteDifference(AbsoluteDifferenceForm2 AbsoluteDifference)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

        AbsoluteDifference( image[0], image[1], image[2] );

        return verifyImage( image[2], static_cast<uint8_t>( intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) );
    }

    bool form3_AbsoluteDifference(AbsoluteDifferenceForm3 AbsoluteDifference)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const std::vector < PenguinV_Image::Image > input = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const PenguinV_Image::Image output = AbsoluteDifference( input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

        return equalSize( output, roiWidth, roiHeight ) &&
            verifyImage( output, static_cast<uint8_t>( intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) );
    }

    bool form4_AbsoluteDifference(AbsoluteDifferenceForm4 AbsoluteDifference)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ), uniformImage( intensity[2] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        AbsoluteDifference( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], image[2], roiX[2], roiY[2], roiWidth, roiHeight );

        return verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
                            static_cast<uint8_t>( intensity[0] > intensity[1] ? intensity[0] - intensity[1] : intensity[1] - intensity[0] ) );
    }

    bool form1_Accumulate(AccumulateForm1 Accumulate)
    {
        const std::vector < uint8_t > intensity = intensityArray( randomValue<uint8_t>( 1, 16 ) );
        std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        std::vector < uint32_t > result( input[0].width() * input[0].height(), 0 );

        for( std::vector < PenguinV_Image::Image >::const_iterator image = input.begin(); image != input.end(); ++image ) {
            Accumulate( *image, result );
        }

        const uint32_t sum = std::accumulate( intensity.begin(), intensity.end(), 0u );

        return std::all_of( result.begin(), result.end(), [&sum]( uint32_t v ) { return v == sum; } );
    }

    bool form2_Accumulate(AccumulateForm2 Accumulate)
    {
        const std::vector < uint8_t > intensity = intensityArray( randomValue<uint8_t>( 1, 16 ) );
        std::vector < PenguinV_Image::Image > input;

        std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t value )
        { input.push_back( uniformImage( value ) ); } );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        std::vector < uint32_t > result( roiWidth * roiHeight, 0 );

        for( size_t imageId = 0; imageId < input.size(); ++imageId ) {
            Accumulate( input[imageId], roiX[imageId], roiY[imageId], roiWidth, roiHeight, result );
        }

        const uint32_t sum = std::accumulate( intensity.begin(), intensity.end(), 0u );

        return std::all_of( result.begin(), result.end(), [&sum]( uint32_t v ) { return v == sum; } );
    }

    bool form1_BinaryDilate(BinaryDilateForm1 BinaryDilate)
    {
        std::vector< uint8_t > fillData( randomValue<uint32_t>(20, 200), 255u );
        fillData.push_back(0u);

        const PenguinV_Image::Image input = randomImage( fillData );
        PenguinV_Image::Image output = input;

        const uint32_t dilationX = randomValue<uint32_t>(1, 5);
        const uint32_t dilationY = randomValue<uint32_t>(1, 5);

        BinaryDilate(output, dilationX, dilationY);

        return equalSize( input, output ) && verifyImage( output, 255u );
    }

    bool form2_BinaryDilate(BinaryDilateForm2 BinaryDilate)
    {
        std::vector< uint8_t > fillData( randomValue<uint32_t>(20, 200), 255u );
        fillData.push_back(0u);

        const PenguinV_Image::Image input = randomImage( fillData );
        PenguinV_Image::Image output = input;

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( output, roiX, roiY, roiWidth, roiHeight );
        if ( !verifyImage(output, roiX, roiY, roiWidth, roiHeight, 0u) ) // full ROI is black, nothing to dilate
            return true;

        const uint32_t dilationX = randomValue<uint32_t>(1, 5);
        const uint32_t dilationY = randomValue<uint32_t>(1, 5);
        
        BinaryDilate( output, roiX, roiY, roiWidth, roiHeight, dilationX, dilationY );

        return verifyImage( output, roiX, roiY, roiWidth, roiHeight, 255u );
    }

    bool form1_BinaryErode(BinaryErodeForm1 BinaryErode)
    {
        std::vector< uint8_t > fillData( randomValue<uint32_t>(20, 200), 0u );
        fillData.push_back(255u);

        const PenguinV_Image::Image input = randomImage( fillData );
        PenguinV_Image::Image output = input;

        const uint32_t dilationX = randomValue<uint32_t>(1, 5);
        const uint32_t dilationY = randomValue<uint32_t>(1, 5);

        BinaryErode(output, dilationX, dilationY);

        return equalSize( input, output ) && verifyImage( output, 0u );
    }

    bool form2_BinaryErode(BinaryErodeForm2 BinaryErode)
    {
        std::vector< uint8_t > fillData( randomValue<uint32_t>(20, 200), 0u );
        fillData.push_back(255u);

        const PenguinV_Image::Image input = randomImage( fillData );
        PenguinV_Image::Image output = input;

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( output, roiX, roiY, roiWidth, roiHeight );
        if ( !verifyImage(output, roiX, roiY, roiWidth, roiHeight, 255u) ) // full ROI is white, nothing to erode
            return true;

        const uint32_t dilationX = randomValue<uint32_t>(1, 5);
        const uint32_t dilationY = randomValue<uint32_t>(1, 5);

        BinaryErode( output, roiX, roiY, roiWidth, roiHeight, dilationX, dilationY );

        return verifyImage( output, roiX, roiY, roiWidth, roiHeight, 0u );
    }

    bool form1_BitwiseAnd(BitwiseAndForm1 BitwiseAnd)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        const PenguinV_Image::Image output = BitwiseAnd( input[0], input[1] );

        return equalSize( input[0], output ) && verifyImage( output, intensity[0] & intensity[1] );
    }

    bool form2_BitwiseAnd(BitwiseAndForm2 BitwiseAnd)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

        BitwiseAnd( image[0], image[1], image[2] );

        return verifyImage( image[2], intensity[0] & intensity[1] );
    }

    bool form3_BitwiseAnd(BitwiseAndForm3 BitwiseAnd)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const std::vector < PenguinV_Image::Image > input = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const PenguinV_Image::Image output = BitwiseAnd( input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

        return equalSize( output, roiWidth, roiHeight ) && verifyImage( output, intensity[0] & intensity[1] );
    }

    bool form4_BitwiseAnd(BitwiseAndForm4 BitwiseAnd)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ), uniformImage( intensity[2] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        BitwiseAnd( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], image[2], roiX[2], roiY[2], roiWidth, roiHeight );

        return verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] & intensity[1] );
    }

    bool form1_BitwiseOr(BitwiseOrForm1 BitwiseOr)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        const PenguinV_Image::Image output = BitwiseOr( input[0], input[1] );

        return equalSize( input[0], output ) && verifyImage( output, intensity[0] | intensity[1] );
    }

    bool form2_BitwiseOr(BitwiseOrForm2 BitwiseOr)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

        BitwiseOr( image[0], image[1], image[2] );

        return verifyImage( image[2], intensity[0] | intensity[1] );
    }

    bool form3_BitwiseOr(BitwiseOrForm3 BitwiseOr)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const std::vector < PenguinV_Image::Image > input = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const PenguinV_Image::Image output = BitwiseOr( input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

        return equalSize( output, roiWidth, roiHeight ) && verifyImage( output, intensity[0] | intensity[1] );
    }

    bool form4_BitwiseOr(BitwiseOrForm4 BitwiseOr)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ), uniformImage( intensity[2] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        BitwiseOr( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], image[2], roiX[2], roiY[2], roiWidth, roiHeight );

        return verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] | intensity[1] );
    }

    bool form1_BitwiseXor(BitwiseXorForm1 BitwiseXor)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        const PenguinV_Image::Image output = BitwiseXor( input[0], input[1] );

        return equalSize( input[0], output ) && verifyImage( output, intensity[0] ^ intensity[1] );
    }

    bool form2_BitwiseXor(BitwiseXorForm2 BitwiseXor)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

        BitwiseXor( image[0], image[1], image[2] );

        return verifyImage( image[2], intensity[0] ^ intensity[1] );
    }

    bool form3_BitwiseXor(BitwiseXorForm3 BitwiseXor)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const std::vector < PenguinV_Image::Image > input = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const PenguinV_Image::Image output = BitwiseXor( input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

        return equalSize( output, roiWidth, roiHeight ) && verifyImage( output, intensity[0] ^ intensity[1] );
    }

    bool form4_BitwiseXor(BitwiseXorForm4 BitwiseXor)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ), uniformImage( intensity[2] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        BitwiseXor( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], image[2], roiX[2], roiY[2], roiWidth, roiHeight );

        return verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight, intensity[0] ^ intensity[1] );
    }

    bool form1_ConvertToGrayScale(ConvertToGrayScaleForm1 ConvertToGrayScale)
    {
        const std::vector < uint8_t > intensity = intensityArray( 1 );
        const PenguinV_Image::Image input = uniformRGBImage( intensity[0] );

        const PenguinV_Image::Image output = ConvertToGrayScale( input );

        return verifyImage( output, intensity[0] );
    }

    bool form2_ConvertToGrayScale(ConvertToGrayScaleForm2 ConvertToGrayScale)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const PenguinV_Image::Image input = uniformRGBImage( intensity[0] );
        PenguinV_Image::Image output( input.width(), input.height() );

        output.fill( intensity[1] );

        ConvertToGrayScale( input, output );

        return verifyImage( output, intensity[0] );
    }

    bool form3_ConvertToGrayScale(ConvertToGrayScaleForm3 ConvertToGrayScale)
    {
        const std::vector < uint8_t > intensity = intensityArray( 1 );
        const PenguinV_Image::Image input  = uniformRGBImage( intensity[0] );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const PenguinV_Image::Image output = ConvertToGrayScale( input, roiX, roiY, roiWidth, roiHeight );

        return verifyImage( output, intensity[0] );
    }

    bool form4_ConvertToGrayScale(ConvertToGrayScaleForm4 ConvertToGrayScale)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const PenguinV_Image::Image input  = uniformRGBImage( intensity[0] );
        PenguinV_Image::Image output = uniformImage   ( intensity[1] );

        std::vector < std::pair <uint32_t, uint32_t> > size( 2 );

        size[0] = imageSize( input );
        size[1] = imageSize( output );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( size, roiX, roiY, roiWidth, roiHeight );

        ConvertToGrayScale( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight );

        return verifyImage( output, roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] );
    }

    bool form1_ConvertToRgb(ConvertToRgbForm1 ConvertToRgb)
    {
        const std::vector < uint8_t > intensity = intensityArray( 1 );
        const PenguinV_Image::Image input = uniformImage( intensity[0] );

        const PenguinV_Image::Image output = ConvertToRgb( input );

        return output.colorCount() == PenguinV_Image::RGB && verifyImage( output, intensity[0] );
    }

    bool form2_ConvertToRgb(ConvertToRgbForm2 ConvertToRgb)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const PenguinV_Image::Image input = uniformImage( intensity[0] );
        PenguinV_Image::Image output( input.width(), input.height(), PenguinV_Image::RGB );

        output.fill( intensity[1] );

        ConvertToRgb( input, output );

        return verifyImage( output, intensity[0] );
    }

    bool form3_ConvertToRgb(ConvertToRgbForm3 ConvertToRgb)
    {
        const std::vector < uint8_t > intensity = intensityArray( 1 );
        const PenguinV_Image::Image input = uniformImage( intensity[0] );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const PenguinV_Image::Image output = ConvertToRgb( input, roiX, roiY, roiWidth, roiHeight );

        return output.colorCount() == PenguinV_Image::RGB && verifyImage( output, intensity[0] );
    }

    bool form4_ConvertToRgb(ConvertToRgbForm4 ConvertToRgb)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const PenguinV_Image::Image input  = uniformImage   ( intensity[0] );
        PenguinV_Image::Image output = uniformRGBImage( intensity[1] );

        std::vector < std::pair <uint32_t, uint32_t> > size( 2 );

        size[0] = imageSize( input );
        size[1] = imageSize( output );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( size, roiX, roiY, roiWidth, roiHeight );

        ConvertToRgb( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight );

        return verifyImage( output, roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] );
    }

    bool form1_Copy(CopyForm1 Copy)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        Copy( input[0], input[1] );

        return verifyImage( input[1], intensity[0] );
    }

    bool form2_Copy(CopyForm2 Copy)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const PenguinV_Image::Image output = Copy( input, roiX, roiY, roiWidth, roiHeight );

        return equalSize( output, roiWidth, roiHeight ) && verifyImage( output, intensity );
    }

    bool form3_Copy(CopyForm3 Copy)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        Copy( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

        return verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] );
    }

    bool form1_Fill(FillForm1 Fill)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        PenguinV_Image::Image image = uniformImage( intensity[0] );

        Fill( image, intensity[1] );

        return verifyImage( image, intensity[1] );
    }

    bool form2_Fill(FillForm2 Fill)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        PenguinV_Image::Image image = uniformImage( intensity[0] );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        Fill( image, roiX, roiY, roiWidth, roiHeight, intensity[1] );

        return verifyImage( image, roiX, roiY, roiWidth, roiHeight, intensity[1] );
    }

    bool form1_Flip(FlipForm1 Flip)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
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

        const PenguinV_Image::Image output = Flip( input, horizontalFlip, verticalFlip );

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

        return true;
    }

    bool form2_Flip(FlipForm2 Flip)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const uint8_t intensityFill = intensityValue();
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

        Flip( input[0], input[1], horizontalFlip, verticalFlip );

        if (verticalFlip) {
            if( !verifyImage( input[1], 0, 0, input[1].width(), input[1].height() / 2 + yCorrection, intensity[0] ) )
                return false;
            if((input[0].height() > 1) && !verifyImage( input[1], 0, input[1].height() / 2 + yCorrection, input[1].width(), input[1].height() / 2, intensityFill ) )
                return false;
        }
        else {
            if( !verifyImage( input[1], 0, 0, input[1].width() / 2 + xCorrection, input[1].height(), intensity[0] ) )
                return false;
            if((input[0].width() > 1) && !verifyImage( input[1], input[1].width() / 2 + xCorrection, 0, input[1].width() / 2, input[1].height(),
                                                       horizontalFlip ? intensityFill : intensity[0] ) )
                return false;
        }

        return true;
    }

    bool form3_Flip(FlipForm3 Flip)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
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

        const PenguinV_Image::Image output = Flip( input, roiX, roiY, roiWidth, roiHeight, horizontalFlip, verticalFlip );

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

        return true;
    }

    bool form4_Flip(FlipForm4 Flip)
    {
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

        Flip( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, horizontalFlip, verticalFlip );

        if (verticalFlip) {
            if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight / 2 + yCorrection, intensity[0] ) )
                return false;
            if((roiHeight > 1) && !verifyImage( image[1], roiX[1], roiY[1] + roiHeight / 2 + yCorrection, roiWidth, roiHeight / 2, intensityFill ) )
                return false;
        }
        else {
            if( !verifyImage( image[1], roiX[1], roiY[1], roiWidth / 2 + xCorrection, roiHeight, intensity[0] ) )
                return false;
            if( (roiWidth > 1) && !verifyImage( image[1], roiX[1] + roiWidth / 2 + xCorrection, roiY[1], roiWidth / 2, roiHeight,
                                                horizontalFlip ? intensityFill : intensity[0] ) )
                return false;
        }

        return true;
    }

    bool form1_GammaCorrection(GammaCorrectionForm1 GammaCorrection)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        const double a     = randomValue <uint32_t>( 100 ) / 100.0;
        const double gamma = randomValue <uint32_t>( 300 ) / 100.0;

        const PenguinV_Image::Image output = GammaCorrection( input, a, gamma );

        const double value = a * pow( intensity / 255.0, gamma ) * 255 + 0.5;
        const uint8_t corrected = (value < 256) ? static_cast<uint8_t>(value) : 255;

        return verifyImage( output, corrected );
    }

    bool form2_GammaCorrection(GammaCorrectionForm2 GammaCorrection)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        const double a     = randomValue <uint32_t>( 100 ) / 100.0;
        const double gamma = randomValue <uint32_t>( 300 ) / 100.0;

        GammaCorrection( input[0], input[1], a, gamma );

        const double value = a * pow( intensity[0] / 255.0, gamma ) * 255 + 0.5;
        const uint8_t corrected = (value < 256) ? static_cast<uint8_t>(value) : 255;

        return verifyImage( input[1], corrected );
    }

    bool form3_GammaCorrection(GammaCorrectionForm3 GammaCorrection)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const double a     = randomValue <uint32_t>( 100 ) / 100.0;
        const double gamma = randomValue <uint32_t>( 300 ) / 100.0;

        const PenguinV_Image::Image output = GammaCorrection( input, roiX, roiY, roiWidth, roiHeight, a, gamma );

        const double value = a * pow( intensity / 255.0, gamma ) * 255 + 0.5;
        const uint8_t corrected = (value < 256) ? static_cast<uint8_t>(value) : 255;

        return equalSize( output, roiWidth, roiHeight ) && verifyImage( output, corrected );
    }

    bool form4_GammaCorrection(GammaCorrectionForm4 GammaCorrection)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        const double a     = randomValue <uint32_t>( 100 ) / 100.0;
        const double gamma = randomValue <uint32_t>( 300 ) / 100.0;

        GammaCorrection( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, a, gamma );

        const double value = a * pow( intensity[0] / 255.0, gamma ) * 255 + 0.5;
        const uint8_t corrected = (value < 256) ? static_cast<uint8_t>(value) : 255;

        return verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, corrected );
    }

    bool form1_GetThreshold(GetThresholdForm1 GetThreshold)
    {
        std::vector < uint8_t > intensity = intensityArray( 2 );
        while (intensity[0] == intensity[1])
            intensity = intensityArray( 2 );

        std::vector< uint32_t > histogram( 256u, 0);
        ++histogram[intensity[0]];
        ++histogram[intensity[1]];

        return GetThreshold(histogram) == std::min(intensity[0], intensity[1]);
    }

    bool form1_Histogram(HistogramForm1 Histogram)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image image = uniformImage( intensity );

        const std::vector < uint32_t > histogram = Histogram( image );

        return histogram.size() == 256u && histogram[intensity] == image.width() * image.height() &&
            std::accumulate( histogram.begin(), histogram.end(), 0u ) == image.width() * image.height();
    }

    bool form2_Histogram(HistogramForm2 Histogram)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image image = uniformImage( intensity );

        std::vector < uint32_t > histogram;
        Histogram( image, histogram );

        return histogram.size() == 256u && histogram[intensity] == image.width() * image.height() &&
            std::accumulate( histogram.begin(), histogram.end(), 0u ) == image.width() * image.height();
    }

    bool form3_Histogram(HistogramForm3 Histogram)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const std::vector < uint32_t > histogram = Histogram( input, roiX, roiY, roiWidth, roiHeight );

        return histogram.size() == 256u && histogram[intensity] == roiWidth * roiHeight &&
            std::accumulate( histogram.begin(), histogram.end(), 0u ) == roiWidth * roiHeight;
    }

    bool form4_Histogram(HistogramForm4 Histogram)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        std::vector < uint32_t > histogram;
        Histogram( input, roiX, roiY, roiWidth, roiHeight, histogram );

        return histogram.size() == 256u && histogram[intensity] == roiWidth * roiHeight &&
            std::accumulate( histogram.begin(), histogram.end(), 0u ) == roiWidth * roiHeight;
    }

    bool form5_Histogram(HistogramForm5 Histogram)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image image = uniformImage( intensity );

        if ( image.height() / 2 == 0 )
            return true;

        PenguinV_Image::Image mask = image.generate( image.width(), image.height(), image.colorCount() );
        fillImage( mask, 0, 0, mask.width(), mask.height() / 2, 255 );
        fillImage( mask, 0, mask.height() / 2, mask.width(), mask.height() - mask.height() / 2, 0 );

        const std::vector < uint32_t > histogram = Histogram( image, mask );

        return histogram.size() == 256u && histogram[intensity] == image.width() * (image.height() / 2) &&
            std::accumulate( histogram.begin(), histogram.end(), 0u ) == image.width() * (image.height() / 2);
    }

    bool form6_Histogram(HistogramForm6 Histogram)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image image = uniformImage( intensity );

        if ( image.height() / 2 == 0 )
            return true;

        PenguinV_Image::Image mask = image.generate( image.width(), image.height(), image.colorCount() );
        fillImage( mask, 0, 0, mask.width(), mask.height() / 2, 255 );
        fillImage( mask, 0, mask.height() / 2, mask.width(), mask.height() - mask.height() / 2, 0 );

        std::vector < uint32_t > histogram;
        Histogram( image, mask, histogram );

        return histogram.size() == 256u && histogram[intensity] == image.width() * (image.height() / 2) &&
            std::accumulate( histogram.begin(), histogram.end(), 0u ) == image.width() * (image.height() / 2);
    }

    bool form7_Histogram(HistogramForm7 Histogram)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        if ( roiHeight / 2 == 0 )
            return true;

        fillImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight / 2, 255 );
        fillImage( image[1], roiX[1], roiY[1] + roiHeight / 2, roiWidth, roiHeight - roiHeight / 2, 0 );

        const std::vector < uint32_t > histogram = Histogram( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

        return histogram.size() == 256u && histogram[intensity[0]] == roiWidth * (roiHeight / 2) &&
            std::accumulate( histogram.begin(), histogram.end(), 0u ) == roiWidth * (roiHeight / 2);
    }

    bool form8_Histogram(HistogramForm8 Histogram)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        if ( roiHeight / 2 == 0 )
            return true;

        fillImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight / 2, 255 );
        fillImage( image[1], roiX[1], roiY[1] + roiHeight / 2, roiWidth, roiHeight - roiHeight / 2, 0 );

        std::vector < uint32_t > histogram;
        Histogram( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, histogram );

        return histogram.size() == 256u && histogram[intensity[0]] == roiWidth * (roiHeight / 2) &&
            std::accumulate( histogram.begin(), histogram.end(), 0u ) == roiWidth * (roiHeight / 2);
    }

    bool form1_Invert(InvertForm1 Invert)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        const PenguinV_Image::Image output = Invert( input );

        return verifyImage( output, ~intensity );
    }

    bool form2_Invert(InvertForm2 Invert)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        Invert( input[0], input[1] );

        return verifyImage( input[1], ~intensity[0] );
    }

    bool form3_Invert(InvertForm3 Invert)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const PenguinV_Image::Image output = Invert( input, roiX, roiY, roiWidth, roiHeight );

        return equalSize( output, roiWidth, roiHeight ) && verifyImage( output, ~intensity );
    }

    bool form4_Invert(InvertForm4 Invert)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        Invert( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

        return verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, ~intensity[0] );
    }

    bool form1_IsEqual(IsEqualForm1 IsEqual)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        return (intensity[0] == intensity[1]) == (IsEqual( input[0], input[1] ));
    }

    bool form2_IsEqual(IsEqualForm2 IsEqual)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        return (intensity[0] == intensity[1]) ==
            (IsEqual( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight ));
    }

    bool form1_LookupTable(LookupTableForm1 LookupTable)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const PenguinV_Image::Image input = randomImage( intensity );

        std::vector < uint8_t > lookupTable( 256, 0 );

        lookupTable[intensity[0]] = intensityValue();
        lookupTable[intensity[1]] = intensityValue();

        const PenguinV_Image::Image output = LookupTable( input, lookupTable );

        std::vector < uint8_t > normalized( 2 );

        normalized[0] = lookupTable[intensity[0]];
        normalized[1] = lookupTable[intensity[1]];

        return verifyImage( output, normalized );
    }

    bool form2_LookupTable(LookupTableForm2 LookupTable)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        PenguinV_Image::Image input  = randomImage( intensity );
        PenguinV_Image::Image output( input.width(), input.height() );

        output.fill( intensityValue() );

        std::vector < uint8_t > lookupTable( 256, 0 );

        lookupTable[intensity[0]] = intensityValue();
        lookupTable[intensity[1]] = intensityValue();

        LookupTable( input, output, lookupTable );

        std::vector < uint8_t > normalized( 2 );

        normalized[0] = lookupTable[intensity[0]];
        normalized[1] = lookupTable[intensity[1]];

        return verifyImage( output, normalized );
    }

    bool form3_LookupTable(LookupTableForm3 LookupTable)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        PenguinV_Image::Image input = uniformImage();

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        fillImage( input, roiX, roiY, roiWidth, roiHeight, intensity );

        std::vector < uint8_t > lookupTable( 256, 0 );

        lookupTable[intensity[0]] = intensityValue();
        lookupTable[intensity[1]] = intensityValue();

        const PenguinV_Image::Image output = LookupTable( input, roiX, roiY, roiWidth, roiHeight, lookupTable );

        std::vector < uint8_t > normalized( 2 );

        normalized[0] = lookupTable[intensity[0]];
        normalized[1] = lookupTable[intensity[1]];

        return equalSize( output, roiWidth, roiHeight ) && verifyImage( output, normalized );
    }

    bool form4_LookupTable(LookupTableForm4 LookupTable)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
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

        LookupTable( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight, lookupTable );

        std::vector < uint8_t > normalized( 2 );

        normalized[0] = lookupTable[intensity[0]];
        normalized[1] = lookupTable[intensity[1]];

        return verifyImage( output, roiX[1], roiY[1], roiWidth, roiHeight, normalized );
    }

    bool form1_Maximum(MaximumForm1 Maximum)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        const PenguinV_Image::Image output = Maximum( input[0], input[1] );

        return equalSize( input[0], output ) &&
            verifyImage( output, intensity[0] > intensity[1] ? intensity[0] : intensity[1] );
    }

    bool form2_Maximum(MaximumForm2 Maximum)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

        Maximum( image[0], image[1], image[2] );

        return verifyImage( image[2], intensity[0] > intensity[1] ? intensity[0] : intensity[1] );
    }

    bool form3_Maximum(MaximumForm3 Maximum)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > input;

        std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t value )
        { input.push_back( uniformImage( value ) ); } );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const PenguinV_Image::Image output = Maximum( input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

        return equalSize( output, roiWidth, roiHeight ) &&
            verifyImage( output, intensity[0] > intensity[1] ? intensity[0] : intensity[1] );
    }

    bool form4_Maximum(MaximumForm4 Maximum)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image;

        std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t value )
        { image.push_back( uniformImage( value ) ); } );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        Maximum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], image[2], roiX[2], roiY[2], roiWidth, roiHeight );

        return verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
                          intensity[0] > intensity[1] ? intensity[0] : intensity[1] );
    }

    bool form1_Minimum(MinimumForm1 Minimum)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        const PenguinV_Image::Image output = Minimum( input[0], input[1] );

        return equalSize( input[0], output ) &&
            verifyImage( output, intensity[0] < intensity[1] ? intensity[0] : intensity[1] );
    }

    bool form2_Minimum(MinimumForm2 Minimum)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

        Minimum( image[0], image[1], image[2] );

        return verifyImage( image[2], intensity[0] < intensity[1] ? intensity[0] : intensity[1] );
    }

    bool form3_Minimum(MinimumForm3 Minimum)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > input;

        std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t value )
        { input.push_back( uniformImage( value ) ); } );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const PenguinV_Image::Image output = Minimum( input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

        return equalSize( output, roiWidth, roiHeight ) &&
            verifyImage( output, intensity[0] < intensity[1] ? intensity[0] : intensity[1] );
    }

    bool form4_Minimum(MinimumForm4 Minimum)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image;

        std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t value )
        { image.push_back( uniformImage( value ) ); } );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        Minimum( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], image[2], roiX[2], roiY[2], roiWidth, roiHeight );

        return verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
                          intensity[0] < intensity[1] ? intensity[0] : intensity[1] );
    }

    bool form1_Normalize(NormalizeForm1 Normalize)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const PenguinV_Image::Image input = randomImage( intensity );

        const PenguinV_Image::Image output = Normalize( input );

        std::vector < uint8_t > normalized( 2 );

        if( intensity[0] == intensity[1] || (input.width() == 1 && input.height() == 1) ) {
            normalized[0] = normalized[1] = intensity[0];
        }
        else {
            normalized[0] = 0;
            normalized[1] = 255;
        }

        return verifyImage( output, normalized );
    }

    bool form2_Normalize(NormalizeForm2 Normalize)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const PenguinV_Image::Image input  = randomImage( intensity );
        PenguinV_Image::Image output( input.width(), input.height() );

        output.fill( intensityValue() );

        Normalize( input, output );

        std::vector < uint8_t > normalized( 2 );

        if( intensity[0] == intensity[1] || (input.width() == 1 && input.height() == 1) ) {
            normalized[0] = normalized[1] = intensity[0];
        }
        else {
            normalized[0] = 0;
            normalized[1] = 255;
        }

        return verifyImage( output, normalized );
    }

    bool form3_Normalize(NormalizeForm3 Normalize)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        PenguinV_Image::Image input = uniformImage();

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        fillImage( input, roiX, roiY, roiWidth, roiHeight, intensity );

        const PenguinV_Image::Image output = Normalize( input, roiX, roiY, roiWidth, roiHeight );

        std::vector < uint8_t > normalized( 2 );

        if( intensity[0] == intensity[1] || (roiWidth == 1 && roiHeight == 1) ) {
            normalized[0] = normalized[1] = intensity[0];
        }
        else {
            normalized[0] = 0;
            normalized[1] = 255;
        }

        return equalSize( output, roiWidth, roiHeight ) && verifyImage( output, normalized );
    }

    bool form4_Normalize(NormalizeForm4 Normalize)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        PenguinV_Image::Image input  = uniformImage();
        PenguinV_Image::Image output = uniformImage();

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;

        std::vector < std::pair < uint32_t, uint32_t > > size( 2 );

        size[0] = imageSize( input );
        size[1] = imageSize( output );

        generateRoi( size, roiX, roiY, roiWidth, roiHeight );

        fillImage( input, roiX[0], roiY[0], roiWidth, roiHeight, intensity );

        Normalize( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight );

        std::vector < uint8_t > normalized( 2 );

        if( intensity[0] == intensity[1] || (roiWidth == 1 && roiHeight == 1) ) {
            normalized[0] = normalized[1] = intensity[0];
        }
        else {
            normalized[0] = 0;
            normalized[1] = 255;
        }

        return verifyImage( output, roiX[1], roiY[1], roiWidth, roiHeight, normalized );
    }

    bool form1_ProjectionProfile(ProjectionProfileForm1 ProjectionProfile)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image image = uniformImage( intensity );

        const bool horizontal = (randomValue<int>(2) == 0);

        std::vector < uint32_t > projection = ProjectionProfile( image, horizontal );

        const uint32_t value = (horizontal ? image.height() : image.width()) * intensity;

        return projection.size() == (horizontal ? image.width() : image.height()) &&
            std::all_of( projection.begin(), projection.end(), [&value]( uint32_t v ) { return value == v; } );
    }

    bool form2_ProjectionProfile(ProjectionProfileForm2 ProjectionProfile)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image image = uniformImage( intensity );

        const bool horizontal = (randomValue<int>(2) == 0);

        std::vector < uint32_t > projection;
        ProjectionProfile( image, horizontal, projection );

        const uint32_t value = (horizontal ? image.height() : image.width()) * intensity;

        return projection.size() == (horizontal ? image.width() : image.height()) &&
            std::all_of( projection.begin(), projection.end(), [&value]( uint32_t v ) { return value == v; } );
    }

    bool form3_ProjectionProfile(ProjectionProfileForm3 ProjectionProfile)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image image = uniformImage( intensity );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        const bool horizontal = (randomValue<int>(2) == 0);

        std::vector < uint32_t > projection = ProjectionProfile( image, roiX, roiY, roiWidth, roiHeight, horizontal );

        uint32_t value = (horizontal ? roiHeight : roiWidth) * intensity;

        return projection.size() == (horizontal ? roiWidth : roiHeight) &&
            std::all_of( projection.begin(), projection.end(), [&value]( uint32_t v ) { return value == v; } );
    }

    bool form4_ProjectionProfile(ProjectionProfileForm4 ProjectionProfile)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image image = uniformImage( intensity );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        const bool horizontal = (randomValue<int>(2) == 0);

        std::vector < uint32_t > projection;
        ProjectionProfile( image, roiX, roiY, roiWidth, roiHeight, horizontal, projection );

        const uint32_t value = (horizontal ? roiHeight : roiWidth) * intensity;

        return projection.size() == (horizontal ? roiWidth : roiHeight) &&
            std::all_of( projection.begin(), projection.end(), [&value]( uint32_t v ) { return value == v; } );
    }

    bool form1_ReplaceChannel(ReplaceChannelForm1 ReplaceChannel)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const PenguinV_Image::Image input = uniformImage( intensity[0] );
        PenguinV_Image::Image output( input.width(), input.height(), PenguinV_Image::RGB );

        output.fill( intensity[1] );

        const uint8_t channelId = randomValue<uint8_t>( 3 );

        ReplaceChannel( input, output, channelId );

        std::vector<uint8_t> verificationArray( 3, intensity[1] );
        verificationArray[channelId] = intensity[0];

        return verifyImage( output, verificationArray, false );
    }

    bool form2_ReplaceChannel(ReplaceChannelForm2 ReplaceChannel)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const PenguinV_Image::Image input  = uniformImage   ( intensity[0] );
        PenguinV_Image::Image output = uniformRGBImage( intensity[1] );

        std::vector < std::pair <uint32_t, uint32_t> > size( 2 );

        size[0] = imageSize( input );
        size[1] = imageSize( output );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( size, roiX, roiY, roiWidth, roiHeight );

        const uint8_t channelId = randomValue<uint8_t>( 3 );

        ReplaceChannel( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight, channelId );

        std::vector<uint8_t> verificationArray( 3, intensity[1] );
        verificationArray[channelId] = intensity[0];

        return verifyImage( output, roiX[1], roiY[1], roiWidth, roiHeight, verificationArray, false );
    }

    bool form1_Resize(ResizeForm1 Resize)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        uint32_t outputWidth  = randomValue<uint32_t>( 1, 2048 );
        uint32_t outputHeight = randomValue<uint32_t>( 1, 2048 );

        const PenguinV_Image::Image output = Resize( input, outputWidth, outputHeight );

        return equalSize( output, outputWidth, outputHeight ) && verifyImage( output, intensity );
    }

    bool form2_Resize(ResizeForm2 Resize)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const PenguinV_Image::Image input  = uniformImage( intensity[0] );
        PenguinV_Image::Image output = uniformImage( intensity[1] );

        Resize( input, output );

        return verifyImage( output, intensity[0] );
    }

    bool form3_Resize(ResizeForm3 Resize)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        uint32_t outputWidth  = randomValue<uint32_t>( 1, 2048 );
        uint32_t outputHeight = randomValue<uint32_t>( 1, 2048 );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const PenguinV_Image::Image output = Resize( input, roiX, roiY, roiWidth, roiHeight, outputWidth, outputHeight );

        return equalSize( output, outputWidth, outputHeight ) && verifyImage( output, intensity );
    }

    bool form4_Resize(ResizeForm4 Resize)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const PenguinV_Image::Image input  = uniformImage( intensity[0] );
        PenguinV_Image::Image output = uniformImage( intensity[1] );

        std::vector < uint32_t > roiX( 2 ), roiY( 2 ), roiWidth( 2 ), roiHeight( 2 );

        generateRoi( input, roiX[0], roiY[0], roiWidth[0], roiHeight[0] );
        generateRoi( output, roiX[1], roiY[1], roiWidth[1], roiHeight[1] );

        Resize( input, roiX[0], roiY[0], roiWidth[0], roiHeight[0], output, roiX[1], roiY[1], roiWidth[1], roiHeight[1] );

        return verifyImage( output, roiX[1], roiY[1], roiWidth[1], roiHeight[1], intensity[0] );
    }

    bool form1_RgbToBgr(RgbToBgrForm1 RgbToBgr)
    {
        PenguinV_Image::Image input = uniformRGBImage( intensityValue() );
        std::vector< uint8_t > intensity = intensityArray( 3 );

        fillImage( input, 0, 0, input.width(), input.height(), intensity );

        const PenguinV_Image::Image output = RgbToBgr( input );

        std::swap( intensity[0], intensity[2] );

        return verifyImage( output, intensity, false );
    }

    bool form2_RgbToBgr(RgbToBgrForm2 RgbToBgr)
    {
        PenguinV_Image::Image input = uniformRGBImage( intensityValue() );
        std::vector< uint8_t > intensity = intensityArray( 3 );

        fillImage( input, 0, 0, input.width(), input.height(), intensity );

        PenguinV_Image::Image output = input.generate( input.width(), input.height(), input.colorCount(), input.alignment() );
        fillImage( output, 0, 0, output.width(), output.height(), intensityValue() );

        RgbToBgr( input, output );

        std::swap( intensity[0], intensity[2] );

        return verifyImage( output, intensity, false );
    }

    bool form3_RgbToBgr(RgbToBgrForm3 RgbToBgr)
    {
        PenguinV_Image::Image input = uniformRGBImage( intensityValue() );
        std::vector< uint8_t > intensity = intensityArray( 3 );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        fillImage( input, roiX, roiY, roiWidth, roiHeight, intensity );

        const PenguinV_Image::Image output = RgbToBgr( input, roiX, roiY, roiWidth, roiHeight );

        std::swap( intensity[0], intensity[2] );

        return equalSize( output, roiWidth, roiHeight ) && verifyImage( output, intensity, false );
    }

    bool form4_RgbToBgr(RgbToBgrForm4 RgbToBgr)
    {
        std::vector < PenguinV_Image::Image > image = { uniformRGBImage( intensityValue() ), uniformRGBImage( intensityValue() ) };
        std::vector< uint8_t > intensity = intensityArray( 3 );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        fillImage( image[0], roiX[0], roiY[0], roiWidth, roiHeight, intensity );

        RgbToBgr( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

        std::swap( intensity[0], intensity[2] );

        return verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, intensity, false );
    }

    bool form1_SetPixel(SetPixelForm1 SetPixel)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        PenguinV_Image::Image image  = uniformImage( intensity[0] );
        const uint32_t x = randomValue<uint32_t>( 0, image.width() );
        const uint32_t y = randomValue<uint32_t>( 0, image.height() );

        SetPixel( image, x, y, intensity[1] );

        return verifyImage( image, x, y, 1, 1, intensity[1] );
    }

    bool form2_SetPixel(SetPixelForm2 SetPixel)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        PenguinV_Image::Image image  = uniformImage( intensity[0] );
        std::vector< uint32_t > X( randomValue<uint32_t>( 1, 100 ) );
        std::vector< uint32_t > Y( X.size() );

        for (size_t j = 0; j < X.size(); j++) {
            X[j] = randomValue<uint32_t>( 0, image.width() );
            Y[j] = randomValue<uint32_t>( 0, image.height() );
        }

        SetPixel( image, X, Y, intensity[1] );

        for (size_t j = 0; j < X.size(); j++) {
            if( !verifyImage( image, X[j], Y[j], 1, 1, intensity[1] ) )
                return false;
        }

        return true;
    }

    bool form1_Shift(ShiftForm1 Shift)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        const double shiftX = randomFloatValue<double>( -static_cast<double>( input.width()  / 4 ), input.width() / 4 , 1 ) + randomFloatValue<double>( 0, 1, 0.01 );
        const double shiftY = randomFloatValue<double>( -static_cast<double>( input.height() / 4 ), input.height() / 4, 1 ) + randomFloatValue<double>( 0, 1, 0.01 );

        const PenguinV_Image::Image output = Shift( input, shiftX, shiftY );

        return verifyImage( output, intensity );
    }

    bool form2_Shift(ShiftForm2 Shift)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        const double shiftX = randomFloatValue<double>( -static_cast<double>( input[0].width()  / 4 ), input[0].width() / 4 , 1 ) + randomFloatValue<double>( 0, 1, 0.01 );
        const double shiftY = randomFloatValue<double>( -static_cast<double>( input[0].height() / 4 ), input[0].height() / 4, 1 ) + randomFloatValue<double>( 0, 1, 0.01 );

        Shift( input[0], input[1], shiftX, shiftY );

        return verifyImage( input[1], intensity[0] );
    }

    bool form3_Shift(ShiftForm3 Shift)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const double maxShiftX = roiX < (input.width()  - roiWidth  - roiX) ? roiX : (input.width()  - roiWidth  - roiX);
        const double maxShiftY = roiY < (input.height() - roiHeight - roiY) ? roiY : (input.height() - roiHeight - roiY);
        const double shiftX = randomFloatValue<double>( -maxShiftX / 4, maxShiftX / 4 , 1 ) + randomFloatValue<double>( 0, 1, 0.01 );
        const double shiftY = randomFloatValue<double>( -maxShiftY / 4, maxShiftY / 4 , 1 ) + randomFloatValue<double>( 0, 1, 0.01 );

        const PenguinV_Image::Image output = Shift( input, roiX, roiY, roiWidth, roiHeight, shiftX, shiftY );

        return equalSize( output, roiWidth, roiHeight ) && verifyImage( output, intensity );
    }

    bool form4_Shift(ShiftForm4 Shift)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        const double maxShiftX = roiX[0] < (image[0].width()  - roiWidth  - roiX[0]) ? roiX[0] : (image[0].width()  - roiWidth  - roiX[0]);
        const double maxShiftY = roiY[0] < (image[0].height() - roiHeight - roiY[0]) ? roiY[0] : (image[0].height() - roiHeight - roiY[0]);
        const double shiftX = randomFloatValue<double>( -maxShiftX / 4, maxShiftX / 4 , 1 ) + randomFloatValue<double>( 0, 1, 0.01 );
        const double shiftY = randomFloatValue<double>( -maxShiftY / 4 , maxShiftY / 4 , 1 ) + randomFloatValue<double>( 0, 1, 0.01 );

        Shift( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, shiftX, shiftY );

        return verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] );
    }

    bool form1_Subtract(SubtractForm1 Subtract)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        const PenguinV_Image::Image output = Subtract( input[0], input[1] );

        return equalSize( input[0], output ) &&
            verifyImage( output, static_cast<uint8_t>( intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0u ) );
    }

    bool form2_Subtract(SubtractForm2 Subtract)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image = uniformImages( intensity );

        Subtract( image[0], image[1], image[2] );

        return verifyImage( image[2], static_cast<uint8_t>( intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0u ) );
    }

    bool form3_Subtract(SubtractForm3 Subtract)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > input;

        std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t value )
        { input.push_back( uniformImage( value ) ); } );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const PenguinV_Image::Image output = Subtract( input[0], roiX[0], roiY[0], input[1], roiX[1], roiY[1], roiWidth, roiHeight );

        return equalSize( output, roiWidth, roiHeight ) &&
            verifyImage( output, static_cast<uint8_t>( intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0u ) );
    }

    bool form4_Subtract(SubtractForm4 Subtract)
    {
        const std::vector < uint8_t > intensity = intensityArray( 3 );
        std::vector < PenguinV_Image::Image > image;

        std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t value )
        { image.push_back( uniformImage( value ) ); } );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        Subtract( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], image[2], roiX[2], roiY[2], roiWidth, roiHeight );

        return verifyImage( image[2], roiX[2], roiY[2], roiWidth, roiHeight,
                            static_cast<uint8_t>( intensity[0] > intensity[1] ? intensity[0] - intensity[1] : 0u ) );
    }

    bool form1_Sum(SumForm1 Sum)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        return Sum( input ) == intensity * input.width() * input.height();
    }

    bool form2_Sum(SumForm2 Sum)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        return Sum( input, roiX, roiY, roiWidth, roiHeight ) == intensity * roiWidth * roiHeight;
    }

    bool form1_Threshold(ThresholdForm1 Threshold)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        const uint8_t threshold = randomValue <uint8_t>( 255 );

        const PenguinV_Image::Image output = Threshold( input, threshold );

        return verifyImage( output, intensity < threshold ? 0 : 255 );
    }

    bool form2_Threshold(ThresholdForm2 Threshold)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        const uint8_t threshold = randomValue <uint8_t>( 255 );

        Threshold( input[0], input[1], threshold );

        return verifyImage( input[1], intensity[0] < threshold ? 0 : 255 );
    }

    bool form3_Threshold(ThresholdForm3 Threshold)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const uint8_t threshold = randomValue <uint8_t>( 255 );

        const PenguinV_Image::Image output = Threshold( input, roiX, roiY, roiWidth, roiHeight, threshold );

        return equalSize( output, roiWidth, roiHeight ) && verifyImage( output, intensity < threshold ? 0 : 255 );
    }

    bool form4_Threshold(ThresholdForm4 Threshold)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > image;

        std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t value )
        { image.push_back( uniformImage( value ) ); } );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        const uint8_t threshold = randomValue <uint8_t>( 255 );

        Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, threshold );

        return verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, intensity[0] < threshold ? 0 : 255 );
    }

    bool form5_Threshold(ThresholdDoubleForm1 Threshold)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        const uint8_t minThreshold = randomValue <uint8_t>( 255 );
        const uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

        const PenguinV_Image::Image output = Threshold( input, minThreshold, maxThreshold );

        return verifyImage( output, intensity < minThreshold || intensity > maxThreshold ? 0 : 255 );
    }

    bool form6_Threshold(ThresholdDoubleForm2 Threshold)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > input = uniformImages( intensity );

        const uint8_t minThreshold = randomValue <uint8_t>( 255 );
        const uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

        Threshold( input[0], input[1], minThreshold, maxThreshold );

        return verifyImage( input[1], intensity[0] < minThreshold || intensity[0] > maxThreshold ? 0 : 255 );
    }

    bool form7_Threshold(ThresholdDoubleForm3 Threshold)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        const uint8_t minThreshold = randomValue <uint8_t>( 255 );
        const uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

        const PenguinV_Image::Image output = Threshold( input, roiX, roiY, roiWidth, roiHeight, minThreshold, maxThreshold );

        return equalSize( output, roiWidth, roiHeight ) &&
            verifyImage( output, intensity < minThreshold || intensity > maxThreshold ? 0 : 255 );
    }

    bool form8_Threshold(ThresholdDoubleForm4 Threshold)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > image;

        std::for_each( intensity.begin(), intensity.end(), [&]( uint8_t value )
        { image.push_back( uniformImage( value ) ); } );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );

        const uint8_t minThreshold = randomValue <uint8_t>( 255 );
        const uint8_t maxThreshold = randomValue <uint8_t>( minThreshold, 255 );

        Threshold( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight, minThreshold, maxThreshold );

        return verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight,
                          intensity[0] < minThreshold || intensity[0] > maxThreshold ? 0 : 255 );
    }

    bool form1_Transpose(TransposeForm1 Transpose)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        const PenguinV_Image::Image output = Transpose( input );

        return equalSize( output, input.height(), input.width() ) && verifyImage( output, intensity );
    }

    bool form2_Transpose(TransposeForm2 Transpose)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const PenguinV_Image::Image input  = uniformImage( intensity[0] );
        PenguinV_Image::Image output( input.height(), input.width() );

        output.fill( intensity[1] );

        Transpose( input, output );

        return verifyImage( output, intensity[0] );
    }

    bool form3_Transpose(TransposeForm3 Transpose)
    {
        const uint8_t intensity = intensityValue();
        const PenguinV_Image::Image input = uniformImage( intensity );

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );

        PenguinV_Image::Image output = Transpose( input, roiX, roiY, roiWidth, roiHeight );

        return equalSize( output, roiHeight, roiWidth )&& verifyImage( output, intensity );
    }

    bool form4_Transpose(TransposeForm4 Transpose)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        const PenguinV_Image::Image input  = uniformImage( intensity[0] );
        PenguinV_Image::Image output = uniformImage( intensity[1] );

        std::vector < std::pair <uint32_t, uint32_t> > size( 2 );

        size[0] = imageSize( input );
        size[1] = std::pair <uint32_t, uint32_t>( output.height(), output.width() );

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( size, roiX, roiY, roiWidth, roiHeight );

        generateOffset( output, roiX[1], roiY[1], roiHeight, roiWidth );

        Transpose( input, roiX[0], roiY[0], output, roiX[1], roiY[1], roiWidth, roiHeight );

        return verifyImage( output, roiX[1], roiY[1], roiHeight, roiWidth, intensity[0] );
    }

    // Filters
    bool form1_Prewitt(PrewittForm1 Prewitt)
    {
        const PenguinV_Image::Image input = uniformImage();
        if ( (input.width() < 3) || (input.height() < 3) )
            return true;

        const PenguinV_Image::Image output = Prewitt( input );

        return equalSize( input, output ) && verifyImage( output, 0u );
    }

    bool form2_Prewitt(PrewittForm2 Prewitt)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > image = uniformImages( intensity );
        if ( (image[0].width() < 3) || (image[0].height() < 3) )
            return true;

        Prewitt( image[0], image[1] );

        return verifyImage( image[1], 0u );
    }

    bool form3_Prewitt(PrewittForm3 Prewitt)
    {
        const PenguinV_Image::Image input = uniformImage();

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );
        if ( (roiWidth < 3) || (roiHeight < 3) )
            return true;

        const PenguinV_Image::Image output = Prewitt( input, roiX, roiY, roiWidth, roiHeight );

        return equalSize( output, roiWidth, roiHeight ) && verifyImage( output, 0u );
    }

    bool form4_Prewitt(PrewittForm4 Prewitt)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );
        if ( (roiWidth < 3) || (roiHeight < 3) )
            return true;

        Prewitt( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

        return verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, 0u );
    }

    bool form1_Sobel(SobelForm1 Sobel)
    {
        const PenguinV_Image::Image input = uniformImage();
        if ( (input.width() < 3) || (input.height() < 3) )
            return true;

        const PenguinV_Image::Image output = Sobel( input );

        return equalSize( input, output ) && verifyImage( output, 0u );
    }

    bool form2_Sobel(SobelForm2 Sobel)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > image = uniformImages( intensity );
        if ( (image[0].width() < 3) || (image[0].height() < 3) )
            return true;

        Sobel( image[0], image[1] );

        return verifyImage( image[1], 0u );
    }

    bool form3_Sobel(SobelForm3 Sobel)
    {
        const PenguinV_Image::Image input = uniformImage();

        uint32_t roiX, roiY, roiWidth, roiHeight;
        generateRoi( input, roiX, roiY, roiWidth, roiHeight );
        if ( (roiWidth < 3) || (roiHeight < 3) )
            return true;

        const PenguinV_Image::Image output = Sobel( input, roiX, roiY, roiWidth, roiHeight );

        return equalSize( output, roiWidth, roiHeight ) && verifyImage( output, 0u );
    }

    bool form4_Sobel(SobelForm4 Sobel)
    {
        const std::vector < uint8_t > intensity = intensityArray( 2 );
        std::vector < PenguinV_Image::Image > image = { uniformImage( intensity[0] ), uniformImage( intensity[1] ) };

        std::vector < uint32_t > roiX, roiY;
        uint32_t roiWidth, roiHeight;
        generateRoi( image, roiX, roiY, roiWidth, roiHeight );
        if ( (roiWidth < 3) || (roiHeight < 3) )
            return true;

        Sobel( image[0], roiX[0], roiY[0], image[1], roiX[1], roiY[1], roiWidth, roiHeight );

        return verifyImage( image[1], roiX[1], roiY[1], roiWidth, roiHeight, 0u );
    }
}

#define FUNCTION_REGISTRATION( function, functionWrapper, counter )                                                             \
struct Register_##functionWrapper                                                                                               \
{                                                                                                                               \
    explicit Register_##functionWrapper( bool makeRegistration )                                                                \
    {                                                                                                                           \
        if( makeRegistration )                                                                                                  \
            FunctionRegistrator::instance().add( functionWrapper, namespaceName + std::string("::") + std::string(#function) +  \
                                                           std::string(" (form ") + std::string(#counter) + std::string(")") ); \
    }                                                                                                                           \
};                                                                                                                              \
const Register_##functionWrapper registrator_##functionWrapper( isSupported );

#define DECLARE_FUNCTION_BODY( functionCall )                       \
    for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {         \
        PrepareFunction(namespaceName);                             \
        const bool returnValue = functionCall;                      \
        CleanupFunction(namespaceName);                             \
        if ( !returnValue )                                         \
            return false;                                           \
    }                                                               \
    return true;

#define SET_FUNCTION_1_FORMS( function )                                                                 \
    bool type1_##function() { DECLARE_FUNCTION_BODY( Function_Template::form1_##function( function ) ) } \
    FUNCTION_REGISTRATION( function, type1_##function, 1 )

#define SET_FUNCTION_2_FORMS( function )                                                                 \
    SET_FUNCTION_1_FORMS( function )                                                                     \
    bool type2_##function() { DECLARE_FUNCTION_BODY( Function_Template::form2_##function( function ) ) } \
    FUNCTION_REGISTRATION( function, type2_##function, 2 )

#define SET_FUNCTION_3_FORMS( function )                                                                 \
    SET_FUNCTION_2_FORMS( function )                                                                     \
    bool type3_##function() { DECLARE_FUNCTION_BODY( Function_Template::form3_##function( function ) ) } \
    FUNCTION_REGISTRATION( function, type3_##function, 3 )

#define SET_FUNCTION_4_FORMS( function )                                                                 \
    SET_FUNCTION_3_FORMS( function )                                                                     \
    bool type4_##function() { DECLARE_FUNCTION_BODY( Function_Template::form4_##function( function ) ) } \
    FUNCTION_REGISTRATION( function, type4_##function, 4 )

#define SET_FUNCTION_8_FORMS( function )                                                                 \
    SET_FUNCTION_4_FORMS( function )                                                                     \
    bool type5_##function() { DECLARE_FUNCTION_BODY( Function_Template::form5_##function( function ) ) } \
    bool type6_##function() { DECLARE_FUNCTION_BODY( Function_Template::form6_##function( function ) ) } \
    bool type7_##function() { DECLARE_FUNCTION_BODY( Function_Template::form7_##function( function ) ) } \
    bool type8_##function() { DECLARE_FUNCTION_BODY( Function_Template::form8_##function( function ) ) } \
    FUNCTION_REGISTRATION( function, type5_##function, 5 )                                               \
    FUNCTION_REGISTRATION( function, type6_##function, 6 )                                               \
    FUNCTION_REGISTRATION( function, type7_##function, 7 )                                               \
    FUNCTION_REGISTRATION( function, type8_##function, 8 )

namespace image_function
{
    using namespace Image_Function;

    const bool isSupported = true;
    const std::string namespaceName = "image_function";

    SET_FUNCTION_4_FORMS( AbsoluteDifference )
    SET_FUNCTION_2_FORMS( Accumulate )
    SET_FUNCTION_2_FORMS( BinaryDilate )
    SET_FUNCTION_2_FORMS( BinaryErode )
    SET_FUNCTION_4_FORMS( BitwiseAnd )
    SET_FUNCTION_4_FORMS( BitwiseOr )
    SET_FUNCTION_4_FORMS( BitwiseXor )
    SET_FUNCTION_4_FORMS( ConvertToGrayScale )
    SET_FUNCTION_4_FORMS( ConvertToRgb )
    SET_FUNCTION_3_FORMS( Copy )
    SET_FUNCTION_2_FORMS( Fill )
    SET_FUNCTION_4_FORMS( Flip )
    SET_FUNCTION_4_FORMS( GammaCorrection )
    SET_FUNCTION_1_FORMS( GetThreshold )
    SET_FUNCTION_8_FORMS( Histogram )
    SET_FUNCTION_4_FORMS( Invert )
    SET_FUNCTION_2_FORMS( IsEqual )
    SET_FUNCTION_4_FORMS( LookupTable )
    SET_FUNCTION_4_FORMS( Maximum )
    SET_FUNCTION_4_FORMS( Minimum )
    SET_FUNCTION_4_FORMS( Normalize )
    SET_FUNCTION_4_FORMS( ProjectionProfile )
    SET_FUNCTION_2_FORMS( ReplaceChannel )
    SET_FUNCTION_4_FORMS( Resize )
    SET_FUNCTION_4_FORMS( RgbToBgr )
    SET_FUNCTION_2_FORMS( SetPixel )
    SET_FUNCTION_4_FORMS( Subtract )
    SET_FUNCTION_2_FORMS( Sum )
    SET_FUNCTION_8_FORMS( Threshold )
    SET_FUNCTION_4_FORMS( Transpose )

    SET_FUNCTION_4_FORMS( Prewitt )
    SET_FUNCTION_4_FORMS( Sobel )
}

namespace function_pool
{
    using namespace Function_Pool;

    const bool isSupported = true;
    const std::string namespaceName = "function_pool";

    SET_FUNCTION_4_FORMS( AbsoluteDifference )
    SET_FUNCTION_4_FORMS( BitwiseAnd )
    SET_FUNCTION_4_FORMS( BitwiseOr )
    SET_FUNCTION_4_FORMS( BitwiseXor )
    SET_FUNCTION_4_FORMS( ConvertToGrayScale )
    SET_FUNCTION_4_FORMS( ConvertToRgb )
    SET_FUNCTION_4_FORMS( GammaCorrection )
    SET_FUNCTION_4_FORMS( Histogram )
    SET_FUNCTION_4_FORMS( Invert )
    SET_FUNCTION_2_FORMS( IsEqual )
    SET_FUNCTION_4_FORMS( LookupTable )
    SET_FUNCTION_4_FORMS( Maximum )
    SET_FUNCTION_4_FORMS( Minimum )
    SET_FUNCTION_4_FORMS( Normalize )
    SET_FUNCTION_4_FORMS( ProjectionProfile )
    SET_FUNCTION_4_FORMS( Resize )
    SET_FUNCTION_4_FORMS( Subtract )
    SET_FUNCTION_2_FORMS( Sum )
    SET_FUNCTION_8_FORMS( Threshold )
}

#ifdef PENGUINV_AVX_SET
namespace avx
{
    using namespace Image_Function_Simd;

    const bool isSupported = isAvxAvailable;
    const std::string namespaceName = "image_function_avx";

    SET_FUNCTION_4_FORMS( AbsoluteDifference )
    SET_FUNCTION_2_FORMS( Accumulate )
    SET_FUNCTION_4_FORMS( BitwiseAnd )
    SET_FUNCTION_4_FORMS( BitwiseOr )
    SET_FUNCTION_4_FORMS( BitwiseXor )
    SET_FUNCTION_4_FORMS( Invert )
    SET_FUNCTION_4_FORMS( Maximum )
    SET_FUNCTION_4_FORMS( Minimum )
    SET_FUNCTION_4_FORMS( ProjectionProfile )
    SET_FUNCTION_4_FORMS( RgbToBgr )
    SET_FUNCTION_4_FORMS( Subtract )
    SET_FUNCTION_2_FORMS( Sum )
    SET_FUNCTION_8_FORMS( Threshold )
}
#endif

#ifdef PENGUINV_NEON_SET
namespace neon
{
    using namespace Image_Function_Simd;

    const bool isSupported = isNeonAvailable;
    const std::string namespaceName = "image_function_neon";

    SET_FUNCTION_4_FORMS( AbsoluteDifference )
    SET_FUNCTION_2_FORMS( Accumulate )
    SET_FUNCTION_4_FORMS( BitwiseAnd )
    SET_FUNCTION_4_FORMS( BitwiseOr )
    SET_FUNCTION_4_FORMS( BitwiseXor )
    SET_FUNCTION_4_FORMS( ConvertToRgb )
    SET_FUNCTION_4_FORMS( Flip )
    SET_FUNCTION_4_FORMS( Invert )
    SET_FUNCTION_4_FORMS( Maximum )
    SET_FUNCTION_4_FORMS( Minimum )
    SET_FUNCTION_4_FORMS( ProjectionProfile )
    SET_FUNCTION_4_FORMS( RgbToBgr )
    SET_FUNCTION_4_FORMS( Subtract )
    SET_FUNCTION_2_FORMS( Sum )
    SET_FUNCTION_8_FORMS( Threshold )
}
#endif

#ifdef PENGUINV_SSE_SET
namespace sse
{
    using namespace Image_Function_Simd;

    const bool isSupported = isSseAvailable;
    const std::string namespaceName = "image_function_sse";

    SET_FUNCTION_4_FORMS( AbsoluteDifference )
    SET_FUNCTION_2_FORMS( Accumulate )
    SET_FUNCTION_4_FORMS( BitwiseAnd )
    SET_FUNCTION_4_FORMS( BitwiseOr )
    SET_FUNCTION_4_FORMS( BitwiseXor )
    SET_FUNCTION_4_FORMS( ConvertToRgb )
    SET_FUNCTION_4_FORMS( ConvertTo16Bit )
    SET_FUNCTION_4_FORMS( Flip )
    SET_FUNCTION_4_FORMS( Invert )
    SET_FUNCTION_4_FORMS( Maximum )
    SET_FUNCTION_4_FORMS( Minimum )
    SET_FUNCTION_4_FORMS( ProjectionProfile )
    SET_FUNCTION_4_FORMS( RgbToBgr )
    SET_FUNCTION_4_FORMS( Subtract )
    SET_FUNCTION_2_FORMS( Sum )
    SET_FUNCTION_8_FORMS( Threshold )
}
#endif

void addTests_Image_Function( UnitTestFramework & framework )
{
    FunctionRegistrator::instance().set( framework );
}
