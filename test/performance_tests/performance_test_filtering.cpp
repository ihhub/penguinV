#include "../../src/filtering.h"
#include "performance_test_filtering.h"
#include "performance_test_helper.h"

namespace
{
    std::pair < double, double > MedianFilter3x3( uint32_t size )
    {
        Performance_Test::TimerContainer timer;

        PenguinV_Image::Image input = Performance_Test::uniformImage( size, size, Performance_Test::randomValue<uint8_t>( 1, 256 ) );
        PenguinV_Image::Image output( input.width(), input.height() );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function::Median( input, output, 3 );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > PrewittFilter( uint32_t size )
    {
        Performance_Test::TimerContainer timer;

        PenguinV_Image::Image input = Performance_Test::uniformImage( size, size, Performance_Test::randomValue<uint8_t>( 1, 256 ) );
        PenguinV_Image::Image output( input.width(), input.height() );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function::Prewitt( input, output );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > SobelFilter( uint32_t size )
    {
        Performance_Test::TimerContainer timer;

        PenguinV_Image::Image input = Performance_Test::uniformImage( size, size, Performance_Test::randomValue<uint8_t>( 1, 256 ) );
        PenguinV_Image::Image output( input.width(), input.height() );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function::Sobel( input, output );

            timer.stop();
        }

        return timer.mean();
    }
}

// Function naming: _functionName_imageSize
#define SET_FUNCTION( function )                                      \
namespace filtering_##function                                        \
{                                                                     \
    std::pair < double, double > _256 () { return function( 256  ); } \
    std::pair < double, double > _512 () { return function( 512  ); } \
    std::pair < double, double > _1024() { return function( 1024 ); } \
    std::pair < double, double > _2048() { return function( 2048 ); } \
}

namespace
{
    SET_FUNCTION( MedianFilter3x3 )
    SET_FUNCTION( PrewittFilter   )
    SET_FUNCTION( SobelFilter     )
}

#define ADD_TEST_FUNCTION( framework, function )    \
ADD_TEST( framework, filtering_##function::_256 );  \
ADD_TEST( framework, filtering_##function::_512 );  \
ADD_TEST( framework, filtering_##function::_1024 ); \
ADD_TEST( framework, filtering_##function::_2048 );

void addTests_Filtering( PerformanceTestFramework & framework )
{
    ADD_TEST_FUNCTION( framework, MedianFilter3x3 )
    ADD_TEST_FUNCTION( framework, PrewittFilter   )
    ADD_TEST_FUNCTION( framework, SobelFilter     )
}
