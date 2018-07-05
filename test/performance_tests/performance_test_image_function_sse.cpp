#include "../../src/image_function_sse.h"
#include "../../src/penguinv/cpu_identification.h"
#include "performance_test_image_function_sse.h"
#include "performance_test_helper.h"

#ifdef PENGUINV_SSE_SET
namespace
{
    void AbsoluteDifference( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Sse::AbsoluteDifference( image[0], image[1], image[2] );

            timer.stop();
        }
    }

    void BitwiseAnd( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Sse::BitwiseAnd( image[0], image[1], image[2] );

            timer.stop();
        }
    }

    void BitwiseOr( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Sse::BitwiseOr( image[0], image[1], image[2] );

            timer.stop();
        }
    }

    void BitwiseXor( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Sse::BitwiseXor( image[0], image[1], image[2] );

            timer.stop();
        }
    }

    void Invert( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Sse::Invert( image[0], image[1] );

            timer.stop();
        }
    }

    void Maximum( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Sse::Maximum( image[0], image[1], image[2] );

            timer.stop();
        }
    }

    void Minimum( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Sse::Minimum( image[0], image[1], image[2] );

            timer.stop();
        }
    }

    void Subtract( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Sse::Subtract( image[0], image[1], image[2] );

            timer.stop();
        }
    }

    void Sum( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        PenguinV_Image::Image image = Performance_Test::uniformImage( size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Sse::Sum( image );

            timer.stop();
        }
    }

    void Threshold( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );
        uint8_t threshold = Performance_Test::randomValue<uint8_t>( 256 );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Sse::Threshold( image[0], image[1], threshold );

            timer.stop();
        }
    }

    void ThresholdDouble( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector < PenguinV_Image::Image > image = Performance_Test::uniformImages( 2, size, size );
        uint8_t minThreshold = Performance_Test::randomValue<uint8_t>( 256 );
        uint8_t maxThreshold = Performance_Test::randomValue<uint8_t>( minThreshold, 256 );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Sse::Threshold( image[0], image[1], minThreshold, maxThreshold );

            timer.stop();
        }
    }
}
#endif

// Function naming: _functionName_imageSize
#define SET_FUNCTION( function )                                                                            \
namespace sse_##function                                                                                    \
{                                                                                                           \
    std::pair < double, double > _256 () { return Performance_Test::runPerformanceTest( function,  256 ); } \
    std::pair < double, double > _512 () { return Performance_Test::runPerformanceTest( function,  512 ); } \
    std::pair < double, double > _1024() { return Performance_Test::runPerformanceTest( function, 1024 ); } \
    std::pair < double, double > _2048() { return Performance_Test::runPerformanceTest( function, 2048 ); } \
}

#ifdef PENGUINV_SSE_SET
namespace
{
    SET_FUNCTION( AbsoluteDifference )
    SET_FUNCTION( BitwiseAnd         )
    SET_FUNCTION( BitwiseOr          )
    SET_FUNCTION( BitwiseXor         )
    SET_FUNCTION( Invert             )
    SET_FUNCTION( Maximum            )
    SET_FUNCTION( Minimum            )
    SET_FUNCTION( Subtract           )
    SET_FUNCTION( Sum                )
    SET_FUNCTION( Threshold          )
    SET_FUNCTION( ThresholdDouble    )
}
#endif

#define ADD_TEST_FUNCTION( framework, function ) \
ADD_TEST( framework, sse_##function::_256 );     \
ADD_TEST( framework, sse_##function::_512 );     \
ADD_TEST( framework, sse_##function::_1024 );    \
ADD_TEST( framework, sse_##function::_2048 );

#ifdef PENGUINV_SSE_SET
void addTests_Image_Function_Sse( PerformanceTestFramework & framework)
{
    if( isSseAvailable ) {
        ADD_TEST_FUNCTION( framework, AbsoluteDifference )
        ADD_TEST_FUNCTION( framework, BitwiseAnd         )
        ADD_TEST_FUNCTION( framework, BitwiseOr          )
        ADD_TEST_FUNCTION( framework, BitwiseXor         )
        ADD_TEST_FUNCTION( framework, Invert             )
        ADD_TEST_FUNCTION( framework, Maximum            )
        ADD_TEST_FUNCTION( framework, Minimum            )
        ADD_TEST_FUNCTION( framework, Subtract           )
        ADD_TEST_FUNCTION( framework, Sum                )
        ADD_TEST_FUNCTION( framework, Threshold          )
        ADD_TEST_FUNCTION( framework, ThresholdDouble    )
    }
}
#else
void addTests_Image_Function_Sse( PerformanceTestFramework & ) {}
#endif
