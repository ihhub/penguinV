#include <vector>
#include "../../../src/opencl/image_function_opencl.h"
#include "../performance_test_framework.h"
#include "performance_test_image_function_opencl.h"
#include "performance_test_helper_opencl.h"

namespace
{
    #define TEST_FUNCTION_LOOP( testFunction )                         \
        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) { \
            timer.start();                                             \
            testFunction;                                              \
            timer.stop();                                              \
        }

    void AbsoluteDifference( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::OpenCL_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_OpenCL::AbsoluteDifference( image[0], image[1], image[2] ) )
    }

    void BitwiseAnd( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::OpenCL_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_OpenCL::BitwiseAnd( image[0], image[1], image[2] ) )
    }

    void BitwiseOr( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::OpenCL_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_OpenCL::BitwiseOr( image[0], image[1], image[2] ) )
    }

    void BitwiseXor( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::OpenCL_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_OpenCL::BitwiseXor( image[0], image[1], image[2] ) )
    }

    void Flip( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::OpenCL_Helper::uniformImages( 2, size, size );

        TEST_FUNCTION_LOOP( Image_Function_OpenCL::Flip( image[0], image[1], true, true ) )
    }

    void Histogram( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::OpenCL_Helper::uniformImages( 1, size, size );

        TEST_FUNCTION_LOOP( Image_Function_OpenCL::Histogram( image[0] ) )
    }

    void Invert( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::OpenCL_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_OpenCL::Invert( image[0], image[1] ) )
    }

    void LookupTable( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::OpenCL_Helper::uniformImages( 2, size, size );
        const std::vector<uint8_t> table( 256, 0);

        TEST_FUNCTION_LOOP( Image_Function_OpenCL::LookupTable( image[0], image[1], table ) )
    }

    void Maximum( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::OpenCL_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_OpenCL::Maximum( image[0], image[1], image[2] ) )
    }

    void Minimum( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::OpenCL_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_OpenCL::Minimum( image[0], image[1], image[2] ) )
    }

    void Subtract( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::OpenCL_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_OpenCL::Subtract( image[0], image[1], image[2] ) )
    }

    void Threshold( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::OpenCL_Helper::uniformImages( 2, size, size );

        TEST_FUNCTION_LOOP( Image_Function_OpenCL::Threshold( image[0], image[1], 128 ) )
    }

    void ThresholdDouble( Performance_Test::TimerContainer & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::OpenCL_Helper::uniformImages( 2, size, size );

        TEST_FUNCTION_LOOP( Image_Function_OpenCL::Threshold( image[0], image[1], 64, 192 ) )
    }
}

// Function naming: function_name_(image_size)_(divider_of_maximim_thread_count)
#define SET_FUNCTION( function )                                                                                 \
namespace opencl_##function                                                                                        \
{                                                                                                                \
    std::pair < double, double >  _256_1() { return Performance_Test::runPerformanceTest( function,  256 ); } \
    std::pair < double, double >  _512_1() { return Performance_Test::runPerformanceTest( function,  512 ); } \
    std::pair < double, double > _1024_1() { return Performance_Test::runPerformanceTest( function, 1024 ); } \
    std::pair < double, double > _2048_1() { return Performance_Test::runPerformanceTest( function, 2048 ); } \
}

 namespace
 {
    SET_FUNCTION( AbsoluteDifference )
    SET_FUNCTION( BitwiseAnd         )
    SET_FUNCTION( BitwiseOr          )
    SET_FUNCTION( BitwiseXor         )
    SET_FUNCTION( Flip               )
    SET_FUNCTION( Histogram          )
    SET_FUNCTION( Invert             )
    SET_FUNCTION( LookupTable        )
    SET_FUNCTION( Maximum            )
    SET_FUNCTION( Minimum            )
    SET_FUNCTION( Subtract           )
    SET_FUNCTION( Threshold          )
    SET_FUNCTION( ThresholdDouble    )
 }

#define ADD_TEST_FUNCTION( framework, function )   \
ADD_TEST( framework, opencl_##function:: _256_1 ); \
ADD_TEST( framework, opencl_##function:: _512_1 ); \
ADD_TEST( framework, opencl_##function::_1024_1 ); \
ADD_TEST( framework, opencl_##function::_2048_1 );

void addTests_Image_Function_OpenCL( PerformanceTestFramework & framework )
{
    ADD_TEST_FUNCTION( framework, AbsoluteDifference )
    ADD_TEST_FUNCTION( framework, BitwiseAnd         )
    ADD_TEST_FUNCTION( framework, BitwiseOr          )
    ADD_TEST_FUNCTION( framework, BitwiseXor         )
    ADD_TEST_FUNCTION( framework, Flip               )
    ADD_TEST_FUNCTION( framework, Histogram          )
    ADD_TEST_FUNCTION( framework, Invert             )
    ADD_TEST_FUNCTION( framework, LookupTable        )
    ADD_TEST_FUNCTION( framework, Maximum            )
    ADD_TEST_FUNCTION( framework, Minimum            )
    ADD_TEST_FUNCTION( framework, Subtract           )
    ADD_TEST_FUNCTION( framework, Threshold          )
    ADD_TEST_FUNCTION( framework, ThresholdDouble    )
}
