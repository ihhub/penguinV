#include <vector>
#include "../../../src/cuda/image_function_cuda.cuh"
#include "../performance_test_framework.h"
#include "performance_test_image_function_cuda.h"
#include "performance_test_helper_cuda.cuh"

namespace
{
    #define TEST_FUNCTION_LOOP( testFunction )                         \
        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) { \
            timer.start();                                             \
            testFunction;                                              \
            timer.stop();                                              \
        }

    void AbsoluteDifference( Performance_Test::Cuda_Helper::TimerContainerCuda & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_Cuda::AbsoluteDifference( image[0], image[1], image[2] ) )
    }

    void BitwiseAnd( Performance_Test::Cuda_Helper::TimerContainerCuda & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_Cuda::BitwiseAnd( image[0], image[1], image[2] ) )
    }

    void BitwiseOr( Performance_Test::Cuda_Helper::TimerContainerCuda & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_Cuda::BitwiseOr( image[0], image[1], image[2] ) )
    }

    void BitwiseXor( Performance_Test::Cuda_Helper::TimerContainerCuda & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_Cuda::BitwiseXor( image[0], image[1], image[2] ) )
    }

    void Flip( Performance_Test::Cuda_Helper::TimerContainerCuda & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::Cuda_Helper::uniformImages( 2, size, size );

        TEST_FUNCTION_LOOP( Image_Function_Cuda::Flip( image[0], image[1], true, true ) )
    }

    void Histogram( Performance_Test::Cuda_Helper::TimerContainerCuda & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::Cuda_Helper::uniformImages( 1, size, size );

        TEST_FUNCTION_LOOP( Image_Function_Cuda::Histogram( image[0] ) )
    }

    void Invert( Performance_Test::Cuda_Helper::TimerContainerCuda & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_Cuda::Invert( image[0], image[1] ) )
    }

    void LookupTable( Performance_Test::Cuda_Helper::TimerContainerCuda & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::Cuda_Helper::uniformImages( 2, size, size );
        const std::vector<uint8_t> table( 256, 0);

        TEST_FUNCTION_LOOP( Image_Function_Cuda::LookupTable( image[0], image[1], table ) )
    }

    void Maximum( Performance_Test::Cuda_Helper::TimerContainerCuda & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_Cuda::Maximum( image[0], image[1], image[2] ) )
    }

    void Minimum( Performance_Test::Cuda_Helper::TimerContainerCuda & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_Cuda::Minimum( image[0], image[1], image[2] ) )
    }

    void Subtract( Performance_Test::Cuda_Helper::TimerContainerCuda & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        TEST_FUNCTION_LOOP( Image_Function_Cuda::Subtract( image[0], image[1], image[2] ) )
    }

    void Threshold( Performance_Test::Cuda_Helper::TimerContainerCuda & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::Cuda_Helper::uniformImages( 2, size, size );

        TEST_FUNCTION_LOOP( Image_Function_Cuda::Threshold( image[0], image[1], 128 ) )
    }

    void ThresholdDouble( Performance_Test::Cuda_Helper::TimerContainerCuda & timer, uint32_t size )
    {
        std::vector<penguinV::Image> image = Performance_Test::Cuda_Helper::uniformImages( 2, size, size );

        TEST_FUNCTION_LOOP( Image_Function_Cuda::Threshold( image[0], image[1], 64, 192 ) )
    }
}

// Function naming: function_name_(image_size)_(divider_of_maximim_thread_count)
#define SET_FUNCTION( function )                                                                                                  \
namespace cuda_##function                                                                                                         \
{                                                                                                                                 \
    std::pair < double, double >  _256_1() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function,  256, 1 ); } \
    std::pair < double, double >  _512_1() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function,  512, 1 ); } \
    std::pair < double, double > _1024_1() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function, 1024, 1 ); } \
    std::pair < double, double > _2048_1() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function, 2048, 1 ); } \
    std::pair < double, double >  _256_2() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function,  256, 2 ); } \
    std::pair < double, double >  _512_2() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function,  512, 2 ); } \
    std::pair < double, double > _1024_2() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function, 1024, 2 ); } \
    std::pair < double, double > _2048_2() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function, 2048, 2 ); } \
    std::pair < double, double >  _256_4() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function,  256, 4 ); } \
    std::pair < double, double >  _512_4() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function,  512, 4 ); } \
    std::pair < double, double > _1024_4() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function, 1024, 4 ); } \
    std::pair < double, double > _2048_4() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function, 2048, 4 ); } \
    std::pair < double, double >  _256_8() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function,  256, 8 ); } \
    std::pair < double, double >  _512_8() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function,  512, 8 ); } \
    std::pair < double, double > _1024_8() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function, 1024, 8 ); } \
    std::pair < double, double > _2048_8() { return Performance_Test::Cuda_Helper::runPerformanceTestCuda( function, 2048, 8 ); } \
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

#define ADD_TEST_FUNCTION( framework, function ) \
ADD_TEST( framework, cuda_##function:: _256_1 ); \
ADD_TEST( framework, cuda_##function:: _512_1 ); \
ADD_TEST( framework, cuda_##function::_1024_1 ); \
ADD_TEST( framework, cuda_##function::_2048_1 ); \
ADD_TEST( framework, cuda_##function:: _256_2 ); \
ADD_TEST( framework, cuda_##function:: _512_2 ); \
ADD_TEST( framework, cuda_##function::_1024_2 ); \
ADD_TEST( framework, cuda_##function::_2048_2 ); \
ADD_TEST( framework, cuda_##function:: _256_4 ); \
ADD_TEST( framework, cuda_##function:: _512_4 ); \
ADD_TEST( framework, cuda_##function::_1024_4 ); \
ADD_TEST( framework, cuda_##function::_2048_4 ); \
ADD_TEST( framework, cuda_##function:: _256_8 ); \
ADD_TEST( framework, cuda_##function:: _512_8 ); \
ADD_TEST( framework, cuda_##function::_1024_8 ); \
ADD_TEST( framework, cuda_##function::_2048_8 );

void addTests_Image_Function_Cuda( PerformanceTestFramework & framework )
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
