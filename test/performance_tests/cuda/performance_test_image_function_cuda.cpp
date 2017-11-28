#include <vector>
#include "../../../src/cuda/image_function_cuda.cuh"
#include "performance_test_image_function_cuda.h"
#include "performance_test_helper_cuda.cuh"

namespace
{
    std::pair < double, double > AbsoluteDifference( uint32_t size, uint32_t threadCount )
    {
        Performance_Test::Cuda_Helper::setCudaThreadCount( threadCount );
        Performance_Test::Cuda_Helper::TimerContainerCuda timer;

        std::vector < Bitmap_Image_Cuda::Image > image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Cuda::AbsoluteDifference( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > BitwiseAnd( uint32_t size, uint32_t threadCount )
    {
        Performance_Test::Cuda_Helper::setCudaThreadCount( threadCount );
        Performance_Test::Cuda_Helper::TimerContainerCuda timer;

        std::vector < Bitmap_Image_Cuda::Image > image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Cuda::BitwiseAnd( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > BitwiseOr( uint32_t size, uint32_t threadCount )
    {
        Performance_Test::Cuda_Helper::setCudaThreadCount( threadCount );
        Performance_Test::Cuda_Helper::TimerContainerCuda timer;

        std::vector < Bitmap_Image_Cuda::Image > image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Cuda::BitwiseOr( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > BitwiseXor( uint32_t size, uint32_t threadCount )
    {
        Performance_Test::Cuda_Helper::setCudaThreadCount( threadCount );
        Performance_Test::Cuda_Helper::TimerContainerCuda timer;

        std::vector < Bitmap_Image_Cuda::Image > image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Cuda::BitwiseXor( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Flip( uint32_t size, uint32_t threadCount )
    {
        Performance_Test::Cuda_Helper::setCudaThreadCount( threadCount );
        Performance_Test::Cuda_Helper::TimerContainerCuda timer;

        std::vector < Bitmap_Image_Cuda::Image > image = Performance_Test::Cuda_Helper::uniformImages( 2, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Cuda::Flip( image[0], image[1], true, true );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Histogram( uint32_t size, uint32_t threadCount )
    {
        Performance_Test::Cuda_Helper::setCudaThreadCount( threadCount );
        Performance_Test::Cuda_Helper::TimerContainerCuda timer;

        std::vector < Bitmap_Image_Cuda::Image > image = Performance_Test::Cuda_Helper::uniformImages( 1, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Cuda::Histogram( image[0] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Invert( uint32_t size, uint32_t threadCount )
    {
        Performance_Test::Cuda_Helper::setCudaThreadCount( threadCount );
        Performance_Test::Cuda_Helper::TimerContainerCuda timer;

        std::vector < Bitmap_Image_Cuda::Image > image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Cuda::Invert( image[0], image[1] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > LookupTable( uint32_t size, uint32_t threadCount )
    {
        Performance_Test::Cuda_Helper::setCudaThreadCount( threadCount );
        Performance_Test::Cuda_Helper::TimerContainerCuda timer;

        std::vector < Bitmap_Image_Cuda::Image > image = Performance_Test::Cuda_Helper::uniformImages( 2, size, size );
        const std::vector<uint8_t> table( 256, 0);


        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Cuda::LookupTable( image[0], image[1], table );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Maximum( uint32_t size, uint32_t threadCount )
    {
        Performance_Test::Cuda_Helper::setCudaThreadCount( threadCount );
        Performance_Test::Cuda_Helper::TimerContainerCuda timer;

        std::vector < Bitmap_Image_Cuda::Image > image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Cuda::Maximum( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Minimum( uint32_t size, uint32_t threadCount )
    {
        Performance_Test::Cuda_Helper::setCudaThreadCount( threadCount );
        Performance_Test::Cuda_Helper::TimerContainerCuda timer;

        std::vector < Bitmap_Image_Cuda::Image > image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Cuda::Minimum( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Subtract( uint32_t size, uint32_t threadCount )
    {
        Performance_Test::Cuda_Helper::setCudaThreadCount( threadCount );
        Performance_Test::Cuda_Helper::TimerContainerCuda timer;

        std::vector < Bitmap_Image_Cuda::Image > image = Performance_Test::Cuda_Helper::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Cuda::Subtract( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Threshold( uint32_t size, uint32_t threadCount )
    {
        Performance_Test::Cuda_Helper::setCudaThreadCount( threadCount );
        Performance_Test::Cuda_Helper::TimerContainerCuda timer;

        std::vector < Bitmap_Image_Cuda::Image > image = Performance_Test::Cuda_Helper::uniformImages( 2, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Cuda::Threshold( image[0], image[1], 128 );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > ThresholdDouble( uint32_t size, uint32_t threadCount )
    {
        Performance_Test::Cuda_Helper::setCudaThreadCount( threadCount );
        Performance_Test::Cuda_Helper::TimerContainerCuda timer;

        std::vector < Bitmap_Image_Cuda::Image > image = Performance_Test::Cuda_Helper::uniformImages( 2, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Image_Function_Cuda::Threshold( image[0], image[1], 64, 192 );

            timer.stop();
        }

        return timer.mean();
    }
}

// Function naming: function_name_(image_size)_(divider_of_maximim_thread_count)
#define CONVERT_PARAMETER( parameter ) _##parameter

#define DECLARE_FUNCTION( function, size, divider )                                                         \
std::pair < double, double > CONVERT_PARAMETER(function)CONVERT_PARAMETER(size)CONVERT_PARAMETER(divider)() \
{                                                                                                           \
    return function( size, Performance_Test::Cuda_Helper::getMaximumCudaThreadCount() / divider );          \
}

#define DECLARE_FUNCTIONS_FIXED_SIZE( function, size, divider1, divider2, divider3, divider4 ) \
DECLARE_FUNCTION( function, size, divider1 )                                                   \
DECLARE_FUNCTION( function, size, divider2 )                                                   \
DECLARE_FUNCTION( function, size, divider3 )                                                   \
DECLARE_FUNCTION( function, size, divider4 )

#define DECLARE_FUNCTIONS( function, size1, size2, size3, size4, divider1, divider2, divider3, divider4 ) \
DECLARE_FUNCTIONS_FIXED_SIZE( function, size1, divider1, divider2, divider3, divider4 )                   \
DECLARE_FUNCTIONS_FIXED_SIZE( function, size2, divider1, divider2, divider3, divider4 )                   \
DECLARE_FUNCTIONS_FIXED_SIZE( function, size3, divider1, divider2, divider3, divider4 )                   \
DECLARE_FUNCTIONS_FIXED_SIZE( function, size4, divider1, divider2, divider3, divider4 )

#define SET_FUNCTION( function ) DECLARE_FUNCTIONS( function, 256, 512, 1024, 2048, 1, 2, 4, 8 );

 namespace cuda
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

#define ADD_TEST_FUNCTIONS_FIXED_SIZE( framework, function, size, divider1, divider2, divider3, divider4 )  \
ADD_TEST( framework, cuda::CONVERT_PARAMETER(function)CONVERT_PARAMETER(size)CONVERT_PARAMETER(divider1) ); \
ADD_TEST( framework, cuda::CONVERT_PARAMETER(function)CONVERT_PARAMETER(size)CONVERT_PARAMETER(divider2) ); \
ADD_TEST( framework, cuda::CONVERT_PARAMETER(function)CONVERT_PARAMETER(size)CONVERT_PARAMETER(divider3) ); \
ADD_TEST( framework, cuda::CONVERT_PARAMETER(function)CONVERT_PARAMETER(size)CONVERT_PARAMETER(divider4) );

#define ADD_FUNCTIONS( framework, function, size1, size2, size3, size4, divider1, divider2, divider3, divider4 ) \
ADD_TEST_FUNCTIONS_FIXED_SIZE( framework, function, size1, divider1, divider2, divider3, divider4 )              \
ADD_TEST_FUNCTIONS_FIXED_SIZE( framework, function, size2, divider1, divider2, divider3, divider4 )              \
ADD_TEST_FUNCTIONS_FIXED_SIZE( framework, function, size3, divider1, divider2, divider3, divider4 )              \
ADD_TEST_FUNCTIONS_FIXED_SIZE( framework, function, size4, divider1, divider2, divider3, divider4 )

#define ADD_TEST_FUNCTION( framework, function ) ADD_FUNCTIONS( framework, function, 256, 512, 1024, 2048, 1, 2, 4, 8 )

namespace Performance_Test
{
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
}
