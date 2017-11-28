#include "../../src/function_pool.h"
#include "performance_test_function_pool.h"
#include "performance_test_helper.h"

namespace
{
    std::pair < double, double > AbsoluteDifference( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        std::vector < Bitmap_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::AbsoluteDifference( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > BitwiseAnd( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        std::vector < Bitmap_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::BitwiseAnd( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > BitwiseOr( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        std::vector < Bitmap_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::BitwiseOr( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > BitwiseXor( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        std::vector < Bitmap_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::BitwiseXor( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > ConvertToColor( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();
    
        Bitmap_Image::Image input  = Performance_Test::uniformImage     ( size, size );
        Bitmap_Image::Image output = Performance_Test::uniformColorImage( size, size );
    
        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();
    
            Function_Pool::ConvertToRgb( input, output );
    
            timer.stop();
        }
    
        return timer.mean();
    }
 
    std::pair < double, double > ConvertToGrayscale( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();
    
        Bitmap_Image::Image input  = Performance_Test::uniformColorImage( size, size );
        Bitmap_Image::Image output = Performance_Test::uniformImage     ( size, size );
    
        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();
    
            Function_Pool::ConvertToGrayScale( input, output );
    
            timer.stop();
        }
    
        return timer.mean();
    }

    std::pair < double, double > GammaCorrection( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        std::vector < Bitmap_Image::Image > image = Performance_Test::uniformImages( 2, size, size );

        double a     = Performance_Test::randomValue <uint32_t>( 100 ) / 100.0;
        double gamma = Performance_Test::randomValue <uint32_t>( 300 ) / 100.0;

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::GammaCorrection( image[0], image[1], a, gamma );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Histogram( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        Bitmap_Image::Image image = Performance_Test::uniformImage( size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::Histogram( image );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Invert( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        std::vector < Bitmap_Image::Image > image = Performance_Test::uniformImages( 2, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::Invert( image[0], image[1] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > LookupTable( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        std::vector < Bitmap_Image::Image > image = Performance_Test::uniformImages( 2, size, size );

        std::vector<uint8_t> table(256, 0);

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::LookupTable( image[0], image[1], table );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Maximum( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        std::vector < Bitmap_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::Maximum( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Minimum( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        std::vector < Bitmap_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::Minimum( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > RgbToBgr( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();
    
        std::vector < Bitmap_Image::Image > image = Performance_Test::uniformColorImages( 2, size, size );
    
        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();
    
            Function_Pool::RgbToBgr( image[0], image[1] );
    
            timer.stop();
        }
    
        return timer.mean();
    }

    std::pair < double, double > ResizeDown( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        Bitmap_Image::Image input  = Performance_Test::uniformImage( size, size );
        Bitmap_Image::Image output = Performance_Test::uniformImage( size / 2, size / 2 );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::Resize( input, output );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > ResizeUp( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        Bitmap_Image::Image input  = Performance_Test::uniformImage( size, size );
        Bitmap_Image::Image output = Performance_Test::uniformImage( size * 2, size * 2 );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::Resize( input, output );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Subtract( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        std::vector < Bitmap_Image::Image > image = Performance_Test::uniformImages( 3, size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::Subtract( image[0], image[1], image[2] );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Sum( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        Bitmap_Image::Image image = Performance_Test::uniformImage( size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::Sum( image );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > Threshold( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        std::vector < Bitmap_Image::Image > image = Performance_Test::uniformImages( 2, size, size );
        uint8_t threshold = Performance_Test::randomValue<uint8_t>( 256 );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::Threshold( image[0], image[1], threshold );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > ThresholdDouble( uint32_t size )
    {
        Performance_Test::TimerContainer timer;
        Performance_Test::setFunctionPoolThreadCount();

        std::vector < Bitmap_Image::Image > image = Performance_Test::uniformImages( 2, size, size );
        uint8_t minThreshold = Performance_Test::randomValue<uint8_t>( 256 );
        uint8_t maxThreshold = Performance_Test::randomValue<uint8_t>( minThreshold, 256 );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Function_Pool::Threshold( image[0], image[1], minThreshold, maxThreshold );

            timer.stop();
        }

        return timer.mean();
    }
}

// Function naming: function_name_(image_size)
#define CONVERT_PARAMETER( parameter ) _##parameter

#define DECLARE_FUNCTION( function, size )                                        \
std::pair < double, double > CONVERT_PARAMETER(function)CONVERT_PARAMETER(size)() \
{                                                                                 \
    return function( size );                                                      \
}

#define DECLARE_FUNCTIONS( function, size1, size2, size3, size4 ) \
DECLARE_FUNCTION( function, size1 )                               \
DECLARE_FUNCTION( function, size2 )                               \
DECLARE_FUNCTION( function, size3 )                               \
DECLARE_FUNCTION( function, size4 )

#define SET_FUNCTION( function ) DECLARE_FUNCTIONS( function, 256, 512, 1024, 2048 );

namespace function_pool
{
    SET_FUNCTION( AbsoluteDifference )
    SET_FUNCTION( BitwiseAnd         )
    SET_FUNCTION( BitwiseOr          )
    SET_FUNCTION( BitwiseXor         )
    SET_FUNCTION( ConvertToColor     )
    SET_FUNCTION( ConvertToGrayscale )
    SET_FUNCTION( GammaCorrection    )
    SET_FUNCTION( Histogram          )
    SET_FUNCTION( Invert             )
    SET_FUNCTION( LookupTable        )
    SET_FUNCTION( Maximum            )
    SET_FUNCTION( Minimum            )
    SET_FUNCTION( RgbToBgr           )
    SET_FUNCTION( ResizeDown         )
    SET_FUNCTION( ResizeUp           )
    SET_FUNCTION( Subtract           )
    SET_FUNCTION( Sum                )
    SET_FUNCTION( Threshold          )
    SET_FUNCTION( ThresholdDouble    )
}

#define ADD_FUNCTIONS( framework, function, size1, size2, size3, size4 )               \
ADD_TEST( framework, function_pool::CONVERT_PARAMETER(function)CONVERT_PARAMETER(size1) ); \
ADD_TEST( framework, function_pool::CONVERT_PARAMETER(function)CONVERT_PARAMETER(size2) ); \
ADD_TEST( framework, function_pool::CONVERT_PARAMETER(function)CONVERT_PARAMETER(size3) ); \
ADD_TEST( framework, function_pool::CONVERT_PARAMETER(function)CONVERT_PARAMETER(size4) );

#define ADD_TEST_FUNCTION( framework, function ) ADD_FUNCTIONS( framework, function, 256, 512, 1024, 2048 )

namespace Performance_Test
{
    void addTests_Function_Pool( PerformanceTestFramework & framework )
    {
        ADD_TEST_FUNCTION( framework, AbsoluteDifference )
        ADD_TEST_FUNCTION( framework, BitwiseAnd         )
        ADD_TEST_FUNCTION( framework, BitwiseOr          )
        ADD_TEST_FUNCTION( framework, BitwiseXor         )
        ADD_TEST_FUNCTION( framework, ConvertToColor     )
        ADD_TEST_FUNCTION( framework, ConvertToGrayscale )
        ADD_TEST_FUNCTION( framework, GammaCorrection    )
        ADD_TEST_FUNCTION( framework, Histogram          )
        ADD_TEST_FUNCTION( framework, Invert             )
        ADD_TEST_FUNCTION( framework, LookupTable        )
        ADD_TEST_FUNCTION( framework, Maximum            )
        ADD_TEST_FUNCTION( framework, Minimum            )
        ADD_TEST_FUNCTION( framework, RgbToBgr           )
        ADD_TEST_FUNCTION( framework, ResizeDown         )
        ADD_TEST_FUNCTION( framework, ResizeUp           )
        ADD_TEST_FUNCTION( framework, Subtract           )
        ADD_TEST_FUNCTION( framework, Sum                )
        ADD_TEST_FUNCTION( framework, Threshold          )
        ADD_TEST_FUNCTION( framework, ThresholdDouble    )
    }
}
