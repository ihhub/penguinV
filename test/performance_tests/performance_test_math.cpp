#include "../../src/image_function_helper.h"
#include "../../src/math/hough_transform.h"
#include "../../src/math/hough_transform_simd.h"
#include "../../src/penguinv/cpu_identification.h"
#include "performance_test_math.h"
#include "performance_test_helper.h"

namespace
{
    typedef bool (*houghFunction)( const std::vector< Point2d > &, double, double, double, double,
                                   std::vector< Point2d > &, std::vector< Point2d > & );
    
    std::pair < double, double > HoughTransformTemplate( houghFunction hough )
    {
        Performance_Test::TimerContainer timer;

        const double angle = 0;
        const double angleTolerance = pvmath::toRadians( 10.0 );
        const double angleStep = pvmath::toRadians( 0.2 );
        const double lineTolerance = 1;

        const uint32_t pointCount = 10000u;
        std::vector< Point2d > point( pointCount );
        for ( uint32_t i = 0; i <pointCount; ++i )
            point[i] = Point2d( static_cast< double >( i ), 0 );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            std::vector< Point2d > pointOnLine;
            std::vector< Point2d > pointOffLine;
            
            timer.start();

            hough( point, angle, angleTolerance, angleStep, lineTolerance, pointOnLine, pointOffLine );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > HoughTransform()
    {
        return HoughTransformTemplate( Image_Function::HoughTransform );
    }
    
    std::pair < double, double > HoughTransformAvx()
    {
        simd::EnableSimd( false );
        simd::EnableAvx( true );
        const std::pair < double, double > result = HoughTransformTemplate( Image_Function_Simd::HoughTransform );
        simd::EnableSimd( true );
        return result;
    }
}

void addTests_Math( PerformanceTestFramework & framework )
{
    ADD_TEST( framework, HoughTransform );
#ifdef PENGUINV_AVX_SET
    ADD_TEST( framework, HoughTransformAvx );
#endif
}
