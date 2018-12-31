#include "../../src/image_function_helper.h"
#include "../../src/math/hough_transform.h"
#include "../../src/math/hough_transform_simd.h"
#include "../../src/penguinv/cpu_identification.h"
#include "performance_test_math.h"
#include "performance_test_helper.h"

namespace
{
    std::pair < double, double > HoughTransform( )
    {
        Performance_Test::TimerContainer timer;

        const double angle = 0;
        const double angleTolerance = static_cast<double>( pvmath::toRadians( 10.0 ) );
        const double angleStep = static_cast<double>( pvmath::toRadians( 0.2 ) );
        const double lineTolerance = 1;

        const double sinVal = sin( angle );
        const double cosVal = cos( angle );

        std::vector< PointBase2D<double> > point( 1000 );
        double i = 0;

        for ( typename std::vector< PointBase2D<double> >::iterator p = point.begin(); p != point.end(); ++p, i++ ) {
                const double x = i;
                const double y = 0;

                p->x = x * cosVal - y * sinVal;
                p->y = x * sinVal + y * cosVal;
            }

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            { // destroy the object within the scope
                std::vector< PointBase2D<double> > pointOnLine;
                std::vector< PointBase2D<double> > pointOffLine;
                Image_Function::HoughTransform( point, angle, angleTolerance, angleStep, lineTolerance, pointOnLine, pointOffLine );
            }

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > HoughTransform_AVX( )
    {
        Performance_Test::TimerContainer timer;

        const double angle = 0;
        const double angleTolerance = static_cast<double>( pvmath::toRadians( 10.0 ) );
        const double angleStep = static_cast<double>( pvmath::toRadians( 0.2 ) );
        const double lineTolerance = 1;

        const double sinVal = sin( angle );
        const double cosVal = cos( angle );

        std::vector< PointBase2D<double> > point( 1000 );
        double i = 0;

        for ( typename std::vector< PointBase2D<double> >::iterator p = point.begin(); p != point.end(); ++p, i++ ) {
                const double x = i;
                const double y = 0;

                p->x = x * cosVal - y * sinVal;
                p->y = x * sinVal + y * cosVal;
            }

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            { // destroy the object within the scope
                std::vector< PointBase2D<double> > pointOnLine;
                std::vector< PointBase2D<double> > pointOffLine;
                Image_Function_Simd::HoughTransform( point, angle, angleTolerance, angleStep, lineTolerance, pointOnLine, pointOffLine );
            }

            timer.stop();
        }

        return timer.mean();
    }
}

void addTests_math( PerformanceTestFramework & framework )
{
    ADD_TEST( framework, HoughTransform );
    #ifdef PENGUINV_AVX_SET
    simd::EnableSimd( false );
    simd::EnableAvx( true );
    ADD_TEST( framework, HoughTransform_AVX );
    #endif
}
