#include "../../src/image_function_helper.h"
#include "../../src/math/hough_transform.h"
#include "../../src/math/hough_transform_simd.h"
#include "../../src/penguinv/cpu_identification.h"
#include "performance_test_math.h"
#include "performance_test_helper.h"

namespace
{
    typedef bool (*houghFunctionFloat)( const std::vector< PointBase2D<float> > &, float, float, float, float,
                                        std::vector< PointBase2D<float> > &, std::vector< PointBase2D<float> > & );

    typedef bool (*houghFunctionDouble)( const std::vector< PointBase2D<double> > &, double, double, double, double,
                                         std::vector< PointBase2D<double> > &, std::vector< PointBase2D<double> > & );
    
    template <typename _Type, typename _Hough>
    std::pair < double, double > HoughTransformTemplate( _Hough hough )
    {
        Performance_Test::TimerContainer timer;

        const _Type angle = 0;
        const _Type angleTolerance = static_cast< _Type >( pvmath::toRadians( 10.0 ) );
        const _Type angleStep = static_cast< _Type >( pvmath::toRadians( 0.2 ) );
        const _Type lineTolerance = 1;

        const uint32_t pointCount = 10000u;
        std::vector< PointBase2D<_Type> > point( pointCount );
        for ( uint32_t i = 0; i < pointCount; ++i )
            point[i] = PointBase2D<_Type>( static_cast< _Type >( i ), 0 );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            std::vector< PointBase2D<_Type> > pointOnLine;
            std::vector< PointBase2D<_Type> > pointOffLine;

            timer.start();

            hough( point, angle, angleTolerance, angleStep, lineTolerance, pointOnLine, pointOffLine );

            timer.stop();
        }

        return timer.mean();
    }

    std::pair < double, double > HoughTransformFloat()
    {
        return HoughTransformTemplate<float, houghFunctionFloat>( Image_Function::HoughTransform );
    }

    std::pair < double, double > HoughTransformDouble()
    {
        return HoughTransformTemplate<double, houghFunctionDouble>( Image_Function::HoughTransform );
    }

    std::pair < double, double > HoughTransformAvxFloat()
    {
        simd::EnableSimd( false );
        simd::EnableAvx( true );
        const std::pair < double, double > result = HoughTransformTemplate<float, houghFunctionFloat>( Image_Function_Simd::HoughTransform );
        simd::EnableSimd( true );
        return result;
    }

    std::pair < double, double > HoughTransformAvxDouble()
    {
        simd::EnableSimd( false );
        simd::EnableAvx( true );
        const std::pair < double, double > result = HoughTransformTemplate<double, houghFunctionDouble>( Image_Function_Simd::HoughTransform );
        simd::EnableSimd( true );
        return result;
    }
}

void addTests_Math( PerformanceTestFramework & framework )
{
    ADD_TEST( framework, HoughTransformFloat );
    ADD_TEST( framework, HoughTransformDouble );
#ifdef PENGUINV_AVX_SET
    ADD_TEST( framework, HoughTransformAvxFloat );
    ADD_TEST( framework, HoughTransformAvxDouble );
#endif
}
