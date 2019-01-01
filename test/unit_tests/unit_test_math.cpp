#include "unit_test_math.h"

#include "unit_test_helper.h"
#include "../../src/math/hough_transform.h"
#include "../../src/math/hough_transform_simd.h"
#include "../../src/penguinv/cpu_identification.h"
#include "../../src/image_function_helper.h"

namespace pvmath
{
    typedef bool (*houghFunction)( const std::vector< Point2d > &, double, double, double, double,
                                   std::vector< Point2d > &, std::vector< Point2d > & );
    
    template <typename _Type>
    bool houghTransformTemplate( houghFunction hough )
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const _Type angle = static_cast<_Type>( toRadians( Unit_Test::randomFloatValue<_Type>(-180, 180, 1 ) ) );
            const _Type angleTolerance = static_cast<_Type>( toRadians( Unit_Test::randomFloatValue<_Type>( 0, 10, 0.1f ) + 0.1f ) );
            const _Type angleStep = angleTolerance / Unit_Test::randomValue( 1, 50 );
            const _Type lineTolerance = Unit_Test::randomFloatValue<_Type>( 0.1f, 5, 0.01f );

            const _Type noiseValue = lineTolerance / 2;
            std::vector< PointBase2D<_Type> > point( Unit_Test::randomValue<uint32_t>( 50u, 100u ) );

            const _Type sinVal = sin( angle );
            const _Type cosVal = cos( angle );

            for ( typename std::vector< PointBase2D<_Type> >::iterator p = point.begin(); p != point.end(); ++p ) {
                const _Type x = Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ) + Unit_Test::randomFloatValue<_Type>( -noiseValue, noiseValue, noiseValue / 10 );
                const _Type y = Unit_Test::randomFloatValue<_Type>( -noiseValue, noiseValue, noiseValue / 10 );

                p->x = x * cosVal - y * sinVal;
                p->y = x * sinVal + y * cosVal;
            }

            std::vector< PointBase2D<_Type> > pointOnLine;
            std::vector< PointBase2D<_Type> > pointOffLine;

            if ( !hough( point, angle, angleTolerance, angleStep, lineTolerance, pointOnLine, pointOffLine ) ||
                 !pointOffLine.empty() )
                return false;
        }

        return true;
    }

    bool houghTransform_double()
    {
        return houghTransformTemplate<double>( Image_Function::HoughTransform );
    }
    
    bool houghTransform_float()
    {
        return houghTransformTemplate<float>( Image_Function::HoughTransform );
    }

    bool houghTransformSimd_double()
    {
        return houghTransformSimdTemplate<double>( Image_Function_Simd::HoughTransform );
    }
    
    bool houghTransformSimd_float()
    {
        return houghTransformSimdTemplate<float>( Image_Function_Simd::HoughTransform );
    }
    
    bool lineConstructor()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const Point2d point1( Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ), Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ) );
            const Point2d point2( Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ), Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ) );
            const Line2d line( point1, point2 );
        }
        return true;
    }
    
    bool parallelLine()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const Point2d point1( Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ), Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ) );
            const Point2d point2( Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ), Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ) );
            const Line2d line1( point1, point2 );
            
            const Point2d offset( Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ), Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ) );
            const bool inverse = ( (i % 2) == 0 );
            const Line2d line2( (inverse ? point1 : point2) + offset, (inverse ? point2 : point1) + offset );
            if ( !line1.isParallel( line2 ) )
                return false;
        }
        return true;
    }
    
    bool lineIntersection()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const Point2d point1( Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ), Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ) );
            const Point2d point2( Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ), Unit_Test::randomFloatValue<double>( -1000, 1000, 0.01 ) );
            const Line2d line1( point1, point2 );
            
            if ( point1 == point2 )
                continue;
            
            const int xCoeff = fabs(point1.x - point2.x) > fabs(point1.y - point2.y) ? -1 : 1;
            const int yCoeff = xCoeff * -1;
            const Line2d line2( Point2d( xCoeff * point1.x, yCoeff * point1.y ), Point2d( xCoeff * point2.x, yCoeff * point2.y ) );
            if ( !line1.isIntersect( line2 ) )
                return false;
        }
        return true;
    }
}

void addTests_Math( UnitTestFramework & framework )
{
    framework.add(pvmath::houghTransform_double, "math::Hough Transform (double)");
    framework.add(pvmath::houghTransform_float, "math::Hough Transform (float)");
    #ifdef PENGUINV_AVX_SET
    simd::EnableSimd( false );
    simd::EnableAvx( true );
    framework.add(pvmath::houghTransformSimd_double, "math::Hough Transform AVX (double)");
    framework.add(pvmath::houghTransformSimd_float, "math::Hough Transform AVX (float)");
    #endif
    framework.add(pvmath::lineConstructor, "math::Line2d constructor");
    framework.add(pvmath::parallelLine, "math::Line2d parallel lines");
    framework.add(pvmath::lineIntersection, "math::Line2d line intersection");
}
