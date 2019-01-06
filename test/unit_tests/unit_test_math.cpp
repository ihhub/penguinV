#include "unit_test_math.h"

#include <cmath>
#include "unit_test_helper.h"
#include "../../src/math/hough_transform.h"

namespace pvmath
{
    template <typename _Type>
    bool houghTransformTemplate()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const _Type angle = static_cast<_Type>( toRadians( Unit_Test::randomFloatValue<_Type>(-180, 180, 1 ) ) );
            const _Type angleTolerance = static_cast<_Type>( toRadians( Unit_Test::randomFloatValue<_Type>( 0, 10, 0.1f ) + 0.1f ) );
            const _Type angleStep = angleTolerance / Unit_Test::randomValue( 1, 50 );
            const _Type lineTolerance = Unit_Test::randomFloatValue<_Type>( 0.1f, 5, 0.01f );

            const _Type noiseValue = lineTolerance / 2;
            std::vector< PointBase2D<_Type> > point( Unit_Test::randomValue<uint32_t>( 50u, 100u ) );

            const _Type sinVal = std::sin( angle );
            const _Type cosVal = std::cos( angle );

            for ( typename std::vector< PointBase2D<_Type> >::iterator p = point.begin(); p != point.end(); ++p ) {
                const _Type x = Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ) + Unit_Test::randomFloatValue<_Type>( -noiseValue, noiseValue, noiseValue / 10 );
                const _Type y = Unit_Test::randomFloatValue<_Type>( -noiseValue, noiseValue, noiseValue / 10 );

                p->x = x * cosVal - y * sinVal;
                p->y = x * sinVal + y * cosVal;
            }

            std::vector< PointBase2D<_Type> > pointOnLine;
            std::vector< PointBase2D<_Type> > pointOffLine;

            if ( !Image_Function::HoughTransform( point, angle, angleTolerance, angleStep, lineTolerance, pointOnLine, pointOffLine ) ||
                 !pointOffLine.empty() )
                return false;
        }

        return true;
    }

    bool houghTransform_double()
    {
        return houghTransformTemplate<double>();
    }

    bool houghTransform_float()
    {
        return houghTransformTemplate<float>();
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

            const Line2d line2( Point2d( -point1.y, point1.x ), Point2d( -point2.y, point2.x ) );
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
    framework.add(pvmath::lineConstructor, "math::Line2d constructor");
    framework.add(pvmath::parallelLine, "math::Line2d parallel lines");
    framework.add(pvmath::lineIntersection, "math::Line2d line intersection");
}
