#include "unit_test_math.h"

#include "../test_helper.h"
#include "../../src/math/hough_transform.h"

namespace pvmath
{
    template <typename _Type>
    bool houghTransformTemplate()
    {
        for( uint32_t i = 0; i < Test_Helper::runCount(); ++i ) {
            const _Type angle = static_cast<_Type>( toRadians( Test_Helper::randomFloatValue<_Type>(-180, 180, 1 ) ) );
            const _Type angleTolerance = static_cast<_Type>( toRadians( Test_Helper::randomFloatValue<_Type>( 0, 10, 0.1f ) + 0.1f ) );
            const _Type angleStep = angleTolerance / Test_Helper::randomValue( 1, 50 );
            const _Type lineTolerance = Test_Helper::randomFloatValue<_Type>( 0.1f, 5, 0.01f );

            const _Type noiseValue = lineTolerance / 2;
            std::vector< PointBase2D<_Type> > point( Test_Helper::randomValue<uint32_t>( 50u, 100u ) );

            const _Type sinVal = sin( angle );
            const _Type cosVal = cos( angle );

            for ( typename std::vector< PointBase2D<_Type> >::iterator p = point.begin(); p != point.end(); ++p ) {
                const _Type x = Test_Helper::randomFloatValue<_Type>( -1000, 1000, 0.01f ) + Test_Helper::randomFloatValue<_Type>( -noiseValue, noiseValue, noiseValue / 10 );
                const _Type y = Test_Helper::randomFloatValue<_Type>( -noiseValue, noiseValue, noiseValue / 10 );

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
        for( uint32_t i = 0; i < Test_Helper::runCount(); ++i ) {
            const Point2d point1( Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ), Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ) );
            const Point2d point2( Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ), Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ) );
            const Line2d line( point1, point2 );
        }
        return true;
    }
    
    bool parallelLine()
    {
        for( uint32_t i = 0; i < Test_Helper::runCount(); ++i ) {
            const Point2d point1( Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ), Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ) );
            const Point2d point2( Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ), Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ) );
            const Line2d line1( point1, point2 );
            
            const Point2d offset( Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ), Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ) );
            const bool inverse = ( (i % 2) == 0 );
            const Line2d line2( (inverse ? point1 : point2) + offset, (inverse ? point2 : point1) + offset );
            if ( !line1.isParallel( line2 ) )
                return false;
        }
        return true;
    }
    
    bool lineIntersection()
    {
        for( uint32_t i = 0; i < Test_Helper::runCount(); ++i ) {
            const Point2d point1( Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ), Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ) );
            const Point2d point2( Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ), Test_Helper::randomFloatValue<double>( -1000, 1000, 0.01 ) );
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
    framework.add(pvmath::lineConstructor, "math::Line2d constructor");
    framework.add(pvmath::parallelLine, "math::Line2d parallel lines");
    framework.add(pvmath::lineIntersection, "math::Line2d line intersection");
}
