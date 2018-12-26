#include "unit_test_math.h"

#include "unit_test_helper.h"
#include "../../src/math/hough_transform.h"

namespace pvmath
{
    bool houghTransform()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const double angle = toRadians( Unit_Test::randomValue(-180, 180, 1 ) );
            const double angleTolerance = Unit_Test::randomValue(0, std::abs(angle / 2.0), 0.1 );
            const double angleStep = angleTolerance / Unit_Test::randomValue( 1, 50 );
            const double lineTolerance = Unit_Test::randomValue( 0.1, 5, 0.01 );

            const double noiseValue = lineTolerance / 2;
            std::vector< Point2d > point( Unit_Test::randomValue<uint32_t>( 50u, 100u ) );

            const double sinVal = sin( angle );
            const double cosVal = cos( angle );

            for ( std::vector< Point2d >::iterator p = point.begin(); p != point.end(); ++p ) {
                const double x = Unit_Test::randomValue( -1000, 1000, 0.01 ) + Unit_Test::randomValue( -noiseValue, noiseValue, 0.01 );
                const double y = Unit_Test::randomValue( -noiseValue, noiseValue, 0.01 );

                p->x = x * cosVal - y * sinVal;
                p->y = x * sinVal + y * cosVal;
            }

            std::vector< Point2d > pointOnLine;
            std::vector< Point2d > pointOffLine;

            if ( !Image_Function::HoughTransform( point, angle, angleTolerance, angleStep, lineTolerance, pointOnLine, pointOffLine ) ||
                 !pointOffLine.empty() )
                return false;
        }

        return true;
    }
    
    bool lineConstructor()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const Point2d point1( Unit_Test::randomValue( -1000, 1000, 0.01 ), Unit_Test::randomValue( -1000, 1000, 0.01 ) );
            const Point2d point2( Unit_Test::randomValue( -1000, 1000, 0.01 ), Unit_Test::randomValue( -1000, 1000, 0.01 ) );
            const Line2d line( point1, point2 );
        }
        return true;
    }
    
    bool parallelLine()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const Point2d point1( Unit_Test::randomValue( -1000, 1000, 0.01 ), Unit_Test::randomValue( -1000, 1000, 0.01 ) );
            const Point2d point2( Unit_Test::randomValue( -1000, 1000, 0.01 ), Unit_Test::randomValue( -1000, 1000, 0.01 ) );
            const Line2d line1( point1, point2 );
            
            const Point2d offset( Unit_Test::randomValue( -1000, 1000, 0.01 ), Unit_Test::randomValue( -1000, 1000, 0.01 ) );
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
            const Point2d point1( Unit_Test::randomValue( -1000, 1000, 0.01 ), Unit_Test::randomValue( -1000, 1000, 0.01 ) );
            const Point2d point2( Unit_Test::randomValue( -1000, 1000, 0.01 ), Unit_Test::randomValue( -1000, 1000, 0.01 ) );
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
    framework.add(pvmath::houghTransform, "math::Hough Transform");
    framework.add(pvmath::lineConstructor, "math::Line2d constructor");
    framework.add(pvmath::parallelLine, "math::Line2d parallel lines");
    framework.add(pvmath::lineIntersection, "math::Line2d line intersection");
}
