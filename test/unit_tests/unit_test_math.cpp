#include "unit_test_math.h"

#include <cmath>
#include "unit_test_helper.h"
#include "../../src/math/hough_transform.h"
#include "../../src/math/haar_transform.h"

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

    template <typename _Type>
    bool haarTransformTemplate()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const uint32_t width  = Unit_Test::randomValue<uint32_t>( 16u, 256u ) * 2; // to make sure that number is divided by 2
            const uint32_t height = Unit_Test::randomValue<uint32_t>( 16u, 256u ) * 2;
            std::vector< _Type > input ( width * height );
            std::vector< _Type > direct( width * height );

            for ( size_t id = 0; id < input.size(); ++id ) {
                input [id] = Unit_Test::randomFloatValue<_Type>( 0, 255, 1.0f );
                direct[id] = Unit_Test::randomFloatValue<_Type>( 0, 255, 1.0f );
            }

            std::vector< _Type > inverse ( width * height );
            Image_Function::HaarDirectTransform ( input, direct, width, height );
            Image_Function::HaarInverseTransform( direct, inverse, width, height );

            for ( size_t id = 0; id < input.size(); ++id ) {
                if (std::fabs(input[id] - inverse[id]) > 0.001f)
                    return false;
            }
        }

        return true;
    }

    bool houghTransformDouble()
    {
        return houghTransformTemplate<double>();
    }

    bool houghTransformFloat()
    {
        return houghTransformTemplate<float>();
    }

    bool haarTransformDouble()
    {
        return haarTransformTemplate<double>();
    }

    bool haarTransformFloat()
    {
        return haarTransformTemplate<float>();
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
    framework.add(pvmath::houghTransformDouble, "math::Hough Transform (double)");
    framework.add(pvmath::houghTransformFloat, "math::Hough Transform (float)");
    framework.add(pvmath::haarTransformDouble, "math::Haar Transform (double)");
    framework.add(pvmath::haarTransformFloat, "math::Haar Transform (float)");
    framework.add(pvmath::lineConstructor, "math::Line2d constructor");
    framework.add(pvmath::parallelLine, "math::Line2d parallel lines");
    framework.add(pvmath::lineIntersection, "math::Line2d line intersection");
}
