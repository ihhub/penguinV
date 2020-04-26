#include "unit_test_math.h"

#include "../../src/math/hough_transform.h"
#include "../../src/math/haar_transform.h"
#include "unit_test_framework.h"
#include "unit_test_helper.h"
#include <cmath>

// Only for opposite() and projection() functions of Line2D class:
namespace pvmathHelper
{
    template <typename _Type>
    bool isEqual( const PointBase2D<_Type> & value1, const PointBase2D<_Type> & value2 )
    {
        return pvmath::isEqual( value1.x, value2.x ) && pvmath::isEqual( value1.y, value2.y );
    }

    // For 90 degree angle we have [4.37113883e-08, 1] values of directories instead of [0, 1].
    // Value 4.37113883e-08 is standard error for float variable.
    // The difference between possible generated points is 2000.
    // We multiply 2000 by 2 * 2 and by 4.37113883e-08 which gives us 0.0001748455532 * 2 --> 0.0004
    template <>
    bool isEqual<float>( const PointBase2D<float> & value1, const PointBase2D<float> & value2 )
    {
        return ( std::fabs( value1.x - value2.x ) < 0.0004 ) && ( std::fabs( value1.y - value2.y ) < 0.0004 );
    }
}

namespace pvmath
{
    template <typename _Type>
    bool houghTransform()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const _Type angle = static_cast<_Type>( toRadians( Unit_Test::randomFloatValue<_Type>(-180, 180, 1 ) ) );
            const _Type angleTolerance = static_cast<_Type>( toRadians( Unit_Test::randomFloatValue<_Type>( 0, 10, 0.1f ) + 0.1f ) );
            const _Type angleStep = angleTolerance / static_cast<_Type>( Unit_Test::randomValue( 10, 50 ) );
            const _Type lineTolerance = Unit_Test::randomFloatValue<_Type>( 0.1f, 5, 0.01f );

            std::vector< PointBase2D<_Type> > point( Unit_Test::randomValue<uint32_t>( 50u, 100u ) );
            const _Type noiseValue = lineTolerance / static_cast<_Type>( 3 * 100 * point.size() );

            const _Type sinVal = std::sin( angle );
            const _Type cosVal = std::cos( angle );

            for ( typename std::vector< PointBase2D<_Type> >::iterator p = point.begin(); p != point.end(); ++p ) {
                const _Type x = Unit_Test::randomFloatValue<_Type>( -100, 100, 0.01f ) + Unit_Test::randomFloatValue<_Type>( -noiseValue, noiseValue, noiseValue / 10 );
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
    bool haarTransform()
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

    template <typename _Type>
    bool lineConstructor()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const PointBase2D<_Type> point1( Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ), Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ) );
            const PointBase2D<_Type> point2( Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ), Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ) );
            const LineBase2D<_Type> line( point1, point2 );
        }
        return true;
    }

    template <typename _Type>
    bool parallelLine()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const PointBase2D<_Type> point1( Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ), Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ) );
            const PointBase2D<_Type> point2( Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ), Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ) );
            const LineBase2D<_Type> line1( point1, point2 );

            const PointBase2D<_Type> offset( Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ), Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ) );
            const bool inverse = ( (i % 2) == 0 );
            const LineBase2D<_Type> line2( (inverse ? point1 : point2) + offset, (inverse ? point2 : point1) + offset );
            if ( !line1.isParallel( line2 ) )
                return false;
        }
        return true;
    }

    template <typename _Type>
    bool lineIntersection()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const PointBase2D<_Type> point1( Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ), Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ) );
            const PointBase2D<_Type> point2( Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ), Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ) );
            const LineBase2D<_Type> line1( point1, point2 );

            if ( point1 == point2 )
                continue;

            const LineBase2D<_Type> line2( PointBase2D<_Type>( -point1.y, point1.x ), PointBase2D<_Type>( -point2.y, point2.x ) );
            PointBase2D<_Type> intersectPoint;
            if ( !line1.isIntersect( line2 ) || !line1.intersection( line2, intersectPoint ) )
                return false;
        }
        return true;
    }

    template <typename _Type>
    bool pointProjection()
    {
        for ( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const PointBase2D<_Type> testPoint( Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ), Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ) );
            const PointBase2D<_Type> pointBase( Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ), Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ) );

            const LineBase2D<_Type> lineX( pointBase, 0 );
            const LineBase2D<_Type> lineY( pointBase, static_cast<_Type>( pvmath::pi / 2 ) );

            const PointBase2D<_Type> resultPointX( testPoint.x, pointBase.y );
            const PointBase2D<_Type> resultPointY( pointBase.x, testPoint.y );

            if ( !pvmathHelper::isEqual( lineX.projection( testPoint ), resultPointX ) || !pvmathHelper::isEqual( lineY.projection( testPoint ), resultPointY ) )
                return false;
        }
        return true;
    }

    template <typename _Type>
    bool pointOpposition()
    {
        for ( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            const PointBase2D<_Type> testPoint( Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ), Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ) );
            const PointBase2D<_Type> pointBase( Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ), Unit_Test::randomFloatValue<_Type>( -1000, 1000, 0.01f ) );

            const LineBase2D<_Type> lineX( pointBase, 0 );
            const LineBase2D<_Type> lineY( pointBase, static_cast<_Type>( pvmath::pi / 2 ) );

            const PointBase2D<_Type> resultPointX( testPoint.x, pointBase.y * 2 - testPoint.y );
            const PointBase2D<_Type> resultPointY( pointBase.x * 2 - testPoint.x, testPoint.y );

            if ( !pvmathHelper::isEqual( lineX.opposite( testPoint ), resultPointX ) || !pvmathHelper::isEqual( lineY.opposite( testPoint ), resultPointY ) )
                return false;
        }
        return true;
    }
}

void addTests_Math( UnitTestFramework & framework )
{
    framework.add( pvmath::houghTransform<double>, "math::Hough Transform (double)" );
    framework.add( pvmath::houghTransform<float>, "math::Hough Transform (float)" );
    framework.add( pvmath::haarTransform<double>, "math::Haar Transform (double)" );
    framework.add( pvmath::haarTransform<float>, "math::Haar Transform (float)" );
    framework.add( pvmath::lineConstructor<double>, "math::Line2d constructor (double)" );
    framework.add( pvmath::lineConstructor<float>, "math::Line2d constructor (float)" );
    framework.add( pvmath::parallelLine<double>, "math::Line2d parallel lines (double)" );
    framework.add( pvmath::parallelLine<float>, "math::Line2d parallel lines (float)" );
    framework.add( pvmath::lineIntersection<double>, "math::Line2d line intersection (double)" );
    framework.add( pvmath::lineIntersection<float>, "math::Line2d line intersection (float)" );
    framework.add( pvmath::pointProjection<double>, "math::Line2d point projection (double)" );
    framework.add( pvmath::pointProjection<float>, "math::Line2d point projection (float)" );
    framework.add( pvmath::pointOpposition<double>, "math::Line2d point opposition (double)" );
    framework.add( pvmath::pointOpposition<float>, "math::Line2d point opposition (float)" );
}
