#include "hough_transform.h"

#include <algorithm>

namespace
{
    const float minimumAngleStep = 0.001f * static_cast<float>( pvmath::pi ) / 180.0f;
    const float minimumLineTolerance = 1e-5f;
}

namespace
{

    template <typename _Type>
    void FindDistance( const std::vector< PointBase2D< _Type > > & input, std::vector < _Type > & distance, _Type cosVal, _Type sinVal )
    {
        _Type * distanceVal = distanceToLine.data();
        const PointBase2D<_Type> * point = input.data();
        const PointBase2D<_Type> * pointEnd = point + inputPointCount;

        for ( ; point != pointEnd; ++point, ++distanceVal )
            (*distanceVal) = point->x * sinVal + point->y * cosVal;
    }

    template <typename _Type>
    void FindDistanceSimd( const std::vector< PointBase2D< _Type > > & input, std::vector < _Type > & distance, _Type cosVal, _Type sinVal )
    {
        FindDistance( input, distance, cosVal, sinVal );
    }

    template <>
    void FindDistanceSimd< float >( const std::vector< PointBase2D< float > > & input, std::vector < float > & distance, float cosVal, float sinVal )
    {
        // some special code here to handle SIMD like
        if ( simd::avx_function ) {
        }
        else if ( simd::sse_function ) {
        }
        else if ( simd::neon_function )
        {
        }
        else {
            FindDistance( input, distance, cosVal, sinVal );
        }
    }

    template <>
    void FindDistanceSimd< double >( const std::vector< PointBase2D< double > > & input, std::vector < double> & distance, double cosVal, double sinVal )
    {
        // some special code here to handle SIMD like
        if ( simd::avx_function ) {
        }
        else if ( simd::sse_function ) {
        }
        else if ( simd::neon_function )
        {
        }
        else {
            FindDistance( input, distance, cosVal, sinVal );
        }
    }

    template <typename _Type>
    bool runHoughTransform( const std::vector< PointBase2D<_Type> > & input, _Type initialAngle, _Type angleTolerance, _Type angleStep,
                            _Type lineTolerance, std::vector< PointBase2D<_Type> > & outOnLine, std::vector< PointBase2D<_Type> > & outOffLine )
    {
        // validate input data
        if ( input.size() < 2u )
            return false;

        if ( angleStep < minimumAngleStep )
            angleStep = minimumAngleStep;

        if ( angleTolerance < minimumAngleStep )
            angleTolerance = minimumAngleStep;

        if ( angleTolerance < angleStep )
            angleTolerance = angleStep;

        if ( lineTolerance < minimumLineTolerance )
            lineTolerance = minimumLineTolerance;

        // find a range of search
        const int angleStepPerSide = static_cast<int>((angleTolerance / angleStep) + 0.5);
        const _Type lineToleranceRange = lineTolerance * 2;

        const size_t inputPointCount = input.size();
        std::vector < _Type > distanceToLine ( inputPointCount );

        int bestAngleId = -angleStepPerSide;
        size_t highestPointCount = 0u;
        _Type averageDistance = 0;

        _Type angleVal = -(initialAngle - angleStep * angleStepPerSide); // this should be an opposite angle

        for ( int angleId = -angleStepPerSide; angleId <= angleStepPerSide; ++angleId, angleVal -= angleStep ) {
            const _Type cosVal = cos( angleVal );
            const _Type sinVal = sin( angleVal );

            // find and sort distances
            _Type * distanceVal = distanceToLine.data();
            const PointBase2D<_Type> * point = input.data();
            const PointBase2D<_Type> * pointEnd = point + inputPointCount;

            for ( ; point != pointEnd; ++point, ++distanceVal )
                (*distanceVal) = point->x * sinVal + point->y * cosVal;

            std::sort( distanceToLine.begin(), distanceToLine.end() );

            // find maximum number of points
            size_t initialPointId = 0u;
            size_t onLinePointCount = 1u;

            for ( size_t pointId = 0u, endPointId = 1u; endPointId < inputPointCount; ++pointId ) {
                const _Type tolerance = lineToleranceRange + distanceToLine[pointId];

                for ( ; endPointId < inputPointCount; ++endPointId ) {
                    if ( tolerance < distanceToLine[endPointId] )
                        break;
                }

                if ( onLinePointCount < endPointId - pointId ) {
                    onLinePointCount = endPointId - pointId;
                    initialPointId = pointId;
                }
            }

            if ( highestPointCount <= onLinePointCount ) {
                const _Type currentDistance = (distanceToLine[initialPointId + onLinePointCount - 1u] + distanceToLine[initialPointId]) / 2;
                if ( highestPointCount < onLinePointCount || std::abs( currentDistance ) < std::abs( averageDistance ) ) {
                    highestPointCount = onLinePointCount;
                    bestAngleId = angleId;
                    averageDistance = currentDistance;
                }
            }
        }

        outOnLine.clear();
        outOffLine.clear();

        angleVal = -(initialAngle + angleStep * bestAngleId);

        const _Type minDistance = averageDistance - lineTolerance;
        const _Type maxDistance = averageDistance + lineTolerance;

        // sort points
        const _Type cosVal = cos( angleVal );
        const _Type sinVal = sin( angleVal );

        _Type * distanceVal = distanceToLine.data();
        const PointBase2D<_Type> * point = input.data();
        const PointBase2D<_Type> * pointEnd = point + inputPointCount;

        for ( ; point != pointEnd; ++point, ++distanceVal ) {
            (*distanceVal) = point->x * sinVal + point->y * cosVal;

            if ( ((*distanceVal) < minDistance) || ((*distanceVal) > maxDistance) )
                outOffLine.push_back( (*point) );
            else
                outOnLine.push_back( (*point) );
        }

        return true;
    }
}

namespace Image_Function_Simd
{
    bool HoughTransform( const std::vector< PointBase2D<double> > & input, double initialAngle, double angleTolerance, double angleStep,
                         double lineTolerance, std::vector< PointBase2D<double> > & outOnLine, std::vector< PointBase2D<double> > & outOffLine )
    {
        return runHoughTransform<double>(input, initialAngle, angleTolerance, angleStep, lineTolerance, outOnLine, outOffLine);
    }

    bool HoughTransform( const std::vector< PointBase2D<float> > & input, float initialAngle, float angleTolerance, float angleStep,
                         float lineTolerance, std::vector< PointBase2D<float> > & outOnLine, std::vector< PointBase2D<float> > & outOffLine )
    {
        return runHoughTransform<float>(input, initialAngle, angleTolerance, angleStep, lineTolerance, outOnLine, outOffLine);
    }
}