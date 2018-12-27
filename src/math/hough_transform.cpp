#include "hough_transform.h"

#include <algorithm>

namespace
{
    const float minimumAngleStep = 0.001 * static_cast<float>( pvmath::pi ) / 180.0;
    const float minimumLineTolerance = 1e-5f;
}

namespace
{
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

            if ( highestPointCount < onLinePointCount ) {
                highestPointCount = onLinePointCount;
                bestAngleId = angleId;
                averageDistance = (distanceToLine[initialPointId + onLinePointCount - 1u] + distanceToLine[initialPointId]) / 2;
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

namespace Image_Function
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
