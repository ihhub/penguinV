#include "hough_transform.h"

#include <algorithm>

namespace
{
    const double minimumAngleStep = 0.001 * pvmath::pi / 180.0;
    const double minimumLineTolerance = 1e-5;
}

namespace Image_Function
{
    bool HoughTransform( const std::vector< Point2d > & input, double initialAngle, double angleTolerance, double angleStep,
                         double lineTolerance, std::vector< Point2d > & outOnLine, std::vector< Point2d > & outOffLine )
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
        const double lineToleranceRange = lineTolerance * 2;

        const size_t inputPointCount = input.size();
        std::vector < double > distanceToLine ( inputPointCount );

        int bestAngleId = -angleStepPerSide;
        size_t highestPointCount = 0u;
        double averageDistance = 0;

        double angleVal = -(initialAngle - angleStep * angleStepPerSide); // this should be an opposite angle

        for ( int angleId = -angleStepPerSide; angleId <= angleStepPerSide; ++angleId, angleVal -= angleStep ) {
            const double cosVal = cos( angleVal );
            const double sinVal = sin( angleVal );

            // find and sort distances
            double * distanceVal = distanceToLine.data();
            const Point2d * point = input.data();
            const Point2d * pointEnd = point + inputPointCount;

            for ( ; point != pointEnd; ++point, ++distanceVal )
                (*distanceVal) = point->x * sinVal + point->y * cosVal;

            std::sort( distanceToLine.begin(), distanceToLine.end() );

            // find maximum number of points
            size_t initialPointId = 0u;
            size_t onLinePointCount = 1u;

            for ( size_t pointId = 0u, endPointId = 1u; endPointId < inputPointCount; ++pointId ) {
                const double tolerance = lineToleranceRange + distanceToLine[pointId];

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

        const double minDistance = averageDistance - lineTolerance;
        const double maxDistance = averageDistance + lineTolerance;

        // sort points
        const double cosVal = cos( angleVal );
        const double sinVal = sin( angleVal );

        double * distanceVal = distanceToLine.data();
        const Point2d * point = input.data();
        const Point2d * pointEnd = point + inputPointCount;

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
