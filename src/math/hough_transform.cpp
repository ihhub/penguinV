#include "hough_transform.h"

#include <algorithm>

namespace
{
    const float minimumAngleStep = 0.001 * pvmath::pi / 180.0;
    const float minimumLineTolerance = 1e-5;
}

namespace
{
    template<>
    bool runHoughTransform<double>( const std::vector< PointBase2D<double> > & input, double initialAngle, double angleTolerance, double angleStep,
                                    double lineTolerance, std::vector< PointBase2D<double> > & outOnLine, std::vector< PointBase2D<double> > & outOffLine )
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
            const PointBase2D<double> * point = input.data();
            const PointBase2D<double> * pointEnd = point + inputPointCount;

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
        const PointBase2D<double> * point = input.data();
        const PointBase2D<double> * pointEnd = point + inputPointCount;

        for ( ; point != pointEnd; ++point, ++distanceVal ) {
            (*distanceVal) = point->x * sinVal + point->y * cosVal;

            if ( ((*distanceVal) < minDistance) || ((*distanceVal) > maxDistance) )
                outOffLine.push_back( (*point) );
            else
                outOnLine.push_back( (*point) );
        }

        return true;
    }

    template<>
    bool runHoughTransform<float>( const std::vector< PointBase2D<float> > & input, float initialAngle, float angleTolerance, float angleStep,
                                   float lineTolerance, std::vector< PointBase2D<float> > & outOnLine, std::vector< PointBase2D<float> > & outOffLine )
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
        const float lineToleranceRange = lineTolerance * 2;

        const size_t inputPointCount = input.size();
        std::vector < float > distanceToLine ( inputPointCount );

        int bestAngleId = -angleStepPerSide;
        size_t highestPointCount = 0u;
        float averageDistance = 0;

        float angleVal = -(initialAngle - angleStep * angleStepPerSide); // this should be an opposite angle

        for ( int angleId = -angleStepPerSide; angleId <= angleStepPerSide; ++angleId, angleVal -= angleStep ) {
            const float cosVal = cos( angleVal );
            const float sinVal = sin( angleVal );

            // find and sort distances
            float * distanceVal = distanceToLine.data();
            const PointBase2D<float> * point = input.data();
            const PointBase2D<float> * pointEnd = point + inputPointCount;

            for ( ; point != pointEnd; ++point, ++distanceVal )
                (*distanceVal) = point->x * sinVal + point->y * cosVal;

            std::sort( distanceToLine.begin(), distanceToLine.end() );

            // find maximum number of points
            size_t initialPointId = 0u;
            size_t onLinePointCount = 1u;

            for ( size_t pointId = 0u, endPointId = 1u; endPointId < inputPointCount; ++pointId ) {
                const float tolerance = lineToleranceRange + distanceToLine[pointId];

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

        const float minDistance = averageDistance - lineTolerance;
        const float maxDistance = averageDistance + lineTolerance;

        // sort points
        const float cosVal = cos( angleVal );
        const float sinVal = sin( angleVal );

        float * distanceVal = distanceToLine.data();
        const PointBase2D<float> * point = input.data();
        const PointBase2D<float> * pointEnd = point + inputPointCount;

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
        runHoughTransform<double>(input, initialAngle, angleTolerance, angleStep, lineTolerance, outOnLine, outOffLine);
    }

    bool HoughTransform( const std::vector< PointBase2D<float> > & input, float initialAngle, float angleTolerance, float angleStep,
                         float lineTolerance, std::vector< PointBase2D<float> > & outOnLine, std::vector< PointBase2D<float> > & outOffLine )
    {
        runHoughTransform<float>(input, initialAngle, angleTolerance, angleStep, lineTolerance, outOnLine, outOffLine);
    }
}
