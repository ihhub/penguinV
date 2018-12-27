#pragma once

#include <vector>
#include "math_base.h"

namespace Image_Function
{
    bool HoughTransform( const std::vector< PointBase2D<double> > & input, double initialAngle, double angleTolerance, double angleStep,
                         double lineTolerance, std::vector< PointBase2D<double> > & outOnLine, std::vector< PointBase2D<double> > & outOffLine );

    bool HoughTransform( const std::vector< PointBase2D<float> > & input, float initialAngle, float angleTolerance, float angleStep,
                         float lineTolerance, std::vector< PointBase2D<float> > & outOnLine, std::vector< PointBase2D<float> > & outOffLine );
}
