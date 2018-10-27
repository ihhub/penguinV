#pragma once

#include <vector>
#include "math_base.h"

namespace Image_Function
{
    // All input data for angles are in radians, line tolerance is in user-defined {Point2d} coordinate system units (usually pixels)
    bool HoughTransform( const std::vector< Point2d > & input, double initialAngle, double angleTolerance, double angleStep, double lineTolerance,
                         std::vector< Point2d > & outOnLine, std::vector< Point2d > & outOffLine );
}
