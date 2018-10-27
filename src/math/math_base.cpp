#include "math_base.h"

namespace pvmath
{
    template <>
    bool isEqual<double>( const double & value1, const double & value2 )
    {
        return abs( value1 - value2 ) < epsilonDouble;
    }

    template <>
    bool isEqual<float>( const float & value1, const float & value2 )
    {
        return fabs( value1 - value2 ) < epsilonFloat;
    }

    double toRadians(double angleDegree)
    {
        return angleDegree * pi / 180;
    }

    double toDegrees(double angleRadians)
    {
        return angleRadians * 180 / pi;
    }
}
