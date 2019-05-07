#include "math_base.h"

namespace pvmath
{
    template <>
    bool isEqual<double>( const double & value1, const double & value2 )
    {
        return std::abs( value1 - value2 ) < epsilonDouble;
    }

    template <>
    bool isEqual<float>( const float & value1, const float & value2 )
    {
        return std::fabs( value1 - value2 ) < epsilonFloat;
    }

    template <>
    bool isEqual<double>( const double & value1, const double & value2, const double epsilonMultiplier )
    {
        return std::abs( value1 - value2 ) < epsilonDouble * epsilonMultiplier;
    }

    template <>
    bool isEqual<float>( const float & value1, const float & value2, const float epsilonMultiplier )
    {
        return std::fabs( value1 - value2 ) < epsilonFloat * epsilonMultiplier;
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
