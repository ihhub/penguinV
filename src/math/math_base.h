#pragma once

#include <cmath>

namespace pvmath
{
    const double pi = std::acos(-1);
    const double epsilonDouble = 1e-10;
    const double epsilonFloat  = 1e-5f;

    template <typename _Type>
    bool isEqual( const _Type & value1, const _Type & value2 )
    {
        return ( value1 == value2 );
    }

    template <>
    bool isEqual<double>( const double & value1, const double & value2 );

    template <>
    bool isEqual<float>( const float & value1, const float & value2 );

    double toRadians(double angleDegree);
    double toDegrees(double angleRadians);
}

template <typename _Type>
struct PointBase2D
{
    PointBase2D( _Type _x = 0, _Type _y = 0 )
        : x( _x )
        , y( _y )
    { }

    bool operator == ( const PointBase2D & point ) const
    {
        return pvmath::isEqual( x, point.x ) && pvmath::isEqual( y, point.y );
    }

    PointBase2D & operator += ( const PointBase2D & point )
    {
        x += point.x;
        y += point.y;
        return *this;
    }

    PointBase2D & operator -= ( const PointBase2D & point )
    {
        x -= point.x;
        y -= point.y;
        return *this;
    }

    PointBase2D operator + ( const PointBase2D & point ) const
    {
        return PointBase2D( x + point.x, y + point.y );
    }

    PointBase2D operator - ( const PointBase2D & point ) const
    {
        return PointBase2D( x - point.x, y - point.y );
    }

    _Type x;
    _Type y;
};

template <typename _Type>
struct PointBase3D : public PointBase2D<_Type>
{
    PointBase3D( _Type _x = 0, _Type _y = 0, _Type _z = 0 )
        : PointBase2D<_Type>( _x, _y )
        , z( _z )
    { }

    bool operator == ( const PointBase3D & point ) const
    {
        return PointBase2D<_Type>::operator==( point ) && pvmath::isEqual( z, point.z );
    }

    PointBase3D & operator += ( const PointBase3D & point )
    {
        PointBase2D<_Type>::operator+=( point );
        z += point.z;
        return *this;
    }

    PointBase3D & operator -= ( const PointBase3D & point )
    {
        PointBase2D<_Type>::operator-=( point );
        z -= point.z;
        return *this;
    }

    PointBase3D operator + ( const PointBase3D & point ) const
    {
        return PointBase3D( PointBase2D<_Type>::x + point.x, PointBase2D<_Type>::y + point.y, z + point.z );
    }

    PointBase3D operator - ( const PointBase3D & point ) const
    {
        return PointBase3D( PointBase2D<_Type>::x - point.x, PointBase2D<_Type>::y - point.y, z - point.z );
    }

    _Type z;
};

typedef PointBase2D<double> Point2d;
typedef PointBase3D<double> Point3d;
