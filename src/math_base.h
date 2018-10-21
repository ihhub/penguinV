#pragma once

#include <cmath>

template <typename _Type>
struct PointBase2D
{
    PointBase2D( _Type _x = 0, _Type _y = 0 )
        : x( _x )
        , y( _y )
    { }

    virtual bool operator == ( const PointBase2D & point ) const
    {
        return (x == point.x) && (y == point.y);
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

    virtual bool operator == ( const PointBase3D & point ) const
    {
        return PointBase2D<_Type>::operator==( point ) && (z == point.z);
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

namespace
{
    const double epsilonDouble = 1e-10;
}

template<>
bool PointBase2D<double>::operator == ( const PointBase2D<double> & point ) const
{
    return fabs(x - point.x) < epsilonDouble && fabs(y - point.y) < epsilonDouble;
}

template<>
bool PointBase3D<double>::operator == ( const PointBase3D<double> & point ) const
{
    return fabs(x - point.x) < epsilonDouble && fabs(y - point.y) < epsilonDouble && fabs(z - point.z) < epsilonDouble;
}

typedef PointBase2D<double> Point2d;
typedef PointBase3D<double> Point3d;
