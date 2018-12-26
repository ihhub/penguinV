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

template <typename _Type>
class LineBase2D
{
public:
    LineBase2D( const PointBase2D<_Type> & point1 = PointBase2D<_Type>(), const PointBase2D<_Type> & point2 = PointBase2D<_Type>() )
        : _position( point1 )
    {
        if ( point1 == point2 ) {
            _direction = PointBase2D<_Type>( 1, 0 ); // we could raise an exception here instead
        }
        else {
            const _Type xDiff = point2.x - point1.x;
            const _Type yDiff = point2.y - point1.y;
            const _Type length = std::sqrt( xDiff * xDiff + yDiff * yDiff ); // here we might need more specific code for non-double cases
            _direction = PointBase2D<_Type>( xDiff / length, yDiff / length );
        }
    }

    // Angle is in radians
    LineBase2D( const PointBase2D<_Type> & position_, _Type angle_ )
        : _position( position_ )
        , _direction( std::cos(angle_), std::sin(angle_) )
    {
    }

    bool operator == ( const LineBase2D & line ) const
    {
        return parallel( line ) && pvmath::isEqual<_Type>( distance(line._position), 0 );
    }

    // This is translation (shift) function
    LineBase2D operator + ( const PointBase2D<_Type> & offset )
    {
        return LineBase2D( _position + offset, angle() );
    }

    LineBase2D & operator += ( const PointBase2D<_Type> & offset )
    {
        _position += offset;
        return *this;
    }

    _Type angle() const
    {
        return std::atan2( _direction.y, _direction.x );
    }

    PointBase2D<_Type> position() const
    {
        return _position;
    }

    bool intersection( const LineBase2D & line, PointBase2D<_Type> & point ) const
    {
        // based on Graphics Gems III, Faster Line Segment Intersection, p. 199-202
        // http://www.realtimerendering.com/resources/GraphicsGems/gems.html#gemsiii
        const _Type denominator = _direction.y * line._direction.x - _direction.x * line._direction.y;
        if (pvmath::isEqual<_Type>(denominator, 0))
            return false; // they are parallel

        const PointBase2D<_Type> offset = _position - line._position;
        const _Type na = (line._direction.y * offset.x - line._direction.x * offset.y) / denominator;
        point = _position + _direction * na;
        return true;
    }

    bool isParallel( const LineBase2D & line )
    {
        const _Type denominator = _direction.y * line._direction.x - _direction.x * line._direction.y;
        return pvmath::isEqual<_Type>(denominator, 0);
    }

    bool isIntersect( const LineBase2D & line ) const
    {
        return !parallel( line );
    }
    
    _Type distance( const PointBase2D<_Type> & point ) const
    {
        // Line equation in the Cartesian coordinate system is
        // y = a * x + b or A * x + B * y + C = 0
        // A distance from a point to a line can be calculated as:
        // |A * x0 + B * y0 + C| / sqrt(A * A + B * B)
        const _Type distanceToLine = _direction.y * (point.x - _position.x) + _direction.x * (_position.y - point.y);
        return (distanceToLine < 0 ? -distanceToLine : distanceToLine);
    }
private:
    PointBase2D<_Type> _position;
    PointBase2D<_Type> _direction;
};

typedef PointBase2D<double> Point2d;
typedef PointBase3D<double> Point3d;
typedef LineBase2D<double> Line2d;
