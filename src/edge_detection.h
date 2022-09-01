#pragma once

#include "image_buffer.h"
#include "math/math_base.h"
#include <vector>

struct EdgeParameter
{
    // Direction of scanning for edges
    enum directionType
    {
        LEFT_TO_RIGHT = 0,
        RIGHT_TO_LEFT,
        TOP_TO_BOTTOM,
        BOTTOM_TO_TOP
    };

    // Type of edge to detect based on gradient: positive means that pixel intensity increases while for negative it decreases. Any: positive and negative
    enum gradientType
    {
        POSITIVE = 4,
        NEGATIVE,
        ANY
    };

    // Type of edge to find per scanning row. First means first edge point in a row, last - last edge point. Set all to catch all edge points in a row
    enum edgeType
    {
        FIRST = 7,
        LAST,
        ALL
    };

    EdgeParameter( directionType _direction = LEFT_TO_RIGHT, gradientType _gradient = ANY, edgeType _edge = ALL, uint32_t _groupFactor = 1u, uint32_t _skipFactor = 1u,
                   uint32_t _contrastCheckLeftSideOffset = 0u, uint32_t _contrastCheckRightSideOffset = 0u, uint8_t _minimumContrast = 10 );

    directionType direction;
    gradientType gradient;
    edgeType edge;
    uint32_t groupFactor; // grouping per row or column (depending on direction) works as a median filter. Default is 1 - no grouping
    uint32_t skipFactor; // skip specific number of rows or columns to do not find edge points on all rows/columns. Default is 1 - no skipping
    // Specify a number of pixels from each side of potential edge point to get pixel intensity needed for contrast verification
    // Such offsets are useful for very smooth edge when pixel intensity increases very slowly per pixel
    // Default values are 0
    uint32_t contrastCheckLeftSideOffset;
    uint32_t contrastCheckRightSideOffset;
    uint8_t minimumContrast; // minimun contrast needed to detect edge

    void verify() const; // self-verification that all parameters are correct
};

template <typename _Type>
class EdgeDetectionBase;

class EdgeDetectionHelper
{
public:
    static void find( EdgeDetectionBase<double> & edgeDetection, const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                      const EdgeParameter & edgeParameter );

    static void find( EdgeDetectionBase<float> & edgeDetection, const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                      const EdgeParameter & edgeParameter );
};

template <typename _Type>
class EdgeDetectionBase
{
public:
    void find( const penguinV::Image & image, const EdgeParameter & edgeParameter = EdgeParameter() )
    {
        find( image, 0, 0, image.width(), image.height(), edgeParameter );
    }

    void find( const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const EdgeParameter & edgeParameter = EdgeParameter() )
    {
        positiveEdgePoint.clear();
        negativeEdgePoint.clear();
        EdgeDetectionHelper::find( *this, image, x, y, width, height, edgeParameter );
    }

    const std::vector<PointBase2D<_Type>> & positiveEdge() const
    {
        return positiveEdgePoint;
    }

    const std::vector<PointBase2D<_Type>> & negativeEdge() const
    {
        return negativeEdgePoint;
    }

    friend class EdgeDetectionHelper;

private:
    std::vector<PointBase2D<_Type>> positiveEdgePoint;
    std::vector<PointBase2D<_Type>> negativeEdgePoint;
};

typedef EdgeDetectionBase<double> EdgeDetection;
