#pragma once

#include <vector>
#include "image_buffer.h"
#include "math/math_base.h"

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

    EdgeParameter( directionType _direction = LEFT_TO_RIGHT, gradientType _gradient = ANY, edgeType _edge = ALL,
                   uint32_t _groupFactor = 1u, uint32_t _skipFactor = 1u, uint32_t _contrastCheckLeftSideOffset = 0u,
                   uint32_t _contrastCheckRightSideOffset = 0u, uint8_t _minimumContrast = 10 );

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

class EdgeDetection
{
public:
    void find( const PenguinV_Image::Image & image, const EdgeParameter & edgeParameter = EdgeParameter() );
    void find( const PenguinV_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const EdgeParameter & edgeParameter = EdgeParameter() );

    const std::vector < Point2d > & positiveEdge() const;
    const std::vector < Point2d > & negativeEdge() const;

private:
    std::vector < Point2d > positiveEdgePoint;
    std::vector < Point2d > negativeEdgePoint;

    void findEdgePoints( std::vector < double > & positive, std::vector < double > & negative, std::vector < int > & data,
                         std::vector < int > & first, std::vector < int > & second, const EdgeParameter & edgeParameter, bool forwardDirection );

    void getDerivatives( const std::vector < int > & image, std::vector < int > & first, std::vector < int > & second ) const;
    void getEdgePoints( std::vector < double > & edge, const std::vector < int > & image, const std::vector < int > & first, const std::vector < int > & second,
                        const EdgeParameter & edgeParameter ) const;
    void removeSimilarPoints( std::vector < double > & edge ) const;
};
