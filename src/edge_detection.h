/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2024                                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#pragma once

#include "image_buffer.h"
#include "math/math_base.h"
#include <vector>

struct EdgeParameter
{
    // Direction of scanning for edges
    enum class DirectionType : uint8_t
    {
        LEFT_TO_RIGHT,
        RIGHT_TO_LEFT,
        TOP_TO_BOTTOM,
        BOTTOM_TO_TOP
    };

    // Type of edge to detect based on gradient: positive means that pixel intensity increases while for negative it decreases. Any: positive and negative
    enum class GradientType : uint8_t
    {
        POSITIVE,
        NEGATIVE,
        ANY
    };

    // Type of edge to find per scanning row. First means first edge point in a row, last - last edge point. Set all to catch all edge points in a row
    enum class EdgeType : uint8_t
    {
        FIRST,
        LAST,
        ALL
    };

    EdgeParameter( DirectionType _direction = DirectionType::LEFT_TO_RIGHT, GradientType _gradient = GradientType::ANY, EdgeType _edge = EdgeType::ALL,
                   uint32_t _groupFactor = 1u, uint32_t _skipFactor = 1u, uint32_t _contrastCheckLeftSideOffset = 0u, uint32_t _contrastCheckRightSideOffset = 0u,
                   uint8_t _minimumContrast = 10 );

    DirectionType direction;

    GradientType gradient;

    EdgeType edge;

    // Grouping per row or column (depending on direction) works as a median filter. Default is 1 - no grouping.
    uint32_t groupFactor;

    // Skip specific number of rows or columns to do not find edge points on all rows/columns. Default is 1 - no skipping.
    uint32_t skipFactor;

    // Specify a number of pixels from each side of potential edge point to get pixel intensity needed for contrast verification
    // Such offsets are useful for very smooth edge when pixel intensity increases very slowly per pixel
    // Default values are 0
    uint32_t contrastCheckLeftSideOffset;
    uint32_t contrastCheckRightSideOffset;

    // Minimun contrast needed to detect edge.
    uint8_t minimumContrast;

    // Self-verification that all parameters are correct.
    void verify() const;
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

using EdgeDetection = EdgeDetectionBase<double>;
