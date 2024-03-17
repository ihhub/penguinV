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

#include "math_base.h"
#include <vector>

namespace Image_Function
{
    bool HoughTransform( const std::vector<PointBase2D<double>> & input, const double initialAngle, double angleTolerance, double angleStep, double lineTolerance,
                         std::vector<PointBase2D<double>> & outOnLine, std::vector<PointBase2D<double>> & outOffLine );

    bool HoughTransform( const std::vector<PointBase2D<float>> & input, const float initialAngle, float angleTolerance, float angleStep, float lineTolerance,
                         std::vector<PointBase2D<float>> & outOnLine, std::vector<PointBase2D<float>> & outOffLine );
}
