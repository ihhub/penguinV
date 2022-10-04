/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2022                                             *
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

#include "../image_buffer.h"
#include "../math/math_base.h"
#include <vector>

// Input image for analysis should not contain pixel intensity values 0 or 255
// If an image contains pixels with such values they would be ignored during correction
class LightCorrection
{
public:
    void analyze( const penguinV::Image & image );
    void correct( penguinV::Image & image ) const;
    void correct( penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height ) const;

    // Returns an array of pixel coordinates on which pixel intensity equal to 0 or 255
    std::vector<PointBase2D<uint32_t>> findIncorrectPixels( const penguinV::Image & image ) const;

private:
    std::vector<uint32_t> _data;
    uint32_t _width;
    uint32_t _height;
};
