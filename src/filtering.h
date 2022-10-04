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

#include "image_buffer.h"
#include <vector>

namespace Image_Function
{
    using namespace penguinV;

    Image Median( const Image & in, uint32_t kernelSize );
    void Median( const Image & in, Image & out, uint32_t kernelSize );
    Image Median( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint32_t kernelSize );
    void Median( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height,
                 uint32_t kernelSize );

    // This filter returns image based on gradient magnitude in both X and Y directions
    Image Prewitt( const Image & in );
    void Prewitt( const Image & in, Image & out );
    Image Prewitt( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
    void Prewitt( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    // This filter returns image based on gradient magnitude in both X and Y directions
    Image Sobel( const Image & in );
    void Sobel( const Image & in, Image & out );
    Image Sobel( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
    void Sobel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

    void GetGaussianKernel( std::vector<float> & filter, uint32_t width, uint32_t height, uint32_t kernelSize, float sigma );
}
