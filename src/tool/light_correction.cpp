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

#include "light_correction.h"
#include "../parameter_validation.h"

void LightCorrection::analyze( const penguinV::Image & image )
{
    Image_Function::ValidateImageParameters( image );

    _width = image.width();
    _height = image.height();

    _data.resize( _width * _height );

    uint8_t minValue = 255u;
    uint8_t maxValue = 0u;

    const uint32_t rowSize = image.rowSize();
    const uint8_t * imageY = image.data();
    const uint8_t * imageYEnd = imageY + _height * rowSize;

    for ( ; imageY != imageYEnd; imageY += rowSize ) {
        const uint8_t * imageX = imageY;
        const uint8_t * imageXEnd = imageX + _width;

        for ( ; imageX != imageXEnd; ++imageX ) {
            if ( ( *imageX ) < minValue && ( *imageX ) > 0 )
                minValue = ( *imageX );

            if ( ( *imageX ) > maxValue && ( *imageX ) < 255u )
                maxValue = ( *imageX );
        }
    }

    if ( minValue >= maxValue )
        throw penguinVException( "Image in invalid. Minimum value cannot be greater or equal to maximum value." );

    imageY = image.data();
    imageYEnd = imageY + _height * rowSize;
    uint32_t * data = _data.data();

    for ( ; imageY != imageYEnd; imageY += rowSize ) {
        const uint8_t * imageX = imageY;
        const uint8_t * imageXEnd = imageX + _width;

        for ( ; imageX != imageXEnd; ++imageX, ++data ) {
            if ( ( *imageX ) > maxValue )
                *data = 4194304u;
            else if ( ( *imageX ) < minValue )
                *data = 4194304u;
            else
                *data = static_cast<unsigned int>( static_cast<double>( ( *imageX ) ) * 4194304u / maxValue ) + 1u;
        }
    }
}

void LightCorrection::correct( penguinV::Image & image ) const
{
    Image_Function::ValidateImageParameters( image );

    correct( image, 0, 0, image.width(), image.height() );
}

void LightCorrection::correct( penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height ) const
{
    Image_Function::ValidateImageParameters( image, x, y, width, height );
    if ( _width == 0 || _height == 0 )
        throw penguinVException( "Image is not being analyzed before calling correction. Analyze first." );

    const uint32_t rowSize = image.rowSize();

    uint8_t * imageY = image.data() + y * rowSize + x;
    const uint8_t * imageYEnd = imageY + height * rowSize;
    const uint32_t * dataY = _data.data() + x + y * _width;

    for ( ; imageY != imageYEnd; imageY += rowSize, dataY += _width ) {
        uint8_t * imageX = imageY;
        const uint8_t * imageXEnd = imageX + width;
        const uint32_t * dataX = dataY;

        for ( ; imageX != imageXEnd; ++imageX, ++dataX ) {
            const uint32_t value = ( ( ( *imageX ) << 22 ) + ( ( *dataX ) >> 1u ) ) / ( *dataX );

            if ( value > 255 )
                *imageX = 255;
            else
                *imageX = static_cast<uint8_t>( value );
        }
    }
}

std::vector<PointBase2D<uint32_t>> LightCorrection::findIncorrectPixels( const penguinV::Image & image ) const
{
    std::vector<PointBase2D<uint32_t>> point;

    const uint32_t rowSize = image.rowSize();
    const uint8_t * imageY = image.data();
    const uint8_t * imageYEnd = imageY + _height * rowSize;

    for ( uint32_t y = 0; imageY != imageYEnd; imageY += rowSize, ++y ) {
        const uint8_t * imageX = imageY;
        const uint8_t * imageXEnd = imageX + _width;

        for ( uint32_t x = 0; imageX != imageXEnd; ++imageX, ++x ) {
            if ( ( ( *imageX ) == 0u ) || ( ( *imageX ) == 255u ) )
                point.emplace_back( x, y );
        }
    }

    return point;
}
