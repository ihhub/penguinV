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

#include "../../ui/ui.h"
#include <algorithm>

template <typename _Type>
void DisplayHistogram( UiWindow & window, const std::vector<_Type> & histogram )
{
    if ( histogram.empty() )
        return;

    const _Type minValue = *std::min_element( histogram.cbegin(), histogram.cend() );
    const _Type maxValue = *std::max_element( histogram.cbegin(), histogram.cend() );
    if ( minValue >= maxValue )
        return;

    const _Type range = maxValue - minValue;

    penguinV::Image image( histogram.size(), 100 );
    image.fill( 0u );

    const uint32_t rowSize = image.rowSize();
    uint8_t * dataX = image.data() + ( image.height() - 1u ) * rowSize;

    for ( size_t i = 0u; i < histogram.size(); ++i, ++dataX ) {
        const _Type limit = ( histogram[i] - minValue ) * 100 / range;
        uint8_t * dataY = dataX;
        for ( _Type y = 0; y < limit; ++y, dataY -= rowSize )
            *dataY = 255;
    }

    window.setImage( image );
    window.show();
}
