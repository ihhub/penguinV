#pragma once

#include <algorithm>
#include "../../ui/ui.h"

template < typename _Type >
void DisplayHistogram( UiWindow & window, const std::vector < _Type > & histogram )
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
        const _Type limit = (histogram[i] - minValue) * 100 / range;
        uint8_t * dataY = dataX;
        for ( _Type y = 0; y < limit; ++y, dataY -= rowSize )
            *dataY = 255;
    }

    window.setImage( image );
    window.show();
}
