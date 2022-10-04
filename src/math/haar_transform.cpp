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

#include "haar_transform.h"
#include "../penguinv_exception.h"
#include <cmath>

namespace
{
    template <typename _Type>
    void directTransform( const std::vector<_Type> & input, std::vector<_Type> & output, size_t width, size_t height )
    {
        if ( input.empty() || input.size() != output.size() || input.size() != width * height || ( width % 2 ) != 0 || ( height % 2 ) != 0 )
            throw penguinVException( "Incorrect input parameters for Haar direct transform" );

        // Direct Haar Matrix
        // |  1/sqrt(2) 1/sqrt(2) |
        // | -1/sqrt(2) 1/sqrt(2) |

        // Having 2 pixel intensities x and y we would have new values
        // a =  x * (1 / sqrt(2)) + y * (1 / sqrt(2)) = (y + x) * (1 / sqrt(2))
        // b = -x * (1 / sqrt(2)) + y * (1 / sqrt(2)) = (y - x) * (1 / sqrt(2))

        const _Type coefficient = 1.0f / std::sqrt( 2.0f );
        std::vector<_Type> temp( input.size() );

        // Transform by X
        const _Type * inputY = input.data();
        const _Type * inputYEnd = input.data() + input.size();
        _Type * outputY = temp.data();

        const size_t halfWidth = width / 2;

        for ( ; inputY != inputYEnd; inputY += width, outputY += width ) {
            const _Type * inputX = inputY;
            const _Type * inputXEnd = inputX + width;
            _Type * outputX = outputY;
            for ( ; inputX != inputXEnd; inputX += 2, ++outputX ) {
                *( outputX ) = ( *( inputX + 1 ) + *( inputX ) ) * coefficient;
                *( outputX + halfWidth ) = ( *( inputX + 1 ) - *( inputX ) ) * coefficient;
            }
        }

        // Transform by Y
        const _Type * inputX = temp.data();
        const _Type * inputXEnd = temp.data() + width;
        _Type * outputX = output.data();

        const size_t halfHeight = ( height / 2 ) * width;

        for ( ; inputX != inputXEnd; ++inputX, ++outputX ) {
            inputY = inputX;
            inputYEnd = inputX + height * width;
            outputY = outputX;
            for ( ; inputY != inputYEnd; inputY += 2 * width, outputY += width ) {
                *( outputY ) = ( *( inputY + width ) + *( inputY ) ) * coefficient;
                *( outputY + halfHeight ) = ( *( inputY + width ) - *( inputY ) ) * coefficient;
            }
        }
    }

    template <typename _Type>
    void inverseTransform( const std::vector<_Type> & input, std::vector<_Type> & output, size_t width, size_t height )
    {
        if ( input.empty() || input.size() != output.size() || input.size() != width * height || ( width % 2 ) != 0 || ( height % 2 ) != 0 )
            throw penguinVException( "Incorrect input parameters for Haar inverse transform" );

        // Inverse Haar Matrix
        // | 1/sqrt(2) -1/sqrt(2) |
        // | 1/sqrt(2)  1/sqrt(2) |

        // Having 2 pixel intensities x and y we would have new values
        // a = x * (1 / sqrt(2)) - y * (1 / sqrt(2)) = (x - y) * (1 / sqrt(2))
        // b = x * (1 / sqrt(2)) + y * (1 / sqrt(2)) = (x + y) * (1 / sqrt(2))

        const _Type coefficient = 1.0f / std::sqrt( 2.0f );
        std::vector<_Type> temp( input.size() );

        // Transform by X
        const _Type * inputY = input.data();
        const _Type * inputYEnd = input.data() + input.size();
        _Type * outputY = temp.data();

        const size_t halfWidth = width / 2;

        for ( ; inputY != inputYEnd; inputY += width, outputY += width ) {
            const _Type * inputX = inputY;
            _Type * outputX = outputY;
            const _Type * outputXEnd = outputY + width;
            for ( ; outputX != outputXEnd; ++inputX, outputX += 2 ) {
                *( outputX ) = ( *( inputX ) - *( inputX + halfWidth ) ) * coefficient;
                *( outputX + 1 ) = ( *( inputX ) + *( inputX + halfWidth ) ) * coefficient;
            }
        }

        // Transform by Y
        const _Type * inputX = temp.data();
        const _Type * inputXEnd = temp.data() + width;
        _Type * outputX = output.data();

        const size_t halfHeight = ( height / 2 ) * width;

        for ( ; inputX != inputXEnd; ++inputX, ++outputX ) {
            inputY = inputX;
            outputY = outputX;
            const _Type * outputYEnd = outputX + height * width;
            for ( ; outputY != outputYEnd; inputY += width, outputY += 2 * width ) {
                *( outputY ) = ( *( inputY ) - *( inputY + halfHeight ) ) * coefficient;
                *( outputY + width ) = ( *( inputY ) + *( inputY + halfHeight ) ) * coefficient;
            }
        }
    }
}

namespace Image_Function
{
    void HaarDirectTransform( const std::vector<double> & input, std::vector<double> & output, size_t width, size_t height )
    {
        directTransform( input, output, width, height );
    }

    void HaarInverseTransform( const std::vector<double> & input, std::vector<double> & output, size_t width, size_t height )
    {
        inverseTransform( input, output, width, height );
    }

    void HaarDirectTransform( const std::vector<float> & input, std::vector<float> & output, size_t width, size_t height )
    {
        directTransform( input, output, width, height );
    }

    void HaarInverseTransform( const std::vector<float> & input, std::vector<float> & output, size_t width, size_t height )
    {
        inverseTransform( input, output, width, height );
    }
}
