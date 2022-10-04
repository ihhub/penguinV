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

#include "unit_test_fft.h"
#include "../../src/fft.h"
#include "unit_test_framework.h"
#include "unit_test_helper.h"

namespace fft
{
    bool RandomImageFFT()
    {
        for ( uint32_t i = 0u; i < 32u; ++i ) { // a special case for FFT because it take a lot of time for execution
            const uint32_t dimension = ( 2u << Unit_Test::randomValue<uint8_t>( 11 ) );

            const penguinV::Image input = Unit_Test::randomImage( dimension, dimension );

            penguinV::Image diracDelta( input.width(), input.height() );
            diracDelta.fill( 0u );
            diracDelta.data()[diracDelta.height() / 2 * diracDelta.rowSize() + diracDelta.width() / 2] = 1u;

            FFT::ComplexData complexDataInput( input );
            FFT::ComplexData complexDataDracDelta( diracDelta );

            FFT::FFTExecutor fftExecutor( input.width(), input.height() );

            fftExecutor.directTransform( complexDataInput );
            fftExecutor.directTransform( complexDataDracDelta );
            fftExecutor.complexMultiplication( complexDataInput, complexDataDracDelta, complexDataInput );
            fftExecutor.inverseTransform( complexDataInput );

            const penguinV::Image output = complexDataInput.get();

            if ( input.height() != output.height() || input.width() != output.width() || input.colorCount() != output.colorCount() )
                return false;

            const uint32_t rowSizeIn = input.rowSize();
            const uint32_t rowSizeOut = output.rowSize();
            const uint32_t width = input.width() * input.colorCount();
            const uint8_t * inY = input.data();
            const uint8_t * outY = output.data();
            const uint8_t * inYEnd = inY + rowSizeIn * input.height();

            for ( ; inY != inYEnd; inY += rowSizeIn, outY += rowSizeOut ) {
                if ( memcmp( inY, outY, width ) != 0 )
                    return false;
            }
        }
        return true;
    }
}

void addTests_FFT( UnitTestFramework & framework )
{
    framework.add( fft::RandomImageFFT, "fft:: random image direct and inverse FFT" );
}
