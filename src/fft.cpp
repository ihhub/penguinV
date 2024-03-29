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

#include "fft.h"
#include "penguinv_exception.h"

namespace FFT
{
    ComplexData::ComplexData() {}

    ComplexData::ComplexData( const penguinV::Image & image )
    {
        set( image );
    }

    ComplexData::ComplexData( const std::vector<float> & data, uint32_t width, uint32_t height )
    {
        resize( width, height );
        set( data );
    }

    ComplexData::ComplexData( const BaseComplexData<kiss_fft_cpx> & data )
    {
        _copy( data );
    }

    ComplexData::ComplexData( ComplexData && data )
    {
        _swap( data );
    }

    ComplexData::~ComplexData()
    {
        _clean();
    }

    void ComplexData::set( const penguinV::Image & image )
    {
        if ( image.empty() || image.colorCount() != 1u )
            throw penguinVException( "Failed to allocate complex data for empty or coloured image" );

        _clean();

        const uint32_t size = image.width() * image.height();

        _allocateData( size * sizeof( kiss_fft_cpx ) );

        _width = image.width();
        _height = image.height();

        // Copy data from input image to FFT array
        const uint32_t rowSize = image.rowSize();

        const uint8_t * inY = image.data();
        kiss_fft_cpx * out = _data;

        const uint8_t * inYEnd = inY + _height * rowSize;

        for ( ; inY != inYEnd; inY += rowSize ) {
            const uint8_t * inX = inY;
            const uint8_t * inXEnd = inX + _width;

            for ( ; inX != inXEnd; ++inX, ++out ) {
                out->r = *inX;
                out->i = 0;
            }
        }
    }

    void ComplexData::set( const std::vector<float> & data )
    {
        if ( data.empty() || _width == 0 || _height == 0 || data.size() != _width * _height )
            throw penguinVException( "Failed to allocate complex data for empty or coloured image" );

        const float * in = data.data();
        kiss_fft_cpx * out = _data;
        const kiss_fft_cpx * outEnd = out + _width * _height;

        for ( ; out != outEnd; ++in, ++out ) {
            out->r = *in;
            out->i = 0;
        }
    }

    penguinV::Image ComplexData::get() const
    {
        if ( empty() )
            return penguinV::Image();

        penguinV::Image image( _width, _height, 1u, 1u );
        uint8_t * out = image.data();

        const uint32_t size = _width * _height;
        const uint32_t middleX = _width / 2;
        const uint32_t middleY = _height / 2;

        for ( uint32_t inY = 0; inY < _height; ++inY ) {
            const uint32_t outY = ( inY < middleY ) ? middleY + inY : inY - middleY;

            const uint32_t posYIn = inY * _width;
            const uint32_t posYOut = outY * _width;
            for ( uint32_t inX = 0; inX < _width; ++inX ) {
                const uint32_t outX = ( inX < middleX ) ? middleX + inX : inX - middleX;
                out[posYOut + outX] = static_cast<uint8_t>( _data[posYIn + inX].r / static_cast<float>( size ) + 0.5 );
            }
        }

        return image;
    }

    void ComplexData::_allocateData( size_t size )
    {
        _data = reinterpret_cast<kiss_fft_cpx *>( cpu_Memory::MemoryAllocator::instance().allocate( size ) );
    }

    void ComplexData::_freeData()
    {
        cpu_Memory::MemoryAllocator::instance().free( _data );
    }

    void ComplexData::_copyData( const BaseComplexData<kiss_fft_cpx> & data )
    {
        memcpy( _data, data.data(), sizeof( kiss_fft_cpx ) * _width * _height );
    }

    FFTExecutor::FFTExecutor( uint32_t width_, uint32_t height_ )
        : _planDirect( 0 )
        , _planInverse( 0 )
    {
        initialize( width_, height_ );
    }

    FFTExecutor::~FFTExecutor()
    {
        _clean();
    }

    void FFTExecutor::directTransform( ComplexData & data )
    {
        directTransform( data, data );
    }

    void FFTExecutor::directTransform( const ComplexData & in, ComplexData & out )
    {
        if ( _planDirect == 0 || !equalSize( *this, in ) || !equalSize( in, out ) )
            throw penguinVException( "Invalid parameters for FFTExecutor::directTransform()" );

        kiss_fftnd( _planDirect, in.data(), out.data() );
    }

    void FFTExecutor::inverseTransform( ComplexData & data )
    {
        inverseTransform( data, data );
    }

    void FFTExecutor::inverseTransform( const ComplexData & in, ComplexData & out )
    {
        if ( _planInverse == 0 || !equalSize( *this, in ) || !equalSize( in, out ) )
            throw penguinVException( "Invalid parameters for FFTExecutor::inverseTransform()" );

        kiss_fftnd( _planInverse, in.data(), out.data() );
    }

    void FFTExecutor::complexMultiplication( const ComplexData & in1, const ComplexData & in2, ComplexData & out ) const
    {
        if ( !equalSize( in1, in2 ) || !equalSize( in1, out ) || in1.width() == 0 || in1.height() == 0 )
            throw penguinVException( "Invalid parameters for FFTExecutor::complexMultiplication" );

        // in1 = A + iB
        // in2 = C + iD
        // out = in1 * (-in2) = (A + iB) * (-C - iD) = - A * C - i(B * C) - i(A * D) + B * D

        const uint32_t size = in1.width() * in1.height();

        const kiss_fft_cpx * in1X = in1.data();
        const kiss_fft_cpx * in2X = in2.data();
        kiss_fft_cpx * outX = out.data();
        const kiss_fft_cpx * outXEnd = outX + size;

        for ( ; outX != outXEnd; ++in1X, ++in2X, ++outX ) {
            outX->r = in1X->r * in2X->r - in1X->i * in2X->i;
            outX->i = in1X->r * in2X->i + in1X->i * in2X->r;
        }
    }

    void FFTExecutor::_makePlans()
    {
        const int dims[2] = { static_cast<int>( _width ), static_cast<int>( _height ) };
        _planDirect = kiss_fftnd_alloc( dims, 2, false, 0, 0 );
        _planInverse = kiss_fftnd_alloc( dims, 2, true, 0, 0 );
    }

    void FFTExecutor::_cleanPlans()
    {
        if ( _planDirect != 0 ) {
            kiss_fft_free( _planDirect );

            _planDirect = 0;
        }

        if ( _planInverse != 0 ) {
            kiss_fft_free( _planInverse );

            _planInverse = 0;
        }
    }
}
