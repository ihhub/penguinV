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

#include "fft_base.h"
#include "../penguinv_exception.h"

namespace FFT
{
    BaseFFTExecutor::BaseFFTExecutor()
        : _width( 0u )
        , _height( 0u )
    {}

    BaseFFTExecutor::~BaseFFTExecutor() {}

    void BaseFFTExecutor::initialize( uint32_t width_, uint32_t height_ )
    {
        if ( width_ == 0 || height_ == 0 )
            throw penguinVException( "Invalid parameters for FFTExecutor::intialize()" );

        _clean();
        _width = width_;
        _height = height_;
        _makePlans();
    }

    uint32_t BaseFFTExecutor::width() const
    {
        return _width;
    }

    uint32_t BaseFFTExecutor::height() const
    {
        return _height;
    }

    void BaseFFTExecutor::_clean()
    {
        _cleanPlans();
        _width = 0u;
        _height = 0u;
    }
}
