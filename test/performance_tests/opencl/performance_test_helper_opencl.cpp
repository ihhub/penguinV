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

#include "performance_test_helper_opencl.h"
#include "../../../src/opencl/opencl_helper.h"

namespace Performance_Test
{
    namespace OpenCL_Helper
    {
        penguinV::Image uniformImage( uint32_t width, uint32_t height )
        {
            return uniformImage( width, height, randomValue<uint8_t>( 256 ) );
        }

        penguinV::Image uniformImage( uint32_t width, uint32_t height, uint8_t value )
        {
            penguinV::ImageOpenCL image( width, height );

            image.fill( value );

            penguinV::Image imageOut;
            imageOut.swap( image );

            return imageOut;
        }

        std::vector<penguinV::Image> uniformImages( uint32_t count, uint32_t width, uint32_t height )
        {
            std::vector<penguinV::Image> image( count );

            for ( std::vector<penguinV::Image>::iterator im = image.begin(); im != image.end(); ++im )
                *im = uniformImage( width, height );

            return image;
        }
    }
}
