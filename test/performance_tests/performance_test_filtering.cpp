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

#include "performance_test_filtering.h"
#include "../../src/filtering.h"
#include "performance_test_framework.h"
#include "performance_test_helper.h"

namespace
{
    typedef void ( *filterFunction )( const penguinV::Image & input, penguinV::Image & output );

    void MedianFilter3x3( const penguinV::Image & input, penguinV::Image & output )
    {
        Image_Function::Median( input, output, 3 );
    }

    void PrewittFilter( const penguinV::Image & input, penguinV::Image & output )
    {
        Image_Function::Prewitt( input, output );
    }

    void SobelFilter( const penguinV::Image & input, penguinV::Image & output )
    {
        Image_Function::Sobel( input, output );
    }

    std::pair<double, double> FilterFunctionTest( filterFunction Filter, uint32_t size )
    {
        Performance_Test::TimerContainer timer;

        penguinV::Image input = Performance_Test::uniformImage( Performance_Test::randomValue<uint8_t>( 1, 256 ), size, size );
        penguinV::Image output( input.width(), input.height() );

        for ( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            Filter( input, output );

            timer.stop();
        }

        return timer.mean();
    }
}

// Function naming: _functionName_imageSize
#define SET_FUNCTION( function )                                                                                                                                         \
    namespace filtering_##function                                                                                                                                       \
    {                                                                                                                                                                    \
        std::pair<double, double> _256()                                                                                                                                 \
        {                                                                                                                                                                \
            return FilterFunctionTest( function, 256 );                                                                                                                  \
        }                                                                                                                                                                \
        std::pair<double, double> _512()                                                                                                                                 \
        {                                                                                                                                                                \
            return FilterFunctionTest( function, 512 );                                                                                                                  \
        }                                                                                                                                                                \
        std::pair<double, double> _1024()                                                                                                                                \
        {                                                                                                                                                                \
            return FilterFunctionTest( function, 1024 );                                                                                                                 \
        }                                                                                                                                                                \
        std::pair<double, double> _2048()                                                                                                                                \
        {                                                                                                                                                                \
            return FilterFunctionTest( function, 2048 );                                                                                                                 \
        }                                                                                                                                                                \
    }

namespace
{
    SET_FUNCTION( MedianFilter3x3 )
    SET_FUNCTION( PrewittFilter )
    SET_FUNCTION( SobelFilter )
}

#define ADD_TEST_FUNCTION( framework, function )                                                                                                                         \
    ADD_TEST( framework, filtering_##function::_256 );                                                                                                                   \
    ADD_TEST( framework, filtering_##function::_512 );                                                                                                                   \
    ADD_TEST( framework, filtering_##function::_1024 );                                                                                                                  \
    ADD_TEST( framework, filtering_##function::_2048 );

void addTests_Filtering( PerformanceTestFramework & framework )
{
    ADD_TEST_FUNCTION( framework, MedianFilter3x3 )
    ADD_TEST_FUNCTION( framework, PrewittFilter )
    ADD_TEST_FUNCTION( framework, SobelFilter )
}
