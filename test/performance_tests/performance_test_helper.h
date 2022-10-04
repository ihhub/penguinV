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

#include "../test_helper.h"
#include <chrono>
#include <cstdlib>
#include <list>
#include <vector>

namespace Performance_Test
{
    using namespace Test_Helper;
    // A class to measure time of individual test
    class BaseTimerContainer
    {
    public:
        BaseTimerContainer();
        ~BaseTimerContainer();

        std::pair<double, double> mean(); // returns mean and sigma values
    protected:
        void push( double value );

    private:
        std::list<double> _time;
    };

    class TimerContainer : public BaseTimerContainer
    {
    public:
        TimerContainer();
        ~TimerContainer();

        void start(); // start time measurement
        void stop(); // stop time measurement
    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> _startTime;
    };

    typedef void ( *performanceFunction )( TimerContainer &, uint32_t );
    std::pair<double, double> runPerformanceTest( performanceFunction function, uint32_t size );
}
