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
#include <map>
#include <string>

#define _STRINGIFY( x ) #x // Macros to convert the name of function to string
#define ADD_TEST( framework, function ) framework.add( function, _STRINGIFY( function ) ); // Macros to call add() function of framework

class PerformanceTestFramework
{
public:
    // pointer to performance test function. Function must return 2 values:
    // - mean value in milliseconds
    // - sigma value in milliseconds
    typedef std::pair<double, double> ( *testFunction )();

    void add( const testFunction test, const std::string & name ); // register function in framework

    void run() const; // run framework performance tests
private:
    std::map<testFunction, std::string> _performanceTest; // container with pointer to functions and their names
};
