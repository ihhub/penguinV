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

#include "performance_test_framework.h"
#include "../../src/penguinv_exception.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <list>

void PerformanceTestFramework::add( const testFunction test, const std::string & name )
{
    _performanceTest.insert( std::pair<testFunction, std::string>( test, name ) );
}

void PerformanceTestFramework::run() const
{
    std::list<std::string> failedFunctionId;
    size_t testId = 1;

    for ( std::map<testFunction, std::string>::const_iterator test = _performanceTest.begin(); test != _performanceTest.end(); ++test, ++testId ) {
        std::cout << "[" << testId << "/" << _performanceTest.size() << "] " << test->second << "... " << std::flush;

        try {
            std::pair<double, double> result = ( test->first )();

            // We must display 3 sigma value because almost all values would be in this range
            std::cout << std::setprecision( 4 ) << result.first << "+/-" << 3 * result.second << " ms" << std::endl;
        }
        catch ( penguinVException & ex ) {
            failedFunctionId.push_back( test->second );
            std::cout << "ERROR : library exception '" << ex.what() << "' raised" << std::endl;
        }
        catch ( std::exception & ex ) {
            failedFunctionId.push_back( test->second );
            std::cout << "ERROR : exception '" << ex.what() << "' raised" << std::endl;
        }
        catch ( ... ) {
            failedFunctionId.push_back( test->second );
            std::cout << "ERROR : unknown exception raised" << std::endl;
        }
    }
}
