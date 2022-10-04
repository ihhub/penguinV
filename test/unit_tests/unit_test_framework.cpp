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

#include "unit_test_framework.h"
#include "../../src/penguinv_exception.h"
#include <algorithm>
#include <iostream>
#include <list>

void UnitTestFramework::add( const testFunction test, const std::string & name )
{
    _unitTest.insert( std::pair<testFunction, std::string>( test, name ) );
}

int UnitTestFramework::run() const
{
    size_t passed = 0u;
    std::list<std::string> failedFunctionId;
    size_t testId = 1u;

    for ( std::map<testFunction, std::string>::const_iterator test = _unitTest.begin(); test != _unitTest.end(); ++test, ++testId ) {
        std::cout << "[" << testId << "/" << _unitTest.size() << "] " << test->second << "... " << std::flush;

        try {
            if ( ( test->first )() ) {
                ++passed;
                std::cout << "OK" << std::endl;
            }
            else {
                failedFunctionId.push_back( test->second );
                std::cout << "FAIL" << std::endl;
            }
        }
        catch ( const penguinVException & ex ) {
            failedFunctionId.push_back( test->second );
            std::cout << "ERROR : library exception '" << ex.what() << "' raised" << std::endl;
        }
        catch ( const std::exception & ex ) {
            failedFunctionId.push_back( test->second );
            std::cout << "ERROR : exception '" << ex.what() << "' raised" << std::endl;
        }
        catch ( ... ) {
            failedFunctionId.push_back( test->second );
            std::cout << "ERROR : unknown exception raised" << std::endl;
        }
    }

    std::cout << passed << " of " << _unitTest.size() << " tests passed." << std::endl;

    if ( !failedFunctionId.empty() ) {
        std::cout << "List of failed tests: " << std::endl;

        for ( std::list<std::string>::const_iterator id = failedFunctionId.begin(); id != failedFunctionId.end(); ++id )
            std::cout << *id << std::endl;

        return -1;
    }

    return 0;
}
