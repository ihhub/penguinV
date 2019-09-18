#include <algorithm>
#include <iostream>
#include <list>
#include "../../src/image_exception.h"
#include "unit_test_framework.h"

void UnitTestFramework::add( const testFunction test, const std::string & name )
{
    _unitTest.insert( std::pair < testFunction, std::string > ( test, name ) );
}

int UnitTestFramework::run() const
{
    size_t passed = 0u;
    std::list < std::string > failedFunctionId;
    size_t testId = 1u;

    for( std::map < testFunction, std::string >::const_iterator test = _unitTest.begin(); test != _unitTest.end(); ++test, ++testId ) {
        std::cout << "["<< testId << "/" << _unitTest.size() << "] " << test->second << "... " << std::flush;

        try {
            if( (test->first)() ) {
                ++passed;
                std::cout << "OK" << std::endl;
            }
            else {
                failedFunctionId.push_back( test->second );
                std::cout << "FAIL" << std::endl;
            }
        }
        catch( const imageException & ex ) {
            failedFunctionId.push_back( test->second );
            std::cout << "ERROR : library exception '" << ex.what() << "' raised" << std::endl;
        }
        catch( const std::exception & ex ) {
            failedFunctionId.push_back( test->second );
            std::cout << "ERROR : exception '" << ex.what() << "' raised" << std::endl;
        }
        catch( ... ) {
            failedFunctionId.push_back( test->second );
            std::cout << "ERROR : unknown exception raised" << std::endl;
        }
    }

    std::cout << passed << " of " << _unitTest.size() << " tests passed." << std::endl;

    if( !failedFunctionId.empty() ) {
        std::cout << "List of failed tests: " << std::endl;

        for( std::list < std::string >::const_iterator id = failedFunctionId.begin(); id != failedFunctionId.end(); ++id )
            std::cout << *id << std::endl;

        return -1;
    }

    return 0;
}
