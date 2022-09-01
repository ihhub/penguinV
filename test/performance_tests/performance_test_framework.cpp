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
