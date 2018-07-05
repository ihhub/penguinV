#pragma once
#include <map>
#include <string>

#define _STRINGIFY(x) #x // Macros to convert the name of function to string
#define ADD_TEST(framework, function) framework.add( function, _STRINGIFY(function) ); // Macros to call add() function of framework

class PerformanceTestFramework
{
public:
    // pointer to performance test function. Function must return 2 values:
    // - mean value in milliseconds
    // - sigma value in milliseconds
    typedef std::pair < double, double > ( *testFunction )();

    void add( const testFunction test, const std::string & name ); // register function in framework

    void run() const; // run framework performance tests
private:
    std::map < testFunction, std::string > _performanceTest; // container with pointer to functions and their names
};
