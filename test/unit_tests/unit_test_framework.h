#pragma once
#include <map>
#include <string>

#define ADD_TEST(framework, function) framework.add( function, #function ); // Macros to call add() function of framework

class UnitTestFramework
{
public:
    typedef bool ( *testFunction )(); // pointer to unit test function. Function must return true in successful case
    void add( const testFunction test, const std::string & name ); // register function in framework

    int run() const; // run framework unit tests
                     // returns 0 when all tests passed
private:
    std::map < testFunction, std::string > _unitTest; // container with pointer to functions and their names
};
