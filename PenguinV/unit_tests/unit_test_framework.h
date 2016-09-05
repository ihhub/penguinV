#pragma once
#include <map>
#include <string>

#define _STRINGIFY(x) #x // Macros to convert the name of function to string
#define ADD_TEST(framework, function) framework.add( function, _STRINGIFY(function) ); // Macros to call add() function of framework

namespace Unit_Test
{
	// pointer to unit test function. Function must return true in successful case
	typedef bool (*testFunction)();

	class UnitTestFramework
	{
	public:
		void add(const testFunction test, const std::string & name ); // register function in framework

		int run() const; // run framework unit tests
						 // returns 0 when all tests passed
	private:
		std::map < testFunction, std::string > _unitTest; // container with pointer to functions and their names
	};
};
