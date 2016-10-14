#pragma once

#include "unit_test_framework.h"

namespace Unit_Test
{
	void addTests_Image_Function_Neon(UnitTestFramework & framework); // function what adds all below tests to framework

	namespace Image_Function_Neon_Test
	{
		bool BitwiseAnd2ParametersTest();
		bool BitwiseAnd3ParametersTest();
		bool BitwiseAnd8ParametersTest();
		bool BitwiseAnd11ParametersTest();

		bool BitwiseOr2ParametersTest();
		bool BitwiseOr3ParametersTest();
		bool BitwiseOr8ParametersTest();
		bool BitwiseOr11ParametersTest();
	};
};
