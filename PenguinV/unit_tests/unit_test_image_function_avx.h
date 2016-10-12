#pragma once

#include "unit_test_framework.h"

namespace Unit_Test
{
	void addTests_Image_Function_Avx(UnitTestFramework & framework); // function what adds all below tests to framework

	namespace Image_Function_Avx_Test
	{
		bool BitwiseAnd2ParametersTest();
		bool BitwiseAnd3ParametersTest();
		bool BitwiseAnd8ParametersTest();
		bool BitwiseAnd11ParametersTest();

		bool BitwiseOr2ParametersTest();
		bool BitwiseOr3ParametersTest();
		bool BitwiseOr8ParametersTest();
		bool BitwiseOr11ParametersTest();

		bool BitwiseXor2ParametersTest();
		bool BitwiseXor3ParametersTest();
		bool BitwiseXor8ParametersTest();
		bool BitwiseXor11ParametersTest();

		bool Invert1ParameterTest();
		bool Invert2ParametersTest();
		bool Invert5ParametersTest();
		bool Invert8ParametersTest();

		bool Maximum2ParametersTest();
		bool Maximum3ParametersTest();
		bool Maximum8ParametersTest();
		bool Maximum11ParametersTest();

		bool Minimum2ParametersTest();
		bool Minimum3ParametersTest();
		bool Minimum8ParametersTest();
		bool Minimum11ParametersTest();

		bool Subtract2ParametersTest();
		bool Subtract3ParametersTest();
		bool Subtract8ParametersTest();
		bool Subtract11ParametersTest();

		bool Sum1ParameterTest();
		bool Sum5ParametersTest();

		bool Threshold2ParametersTest();
		bool Threshold3ParametersTest();
		bool Threshold6ParametersTest();
		bool Threshold9ParametersTest();

		bool ThresholdDouble3ParametersTest();
		bool ThresholdDouble4ParametersTest();
		bool ThresholdDouble7ParametersTest();
		bool ThresholdDouble10ParametersTest();
	};
};
