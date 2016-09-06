#pragma once

#include "unit_test_framework.h"

namespace Unit_Test
{
	void addTests_Image_Function_Avx(UnitTestFramework & framework); // function what adds all below tests to framework

	bool BitwiseAndAvx2ParametersTest();
	bool BitwiseAndAvx3ParametersTest();
	bool BitwiseAndAvx8ParametersTest();
	bool BitwiseAndAvx11ParametersTest();

	bool BitwiseOrAvx2ParametersTest();
	bool BitwiseOrAvx3ParametersTest();
	bool BitwiseOrAvx8ParametersTest();
	bool BitwiseOrAvx11ParametersTest();

	bool BitwiseXorAvx2ParametersTest();
	bool BitwiseXorAvx3ParametersTest();
	bool BitwiseXorAvx8ParametersTest();
	bool BitwiseXorAvx11ParametersTest();

	bool InvertAvx1ParameterTest();
	bool InvertAvx2ParametersTest();
	bool InvertAvx5ParametersTest();
	bool InvertAvx8ParametersTest();

	bool MaximumAvx2ParametersTest();
	bool MaximumAvx3ParametersTest();
	bool MaximumAvx8ParametersTest();
	bool MaximumAvx11ParametersTest();

	bool MinimumAvx2ParametersTest();
	bool MinimumAvx3ParametersTest();
	bool MinimumAvx8ParametersTest();
	bool MinimumAvx11ParametersTest();

	bool SubtractAvx2ParametersTest();
	bool SubtractAvx3ParametersTest();
	bool SubtractAvx8ParametersTest();
	bool SubtractAvx11ParametersTest();

	bool ThresholdAvx2ParametersTest();
	bool ThresholdAvx3ParametersTest();
	bool ThresholdAvx6ParametersTest();
	bool ThresholdAvx9ParametersTest();
};
