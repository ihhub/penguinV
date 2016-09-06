#pragma once

#include "unit_test_framework.h"

namespace Unit_Test
{
	void addTests_Image_Function_Sse(UnitTestFramework & framework); // function what adds all below tests to framework

	bool BitwiseAndSse2ParametersTest();
	bool BitwiseAndSse3ParametersTest();
	bool BitwiseAndSse8ParametersTest();
	bool BitwiseAndSse11ParametersTest();

	bool BitwiseOrSse2ParametersTest();
	bool BitwiseOrSse3ParametersTest();
	bool BitwiseOrSse8ParametersTest();
	bool BitwiseOrSse11ParametersTest();

	bool BitwiseXorSse2ParametersTest();
	bool BitwiseXorSse3ParametersTest();
	bool BitwiseXorSse8ParametersTest();
	bool BitwiseXorSse11ParametersTest();

	bool InvertSse1ParameterTest();
	bool InvertSse2ParametersTest();
	bool InvertSse5ParametersTest();
	bool InvertSse8ParametersTest();

	bool MaximumSse2ParametersTest();
	bool MaximumSse3ParametersTest();
	bool MaximumSse8ParametersTest();
	bool MaximumSse11ParametersTest();

	bool MinimumSse2ParametersTest();
	bool MinimumSse3ParametersTest();
	bool MinimumSse8ParametersTest();
	bool MinimumSse11ParametersTest();

	bool SubtractSse2ParametersTest();
	bool SubtractSse3ParametersTest();
	bool SubtractSse8ParametersTest();
	bool SubtractSse11ParametersTest();

	bool ThresholdSse2ParametersTest();
	bool ThresholdSse3ParametersTest();
	bool ThresholdSse6ParametersTest();
	bool ThresholdSse9ParametersTest();
};
