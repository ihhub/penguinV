#pragma once

#include "unit_test_framework.h"

namespace Unit_Test
{
	void addTests_Image_Function(UnitTestFramework & framework); // function what adds all below tests to framework

	namespace Image_Function_Test
	{
		bool AbsoluteDifference2ParametersTest();
		bool AbsoluteDifference3ParametersTest();
		bool AbsoluteDifference8ParametersTest();
		bool AbsoluteDifference11ParametersTest();

		bool Accumulate2ParametersTest();
		bool Accumulate6ParametersTest();

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

		bool Copy2ParametersTest();
		bool Copy5ParametersTest();
		bool Copy8ParametersTest();

		bool ConvertToGray2ParametersTest();
		bool ConvertToGray8ParametersTest();
		bool ConvertToColor2ParametersTest();
		bool ConvertToColor8ParametersTest();

		bool Fill2ParametersTest();
		bool Fill6ParametersTest();

		bool GammaCorrection3ParametersTest();
		bool GammaCorrection4ParametersTest();
		bool GammaCorrection7ParametersTest();
		bool GammaCorrection10ParametersTest();

		bool Histogram1ParameterTest();
		bool Histogram2ParametersTest();
		bool Histogram4ParametersTest();
		bool Histogram5ParametersTest();

		bool Invert1ParameterTest();
		bool Invert2ParametersTest();
		bool Invert5ParametersTest();
		bool Invert8ParametersTest();

		bool IsEqual2ParametersTest();
		bool IsEqual8ParametersTest();

		bool LookupTable2ParametersTest();
		bool LookupTable3ParametersTest();
		bool LookupTable6ParametersTest();
		bool LookupTable9ParametersTest();

		bool Maximum2ParametersTest();
		bool Maximum3ParametersTest();
		bool Maximum8ParametersTest();
		bool Maximum11ParametersTest();

		bool Minimum2ParametersTest();
		bool Minimum3ParametersTest();
		bool Minimum8ParametersTest();
		bool Minimum11ParametersTest();

		bool Normalize1ParameterTest();
		bool Normalize2ParametersTest();
		bool Normalize5ParametersTest();
		bool Normalize8ParametersTest();

		bool ProjectionProfile2ParametersTest();
		bool ProjectionProfile3ParametersTest();
		bool ProjectionProfile6ParametersTest();
		bool ProjectionProfile7ParametersTest();

		bool Resize2ParametersTest();
		bool Resize3ParametersTest();
		bool Resize7ParametersTest();
		bool Resize9ParametersTest();

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

		bool Transpose1ParameterTest();
		bool Transpose2ParametersTest();
		bool Transpose5ParametersTest();
		bool Transpose8ParametersTest();
	};
};
