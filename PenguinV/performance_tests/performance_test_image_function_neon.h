#pragma once

#include "performance_test_framework.h"

namespace Performance_Test
{
	void addTests_Image_Function_Neon(PerformanceTestFramework & framework); // function what adds all below tests to framework

	namespace Image_Function_Neon_Test
	{
		std::pair < double, double > BitwiseAndSize256();
		std::pair < double, double > BitwiseAndSize512();
		std::pair < double, double > BitwiseAndSize1024();
		std::pair < double, double > BitwiseAndSize2048();

		std::pair < double, double > BitwiseOrSize256();
		std::pair < double, double > BitwiseOrSize512();
		std::pair < double, double > BitwiseOrSize1024();
		std::pair < double, double > BitwiseOrSize2048();

		std::pair < double, double > InvertSize256();
		std::pair < double, double > InvertSize512();
		std::pair < double, double > InvertSize1024();
		std::pair < double, double > InvertSize2048();

		std::pair < double, double > MaximumSize256();
		std::pair < double, double > MaximumSize512();
		std::pair < double, double > MaximumSize1024();
		std::pair < double, double > MaximumSize2048();

		std::pair < double, double > MinimumSize256();
		std::pair < double, double > MinimumSize512();
		std::pair < double, double > MinimumSize1024();
		std::pair < double, double > MinimumSize2048();

		std::pair < double, double > SubtractSize256();
		std::pair < double, double > SubtractSize512();
		std::pair < double, double > SubtractSize1024();
		std::pair < double, double > SubtractSize2048();

		std::pair < double, double > ThresholdSize256();
		std::pair < double, double > ThresholdSize512();
		std::pair < double, double > ThresholdSize1024();
		std::pair < double, double > ThresholdSize2048();

		std::pair < double, double > ThresholdDoubleSize256();
		std::pair < double, double > ThresholdDoubleSize512();
		std::pair < double, double > ThresholdDoubleSize1024();
		std::pair < double, double > ThresholdDoubleSize2048();
	};
};
