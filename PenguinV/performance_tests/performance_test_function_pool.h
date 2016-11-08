#pragma once

#include "performance_test_framework.h"

namespace Performance_Test
{
	void addTests_Function_Pool(PerformanceTestFramework & framework); // function what adds all below tests to framework

	namespace Function_Pool_Test
	{
		std::pair < double, double > AbsoluteDifferenceSize256();
		std::pair < double, double > AbsoluteDifferenceSize512();
		std::pair < double, double > AbsoluteDifferenceSize1024();
		std::pair < double, double > AbsoluteDifferenceSize2048();

		std::pair < double, double > BitwiseAndSize256();
		std::pair < double, double > BitwiseAndSize512();
		std::pair < double, double > BitwiseAndSize1024();
		std::pair < double, double > BitwiseAndSize2048();

		std::pair < double, double > BitwiseOrSize256();
		std::pair < double, double > BitwiseOrSize512();
		std::pair < double, double > BitwiseOrSize1024();
		std::pair < double, double > BitwiseOrSize2048();

		std::pair < double, double > BitwiseXorSize256();
		std::pair < double, double > BitwiseXorSize512();
		std::pair < double, double > BitwiseXorSize1024();
		std::pair < double, double > BitwiseXorSize2048();

		std::pair < double, double > GammaCorrectionSize256();
		std::pair < double, double > GammaCorrectionSize512();
		std::pair < double, double > GammaCorrectionSize1024();
		std::pair < double, double > GammaCorrectionSize2048();

		std::pair < double, double > HistogramSize256();
		std::pair < double, double > HistogramSize512();
		std::pair < double, double > HistogramSize1024();
		std::pair < double, double > HistogramSize2048();

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

		std::pair < double, double > SumSize256();
		std::pair < double, double > SumSize512();
		std::pair < double, double > SumSize1024();
		std::pair < double, double > SumSize2048();
	};
};
