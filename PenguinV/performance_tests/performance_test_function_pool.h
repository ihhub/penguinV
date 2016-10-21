#pragma once

#include "performance_test_framework.h"

namespace Performance_Test
{
	void addTests_Function_Pool(PerformanceTestFramework & framework); // function what adds all below tests to framework

	namespace Function_Pool_Test
	{
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
	};
};
