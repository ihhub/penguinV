#pragma once

#include "performance_test_framework.h"

namespace Performance_Test
{
    void addTests_Filtering( PerformanceTestFramework & framework ); // function what adds all below tests to framework

    namespace Filtering_Test
    {
        std::pair < double, double > MedianFilterSize256();
        std::pair < double, double > MedianFilterSize512();
        std::pair < double, double > MedianFilterSize1024();
        std::pair < double, double > MedianFilterSize2048();
    }
}
