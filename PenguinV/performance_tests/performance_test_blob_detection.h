#pragma once

#include "performance_test_framework.h"

namespace Performance_Test
{
    void addTests_Blob_Detection( PerformanceTestFramework & framework ); // function what adds all below tests to framework

    namespace Blob_Detection_Test
    {
        std::pair < double, double > SolidImageSize256();
        std::pair < double, double > SolidImageSize512();
        std::pair < double, double > SolidImageSize1024();
        std::pair < double, double > SolidImageSize2048();
    };
};
