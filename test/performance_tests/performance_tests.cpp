// This application is designed to run performance tests on penguinV library
#include <iostream>
#include "performance_test_blob_detection.h"
#include "performance_test_filtering.h"
#include "performance_test_framework.h"
#include "performance_test_function_pool.h"
#include "performance_test_image_function.h"
#include "performance_test_image_function_avx.h"
#include "performance_test_image_function_neon.h"
#include "performance_test_image_function_sse.h"

int main()
{
    // The main purpose of this application is to test everything within library
    // To do this we need an engine (framework) and a bunch of tests

    // We create a framework
    Performance_Test::PerformanceTestFramework framework;

    // We add tests
    Performance_Test::addTests_Blob_Detection     ( framework );
    Performance_Test::addTests_Filtering          ( framework );
    Performance_Test::addTests_Function_Pool      ( framework );
    Performance_Test::addTests_Image_Function     ( framework );
    Performance_Test::addTests_Image_Function_Avx ( framework );
    Performance_Test::addTests_Image_Function_Neon( framework );
    Performance_Test::addTests_Image_Function_Sse ( framework );

    // Just run the framework what will handle all tests
    framework.run();

    return 0;
}
