// This application is designed to run performance tests on penguinV library
#include <iostream>
#include "performance_test_image_function_cuda.h"
#include "../performance_test_framework.h"

int main()
{
    // The main purpose of this application is to test everything within library
    // To do this we need an engine (framework) and a bunch of tests

    // We create a framework
    Performance_Test::PerformanceTestFramework framework;

    // We add tests
    Performance_Test::addTests_Image_Function_Cuda( framework );

    // Just run the framework what will handle all tests
    framework.run();

    return 0;
}
