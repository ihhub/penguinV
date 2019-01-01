// This application is designed to run performance tests on penguinV library
#include <iostream>
#include "performance_test_blob_detection.h"
#include "performance_test_math.h"
#include "performance_test_filtering.h"
#include "performance_test_framework.h"
#include "performance_test_image_function.h"

int main()
{
    // The main purpose of this application is to test everything within library
    // To do this we need an engine (framework) and a bunch of tests

    // We create a framework
    PerformanceTestFramework framework;

    // We add tests
    addTests_Blob_Detection     ( framework );
    addTests_Filtering          ( framework );
    addTests_Image_Function     ( framework );
    addTests_Math               ( framework );

    // Just run the framework what will handle all tests
    framework.run();

    return 0;
}
