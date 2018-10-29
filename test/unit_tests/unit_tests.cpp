// This application is designed to run unit tests on penguinV library
#include <iostream>
#include "unit_test_bitmap.h"
#include "unit_test_blob_detection.h"
#include "unit_test_fft.h"
#include "unit_test_framework.h"
#include "unit_test_image_buffer.h"
#include "unit_test_image_function.h"
#include "unit_test_math.h"

int main()
{
    // The main purpose of this application is to test everything within library
    // To do this we need an engine (framework) and a bunch of tests

    // We create a framework
    UnitTestFramework framework;

    // We add tests
    addTests_Bitmap         ( framework );
    addTests_Blob_Detection ( framework );
    addTests_Image_Buffer   ( framework );
    addTests_Image_Function ( framework );
    addTests_Math           ( framework );
    addTests_FFT( framework );

    // Just run the framework what will handle all tests
    return framework.run();
}
