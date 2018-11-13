// This application is designed to run unit tests on penguinV library
#include <cstdlib>
#include "unit_test_bitmap.h"
#include "unit_test_jpeg_ops.h"
#include "unit_test_blob_detection.h"
#include "unit_test_fft.h"
#include "unit_test_framework.h"
#include "unit_test_helper.h"
#include "unit_test_image_buffer.h"
#include "unit_test_image_function.h"
#include "unit_test_math.h"

int main( int argc, char * argv[] )
{
    if ( argc >= 2 ) {
        const int testCount = std::atoi( argv[1] );
        if ( testCount > 0 )
            Unit_Test::setRunCount( static_cast<uint32_t>( testCount ) );
    }

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
    addTests_FFT            ( framework );
    addTests_Jpeg           ( framework );

    // Just run the framework what will handle all tests
    return framework.run();
}
