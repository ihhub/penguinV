#include <iostream>
#include "performance_test_blob_detection.h"
#include "performance_test_filtering.h"
#include "performance_test_framework.h"
#include "performance_test_helper.h"
#include "performance_test_image_function.h"

int main( int argc, char* argv[] )
{
    Performance_Test::setRunCount( argc, argv, 128 );

    cpu_Memory::MemoryAllocator::instance().reserve( 32 * 1024 * 1024 );

    PerformanceTestFramework framework;
    addTests_Blob_Detection     ( framework );
    addTests_Filtering          ( framework );
    addTests_Image_Function     ( framework );
    framework.run();

    return 0;
}
