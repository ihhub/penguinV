#include "unit_test_bitmap.h"
#include "unit_test_blob_detection.h"
#include "unit_test_edge_detection.h"
#include "unit_test_fft.h"
#include "unit_test_framework.h"
#include "unit_test_helper.h"
#include "unit_test_image_buffer.h"
#include "unit_test_image_function.h"
#include "unit_test_math.h"
#include "unit_test_png.h"

int main( int argc, char * argv[] )
{
    Unit_Test::setRunCount( argc, argv, 1001 );

    cpu_Memory::MemoryAllocator::instance().reserve( 32 * 1024 * 1024 ); // reserve preallocated memory

    UnitTestFramework framework;
    addTests_Bitmap         ( framework );
    addTests_Png            ( framework );
    addTests_Blob_Detection ( framework );
    addTests_Edge_Detection ( framework );
    addTests_Image_Buffer   ( framework );
    addTests_Image_Function ( framework );
    addTests_Math           ( framework );
    addTests_FFT            ( framework );
    return framework.run();
}
