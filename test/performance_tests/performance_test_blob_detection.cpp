#include "../../src/blob_detection.h"
#include "performance_test_blob_detection.h"
#include "performance_test_helper.h"
#include "../test_helper.h"

namespace
{
    std::pair < double, double > SolidImage( uint32_t size )
    {
        Performance_Test::TimerContainer timer;

        PenguinV_Image::Image image = Test_Helper::uniformImage( size, size, Test_Helper::randomValue<uint8_t>( 1, 256 ) );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            { // destroy the object within the scope
                Blob_Detection::BlobDetection detection;

                detection.find( image );
            }

            timer.stop();
        }

        return timer.mean();
    }
}

// Function naming: _functionName_imageSize
#define SET_FUNCTION( function )                                      \
namespace blob_detection_##function                                   \
{                                                                     \
    std::pair < double, double > _256 () { return function( 256  ); } \
    std::pair < double, double > _512 () { return function( 512  ); } \
    std::pair < double, double > _1024() { return function( 1024 ); } \
    std::pair < double, double > _2048() { return function( 2048 ); } \
}

namespace
{
    SET_FUNCTION( SolidImage )
}

#define ADD_TEST_FUNCTION( framework, function )         \
ADD_TEST( framework, blob_detection_##function::_256 );  \
ADD_TEST( framework, blob_detection_##function::_512 );  \
ADD_TEST( framework, blob_detection_##function::_1024 ); \
ADD_TEST( framework, blob_detection_##function::_2048 );

void addTests_Blob_Detection( PerformanceTestFramework & framework )
{
    ADD_TEST_FUNCTION( framework, SolidImage )
}
