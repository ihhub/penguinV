#include "../../src/blob_detection.h"
#include "performance_test_blob_detection.h"
#include "performance_test_helper.h"

namespace
{
    std::pair < double, double > SolidImage( uint32_t size )
    {
        Performance_Test::TimerContainer timer;

        Bitmap_Image::Image image = Performance_Test::uniformImage( size, size, Performance_Test::randomValue<uint8_t>( 1, 256 ) );

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

// Function naming: function_name_(image_size)
#define CONVERT_PARAMETER( parameter ) _##parameter

#define DECLARE_FUNCTION( function, size )                                        \
std::pair < double, double > CONVERT_PARAMETER(function)CONVERT_PARAMETER(size)() \
{                                                                                 \
    return function( size );                                                      \
}

#define DECLARE_FUNCTIONS( function, size1, size2, size3, size4 ) \
DECLARE_FUNCTION( function, size1 )                               \
DECLARE_FUNCTION( function, size2 )                               \
DECLARE_FUNCTION( function, size3 )                               \
DECLARE_FUNCTION( function, size4 )

#define SET_FUNCTION( function ) DECLARE_FUNCTIONS( function, 256, 512, 1024, 2048 );

namespace blob_detection
{
    SET_FUNCTION( SolidImage )
}

#define ADD_FUNCTIONS( framework, function, size1, size2, size3, size4 )                    \
ADD_TEST( framework, blob_detection::CONVERT_PARAMETER(function)CONVERT_PARAMETER(size1) ); \
ADD_TEST( framework, blob_detection::CONVERT_PARAMETER(function)CONVERT_PARAMETER(size2) ); \
ADD_TEST( framework, blob_detection::CONVERT_PARAMETER(function)CONVERT_PARAMETER(size3) ); \
ADD_TEST( framework, blob_detection::CONVERT_PARAMETER(function)CONVERT_PARAMETER(size4) );

#define ADD_TEST_FUNCTION( framework, function ) ADD_FUNCTIONS( framework, function, 256, 512, 1024, 2048 )

namespace Performance_Test
{
    void addTests_Blob_Detection( PerformanceTestFramework & framework )
    {
        ADD_TEST_FUNCTION( framework, SolidImage )
    }
}
