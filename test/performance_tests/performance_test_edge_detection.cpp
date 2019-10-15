#include "../../src/edge_detection.h"
#include "performance_test_edge_detection.h"
#include "performance_test_helper.h"

namespace
{
    std::pair < double, double > SolidImage( uint32_t size )
    {
        Performance_Test::TimerContainer timer;

        PenguinV::Image image = Performance_Test::randomImage( size, size );

        for( uint32_t i = 0; i < Performance_Test::runCount(); ++i ) {
            timer.start();

            { // destroy the object within the scope
                EdgeDetection detection;

                detection.find( image );
            }

            timer.stop();
        }

        return timer.mean();
    }
}

// Function naming: _functionName_imageSize
#define SET_FUNCTION( function )                                      \
namespace edge_detection_##function                                   \
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
ADD_TEST( framework, edge_detection_##function::_256 );  \
ADD_TEST( framework, edge_detection_##function::_512 );  \
ADD_TEST( framework, edge_detection_##function::_1024 ); \
ADD_TEST( framework, edge_detection_##function::_2048 );

void addTests_Edge_Detection( PerformanceTestFramework & framework )
{
    ADD_TEST_FUNCTION( framework, SolidImage )
}
