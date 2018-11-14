#include "unit_test_edge_detection.h"
#include "unit_test_helper.h"
#include "../../src/edge_detection.h"

namespace edge_detection
{
    bool DetectHorizontalEdge()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            PenguinV_Image::Image image = Unit_Test::blackImage();

            uint32_t roiX, roiY;
            uint32_t roiWidth, roiHeight;
            Unit_Test::generateRoi( image, roiX, roiY, roiWidth, roiHeight );

            const uint32_t roiXEnd = roiX + roiWidth;

            if ( roiX <= 1u && (roiXEnd + 2u) >= image.width() )
                continue;

            Unit_Test::fillImage( image, roiX, roiY, roiWidth, roiHeight, Unit_Test::randomValue<uint8_t>( 64, 256 ) );

            EdgeDetection edgeDetection;
            edgeDetection.find( image, EdgeParameter( EdgeParameter::LEFT_TO_RIGHT) );

            const std::vector< Point2d > & positive = edgeDetection.positiveEdge();
            const std::vector< Point2d > & negative = edgeDetection.negativeEdge();

            if ( ( (roiX > 1u) && (positive.size() != roiHeight) ) || ( ((roiXEnd + 2u) < image.width()) && (negative.size() != roiHeight) ) )
                return false;

            for ( std::vector< Point2d >::const_iterator point = positive.cbegin(); point != positive.cend(); ++point ) {
                if ( fabs( point->x - roiX ) > 1.0 )
                    return false;
            }

            for ( std::vector< Point2d >::const_iterator point = negative.cbegin(); point != negative.cend(); ++point ) {
                if ( fabs( point->x - roiXEnd ) > 1.0 )
                    return false;
            }
        }

        return true;
    }

    bool DetectVerticalEdge()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            PenguinV_Image::Image image = Unit_Test::blackImage();

            uint32_t roiX, roiY;
            uint32_t roiWidth, roiHeight;
            Unit_Test::generateRoi( image, roiX, roiY, roiWidth, roiHeight );

            const uint32_t roiYEnd = roiY + roiHeight;

            if ( roiY <= 1u && (roiYEnd + 2u) >= image.height() )
                continue;

            Unit_Test::fillImage( image, roiX, roiY, roiWidth, roiHeight, Unit_Test::randomValue<uint8_t>( 64, 256 ) );

            EdgeDetection edgeDetection;
            edgeDetection.find( image, EdgeParameter( EdgeParameter::TOP_TO_BOTTOM) );

            const std::vector< Point2d > & positive = edgeDetection.positiveEdge();
            const std::vector< Point2d > & negative = edgeDetection.negativeEdge();

            if ( ( (roiY > 1u) && (positive.size() != roiWidth) ) || ( ((roiYEnd + 2u) < image.height()) && (negative.size() != roiWidth) ) )
                return false;

            for ( std::vector< Point2d >::const_iterator point = positive.cbegin(); point != positive.cend(); ++point ) {
                if ( fabs( point->y - roiY ) > 1.0 )
                    return false;
            }

            for ( std::vector< Point2d >::const_iterator point = negative.cbegin(); point != negative.cend(); ++point ) {
                if ( fabs( point->y - roiYEnd ) > 1.0 )
                    return false;
            }
        }

        return true;
    }
}

void addTests_Edge_Detection( UnitTestFramework & framework )
{
    framework.add( edge_detection::DetectHorizontalEdge, "edge_detection::Detect horizontal edges" );
    framework.add( edge_detection::DetectVerticalEdge, "edge_detection::Detect vertical edges" );
}
