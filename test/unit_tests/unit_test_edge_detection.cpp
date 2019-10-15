#include "unit_test_edge_detection.h"
#include "unit_test_helper.h"
#include "../../src/edge_detection.h"

namespace edge_detection
{
    template <typename _Type>
    bool DetectHorizontalEdge()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            PenguinV::Image image = Unit_Test::blackImage();

            uint32_t roiX, roiY;
            uint32_t roiWidth, roiHeight;
            Unit_Test::generateRoi( image, roiX, roiY, roiWidth, roiHeight );

            const uint32_t roiXEnd = roiX + roiWidth;
            const bool isValidLeftEdge  = ( roiX > 1u ) && ( roiX    + 1u < image.width() );
            const bool isValidRightEdge = ( roiX > 0u ) && ( roiXEnd + 2u < image.width() );

            if ( !isValidLeftEdge && !isValidRightEdge )
                continue;

            Unit_Test::fillImage( image, roiX, roiY, roiWidth, roiHeight, Unit_Test::randomValue<uint8_t>( 64, 256 ) );

            EdgeDetectionBase<_Type> edgeDetection;
            edgeDetection.find( image, EdgeParameter( EdgeParameter::LEFT_TO_RIGHT) );

            const std::vector< PointBase2D<_Type> > & positive = edgeDetection.positiveEdge();
            const std::vector< PointBase2D<_Type> > & negative = edgeDetection.negativeEdge();

            if ( ( isValidLeftEdge && (positive.size() != roiHeight) ) || ( isValidRightEdge && (negative.size() != roiHeight) ) )
                return false;

            for ( typename std::vector< PointBase2D<_Type> >::const_iterator point = positive.cbegin(); point != positive.cend(); ++point ) {
                if ( fabs( point->x - static_cast<_Type>( roiX ) ) > 1.0 )
                    return false;
            }

            for ( typename std::vector< PointBase2D<_Type> >::const_iterator point = negative.cbegin(); point != negative.cend(); ++point ) {
                if ( fabs( point->x - static_cast<_Type>( roiXEnd ) ) > 1.0 )
                    return false;
            }
        }

        return true;
    }

    template <typename _Type>
    bool DetectVerticalEdge()
    {
        for( uint32_t i = 0; i < Unit_Test::runCount(); ++i ) {
            PenguinV::Image image = Unit_Test::blackImage();

            uint32_t roiX, roiY;
            uint32_t roiWidth, roiHeight;
            Unit_Test::generateRoi( image, roiX, roiY, roiWidth, roiHeight );

            const uint32_t roiYEnd = roiY + roiHeight;
            const bool isValidTopEdge    = ( roiY > 1u ) && ( roiY    + 1u < image.height() );
            const bool isValidBottomEdge = ( roiY > 0u ) && ( roiYEnd + 2u < image.height() );

            if ( !isValidTopEdge && !isValidBottomEdge )
                continue;

            Unit_Test::fillImage( image, roiX, roiY, roiWidth, roiHeight, Unit_Test::randomValue<uint8_t>( 64, 256 ) );

            EdgeDetectionBase<_Type> edgeDetection;
            edgeDetection.find( image, EdgeParameter( EdgeParameter::TOP_TO_BOTTOM) );

            const std::vector< PointBase2D<_Type> > & positive = edgeDetection.positiveEdge();
            const std::vector< PointBase2D<_Type> > & negative = edgeDetection.negativeEdge();

            if ( ( isValidTopEdge && (positive.size() != roiWidth) ) || ( isValidBottomEdge && (negative.size() != roiWidth) ) )
                return false;

            for ( typename std::vector< PointBase2D<_Type> >::const_iterator point = positive.cbegin(); point != positive.cend(); ++point ) {
                if ( fabs( point->y - static_cast<_Type>( roiY ) ) > 1.0 )
                    return false;
            }

            for ( typename std::vector< PointBase2D<_Type> >::const_iterator point = negative.cbegin(); point != negative.cend(); ++point ) {
                if ( fabs( point->y - static_cast<_Type>( roiYEnd ) ) > 1.0 )
                    return false;
            }
        }

        return true;
    }
}

void addTests_Edge_Detection( UnitTestFramework & framework )
{
    framework.add( edge_detection::DetectHorizontalEdge<double>, "edge_detection::Detect horizontal edges (double)" );
    framework.add( edge_detection::DetectHorizontalEdge<float>, "edge_detection::Detect horizontal edges (float)" );
    framework.add( edge_detection::DetectVerticalEdge<double>, "edge_detection::Detect vertical edges (double)" );
    framework.add( edge_detection::DetectVerticalEdge<float>, "edge_detection::Detect vertical edges (float)" );
}
