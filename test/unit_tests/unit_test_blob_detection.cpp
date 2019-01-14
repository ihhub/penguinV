#include "unit_test_blob_detection.h"
#include "../test_helper.h"
#include "../../src/blob_detection.h"
#include "../../src/image_function.h"

namespace blob_detection
{
    bool Detect1Blob()
    {
        for( uint32_t i = 0; i < Test_Helper::runCount(); ++i ) {
            PenguinV_Image::Image image = Test_Helper::blackImage();

            uint32_t roiX, roiY;
            uint32_t roiWidth, roiHeight;
            Test_Helper::generateRoi( image, roiX, roiY, roiWidth, roiHeight );

            Test_Helper::fillImage( image, roiX, roiY, roiWidth, roiHeight, Test_Helper::randomValue<uint8_t>( 1, 256 ) );

            Blob_Detection::BlobDetection detection;
            detection.find( image );

            const uint32_t contour = ((roiWidth > 1) && (roiHeight > 2)) ? (2 * roiWidth + 2 * (roiHeight - 2)) : (roiWidth * roiHeight);

            if( detection().size() != 1 || detection()[0].width() != roiWidth ||
                detection()[0].height() != roiHeight || detection()[0].size() != roiWidth * roiHeight ||
                detection()[0].contourX().size() != contour ||
                detection()[0].edgeX   ().size() != contour )
                return false;
        }

        return true;
    }
}


void addTests_Blob_Detection( UnitTestFramework & framework )
{
    framework.add( blob_detection::Detect1Blob, "blob_detection::Detect one blob" );
}
