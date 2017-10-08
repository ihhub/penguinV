#include "unit_test_blob_detection.h"
#include "unit_test_helper.h"
#include "../Library/blob_detection.h"
#include "../Library/image_function.h"

namespace Unit_Test
{
    void addTests_Blob_Detection( UnitTestFramework & framework )
    {
        ADD_TEST( framework, Blob_Detection_Test::Detect1Blob );
    }

    namespace Blob_Detection_Test
    {
        bool Detect1Blob()
        {
            for( size_t i = 0; i < runCount(); ++i ) {
                Bitmap_Image::Image image = blackImage();

                size_t roiX, roiY;
                size_t roiWidth, roiHeight;

                generateRoi( image, roiX, roiY, roiWidth, roiHeight );

                uint8_t intensity = randomValue<uint8_t>( 1, 255 );

                fillImage( image, roiX, roiY, roiWidth, roiHeight, intensity );

                Blob_Detection::BlobDetection detection;

                detection.find( image );

                size_t contour = roiWidth > 1 && roiHeight > 2 ? 2 * roiWidth + 2 * (roiHeight - 2) : roiWidth * roiHeight;

                if( detection().size() != 1 || detection()[0].width() != roiWidth ||
                    detection()[0].height() != roiHeight || detection()[0].size() != roiWidth * roiHeight ||
                    detection()[0].contourX().size() != contour ||
                    detection()[0].edgeX   ().size() != contour )
                    return false;
            }

            return true;
        }
    };
};
