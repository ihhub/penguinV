#include "unit_test_blob_detection.h"
#include "unit_test_helper.h"
#include "../Library/blob_detection.h"
#include "../Library/image_function.h"

namespace Unit_Test
{
	void addTests_Blob_Detection(UnitTestFramework & framework)
	{
		ADD_TEST( framework, Blob_Detection_Test::Detect1Blob );
	}

	namespace Blob_Detection_Test
	{
		bool Detect1Blob()
		{
			for( uint32_t i = 0; i < runCount(); ++i ) {
				Bitmap_Image::Image image = blackImage();

				uint32_t roiX, roiY;
				uint32_t roiWidth, roiHeight;

				generateRoi( image, roiX, roiY, roiWidth, roiHeight );

				uint8_t intensity = randomValue<uint8_t>(1, 255);

				fillImage( image, roiX, roiY, roiWidth, roiHeight, intensity );

				Blob_Detection::BlobDetection detection;

				detection.find( image );

				if( detection.get().size() != 1 || detection.get()[0].width() != roiWidth ||
					detection.get()[0].height() != roiHeight || detection.get()[0].size() != roiWidth * roiHeight )
					return false;

			}

			return true;
		}
	};
};
