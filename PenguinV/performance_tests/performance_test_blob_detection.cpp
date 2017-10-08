#include "../Library/blob_detection.h"
#include "performance_test_blob_detection.h"
#include "performance_test_helper.h"

namespace Performance_Test
{
    void addTests_Blob_Detection( PerformanceTestFramework & framework )
    {
        ADD_TEST( framework, Blob_Detection_Test::SolidImageSize256 );
        ADD_TEST( framework, Blob_Detection_Test::SolidImageSize512 );
        ADD_TEST( framework, Blob_Detection_Test::SolidImageSize1024 );
        ADD_TEST( framework, Blob_Detection_Test::SolidImageSize2048 );
    }

    namespace Blob_Detection_Test
    {
        std::pair < double, double > SolidImageSize256()
        {
            TimerContainer timer;

            Bitmap_Image::Image image = uniformImage( 256, 256, randomValue<uint8_t>( 1, 256 ) );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                { // this we do to free resources of created object what is important for speed too
                    Blob_Detection::BlobDetection detection;

                    detection.find( image );
                }

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SolidImageSize512()
        {
            TimerContainer timer;

            Bitmap_Image::Image image = uniformImage( 512, 512, randomValue<uint8_t>( 1, 256 ) );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                { // this we do to free resources of created object what is important for speed too
                    Blob_Detection::BlobDetection detection;

                    detection.find( image );
                }

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SolidImageSize1024()
        {
            TimerContainer timer;

            Bitmap_Image::Image image = uniformImage( 1024, 1024, randomValue<uint8_t>( 1, 256 ) );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                { // this we do to free resources of created object what is important for speed too
                    Blob_Detection::BlobDetection detection;

                    detection.find( image );
                }

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SolidImageSize2048()
        {
            TimerContainer timer;

            Bitmap_Image::Image image = uniformImage( 2048, 2048, randomValue<uint8_t>( 1, 256 ) );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                { // this we do to free resources of created object what is important for speed too
                    Blob_Detection::BlobDetection detection;

                    detection.find( image );
                }

                timer.stop();
            }

            return timer.mean();
        }
    };
};
