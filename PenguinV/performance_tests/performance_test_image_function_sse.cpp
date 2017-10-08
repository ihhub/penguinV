#include "../Library/image_function_sse.h"
#include "performance_test_image_function_sse.h"
#include "performance_test_helper.h"

namespace Performance_Test
{
    void addTests_Image_Function_Sse( PerformanceTestFramework & framework )
    {
        ADD_TEST( framework, Image_Function_Sse_Test::AbsoluteDifferenceSize256 );
        ADD_TEST( framework, Image_Function_Sse_Test::AbsoluteDifferenceSize512 );
        ADD_TEST( framework, Image_Function_Sse_Test::AbsoluteDifferenceSize1024 );
        ADD_TEST( framework, Image_Function_Sse_Test::AbsoluteDifferenceSize2048 );

        ADD_TEST( framework, Image_Function_Sse_Test::BitwiseAndSize256 );
        ADD_TEST( framework, Image_Function_Sse_Test::BitwiseAndSize512 );
        ADD_TEST( framework, Image_Function_Sse_Test::BitwiseAndSize1024 );
        ADD_TEST( framework, Image_Function_Sse_Test::BitwiseAndSize2048 );

        ADD_TEST( framework, Image_Function_Sse_Test::BitwiseOrSize256 );
        ADD_TEST( framework, Image_Function_Sse_Test::BitwiseOrSize512 );
        ADD_TEST( framework, Image_Function_Sse_Test::BitwiseOrSize1024 );
        ADD_TEST( framework, Image_Function_Sse_Test::BitwiseOrSize2048 );

        ADD_TEST( framework, Image_Function_Sse_Test::BitwiseXorSize256 );
        ADD_TEST( framework, Image_Function_Sse_Test::BitwiseXorSize512 );
        ADD_TEST( framework, Image_Function_Sse_Test::BitwiseXorSize1024 );
        ADD_TEST( framework, Image_Function_Sse_Test::BitwiseXorSize2048 );

        ADD_TEST( framework, Image_Function_Sse_Test::InvertSize256 );
        ADD_TEST( framework, Image_Function_Sse_Test::InvertSize512 );
        ADD_TEST( framework, Image_Function_Sse_Test::InvertSize1024 );
        ADD_TEST( framework, Image_Function_Sse_Test::InvertSize2048 );

        ADD_TEST( framework, Image_Function_Sse_Test::MaximumSize256 );
        ADD_TEST( framework, Image_Function_Sse_Test::MaximumSize512 );
        ADD_TEST( framework, Image_Function_Sse_Test::MaximumSize1024 );
        ADD_TEST( framework, Image_Function_Sse_Test::MaximumSize2048 );

        ADD_TEST( framework, Image_Function_Sse_Test::MinimumSize256 );
        ADD_TEST( framework, Image_Function_Sse_Test::MinimumSize512 );
        ADD_TEST( framework, Image_Function_Sse_Test::MinimumSize1024 );
        ADD_TEST( framework, Image_Function_Sse_Test::MinimumSize2048 );

        ADD_TEST( framework, Image_Function_Sse_Test::SubtractSize256 );
        ADD_TEST( framework, Image_Function_Sse_Test::SubtractSize512 );
        ADD_TEST( framework, Image_Function_Sse_Test::SubtractSize1024 );
        ADD_TEST( framework, Image_Function_Sse_Test::SubtractSize2048 );

        ADD_TEST( framework, Image_Function_Sse_Test::SumSize256 );
        ADD_TEST( framework, Image_Function_Sse_Test::SumSize512 );
        ADD_TEST( framework, Image_Function_Sse_Test::SumSize1024 );
        ADD_TEST( framework, Image_Function_Sse_Test::SumSize2048 );

        ADD_TEST( framework, Image_Function_Sse_Test::ThresholdSize256 );
        ADD_TEST( framework, Image_Function_Sse_Test::ThresholdSize512 );
        ADD_TEST( framework, Image_Function_Sse_Test::ThresholdSize1024 );
        ADD_TEST( framework, Image_Function_Sse_Test::ThresholdSize2048 );

        ADD_TEST( framework, Image_Function_Sse_Test::ThresholdDoubleSize256 );
        ADD_TEST( framework, Image_Function_Sse_Test::ThresholdDoubleSize512 );
        ADD_TEST( framework, Image_Function_Sse_Test::ThresholdDoubleSize1024 );
        ADD_TEST( framework, Image_Function_Sse_Test::ThresholdDoubleSize2048 );
    }

    namespace Image_Function_Sse_Test
    {
        std::pair < double, double > AbsoluteDifferenceSize256()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::AbsoluteDifference( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > AbsoluteDifferenceSize512()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::AbsoluteDifference( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > AbsoluteDifferenceSize1024()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::AbsoluteDifference( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > AbsoluteDifferenceSize2048()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::AbsoluteDifference( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseAndSize256()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::BitwiseAnd( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseAndSize512()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::BitwiseAnd( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseAndSize1024()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::BitwiseAnd( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseAndSize2048()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::BitwiseAnd( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseOrSize256()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::BitwiseOr( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseOrSize512()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::BitwiseOr( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseOrSize1024()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::BitwiseOr( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseOrSize2048()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::BitwiseOr( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseXorSize256()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::BitwiseXor( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseXorSize512()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::BitwiseXor( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseXorSize1024()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::BitwiseXor( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseXorSize2048()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::BitwiseXor( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > InvertSize256()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Invert( image[0], image[1] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > InvertSize512()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Invert( image[0], image[1] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > InvertSize1024()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Invert( image[0], image[1] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > InvertSize2048()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Invert( image[0], image[1] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MaximumSize256()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Maximum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MaximumSize512()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Maximum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MaximumSize1024()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Maximum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MaximumSize2048()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Maximum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MinimumSize256()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Minimum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MinimumSize512()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Minimum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MinimumSize1024()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Minimum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MinimumSize2048()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Minimum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SubtractSize256()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Subtract( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SubtractSize512()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Subtract( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SubtractSize1024()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Subtract( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SubtractSize2048()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Subtract( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SumSize256()
        {
            TimerContainer timer;

            Bitmap_Image::Image image = uniformImage( 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Sum( image );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SumSize512()
        {
            TimerContainer timer;

            Bitmap_Image::Image image = uniformImage( 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Sum( image );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SumSize1024()
        {
            TimerContainer timer;

            Bitmap_Image::Image image = uniformImage( 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Sum( image );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SumSize2048()
        {
            TimerContainer timer;

            Bitmap_Image::Image image = uniformImage( 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Sum( image );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdSize256()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 256, 256 );
            uint8_t threshold = randomValue<uint8_t>( 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Threshold( image[0], image[1], threshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdSize512()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 512, 512 );
            uint8_t threshold = randomValue<uint8_t>( 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Threshold( image[0], image[1], threshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdSize1024()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 1024, 1024 );
            uint8_t threshold = randomValue<uint8_t>( 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Threshold( image[0], image[1], threshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdSize2048()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 2048, 2048 );
            uint8_t threshold = randomValue<uint8_t>( 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Threshold( image[0], image[1], threshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdDoubleSize256()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 256, 256 );
            uint8_t minThreshold = randomValue<uint8_t>( 256 );
            uint8_t maxThreshold = randomValue<uint8_t>( minThreshold, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Threshold( image[0], image[1], minThreshold, maxThreshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdDoubleSize512()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 512, 512 );
            uint8_t minThreshold = randomValue<uint8_t>( 256 );
            uint8_t maxThreshold = randomValue<uint8_t>( minThreshold, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Threshold( image[0], image[1], minThreshold, maxThreshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdDoubleSize1024()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 1024, 1024 );
            uint8_t minThreshold = randomValue<uint8_t>( 256 );
            uint8_t maxThreshold = randomValue<uint8_t>( minThreshold, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Threshold( image[0], image[1], minThreshold, maxThreshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdDoubleSize2048()
        {
            TimerContainer timer;

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 2048, 2048 );
            uint8_t minThreshold = randomValue<uint8_t>( 256 );
            uint8_t maxThreshold = randomValue<uint8_t>( minThreshold, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Sse::Threshold( image[0], image[1], minThreshold, maxThreshold );

                timer.stop();
            }

            return timer.mean();
        }
    };
};
