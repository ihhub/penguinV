#include "../Library/filtering.h"
#include "performance_test_filtering.h"
#include "performance_test_helper.h"

namespace Performance_Test
{
    void addTests_Filtering( PerformanceTestFramework & framework )
    {
        ADD_TEST( framework, Filtering_Test::MedianFilterSize256 );
        ADD_TEST( framework, Filtering_Test::MedianFilterSize512 );
        ADD_TEST( framework, Filtering_Test::MedianFilterSize1024 );
        ADD_TEST( framework, Filtering_Test::MedianFilterSize2048 );
    }

    namespace Filtering_Test
    {
        std::pair < double, double > MedianFilterSize256()
        {
            TimerContainer timer;

            Bitmap_Image::Image input = uniformImage( 256, 256, randomValue<uint8_t>( 1, 256 ) );
            Bitmap_Image::Image output( input.width(), input.height() );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function::Filtering::Median( input, output, 3 );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MedianFilterSize512()
        {
            TimerContainer timer;

            Bitmap_Image::Image input = uniformImage( 512, 512, randomValue<uint8_t>( 1, 256 ) );
            Bitmap_Image::Image output( input.width(), input.height() );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function::Filtering::Median( input, output, 3 );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MedianFilterSize1024()
        {
            TimerContainer timer;

            Bitmap_Image::Image input = uniformImage( 1024, 1024, randomValue<uint8_t>( 1, 256 ) );
            Bitmap_Image::Image output( input.width(), input.height() );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function::Filtering::Median( input, output, 3 );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MedianFilterSize2048()
        {
            TimerContainer timer;

            Bitmap_Image::Image input = uniformImage( 2048, 2048, randomValue<uint8_t>( 1, 256 ) );
            Bitmap_Image::Image output( input.width(), input.height() );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function::Filtering::Median( input, output, 3 );

                timer.stop();
            }

            return timer.mean();
        }
    }
};
