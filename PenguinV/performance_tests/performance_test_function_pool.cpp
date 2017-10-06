#include "../Library/function_pool.h"
#include "performance_test_function_pool.h"
#include "performance_test_helper.h"

namespace Performance_Test
{
    void addTests_Function_Pool( PerformanceTestFramework & framework )
    {
        ADD_TEST( framework, Function_Pool_Test::AbsoluteDifferenceSize256 );
        ADD_TEST( framework, Function_Pool_Test::AbsoluteDifferenceSize512 );
        ADD_TEST( framework, Function_Pool_Test::AbsoluteDifferenceSize1024 );
        ADD_TEST( framework, Function_Pool_Test::AbsoluteDifferenceSize2048 );

        ADD_TEST( framework, Function_Pool_Test::BitwiseAndSize256 );
        ADD_TEST( framework, Function_Pool_Test::BitwiseAndSize512 );
        ADD_TEST( framework, Function_Pool_Test::BitwiseAndSize1024 );
        ADD_TEST( framework, Function_Pool_Test::BitwiseAndSize2048 );

        ADD_TEST( framework, Function_Pool_Test::BitwiseOrSize256 );
        ADD_TEST( framework, Function_Pool_Test::BitwiseOrSize512 );
        ADD_TEST( framework, Function_Pool_Test::BitwiseOrSize1024 );
        ADD_TEST( framework, Function_Pool_Test::BitwiseOrSize2048 );

        ADD_TEST( framework, Function_Pool_Test::BitwiseXorSize256 );
        ADD_TEST( framework, Function_Pool_Test::BitwiseXorSize512 );
        ADD_TEST( framework, Function_Pool_Test::BitwiseXorSize1024 );
        ADD_TEST( framework, Function_Pool_Test::BitwiseXorSize2048 );

        ADD_TEST( framework, Function_Pool_Test::GammaCorrectionSize256 );
        ADD_TEST( framework, Function_Pool_Test::GammaCorrectionSize512 );
        ADD_TEST( framework, Function_Pool_Test::GammaCorrectionSize1024 );
        ADD_TEST( framework, Function_Pool_Test::GammaCorrectionSize2048 );

        ADD_TEST( framework, Function_Pool_Test::HistogramSize256 );
        ADD_TEST( framework, Function_Pool_Test::HistogramSize512 );
        ADD_TEST( framework, Function_Pool_Test::HistogramSize1024 );
        ADD_TEST( framework, Function_Pool_Test::HistogramSize2048 );

        ADD_TEST( framework, Function_Pool_Test::InvertSize256 );
        ADD_TEST( framework, Function_Pool_Test::InvertSize512 );
        ADD_TEST( framework, Function_Pool_Test::InvertSize1024 );
        ADD_TEST( framework, Function_Pool_Test::InvertSize2048 );

        ADD_TEST( framework, Function_Pool_Test::LookupTable256 );
        ADD_TEST( framework, Function_Pool_Test::LookupTable512 );
        ADD_TEST( framework, Function_Pool_Test::LookupTable1024 );
        ADD_TEST( framework, Function_Pool_Test::LookupTable2048 );

        ADD_TEST( framework, Function_Pool_Test::MaximumSize256 );
        ADD_TEST( framework, Function_Pool_Test::MaximumSize512 );
        ADD_TEST( framework, Function_Pool_Test::MaximumSize1024 );
        ADD_TEST( framework, Function_Pool_Test::MaximumSize2048 );

        ADD_TEST( framework, Function_Pool_Test::MinimumSize256 );
        ADD_TEST( framework, Function_Pool_Test::MinimumSize512 );
        ADD_TEST( framework, Function_Pool_Test::MinimumSize1024 );
        ADD_TEST( framework, Function_Pool_Test::MinimumSize2048 );

        ADD_TEST( framework, Function_Pool_Test::ResizeSize256to128 );
        ADD_TEST( framework, Function_Pool_Test::ResizeSize256to512 );
        ADD_TEST( framework, Function_Pool_Test::ResizeSize512to256 );
        ADD_TEST( framework, Function_Pool_Test::ResizeSize512to1024 );
        ADD_TEST( framework, Function_Pool_Test::ResizeSize1024to512 );
        ADD_TEST( framework, Function_Pool_Test::ResizeSize1024to2048 );
        ADD_TEST( framework, Function_Pool_Test::ResizeSize2048to1024 );
        ADD_TEST( framework, Function_Pool_Test::ResizeSize2048to4096 );

        ADD_TEST( framework, Function_Pool_Test::SubtractSize256 );
        ADD_TEST( framework, Function_Pool_Test::SubtractSize512 );
        ADD_TEST( framework, Function_Pool_Test::SubtractSize1024 );
        ADD_TEST( framework, Function_Pool_Test::SubtractSize2048 );

        ADD_TEST( framework, Function_Pool_Test::SumSize256 );
        ADD_TEST( framework, Function_Pool_Test::SumSize512 );
        ADD_TEST( framework, Function_Pool_Test::SumSize1024 );
        ADD_TEST( framework, Function_Pool_Test::SumSize2048 );

        ADD_TEST( framework, Function_Pool_Test::ThresholdSize256 );
        ADD_TEST( framework, Function_Pool_Test::ThresholdSize512 );
        ADD_TEST( framework, Function_Pool_Test::ThresholdSize1024 );
        ADD_TEST( framework, Function_Pool_Test::ThresholdSize2048 );

        ADD_TEST( framework, Function_Pool_Test::ThresholdDoubleSize256 );
        ADD_TEST( framework, Function_Pool_Test::ThresholdDoubleSize512 );
        ADD_TEST( framework, Function_Pool_Test::ThresholdDoubleSize1024 );
        ADD_TEST( framework, Function_Pool_Test::ThresholdDoubleSize2048 );
    }

    namespace Function_Pool_Test
    {
        std::pair < double, double > AbsoluteDifferenceSize256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::AbsoluteDifference( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > AbsoluteDifferenceSize512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::AbsoluteDifference( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > AbsoluteDifferenceSize1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::AbsoluteDifference( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > AbsoluteDifferenceSize2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::AbsoluteDifference( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseAndSize256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::BitwiseAnd( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseAndSize512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::BitwiseAnd( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseAndSize1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::BitwiseAnd( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseAndSize2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::BitwiseAnd( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseOrSize256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::BitwiseOr( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseOrSize512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::BitwiseOr( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseOrSize1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::BitwiseOr( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseOrSize2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::BitwiseOr( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseXorSize256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::BitwiseXor( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseXorSize512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::BitwiseXor( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseXorSize1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::BitwiseXor( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseXorSize2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::BitwiseXor( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > GammaCorrectionSize256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 256, 256 );

            double a     = randomValue <uint32_t>( 100 ) / 100.0;
            double gamma = randomValue <uint32_t>( 300 ) / 100.0;

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::GammaCorrection( image[0], image[1], a, gamma );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > GammaCorrectionSize512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 512, 512 );

            double a     = randomValue <uint32_t>( 100 ) / 100.0;
            double gamma = randomValue <uint32_t>( 300 ) / 100.0;

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::GammaCorrection( image[0], image[1], a, gamma );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > GammaCorrectionSize1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 1024, 1024 );

            double a     = randomValue <uint32_t>( 100 ) / 100.0;
            double gamma = randomValue <uint32_t>( 300 ) / 100.0;

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::GammaCorrection( image[0], image[1], a, gamma );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > GammaCorrectionSize2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 2048, 2048 );

            double a     = randomValue <uint32_t>( 100 ) / 100.0;
            double gamma = randomValue <uint32_t>( 300 ) / 100.0;

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::GammaCorrection( image[0], image[1], a, gamma );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > HistogramSize256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image image = uniformImage( 256, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Histogram( image );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > HistogramSize512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image image = uniformImage( 512, 512 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Histogram( image );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > HistogramSize1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image image = uniformImage( 1024, 1024 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Histogram( image );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > HistogramSize2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image image = uniformImage( 2048, 2048 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Histogram( image );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > InvertSize256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 256, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Invert( image[0], image[1] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > InvertSize512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 512, 512 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Invert( image[0], image[1] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > InvertSize1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 1024, 1024 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Invert( image[0], image[1] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > InvertSize2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 2048, 2048 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Invert( image[0], image[1] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > LookupTable256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 256, 256 );

            std::vector<uint8_t> table(256, 0);

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::LookupTable( image[0], image[1], table );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > LookupTable512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 512, 512 );

            std::vector<uint8_t> table(256, 0);

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::LookupTable( image[0], image[1], table );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > LookupTable1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 1024, 1024 );

            std::vector<uint8_t> table(256, 0);

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::LookupTable( image[0], image[1], table );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > LookupTable2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 2048, 2048 );

            std::vector<uint8_t> table(256, 0);

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::LookupTable( image[0], image[1], table );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MaximumSize256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Maximum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MaximumSize512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Maximum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MaximumSize1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Maximum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MaximumSize2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Maximum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MinimumSize256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Minimum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MinimumSize512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Minimum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MinimumSize1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Minimum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MinimumSize2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Minimum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ResizeSize256to128()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image input  = uniformImage( 256, 256 );
            Bitmap_Image::Image output = uniformImage( 128, 128 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Resize( input, output );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ResizeSize256to512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image input  = uniformImage( 256, 256 );
            Bitmap_Image::Image output = uniformImage( 512, 512 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Resize( input, output );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ResizeSize512to256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image input  = uniformImage( 512, 512 );
            Bitmap_Image::Image output = uniformImage( 256, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Resize( input, output );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ResizeSize512to1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image input  = uniformImage( 512, 512 );
            Bitmap_Image::Image output = uniformImage( 1024, 1024 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Resize( input, output );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ResizeSize1024to512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image input  = uniformImage( 1024, 1024 );
            Bitmap_Image::Image output = uniformImage( 512, 512 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Resize( input, output );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ResizeSize1024to2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image input  = uniformImage( 1024, 1024 );
            Bitmap_Image::Image output = uniformImage( 2048, 2048 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Resize( input, output );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ResizeSize2048to1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image input  = uniformImage( 2048, 2048 );
            Bitmap_Image::Image output = uniformImage( 1024, 1024 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Resize( input, output );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ResizeSize2048to4096()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image input  = uniformImage( 2048, 2048 );
            Bitmap_Image::Image output = uniformImage( 4096, 4096 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Resize( input, output );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SubtractSize256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 256, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Subtract( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SubtractSize512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 512, 512 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Subtract( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SubtractSize1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 1024, 1024 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Subtract( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SubtractSize2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 3, 2048, 2048 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Subtract( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SumSize256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image image = uniformImage( 256, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Sum( image );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SumSize512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image image = uniformImage( 512, 512 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Sum( image );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SumSize1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image image = uniformImage( 1024, 1024 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Sum( image );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SumSize2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            Bitmap_Image::Image image = uniformImage( 2048, 2048 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Sum( image );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdSize256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 256, 256 );
            uint8_t threshold = randomValue<uint8_t>( 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Threshold( image[0], image[1], threshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdSize512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 512, 512 );
            uint8_t threshold = randomValue<uint8_t>( 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Threshold( image[0], image[1], threshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdSize1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 1024, 1024 );
            uint8_t threshold = randomValue<uint8_t>( 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Threshold( image[0], image[1], threshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdSize2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 2048, 2048 );
            uint8_t threshold = randomValue<uint8_t>( 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Threshold( image[0], image[1], threshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdDoubleSize256()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 256, 256 );
            uint8_t minThreshold = randomValue<uint8_t>( 256 );
            uint8_t maxThreshold = randomValue<uint8_t>( minThreshold, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Threshold( image[0], image[1], minThreshold, maxThreshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdDoubleSize512()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 512, 512 );
            uint8_t minThreshold = randomValue<uint8_t>( 256 );
            uint8_t maxThreshold = randomValue<uint8_t>( minThreshold, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Threshold( image[0], image[1], minThreshold, maxThreshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdDoubleSize1024()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 1024, 1024 );
            uint8_t minThreshold = randomValue<uint8_t>( 256 );
            uint8_t maxThreshold = randomValue<uint8_t>( minThreshold, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Threshold( image[0], image[1], minThreshold, maxThreshold );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > ThresholdDoubleSize2048()
        {
            TimerContainer timer;
            setFunctionPoolThreadCount();

            std::vector < Bitmap_Image::Image > image = uniformImages( 2, 2048, 2048 );
            uint8_t minThreshold = randomValue<uint8_t>( 256 );
            uint8_t maxThreshold = randomValue<uint8_t>( minThreshold, 256 );

            for( uint32_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Function_Pool::Threshold( image[0], image[1], minThreshold, maxThreshold );

                timer.stop();
            }

            return timer.mean();
        }
    }
}
