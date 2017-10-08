#include <vector>
#include "../../Library/cuda/image_function_cuda.cuh"
#include "performance_test_image_function_cuda.h"
#include "performance_test_helper_cuda.cuh"

namespace Performance_Test
{
    void addTests_Image_Function_Cuda( PerformanceTestFramework & framework )
    {
        ADD_TEST( framework, Image_Function_Cuda_Test::AbsoluteDifferenceSize256 );
        ADD_TEST( framework, Image_Function_Cuda_Test::AbsoluteDifferenceSize512 );
        ADD_TEST( framework, Image_Function_Cuda_Test::AbsoluteDifferenceSize1024 );
        ADD_TEST( framework, Image_Function_Cuda_Test::AbsoluteDifferenceSize2048 );

        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseAndSize256 );
        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseAndSize512 );
        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseAndSize1024 );
        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseAndSize2048 );

        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseOrSize256 );
        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseOrSize512 );
        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseOrSize1024 );
        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseOrSize2048 );

        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseXorSize256 );
        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseXorSize512 );
        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseXorSize1024 );
        ADD_TEST( framework, Image_Function_Cuda_Test::BitwiseXorSize2048 );

        ADD_TEST( framework, Image_Function_Cuda_Test::InvertSize256 );
        ADD_TEST( framework, Image_Function_Cuda_Test::InvertSize512 );
        ADD_TEST( framework, Image_Function_Cuda_Test::InvertSize1024 );
        ADD_TEST( framework, Image_Function_Cuda_Test::InvertSize2048 );

        ADD_TEST( framework, Image_Function_Cuda_Test::MaximumSize256 );
        ADD_TEST( framework, Image_Function_Cuda_Test::MaximumSize512 );
        ADD_TEST( framework, Image_Function_Cuda_Test::MaximumSize1024 );
        ADD_TEST( framework, Image_Function_Cuda_Test::MaximumSize2048 );

        ADD_TEST( framework, Image_Function_Cuda_Test::MinimumSize256 );
        ADD_TEST( framework, Image_Function_Cuda_Test::MinimumSize512 );
        ADD_TEST( framework, Image_Function_Cuda_Test::MinimumSize1024 );
        ADD_TEST( framework, Image_Function_Cuda_Test::MinimumSize2048 );

        ADD_TEST( framework, Image_Function_Cuda_Test::SubtractSize256 );
        ADD_TEST( framework, Image_Function_Cuda_Test::SubtractSize512 );
        ADD_TEST( framework, Image_Function_Cuda_Test::SubtractSize1024 );
        ADD_TEST( framework, Image_Function_Cuda_Test::SubtractSize2048 );
    }

    namespace Image_Function_Cuda_Test
    {
        std::pair < double, double > AbsoluteDifferenceSize256()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::AbsoluteDifference( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > AbsoluteDifferenceSize512()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::AbsoluteDifference( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > AbsoluteDifferenceSize1024()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::AbsoluteDifference( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > AbsoluteDifferenceSize2048()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::AbsoluteDifference( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseAndSize256()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::BitwiseAnd( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseAndSize512()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::BitwiseAnd( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseAndSize1024()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::BitwiseAnd( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseAndSize2048()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::BitwiseAnd( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseOrSize256()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::BitwiseOr( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseOrSize512()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::BitwiseOr( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseOrSize1024()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::BitwiseOr( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseOrSize2048()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::BitwiseOr( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseXorSize256()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::BitwiseXor( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseXorSize512()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::BitwiseXor( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseXorSize1024()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::BitwiseXor( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > BitwiseXorSize2048()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::BitwiseXor( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > InvertSize256()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Invert( image[0], image[1] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > InvertSize512()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Invert( image[0], image[1] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > InvertSize1024()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Invert( image[0], image[1] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > InvertSize2048()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Invert( image[0], image[1] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MaximumSize256()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Maximum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MaximumSize512()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Maximum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MaximumSize1024()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Maximum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MaximumSize2048()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Maximum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MinimumSize256()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Minimum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MinimumSize512()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Minimum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MinimumSize1024()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Minimum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > MinimumSize2048()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Minimum( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SubtractSize256()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 256, 256 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Subtract( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SubtractSize512()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 512, 512 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Subtract( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SubtractSize1024()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 1024, 1024 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Subtract( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }

        std::pair < double, double > SubtractSize2048()
        {
            Cuda_Helper::TimerContainerCuda timer;

            std::vector < Bitmap_Image_Cuda::Image > image = Cuda_Helper::uniformImages( 3, 2048, 2048 );

            for( size_t i = 0; i < runCount(); ++i ) {
                timer.start();

                Image_Function_Cuda::Subtract( image[0], image[1], image[2] );

                timer.stop();
            }

            return timer.mean();
        }
    };
};
