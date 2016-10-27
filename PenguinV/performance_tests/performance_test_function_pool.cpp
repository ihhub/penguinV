#include "../Library/function_pool.h"
#include "performance_test_function_pool.h"
#include "performance_test_helper.h"

namespace Performance_Test
{
	void addTests_Function_Pool(PerformanceTestFramework & framework)
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

		ADD_TEST( framework, Function_Pool_Test::InvertSize256 );
		ADD_TEST( framework, Function_Pool_Test::InvertSize512 );
		ADD_TEST( framework, Function_Pool_Test::InvertSize1024 );
		ADD_TEST( framework, Function_Pool_Test::InvertSize2048 );

		ADD_TEST( framework, Function_Pool_Test::MaximumSize256 );
		ADD_TEST( framework, Function_Pool_Test::MaximumSize512 );
		ADD_TEST( framework, Function_Pool_Test::MaximumSize1024 );
		ADD_TEST( framework, Function_Pool_Test::MaximumSize2048 );

		ADD_TEST( framework, Function_Pool_Test::MinimumSize256 );
		ADD_TEST( framework, Function_Pool_Test::MinimumSize512 );
		ADD_TEST( framework, Function_Pool_Test::MinimumSize1024 );
		ADD_TEST( framework, Function_Pool_Test::MinimumSize2048 );

		ADD_TEST( framework, Function_Pool_Test::SubtractSize256 );
		ADD_TEST( framework, Function_Pool_Test::SubtractSize512 );
		ADD_TEST( framework, Function_Pool_Test::SubtractSize1024 );
		ADD_TEST( framework, Function_Pool_Test::SubtractSize2048 );

		ADD_TEST( framework, Function_Pool_Test::SumSize256 );
		ADD_TEST( framework, Function_Pool_Test::SumSize512 );
		ADD_TEST( framework, Function_Pool_Test::SumSize1024 );
		ADD_TEST( framework, Function_Pool_Test::SumSize2048 );
	}

	namespace Function_Pool_Test
	{
		std::pair < double, double > AbsoluteDifferenceSize256()
		{
			TimerContainer timer;
			setFunctionPoolThreadCount();

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Function_Pool::BitwiseXor( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > InvertSize256()
		{
			TimerContainer timer;
			setFunctionPoolThreadCount();

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 256, 256);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 512, 512);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 1024, 1024);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Function_Pool::Invert( image[0], image[1] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > MaximumSize256()
		{
			TimerContainer timer;
			setFunctionPoolThreadCount();

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Function_Pool::Minimum( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > SubtractSize256()
		{
			TimerContainer timer;
			setFunctionPoolThreadCount();

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

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

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

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

			Bitmap_Image::Image image = uniformImage(256, 256);

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

			Bitmap_Image::Image image = uniformImage(512, 512);

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

			Bitmap_Image::Image image = uniformImage(1024, 1024);

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

			Bitmap_Image::Image image = uniformImage(2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Function_Pool::Sum( image );

				timer.stop();
			}

			return timer.mean();
		}
	};
};
