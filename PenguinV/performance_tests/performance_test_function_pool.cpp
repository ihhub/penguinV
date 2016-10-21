#include "../Library/function_pool.h"
#include "performance_test_function_pool.h"
#include "performance_test_helper.h"

namespace Performance_Test
{
	void addTests_Function_Pool(PerformanceTestFramework & framework)
	{
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
	}

	namespace Function_Pool_Test
	{
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
	};
};
