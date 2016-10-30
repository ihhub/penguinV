#include <vector>
#include "../../Library/cuda/image_function_cuda.cuh"
#include "../performance_test_helper.h"
#include "performance_test_image_function_cuda.h"
#include "performance_test_helper_cuda.h"

namespace Performance_Test
{
	void addTests_Image_Function_Cuda(PerformanceTestFramework & framework)
	{
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
	}

	namespace Image_Function_Cuda_Test
	{
		std::pair < double, double > BitwiseAndSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::BitwiseAnd( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseAndSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::BitwiseAnd( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseAndSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::BitwiseAnd( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseAndSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::BitwiseAnd( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseOrSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::BitwiseOr( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseOrSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::BitwiseOr( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseOrSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::BitwiseOr( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseOrSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::BitwiseOr( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseXorSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::BitwiseXor( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseXorSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::BitwiseXor( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseXorSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::BitwiseXor( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseXorSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::BitwiseXor( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > InvertSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::Invert( image[0], image[1] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > InvertSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::Invert( image[0], image[1] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > InvertSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::Invert( image[0], image[1] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > InvertSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image_Cuda::ImageCuda > image = Cuda::uniformImages(3, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function_Cuda::Invert( image[0], image[1] );

				timer.stop();
			}

			return timer.mean();
		}
	};
};
