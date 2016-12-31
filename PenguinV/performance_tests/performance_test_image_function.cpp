#include "../Library/image_function.h"
#include "performance_test_image_function.h"
#include "performance_test_helper.h"

namespace Performance_Test
{
	void addTests_Image_Function(PerformanceTestFramework & framework)
	{
		ADD_TEST( framework, Image_Function_Test::AbsoluteDifferenceSize256 );
		ADD_TEST( framework, Image_Function_Test::AbsoluteDifferenceSize512 );
		ADD_TEST( framework, Image_Function_Test::AbsoluteDifferenceSize1024 );
		ADD_TEST( framework, Image_Function_Test::AbsoluteDifferenceSize2048 );

		ADD_TEST( framework, Image_Function_Test::BitwiseAndSize256 );
		ADD_TEST( framework, Image_Function_Test::BitwiseAndSize512 );
		ADD_TEST( framework, Image_Function_Test::BitwiseAndSize1024 );
		ADD_TEST( framework, Image_Function_Test::BitwiseAndSize2048 );

		ADD_TEST( framework, Image_Function_Test::BitwiseOrSize256 );
		ADD_TEST( framework, Image_Function_Test::BitwiseOrSize512 );
		ADD_TEST( framework, Image_Function_Test::BitwiseOrSize1024 );
		ADD_TEST( framework, Image_Function_Test::BitwiseOrSize2048 );

		ADD_TEST( framework, Image_Function_Test::BitwiseXorSize256 );
		ADD_TEST( framework, Image_Function_Test::BitwiseXorSize512 );
		ADD_TEST( framework, Image_Function_Test::BitwiseXorSize1024 );
		ADD_TEST( framework, Image_Function_Test::BitwiseXorSize2048 );

		ADD_TEST( framework, Image_Function_Test::ConvertToColorSize256 );
		ADD_TEST( framework, Image_Function_Test::ConvertToColorSize512 );
		ADD_TEST( framework, Image_Function_Test::ConvertToColorSize1024 );
		ADD_TEST( framework, Image_Function_Test::ConvertToColorSize2048 );

		ADD_TEST( framework, Image_Function_Test::ConvertToGrayscaleSize256 );
		ADD_TEST( framework, Image_Function_Test::ConvertToGrayscaleSize512 );
		ADD_TEST( framework, Image_Function_Test::ConvertToGrayscaleSize1024 );
		ADD_TEST( framework, Image_Function_Test::ConvertToGrayscaleSize2048 );

		ADD_TEST( framework, Image_Function_Test::FillSize256 );
		ADD_TEST( framework, Image_Function_Test::FillSize512 );
		ADD_TEST( framework, Image_Function_Test::FillSize1024 );
		ADD_TEST( framework, Image_Function_Test::FillSize2048 );

		ADD_TEST( framework, Image_Function_Test::GammaCorrectionSize256 );
		ADD_TEST( framework, Image_Function_Test::GammaCorrectionSize512 );
		ADD_TEST( framework, Image_Function_Test::GammaCorrectionSize1024 );
		ADD_TEST( framework, Image_Function_Test::GammaCorrectionSize2048 );

		ADD_TEST( framework, Image_Function_Test::HistogramSize256 );
		ADD_TEST( framework, Image_Function_Test::HistogramSize512 );
		ADD_TEST( framework, Image_Function_Test::HistogramSize1024 );
		ADD_TEST( framework, Image_Function_Test::HistogramSize2048 );

		ADD_TEST( framework, Image_Function_Test::InvertSize256 );
		ADD_TEST( framework, Image_Function_Test::InvertSize512 );
		ADD_TEST( framework, Image_Function_Test::InvertSize1024 );
		ADD_TEST( framework, Image_Function_Test::InvertSize2048 );

		ADD_TEST( framework, Image_Function_Test::MaximumSize256 );
		ADD_TEST( framework, Image_Function_Test::MaximumSize512 );
		ADD_TEST( framework, Image_Function_Test::MaximumSize1024 );
		ADD_TEST( framework, Image_Function_Test::MaximumSize2048 );

		ADD_TEST( framework, Image_Function_Test::MinimumSize256 );
		ADD_TEST( framework, Image_Function_Test::MinimumSize512 );
		ADD_TEST( framework, Image_Function_Test::MinimumSize1024 );
		ADD_TEST( framework, Image_Function_Test::MinimumSize2048 );

		ADD_TEST( framework, Image_Function_Test::ResizeSize256to128 );
		ADD_TEST( framework, Image_Function_Test::ResizeSize256to512 );
		ADD_TEST( framework, Image_Function_Test::ResizeSize512to256 );
		ADD_TEST( framework, Image_Function_Test::ResizeSize512to1024 );
		ADD_TEST( framework, Image_Function_Test::ResizeSize1024to512 );
		ADD_TEST( framework, Image_Function_Test::ResizeSize1024to2048 );
		ADD_TEST( framework, Image_Function_Test::ResizeSize2048to1024 );
		ADD_TEST( framework, Image_Function_Test::ResizeSize2048to4096 );

		ADD_TEST( framework, Image_Function_Test::SubtractSize256 );
		ADD_TEST( framework, Image_Function_Test::SubtractSize512 );
		ADD_TEST( framework, Image_Function_Test::SubtractSize1024 );
		ADD_TEST( framework, Image_Function_Test::SubtractSize2048 );

		ADD_TEST( framework, Image_Function_Test::SumSize256 );
		ADD_TEST( framework, Image_Function_Test::SumSize512 );
		ADD_TEST( framework, Image_Function_Test::SumSize1024 );
		ADD_TEST( framework, Image_Function_Test::SumSize2048 );

		ADD_TEST( framework, Image_Function_Test::ThresholdSize256 );
		ADD_TEST( framework, Image_Function_Test::ThresholdSize512 );
		ADD_TEST( framework, Image_Function_Test::ThresholdSize1024 );
		ADD_TEST( framework, Image_Function_Test::ThresholdSize2048 );

		ADD_TEST( framework, Image_Function_Test::ThresholdDoubleSize256 );
		ADD_TEST( framework, Image_Function_Test::ThresholdDoubleSize512 );
		ADD_TEST( framework, Image_Function_Test::ThresholdDoubleSize1024 );
		ADD_TEST( framework, Image_Function_Test::ThresholdDoubleSize2048 );
	}

	namespace Image_Function_Test
	{
		std::pair < double, double > AbsoluteDifferenceSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::AbsoluteDifference( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > AbsoluteDifferenceSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::AbsoluteDifference( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > AbsoluteDifferenceSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::AbsoluteDifference( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > AbsoluteDifferenceSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::AbsoluteDifference( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseAndSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::BitwiseAnd( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseAndSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::BitwiseAnd( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseAndSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::BitwiseAnd( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseAndSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::BitwiseAnd( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseOrSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::BitwiseOr( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseOrSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::BitwiseOr( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseOrSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::BitwiseOr( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseOrSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::BitwiseOr( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseXorSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::BitwiseXor( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseXorSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::BitwiseXor( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseXorSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::BitwiseXor( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > BitwiseXorSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::BitwiseXor( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ConvertToColorSize256()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformImage     (256, 256);
			Bitmap_Image::Image output = uniformColorImage(256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::ConvertToRgb( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ConvertToColorSize512()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformImage     (512, 512);
			Bitmap_Image::Image output = uniformColorImage(512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::ConvertToRgb( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ConvertToColorSize1024()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformImage     (1024, 1024);
			Bitmap_Image::Image output = uniformColorImage(1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::ConvertToRgb( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ConvertToColorSize2048()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformImage     (2048, 2048);
			Bitmap_Image::Image output = uniformColorImage(2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::ConvertToRgb( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ConvertToGrayscaleSize256()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformColorImage(256, 256);
			Bitmap_Image::Image output = uniformImage     (256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::ConvertToGrayScale( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ConvertToGrayscaleSize512()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformColorImage(512, 512);
			Bitmap_Image::Image output = uniformImage     (512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::ConvertToGrayScale( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ConvertToGrayscaleSize1024()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformColorImage(1024, 1024);
			Bitmap_Image::Image output = uniformImage     (1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::ConvertToGrayScale( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ConvertToGrayscaleSize2048()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformColorImage(2048, 2048);
			Bitmap_Image::Image output = uniformImage     (2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::ConvertToGrayScale( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > FillSize256()
		{
			TimerContainer timer;

			Bitmap_Image::Image image = uniformImage(256, 256);
			uint8_t value = randomValue<uint8_t>(256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Fill( image, value );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > FillSize512()
		{
			TimerContainer timer;

			Bitmap_Image::Image image = uniformImage(512, 512);
			uint8_t value = randomValue<uint8_t>(256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Fill( image, value );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > FillSize1024()
		{
			TimerContainer timer;

			Bitmap_Image::Image image = uniformImage(1024, 1024);
			uint8_t value = randomValue<uint8_t>(256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Fill( image, value );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > FillSize2048()
		{
			TimerContainer timer;

			Bitmap_Image::Image image = uniformImage(2048, 2048);
			uint8_t value = randomValue<uint8_t>(256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Fill( image, value );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > GammaCorrectionSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 256, 256);

			double a     = randomValue <uint32_t>(100) / 100.0;
			double gamma = randomValue <uint32_t>(300) / 100.0;

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::GammaCorrection( image[0], image[1], a, gamma );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > GammaCorrectionSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 512, 512);

			double a     = randomValue <uint32_t>(100) / 100.0;
			double gamma = randomValue <uint32_t>(300) / 100.0;

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::GammaCorrection( image[0], image[1], a, gamma );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > GammaCorrectionSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 1024, 1024);

			double a     = randomValue <uint32_t>(100) / 100.0;
			double gamma = randomValue <uint32_t>(300) / 100.0;

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::GammaCorrection( image[0], image[1], a, gamma );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > GammaCorrectionSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 2048, 2048);

			double a     = randomValue <uint32_t>(100) / 100.0;
			double gamma = randomValue <uint32_t>(300) / 100.0;

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::GammaCorrection( image[0], image[1], a, gamma );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > HistogramSize256()
		{
			TimerContainer timer;

			Bitmap_Image::Image image = uniformImage(256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Histogram( image );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > HistogramSize512()
		{
			TimerContainer timer;

			Bitmap_Image::Image image = uniformImage(512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Histogram( image );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > HistogramSize1024()
		{
			TimerContainer timer;

			Bitmap_Image::Image image = uniformImage(1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Histogram( image );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > HistogramSize2048()
		{
			TimerContainer timer;

			Bitmap_Image::Image image = uniformImage(2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Histogram( image );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > InvertSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Invert( image[0], image[1] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > InvertSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Invert( image[0], image[1] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > InvertSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Invert( image[0], image[1] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > InvertSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Invert( image[0], image[1] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > MaximumSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Maximum( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > MaximumSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Maximum( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > MaximumSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Maximum( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > MaximumSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Maximum( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > MinimumSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Minimum( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > MinimumSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Minimum( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > MinimumSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Minimum( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > MinimumSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Minimum( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ResizeSize256to128()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformImage(256, 256);
			Bitmap_Image::Image output = uniformImage(128, 128);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Resize( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ResizeSize256to512()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformImage(256, 256);
			Bitmap_Image::Image output = uniformImage(512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Resize( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ResizeSize512to256()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformImage(512, 512);
			Bitmap_Image::Image output = uniformImage(256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Resize( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ResizeSize512to1024()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformImage(512, 512);
			Bitmap_Image::Image output = uniformImage(1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Resize( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ResizeSize1024to512()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformImage(1024, 1024);
			Bitmap_Image::Image output = uniformImage(512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Resize( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ResizeSize1024to2048()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformImage(1024, 1024);
			Bitmap_Image::Image output = uniformImage(2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Resize( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ResizeSize2048to1024()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformImage(2048, 2048);
			Bitmap_Image::Image output = uniformImage(1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Resize( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ResizeSize2048to4096()
		{
			TimerContainer timer;

			Bitmap_Image::Image input  = uniformImage(2048, 2048);
			Bitmap_Image::Image output = uniformImage(4096, 4096);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Resize( input, output );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > SubtractSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Subtract( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > SubtractSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Subtract( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > SubtractSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Subtract( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > SubtractSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(3, 2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Subtract( image[0], image[1], image[2] );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > SumSize256()
		{
			TimerContainer timer;

			Bitmap_Image::Image image = uniformImage(256, 256);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Sum( image );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > SumSize512()
		{
			TimerContainer timer;

			Bitmap_Image::Image image = uniformImage(512, 512);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Sum( image );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > SumSize1024()
		{
			TimerContainer timer;

			Bitmap_Image::Image image = uniformImage(1024, 1024);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Sum( image );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > SumSize2048()
		{
			TimerContainer timer;

			Bitmap_Image::Image image = uniformImage(2048, 2048);

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Sum( image );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ThresholdSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 256, 256);
			uint8_t threshold = randomValue<uint8_t>( 256 );

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Threshold( image[0], image[1], threshold );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ThresholdSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 512, 512);
			uint8_t threshold = randomValue<uint8_t>( 256 );

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Threshold( image[0], image[1], threshold );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ThresholdSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 1024, 1024);
			uint8_t threshold = randomValue<uint8_t>( 256 );

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Threshold( image[0], image[1], threshold );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ThresholdSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 2048, 2048);
			uint8_t threshold = randomValue<uint8_t>( 256 );

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Threshold( image[0], image[1], threshold );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ThresholdDoubleSize256()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 256, 256);
			uint8_t minThreshold = randomValue<uint8_t>( 256 );
			uint8_t maxThreshold = randomValue<uint8_t>( minThreshold, 256 );

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Threshold( image[0], image[1], minThreshold, maxThreshold );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ThresholdDoubleSize512()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 512, 512);
			uint8_t minThreshold = randomValue<uint8_t>( 256 );
			uint8_t maxThreshold = randomValue<uint8_t>( minThreshold, 256 );

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Threshold( image[0], image[1], minThreshold, maxThreshold );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ThresholdDoubleSize1024()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 1024, 1024);
			uint8_t minThreshold = randomValue<uint8_t>( 256 );
			uint8_t maxThreshold = randomValue<uint8_t>( minThreshold, 256 );

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Threshold( image[0], image[1], minThreshold, maxThreshold );

				timer.stop();
			}

			return timer.mean();
		}

		std::pair < double, double > ThresholdDoubleSize2048()
		{
			TimerContainer timer;

			std::vector < Bitmap_Image::Image > image = uniformImages(2, 2048, 2048);
			uint8_t minThreshold = randomValue<uint8_t>( 256 );
			uint8_t maxThreshold = randomValue<uint8_t>( minThreshold, 256 );

			for( uint32_t i = 0; i < runCount(); ++i ) {
				timer.start();

				Image_Function::Threshold( image[0], image[1], minThreshold, maxThreshold );

				timer.stop();
			}

			return timer.mean();
		}
	};
};
