#include "unit_test_image_buffer.h"
#include "unit_test_helper.h"
#include "../Library/image_function.h"

namespace Unit_Test
{
	void addTests_Image_Buffer(UnitTestFramework & framework)
	{
		ADD_TEST( framework, ImageTemplateEmptyConstructorTest );

		ADD_TEST( framework, ImageTemplateConstructor2ParametersTest );

		ADD_TEST( framework, ImageTemplateConstructor3ParametersTest );

		ADD_TEST( framework, ImageTemplateConstructor4ParametersTest );

		ADD_TEST( framework, ImageTemplateCopyConstructorU8Test );
		ADD_TEST( framework, ImageTemplateCopyConstructorU16Test );
		ADD_TEST( framework, ImageTemplateCopyConstructorU32Test );
		ADD_TEST( framework, ImageTemplateCopyConstructorU64Test );
		ADD_TEST( framework, ImageTemplateCopyConstructorS8Test );
		ADD_TEST( framework, ImageTemplateCopyConstructorS16Test );
		ADD_TEST( framework, ImageTemplateCopyConstructorS32Test );
		ADD_TEST( framework, ImageTemplateCopyConstructorS64Test );
		ADD_TEST( framework, ImageTemplateCopyConstructorFTest );
		ADD_TEST( framework, ImageTemplateCopyConstructorDTest );

		ADD_TEST( framework, ImageTemplateNullAssignmentTest );

		ADD_TEST( framework, ImageTemplateAssignmentOperatorU8Test );
		ADD_TEST( framework, ImageTemplateAssignmentOperatorU16Test );
		ADD_TEST( framework, ImageTemplateAssignmentOperatorU32Test );
		ADD_TEST( framework, ImageTemplateAssignmentOperatorU64Test );
		ADD_TEST( framework, ImageTemplateAssignmentOperatorS8Test );
		ADD_TEST( framework, ImageTemplateAssignmentOperatorS16Test );
		ADD_TEST( framework, ImageTemplateAssignmentOperatorS32Test );
		ADD_TEST( framework, ImageTemplateAssignmentOperatorS64Test );
		ADD_TEST( framework, ImageTemplateAssignmentOperatorFTest );
		ADD_TEST( framework, ImageTemplateAssignmentOperatorDTest );
	}

	bool ImageTemplateEmptyConstructorTest()
	{
		Template_Image::ImageTemplate < uint8_t > image;

		return isEmpty( image );
	}

	bool ImageTemplateConstructor2ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(2048);
			uint32_t height     = randomValue<uint32_t>(2048);
	
			if( !equalSize( Template_Image::ImageTemplate < uint8_t >( width, height ), width, height, rowSize(width), 1, 1 ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateConstructor3ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(2048);
			uint32_t height     = randomValue<uint32_t>(2048);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
	
			if( !equalSize( Template_Image::ImageTemplate < uint8_t >( width, height, colorCount ), width, height,
				rowSize(width, colorCount), colorCount, 1 ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateConstructor4ParametersTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(2048);
			uint32_t height     = randomValue<uint32_t>(2048);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);
	
			if( !equalSize( Template_Image::ImageTemplate < uint8_t >( width, height, colorCount, alignment ), width, height,
							rowSize(width, colorCount, alignment), colorCount, alignment ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateCopyConstructorU8Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < uint8_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < uint8_t > image_copy( image );
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateCopyConstructorU16Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < uint16_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < uint16_t > image_copy( image );
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateCopyConstructorU32Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < uint32_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < uint32_t > image_copy( image );
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateCopyConstructorU64Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < uint64_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < uint64_t > image_copy( image );
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateCopyConstructorS8Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < int8_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < int8_t > image_copy( image );
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateCopyConstructorS16Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < int16_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < int16_t > image_copy( image );
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateCopyConstructorS32Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < int32_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < int32_t > image_copy( image );
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateCopyConstructorS64Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < int64_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < int64_t > image_copy( image );
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateCopyConstructorFTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < float > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < float > image_copy( image );
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateCopyConstructorDTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < double > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < double > image_copy( image );
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateNullAssignmentTest()
	{
		try {
			Template_Image::ImageTemplate < uint8_t > image;

			uint8_t fakeArray[1];
			uint8_t fakeValue = rand() % 2;
			if( fakeValue == 1 )
				fakeValue = 0;

			image.assign( fakeArray, fakeValue, fakeValue, fakeValue, fakeValue );
		}
		catch(imageException &) {
			return true;
		}
		
		return false;
	}

	bool ImageTemplateAssignmentOperatorU8Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < uint8_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < uint8_t > image_copy;

			image_copy = image;
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateAssignmentOperatorU16Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < uint16_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < uint16_t > image_copy;

			image_copy = image;
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateAssignmentOperatorU32Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < uint32_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < uint32_t > image_copy;

			image_copy = image;
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateAssignmentOperatorU64Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < uint64_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < uint64_t > image_copy;

			image_copy = image;
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateAssignmentOperatorS8Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < int8_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < int8_t > image_copy;

			image_copy = image;
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateAssignmentOperatorS16Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < int16_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < int16_t > image_copy;

			image_copy = image;
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateAssignmentOperatorS32Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < int32_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < int32_t > image_copy;

			image_copy = image;
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateAssignmentOperatorS64Test()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < int64_t > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < int64_t > image_copy;

			image_copy = image;
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateAssignmentOperatorFTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < float > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < float > image_copy;

			image_copy = image;
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}

	bool ImageTemplateAssignmentOperatorDTest()
	{
		for( uint32_t i = 0; i < runCount(); ++i ) {
			uint32_t width      = randomValue<uint32_t>(1024);
			uint32_t height     = randomValue<uint32_t>(1024);
			uint8_t  colorCount = randomValue<uint8_t >(1, 4);
			uint8_t  alignment  = randomValue<uint8_t >(1, 32);

			Template_Image::ImageTemplate < double > image(width, height, colorCount, alignment);
			Template_Image::ImageTemplate < double > image_copy;

			image_copy = image;
	
			if( !equalSize( image, image_copy ) || !equalData( image, image_copy ) )
				return false;
		}

		return true;
	}
};
