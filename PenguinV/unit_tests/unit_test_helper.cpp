#include "unit_test_helper.h"
#include "../Library/image_function.h"

namespace Unit_Test
{
	Bitmap_Image::Image uniformImage(uint8_t value)
	{
		Bitmap_Image::Image image( randomValue<uint32_t>( 1, 2048 ), randomValue<uint32_t>( 1, 2048 ) );

		image.fill( value );

		return image;
	}

	Bitmap_Image::Image uniformImage()
	{
		return uniformImage( randomValue<uint8_t>( 256 ) );
	}

	Bitmap_Image::Image blackImage()
	{
		return uniformImage( 0u );
	}

	Bitmap_Image::Image whiteImage()
	{
		return uniformImage( 255u );
	}

	Bitmap_Image::Image randomImage()
	{
		Bitmap_Image::Image image( randomValue<uint32_t>( 1, 2048 ), randomValue<uint32_t>( 1, 2048 ) );

		uint8_t * outY = image.data();
		const uint8_t * outYEnd = outY + image.height() * image.rowSize();

		for( ; outY != outYEnd; outY += image.rowSize() ) {
			uint8_t * outX = outY;
			const uint8_t * outXEnd = outX + image.width();

			for( ; outX != outXEnd; ++outX )
				(*outX) = randomValue<uint8_t>( 256 );
		}

		return image;
	}

	std::vector < Bitmap_Image::Image > uniformImages( uint32_t images )
	{
		if( images == 0 )
			throw imageException( "Invalid parameter" );

		std::vector < Bitmap_Image::Image > image;

		image.push_back( uniformImage() );

		image.resize( images );

		for( size_t i = 1; i < image.size(); ++i ) {
			image[i].resize( image[0].width(), image[0].height() );
			image[i].fill( randomValue<uint8_t>( 256 ) );
		}

		return image;
	}

	template <typename data>
	std::vector < data > generateArray( uint32_t size, int maximumValue )
	{
		std::vector < data > fillArray( size );

		std::for_each( fillArray.begin(), fillArray.end(), [&](data & value){ value = randomValue<data>( maximumValue ); } );

		return fillArray;
	}

	uint8_t intensityValue()
	{
		return randomValue<uint8_t>(255);
	}

	std::vector < uint8_t > intensityArray( uint32_t size )
	{
		return generateArray<uint8_t>( size, 256 );
	}

	std::vector < Bitmap_Image::Image > uniformImages( std::vector < uint8_t > intensityValue )
	{
		if( intensityValue.size() == 0 )
			throw imageException( "Invalid parameter" );

		std::vector < Bitmap_Image::Image > image;

		image.push_back( uniformImage(intensityValue[0]) );

		image.resize( intensityValue.size() );

		for( size_t i = 1; i < image.size(); ++i ) {
			image[i].resize( image[0].width(), image[0].height() );
			image[i].fill( intensityValue[i] );
		}

		return image;
	}

	bool equalSize( const Bitmap_Image::Image & image, uint32_t width, uint32_t height )
	{
		return image.width() == width && image.height() == height && !image.empty();
	}

	bool verifyImage( const Bitmap_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value )
	{
		Image_Function::ParameterValidation( image, x, y, width, height );

		const uint8_t * outputY = image.data() + y * image.rowSize() + x;
		const uint8_t * endY    = outputY + image.rowSize() * height;

		for( ; outputY != endY; outputY += image.rowSize() ) {
			if( std::any_of( outputY, outputY + width,
				[value](const uint8_t & v){ return v != value; } ) )
				return false;
		}

		return true;
	}

	bool verifyImage( const Bitmap_Image::Image & image, uint8_t value )
	{
		return verifyImage( image, 0, 0, image.width(), image.height(), value );
	}

	void fillImage( Bitmap_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value )
	{
		Image_Function::Fill( image, x, y, width, height, value );
	}

	void generateRoi( const Bitmap_Image::Image & image, uint32_t & x, uint32_t & y, uint32_t & width, uint32_t & height )
	{
		width  = randomValue<uint32_t>( 1, image.width()  + 1 );
		height = randomValue<uint32_t>( 1, image.height() + 1 );

		x = randomValue<uint32_t>( image.width()  - width  );
		y = randomValue<uint32_t>( image.height() - height );
	}

	void generateRoi( const std::vector < Bitmap_Image::Image > & image, std::vector < uint32_t > & x, std::vector < uint32_t > & y,
					  uint32_t & width, uint32_t & height )
	{
		uint32_t maximumWidth  = 0;
		uint32_t maximumHeight = 0;

		for( std::vector < Bitmap_Image::Image >::const_iterator im = image.begin(); im != image.end(); ++im ) {
			if( maximumWidth == 0 )
				maximumWidth = im->width();
			else if( maximumWidth > im->width() )
				maximumWidth = im->width();

			if( maximumHeight == 0 )
				maximumHeight = im->height();
			else if( maximumHeight > im->height() )
				maximumHeight = im->height();
		}

		width  = randomValue<uint32_t>( 1, maximumWidth  + 1 );
		height = randomValue<uint32_t>( 1, maximumHeight + 1 );

		x.resize( image.size() );
		y.resize( image.size() );

		for( size_t i = 0; i < image.size(); ++i ) {
			x[i] = randomValue<uint32_t>( image[i].width()  - width  );
			y[i] = randomValue<uint32_t>( image[i].height() - height );
		}
	}

	uint32_t rowSize(uint32_t width, uint8_t colorCount, uint8_t alignment)
	{
		uint32_t size = width * colorCount;
		if( size % alignment != 0 )
			size = ((size / alignment) + 1) * alignment;

		return size;
	}

	uint32_t runCount()
	{
		return 1024; // some magic number for loop. Higher value = higher chance to verify all possible situations
	}
};
