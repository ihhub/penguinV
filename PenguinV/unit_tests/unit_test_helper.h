#pragma once

#include <algorithm>
#include <cstdlib>
#include <vector>
#include "../Library/image_buffer.h"
#include "../Library/image_exception.h"

// A bunch of functions to help writing unit tests
namespace Unit_Test
{
	// Generate images
	Bitmap_Image::Image uniformImage();
	Bitmap_Image::Image uniformImage(uint8_t value);
	Bitmap_Image::ColorImage uniformColorImage();
	Bitmap_Image::ColorImage uniformColorImage(uint8_t value);
	Bitmap_Image::Image blackImage();
	Bitmap_Image::Image whiteImage();
	Bitmap_Image::Image randomImage();
	Bitmap_Image::Image randomImage(const std::vector <uint8_t> & value);
	std::vector < Bitmap_Image::Image > uniformImages( uint32_t images );
	std::vector < Bitmap_Image::Image > uniformImages( std::vector < uint8_t > intensityValue );
	
	// Generate pixel intensity values
	uint8_t intensityValue();
	std::vector < uint8_t > intensityArray( uint32_t size );

	// Image size and ROI verification
	template <typename data>
	bool equalSize( const data & image1, const data & image2 )
	{
		return image1.height() == image2.height() && image1.width() == image2.width() &&
			   image1.alignment() == image2.alignment() && image1.colorCount() == image2.colorCount()
			   && image1.rowSize() == image2.rowSize();
	};

	template <typename data>
	bool equalSize( const data & image, uint32_t width, uint32_t height, uint32_t rowSize, uint8_t colorCount,
					uint8_t alignment )
	{
		return ( (width == 0 || height == 0) && image.data() == nullptr && image.width() == 0 && image.height() == 0 &&
				  image.colorCount() == colorCount && image.alignment() == alignment && image.rowSize() == 0 ) ||
				( width == image.width() && height == image.height() && colorCount == image.colorCount() &&
				  alignment == image.alignment() && rowSize == image.rowSize() );
	};

	bool equalSize( const Bitmap_Image::Image & image, uint32_t width, uint32_t height );

	template <typename data>
	bool equalData( const Template_Image::ImageTemplate < data > & image1, const Template_Image::ImageTemplate < data > & image2 )
	{
		return memcmp(image1.data(), image2.data(), sizeof(data) * image1.height() * image1.rowSize()) == 0;
	};

	template <typename data>
	bool isEmpty( const Template_Image::ImageTemplate < data > & image)
	{
		return image.data() == nullptr && image.width() == 0 && image.height() == 0 &&
			   image.colorCount() == 1 && image.alignment() == 1 && image.rowSize() == 0;
	};

	template <uint8_t byteCount>
	bool verifyImage( const Bitmap_Image::BitmapImage < byteCount > & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value )
	{
		if( image.empty() || width == 0 || height == 0 || x + width > image.width() || y + height > image.height() )
			throw imageException("Bad input parameters in image function");

		const uint8_t * outputY = image.data() + y * image.rowSize() + x * image.colorCount();
		const uint8_t * endY    = outputY + image.rowSize() * height;

		for( ; outputY != endY; outputY += image.rowSize() ) {
			const uint8_t * outputX = outputY;
			const uint8_t * endX    = outputX + width * image.colorCount();

			for( ; outputX != endX; ++outputX ) {
				if( (*outputX) != value )
					return false;
			}
		}

		return true;
	};

	template <uint8_t byteCount>
	bool verifyImage( const Bitmap_Image::BitmapImage < byteCount > & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
					  const std::vector < uint8_t > & value )
	{
		if( image.empty() || width == 0 || height == 0 || x + width > image.width() || y + height > image.height() )
			throw imageException("Bad input parameters in image function");

		const uint8_t * outputY = image.data() + y * image.rowSize() + x * image.colorCount();
		const uint8_t * endY    = outputY + image.rowSize() * height;

		for( ; outputY != endY; outputY += image.rowSize() ) {
			const uint8_t * outputX = outputY;
			const uint8_t * endX    = outputX + width * image.colorCount();

			for( ; outputX != endX; ++outputX ) {
				bool equal = false;
				
				for( std::vector < uint8_t >::const_iterator v = value.begin(); v != value.end(); ++v ) {
					if( (*outputX) == (*v) ) {
						equal = true;
						break;
					}
				}

				if( !equal )
					return false;
			}
		}

		return true;
	};

	template <uint8_t byteCount>
	bool verifyImage( const Bitmap_Image::BitmapImage < byteCount > & image, uint8_t value )
	{
		return verifyImage( image, 0, 0, image.width(), image.height(), value );
	};

	template <uint8_t byteCount>
	bool verifyImage( const Bitmap_Image::BitmapImage < byteCount > & image, const std::vector < uint8_t > & value )
	{
		return verifyImage( image, 0, 0, image.width(), image.height(), value );
	};

	// Fill image ROI with specific intensity
	void fillImage( Bitmap_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value );
	void fillImage( Bitmap_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
					const std::vector < uint8_t > & value );

	// Generate and return ROI based on full image size
	void generateRoi( const Bitmap_Image::Image & image, uint32_t & x, uint32_t & y, uint32_t & width, uint32_t & height );
	void generateRoi( const std::vector < Bitmap_Image::Image > & image, std::vector < uint32_t > & x, std::vector < uint32_t > & y,
					  uint32_t & width, uint32_t & height );
	// first element in pair structure is width, second - height
	void generateRoi( const std::vector < std::pair< uint32_t, uint32_t > > & imageSize, std::vector < uint32_t > & x,
					  std::vector < uint32_t > & y, uint32_t & width, uint32_t & height );

	template <uint8_t byteCount>
	std::pair <uint32_t, uint32_t> imageSize( const Bitmap_Image::BitmapImage < byteCount > & image )
	{
		return std::pair <uint32_t, uint32_t>( image.width(), image.height() );
	};

	// Return calculated row size
	uint32_t rowSize(uint32_t width, uint8_t colorCount = 1, uint8_t alignment = 1);

	uint32_t runCount(); // fixed value for all test loops

	// Return random value for specific range or variable type
	template <typename data>
	data randomValue(int maximum)
	{
		if( maximum <= 0 )
			return 0;
		else
			return static_cast<data>( rand() ) % maximum;
	};

	template <typename data>
	data randomValue(data minimum, int maximum)
	{
		if( maximum <= 0 ) {
			return 0;
		}
		else {
			data value = static_cast<data>( rand() ) % maximum;

			if( value < minimum )
				value = minimum;

			return value;
		}
	};
};
