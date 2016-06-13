#pragma once

#include "image_buffer.h"
#include <vector>

namespace Image_Function
{
	using namespace Bitmap_Image;

	template <uint8_t bytes>
	void ParameterValidation( const BitmapImage <bytes> & image1 );
	template <uint8_t bytes1, uint8_t bytes2>
	void ParameterValidation( const BitmapImage <bytes1> & image1, const BitmapImage <bytes2> & image2 );
	template <uint8_t bytes1, uint8_t bytes2, uint8_t bytes3>
	void ParameterValidation( const BitmapImage <bytes1> & image1, const BitmapImage <bytes2> & image2, const BitmapImage <bytes3> & image3 );
	template <uint8_t bytes>
	void ParameterValidation( const BitmapImage <bytes> & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height );
	template <uint8_t bytes1, uint8_t bytes2>
	void ParameterValidation( const BitmapImage <bytes1> & image1, uint32_t startX1, uint32_t startY1, const BitmapImage <bytes2> & image2,
							  uint32_t startX2, uint32_t startY2, uint32_t width, uint32_t height );
	template <uint8_t bytes1, uint8_t bytes2, uint8_t bytes3>
	void ParameterValidation( const BitmapImage <bytes1> & image1, uint32_t startX1, uint32_t startY1, const BitmapImage <bytes2> & image2,
							  uint32_t startX2, uint32_t startY2, const BitmapImage <bytes3> & image3, uint32_t startX3, uint32_t startY3,
							  uint32_t width, uint32_t height );


	Image BitwiseAnd( const Image & in1, const Image & in2 );
	void  BitwiseAnd( const Image & in1, const Image & in2, Image & out );
	Image BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					  uint32_t width, uint32_t height );
	void  BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

	Image BitwiseOr( const Image & in1, const Image & in2 );
	void  BitwiseOr( const Image & in1, const Image & in2, Image & out );
	Image BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 uint32_t width, uint32_t height );
	void  BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

	Image BitwiseXor( const Image & in1, const Image & in2 );
	void  BitwiseXor( const Image & in1, const Image & in2, Image & out );
	Image BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					  uint32_t width, uint32_t height );
	void  BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

	// ImageInvert function is Bitwise NOT operation. But to make function name more user-friendly we named it like this
	Image Invert( const Image & in );
	void  Invert( const Image & in, Image & out );
	Image Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
	void  Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
				  uint32_t width, uint32_t height );	

	void  Copy( const Image & in, Image & out );
	Image Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
	void  Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
				uint32_t width, uint32_t height );

	uint8_t GetPixel( const Image & image, uint32_t x, uint32_t y );
	void    SetPixel( Image & image, uint32_t x, uint32_t y, uint8_t value );

	void Convert( const Image & in, ColorImage & out );
	void Convert( const ColorImage & in, Image & out );
	void Convert( const Image & in, uint32_t startXIn, uint32_t startYIn, ColorImage & out, uint32_t startXOut, uint32_t startYOut,
				  uint32_t width, uint32_t height );
	void Convert( const ColorImage & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
				  uint32_t width, uint32_t height );

	// Thresholding works in such way:
		// if pixel intensity on input image is          less (  < ) than threshold then set pixel intensity on output image as 0
		// if pixel intensity on input image is equal or more ( >= ) than threshold then set pixel intensity on output image as 255
	Image Threshold( const Image & in, uint8_t threshold );
	void  Threshold( const Image & in, Image & out, uint8_t threshold );
	Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t threshold );
	void  Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					 uint32_t width, uint32_t height, uint8_t threshold );
	
	std::vector < uint32_t > Histogram( const Image & image );
	void                     Histogram( const Image & image, std::vector < uint32_t > & histogram );
	std::vector < uint32_t > Histogram( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height );
	void                     Histogram( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height,
										std::vector < uint32_t > & histogram );

	Image Subtract( const Image & in1, const Image & in2 );
	void  Subtract( const Image & in1, const Image & in2, Image & out );
	Image Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					uint32_t width, uint32_t height );
	void  Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

	Image Minimum( const Image & in1, const Image & in2 );
	void  Minimum( const Image & in1, const Image & in2, Image & out );
	Image Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   uint32_t width, uint32_t height );
	void  Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

	Image Maximum( const Image & in1, const Image & in2 );
	void  Maximum( const Image & in1, const Image & in2, Image & out );
	Image Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   uint32_t width, uint32_t height );
	void  Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height );

	Image Normalize( const Image & in );
	void  Normalize( const Image & in, Image & out );
	Image Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height );
	void  Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					 uint32_t width, uint32_t height );

	// Make sure that your image is not so big to do not have overloaded value
	uint32_t Sum( const Image & image );
	uint32_t Sum( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height );

	void Fill( Image & image, uint8_t value );
	void Fill( Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height, uint8_t value );
};