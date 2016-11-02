#pragma once

#include <stdint.h>
#include "../image_buffer.h"
#include "image_buffer_cuda.cuh"

namespace Image_Function_Cuda
{
	using namespace Bitmap_Image_Cuda;

	template <uint8_t bytes>
	void ParameterValidation( const BitmapImageCuda <bytes> & image1 );
	template <uint8_t bytes1, uint8_t bytes2>
	void ParameterValidation( const BitmapImageCuda <bytes1> & image1, const BitmapImageCuda <bytes2> & image2 );
	template <uint8_t bytes1, uint8_t bytes2, uint8_t bytes3>
	void ParameterValidation( const BitmapImageCuda <bytes1> & image1, const BitmapImageCuda <bytes2> & image2, const BitmapImageCuda <bytes3> & image3 );

	ImageCuda AbsoluteDifference( const ImageCuda & in1, const ImageCuda & in2 );
	void      AbsoluteDifference( const ImageCuda & in1, const ImageCuda & in2, ImageCuda & out );

	ImageCuda BitwiseAnd( const ImageCuda & in1, const ImageCuda & in2 );
	void      BitwiseAnd( const ImageCuda & in1, const ImageCuda & in2, ImageCuda & out );

	ImageCuda BitwiseOr( const ImageCuda & in1, const ImageCuda & in2 );
	void      BitwiseOr( const ImageCuda & in1, const ImageCuda & in2, ImageCuda & out );

	ImageCuda BitwiseXor( const ImageCuda & in1, const ImageCuda & in2 );
	void      BitwiseXor( const ImageCuda & in1, const ImageCuda & in2, ImageCuda & out );

	void Convert( const Bitmap_Image::Image & in, ImageCuda & out );
	void Convert( const ImageCuda & in, Bitmap_Image::Image & out );

	// Invert function is Bitwise NOT operation. But to make function name more user-friendly we named it like this
	ImageCuda Invert( const ImageCuda & in );
	void      Invert( const ImageCuda & in, ImageCuda & out );

	ImageCuda Maximum( const ImageCuda & in1, const ImageCuda & in2 );
	void      Maximum( const ImageCuda & in1, const ImageCuda & in2, ImageCuda & out );

	ImageCuda Minimum( const ImageCuda & in1, const ImageCuda & in2 );
	void      Minimum( const ImageCuda & in1, const ImageCuda & in2, ImageCuda & out );

	ImageCuda Subtract( const ImageCuda & in1, const ImageCuda & in2 );
	void      Subtract( const ImageCuda & in1, const ImageCuda & in2, ImageCuda & out );
};
