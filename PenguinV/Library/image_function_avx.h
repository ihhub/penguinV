#pragma once

#include "image_function.h"

// Utilize these image functions only if your CPU supports AXV 2.0 !!!

// These functions contain AXV 2.0 code through Intel Intrinsics functions
// Functions have totally same results like normal functions but they are faster!
// You will have speed up compare to normal functions if the width of inspection area is bigger than 32 pixels
// because 32 pixels is minimum width what AVX 2.0 function can process
// Anyway you do not need to care about this. Everything is done inside. Just use it if you want to have faster code

namespace Image_Function_Avx
{
	using namespace Bitmap_Image;

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
};
