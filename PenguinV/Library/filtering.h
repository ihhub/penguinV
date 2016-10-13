#pragma once

#include "image_buffer.h"

namespace Image_Function
{
	namespace Filtering
	{
		using namespace Bitmap_Image;

		Image Median( const Image & in, uint32_t kernelSize );
		void  Median( const Image & in, Image & out, uint32_t kernelSize );
		Image Median( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint32_t kernelSize );
		void  Median( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					  uint32_t width, uint32_t height, uint32_t kernelSize );
	};
};
