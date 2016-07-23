#include <emmintrin.h>
#include "image_function_sse.h"

namespace Image_Function_Sse
{
	// We are not sure that input data is aligned by 16 bytes so we use loadu() functions instead of load()
	// You can change it in case if your application has always aligned by 16 images images and areas (ROIs - regions of interest)

	void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		// image width is less than 16 bytes so no use to utilize SSE :(
		if (width < 16u) {
			Image_Function::BitwiseAnd(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
			return;
		}

		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

		uint32_t rowSizeIn1 = in1.rowSize();
		uint32_t rowSizeIn2 = in2.rowSize();
		uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1;
		const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		uint32_t sseWidth = width / 16u;
		uint32_t totalSseWidth = sseWidth * 16u;
		uint32_t nonSseWidth = width - totalSseWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
		
			const __m128i * src1 = reinterpret_cast < const __m128i* > (in1Y);
			const __m128i * src2 = reinterpret_cast < const __m128i* > (in2Y);
			__m128i       * dst  = reinterpret_cast <       __m128i* > (outY);

			const __m128i * src1End = src1 + sseWidth;

			for( ; src1 != src1End; ++src1, ++src2, ++dst )
				_mm_storeu_si128(dst, _mm_and_si128( _mm_loadu_si128(src1), _mm_loadu_si128(src2) ) );


			if( nonSseWidth > 0 ) {

				const uint8_t * in1X = in1Y + totalSseWidth;
				const uint8_t * in2X = in2Y + totalSseWidth;
				uint8_t       * outX = outY + totalSseWidth;

				const uint8_t * outXEnd = outX + nonSseWidth;

				for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
					(*outX) = (*in1X) & (*in2X);
			}
		
		}
	}

	Image BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					  uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseAnd( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		BitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseAnd( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseAnd( in1, in2, out );

		return out;
	}

	void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		// image width is less than 16 bytes so no use to utilize SSE :(
		if (width < 16u) {
			Image_Function::BitwiseOr(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
			return;
		}

		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

		uint32_t rowSizeIn1 = in1.rowSize();
		uint32_t rowSizeIn2 = in2.rowSize();
		uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1;
		const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		uint32_t sseWidth = width / 16u;
		uint32_t totalSseWidth = sseWidth * 16u;
		uint32_t nonSseWidth = width - totalSseWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
		
			const __m128i * src1 = reinterpret_cast < const __m128i* > (in1Y);
			const __m128i * src2 = reinterpret_cast < const __m128i* > (in2Y);
			__m128i       * dst  = reinterpret_cast <       __m128i* > (outY);

			const __m128i * src1End = src1 + sseWidth;

			for( ; src1 != src1End; ++src1, ++src2, ++dst )
				_mm_storeu_si128(dst, _mm_or_si128( _mm_loadu_si128(src1), _mm_loadu_si128(src2) ) );


			if( nonSseWidth > 0 ) {

				const uint8_t * in1X = in1Y + totalSseWidth;
				const uint8_t * in2X = in2Y + totalSseWidth;
				uint8_t       * outX = outY + totalSseWidth;

				const uint8_t * outXEnd = outX + nonSseWidth;

				for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
					(*outX) = (*in1X) | (*in2X);
			}
		
		}
	}

	Image BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseOr( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		BitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseOr( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseOr( in1, in2, out );

		return out;
	}

	void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		// image width is less than 16 bytes so no use to utilize SSE :(
		if (width < 16u) {
			Image_Function::BitwiseXor(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
			return;
		}

		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

		uint32_t rowSizeIn1 = in1.rowSize();
		uint32_t rowSizeIn2 = in2.rowSize();
		uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1;
		const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		uint32_t sseWidth = width / 16u;
		uint32_t totalSseWidth = sseWidth * 16u;
		uint32_t nonSseWidth = width - totalSseWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
		
			const __m128i * src1 = reinterpret_cast < const __m128i* > (in1Y);
			const __m128i * src2 = reinterpret_cast < const __m128i* > (in2Y);
			__m128i       * dst  = reinterpret_cast <       __m128i* > (outY);

			const __m128i * src1End = src1 + sseWidth;

			for( ; src1 != src1End; ++src1, ++src2, ++dst )
				_mm_storeu_si128(dst, _mm_xor_si128( _mm_loadu_si128(src1), _mm_loadu_si128(src2) ) );


			if( nonSseWidth > 0 ) {

				const uint8_t * in1X = in1Y + totalSseWidth;
				const uint8_t * in2X = in2Y + totalSseWidth;
				uint8_t       * outX = outY + totalSseWidth;

				const uint8_t * outXEnd = outX + nonSseWidth;

				for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
					(*outX) = (*in1X) ^ (*in2X);
			}
		
		}
	}

	Image BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					  uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseXor( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		BitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseXor( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseXor( in1, in2, out );

		return out;
	}

};
