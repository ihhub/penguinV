#include <immintrin.h>
#include "image_function_avx.h"
#include "image_function_sse.h"

namespace Image_Function_Avx
{
	// We are not sure that input data is aligned by 16 bytes so we use loadu() functions instead of load()
	// You can change it in case if your application has always aligned by 32 images images and areas (ROIs - regions of interest)

	// All processors what support AVX 2.0 support SSE too

	void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		// image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
		if (width < 32u) {
			Image_Function_Sse::BitwiseAnd(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
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

		uint32_t sseWidth = width / 32u;
		uint32_t totalSseWidth = sseWidth * 32u;
		uint32_t nonSseWidth = width - totalSseWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
		
			const __m256i * src1 = reinterpret_cast < const __m256i* > (in1Y);
			const __m256i * src2 = reinterpret_cast < const __m256i* > (in2Y);
			__m256i       * dst  = reinterpret_cast <       __m256i* > (outY);

			const __m256i * src1End = src1 + sseWidth;

			for( ; src1 != src1End; ++src1, ++src2, ++dst )
				_mm256_storeu_si256(dst, _mm256_and_si256( _mm256_loadu_si256(src1), _mm256_loadu_si256(src2) ) );

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
		// image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
		if (width < 32u) {
			Image_Function_Sse::BitwiseOr(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
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

		uint32_t sseWidth = width / 32u;
		uint32_t totalSseWidth = sseWidth * 32u;
		uint32_t nonSseWidth = width - totalSseWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
		
			const __m256i * src1 = reinterpret_cast < const __m256i* > (in1Y);
			const __m256i * src2 = reinterpret_cast < const __m256i* > (in2Y);
			__m256i       * dst  = reinterpret_cast <       __m256i* > (outY);

			const __m256i * src1End = src1 + sseWidth;

			for( ; src1 != src1End; ++src1, ++src2, ++dst )
				_mm256_storeu_si256(dst, _mm256_or_si256( _mm256_loadu_si256(src1), _mm256_loadu_si256(src2) ) );

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
		// image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
		if (width < 32u) {
			Image_Function_Sse::BitwiseXor(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
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

		uint32_t sseWidth = width / 32u;
		uint32_t totalSseWidth = sseWidth * 32u;
		uint32_t nonSseWidth = width - totalSseWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
		
			const __m256i * src1 = reinterpret_cast < const __m256i* > (in1Y);
			const __m256i * src2 = reinterpret_cast < const __m256i* > (in2Y);
			__m256i       * dst  = reinterpret_cast <       __m256i* > (outY);

			const __m256i * src1End = src1 + sseWidth;

			for( ; src1 != src1End; ++src1, ++src2, ++dst )
				_mm256_storeu_si256(dst, _mm256_xor_si256( _mm256_loadu_si256(src1), _mm256_loadu_si256(src2) ) );

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
