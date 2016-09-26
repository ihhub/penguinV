#include <immintrin.h>
#include "image_function_avx.h"
#include "image_function_sse.h"

// This unnamed namespace contains all necessary information to reduce bugs in SIMD function writing
namespace
{
	const uint32_t simdSize = 32u;
	typedef __m256i simd;
};

namespace Image_Function_Avx
{
	// We are not sure that input data is aligned by 16 bytes so we use loadu() functions instead of load()
	// You can change it in case if your application has always aligned by 32 images images and areas (ROIs - regions of interest)

	// All processors what support AVX 2.0 support SSE too

	Image BitwiseAnd( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseAnd( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		BitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					  uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		// image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
		if (width < simdSize) {
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

		uint32_t simdWidth = width / simdSize;
		uint32_t totalSimdWidth = simdWidth * simdSize;
		uint32_t nonSimdWidth = width - totalSimdWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
			const simd * src1 = reinterpret_cast < const simd* > (in1Y);
			const simd * src2 = reinterpret_cast < const simd* > (in2Y);
			simd       * dst  = reinterpret_cast <       simd* > (outY);

			const simd * src1End = src1 + simdWidth;

			for( ; src1 != src1End; ++src1, ++src2, ++dst )
				_mm256_storeu_si256(dst, _mm256_and_si256( _mm256_loadu_si256(src1), _mm256_loadu_si256(src2) ) );

			if( nonSimdWidth > 0 ) {
				const uint8_t * in1X = in1Y + totalSimdWidth;
				const uint8_t * in2X = in2Y + totalSimdWidth;
				uint8_t       * outX = outY + totalSimdWidth;

				const uint8_t * outXEnd = outX + nonSimdWidth;

				for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
					(*outX) = (*in1X) & (*in2X);
			}
		}
	}

	Image BitwiseOr( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseOr( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		BitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		// image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
		if (width < simdSize) {
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

		uint32_t simdWidth = width / simdSize;
		uint32_t totalSimdWidth = simdWidth * simdSize;
		uint32_t nonSimdWidth = width - totalSimdWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
			const simd * src1 = reinterpret_cast < const simd* > (in1Y);
			const simd * src2 = reinterpret_cast < const simd* > (in2Y);
			simd       * dst  = reinterpret_cast <       simd* > (outY);

			const simd * src1End = src1 + simdWidth;

			for( ; src1 != src1End; ++src1, ++src2, ++dst )
				_mm256_storeu_si256(dst, _mm256_or_si256( _mm256_loadu_si256(src1), _mm256_loadu_si256(src2) ) );

			if( nonSimdWidth > 0 ) {
				const uint8_t * in1X = in1Y + totalSimdWidth;
				const uint8_t * in2X = in2Y + totalSimdWidth;
				uint8_t       * outX = outY + totalSimdWidth;

				const uint8_t * outXEnd = outX + nonSimdWidth;

				for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
					(*outX) = (*in1X) | (*in2X);
			}
		}
	}

	Image BitwiseXor( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseXor( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		BitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					  uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		// image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
		if (width < simdSize) {
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

		uint32_t simdWidth = width / simdSize;
		uint32_t totalSimdWidth = simdWidth * simdSize;
		uint32_t nonSimdWidth = width - totalSimdWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
			const simd * src1 = reinterpret_cast < const simd* > (in1Y);
			const simd * src2 = reinterpret_cast < const simd* > (in2Y);
			simd       * dst  = reinterpret_cast <       simd* > (outY);

			const simd * src1End = src1 + simdWidth;

			for( ; src1 != src1End; ++src1, ++src2, ++dst )
				_mm256_storeu_si256(dst, _mm256_xor_si256( _mm256_loadu_si256(src1), _mm256_loadu_si256(src2) ) );

			if( nonSimdWidth > 0 ) {
				const uint8_t * in1X = in1Y + totalSimdWidth;
				const uint8_t * in2X = in2Y + totalSimdWidth;
				uint8_t       * outX = outY + totalSimdWidth;

				const uint8_t * outXEnd = outX + nonSimdWidth;

				for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
					(*outX) = (*in1X) ^ (*in2X);
			}
		}
	}

	Image Invert( const Image & in )
	{
		Image_Function::ParameterValidation( in );

		Image out( in.width(), in.height() );

		Invert( in, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Invert( const Image & in, Image & out )
	{
		Image_Function::ParameterValidation( in, out );

		Invert( in, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Invert( in, startXIn, startYIn, out, 0, 0, width, height );

		return out;
	}

	void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
				  uint32_t width, uint32_t height )
	{
		// image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
		if (width < simdSize) {
			Image_Function_Sse::Invert(in, startXIn, startYIn, out, startXOut, startYOut, width, height);
			return;
		}

		Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

		uint32_t rowSizeIn  = in.rowSize();
		uint32_t rowSizeOut = out.rowSize();

		const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		uint32_t simdWidth = width / simdSize;
		uint32_t totalSimdWidth = simdWidth * simdSize;
		uint32_t nonSimdWidth = width - totalSimdWidth;

		simd mask = _mm256_set_epi8( 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu,
									 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu,
									 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu,
									 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu );

		for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
			const simd * src1 = reinterpret_cast < const simd* > (inY);
			simd       * dst  = reinterpret_cast <       simd* > (outY);

			const simd * src1End = src1 + simdWidth;

			for( ; src1 != src1End; ++src1, ++dst )
				_mm256_storeu_si256( dst, _mm256_andnot_si256(_mm256_loadu_si256(src1), mask) );

			if( nonSimdWidth > 0 ) {
				const uint8_t * inX  = inY  + totalSimdWidth;
				uint8_t       * outX = outY + totalSimdWidth;

				const uint8_t * outXEnd = outX + nonSimdWidth;

				for( ; outX != outXEnd; ++outX, ++inX )
					(*outX) = ~(*inX);
			}
		}
	}

	Image Maximum( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		Maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Maximum( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		Maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		Maximum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		// image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
		if (width < simdSize) {
			Image_Function_Sse::Maximum(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
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

		uint32_t simdWidth = width / simdSize;
		uint32_t totalSimdWidth = simdWidth * simdSize;
		uint32_t nonSimdWidth = width - totalSimdWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
			const simd * src1 = reinterpret_cast < const simd* > (in1Y);
			const simd * src2 = reinterpret_cast < const simd* > (in2Y);
			simd       * dst  = reinterpret_cast <       simd* > (outY);

			const simd * src1End = src1 + simdWidth;

			for( ; src1 != src1End; ++src1, ++src2, ++dst )
				_mm256_storeu_si256(dst, _mm256_max_epu8( _mm256_loadu_si256(src1), _mm256_loadu_si256(src2) ) );

			if( nonSimdWidth > 0 ) {
				const uint8_t * in1X = in1Y + totalSimdWidth;
				const uint8_t * in2X = in2Y + totalSimdWidth;
				uint8_t       * outX = outY + totalSimdWidth;

				const uint8_t * outXEnd = outX + nonSimdWidth;

				for( ; outX != outXEnd; ++outX, ++in1X, ++in2X ) {
					if( (*in2X) < (*in1X) )
						(*outX) = (*in1X);
					else
						(*outX) = (*in2X);
				}
			}
		}
	}

	Image Minimum( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		Minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Minimum( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		Minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		Minimum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		// image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
		if (width < simdSize) {
			Image_Function_Sse::Minimum(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
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

		uint32_t simdWidth = width / simdSize;
		uint32_t totalSimdWidth = simdWidth * simdSize;
		uint32_t nonSimdWidth = width - totalSimdWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
			const simd * src1 = reinterpret_cast < const simd* > (in1Y);
			const simd * src2 = reinterpret_cast < const simd* > (in2Y);
			simd       * dst  = reinterpret_cast <       simd* > (outY);

			const simd * src1End = src1 + simdWidth;

			for( ; src1 != src1End; ++src1, ++src2, ++dst )
				_mm256_storeu_si256(dst, _mm256_min_epu8( _mm256_loadu_si256(src1), _mm256_loadu_si256(src2) ) );

			if( nonSimdWidth > 0 ) {
				const uint8_t * in1X = in1Y + totalSimdWidth;
				const uint8_t * in2X = in2Y + totalSimdWidth;
				uint8_t       * outX = outY + totalSimdWidth;

				const uint8_t * outXEnd = outX + nonSimdWidth;

				for( ; outX != outXEnd; ++outX, ++in1X, ++in2X ) {
					if( (*in2X) > (*in1X) )
						(*outX) = (*in1X);
					else
						(*outX) = (*in2X);
				}
			}
		}
	}

	Image Subtract( const Image & in1, const Image & in2 )
	{
		Image_Function::ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		Subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Subtract( const Image & in1, const Image & in2, Image & out )
	{
		Image_Function::ParameterValidation( in1, in2, out );

		Subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					uint32_t width, uint32_t height )
	{
		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		Subtract( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		// image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
		if (width < simdSize) {
			Image_Function_Sse::Subtract(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
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

		uint32_t simdWidth = width / simdSize;
		uint32_t totalSimdWidth = simdWidth * simdSize;
		uint32_t nonSimdWidth = width - totalSimdWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
			const simd * src1 = reinterpret_cast < const simd* > (in1Y);
			const simd * src2 = reinterpret_cast < const simd* > (in2Y);
			simd       * dst  = reinterpret_cast <       simd* > (outY);

			const simd * src1End = src1 + simdWidth;

			for( ; src1 != src1End; ++src1, ++src2, ++dst ) {
				simd data = _mm256_loadu_si256(src1);
				_mm256_storeu_si256(dst, _mm256_sub_epi8(data, _mm256_min_epu8( data, _mm256_loadu_si256(src2)) ) );
			}

			if( nonSimdWidth > 0 ) {
				const uint8_t * in1X = in1Y + totalSimdWidth;
				const uint8_t * in2X = in2Y + totalSimdWidth;
				uint8_t       * outX = outY + totalSimdWidth;

				const uint8_t * outXEnd = outX + nonSimdWidth;

				for( ; outX != outXEnd; ++outX, ++in1X, ++in2X ) {
					if( (*in2X) > (*in1X) )
						(*outX) = 0;
					else
						(*outX) = (*in1X) - (*in2X);
				}
			}
		}
	}

	Image Threshold( const Image & in, uint8_t threshold )
	{
		Image_Function::ParameterValidation( in );

		Image out( in.width(), in.height() );

		Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), threshold );

		return out;
	}

	void Threshold( const Image & in, Image & out, uint8_t threshold )
	{
		Image_Function::ParameterValidation( in, out );

		Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), threshold );
	}

	Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t threshold )
	{
		Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Threshold( in, startXIn, startYIn, out, 0, 0, width, height, threshold );

		return out;
	}

	void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					uint32_t width, uint32_t height, uint8_t threshold )
	{
		// image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
		if (width < simdSize) {
			Image_Function_Sse::Threshold(in, startXIn, startYIn, out, startXOut, startYOut, width, height, threshold);
			return;
		}

		Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

		// AVX does not have command "great or equal to" so we have 2 situations:
		// when threshold value is 0 and it is not
		if( threshold > 0 ) {
			uint32_t rowSizeIn  = in.rowSize();
			uint32_t rowSizeOut = out.rowSize();

			const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
			uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

			const uint8_t * outYEnd = outY + height * rowSizeOut;

			uint32_t simdWidth = width / simdSize;
			uint32_t totalSimdWidth = simdWidth * simdSize;
			uint32_t nonSimdWidth = width - totalSimdWidth;

			simd mask = _mm256_set_epi8( 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u,
										 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u,
										 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u,
										 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u );

			simd compare = _mm256_set_epi8(
				(threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
				(threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
				(threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
				(threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
				(threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
				(threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
				(threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
				(threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u );

			for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
				const simd * src1 = reinterpret_cast < const simd* > (inY);
				simd       * dst  = reinterpret_cast <       simd* > (outY);

				const simd * src1End = src1 + simdWidth;

				for( ; src1 != src1End; ++src1, ++dst )
					_mm256_storeu_si256( dst, _mm256_cmpgt_epi8(_mm256_xor_si256( _mm256_loadu_si256(src1), mask ), compare) );

				if( nonSimdWidth > 0 ) {
					const uint8_t * inX  = inY  + totalSimdWidth;
					uint8_t       * outX = outY + totalSimdWidth;

					const uint8_t * outXEnd = outX + nonSimdWidth;

					for( ; outX != outXEnd; ++outX, ++inX )
						(*outX) = (*inX) < threshold ? 0 : 255;
				}
			}
		}
		else {
			uint32_t rowSizeOut = out.rowSize();

			uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;
			const uint8_t * outYEnd = outY + height * rowSizeOut;

			for( ; outY != outYEnd; outY += rowSizeOut ) 
				memset( outY, 255u, sizeof(uint8_t) * width );
		}
	}
};
