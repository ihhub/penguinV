#include <arm_neon.h>
#include "image_function.h"
#include "image_function_neon.h"

// This unnamed namespace contains all necessary information to reduce bugs in SIMD function writing
namespace
{
	const uint32_t simdSize = 16u;
};

namespace Image_Function_Neon
{
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
		// image width is less than 16 bytes so no use to utilize NEON :(
		if (width < simdSize) {
			Image_Function::BitwiseAnd(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
			return;
		}

		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
		Image_Function::VerifyGrayScaleImage( in1, in2, out );

		const uint32_t rowSizeIn1 = in1.rowSize();
		const uint32_t rowSizeIn2 = in2.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1;
		const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		const uint32_t simdWidth = width / simdSize;
		const uint32_t totalSimdWidth = simdWidth * simdSize;
		const uint32_t nonSimdWidth = width - totalSimdWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
			const uint8_t * src1 = in1Y;
			const uint8_t * src2 = in2Y;
			uint8_t       * dst  = outY;

			const uint8_t * src1End = src1 + totalSimdWidth;

			for( ; src1 != src1End; src1 += simdSize, src2 += simdSize, dst += simdSize )
				vst1q_u8( dst, vandq_u8( vld1q_u8( src1 ), vld1q_u8( src2 ) ) );

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
		// image width is less than 16 bytes so no use to utilize NEON :(
		if (width < simdSize) {
			Image_Function::BitwiseOr(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
			return;
		}

		Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
		Image_Function::VerifyGrayScaleImage( in1, in2, out );

		const uint32_t rowSizeIn1 = in1.rowSize();
		const uint32_t rowSizeIn2 = in2.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1;
		const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		const uint32_t simdWidth = width / simdSize;
		const uint32_t totalSimdWidth = simdWidth * simdSize;
		const uint32_t nonSimdWidth = width - totalSimdWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
			const uint8_t * src1 = in1Y;
			const uint8_t * src2 = in2Y;
			uint8_t       * dst  = outY;

			const uint8_t * src1End = src1 + totalSimdWidth;

			for( ; src1 != src1End; src1 += simdSize, src2 += simdSize, dst += simdSize )
				vst1q_u8( dst, vorrq_u8( vld1q_u8( src1 ), vld1q_u8( src2 ) ) );

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
		// image width is less than 16 bytes so no use to utilize NEON :(
		if (width < simdSize) {
			Image_Function::Invert(in, startXIn, startYIn, out, startXOut, startYOut, width, height);
			return;
		}

		Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
		Image_Function::VerifyGrayScaleImage( in, out );

		const uint32_t rowSizeIn  = in.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		const uint32_t simdWidth = width / simdSize;
		const uint32_t totalSimdWidth = simdWidth * simdSize;
		const uint32_t nonSimdWidth = width - totalSimdWidth;

		for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
			const uint8_t * src1 = inY;
			uint8_t       * dst  = outY;

			const uint8_t * src1End = src1 + totalSimdWidth;

			for( ; src1 != src1End; src1 += simdSize, dst += simdSize )
				vst1q_u8( dst, vmvnq_u8( vld1q_u8( src1 ) ) );

			if( nonSimdWidth > 0 ) {
				const uint8_t * inX  = inY  + totalSimdWidth;
				uint8_t       * outX = outY + totalSimdWidth;

				const uint8_t * outXEnd = outX + nonSimdWidth;

				for( ; outX != outXEnd; ++outX, ++inX )
					(*outX) = ~(*inX);
			}
		}
	}

	Image Maximum(const Image & in1, const Image & in2)
	{
		Image_Function::ParameterValidation(in1, in2);

		Image out(in1.width(), in1.height());

		Maximum(in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height());

		return out;
	}

	void Maximum(const Image & in1, const Image & in2, Image & out)
	{
		Image_Function::ParameterValidation(in1, in2, out);

		Maximum(in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height());
	}

	Image Maximum(const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
		uint32_t width, uint32_t height)
	{
		Image_Function::ParameterValidation(in1, startX1, startY1, in2, startX2, startY2, width, height);

		Image out(width, height);

		Maximum(in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height());

		return out;
	}

	void Maximum(const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
		Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height)
	{
		// image width is less than 16 bytes so no use to utilize NEON :(
		if (width < simdSize) {
			Image_Function::Maximum(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
			return;
		}

		Image_Function::ParameterValidation(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
		Image_Function::VerifyGrayScaleImage( in1, in2, out );

		const uint32_t rowSizeIn1 = in1.rowSize();
		const uint32_t rowSizeIn2 = in2.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1;
		const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		const uint32_t simdWidth = width / simdSize;
		const uint32_t totalSimdWidth = simdWidth * simdSize;
		const uint32_t nonSimdWidth = width - totalSimdWidth;

		for (; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2) {
			const uint8_t * src1 = in1Y;
			const uint8_t * src2 = in2Y;
			uint8_t       * dst  = outY;

			const uint8_t * src1End = src1 + totalSimdWidth;

			for( ; src1 != src1End; src1 += simdSize, src2 += simdSize, dst += simdSize )
				vst1q_u8( dst, vmaxq_u8( vld1q_u8(src1), vld1q_u8(src2) ) );

			if (nonSimdWidth > 0) {
				const uint8_t * in1X = in1Y + totalSimdWidth;
				const uint8_t * in2X = in2Y + totalSimdWidth;
				uint8_t       * outX = outY + totalSimdWidth;

				const uint8_t * outXEnd = outX + nonSimdWidth;

				for (; outX != outXEnd; ++outX, ++in1X, ++in2X) {
					if ((*in2X) < (*in1X))
						(*outX) = (*in1X);
					else
						(*outX) = (*in2X);
				}
			}
		}
	}

	Image Minimum(const Image & in1, const Image & in2)
	{
		Image_Function::ParameterValidation(in1, in2);

		Image out(in1.width(), in1.height());

		Minimum(in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height());

		return out;
	}

	void Minimum(const Image & in1, const Image & in2, Image & out)
	{
		Image_Function::ParameterValidation(in1, in2, out);

		Minimum(in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height());
	}

	Image Minimum(const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
		uint32_t width, uint32_t height)
	{
		Image_Function::ParameterValidation(in1, startX1, startY1, in2, startX2, startY2, width, height);

		Image out(width, height);

		Minimum(in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height());

		return out;
	}

	void Minimum(const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
		Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height)
	{
		// image width is less than 16 bytes so no use to utilize NEON :(
		if (width < simdSize) {
			Image_Function::Minimum(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
			return;
		}

		Image_Function::ParameterValidation(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
		Image_Function::VerifyGrayScaleImage( in1, in2, out );

		const uint32_t rowSizeIn1 = in1.rowSize();
		const uint32_t rowSizeIn2 = in2.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1;
		const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		const uint32_t simdWidth = width / simdSize;
		const uint32_t totalSimdWidth = simdWidth * simdSize;
		const uint32_t nonSimdWidth = width - totalSimdWidth;

		for (; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2) {
			const uint8_t * src1 = in1Y;
			const uint8_t * src2 = in2Y;
			uint8_t       * dst  = outY;

			const uint8_t * src1End = src1 + totalSimdWidth;

			for( ; src1 != src1End; src1 += simdSize, src2 += simdSize, dst += simdSize )
				vst1q_u8( dst, vminq_u8( vld1q_u8(src1), vld1q_u8(src2) ) );

			if (nonSimdWidth > 0) {
				const uint8_t * in1X = in1Y + totalSimdWidth;
				const uint8_t * in2X = in2Y + totalSimdWidth;
				uint8_t       * outX = outY + totalSimdWidth;

				const uint8_t * outXEnd = outX + nonSimdWidth;

				for (; outX != outXEnd; ++outX, ++in1X, ++in2X) {
					if ((*in2X) > (*in1X))
						(*outX) = (*in1X);
					else
						(*outX) = (*in2X);
				}
			}
		}
	}

	Image Subtract(const Image & in1, const Image & in2)
	{
		Image_Function::ParameterValidation(in1, in2);

		Image out(in1.width(), in1.height());

		Subtract(in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height());

		return out;
	}

	void Subtract(const Image & in1, const Image & in2, Image & out)
	{
		Image_Function::ParameterValidation(in1, in2, out);

		Subtract(in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height());
	}

	Image Subtract(const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
		uint32_t width, uint32_t height)
	{
		Image_Function::ParameterValidation(in1, startX1, startY1, in2, startX2, startY2, width, height);

		Image out(width, height);

		Subtract(in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height());

		return out;
	}

	void Subtract(const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
		Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height)
	{
		// image width is less than 16 bytes so no use to utilize NEON :(
		if (width < simdSize) {
			Image_Function::Subtract(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
			return;
		}

		Image_Function::ParameterValidation(in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height);
		Image_Function::VerifyGrayScaleImage( in1, in2, out );

		const uint32_t rowSizeIn1 = in1.rowSize();
		const uint32_t rowSizeIn2 = in2.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1;
		const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		const uint32_t simdWidth = width / simdSize;
		const uint32_t totalSimdWidth = simdWidth * simdSize;
		const uint32_t nonSimdWidth = width - totalSimdWidth;

		for (; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2) {
			const uint8_t * src1 = in1Y;
			const uint8_t * src2 = in2Y;
			uint8_t       * dst  = outY;

			const uint8_t * src1End = src1 + totalSimdWidth;

			for( ; src1 != src1End; src1 += simdSize, src2 += simdSize, dst += simdSize ) {
				uint8x16_t data = vld1q_u8(src1);
				vst1q_u8( dst, vsubq_u8( data, vminq_u8( data, vld1q_u8(src2) ) ) );
			}

			if (nonSimdWidth > 0) {
				const uint8_t * in1X = in1Y + totalSimdWidth;
				const uint8_t * in2X = in2Y + totalSimdWidth;
				uint8_t       * outX = outY + totalSimdWidth;

				const uint8_t * outXEnd = outX + nonSimdWidth;

				for (; outX != outXEnd; ++outX, ++in1X, ++in2X) {
					if ((*in2X) > (*in1X))
						(*outX) = 0;
					else
						(*outX) = (*in1X) - (*in2X);
				}
			}
		}
	}
};
