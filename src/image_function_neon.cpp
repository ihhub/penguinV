#include "image_function_neon.h"

#ifdef PENGUINV_NEON_SET

#include <arm_neon.h>
#include "image_function.h"
#include "parameter_validation.h"

// This unnamed namespace contains all necessary information to reduce bugs in SIMD function writing
namespace
{
    const uint32_t simdSize = 16u;
    typedef uint8x16_t simd;
}

namespace Image_Function_Neon
{
    Image AbsoluteDifference( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::AbsoluteDifference( AbsoluteDifference, in1, in2 );
    }

    void AbsoluteDifference( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::AbsoluteDifference( AbsoluteDifference, in1, in2, out );
    }

    Image AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                              uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::AbsoluteDifference( AbsoluteDifference, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                             Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::AbsoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

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
                vst1q_u8( dst, vabdq_u8( vld1q_u8( src1 ), vld1q_u8( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                    (*outX) = (*in2X) > (*in1X) ? (*in2X) - (*in1X) : (*in1X) - (*in2X);
            }
        }
    }

    Image BitwiseAnd( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::BitwiseAnd( BitwiseAnd, in1, in2 );
    }

    void BitwiseAnd( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::BitwiseAnd( BitwiseAnd, in1, in2, out );
    }

    Image BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseAnd( BitwiseAnd, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

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
        return Image_Function_Helper::BitwiseOr( BitwiseOr, in1, in2 );
    }

    void BitwiseOr( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::BitwiseOr( BitwiseOr, in1, in2, out );
    }

    Image BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseOr( BitwiseOr, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

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

    Image BitwiseXor( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::BitwiseXor( BitwiseXor, in1, in2 );
    }

    void BitwiseXor( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::BitwiseXor( BitwiseXor, in1, in2, out );
    }

    Image BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseXor( BitwiseXor, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

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
                vst1q_u8( dst, veorq_u8( vld1q_u8( src1 ), vld1q_u8( src2 ) ) );

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
        return Image_Function_Helper::Invert( Invert, in );
    }

    void Invert( const Image & in, Image & out )
    {
        Image_Function_Helper::Invert( Invert, in, out );
    }

    Image Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Invert( Invert, in, startXIn, startYIn, width, height );
    }

    void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                 uint32_t width, uint32_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::Invert( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

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

    Image Maximum( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::Maximum( Maximum, in1, in2 );
    }

    void Maximum( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::Maximum( Maximum, in1, in2, out );
    }

    Image Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Maximum( Maximum, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::Maximum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

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
                vst1q_u8( dst, vmaxq_u8( vld1q_u8( src1 ), vld1q_u8( src2 ) ) );

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
        return Image_Function_Helper::Minimum( Minimum, in1, in2 );
    }

    void Minimum( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::Minimum( Minimum, in1, in2, out );
    }

    Image Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Minimum( Minimum, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::Minimum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

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
                vst1q_u8( dst, vminq_u8( vld1q_u8( src1 ), vld1q_u8( src2 ) ) );

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
        return Image_Function_Helper::Subtract( Subtract, in1, in2 );
    }

    void Subtract( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::Subtract( Subtract, in1, in2, out );
    }

    Image Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Subtract( Subtract, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::Subtract( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const uint8_t * src1 = in1Y;
            const uint8_t * src2 = in2Y;
            uint8_t       * dst  = outY;

            const uint8_t * src1End = src1 + totalSimdWidth;

            for( ; src1 != src1End; src1 += simdSize, src2 += simdSize, dst += simdSize ) {
                const simd data = vld1q_u8( src1 );
                vst1q_u8( dst, vsubq_u8( data, vminq_u8( data, vld1q_u8( src2 ) ) ) );
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
        return Image_Function_Helper::Threshold( Threshold, in, threshold );
    }

    void Threshold( const Image & in, Image & out, uint8_t threshold )
    {
        Image_Function_Helper::Threshold( Threshold, in, out, threshold );
    }

    Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t threshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, startXIn, startYIn, width, height, threshold );
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t threshold )
    {
        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width < simdSize ) {
            Image_Function::Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, threshold );
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

        const uint8_t thresholdValue[16] ={ threshold, threshold, threshold, threshold, threshold, threshold, threshold, threshold,
                                            threshold, threshold, threshold, threshold, threshold, threshold, threshold, threshold };
        const simd compare = vld1q_u8( thresholdValue );

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * src = inY;
            uint8_t       * dst = outY;

            const uint8_t * srcEnd = src + totalSimdWidth;

            for( ; src != srcEnd; src += simdSize, dst += simdSize )
                vst1q_u8( dst, vcgeq_u8( vld1q_u8( src ), compare ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * inX  = inY  + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++inX )
                    (*outX) = (*inX) < threshold ? 0 : 255;
            }
        }
    }

    Image Threshold( const Image & in, uint8_t minThreshold, uint8_t maxThreshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, minThreshold, maxThreshold );
    }

    void Threshold( const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function_Helper::Threshold( Threshold, in, out, minThreshold, maxThreshold );
    }

    Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t minThreshold,
                     uint8_t maxThreshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, startXIn, startYIn, width, height, minThreshold, maxThreshold );
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold )
    {
        // image width is less than 16 bytes so no use to utilize NEOD :(
        if( width < simdSize ) {
            Image_Function::Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold );
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

        const uint8_t thresholdMinValue[16] ={ minThreshold, minThreshold, minThreshold, minThreshold,
                                               minThreshold, minThreshold, minThreshold, minThreshold,
                                               minThreshold, minThreshold, minThreshold, minThreshold,
                                               minThreshold, minThreshold, minThreshold, minThreshold };
        const simd compareMin = vld1q_u8( thresholdMinValue );

        const uint8_t thresholdMaxValue[16] ={ maxThreshold, maxThreshold, maxThreshold, maxThreshold,
                                               maxThreshold, maxThreshold, maxThreshold, maxThreshold,
                                               maxThreshold, maxThreshold, maxThreshold, maxThreshold,
                                               maxThreshold, maxThreshold, maxThreshold, maxThreshold };
        const simd compareMax = vld1q_u8( thresholdMaxValue );

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * src = inY;
            uint8_t       * dst = outY;

            const uint8_t * srcEnd = src + totalSimdWidth;

            for( ; src != srcEnd; src += simdSize, dst += simdSize ) {
                const simd data = vld1q_u8( src );
                vst1q_u8( dst, vandq_u8( vcgeq_u8( data, compareMin ), vcleq_u8( data, compareMax ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * inX  = inY  + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++inX )
                    (*outX) = (*inX) < minThreshold || (*inX) > maxThreshold ? 0 : 255;
            }
        }
    }
}
#endif
