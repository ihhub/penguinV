#include <arm_neon.h>
#include "image_function.h"
#include "image_function_neon.h"

// This unnamed namespace contains all necessary information to reduce bugs in SIMD function writing
namespace
{
    const size_t simdSize = 16u;
    typedef uint8x16_t simd;
};

namespace Image_Function_Neon
{
    Image AbsoluteDifference( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        AbsoluteDifference( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void AbsoluteDifference( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        AbsoluteDifference( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image AbsoluteDifference( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                              size_t width, size_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        AbsoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void AbsoluteDifference( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                             Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::AbsoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const size_t rowSizeIn1 = in1.rowSize();
        const size_t rowSizeIn2 = in2.rowSize();
        const size_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        const size_t simdWidth = width / simdSize;
        const size_t totalSimdWidth = simdWidth * simdSize;
        const size_t nonSimdWidth = width - totalSimdWidth;

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

    Image BitwiseAnd( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                      size_t width, size_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseAnd( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                     Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const size_t rowSizeIn1 = in1.rowSize();
        const size_t rowSizeIn2 = in2.rowSize();
        const size_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        const size_t simdWidth = width / simdSize;
        const size_t totalSimdWidth = simdWidth * simdSize;
        const size_t nonSimdWidth = width - totalSimdWidth;

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

    Image BitwiseOr( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                     size_t width, size_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseOr( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                    Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const size_t rowSizeIn1 = in1.rowSize();
        const size_t rowSizeIn2 = in2.rowSize();
        const size_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        const size_t simdWidth = width / simdSize;
        const size_t totalSimdWidth = simdWidth * simdSize;
        const size_t nonSimdWidth = width - totalSimdWidth;

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

    Image BitwiseXor( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                      size_t width, size_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseXor( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                     Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const size_t rowSizeIn1 = in1.rowSize();
        const size_t rowSizeIn2 = in2.rowSize();
        const size_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        const size_t simdWidth = width / simdSize;
        const size_t totalSimdWidth = simdWidth * simdSize;
        const size_t nonSimdWidth = width - totalSimdWidth;

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

    Image Invert( const Image & in, size_t startXIn, size_t startYIn, size_t width, size_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height );

        Invert( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    void Invert( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                 size_t width, size_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::Invert( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const size_t rowSizeIn  = in.rowSize();
        const size_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        const size_t simdWidth = width / simdSize;
        const size_t totalSimdWidth = simdWidth * simdSize;
        const size_t nonSimdWidth = width - totalSimdWidth;

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

    Image Maximum( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                   size_t width, size_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        Maximum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Maximum( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                  Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::Maximum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const size_t rowSizeIn1 = in1.rowSize();
        const size_t rowSizeIn2 = in2.rowSize();
        const size_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        const size_t simdWidth = width / simdSize;
        const size_t totalSimdWidth = simdWidth * simdSize;
        const size_t nonSimdWidth = width - totalSimdWidth;

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

    Image Minimum( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                   size_t width, size_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        Minimum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Minimum( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                  Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::Minimum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const size_t rowSizeIn1 = in1.rowSize();
        const size_t rowSizeIn2 = in2.rowSize();
        const size_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        const size_t simdWidth = width / simdSize;
        const size_t totalSimdWidth = simdWidth * simdSize;
        const size_t nonSimdWidth = width - totalSimdWidth;

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

    Image Subtract( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                    size_t width, size_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        Subtract( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Subtract( const Image & in1, size_t startX1, size_t startY1, const Image & in2, size_t startX2, size_t startY2,
                   Image & out, size_t startXOut, size_t startYOut, size_t width, size_t height )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );

        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width * colorCount < simdSize ) {
            Image_Function::Subtract( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        width = width * colorCount;

        const size_t rowSizeIn1 = in1.rowSize();
        const size_t rowSizeIn2 = in2.rowSize();
        const size_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        const size_t simdWidth = width / simdSize;
        const size_t totalSimdWidth = simdWidth * simdSize;
        const size_t nonSimdWidth = width - totalSimdWidth;

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

    Image Threshold( const Image & in, size_t startXIn, size_t startYIn, size_t width, size_t height, uint8_t threshold )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height );

        Threshold( in, startXIn, startYIn, out, 0, 0, width, height, threshold );

        return out;
    }

    void Threshold( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                    size_t width, size_t height, uint8_t threshold )
    {
        // image width is less than 16 bytes so no use to utilize NEON :(
        if( width < simdSize ) {
            Image_Function::Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, threshold );
            return;
        }

        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in, out );

        const size_t rowSizeIn  = in.rowSize();
        const size_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        const size_t simdWidth = width / simdSize;
        const size_t totalSimdWidth = simdWidth * simdSize;
        const size_t nonSimdWidth = width - totalSimdWidth;

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
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), minThreshold, maxThreshold );

        return out;
    }

    void Threshold( const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in, out );

        Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), minThreshold, maxThreshold );
    }

    Image Threshold( const Image & in, size_t startXIn, size_t startYIn, size_t width, size_t height, uint8_t minThreshold,
                     uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height );

        Threshold( in, startXIn, startYIn, out, 0, 0, width, height, minThreshold, maxThreshold );

        return out;
    }

    void Threshold( const Image & in, size_t startXIn, size_t startYIn, Image & out, size_t startXOut, size_t startYOut,
                    size_t width, size_t height, uint8_t minThreshold, uint8_t maxThreshold )
    {
        // image width is less than 16 bytes so no use to utilize NEOD :(
        if( width < simdSize ) {
            Image_Function::Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold );
            return;
        }

        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in, out );

        const size_t rowSizeIn  = in.rowSize();
        const size_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        const size_t simdWidth = width / simdSize;
        const size_t totalSimdWidth = simdWidth * simdSize;
        const size_t nonSimdWidth = width - totalSimdWidth;

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
};
