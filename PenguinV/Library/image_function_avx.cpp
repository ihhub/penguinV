#include <immintrin.h>
#include "image_function_avx.h"
#include "image_function_sse.h"

// This unnamed namespace contains all necessary information to reduce bugs in SIMD function writing
namespace
{
    const size_t simdSize = 32u;
    typedef __m256i simd;
};

namespace Image_Function_Avx
{
    // We are not sure that input data is aligned by 16 bytes so we use loadu() functions instead of load()
    // You can change it in case if your application has always aligned by 32 images images and areas (ROIs - regions of interest)

    // All processors what support AVX 2.0 support SSE too

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

        // image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
        if( width * colorCount < simdSize ) {
            Image_Function_Sse::AbsoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
            return;
        }

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in1, in2, out );

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
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst ) {
                simd data1 = _mm256_loadu_si256( src1 );
                simd data2 = _mm256_loadu_si256( src2 );
                _mm256_storeu_si256( dst, _mm256_sub_epi8( _mm256_max_epu8( data1, data2 ), _mm256_min_epu8( data1, data2 ) ) );
            }

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

        // image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
        if( width * colorCount < simdSize ) {
            Image_Function_Sse::BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
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
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm256_storeu_si256( dst, _mm256_and_si256( _mm256_loadu_si256( src1 ), _mm256_loadu_si256( src2 ) ) );

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

        // image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
        if( width * colorCount < simdSize ) {
            Image_Function_Sse::BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
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
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm256_storeu_si256( dst, _mm256_or_si256( _mm256_loadu_si256( src1 ), _mm256_loadu_si256( src2 ) ) );

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

        // image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
        if( width * colorCount < simdSize ) {
            Image_Function_Sse::BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
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
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm256_storeu_si256( dst, _mm256_xor_si256( _mm256_loadu_si256( src1 ), _mm256_loadu_si256( src2 ) ) );

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

        // image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
        if( width * colorCount < simdSize ) {
            Image_Function_Sse::Invert( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
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

        const simd mask = _mm256_set_epi8(
            0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu,
            0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu,
            0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu,
            0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu );

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const simd * src1 = reinterpret_cast <const simd*> (inY);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++dst )
                _mm256_storeu_si256( dst, _mm256_andnot_si256( _mm256_loadu_si256( src1 ), mask ) );

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

        // image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
        if( width * colorCount < simdSize ) {
            Image_Function_Sse::Maximum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
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
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm256_storeu_si256( dst, _mm256_max_epu8( _mm256_loadu_si256( src1 ), _mm256_loadu_si256( src2 ) ) );

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

        // image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
        if( width * colorCount < simdSize ) {
            Image_Function_Sse::Minimum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
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
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm256_storeu_si256( dst, _mm256_min_epu8( _mm256_loadu_si256( src1 ), _mm256_loadu_si256( src2 ) ) );

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

        // image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
        if( width * colorCount < simdSize ) {
            Image_Function_Sse::Subtract( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
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
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst ) {
                simd data = _mm256_loadu_si256( src1 );
                _mm256_storeu_si256( dst, _mm256_sub_epi8( data, _mm256_min_epu8( data, _mm256_loadu_si256( src2 ) ) ) );
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

    size_t Sum( const Image & image )
    {
        return Sum( image, 0, 0, image.width(), image.height() );
    }

    size_t Sum( const Image & image, size_t x, size_t y, size_t width, size_t height )
    {
        // image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
        if( width < simdSize ) {
            return Image_Function_Sse::Sum( image, x, y, width, height );
        }

        Image_Function::ParameterValidation( image, x, y, width, height );
        Image_Function::VerifyGrayScaleImage( image );

        const size_t rowSize = image.rowSize();

        const uint8_t * imageY    = image.data() + y * rowSize + x;
        const uint8_t * imageYEnd = imageY + height * rowSize;

        size_t sum = 0;

        const size_t simdWidth = width / simdSize;
        const size_t totalSimdWidth = simdWidth * simdSize;
        const size_t nonSimdWidth = width - totalSimdWidth;

        simd simdSum = _mm256_setzero_si256();
        simd zero    = _mm256_setzero_si256();

        for( ; imageY != imageYEnd; imageY += rowSize ) {
            const simd * src    = reinterpret_cast <const simd*> (imageY);
            const simd * srcEnd = src + simdWidth;

            for( ; src != srcEnd; ++src ) {
                simd data = _mm256_loadu_si256( src );

                simd dataLo  = _mm256_unpacklo_epi8( data, zero );
                simd dataHi  = _mm256_unpackhi_epi8( data, zero );
                simd sumLoHi = _mm256_add_epi16( dataLo, dataHi );

                simdSum = _mm256_add_epi32( simdSum, _mm256_add_epi32( _mm256_unpacklo_epi16( sumLoHi, zero ),
                                                                       _mm256_unpackhi_epi16( sumLoHi, zero ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageX    = imageY + totalSimdWidth;
                const uint8_t * imageXEnd = imageX + nonSimdWidth;

                for( ; imageX != imageXEnd; ++imageX )
                    sum += (*imageX);
            }
        }

        size_t output[8] ={ 0 };

        _mm256_storeu_si256( reinterpret_cast <simd*>(output), simdSum );

        return sum + output[0] + output[1] + output[2] + output[3] + output[4] + output[5] + output[6] + output[7];
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
        // image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
        if( width < simdSize ) {
            Image_Function_Sse::Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, threshold );
            return;
        }

        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in, out );

        // AVX does not have command "great or equal to" so we have 2 situations:
        // when threshold value is 0 and it is not
        if( threshold > 0 ) {
            const size_t rowSizeIn  = in.rowSize();
            const size_t rowSizeOut = out.rowSize();

            const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
            uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

            const uint8_t * outYEnd = outY + height * rowSizeOut;

            const size_t simdWidth = width / simdSize;
            const size_t totalSimdWidth = simdWidth * simdSize;
            const size_t nonSimdWidth = width - totalSimdWidth;

            const simd mask = _mm256_set_epi8(
                0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u,
                0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u,
                0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u,
                0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u );

            const simd compare = _mm256_set_epi8(
                (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
                (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
                (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
                (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
                (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
                (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
                (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u,
                (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u, (threshold - 1) ^ 0x80u );

            for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
                const simd * src1 = reinterpret_cast <const simd*> (inY);
                simd       * dst  = reinterpret_cast <simd*> (outY);

                const simd * src1End = src1 + simdWidth;

                for( ; src1 != src1End; ++src1, ++dst )
                    _mm256_storeu_si256( dst, _mm256_cmpgt_epi8( _mm256_xor_si256( _mm256_loadu_si256( src1 ), mask ), compare ) );

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
            const size_t rowSizeOut = out.rowSize();

            uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;
            const uint8_t * outYEnd = outY + height * rowSizeOut;

            for( ; outY != outYEnd; outY += rowSizeOut )
                memset( outY, 255u, sizeof( uint8_t ) * width );
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
        // image width is less than 32 bytes so no use to utilize AVX 2.0 :( Let's try SSE!
        if( width < simdSize ) {
            Image_Function_Sse::Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold );
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

        const simd shiftMask = _mm256_set_epi8(
            0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u,
            0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u,
            0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u,
            0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u, 0x80u );

        const simd notMask = _mm256_set_epi8(
            0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu,
            0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu,
            0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu,
            0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu, 0xffu );

        const simd maxCompare = _mm256_set_epi8(
            maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u,
            maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u,
            maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u,
            maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u,
            maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u,
            maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u,
            maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u,
            maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u, maxThreshold ^ 0x80u );

        if( minThreshold > 0 ) {
            const simd minCompare = _mm256_set_epi8(
                (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u,
                (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u,
                (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u,
                (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u,
                (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u,
                (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u,
                (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u,
                (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u, (minThreshold - 1) ^ 0x80u );


            for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
                const simd * src1 = reinterpret_cast <const simd*> (inY);
                simd       * dst  = reinterpret_cast <simd*> (outY);

                const simd * src1End = src1 + simdWidth;

                for( ; src1 != src1End; ++src1, ++dst ) {
                    simd data = _mm256_xor_si256( _mm256_loadu_si256( src1 ), shiftMask );

                    _mm256_storeu_si256( dst, _mm256_and_si256(
                        _mm256_andnot_si256(
                            _mm256_cmpgt_epi8( data, maxCompare ), notMask ),
                        _mm256_cmpgt_epi8( data, minCompare ) ) );
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
        else {
            for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
                const simd * src1 = reinterpret_cast <const simd*> (inY);
                simd       * dst  = reinterpret_cast <simd*> (outY);

                const simd * src1End = src1 + simdWidth;

                for( ; src1 != src1End; ++src1, ++dst ) {
                    simd data = _mm256_xor_si256( _mm256_loadu_si256( src1 ), shiftMask );

                    _mm256_storeu_si256( dst, _mm256_andnot_si256( _mm256_cmpgt_epi8( data, maxCompare ), notMask ) );
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
};
