#include "image_function_simd.h"

#include "image_function.h"
#include "image_function_helper.h"
#include "parameter_validation.h"
#include "penguinv/cpu_identification.h"

#ifdef PENGUINV_AVX_SET
#include <immintrin.h>
#endif

#ifdef PENGUINV_SSE_SET
#include <emmintrin.h>
#endif

#ifdef PENGUINV_NEON_SET
#include <arm_neon.h>
#endif

namespace
{
    struct FunctionRegistrator
    {
        Image_Function_Helper::FunctionTableHolder table;

        FunctionRegistrator()
        {
            table.AbsoluteDifference = &Image_Function_Simd::AbsoluteDifference;
            table.Accumulate         = &Image_Function_Simd::Accumulate;
            table.BitwiseAnd         = &Image_Function_Simd::BitwiseAnd;
            table.BitwiseOr          = &Image_Function_Simd::BitwiseOr;
            table.BitwiseXor         = &Image_Function_Simd::BitwiseXor;
            table.Invert             = &Image_Function_Simd::Invert;
            table.Maximum            = &Image_Function_Simd::Maximum;
            table.Minimum            = &Image_Function_Simd::Minimum;
            table.ProjectionProfile  = &Image_Function_Simd::ProjectionProfile;
            table.Subtract           = &Image_Function_Simd::Subtract;
            table.Sum                = &Image_Function_Simd::Sum;
            table.Threshold          = &Image_Function_Simd::Threshold;
            table.Threshold2         = &Image_Function_Simd::Threshold;

            Image_Function_Helper::registerFunctionTable( PenguinV_Image::Image(), table, true );
        }
    };

    const FunctionRegistrator functionRegistrator;
}

namespace avx
{
    const uint32_t simdSize = 32u;

#ifdef PENGUINV_AVX_SET
    typedef __m256i simd;

    // We are not sure that input data is aligned by 32 bytes so we use loadu() functions instead of load()

    void AbsoluteDifference( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                             uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    void Accumulate( uint32_t rowSize, const uint8_t * imageY, const uint8_t * imageYEnd, uint32_t * outY, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        simd zero = _mm256_setzero_si256();

        const uint32_t width = totalSimdWidth + nonSimdWidth;

        for( ; imageY != imageYEnd; imageY += rowSize, outY += width ) {
            const simd * src    = reinterpret_cast <const simd*> (imageY);
            const simd * srcEnd = src + simdWidth;
            simd       * dst    = reinterpret_cast <simd*> (outY);

            for( ; src != srcEnd; ++src ) {
                simd data = _mm256_loadu_si256( src );

                const simd dataLo  = _mm256_unpacklo_epi8( data, zero );
                const simd dataHi  = _mm256_unpackhi_epi8( data, zero );

                const simd data_1 = _mm256_unpacklo_epi16( dataLo, zero );
                const simd data_2 = _mm256_unpackhi_epi16( dataLo, zero );
                const simd data_3 = _mm256_unpacklo_epi16( dataHi, zero );
                const simd data_4 = _mm256_unpackhi_epi16( dataHi, zero );

                _mm256_storeu_si256( dst, _mm256_add_epi32( data_1, _mm256_loadu_si256( dst ) ) );
                ++dst;
                _mm256_storeu_si256( dst, _mm256_add_epi32( data_2, _mm256_loadu_si256( dst ) ) );
                ++dst;
                _mm256_storeu_si256( dst, _mm256_add_epi32( data_3, _mm256_loadu_si256( dst ) ) );
                ++dst;
                _mm256_storeu_si256( dst, _mm256_add_epi32( data_4, _mm256_loadu_si256( dst ) ) );
                ++dst;
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageX    = imageY + totalSimdWidth;
                const uint8_t * imageXEnd = imageX + nonSimdWidth;
                uint32_t      * outX      = outY + totalSimdWidth;

                for( ; imageX != imageXEnd; ++imageX, ++outX )
                    (*outX) += (*imageX);
            }   
        }
    }

    void BitwiseAnd( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    void BitwiseOr( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                    uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    void BitwiseXor( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    void Invert( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd,
                 uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const char maskValue = static_cast<char>(0xffu);
        const simd mask = _mm256_set_epi8(
            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue );

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

    void Maximum( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                  uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    void Minimum( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                  uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    void ProjectionProfile( uint32_t rowSize, const uint8_t * imageStart, uint32_t height, bool horizontal,
                            uint32_t * out, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const simd zero = _mm256_setzero_si256();

        if( horizontal ) {
            const uint8_t * imageSimdXEnd = imageStart + totalSimdWidth;

            for( ; imageStart != imageSimdXEnd; imageStart += simdSize, out += simdSize ) {
                const uint8_t * imageSimdY = imageStart;
                const uint8_t * imageSimdYEnd = imageSimdY + height * rowSize;
                simd simdSum_1 = _mm256_setzero_si256();
                simd simdSum_2 = _mm256_setzero_si256();
                simd simdSum_3 = _mm256_setzero_si256();
                simd simdSum_4 = _mm256_setzero_si256();

                simd * dst = reinterpret_cast <simd*> (out);

                for( ; imageSimdY != imageSimdYEnd; imageSimdY += rowSize) {
                    const simd * src    = reinterpret_cast <const simd*> (imageSimdY);

                    const simd data = _mm256_loadu_si256( src );

                    const simd dataLo  = _mm256_unpacklo_epi8( data, zero );
                    const simd dataHi  = _mm256_unpackhi_epi8( data, zero );

                    const simd data_1 = _mm256_unpacklo_epi16( dataLo, zero );
                    const simd data_2 = _mm256_unpackhi_epi16( dataLo, zero );
                    const simd data_3 = _mm256_unpacklo_epi16( dataHi, zero );
                    const simd data_4 = _mm256_unpackhi_epi16( dataHi, zero );
                    simdSum_1 = _mm256_add_epi32( data_1, simdSum_1 );
                    simdSum_2 = _mm256_add_epi32( data_2, simdSum_2 );
                    simdSum_3 = _mm256_add_epi32( data_3, simdSum_3 );
                    simdSum_4 = _mm256_add_epi32( data_4, simdSum_4 );
                }

                _mm256_storeu_si256( dst, _mm256_add_epi32( simdSum_1, _mm256_loadu_si256( dst ) ) );
                ++dst;
                _mm256_storeu_si256( dst, _mm256_add_epi32( simdSum_2, _mm256_loadu_si256( dst ) ) );
                ++dst;
                _mm256_storeu_si256( dst, _mm256_add_epi32( simdSum_3, _mm256_loadu_si256( dst ) ) );
                ++dst;
                _mm256_storeu_si256( dst, _mm256_add_epi32( simdSum_4, _mm256_loadu_si256( dst ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t* imageXEnd = imageStart + nonSimdWidth;

                for( ; imageStart != imageXEnd; ++imageStart, ++out ) {
                    const uint8_t * imageY    = imageStart;
                    const uint8_t * imageYEnd = imageY + height * rowSize;

                    for( ; imageY != imageYEnd; imageY += rowSize )
                        (*out) += (*imageY);
                }
            }
        }
        else {
            const uint8_t * imageYEnd = imageStart + height * rowSize;

            for( ; imageStart != imageYEnd; imageStart += rowSize, ++out ) {
                const simd * src    = reinterpret_cast <const simd*> (imageStart);
                const simd * srcEnd = src + simdWidth;
                simd simdSum = _mm256_setzero_si256();

                for( ; src != srcEnd; ++src ) {
                    simd data = _mm256_loadu_si256( src );

                    simd dataLo  = _mm256_unpacklo_epi8( data, zero );
                    simd dataHi  = _mm256_unpackhi_epi8( data, zero );
                    simd sumLoHi = _mm256_add_epi16( dataLo, dataHi );

                    simdSum = _mm256_add_epi32( simdSum, _mm256_add_epi32( _mm256_unpacklo_epi16( sumLoHi, zero ),
                                                                           _mm256_unpackhi_epi16( sumLoHi, zero ) ) );
                }

                if( nonSimdWidth > 0 ) {
                    const uint8_t * imageX    = imageStart + totalSimdWidth;
                    const uint8_t * imageXEnd = imageX + nonSimdWidth;

                    for( ; imageX != imageXEnd; ++imageX )
                        (*out) += (*imageX);
                }

                uint32_t output[8] = { 0 };
                _mm256_storeu_si256( reinterpret_cast <simd*>(output), simdSum );
                
                (*out) += output[0] + output[1] + output[2] + output[3] + output[4] + output[5] + output[6] + output[7];
            }
        }
    }

    void Subtract( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                   uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    uint32_t Sum( uint32_t rowSize, const uint8_t * imageY,const uint8_t * imageYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        uint32_t sum = 0;
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

        uint32_t output[8] ={ 0 };

        _mm256_storeu_si256( reinterpret_cast <simd*>(output), simdSum );

        return sum + output[0] + output[1] + output[2] + output[3] + output[4] + output[5] + output[6] + output[7];
    }

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t threshold,
                    uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        // AVX does not have command "great or equal to" so we have 2 situations:
        // when threshold value is 0 and it is not
        if( threshold > 0 ) {
            const char maskValue = static_cast<char>(0x80u);
            const simd mask = _mm256_set_epi8(
                maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
                maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
                maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
                maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue );

            const char compareValue = static_cast<char>((threshold - 1) ^ 0x80u);
            const simd compare = _mm256_set_epi8(
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue );

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
            for( ; outY != outYEnd; outY += rowSizeOut )
                memset( outY, 255u, sizeof( uint8_t ) * (totalSimdWidth + nonSimdWidth) );
        }
    }

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t minThreshold, uint8_t maxThreshold,
                    uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const char shiftMaskValue = static_cast<char>(0x80u);
        const simd shiftMask = _mm256_set_epi8(
            shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue,
            shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue,
            shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue,
            shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue );

        const char notMaskValue = static_cast<char>(0xffu);
        const simd notMask = _mm256_set_epi8(
            notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue,
            notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue,
            notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue,
            notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue );

        const char maxCompareValue = static_cast<char>(maxThreshold ^ 0x80u);
        const simd maxCompare = _mm256_set_epi8(
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue );

        if( minThreshold > 0 ) {
            const char minCompareValue = static_cast<char>((minThreshold - 1) ^ 0x80u);
            const simd minCompare = _mm256_set_epi8(
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue );


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
#endif
}

namespace sse
{
    const uint32_t simdSize = 16u;

#ifdef PENGUINV_SSE_SET
    typedef __m128i simd;

    // We are not sure that input data is aligned by 16 bytes so we use loadu() functions instead of load()

    void AbsoluteDifference( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                             uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst ) {
                simd data1 = _mm_loadu_si128( src1 );
                simd data2 = _mm_loadu_si128( src2 );
                _mm_storeu_si128( dst, _mm_sub_epi8( _mm_max_epu8( data1, data2 ), _mm_min_epu8( data1, data2 ) ) );
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

    void Accumulate( uint32_t rowSize, const uint8_t * imageY, const uint8_t * imageYEnd, uint32_t * outY, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        simd zero = _mm_setzero_si128();

        const uint32_t width = totalSimdWidth + nonSimdWidth;

        for( ; imageY != imageYEnd; imageY += rowSize, outY += width ) {
            const simd * src    = reinterpret_cast <const simd*> (imageY);
            const simd * srcEnd = src + simdWidth;
            simd       * dst    = reinterpret_cast <simd*> (outY);

            for( ; src != srcEnd; ++src ) {
                simd data = _mm_loadu_si128( src );

                const simd dataLo  = _mm_unpacklo_epi8( data, zero );
                const simd dataHi  = _mm_unpackhi_epi8( data, zero );

                const simd data_1 = _mm_unpacklo_epi16( dataLo, zero );
                const simd data_2 = _mm_unpackhi_epi16( dataLo, zero );
                const simd data_3 = _mm_unpacklo_epi16( dataHi, zero );
                const simd data_4 = _mm_unpackhi_epi16( dataHi, zero );

                _mm_storeu_si128( dst, _mm_add_epi32( data_1, _mm_loadu_si128( dst ) ) );
                ++dst;
                _mm_storeu_si128( dst, _mm_add_epi32( data_2, _mm_loadu_si128( dst ) ) );
                ++dst;
                _mm_storeu_si128( dst, _mm_add_epi32( data_3, _mm_loadu_si128( dst ) ) );
                ++dst;
                _mm_storeu_si128( dst, _mm_add_epi32( data_4, _mm_loadu_si128( dst ) ) );
                ++dst;
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageX    = imageY + totalSimdWidth;
                const uint8_t * imageXEnd = imageX + nonSimdWidth;
                uint32_t      * outX      = outY + totalSimdWidth;

                for( ; imageX != imageXEnd; ++imageX, ++outX )
                    (*outX) += (*imageX);
            }   
        }
    }

    void BitwiseAnd( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm_storeu_si128( dst, _mm_and_si128( _mm_loadu_si128( src1 ), _mm_loadu_si128( src2 ) ) );

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

    void BitwiseOr( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                    uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm_storeu_si128( dst, _mm_or_si128( _mm_loadu_si128( src1 ), _mm_loadu_si128( src2 ) ) );

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

    void BitwiseXor( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm_storeu_si128( dst, _mm_xor_si128( _mm_loadu_si128( src1 ), _mm_loadu_si128( src2 ) ) );

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

    void Invert( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd,
                 uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const char maskValue = static_cast<char>(0xffu);
        const simd mask = _mm_set_epi8( maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
                                        maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue );

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const simd * src1 = reinterpret_cast <const simd*> (inY);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++dst )
                _mm_storeu_si128( dst, _mm_andnot_si128( _mm_loadu_si128( src1 ), mask ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * inX  = inY  + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++inX )
                    (*outX) = ~(*inX);
            }
        }
    }

    void Maximum( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                  uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm_storeu_si128( dst, _mm_max_epu8( _mm_loadu_si128( src1 ), _mm_loadu_si128( src2 ) ) );

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

    void Minimum( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                  uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm_storeu_si128( dst, _mm_min_epu8( _mm_loadu_si128( src1 ), _mm_loadu_si128( src2 ) ) );

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

    void ProjectionProfile( uint32_t rowSize, const uint8_t * imageStart, uint32_t height, bool horizontal,
                            uint32_t * out, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const simd zero = _mm_setzero_si128();

        if( horizontal ) {
            const uint8_t * imageSimdXEnd = imageStart + totalSimdWidth;

            for( ; imageStart != imageSimdXEnd; imageStart += simdSize, out += simdSize ) {
                const uint8_t * imageSimdY    = imageStart;
                const uint8_t * imageSimdYEnd = imageSimdY + height * rowSize;
                simd simdSum_1 = _mm_setzero_si128();
                simd simdSum_2 = _mm_setzero_si128();
                simd simdSum_3 = _mm_setzero_si128();
                simd simdSum_4 = _mm_setzero_si128();

                simd * dst = reinterpret_cast <simd*> (out);

                for( ; imageSimdY != imageSimdYEnd; imageSimdY += rowSize) {
                    const simd * src    = reinterpret_cast <const simd*> (imageSimdY);

                    const simd data = _mm_loadu_si128( src );

                    const simd dataLo  = _mm_unpacklo_epi8( data, zero );
                    const simd dataHi  = _mm_unpackhi_epi8( data, zero );

                    const simd data_1 = _mm_unpacklo_epi16( dataLo, zero );
                    const simd data_2 = _mm_unpackhi_epi16( dataLo, zero );
                    const simd data_3 = _mm_unpacklo_epi16( dataHi, zero );
                    const simd data_4 = _mm_unpackhi_epi16( dataHi, zero );
                    simdSum_1 = _mm_add_epi32( data_1, simdSum_1 );
                    simdSum_2 = _mm_add_epi32( data_2, simdSum_2 );
                    simdSum_3 = _mm_add_epi32( data_3, simdSum_3 );
                    simdSum_4 = _mm_add_epi32( data_4, simdSum_4 );
                }

                _mm_storeu_si128( dst, _mm_add_epi32( simdSum_1, _mm_loadu_si128( dst ) ) );
                ++dst;
                _mm_storeu_si128( dst, _mm_add_epi32( simdSum_2, _mm_loadu_si128( dst ) ) );
                ++dst;
                _mm_storeu_si128( dst, _mm_add_epi32( simdSum_3, _mm_loadu_si128( dst ) ) );
                ++dst;
                _mm_storeu_si128( dst, _mm_add_epi32( simdSum_4, _mm_loadu_si128( dst ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageXEnd = imageStart + nonSimdWidth;

                for( ; imageStart != imageXEnd; ++imageStart, ++out ) {
                    const uint8_t * imageY    = imageStart;
                    const uint8_t * imageYEnd = imageY + height * rowSize;

                    for( ; imageY != imageYEnd; imageY += rowSize )
                        (*out) += (*imageY);
                }
            }
        }
        else {
            const uint8_t * imageYEnd = imageStart + height * rowSize;

            for( ; imageStart != imageYEnd; imageStart += rowSize, ++out ) {
                const simd * src    = reinterpret_cast <const simd*> (imageStart);
                const simd * srcEnd = src + simdWidth;
                simd simdSum = _mm_setzero_si128();

                for( ; src != srcEnd; ++src ) {
                    simd data = _mm_loadu_si128( src );

                    simd dataLo  = _mm_unpacklo_epi8( data, zero );
                    simd dataHi  = _mm_unpackhi_epi8( data, zero );
                    simd sumLoHi = _mm_add_epi16( dataLo, dataHi );

                    simdSum = _mm_add_epi32( simdSum, _mm_add_epi32( _mm_unpacklo_epi16( sumLoHi, zero ),
                                                                     _mm_unpackhi_epi16( sumLoHi, zero ) ) );
                }

                if( nonSimdWidth > 0 ) {
                    const uint8_t * imageX    = imageStart + totalSimdWidth;
                    const uint8_t * imageXEnd = imageX + nonSimdWidth;

                    for( ; imageX != imageXEnd; ++imageX )
                        (*out) += (*imageX);
                }

                uint32_t output[4] = { 0 };
                _mm_storeu_si128( reinterpret_cast <simd*>(output), simdSum );
                
                (*out) += output[0] + output[1] + output[2] + output[3];
            }
        }
    }

    void Subtract( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                   uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst ) {
                simd data = _mm_loadu_si128( src1 );
                _mm_storeu_si128( dst, _mm_sub_epi8( data, _mm_min_epu8( data, _mm_loadu_si128( src2 ) ) ) );
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

    uint32_t Sum( uint32_t rowSize, const uint8_t * imageY,const uint8_t * imageYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        uint32_t sum = 0;
        simd simdSum = _mm_setzero_si128();
        simd zero    = _mm_setzero_si128();

        for( ; imageY != imageYEnd; imageY += rowSize ) {
            const simd * src    = reinterpret_cast <const simd*> (imageY);
            const simd * srcEnd = src + simdWidth;

            for( ; src != srcEnd; ++src ) {
                simd data = _mm_loadu_si128( src );

                simd dataLo  = _mm_unpacklo_epi8( data, zero );
                simd dataHi  = _mm_unpackhi_epi8( data, zero );
                simd sumLoHi = _mm_add_epi16( dataLo, dataHi );

                simdSum = _mm_add_epi32( simdSum, _mm_add_epi32( _mm_unpacklo_epi16( sumLoHi, zero ),
                                                                 _mm_unpackhi_epi16( sumLoHi, zero ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageX    = imageY + totalSimdWidth;
                const uint8_t * imageXEnd = imageX + nonSimdWidth;

                for( ; imageX != imageXEnd; ++imageX )
                    sum += (*imageX);
            }
        }

        uint32_t output[4] ={ 0 };

        _mm_storeu_si128( reinterpret_cast <simd*>(output), simdSum );

        return sum + output[0] + output[1] + output[2] + output[3];
    }

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t threshold,
                    uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        // SSE does not have command "great or equal to" so we have 2 situations:
        // when threshold value is 0 and it is not
        if( threshold > 0 ) {
            const char maskValue = static_cast<char>(0x80u);
            const simd mask = _mm_set_epi8( maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
                                            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue );

            const char compareValue = static_cast<char>((threshold - 1) ^ 0x80u);
            const simd compare = _mm_set_epi8(
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue );

            for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
                const simd * src1 = reinterpret_cast <const simd*> (inY);
                simd       * dst  = reinterpret_cast <simd*> (outY);

                const simd * src1End = src1 + simdWidth;

                for( ; src1 != src1End; ++src1, ++dst )
                    _mm_storeu_si128( dst, _mm_cmpgt_epi8( _mm_xor_si128( _mm_loadu_si128( src1 ), mask ), compare ) );

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
            for( ; outY != outYEnd; outY += rowSizeOut )
                memset( outY, 255u, sizeof( uint8_t ) * (totalSimdWidth + nonSimdWidth) );
        }
    }

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t minThreshold, uint8_t maxThreshold,
                    uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const char shiftMaskValue = static_cast<char>(0x80u);
        const simd shiftMask = _mm_set_epi8( shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue,
                                             shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue,
                                             shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue,
                                             shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue );

        const char notMaskValue = static_cast<char>(0xffu);
        const simd notMask = _mm_set_epi8( notMaskValue, notMaskValue, notMaskValue, notMaskValue,
                                           notMaskValue, notMaskValue, notMaskValue, notMaskValue,
                                           notMaskValue, notMaskValue, notMaskValue, notMaskValue,
                                           notMaskValue, notMaskValue, notMaskValue, notMaskValue );

        const char minCompareValue = static_cast<char>(minThreshold ^ 0x80u);
        const simd minCompare = _mm_set_epi8(
            minCompareValue, minCompareValue, minCompareValue, minCompareValue,
            minCompareValue, minCompareValue, minCompareValue, minCompareValue,
            minCompareValue, minCompareValue, minCompareValue, minCompareValue,
            minCompareValue, minCompareValue, minCompareValue, minCompareValue );

        const char maxCompareValue = static_cast<char>(maxThreshold ^ 0x80u);
        const simd maxCompare = _mm_set_epi8(
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue );

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const simd * src1 = reinterpret_cast <const simd*> (inY);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++dst ) {
                simd data = _mm_xor_si128( _mm_loadu_si128( src1 ), shiftMask );

                _mm_storeu_si128( dst, _mm_andnot_si128(
                    _mm_or_si128(
                        _mm_cmplt_epi8( data, minCompare ),
                        _mm_cmpgt_epi8( data, maxCompare ) ),
                    notMask ) );
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
#endif
}

namespace neon
{
    const uint32_t simdSize = 16u;

#ifdef PENGUINV_NEON_SET
    typedef uint8x16_t simd;

    void AbsoluteDifference( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                             uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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
    
    void Accumulate( uint32_t rowSize, const uint8_t * imageY, const uint8_t * imageYEnd, uint32_t * outY, uint32_t simdWidth,
                     uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const uint8x8_t zero_8 = vdup_n_u8(0);

        const uint32_t width = totalSimdWidth + nonSimdWidth;

        for( ; imageY != imageYEnd; imageY += rowSize, outY += width ) {
            const uint8_t * src    = imageY;
            const uint8_t * srcEnd = src + totalSimdWidth;
            uint32_t      * dst    = outY;

            for( ; src != srcEnd; src+= simdSize ) {
                uint8x16_t data = vld1q_u8( src );

                const uint16x8_t dataLo  = vaddl_u8( vget_low_u8(data), zero_8 );
                const uint16x8_t dataHi  = vaddl_u8( vget_high_u8(data), zero_8 );

                vst1q_u32( dst, vaddw_u16( vld1q_u32( dst ), vget_low_u16(dataLo) ) );
                dst += 4;
                vst1q_u32( dst, vaddw_u16( vld1q_u32( dst ), vget_high_u16(dataLo) ) );
                dst += 4;
                vst1q_u32( dst, vaddw_u16( vld1q_u32( dst ), vget_low_u16(dataHi) ) );
                dst += 4;
                vst1q_u32( dst, vaddw_u16( vld1q_u32( dst ), vget_high_u16(dataHi) ) );
                dst += 4;
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageX    = imageY + totalSimdWidth;
                const uint8_t * imageXEnd = imageX + nonSimdWidth;
                uint32_t      * outX      = outY + totalSimdWidth;

                for( ; imageX != imageXEnd; ++imageX, ++outX )
                    (*outX) += (*imageX);
            }
        }
    }

    void BitwiseAnd( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    void BitwiseOr( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                    uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    void BitwiseXor( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    void Invert( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd,
                 uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    void Maximum( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                  uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    void Minimum( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                  uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    void ProjectionProfile( uint32_t rowSize, const uint8_t * imageStart, uint32_t height, bool horizontal,
                            uint32_t * out, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const uint8x8_t zero = vdup_n_u8(0);
        const uint16x4_t zero_16 = vdup_n_u16(0);

        if( horizontal ) {
            const uint8_t * imageSimdXEnd = imageStart + totalSimdWidth;

            for( ; imageStart != imageSimdXEnd; imageStart += simdSize, out += simdSize ) {
                const uint8_t * imageSimdY    = imageStart;
                const uint8_t * imageSimdYEnd = imageSimdY + height * rowSize;
                uint32x4_t simdSum_1 = vdupq_n_u32(0);
                uint32x4_t simdSum_2 = vdupq_n_u32(0);
                uint32x4_t simdSum_3 = vdupq_n_u32(0);
                uint32x4_t simdSum_4 = vdupq_n_u32(0);

                uint32_t * dst = out;

                for( ; imageSimdY != imageSimdYEnd; imageSimdY += rowSize) {
                    const uint8_t * src = imageSimdY;

                    const uint8x16_t data = vld1q_u8( src );

                    const uint16x8_t dataLo  = vaddl_u8( vget_low_u8(data), zero );
                    const uint16x8_t dataHi  = vaddl_u8( vget_high_u8(data), zero );

                    const uint32x4_t data_1  = vaddl_u16( vget_low_u16(dataLo), zero_16 );
                    const uint32x4_t data_2  = vaddl_u16( vget_high_u16(dataLo), zero_16 );
                    const uint32x4_t data_3  = vaddl_u16( vget_low_u16(dataLo), zero_16 );
                    const uint32x4_t data_4  = vaddl_u16( vget_high_u16(dataLo), zero_16 );

                    simdSum_1 = vaddq_u32( simdSum_1, data_1  );
                    simdSum_2 = vaddq_u32( simdSum_2, data_2 );
                    simdSum_3 = vaddq_u32( simdSum_3, data_3  );
                    simdSum_4 = vaddq_u32( simdSum_4, data_4 );
                }

                vst1q_u32( dst, vaddq_u32( vld1q_u32( dst ), simdSum_1 ) );
                dst += 4;
                vst1q_u32( dst, vaddq_u32( vld1q_u32( dst ), simdSum_2 ) );
                dst += 4;
                vst1q_u32( dst, vaddq_u32( vld1q_u32( dst ), simdSum_3 ) );
                dst += 4;
                vst1q_u32( dst, vaddq_u32( vld1q_u32( dst ), simdSum_4 ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageXEnd = imageStart + nonSimdWidth;

                for( ; imageStart != imageXEnd; ++imageStart, ++out ) {
                    const uint8_t * imageY    = imageStart;
                    const uint8_t * imageYEnd = imageY + height * rowSize;

                    for( ; imageY != imageYEnd; imageY += rowSize )
                        (*out) += (*imageY);
                }
            }
        }
        else {
            const uint8_t * imageYEnd = imageStart + height * rowSize;

            for( ; imageStart != imageYEnd; imageStart += rowSize, ++out ) {
                const uint8_t * src    = imageStart;
                const uint8_t * srcEnd = src + simdWidth*simdSize;
                uint32x4_t simdSum = vdupq_n_u32(0);

                for( ; src != srcEnd; src += simdSize ) {
                    const uint8x16_t data = vld1q_u8( src );

                    const uint16x8_t dataLo  = vaddl_u8( vget_low_u8(data), zero );
                    const uint16x8_t dataHi  = vaddl_u8( vget_high_u8(data), zero );
                    const uint16x8_t sumLoHi  = vaddq_u16( dataHi, dataLo );

                    const uint32x4_t sum = vaddl_u16( vadd_u16( vget_low_u16(sumLoHi), 
                                                                vget_high_u16(sumLoHi) ),
                                                                zero_16 );

                    simdSum = vaddq_u32( simdSum, sum );
                }

                if( nonSimdWidth > 0 ) {
                    const uint8_t * imageX    = imageStart + totalSimdWidth;
                    const uint8_t * imageXEnd = imageX + nonSimdWidth;

                    for( ; imageX != imageXEnd; ++imageX )
                        (*out) += (*imageX);
                }

                uint32_t output[4] = { 0 };
                vst1q_u32( output, simdSum );
                
                (*out) += output[0] + output[1] + output[2] + output[3];
            }
        }
    }

    void Subtract( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                   uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    uint32_t Sum( uint32_t rowSize, const uint8_t * imageY, const uint8_t * imageYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        uint32_t sum = 0;
        uint32x4_t simdSum = vdupq_n_u32(0);

        for( ; imageY != imageYEnd; imageY += rowSize ) {
            const uint8_t * src    = imageY;
            const uint8_t * srcEnd = src + simdWidth;

            for( ; src != srcEnd; ++src ) {
                const uint8x16_t data = vld1q_u8(src);
                const uint16x8_t data8Sum = vaddl_u8(vget_high_u8(data), vget_low_u8(data));
                const uint32x4_t data16Sum = vaddl_u16(vget_high_u16(data8Sum), vget_low_u16(data8Sum));
                simdSum = vaddq_u32(simdSum, data16Sum);
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageX    = imageY + totalSimdWidth;
                const uint8_t * imageXEnd = imageX + nonSimdWidth;

                for( ; imageX != imageXEnd; ++imageX )
                    sum += (*imageX);
            }
        }

        uint32_t output[4] = { 0 };
        vst1q_u32(output, simdSum);
        return (sum + output[0] + output[1] + output[2] + output[3]);
    }

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t threshold,
                    uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t minThreshold, uint8_t maxThreshold,
                    uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
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
#endif
}

namespace simd
{
    enum SIMDType
    {
        avx_function,
        sse_function,
        neon_function,
        cpu_function
    };

    bool isAvxEnabled = true;
    bool isSseEnabled = true;
    bool isNeonEnabled = true;

    SIMDType actualSimdType()
    {
        #ifdef PENGUINV_AVX_SET
        if ( isAvxAvailable && isAvxEnabled )
            return avx_function;
        #endif

        #ifdef PENGUINV_SSE_SET
        if ( isSseAvailable && isSseEnabled )
            return sse_function;
        #endif

        #ifdef PENGUINV_NEON_SET
        if ( isNeonAvailable && isNeonEnabled )
            return neon_function;
        #endif

        return cpu_function;
    }

    uint32_t getSimdSize( SIMDType simdType )
    {
        if ( simdType == avx_function )
            return avx::simdSize;
        if ( simdType == sse_function )
            return sse::simdSize;
        if ( simdType == neon_function )
            return neon::simdSize;
        if ( simdType == cpu_function )
            return 1u;

        return 0u;
    }

#ifdef PENGUINV_AVX_SET
#define AVX_CODE( code )          \
if ( simdType == avx_function ) { \
    code;                         \
    return;                       \
}
#else
#define AVX_CODE( code )
#endif

#ifdef PENGUINV_SSE_SET
#define SSE_CODE( code )          \
if ( simdType == sse_function ) { \
    code;                         \
    return;                       \
}
#else
#define SSE_CODE( code )
#endif

#ifdef PENGUINV_NEON_SET
#define NEON_CODE( code )          \
if ( simdType == neon_function ) { \
    code;                          \
    return;                        \
}
#else
#define NEON_CODE( code )
#endif

    using namespace PenguinV_Image;

    void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                             Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        if( (simdType == cpu_function) || (width * colorCount < simdSize) ) {
            AVX_CODE( AbsoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, sse_function ); )

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

        AVX_CODE( avx::AbsoluteDifference( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::AbsoluteDifference( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::AbsoluteDifference( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void Accumulate( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector<uint32_t> & result, SIMDType simdType )
    {
        const uint32_t simdSize = getSimdSize( simdType );
        const uint8_t colorCount = image.colorCount();

        if( (simdType == cpu_function) || ((width * height * colorCount) < simdSize) ) {
            Image_Function::Accumulate(image, x, y, width, height, result);
            return;
        }

        Image_Function::ParameterValidation( image, x, y, width, height );
        Image_Function::VerifyGrayScaleImage( image );

        if( result.size() != width * height * colorCount )
            throw imageException( "Array size is not equal to image ROI (width * height) size" );

        const uint32_t rowSize = image.rowSize();

        const uint8_t * imageY    = image.data() + y * rowSize + x * colorCount;
        const uint8_t * imageYEnd = imageY + height * rowSize;

        uint32_t * outY = result.data();
        width = width * colorCount;

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::Accumulate( rowSize, imageY, imageYEnd, outY, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Accumulate( rowSize, imageY, imageYEnd, outY, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Accumulate( rowSize, imageY, imageYEnd, outY, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        if( (simdType == cpu_function) || (width * colorCount < simdSize) ) {
            AVX_CODE( BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, sse_function ); )

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

        AVX_CODE( avx::BitwiseAnd( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::BitwiseAnd( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::BitwiseAnd( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        if( (simdType == cpu_function) || (width * colorCount < simdSize) ) {
            AVX_CODE( BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, sse_function ); )

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

        AVX_CODE( avx::BitwiseOr( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::BitwiseOr( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::BitwiseOr( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        if( (simdType == cpu_function) || (width * colorCount < simdSize) ) {
            AVX_CODE( BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, sse_function ); )

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

        AVX_CODE( avx::BitwiseXor( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::BitwiseXor( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::BitwiseXor( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                 uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in, out );
        const uint32_t simdSize = getSimdSize( simdType );

        if( (simdType == cpu_function) || (width * colorCount < simdSize) ) {
            AVX_CODE( Invert( in, startXIn, startYIn, out, startXOut, startYOut, width, height, sse_function ); )

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

        AVX_CODE( avx::Invert( rowSizeIn, rowSizeOut, inY, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Invert( rowSizeIn, rowSizeOut, inY, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Invert( rowSizeIn, rowSizeOut, inY, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        if( (simdType == cpu_function) || (width * colorCount < simdSize) ) {
            AVX_CODE( Maximum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, sse_function ); )

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

        AVX_CODE( avx::Maximum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Maximum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Maximum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        if( (simdType == cpu_function) || (width * colorCount < simdSize) ) {
            AVX_CODE( Minimum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, sse_function ); )

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

        AVX_CODE( avx::Minimum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Minimum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Minimum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                            std::vector < uint32_t > & projection, SIMDType simdType )
    {
        const uint32_t simdSize = getSimdSize( simdType );
        const uint8_t colorCount = image.colorCount();

        if( (simdType == cpu_function) || ((width * height * colorCount) < simdSize) ) {
            AVX_CODE( ProjectionProfile( image, x, y, width, height, horizontal, projection, sse_function ); )

            Image_Function::ProjectionProfile( image, x, y, width, height, horizontal, projection );
            return;
        }
        Image_Function::ParameterValidation( image, x, y, width, height );

        projection.resize( horizontal ? width * colorCount : height );
        std::fill( projection.begin(), projection.end(), 0u );
        uint32_t * out = projection.data();

        const uint32_t rowSize = image.rowSize();
        width = width * colorCount;
        const uint8_t * imageStart = image.data() + y * rowSize + x * colorCount; 

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::ProjectionProfile( rowSize, imageStart, height, horizontal, out, simdWidth, totalSimdWidth, nonSimdWidth ) )
        SSE_CODE( sse::ProjectionProfile( rowSize, imageStart, height, horizontal, out, simdWidth, totalSimdWidth, nonSimdWidth ) )
        NEON_CODE( neon::ProjectionProfile( rowSize, imageStart, height, horizontal, out, simdWidth, totalSimdWidth, nonSimdWidth ) )
    }

    void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        if( (simdType == cpu_function) || (width * colorCount < simdSize) ) {
            AVX_CODE( Subtract( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, sse_function ); )

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

        AVX_CODE( avx::Subtract( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Subtract( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Subtract( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint32_t simdSize = getSimdSize( simdType );

        if( (simdType == cpu_function) || (width < simdSize) ) {
            #ifdef PENGUINV_AVX_SET
            if ( simdType == avx_function )
                return Sum( image, x, y, width, height, sse_function );
            #endif

            return Image_Function::Sum( image, x, y, width, height );
        }

        Image_Function::ParameterValidation( image, x, y, width, height );
        Image_Function::VerifyGrayScaleImage( image );

        const uint32_t rowSize = image.rowSize();

        const uint8_t * imageY    = image.data() + y * rowSize + x;
        const uint8_t * imageYEnd = imageY + height * rowSize;

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        #ifdef PENGUINV_AVX_SET
        if ( simdType == avx_function )
            return avx::Sum( rowSize, imageY, imageYEnd, simdWidth, totalSimdWidth, nonSimdWidth );
        #endif
        #ifdef PENGUINV_SSE_SET
        if ( simdType == sse_function )
            return sse::Sum( rowSize, imageY, imageYEnd, simdWidth, totalSimdWidth, nonSimdWidth );
        #endif
        #ifdef PENGUINV_NEON_SET
        if (simdType == neon_function)
            return neon::Sum( rowSize, imageY, imageYEnd, simdWidth, totalSimdWidth, nonSimdWidth );
        #endif

        return 0u;
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t threshold, SIMDType simdType )
    {
        const uint32_t simdSize = getSimdSize( simdType );

        if( (simdType == cpu_function) || (width < simdSize) ) {
            AVX_CODE( Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, threshold, sse_function ); )

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

        AVX_CODE( avx::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, threshold, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, threshold, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, threshold, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold, SIMDType simdType )
    {
        const uint32_t simdSize = getSimdSize( simdType );

        if( (simdType == cpu_function) || (width < simdSize) ) {
            AVX_CODE( Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold, sse_function ); )

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

        AVX_CODE( avx::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, minThreshold, maxThreshold, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, minThreshold, maxThreshold, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, minThreshold, maxThreshold, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }
}

namespace Image_Function_Simd
{
    namespace Simd_Activation
    {
        void EnableSimd( bool enable )
        {
            EnableAvx( enable );
            EnableSse( enable );
            EnableNeon( enable );
        }

        void EnableAvx( bool enable )
        {
            simd::isAvxEnabled = enable;
        }

        void EnableSse( bool enable )
        {
            simd::isSseEnabled = enable;
        }

        void EnableNeon( bool enable )
        {
            simd::isNeonEnabled = enable;
        }
    }

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
        simd::AbsoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
    }

    void Accumulate( const Image & image, std::vector < uint32_t > & result )
    {
        Image_Function_Helper::Accumulate( Accumulate, image, result );
    }

    void Accumulate( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & result )
    {
        simd::Accumulate(image, x, y, width, height, result, simd::actualSimdType());
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
        simd::BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::Invert( in, startXIn, startYIn, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::Maximum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::Minimum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
    }

    std::vector < uint32_t > ProjectionProfile( const Image & image, bool horizontal )
    {
        return Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, horizontal );
    }
    void ProjectionProfile( const Image & image, bool horizontal, std::vector < uint32_t > & projection )
    {
        Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, horizontal, projection );
    }
    std::vector < uint32_t > ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal )
    {
        return Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, x, y, width, height, horizontal );
    }
    void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal, 
                            std::vector < uint32_t > & projection )
    {
        simd::ProjectionProfile( image, x, y, width, height, horizontal, projection, simd::actualSimdType() );
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
        simd::Subtract( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
    }

    uint32_t Sum( const Image & image )
    {
        return Sum( image, 0, 0, image.width(), image.height() );
    }

    uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        return simd::Sum( image, x, y, width, height, simd::actualSimdType() );
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
        simd::Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, threshold, simd::actualSimdType() );
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
        simd::Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold, simd::actualSimdType() );
    }
}
