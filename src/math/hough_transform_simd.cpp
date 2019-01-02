#include "hough_transform_simd.h"
#include "../image_function_helper.h"
#include "../penguinv/cpu_identification.h"

#ifdef PENGUINV_AVX_SET
#include <immintrin.h>
#endif

#ifdef PENGUINV_SSE_SET
#include <emmintrin.h>
#endif

#ifdef PENGUINV_NEON_SET
#include <arm_neon.h>
#endif

#include <algorithm>
#include <cmath>

namespace
{
    const bool float_layout_check  = std::is_standard_layout<PointBase2D<float>>::value;
    const bool double_layout_check = std::is_standard_layout<PointBase2D<double>>::value;
    const uint32_t avx_double = 4u;
    const uint32_t avx_float = 8u;
    const uint32_t sse_double = 2u;
    const uint32_t sse_float = 4u;
    const uint32_t neon_double = 2u;
    const uint32_t neon_float = 4u;

    const float minimumAngleStep = 0.001f * static_cast<float>( pvmath::pi ) / 180.0f;
    const float minimumLineTolerance = 1e-5f;
    
    template <typename _Type>
    void FindDistance( const std::vector< PointBase2D< _Type > > & input, std::vector < _Type > & distance, _Type cosVal, _Type sinVal )
    {
        _Type * distanceVal = distance.data();
        const PointBase2D<_Type> * point = input.data();
        const PointBase2D<_Type> * pointEnd = point + input.size();

        for ( ; point != pointEnd; ++point, ++distanceVal )
            (*distanceVal) = point->x * sinVal + point->y * cosVal;
    }

    template <typename _Type>
    void FindDistanceSimd( const std::vector< PointBase2D< _Type > > & input, std::vector < _Type > & distance, _Type cosVal, _Type sinVal, const size_t inputPointCount )
    {
        FindDistance( input, distance, cosVal, sinVal, inputPointCount );
    }

    template <>
    void FindDistanceSimd< float >( const std::vector< PointBase2D< float > > & input, std::vector < float > & distance, float cosVal, float sinVal, const size_t inputPointCount )
    {
        simd::SIMDType simdType = simd::actualSimdType();

        if ( simdType == simd::avx_function && float_layout_check ) {
            #ifdef PENGUINV_AVX_SET
            const uint32_t simdWidth = (inputPointCount*2) / (avx_float*2);
            const uint32_t totalSimdWidth = simdWidth * (avx_float*2);
            const uint32_t nonSimdWidth = (inputPointCount*2) - totalSimdWidth;

            const float * point = reinterpret_cast<const float*>(input.data());
            const float * PointEndSimd = point + totalSimdWidth;

            float * distanceVal = distance.data();

            const __m256 coeff = _mm256_set_ps(cosVal, sinVal, cosVal, sinVal, cosVal, sinVal, cosVal, sinVal);
            const __m256i ctrl = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

            for( ;point != PointEndSimd; point += avx_float, distanceVal += avx_float )
            {
                __m256 src1 = _mm256_loadu_ps(point);
                point += avx_float;
                __m256 src2 = _mm256_loadu_ps(point);

                src1 = _mm256_mul_ps (src1, coeff);
                src2 = _mm256_mul_ps (src2, coeff);

                __m256 result = _mm256_hadd_ps (src1, src2);
                result = _mm256_permutevar8x32_ps (result, ctrl);

                _mm256_storeu_ps(distanceVal, result);
            }

            if( nonSimdWidth > 0 )
            {
                const PointBase2D<float> * pointStruct = input.data() + totalSimdWidth / 2;
                const PointBase2D<float> * PointEnd = input.data() + inputPointCount;

                for ( ; pointStruct != PointEnd; ++pointStruct, ++distanceVal )
                    (*distanceVal) = pointStruct->x * sinVal + pointStruct->y * cosVal;
            }
            #endif
        }
        else if ( simdType == simd::sse_function && float_layout_check ) {
        }
        else if ( simdType == simd::neon_function && float_layout_check ){
        }
        else {
            FindDistance( input, distance, cosVal, sinVal );
        }
    }

    template <>
    void FindDistanceSimd< double >( const std::vector< PointBase2D< double > > & input, std::vector < double> & distance, double cosVal, double sinVal, const size_t inputPointCount )
    {
        simd::SIMDType simdType = simd::actualSimdType();

        if ( simdType == simd::avx_function && double_layout_check ) {
            #ifdef PENGUINV_AVX_SET
            const uint32_t simdWidth = (inputPointCount*2) / (avx_double*2);
            const uint32_t totalSimdWidth = simdWidth * (avx_double*2);
            const uint32_t nonSimdWidth = (inputPointCount*2) - totalSimdWidth;

            const double * point = reinterpret_cast<const double*>(input.data());
            const double * PointEndSimd = point + totalSimdWidth;

            double * distanceVal = distance.data();

            const __m256d coeff = _mm256_set_pd(cosVal, sinVal, cosVal, sinVal);

            for( ;point != PointEndSimd; point += avx_double, distanceVal += avx_double )
            {
                __m256d src1 = _mm256_loadu_pd(point);
                point += avx_double;
                __m256d src2 = _mm256_loadu_pd(point);

                src1 = _mm256_mul_pd(src1, coeff);
                src2 = _mm256_mul_pd(src2, coeff);

                __m256d result = _mm256_hadd_pd(src1, src2);
                result = _mm256_permute4x64_pd(result, 0b11011000);

                _mm256_storeu_pd(distanceVal, result);
            }

            if( nonSimdWidth > 0 )
            {
                const PointBase2D<double> * pointStruct = input.data() + totalSimdWidth / 2;
                const PointBase2D<double> * PointEnd = input.data() + inputPointCount;

                for ( ; pointStruct != PointEnd; ++pointStruct, ++distanceVal )
                    (*distanceVal) = pointStruct->x * sinVal + pointStruct->y * cosVal;
            }
            #endif
        }
        else if ( simdType == simd::sse_function && double_layout_check ) {
        }
        else if ( simdType == simd::neon_function && double_layout_check )
        {
        }
        else {
            FindDistance( input, distance, cosVal, sinVal );
        }
    }

    template <typename _Type>
    bool runHoughTransform( const std::vector< PointBase2D<_Type> > & input, _Type initialAngle, _Type angleTolerance, _Type angleStep,
                            _Type lineTolerance, std::vector< PointBase2D<_Type> > & outOnLine, std::vector< PointBase2D<_Type> > & outOffLine )
    {
        // validate input data
        if ( input.size() < 2u )
            return false;

        if ( angleStep < minimumAngleStep )
            angleStep = minimumAngleStep;

        if ( angleTolerance < minimumAngleStep )
            angleTolerance = minimumAngleStep;

        if ( angleTolerance < angleStep )
            angleTolerance = angleStep;

        if ( lineTolerance < minimumLineTolerance )
            lineTolerance = minimumLineTolerance;

        // find a range of search
        const int angleStepPerSide = static_cast<int>((angleTolerance / angleStep) + 0.5);
        const _Type lineToleranceRange = lineTolerance * 2;

        const size_t inputPointCount = input.size();
        std::vector < _Type > distanceToLine ( inputPointCount );

        int bestAngleId = -angleStepPerSide;
        size_t highestPointCount = 0u;
        _Type averageDistance = 0;

        _Type angleVal = -(initialAngle - angleStep * angleStepPerSide); // this should be an opposite angle

        for ( int angleId = -angleStepPerSide; angleId <= angleStepPerSide; ++angleId, angleVal -= angleStep ) {
            const _Type cosVal = std::cos( angleVal );
            const _Type sinVal = std::sin( angleVal );

            // find and sort distances
            FindDistanceSimd<_Type>(input, distanceToLine, cosVal, sinVal, inputPointCount);

            std::sort( distanceToLine.begin(), distanceToLine.end() );

            // find maximum number of points
            size_t initialPointId = 0u;
            size_t onLinePointCount = 1u;

            for ( size_t pointId = 0u, endPointId = 1u; endPointId < inputPointCount; ++pointId ) {
                const _Type tolerance = lineToleranceRange + distanceToLine[pointId];

                for ( ; endPointId < inputPointCount; ++endPointId ) {
                    if ( tolerance < distanceToLine[endPointId] )
                        break;
                }

                if ( onLinePointCount < endPointId - pointId ) {
                    onLinePointCount = endPointId - pointId;
                    initialPointId = pointId;
                }
            }

            if ( highestPointCount <= onLinePointCount ) {
                const _Type currentDistance = (distanceToLine[initialPointId + onLinePointCount - 1u] + distanceToLine[initialPointId]) / 2;
                if ( highestPointCount < onLinePointCount || std::abs( currentDistance ) < std::abs( averageDistance ) ) {
                    highestPointCount = onLinePointCount;
                    bestAngleId = angleId;
                    averageDistance = currentDistance;
                }
            }
        }

        outOnLine.clear();
        outOffLine.clear();

        angleVal = -(initialAngle + angleStep * bestAngleId);

        const _Type minDistance = averageDistance - lineTolerance;
        const _Type maxDistance = averageDistance + lineTolerance;

        // sort points
        const _Type cosVal = std::cos( angleVal );
        const _Type sinVal = std::sin( angleVal );

        _Type * distanceVal = distanceToLine.data();

        FindDistanceSimd<_Type>(input, distanceToLine, cosVal, sinVal, inputPointCount);

        const PointBase2D<_Type> * point = input.data();
        const PointBase2D<_Type> * pointEnd = point + inputPointCount;

        for ( ; point != pointEnd; ++point, ++distanceVal ) {
            if ( ((*distanceVal) < minDistance) || ((*distanceVal) > maxDistance) )
                outOffLine.push_back( (*point) );
            else
                outOnLine.push_back( (*point) );
        }

        return true;
    }
}

namespace Image_Function_Simd
{
    bool HoughTransform( const std::vector< PointBase2D<double> > & input, double initialAngle, double angleTolerance, double angleStep,
                         double lineTolerance, std::vector< PointBase2D<double> > & outOnLine, std::vector< PointBase2D<double> > & outOffLine )
    {
        return runHoughTransform<double>(input, initialAngle, angleTolerance, angleStep, lineTolerance, outOnLine, outOffLine);
    }

    bool HoughTransform( const std::vector< PointBase2D<float> > & input, float initialAngle, float angleTolerance, float angleStep,
                         float lineTolerance, std::vector< PointBase2D<float> > & outOnLine, std::vector< PointBase2D<float> > & outOffLine )
    {
        return runHoughTransform<float>(input, initialAngle, angleTolerance, angleStep, lineTolerance, outOnLine, outOffLine);
    }
}
