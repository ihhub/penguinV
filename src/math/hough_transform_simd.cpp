#include "hough_transform_simd.h"

#include <algorithm>
#include <cmath>

#include "../penguinv/cpu_identification.h"
#include "../image_function_helper.h"

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
    const bool isFloatAvailable  = std::is_standard_layout<PointBase2D<float>>::value;
    const bool isDoubleAvailable = std::is_standard_layout<PointBase2D<double>>::value;
    const size_t avxDouble  = 4u;
    const size_t avxFloat   = 8u;
    const size_t sseDouble  = 2u;
    const size_t sseFloat   = 4u;
    const size_t neonDouble = 2u;
    const size_t neonFloat  = 4u;

    const float minimumAngleStep = 0.001f * static_cast<float>( pvmath::pi ) / 180.0f;
    const float minimumLineTolerance = 1e-5f;
    
    template <typename _Type>
    void FindDistance( const std::vector< PointBase2D< _Type > > & input, std::vector < _Type > & distance, _Type cosVal, _Type sinVal, size_t offset = 0u )
    {
        if ( offset >= input.size() )
            throw imageException( "distance offset is superior to the input size" );

        _Type * distanceVal = distance.data() + offset;
        const PointBase2D<_Type> * point = input.data() + offset;
        const PointBase2D<_Type> * pointEnd = input.data() + input.size();

        for ( ; point != pointEnd; ++point, ++distanceVal )
            (*distanceVal) = point->x * sinVal + point->y * cosVal;
    }

    template <typename _Type>
    void FindDistanceSimd( const std::vector< PointBase2D< _Type > > & input, std::vector < _Type > & distance, _Type cosVal, _Type sinVal )
    {
        FindDistance( input, distance, cosVal, sinVal );
    }

    template <>
    void FindDistanceSimd< float >( const std::vector< PointBase2D< float > > & input, std::vector < float > & distance, float cosVal, float sinVal )
    {
        const simd::SIMDType simdType = simd::actualSimdType();

#ifdef PENGUINV_AVX_SET
        if ( isFloatAvailable && simdType == simd::avx_function ) {
            const size_t simdWidth = input.size() / avxFloat;
            const size_t totalSimdWidth = simdWidth * (avxFloat * 2);
            const size_t nonSimdWidth = (input.size() * 2) - totalSimdWidth;

            const float * point = reinterpret_cast<const float*>(input.data());
            const float * PointEndSimd = point + totalSimdWidth;

            float * distanceVal = distance.data();

            const float coefficients[8] = { cosVal, sinVal, cosVal, sinVal, cosVal, sinVal, cosVal, sinVal };
            const __m256 coeff = _mm256_loadu_ps( coefficients );
            const __m256i ctrl = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

            for( ;point != PointEndSimd; point += avxFloat, distanceVal += avxFloat ) {
                __m256 src1 = _mm256_loadu_ps(point);
                point += avxFloat;
                __m256 src2 = _mm256_loadu_ps(point);

                src1 = _mm256_mul_ps(src1, coeff);
                src2 = _mm256_mul_ps(src2, coeff);

                __m256 result = _mm256_hadd_ps(src1, src2);
                result = _mm256_permutevar8x32_ps(result, ctrl);

                _mm256_storeu_ps(distanceVal, result);
            }

            if( nonSimdWidth > 0 )
                FindDistance( input, distance, cosVal, sinVal, totalSimdWidth / 2 );

            return; // we have to return after execution to do not proceed to the rest of the code
        }
#endif

#ifdef PENGUINV_SSE_SET
        if ( isFloatAvailable && simdType == simd::sse_function ) {
            return; // we have to return after execution to do not proceed to the rest of the code
        }
#endif

#ifdef PENGUINV_NEON_SET
        if ( isFloatAvailable && simdType == simd::neon_function ) {
            return; // we have to return after execution to do not proceed to the rest of the code
        }
#endif

        FindDistance( input, distance, cosVal, sinVal ); // no SIMD found, run original CPU code
    }

    template <>
    void FindDistanceSimd< double >( const std::vector< PointBase2D< double > > & input, std::vector < double> & distance, double cosVal, double sinVal )
    {
        const simd::SIMDType simdType = simd::actualSimdType();

#ifdef PENGUINV_AVX_SET
        if ( isDoubleAvailable && simdType == simd::avx_function ) {
            const size_t simdWidth = input.size() / avxDouble;
            const size_t totalSimdWidth = simdWidth * (avxDouble * 2);
            const size_t nonSimdWidth = (input.size() * 2) - totalSimdWidth;

            const double * point = reinterpret_cast<const double*>(input.data());
            const double * PointEndSimd = point + totalSimdWidth;

            double * distanceVal = distance.data();

            const double coefficients[8] = { cosVal, sinVal, cosVal, sinVal };
            const __m256d coeff = _mm256_loadu_pd( coefficients );

            for( ;point != PointEndSimd; point += avxDouble, distanceVal += avxDouble ) {
                __m256d src1 = _mm256_loadu_pd(point);
                point += avxDouble;
                __m256d src2 = _mm256_loadu_pd(point);

                src1 = _mm256_mul_pd(src1, coeff);
                src2 = _mm256_mul_pd(src2, coeff);

                __m256d result = _mm256_hadd_pd(src1, src2);
                result = _mm256_permute4x64_pd(result, 0b11011000);

                _mm256_storeu_pd(distanceVal, result);
            }

            if( nonSimdWidth > 0 )
                FindDistance( input, distance, cosVal, sinVal, totalSimdWidth / 2 );

            return; // we have to return after execution to do not proceed to the rest of the code
        }
#endif

#ifdef PENGUINV_SSE_SET
        if ( isDoubleAvailable && simdType == simd::sse_function ) {
            return; // we have to return after execution to do not proceed to the rest of the code
        }
#endif

#ifdef PENGUINV_NEON_SET
        if ( isDoubleAvailable && simdType == simd::neon_function ) {
            return; // we have to return after execution to do not proceed to the rest of the code
        }
#endif

        FindDistance( input, distance, cosVal, sinVal ); // no SIMD found, run original CPU code
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
            FindDistanceSimd<_Type>(input, distanceToLine, cosVal, sinVal);

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

        FindDistanceSimd<_Type>(input, distanceToLine, cosVal, sinVal);

        _Type * distanceVal = distanceToLine.data();
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