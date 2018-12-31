#include "hough_transform.h"
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

namespace
{
    const uint32_t avx_size = 32u;
    const uint32_t sse_size = 16u;
    const uint32_t neon_size = 16u;
}

namespace
{
    const float minimumAngleStep = 0.001f * static_cast<float>( pvmath::pi ) / 180.0f;
    const float minimumLineTolerance = 1e-5f;
}

namespace
{

    template <typename _Type>
    void FindDistance( const std::vector< PointBase2D< _Type > > & input, std::vector < _Type > & distance, _Type cosVal, _Type sinVal, const size_t inputPointCount )
    {
        _Type * distanceVal = distance.data();
        const PointBase2D<_Type> * point = input.data();
        const PointBase2D<_Type> * pointEnd = point + inputPointCount;

        for ( ; point != pointEnd; ++point, ++distanceVal )
            (*distanceVal) = point->x * sinVal + point->y * cosVal;
    }

    template <typename _Type>
    void FindDistanceSimd( const std::vector< PointBase2D< _Type > > & input, std::vector < _Type > & distance, _Type cosVal, _Type sinVal, const size_t inputPointCount, simd::SIMDType simdType )
    {
        FindDistance( input, distance, cosVal, sinVal, inputPointCount );
    }

    template <>
    void FindDistanceSimd< float >( const std::vector< PointBase2D< float > > & input, std::vector < float > & distance, float cosVal, float sinVal, const size_t inputPointCount, simd::SIMDType simdType )
    {
        // some special code here to handle SIMD like
        if ( simd::avx_function ) {
        }
        else if ( simd::sse_function ) {
        }
        else if ( simd::neon_function ){
        }
        else {
            FindDistance( input, distance, cosVal, sinVal, inputPointCount );
        }
    }

    template <>
    void FindDistanceSimd< double >( const std::vector< PointBase2D< double > > & input, std::vector < double> & distance, double cosVal, double sinVal, const size_t inputPointCount, simd::SIMDType simdType )
    {
        if ( simd::avx_function ) {
            #ifdef PENGUINV_AVX_SET
            const uint32_t simdWidth = inputPointCount / (avx_size*2);
            const uint32_t totalSimdWidth = simdWidth * (avx_size*2);
            const uint32_t nonSimdWidth = inputPointCount - totalSimdWidth;

            const double * point = input.data();
            const double * PointEndSimd = point + totalSimdWidth;

            const double * dst = distance.data()

            const __m256d coeff = _mm256_set_pd (cosval, sinval, cosval, sinval)

            for(;point != PointEndSimd; point += avx_size, dst += 4)
            {
                __m256d src1 = _mm256_load_pd(point);
                point += avx_size;
                __m256d src2 = _mm256_load_pd(point);

                src1 = _mm256_mul_pd (src1, coeff);
                src2 = _mm256_mul_pd (src2, coeff);

                __m256d resul = _mm256_hadd_pd (src1, src2);
                resul = _mm256_permute4x64_pd (resul, 0b11011000);

                _mm256_store_pd(dst, resul);
            }

            if(nonSimdWidth > 0)
            {
                const double * PointEnd = PointEndSimd + nonSimdWidth;
                for ( ; point != pointEnd; ++point, ++dst)
                    (*dst) = point->x * sinVal + point->y * cosVal;
            }
            #endif
        }
        else if ( simd::sse_function ) {
        }
        else if ( simd::neon_function )
        {
        }
        else {
            FindDistance( input, distance, cosVal, sinVal, inputPointCount );
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
            const _Type cosVal = cos( angleVal );
            const _Type sinVal = sin( angleVal );

            // find and sort distances
            if(std::is_standard_layout<PointBase2D<_Type>>::value)
            {
                FindDistanceSimd<_Type>(input, distanceToLine, cosVal, sinVal, simd::actualSimdType())
            }
            else
            {
                FindDistance<_Type>( input, distanceToLine, cosVal, sinVal )
            }

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
        const _Type cosVal = cos( angleVal );
        const _Type sinVal = sin( angleVal );

        _Type * distanceVal = distanceToLine.data();
        const PointBase2D<_Type> * point = input.data();
        const PointBase2D<_Type> * pointEnd = point + inputPointCount;

        for ( ; point != pointEnd; ++point, ++distanceVal ) {
            (*distanceVal) = point->x * sinVal + point->y * cosVal;

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