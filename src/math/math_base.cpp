#include "math_base.h"
#include "../penguinv_exception.h"

namespace pvmath
{
    template <>
    bool isEqual<double>( const double & value1, const double & value2 )
    {
        return std::abs( value1 - value2 ) < epsilonDouble;
    }

    template <>
    bool isEqual<float>( const float & value1, const float & value2 )
    {
        return std::fabs( value1 - value2 ) < epsilonFloat;
    }

    template <>
    bool isEqual<double>( const double & value1, const double & value2, const double epsilonMultiplier )
    {
        return std::abs( value1 - value2 ) < epsilonDouble * epsilonMultiplier;
    }

    template <>
    bool isEqual<float>( const float & value1, const float & value2, const float epsilonMultiplier )
    {
        return std::fabs( value1 - value2 ) < epsilonFloat * epsilonMultiplier;
    }

    double toRadians( double angleDegree )
    {
        return angleDegree * pi / 180;
    }

    double toDegrees( double angleRadians )
    {
        return angleRadians * 180 / pi;
    }

    void getMatrixRoots( const std::vector<double> & squareMatrix, const std::vector<double> freeTerms, std::vector<double> & roots )
    {
        if ( squareMatrix.empty() || freeTerms.empty() || squareMatrix.size() != freeTerms.size() * freeTerms.size() || freeTerms.size() < 2 )
            throw penguinVException( "Invalid parameters for getMatrixRoots function" );

        const size_t size = freeTerms.size();
        std::vector<size_t> matrixIndices( size * size );

        for ( size_t i = 0; i < size; ++i )
            for ( size_t j = 0; j < size; ++j )
                matrixIndices[i + j * size] = i;

        std::vector<double> matrix( squareMatrix );
        std::vector<double> terms( freeTerms );

        // Direct method
        for ( size_t i = 0u; i < size; ++i ) {
            size_t maxRowId = i;
            size_t maxColumnId = i;
            double maxCoeff = 0;

            for ( size_t j = i; j < size; ++j ) {
                for ( size_t k = i; k < size; ++k ) {
                    const double value = fabs( matrix[j + k * size] );
                    if ( value > maxCoeff ) {
                        maxCoeff = value;
                        maxColumnId = j;
                        maxRowId = k;
                    }
                }
            }

            if ( maxCoeff < 1e-10 )
                throw penguinVException( "Invalid matrix" );

            if ( maxColumnId != i ) {
                for ( size_t k = 0; k < size; ++k ) {
                    std::swap( matrix[i + k * size], matrix[maxColumnId + k * size] );
                    std::swap( matrixIndices[i + k * size], matrixIndices[maxColumnId + k * size] );
                }
            }

            if ( maxRowId != i ) {
                for ( size_t j = 0; j < size; ++j ) {
                    std::swap( matrix[j + i * size], matrix[j + maxRowId * size] );
                    std::swap( matrixIndices[j + i * size], matrixIndices[j + maxRowId * size] );
                }

                std::swap( terms[i], terms[maxRowId] );
            }

            for ( size_t j = i + 1; j < size; ++j ) {
                double multiplicator = matrix[i + j * size] / matrix[i + i * size];

                for ( size_t k = i; k < size; ++k )
                    matrix[k + j * size] = matrix[k + j * size] - matrix[k + i * size] * multiplicator;

                terms[j] = terms[j] - terms[i] * multiplicator;
            }
        }

        // Inverse method
        std::vector<double> unsortedRoots( size );
        for ( size_t i = size - 1;; --i ) {
            if ( fabs( matrix[i + i * size] ) < 1e-10 ) {
                unsortedRoots[i] = 0;
            }
            else {
                double temp = 0;
                for ( size_t k = i + 1; k < size; ++k )
                    temp += unsortedRoots[k] * matrix[k + i * size];

                unsortedRoots[i] = ( terms[i] - temp ) / matrix[i + i * size];
            }

            if ( i == 0 )
                break;
        }

        // Results
        for ( size_t i = 0; i < size; ++i )
            roots[matrixIndices[i + i * size]] = unsortedRoots[i];
    }
}
