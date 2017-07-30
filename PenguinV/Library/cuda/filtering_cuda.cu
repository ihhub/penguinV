#include <cuda_runtime.h>
#include <math.h>
#include "filtering_cuda.cuh"
#include "../image_buffer.h"
#include "../image_exception.h"

namespace Image_Function_Cuda
{
    namespace Filtering
    {
        Image Gaussian( const Image & in, float sigma )
        {
            Image_Function::ParameterValidation( in );

            Image out( in.width(), in.height() );

            Gaussian( in, out, sigma );

            return out;
        }

        void Gaussian( const Image & in, Image & out, float sigma )
        {
            Image_Function::ParameterValidation( in, out );

            if( sigma < 0 )
                throw imageException( "Sigma value cannot be negative" );

            FFT_Cuda::ComplexData image( in );
            FFT_Cuda::ComplexData filter = GetGaussianKernel( in.width(), in.height(), sigma );

            FFT_Cuda::FFTExecutor executor( in.width(), in.height() );

            executor.directTransform( image );
            executor.directTransform( filter );
            
            executor.complexMultiplication( image, filter, image );

            executor.inverseTransform( image );

            out = image.get();
        }

        FFT_Cuda::ComplexData GetGaussianKernel( uint32_t width, uint32_t height, float sigma )
        {
            if( width < 3 || height < 3 || sigma < 0 )
                throw imageException( "Incorrect input parameters for Gaussian filter kernel" );

            const uint32_t size = width * height;

            std::vector<float> data( size, 0 );

            uint32_t filterSize = static_cast<uint32_t>(3 * sigma + 0.5);
            if( (filterSize * 2 + 1) > width )
                filterSize = (width  - 1) / 2;
            if( (filterSize * 2 + 1) > height )
                filterSize = (height - 1) / 2;
            if( filterSize == 0 )
                filterSize = 1;

            static const float pi = 3.1415926536f;
            const float doubleSigma = sigma * 2;

            float * y = data.data() + (height / 2 - filterSize) * width + width / 2 - filterSize;
            const float * endY = y + (2 * filterSize + 1) * width;

            float sum = 0;

            for( int32_t posY = -static_cast<int32_t>(filterSize) ; y != endY; y += width, ++posY ) {
                float * x = y;
                const float * endX = x + 2 * filterSize + 1;

                for( int32_t posX = -static_cast<int32_t>(filterSize) ; x != endX; ++x, ++posX ) {
                    *x = 1.0f / ( pi * doubleSigma ) * exp( -(posX * posX + posY * posY) / doubleSigma );
                    sum += *x;
                }
            }

            const float normalization = 1.0f / sum;

            y = data.data() + (height / 2 - filterSize) * width + width / 2 - filterSize;

            for( int32_t posY = -static_cast<int32_t>(filterSize) ; y != endY; y += width, ++posY ) {
                float * x = y;
                const float * endX = x + 2 * filterSize + 1;

                for( int32_t posX = -static_cast<int32_t>(filterSize) ; x != endX; ++x, ++posX ) {
                    *x *= normalization;
                }
            }

            Cuda_Types::Array<float> cudaData( data );

            FFT_Cuda::ComplexData complexData;
            complexData.resize( width, height );
            complexData.set( cudaData );

            return complexData;
        }
    }
}
