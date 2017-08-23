#include <cuda_runtime.h>
#include <math.h>
#include "filtering_cuda.cuh"
#include "../image_buffer.h"
#include "../image_exception.h"

namespace Image_Function_Cuda
{
    namespace Filtering
    {
        Image Gaussian( const Image & in, uint32_t kernelSize, float sigma )
        {
            Image_Function::ParameterValidation( in );

            Image out( in.width(), in.height() );

            Gaussian( in, out, kernelSize, sigma );

            return out;
        }

        void Gaussian( const Image & in, Image & out, uint32_t kernelSize,  float sigma )
        {
            Image_Function::ParameterValidation( in, out );

            if( sigma < 0 )
                throw imageException( "Sigma value cannot be negative" );

            FFT_Cuda::ComplexData image( in );
            FFT_Cuda::ComplexData filter = GetGaussianKernel( in.width(), in.height(), kernelSize, sigma );

            FFT_Cuda::FFTExecutor executor( in.width(), in.height() );

            executor.directTransform( image );
            executor.directTransform( filter );

            executor.complexMultiplication( image, filter, image );

            executor.inverseTransform( image );

            out = image.get();
        }

        FFT_Cuda::ComplexData GetGaussianKernel( uint32_t width, uint32_t height, uint32_t kernelSize, float sigma )
        {
            if( width < 3 || height < 3 || kernelSize == 0 || width < (kernelSize * 2 + 1) || height < (kernelSize * 2 + 1) || sigma < 0 )
                throw imageException( "Incorrect input parameters for Gaussian filter kernel" );

            const uint32_t size = width * height;

            std::vector<float> data( size, 0 );

            static const float pi = 3.1415926536f;
            const float doubleSigma = sigma * 2;

            float * y = data.data() + (height / 2 - kernelSize) * width + width / 2 - kernelSize;
            const float * endY = y + (2 * kernelSize + 1) * width;

            float sum = 0;

            for( int32_t posY = -static_cast<int32_t>(kernelSize) ; y != endY; y += width, ++posY ) {
                float * x = y;
                const float * endX = x + 2 * kernelSize + 1;

                for( int32_t posX = -static_cast<int32_t>(kernelSize) ; x != endX; ++x, ++posX ) {
                    *x = 1.0f / (pi * doubleSigma) * exp( -(posX * posX + posY * posY) / doubleSigma );
                    sum += *x;
                }
            }

            const float normalization = 1.0f / sum;

            y = data.data() + (height / 2 - kernelSize) * width + width / 2 - kernelSize;

            for( int32_t posY = -static_cast<int32_t>(kernelSize) ; y != endY; y += width, ++posY ) {
                float * x = y;
                const float * endX = x + 2 * kernelSize + 1;

                for( int32_t posX = -static_cast<int32_t>(kernelSize) ; x != endX; ++x, ++posX ) {
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
