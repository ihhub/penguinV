#include <cmath>
#include <cuda_runtime.h>
#include "filtering_cuda.cuh"
#include "../filtering.h"
#include "../image_buffer.h"
#include "../image_exception.h"
#include "../parameter_validation.h"

namespace Image_Function_Cuda
{
    Image Gaussian( const Image & in, uint32_t kernelSize, float sigma )
    {
        Image_Function::ParameterValidation( in );

        ImageCuda out( in.width(), in.height() );

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
        std::vector<float> data;
        Image_Function::GetGaussianKernel( data, width, height, kernelSize, sigma );

        multiCuda::Array<float> cudaData( data );

        FFT_Cuda::ComplexData complexData;
        complexData.resize( width, height );
        complexData.set( cudaData );

        return complexData;
    }
}
