#pragma once

#include "cuda_fft.cuh"

namespace Image_Function_Cuda
{
    namespace Filtering
    {
        using namespace PenguinV_Image;

        Image Gaussian( const Image & in, uint32_t kernelSize, float sigma );
        void  Gaussian( const Image & in, Image & out, uint32_t kernelSize, float sigma );

        FFT_Cuda::ComplexData GetGaussianKernel( uint32_t width, uint32_t height, uint32_t kernelSize, float sigma );
    }
}
