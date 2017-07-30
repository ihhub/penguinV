#pragma once

#include "cuda_fft.cuh"

namespace Image_Function_Cuda
{
    namespace Filtering
    {
        using namespace Bitmap_Image_Cuda;

        Image Gaussian( const Image & in, float sigma );
        void  Gaussian( const Image & in, Image & out, float sigma );

        FFT_Cuda::ComplexData GetGaussianKernel( uint32_t width, uint32_t height, float sigma );
    }
}
