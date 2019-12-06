#pragma once

#if !defined(_MSC_VER) && !defined(_WIN32)

#include <cufft.h>
#include <stdint.h>
#include "cuda_types.cuh"
#include "../math/fft_base.h"
#include "image_buffer_cuda.cuh"

namespace FFT_Cuda
{
    // This class store complex ([Re, Im]) data in GPU memory
    // It is used for Fast Fourier Transform
    class ComplexData : public FFT::BaseComplexData<cufftComplex>
    {
    public:
        ComplexData();
        ComplexData( const penguinV::Image & image );

        ComplexData( const ComplexData & data );
        ComplexData( ComplexData && data );

        ~ComplexData();

        void set( const penguinV::Image & image );
        void set( const multiCuda::Array<float> & data );

        // This function returns normalized image with swapped quadrants
        penguinV::Image get() const;

    private:
        void _allocateData( size_t size ) override;
        void _freeData() override;
        void _copyData( const BaseComplexData<cufftComplex> & data ) override;
    };

    // The class for FFT commands execution like:
    // conversion from original domain of data to frequence domain and vice versa,
    // complex multiplication in frequency domain (convolution)
    class FFTExecutor : public FFT::BaseFFTExecutor
    {
    public:
        FFTExecutor( uint32_t width_ = 0u, uint32_t height_ = 0u );
        ~FFTExecutor();

        // conversion from original domain of data to frequency domain
        void directTransform( ComplexData & data );
        void directTransform( ComplexData & in, ComplexData & out );

        // conversion from frequence domain of data to original domain
        void inverseTransform( ComplexData & data );
        void inverseTransform( ComplexData & in, ComplexData & out );

        void complexMultiplication( const ComplexData & in1, ComplexData & in2, ComplexData & out ) const;

    private:
        cufftHandle _plan;

        void _cleanPlans() override;
        void _makePlans() override;
    };
}
#endif
