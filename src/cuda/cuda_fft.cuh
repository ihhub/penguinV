#pragma once

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
        ComplexData( const PenguinV_Image::Image & image );

        ComplexData( const ComplexData & data );
        ComplexData( ComplexData && data );

        ~ComplexData();

        void set( const PenguinV_Image::Image & image );
        void set( const multiCuda::Array<float> & data );

        // This function returns normalized image with swapped quadrants
        PenguinV_Image::Image get() const;

    private:
        void _allocateData(size_t nBytes) override;
        void _freeData() override;
        void _copyData(const BaseComplexData<cufftComplex> & data) override;
    };

    // The class for FFT commands execution like:
    // conversion from original domain of data to frequence domain and vice versa,
    // complex multiplication in frequency domain (convolution)
    class FFTExecutor : public FFT::BaseFFTExecutor
    {
    public:
        FFTExecutor();
        FFTExecutor( uint32_t width_, uint32_t height_ );
        ~FFTExecutor();
    
        bool dimensionsMatch( const ComplexData & data) const;
        using BaseFFTExecutor::dimensionsMatch;

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
        void _makePlans(const uint32_t width_, const uint32_t height_) override;
    };
}
