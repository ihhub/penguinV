#pragma once
#include <vector>
#include "image_buffer.h"
#include "math/fft_base.h"
#include "thirdparty/kissfft/kiss_fft.h"
#include "thirdparty/kissfft/kiss_fftnd.h"

namespace FFT
{
    // This class store complex ([real, imaginary]) data in CPU memory
    // It is used for Fast Fourier Transform
    class ComplexData : public BaseComplexData<kiss_fft_cpx>
    {
    public:
        ComplexData();
        ComplexData( const penguinV::Image & image );
        ComplexData( const std::vector<float> & data, uint32_t width, uint32_t height );

        ComplexData( const BaseComplexData<kiss_fft_cpx> & data );
        ComplexData( ComplexData && data );

        ~ComplexData();

        void set( const penguinV::Image & image );
        void set( const std::vector<float> & data );

        // This function returns normalized image with swapped quadrants
        penguinV::Image get() const;

    private:
        void _allocateData( size_t size ) override;
        void _freeData() override;
        void _copyData( const BaseComplexData<kiss_fft_cpx> & data ) override;
    };

    // The class for FFT command execution:
    // - conversion from original domain of data to frequency domain and vice versa
    // - complex multiplication in frequency domain (convolution)
    class FFTExecutor : public BaseFFTExecutor
    {
    public:
        FFTExecutor( uint32_t width_ = 0u, uint32_t height_ = 0u );
        ~FFTExecutor();

        // conversion from original domain of data to frequence domain
        void directTransform( ComplexData & data );
        void directTransform( const ComplexData & in, ComplexData & out );

        // conversion from frequence domain of data to original domain
        void inverseTransform( ComplexData & data );
        void inverseTransform( const ComplexData & in, ComplexData & out );

        void complexMultiplication( const ComplexData & in1, const ComplexData & in2, ComplexData & out ) const;

    private:
        kiss_fftnd_cfg _planDirect;
        kiss_fftnd_cfg _planInverse;

        void _makePlans() override;
        void _cleanPlans() override;
    };
}
