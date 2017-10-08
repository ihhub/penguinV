#pragma once
#include <vector>
#include "image_buffer.h"
#include "thirdparty/kissfft/kiss_fft.h"
#include "thirdparty/kissfft/kiss_fftnd.h"

namespace FFT
{
    // This class store complex ([Re, Im]) data in GPU memory
    // It is used for Fast Fourier Transform
    class ComplexData
    {
    public:
        ComplexData();
        ComplexData( const Bitmap_Image::Image & image );

        ComplexData( const ComplexData & data );
        ComplexData( ComplexData && data );

        ComplexData & operator=( const ComplexData & data );
        ComplexData & operator=( ComplexData && data );

        ~ComplexData();

        void set( const Bitmap_Image::Image & image );
        void set( const std::vector<float> & data );

        // This function returns normalized image with swapped quadrants
        Bitmap_Image::Image get() const;

        void resize( size_t width_, size_t height_ );

        kiss_fft_cpx * data(); // returns a pointer to data
        const kiss_fft_cpx * data() const; // returns to data
        size_t width() const; // width of array
        size_t height() const; // height of array
        bool empty() const; // returns true is array is empty (unullocated)

    private:
        kiss_fft_cpx * _data;
        size_t _width;
        size_t _height;

        void _clean();

        void _copy( const ComplexData & data );
        void _swap( ComplexData & data );
    };

    // The class for FFT commands execution like:
    // conversion from original domain of data to frequence domain and vice versa,
    // complex multiplication in frequency domain (convolution)
    class FFTExecutor
    {
    public:
        FFTExecutor();
        FFTExecutor( size_t width_, size_t height_ );
        ~FFTExecutor();

        void initialize( size_t width_, size_t height_ );

        size_t width() const;
        size_t height() const;

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
        size_t _width;
        size_t _height;

        void _clean();
    };
}
