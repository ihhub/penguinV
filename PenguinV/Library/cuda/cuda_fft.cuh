#pragma once

#include <cufft.h>
#include <stdint.h>
#include "cuda_types.cuh"
#include "image_buffer_cuda.cuh"

namespace FFT_Cuda
{
    using namespace Bitmap_Image_Cuda;

    // This class store complex ([Re, Im]) data in GPU memory
    // It is used for Fast Fourier Transform
    class ComplexData
    {
    public:
        ComplexData();
        ComplexData( const Bitmap_Image_Cuda::Image & image );

        ComplexData( const ComplexData & data );
        ComplexData( ComplexData && data );

        ComplexData & operator=( const ComplexData & data );
        ComplexData & operator=( ComplexData && data );

        ~ComplexData();

        void set( const Bitmap_Image_Cuda::Image & image );
        void set( const Cuda_Types::Array<float> & data );

        // This function returns normalized image with swapped quadrants
        Bitmap_Image_Cuda::Image get() const;

        void resize( uint32_t width_, uint32_t height_ );

        cufftComplex * data(); // returns a pointer to GPU memory
        const cufftComplex * data() const; // returns a pointer to GPU memory
        uint32_t width() const; // width of array
        uint32_t height() const; // height of array
        bool empty() const; // returns true is array is empty (unullocated)

    private:
        cufftComplex * _data;
        uint32_t _width;
        uint32_t _height;

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
        FFTExecutor( uint32_t width_, uint32_t height_ );
        ~FFTExecutor();

        void initialize( uint32_t width_, uint32_t height_ );

        uint32_t width() const;
        uint32_t height() const;

        // conversion from original domain of data to frequence domain
        void directTransform( ComplexData & data );
        void directTransform( ComplexData & in, ComplexData & out );

        // conversion from frequence domain of data to original domain
        void inverseTransform( ComplexData & data );
        void inverseTransform( ComplexData & in, ComplexData & out );

        void complexMultiplication( ComplexData & in1, ComplexData & in2, ComplexData & out ) const;

    private:
        cufftHandle _plan;
        uint32_t _width;
        uint32_t _height;

        void _clean();
    };
}
