#pragma once

#include <stdint.h>
#include "../image_buffer.h"
#include "cuda_helper.cuh"
#include "image_buffer_cuda.cuh"

namespace Image_Function_Cuda
{
    using namespace Bitmap_Image_Cuda;

    Image AbsoluteDifference( const Image & in1, const Image & in2, cudaStream_t stream = Cuda::getCudaStream() );
    void  AbsoluteDifference( const Image & in1, const Image & in2, Image & out, cudaStream_t stream = Cuda::getCudaStream() );

    Image BitwiseAnd( const Image & in1, const Image & in2, cudaStream_t stream = Cuda::getCudaStream() );
    void  BitwiseAnd( const Image & in1, const Image & in2, Image & out, cudaStream_t stream = Cuda::getCudaStream() );

    Image BitwiseOr( const Image & in1, const Image & in2, cudaStream_t stream = Cuda::getCudaStream() );
    void  BitwiseOr( const Image & in1, const Image & in2, Image & out, cudaStream_t stream = Cuda::getCudaStream() );

    Image BitwiseXor( const Image & in1, const Image & in2, cudaStream_t stream = Cuda::getCudaStream() );
    void  BitwiseXor( const Image & in1, const Image & in2, Image & out, cudaStream_t stream = Cuda::getCudaStream() );

    // To archive maximum performance it is recommended that Bitmap_Image::Image has 1 pixel alignment
    // or (width * color count) will be equal to row size
    Image ConvertToCuda( const Bitmap_Image::Image & in );
    void  ConvertToCuda( const Bitmap_Image::Image & in, Image & out );
    Bitmap_Image::Image ConvertFromCuda( const Image & in );
    void                ConvertFromCuda( const Image & in, Bitmap_Image::Image & out );

    Image ConvertToGrayScale( const Image & in, cudaStream_t stream = Cuda::getCudaStream() );
    void  ConvertToGrayScale( const Image & in, Image & out, cudaStream_t stream = Cuda::getCudaStream() );

    Image ConvertToRgb( const Image & in, cudaStream_t stream = Cuda::getCudaStream() );
    void  ConvertToRgb( const Image & in, Image & out, cudaStream_t stream = Cuda::getCudaStream() );

    void  Copy( const Image & in, Image & out );

    Image ExtractChannel( const Image & in, uint8_t channelId, cudaStream_t stream = Cuda::getCudaStream() );
    void  ExtractChannel( const Image & in, Image & out, uint8_t channelId, cudaStream_t stream = Cuda::getCudaStream() );

    void  Fill( Image & image, uint8_t value, cudaStream_t stream = Cuda::getCudaStream() );

    // Make sure that input parameters such as input and output images are not same image!
    // horizontal flip: left-right --> right-left
    // vertical flip: top-bottom --> bottom-top
    Image Flip( const Image & in, bool horizontal, bool vertical, cudaStream_t stream = Cuda::getCudaStream() );
    void  Flip( const Image & in, Image & out, bool horizontal, bool vertical, cudaStream_t stream = Cuda::getCudaStream() );

    // Gamma correction works by formula:
    // output = A * ((input / 255) ^ gamma) * 255, where A - multiplication, gamma - power base. Both values must be greater than 0
    // Usually people set A as 1
    Image GammaCorrection( const Image & in, double a, double gamma, cudaStream_t stream = Cuda::getCudaStream() );
    void  GammaCorrection( const Image & in, Image & out, double a, double gamma, cudaStream_t stream = Cuda::getCudaStream() );

    uint8_t GetThreshold( const std::vector < uint32_t > & histogram );

    std::vector < uint32_t > Histogram( const Image & image, cudaStream_t stream = Cuda::getCudaStream() );
    void                     Histogram( const Image & image, std::vector < uint32_t > & histogram, cudaStream_t stream = Cuda::getCudaStream() );

    // Invert function is Bitwise NOT operation. But to make function name more user-friendly we named it like this
    Image Invert( const Image & in, cudaStream_t stream = Cuda::getCudaStream() );
    void  Invert( const Image & in, Image & out, cudaStream_t stream = Cuda::getCudaStream() );

    Image LookupTable( const Image & in, const std::vector < uint8_t > & table, cudaStream_t stream = Cuda::getCudaStream() );
    void  LookupTable( const Image & in, Image & out, const std::vector < uint8_t > & table, cudaStream_t stream = Cuda::getCudaStream() );

    Image Maximum( const Image & in1, const Image & in2, cudaStream_t stream = Cuda::getCudaStream() );
    void  Maximum( const Image & in1, const Image & in2, Image & out, cudaStream_t stream = Cuda::getCudaStream() );

    Image Minimum( const Image & in1, const Image & in2, cudaStream_t stream = Cuda::getCudaStream() );
    void  Minimum( const Image & in1, const Image & in2, Image & out, cudaStream_t stream = Cuda::getCudaStream() );

    Image Subtract( const Image & in1, const Image & in2, cudaStream_t stream = Cuda::getCudaStream() );
    void  Subtract( const Image & in1, const Image & in2, Image & out, cudaStream_t stream = Cuda::getCudaStream() );

    // Thresholding works in such way:
        // if pixel intensity on input image is          less (  < ) than threshold then set pixel intensity on output image as 0
        // if pixel intensity on input image is equal or more ( >= ) than threshold then set pixel intensity on output image as 255
    Image Threshold( const Image & in, uint8_t threshold, cudaStream_t stream = Cuda::getCudaStream() );
    void  Threshold( const Image & in, Image & out, uint8_t threshold, cudaStream_t stream = Cuda::getCudaStream() );

    // Thresholding works in such way:
        // if pixel intensity on input image is less ( < ) than minimum threshold or more ( > ) than maximum threshold
        // then      set pixel intensity on output image as 0
        // otherwise set pixel intensity on output image as 255
    Image Threshold( const Image & in, uint8_t minThreshold, uint8_t maxThreshold, cudaStream_t stream = Cuda::getCudaStream() );
    void  Threshold( const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold, cudaStream_t stream = Cuda::getCudaStream() );
};
