#include <cuda_runtime.h>
#include <math.h>
#include "image_function_cuda.cuh"
#include "../parameter_validation.h"
#include "../thirdparty/multicuda/src/cuda_types.cuh"
#include "../thirdparty/multicuda/src/cuda_helper.cuh"

namespace
{
    // The list of CUDA device functions on device side
    __global__ void absoluteDifferenceCuda( const uint8_t * in1, uint32_t rowSizeIn1, const uint8_t * in2, uint32_t rowSizeIn2,
                                            uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            const uint8_t * in1X = in1 + y * rowSizeIn1 + x;
            const uint8_t * in2X = in2 + y * rowSizeIn2 + x;
            uint8_t * outX = out + y * rowSizeOut + x;
            (*outX) = ((*in1X) > ( *in2X )) ? ((*in1X) - (*in2X)) : ((*in2X) - (*in1X));
        }
    }

    __global__ void bitwiseAndCuda( const uint8_t * in1, uint32_t rowSizeIn1, const uint8_t * in2, uint32_t rowSizeIn2,
                                    uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            const uint32_t idIn1 = y * rowSizeIn1 + x;
            const uint32_t idIn2 = y * rowSizeIn2 + x;
            const uint32_t idOut = y * rowSizeOut + x;
            out[idOut] = in1[idIn1] & in2[idIn2];
        }
    }

    __global__ void bitwiseOrCuda( const uint8_t * in1, uint32_t rowSizeIn1, const uint8_t * in2, uint32_t rowSizeIn2,
                                   uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            const uint32_t idIn1 = y * rowSizeIn1 + x;
            const uint32_t idIn2 = y * rowSizeIn2 + x;
            const uint32_t idOut = y * rowSizeOut + x;
            out[idOut] = in1[idIn1] | in2[idIn2];
        }
    }

    __global__ void bitwiseXorCuda( const uint8_t * in1, uint32_t rowSizeIn1, const uint8_t * in2, uint32_t rowSizeIn2,
                                    uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            const uint32_t idIn1 = y * rowSizeIn1 + x;
            const uint32_t idIn2 = y * rowSizeIn2 + x;
            const uint32_t idOut = y * rowSizeOut + x;
            out[idOut] = in1[idIn1] ^ in2[idIn2];
        }
    }

    __global__ void convertToGrayScaleCuda( const uint8_t * in, uint32_t rowSizeIn, uint8_t colorCount, uint8_t * out, uint32_t rowSizeOut,
                                            uint32_t width, uint32_t height )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            const uint8_t * data = in + y * rowSizeIn + x * colorCount;
            const uint8_t * dataEnd = data + colorCount;

            uint32_t sum = 0;
            for ( ; data != dataEnd; ++data )
            {
                sum += (*data);
            }

            const uint32_t id = y * rowSizeOut + x;
            out[id] = static_cast<uint8_t>(sum / colorCount);
        }
    }

    __global__ void convertToRgbCuda( const uint8_t * in, uint32_t rowSizeIn, uint8_t * out, uint32_t rowSizeOut, uint8_t colorCount,
                                      uint32_t width, uint32_t height )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            const uint8_t * dataIn = in + y * rowSizeIn + x;

            uint8_t * dataOut = out + (width * y + x) * colorCount;
            const uint8_t * dataOutEnd = dataOut + colorCount;

            for ( ; dataOut != dataOutEnd; ++dataOut )
            {
                (*dataOut) = (*dataIn);
            }
        }
    }

    __global__ void copyCuda( const uint8_t * in, uint32_t rowSizeIn, uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            out[y * rowSizeOut + x] = in[y * rowSizeIn + x];
        }
    }

    __global__ void extractChannelCuda( const uint8_t * in, uint32_t rowSizeIn, uint8_t colorCount, uint8_t * out, uint32_t rowSizeOut,
                                        uint32_t width, uint32_t height, uint8_t channelId )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height )
            out[y * rowSizeOut + x] = in[y * rowSizeIn + x * colorCount + channelId];
    }

    __global__ void fillCuda( uint8_t * data, uint32_t rowSize, uint32_t width, uint32_t height, uint8_t value )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height )
            data[y * rowSize + x] = value;
    }

    __global__ void flipCuda( const uint8_t * in, uint32_t rowSizeIn, uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height,
                              bool horizontal, bool vertical )
    {
        const uint32_t inX = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t inY = blockDim.y * blockIdx.y + threadIdx.y;

        if ( inX < width && inY < height ) {
            const uint32_t outX = horizontal ? (width  - 1 - inX) : inX;
            const uint32_t outY = vertical   ? (height - 1 - inY) : inY;

            out[outY * rowSizeOut + outX] = in[inY * rowSizeIn + inX];
        }
    }

    __global__ void histogramCuda( const uint8_t * data, uint32_t rowSize, uint32_t width, uint32_t height, uint32_t * histogram )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            const uint32_t id = y * rowSize + x;
            atomicAdd( &histogram[data[id]], 1 );
        }
    }

    __global__ void invertCuda( const uint8_t * in, uint32_t rowSizeIn, uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            out[y * rowSizeOut + x] = ~in[y * rowSizeIn + x];
        }
    }

    __global__ void lookupTableCuda( const uint8_t * in, uint32_t rowSizeIn, uint8_t * out, uint32_t rowSizeOut,
                                     uint32_t width, uint32_t height, uint8_t * table )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            out[y * rowSizeOut + x] = table[in[y * rowSizeIn + x]];
        }
    }

    __global__ void maximumCuda( const uint8_t * in1, uint32_t rowSizeIn1, const uint8_t * in2, uint32_t rowSizeIn2,
                                 uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            const uint8_t * in1X = in1 + y * rowSizeIn1 + x;
            const uint8_t * in2X = in2 + y * rowSizeIn2 + x;
            uint8_t * outX = out + y * rowSizeOut + x;
            (*outX) = ((*in1X) > ( *in2X )) ? (*in1X) : (*in2X);
        }
    }

    __global__ void minimumCuda( const uint8_t * in1, uint32_t rowSizeIn1, const uint8_t * in2, uint32_t rowSizeIn2,
                                 uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            const uint8_t * in1X = in1 + y * rowSizeIn1 + x;
            const uint8_t * in2X = in2 + y * rowSizeIn2 + x;
            uint8_t * outX = out + y * rowSizeOut + x;
            (*outX) = ((*in1X) < (*in2X)) ? (*in1X) : (*in2X);
        }
    }

    __global__ void subtractCuda( const uint8_t * in1, uint32_t rowSizeIn1, const uint8_t * in2, uint32_t rowSizeIn2,
                                  uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            const uint8_t * in1X = in1 + y * rowSizeIn1 + x;
            const uint8_t * in2X = in2 + y * rowSizeIn2 + x;
            uint8_t * outX = out + y * rowSizeOut + x;
            (*outX) = ((*in1X) > ( *in2X )) ? ((*in1X) - (*in2X)) : 0;
        }
    }

    __global__ void thresholdCuda( const uint8_t * in, uint32_t rowSizeIn, uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height,
                                   uint8_t threshold )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            out[y * rowSizeOut + x] = (in[y * rowSizeIn + x] < threshold) ? 0 : 255;
        }
    }

    __global__ void thresholdCuda( const uint8_t * in, uint32_t rowSizeIn, uint8_t * out, uint32_t rowSizeOut, uint32_t width, uint32_t height,
                                   uint8_t minThreshold, uint8_t maxThreshold )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if ( x < width && y < height ) {
            const uint32_t idIn = y * rowSizeIn + x;
            out[y * rowSizeOut + x] = ((in[idIn] < minThreshold) || (in[idIn] > maxThreshold)) ? 0 : 255;
        }
    }
}

namespace Image_Function_Cuda
{
    Image AbsoluteDifference( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        AbsoluteDifference( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void AbsoluteDifference( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        AbsoluteDifference( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                              uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        AbsoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                             Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        launchKernel2D( absoluteDifferenceCuda, width, height,
                        in1Y, rowSizeIn1, in2Y, rowSizeIn2, outY, rowSizeOut, width, height );
    }

    Image BitwiseAnd( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        BitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseAnd( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        BitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        launchKernel2D( bitwiseAndCuda, width, height,
                        in1Y, rowSizeIn1, in2Y, rowSizeIn2, outY, rowSizeOut, width, height );
    }

    Image BitwiseOr( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        BitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseOr( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        BitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        launchKernel2D( bitwiseOrCuda, width, height,
                        in1Y, rowSizeIn1, in2Y, rowSizeIn2, outY, rowSizeOut, width, height );
    }

    Image BitwiseXor( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        BitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseXor( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        BitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        launchKernel2D( bitwiseXorCuda, width, height,
                        in1Y, rowSizeIn1, in2Y, rowSizeIn2, outY, rowSizeOut, width, height );
    }

    Image ConvertToCuda( const PenguinV_Image::Image & in )
    {
        Image out( in.width(), in.height(), in.colorCount() );

        ConvertToCuda( in, out );

        return out;
    }

    void ConvertToCuda( const PenguinV_Image::Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        if ( in.width() != out.width() || in.height() != out.height() ||
             in.colorCount() != out.colorCount() )
            throw imageException( "Bad input parameters in image function" );

        if ( in.alignment() == 1u || (in.rowSize() == in.width() * in.colorCount()) )
        {
            const uint32_t size = in.rowSize() * in.height();

            if ( !multiCuda::cudaSafeCheck( cudaMemcpy( out.data(), in.data(), size * sizeof( uint8_t ), cudaMemcpyHostToDevice ) ) )
                throw imageException( "Cannot copy a memory to CUDA device" );
        }
        else
        {
            if ( !multiCuda::cudaSafeCheck( cudaMemcpy2D( out.data(), out.rowSize(), in.data(), in.rowSize(),
                                                          in.colorCount() * in.width(), in.height(), cudaMemcpyHostToDevice ) ) )
                throw imageException( "Cannot copy a memory to CUDA device" );
        }
    }

    PenguinV_Image::Image ConvertFromCuda( const Image & in )
    {
        PenguinV_Image::Image out( in.width(), in.height(), in.colorCount(), 1u );

        ConvertFromCuda( in, out );

        return out;
    }

    void ConvertFromCuda( const Image & in, PenguinV_Image::Image & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        if ( in.width() != out.width() || in.height() != out.height() ||
             in.colorCount() != out.colorCount() )
            throw imageException( "Bad input parameters in image function" );

        if ( out.alignment() == 1u || (out.rowSize() == out.width() * out.colorCount()) )
        {
            const uint32_t size = in.rowSize() * in.height();

            if ( !multiCuda::cudaSafeCheck( cudaMemcpy( out.data(), in.data(), size, cudaMemcpyDeviceToHost ) ) )
                throw imageException( "Cannot copy a memory from CUDA device" );
        }
        else
        {
            if ( !multiCuda::cudaSafeCheck( cudaMemcpy2D( out.data(), out.rowSize(), in.data(), in.rowSize(),
                                                          in.colorCount() * in.width(), in.height(), cudaMemcpyDeviceToHost ) ) )
                throw imageException( "Cannot copy a memory to CUDA device" );
        }
    }

    Image ConvertToGrayScale( const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        ConvertToGrayScale( in, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void ConvertToGrayScale( const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        ConvertToGrayScale( in, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height );

        ConvertToGrayScale( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                             uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( out );

        if ( in.colorCount() == GRAY_SCALE ) {
            Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            return;
        }

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t colorCount = in.colorCount();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        launchKernel2D( convertToGrayScaleCuda, width, height,
                        inY, rowSizeIn, colorCount, outY, rowSizeOut, width, height );
    }

    Image ConvertToRgb( const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height(), RGB );

        ConvertToRgb( in, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void ConvertToRgb( const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        ConvertToRgb( in, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height, RGB );

        ConvertToRgb( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyColoredImage  ( out );

        if ( in.colorCount() == RGB ) {
            Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            return;
        }

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t colorCount = out.colorCount();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        launchKernel2D( convertToRgbCuda, width, height,
                        inY, rowSizeIn, outY, rowSizeOut, colorCount, width, height );
    }

    void Copy( const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        out = in;
    }

    Image Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height );

        Copy( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    void Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
               uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

        const uint8_t colorCount  = Image_Function::CommonColorCount( in, out );
        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        width = width * colorCount;

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        launchKernel2D( copyCuda, width, height,
                        inY, rowSizeIn, outY, rowSizeOut, width, height );
    }

    Image ExtractChannel( const Image & in, uint8_t channelId )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        ExtractChannel( in, 0, 0, out, 0, 0, in.width(), in.height(), channelId );

        return out;
    }

    void ExtractChannel( const Image & in, Image & out, uint8_t channelId )
    {
        Image_Function::ParameterValidation( in, out );

        ExtractChannel( in, 0, 0, out, 0, 0, in.width(), in.height(), channelId );
    }

    Image ExtractChannel( const Image & in, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t channelId )
    {
        Image_Function::ParameterValidation( in, x, y, width, height );

        Image out( width, height );

        ExtractChannel( in, x, y, out, 0, 0, width, height, channelId );

        return out;
    }

    void ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
                         uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( out );

        if ( channelId >= in.colorCount() )
            throw imageException( "Channel ID for color image is greater than channel count in input image" );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t colorCount = in.colorCount();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn * colorCount + channelId;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        launchKernel2D( extractChannelCuda, width, height,
                        inY, rowSizeIn, colorCount, outY, rowSizeOut, width, height, channelId );
    }

    void Fill( Image & image, uint8_t value )
    {
        image.fill( value );
    }

    void Fill( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value )
    {
        Image_Function::ParameterValidation( image, x, y, width, height );
        Image_Function::VerifyGrayScaleImage( image );

        const uint32_t rowSize = image.rowSize();

        uint8_t * imageY = image.data() + y * rowSize + x;

        launchKernel2D( fillCuda, width, height,
                        imageY, rowSize, width, height, value );
    }

    Image Flip( const Image & in, bool horizontal, bool vertical )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        Flip( in, out, horizontal, vertical );

        return out;
    }

    void  Flip( const Image & in, Image & out, bool horizontal, bool vertical )
    {
        Image_Function::ParameterValidation( in, out );
        Image_Function::VerifyGrayScaleImage( in, out );

        if ( !horizontal && !vertical ) {
            Copy( in, out );
        }
        else {
            launchKernel2D( flipCuda, out.width(), out.height(),
                            in.data(), in.rowSize(), out.data(), out.rowSize(), out.width(), out.height(), horizontal, vertical );
        }
    }

    Image GammaCorrection( const Image & in, double a, double gamma )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        GammaCorrection( in, 0, 0, out, 0, 0, out.width(), out.height(), a, gamma );

        return out;
    }

    void GammaCorrection( const Image & in, Image & out, double a, double gamma )
    {
        Image_Function::ParameterValidation( in, out );

        GammaCorrection( in, 0, 0, out, 0, 0, out.width(), out.height(), a, gamma );
    }

    Image GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double a, double gamma )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height );

        GammaCorrection( in, startXIn, startYIn, out, 0, 0, width, height, a, gamma );

        return out;
    }

    void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                          uint32_t width, uint32_t height, double a, double gamma )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in, out );

        if ( a < 0 || gamma < 0 )
            throw imageException( "Gamma correction parameters are invalid" );

        // We precalculate all values and store them in lookup table
        std::vector < uint8_t > value( 256, 255u );

        for ( uint16_t i = 0; i < 256; ++i ) {
            double data = a * pow( i / 255.0, gamma ) * 255 + 0.5;

            if ( data < 256 )
                value[i] = static_cast<uint8_t>(data);
        }

        LookupTable( in, startXIn, startYIn, out, startXOut, startYOut, width, height, value );
    }

    uint8_t GetThreshold( const std::vector < uint32_t > & histogram )
    {
        if ( histogram.size() != 256 )
            throw imageException( "Histogram size is not 256" );

        // It is well-known Otsu's method to find threshold
        uint32_t pixelCount = histogram[0] + histogram[1];
        uint32_t sum = histogram[1];
        for ( uint16_t i = 2; i < 256; ++i ) {
            sum = sum + i * histogram[i];
            pixelCount += histogram[i];
        }

        uint32_t sumTemp = 0;
        uint32_t pixelCountTemp = 0;

        double maximumSigma = -1;

        uint8_t threshold = 0;

        for ( uint16_t i = 0; i < 256; ++i ) {
            pixelCountTemp += histogram[i];

            if ( pixelCountTemp > 0 && pixelCountTemp != pixelCount ) {
                sumTemp += i * histogram[i];

                double w1 = static_cast<double>(pixelCountTemp) / pixelCount;
                double a  = static_cast<double>(sumTemp) / pixelCountTemp -
                    static_cast<double>(sum - sumTemp) / (pixelCount - pixelCountTemp);
                double sigma = w1 * (1 - w1) * a * a;

                if ( sigma > maximumSigma ) {
                    maximumSigma = sigma;
                    threshold = static_cast <uint8_t>(i);
                }
            }
        }

        return threshold;
    }

    std::vector < uint32_t > Histogram( const Image & image )
    {
        std::vector < uint32_t > histogram;

        Histogram( image, 0, 0, image.width(), image.height(), histogram );

        return histogram;
    }

    void Histogram( const Image & image, std::vector < uint32_t > & histogram )
    {
        Histogram( image, 0, 0, image.width(), image.height(), histogram );
    }

    std::vector < uint32_t > Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( image, x, y, width, height );

        std::vector < uint32_t > histogram;

        Histogram( image, x, y, width, height, histogram );

        return histogram;
    }

    void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & histogram )
    {
        Image_Function::ParameterValidation( image, x, y, width, height );
        Image_Function::VerifyGrayScaleImage( image );

        histogram.resize( 256u );
        std::fill( histogram.begin(), histogram.end(), 0u );

        const uint32_t rowSize = image.rowSize();

        const uint8_t * imageY = image.data() + y * rowSize + x;

        multiCuda::Array< uint32_t > tableCuda( histogram );

        launchKernel2D( histogramCuda, width, height,
                        imageY, rowSize, width, height, tableCuda.data() );

        histogram = tableCuda.get();
    }

    Image Invert( const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        Invert( in, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Invert( const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        Invert( in, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height );

        Invert( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                 uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

        const uint8_t colorCount = Image_Function::CommonColorCount( in, out );
        width = width * colorCount;

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        launchKernel2D( invertCuda, width, height,
                        inY, rowSizeIn, outY, rowSizeOut, width, height );
    }

    Image LookupTable( const Image & in, const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        LookupTable( in, 0, 0, out, 0, 0, out.width(), out.height(), table );

        return out;
    }

    void LookupTable( const Image & in, Image & out, const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in, out );

        LookupTable( in, 0, 0, out, 0, 0, out.width(), out.height(), table );
    }

    Image LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                       const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height );

        LookupTable( in, startXIn, startYIn, out, 0, 0, width, height, table );

        return out;
    }

    void LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                      uint32_t width, uint32_t height, const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in, out );

        if ( table.size() != 256u )
            throw imageException( "Lookup table size is not equal to 256" );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        multiCuda::Array< uint8_t > tableCuda( table );

        launchKernel2D( lookupTableCuda, width, height,
                        inY, rowSizeIn, outY, rowSizeOut, width, height, tableCuda.data() );
    }

    Image Maximum( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        Maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Maximum( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        Maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        Maximum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        launchKernel2D( maximumCuda, width, height,
                        in1Y, rowSizeIn1, in2Y, rowSizeIn2, outY, rowSizeOut, width, height );
    }

    Image Minimum( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        Minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Minimum( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        Minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        Minimum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        launchKernel2D( minimumCuda, width, height,
                        in1Y, rowSizeIn1, in2Y, rowSizeIn2, outY, rowSizeOut, width, height );
    }

    Image Subtract( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        Subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Subtract( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        Subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        Image out( width, height );

        Subtract( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        width = width * colorCount;

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        launchKernel2D( subtractCuda, width, height,
                        in1Y, rowSizeIn1, in2Y, rowSizeIn2, outY, rowSizeOut, width, height );
    }

    Image Threshold( const Image & in, uint8_t threshold )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), threshold );

        return out;
    }

    void Threshold( const Image & in, Image & out, uint8_t threshold )
    {
        Image_Function::ParameterValidation( in, out );

        Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), threshold );
    }

    Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t threshold )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height );

        Threshold( in, startXIn, startYIn, out, 0, 0, width, height, threshold );

        return out;
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t threshold )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        launchKernel2D( thresholdCuda, width, height,
                        inY, rowSizeIn, outY, rowSizeOut, width, height, threshold );
    }

    Image Threshold( const Image & in, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), minThreshold, maxThreshold );

        return out;
    }

    void Threshold( const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in, out );

        Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), minThreshold, maxThreshold );
    }

    Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t minThreshold,
                     uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height );

        Threshold( in, startXIn, startYIn, out, 0, 0, width, height, minThreshold, maxThreshold );

        return out;
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        launchKernel2D( thresholdCuda, width, height,
                        inY, rowSizeIn, outY, rowSizeOut, width, height, minThreshold, maxThreshold );
    }
}
