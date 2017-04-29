#include <cuda_runtime.h>
#include <math.h>
#include "../image_function.h"
#include "image_function_cuda.cuh"

namespace
{
    // Helper function which should return proper arguments for CUDA device functions
    void getKernelParameters( int & threadsPerBlock, int & blocksPerGrid, uint32_t size )
    {
        if( size < 256 ) {
            threadsPerBlock = size;
            blocksPerGrid = 1;
        }
        else {
            threadsPerBlock = 256;
            blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        }
    };

    // The list of CUDA device functions
    __global__ void absoluteDifference( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] > in2[id] ? in1[id] - in2[id] : in2[id] - in1[id];
        }
    };

    __global__ void bitwiseAnd( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] & in2[id];
        }
    };

    __global__ void bitwiseOr( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] | in2[id];
        }
    };

    __global__ void bitwiseXor( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] ^ in2[id];
        }
    };

    __global__ void fill( uint8_t * data, uint8_t value, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            data[id] = value;
        }
    };

    __global__ void gammaCorrection( const uint8_t * in, uint8_t * out, uint32_t size, double a, float gamma )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ uint8_t value[256];

        if( threadIdx.x == 0 ) {
            for( uint16_t i = 0; i < 256; ++i ) {
                double data = a * __powf( __fdividef( (float)i, 255.0f ), gamma ) * 255 + 0.5;

                if( data < 255 )
                    value[i] = static_cast<uint8_t>(data);
                else
                    value[i] = 255;
            }
        }

        __syncthreads();

        if( id < size ) {
            out[id] = value[in[id]];
        }
    };

    __global__ void invert( const uint8_t * in, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = ~in[id];
        }
    };

    __global__ void maximum( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] > in2[id] ? in1[id] : in2[id];
        }
    };

    __global__ void minimum( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] < in2[id] ? in1[id] : in2[id];
        }
    };

    __global__ void subtract( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] > in2[id] ? in1[id] - in2[id] : 0;
        }
    };
};

namespace Image_Function_Cuda
{
    Image AbsoluteDifference( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        AbsoluteDifference( in1, in2, out );

        return out;
    }

    void AbsoluteDifference( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        int threadsPerBlock = 1, blocksPerGrid = 1;
        getKernelParameters( threadsPerBlock, blocksPerGrid, out.width() * out.height() );

        absoluteDifference<<<blocksPerGrid, threadsPerBlock>>>(in1.data(), in2.data(), out.data(), out.width() * out.height());
        cudaError_t error = cudaGetLastError();
        if( error != cudaSuccess )
            throw imageException( "Failed to launch CUDA kernel" );
    }

    Image BitwiseAnd( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        BitwiseAnd( in1, in2, out );

        return out;
    }

    void BitwiseAnd( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        int threadsPerBlock = 1, blocksPerGrid = 1;
        getKernelParameters( threadsPerBlock, blocksPerGrid, out.width() * out.height() );

        bitwiseAnd<<<blocksPerGrid, threadsPerBlock>>>(in1.data(), in2.data(), out.data(), out.width() * out.height());
        cudaError_t error = cudaGetLastError();
        if( error != cudaSuccess )
            throw imageException( "Failed to launch CUDA kernel" );
    }

    Image BitwiseOr( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        BitwiseOr( in1, in2, out );

        return out;
    }

    void BitwiseOr( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        int threadsPerBlock = 1, blocksPerGrid = 1;
        getKernelParameters( threadsPerBlock, blocksPerGrid, out.width() * out.height() );

        bitwiseOr<<<blocksPerGrid, threadsPerBlock>>>(in1.data(), in2.data(), out.data(), out.width() * out.height());
        cudaError_t error = cudaGetLastError();
        if( error != cudaSuccess )
            throw imageException( "Failed to launch CUDA kernel" );
    }

    Image BitwiseXor( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        BitwiseXor( in1, in2, out );

        return out;
    }

    void BitwiseXor( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        int threadsPerBlock = 1, blocksPerGrid = 1;
        getKernelParameters( threadsPerBlock, blocksPerGrid, out.width() * out.height() );

        bitwiseXor<<<blocksPerGrid, threadsPerBlock>>>(in1.data(), in2.data(), out.data(), out.width() * out.height());
        cudaError_t error = cudaGetLastError();
        if( error != cudaSuccess )
            throw imageException( "Failed to launch CUDA kernel" );
    }

    void Convert( const Bitmap_Image::Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        if( in.width() != out.width() || in.height() != out.height() )
            throw imageException( "Bad input parameters in image function" );

        uint32_t rowSizeIn  = in.rowSize();
        uint32_t rowSizeOut = out.width();

        const uint8_t * Y    = in.data();
        const uint8_t * YEnd = Y + in.height() * rowSizeIn;

        uint8_t * cudaY = out.data();

        for( ; Y != YEnd; Y += rowSizeIn, cudaY += rowSizeOut ) {
            cudaError_t error = cudaMemcpy( cudaY, Y, out.width() * sizeof( uint8_t ), cudaMemcpyHostToDevice );
            if( error != cudaSuccess )
                throw imageException( "Cannot copy a memory to CUDA device" );
        }
    }

    void Convert( const Image & in, Bitmap_Image::Image & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        if( in.width() != out.width() || in.height() != out.height() )
            throw imageException( "Bad input parameters in image function" );

        uint32_t rowSizeIn  = in.width();
        uint32_t rowSizeOut = out.rowSize();

        uint8_t * Y    = out.data();
        const uint8_t * YEnd = Y + out.height() * rowSizeOut;

        const uint8_t * cudaY = in.data();

        for( ; Y != YEnd; Y += rowSizeOut, cudaY += rowSizeIn ) {
            cudaError_t error = cudaMemcpy( Y, cudaY, in.width() * sizeof( uint8_t ), cudaMemcpyDeviceToHost );
            if( error != cudaSuccess )
                throw imageException( "Cannot copy a memory from CUDA device" );
        }
    }

    void Copy( const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        out = in;
    }

    void Fill( Image & image, uint8_t value )
    {
        Image_Function::ParameterValidation( image );

        int threadsPerBlock = 1, blocksPerGrid = 1;
        getKernelParameters( threadsPerBlock, blocksPerGrid, image.width() * image.height() );

        fill<<<blocksPerGrid, threadsPerBlock>>>(image.data(), value, image.width() * image.height());
        cudaError_t error = cudaGetLastError();
        if( error != cudaSuccess )
            throw imageException( "Failed to launch CUDA kernel" );
    }

    Image GammaCorrection( const Image & in, double a, double gamma )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        GammaCorrection( in, out, a, gamma );

        return out;
    }

    void GammaCorrection( const Image & in, Image & out, double a, double gamma )
    {
        Image_Function::ParameterValidation( in, out );

        if( a < 0 || gamma < 0 )
            throw imageException( "Bad input parameters in image function" );

        int threadsPerBlock = 1, blocksPerGrid = 1;
        getKernelParameters( threadsPerBlock, blocksPerGrid, out.width() * out.height() );

        gammaCorrection<<<blocksPerGrid, threadsPerBlock>>>(in.data(), out.data(), out.width() * out.height(), a, static_cast<float>(gamma));
        cudaError_t error = cudaGetLastError();
        if( error != cudaSuccess )
            throw imageException( "Failed to launch CUDA kernel" );
    }

    Image Invert( const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        Invert( in, out );

        return out;
    }

    void Invert( const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        int threadsPerBlock = 1, blocksPerGrid = 1;
        getKernelParameters( threadsPerBlock, blocksPerGrid, out.width() * out.height() );

        invert<<<blocksPerGrid, threadsPerBlock>>>(in.data(), out.data(), out.width() * out.height());
        cudaError_t error = cudaGetLastError();
        if( error != cudaSuccess )
            throw imageException( "Failed to launch CUDA kernel" );
    }

    Image Maximum( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        Maximum( in1, in2, out );

        return out;
    }

    void Maximum( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        int threadsPerBlock = 1, blocksPerGrid = 1;
        getKernelParameters( threadsPerBlock, blocksPerGrid, out.width() * out.height() );

        maximum<<<blocksPerGrid, threadsPerBlock>>>(in1.data(), in2.data(), out.data(), out.width() * out.height());
        cudaError_t error = cudaGetLastError();
        if( error != cudaSuccess )
            throw imageException( "Failed to launch CUDA kernel" );
    }

    Image Minimum( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        Minimum( in1, in2, out );

        return out;
    }

    void Minimum( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        int threadsPerBlock = 1, blocksPerGrid = 1;
        getKernelParameters( threadsPerBlock, blocksPerGrid, out.width() * out.height() );

        minimum<<<blocksPerGrid, threadsPerBlock>>>(in1.data(), in2.data(), out.data(), out.width() * out.height());
        cudaError_t error = cudaGetLastError();
        if( error != cudaSuccess )
            throw imageException( "Failed to launch CUDA kernel" );
    }

    Image Subtract( const Image & in1, const Image & in2 )
    {
        Image_Function::ParameterValidation( in1, in2 );

        Image out( in1.width(), in1.height() );

        Subtract( in1, in2, out );

        return out;
    }

    void Subtract( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function::ParameterValidation( in1, in2, out );

        int threadsPerBlock = 1, blocksPerGrid = 1;
        getKernelParameters( threadsPerBlock, blocksPerGrid, out.width() * out.height() );

        subtract<<<blocksPerGrid, threadsPerBlock>>>(in1.data(), in2.data(), out.data(), out.width() * out.height());
        cudaError_t error = cudaGetLastError();
        if( error != cudaSuccess )
            throw imageException( "Failed to launch CUDA kernel" );
    }
};
