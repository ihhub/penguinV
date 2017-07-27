#include <cuda_runtime.h>
#include <math.h>
#include "cuda_types.cuh"
#include "image_function_cuda.cuh"

namespace
{
    struct KernelParameters
    {
        KernelParameters( int threadsPerBlock_, int blocksPerGrid_  )
            : threadsPerBlock( threadsPerBlock_ )
            , blocksPerGrid  ( blocksPerGrid_   )
        {};

        int threadsPerBlock;
        int blocksPerGrid;
    };

    // Helper function which should return proper arguments for CUDA device functions
    KernelParameters getKernelParameters( uint32_t size )
    {
        static const int threadsPerBlock = 256;
        return KernelParameters( threadsPerBlock, (size + threadsPerBlock - 1) / threadsPerBlock );
    };

    // Validation of last occured error in functions on host side
    void ValidateLastError()
    {
        cudaError_t error = cudaGetLastError();
        if( error != cudaSuccess )
            throw imageException( "Failed to launch CUDA kernel" );
    };

    // The list of CUDA device functions on device side
    __global__ void absoluteDifferenceCuda( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] > in2[id] ? in1[id] - in2[id] : in2[id] - in1[id];
        }
    };

    __global__ void bitwiseAndCuda( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] & in2[id];
        }
    };

    __global__ void bitwiseOrCuda( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] | in2[id];
        }
    };

    __global__ void bitwiseXorCuda( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] ^ in2[id];
        }
    };

    __global__ void convertToGrayScaleCuda( const uint8_t * in, uint8_t * out, uint32_t size, uint32_t width, uint8_t colorCount )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            uint32_t x = id % width;
            uint32_t y = id / width;

            uint32_t sum = 0;

            const uint8_t * data = in + (width * y + x) * colorCount;

            for( uint8_t i = 0; i < colorCount; ++i, ++data )
            {
                sum += (*data);
            }

            out[id] = static_cast<uint8_t>(sum / colorCount);
        }
    };

    __global__ void fillCuda( uint8_t * data, uint8_t value, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            data[id] = value;
        }
    };

    __global__ void histogramCuda( const uint8_t * data, uint32_t size, uint32_t * histogram )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            atomicAdd( &histogram[data[id]], 1 );
        }
    };

    __global__ void invertCuda( const uint8_t * in, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = ~in[id];
        }
    };

    __global__ void lookupTableCuda( const uint8_t * in, uint8_t * out, uint32_t size, uint8_t * table )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = table[in[id]];
        }
    };

    __global__ void maximumCuda( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] > in2[id] ? in1[id] : in2[id];
        }
    };

    __global__ void minimumCuda( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] < in2[id] ? in1[id] : in2[id];
        }
    };

    __global__ void subtractCuda( const uint8_t * in1, const uint8_t * in2, uint8_t * out, uint32_t size )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in1[id] > in2[id] ? in1[id] - in2[id] : 0;
        }
    };

    __global__ void thresholdCuda( const uint8_t * in, uint8_t * out, uint32_t size, uint8_t threshold )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in[id] < threshold ? 0 : 255;
        }
    };

    __global__ void thresholdCuda( const uint8_t * in, uint8_t * out, uint32_t size, uint8_t minThreshold, uint8_t maxThreshold )
    {
        uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;

        if( id < size ) {
            out[id] = in[id] < minThreshold || in[id] > maxThreshold ? 0 : 255;
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

        const uint32_t size = out.rowSize() * out.height();
        const KernelParameters kernel = getKernelParameters( size );

        absoluteDifferenceCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>( in1.data(), in2.data(), out.data(), size );
        
        ValidateLastError();
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

        const uint32_t size = out.rowSize() * out.height();
        const KernelParameters kernel = getKernelParameters( size );

        bitwiseAndCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>(in1.data(), in2.data(), out.data(), size);
        
        ValidateLastError();
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

        const uint32_t size = out.rowSize() * out.height();
        const KernelParameters kernel = getKernelParameters( size );

        bitwiseOrCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>(in1.data(), in2.data(), out.data(), size);
        
        ValidateLastError();
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

        const uint32_t size = out.rowSize() * out.height();
        const KernelParameters kernel = getKernelParameters( size );

        bitwiseXorCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>(in1.data(), in2.data(), out.data(), size);
        
        ValidateLastError();
    }

    Image ConvertToCuda( const Bitmap_Image::Image & in )
    {
        Image out( in.width(), in.height(), in.colorCount() );

        ConvertToCuda( in, out );

        return out;
    }

    void ConvertToCuda( const Bitmap_Image::Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        if( in.width() != out.width() || in.height() != out.height() ||
            in.colorCount() != out.colorCount())
            throw imageException( "Bad input parameters in image function" );

        if (in.alignment() == 1u || (in.rowSize() == in.width() * in.colorCount()))
        {
            const uint32_t size = in.rowSize() * in.height();

            cudaError_t error = cudaMemcpy( out.data(), in.data(), size, cudaMemcpyHostToDevice );
            if( error != cudaSuccess )
                throw imageException( "Cannot copy a memory to CUDA device" );
        }
        else
        {
            cudaError_t error = cudaMemcpy2D( out.data(), out.rowSize(), in.data(), in.rowSize(), in.colorCount() * in.width(), in.height(), cudaMemcpyHostToDevice);
            if( error != cudaSuccess )
                throw imageException( "Cannot copy a memory to CUDA device" );
        }
    }

    Bitmap_Image::Image ConvertFromCuda( const Image & in )
    {
        Bitmap_Image::Image out( in.width(), in.height(), in.colorCount(), 1u );

        ConvertFromCuda( in, out );

        return out;
    }

    void ConvertFromCuda( const Image & in, Bitmap_Image::Image & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        if( in.width() != out.width() || in.height() != out.height() ||
            in.colorCount() != out.colorCount())
            throw imageException( "Bad input parameters in image function" );

        if (out.alignment() == 1u || (out.rowSize() == out.width() * out.colorCount()))
        {
            const uint32_t size = in.rowSize() * in.height();

            cudaError_t error = cudaMemcpy( out.data(), in.data(), size, cudaMemcpyDeviceToHost );
            if( error != cudaSuccess )
                throw imageException( "Cannot copy a memory from CUDA device" );
        }
        else
        {
            cudaError_t error = cudaMemcpy2D( out.data(), out.rowSize(), in.data(), in.rowSize(), in.colorCount() * in.width(), in.height(), cudaMemcpyDeviceToHost);
            if( error != cudaSuccess )
                throw imageException( "Cannot copy a memory to CUDA device" );
        }
    }

    Image ConvertToGrayScale( const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        ConvertToGrayScale( in, out );

        return out;
    }

    void ConvertToGrayScale( const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );
        Image_Function::VerifyGrayScaleImage( out );

        if( in.colorCount() == GRAY_SCALE ) {
            Copy( in, out );
            return;
        }

        const uint32_t size = out.width() * out.height();
        const KernelParameters kernel = getKernelParameters( size );

        convertToGrayScaleCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>(in.data(), out.data(), size, in.width(), in.colorCount());

        ValidateLastError();
    }

    void Copy( const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        out = in;
    }

    void Fill( Image & image, uint8_t value )
    {
        Image_Function::ParameterValidation( image );

        const uint32_t size = image.rowSize() * image.height();
        const KernelParameters kernel = getKernelParameters( size );

        fillCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>(image.data(), value, size);
        
        ValidateLastError();
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

        // We precalculate all values and store them in lookup table
        std::vector < uint8_t > value( 256 );

        for( uint16_t i = 0; i < 256; ++i ) {
            double data = a * pow( i / 255.0, gamma ) * 255 + 0.5;

            if( data < 256 )
                value[i] = static_cast<uint8_t>(data);
            else
                value[i] = 255;
        }

        LookupTable( in, out, value );
    }

    uint8_t GetThreshold( const std::vector < uint32_t > & histogram )
    {
        if( histogram.size() != 256 )
            throw imageException( "Histogram size is not 256" );

        // It is well-known Otsu's method to find threshold
        uint32_t pixelCount = histogram[0] + histogram[1];
        uint32_t sum = histogram[1];
        for( uint16_t i = 2; i < 256; ++i ) {
            sum = sum + i * histogram[i];
            pixelCount += histogram[i];
        }

        uint32_t sumTemp = 0;
        uint32_t pixelCountTemp = 0;

        double maximumSigma = -1;

        uint8_t threshold = 0;

        for( uint16_t i = 0; i < 256; ++i ) {
            pixelCountTemp += histogram[i];

            if( pixelCountTemp > 0 && pixelCountTemp != pixelCount ) {
                sumTemp += i * histogram[i];

                double w1 = static_cast<double>(pixelCountTemp) / pixelCount;
                double a  = static_cast<double>(sumTemp) / pixelCountTemp -
                    static_cast<double>(sum - sumTemp) / (pixelCount - pixelCountTemp);
                double sigma = w1 * (1 - w1) * a * a;

                if( sigma > maximumSigma ) {
                    maximumSigma = sigma;
                    threshold = static_cast <uint8_t>(i);
                }
            }
        }

        return threshold;
    }

    std::vector < uint32_t > Histogram( const Image & image )
    {
        Image_Function::ParameterValidation( image );

        std::vector < uint32_t > histogram;

        Histogram( image, histogram );

        return histogram;
    }

    void Histogram( const Image & image, std::vector < uint32_t > & histogram )
    {
        Image_Function::ParameterValidation( image );
        Image_Function::VerifyGrayScaleImage( image );

        histogram.resize( 256u );
        std::fill( histogram.begin(), histogram.end(), 0u );

        Cuda_Types::Array< uint32_t > tableCuda( histogram );

        const uint32_t size = image.width() * image.height();
        const KernelParameters kernel = getKernelParameters( size );

        histogramCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>(image.data(), size, &tableCuda);

        ValidateLastError();

        histogram = tableCuda.get();
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

        const uint32_t size = out.rowSize() * out.height();
        const KernelParameters kernel = getKernelParameters( size );

        invertCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>(in.data(), out.data(), size);
        
        ValidateLastError();
    }

    Image LookupTable( const Image & in, const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        LookupTable( in, out, table );

        return out;
    }
    
    void  LookupTable( const Image & in, Image & out, const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in, out );
        Image_Function::VerifyGrayScaleImage( in, out );

        if( table.size() != 256u )
            throw imageException( "Lookup table size is not equal to 256" );

        Cuda_Types::Array< uint8_t > tableCuda( table );

        const uint32_t size = out.rowSize() * out.height();
        const KernelParameters kernel = getKernelParameters( size );

        lookupTableCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>(in.data(), out.data(), size, &tableCuda);

        ValidateLastError();
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

        const uint32_t size = out.rowSize() * out.height();
        const KernelParameters kernel = getKernelParameters( size );

        maximumCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>(in1.data(), in2.data(), out.data(), size);
        
        ValidateLastError();
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

        const uint32_t size = out.rowSize() * out.height();
        const KernelParameters kernel = getKernelParameters( size );

        minimumCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>(in1.data(), in2.data(), out.data(), size);
        
        ValidateLastError();
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

        const uint32_t size = out.rowSize() * out.height();
        const KernelParameters kernel = getKernelParameters( size );

        subtractCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>(in1.data(), in2.data(), out.data(), size);
        
        ValidateLastError();
    }

    Image Threshold( const Image & in, uint8_t threshold )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        Threshold( in, out, threshold );

        return out;
    }

    void Threshold( const Image & in, Image & out, uint8_t threshold )
    {
        Image_Function::ParameterValidation( in, out );
        Image_Function::VerifyGrayScaleImage( in, out );

        const uint32_t size = out.width() * out.height();
        const KernelParameters kernel = getKernelParameters( size );

        thresholdCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>(in.data(), out.data(), size, threshold);
        
        ValidateLastError();
    }

    Image Threshold( const Image & in, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        Threshold( in, out, minThreshold, maxThreshold );

        return out;
    }

    void Threshold( const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function::ParameterValidation( in, out );
        Image_Function::VerifyGrayScaleImage( in, out );

        const uint32_t size = out.width() * out.height();
        const KernelParameters kernel = getKernelParameters( size );

        thresholdCuda<<<kernel.blocksPerGrid, kernel.threadsPerBlock>>>(in.data(), out.data(), size, minThreshold, maxThreshold);
        
        ValidateLastError();
    }
};
