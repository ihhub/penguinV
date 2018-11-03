#include <cuda_runtime.h>
#include "cuda_fft.cuh"
#include "cuda_helper.cuh"
#include "../image_exception.h"

namespace
{
    __global__ void copyFromImageCuda( const uint8_t * in, cufftComplex * out, uint32_t width, uint32_t height )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if( x < width && y < height ) {
            const uint32_t id = y * width + x;
            out[id].x = in[id];
            out[id].y = 0;
        }
    }

    __global__ void copyFromFloatCuda( const float * in, cufftComplex * out, uint32_t width, uint32_t height )
    {
        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if( x < width && y < height ) {
            const uint32_t id = y * width + x;
            out[id].x = in[id];
            out[id].y = 0;
        }
    }

    __global__ void copyToImageCuda( const cufftComplex * in, uint8_t * out, float size, uint32_t width, uint32_t height )
    {
        const uint32_t inX = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t inY = blockDim.y * blockIdx.y + threadIdx.y;

        if( inX < width && inY < height ) {
            const uint32_t id = inY * width + inX;

            const uint32_t middleX = width / 2;
            const uint32_t middleY = height / 2;

            const uint32_t outX = (inX < middleX) ? middleX + inX : inX - middleX;
            const uint32_t outY = (inY < middleY) ? middleY + inY : inY - middleY;

            out[outY * width + outX] = static_cast<uint8_t>(in[id].x / size + 0.5f);
        }
    }

    __global__ void complexMultiplicationCuda( const cufftComplex * in1, const cufftComplex * in2, cufftComplex * out, uint32_t width, uint32_t height )
    {
        // in1 = A + iB
        // in2 = C + iD
        // out = in1 * (-in2) = (A + iB) * (-C - iD) = - A * C - i(B * C) - i(A * D) + B * D

        const uint32_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const uint32_t y = blockDim.y * blockIdx.y + threadIdx.y;

        if( x < width && y < height ) {
            const uint32_t id = y * width + x;
            out[id].x = in1[id].x * in2[id].x - in1[id].y * in2[id].y;
            out[id].y = in1[id].x * in2[id].y + in1[id].y * in2[id].x;
        }
    }
}

namespace FFT_Cuda
{
    ComplexData::ComplexData()
        : _data  ( NULL )
        , _width ( 0 )
        , _height( 0 )
    {
    }

    ComplexData::ComplexData( const PenguinV_Image::Image & image )
        : _data  ( NULL )
        , _width ( 0 )
        , _height( 0 )
    {
        set( image );
    }

    ComplexData::ComplexData( const ComplexData & data )
        : _data  ( NULL )
        , _width ( 0 )
        , _height( 0 )
    {
        _copy( data );
    }

    ComplexData::ComplexData( ComplexData && data )
        : _data  ( NULL )
        , _width ( 0 )
        , _height( 0 )
    {
        _swap( data );
    }

    ComplexData & ComplexData::operator=( const ComplexData & data )
    {
        _copy( data );

        return *this;
    }

    ComplexData & ComplexData::operator=( ComplexData && data )
    {
        _swap( data );

        return *this;
    }

    ComplexData::~ComplexData()
    {
        _clean();
    }

    void ComplexData::set( const PenguinV_Image::Image & image )
    {
        if( image.empty() || image.colorCount() != 1u )
            throw imageException( "Failed to allocate complex data for empty or coloured image" );

        _clean();

        multiCuda::cudaCheck( cudaMalloc( &_data, (image.width() * image.height()) * sizeof( cufftComplex ) ) );

        _width  = image.width();
        _height = image.height();

        launchKernel2D( copyFromImageCuda, _width, _height,
                        image.data(), _data, _width, _height );
    }

    void ComplexData::set( const multiCuda::Array<float> & data )
    {
        if( data.empty() || _width == 0 || _height == 0 || data.size() != _width * _height )
            throw imageException( "Failed to allocate complex data for empty or coloured image" );

        launchKernel2D( copyFromFloatCuda, _width, _height,
                        data.data(), _data, _width, _height );
    }

    PenguinV_Image::Image ComplexData::get() const
    {
        if( empty() )
            return PenguinV_Image::Image();

        PenguinV_Image::ImageCuda image( _width, _height );

        const float size = static_cast<float>(image.width() * image.height());

        launchKernel2D( copyToImageCuda, _width, _height,
                        _data, image.data(), size, _width, _height );

        return image;
    }

    void ComplexData::resize( uint32_t width_, uint32_t height_ )
    {
        if( (width_ != _width || height_ != _height) && width_ != 0 && height_ != 0 ) {
            _clean();

            multiCuda::cudaCheck( cudaMalloc( &_data, (width_ * height_) * sizeof( cufftComplex ) ) );

            _width  = width_;
            _height = height_;
        }
    }

    cufftComplex * ComplexData::data()
    {
        return _data;
    }

    const cufftComplex * ComplexData::data() const
    {
        return _data;
    }

    uint32_t ComplexData::width() const
    {
        return _width;
    }

    uint32_t ComplexData::height() const
    {
        return _height;
    }

    bool ComplexData::empty() const
    {
        return _data == NULL;
    }

    void ComplexData::_clean()
    {
        if( _data != NULL ) {
            cudaFree( _data );
            _data = NULL;
        }

        _width  = 0;
        _height = 0;
    }

    void ComplexData::_copy( const ComplexData & data )
    {
        _clean();

        resize( data._width, data._height );

        if( !empty() ) {
            if( !multiCuda::cudaSafeCheck( cudaMemcpy( _data, data._data, _width * _height * sizeof( cufftComplex ), cudaMemcpyDeviceToDevice ) ) )
                throw imageException( "Cannot copy a memory to CUDA device" );
        }
    }

    void ComplexData::_swap( ComplexData & data )
    {
        std::swap( _data, data._data );
        std::swap( _width, data._width );
        std::swap( _height, data._height );
    }

    FFTExecutor::FFTExecutor()
        : _plan  ( 0 )
        , _width ( 0 )
        , _height( 0 )
    {
    }

    FFTExecutor::FFTExecutor( uint32_t width_, uint32_t height_ )
        : _plan  ( 0 )
        , _width ( 0 )
        , _height( 0 )
    {
        initialize( width_, height_ );
    }

    FFTExecutor::~FFTExecutor()
    {
        _clean();
    }

    void FFTExecutor::initialize( uint32_t width_, uint32_t height_ )
    {
        if( width_ == 0 || height_ == 0 )
            throw imageException( "Invalid parameters for FFTExecutor" );

        _clean();

        if( cufftPlan2d( &_plan, width_, height_, CUFFT_C2C ) != CUFFT_SUCCESS )
            throw imageException( "Cannot create FFT plan on CUDA device" );

        _width  = width_;
        _height = height_;
    }

    uint32_t FFTExecutor::width() const
    {
        return _width;
    }

    uint32_t FFTExecutor::height() const
    {
        return _height;
    }

    void FFTExecutor::directTransform( ComplexData & data )
    {
        directTransform( data, data );
    }

    void FFTExecutor::directTransform( ComplexData & in, ComplexData & out )
    {
        if( _plan == 0 || _width != in.width() || _height != in.height() || _width != out.width() || _height != out.height() )
            throw imageException( "Invalid parameters for FFTExecutor" );

        if( cufftExecC2C( _plan, in.data(), out.data(), CUFFT_FORWARD ) != CUFFT_SUCCESS )
            throw imageException( "Cannot execute direct FFT transform on CUDA device" );
    }

    void FFTExecutor::inverseTransform( ComplexData & data )
    {
        inverseTransform( data, data );
    }

    void FFTExecutor::inverseTransform( ComplexData & in, ComplexData & out )
    {
        if( _plan == 0 || _width != in.width() || _height != in.height() || _width != out.width() || _height != out.height() )
            throw imageException( "Invalid parameters for FFTExecutor" );

        if( cufftExecC2C( _plan, in.data(), out.data(), CUFFT_INVERSE ) != CUFFT_SUCCESS )
            throw imageException( "Cannot execute inverse FFT transform on CUDA device" );
    }

    void FFTExecutor::complexMultiplication( ComplexData & in1, ComplexData & in2, ComplexData & out ) const
    {
        if( in1.width() != in2.width() || in1.height() != in2.height() || in1.width() != out.width() || in1.height() != out.height() ||
            in1.width() == 0 || in1.height() == 0 )
            throw imageException( "Invalid parameters for FFTExecutor" );

        launchKernel2D( complexMultiplicationCuda, _width, _height,
                        in1.data(), in2.data(), out.data(), _width, _height );
    }

    void FFTExecutor::_clean()
    {
        if( _plan != 0 ) {
            cufftDestroy( _plan );

            _plan = 0;
        }

        _width  = 0;
        _height = 0;
    }
}
