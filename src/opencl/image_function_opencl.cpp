#include <math.h>
#include "../thirdparty/multicl/src/opencl_device.h"
#include "../thirdparty/multicl/src/opencl_helper.h"
#include "image_function_opencl.h"
#include "../image_function_helper.h"
#include "../parameter_validation.h"

namespace
{
    const std::string programCode = R"(
        // OpenCL kernels for PenguinV library
        #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

        __kernel void absoluteDifferenceOpenCL( __global const unsigned char * in1, __global const unsigned char * in2, __global unsigned char * out, unsigned int width, unsigned int height )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;
                out[id] = in1[id] > in2[id] ? in1[id] - in2[id] : in2[id] - in1[id];
            }
        }

        __kernel void bitwiseAndOpenCL( __global const unsigned char * in1, __global const unsigned char * in2, __global unsigned char * out, unsigned int width, unsigned int height )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;
                out[id] = in1[id] & in2[id];
            }
        }

        __kernel void bitwiseOrOpenCL( __global const unsigned char * in1, __global const unsigned char * in2, __global unsigned char * out, unsigned int width, unsigned int height )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;
                out[id] = in1[id] | in2[id];
            }
        }

        __kernel void bitwiseXorOpenCL( __global const unsigned char * in1, __global const unsigned char * in2, __global unsigned char * out, unsigned int width, unsigned int height )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;
                out[id] = in1[id] ^ in2[id];
            }
        }

        __kernel void convertToGrayScaleOpenCL( __global const unsigned char * in, __global unsigned char * out, unsigned int width, unsigned int height, unsigned char colorCount )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;

                unsigned int sum = 0;
                __global const unsigned char * data = in + (width * y + x) * colorCount;

                for( unsigned char i = 0; i < colorCount; ++i, ++data )
                {
                    sum += (*data);
                }

                out[id] = sum / colorCount;
            }
        }

        __kernel void convertToRgbOpenCL( __global const unsigned char * in, __global unsigned char * out, unsigned int width, unsigned int height, unsigned char colorCount )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;

                __global unsigned char * data = out + (width * y + x) * colorCount;

                for( unsigned char i = 0; i < colorCount; ++i, ++data )
                {
                    (*data) = in[id];
                }
            }
        }

        __kernel void extractChannelOpenCL( __global const unsigned char * in, __global unsigned char * out, unsigned int width, unsigned int height, unsigned char channelCount, unsigned char channelId )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height )
                out[y * width + x] = in[(y * width + x) * channelCount + channelId];
        }

        __kernel void fillOpenCL( __global unsigned char * data, unsigned char value, unsigned int size )
        {
            size_t id = get_global_id(0);

            if( id < size ) {
                data[id] = value;
            }
        }

        __kernel void flipOpenCL( __global const unsigned char * in, __global unsigned char * out, unsigned int width, unsigned int height, unsigned char horizontal, unsigned char vertical )
        {
            const size_t inX = get_global_id(0);
            const size_t inY = get_global_id(1);

            if( inX < width && inY < height ) {
                const size_t outX = (horizontal != 0) ? (width  - 1 - inX) : inX;
                const size_t outY = (vertical != 0)   ? (height - 1 - inY) : inY;

                out[outY * width + outX] = in[inY * width + inX];
            }
        }

        __kernel void histogramOpenCL( __global const unsigned char * data, unsigned int width, unsigned int height, volatile __global unsigned int * histogram )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;
                atomic_add( &histogram[data[id]], 1 );
            }
        }

        __kernel void invertOpenCL( __global const unsigned char * in, __global unsigned char * out, unsigned int width, unsigned int height )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;
                out[id] = ~in[id];
            }
        }

        __kernel void lookupTableOpenCL( __global const unsigned char * in, __global unsigned char * out, unsigned int width, unsigned int height, __global unsigned char * table )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;
                out[id] = table[in[id]];
            }
        }

        __kernel void maximumOpenCL( __global const unsigned char * in1, __global const unsigned char * in2, __global unsigned char * out, unsigned int width, unsigned int height )
        {
           const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;
                out[id] = in1[id] > in2[id] ? in1[id] : in2[id];
            }
        }

        __kernel void minimumOpenCL( __global const unsigned char * in1, __global const unsigned char * in2, __global unsigned char * out, unsigned int width, unsigned int height )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;
                out[id] = in1[id] < in2[id] ? in1[id] : in2[id];
            }
        }

        __kernel void subtractOpenCL( __global const unsigned char * in1, __global const unsigned char * in2, __global unsigned char * out, unsigned int width, unsigned int height )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;
                out[id] = in1[id] > in2[id] ? in1[id] - in2[id] : 0;
            }
        }

        __kernel void thresholdOpenCL( __global const unsigned char * in, __global unsigned char * out, unsigned int width, unsigned int height, unsigned char threshold )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;
                out[id] = in[id] < threshold ? 0 : 255;
            }
        }

        __kernel void thresholdOpenCL2( __global const unsigned char * in, __global unsigned char * out, unsigned int width, unsigned int height, unsigned char minThreshold, unsigned char maxThreshold )
        {
            const size_t x = get_global_id(0);
            const size_t y = get_global_id(1);

            if( x < width && y < height ) {
                const size_t id = y * width + x;
                out[id] = in[id] < minThreshold || in[id] > maxThreshold ? 0 : 255;
            }
        }
        )";

    multiCL::OpenCLProgram GetProgram()
    {
        return multiCL::OpenCLProgram( multiCL::OpenCLDeviceManager::instance().device().context(), programCode );
    }
}

namespace Image_Function_OpenCL
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

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "absoluteDifferenceOpenCL");
        kernel.setArgument( in1.data(), in2.data(), out.data(), out.rowSize(), out.height() );

        multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );
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

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "bitwiseAndOpenCL");
        kernel.setArgument( in1.data(), in2.data(), out.data(), out.rowSize(), out.height() );

        multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );
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

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "bitwiseOrOpenCL");
        kernel.setArgument( in1.data(), in2.data(), out.data(), out.rowSize(), out.height() );

        multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );
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

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "bitwiseXorOpenCL");
        kernel.setArgument( in1.data(), in2.data(), out.data(), out.rowSize(), out.height() );

        multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );
    }

    Image ConvertToOpenCL( const PenguinV_Image::Image & in )
    {
        Image out( in.width(), in.height(), in.colorCount() );

        ConvertToOpenCL( in, out );

        return out;
    }

    void ConvertToOpenCL( const PenguinV_Image::Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        if( in.width() != out.width() || in.height() != out.height() ||
            in.colorCount() != out.colorCount())
            throw imageException( "Bad input parameters in image function" );

        if (in.alignment() == 1u || (in.rowSize() == in.width() * in.colorCount()))
        {
            const uint32_t size = in.rowSize() * in.height();

            multiCL::writeBuffer( out.data(), size * sizeof(uint8_t), in.data() );
        }
        else
        {
            const size_t origin[3] = {0, 0, 0};
            const size_t region[3] = {in.width(), in.height(), 1};

            multiCL::openCLCheck( clEnqueueWriteImage( multiCL::OpenCLDeviceManager::instance().device().queue()(), out.data(), CL_TRUE,
                                                      origin, region, in.rowSize(), 0, in.data(), 0, NULL, NULL ) );
        }
    }

    PenguinV_Image::Image ConvertFromOpenCL( const Image & in )
    {
        PenguinV_Image::Image out( in.width(), in.height(), in.colorCount(), 1u );

        ConvertFromOpenCL( in, out );

        return out;
    }

    void ConvertFromOpenCL( const Image & in, PenguinV_Image::Image & out )
    {
        Image_Function::ParameterValidation( in );
        Image_Function::ParameterValidation( out );

        if( in.width() != out.width() || in.height() != out.height() ||
            in.colorCount() != out.colorCount())
            throw imageException( "Bad input parameters in image function" );

        if (out.alignment() == 1u || (out.rowSize() == out.width() * out.colorCount()))
        {
            const uint32_t size = in.rowSize() * in.height();

            multiCL::readBuffer( in.data(), size * sizeof(uint8_t), out.data() );
        }
        else
        {
            const size_t origin[3] = {0, 0, 0};
            const size_t region[3] = {out.width(), out.height(), 1};

            multiCL::openCLCheck( clEnqueueReadImage( multiCL::OpenCLDeviceManager::instance().device().queue()(), in.data(), CL_TRUE,
                                                      origin, region, out.rowSize(), 0, out.data(), 0, NULL, NULL ) );
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

        if( in.colorCount() == PenguinV_Image::GRAY_SCALE ) {
            Copy( in, out );
            return;
        }

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "convertToGrayScaleOpenCL");
        kernel.setArgument( in.data(), out.data(), in.width(), in.height(), in.colorCount() );

        multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );
    }

    Image ConvertToRgb( const Image & in )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height(), PenguinV_Image::RGB );

        ConvertToRgb( in, out );

        return out;
    }

    void ConvertToRgb( const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );
        Image_Function::VerifyRGBImage( out );

        if( in.colorCount() == PenguinV_Image::RGB ) {
            Copy( in, out );
        }
        else {
            const multiCL::OpenCLProgram & program = GetProgram();
            multiCL::OpenCLKernel kernel( program, "convertToRgbOpenCL");
            kernel.setArgument( in.data(), out.data(), in.width(), in.height(), out.colorCount() );

            multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );
        }
    }

    void Copy( const Image & in, Image & out )
    {
        Image_Function::ParameterValidation( in, out );

        out = in;
    }

    Image ExtractChannel( const Image & in, uint8_t channelId )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        ExtractChannel( in, out, channelId );

        return out;
    }

    void ExtractChannel( const Image & in, Image & out, uint8_t channelId )
    {
        Image_Function::ParameterValidation( in, out );
        Image_Function::VerifyGrayScaleImage( out );

        if( channelId >= in.colorCount() )
            throw imageException( "Channel ID for color image is greater than channel count in input image" );

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "extractChannelOpenCL");
        kernel.setArgument( in.data(), out.data(), out.width(), out.height(), in.colorCount(), channelId );

        multiCL::launchKernel2D( kernel, out.width(), out.height() );
    }

    void Fill( Image & image, uint8_t value )
    {
        Image_Function::ParameterValidation( image );

        const uint32_t size = image.rowSize() * image.height();

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "fillOpenCL");
        kernel.setArgument( image.data(), value, size );

        multiCL::launchKernel1D( kernel, size );
    }

    Image Flip( const Image & in, bool horizontal, bool vertical )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        Flip( in, out, horizontal, vertical );

        return out;
    }

    void Flip( const Image & in, Image & out, bool horizontal, bool vertical )
    {
        Image_Function::ParameterValidation( in, out );
        Image_Function::VerifyGrayScaleImage( in, out );

        if( !horizontal && !vertical ) {
            Copy( in, out );
        }
        else {
            const multiCL::OpenCLProgram & program = GetProgram();
            multiCL::OpenCLKernel kernel( program, "flipOpenCL");
            kernel.setArgument( in.data(), out.data(), out.rowSize(), out.height(), horizontal, vertical );

            multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );
        }
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
        return Image_Function_Helper::GetThreshold( histogram );
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

        cl_mem histogramOpenCL = multiCL::MemoryManager::memory().allocate<uint32_t>( histogram.size() );
        multiCL::writeBuffer( histogramOpenCL, sizeof( uint32_t ) * histogram.size(), histogram.data() );

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "histogramOpenCL");
        kernel.setArgument( image.data(), image.rowSize(), image.height(), histogramOpenCL );

        multiCL::launchKernel2D( kernel, image.rowSize(), image.height() );

        multiCL::readBuffer( histogramOpenCL, sizeof( uint32_t ) * histogram.size(), histogram.data() );
        multiCL::MemoryManager::memory().free( histogramOpenCL );
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

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "invertOpenCL");
        kernel.setArgument( in.data(), out.data(), out.rowSize(), out.height() );

        multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );
    }

    Image LookupTable( const Image & in, const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in );

        Image out( in.width(), in.height() );

        LookupTable( in, out, table );

        return out;
    }

    void LookupTable( const Image & in, Image & out, const std::vector < uint8_t > & table )
    {
        Image_Function::ParameterValidation( in, out );
        Image_Function::VerifyGrayScaleImage( in, out );

        if( table.size() != 256u )
            throw imageException( "Lookup table size is not equal to 256" );

        cl_mem tableOpenCL = multiCL::MemoryManager::memory().allocate<uint8_t>( table.size() );
        multiCL::writeBuffer( tableOpenCL, sizeof(uint8_t) * table.size(), table.data() );

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "lookupTableOpenCL");
        kernel.setArgument( in.data(), out.data(), out.rowSize(), out.height(), tableOpenCL );

        multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );

        multiCL::MemoryManager::memory().free( tableOpenCL );
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

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "maximumOpenCL");
        kernel.setArgument( in1.data(), in2.data(), out.data(), out.rowSize(), out.height() );

        multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );
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

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "minimumOpenCL");
        kernel.setArgument( in1.data(), in2.data(), out.data(), out.rowSize(), out.height() );

        multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );
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

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "subtractOpenCL");
        kernel.setArgument( in1.data(), in2.data(), out.data(), out.rowSize(), out.height() );

        multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );
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

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "thresholdOpenCL");
        kernel.setArgument( in.data(), out.data(), out.rowSize(), out.height(), threshold );

        multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );
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

        const multiCL::OpenCLProgram & program = GetProgram();
        multiCL::OpenCLKernel kernel( program, "thresholdOpenCL2");
        kernel.setArgument( in.data(), out.data(), out.rowSize(), out.height(), minThreshold, maxThreshold );

        multiCL::launchKernel2D( kernel, out.rowSize(), out.height() );
    }
}
