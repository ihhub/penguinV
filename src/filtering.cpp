#include "filtering.h"
#include "image_function.h"
#include "parameter_validation.h"
#include <cmath>

namespace
{
    // We could precalculate all values so our filter would be faster
    // This structure needs some optimisation in size as we store all values in symmetric matrix
    // We could reduce the size of the array by storing only a half of the matrix
    template <uint32_t sizeInBytes>
    struct FilterKernel
    {
        explicit FilterKernel( float multiplier_ )
        {
            const uint32_t size = 256u * sizeInBytes;
            for ( uint32_t i = 0; i < size; ++i ) {
                for ( uint32_t j = i; j < size; ++j )
                    kernel[i][j] = kernel[j][i] = static_cast<uint8_t>( sqrtf( static_cast<float>( i * i + j * j ) ) * multiplier_ + 0.5f );
            }
        }

        uint8_t kernel[256u * sizeInBytes][256u * sizeInBytes];
    };
} // namespace

namespace Image_Function
{
    Image Median( const Image & in, uint32_t kernelSize )
    {
        ParameterValidation( in );

        Image out( in.width(), in.height() );

        Median( in, 0, 0, out, 0, 0, out.width(), out.height(), kernelSize );

        return out;
    }

    void Median( const Image & in, Image & out, uint32_t kernelSize )
    {
        ParameterValidation( in, out );

        Median( in, 0, 0, out, 0, 0, out.width(), out.height(), kernelSize );
    }

    Image Median( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint32_t kernelSize )
    {
        ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height );

        Median( in, startXIn, startYIn, out, 0, 0, width, height, kernelSize );

        return out;
    }

    void Median( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height,
                 uint32_t kernelSize )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyGrayScaleImage( in, out );

        if ( kernelSize < 3 || kernelSize % 2 == 0 || kernelSize >= width || kernelSize >= height )
            throw imageException( "Kernel size for filter is not correct" );

        // Border's problem is well-known problem which can be solved in different ways
        // We just copy parts of original image without applying filtering
        Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, kernelSize / 2 );
        Copy( in, startXIn, startYIn + height - kernelSize / 2, out, startXOut, startYOut + height - kernelSize / 2, width, kernelSize / 2 );
        Copy( in, startXIn, startYIn + kernelSize / 2, out, startXOut, startYOut + kernelSize / 2, kernelSize / 2, height - ( kernelSize - 1 ) );
        Copy( in, startXIn + width - kernelSize / 2, startYIn + kernelSize / 2, out, startXOut + width - kernelSize / 2, startYOut + kernelSize / 2, kernelSize / 2,
              height - ( kernelSize - 1 ) );

        std::vector<uint8_t> data( kernelSize * kernelSize );
        uint8_t * dataFirstValue = data.data();
        uint8_t * medianValue = dataFirstValue + data.size() / 2;
        uint8_t * dataLastValue = dataFirstValue + data.size();

        const uint32_t rowSizeIn = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY = in.data() + startYIn * rowSizeIn + startXIn;
        uint8_t * outY = out.data() + ( startYOut + kernelSize / 2 ) * rowSizeOut + startXOut + kernelSize / 2;

        width = width - ( kernelSize - 1 );
        height = height - ( kernelSize - 1 );

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for ( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX = inY;
            uint8_t * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for ( ; outX != outXEnd; ++outX, ++inX ) {
                uint8_t * value = data.data();

                const uint8_t * inYRead = inX;
                const uint8_t * inYReadEnd = inYRead + kernelSize * rowSizeIn;

                for ( ; inYRead != inYReadEnd; inYRead += rowSizeIn ) {
                    const uint8_t * inXRead = inYRead;
                    const uint8_t * inXReadEnd = inXRead + kernelSize;

                    for ( ; inXRead != inXReadEnd; ++inXRead, ++value )
                        *value = *inXRead;
                }

                std::nth_element( dataFirstValue, medianValue, dataLastValue );
                ( *outX ) = *medianValue;
            }
        }
    }

    Image Prewitt( const Image & in )
    {
        ParameterValidation( in );

        Image out( in.width(), in.height() );

        Prewitt( in, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Prewitt( const Image & in, Image & out )
    {
        ParameterValidation( in, out );

        Prewitt( in, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Prewitt( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height );

        Sobel( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    void Prewitt( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyGrayScaleImage( in, out );

        if ( width < 3 || height < 3 )
            throw imageException( "Input image is very small for Sobel filter to be applied" );

        const uint32_t startXOffset = ( startXIn > 0u ) ? 1u : 0u;
        const uint32_t startYOffset = ( startYIn > 0u ) ? 1u : 0u;
        const uint32_t endXOffset = ( ( startXIn + width + 1u ) < in.width() ) ? 1u : 0u;
        const uint32_t endYOffset = ( ( startYIn + height + 1u ) < in.height() ) ? 1u : 0u;

        // Create the map of gradient values
        const uint32_t gradientWidth = width - ( ( ( startXOffset > 0u ) ? 0u : 1u ) + ( ( endXOffset > 0u ) ? 0u : 1u ) );
        const uint32_t gradientHeight = height - ( ( ( startYOffset > 0u ) ? 0u : 1u ) + ( ( endYOffset > 0u ) ? 0u : 1u ) );

        const float multiplier = 1.0f / ( 3.0f * sqrtf( 2.0f ) );

        const uint32_t rowSizeIn = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY = in.data() + ( startYIn + 1u - startYOffset ) * rowSizeIn + startXIn + 1u - startXOffset;
        uint8_t * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * inYEnd = inY + gradientHeight * rowSizeIn;

        if ( startYOffset == 0u ) {
            // fill top row with zeros
            memset( outY, 0, width );
            outY += rowSizeOut;
        }

        const uint32_t yPlusX = rowSizeIn + 1u;
        const uint32_t yMinusX = rowSizeIn - 1u;

        static const FilterKernel<3u> kernel( multiplier );

        for ( ; inY != inYEnd; inY += rowSizeIn, outY += rowSizeOut ) {
            uint8_t * outX = outY;

            if ( startXOffset == 0u ) {
                // set first pixel in row to 0
                *outX = 0;
                ++outX;
            }

            const uint8_t * inX = inY;
            const uint8_t * inXEnd = inX + gradientWidth;

            for ( ; inX != inXEnd; ++inX, ++outX ) {
                //      | +1  0 -1|
                // Gx = | +1  0 -1|
                //      | +1  0 -1|

                //      | +1 +1 +1|
                // Gy = |  0  0  0|
                //      | -1 -1 -1|

                const int32_t partialG = static_cast<int32_t>( *( inX - yPlusX ) ) - *( inX + yPlusX ); // [y - 1; x - 1] - [y + 1; x + 1]

                const int32_t gX = partialG + *( inX - 1 ) + *( inX + yMinusX ) - *( inX - yMinusX ) - *( inX + 1 );
                const int32_t gY = partialG + *( inX - rowSizeIn ) + *( inX - yMinusX ) - *( inX + yMinusX ) - *( inX + rowSizeIn );

                *outX = kernel.kernel[( gX < 0 ) ? -gX : gX][( gY < 0 ) ? -gY : gY];
            }

            if ( endXOffset == 0u ) {
                // set last pixel in row to 0
                *outX = 0;
            }
        }

        if ( endYOffset == 0u ) {
            // fill bottom row with zeros
            memset( outY, 0, width );
        }
    }

    Image Sobel( const Image & in )
    {
        ParameterValidation( in );

        Image out( in.width(), in.height() );

        Sobel( in, 0, 0, out, 0, 0, out.width(), out.height() );

        return out;
    }

    void Sobel( const Image & in, Image & out )
    {
        ParameterValidation( in, out );

        Sobel( in, 0, 0, out, 0, 0, out.width(), out.height() );
    }

    Image Sobel( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, width, height );

        Image out( width, height );

        Sobel( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    void Sobel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyGrayScaleImage( in, out );

        if ( width < 3 || height < 3 )
            throw imageException( "Input image is very small for Sobel filter to be applied" );

        const uint32_t startXOffset = ( startXIn > 0u ) ? 1u : 0u;
        const uint32_t startYOffset = ( startYIn > 0u ) ? 1u : 0u;
        const uint32_t endXOffset = ( ( startXIn + width + 1u ) < in.width() ) ? 1u : 0u;
        const uint32_t endYOffset = ( ( startYIn + height + 1u ) < in.height() ) ? 1u : 0u;

        // Create the map of gradient values
        const uint32_t gradientWidth = width - ( ( ( startXOffset > 0u ) ? 0u : 1u ) + ( ( endXOffset > 0u ) ? 0u : 1u ) );
        const uint32_t gradientHeight = height - ( ( ( startYOffset > 0u ) ? 0u : 1u ) + ( ( endYOffset > 0u ) ? 0u : 1u ) );

        const float multiplier = 1.0f / ( 4.0f * sqrtf( 2.0f ) );

        const uint32_t rowSizeIn = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY = in.data() + ( startYIn + 1u - startYOffset ) * rowSizeIn + startXIn + 1u - startXOffset;
        uint8_t * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * inYEnd = inY + gradientHeight * rowSizeIn;

        if ( startYOffset == 0u ) {
            // fill top row with zeros
            memset( outY, 0, width );
            outY += rowSizeOut;
        }

        const uint32_t yPlusX = rowSizeIn + 1u;
        const uint32_t yMinusX = rowSizeIn - 1u;
        static const FilterKernel<4u> kernel( multiplier );

        for ( ; inY != inYEnd; inY += rowSizeIn, outY += rowSizeOut ) {
            uint8_t * outX = outY;

            if ( startXOffset == 0u ) {
                // set first pixel in row to 0
                *outX = 0;
                ++outX;
            }

            const uint8_t * inX = inY;
            const uint8_t * inXEnd = inX + gradientWidth;

            for ( ; inX != inXEnd; ++inX, ++outX ) {
                //      | +1  0 -1|
                // Gx = | +2  0 -2|
                //      | +1  0 -1|

                //      | +1 +2 +1|
                // Gy = |  0  0  0|
                //      | -1 -2 -1|

                const int32_t partialG = static_cast<int32_t>( *( inX - yPlusX ) ) - *( inX + yPlusX ); // [y - 1; x - 1] - [y + 1; x + 1]

                const int32_t gX = partialG + *( inX + yMinusX ) - *( inX - yMinusX ) + 2 * ( static_cast<int32_t>( *( inX - 1 ) ) - ( *( inX + 1 ) ) );
                const int32_t gY = partialG + *( inX - yMinusX ) - *( inX + yMinusX ) + 2 * ( static_cast<int32_t>( *( inX - rowSizeIn ) ) - ( *( inX + rowSizeIn ) ) );

                *outX = kernel.kernel[( gX < 0 ) ? -gX : gX][( gY < 0 ) ? -gY : gY];
            }

            if ( endXOffset == 0u ) {
                // set last pixel in row to 0
                *outX = 0;
            }
        }

        if ( endYOffset == 0u ) {
            // fill bottom row with zeros
            memset( outY, 0, width );
        }
    }

    void GetGaussianKernel( std::vector<float> & filter, uint32_t width, uint32_t height, uint32_t kernelSize, float sigma )
    {
        if ( width < 3 || height < 3 || kernelSize == 0 || width < ( kernelSize * 2 + 1 ) || height < ( kernelSize * 2 + 1 ) || sigma < 0 )
            throw imageException( "Incorrect input parameters for Gaussian filter kernel" );

        filter.resize( width * height );

        std::fill( filter.begin(), filter.end(), 0.0f );

        static const float pi = 3.1415926536f;
        const float doubleSigma = sigma * 2;
        const float doubleSigmaPiInv = 1.0f / ( doubleSigma * pi );
        const uint32_t twiceKernelSizePlusOne = 2 * kernelSize + 1;

        float * y = filter.data() + ( height / 2 - kernelSize ) * width + width / 2 - kernelSize;
        const float * endY = y + twiceKernelSizePlusOne * width;

        float sum = 0;

        for ( int32_t posY = -static_cast<int32_t>( kernelSize ); y != endY; y += width, ++posY ) {
            float * x = y;
            const float * endX = x + twiceKernelSizePlusOne;
            const int32_t posY2 = posY * posY;

            for ( int32_t posX = -static_cast<int32_t>( kernelSize ); x != endX; ++x, ++posX ) {
                *x = doubleSigmaPiInv * expf( -static_cast<float>( posX * posX + posY2 ) / doubleSigma );
                sum += *x;
            }
        }

        const float normalization = 1.0f / sum;

        y = filter.data() + ( height / 2 - kernelSize ) * width + width / 2 - kernelSize;

        for ( int32_t posY = -static_cast<int32_t>( kernelSize ); y != endY; y += width, ++posY ) {
            float * x = y;
            const float * endX = x + twiceKernelSizePlusOne;

            for ( int32_t posX = -static_cast<int32_t>( kernelSize ); x != endX; ++x, ++posX ) {
                *x *= normalization;
            }
        }
    }
} // namespace Image_Function
