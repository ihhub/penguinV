#include <math.h>
#include "filtering.h"
#include "image_function.h"

namespace Image_Function
{
    namespace Filtering
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

        void Median( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                     uint32_t width, uint32_t height, uint32_t kernelSize )
        {
            ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            VerifyGrayScaleImage( in, out );

            if( kernelSize < 3 || kernelSize % 2 == 0 || kernelSize >= width || kernelSize >= height )
                throw imageException( "Kernel size for filter is not correct" );

            // Border's problem is well-known problem which can be solved in different ways
            // We just copy parts of original image without applying filtering
            Copy( in, startXIn, startYIn, out, startXOut,
                  startYOut, width, kernelSize / 2 );
            Copy( in, startXIn, startYIn + height - kernelSize / 2, out, startXOut,
                  startYOut + height - kernelSize / 2, width, kernelSize / 2 );
            Copy( in, startXIn, startYIn + kernelSize / 2, out, startXOut,
                  startYOut + kernelSize / 2, kernelSize / 2, height - (kernelSize - 1) );
            Copy( in, startXIn + width - kernelSize / 2, startYIn + kernelSize / 2, out, startXOut + width - kernelSize / 2,
                  startYOut + kernelSize / 2, kernelSize / 2, height - (kernelSize - 1) );

            std::vector < uint8_t > data( kernelSize * kernelSize );
            const size_t dataMedianPosition = data.size() / 2;

            const uint32_t rowSizeIn  = in.rowSize();
            const uint32_t rowSizeOut = out.rowSize();

            const uint8_t * inY  = in.data()  + startYIn                     * rowSizeIn  + startXIn;
            uint8_t       * outY = out.data() + (startYOut + kernelSize / 2) * rowSizeOut + startXOut + kernelSize / 2;

            width  = width  - (kernelSize - 1);
            height = height - (kernelSize - 1);

            const uint8_t * outYEnd = outY + height * rowSizeOut;

            for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
                const uint8_t * inX = inY;
                uint8_t       * outX = outY;

                const uint8_t * outXEnd = outX + width;

                for( ; outX != outXEnd; ++outX, ++inX ) {
                    uint8_t * value = data.data();

                    const uint8_t * inYRead    = inX;
                    const uint8_t * inYReadEnd = inYRead + kernelSize * rowSizeIn;

                    for( ; inYRead != inYReadEnd; inYRead += rowSizeIn ) {
                        const uint8_t * inXRead    = inYRead;
                        const uint8_t * inXReadEnd = inXRead + kernelSize;

                        for( ; inXRead != inXReadEnd; ++inXRead, ++value )
                            *value = *inXRead;
                    }

                    std::nth_element( data.begin(), data.begin() + dataMedianPosition, data.end() );

                    (*outX) = data[dataMedianPosition];
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

        void Prewitt( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                      uint32_t width, uint32_t height )
        {
            ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            VerifyGrayScaleImage( in, out );

            if( width < 3 || height < 3 )
                throw imageException( "Input image is very small for Sobel filter to be applied" );

            // Create the map of gradient values
            const uint32_t gradientWidth  = width  - 2;
            const uint32_t gradientHeight = height - 2;

            const float multiplier = 255.0f / sqrtf( 1170450.0f ); // 1170450 is 2 * (255 * 3) * (255 * 3);

            const uint32_t rowSizeIn  = in.rowSize();
            const uint32_t rowSizeOut = out.rowSize();

            const uint8_t * inY  = in.data()  + (startYIn + 1) * rowSizeIn + startXIn + 1;
            uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

            const uint8_t * inYEnd = inY + gradientHeight * rowSizeIn;

            // fill top row with zeros
            memset( outY, 0, width );
            outY += rowSizeOut;

            for( ; inY != inYEnd; inY += rowSizeIn, outY += rowSizeOut ) {
                // set first pixel in row to 0
                *outY = 0;

                uint8_t       * outX    = outY + 1;
                const uint8_t * inX = inY;
                const uint8_t * inXEnd = inX + gradientWidth;

                for( ; inX != inXEnd; ++inX, ++outX ) {
                    //      | +1  0 -1|
                    // Gx = | +1  0 -1|
                    //      | +1  0 -1|

                    //      | +1 +1 +1|
                    // Gy = |  0  0  0|
                    //      | -1 -1 -1|

                    const int32_t gX = *(inX - rowSizeIn - 1) + *(inX - 1) + *(inX + rowSizeIn - 1) -
                                       *(inX - rowSizeIn + 1) - *(inX + 1) - *(inX + rowSizeIn + 1);

                    const int32_t gY = *(inX - rowSizeIn - 1) + *(inX - rowSizeIn) + *(inX - rowSizeIn + 1) -
                                       *(inX + rowSizeIn - 1) - *(inX + rowSizeIn) - *(inX + rowSizeIn + 1);

                    *outX = static_cast<uint8_t>(sqrtf( static_cast<float>(gX * gX + gY + gY) ) * multiplier + 0.5f);
                }

                // set last pixel in row to 0
                *outX = 0;
            }

            // fill bottom row with zeros
            memset( outY, 0, width );
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

        void Sobel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height )
        {
            ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            VerifyGrayScaleImage( in, out );

            if( width < 3 || height < 3 )
                throw imageException( "Input image is very small for Sobel filter to be applied" );

            // Create the map of gradient values
            const uint32_t gradientWidth  = width  - 2;
            const uint32_t gradientHeight = height - 2;

            const float multiplier = 255.0f / sqrtf( 2080800.0f ); // 2080800 is 2 * (255 * 4) * (255 * 4);

            const uint32_t rowSizeIn  = in.rowSize();
            const uint32_t rowSizeOut = out.rowSize();

            const uint8_t * inY  = in.data()  + (startYIn + 1) * rowSizeIn + startXIn + 1;
            uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

            const uint8_t * inYEnd = inY + gradientHeight * rowSizeIn;

            // fill top row with zeros
            memset( outY, 0, width );
            outY += rowSizeOut;

            for( ; inY != inYEnd; inY += rowSizeIn, outY += rowSizeOut ) {
                // set first pixel in row to 0
                *outY = 0;

                uint8_t       * outX    = outY + 1;
                const uint8_t * inX = inY;
                const uint8_t * inXEnd = inX + gradientWidth;

                for( ; inX != inXEnd; ++inX, ++outX ) {
                    //      | +1  0 -1|
                    // Gx = | +2  0 -2|
                    //      | +1  0 -1|

                    //      | +1 +2 +1|
                    // Gy = |  0  0  0|
                    //      | -1 -2 -1|

                    const int32_t gX = *(inX - rowSizeIn - 1) + 2 * (*(inX - 1)) + *(inX + rowSizeIn - 1) -
                                       *(inX - rowSizeIn + 1) - 2 * (*(inX + 1)) - *(inX + rowSizeIn + 1);

                    const int32_t gY = *(inX - rowSizeIn - 1) + 2 * (*(inX - rowSizeIn)) + *(inX - rowSizeIn + 1) -
                                       *(inX + rowSizeIn - 1) - 2 * (*(inX + rowSizeIn)) - *(inX + rowSizeIn + 1);

                    *outX = static_cast<uint8_t>(sqrtf( static_cast<float>(gX * gX + gY + gY) ) * multiplier + 0.5f);
                }

                // set last pixel in row to 0
                *outX = 0;
            }

            // fill bottom row with zeros
            memset( outY, 0, width );
        }

        void GetGaussianKernel( std::vector<float> & filter, uint32_t width, uint32_t height, uint32_t kernelSize, float sigma )
        {
            if( width < 3 || height < 3 || kernelSize == 0 || width < (kernelSize * 2 + 1) || height < (kernelSize * 2 + 1) || sigma < 0 )
                throw imageException( "Incorrect input parameters for Gaussian filter kernel" );

            const uint32_t size = width * height;

            filter.resize( size );

            std::fill( filter.begin(), filter.end(), 0.0f );

            static const float pi = 3.1415926536f;
            const float doubleSigma = sigma * 2;

            float * y = filter.data() + (height / 2 - kernelSize) * width + width / 2 - kernelSize;
            const float * endY = y + (2 * kernelSize + 1) * width;

            float sum = 0;

            for( int32_t posY = -static_cast<int32_t>(kernelSize) ; y != endY; y += width, ++posY ) {
                float * x = y;
                const float * endX = x + 2 * kernelSize + 1;

                for( int32_t posX = -static_cast<int32_t>(kernelSize) ; x != endX; ++x, ++posX ) {
                    *x = 1.0f / (pi * doubleSigma) * exp( -(posX * posX + posY * posY) / doubleSigma );
                    sum += *x;
                }
            }

            const float normalization = 1.0f / sum;

            y = filter.data() + (height / 2 - kernelSize) * width + width / 2 - kernelSize;

            for( int32_t posY = -static_cast<int32_t>(kernelSize) ; y != endY; y += width, ++posY ) {
                float * x = y;
                const float * endX = x + 2 * kernelSize + 1;

                for( int32_t posX = -static_cast<int32_t>(kernelSize) ; x != endX; ++x, ++posX ) {
                    *x *= normalization;
                }
            }
        }
    };
};
