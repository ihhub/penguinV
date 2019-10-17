#include <cmath>
#include "image_function.h"
#include "parameter_validation.h"
#include "image_function_helper.h"

namespace
{
    struct FunctionRegistrator
    {
        Image_Function_Helper::FunctionTableHolder table;
        
        FunctionRegistrator()
        {
            table.AbsoluteDifference = &Image_Function::AbsoluteDifference;
            table.Accumulate         = &Image_Function::Accumulate;
            table.BitwiseAnd         = &Image_Function::BitwiseAnd;
            table.BitwiseOr          = &Image_Function::BitwiseOr;
            table.BitwiseXor         = &Image_Function::BitwiseXor;
            table.ConvertTo16Bit     = Image_Function::ConvertTo16Bit;
            table.ConvertTo8Bit      = Image_Function::ConvertTo8Bit;
            table.ConvertToGrayScale = &Image_Function::ConvertToGrayScale;
            table.ConvertToRgb       = &Image_Function::ConvertToRgb;
            table.Copy               = &Image_Function::Copy;
            table.ExtractChannel     = &Image_Function::ExtractChannel;
            table.Fill               = &Image_Function::Fill;
            table.Flip               = &Image_Function::Flip;
            table.GammaCorrection    = &Image_Function::GammaCorrection;
            table.GetPixel           = &Image_Function::GetPixel;
            table.Histogram          = &Image_Function::Histogram;
            table.Invert             = &Image_Function::Invert;
            table.IsEqual            = &Image_Function::IsEqual;
            table.LookupTable        = &Image_Function::LookupTable;
            table.Maximum            = &Image_Function::Maximum;
            table.Merge              = &Image_Function::Merge;
            table.Minimum            = &Image_Function::Minimum;
            table.Normalize          = &Image_Function::Normalize;
            table.ProjectionProfile  = &Image_Function::ProjectionProfile;
            table.Resize             = &Image_Function::Resize;
            table.RgbToBgr           = &Image_Function::RgbToBgr;
            table.RgbToRgba          = &Image_Function::RgbToRgba;
            table.RgbaToRgb          = &Image_Function::RgbaToRgb;
            table.SetPixel           = &Image_Function::SetPixel;
            table.SetPixel2          = &Image_Function::SetPixel;
            table.Shift              = &Image_Function::Shift;
            table.Split              = &Image_Function::Split;
            table.Subtract           = &Image_Function::Subtract;
            table.Sum                = &Image_Function::Sum;
            table.Threshold          = &Image_Function::Threshold;
            table.Threshold2         = &Image_Function::Threshold;
            table.Transpose          = &Image_Function::Transpose;

            ImageTypeManager::instance().setFunctionTable( penguinV::Image().type(), table );
        }
    };

    const FunctionRegistrator functionRegistrator;

    void Dilate( penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint32_t dilationX, uint32_t dilationY, uint8_t value )
    {
        Image_Function::ParameterValidation( image, x, y, width, height );
        Image_Function::VerifyGrayScaleImage( image );

        if ( dilationX == 0u && dilationY == 0u )
            return;

        if( dilationX > width / 2 )
            dilationX = width / 2;
        if( dilationY > height / 2 )
            dilationY = height / 2;

        const uint32_t rowSize = image.rowSize();

        if ( dilationX > 0u ) {
            const int32_t dilateX = static_cast<int32_t>(dilationX);

            uint8_t ** startPos = new uint8_t *[2 * width];
            uint8_t ** endPos = startPos + width;

            uint8_t * imageY    = image.data() + y * rowSize + x;
            uint8_t * imageYEnd = imageY + height * rowSize;

            for( ; imageY != imageYEnd; imageY += rowSize ) {
                uint32_t pairCount = 0u;

                uint8_t previousValue = *imageY;

                uint8_t * imageXStart = imageY;
                uint8_t * imageX      = imageXStart + 1;
                uint8_t * imageXEnd   = imageXStart + width;

                for( ; imageX != imageXEnd; ++imageX ) {
                    if( (*imageX) != previousValue ) {
                        if( imageX - imageXStart < dilateX )
                            startPos[pairCount] = imageXStart;
                        else
                            startPos[pairCount] = imageX - dilateX;

                        if ( imageXEnd - imageX < dilateX ) {
                            endPos[pairCount++] = imageXEnd;
                            break;
                        }

                        endPos[pairCount++] = imageX + dilateX;
                        previousValue = 0xFFu ^ previousValue;
                    }
                }

                for( uint32_t i = 0u; i < pairCount; ++i ) {
                    imageX    = startPos[i];
                    imageXEnd = endPos[i];

                    for( ; imageX != imageXEnd; ++imageX )
                        (*imageX) = value;
                }
            }

            delete[] startPos;
        }

        if( dilationY > 0u ) {
            uint8_t ** startPos = new uint8_t *[2 * height];
            uint8_t ** endPos = startPos + height;

            uint8_t * imageX    = image.data() + y * rowSize + x;
            uint8_t * imageXEnd = imageX + width;

            for( ; imageX != imageXEnd; ++imageX ) {
                uint32_t pairCount = 0u;

                uint8_t previousValue = *imageX;

                uint8_t * imageYStart = imageX;
                uint8_t * imageY      = imageYStart + rowSize;
                uint8_t * imageYEnd   = imageYStart + height * rowSize;

                for( ; imageY != imageYEnd; imageY += rowSize ) {
                    if( (*imageY) != previousValue ) {
                        const uint32_t rowId = static_cast<uint32_t>(imageY - imageYStart) / rowSize;

                        if( rowId < dilationY )
                            startPos[pairCount] = imageYStart;
                        else
                            startPos[pairCount] = imageY - dilationY * rowSize;

                        if ( height - rowId < dilationY ) {
                            endPos[pairCount++] = imageYEnd;
                            break;
                        }

                        endPos[pairCount++] = imageY + dilationY * rowSize;
                        previousValue = 0xFFu ^ previousValue;
                    }
                }

                for( uint32_t i = 0u; i < pairCount; ++i ) {
                    imageY    = startPos[i];
                    imageYEnd = endPos[i];

                    for( ; imageY != imageYEnd; imageY += rowSize )
                        (*imageY) = value;
                }
            }

            delete[] startPos;
        }
    }
}

namespace Image_Function
{
    Image AbsoluteDifference( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::AbsoluteDifference( AbsoluteDifference, in1, in2 );
    }

    void AbsoluteDifference( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::AbsoluteDifference( AbsoluteDifference, in1, in2, out );
    }

    Image AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                              uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::AbsoluteDifference( AbsoluteDifference, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                             Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount  = CommonColorCount( in1, in2, out );
        width = width * colorCount;

        OptimiseRoi( width, height, in1, in2, out );

        const uint32_t rowSize1   = in1.rowSize();
        const uint32_t rowSize2   = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = static_cast<uint8_t>( (*in2X) > (*in1X) ? (*in2X) - (*in1X) : (*in1X) - (*in2X) );
        }
    }

    void Accumulate( const Image & image, std::vector < uint32_t > & result )
    {
        Image_Function_Helper::Accumulate( Accumulate, image, result );
    }

    void Accumulate( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & result )
    {
        ParameterValidation( image, x, y, width, height );

        const uint8_t colorCount = image.colorCount();
        width = width * colorCount;

        OptimiseRoi( width, height, image );

        if( result.size() != width * height )
            throw imageException( "Array size is not equal to image ROI (width * height) size" );

        const uint32_t rowSize = image.rowSize();

        const uint8_t * imageY    = image.data() + y * rowSize + x * colorCount;
        const uint8_t * imageYEnd = imageY + height * rowSize;
        std::vector < uint32_t >::iterator v = result.begin();

        for( ; imageY != imageYEnd; imageY += rowSize ) {
            const uint8_t * imageX    = imageY;
            const uint8_t * imageXEnd = imageX + width;

            for( ; imageX != imageXEnd; ++imageX, ++v )
                *v += (*imageX);
        }
    }

    void BinaryDilate( Image & image, uint32_t dilationX, uint32_t dilationY )
    {
        ParameterValidation( image );

        BinaryDilate( image, 0, 0, image.width(), image.height(), dilationX, dilationY );
    }

    void BinaryDilate( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint32_t dilationX, uint32_t dilationY )
    {
        Dilate( image, x, y, width, height, dilationX, dilationY, 255u );
    }

    void BinaryErode( Image & image, uint32_t erosionX, uint32_t erosionY )
    {
        ParameterValidation( image );

        BinaryErode( image, 0, 0, image.width(), image.height(), erosionX, erosionY );
    }

    void BinaryErode( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint32_t erosionX, uint32_t erosionY )
    {
        Dilate( image, x, y, width, height, erosionX, erosionY, 0u );
    }

    Image BitwiseAnd( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::BitwiseAnd( BitwiseAnd, in1, in2 );
    }

    void BitwiseAnd( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::BitwiseAnd( BitwiseAnd, in1, in2, out );
    }

    Image BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseAnd( BitwiseAnd, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount  = CommonColorCount( in1, in2, out );
        width = width * colorCount;

        OptimiseRoi( width, height, in1, in2, out );

        const uint32_t rowSize1   = in1.rowSize();
        const uint32_t rowSize2   = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = (*in1X) & (*in2X);
        }
    }

    Image BitwiseOr( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::BitwiseOr( BitwiseOr, in1, in2 );
    }

    void BitwiseOr( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::BitwiseOr( BitwiseOr, in1, in2, out );
    }

    Image BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseOr( BitwiseOr, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount  = CommonColorCount( in1, in2, out );
        width = width * colorCount;

        OptimiseRoi( width, height, in1, in2, out );

        const uint32_t rowSize1   = in1.rowSize();
        const uint32_t rowSize2   = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = (*in1X) | (*in2X);
        }
    }

    Image BitwiseXor( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::BitwiseXor( BitwiseXor, in1, in2 );
    }

    void BitwiseXor( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::BitwiseXor( BitwiseXor, in1, in2, out );
    }

    Image BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                      uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::BitwiseXor( BitwiseXor, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount  = CommonColorCount( in1, in2, out );
        width = width * colorCount;

        OptimiseRoi( width, height, in1, in2, out );

        const uint32_t rowSize1   = in1.rowSize();
        const uint32_t rowSize2   = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = (*in1X) ^ (*in2X);
        }
    }

    Image16Bit ConvertTo16Bit( const Image & in )
    {
        Image16Bit out = Image16Bit().generate( in.width(), in.height(), in.colorCount() );
        ConvertTo16Bit( in, 0, 0, out, 0, 0, in.width(), in.height() );

        return out;
    }

    void ConvertTo16Bit( const Image & in, Image16Bit & out )
    {
        ParameterValidation( in );
        ParameterValidation( out );

        ConvertTo16Bit( in, 0, 0, out, 0, 0, in.width(), in.height() );
    }

    Image16Bit ConvertTo16Bit( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, width, height );

        Image16Bit out = Image16Bit().generate( width, height, in.colorCount() );
        ConvertTo16Bit( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    void ConvertTo16Bit( const Image & in, uint32_t startXIn, uint32_t startYIn, Image16Bit & out, uint32_t startXOut, uint32_t startYOut,
                         uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, width, height );
        ParameterValidation( out, startXOut, startYOut, width, height );
        if ( in.colorCount() != out.colorCount() )
            throw imageException( "Color counts of images are different" );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t colorCount = in.colorCount();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint16_t      * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint16_t * outYEnd = outY + height * rowSizeOut;

        width = width * colorCount;

        for ( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t  * inX  = inY;
            uint16_t       * outX = outY;
            const uint16_t * outXEnd = outX + width;

            for ( ; outX != outXEnd; ++outX, ++inX )
                *outX = static_cast<uint16_t>( (*inX) << 8 );
        }
    }

    Image ConvertTo8Bit( const Image16Bit & in )
    {
        Image out = Image().generate( in.width(), in.height(), in.colorCount() );
        ConvertTo8Bit( in, 0, 0, out, 0, 0, in.width(), in.height() );

        return out;
    }

    void ConvertTo8Bit( const Image16Bit & in, Image & out )
    {
        ParameterValidation( in );
        ParameterValidation( out );

        ConvertTo8Bit( in, 0, 0, out, 0, 0, in.width(), in.height() );
    }

    Image ConvertTo8Bit( const Image16Bit & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, width, height );

        Image out = Image().generate( width, height, in.colorCount() );
        ConvertTo8Bit( in, startXIn, startYIn, out, 0, 0, width, height );

        return out;
    }

    void ConvertTo8Bit( const Image16Bit & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                        uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, width, height );
        ParameterValidation( out, startXOut, startYOut, width, height );
        if ( in.colorCount() != out.colorCount() )
            throw imageException( "Color counts of images are different" );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t colorCount = in.colorCount();

        const uint16_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t      * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        width = width * colorCount;

        for ( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint16_t * inX  = inY;
            uint8_t        * outX = outY;
            const uint8_t  * outXEnd = outX + width;

            for ( ; outX != outXEnd; ++outX, ++inX )
                *outX = static_cast<uint8_t>( (*inX) >> 8 );
        }
    }

    Image ConvertToGrayScale( const Image & in )
    {
        return Image_Function_Helper::ConvertToGrayScale( ConvertToGrayScale, in );
    }

    void ConvertToGrayScale( const Image & in, Image & out )
    {
        Image_Function_Helper::ConvertToGrayScale( ConvertToGrayScale, in, out );
    }

    Image ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::ConvertToGrayScale( ConvertToGrayScale, in, startXIn, startYIn, width, height );
    }

    void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                             uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyGrayScaleImage( out );

        if( in.colorCount() == GRAY_SCALE ) {
            Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            return;
        }

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t colorCount = RGB;

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, inX += colorCount )
                (*outX) = static_cast <uint8_t>( ( *(inX) + *(inX + 1) + *(inX + 2) ) / 3 ); // average of red, green and blue components
        }
    }

    Image ConvertToRgb( const Image & in )
    {
        return Image_Function_Helper::ConvertToRgb( ConvertToRgb, in );
    }

    void ConvertToRgb( const Image & in, Image & out )
    {
        Image_Function_Helper::ConvertToRgb( ConvertToRgb, in, out );
    }

    Image ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::ConvertToRgb( ConvertToRgb, in, startXIn, startYIn, width, height );
    }

    void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyRGBImage     ( out );

        if( in.colorCount() == RGB ) {
            Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            return;
        }

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t colorCount = RGB;

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        width = width * colorCount;

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; outX += colorCount, ++inX )
                memset( outX, (*inX), sizeof( uint8_t ) * colorCount );
        }
    }

    void Copy( const Image & in, Image & out )
    {
        ParameterValidation( in, out );

        out = in;
    }

    Image Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Copy( Copy, in, startXIn, startYIn, width, height );
    }

    void Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
               uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

        const uint8_t colorCount  = CommonColorCount( in, out );
        width = width * colorCount;

        OptimiseRoi( width, height, in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn )
            memcpy( outY, inY, sizeof( uint8_t ) * width );
    }

    Image ExtractChannel( const Image & in, uint8_t channelId )
    {
        return Image_Function_Helper::ExtractChannel( ExtractChannel, in, channelId );
    }

    void ExtractChannel( const Image & in, Image & out, uint8_t channelId )
    {
        Image_Function_Helper::ExtractChannel( ExtractChannel, in, out, channelId );
    }

    Image ExtractChannel( const Image & in, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t channelId )
    {
        return Image_Function_Helper::ExtractChannel( ExtractChannel, in, x, y, width, height, channelId );
    }

    void ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
                         uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyGrayScaleImage( out );

        if( channelId >= in.colorCount() )
            throw imageException( "Channel ID for color image is greater than channel count in input image" );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t colorCount = in.colorCount();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn * colorCount + channelId;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, inX += colorCount )
                (*outX) = *(inX);
        }
    }

    void Fill( Image & image, uint8_t value )
    {
        image.fill( value );
    }

    void Fill( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value )
    {
        ParameterValidation( image, x, y, width, height );

        const uint8_t colorCount = image.colorCount();
        width = width * colorCount;

        OptimiseRoi( width, height, image );

        const uint32_t rowSize = image.rowSize();

        uint8_t * imageY = image.data() + y * rowSize + x * colorCount;
        const uint8_t * imageYEnd = imageY + height * rowSize;

        for( ; imageY != imageYEnd; imageY += rowSize )
            memset( imageY, value, sizeof( uint8_t ) * width );
    }

    Image Flip( const Image & in, bool horizontal, bool vertical )
    {
        return Image_Function_Helper::Flip( Flip, in, horizontal, vertical );
    }

    void Flip( const Image & in, Image & out, bool horizontal, bool vertical )
    {
        Image_Function_Helper::Flip( Flip, in, out, horizontal, vertical );
    }

    Image Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                bool horizontal, bool vertical )
    {
        return Image_Function_Helper::Flip( Flip, in, startXIn, startYIn, width, height, horizontal, vertical );
    }

    void Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
               uint32_t width, uint32_t height, bool horizontal, bool vertical )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyGrayScaleImage( in, out );

        if( !horizontal && !vertical ) {
            Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        }
        else {
            const uint32_t rowSizeIn  = in.rowSize();
            const uint32_t rowSizeOut = out.rowSize();

            const uint8_t * inY    = in.data() + startYIn * rowSizeIn + startXIn;
            const uint8_t * inYEnd = inY + height * rowSizeIn;

            if( horizontal && !vertical ) {
                uint8_t * outY = out.data() + startYOut * rowSizeOut + startXOut + width - 1;

                for( ; inY != inYEnd; inY += rowSizeIn, outY += rowSizeOut ) {
                    const uint8_t * inX    = inY;
                    uint8_t       * outX   = outY;
                    const uint8_t * inXEnd = inX + width;

                    for( ; inX != inXEnd; ++inX, --outX )
                        (*outX) = (*inX);
                }
            }
            else if( !horizontal && vertical ) {
                uint8_t * outY = out.data() + (startYOut + height - 1) * rowSizeOut + startXOut;

                for( ; inY != inYEnd; inY += rowSizeIn, outY -= rowSizeOut )
                    memcpy( outY, inY, sizeof( uint8_t ) * width );
            }
            else {
                uint8_t * outY = out.data() + (startYOut + height - 1) * rowSizeOut + startXOut + width - 1;

                for( ; inY != inYEnd; inY += rowSizeIn, outY -= rowSizeOut ) {
                    const uint8_t * inX    = inY;
                    uint8_t       * outX   = outY;
                    const uint8_t * inXEnd = inX + width;

                    for( ; inX != inXEnd; ++inX, --outX )
                        (*outX) = (*inX);
                }
            }
        }
    }

    Image GammaCorrection( const Image & in, double a, double gamma )
    {
        return Image_Function_Helper::GammaCorrection( GammaCorrection, in, a, gamma );
    }

    void GammaCorrection( const Image & in, Image & out, double a, double gamma )
    {
        Image_Function_Helper::GammaCorrection( GammaCorrection, in, out, a, gamma );
    }

    Image GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double a, double gamma )
    {
        return Image_Function_Helper::GammaCorrection( GammaCorrection, in, startXIn, startYIn, width, height, a, gamma );
    }

    void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                          uint32_t width, uint32_t height, double a, double gamma )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

        const std::vector<uint8_t> & value = Image_Function_Helper::GetGammaCorrectionLookupTable( a, gamma );

        LookupTable( in, startXIn, startYIn, out, startXOut, startYOut, width, height, value );
    }

    uint8_t GetPixel( const Image & image, uint32_t x, uint32_t y )
    {
        if( image.empty() || x >= image.width() || y >= image.height() || image.colorCount() != GRAY_SCALE )
            throw imageException( "Position of point [x, y] is out of image" );

        return *(image.data() + y * image.rowSize() + x);
    }

    uint8_t GetThreshold( const std::vector < uint32_t > & histogram )
    {
        return Image_Function_Helper::GetThreshold( histogram );
    }

    std::vector < uint32_t > Histogram( const Image & image )
    {
        return Image_Function_Helper::Histogram( Histogram, image );
    }

    void Histogram( const Image & image, std::vector < uint32_t > & histogram )
    {
        Image_Function_Helper::Histogram( Histogram, image, histogram );
    }

    std::vector < uint32_t > Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Histogram( Histogram, image, x, y, width, height );
    }

    void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & histogram )
    {
        ParameterValidation( image, x, y, width, height );
        OptimiseRoi( width, height, image );

        const uint8_t colorCount = image.colorCount();

        histogram.resize( 256u * colorCount );
        std::fill( histogram.begin(), histogram.end(), 0u );

        const uint32_t rowSize = image.rowSize();

        const uint8_t * imageY = image.data() + y * rowSize + x * colorCount;
        const uint8_t * imageYEnd = imageY + height * rowSize;

        if ( colorCount == 1 ) {
            for ( ; imageY != imageYEnd; imageY += rowSize ) {
                const uint8_t * imageX = imageY;
                const uint8_t * imageXEnd = imageX + width;

                for ( ; imageX != imageXEnd; ++imageX )
                    ++histogram[*imageX];
            }
        }
        else {
            for ( ; imageY != imageYEnd; imageY += rowSize ) {
                const uint8_t * imageX = imageY;
                const uint8_t * imageXEnd = imageX + width;

                for ( ; imageX != imageXEnd; imageX += colorCount ) {
                    for ( uint8_t colorChannel = 0; colorChannel < colorCount; ++colorChannel )
                        ++histogram[*( imageX + colorChannel ) * colorCount + colorChannel];
                }
            }
        }
    }

    std::vector < uint32_t > Histogram( const Image & image, const Image & mask )
    {
        return Image_Function_Helper::Histogram( Histogram, image, mask );
    }

    void Histogram( const Image & image, const Image & mask, std::vector < uint32_t > & histogram )
    {
        Image_Function_Helper::Histogram( Histogram, image, mask, histogram );
    }

    std::vector < uint32_t > Histogram( const Image & image, uint32_t x, uint32_t y, const Image & mask, uint32_t maskX, uint32_t maskY,
                                        uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Histogram( Histogram, image, x, y, mask, maskX, maskY, width, height );
    }

    void Histogram( const Image & image, uint32_t x, uint32_t y, const Image & mask, uint32_t maskX, uint32_t maskY, uint32_t width, uint32_t height,
                    std::vector < uint32_t > & histogram )
    {
        ParameterValidation( image, x, y, mask, maskX, maskY, width, height );
        OptimiseRoi( width, height, image, mask );

        const uint8_t colorCount = image.colorCount();

        histogram.resize( 256u * colorCount );
        std::fill( histogram.begin(), histogram.end(), 0u );

        const uint32_t rowSize     = image.rowSize();
        const uint32_t rowSizeMask = mask.rowSize();

        const uint8_t * imageY = image.data() + y * rowSize + x * colorCount;
        const uint8_t * imageYMask = mask.data() + maskY * rowSizeMask + maskX;
        const uint8_t * imageYEnd  = imageY + height * rowSize;

        if ( colorCount == 1 ) {
            for ( ; imageY != imageYEnd; imageY += rowSize, imageYMask += rowSizeMask ) {
                const uint8_t * imageX = imageY;
                const uint8_t * imageXMask = imageYMask;
                const uint8_t * imageXEnd = imageX + width;

                for ( ; imageX != imageXEnd; ++imageX, ++imageXMask ) {
                    if ( ( *imageXMask ) > 0 )
                        ++histogram[*imageX];
                }
            }
        }
        else {
            for ( ; imageY != imageYEnd; imageY += rowSize, imageYMask += rowSizeMask ) {
                const uint8_t * imageX = imageY;
                const uint8_t * imageXMask = imageYMask;
                const uint8_t * imageXEnd = imageX + width;

                for ( ; imageX != imageXEnd; imageX += colorCount, ++imageXMask ) {
                    if ( ( *imageXMask ) > 0 ) {
                        for ( uint8_t colorChannel = 0; colorChannel < colorCount; ++colorChannel )
                            ++histogram[*( imageX + colorChannel ) * colorCount + colorChannel];
                    }
                }
            }
        }
    }

    Image Invert( const Image & in )
    {
        return Image_Function_Helper::Invert( Invert, in );
    }

    void Invert( const Image & in, Image & out )
    {
        Image_Function_Helper::Invert( Invert, in, out );
    }

    Image Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Invert( Invert, in, startXIn, startYIn, width, height );
    }

    void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                 uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

        const uint8_t colorCount  = CommonColorCount( in, out );
        width = width * colorCount;

        OptimiseRoi( width, height, in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++inX )
                (*outX) = static_cast<uint8_t>( ~(*inX) );
        }
    }

    bool IsBinary( const Image & image )
    {
        ParameterValidation( image );

        return IsBinary( image, 0u, 0u, image.width(), image.height() );
    }

    bool IsBinary( const Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height )
    {
        ParameterValidation( image, startX, startY, width, height );
        VerifyGrayScaleImage( image );

        const std::vector< uint32_t > histogram = Histogram( image, startX, startY, width, height );

        size_t counter = 0u;

        for( std::vector< uint32_t >::const_iterator value = histogram.begin(); value != histogram.end(); ++value ) {
            if( (*value) > 0u )
                ++counter;
        }

        return (counter < 3u);
    }

    bool IsEqual( const Image & in1, const Image & in2 )
    {
        ParameterValidation( in1, in2 );

        return IsEqual( in1, 0, 0, in2, 0, 0, in1.width(), in1.height() );
    }

    bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  uint32_t width, uint32_t height )
    {
        ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        const uint8_t colorCount = CommonColorCount( in1, in2 );
        width = width * colorCount;

        OptimiseRoi( width, height, in1, in2 );

        const uint32_t rowSize1  = in1.rowSize();
        const uint32_t rowSize2  = in2.rowSize();

        const uint8_t * in1Y = in1.data() + startY1 * rowSize1 + startX1 * colorCount;
        const uint8_t * in2Y = in2.data() + startY2 * rowSize2 + startX2 * colorCount;

        const uint8_t * in1YEnd = in1Y + height * rowSize1;

        for( ; in1Y != in1YEnd; in1Y += rowSize1, in2Y += rowSize2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            const uint8_t * in1XEnd = in1X + width;

            for( ; in1X != in1XEnd; ++in1X, ++in2X ) {
                if( (*in1X) != (*in2X) )
                    return false;
            }
        }

        return true;
    }

    Image LookupTable( const Image & in, const std::vector < uint8_t > & table )
    {
        return Image_Function_Helper::LookupTable( LookupTable, in, table );
    }

    void LookupTable( const Image & in, Image & out, const std::vector < uint8_t > & table )
    {
        Image_Function_Helper::LookupTable( LookupTable, in, out, table );
    }

    Image LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
                       const std::vector < uint8_t > & table )
    {
        return Image_Function_Helper::LookupTable( LookupTable, in, startXIn, startYIn, width, height, table );
    }

    void LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                      uint32_t width, uint32_t height, const std::vector < uint8_t > & table )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

        if( table.size() != 256u )
            throw imageException( "Lookup table size is not equal to 256" );

        const uint8_t colorCount = CommonColorCount( in, out );
        width = width * colorCount;

        OptimiseRoi( width, height, in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++inX )
                (*outX) = table[*inX];
        }
    }

    Image Maximum( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::Maximum( Maximum, in1, in2 );
    }

    void Maximum( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::Maximum( Maximum, in1, in2, out );
    }

    Image Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Maximum( Maximum, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount  = CommonColorCount( in1, in2, out );
        width = width * colorCount;

        OptimiseRoi( width, height, in1, in2, out );

        const uint32_t rowSize1   = in1.rowSize();
        const uint32_t rowSize2   = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = (*in2X) < (*in1X) ? (*in1X) : (*in2X);
        }
    }

    Image Merge( const Image & in1, const Image & in2, const Image & in3 )
    {
        return Image_Function_Helper::Merge( Merge, in1, in2, in3 );
    }

    void Merge( const Image & in1, const Image & in2, const Image & in3, Image & out )
    {
        Image_Function_Helper::Merge( Merge, in1, in2, in3, out );
    }

    Image Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                 const Image & in3, uint32_t startXIn3, uint32_t startYIn3, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Merge( Merge, in1, startXIn1, startYIn1, in2, startXIn2, startYIn2,
                                             in3, startXIn3, startYIn3, width, height  );
    }

    void Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                const Image & in3, uint32_t startXIn3, uint32_t startYIn3, Image & out, uint32_t startXOut, uint32_t startYOut,
                uint32_t width, uint32_t height )
    {
        ParameterValidation ( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3, width, height );
        ParameterValidation ( out, startXOut, startYOut, width, height );
        VerifyGrayScaleImage( in1, in2, in3 );
        VerifyRGBImage      ( out );

        const uint8_t colorCount = RGB;

        if( colorCount != out.colorCount() )
            throw imageException( "Color image is not 3-colored image" );

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeIn3 = in3.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        width = width * colorCount;

        const uint8_t * in1Y = in1.data() + startYIn1 * rowSizeIn1 + startXIn1;
        const uint8_t * in2Y = in2.data() + startYIn2 * rowSizeIn2 + startXIn2;
        const uint8_t * in3Y = in3.data() + startYIn3 * rowSizeIn3 + startXIn3;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2, in3Y += rowSizeIn3 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            const uint8_t * in3X = in3Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ) {
                *(outX++) = *(in1X++);
                *(outX++) = *(in2X++);
                *(outX++) = *(in3X++);
            }
        }
    }

    Image Minimum( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::Minimum( Minimum, in1, in2 );
    }

    void Minimum( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::Minimum( Minimum, in1, in2, out );
    }

    Image Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Minimum( Minimum, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount  = CommonColorCount( in1, in2, out );
        width = width * colorCount;

        OptimiseRoi( width, height, in1, in2, out );

        const uint32_t rowSize1   = in1.rowSize();
        const uint32_t rowSize2   = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = (*in2X) > (*in1X) ? (*in1X) : (*in2X);
        }
    }

    Image Normalize( const Image & in )
    {
        return Image_Function_Helper::Normalize( Normalize, in );
    }

    void Normalize( const Image & in, Image & out )
    {
        Image_Function_Helper::Normalize( Normalize, in, out );
    }

    Image Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Normalize( Normalize, in, startXIn, startYIn, width, height );
    }

    void Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

        const uint8_t colorCount = CommonColorCount( in, out );
        const uint32_t rowSizeIn = in.rowSize();

        const uint8_t * inY    = in.data()  + startYIn  * rowSizeIn + startXIn * colorCount;
        const uint8_t * inYEnd = inY + height * rowSizeIn;

        uint8_t minimum = 255;
        uint8_t maximum = 0;

        const uint32_t realWidth = width * colorCount;

        for( ; inY != inYEnd; inY += rowSizeIn ) {
            const uint8_t * inX = inY;
            const uint8_t * inXEnd = inX + realWidth;

            for( ; inX != inXEnd; ++inX ) {
                if( minimum > (*inX) )
                    minimum = (*inX);

                if( maximum < (*inX) )
                    maximum = (*inX);
            }
        }

        if( (minimum == 0 && maximum == 255) || (minimum == maximum) ) {
            Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        }
        else {
            const double correction = 255.0 / (maximum - minimum);

            // We precalculate all values and store them in lookup table
            std::vector < uint8_t > value( 256 );

            for( uint16_t i = 0; i < 256; ++i )
                value[i] = static_cast <uint8_t>((i - minimum) * correction + 0.5);

            LookupTable( in, startXIn, startYIn, out, startXOut, startYOut, width, height, value );
        }
    }

    std::vector < uint32_t > ProjectionProfile( const Image & image, bool horizontal )
    {
        return Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, horizontal );
    }

    void ProjectionProfile( const Image & image, bool horizontal, std::vector < uint32_t > & projection )
    {
        ProjectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );
    }

    std::vector < uint32_t > ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal )
    {
        return Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, x, y, width, height, horizontal );
    }

    void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                            std::vector < uint32_t > & projection )
    {
        ParameterValidation( image, x, y, width, height );

        const uint8_t colorCount = image.colorCount();

        projection.resize( horizontal ? width * colorCount : height );
        std::fill( projection.begin(), projection.end(), 0u );

        const uint32_t rowSize = image.rowSize();

        width = width * colorCount;

        if( horizontal ) {
            const uint8_t * imageX = image.data() + y * rowSize + x * colorCount;
            const uint8_t * imageXEnd = imageX + width;

            std::vector < uint32_t > ::iterator data = projection.begin();

            for( ; imageX != imageXEnd; ++imageX, ++data ) {
                const uint8_t * imageY    = imageX;
                const uint8_t * imageYEnd = imageY + height * rowSize;

                for( ; imageY != imageYEnd; imageY += rowSize )
                    (*data) += (*imageY);
            }
        }
        else {
            const uint8_t * imageY = image.data() + y * rowSize + x * colorCount;
            const uint8_t * imageYEnd = imageY + height * rowSize;

            std::vector < uint32_t > ::iterator data = projection.begin();

            for( ; imageY != imageYEnd; imageY += rowSize, ++data ) {
                const uint8_t * imageX    = imageY;
                const uint8_t * imageXEnd = imageX + width;

                for( ; imageX != imageXEnd; ++imageX )
                    (*data) += (*imageX);
            }
        }
    }

    void ReplaceChannel( const Image & channel, Image & rgb, uint8_t channelId )
    {
        ParameterValidation( channel, rgb );

        ReplaceChannel( channel, 0, 0, rgb, 0, 0, channel.width(), channel.height(), channelId );
    }

    void ReplaceChannel( const Image & channel, uint32_t startXChannel, uint32_t startYChannel, Image & rgb, uint32_t startXRgb, uint32_t startYRgb,
                         uint32_t width, uint32_t height, uint8_t channelId )
    {
        ParameterValidation( channel, startXChannel, startYChannel, rgb, startXRgb, startYRgb, width, height );
        VerifyGrayScaleImage( channel );
        VerifyRGBImage( rgb );

        if ( channelId >= RGB )
            throw imageException( "Channel ID is greater than number of channels in RGB image" );

        const uint32_t rowSizeIn  = channel.rowSize();
        const uint32_t rowSizeOut = rgb.rowSize();

        const uint8_t colorCount = RGB;

        const uint8_t * inY  = channel.data() + startYChannel * rowSizeIn  + startXChannel;
        uint8_t       * outY = rgb.data() + startYRgb * rowSizeOut + startXRgb * colorCount + channelId;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        width = width * colorCount;

        for ( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for ( ; outX != outXEnd; outX += colorCount, ++inX )
                *outX = *inX;
        }
    }

    Image Resize( const Image & in, uint32_t widthOut, uint32_t heightOut )
    {
        return Image_Function_Helper::Resize( Resize, in, widthOut, heightOut );
    }

    void Resize( const Image & in, Image & out )
    {
        Image_Function_Helper::Resize( Resize, in, out );
    }

    Image Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                  uint32_t widthOut, uint32_t heightOut )
    {
        return Image_Function_Helper::Resize( Resize, in, startXIn, startYIn, widthIn, heightIn, widthOut, heightOut );
    }

    void Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
                 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut )
    {
        ParameterValidation( in, startXIn, startYIn, widthIn, heightIn );
        ParameterValidation( out, startXOut, startYOut, widthOut, heightOut );
        VerifyGrayScaleImage( in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + heightOut * rowSizeOut;

        uint32_t idY = 0;

        // Precalculation of X position
        std::vector < uint32_t > positionX( widthOut );
        for( uint32_t x = 0; x < widthOut; ++x )
            positionX[x] = x * widthIn / widthOut;

        for( ; outY != outYEnd; outY += rowSizeOut, ++idY ) {
            const uint8_t * inX  = inY + (idY * heightIn / heightOut) * rowSizeIn;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + widthOut;

            const uint32_t * idX = positionX.data();

            for( ; outX != outXEnd; ++outX, ++idX )
                (*outX) = *(inX + (*idX));
        }
    }

    Image RgbToBgr( const Image & in )
    {
        return Image_Function_Helper::RgbToBgr( RgbToBgr, in );
    }

    void RgbToBgr( const Image & in, Image & out )
    {
        Image_Function_Helper::RgbToBgr( RgbToBgr, in, out );
    }

    Image RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::RgbToBgr( RgbToBgr, in, startXIn, startYIn, width, height );
    }

    void RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                   uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyRGBImage     ( in, out );

        const uint8_t colorCount = RGB;
        width = width * colorCount;

        OptimiseRoi( width, height, in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; outX += colorCount, inX += colorCount ) {
                *(outX + 2) = *(inX);
                *(outX + 1) = *(inX + 1);
                *(outX) = *(inX + 2);
            }
        }
    }

    Image RgbToRgba( const Image & in )
    {
        return Image_Function_Helper::RgbToRgba( RgbToRgba, in );
    }

    void RgbToRgba( const Image & in, Image & out )
    {
        Image_Function_Helper::RgbToRgba( RgbToRgba, in, out );
    }

    Image RgbToRgba( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::RgbToRgba( RgbToRgba, in, startXIn, startYIn, width, height );
    }

    void RgbToRgba( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyRGBImage     ( in );
        VerifyRGBAImage    ( out );

        const uint8_t colorCountIn = RGB;
        const uint8_t colorCountOut = RGBA;
        width = width * colorCountOut;

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCountIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCountOut;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; outX += colorCountOut, inX += colorCountIn ) {
                *(outX) = *(inX);
                *(outX + 1) = *(inX + 1);
                *(outX + 2) = *(inX + 2);
                *(outX + 3) = 255u;
            }
        }
    }

    Image RgbaToRgb( const Image & in )
    {
        return Image_Function_Helper::RgbaToRgb( RgbaToRgb, in );
    }

    void RgbaToRgb( const Image & in, Image & out )
    {
        Image_Function_Helper::RgbaToRgb( RgbaToRgb, in, out );
    }

    Image RgbaToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::RgbaToRgb( RgbaToRgb, in, startXIn, startYIn, width, height );
    }

    void RgbaToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyRGBAImage    ( in );
        VerifyRGBImage     ( out );

        const uint8_t colorCountIn = RGBA;
        const uint8_t colorCountOut = RGB;
        width = width * colorCountOut;

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCountIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCountOut;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; outX += colorCountOut, inX += colorCountIn ) {
                *(outX) = *(inX);
                *(outX + 1) = *(inX + 1);
                *(outX + 2) = *(inX + 2);
            }
        }
    }

    Image Rotate( const Image & in, double centerX, double centerY, double angle )
    {
        return Image_Function_Helper::Rotate( Rotate, in, centerX, centerY, angle );
    }

    void Rotate( const Image & in, double centerXIn, double centerYIn, Image & out, double centerXOut, double centerYOut, double angle )
    {
        Image_Function_Helper::Rotate( Rotate, in, centerXIn, centerYIn, out, centerXOut, centerYOut, angle );
    }

    Image Rotate( const Image & in, uint32_t x, uint32_t y, double centerX, double centerY, uint32_t width, uint32_t height, double angle )
    {
        return Image_Function_Helper::Rotate( Rotate, in, x, y, centerX, centerY, width, height, angle );
    }

    void Rotate( const Image & in, uint32_t startXIn, uint32_t startYIn, double centerXIn, double centerYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                 double centerXOut, double centerYOut, uint32_t width, uint32_t height, double angle )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyGrayScaleImage( in, out );

        const double cosAngle = cos( angle );
        const double sinAngle = sin( angle );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data();
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;
        const uint8_t * outYEnd = outY + height * rowSizeOut;

        double inXPos = -(cosAngle * centerXOut + sinAngle * centerYOut) + centerXIn;
        double inYPos = -(-sinAngle * centerXOut + cosAngle * centerYOut) + centerYIn;

        const double minX = startXIn;
        const double minY = startYIn;
        const uint32_t maxX = startXIn + width - 1u;
        const uint32_t maxY = startYIn + height - 1u;

        for( ; outY != outYEnd; outY += rowSizeOut, inXPos += sinAngle, inYPos += cosAngle ) {
            uint8_t       * outX = outY;
            const uint8_t * outXEnd = outX + width;

            double posX = inXPos;
            double posY = inYPos;

            for( ; outX != outXEnd; ++outX, posX += cosAngle, posY -= sinAngle ) {
                if( posX < minX || posY < minY ) {
                    (*outX) = 0; // we actually do not know what is beyond an image so we set value 0
                }
                else {
                    const uint32_t x = static_cast<uint32_t>(posX);
                    const uint32_t y = static_cast<uint32_t>(posY);

                    if( x >= maxX || y >= maxY ) {
                        (*outX) = 0; // we actually do not know what is beyond an image so we set value 0
                    }
                    else {
                        const uint8_t * inX = inY + y * rowSizeIn + x;

                        // we use bilinear approximation to find pixel intensity value
                        const double coeffX = posX - x;
                        const double coeffY = posY - y;

                        // Take a weighted mean of four pixels. Use offset of 0.5
                        // so that integer conversion leads to rounding instead of 
                        // simple truncation.
                        const double sum = *(inX) * (1 - coeffX) * (1 - coeffY) + *(inX + 1) * (coeffX) * (1 - coeffY) +
                                           *(inX + rowSizeIn) * (1 - coeffX) * (coeffY) + *(inX + rowSizeIn + 1) * (coeffX) * (coeffY) + 0.5;

                        (*outX) = static_cast<uint8_t>(sum);
                    }
                }
            }
        }
    }

    void SetPixel( Image & image, uint32_t x, uint32_t y, uint8_t value )
    {
        if( image.empty() || x >= image.width() || y >= image.height() || image.colorCount() != GRAY_SCALE )
            throw imageException( "Position of point [x, y] is out of image" );

        *(image.data() + y * image.rowSize() + x) = value;
    }

    void SetPixel( Image & image, const std::vector < uint32_t > & X, const std::vector < uint32_t > & Y, uint8_t value )
    {
        if( image.empty() || X.empty() || X.size() != Y.size() || image.colorCount() != GRAY_SCALE )
            throw imageException( "Bad input parameters in image function" );

        const uint32_t rowSize = image.rowSize();
        uint8_t * data = image.data();

        std::vector < uint32_t >::const_iterator x   = X.begin();
        std::vector < uint32_t >::const_iterator y   = Y.begin();
        std::vector < uint32_t >::const_iterator end = X.end();

        const uint32_t width  = image.width();
        const uint32_t height = image.height();

        for( ; x != end; ++x, ++y ) {
            if( (*x) >= width || (*y) >= height )
                throw imageException( "Position of point [x, y] is out of image" );

            *(data + (*y) * rowSize + (*x)) = value;
        }
    }

    Image Shift( const Image & in, double shiftX, double shiftY )
    {
        return Image_Function_Helper::Shift( Shift, in, shiftX, shiftY );
    }

    void Shift( const Image & in, Image & out, double shiftX, double shiftY )
    {
        Image_Function_Helper::Shift( Shift, in, out, shiftX, shiftY );
    }

    Image Shift( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double shiftX, double shiftY )
    {
        return Image_Function_Helper::Shift( Shift, in, startXIn, startYIn, width, height, shiftX, shiftY );
    }

    void Shift( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                uint32_t width, uint32_t height, double shiftX, double shiftY )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyGrayScaleImage( in, out );

        if ( (fabs(shiftX) > width - 1) || (fabs(shiftY) > height - 1) )
            throw imageException("Shift value by value bigger than ROI");

        // take a note that we use an opposite values
        int32_t shiftXIntegral = -static_cast<int32_t>( shiftX );
        int32_t shiftYIntegral = -static_cast<int32_t>( shiftY );

        shiftX = shiftX + shiftXIntegral;
        shiftY = shiftY + shiftYIntegral;

        if ( shiftX > 0.0 )
            --shiftXIntegral;

        if ( shiftY > 0.0 )
            --shiftYIntegral;

        const double coeffX = ( shiftX < 0.0 ) ? (1.0 + shiftX) : shiftX;
        const double coeffY = ( shiftY < 0.0 ) ? (1.0 + shiftY) : shiftY;

        // 2^23 = 8388608, 2^23 * 256 is a integer limit
        const uint32_t coeff0 = static_cast<uint32_t>((      coeffX) * (      coeffY) * 16384 + 0.5);
        const uint32_t coeff1 = static_cast<uint32_t>((1.0 - coeffX) * (      coeffY) * 16384 + 0.5);
        const uint32_t coeff2 = static_cast<uint32_t>((1.0 - coeffX) * (1.0 - coeffY) * 16384 + 0.5);
        const uint32_t coeff3 = static_cast<uint32_t>((      coeffX) * (1.0 - coeffY) * 16384 + 0.5);

        const uint32_t limitX = in.width()  - 1u; // we need 2 subsequent pixels so we cannot use last pixel
        const uint32_t limitY = in.height() - 1u;

        const uint32_t emptyLeftArea  = (shiftXIntegral < 0 && startXIn < static_cast<uint32_t>(-shiftXIntegral)) ?
                                        (static_cast<uint32_t>(-shiftXIntegral) - startXIn) : 0u;
        const uint32_t emptyRightArea = (shiftXIntegral > 0 && limitX < startXIn + width + static_cast<uint32_t>(shiftXIntegral)) ?
                                        (startXIn + width + static_cast<uint32_t>(shiftXIntegral) - limitX) : 0u;
        const uint32_t realWidth = width - emptyLeftArea - emptyRightArea;

        const uint32_t emptyTopArea    = (shiftYIntegral < 0 && startYIn < static_cast<uint32_t>(-shiftYIntegral)) ?
                                         (static_cast<uint32_t>(-shiftYIntegral) - startYIn) : 0u;
        const uint32_t emptyBottomArea = (shiftYIntegral > 0 && limitY < startYIn + height + static_cast<uint32_t>(shiftYIntegral)) ?
                                         (startYIn + height + static_cast<uint32_t>(shiftYIntegral) - limitY) : 0u;
        const uint32_t realHeight = height - emptyTopArea - emptyBottomArea;

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        inY += shiftXIntegral + shiftYIntegral * static_cast<int32_t>( rowSizeIn );

        const uint8_t * outYEnd = outY + emptyTopArea * rowSizeOut;
        for( ; outY != outYEnd; outY+=rowSizeOut )
            memset( outY, 0, width );

        inY += emptyTopArea * rowSizeIn;

        outYEnd = outY + realHeight * rowSizeOut;
        for( ; outY != outYEnd; outY+=rowSizeOut, inY+=rowSizeIn ) {
            const uint8_t * inX = inY;
            uint8_t * outX = outY;
            const uint8_t * outXEnd = outX + emptyLeftArea + realWidth;

            memset( outX, 0, emptyLeftArea );
            inX += emptyLeftArea;

            for( ; outX != outXEnd; ++outX ) {
                uint32_t data  = (*(inX))  * coeff0;
                data += (*(inX+rowSizeIn)) * coeff3;
                data += (*(++inX))         * coeff1; // here we increment inX
                data += (*(inX+rowSizeIn)) * coeff2;

                *outX = static_cast<uint8_t>((data + 8192) >> 14); // 8192 is 0.5 after calculations
            }

            memset( outX, 0, emptyRightArea );
        }

        outYEnd = outY + emptyBottomArea * rowSizeOut;
        for( ; outY != outYEnd; outY+=rowSizeOut )
            memset( outY, 0, width );
    }

    void Split( const Image & in, Image & out1, Image & out2, Image & out3 )
    {
        ParameterValidation( out1, out2, out3 );
        ParameterValidation( in, out1 );

        Split( in, 0, 0, out1, 0, 0, out2, 0, 0, out3, 0, 0, in.width(), in.height() );
    }

    void Split( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out1, uint32_t startXOut1, uint32_t startYOut1,
                Image & out2, uint32_t startXOut2, uint32_t startYOut2, Image & out3, uint32_t startXOut3, uint32_t startYOut3,
                uint32_t width, uint32_t height )
    {
        ParameterValidation ( in, startXIn, startYIn, width, height );
        ParameterValidation ( out1, startXOut1, startYOut1, out2, startXOut2, startYOut2, out3, startXOut3, startYOut3, width, height );
        VerifyRGBImage      ( in );
        VerifyGrayScaleImage( out1, out2, out3 );

        const uint8_t colorCount = RGB;

        const uint32_t rowSizeIn   = in.rowSize();
        const uint32_t rowSizeOut1 = out1.rowSize();
        const uint32_t rowSizeOut2 = out2.rowSize();
        const uint32_t rowSizeOut3 = out3.rowSize();

        width = width * colorCount;

        const uint8_t * inY = in.data() + startYIn * rowSizeIn + startXIn * colorCount;
        uint8_t * out1Y = out1.data() + startYOut1 * rowSizeOut1 + startXOut1;
        uint8_t * out2Y = out2.data() + startYOut2 * rowSizeOut2 + startXOut2;
        uint8_t * out3Y = out3.data() + startYOut3 * rowSizeOut3 + startXOut3;

        const uint8_t * inYEnd = inY + height * rowSizeIn;

        for( ; inY != inYEnd; inY += rowSizeIn, out1Y += rowSizeOut1, out2Y += rowSizeOut2, out3Y += rowSizeOut3 ) {
            const uint8_t * inX = inY;
            uint8_t * out1X = out1Y;
            uint8_t * out2X = out2Y;
            uint8_t * out3X = out3Y;

            const uint8_t * inXEnd = inX + width;

            for( ; inX != inXEnd; ) {
                *(out1X++) = *(inX++);
                *(out2X++) = *(inX++);
                *(out3X++) = *(inX++);
            }
        }
    }

    Image Subtract( const Image & in1, const Image & in2 )
    {
        return Image_Function_Helper::Subtract( Subtract, in1, in2 );
    }

    void Subtract( const Image & in1, const Image & in2, Image & out )
    {
        Image_Function_Helper::Subtract( Subtract, in1, in2, out );
    }

    Image Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Subtract( Subtract, in1, startX1, startY1, in2, startX2, startY2, width, height );
    }

    void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
    {
        ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

        const uint8_t colorCount  = CommonColorCount( in1, in2, out );
        width = width * colorCount;

        OptimiseRoi( width, height, in1, in1, out );

        const uint32_t rowSize1   = in1.rowSize();
        const uint32_t rowSize2   = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = static_cast<uint8_t>( (*in2X) > (*in1X) ? 0u : static_cast<uint32_t>(*in1X) - static_cast<uint32_t>(*in2X) );
        }
    }

    uint32_t Sum( const Image & image )
    {
        return Sum( image, 0, 0, image.width(), image.height() );
    }

    uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        ParameterValidation( image, x, y, width, height );
        VerifyGrayScaleImage( image );
        OptimiseRoi( width, height, image );

        const uint32_t rowSize = image.rowSize();

        const uint8_t * imageY    = image.data() + y * rowSize + x;
        const uint8_t * imageYEnd = imageY + height * rowSize;

        uint32_t sum = 0;

        for( ; imageY != imageYEnd; imageY += rowSize ) {
            const uint8_t * imageX    = imageY;
            const uint8_t * imageXEnd = imageX + width;

            for( ; imageX != imageXEnd; ++imageX )
                sum += (*imageX);
        }

        return sum;
    }

    Image Threshold( const Image & in, uint8_t threshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, threshold );
    }

    void Threshold( const Image & in, Image & out, uint8_t threshold )
    {
        Image_Function_Helper::Threshold( Threshold, in, out, threshold );
    }

    Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t threshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, startXIn, startYIn, width, height, threshold );
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t threshold )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyGrayScaleImage( in, out );
        OptimiseRoi( width, height, in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++inX )
                (*outX) = (*inX) < threshold ? 0 : 255;
        }
    }

    Image Threshold( const Image & in, uint8_t minThreshold, uint8_t maxThreshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, minThreshold, maxThreshold );
    }

    void Threshold( const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold )
    {
        Image_Function_Helper::Threshold( Threshold, in, out, minThreshold, maxThreshold );
    }

    Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t minThreshold,
                     uint8_t maxThreshold )
    {
        return Image_Function_Helper::Threshold( Threshold, in, startXIn, startYIn, width, height, minThreshold, maxThreshold );
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold )
    {
        ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        VerifyGrayScaleImage( in, out );
        OptimiseRoi( width, height, in, out );

        if( minThreshold > maxThreshold )
            throw imageException( "Minimum threshold value is bigger than maximum threshold value" );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++inX )
                (*outX) = (*inX) < minThreshold || (*inX) > maxThreshold ? 0 : 255;
        }
    }

    Image Transpose( const Image & in )
    {
        return Image_Function_Helper::Transpose( Transpose, in );
    }

    void Transpose( const Image & in, Image & out )
    {
        Image_Function_Helper::Transpose( Transpose, in, out );
    }

    Image Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
    {
        return Image_Function_Helper::Transpose( Transpose, in, startXIn, startYIn, width, height );
    }

    void Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height )
    {
        ParameterValidation( in, startXIn, startYIn, width, height );
        ParameterValidation( out, startXOut, startYOut, height, width );
        VerifyGrayScaleImage( in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inX  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + width * rowSizeOut;

        for( ; outY != outYEnd; outY += rowSizeOut, ++inX ) {
            const uint8_t * inY  = inX;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + height;

            for( ; outX != outXEnd; ++outX, inY += rowSizeIn )
                (*outX) = *(inY);
        }
    }
}
