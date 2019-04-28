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
            table.SetPixel           = &Image_Function::SetPixel;
            table.SetPixel2          = &Image_Function::SetPixel;
            table.Shift              = &Image_Function::Shift;
            table.Split              = &Image_Function::Split;
            table.Subtract           = &Image_Function::Subtract;
            table.Sum                = &Image_Function::Sum;
            table.Threshold          = &Image_Function::Threshold;
            table.Threshold2         = &Image_Function::Threshold;
            table.Transpose          = &Image_Function::Transpose;

            ImageTypeManager::instance().setFunctionTable( PenguinV_Image::Image().type(), table );
        }
    };

    const FunctionRegistrator functionRegistrator;

    void Dilate( PenguinV_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint32_t dilationX, uint32_t dilationY, uint8_t value )
    {
        Image_Function::ParameterValidation( image, x, y, width, height );
        Image_Function::VerifyGrayScaleImage( image );


        if( dilationX > width / 2 )
            dilationX = width / 2;
        if( dilationY > height / 2 )
            dilationY = height / 2;

        if( dilationX > 0u ) {
            const int32_t dilateX = static_cast<int32_t>(dilationX);

            uint8_t ** startPos = new uint8_t *[2 * width];
            uint8_t ** endPos = startPos + width;

            const uint32_t rowSize = image.rowSize();
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

                        if( imageXEnd - imageX < dilateX )
                            endPos[pairCount] = imageXEnd;
                        else
                            endPos[pairCount] = imageX + dilateX;

                        previousValue = 0xFFu ^ previousValue;
                        ++pairCount;
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

            const uint32_t rowSize = image.rowSize();
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

                        if( height - rowId < dilationY )
                            endPos[pairCount] = imageYEnd;
                        else
                            endPos[pairCount] = imageY + dilationY * rowSize;

                        previousValue = 0xFFu ^ previousValue;
                        ++pairCount;
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

namespace cpu
{
    using namespace PenguinV_Image;

    void AbsoluteDifference( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                             uint8_t * outY, const uint8_t * outYEnd, uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = static_cast<uint8_t>( (*in2X) > (*in1X) ? (*in2X) - (*in1X) : (*in1X) - (*in2X) );
        }
    }

    void Accumulate( uint32_t rowSize, const uint8_t * imageY, const uint8_t * imageYEnd, uint32_t * outY, std::vector<uint32_t> & result, uint32_t width )
    {
        std::vector < uint32_t >::iterator v = result.begin();

        for( ; imageY != imageYEnd; imageY += rowSize ) {
            const uint8_t * imageX    = imageY;
            const uint8_t * imageXEnd = imageX + width;

            for( ; imageX != imageXEnd; ++imageX, ++v )
                *v += (*imageX);
        }
    }

    void BitwiseAnd( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = (*in1X) & (*in2X);
        }
    }

    void BitwiseOr( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                    uint8_t * outY, const uint8_t * outYEnd, uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = (*in1X) | (*in2X);
        }
    }

    void BitwiseXor( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = (*in1X) ^ (*in2X);
        }
    }

    

    void ConvertTo16Bit( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint16_t * outY, const uint16_t * outYEnd, 
                         uint32_t width )
    {
        for ( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t  * inX  = inY;
            uint16_t       * outX = outY;
            const uint16_t * outXEnd = outX + width;

            for ( ; outX != outXEnd; ++outX, ++inX )
                *outX = (*inX) << 8;
        }
    }

    void ConvertTo8Bit( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint16_t * inY, uint8_t * outY, const uint8_t * outYEnd, 
                         uint32_t width )
    {
        for ( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint16_t * inX  = inY;
            uint8_t        * outX = outY;
            const uint8_t  * outXEnd = outX + width;

            for ( ; outX != outXEnd; ++outX, ++inX )
                *outX = (*inX) >> 8;
        }
    }

    void ConvertToGrayScale( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd,
                             const uint8_t colorCount, uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, inX += colorCount )
                (*outX) = static_cast <uint8_t>((*(inX) + *(inX + 1) + *(inX + 2)) / 3u); // average of red, green and blue components
        }
    }

    void ConvertToRgb( uint8_t * outY, const uint8_t * outYEnd, const uint8_t * inY, uint32_t rowSizeOut, uint32_t rowSizeIn,
                       const uint8_t colorCount, uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; outX += colorCount, ++inX )
                memset( outX, (*inX), sizeof( uint8_t ) * colorCount );
        }
    }

    void Copy( uint8_t * outY, const uint8_t * outYEnd, const uint8_t * inY, uint32_t rowSizeOut, uint32_t rowSizeIn, uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn )
            memcpy( outY, inY, sizeof( uint8_t ) * width );
    }

    void ExtractChannel( uint8_t * outY, const uint8_t * outYEnd, const uint8_t * inY, uint32_t rowSizeOut, uint32_t rowSizeIn,
                         const uint8_t colorCount, uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, inX += colorCount )
                (*outX) = *(inX);
        }
    }

    void Fill( uint8_t * imageY, const uint8_t * imageYEnd, const uint32_t rowSize, uint8_t value, uint32_t width )
    {
        for( ; imageY != imageYEnd; imageY += rowSize )
            memset( imageY, value, sizeof( uint8_t ) * width );
    }

    void Flip( Image_Function::Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t height, const uint32_t rowSizeIn, const uint32_t rowSizeOut, 
               const uint8_t * inY, const uint8_t * inYEnd, bool horizontal, bool vertical, const uint32_t width )
    {
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

    uint8_t GetPixel( const Image & image, uint32_t x, uint32_t y )
    {
       return *(image.data() + y * image.rowSize() + x);
    }

    void GammaCorrection( std::vector < uint8_t > & value , double a, double gamma )
    {
        for( uint16_t i = 0; i < 256; ++i ) {
            double data = a * pow( i / 255.0, gamma ) * 255 + 0.5;

            if( data < 256 )
                value[i] = static_cast<uint8_t>(data);
        }
    }

    void Histogram( uint32_t rowSize, const uint8_t * imageY, const uint8_t * imageYEnd, uint32_t width, std::vector < uint32_t > & histogram )
    {
        for( ; imageY != imageYEnd; imageY += rowSize ) {
            const uint8_t * imageX    = imageY;
            const uint8_t * imageXEnd = imageX + width;

            for( ; imageX != imageXEnd; ++imageX )
                ++histogram[*imageX];
        }
    }

    void Histogram( uint32_t rowSize, uint32_t rowSizeMask, const uint8_t * imageY, const uint8_t * imageYEnd, const uint8_t * imageYMask, 
                    uint32_t width, std::vector < uint32_t > & histogram )
    {
        for( ; imageY != imageYEnd; imageY += rowSize, imageYMask += rowSizeMask ) {
            const uint8_t * imageX     = imageY;
            const uint8_t * imageXMask = imageYMask;
            const uint8_t * imageXEnd  = imageX + width;

            for( ; imageX != imageXEnd; ++imageX, ++imageXMask ) {
                if( (*imageXMask) > 0 )
                    ++histogram[*imageX];
            }
        }
    }

    void Invert( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd,
                 uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++inX )
                (*outX) = ~(*inX);
        }
    }

    bool IsBinary( const std::vector< uint32_t > &histogram )
    {
        size_t counter = 0u;

        for( std::vector< uint32_t >::const_iterator value = histogram.begin(); value != histogram.end(); ++value ) {
            if( (*value) > 0u )
                ++counter;
        }

        return (counter < 3u);
    }

    bool IsEqual( uint32_t rowSize1, uint32_t rowSize2, const uint8_t * in1Y, const uint8_t * in2Y, const uint8_t * in1YEnd, 
                  uint32_t width )
    {
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

    void LookupTable( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd,
                      const std::vector < uint8_t > & table, uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++inX )
                (*outX) = table[*inX];
        }
    }

    void Maximum( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                  uint8_t * outY, const uint8_t * outYEnd, uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = (*in2X) < (*in1X) ? (*in1X) : (*in2X);
        }
    }

    void Merge( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeIn3, uint32_t rowSizeOut, const uint8_t * in1Y, 
                const uint8_t * in2Y, const uint8_t * in3Y, uint8_t * outY, const uint8_t * outYEnd, uint32_t width )
    {
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

    void Minimum( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                  uint8_t * outY, const uint8_t * outYEnd, uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = (*in2X) > (*in1X) ? (*in1X) : (*in2X);
        }
    }

    void Normalize( const uint32_t rowSizeIn, const uint8_t * inY, const uint8_t * inYEnd, uint8_t minimum, uint8_t maximum, uint32_t realWidth )
    {
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
    }

    void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t rowSize, uint8_t colorCount, uint32_t width, 
                            uint32_t height, bool horizontal, std::vector < uint32_t > & projection )
    {
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

    void ReplaceChannel( uint8_t * outY, const uint8_t * outYEnd, const uint8_t * inY, uint32_t rowSizeOut, uint32_t rowSizeIn,
                         const uint8_t colorCount, uint32_t width )
    {
        for ( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for ( ; outX != outXEnd; outX += colorCount, ++inX )
                *outX = *inX;
        }
    }

    void Resize( uint8_t * outY, const uint8_t * outYEnd, const uint8_t * inY, uint32_t widthIn, uint32_t rowSizeOut, uint32_t rowSizeIn,
                 uint32_t idY, uint32_t heightIn, uint32_t widthOut, uint32_t heightOut )
    {
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

    void RgbToBgr( uint8_t * outY, const uint8_t * outYEnd, const uint8_t * inY, uint32_t rowSizeOut, uint32_t rowSizeIn,
                   const uint8_t colorCount, uint32_t width )
    {
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

    void Rotate( uint8_t * outY, const uint8_t * outYEnd, const uint8_t * inY, uint32_t rowSizeOut, uint32_t rowSizeIn,
                 const double cosAngle, const double sinAngle, double inXPos, double inYPos, uint32_t height, uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, inXPos += sinAngle, inYPos += cosAngle ) {
            uint8_t       * outX = outY;
            const uint8_t * outXEnd = outX + width;

            double posX = inXPos;
            double posY = inYPos;

            for( ; outX != outXEnd; ++outX, posX += cosAngle, posY -= sinAngle ) {
                if( posX < 0 || posY < 0 ) {
                    (*outX) = 0; // we actually do not know what is beyond an image so we set value 0
                }
                else {
                    uint32_t x = static_cast<uint32_t>(posX);
                    uint32_t y = static_cast<uint32_t>(posY);

                    if( x >= width - 1 || y >= height - 1 ) {
                        (*outX) = 0; // we actually do not know what is beyond an image so we set value 0
                    }
                    else {
                        const uint8_t * inX = inY + y * rowSizeIn + x;

                        // we use bilinear approximation to find pixel intensity value

                        double coeffX = posX - x;
                        double coeffY = posY - y;

                        // Take a weighted mean of four pixels. Use offset of 0.5
                        // so that integer conversion leads to rounding instead of 
                        // simple truncation.

                        double sum = (*inX) * (1 - coeffX) * (1 - coeffY) +
                            *(inX + 1) * (coeffX) * (1 - coeffY) +
                            *(inX + rowSizeIn) * (1 - coeffX) * (coeffY) +
                            *(inX + rowSizeIn + 1) * (coeffX) * (coeffY) + 
                            0.5;

                        (*outX) = static_cast<uint8_t>(sum);
                    }
                }
            }
        }
    }

    void SetPixel( Image & image, uint32_t x, uint32_t y, uint8_t value )
    {
        *(image.data() + y * image.rowSize() + x) = value;
    }

    void SetPixel( Image & image, const std::vector < uint32_t > & X, const std::vector < uint32_t > & Y, uint8_t value, 
                   uint8_t * data, uint32_t rowSize, uint32_t height, uint32_t width )
    {
        std::vector < uint32_t >::const_iterator x   = X.begin();
        std::vector < uint32_t >::const_iterator y   = Y.begin();
        std::vector < uint32_t >::const_iterator end = X.end();

        for( ; x != end; ++x, ++y ) {
            if( (*x) >= width || (*y) >= height )
                throw imageException( "Position of point [x, y] is out of image" );

            *(data + (*y) * rowSize + (*x)) = value;
        }
    }

    void Shift( uint8_t * outY, const uint8_t * outYEnd, const uint8_t * inY, uint32_t rowSizeOut, uint32_t rowSizeIn,
                const uint32_t coeff0, const uint32_t coeff1, const uint32_t coeff2, const uint32_t coeff3,
                const uint32_t emptyTopArea, const uint32_t emptyLeftArea, const uint32_t emptyRightArea, const uint32_t emptyBottomArea,
                const uint32_t realHeight, const uint32_t realWidth, uint32_t height, uint32_t width )
    {
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

    void Split( uint32_t rowSizeOut1, uint32_t rowSizeOut2, uint32_t rowSizeOut3, uint32_t rowSizeIn, const uint8_t * inY, 
                uint8_t * out1Y, uint8_t * out2Y, uint8_t * out3Y, const uint8_t * inYEnd, uint32_t width )
    {
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

    void Subtract( uint32_t rowSize1, uint32_t rowSize2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                   uint8_t * outY, const uint8_t * outYEnd, uint32_t width )
    {
       for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
            const uint8_t * in1X = in1Y;
            const uint8_t * in2X = in2Y;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                (*outX) = static_cast<uint8_t>( (*in2X) > (*in1X) ? 0u : (*in1X) - (*in2X) );
        }
    }

    uint32_t Sum( uint32_t rowSize, const uint8_t * imageY,const uint8_t * imageYEnd, uint32_t width )
    {
        uint32_t sum = 0;

        for( ; imageY != imageYEnd; imageY += rowSize ) {
            const uint8_t * imageX    = imageY;
            const uint8_t * imageXEnd = imageX + width;

            for( ; imageX != imageXEnd; ++imageX )
                sum += (*imageX);
        }

        return sum;
    }

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t threshold,
                    uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++inX )
                (*outX) = (*inX) < threshold ? 0 : 255;
        }
    }

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t minThreshold, 
                    uint8_t maxThreshold, uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + width;

            for( ; outX != outXEnd; ++outX, ++inX )
                (*outX) = (*inX) < minThreshold || (*inX) > maxThreshold ? 0 : 255;
        }
    }

    void Transpose( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inX, uint8_t * outY, const uint8_t * outYEnd, uint8_t height,
                    uint32_t width )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, ++inX ) {
            const uint8_t * inY  = inX;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + height;

            for( ; outX != outXEnd; ++outX, inY += rowSizeIn )
                (*outX) = *(inY);
        }
    }
}

namespace avx
{
    const uint32_t simdSize = 32u;

#ifdef PENGUINV_AVX_SET
    typedef __m256i simd;

    // We are not sure that input data is aligned by 32 bytes so we use loadu() functions instead of load()

    void AbsoluteDifference( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                             uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst ) {
                simd data1 = _mm256_loadu_si256( src1 );
                simd data2 = _mm256_loadu_si256( src2 );
                _mm256_storeu_si256( dst, _mm256_sub_epi8( _mm256_max_epu8( data1, data2 ), _mm256_min_epu8( data1, data2 ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                    (*outX) = static_cast<uint8_t>( (*in2X) > (*in1X) ? (*in2X) - (*in1X) : (*in1X) - (*in2X) );
            }
        }
    }

    void Accumulate( uint32_t rowSize, const uint8_t * imageY, const uint8_t * imageYEnd, uint32_t * outY, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        simd zero = _mm256_setzero_si256();

        const uint32_t width = totalSimdWidth + nonSimdWidth;

        for( ; imageY != imageYEnd; imageY += rowSize, outY += width ) {
            const simd * src    = reinterpret_cast <const simd*> (imageY);
            const simd * srcEnd = src + simdWidth;
            simd       * dst    = reinterpret_cast <simd*> (outY);

            for( ; src != srcEnd; ++src ) {
                simd data = _mm256_loadu_si256( src );

                const simd dataLo  = _mm256_unpacklo_epi8( data, zero );
                const simd dataHi  = _mm256_unpackhi_epi8( data, zero );

                const simd data_1 = _mm256_unpacklo_epi16( dataLo, zero );
                const simd data_2 = _mm256_unpackhi_epi16( dataLo, zero );
                const simd data_3 = _mm256_unpacklo_epi16( dataHi, zero );
                const simd data_4 = _mm256_unpackhi_epi16( dataHi, zero );

                _mm256_storeu_si256( dst, _mm256_add_epi32( data_1, _mm256_loadu_si256( dst ) ) );
                ++dst;
                _mm256_storeu_si256( dst, _mm256_add_epi32( data_2, _mm256_loadu_si256( dst ) ) );
                ++dst;
                _mm256_storeu_si256( dst, _mm256_add_epi32( data_3, _mm256_loadu_si256( dst ) ) );
                ++dst;
                _mm256_storeu_si256( dst, _mm256_add_epi32( data_4, _mm256_loadu_si256( dst ) ) );
                ++dst;
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageX    = imageY + totalSimdWidth;
                const uint8_t * imageXEnd = imageX + nonSimdWidth;
                uint32_t      * outX      = outY + totalSimdWidth;

                for( ; imageX != imageXEnd; ++imageX, ++outX )
                    (*outX) += (*imageX);
            }   
        }
    }

    void BitwiseAnd( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm256_storeu_si256( dst, _mm256_and_si256( _mm256_loadu_si256( src1 ), _mm256_loadu_si256( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                    (*outX) = (*in1X) & (*in2X);
            }
        }
    }

    void BitwiseOr( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                    uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm256_storeu_si256( dst, _mm256_or_si256( _mm256_loadu_si256( src1 ), _mm256_loadu_si256( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                    (*outX) = (*in1X) | (*in2X);
            }
        }
    }

    void BitwiseXor( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm256_storeu_si256( dst, _mm256_xor_si256( _mm256_loadu_si256( src1 ), _mm256_loadu_si256( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                    (*outX) = (*in1X) ^ (*in2X);
            }
        }
    }

    void Invert( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd,
                 uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const char maskValue = static_cast<char>(0xffu);
        const simd mask = _mm256_set_epi8(
            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue );

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const simd * src1 = reinterpret_cast <const simd*> (inY);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++dst )
                _mm256_storeu_si256( dst, _mm256_andnot_si256( _mm256_loadu_si256( src1 ), mask ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * inX  = inY  + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++inX )
                    (*outX) = ~(*inX);
            }
        }
    }

    void Maximum( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                  uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm256_storeu_si256( dst, _mm256_max_epu8( _mm256_loadu_si256( src1 ), _mm256_loadu_si256( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X ) {
                    if( (*in2X) < (*in1X) )
                        (*outX) = (*in1X);
                    else
                        (*outX) = (*in2X);
                }
            }
        }
    }

    void Minimum( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                  uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm256_storeu_si256( dst, _mm256_min_epu8( _mm256_loadu_si256( src1 ), _mm256_loadu_si256( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X ) {
                    if( (*in2X) > (*in1X) )
                        (*outX) = (*in1X);
                    else
                        (*outX) = (*in2X);
                }
            }
        }
    }

    void ProjectionProfile( uint32_t rowSize, const uint8_t * imageStart, uint32_t height, bool horizontal,
                            uint32_t * out, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const simd zero = _mm256_setzero_si256();

        if( horizontal ) {
            const uint8_t * imageSimdXEnd = imageStart + totalSimdWidth;

            for( ; imageStart != imageSimdXEnd; imageStart += simdSize, out += simdSize ) {
                const uint8_t * imageSimdY = imageStart;
                const uint8_t * imageSimdYEnd = imageSimdY + height * rowSize;
                simd simdSum_1 = _mm256_setzero_si256();
                simd simdSum_2 = _mm256_setzero_si256();
                simd simdSum_3 = _mm256_setzero_si256();
                simd simdSum_4 = _mm256_setzero_si256();

                simd * dst = reinterpret_cast <simd*> (out);

                for( ; imageSimdY != imageSimdYEnd; imageSimdY += rowSize) {
                    const simd * src    = reinterpret_cast <const simd*> (imageSimdY);

                    const simd data = _mm256_loadu_si256( src );

                    const simd dataLo  = _mm256_unpacklo_epi8( data, zero );
                    const simd dataHi  = _mm256_unpackhi_epi8( data, zero );

                    const simd data_1 = _mm256_unpacklo_epi16( dataLo, zero );
                    const simd data_2 = _mm256_unpackhi_epi16( dataLo, zero );
                    const simd data_3 = _mm256_unpacklo_epi16( dataHi, zero );
                    const simd data_4 = _mm256_unpackhi_epi16( dataHi, zero );
                    simdSum_1 = _mm256_add_epi32( data_1, simdSum_1 );
                    simdSum_2 = _mm256_add_epi32( data_2, simdSum_2 );
                    simdSum_3 = _mm256_add_epi32( data_3, simdSum_3 );
                    simdSum_4 = _mm256_add_epi32( data_4, simdSum_4 );
                }

                _mm256_storeu_si256( dst, _mm256_add_epi32( simdSum_1, _mm256_loadu_si256( dst ) ) );
                ++dst;
                _mm256_storeu_si256( dst, _mm256_add_epi32( simdSum_2, _mm256_loadu_si256( dst ) ) );
                ++dst;
                _mm256_storeu_si256( dst, _mm256_add_epi32( simdSum_3, _mm256_loadu_si256( dst ) ) );
                ++dst;
                _mm256_storeu_si256( dst, _mm256_add_epi32( simdSum_4, _mm256_loadu_si256( dst ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t* imageXEnd = imageStart + nonSimdWidth;

                for( ; imageStart != imageXEnd; ++imageStart, ++out ) {
                    const uint8_t * imageY    = imageStart;
                    const uint8_t * imageYEnd = imageY + height * rowSize;

                    for( ; imageY != imageYEnd; imageY += rowSize )
                        (*out) += (*imageY);
                }
            }
        }
        else {
            const uint8_t * imageYEnd = imageStart + height * rowSize;

            for( ; imageStart != imageYEnd; imageStart += rowSize, ++out ) {
                const simd * src    = reinterpret_cast <const simd*> (imageStart);
                const simd * srcEnd = src + simdWidth;
                simd simdSum = _mm256_setzero_si256();

                for( ; src != srcEnd; ++src ) {
                    simd data = _mm256_loadu_si256( src );

                    simd dataLo  = _mm256_unpacklo_epi8( data, zero );
                    simd dataHi  = _mm256_unpackhi_epi8( data, zero );
                    simd sumLoHi = _mm256_add_epi16( dataLo, dataHi );

                    simdSum = _mm256_add_epi32( simdSum, _mm256_add_epi32( _mm256_unpacklo_epi16( sumLoHi, zero ),
                                                                           _mm256_unpackhi_epi16( sumLoHi, zero ) ) );
                }

                if( nonSimdWidth > 0 ) {
                    const uint8_t * imageX    = imageStart + totalSimdWidth;
                    const uint8_t * imageXEnd = imageX + nonSimdWidth;

                    for( ; imageX != imageXEnd; ++imageX )
                        (*out) += (*imageX);
                }

                uint32_t output[8] = { 0 };
                _mm256_storeu_si256( reinterpret_cast <simd*>(output), simdSum );
                
                (*out) += output[0] + output[1] + output[2] + output[3] + output[4] + output[5] + output[6] + output[7];
            }
        }
    }

    void RgbToBgr( uint8_t * outY, const uint8_t * inY, const uint8_t * outYEnd, uint32_t rowSizeOut, uint32_t rowSizeIn, 
                   const uint8_t colorCount, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const simd ctrl = _mm256_setr_epi8(2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, 15, 
                                           16, 17, 20, 19, 18, 23, 22, 21, 26, 25, 24, 29, 28, 27, 30, 31);
        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + totalSimdWidth;

            for( ; outX != outXEnd; outX += simdWidth, inX += simdWidth ) {
                const simd * src = reinterpret_cast<const simd*> (inX);
                simd * dst = reinterpret_cast<simd*> (outX);
                simd result = _mm256_loadu_si256( src );
                result = _mm256_shuffle_epi8( result, ctrl );
                _mm256_storeu_si256( dst, result );
                *(outX + 15) = *(inX + 17);
                *(outX + 17) = *(inX + 15);
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * outXEndNonSimd = outXEnd + nonSimdWidth;
                for( ; outX != outXEndNonSimd; outX += colorCount, inX += colorCount ) {
                    *(outX + 2) = *(inX);
                    *(outX + 1) = *(inX + 1);
                    *(outX) = *(inX + 2);
                }
            }
        }
    }

    void Subtract( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                   uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst ) {
                simd data = _mm256_loadu_si256( src1 );
                _mm256_storeu_si256( dst, _mm256_sub_epi8( data, _mm256_min_epu8( data, _mm256_loadu_si256( src2 ) ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X ) {
                    if( (*in2X) > (*in1X) )
                        (*outX) = 0;
                    else
                        (*outX) = (*in1X) - (*in2X);
                }
            }
        }
    }

    uint32_t Sum( uint32_t rowSize, const uint8_t * imageY,const uint8_t * imageYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        uint32_t sum = 0;
        simd simdSum = _mm256_setzero_si256();
        simd zero    = _mm256_setzero_si256();

        for( ; imageY != imageYEnd; imageY += rowSize ) {
            const simd * src    = reinterpret_cast <const simd*> (imageY);
            const simd * srcEnd = src + simdWidth;

            for( ; src != srcEnd; ++src ) {
                simd data = _mm256_loadu_si256( src );

                simd dataLo  = _mm256_unpacklo_epi8( data, zero );
                simd dataHi  = _mm256_unpackhi_epi8( data, zero );
                simd sumLoHi = _mm256_add_epi16( dataLo, dataHi );

                simdSum = _mm256_add_epi32( simdSum, _mm256_add_epi32( _mm256_unpacklo_epi16( sumLoHi, zero ),
                                                                       _mm256_unpackhi_epi16( sumLoHi, zero ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageX    = imageY + totalSimdWidth;
                const uint8_t * imageXEnd = imageX + nonSimdWidth;

                for( ; imageX != imageXEnd; ++imageX )
                    sum += (*imageX);
            }
        }

        uint32_t output[8] ={ 0 };

        _mm256_storeu_si256( reinterpret_cast <simd*>(output), simdSum );

        return sum + output[0] + output[1] + output[2] + output[3] + output[4] + output[5] + output[6] + output[7];
    }

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t threshold,
                    uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        // AVX does not have command "great or equal to" so we have 2 situations:
        // when threshold value is 0 and it is not
        if( threshold > 0 ) {
            const char maskValue = static_cast<char>(0x80u);
            const simd mask = _mm256_set_epi8(
                maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
                maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
                maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
                maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue );

            const char compareValue = static_cast<char>((threshold - 1) ^ 0x80u);
            const simd compare = _mm256_set_epi8(
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue );

            for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
                const simd * src1 = reinterpret_cast <const simd*> (inY);
                simd       * dst  = reinterpret_cast <simd*> (outY);

                const simd * src1End = src1 + simdWidth;

                for( ; src1 != src1End; ++src1, ++dst )
                    _mm256_storeu_si256( dst, _mm256_cmpgt_epi8( _mm256_xor_si256( _mm256_loadu_si256( src1 ), mask ), compare ) );

                if( nonSimdWidth > 0 ) {
                    const uint8_t * inX  = inY  + totalSimdWidth;
                    uint8_t       * outX = outY + totalSimdWidth;

                    const uint8_t * outXEnd = outX + nonSimdWidth;

                    for( ; outX != outXEnd; ++outX, ++inX )
                        (*outX) = (*inX) < threshold ? 0 : 255;
                }
            }
        }
        else {
            for( ; outY != outYEnd; outY += rowSizeOut )
                memset( outY, 255u, sizeof( uint8_t ) * (totalSimdWidth + nonSimdWidth) );
        }
    }

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t minThreshold, uint8_t maxThreshold,
                    uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const char shiftMaskValue = static_cast<char>(0x80u);
        const simd shiftMask = _mm256_set_epi8(
            shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue,
            shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue,
            shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue,
            shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue );

        const char notMaskValue = static_cast<char>(0xffu);
        const simd notMask = _mm256_set_epi8(
            notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue,
            notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue,
            notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue,
            notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue, notMaskValue );

        const char maxCompareValue = static_cast<char>(maxThreshold ^ 0x80u);
        const simd maxCompare = _mm256_set_epi8(
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue );

        if( minThreshold > 0 ) {
            const char minCompareValue = static_cast<char>((minThreshold - 1) ^ 0x80u);
            const simd minCompare = _mm256_set_epi8(
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue,
                minCompareValue, minCompareValue, minCompareValue, minCompareValue );


            for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
                const simd * src1 = reinterpret_cast <const simd*> (inY);
                simd       * dst  = reinterpret_cast <simd*> (outY);

                const simd * src1End = src1 + simdWidth;

                for( ; src1 != src1End; ++src1, ++dst ) {
                    simd data = _mm256_xor_si256( _mm256_loadu_si256( src1 ), shiftMask );

                    _mm256_storeu_si256( dst, _mm256_and_si256(
                        _mm256_andnot_si256(
                            _mm256_cmpgt_epi8( data, maxCompare ), notMask ),
                        _mm256_cmpgt_epi8( data, minCompare ) ) );
                }

                if( nonSimdWidth > 0 ) {
                    const uint8_t * inX  = inY  + totalSimdWidth;
                    uint8_t       * outX = outY + totalSimdWidth;

                    const uint8_t * outXEnd = outX + nonSimdWidth;

                    for( ; outX != outXEnd; ++outX, ++inX )
                        (*outX) = (*inX) < minThreshold || (*inX) > maxThreshold ? 0 : 255;
                }
            }
        }
        else {
            for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
                const simd * src1 = reinterpret_cast <const simd*> (inY);
                simd       * dst  = reinterpret_cast <simd*> (outY);

                const simd * src1End = src1 + simdWidth;

                for( ; src1 != src1End; ++src1, ++dst ) {
                    simd data = _mm256_xor_si256( _mm256_loadu_si256( src1 ), shiftMask );

                    _mm256_storeu_si256( dst, _mm256_andnot_si256( _mm256_cmpgt_epi8( data, maxCompare ), notMask ) );
                }

                if( nonSimdWidth > 0 ) {
                    const uint8_t * inX  = inY  + totalSimdWidth;
                    uint8_t       * outX = outY + totalSimdWidth;

                    const uint8_t * outXEnd = outX + nonSimdWidth;

                    for( ; outX != outXEnd; ++outX, ++inX )
                        (*outX) = (*inX) < minThreshold || (*inX) > maxThreshold ? 0 : 255;
                }
            }
        }
    }
#endif
}

namespace sse
{
    const uint32_t simdSize = 16u;

#ifdef PENGUINV_SSE_SET
    typedef __m128i simd;

    // We are not sure that input data is aligned by 16 bytes so we use loadu() functions instead of load()

    void AbsoluteDifference( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                             uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst ) {
                simd data1 = _mm_loadu_si128( src1 );
                simd data2 = _mm_loadu_si128( src2 );
                _mm_storeu_si128( dst, _mm_sub_epi8( _mm_max_epu8( data1, data2 ), _mm_min_epu8( data1, data2 ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                    (*outX) = static_cast<uint8_t>( (*in2X) > (*in1X) ? (*in2X) - (*in1X) : (*in1X) - (*in2X) );
            }
        }
    }

    void Accumulate( uint32_t rowSize, const uint8_t * imageY, const uint8_t * imageYEnd, uint32_t * outY, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        simd zero = _mm_setzero_si128();

        const uint32_t width = totalSimdWidth + nonSimdWidth;

        for( ; imageY != imageYEnd; imageY += rowSize, outY += width ) {
            const simd * src    = reinterpret_cast <const simd*> (imageY);
            const simd * srcEnd = src + simdWidth;
            simd       * dst    = reinterpret_cast <simd*> (outY);

            for( ; src != srcEnd; ++src ) {
                simd data = _mm_loadu_si128( src );

                const simd dataLo  = _mm_unpacklo_epi8( data, zero );
                const simd dataHi  = _mm_unpackhi_epi8( data, zero );

                const simd data_1 = _mm_unpacklo_epi16( dataLo, zero );
                const simd data_2 = _mm_unpackhi_epi16( dataLo, zero );
                const simd data_3 = _mm_unpacklo_epi16( dataHi, zero );
                const simd data_4 = _mm_unpackhi_epi16( dataHi, zero );

                _mm_storeu_si128( dst, _mm_add_epi32( data_1, _mm_loadu_si128( dst ) ) );
                ++dst;
                _mm_storeu_si128( dst, _mm_add_epi32( data_2, _mm_loadu_si128( dst ) ) );
                ++dst;
                _mm_storeu_si128( dst, _mm_add_epi32( data_3, _mm_loadu_si128( dst ) ) );
                ++dst;
                _mm_storeu_si128( dst, _mm_add_epi32( data_4, _mm_loadu_si128( dst ) ) );
                ++dst;
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageX    = imageY + totalSimdWidth;
                const uint8_t * imageXEnd = imageX + nonSimdWidth;
                uint32_t      * outX      = outY + totalSimdWidth;

                for( ; imageX != imageXEnd; ++imageX, ++outX )
                    (*outX) += (*imageX);
            }   
        }
    }

    void BitwiseAnd( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm_storeu_si128( dst, _mm_and_si128( _mm_loadu_si128( src1 ), _mm_loadu_si128( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                    (*outX) = (*in1X) & (*in2X);
            }
        }
    }

    void BitwiseOr( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                    uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm_storeu_si128( dst, _mm_or_si128( _mm_loadu_si128( src1 ), _mm_loadu_si128( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                    (*outX) = (*in1X) | (*in2X);
            }
        }
    }

    void BitwiseXor( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm_storeu_si128( dst, _mm_xor_si128( _mm_loadu_si128( src1 ), _mm_loadu_si128( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                    (*outX) = (*in1X) ^ (*in2X);
            }
        }
    }

#ifdef PENGUINV_SSSE3_SET
    void ConvertToRgb( uint8_t * outY, const uint8_t * outYEnd, const uint8_t * inY, uint32_t rowSizeOut, uint32_t rowSizeIn,
                       uint8_t colorCount, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const simd ctrl1 = _mm_setr_epi8( 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5 );
        const simd ctrl2 = _mm_setr_epi8( 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10 );
        const simd ctrl3 = _mm_setr_epi8( 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15 );
        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const simd * src = reinterpret_cast<const simd*>(inY);
            simd       * dst = reinterpret_cast<simd*>(outY);

            const simd * srcEnd = src + simdWidth;

            for( ; src != srcEnd; ++src ) {
                const simd src1 = _mm_loadu_si128( src );

                _mm_storeu_si128( dst++, _mm_shuffle_epi8( src1, ctrl1 ) );
                _mm_storeu_si128( dst++, _mm_shuffle_epi8( src1, ctrl2 ) );
                _mm_storeu_si128( dst++, _mm_shuffle_epi8( src1, ctrl3 ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * inX  = inY + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth*colorCount;

                const uint8_t * inXEnd = inX + nonSimdWidth;

                for( ; inX != inXEnd; outX += colorCount, ++inX )
                    memset( outX, (*inX), sizeof( uint8_t ) * colorCount );
            }
        }
    }
#endif

#ifdef PENGUINV_SSSE3_SET
    void Flip( Image_Function::Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, const uint32_t rowSizeIn, 
               const uint32_t rowSizeOut, const uint8_t * inY, const uint8_t * inYEnd, bool horizontal, bool vertical, const uint32_t simdWidth, 
               const uint32_t totalSimdWidth, const uint32_t nonSimdWidth )
    {
        const simd ctrl = _mm_setr_epi8( 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 );

        if( horizontal && !vertical ) {
            uint8_t * outYSimd = out.data() + startYOut * rowSizeOut + startXOut + width - simdSize;
            uint8_t * outY = out.data() + startYOut * rowSizeOut + startXOut + width - 1;

            for( ; inY != inYEnd; inY += rowSizeIn, outY += rowSizeOut, outYSimd += rowSizeOut ) {
                const simd * inXSimd    = reinterpret_cast<const simd*>(inY);
                simd       * outXSimd   = reinterpret_cast<simd*>(outYSimd);
                const simd * inXEndSimd = inXSimd + simdWidth;

                for( ; inXSimd != inXEndSimd; ++inXSimd, --outXSimd )
                    _mm_storeu_si128( outXSimd, _mm_shuffle_epi8( _mm_loadu_si128( inXSimd ), ctrl ) );
                        
                if(nonSimdWidth > 0)
                {
                    const uint8_t * inX    = inY + totalSimdWidth;
                    uint8_t       * outX   = outY - totalSimdWidth;
                    const uint8_t * inXEnd = inX + nonSimdWidth;

                    for( ; inX != inXEnd; ++inX, --outX )
                        (*outX) = (*inX);
                }
            }
        }
        else if( !horizontal && vertical ) {
            uint8_t * outY = out.data() + (startYOut + height - 1) * rowSizeOut + startXOut;

            for( ; inY != inYEnd; inY += rowSizeIn, outY -= rowSizeOut )
                memcpy( outY, inY, sizeof( uint8_t ) * width );
        }
        else {
            uint8_t * outYSimd = out.data() + (startYOut + height - 1) * rowSizeOut + startXOut + width - simdSize;
            uint8_t * outY = out.data() + (startYOut + height - 1) * rowSizeOut + startXOut + width - 1;

            for( ; inY != inYEnd; inY += rowSizeIn, outY -= rowSizeOut, outYSimd -= rowSizeOut ) {
                const simd * inXSimd    = reinterpret_cast<const simd*>(inY);
                simd       * outXSimd   = reinterpret_cast<simd*>(outYSimd);
                const simd * inXEndSimd = inXSimd + simdWidth;

                for( ; inXSimd != inXEndSimd; ++inXSimd, --outXSimd )
                    _mm_storeu_si128( outXSimd, _mm_shuffle_epi8( _mm_loadu_si128( inXSimd ), ctrl ) );
                        
                if(nonSimdWidth > 0)
                {
                    const uint8_t * inX    = inY + totalSimdWidth;
                    uint8_t       * outX   = outY - totalSimdWidth;
                    const uint8_t * inXEnd = inX + nonSimdWidth;

                    for( ; inX != inXEnd; ++inX, --outX )
                        (*outX) = (*inX);
                }
            }
        }
    }
#endif

    void Invert( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd,
                 uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const char maskValue = static_cast<char>(0xffu);
        const simd mask = _mm_set_epi8( maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
                                        maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue );

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const simd * src1 = reinterpret_cast <const simd*> (inY);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++dst )
                _mm_storeu_si128( dst, _mm_andnot_si128( _mm_loadu_si128( src1 ), mask ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * inX  = inY  + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++inX )
                    (*outX) = ~(*inX);
            }
        }
    }

    void Maximum( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                  uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm_storeu_si128( dst, _mm_max_epu8( _mm_loadu_si128( src1 ), _mm_loadu_si128( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X ) {
                    if( (*in2X) < (*in1X) )
                        (*outX) = (*in1X);
                    else
                        (*outX) = (*in2X);
                }
            }
        }
    }

    void Minimum( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                  uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst )
                _mm_storeu_si128( dst, _mm_min_epu8( _mm_loadu_si128( src1 ), _mm_loadu_si128( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X ) {
                    if( (*in2X) > (*in1X) )
                        (*outX) = (*in1X);
                    else
                        (*outX) = (*in2X);
                }
            }
        }
    }

    void ProjectionProfile( uint32_t rowSize, const uint8_t * imageStart, uint32_t height, bool horizontal,
                            uint32_t * out, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const simd zero = _mm_setzero_si128();

        if( horizontal ) {
            const uint8_t * imageSimdXEnd = imageStart + totalSimdWidth;

            for( ; imageStart != imageSimdXEnd; imageStart += simdSize, out += simdSize ) {
                const uint8_t * imageSimdY    = imageStart;
                const uint8_t * imageSimdYEnd = imageSimdY + height * rowSize;
                simd simdSum_1 = _mm_setzero_si128();
                simd simdSum_2 = _mm_setzero_si128();
                simd simdSum_3 = _mm_setzero_si128();
                simd simdSum_4 = _mm_setzero_si128();

                simd * dst = reinterpret_cast <simd*> (out);

                for( ; imageSimdY != imageSimdYEnd; imageSimdY += rowSize) {
                    const simd * src    = reinterpret_cast <const simd*> (imageSimdY);

                    const simd data = _mm_loadu_si128( src );

                    const simd dataLo  = _mm_unpacklo_epi8( data, zero );
                    const simd dataHi  = _mm_unpackhi_epi8( data, zero );

                    const simd data_1 = _mm_unpacklo_epi16( dataLo, zero );
                    const simd data_2 = _mm_unpackhi_epi16( dataLo, zero );
                    const simd data_3 = _mm_unpacklo_epi16( dataHi, zero );
                    const simd data_4 = _mm_unpackhi_epi16( dataHi, zero );
                    simdSum_1 = _mm_add_epi32( data_1, simdSum_1 );
                    simdSum_2 = _mm_add_epi32( data_2, simdSum_2 );
                    simdSum_3 = _mm_add_epi32( data_3, simdSum_3 );
                    simdSum_4 = _mm_add_epi32( data_4, simdSum_4 );
                }

                _mm_storeu_si128( dst, _mm_add_epi32( simdSum_1, _mm_loadu_si128( dst ) ) );
                ++dst;
                _mm_storeu_si128( dst, _mm_add_epi32( simdSum_2, _mm_loadu_si128( dst ) ) );
                ++dst;
                _mm_storeu_si128( dst, _mm_add_epi32( simdSum_3, _mm_loadu_si128( dst ) ) );
                ++dst;
                _mm_storeu_si128( dst, _mm_add_epi32( simdSum_4, _mm_loadu_si128( dst ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageXEnd = imageStart + nonSimdWidth;

                for( ; imageStart != imageXEnd; ++imageStart, ++out ) {
                    const uint8_t * imageY    = imageStart;
                    const uint8_t * imageYEnd = imageY + height * rowSize;

                    for( ; imageY != imageYEnd; imageY += rowSize )
                        (*out) += (*imageY);
                }
            }
        }
        else {
            const uint8_t * imageYEnd = imageStart + height * rowSize;

            for( ; imageStart != imageYEnd; imageStart += rowSize, ++out ) {
                const simd * src    = reinterpret_cast <const simd*> (imageStart);
                const simd * srcEnd = src + simdWidth;
                simd simdSum = _mm_setzero_si128();

                for( ; src != srcEnd; ++src ) {
                    simd data = _mm_loadu_si128( src );

                    simd dataLo  = _mm_unpacklo_epi8( data, zero );
                    simd dataHi  = _mm_unpackhi_epi8( data, zero );
                    simd sumLoHi = _mm_add_epi16( dataLo, dataHi );

                    simdSum = _mm_add_epi32( simdSum, _mm_add_epi32( _mm_unpacklo_epi16( sumLoHi, zero ),
                                                                     _mm_unpackhi_epi16( sumLoHi, zero ) ) );
                }

                if( nonSimdWidth > 0 ) {
                    const uint8_t * imageX    = imageStart + totalSimdWidth;
                    const uint8_t * imageXEnd = imageX + nonSimdWidth;

                    for( ; imageX != imageXEnd; ++imageX )
                        (*out) += (*imageX);
                }

                uint32_t output[4] = { 0 };
                _mm_storeu_si128( reinterpret_cast <simd*>(output), simdSum );
                
                (*out) += output[0] + output[1] + output[2] + output[3];
            }
        }
    }

#ifdef PENGUINV_SSSE3_SET
    void RgbToBgr( uint8_t * outY, const uint8_t * inY, const uint8_t * outYEnd, uint32_t rowSizeOut, uint32_t rowSizeIn, 
                   const uint8_t colorCount, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const simd ctrl = _mm_set_epi8(15, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2);
        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + totalSimdWidth;

            for( ; outX != outXEnd; outX += simdWidth, inX += simdWidth ) {
                const simd * src = reinterpret_cast<const simd*> (inX);
                simd * dst = reinterpret_cast<simd*> (outX);
                simd result = _mm_loadu_si128( src );
                result = _mm_shuffle_epi8( result, ctrl );
                _mm_storeu_si128( dst, result );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * outXEndNonSimd = outXEnd + nonSimdWidth;
                for( ; outX != outXEndNonSimd; outX += colorCount, inX += colorCount ) {
                    *(outX + 2) = *(inX);
                    *(outX + 1) = *(inX + 1);
                    *(outX) = *(inX + 2);
                }
            }
        }
    }
#endif

    void Subtract( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                   uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const simd * src1 = reinterpret_cast <const simd*> (in1Y);
            const simd * src2 = reinterpret_cast <const simd*> (in2Y);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++src2, ++dst ) {
                simd data = _mm_loadu_si128( src1 );
                _mm_storeu_si128( dst, _mm_sub_epi8( data, _mm_min_epu8( data, _mm_loadu_si128( src2 ) ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X ) {
                    if( (*in2X) > (*in1X) )
                        (*outX) = 0u;
                    else
                        (*outX) = static_cast<uint8_t>( (*in1X) - (*in2X) );
                }
            }
        }
    }

    uint32_t Sum( uint32_t rowSize, const uint8_t * imageY, const uint8_t * imageYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        uint32_t sum = 0;
        simd simdSum = _mm_setzero_si128();
        simd zero    = _mm_setzero_si128();

        for( ; imageY != imageYEnd; imageY += rowSize ) {
            const simd * src    = reinterpret_cast <const simd*> (imageY);
            const simd * srcEnd = src + simdWidth;

            for( ; src != srcEnd; ++src ) {
                simd data = _mm_loadu_si128( src );

                simd dataLo  = _mm_unpacklo_epi8( data, zero );
                simd dataHi  = _mm_unpackhi_epi8( data, zero );
                simd sumLoHi = _mm_add_epi16( dataLo, dataHi );

                simdSum = _mm_add_epi32( simdSum, _mm_add_epi32( _mm_unpacklo_epi16( sumLoHi, zero ),
                                                                 _mm_unpackhi_epi16( sumLoHi, zero ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageX    = imageY + totalSimdWidth;
                const uint8_t * imageXEnd = imageX + nonSimdWidth;

                for( ; imageX != imageXEnd; ++imageX )
                    sum += (*imageX);
            }
        }

        uint32_t output[4] ={ 0 };

        _mm_storeu_si128( reinterpret_cast <simd*>(output), simdSum );

        return sum + output[0] + output[1] + output[2] + output[3];
    }

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t threshold,
                    uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        // SSE does not have command "great or equal to" so we have 2 situations:
        // when threshold value is 0 and it is not
        if( threshold > 0 ) {
            const char maskValue = static_cast<char>(0x80u);
            const simd mask = _mm_set_epi8( maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
                                            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue );

            const char compareValue = static_cast<char>((threshold - 1) ^ 0x80u);
            const simd compare = _mm_set_epi8(
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue,
                compareValue, compareValue, compareValue, compareValue );

            for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
                const simd * src1 = reinterpret_cast <const simd*> (inY);
                simd       * dst  = reinterpret_cast <simd*> (outY);

                const simd * src1End = src1 + simdWidth;

                for( ; src1 != src1End; ++src1, ++dst )
                    _mm_storeu_si128( dst, _mm_cmpgt_epi8( _mm_xor_si128( _mm_loadu_si128( src1 ), mask ), compare ) );

                if( nonSimdWidth > 0 ) {
                    const uint8_t * inX  = inY  + totalSimdWidth;
                    uint8_t       * outX = outY + totalSimdWidth;

                    const uint8_t * outXEnd = outX + nonSimdWidth;

                    for( ; outX != outXEnd; ++outX, ++inX )
                        (*outX) = (*inX) < threshold ? 0 : 255;
                }
            }
        }
        else {
            for( ; outY != outYEnd; outY += rowSizeOut )
                memset( outY, 255u, sizeof( uint8_t ) * (totalSimdWidth + nonSimdWidth) );
        }
    }

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t minThreshold, uint8_t maxThreshold,
                    uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const char shiftMaskValue = static_cast<char>(0x80u);
        const simd shiftMask = _mm_set_epi8( shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue,
                                             shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue,
                                             shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue,
                                             shiftMaskValue, shiftMaskValue, shiftMaskValue, shiftMaskValue );

        const char notMaskValue = static_cast<char>(0xffu);
        const simd notMask = _mm_set_epi8( notMaskValue, notMaskValue, notMaskValue, notMaskValue,
                                           notMaskValue, notMaskValue, notMaskValue, notMaskValue,
                                           notMaskValue, notMaskValue, notMaskValue, notMaskValue,
                                           notMaskValue, notMaskValue, notMaskValue, notMaskValue );

        const char minCompareValue = static_cast<char>(minThreshold ^ 0x80u);
        const simd minCompare = _mm_set_epi8(
            minCompareValue, minCompareValue, minCompareValue, minCompareValue,
            minCompareValue, minCompareValue, minCompareValue, minCompareValue,
            minCompareValue, minCompareValue, minCompareValue, minCompareValue,
            minCompareValue, minCompareValue, minCompareValue, minCompareValue );

        const char maxCompareValue = static_cast<char>(maxThreshold ^ 0x80u);
        const simd maxCompare = _mm_set_epi8(
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue,
            maxCompareValue, maxCompareValue, maxCompareValue, maxCompareValue );

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const simd * src1 = reinterpret_cast <const simd*> (inY);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++dst ) {
                simd data = _mm_xor_si128( _mm_loadu_si128( src1 ), shiftMask );

                _mm_storeu_si128( dst, _mm_andnot_si128(
                    _mm_or_si128(
                        _mm_cmplt_epi8( data, minCompare ),
                        _mm_cmpgt_epi8( data, maxCompare ) ),
                    notMask ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * inX  = inY  + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++inX )
                    (*outX) = (*inX) < minThreshold || (*inX) > maxThreshold ? 0 : 255;
            }
        }
    }
#endif
}

namespace neon
{
    const uint32_t simdSize = 16u;

#ifdef PENGUINV_NEON_SET
    typedef uint8x16_t simd;

    void AbsoluteDifference( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                             uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const uint8_t * src1 = in1Y;
            const uint8_t * src2 = in2Y;
            uint8_t       * dst  = outY;

            const uint8_t * src1End = src1 + totalSimdWidth;

            for( ; src1 != src1End; src1 += simdSize, src2 += simdSize, dst += simdSize )
                vst1q_u8( dst, vabdq_u8( vld1q_u8( src1 ), vld1q_u8( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                    (*outX) = (*in2X) > (*in1X) ? (*in2X) - (*in1X) : (*in1X) - (*in2X);
            }
        }
    }
    
    void Accumulate( uint32_t rowSize, const uint8_t * imageY, const uint8_t * imageYEnd, uint32_t * outY, uint32_t simdWidth,
                     uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const uint8x8_t zero_8 = vdup_n_u8(0);

        const uint32_t width = totalSimdWidth + nonSimdWidth;

        for( ; imageY != imageYEnd; imageY += rowSize, outY += width ) {
            const uint8_t * src    = imageY;
            const uint8_t * srcEnd = src + totalSimdWidth;
            uint32_t      * dst    = outY;

            for( ; src != srcEnd; src+= simdSize ) {
                uint8x16_t data = vld1q_u8( src );

                const uint16x8_t dataLo  = vaddl_u8( vget_low_u8(data), zero_8 );
                const uint16x8_t dataHi  = vaddl_u8( vget_high_u8(data), zero_8 );

                vst1q_u32( dst, vaddw_u16( vld1q_u32( dst ), vget_low_u16(dataLo) ) );
                dst += 4;
                vst1q_u32( dst, vaddw_u16( vld1q_u32( dst ), vget_high_u16(dataLo) ) );
                dst += 4;
                vst1q_u32( dst, vaddw_u16( vld1q_u32( dst ), vget_low_u16(dataHi) ) );
                dst += 4;
                vst1q_u32( dst, vaddw_u16( vld1q_u32( dst ), vget_high_u16(dataHi) ) );
                dst += 4;
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageX    = imageY + totalSimdWidth;
                const uint8_t * imageXEnd = imageX + nonSimdWidth;
                uint32_t      * outX      = outY + totalSimdWidth;

                for( ; imageX != imageXEnd; ++imageX, ++outX )
                    (*outX) += (*imageX);
            }
        }
    }

    void BitwiseAnd( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const uint8_t * src1 = in1Y;
            const uint8_t * src2 = in2Y;
            uint8_t       * dst  = outY;

            const uint8_t * src1End = src1 + totalSimdWidth;

            for( ; src1 != src1End; src1 += simdSize, src2 += simdSize, dst += simdSize )
                vst1q_u8( dst, vandq_u8( vld1q_u8( src1 ), vld1q_u8( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                    (*outX) = (*in1X) & (*in2X);
            }
        }
    }

    void BitwiseOr( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                    uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const uint8_t * src1 = in1Y;
            const uint8_t * src2 = in2Y;
            uint8_t       * dst  = outY;

            const uint8_t * src1End = src1 + totalSimdWidth;

            for( ; src1 != src1End; src1 += simdSize, src2 += simdSize, dst += simdSize )
                vst1q_u8( dst, vorrq_u8( vld1q_u8( src1 ), vld1q_u8( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                    (*outX) = (*in1X) | (*in2X);
            }
        }
    }

    void BitwiseXor( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                     uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const uint8_t * src1 = in1Y;
            const uint8_t * src2 = in2Y;
            uint8_t       * dst  = outY;

            const uint8_t * src1End = src1 + totalSimdWidth;

            for( ; src1 != src1End; src1 += simdSize, src2 += simdSize, dst += simdSize )
                vst1q_u8( dst, veorq_u8( vld1q_u8( src1 ), vld1q_u8( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
                    (*outX) = (*in1X) ^ (*in2X);
            }
        }
    }

    void ConvertToRgb( uint8_t * outY, const uint8_t * outYEnd, const uint8_t * inY, uint32_t rowSizeOut, uint32_t rowSizeIn,
                       uint8_t colorCount, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const uint8_t ctrl1_array[8] = {0, 0, 0, 1, 1, 1, 2, 2};
        const uint8_t ctrl2_array[8] = {2, 3, 3, 3, 4, 4, 4, 5};
        const uint8_t ctrl3_array[8] = {5, 5, 6, 6, 6, 7, 7, 7};

        const uint8x8_t ctrl1 = vld1_u8( ctrl1_array );
        const uint8x8_t ctrl2 = vld1_u8( ctrl2_array );
        const uint8x8_t ctrl3 = vld1_u8( ctrl3_array );

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * src = inY;
            uint8_t       * dst = outY;

            const uint8_t * srcEnd = src + simdWidth;

            for( ; src != srcEnd; ++src ) {
                const uint8x8_t src1 = vld1_u8( src );

                vst1_u8( dst, vtbl1_u8( src1, ctrl1 ) );
                dst += 8;
                vst1_u8( dst, vtbl1_u8( src1, ctrl2 ) );
                dst += 8;
                vst1_u8( dst, vtbl1_u8( src1, ctrl3 ) );
                dst += 8;
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * inX  = inY + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth*colorCount;

                const uint8_t * inXEnd = inX + nonSimdWidth;

                for( ; inX != inXEnd; outX += colorCount, ++inX )
                    memset( outX, (*inX), sizeof( uint8_t ) * colorCount );
            }
        }
    }

    void Flip( Image_Function::Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, const uint32_t rowSizeIn, 
               const uint32_t rowSizeOut, const uint8_t * inY, const uint8_t * inYEnd, bool horizontal, bool vertical, const uint32_t simdWidth, 
               const uint32_t totalSimdWidth, const uint32_t nonSimdWidth )
    {
        if( horizontal && !vertical ) {
            uint8_t * outYSimd = out.data() + startYOut * rowSizeOut + startXOut + width - 8;
            uint8_t * outY = out.data() + startYOut * rowSizeOut + startXOut + width - 1;

            for( ; inY != inYEnd; inY += rowSizeIn, outY += rowSizeOut, outYSimd += rowSizeOut ) {
                const uint8_t * inXSimd    = inY;
                uint8_t       * outXSimd   = outYSimd;
                const uint8_t * inXEndSimd = inXSimd + simdWidth * 8;

                for( ; inXSimd != inXEndSimd; inXSimd += 8, outXSimd -= 8 )
                    vst1_u8( outXSimd, vrev64_u8( vld1_u8( inXSimd ) ) );
                        
                if( nonSimdWidth > 0 ) {
                    const uint8_t * inX    = inY + totalSimdWidth;
                    uint8_t       * outX   = outY - totalSimdWidth;
                    const uint8_t * inXEnd = inX + nonSimdWidth;

                    for( ; inX != inXEnd; ++inX, --outX )
                        (*outX) = (*inX);
                }
            }
        }
        else if( !horizontal && vertical ) {
            uint8_t * outY = out.data() + (startYOut + height - 1) * rowSizeOut + startXOut;

            for( ; inY != inYEnd; inY += rowSizeIn, outY -= rowSizeOut )
                memcpy( outY, inY, sizeof( uint8_t ) * width );
        }
        else {
            uint8_t * outYSimd = out.data() + (startYOut + height - 1) * rowSizeOut + startXOut + width - 8;
            uint8_t * outY = out.data() + (startYOut + height - 1) * rowSizeOut + startXOut + width - 1;

            for( ; inY != inYEnd; inY += rowSizeIn, outY -= rowSizeOut, outYSimd -= rowSizeOut ) {
                const uint8_t * inXSimd    = inY;
                uint8_t       * outXSimd   = outYSimd;
                const uint8_t * inXEndSimd = inXSimd + simdWidth * 8;

                for( ; inXSimd != inXEndSimd; inXSimd += 8, outXSimd -= 8 )
                    vst1_u8( outXSimd, vrev64_u8( vld1_u8( inXSimd ) ) );
                        
                if( nonSimdWidth > 0 ) {
                    const uint8_t * inX    = inY + totalSimdWidth;
                    uint8_t       * outX   = outY - totalSimdWidth;
                    const uint8_t * inXEnd = inX + nonSimdWidth;

                    for( ; inX != inXEnd; ++inX, --outX )
                        (*outX) = (*inX);
                }
            }
        }
    }

    void Invert( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd,
                 uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const char maskValue = static_cast<char>(0xffu);
        const simd mask = _mm256_set_epi8(
            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue,
            maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue, maskValue );

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const simd * src1 = reinterpret_cast <const simd*> (inY);
            simd       * dst  = reinterpret_cast <simd*> (outY);

            const simd * src1End = src1 + simdWidth;

            for( ; src1 != src1End; ++src1, ++dst )
                _mm256_storeu_si256( dst, _mm256_andnot_si256( _mm256_loadu_si256( src1 ), mask ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * inX  = inY  + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++inX )
                    (*outX) = ~(*inX);
            }
        }
    }

    void Minimum( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                  uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const uint8_t * src1 = in1Y;
            const uint8_t * src2 = in2Y;
            uint8_t       * dst  = outY;

            const uint8_t * src1End = src1 + totalSimdWidth;

            for( ; src1 != src1End; src1 += simdSize, src2 += simdSize, dst += simdSize )
                vst1q_u8( dst, vminq_u8( vld1q_u8( src1 ), vld1q_u8( src2 ) ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X ) {
                    if( (*in2X) > (*in1X) )
                        (*outX) = (*in1X);
                    else
                        (*outX) = (*in2X);
                }
            }
        }
    }

    void ProjectionProfile( uint32_t rowSize, const uint8_t * imageStart, uint32_t height, bool horizontal,
                            uint32_t * out, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const uint8x8_t zero = vdup_n_u8(0);
        const uint16x4_t zero_16 = vdup_n_u16(0);

        if( horizontal ) {
            const uint8_t * imageSimdXEnd = imageStart + totalSimdWidth;

            for( ; imageStart != imageSimdXEnd; imageStart += simdSize, out += simdSize ) {
                const uint8_t * imageSimdY    = imageStart;
                const uint8_t * imageSimdYEnd = imageSimdY + height * rowSize;
                uint32x4_t simdSum_1 = vdupq_n_u32(0);
                uint32x4_t simdSum_2 = vdupq_n_u32(0);
                uint32x4_t simdSum_3 = vdupq_n_u32(0);
                uint32x4_t simdSum_4 = vdupq_n_u32(0);

                uint32_t * dst = out;

                for( ; imageSimdY != imageSimdYEnd; imageSimdY += rowSize) {
                    const uint8_t * src = imageSimdY;

                    const uint8x16_t data = vld1q_u8( src );

                    const uint16x8_t dataLo = vaddl_u8( vget_low_u8(data), zero );
                    const uint16x8_t dataHi = vaddl_u8( vget_high_u8(data), zero );

                    const uint32x4_t data_1 = vaddl_u16( vget_low_u16 (dataLo), zero_16 );
                    const uint32x4_t data_2 = vaddl_u16( vget_high_u16(dataLo), zero_16 );
                    const uint32x4_t data_3 = vaddl_u16( vget_low_u16 (dataLo), zero_16 );
                    const uint32x4_t data_4 = vaddl_u16( vget_high_u16(dataLo), zero_16 );

                    simdSum_1 = vaddq_u32( simdSum_1, data_1 );
                    simdSum_2 = vaddq_u32( simdSum_2, data_2 );
                    simdSum_3 = vaddq_u32( simdSum_3, data_3 );
                    simdSum_4 = vaddq_u32( simdSum_4, data_4 );
                }

                vst1q_u32( dst, vaddq_u32( vld1q_u32( dst ), simdSum_1 ) );
                dst += 4;
                vst1q_u32( dst, vaddq_u32( vld1q_u32( dst ), simdSum_2 ) );
                dst += 4;
                vst1q_u32( dst, vaddq_u32( vld1q_u32( dst ), simdSum_3 ) );
                dst += 4;
                vst1q_u32( dst, vaddq_u32( vld1q_u32( dst ), simdSum_4 ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageXEnd = imageStart + nonSimdWidth;

                for( ; imageStart != imageXEnd; ++imageStart, ++out ) {
                    const uint8_t * imageY    = imageStart;
                    const uint8_t * imageYEnd = imageY + height * rowSize;

                    for( ; imageY != imageYEnd; imageY += rowSize )
                        (*out) += (*imageY);
                }
            }
        }
        else {
            const uint8_t * imageYEnd = imageStart + height * rowSize;

            for( ; imageStart != imageYEnd; imageStart += rowSize, ++out ) {
                const uint8_t * src    = imageStart;
                const uint8_t * srcEnd = src + simdWidth*simdSize;
                uint32x4_t simdSum = vdupq_n_u32(0);

                for( ; src != srcEnd; src += simdSize ) {
                    const uint8x16_t data = vld1q_u8( src );

                    const uint16x8_t dataLo  = vaddl_u8( vget_low_u8(data), zero );
                    const uint16x8_t dataHi  = vaddl_u8( vget_high_u8(data), zero );
                    const uint16x8_t sumLoHi = vaddq_u16( dataHi, dataLo );

                    const uint32x4_t sum = vaddl_u16( vadd_u16( vget_low_u16(sumLoHi),
                                                                vget_high_u16(sumLoHi) ),
                                                                zero_16 );

                    simdSum = vaddq_u32( simdSum, sum );
                }

                if( nonSimdWidth > 0 ) {
                    const uint8_t * imageX    = imageStart + totalSimdWidth;
                    const uint8_t * imageXEnd = imageX + nonSimdWidth;

                    for( ; imageX != imageXEnd; ++imageX )
                        (*out) += (*imageX);
                }

                uint32_t output[4] = { 0 };
                vst1q_u32( output, simdSum );

                (*out) += output[0] + output[1] + output[2] + output[3];
            }
        }
    }

    void RgbToBgr( uint8_t * outY, const uint8_t * inY, const uint8_t * outYEnd, uint32_t rowSizeOut, uint32_t rowSizeIn, 
                   const uint8_t colorCount, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const uint8_t ctrl_array[8] = {2, 1, 0, 5, 4, 3, 6, 7};
        const uint8x8_t ctrl = vld1_u8( ctrl_array );
        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * inX  = inY;
            uint8_t       * outX = outY;

            const uint8_t * outXEnd = outX + totalSimdWidth;

            for( ; outX != outXEnd; outX += simdWidth, inX += simdWidth ) {
                uint8x8_t result = vld1_u8( inX );
                result = vtbl1_u8( result, ctrl );
                vst1_u8( outX, result );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * outXEndNonSimd = outXEnd + nonSimdWidth;
                for( ; outX != outXEndNonSimd; outX += colorCount, inX += colorCount ) {
                    *(outX + 2) = *(inX);
                    *(outX + 1) = *(inX + 1);
                    *(outX) = *(inX + 2);
                }
            }
        }
    }

    void Subtract( uint32_t rowSizeIn1, uint32_t rowSizeIn2, uint32_t rowSizeOut, const uint8_t * in1Y, const uint8_t * in2Y,
                   uint8_t * outY, const uint8_t * outYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2 ) {
            const uint8_t * src1 = in1Y;
            const uint8_t * src2 = in2Y;
            uint8_t       * dst  = outY;

            const uint8_t * src1End = src1 + totalSimdWidth;

            for( ; src1 != src1End; src1 += simdSize, src2 += simdSize, dst += simdSize ) {
                const simd data = vld1q_u8( src1 );
                vst1q_u8( dst, vsubq_u8( data, vminq_u8( data, vld1q_u8( src2 ) ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * in1X = in1Y + totalSimdWidth;
                const uint8_t * in2X = in2Y + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++in1X, ++in2X ) {
                    if( (*in2X) > (*in1X) )
                        (*outX) = 0;
                    else
                        (*outX) = (*in1X) - (*in2X);
                }
            }
        }
    }

    uint32_t Sum( uint32_t rowSize, const uint8_t * imageY, const uint8_t * imageYEnd, uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        uint32_t sum = 0;
        uint32x4_t simdSum = vdupq_n_u32(0);

        for( ; imageY != imageYEnd; imageY += rowSize ) {
            const uint8_t * src    = imageY;
            const uint8_t * srcEnd = src + simdWidth;

            for( ; src != srcEnd; ++src ) {
                const uint8x16_t data = vld1q_u8(src);
                const uint16x8_t data8Sum = vaddl_u8(vget_high_u8(data), vget_low_u8(data));
                const uint32x4_t data16Sum = vaddl_u16(vget_high_u16(data8Sum), vget_low_u16(data8Sum));
                simdSum = vaddq_u32(simdSum, data16Sum);
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * imageX    = imageY + totalSimdWidth;
                const uint8_t * imageXEnd = imageX + nonSimdWidth;

                for( ; imageX != imageXEnd; ++imageX )
                    sum += (*imageX);
            }
        }

        uint32_t output[4] = { 0 };
        vst1q_u32(output, simdSum);
        return (sum + output[0] + output[1] + output[2] + output[3]);
    }

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t threshold,
                    uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const uint8_t thresholdValue[16] ={ threshold, threshold, threshold, threshold, threshold, threshold, threshold, threshold,
                                            threshold, threshold, threshold, threshold, threshold, threshold, threshold, threshold };
        const simd compare = vld1q_u8( thresholdValue );

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * src = inY;
            uint8_t       * dst = outY;

            const uint8_t * srcEnd = src + totalSimdWidth;

            for( ; src != srcEnd; src += simdSize, dst += simdSize )
                vst1q_u8( dst, vcgeq_u8( vld1q_u8( src ), compare ) );

            if( nonSimdWidth > 0 ) {
                const uint8_t * inX  = inY  + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++inX )
                    (*outX) = (*inX) < threshold ? 0 : 255;
            }
        }
    }

    void Threshold( uint32_t rowSizeIn, uint32_t rowSizeOut, const uint8_t * inY, uint8_t * outY, const uint8_t * outYEnd, uint8_t minThreshold, uint8_t maxThreshold,
                    uint32_t simdWidth, uint32_t totalSimdWidth, uint32_t nonSimdWidth )
    {
        const uint8_t thresholdMinValue[16] ={ minThreshold, minThreshold, minThreshold, minThreshold,
                                               minThreshold, minThreshold, minThreshold, minThreshold,
                                               minThreshold, minThreshold, minThreshold, minThreshold,
                                               minThreshold, minThreshold, minThreshold, minThreshold };
        const simd compareMin = vld1q_u8( thresholdMinValue );

        const uint8_t thresholdMaxValue[16] ={ maxThreshold, maxThreshold, maxThreshold, maxThreshold,
                                               maxThreshold, maxThreshold, maxThreshold, maxThreshold,
                                               maxThreshold, maxThreshold, maxThreshold, maxThreshold,
                                               maxThreshold, maxThreshold, maxThreshold, maxThreshold };
        const simd compareMax = vld1q_u8( thresholdMaxValue );

        for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
            const uint8_t * src = inY;
            uint8_t       * dst = outY;

            const uint8_t * srcEnd = src + totalSimdWidth;

            for( ; src != srcEnd; src += simdSize, dst += simdSize ) {
                const simd data = vld1q_u8( src );
                vst1q_u8( dst, vandq_u8( vcgeq_u8( data, compareMin ), vcleq_u8( data, compareMax ) ) );
            }

            if( nonSimdWidth > 0 ) {
                const uint8_t * inX  = inY  + totalSimdWidth;
                uint8_t       * outX = outY + totalSimdWidth;

                const uint8_t * outXEnd = outX + nonSimdWidth;

                for( ; outX != outXEnd; ++outX, ++inX )
                    (*outX) = (*inX) < minThreshold || (*inX) > maxThreshold ? 0 : 255;
            }
        }
    }
#endif
}

namespace simd
{
    uint32_t getSimdSize( SIMDType simdType )
    {
        if ( simdType == avx_function )
            return avx::simdSize;
        if ( simdType == sse_function )
            return sse::simdSize;
        if ( simdType == neon_function )
            return neon::simdSize;
        if ( simdType == cpu_function )
            return 1u;

        return 0u;
    }

#ifdef PENGUINV_AVX_SET
#define AVX_CODE( code )          \
if ( simdType == avx_function ) { \
    code;                         \
    return;                       \
}
#else
#define AVX_CODE( code )
#endif

#ifdef PENGUINV_SSE_SET
#define SSE_CODE( code )          \
if ( simdType == sse_function ) { \
    code;                         \
    return;                       \
}

#ifdef PENGUINV_SSSE3_SET
#define SSSE3_CODE( code )        \
if ( simdType == sse_function ) { \
    code;                         \
    return;                       \
}
#else
#define	SSSE3_CODE( code )
#endif

#else
#define SSE_CODE( code )
#define SSSE3_CODE( code )
#endif

#ifdef PENGUINV_NEON_SET
#define NEON_CODE( code )          \
if ( simdType == neon_function ) { \
    code;                          \
    return;                        \
}
#else
#define NEON_CODE( code )
#endif

#define CPU_CODE( code )           \
if ( simdType == cpu_function ) {  \
    code;                          \
    return;                        \
}

#define SIMD_CHECK( code, condition )    \
if ( (condition) < simdSize ) {          \
    code;                                \
    return;                              \
}

    using namespace PenguinV_Image;

    void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                             Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
        width = width * colorCount;

        Image_Function::OptimiseRoi( width, height, in1, in2, out );

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        CPU_CODE( cpu::AbsoluteDifference( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width ); )
        SIMD_CHECK( cpu::AbsoluteDifference( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width );, width)

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::AbsoluteDifference( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::AbsoluteDifference( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::AbsoluteDifference( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void Accumulate( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector<uint32_t> & result, SIMDType simdType )
    {
        const uint32_t simdSize = getSimdSize( simdType );
        const uint8_t colorCount = image.colorCount();

        Image_Function::ParameterValidation( image, x, y, width, height );
        Image_Function::VerifyGrayScaleImage( image );
        width = width * colorCount;

        if( result.size() != width * height )
            throw imageException( "Array size is not equal to image ROI (width * height) size" );

        Image_Function::OptimiseRoi( width, height, image );

        const uint32_t rowSize = image.rowSize();

        const uint8_t * imageY    = image.data() + y * rowSize + x * colorCount;
        const uint8_t * imageYEnd = imageY + height * rowSize;

        uint32_t * outY = result.data();

        CPU_CODE( cpu::Accumulate( rowSize, imageY, imageYEnd, outY, result, width ); )
        SIMD_CHECK( cpu::Accumulate( rowSize, imageY, imageYEnd, outY, result, width );, width * height )

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::Accumulate( rowSize, imageY, imageYEnd, outY, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Accumulate( rowSize, imageY, imageYEnd, outY, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Accumulate( rowSize, imageY, imageYEnd, outY, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
        width = width * colorCount;

        Image_Function::OptimiseRoi( width, height, in1, in2, out );

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        CPU_CODE( cpu::BitwiseAnd( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width ); )
        SIMD_CHECK( cpu::BitwiseAnd( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width );, width )

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::BitwiseAnd( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::BitwiseAnd( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::BitwiseAnd( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                    Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
        width = width * colorCount;

        Image_Function::OptimiseRoi( width, height, in1, in2, out );

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        AVX_CODE( cpu::BitwiseOr( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width ); )
        SIMD_CHECK( cpu::BitwiseOr( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width );, width )

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::BitwiseOr( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::BitwiseOr( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::BitwiseOr( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                     Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
        width = width * colorCount;

        Image_Function::OptimiseRoi( width, height, in1, in2, out );

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        CPU_CODE( cpu::BitwiseXor( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width ); )
        SIMD_CHECK( cpu::BitwiseXor( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width );, width )

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::BitwiseXor( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::BitwiseXor( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::BitwiseXor( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void ConvertTo16Bit( const Image & in, uint32_t startXIn, uint32_t startYIn, Image16Bit & out, uint32_t startXOut, uint32_t startYOut,
                         uint32_t width, uint32_t height, SIMDType simdType )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );
        Image_Function::ParameterValidation( out, startXOut, startYOut, width, height );
        if ( in.colorCount() != out.colorCount() )
            throw imageException( "Color counts of images are different" );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t colorCount = in.colorCount();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint16_t      * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint16_t * outYEnd = outY + height * rowSizeOut;

        width = width * colorCount;

        cpu::ConvertTo16Bit( rowSizeIn, rowSizeOut, inY, outY, outYEnd, width );
    }

    void ConvertTo8Bit( const Image16Bit & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                        uint32_t width, uint32_t height, SIMDType simdType )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );
        Image_Function::ParameterValidation( out, startXOut, startYOut, width, height );
        if ( in.colorCount() != out.colorCount() )
            throw imageException( "Color counts of images are different" );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t colorCount = in.colorCount();

        const uint16_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t      * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        width = width * colorCount;

        cpu::ConvertTo8Bit( rowSizeIn, rowSizeOut, inY, outY, outYEnd, width );
    }

     void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                             uint32_t width, uint32_t height, SIMDType simdType )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( out );

        if( in.colorCount() == GRAY_SCALE ) {
            Image_Function::Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            return;
        }

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t colorCount = RGB;

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        cpu::ConvertToGrayScale( rowSizeIn, rowSizeOut, inY, outY, outYEnd, colorCount, width );
    }

    void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                       uint32_t width, uint32_t height, SIMDType simdType )
    {
        uint32_t simdSize = getSimdSize( simdType );
        if( simdType == neon_function ) // for neon, because the algorithm used work with packet of 64 bit
            simdSize = 8u;
        
        const uint8_t colorCount = RGB;

        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyRGBImage     ( out );

        if( in.colorCount() == RGB ) {
            Image_Function::Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
            return;
        }

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        if( (simdType == cpu_function) || (simdType == avx_function) || (width < simdSize) ) {
            AVX_CODE( ConvertToRgb( in, startXIn, startYIn, out, startXOut, startYOut, width, height, sse_function ); )

            cpu::ConvertToRgb( outY, outYEnd, inY, rowSizeOut, rowSizeIn, colorCount, width );
            return;
        }

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        SSSE3_CODE( sse::ConvertToRgb( outY, outYEnd, inY, rowSizeOut, rowSizeIn, colorCount, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::ConvertToRgb( outY, outYEnd, inY, rowSizeOut, rowSizeIn, colorCount, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
               uint32_t width, uint32_t height, SIMDType simdType )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

        const uint8_t colorCount  = Image_Function::CommonColorCount( in, out );
        width = width * colorCount;

        Image_Function::OptimiseRoi( width, height, in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        cpu::Copy( outY, outYEnd, inY, rowSizeOut, rowSizeIn, width );
    }

    void ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
                         uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId, SIMDType simdType )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( out );

        if( channelId >= in.colorCount() )
            throw imageException( "Channel ID for color image is greater than channel count in input image" );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t colorCount = in.colorCount();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn * colorCount + channelId;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        cpu::ExtractChannel( outY, outYEnd, inY, rowSizeOut, rowSizeIn, colorCount, width );
    }

    void Fill( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value, SIMDType simdType )
    {
        Image_Function::ParameterValidation( image, x, y, width, height );

        const uint8_t colorCount = image.colorCount();
        width = width * colorCount;

        Image_Function::OptimiseRoi( width, height, image );

        const uint32_t rowSize = image.rowSize();

        uint8_t * imageY = image.data() + y * rowSize + x * colorCount;
        const uint8_t * imageYEnd = imageY + height * rowSize;

        cpu::Fill( imageY, imageYEnd, rowSize, value, width );
    }

    void Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
               uint32_t width, uint32_t height, bool horizontal, bool vertical, SIMDType simdType )
    {
        uint32_t simdSize = getSimdSize( simdType );

        if( simdType == neon_function ) // for neon, because the algorithm used work with packet of 64 bit
            simdSize = 8u;

        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in, out );

        if( !horizontal && !vertical ) {
            Image_Function::Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        }
        else {
            const uint32_t rowSizeIn  = in.rowSize();
            const uint32_t rowSizeOut = out.rowSize();

            const uint8_t * inY    = in.data() + startYIn * rowSizeIn + startXIn;
            const uint8_t * inYEnd = inY + height * rowSizeIn;

            if( (simdType == cpu_function) || (simdType == avx_function) || (width < simdSize) ) {
                AVX_CODE( Flip( in, startXIn, startYIn, out, startXOut, startYOut, width, height, horizontal, vertical, sse_function ); )

                cpu::Flip( out, startXOut, startYOut, height, rowSizeIn, rowSizeOut, inY, inYEnd, horizontal, 
                           vertical, width );
                return;
            }

            const uint32_t simdWidth = width / simdSize;
            const uint32_t totalSimdWidth = simdWidth * simdSize;
            const uint32_t nonSimdWidth = width - totalSimdWidth;

            SSSE3_CODE( sse::Flip( out, startXOut, startYOut, width, height, rowSizeIn, rowSizeOut, inY, inYEnd, horizontal, 
                                   vertical, simdWidth, totalSimdWidth, nonSimdWidth ); )
            NEON_CODE( neon::Flip( out, startXOut, startYOut, width, height, rowSizeIn, rowSizeOut, inY, inYEnd, horizontal, 
                                   vertical, simdWidth, totalSimdWidth, nonSimdWidth ); )
        }
    }

    uint8_t GetPixel( const Image & image, uint32_t x, uint32_t y, SIMDType simdType )
    {
        if( image.empty() || x >= image.width() || y >= image.height() || image.colorCount() != GRAY_SCALE )
            throw imageException( "Position of point [x, y] is out of image" );

        return cpu::GetPixel( image, x, y );
    }

    void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                          uint32_t width, uint32_t height, double a, double gamma, SIMDType simdType )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

        if( a < 0 || gamma < 0 )
            throw imageException( "Gamma correction parameters are invalid" );

        // We precalculate all values and store them in lookup table
        std::vector < uint8_t > value( 256, 255u );

        cpu::GammaCorrection( value, a, gamma );

        Image_Function::LookupTable( in, startXIn, startYIn, out, startXOut, startYOut, width, height, value );
    }

    void Histogram( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & histogram, SIMDType simdType )
    {
        Image_Function::ParameterValidation( image, x, y, width, height );
        Image_Function::VerifyGrayScaleImage( image );
        Image_Function::OptimiseRoi( width, height, image );

        histogram.resize( 256u );
        std::fill( histogram.begin(), histogram.end(), 0u );

        const uint32_t rowSize = image.rowSize();

        const uint8_t * imageY = image.data() + y * rowSize + x;
        const uint8_t * imageYEnd = imageY + height * rowSize;

        cpu::Histogram( rowSize, imageY, imageYEnd, width, histogram );
    }

    void Histogram( const Image & image, uint32_t x, uint32_t y, const Image & mask, uint32_t maskX, uint32_t maskY, uint32_t width, uint32_t height,
                    std::vector < uint32_t > & histogram, SIMDType simdType )
    {
        Image_Function::ParameterValidation( image, x, y, mask, maskX, maskY, width, height );
        Image_Function::VerifyGrayScaleImage( image, mask );
        Image_Function::OptimiseRoi( width, height, image, mask );

        histogram.resize( 256u );
        std::fill( histogram.begin(), histogram.end(), 0u );

        const uint32_t rowSize     = image.rowSize();
        const uint32_t rowSizeMask = mask.rowSize();

        const uint8_t * imageY     = image.data() + y * rowSize + x;
        const uint8_t * imageYMask = mask.data() + maskY * rowSizeMask + maskX;
        const uint8_t * imageYEnd  = imageY + height * rowSize;

        cpu::Histogram( rowSize, rowSizeMask, imageY, imageYEnd, imageYMask, width, histogram );
    }

    void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                 uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in, out );
        const uint32_t simdSize = getSimdSize( simdType );

        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        width = width * colorCount;

        Image_Function::OptimiseRoi( width, height, in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        CPU_CODE( cpu::Invert( rowSizeIn, rowSizeOut, inY, outY, outYEnd, width ); )
        SIMD_CHECK( cpu::Invert( rowSizeIn, rowSizeOut, inY, outY, outYEnd, width );, width )

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::Invert( rowSizeIn, rowSizeOut, inY, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Invert( rowSizeIn, rowSizeOut, inY, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Invert( rowSizeIn, rowSizeOut, inY, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    bool IsBinary( const Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height, SIMDType simdType )
    {
        Image_Function::ParameterValidation( image, startX, startY, width, height );
        Image_Function::VerifyGrayScaleImage( image );

        const std::vector< uint32_t > histogram = Image_Function::Histogram( image, startX, startY, width, height );

        return cpu::IsBinary( histogram );
    }

    bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  uint32_t width, uint32_t height, SIMDType simdType )
    {
        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2 );
        width = width * colorCount;

        Image_Function::OptimiseRoi( width, height, in1, in2 );

        const uint32_t rowSize1  = in1.rowSize();
        const uint32_t rowSize2  = in2.rowSize();

        const uint8_t * in1Y = in1.data() + startY1 * rowSize1 + startX1 * colorCount;
        const uint8_t * in2Y = in2.data() + startY2 * rowSize2 + startX2 * colorCount;

        const uint8_t * in1YEnd = in1Y + height * rowSize1;

        return cpu::IsEqual( rowSize1, rowSize2, in1Y, in2Y, in1YEnd, width );
    }

    void LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                      uint32_t width, uint32_t height, const std::vector < uint8_t > & table, SIMDType simdType )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

        if( table.size() != 256u )
            throw imageException( "Lookup table size is not equal to 256" );

        const uint8_t colorCount = Image_Function::CommonColorCount( in, out );
        width = width * colorCount;

        Image_Function::OptimiseRoi( width, height, in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        cpu::LookupTable( rowSizeIn, rowSizeOut, inY, outY, outYEnd, table, width );
    }

    void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
        width = width * colorCount;

        Image_Function::OptimiseRoi( width, height, in1, in2, out );

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        CPU_CODE( cpu::Maximum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width ); )
        SIMD_CHECK( cpu::Maximum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width );, width )

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::Maximum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Maximum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Maximum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
                const Image & in3, uint32_t startXIn3, uint32_t startYIn3, Image & out, uint32_t startXOut, uint32_t startYOut,
                uint32_t width, uint32_t height, SIMDType simdType )
    {
        Image_Function::ParameterValidation ( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3, width, height );
        Image_Function::ParameterValidation ( out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in1, in2, in3 );
        Image_Function::VerifyRGBImage      ( out );

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

        cpu::Merge( rowSizeIn1, rowSizeIn2, rowSizeIn3, rowSizeOut, in1Y, in2Y, in3Y, outY, outYEnd, width );
    }

    void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
        width = width * colorCount;

        Image_Function::OptimiseRoi( width, height, in1, in2, out );

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        CPU_CODE( cpu::Minimum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width ); )
        SIMD_CHECK( cpu::Minimum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width );, width )

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::Minimum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Minimum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Minimum( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, SIMDType simdType )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

        const uint8_t colorCount = Image_Function::CommonColorCount( in, out );
        const uint32_t rowSizeIn = in.rowSize();

        const uint8_t * inY    = in.data()  + startYIn  * rowSizeIn + startXIn * colorCount;
        const uint8_t * inYEnd = inY + height * rowSizeIn;

        uint8_t minimum = 255;
        uint8_t maximum = 0;

        const uint32_t realWidth = width * colorCount;

        cpu::Normalize( rowSizeIn, inY, inYEnd, minimum, maximum, realWidth );

        if( (minimum == 0 && maximum == 255) || (minimum == maximum) ) {
            Image_Function::Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        }
        else {
            const double correction = 255.0 / (maximum - minimum);

            // We precalculate all values and store them in lookup table
            std::vector < uint8_t > value( 256 );

            for( uint16_t i = 0; i < 256; ++i )
                value[i] = static_cast <uint8_t>((i - minimum) * correction + 0.5);

            Image_Function::LookupTable( in, startXIn, startYIn, out, startXOut, startYOut, width, height, value );
        }
    }

    void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal,
                            std::vector < uint32_t > & projection, SIMDType simdType )
    {
        const uint32_t simdSize = getSimdSize( simdType );
        const uint8_t colorCount = image.colorCount();

        Image_Function::ParameterValidation( image, x, y, width, height );
        width = width * colorCount;

        projection.resize( horizontal ? width : height );
        std::fill( projection.begin(), projection.end(), 0u );
        uint32_t * out = projection.data();

        const uint32_t rowSize = image.rowSize();

        const uint8_t * imageStart = image.data() + y * rowSize + x * colorCount;

        CPU_CODE( cpu::ProjectionProfile( image, x, y, rowSize, colorCount, width, height, horizontal, projection ); )
        SIMD_CHECK( cpu::ProjectionProfile( image, x, y, rowSize, colorCount, width, height, horizontal, projection );, width * height )

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::ProjectionProfile( rowSize, imageStart, height, horizontal, out, simdWidth, totalSimdWidth, nonSimdWidth ) )
        SSE_CODE( sse::ProjectionProfile( rowSize, imageStart, height, horizontal, out, simdWidth, totalSimdWidth, nonSimdWidth ) )
        NEON_CODE( neon::ProjectionProfile( rowSize, imageStart, height, horizontal, out, simdWidth, totalSimdWidth, nonSimdWidth ) )
    }

    void ReplaceChannel( const Image & channel, uint32_t startXChannel, uint32_t startYChannel, Image & rgb, uint32_t startXRgb, uint32_t startYRgb,
                         uint32_t width, uint32_t height, uint8_t channelId, SIMDType simdType )
    {
        Image_Function::ParameterValidation( channel, startXChannel, startYChannel, rgb, startXRgb, startYRgb, width, height );
        Image_Function::VerifyGrayScaleImage( channel );
        Image_Function::VerifyRGBImage( rgb );

        if ( channelId >= RGB )
            throw imageException( "Channel ID is greater than number of channels in RGB image" );

        const uint32_t rowSizeIn  = channel.rowSize();
        const uint32_t rowSizeOut = rgb.rowSize();

        const uint8_t colorCount = RGB;

        const uint8_t * inY  = channel.data() + startYChannel * rowSizeIn  + startXChannel;
        uint8_t       * outY = rgb.data() + startYRgb * rowSizeOut + startXRgb * colorCount + channelId;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        width = width * colorCount;

        cpu::ReplaceChannel( outY, outYEnd, inY, rowSizeOut, rowSizeIn, colorCount, width );
    }

    void Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn, Image & out, 
                 uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut, SIMDType simdType )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, widthIn, heightIn );
        Image_Function::ParameterValidation( out, startXOut, startYOut, widthOut, heightOut );
        Image_Function::VerifyGrayScaleImage( in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + heightOut * rowSizeOut;

        uint32_t idY = 0;

        cpu::Resize(outY, outYEnd, inY, widthIn, rowSizeOut, rowSizeIn, idY, heightIn, widthOut, heightOut);
    }

    void RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                   uint32_t width, uint32_t height, SIMDType simdType )
    {
        uint32_t simdSize = getSimdSize( simdType );

        if( simdType == neon_function ) // for neon, because the algorithm used work with packet of 64 bit
            simdSize = 8u;

        const uint8_t colorCount = RGB;

        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyRGBImage     ( in, out );
        width = width * colorCount;

        Image_Function::OptimiseRoi( width, height, in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        CPU_CODE( cpu::RgbToBgr( outY, outYEnd, inY, rowSizeOut, rowSizeIn, colorCount, width); )
        SIMD_CHECK( cpu::RgbToBgr( outY, outYEnd, inY, rowSizeOut, rowSizeIn, colorCount, width);, width )

        const uint32_t rgbSimdSize = (simdSize / colorCount) * colorCount;
        const uint32_t simdWidth = width / rgbSimdSize;
        uint32_t totalSimdWidth = simdWidth * rgbSimdSize;
        uint32_t nonSimdWidth = width - totalSimdWidth;

        // to prevent unallowed access to memory
        if( nonSimdWidth < (simdSize % colorCount) ) {
            totalSimdWidth -= rgbSimdSize;
            nonSimdWidth += rgbSimdSize;
        }

        AVX_CODE( avx::RgbToBgr( outY, inY, outYEnd, rowSizeOut, rowSizeIn, colorCount, rgbSimdSize, totalSimdWidth, nonSimdWidth ); )
        SSSE3_CODE( sse::RgbToBgr( outY, inY, outYEnd, rowSizeOut, rowSizeIn, colorCount, rgbSimdSize, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::RgbToBgr( outY, inY, outYEnd, rowSizeOut, rowSizeIn, colorCount, rgbSimdSize, totalSimdWidth, nonSimdWidth ); )
    }

    void Rotate( const Image & in, double centerXIn, double centerYIn, Image & out, double centerXOut, double centerYOut, double angle,
                 SIMDType simdType )
    {
        Image_Function::ParameterValidation( in, out );
        Image_Function::VerifyGrayScaleImage( in, out );

        const double cosAngle = cos( angle );
        const double sinAngle = sin( angle );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint32_t width  = in.width();
        const uint32_t height = in.height();

        const uint8_t * inY  = in.data();
        uint8_t       * outY = out.data();
        const uint8_t * outYEnd = outY + height * rowSizeOut;

        double inXPos = -(cosAngle * centerXOut + sinAngle * centerYOut) + centerXIn;
        double inYPos = -(-sinAngle * centerXOut + cosAngle * centerYOut) + centerYIn;

        cpu::Rotate( outY, outYEnd, inY, rowSizeOut, rowSizeIn, cosAngle, sinAngle, inXPos, inYPos, height, width);
    }

    void SetPixel( Image & image, uint32_t x, uint32_t y, uint8_t value, SIMDType simdType )
    {
        if( image.empty() || x >= image.width() || y >= image.height() || image.colorCount() != GRAY_SCALE )
            throw imageException( "Position of point [x, y] is out of image" );

        cpu::SetPixel( image, x, y, value );
    }

    void SetPixel( Image & image, const std::vector < uint32_t > & X, const std::vector < uint32_t > & Y, uint8_t value, SIMDType simdType )
    {
        if( image.empty() || X.empty() || X.size() != Y.size() || image.colorCount() != GRAY_SCALE )
            throw imageException( "Bad input parameters in image function" );

        const uint32_t rowSize = image.rowSize();
        uint8_t * data = image.data();

        const uint32_t width  = image.width();
        const uint32_t height = image.height();

        cpu::SetPixel(image, X, Y, value, data, rowSize, height, width);
    }

    void Shift( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                uint32_t width, uint32_t height, double shiftX, double shiftY, SIMDType simdType )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in, out );

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
        
        cpu::Shift( outY, outYEnd, inY, rowSizeOut, rowSizeIn, coeff0, coeff1, coeff2, coeff3, emptyTopArea, emptyLeftArea, emptyRightArea,
                    emptyBottomArea, realHeight, realWidth, height, width );
    }

    void Split( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out1, uint32_t startXOut1, uint32_t startYOut1,
                Image & out2, uint32_t startXOut2, uint32_t startYOut2, Image & out3, uint32_t startXOut3, uint32_t startYOut3,
                uint32_t width, uint32_t height, SIMDType simdType )
    {
        Image_Function::ParameterValidation ( in, startXIn, startYIn, width, height );
        Image_Function::ParameterValidation ( out1, startXOut1, startYOut1, out2, startXOut2, startYOut2, out3, startXOut3, startYOut3, width, height );
        Image_Function::VerifyRGBImage      ( in );
        Image_Function::VerifyGrayScaleImage( out1, out2, out3 );

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

        cpu::Split(rowSizeOut1, rowSizeOut2, rowSizeOut3, rowSizeIn, inY, out1Y, out2Y, out3Y, inYEnd, width);
    }

    void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint8_t colorCount = Image_Function::CommonColorCount( in1, in2, out );
        const uint32_t simdSize = getSimdSize( simdType );

        Image_Function::ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
        width = width * colorCount;

        Image_Function::OptimiseRoi( width, height, in1, in2, out );

        const uint32_t rowSizeIn1 = in1.rowSize();
        const uint32_t rowSizeIn2 = in2.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * in1Y = in1.data() + startY1   * rowSizeIn1 + startX1   * colorCount;
        const uint8_t * in2Y = in2.data() + startY2   * rowSizeIn2 + startX2   * colorCount;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        CPU_CODE( cpu::Subtract( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width ); )
        SIMD_CHECK( cpu::Subtract( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, width );, width )

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::Subtract( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Subtract( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Subtract( rowSizeIn1, rowSizeIn2, rowSizeOut, in1Y, in2Y, outY, outYEnd, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, SIMDType simdType )
    {
        const uint32_t simdSize = getSimdSize( simdType );

        Image_Function::ParameterValidation( image, x, y, width, height );
        Image_Function::VerifyGrayScaleImage( image );

        Image_Function::OptimiseRoi( width, height, image );

        const uint32_t rowSize = image.rowSize();

        const uint8_t * imageY    = image.data() + y * rowSize + x;
        const uint8_t * imageYEnd = imageY + height * rowSize;

        if( (simdType == cpu_function) || (width < simdSize) ) {
            #ifdef PENGUINV_AVX_SET
            if ( simdType == avx_function )
                return Sum( rowSize, imageY, imageYEnd, width );
            #endif

            return cpu::Sum( rowSize, imageY, imageYEnd, width );
        }

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        #ifdef PENGUINV_AVX_SET
        if ( simdType == avx_function )
            return avx::Sum( rowSize, imageY, imageYEnd, simdWidth, totalSimdWidth, nonSimdWidth );
        #endif
        #ifdef PENGUINV_SSE_SET
        if ( simdType == sse_function )
            return sse::Sum( rowSize, imageY, imageYEnd, simdWidth, totalSimdWidth, nonSimdWidth );
        #endif
        #ifdef PENGUINV_NEON_SET
        if (simdType == neon_function)
            return neon::Sum( rowSize, imageY, imageYEnd, simdWidth, totalSimdWidth, nonSimdWidth );
        #endif

        return 0u;
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t threshold, SIMDType simdType )
    {
        const uint32_t simdSize = getSimdSize( simdType );

        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in, out );

        Image_Function::OptimiseRoi( width, height, in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        CPU_CODE( cpu::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, threshold, width ); )
        SIMD_CHECK( cpu::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, threshold, width );, width )

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, threshold, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, threshold, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, threshold, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold, SIMDType simdType )
    {
        const uint32_t simdSize = getSimdSize( simdType );

        Image_Function::ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
        Image_Function::VerifyGrayScaleImage( in, out );

        Image_Function::OptimiseRoi( width, height, in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + height * rowSizeOut;

        CPU_CODE( cpu::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, minThreshold, maxThreshold, width ); )
        SIMD_CHECK( cpu::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, minThreshold, maxThreshold, width );, width )

        const uint32_t simdWidth = width / simdSize;
        const uint32_t totalSimdWidth = simdWidth * simdSize;
        const uint32_t nonSimdWidth = width - totalSimdWidth;

        AVX_CODE( avx::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, minThreshold, maxThreshold, simdWidth, totalSimdWidth, nonSimdWidth ); )
        SSE_CODE( sse::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, minThreshold, maxThreshold, simdWidth, totalSimdWidth, nonSimdWidth ); )
        NEON_CODE( neon::Threshold( rowSizeIn, rowSizeOut, inY, outY, outYEnd, minThreshold, maxThreshold, simdWidth, totalSimdWidth, nonSimdWidth ); )
    }

    void Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
                    uint32_t width, uint32_t height, SIMDType simdType )
    {
        Image_Function::ParameterValidation( in, startXIn, startYIn, width, height );
        Image_Function::ParameterValidation( out, startXOut, startYOut, height, width );
        Image_Function::VerifyGrayScaleImage( in, out );

        const uint32_t rowSizeIn  = in.rowSize();
        const uint32_t rowSizeOut = out.rowSize();

        const uint8_t * inX  = in.data()  + startYIn  * rowSizeIn  + startXIn;
        uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

        const uint8_t * outYEnd = outY + width * rowSizeOut;

        cpu::Transpose( rowSizeIn, rowSizeOut, inX, outY, outYEnd, height, width );
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
        simd::AbsoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
    }

    void Accumulate( const Image & image, std::vector < uint32_t > & result )
    {
        Image_Function_Helper::Accumulate( Accumulate, image, result );
    }

    void Accumulate( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & result )
    {
        simd::Accumulate(image, x, y, width, height, result, simd::actualSimdType());
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
        simd::BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::ConvertTo16Bit( in, startXIn, startYIn, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::ConvertTo8Bit( in, startXIn, startYIn, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
       simd::ConvertToGrayScale( in, startXIn, startYIn, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::ConvertToRgb( in, startXIn, startYIn, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::ExtractChannel( in, startXIn, startYIn, out, startXOut, startYOut, width, height, channelId, simd::actualSimdType() );
    }

    void Fill( Image & image, uint8_t value )
    {
        image.fill( value );
    }

    void Fill( Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t value )
    {
        simd::Fill( image, x, y, width, height, value, simd::actualSimdType() );
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
        simd::Flip(in, startXIn, startYIn, out, startXOut, startYOut, width, height, horizontal, vertical, simd::actualSimdType());
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
        simd::GammaCorrection( in, startXIn, startYIn, out, startXOut, startYOut, width, height, a, gamma, simd::actualSimdType() );
    }

    uint8_t GetPixel( const Image & image, uint32_t x, uint32_t y )
    {
        return cpu::GetPixel( image, x, y );
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
        simd::Histogram( image, x, y, width, height, histogram, simd::actualSimdType() );
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
        simd::Histogram( image, x, y, mask, maskX, maskY, width, height, histogram, simd::actualSimdType() );
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
        simd::Invert( in, startXIn, startYIn, out, startXOut, startYOut, width, height, simd::actualSimdType() );
    }

    bool IsBinary( const Image & image )
    {
        ParameterValidation( image );

        return IsBinary( image, 0u, 0u, image.width(), image.height() );
    }

    bool IsBinary( const Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height )
    {
        return simd::IsBinary( image, startX, startY, width, height, simd::actualSimdType() );
    }

    bool IsEqual( const Image & in1, const Image & in2 )
    {
        ParameterValidation( in1, in2 );

        return IsEqual( in1, 0, 0, in2, 0, 0, in1.width(), in1.height() );
    }

    bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
                  uint32_t width, uint32_t height )
    {
        return simd::IsEqual( in1, startX1, startY1, in2, startX2, startY2, width, height, simd::actualSimdType() );
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
        simd::LookupTable( in, startXIn, startYIn, out, startXOut, startYOut, width, height, table, simd::actualSimdType() );
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
        simd::Maximum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::Merge( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3, out,
                    startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::Minimum( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
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
        simd::Normalize( in, startXIn, startYIn, out, startXOut, startYOut, width, height, simd::actualSimdType() );
    }

    std::vector < uint32_t > ProjectionProfile( const Image & image, bool horizontal )
    {
        return Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, horizontal );
    }
    void ProjectionProfile( const Image & image, bool horizontal, std::vector < uint32_t > & projection )
    {
        Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, horizontal, projection );
    }
    std::vector < uint32_t > ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal )
    {
        return Image_Function_Helper::ProjectionProfile( ProjectionProfile, image, x, y, width, height, horizontal );
    }
    void ProjectionProfile( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool horizontal, 
                            std::vector < uint32_t > & projection )
    {
        simd::ProjectionProfile( image, x, y, width, height, horizontal, projection, simd::actualSimdType() );
    }

    void ReplaceChannel( const Image & channel, Image & rgb, uint8_t channelId )
    {
        ParameterValidation( channel, rgb );

        ReplaceChannel( channel, 0, 0, rgb, 0, 0, channel.width(), channel.height(), channelId );
    }

    void ReplaceChannel( const Image & channel, uint32_t startXChannel, uint32_t startYChannel, Image & rgb, uint32_t startXRgb, uint32_t startYRgb,
                         uint32_t width, uint32_t height, uint8_t channelId )
    {
        simd::ReplaceChannel( channel, startXChannel, startYChannel, rgb, startXRgb, startYRgb, width, height, channelId, simd::actualSimdType() );
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
        simd::Resize( in, startXIn, startYIn, widthIn, heightIn, out, startXOut, startYOut, widthOut, heightOut, simd::actualSimdType() );
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
        simd::RgbToBgr( in, startXIn, startYIn, out, startXOut, startYOut, width, height, simd::actualSimdType() );
    }

    void Rotate( const Image & in, double centerXIn, double centerYIn, Image & out, double centerXOut, double centerYOut, double angle )
    {
        simd::Rotate( in, centerXIn, centerYIn, out, centerXOut, centerYOut, angle, simd::actualSimdType() );
    }

    void SetPixel( Image & image, uint32_t x, uint32_t y, uint8_t value )
    {
        simd::SetPixel( image, x, y, value, simd::actualSimdType() );
    }

    void SetPixel( Image & image, const std::vector < uint32_t > & X, const std::vector < uint32_t > & Y, uint8_t value )
    {
        simd::SetPixel( image, X, Y, value, simd::actualSimdType() );
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
        simd::Shift( in, startXIn, startYIn, out, startXOut, startYOut, width, height, shiftX, shiftY, simd::actualSimdType() );
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
        simd::Split( in, startXIn, startYIn, out1, startXOut1, startYOut1, out2, startXOut2, startYOut2,
                     out3, startXOut3, startYOut3, width, height, simd::actualSimdType() );
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
        simd::Subtract( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height, simd::actualSimdType() );
    }

    uint32_t Sum( const Image & image )
    {
        return Sum( image, 0, 0, image.width(), image.height() );
    }

    uint32_t Sum( const Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height )
    {
        return simd::Sum( image, x, y, width, height, simd::actualSimdType() );
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
        simd::Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, threshold, simd::actualSimdType() );
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
        simd::Threshold( in, startXIn, startYIn, out, startXOut, startYOut, width, height, minThreshold, maxThreshold, simd::actualSimdType() );
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
        simd::Transpose( in, startXIn, startYIn, out, startXOut, startYOut, width, height, simd::actualSimdType() );
    }
}