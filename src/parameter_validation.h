#pragma once
#include "image_buffer.h"

namespace Image_Function
{
    template <typename TImage>
    uint8_t CommonColorCount( const TImage & image1, const TImage & image2 )
    {
        if( image1.colorCount() != image2.colorCount() )
            throw imageException( "Color counts of images are different" );

        return image1.colorCount();
    }

    template <typename TImage>
    uint8_t CommonColorCount( const TImage & image1, const TImage & image2, const TImage & image3 )
    {
        if( image1.colorCount() != image2.colorCount() || image1.colorCount() != image3.colorCount() )
            throw imageException( "Color counts of images are different" );

        return image1.colorCount();
    }

    template <typename TImage>
    bool IsCorrectColorCount( const TImage & image )
    {
        return image.colorCount() == PenguinV_Image::GRAY_SCALE || image.colorCount() == PenguinV_Image::RGB || image.colorCount() == PenguinV_Image::RGBA;
    }

    template <typename TImage>
    void VerifyGenericImage( const TImage & image )
    {
        if ( image.empty())
            throw imageException( "Bad input parameters in image function: image is empty" );
        if ( !IsCorrectColorCount( image ))
            throw imageException( "Bad input parameters in image function: image has invalid color count" );
    }

    template <typename TImage, typename... Args>
    void VerifyGenericImage( const TImage & image, Args... args)
    {
        VerifyGenericImage( image );
        VerifyGenericImage( args... );
    }

    template <typename TImage>
    void VerifyRGBImage( const TImage & image )
    {
        if( image.colorCount() != PenguinV_Image::RGB )
            throw imageException( "Bad input parameters in image function: colored image has different than 3 color channels" );
    }

    template <typename TImage, typename... Args>
    void VerifyRGBImage( const TImage & image, Args... args)
    {
        VerifyRGBImage( image );
        VerifyRGBImage( args... );
    }

    template <typename TImage>
    void VerifyRGBAImage( const TImage & image )
    {
        if( image.colorCount() != PenguinV_Image::RGBA )
            throw imageException( "Bad input parameters in image function: RGBA image has different than 4 color channels" );
    }

    template <typename TImage, typename... Args>
    void VerifyRGBAImage( const TImage & image, Args... args)
    {
        VerifyRGBAImage( image );
        VerifyRGBAImage( args... );
    }

    template <typename TImage>
    void VerifyGrayScaleImage( const TImage & image1 )
    {
        if( image1.colorCount() != PenguinV_Image::GRAY_SCALE )
            throw imageException( "Bad input parameters in image function: gray-scaled image has more than 1 color channels" );
    }

    template <typename TImage, typename... Args>
    void VerifyGrayScaleImage( const TImage & image, Args... args)
    {
        VerifyGrayScaleImage( image );
        VerifyGrayScaleImage( args... );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image )
    {
        VerifyGenericImage( image );
    }

    template <typename TImage, typename... Args>
    void ParameterValidation( const TImage & image, Args... args)
    {
        ParameterValidation( image );
        ParameterValidation( args... );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height )
    {
        VerifyGenericImage( image );
        if( width == 0 || height == 0 || startX + width > image.width() || startY + height > image.height() )
            throw imageException( "Bad input parameters in image function: invalid image section size" );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image1, uint32_t startX1, uint32_t startY1,
                              const TImage & image2, uint32_t startX2, uint32_t startY2,
                              uint32_t width, uint32_t height )
    {
        ParameterValidation ( image1, startX1, startY1, width, height );
        ParameterValidation ( image2, startX2, startY2, width, height );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image1, uint32_t startX1, uint32_t startY1,
                              const TImage & image2, uint32_t startX2, uint32_t startY2,
                              const TImage & image3, uint32_t startX3, uint32_t startY3,
                              uint32_t width, uint32_t height )
    {
        ParameterValidation ( image1, startX1, startY1, width, height );
        ParameterValidation ( image2, startX2, startY2, width, height );
        ParameterValidation ( image3, startX3, startY3, width, height );
    }
}
