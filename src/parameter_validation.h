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
        return image.colorCount() == PenguinV_Image::GRAY_SCALE || image.colorCount() == PenguinV_Image::RGB;
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
    void VerifyGrayScaleImage( const TImage & image )
    {
        if( image.colorCount() != PenguinV_Image::GRAY_SCALE )
            throw imageException( "Bad input parameters in image function: gray-scaled image has more than 1 color channels" );
    }

    template <typename TImage, typename... Args>
    void VerifyGrayScaleImage( const TImage & image, Args... args)
    {
        VerifyGrayScaleImage( image );
        VerifyGrayScaleImage( args... );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image1 )
    {
        if( image1.empty() || !IsCorrectColorCount( image1 ) )
            throw imageException( "Bad input parameters in image function" );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image1, const TImage & image2 )
    {
        if( image1.empty() || image2.empty() || !IsCorrectColorCount( image1 ) || !IsCorrectColorCount( image2 ) ||
            image1.width() != image2.width() || image1.height() != image2.height() )
            throw imageException( "Bad input parameters in image function" );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image1, const TImage & image2, const TImage & image3 )
    {
        if( image1.empty() || image2.empty() || image3.empty() || !IsCorrectColorCount( image1 ) || !IsCorrectColorCount( image2 ) ||
            !IsCorrectColorCount( image3 ) || image1.width() != image2.width() || image1.height() != image2.height() ||
            image1.width() != image3.width() || image1.height() != image3.height() )
            throw imageException( "Bad input parameters in image function" );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height )
    {
        if( image.empty() || !IsCorrectColorCount( image ) || width == 0 || height == 0 || startX + width > image.width() || startY + height > image.height() )
            throw imageException( "Bad input parameters in image function" );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image1, uint32_t startX1, uint32_t startY1,
                              const TImage & image2, uint32_t startX2, uint32_t startY2,
                              uint32_t width, uint32_t height )
    {
        if( image1.empty() || image2.empty() || !IsCorrectColorCount( image1 ) || !IsCorrectColorCount( image2 ) || width == 0 || height == 0 ||
            startX1 + width > image1.width() || startY1 + height > image1.height() ||
            startX2 + width > image2.width() || startY2 + height > image2.height() )
            throw imageException( "Bad input parameters in image function" );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image1, uint32_t startX1, uint32_t startY1,
                              const TImage & image2, uint32_t startX2, uint32_t startY2,
                              const TImage & image3, uint32_t startX3, uint32_t startY3,
                              uint32_t width, uint32_t height )
    {
        if( image1.empty() || image2.empty() || image3.empty() || !IsCorrectColorCount( image1 ) || !IsCorrectColorCount( image2 ) ||
            !IsCorrectColorCount( image3 ) || width == 0 || height == 0 ||
            startX1 + width > image1.width() || startY1 + height > image1.height() ||
            startX2 + width > image2.width() || startY2 + height > image2.height() ||
            startX3 + width > image3.width() || startY3 + height > image3.height() )
            throw imageException( "Bad input parameters in image function" );
    }
}
