#pragma once

#include <limits>
#include <utility>

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
        return image.colorCount() == PenguinV::GRAY_SCALE || image.colorCount() == PenguinV::RGB || image.colorCount() == PenguinV::RGBA;
    }

    template <typename TImage>
    void VerifyRGBImage( const TImage & image )
    {
        if( image.colorCount() != PenguinV::RGB )
            throw imageException( "Bad input parameters in image function: colored image has different than 3 color channels" );
    }

    template <typename TImage>
    void VerifyRGBAImage( const TImage & image )
    {
        if( image.colorCount() != PenguinV::RGBA )
            throw imageException( "Bad input parameters in image function: colored image has different than 4 color channels" );
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
        if( image.colorCount() != PenguinV::GRAY_SCALE )
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

    template <typename TImage, typename... Args>
    void ParameterValidation( const TImage & image1, const TImage & image2, Args... args )
    {
        ParameterValidation( image1, image2 );
        ParameterValidation( image2, args... );
    }

    template <typename _Type>
    std::pair<_Type, _Type> ExtractRoiSize( _Type width, _Type height )
    {
        return std::pair<_Type, _Type>( width, height );
    }

    template <typename TImage, typename... Args>
    std::pair<uint32_t, uint32_t> ExtractRoiSize( const TImage &, uint32_t, uint32_t, Args... args )
    {
        return ExtractRoiSize( args... );
    }

    template <typename TImage>
    void ParameterValidation( const TImage & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height )
    {
        if( image.empty() || !IsCorrectColorCount( image ) || width == 0 || height == 0 || startX + width > image.width() || startY + height > image.height() ||
            startX + width < width || startY + height < height )
            throw imageException( "Bad input parameters in image function" );
    }

    template <typename TImage, typename... Args>
    void ParameterValidation( const TImage & image1, uint32_t startX1, uint32_t startY1, Args... args )
    {
        const std::pair<uint32_t, uint32_t> & dimensions = ExtractRoiSize( args... );

        ParameterValidation( image1, startX1, startY1, dimensions.first, dimensions.second );
        ParameterValidation( args... );
    }

    template <typename TImage>
    bool IsFullImageRow( uint32_t width, const TImage & image )
    {
        return image.rowSize() == width;
    }

    template <typename TImage, typename... Args>
    bool IsFullImageRow( uint32_t width, const TImage & image, Args... args )
    {
        if( !IsFullImageRow( width, image ) )
            return false;

        return IsFullImageRow( width, args... );
    }

    template <typename TImage, typename... Args>
    void OptimiseRoi( uint32_t & width, uint32_t & height, const TImage & image, Args... args )
    {
        if( IsFullImageRow(width, image, args...) && ( width < (std::numeric_limits<uint32_t>::max() / height) ) ) {
            width = width * height;
            height = 1u;
        }
    }
}
