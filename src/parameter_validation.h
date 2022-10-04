/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2022                                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#pragma once

#include <limits>
#include <utility>

namespace Image_Function
{
    template <typename TImage>
    uint8_t CheckCommonColorCount( const TImage & image1, const TImage & image2 )
    {
        if ( image1.colorCount() != image2.colorCount() )
            throw penguinVException( "The number of color channels in images is different" );

        return image1.colorCount();
    }

    template <typename TImage>
    uint8_t CheckCommonColorCount( const TImage & image1, const TImage & image2, const TImage & image3 )
    {
        if ( image1.colorCount() != image2.colorCount() || image1.colorCount() != image3.colorCount() )
            throw penguinVException( "The number of color channels in images is different" );

        return image1.colorCount();
    }

    template <typename TImage>
    bool IsCorrectColorCount( const TImage & image )
    {
        return image.colorCount() == penguinV::GRAY_SCALE || image.colorCount() == penguinV::RGB || image.colorCount() == penguinV::RGBA;
    }

    template <typename TImage>
    void VerifyRGBImage( const TImage & image )
    {
        if ( image.colorCount() != penguinV::RGB )
            throw penguinVException( "Bad input parameters in image function: colored image has different than 3 color channels" );
    }

    template <typename TImage>
    void VerifyRGBAImage( const TImage & image )
    {
        if ( image.colorCount() != penguinV::RGBA )
            throw penguinVException( "Bad input parameters in image function: colored image has different than 4 color channels" );
    }

    template <typename TImage, typename... Args>
    void VerifyRGBImage( const TImage & image, Args... args )
    {
        VerifyRGBImage( image );
        VerifyRGBImage( args... );
    }

    template <typename TImage>
    void VerifyGrayScaleImage( const TImage & image )
    {
        if ( image.colorCount() != penguinV::GRAY_SCALE )
            throw penguinVException( "Bad input parameters in image function: gray-scaled image has more than 1 color channels" );
    }

    template <typename TImage, typename... Args>
    void VerifyGrayScaleImage( const TImage & image, Args... args )
    {
        VerifyGrayScaleImage( image );
        VerifyGrayScaleImage( args... );
    }

    template <typename TImage>
    void ValidateImageParameters( const TImage & image1 )
    {
        if ( image1.empty() || !IsCorrectColorCount( image1 ) )
            throw penguinVException( "Bad input parameters in image function" );
    }

    template <typename TImage>
    void ValidateImageParameters( const TImage & image1, const TImage & image2 )
    {
        if ( image1.empty() || image2.empty() || !IsCorrectColorCount( image1 ) || !IsCorrectColorCount( image2 ) || image1.width() != image2.width()
             || image1.height() != image2.height() )
            throw penguinVException( "Bad input parameters in image function" );
    }

    template <typename TImage, typename... Args>
    void ValidateImageParameters( const TImage & image1, const TImage & image2, Args... args )
    {
        ValidateImageParameters( image1, image2 );
        ValidateImageParameters( image2, args... );
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
    void ValidateImageParameters( const TImage & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height )
    {
        if ( image.empty() || !IsCorrectColorCount( image ) || width == 0 || height == 0 || startX + width > image.width() || startY + height > image.height()
             || startX + width < width || startY + height < height )
            throw penguinVException( "Bad input parameters in image function" );
    }

    template <typename TImage, typename... Args>
    void ValidateImageParameters( const TImage & image1, uint32_t startX1, uint32_t startY1, Args... args )
    {
        const std::pair<uint32_t, uint32_t> & dimensions = ExtractRoiSize( args... );

        ValidateImageParameters( image1, startX1, startY1, dimensions.first, dimensions.second );
        ValidateImageParameters( args... );
    }

    template <typename TImage>
    bool IsFullImageRow( uint32_t width, const TImage & image )
    {
        return image.rowSize() == width;
    }

    template <typename TImage, typename... Args>
    bool IsFullImageRow( uint32_t width, const TImage & image, Args... args )
    {
        if ( !IsFullImageRow( width, image ) )
            return false;

        return IsFullImageRow( width, args... );
    }

    template <typename TImage, typename... Args>
    void OptimiseRoi( uint32_t & width, uint32_t & height, const TImage & image, Args... args )
    {
        if ( IsFullImageRow( width, image, args... ) && ( width < ( std::numeric_limits<uint32_t>::max() / height ) ) ) {
            width = width * height;
            height = 1u;
        }
    }
}
