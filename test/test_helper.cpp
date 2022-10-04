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

#include "test_helper.h"
#include "../src/parameter_validation.h"

namespace
{
    uint32_t randomSize()
    {
        return Test_Helper::randomValue<uint32_t>( 1, 2048 );
    }

    template <typename _Type>
    penguinV::ImageTemplate<_Type> generateImage( uint32_t width, uint32_t height, uint8_t colorCount, _Type value,
                                                  const penguinV::ImageTemplate<_Type> & reference = penguinV::ImageTemplate<_Type>() )
    {
        penguinV::ImageTemplate<_Type> image = reference.generate( width, height, colorCount );

        image.fill( value );

        return image;
    }

    void fillRandomData( penguinV::Image & image )
    {
        uint32_t height = image.height();
        uint32_t width = image.width() * image.colorCount();
        Image_Function::OptimiseRoi( width, height, image );

        const uint32_t rowSize = image.rowSize();
        uint8_t * outY = image.data();
        const uint8_t * outYEnd = outY + height * rowSize;

        for ( ; outY != outYEnd; outY += rowSize ) {
            uint8_t * outX = outY;
            const uint8_t * outXEnd = outX + width;

            for ( ; outX != outXEnd; ++outX )
                ( *outX ) = Test_Helper::randomValue<uint8_t>( 256 );
        }
    }

    uint32_t testRunCount = 100; // some magic number for loop. Higher value = higher chance to verify all possible situations
}

namespace Test_Helper
{
    penguinV::Image uniformImage( uint32_t width, uint32_t height, const penguinV::Image & reference )
    {
        return uniformImage( randomValue<uint8_t>( 256 ), width, height, reference );
    }

    penguinV::Image uniformImage( uint8_t value, uint32_t width, uint32_t height, const penguinV::Image & reference )
    {
        return generateImage<uint8_t>( ( width > 0u ) ? width : randomSize(), ( height > 0u ) ? height : randomSize(), penguinV::GRAY_SCALE, value, reference );
    }

    penguinV::Image16Bit uniformImage16Bit( uint16_t value, uint32_t width, uint32_t height, const penguinV::Image16Bit & reference )
    {
        return generateImage<uint16_t>( ( width > 0u ) ? width : randomSize(), ( height > 0u ) ? height : randomSize(), penguinV::GRAY_SCALE, value, reference );
    }

    penguinV::Image uniformRGBImage( const penguinV::Image & reference )
    {
        return uniformRGBImage( randomValue<uint8_t>( 256 ), reference );
    }

    penguinV::Image uniformRGBImage( uint8_t value, const penguinV::Image & reference )
    {
        return generateImage<uint8_t>( randomSize(), randomSize(), penguinV::RGB, value, reference );
    }

    penguinV::Image uniformRGBImage( uint32_t width, uint32_t height )
    {
        return uniformRGBImage( width, height, randomValue<uint8_t>( 256 ) );
    }

    penguinV::Image uniformRGBImage( uint32_t width, uint32_t height, uint8_t value )
    {
        return generateImage<uint8_t>( width, height, penguinV::RGB, value );
    }

    penguinV::Image uniformRGBAImage( uint32_t width, uint32_t height )
    {
        return uniformRGBAImage( width, height, randomValue<uint8_t>( 256 ) );
    }

    penguinV::Image uniformRGBAImage( uint32_t width, uint32_t height, uint8_t value )
    {
        return generateImage<uint8_t>( width, height, penguinV::RGBA, value );
    }

    penguinV::Image uniformRGBAImage( const penguinV::Image & reference )
    {
        return uniformRGBAImage( randomValue<uint8_t>( 256 ), reference );
    }

    penguinV::Image uniformRGBAImage( uint8_t value, const penguinV::Image & reference )
    {
        return generateImage<uint8_t>( randomSize(), randomSize(), penguinV::RGBA, value, reference );
    }

    std::vector<penguinV::Image> uniformImages( uint32_t count, uint32_t width, uint32_t height )
    {
        std::vector<penguinV::Image> image( count );

        for ( std::vector<penguinV::Image>::iterator im = image.begin(); im != image.end(); ++im )
            *im = uniformImage( width, height );

        return image;
    }

    std::vector<penguinV::Image> uniformRGBImages( uint32_t count, uint32_t width, uint32_t height )
    {
        std::vector<penguinV::Image> image( count );

        for ( std::vector<penguinV::Image>::iterator im = image.begin(); im != image.end(); ++im )
            *im = uniformRGBImage( width, height );

        return image;
    }

    std::vector<penguinV::Image> uniformImages( uint32_t images, const penguinV::Image & reference )
    {
        if ( images == 0 )
            throw penguinVException( "Invalid parameter: number of images is 0" );

        std::vector<uint8_t> intesity( images );
        for ( size_t i = 0u; i < intesity.size(); ++i )
            intesity[i] = randomValue<uint8_t>( 256 );

        return uniformImages( intesity, reference );
    }

    std::vector<penguinV::Image> uniformImages( const std::vector<uint8_t> & intensityValue, const penguinV::Image & reference )
    {
        if ( intensityValue.size() == 0 )
            throw penguinVException( "Invalid parameter" );

        std::vector<penguinV::Image> image;

        image.push_back( uniformImage( intensityValue[0], 0, 0, reference ) );

        image.resize( intensityValue.size() );

        for ( size_t i = 1u; i < image.size(); ++i ) {
            image[i] = reference.generate( image[0].width(), image[0].height() );
            image[i].fill( intensityValue[i] );
        }

        return image;
    }

    penguinV::Image randomImage( uint32_t width, uint32_t height )
    {
        penguinV::Image image( ( width == 0 ) ? randomSize() : width, ( height == 0 ) ? randomSize() : height );

        fillRandomData( image );

        return image;
    }

    penguinV::Image randomRGBImage( const penguinV::Image & reference )
    {
        penguinV::Image image = reference.generate( randomSize(), randomSize(), penguinV::RGB );

        fillRandomData( image );

        return image;
    }

    penguinV::Image randomImage( const std::vector<uint8_t> & value )
    {
        if ( value.empty() )
            return randomImage();

        penguinV::Image image( randomSize(), randomSize() );

        uint32_t height = image.height();
        uint32_t width = image.width();

        const size_t valueSize = value.size();

        if ( valueSize <= width && ( width % static_cast<uint32_t>( valueSize ) ) == 0 )
            Image_Function::OptimiseRoi( width, height, image );

        const uint32_t rowSize = image.rowSize();
        uint8_t * outY = image.data();
        const uint8_t * outYEnd = outY + height * rowSize;

        size_t id = 0;

        for ( ; outY != outYEnd; outY += rowSize ) {
            uint8_t * outX = outY;
            const uint8_t * outXEnd = outX + width;

            for ( ; outX != outXEnd; ++outX ) {
                ( *outX ) = value[id++];
                if ( id == valueSize )
                    id = 0u;
            }
        }

        return image;
    }

    uint32_t runCount()
    {
        return testRunCount;
    }

    void setRunCount( int argc, char * argv[], uint32_t count )
    {
        testRunCount = count;
        if ( argc >= 2 ) {
            const int testCount = std::atoi( argv[1] );
            if ( testCount > 0 )
                testRunCount = static_cast<uint32_t>( testCount );
        }
    }
}
