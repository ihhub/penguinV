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
#include "../image_buffer.h"
#include "../parameter_validation.h"
#include <fstream>

namespace Raw_Operation
{
    template <typename _Type>
    penguinV::ImageTemplate<_Type> Load( const std::string & path, uint32_t width, uint32_t height, uint8_t colorCount )
    {
        if ( path.empty() || width == 0 || height == 0 || colorCount == 0 )
            throw penguinVException( "Incorrect parameters for raw image loading" );

        std::fstream file;
        file.open( path, std::fstream::in | std::fstream::binary );

        if ( !file )
            return penguinV::ImageTemplate<_Type>();

        file.seekg( 0, file.end );
        std::streamoff length = file.tellg();

        const uint32_t overallImageSize = width * height * colorCount * static_cast<uint32_t>( sizeof( _Type ) );

        if ( length != overallImageSize )
            return penguinV::ImageTemplate<_Type>();

        file.seekg( 0, file.beg );

        penguinV::ImageTemplate<_Type> image( width, height, colorCount );

        size_t dataToRead = overallImageSize;
        size_t dataReaded = 0;

        const size_t blockSize = 4 * 1024 * 1024; // read by 4 MB blocks

        char * output = reinterpret_cast<char *>( image.data() );
        while ( dataToRead > 0 ) {
            size_t readSize = dataToRead > blockSize ? blockSize : dataToRead;

            file.read( output + dataReaded, static_cast<std::streamsize>( readSize ) );

            dataReaded += readSize;
            dataToRead -= readSize;
        }

        return image;
    }

    template <typename _Type>
    void Save( const std::string & path, const penguinV::ImageTemplate<_Type> & image )
    {
        Image_Function::ValidateImageParameters( image );

        std::fstream file;
        file.open( path, std::fstream::out | std::fstream::trunc | std::fstream::binary );

        if ( !file )
            throw penguinVException( "Cannot create file for saving" );

        size_t dataToWrite = sizeof( _Type ) * image.rowSize() * image.height();
        size_t dataWritten = 0;

        const size_t blockSize = 4 * 1024 * 1024; // read by 4 MB blocks

        const char * output = reinterpret_cast<const char *>( image.data() );
        while ( dataToWrite > 0 ) {
            size_t writeSize = dataToWrite > blockSize ? blockSize : dataToWrite;

            file.write( output + dataWritten, static_cast<std::streamsize>( writeSize ) );
            file.flush();

            dataWritten += writeSize;
            dataToWrite -= writeSize;
        }

        if ( !file )
            throw penguinVException( "failed to write data into file" );
    }

    template <typename _Type>
    void LittleEndianToBigEndian( penguinV::ImageTemplate<_Type> & image )
    {
        Image_Function::ValidateImageParameters( image );
        if ( sizeof( _Type ) < 2 )
            return;

        const size_t stepSize = sizeof( _Type );
        _Type * data = image.data();
        const _Type * end = data + image.rowSize() * image.height();
        for ( ; data != end; ++data ) {
            uint8_t * left = reinterpret_cast<uint8_t *>( data );
            uint8_t * right = left + stepSize - 1;
            for ( ; left < right; ++left, --right ) {
                uint8_t temp = *left;
                *left = *right;
                *right = temp;
            }
        }
    }
}
