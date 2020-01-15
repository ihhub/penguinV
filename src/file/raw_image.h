#pragma once
#include <fstream>
#include "../image_buffer.h"
#include "../parameter_validation.h"

namespace Raw_Operation
{
    template <typename _Type>
    penguinV::Image Load( const std::string & path, uint32_t width, uint32_t height, uint8_t colorCount )
    {
        if ( path.empty() || width == 0 || height == 0 || colorCount == 0 )
            throw imageException( "Incorrect parameters for raw image loading" );

        std::fstream file;
        file.open( path, std::fstream::in | std::fstream::binary );

        if ( !file )
            return penguinV::Image().generate<_Type>();

        file.seekg( 0, file.end );
        std::streamoff length = file.tellg();

        const uint32_t overallImageSize = width * height * colorCount * static_cast<uint32_t>( sizeof( _Type ) );

        if ( length != overallImageSize )
            return penguinV::Image().generate<_Type>();

        file.seekg( 0, file.beg );

        penguinV::Image image = penguinV::Image().generate<_Type>( width, height, colorCount );

        size_t dataToRead = overallImageSize;
        size_t dataReaded = 0;

        const size_t blockSize = 4 * 1024 * 1024; // read by 4 MB blocks

        while ( dataToRead > 0 ) {
            size_t readSize = dataToRead > blockSize ? blockSize : dataToRead;

            file.read( reinterpret_cast<char *>(image.data() + dataReaded), static_cast<std::streamsize>(readSize) );

            dataReaded += readSize;
            dataToRead -= readSize;
        }

        return image;
    }

    void Save( const std::string & path, const penguinV::Image & image )
    {
        Image_Function::ParameterValidation( image );

        std::vector < uint8_t > pallete;

        std::fstream file;
        file.open( path, std::fstream::out | std::fstream::trunc | std::fstream::binary );

        if(  !file )
            throw imageException( "Cannot create file for saving" );

        size_t dataToWrite = image.rowSize() * image.height() * image.dataSize();
        size_t dataWritten = 0;

        const size_t blockSize = 4 * 1024 * 1024; // read by 4 MB blocks

        while ( dataToWrite > 0 ) {
            size_t writeSize = dataToWrite > blockSize ? blockSize : dataToWrite;

            file.write( reinterpret_cast<const char *>(image.data() + dataWritten), static_cast<std::streamsize>( writeSize ) );
            file.flush();

            dataWritten += writeSize;
            dataToWrite -= writeSize;
        }

        if( !file )
            throw imageException( "failed to write data into file" );
    }

    template <typename _Type>
    void LittleEndianToBigEndian( penguinV::Image & image )
    {
        Image_Function::ParameterValidation( image );
        if ( sizeof( _Type ) < 2 )
            return;

        const size_t stepSize = sizeof( _Type );
        _Type * data = image.data<_Type>();
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
