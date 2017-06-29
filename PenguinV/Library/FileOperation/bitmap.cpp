#include <fstream>
#include <memory>
#include <vector>
#include "bitmap.h"
#include "../image_exception.h"

namespace Bitmap_Operation
{
    const static uint8_t BITMAP_ALIGNMENT = 4; // this is standard alignment of bitmap images

    void flipData( Bitmap_Image::Image & image )
    {
        if( image.empty() || image.height() < 2u )
            return;

        const uint32_t rowSize = image.rowSize();
        const uint32_t height  = image.height();

        std::vector < uint8_t > temp( rowSize );

        uint8_t * start = image.data();
        uint8_t * end   = image.data() + rowSize * (height - 1);

        for( uint32_t rowId = 0; rowId < height / 2; ++rowId, start += rowSize, end -= rowSize ) {
            memcpy( temp.data(), start, sizeof( uint8_t ) * rowSize );
            memcpy( start, end, sizeof( uint8_t ) * rowSize );
            memcpy( end, temp.data(), sizeof( uint8_t ) * rowSize );
        }
    }

    // Seems like very complicated structure but we did this to avoid compiler specific code for bitmap structures :(
    template <typename valueType>
    void get_value( const std::vector < uint8_t > & data, size_t & offset, valueType & value )
    {
        value = *(reinterpret_cast<const valueType *>(data.data() + offset));
        offset += sizeof( valueType );
    };

    template <typename valueType>
    void set_value( std::vector < uint8_t > & data, size_t & offset, const valueType & value )
    {
        memcpy( data.data() + offset, reinterpret_cast<const uint8_t *>(&value), sizeof( valueType ) );
        offset += sizeof( valueType );
    };

    struct BitmapFileHeader
    {
        BitmapFileHeader()
            : bfType     ( 0x4D42 ) // bitmap identifier
            , bfSize     ( 0 )      // total size of image
            , bfReserved1( 0 )      // not in use
            , bfReserved2( 0 )      // not in use
            , bfOffBits  ( 0 )      // offset from beggining of file to image data
            , overallSize( 14 )     // real size of this structure for bitmap format
        { };

        uint16_t bfType;
        uint32_t bfSize;
        uint16_t bfReserved1;
        uint16_t bfReserved2;
        uint32_t bfOffBits;

        uint32_t overallSize;

        void set( const std::vector < uint8_t > & data )
        {
            size_t offset = 0;
            get_value( data, offset, bfType );
            get_value( data, offset, bfSize );
            get_value( data, offset, bfReserved1 );
            get_value( data, offset, bfReserved2 );
            get_value( data, offset, bfOffBits );
        };

        void get( std::vector < uint8_t > & data )
        {
            size_t offset = 0;
            set_value( data, offset, bfType );
            set_value( data, offset, bfSize );
            set_value( data, offset, bfReserved1 );
            set_value( data, offset, bfReserved2 );
            set_value( data, offset, bfOffBits );
        };
    };

    struct BitmapDibHeader
    {
        virtual void set( const std::vector < uint8_t > & ) = 0;
        virtual void get( std::vector < uint8_t > & )      = 0;

        virtual uint32_t width()      = 0;
        virtual uint32_t height()     = 0;
        virtual uint8_t  colorCount() = 0;
        virtual uint32_t size()       = 0;

        virtual void setWidth( uint32_t )      = 0;
        virtual void setHeight( uint32_t )     = 0;
        virtual void setColorCount( uint16_t ) = 0;
        virtual void setImageSize( uint32_t )  = 0;

        virtual bool validate( const BitmapFileHeader & ) = 0;
    };

    struct BitmapCoreHeader : public BitmapDibHeader
    {
        BitmapCoreHeader()
            : bcSize     ( 12 ) // the size of this structure
            , bcWidth    ( 0 )  // width of image
            , bcHeight   ( 0 )  // height of image
            , bcPlanes   ( 1 )  // number of colour planes (always 1)
            , bcBitCount ( 0 )  // bits per pixel
            , overallSize( 12 ) // real size of this structure for bitmap format
        { };

        uint32_t bcSize;
        uint16_t bcWidth;
        uint16_t bcHeight;
        uint16_t bcPlanes;
        uint16_t bcBitCount;

        uint32_t overallSize;

        void set( const std::vector < uint8_t > & data )
        {
            size_t offset = 0;
            get_value( data, offset, bcSize );
            get_value( data, offset, bcWidth );
            get_value( data, offset, bcHeight );
            get_value( data, offset, bcPlanes );
            get_value( data, offset, bcBitCount );
        };

        void get( std::vector < uint8_t > & data )
        {
            size_t offset = 0;
            set_value( data, offset, bcSize );
            set_value( data, offset, bcWidth );
            set_value( data, offset, bcHeight );
            set_value( data, offset, bcPlanes );
            set_value( data, offset, bcBitCount );
        };

        virtual uint32_t width()
        {
            return bcWidth;
        };

        virtual uint32_t height()
        {
            return bcHeight;
        };

        virtual uint8_t colorCount()
        {
            return static_cast<uint8_t>(bcBitCount / 8u);
        };

        virtual uint32_t size()
        {
            return overallSize;
        };

        virtual void setWidth( uint32_t w )
        {
            bcWidth = static_cast<uint16_t>(w);
        };

        virtual void setHeight( uint32_t h )
        {
            bcHeight = static_cast<uint16_t>(h);
        };

        virtual void setColorCount( uint16_t c )
        {
            bcBitCount = c * 8u;
        };

        virtual void setImageSize( uint32_t )
        {
        };

        virtual bool validate( const BitmapFileHeader & header )
        {
            return !(bcBitCount == 8u || bcBitCount == 24u || bcBitCount == 32u) ||
                bcWidth == 0 || bcHeight == 0 || bcSize != BitmapCoreHeader().bcSize ||
                bcPlanes != BitmapCoreHeader().bcPlanes || header.bfOffBits < overallSize + header.overallSize;
        };
    };

    struct BitmapInfoHeader : public BitmapDibHeader
    {
        BitmapInfoHeader()
            : biSize         ( 40 ) // the size of this structure
            , biWidth        ( 0 )  // width of image
            , biHeight       ( 0 )  // height of image
            , biPlanes       ( 1 )  // number of colour planes (always 1)
            , biBitCount     ( 8 )  // bits per pixel
            , biCompress     ( 0 )  // compression type
            , biSizeImage    ( 0 )  // image data size in bytes
            , biXPelsPerMeter( 0 )  // pixels per meter in X direction
            , biYPelsPerMeter( 0 )  // pixels per meter in Y direction
            , biClrUsed      ( 0 )  // Number of colours used
            , biClrImportant ( 0 )  // important colours
            , overallSize    ( 40 ) // real size of this structure for bitmap format
        { };

        uint32_t biSize;
        int32_t  biWidth;
        int32_t  biHeight;
        uint16_t biPlanes;
        uint16_t biBitCount;
        uint32_t biCompress;
        uint32_t biSizeImage;
        int32_t  biXPelsPerMeter;
        int32_t  biYPelsPerMeter;
        uint32_t biClrUsed;
        uint32_t biClrImportant;

        uint32_t overallSize;

        void set( const std::vector < uint8_t > & data )
        {
            size_t offset = 0;
            get_value( data, offset, biSize );
            get_value( data, offset, biWidth );
            get_value( data, offset, biHeight );
            get_value( data, offset, biPlanes );
            get_value( data, offset, biBitCount );
            get_value( data, offset, biCompress );
            get_value( data, offset, biSizeImage );
            get_value( data, offset, biXPelsPerMeter );
            get_value( data, offset, biYPelsPerMeter );
            get_value( data, offset, biClrUsed );
            get_value( data, offset, biClrImportant );
        };

        void get( std::vector < uint8_t > & data )
        {
            size_t offset = 0;
            set_value( data, offset, biSize );
            set_value( data, offset, biWidth );
            set_value( data, offset, biHeight );
            set_value( data, offset, biPlanes );
            set_value( data, offset, biBitCount );
            set_value( data, offset, biCompress );
            set_value( data, offset, biSizeImage );
            set_value( data, offset, biXPelsPerMeter );
            set_value( data, offset, biYPelsPerMeter );
            set_value( data, offset, biClrUsed );
            set_value( data, offset, biClrImportant );
        };

        virtual uint32_t width()
        {
            return static_cast<uint32_t>(biWidth);
        };

        virtual uint32_t height()
        {
            return static_cast<uint32_t>(biHeight);
        };

        virtual uint8_t colorCount()
        {
            return static_cast<uint8_t>(biBitCount / 8u);
        };

        virtual uint32_t size()
        {
            return overallSize;
        };

        virtual void setWidth( uint32_t w )
        {
            biWidth = static_cast<int32_t>(w);
        };

        virtual void setHeight( uint32_t h )
        {
            biHeight = static_cast<int32_t>(h);
        };

        virtual void setColorCount( uint16_t c )
        {
            biBitCount = c * 8u;
        };

        virtual void setImageSize( uint32_t s )
        {
            biSizeImage = s;
        };

        virtual bool validate( const BitmapFileHeader & header )
        {
            uint32_t rowSize = width() * colorCount();
            if( rowSize % BITMAP_ALIGNMENT != 0 )
                rowSize = (rowSize / BITMAP_ALIGNMENT + 1) * BITMAP_ALIGNMENT;

            return !(biBitCount == 8u || biBitCount == 24u || biBitCount == 32u) ||
                biWidth <= 0 || biHeight <= 0 || biSize != BitmapInfoHeader().biSize ||
                (biSizeImage != static_cast<uint32_t>(rowSize * biHeight) && biSizeImage != 0) ||
                biPlanes != BitmapInfoHeader().biPlanes || header.bfOffBits < overallSize + header.overallSize;
        };
    };

    BitmapDibHeader * getInfoHeader( uint32_t size )
    {
        switch( size )
        {
            case 12u:
                return new BitmapCoreHeader;
            case 40u:
                return new BitmapInfoHeader;
            default:
                return nullptr;
        };
    };

    Bitmap_Image::Image Load( const std::string & path )
    {
        if( path.empty() )
            throw imageException( "Incorrect parameters for bitmap loading" );

        std::fstream file;
        file.open( path, std::fstream::in | std::fstream::binary );

        if( !file )
            return Bitmap_Image::Image();

        file.seekg( 0, file.end );
        std::streamoff length = file.tellg();

        if( length == std::char_traits<char>::pos_type( -1 ) ||
            static_cast<size_t>(length) < sizeof( BitmapFileHeader ) + sizeof( BitmapInfoHeader ) )
            return Bitmap_Image::Image();

        file.seekg( 0, file.beg );

        // read bitmap header
        BitmapFileHeader header;

        std::vector < uint8_t > data( sizeof( BitmapFileHeader ) );

        file.read( reinterpret_cast<char *>(data.data()), header.overallSize );

        header.set( data );

        // we suppose to compare header.bfSize and length but some editors don't put correct information
        if( header.bfType != BitmapFileHeader().bfType || header.bfOffBits >= length )
            return Bitmap_Image::Image();

        // read the size of dib header
        data.resize( 4u );

        file.read( reinterpret_cast<char *>(data.data()), data.size() );

        size_t dibHeaderOffset = 0;
        uint32_t dibHeaderSize = 0;

        get_value( data, dibHeaderOffset, dibHeaderSize );

        // create proper dib header 
        std::unique_ptr <BitmapDibHeader> info( getInfoHeader( dibHeaderSize ) );

        if( info.get() == nullptr )
            return Bitmap_Image::Image();

        data.resize( dibHeaderSize );

        // read bitmap info
        file.read( reinterpret_cast<char *>(data.data() + sizeof( dibHeaderSize )), data.size() - sizeof( dibHeaderSize ) );

        info->set( data );

        if( info->validate( header ) )
            return Bitmap_Image::Image();

        uint32_t rowSize = info->width() * info->colorCount();
        if( rowSize % BITMAP_ALIGNMENT != 0 )
            rowSize = (rowSize / BITMAP_ALIGNMENT + 1) * BITMAP_ALIGNMENT;

        if( length != header.bfOffBits + rowSize * info->height() )
            return Bitmap_Image::Image();

        uint32_t palleteSize = header.bfOffBits - info->size() - header.overallSize;

        if( palleteSize > 0 ) {
            std::vector < uint8_t > pallete( palleteSize );

            size_t dataToRead = palleteSize;
            size_t dataReaded = 0;

            const size_t blockSize = 4 * 1024 * 1024; // read by 4 MB blocks

            while( dataToRead > 0 ) {
                size_t readSize = dataToRead > blockSize ? blockSize : dataToRead;

                file.read( reinterpret_cast<char *>(pallete.data() + dataReaded), readSize );

                dataReaded += readSize;
                dataToRead -= readSize;
            }
        }

        Bitmap_Image::Image image( info->width(), info->height(), info->colorCount(), BITMAP_ALIGNMENT );

        size_t dataToRead = rowSize * info->height();
        size_t dataReaded = 0;

        const size_t blockSize = 4 * 1024 * 1024; // read by 4 MB blocks

        while( dataToRead > 0 ) {
            size_t readSize = dataToRead > blockSize ? blockSize : dataToRead;

            file.read( reinterpret_cast<char *>(image.data() + dataReaded), readSize );

            dataReaded += readSize;
            dataToRead -= readSize;
        }

        // thanks to bitmap creators image is stored in wrong flipped format so we have to flip back
        flipData( image );

        return image;
    }

    void Load( const std::string & path, Bitmap_Image::Image & raw )
    {
        raw = Load( path );
    }

    void Save( const std::string & path, const Bitmap_Image::Image & image )
    {
        Save( path, image, 0, 0, image.width(), image.height() );
    }

    void Save( const std::string & path, const Bitmap_Image::Image & image, uint32_t startX, uint32_t startY,
               uint32_t width, uint32_t height )
    {
        Image_Function::ParameterValidation( image, startX, startY, width, height );

        uint32_t palleteSize = 0;
        std::vector < uint8_t > pallete;

        // Create a pallete only for gray-scale image
        if( image.colorCount() == 1u ) {
            palleteSize = 1024u;
            pallete.resize( palleteSize );

            uint8_t * palleteData = pallete.data();
            uint8_t * palleteEnd = palleteData + pallete.size();

            for( uint8_t i = 0; palleteData != palleteEnd; ++i, ++palleteData ) {
                *palleteData++ = i;
                *palleteData++ = i;
                *palleteData++ = i;
            }
        }

        uint32_t lineLength = width * image.colorCount();
        if( lineLength % BITMAP_ALIGNMENT != 0 )
            lineLength = (lineLength / BITMAP_ALIGNMENT + 1) * BITMAP_ALIGNMENT;

        BitmapFileHeader header;
        BitmapInfoHeader info;

        header.bfSize    = header.overallSize + info.size() + palleteSize + lineLength * height;
        header.bfOffBits = header.overallSize + info.size() + palleteSize;

        info.setWidth     ( width );
        info.setHeight    ( height );
        info.setColorCount( image.colorCount() );
        info.setImageSize ( lineLength * height );

        std::fstream file;
        file.open( path, std::fstream::out | std::fstream::trunc | std::fstream::binary );

        if( !file )
            throw imageException( "Cannot create file for saving" );

        std::vector < uint8_t > data( sizeof( BitmapFileHeader ) );

        header.get( data );
        file.write( reinterpret_cast<const char *>(data.data()), header.overallSize );

        data.resize( sizeof( BitmapInfoHeader ) );

        info.get( data );
        file.write( reinterpret_cast<const char *>(data.data()), info.size() );

        if( !pallete.empty() )
            file.write( reinterpret_cast<const char *>(pallete.data()), pallete.size() );

        file.flush();

        uint32_t rowSize = image.rowSize();

        const uint8_t * imageY = image.data() + (startY + height - 1) * rowSize + startX;

        std::vector < uint8_t > temp( lineLength, 0 );

        for( uint32_t rowId = 0; rowId < height; ++rowId, imageY -= rowSize ) {
            memcpy( temp.data(), imageY, sizeof( uint8_t ) * width );

            file.write( reinterpret_cast<const char *>(temp.data()), lineLength );
            file.flush();
        }

        if( !file )
            throw imageException( "failed to write data into file" );
    }
}
