#include <fstream>
#include <vector>
#include "bitmap.h"
#include "../image_exception.h"

namespace Bitmap_Operation
{
	// Seems like very complicated structure but we did this to avoid compiler specific code for bitmap structures :(
	template <typename valueType>
	void get_value( const std::vector < uint8_t > & data, size_t & offset, valueType & value )
	{
		value = *(reinterpret_cast< const valueType * >(data.data() + offset));
		offset += sizeof(valueType);
	};

	template <typename valueType>
	void set_value( std::vector < uint8_t > & data, size_t & offset, const valueType & value )
	{
		memcpy( data.data() + offset, reinterpret_cast< const uint8_t * >(&value), sizeof(valueType ) );
		offset += sizeof(valueType);
	};

	struct BitmapFileHeader
	{
		BitmapFileHeader()
			: bfType     (0x4D42) // bitmap identifier
			, bfSize     (0)      // total size of image
			, bfReserved1(0)      // not in use
			, bfReserved2(0)      // not in use
			, bfOffBits  (0)      // offset from beggining of file to image data
			, overallSize(14)     // real size of this structure for bitmap format
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
			get_value( data, offset, bfType      );
			get_value( data, offset, bfSize      );
			get_value( data, offset, bfReserved1 );
			get_value( data, offset, bfReserved2 );
			get_value( data, offset, bfOffBits   );
		};

		void get( std::vector < uint8_t > & data )
		{
			size_t offset = 0;
			set_value( data, offset, bfType      );
			set_value( data, offset, bfSize      );
			set_value( data, offset, bfReserved1 );
			set_value( data, offset, bfReserved2 );
			set_value( data, offset, bfOffBits   );
		};
	};

	struct BitmapInfoHeader
	{
		BitmapInfoHeader()
			: biSize         (40) // the size of this structure
			, biWidth        (0)  // width of image
			, biHeight       (0)  // height of image
			, biPlanes       (1)  // number of colour planes (always 1)
			, biBitCount     (8)  // bits per pixel
			, biCompress     (0)  // compression type
			, biSizeImage    (0)  // image data size in bytes
			, biXPelsPerMeter(0)  // pixels per meter in X direction
			, biYPelsPerMeter(0)  // pixels per meter in Y direction
			, biClrUsed      (0)  // Number of colours used
			, biClrImportant (0)  // important colours
			, overallSize    (40) // real size of this structure for bitmap format
		{};

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
			get_value( data, offset, biSize          );
			get_value( data, offset, biWidth         );
			get_value( data, offset, biHeight        );
			get_value( data, offset, biPlanes        );
			get_value( data, offset, biBitCount      );
			get_value( data, offset, biCompress      );
			get_value( data, offset, biSizeImage     );
			get_value( data, offset, biXPelsPerMeter );
			get_value( data, offset, biYPelsPerMeter );
			get_value( data, offset, biClrUsed       );
			get_value( data, offset, biClrImportant  );
		};

		void get( std::vector < uint8_t > & data )
		{
			size_t offset = 0;
			set_value( data, offset, biSize          );
			set_value( data, offset, biWidth         );
			set_value( data, offset, biHeight        );
			set_value( data, offset, biPlanes        );
			set_value( data, offset, biBitCount      );
			set_value( data, offset, biCompress      );
			set_value( data, offset, biSizeImage     );
			set_value( data, offset, biXPelsPerMeter );
			set_value( data, offset, biYPelsPerMeter );
			set_value( data, offset, biClrUsed       );
			set_value( data, offset, biClrImportant  );
		};
	};

	BitmapRawImage Load(std::string path)
	{
		if( path.empty() )
			throw imageException("Incorrect parameters for bitmap loading");

		std::basic_fstream <uint8_t> file;
		file.open( path, std::fstream::in | std::fstream::binary );

		if( !file )
			return BitmapRawImage();

		file.seekg(0, file.end);
		std::streamoff length = file.tellg();

		if( length == std::char_traits<char>::pos_type(-1) ||
			static_cast<size_t>(length) < sizeof(BitmapFileHeader) + sizeof(BitmapInfoHeader) )
			return BitmapRawImage();

		file.seekg(0, file.beg);

		// read bitmap header
		BitmapFileHeader header;

		std::vector < uint8_t > data( sizeof(BitmapFileHeader) );

		file.read( data.data(), header.overallSize );

		header.set( data );

		// we suppose to compare header.bfSize and length but some editors don't put correct information
		if( header.bfType != BitmapFileHeader().bfType || header.bfOffBits >= length )
			return BitmapRawImage();

		// read bitmap info
		BitmapInfoHeader info;

		data.resize( sizeof(BitmapInfoHeader) );

		file.read( data.data(), info.overallSize );

		info.set( data );

		if( !(info.biBitCount == 8u || info.biBitCount == 24u) ||
			info.biWidth <= 0 || info.biHeight <= 0 || info.biSize != BitmapInfoHeader().biSize ||
			(info.biSizeImage != static_cast<uint32_t>(info.biWidth * info.biHeight) && info.biSizeImage != 0 ) ||
			info.biPlanes != BitmapInfoHeader().biPlanes || header.bfOffBits < info.overallSize - header.overallSize)
			return BitmapRawImage();

		uint32_t rowSize = static_cast<uint32_t>(info.biWidth) * static_cast<uint8_t>(info.biBitCount / 8u);
		if( rowSize % Bitmap_Image::BITMAP_ALIGNMENT != 0 )
			rowSize = (rowSize / Bitmap_Image::BITMAP_ALIGNMENT + 1) * Bitmap_Image::BITMAP_ALIGNMENT;

		if( length != header.bfOffBits + rowSize * static_cast<uint32_t>(info.biHeight) )
			return BitmapRawImage();

		BitmapRawImage raw;

		uint32_t palleteSize = header.bfOffBits - info.overallSize - header.overallSize;

		if( palleteSize > 0 ) {
			raw.allocatePallete( palleteSize );

			size_t dataToRead = palleteSize;
			size_t dataReaded = 0;

			const size_t blockSize = 4 * 1024 * 1024; // read by 4 MB blocks

			while( dataToRead > 0 ) {
				size_t readSize = dataToRead > blockSize ? blockSize : dataToRead;

				file.read( raw.pallete() + dataReaded, readSize );

				dataReaded += readSize;
				dataToRead -= readSize;
			}
		}

		raw.allocateImage( static_cast<uint32_t>(info.biWidth), static_cast<uint32_t>(info.biHeight),
						   static_cast<uint8_t>(info.biBitCount / 8u), Bitmap_Image::BITMAP_ALIGNMENT );

		size_t dataToRead =  rowSize * static_cast < size_t > (info.biHeight);
		size_t dataReaded = 0;

		const size_t blockSize = 4 * 1024 * 1024; // read by 4 MB blocks

		while( dataToRead > 0 ) {
			size_t readSize = dataToRead > blockSize ? blockSize : dataToRead;

			file.read( raw.data() + dataReaded, readSize );

			dataReaded += readSize;
			dataToRead -= readSize;
		}

		// thanks to bitmap creators image is stored in wrong flipped format so we have to flip back
		raw.flipData();

		return raw;
	}

	void Load(std::string path, BitmapRawImage & raw)
	{
		raw = Load(path);
	}

	void Save( std::string path, Template_Image::ImageTemplate < uint8_t > & image )
	{
		Save( path, image, 0, 0, image.width(), image.height() );
	}

	void Save( std::string path, Template_Image::ImageTemplate < uint8_t > & image, uint32_t startX, uint32_t startY,
			   uint32_t width, uint32_t height )
	{
		if( path.empty() || image.empty() || !(image.colorCount() == 1u || image.colorCount() == 3u)  )
			throw imageException("Incorrect parameters for bitmap saving");

		uint32_t palleteSize = 0;
		std::vector < uint8_t > pallete;

		// Create pallete only for gray-scale image
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
		if( lineLength % Bitmap_Image::BITMAP_ALIGNMENT != 0 )
			lineLength = (lineLength / Bitmap_Image::BITMAP_ALIGNMENT + 1) * Bitmap_Image::BITMAP_ALIGNMENT;

		BitmapFileHeader header;
		BitmapInfoHeader info;

		header.bfSize    = header.overallSize + info.overallSize + palleteSize + lineLength * height;
		header.bfOffBits = header.overallSize + info.overallSize + palleteSize;
		
		info.biWidth     = static_cast<int32_t>(width);
		info.biHeight    = static_cast<int32_t>(height);
		info.biBitCount  = 8u * image.colorCount();
		info.biSizeImage = width * height;

		std::basic_fstream <uint8_t> file;
		file.open( path, std::fstream::out | std::fstream::trunc | std::fstream::binary );

		if( !file )
			throw imageException("Cannot create file for saving");

		std::vector < uint8_t > data( sizeof(BitmapFileHeader) );

		header.get( data );
		file.write( data.data(), header.overallSize );

		data.resize( sizeof(BitmapInfoHeader) );

		info.get( data );
		file.write( data.data(), info.overallSize );

		if( !pallete.empty() )
			file.write( pallete.data(), pallete.size() );

		file.flush();

		uint32_t rowSize = image.rowSize();

		const uint8_t * imageY = image.data() + (startY + height - 1) * rowSize + startX;

		std::vector < uint8_t > temp( lineLength, 0 );

		for( uint32_t rowId = 0; rowId < height; ++rowId, imageY -= rowSize ) {
			memcpy( temp.data(), imageY, sizeof(uint8_t) * width );
			
			file.write( temp.data(), lineLength );
			file.flush();
		}

		if( !file )
			throw imageException("failed to write data into file");
	}
}
