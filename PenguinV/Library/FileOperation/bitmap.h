#include "../image_buffer.h"
#include "../image_function.h"

namespace Bitmap_Operation
{
	template <typename TColorDepth>
	class RawImageTemplate
	{
	public:
		RawImageTemplate()
			: _width      (0)       // width of image
			, _height     (0)       // height of image
			, _colorCount (1)       // number of colors per pixel
			, _alignment  (1)       // some formats require that row size must be a multiple of some value (alignment)
			, _rowSize    (0)       // size of single row on image, usually it is equal to width
			, _data       (nullptr) // pointer to image data
			, _freeData   (true)    // flag to clear data if no conversion to image made
			, _pallete    (nullptr) // pointer to image pallete
			, _palleteSize(0)       // the size of pallete elements
			, _freePallete(true)    // flag to clear pallete
		{ }

		RawImageTemplate(const RawImageTemplate & raw)
			: _width      (0)
			, _height     (0)
			, _colorCount (1)
			, _alignment  (1)
			, _rowSize    (0)
			, _data       (nullptr)
			, _freeData   (true)
			, _pallete    (nullptr)
			, _palleteSize(0)
			, _freePallete(true)
		{
			allocateImage( raw._width, raw._height, raw._colorCount, raw._alignment );
			allocatePallete( raw._palleteSize );

			_copy(raw);
		}

		RawImageTemplate( RawImageTemplate && raw )
			: _width      (0)
			, _height     (0)
			, _colorCount (1)
			, _alignment  (1)
			, _rowSize    (0)
			, _data       (nullptr)
			, _freeData   (true)
			, _pallete    (nullptr)
			, _palleteSize(0)
			, _freePallete(true)
		{
			_swap( raw );
		}

		~RawImageTemplate()
		{
			_clearImage();
			_clearPallete();
		}

		RawImageTemplate & operator=(const RawImageTemplate & raw)
		{
			allocateImage( raw._width, raw._height, raw._colorCount, raw._alignment );
			allocatePallete( raw._palleteSize );

			_copy(raw);

			return (*this);
		}

		RawImageTemplate & operator=(RawImageTemplate && raw)
		{
			_swap( raw );

			return (*this);
		}

		TColorDepth * data() const
		{
			return _data;
		}

		TColorDepth * data()
		{
			return _data;
		}

		TColorDepth * pallete() const
		{
			return _pallete;
		}

		TColorDepth * pallete()
		{
			return _pallete;
		}

		void allocateImage( uint32_t width, uint32_t height, uint8_t colorCount, uint8_t alignment )
		{
			_clearImage();

			if( width > 0 && height > 0 && colorCount > 0 && alignment > 0 ) {
				_width      = width;
				_height     = height;
				_colorCount = colorCount;
				_alignment  = alignment;

				_rowSize = width * colorCount;
				if( _rowSize % alignment != 0 )
					_rowSize = (_rowSize / alignment + 1) * alignment;

				_data = new TColorDepth [_height * _rowSize];
				_freeData = true;
			}
		}

		void allocatePallete( uint32_t size )
		{
			_clearPallete();

			if( size > 0 ) {
				_palleteSize = size;

				_pallete = new TColorDepth [_palleteSize];
				_freePallete = true;
			}
		}

		void assignImage( Template_Image::ImageTemplate <TColorDepth> & image )
		{
			if( _availableImage() ) {
				image.assign( _data, _width, _height, _colorCount, _alignment );
				_freeData = false;
			}
		}

		void flipData()
		{
			if( _availableImage() ) {

				std::vector < TColorDepth > temp( _rowSize );

				TColorDepth * start = _data;
				TColorDepth * end   = _data + _rowSize * (_height - 1);

				for( uint32_t rowId = 0; rowId < _height / 2; ++rowId, start += _rowSize, end -= _rowSize ) {
					memcpy( temp.data(), start      , sizeof( TColorDepth ) * _rowSize );
					memcpy( start      , end        , sizeof( TColorDepth ) * _rowSize );
					memcpy( end        , temp.data(), sizeof( TColorDepth ) * _rowSize );
				}
			}
		}
	protected:
		uint32_t _width;
		uint32_t _height;
		uint8_t  _colorCount;
		uint8_t  _alignment;
		uint32_t _rowSize;

		TColorDepth * _data;
		bool _freeData;

		TColorDepth * _pallete;
		uint32_t _palleteSize;
		bool _freePallete;

		void _clearImage()
		{
			if( _freeData && _data != nullptr ) {
				delete [] _data;
				_data = nullptr;
			}

			_width      = 0;
			_height     = 0;
			_colorCount = 1;
			_alignment  = 1;
		}

		void _clearPallete()
		{
			if( _freePallete && _pallete != nullptr ) {
				delete [] _pallete;
				_pallete = nullptr;
			}

			_palleteSize = 0;
		}

		bool _availableImage() const
		{
			return _width > 0 && _height > 0 && _colorCount > 0 && _alignment > 0 && _rowSize > 0 && _data != nullptr && _freeData;
		}

		void _swap(RawImageTemplate & raw)
		{
			_width      = raw._width;
			_height     = raw._height;
			_colorCount = raw._colorCount;
			_alignment  = raw._alignment;
			_rowSize    = raw._rowSize;

			std::swap( _data, raw._data );
			std::swap( _freeData, raw._freeData );
		}

		void _copy(const RawImageTemplate & raw)
		{
			if( _rowSize == raw._rowSize && _height == raw._height) {
				if( raw._data != nullptr && _data != nullptr )
					memcpy( _data, raw._data, sizeof(TColorDepth) * _rowSize * _height );
			}
			else {
				throw imageException("Invalid raw image to copy");
			}

			if( _palleteSize == raw._palleteSize ) {
				if( raw._pallete != nullptr && _pallete != nullptr )
					memcpy( _pallete, raw._pallete, _palleteSize * sizeof(TColorDepth) );
			}
			else {
				throw imageException("Invalid pallete to copy");
			}
		}
	};

	class BitmapRawImage : public RawImageTemplate <uint8_t>
	{
	public:
		BitmapRawImage()
			: RawImageTemplate()
		{
		}

		BitmapRawImage(const BitmapRawImage & raw)
			: RawImageTemplate(raw)
		{
		}

		BitmapRawImage(BitmapRawImage && raw)
			: RawImageTemplate()
		{
			_swap(raw);
		}

		BitmapRawImage & operator=(const BitmapRawImage & raw)
		{
			RawImageTemplate::operator=( raw );

			return (*this);
		}

		BitmapRawImage & operator=(BitmapRawImage && raw)
		{
			_swap( raw );

			return (*this);
		}

		~BitmapRawImage()
		{
		}

		bool isGrayScale() const
		{
			return _availableImage() && _colorCount == 1u;
		}
		
		bool isColor() const
		{
			return _availableImage() && _colorCount == 3u;
		}

		// copy data to image if data is the same format as image
		void operator > ( Bitmap_Image::Image & image )
		{
			if( isGrayScale() )
				assignImage( image );
		}

		// copy data to image if data is the same format as image
		void operator > ( Bitmap_Image::ColorImage & image )
		{
			if( isColor() )
				assignImage( image );
		}

		// forcefully copy data to image with needed conversion
		void operator >> ( Bitmap_Image::Image & image )
		{
			if( isGrayScale() ) {
				assignImage( image );
			}
			else if( isColor() ) {
				Bitmap_Image::ColorImage colorImage( _width, _height );

				assignImage( colorImage );

				image.resize( _width, _height );

				Image_Function::Convert( colorImage, image );
			}
		}

		// forcefully copy data to image with needed conversion
		void operator >> ( Bitmap_Image::ColorImage & image )
		{
			if( isColor() ) {
				assignImage( image );
			}
			else if( isGrayScale() ) {
				Bitmap_Image::Image grayImage( _width, _height );

				assignImage( grayImage );

				image.resize( _width, _height );

				Image_Function::Convert( grayImage, image );
			}
		}
	};
	
	// Below functions support only Bitmap_Image::Image and Bitmap_Image::ColorImage classes
	BitmapRawImage Load(std::string path);
	void           Load(std::string path, BitmapRawImage & raw);

	void Save( std::string path, Template_Image::ImageTemplate < uint8_t > & image );
	void Save( std::string path, Template_Image::ImageTemplate < uint8_t > & image, uint32_t startX, uint32_t startY,
			   uint32_t width, uint32_t height );
};
