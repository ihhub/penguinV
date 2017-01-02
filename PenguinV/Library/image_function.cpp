#include <cmath>
#include "image_function.h"

namespace Image_Function
{
	Image AbsoluteDifference( const Image & in1, const Image & in2 )
	{
		ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		AbsoluteDifference( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void AbsoluteDifference( const Image & in1, const Image & in2, Image & out )
	{
		ParameterValidation( in1, in2, out );

		AbsoluteDifference( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
							  uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		AbsoluteDifference( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void AbsoluteDifference( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
							 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

		const uint8_t colorCount  = CommonColorCount( in1, in2, out );
		const uint32_t rowSize1   = in1.rowSize();
		const uint32_t rowSize2   = in2.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
		const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		width = width * colorCount;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
			const uint8_t * in1X = in1Y;
			const uint8_t * in2X = in2Y;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
				(*outX) = (*in2X) > (*in1X) ? (*in2X) - (*in1X) : (*in1X) - (*in2X);
		}
	}

	void Accumulate( const Image & image, std::vector < uint32_t > & result )
	{
		ParameterValidation( image );

		Accumulate( image, 0, 0, image.width(), image.height(), result );
	}

	void Accumulate( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & result )
	{
		ParameterValidation( image, x, y, width, height );
		VerifyGrayScaleImage( image );

		if( result.size() != width * height )
			throw imageException("Array size is not equal to image ROI (width * height) size");

		const uint32_t rowSize = image.rowSize();

		const uint8_t * imageY    = image.data() + y * rowSize + x;
		const uint8_t * imageYEnd = imageY + height * rowSize;
		std::vector < uint32_t >::iterator v = result.begin();

		for( ; imageY != imageYEnd; imageY += rowSize ) {
			const uint8_t * imageX    = imageY;
			const uint8_t * imageXEnd = imageX + width;

			for( ; imageX != imageXEnd; ++imageX, ++v )
				*v += (*imageX);
		}
	}

	Image BitwiseAnd( const Image & in1, const Image & in2 )
	{
		ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseAnd( const Image & in1, const Image & in2, Image & out )
	{
		ParameterValidation( in1, in2, out );

		BitwiseAnd( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					  uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseAnd( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseAnd( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );

		const uint8_t colorCount  = CommonColorCount( in1, in2, out );
		const uint32_t rowSize1   = in1.rowSize();
		const uint32_t rowSize2   = in2.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
		const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		width = width * colorCount;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
			const uint8_t * in1X = in1Y;
			const uint8_t * in2X = in2Y;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
				(*outX) = (*in1X) & (*in2X);
		}
	}

	Image BitwiseOr( const Image & in1, const Image & in2 )
	{
		ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseOr( const Image & in1, const Image & in2, Image & out )
	{
		ParameterValidation( in1, in2, out );

		BitwiseOr( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseOr( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseOr( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
		
		const uint8_t colorCount  = CommonColorCount( in1, in2, out );
		const uint32_t rowSize1   = in1.rowSize();
		const uint32_t rowSize2   = in2.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
		const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		width = width * colorCount;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
			const uint8_t * in1X = in1Y;
			const uint8_t * in2X = in2Y;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
				(*outX) = (*in1X) | (*in2X);
		}
	}

	Image BitwiseXor( const Image & in1, const Image & in2 )
	{
		ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		BitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseXor( const Image & in1, const Image & in2, Image & out )
	{
		ParameterValidation( in1, in2, out );

		BitwiseXor( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					  uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		BitwiseXor( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void BitwiseXor( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
		
		const uint8_t colorCount  = CommonColorCount( in1, in2, out );
		const uint32_t rowSize1   = in1.rowSize();
		const uint32_t rowSize2   = in2.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
		const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		width = width * colorCount;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
			const uint8_t * in1X = in1Y;
			const uint8_t * in2X = in2Y;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
				(*outX) = (*in1X) ^ (*in2X);
		}
	}

	Image ConvertToGrayScale( const Image & in )
	{
		ParameterValidation( in );

		Image out( in.width(), in.height() );

		ConvertToGrayScale(in, 0, 0, out, 0, 0, out.width(), out.height());

		return out;
	}

	void ConvertToGrayScale( const Image & in, Image & out )
	{
		ParameterValidation( in, out );

		ConvertToGrayScale(in, 0, 0, out, 0, 0, out.width(), out.height());
	}

	Image ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		ConvertToGrayScale( in, startXIn, startYIn, out, 0, 0, width, height );

		return out;
	}

	void ConvertToGrayScale( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
							 uint32_t width, uint32_t height )
	{
		ParameterValidation(in, startXIn, startYIn, out, startXOut, startYOut, width, height);
		VerifyGrayScaleImage( out );

		if( in.colorCount() == GRAY_SCALE ) {
			Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
			return;
		}

		const uint32_t rowSizeIn  = in.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t colorCount = RGB;

		const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn * colorCount;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		for (; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn) {
			const uint8_t * inX = inY;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for (; outX != outXEnd; ++outX, inX += colorCount)
				(*outX) = static_cast <uint8_t>( ( *(inX) + *(inX + 1) + *(inX + 2) ) / 3u ); // average of red, green and blue components
		}
	}

	Image ConvertToRgb( const Image & in )
	{
		ParameterValidation( in );

		Image out( in.width(), in.height(), RGB );

		ConvertToRgb(in, 0, 0, out, 0, 0, out.width(), out.height());

		return out;
	}

	void ConvertToRgb( const Image & in, Image & out )
	{
		ParameterValidation( in, out );

		ConvertToRgb(in, 0, 0, out, 0, 0, out.width(), out.height());
	}

	Image ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height, RGB );

		ConvertToRgb( in, startXIn, startYIn, out, 0, 0, width, height );

		return out;
	}

	void ConvertToRgb( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					   uint32_t width, uint32_t height )
	{
		ParameterValidation(in, startXIn, startYIn, out, startXOut, startYOut, width, height);
		VerifyColoredImage  ( out );

		if( in.colorCount() == RGB ) {
			Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
			return;
		}

		const uint32_t rowSizeIn  = in.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t colorCount = RGB;

		const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		for (; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn) {
			const uint8_t * inX  = inY;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width * colorCount;

			for (; outX != outXEnd; outX += colorCount, ++inX)
				memset( outX, (*inX), sizeof(uint8_t) * colorCount );
		}
	}

	void Copy( const Image & in, Image & out )
	{
		ParameterValidation( in, out );

		out = in;
	}

	Image Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Copy( in, startXIn, startYIn, out, 0, 0, width, height );

		return out;
	}

	void Copy( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
			   uint32_t width, uint32_t height )
	{
		ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );

		const uint8_t colorCount  = CommonColorCount( in, out );
		const uint32_t rowSizeIn  = in.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		width = width * colorCount;

		const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn )
			memcpy( outY, inY, sizeof(uint8_t) * width );
	}

	Image ExtractChannel( const Image & in, uint8_t channelId )
	{
		ParameterValidation( in );

		Image out( in.width(), in.height() );

		ExtractChannel( in, 0, 0, out, 0, 0, in.width(), in.height(), channelId );

		return out;
	}

	void ExtractChannel( const Image & in, Image & out, uint8_t channelId )
	{
		ParameterValidation( in, out );

		ExtractChannel( in, 0, 0, out, 0, 0, in.width(), in.height(), channelId );
	}

	Image ExtractChannel( const Image & in, uint32_t x, uint32_t y, uint32_t width, uint32_t height, uint8_t channelId )
	{
		ParameterValidation( in, x, y, width, height );

		Image out( width, height );

		ExtractChannel( in, x, y, out, 0, 0, width, height, channelId );

		return out;
	}

	void ExtractChannel( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut,
						 uint32_t startYOut, uint32_t width, uint32_t height, uint8_t channelId )
	{
		ParameterValidation(in, startXIn, startYIn, out, startXOut, startYOut, width, height);
		VerifyGrayScaleImage( out );

		if( channelId >= in.colorCount() )
			throw imageException("Channel ID for color image is greater than channel count in input image");

		const uint32_t rowSizeIn  = in.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t colorCount = in.colorCount();

		const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn * colorCount + channelId;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		for (; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn) {
			const uint8_t * inX  = inY;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for (; outX != outXEnd; ++outX, inX += colorCount)
				(*outX) = *(inX);
		}
	}

	void Fill( Image & image, uint8_t value )
	{
		image.fill( value );
	}

	void Fill( Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height, uint8_t value )
	{
		ParameterValidation( image, x, y, width, height );
		VerifyGrayScaleImage( image );

		const uint32_t rowSize = image.rowSize();

		uint8_t * imageY = image.data() + y * rowSize + x;
		const uint8_t * imageYEnd = imageY + height * rowSize;

		for( ; imageY != imageYEnd; imageY += rowSize )
			memset( imageY, value, sizeof(uint8_t) * width );
	}

	Image Flip( const Image & in, bool horizontal, bool vertical )
	{
		ParameterValidation( in );

		Image out( in.width(), in.height() );

		Flip( in, 0, 0, out, 0, 0, out.width(), out.height(), horizontal, vertical );

		return out;
	}

	void Flip( const Image & in, Image & out, bool horizontal, bool vertical )
	{
		ParameterValidation( in, out );

		Flip( in, 0, 0, out, 0, 0, out.width(), out.height(), horizontal, vertical );
	}

	Image Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
				bool horizontal, bool vertical)
	{
		ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Flip( in, startXIn, startYIn, out, 0, 0, width, height, horizontal, vertical );

		return out;
	}

	void Flip( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
			   uint32_t width, uint32_t height, bool horizontal, bool vertical )
	{
		ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
		VerifyGrayScaleImage( in, out );

		if( !horizontal && !vertical ) {
			Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
		}
		else {
			const uint32_t rowSizeIn  = in.rowSize();
			const uint32_t rowSizeOut = out.rowSize();

			const uint8_t * inY    = in.data() + startYIn * rowSizeIn + startXIn;
			const uint8_t * inYEnd = inY + height * rowSizeIn;

			if( horizontal && !vertical ) {
				uint8_t * outY = out.data() + startYOut * rowSizeOut + startXOut + width - 1;

				for( ; inY != inYEnd; inY += rowSizeIn, outY += rowSizeOut ) {
					const uint8_t * inX    = inY;
					uint8_t       * outX   = outY;
					const uint8_t * inXEnd = inX + width;

					for( ; inX != inXEnd; ++inX, --outX )
						(*outX) = (*inX);
				}
			}
			else if( !horizontal && vertical ) {
				uint8_t * outY = out.data() + (startYOut + height - 1) * rowSizeOut + startXOut;

				for( ; inY != inYEnd; inY += rowSizeIn, outY -= rowSizeOut )
					memcpy( outY, inY, sizeof(uint8_t) * width );
			}
			else {
				uint8_t * outY = out.data() + (startYOut + height - 1) * rowSizeOut + startXOut + width - 1;

				for( ; inY != inYEnd; inY += rowSizeIn, outY -= rowSizeOut ) {
					const uint8_t * inX    = inY;
					uint8_t       * outX   = outY;
					const uint8_t * inXEnd = inX + width;

					for( ; inX != inXEnd; ++inX, --outX )
						(*outX) = (*inX);
				}
			}
		}
	}

	Image GammaCorrection( const Image & in, double a, double gamma )
	{
		ParameterValidation( in );

		Image out( in.width(), in.height() );

		GammaCorrection( in, 0, 0, out, 0, 0, out.width(), out.height(), a, gamma );

		return out;
	}

	void GammaCorrection( const Image & in, Image & out, double a, double gamma )
	{
		ParameterValidation( in, out );

		GammaCorrection( in, 0, 0, out, 0, 0, out.width(), out.height(), a, gamma );
	}

	Image GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, double a, double gamma )
	{
		ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		GammaCorrection( in, startXIn, startYIn, out, 0, 0, width, height, a, gamma );

		return out;
	}

	void GammaCorrection( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
						  uint32_t width, uint32_t height, double a, double gamma )
	{
		ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
		VerifyGrayScaleImage( in, out );

		if( a < 0 || gamma < 0 )
			throw imageException("Gamma correction parameters are invalid");

		// We precalculate all values and store them in lookup table
		std::vector < uint8_t > value(256);

		for( uint16_t i = 0; i < 256; ++i ) {
			double data = a * pow( i / 255.0, gamma ) * 255 + 0.5;

			if( data < 256 )
				value[i] = static_cast<uint8_t>(data);
			else
				value[i] = 255;
		}

		LookupTable( in, startXIn, startYIn, out, startXOut, startYOut, width, height, value );
	}

	uint8_t GetPixel( const Image & image, uint32_t x, uint32_t y )
	{
		if( image.empty() || x >= image.width() || y >= image.height() || image.colorCount() != GRAY_SCALE )
			throw imageException("Position of point [x, y] is out of image");

		return *(image.data() + y * image.rowSize() + x);
	}

	uint8_t GetThreshold( const std::vector < uint32_t > & histogram )
	{
		if( histogram.size() != 256 )
			throw imageException("Histogram size is not 256");

		uint8_t threshold = 0;

		// It is well-known Otsu's method to find threshold
		uint32_t pixelCount = histogram[0] + histogram[1];
		uint32_t sum = histogram[1];
		for(uint16_t i = 2; i < 256; ++i) {
			sum  = sum  + i * histogram[i];
			pixelCount += histogram[i];
		}

		uint32_t sumTemp = 0;
		uint32_t pixelCountTemp = 0;
		
		double maximumSigma = -1;

		for(uint16_t i = 0; i < 256; ++i) {
			pixelCountTemp += histogram[i];

			if(pixelCountTemp > 0 && pixelCountTemp != pixelCount) {
				sumTemp += i * histogram[i];

				double w1 = static_cast<double>(pixelCountTemp) / pixelCount;
				double a  = static_cast<double>(sumTemp       ) / pixelCountTemp -
						    static_cast<double>(sum - sumTemp ) / (pixelCount - pixelCountTemp);
				double sigma = w1 * (1 - w1) * a * a;

				if(sigma > maximumSigma) {
					maximumSigma = sigma;
					threshold = static_cast < uint8_t >(i);
				}
			}
		}

		return threshold;
	}

	std::vector < uint32_t > Histogram( const Image & image )
	{
		std::vector < uint32_t > histogram;

		Histogram( image, 0, 0, image.width(), image.height(), histogram );

		return histogram;
	}

	void Histogram( const Image & image, std::vector < uint32_t > & histogram )
	{
		Histogram( image, 0, 0, image.width(), image.height(), histogram );
	}

	std::vector < uint32_t > Histogram( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height )
	{
		ParameterValidation( image, x, y, width, height );

		std::vector < uint32_t > histogram;

		Histogram( image, x, y, width, height, histogram );

		return histogram;
	}

	void Histogram( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height, std::vector < uint32_t > & histogram )
	{
		ParameterValidation( image, x, y, width, height );
		VerifyGrayScaleImage( image );

		histogram.resize( 256u );
		std::fill( histogram.begin(), histogram.end(), 0u );

		const uint32_t rowSize = image.rowSize();

		const uint8_t * imageY = image.data() + y * rowSize + x;
		const uint8_t * imageYEnd = imageY + height * rowSize;

		for( ; imageY != imageYEnd; imageY += rowSize ) {
			const uint8_t * imageX    = imageY;
			const uint8_t * imageXEnd = imageX + width;

			for( ; imageX != imageXEnd; ++imageX )
				++histogram[*imageX];
		}
	}

	Image Invert( const Image & in )
	{
		ParameterValidation( in );

		Image out( in.width(), in.height() );

		Invert( in, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Invert( const Image & in, Image & out )
	{
		ParameterValidation( in, out );

		Invert( in, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Invert( in, startXIn, startYIn, out, 0, 0, width, height );

		return out;
	}

	void Invert( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
				 uint32_t width, uint32_t height )
	{
		ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
		
		const uint8_t colorCount  = CommonColorCount( in, out );
		const uint32_t rowSizeIn  = in.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		width = width * colorCount;

		for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
			const uint8_t * inX  = inY;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for( ; outX != outXEnd; ++outX, ++inX )
				(*outX) = ~(*inX);
		}
	}

	bool IsEqual( const Image & in1, const Image & in2 )
	{
		ParameterValidation( in1, in2 );

		return IsEqual( in1, 0, 0, in2, 0, 0, in1.width(), in1.height() );
	}

	bool IsEqual( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				  uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );
		
		const uint8_t colorCount = CommonColorCount( in1, in2 );
		const uint32_t rowSize1  = in1.rowSize();
		const uint32_t rowSize2  = in2.rowSize();

		const uint8_t * in1Y = in1.data() + startY1 * rowSize1 + startX1 * colorCount;
		const uint8_t * in2Y = in2.data() + startY2 * rowSize2 + startX2 * colorCount;

		const uint8_t * in1YEnd = in1Y + height * rowSize1;

		width = width * colorCount;

		for( ; in1Y != in1YEnd; in1Y += rowSize1, in2Y += rowSize2 ) {
			const uint8_t * in1X = in1Y;
			const uint8_t * in2X = in2Y;
			const uint8_t * in1XEnd = in1X + width;

			for( ; in1X != in1XEnd; ++in1X, ++in2X ) {
				if( (*in1X) != (*in2X) )
					return false;
			}
		}

		return true;
	}

	Image LookupTable( const Image & in, const std::vector < uint8_t > & table )
	{
		ParameterValidation( in );

		Image out( in.width(), in.height() );

		LookupTable( in, 0, 0, out, 0, 0, out.width(), out.height(), table );

		return out;
	}

	void LookupTable( const Image & in, Image & out, const std::vector < uint8_t > & table )
	{
		ParameterValidation( in, out );

		LookupTable( in, 0, 0, out, 0, 0, out.width(), out.height(), table );
	}

	Image LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height,
					   const std::vector < uint8_t > & table )
	{
		ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		LookupTable( in, startXIn, startYIn, out, 0, 0, width, height, table );

		return out;
	}
	
	void LookupTable( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					  uint32_t width, uint32_t height, const std::vector < uint8_t > & table )
	{
		ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
		VerifyGrayScaleImage( in, out );

		if( table.size() != 256u )
			throw imageException("Lookup table size is not equal to 256");

		const uint32_t rowSizeIn  = in.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
			const uint8_t * inX  = inY;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for( ; outX != outXEnd; ++outX, ++inX )
				(*outX) = table[*inX];
		}
	}

	Image Maximum( const Image & in1, const Image & in2 )
	{
		ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		Maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Maximum( const Image & in1, const Image & in2, Image & out )
	{
		ParameterValidation( in1, in2, out );

		Maximum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		Maximum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}
	
	void Maximum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
		
		const uint8_t colorCount  = CommonColorCount( in1, in2, out );
		const uint32_t rowSize1   = in1.rowSize();
		const uint32_t rowSize2   = in2.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
		const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		width = width * colorCount;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
			const uint8_t * in1X = in1Y;
			const uint8_t * in2X = in2Y;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
				(*outX) = (*in2X) < (*in1X) ? (*in1X) : (*in2X);
		}
	}

	Image Merge( const Image & in1, const Image & in2, const Image & in3 )
	{
		ParameterValidation( in1, in2, in3 );

		Image out( in1.width(), in1.height(), RGB );

		Merge( in1, 0, 0, in2, 0, 0, in3, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Merge( const Image & in1, const Image & in2, const Image & in3, Image & out )
	{
		ParameterValidation( in1, in2, in3 );
		ParameterValidation( in1, out );

		Merge( in1, 0, 0, in2, 0, 0, in3, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
				 const Image & in3, uint32_t startXIn3, uint32_t startYIn3, uint32_t width, uint32_t height)
	{
		ParameterValidation(in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3, width, height);

		Image out( width, height, RGB );

		Merge( in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3, out, 0, 0, width, height );

		return out;
	}

	void Merge( const Image & in1, uint32_t startXIn1, uint32_t startYIn1, const Image & in2, uint32_t startXIn2, uint32_t startYIn2,
				const Image & in3, uint32_t startXIn3, uint32_t startYIn3, Image & out, uint32_t startXOut, uint32_t startYOut,
				uint32_t width, uint32_t height )
	{
		ParameterValidation(in1, startXIn1, startYIn1, in2, startXIn2, startYIn2, in3, startXIn3, startYIn3, width, height);
		ParameterValidation(out, startXOut, startYOut, width, height);
		VerifyGrayScaleImage( in1, in2, in3 );
		VerifyColoredImage  ( out );

		const uint8_t colorCount = RGB;

		if (colorCount != out.colorCount() )
			throw imageException("Color image is not 3-colored image");

		const uint32_t rowSizeIn1 = in1.rowSize();
		const uint32_t rowSizeIn2 = in2.rowSize();
		const uint32_t rowSizeIn3 = in3.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		width = width * colorCount;

		const uint8_t * in1Y = in1.data() + startYIn1 * rowSizeIn1 + startXIn1;
		const uint8_t * in2Y = in2.data() + startYIn2 * rowSizeIn2 + startXIn2;
		const uint8_t * in3Y = in3.data() + startYIn3 * rowSizeIn3 + startXIn3;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		for (; outY != outYEnd; outY += rowSizeOut, in1Y += rowSizeIn1, in2Y += rowSizeIn2, in3Y += rowSizeIn3) {
			const uint8_t * in1X = in1Y;
			const uint8_t * in2X = in2Y;
			const uint8_t * in3X = in3Y;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for (; outX != outXEnd; ) {
				*(outX++) = *(in1X++);
				*(outX++) = *(in2X++);
				*(outX++) = *(in3X++);
			}
		}
	}

	Image Minimum( const Image & in1, const Image & in2 )
	{
		ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		Minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Minimum( const Image & in1, const Image & in2, Image & out )
	{
		ParameterValidation( in1, in2, out );

		Minimum( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		Minimum( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Minimum( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				  Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
		
		const uint8_t colorCount  = CommonColorCount( in1, in2, out );
		const uint32_t rowSize1   = in1.rowSize();
		const uint32_t rowSize2   = in2.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
		const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		width = width * colorCount;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
			const uint8_t * in1X = in1Y;
			const uint8_t * in2X = in2Y;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
				(*outX) = (*in2X) > (*in1X) ? (*in1X) : (*in2X);
		}
	}

	Image Normalize( const Image & in )
	{
		ParameterValidation( in );

		Image out( in.width(), in.height() );

		Normalize( in, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Normalize( const Image & in, Image & out )
	{
		ParameterValidation( in, out );

		Normalize( in, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Normalize( in, startXIn, startYIn, out, 0, 0, width, height );

		return out;
	}
	
	void Normalize( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					uint32_t width, uint32_t height )
	{
		ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
		VerifyGrayScaleImage( in, out );

		const uint32_t rowSizeIn = in.rowSize();

		const uint8_t * inY    = in.data()  + startYIn  * rowSizeIn  + startXIn;
		const uint8_t * inYEnd = inY + height * rowSizeIn;

		uint8_t minimum = 255;
		uint8_t maximum = 0;

		for( ; inY != inYEnd; inY += rowSizeIn ) {
			const uint8_t * inX = inY;
			const uint8_t * inXEnd = inX + width;

			for( ; inX != inXEnd; ++inX ) {
				if( minimum > (*inX) )
					minimum = (*inX);

				if( maximum < (*inX) )
					maximum = (*inX);
			}
		}

		if( (minimum == 0 && maximum == 255) || (minimum == maximum) ) {
			Copy( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
		}
		else {
			const double correction = 255.0 / ( maximum - minimum );

			// We precalculate all values and store them in lookup table
			std::vector < uint8_t > value(256);

			for( uint16_t i = 0; i < 256; ++i )
				value[i] = static_cast < uint8_t >( ( i - minimum ) * correction + 0.5);

			LookupTable( in, startXIn, startYIn, out, startXOut, startYOut, width, height, value );
		}
	}

	std::vector < uint32_t > ProjectionProfile( const Image & image, bool horizontal )
	{
		std::vector < uint32_t > projection;

		ProjectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );

		return projection;
	}

	void ProjectionProfile( const Image & image, bool horizontal, std::vector < uint32_t > & projection )
	{
		ProjectionProfile( image, 0, 0, image.width(), image.height(), horizontal, projection );
	}

	std::vector < uint32_t > ProjectionProfile( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height, bool horizontal )
	{
		std::vector < uint32_t > projection;

		ProjectionProfile( image, x, y, width, height, horizontal, projection );

		return projection;
	}

	void ProjectionProfile( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height, bool horizontal,
							std::vector < uint32_t > & projection )
	{
		ParameterValidation( image, x, y, width, height );
		VerifyGrayScaleImage( image );

		projection.resize( horizontal ? width : height );
		std::fill( projection.begin(), projection.end(), 0u );

		const uint32_t rowSize = image.rowSize();

		if( horizontal ) {
			const uint8_t * imageX = image.data() + y * rowSize + x;
			const uint8_t * imageXEnd = imageX + width;

			std::vector < uint32_t > :: iterator data = projection.begin();

			for( ; imageX != imageXEnd; ++imageX, ++data ) {
				const uint8_t * imageY    = imageX;
				const uint8_t * imageYEnd = imageY + height * rowSize;

				for( ; imageY != imageYEnd; imageY += rowSize )
					(*data) += (*imageY);
			}
		}
		else {
			const uint8_t * imageY = image.data() + y * rowSize + x;
			const uint8_t * imageYEnd = imageY + height * rowSize;

			std::vector < uint32_t > :: iterator data = projection.begin();

			for( ; imageY != imageYEnd; imageY += rowSize, ++data ) {
				const uint8_t * imageX    = imageY;
				const uint8_t * imageXEnd = imageX + width;

				for( ; imageX != imageXEnd; ++imageX )
					(*data) += (*imageX);
			}
		}
	}

	Image Resize( const Image & in, uint32_t widthOut, uint32_t heightOut )
	{
		ParameterValidation( in );

		Image out( widthOut, heightOut );

		Resize( in, 0, 0, in.width(), in.height(), out, 0, 0, widthOut, heightOut );

		return out;
	}

	void Resize( const Image & in, Image & out )
	{
		ParameterValidation( in );
		ParameterValidation( out );

		Resize( in, 0, 0, in.width(), in.height(), out, 0, 0, out.width(), out.height() );
	}

	Image Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
				  uint32_t widthOut, uint32_t heightOut )
	{
		ParameterValidation( in, startXIn, startYIn, widthIn, heightIn );

		Image out( widthOut, heightOut );

		Resize( in, startXIn, startYIn, widthIn, heightIn, out, 0, 0, widthOut, heightOut );

		return out;
	}

	void Resize( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t widthIn, uint32_t heightIn,
				 Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t widthOut, uint32_t heightOut )
	{
		ParameterValidation( in,  startXIn,  startYIn,  widthIn,  heightIn );
		ParameterValidation( out, startXOut, startYOut, widthOut, heightOut );
		VerifyGrayScaleImage( in, out );

		const uint32_t rowSizeIn  = in.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + heightOut * rowSizeOut;

		uint32_t idY = 0;

		for( ; outY != outYEnd; outY += rowSizeOut, ++idY ) {
			const uint8_t * inX  = inY + (idY * heightIn / heightOut) * rowSizeIn;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + widthOut;

			uint32_t idX = 0;

			for( ; outX != outXEnd; ++outX, ++idX )
				(*outX) = *(inX + idX * widthIn / widthOut);
		}
	}

	Image RgbToBgr(const Image & in)
	{
		ParameterValidation(in);

		Image out(in.width(), in.height(), 3u);

		RgbToBgr(in, 0, 0, out, 0, 0, in.width(), in.height());

		return out;
	}

	void RgbToBgr(const Image & in, Image & out)
	{
		ParameterValidation(in, out);

		RgbToBgr(in, 0, 0, out, 0, 0, in.width(), in.height());
	}

	Image RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		ParameterValidation(in, startXIn, startYIn, width, height);

		Image out(width, height, 3u);

		RgbToBgr(in, startXIn, startYIn, out, 0, 0, width, height);

		return out;
	}

	void RgbToBgr( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
				  uint32_t width, uint32_t height )
	{
		ParameterValidation(in, startXIn, startYIn, out, startXOut, startYOut, width, height);
		VerifyColoredImage( in, out );

		const uint8_t colorCount = RGB;

		const uint32_t rowSizeIn  = in.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		width = width * colorCount;

		const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn  * colorCount;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
			const uint8_t * inX  = inY;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for( ; outX != outXEnd; outX += colorCount, inX += colorCount ) {
				*(outX + 2) = *(inX    );
				*(outX + 1) = *(inX + 1);
				*(outX    ) = *(inX + 2);
			}
		}
	}

	void Rotate( const Image & in, double centerXIn, double centerYIn, Image & out, double centerXOut, double centerYOut, double angle )
	{
		ParameterValidation( in, out );
		VerifyGrayScaleImage( in, out );

		const double cosAngle = cos(angle);
		const double sinAngle = sin(angle);

		const uint32_t rowSizeIn  = in.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint32_t width  = in.width();
		const uint32_t height = in.height();

		const uint8_t * inY  = in.data();
		uint8_t       * outY = out.data();
		const uint8_t * outYEnd = outY + height * rowSizeOut;
	
		double inXPos = -( cosAngle * centerXOut + sinAngle * centerYOut) + centerXIn;
		double inYPos = -(-sinAngle * centerXOut + cosAngle * centerYOut) + centerYIn;

		for( ; outY != outYEnd; outY += rowSizeOut, inXPos += sinAngle, inYPos += cosAngle ) {
			uint8_t       * outX = outY;
			const uint8_t * outXEnd = outX + width;

			double posX = inXPos;
			double posY = inYPos;

			for( ; outX != outXEnd; ++outX, posX += cosAngle, posY -= sinAngle ) {
				if( posX < 0 || posY < 0 ) {
					(*outX) = 0; // we actually do not know what is beyond an image so we set value 0
				}
				else {
					uint32_t x = static_cast<uint32_t>(posX);
					uint32_t y = static_cast<uint32_t>(posY);

					if( x >= width - 1 || y >= height - 1 ) {
						(*outX) = 0; // we actually do not know what is beyond an image so we set value 0
					}
					else {
						const uint8_t * inX = inY + y * rowSizeIn + x;

						// we use bilinear approximation to find pixel intensity value
						double coeffX = posX - x;
						double coeffY = posY - y;

						double sum = (*inX                ) * (1 - coeffX) * (1 - coeffY) +
									 (*inX + 1            ) * (    coeffX) * (1 - coeffY) +
									 (*inX + rowSizeIn    ) * (1 - coeffX) * (    coeffY) +
									 (*inX + rowSizeIn + 1) * (    coeffX) * (    coeffY) + 0.5;

						(*outX) = static_cast<uint8_t>( sum );
					}
				}
			}
		}
	}

	void SetPixel( Image & image, uint32_t x, uint32_t y, uint8_t value )
	{
		if( image.empty() || x >= image.width() || y >= image.height() || image.colorCount() != GRAY_SCALE )
			throw imageException("Position of point [x, y] is out of image");

		*(image.data() + y * image.rowSize() + x) = value;
	}

	void SetPixel( Image & image, const std::vector < uint32_t > & X, const std::vector < uint32_t > & Y, uint8_t value )
	{
		if( image.empty() || X.empty() || X.size() != Y.size() || image.colorCount() != GRAY_SCALE )
			throw imageException("Bad input parameters in image function");

		const uint32_t rowSize = image.rowSize();
		uint8_t * data = image.data();

		std::vector < uint32_t >::const_iterator x   = X.begin();
		std::vector < uint32_t >::const_iterator y   = Y.begin();
		std::vector < uint32_t >::const_iterator end = X.end();

		const uint32_t width  = image.width();
		const uint32_t height = image.height();

		for( ; x != end; ++x, ++y ) {
			if( (*x) >= width || (*y) >= height )
				throw imageException("Position of point [x, y] is out of image");

			*(data + (*y) * rowSize + (*x)) = value;
		}
	}

	void Split( const Image & in, Image & out1, Image & out2, Image & out3 )
	{
		ParameterValidation( out1, out2, out3 );
		ParameterValidation( in, out1 );

		Split( in, 0, 0, out1, 0, 0, out2, 0, 0, out3, 0, 0, in.width(), in.height() );
	}

	void Split( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out1, uint32_t startXOut1, uint32_t startYOut1,
				Image & out2, uint32_t startXOut2, uint32_t startYOut2, Image & out3, uint32_t startXOut3, uint32_t startYOut3,
				uint32_t width, uint32_t height)
	{
		ParameterValidation(in, startXIn, startYIn, width, height);
		ParameterValidation(out1, startXOut1, startYOut1, out2, startXOut2, startYOut2, out3, startXOut3, startYOut3, width, height);
		VerifyColoredImage  ( in );
		VerifyGrayScaleImage( out1, out2, out3 );

		const uint8_t colorCount = RGB;

		const uint32_t rowSizeIn   = in  .rowSize();
		const uint32_t rowSizeOut1 = out1.rowSize();
		const uint32_t rowSizeOut2 = out2.rowSize();
		const uint32_t rowSizeOut3 = out3.rowSize();
		
		width = width * colorCount;

		const uint8_t * inY = in.data() + startYIn * rowSizeIn + startXIn * colorCount;
		uint8_t * out1Y = out1.data() + startYOut1 * rowSizeOut1 + startXOut1;
		uint8_t * out2Y = out2.data() + startYOut2 * rowSizeOut2 + startXOut2;
		uint8_t * out3Y = out3.data() + startYOut3 * rowSizeOut3 + startXOut3;

		const uint8_t * inYEnd = inY + height * rowSizeIn;

		for( ; inY != inYEnd; inY += rowSizeIn, out1Y += rowSizeOut1, out2Y += rowSizeOut2, out3Y += rowSizeOut3 ) {
			const uint8_t * inX = inY;
			uint8_t * out1X = out1Y;
			uint8_t * out2X = out2Y;
			uint8_t * out3X = out3Y;

			const uint8_t * inXEnd = inX + width;

			for (; inX != inXEnd; ) {
				*(out1X++) = *(inX++);
				*(out2X++) = *(inX++);
				*(out3X++) = *(inX++);
			}
		}
	}

	Image Subtract( const Image & in1, const Image & in2 )
	{
		ParameterValidation( in1, in2 );

		Image out( in1.width(), in1.height() );

		Subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Subtract( const Image & in1, const Image & in2, Image & out )
	{
		ParameterValidation( in1, in2, out );

		Subtract( in1, 0, 0, in2, 0, 0, out, 0, 0, out.width(), out.height() );
	}

	Image Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
					uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, width, height );

		Image out( width, height );

		Subtract( in1, startX1, startY1, in2, startX2, startY2, out, 0, 0, out.width(), out.height() );

		return out;
	}

	void Subtract( const Image & in1, uint32_t startX1, uint32_t startY1, const Image & in2, uint32_t startX2, uint32_t startY2,
				   Image & out, uint32_t startXOut, uint32_t startYOut, uint32_t width, uint32_t height )
	{
		ParameterValidation( in1, startX1, startY1, in2, startX2, startY2, out, startXOut, startYOut, width, height );
		
		const uint8_t colorCount  = CommonColorCount( in1, in2, out );
		const uint32_t rowSize1   = in1.rowSize();
		const uint32_t rowSize2   = in2.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * in1Y = in1.data() + startY1   * rowSize1   + startX1   * colorCount;
		const uint8_t * in2Y = in2.data() + startY2   * rowSize2   + startX2   * colorCount;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut * colorCount;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		width = width * colorCount;

		for( ; outY != outYEnd; outY += rowSizeOut, in1Y += rowSize1, in2Y += rowSize2 ) {
			const uint8_t * in1X = in1Y;
			const uint8_t * in2X = in2Y;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for( ; outX != outXEnd; ++outX, ++in1X, ++in2X )
				(*outX) = (*in2X) > (*in1X) ? 0 : (*in1X) - (*in2X);		
		}
	}

	uint32_t Sum( const Image & image )
	{
		return Sum( image, 0, 0, image.width(), image.height() );
	}

	uint32_t Sum( const Image & image, uint32_t x, int32_t y, uint32_t width, uint32_t height )
	{
		ParameterValidation( image, x, y, width, height );
		VerifyGrayScaleImage( image );

		const uint32_t rowSize = image.rowSize();

		const uint8_t * imageY    = image.data() + y * rowSize + x;
		const uint8_t * imageYEnd = imageY + height * rowSize;

		uint32_t sum = 0;

		for( ; imageY != imageYEnd; imageY += rowSize ) {
			const uint8_t * imageX    = imageY;
			const uint8_t * imageXEnd = imageX + width;

			for( ; imageX != imageXEnd; ++imageX )
				sum += (*imageX);
		}

		return sum;
	}

	Image Threshold( const Image & in, uint8_t threshold )
	{
		ParameterValidation( in );

		Image out( in.width(), in.height() );

		Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), threshold );

		return out;
	}

	void Threshold( const Image & in, Image & out, uint8_t threshold )
	{
		ParameterValidation( in, out );

		Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), threshold );
	}

	Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t threshold )
	{
		ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Threshold( in, startXIn, startYIn, out, 0, 0, width, height, threshold );

		return out;
	}

	void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					uint32_t width, uint32_t height, uint8_t threshold )
	{
		ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
		VerifyGrayScaleImage( in, out );

		const uint32_t rowSizeIn  = in.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
			const uint8_t * inX  = inY;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for( ; outX != outXEnd; ++outX, ++inX )
				(*outX) = (*inX) < threshold ? 0 : 255;
		}
	}

	Image Threshold( const Image & in, uint8_t minThreshold, uint8_t maxThreshold )
	{
		ParameterValidation( in );

		Image out( in.width(), in.height() );

		Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), minThreshold, maxThreshold );

		return out;
	}

	void Threshold( const Image & in, Image & out, uint8_t minThreshold, uint8_t maxThreshold )
	{
		ParameterValidation( in, out );

		Threshold( in, 0, 0, out, 0, 0, out.width(), out.height(), minThreshold, maxThreshold );
	}

	Image Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height, uint8_t minThreshold,
					 uint8_t maxThreshold )
	{
		ParameterValidation( in, startXIn, startYIn, width, height );

		Image out( width, height );

		Threshold( in, startXIn, startYIn, out, 0, 0, width, height, minThreshold, maxThreshold );

		return out;
	}

	void Threshold( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					uint32_t width, uint32_t height, uint8_t minThreshold, uint8_t maxThreshold )
	{
		ParameterValidation( in, startXIn, startYIn, out, startXOut, startYOut, width, height );
		VerifyGrayScaleImage( in, out );

		if( minThreshold > maxThreshold )
			throw imageException("Minimum threshold value is bigger than maximum threshold value");

		const uint32_t rowSizeIn  = in.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * inY  = in.data()  + startYIn  * rowSizeIn  + startXIn;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + height * rowSizeOut;

		for( ; outY != outYEnd; outY += rowSizeOut, inY += rowSizeIn ) {
			const uint8_t * inX  = inY;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + width;

			for( ; outX != outXEnd; ++outX, ++inX )
				(*outX) = (*inX) < minThreshold || (*inX) > maxThreshold ? 0 : 255;
		}
	}

	Image Transpose( const Image & in )
	{
		ParameterValidation( in );

		Image out(in.height(), in.width());

		Transpose( in, 0, 0, out, 0, 0, in.width(), in.height() );

		return out;
	}

	void Transpose( const Image & in, Image & out )
	{
		ParameterValidation( in );
		ParameterValidation( out );

		Transpose( in, 0, 0, out, 0, 0, in.width(), in.height() );
	}

	Image Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, uint32_t width, uint32_t height )
	{
		ParameterValidation( in, startXIn, startYIn, width, height );

		Image out(height, width);

		Transpose( in, startXIn, startYIn, out, 0, 0, width, height );

		return out;
	}
	
	void Transpose( const Image & in, uint32_t startXIn, uint32_t startYIn, Image & out, uint32_t startXOut, uint32_t startYOut,
					uint32_t width, uint32_t height )
	{
		ParameterValidation( in,  startXIn,  startYIn,  width,  height );
		ParameterValidation( out, startXOut, startYOut, height, width  );
		VerifyGrayScaleImage( in, out );

		const uint32_t rowSizeIn  = in.rowSize();
		const uint32_t rowSizeOut = out.rowSize();

		const uint8_t * inX  = in.data()  + startYIn  * rowSizeIn  + startXIn;
		uint8_t       * outY = out.data() + startYOut * rowSizeOut + startXOut;

		const uint8_t * outYEnd = outY + width * rowSizeOut;

		for( ; outY != outYEnd; outY += rowSizeOut, ++inX ) {
			const uint8_t * inY  = inX;
			uint8_t       * outX = outY;

			const uint8_t * outXEnd = outX + height;

			for( ; outX != outXEnd; ++outX, inY += rowSizeIn )
				(*outX) = *(inY);
		}
	}
};
