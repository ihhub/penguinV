#include <queue>
#include <numeric>
#include "blob_detection.h"
#include "image_function.h"

namespace
{
	template <typename Data>
	void listToVector( std::list <Data> & l, std::vector <Data> & v, uint32_t offset )
	{
		std::vector < Data > temp ( l.begin(), l.end() );

		std::for_each( temp.begin(), temp.end(), [&](uint32_t & value) { value = value + offset - 1; } ); // note that we extract value 1!

		std::swap(v, temp);
		l.clear();
	}

	const double pi = 3.1415926536;
};

namespace Blob_Detection
{
	const std::vector < uint32_t > & BlobInfo::pointX() const
	{
		return _pointX;
	}

	const std::vector < uint32_t > & BlobInfo::pointY() const
	{
		return _pointY;
	}

	const std::vector < uint32_t > & BlobInfo::contourX() const
	{
		return _contourX;
	}

	const std::vector < uint32_t > & BlobInfo::contourY() const
	{
		return _contourY;
	}

	const std::vector < uint32_t > & BlobInfo::edgeX() const
	{
		return _edgeX;
	}

	const std::vector < uint32_t > & BlobInfo::edgeY() const
	{
		return _edgeY;
	}

	Area BlobInfo::area()
	{
		_getArea();

		return _area.value;
	}

	Area BlobInfo::area() const
	{
		return _area.value;
	}

	Point BlobInfo::center()
	{
		_getCenter();

		return _center.value;
	}

	Point BlobInfo::center() const
	{
		return _center.value;
	}

	double BlobInfo::circularity()
	{
		_getCircularity();

		return _circularity.value;
	}

	double BlobInfo::circularity() const
	{
		return _circularity.value;
	}

	double BlobInfo::elongation()
	{
		_getElongation();

		return _elongation.value;
	}

	double BlobInfo::elongation() const
	{
		return _elongation.value;
	}

	uint32_t BlobInfo::height()
	{
		_getHeight();

		return _height.value;
	}

	uint32_t BlobInfo::height() const
	{
		return _height.value;
	}

	double BlobInfo::length()
	{
		_getLength();

		return _length.value;
	}

	double BlobInfo::length() const
	{
		return _length.value;
	}

	size_t BlobInfo::size() const
	{
		return _pointX.size();
	}

	uint32_t BlobInfo::width()
	{
		_getWidth();

		return _width.value;
	}
	
	uint32_t BlobInfo::width() const
	{
		return _width.value;
	}

	bool BlobInfo::isSolid() const
	{
		return _contourX.size() < _edgeX.size();
	}

	void BlobInfo::_getArea()
	{
		if( !_pointX.empty() && !_pointY.empty() && !_area.found ) {
			_area.value.left   = *( std::min_element( _contourX.begin(), _contourX.end() ) );
			_area.value.right  = *( std::max_element( _contourX.begin(), _contourX.end() ) ) + 1; // note that we add 1
			_area.value.top    = *( std::min_element( _contourY.begin(), _contourY.end() ) );
			_area.value.bottom = *( std::max_element( _contourY.begin(), _contourY.end() ) ) + 1; // note that we add 1

			_area.found = true;
		}
	}

	void BlobInfo::_getCenter()
	{
		if( !_pointX.empty() && !_pointY.empty() && !_center.found ) {
			_center.value.x = static_cast <double>( std::accumulate( _pointX.begin(), _pointX.end(), 0 ) ) /
							  static_cast <double>( size() );
			_center.value.y = static_cast <double>( std::accumulate( _pointY.begin(), _pointY.end(), 0 ) ) /
							  static_cast <double>( size() );

			_center.found = true;
		}
	}

	void BlobInfo::_getCircularity()
	{
		if( !_pointX.empty() && !_contourX.empty() && !_circularity.found ) {
			
			// this formula doesn't work properly :(
			//_circularity.value = ( 2 * sqrt(pi * static_cast<double>(size())) ) / static_cast<double>(_contourX.size());

			double radius = sqrt(static_cast<double>(size()) / pi);
			_getCenter();

			double difference = 0;

			std::vector < uint32_t >::const_iterator x   = _contourX.begin();
			std::vector < uint32_t >::const_iterator y   = _contourY.begin();
			std::vector < uint32_t >::const_iterator end = _contourX.end();

			for( ; x != end; ++x, ++y ) {
				difference += fabs( sqrt( (*x - _center.value.x) * (*x - _center.value.x) + 
										  (*y - _center.value.y) * (*y - _center.value.y) ) - radius );
			}

			_circularity.value = 1 - difference / ( static_cast<double>(_contourX.size()) * radius );
			
			_circularity.found = true;
		}
	}

	void BlobInfo::_getElongation()
	{
		if( !_contourX.empty() && !_contourY.empty() && !_elongation.found ) {
			if( _contourX.size() > 1 ) {
			
				std::vector < uint32_t >::const_iterator x   = _contourX.cbegin();
				std::vector < uint32_t >::const_iterator y   = _contourY.cbegin();
				std::vector < uint32_t >::const_iterator end = _contourX.cend();

				uint32_t maximumDistance = 0;

				Point startPoint, endPoint;

				for( ; x != (end - 1); ++x, ++y ) {

					std::vector < uint32_t >::const_iterator xx = x + 1;
					std::vector < uint32_t >::const_iterator yy = y + 1;

					for( ; xx != end; ++xx, ++yy ) {

						uint32_t distance = (*x - *xx) * (*x - *xx) + (*y - *yy) * (*y - *yy);

						if( maximumDistance < distance ) {
							maximumDistance = distance;

							startPoint.x = *x;
							startPoint.y = *y;

							endPoint.x = *xx;
							endPoint.y = *xx;
						}
					}

				}

				double length = sqrt( static_cast<double>( maximumDistance ) );
				double angle  = -atan2( static_cast< double >( endPoint.y - startPoint.y ),
										static_cast< double >( endPoint.x - startPoint.x ) );

				double _cos = cos(angle);
				double _sin = sin(angle);

				std::vector < double > contourYTemp( _contourY.begin(), _contourY.end() );

				std::vector < uint32_t >::const_iterator xRotated   = _contourX.begin();
				std::vector < double >::iterator yRotated           = contourYTemp.begin();
				std::vector < uint32_t >::const_iterator endRotated = _contourX.end();

				for( ; xRotated != endRotated; ++xRotated, ++yRotated )
					(*yRotated) = (*xRotated - startPoint.x) * _sin + (*yRotated - startPoint.y) * _cos;

				double height = *(std::max_element(contourYTemp.begin(), contourYTemp.end() ) ) -
								*(std::min_element(contourYTemp.begin(), contourYTemp.end() ) );

				if( height < 1 )
					height = 1;

				_elongation.value = length / height;
			}
			else {
				_elongation.value = 1;
			}

			_elongation.found = true;
		}
	}

	void BlobInfo::_getHeight()
	{
		if( !_pointY.empty() && !_height.found ) {
			_height.value = *( std::max_element( _contourY.begin(), _contourY.end() ) ) -
				*( std::min_element( _contourY.begin(), _contourY.end() ) );

			_height.found = true;
		}
	}

	void BlobInfo::_getLength()
	{
		if( !_contourX.empty() && !_contourY.empty() && !_length.found ) {
			if( _contourX.size() > 1 ) {

				std::vector < uint32_t >::const_iterator x   = _contourX.cbegin();
				std::vector < uint32_t >::const_iterator y   = _contourY.cbegin();
				std::vector < uint32_t >::const_iterator end = _contourX.cend();

				uint32_t maximumDistance = 0;

				for( ; x != (end - 1); ++x, ++y ) {

					std::vector < uint32_t >::const_iterator xx = x + 1;
					std::vector < uint32_t >::const_iterator yy = y + 1;

					for( ; xx != end; ++xx, ++yy ) {

						uint32_t distance = (*x - *xx) * (*x - *xx) + (*y - *yy) * (*y - *yy);

						if( maximumDistance < distance )
							maximumDistance = distance;
					}

				}

				_length.value = sqrt( static_cast<double>( maximumDistance ) );

			}
			else {
				_length.value = 0.5;
			}

			_length.found = true;
		}
	}

	void BlobInfo::_getWidth()
	{
		if( !_contourX.empty() && !_width.found ) {
			_width.value = *( std::max_element( _contourX.begin(), _contourX.end() ) ) -
				*( std::min_element( _contourX.begin(), _contourX.end() ) );

			_width.found = true;
		}
	}

	void BlobInfo::_preparePoints(uint32_t offsetX, uint32_t offsetY)
	{
		listToVector( _tempPointX  , _pointX  , offsetX );
		listToVector( _tempPointY  , _pointY  , offsetY );
		listToVector( _tempContourX, _contourX, offsetX );
		listToVector( _tempContourY, _contourY, offsetY );
		listToVector( _tempEdgeX   , _edgeX   , offsetX );
		listToVector( _tempEdgeY   , _edgeY   , offsetY );
	}


	const std::vector < BlobInfo > BlobDetection::find( const Bitmap_Image::Image & image, BlobParameters parameter, uint8_t threshold )
	{
		return find( image, 0, 0, image.width(), image.height(), parameter, threshold );
	}

	const std::vector < BlobInfo > BlobDetection::find( const Bitmap_Image::Image & image, uint32_t x, int32_t y, uint32_t width,
														uint32_t height, BlobParameters parameter, uint8_t threshold )
	{
		Image_Function::ParameterValidation( image, x, y, width, height );

		parameter._verify();

		_blob.clear();

		// Create a map of found pixels:
		// 0 - no pixel
		// 1 - not used pixel
		// 2 - found pixel
		// 3 - edge point
		// 4 - contour point
		// we make the area by 2 pixels in each direction bigger so we don't need to check borders of map
		std::vector < uint8_t > imageMap( (width + 2) * (height + 2), 0 );

		uint32_t rowSize = image.rowSize();

		const uint8_t * imageY    = image.data() + y * rowSize + x;
		const uint8_t * imageYEnd = imageY + height * rowSize;

		const uint32_t mapWidth = width + 2;

		std::vector < uint8_t >::iterator mapValueX = imageMap.begin() + mapWidth + 1;

		for( ; imageY != imageYEnd; imageY += rowSize, mapValueX += mapWidth ) {

			const uint8_t * imageX = imageY;
			const uint8_t * imageXEnd = imageX + width;

			std::vector < uint8_t >::iterator mapValueY = mapValueX;

			for( ; imageX != imageXEnd; ++imageX, ++mapValueY ) {
				*mapValueY = (*imageX) < threshold ? 0 : 1;
			}
		}

		// find all blobs
		std::list < BlobInfo > foundBlob;

		mapValueX = imageMap.begin() + mapWidth;
		std::vector < uint8_t >::const_iterator endMap = imageMap.end() - mapWidth;

		for( ; mapValueX != endMap; ++mapValueX ) {
		
			if( *mapValueX == 1 ) { // blob found!
				foundBlob.push_back( BlobInfo() );

				uint32_t relativePosition = static_cast<uint32_t>( mapValueX - imageMap.begin() );

				foundBlob.back()._tempPointX.push_back( relativePosition % mapWidth );
				foundBlob.back()._tempPointY.push_back( relativePosition / mapWidth );

				*mapValueX = 2;

				std::list < uint32_t >::iterator xIter = foundBlob.back()._tempPointX.begin();
				std::list < uint32_t >::iterator yIter = foundBlob.back()._tempPointY.begin();

				do {
					uint32_t xMap = *xIter;
					uint32_t yMap = *yIter;

					std::vector < uint8_t >::iterator position = imageMap.begin() + yMap * mapWidth + xMap;

					uint8_t neighbourCount = 0;

					position = position - 1 - mapWidth;
					if( *(position) > 0 ) {
						if( *(position) == 1 ) {
							foundBlob.back()._tempPointX.push_back( xMap - 1 );
							foundBlob.back()._tempPointY.push_back( yMap - 1 );
							*(position) = 2;
						}
						++neighbourCount;
					}

					++position;
					if( *(position) > 0 ) {
						if( *(position) == 1 ) {
							foundBlob.back()._tempPointX.push_back( xMap     );
							foundBlob.back()._tempPointY.push_back( yMap - 1 );
							*(position) = 2;
						}
						++neighbourCount;
					}

					++position;
					if( *(position) > 0 ) {
						if( *(position) == 1 ) {
							foundBlob.back()._tempPointX.push_back( xMap + 1 );
							foundBlob.back()._tempPointY.push_back( yMap - 1 );
							*(position) = 2;
						}
						++neighbourCount;
					}

					position = position - 2 + mapWidth;
					if( *(position) > 0 ) {
						if( *(position) == 1 ) {
							foundBlob.back()._tempPointX.push_back( xMap - 1 );
							foundBlob.back()._tempPointY.push_back( yMap     );
							*(position) = 2;
						}
						++neighbourCount;
					}

					position = position + 2;
					if( *(position) > 0 ) {
						if( *(position) == 1 ) {
							foundBlob.back()._tempPointX.push_back( xMap + 1 );
							foundBlob.back()._tempPointY.push_back( yMap     );
							*(position) = 2;
						}
						++neighbourCount;
					}

					position = position - 2 + mapWidth;
					if( *(position) > 0 ) {
						if( *(position) == 1 ) {
							foundBlob.back()._tempPointX.push_back( xMap - 1 );
							foundBlob.back()._tempPointY.push_back( yMap + 1 );
							*(position) = 2;
						}
						++neighbourCount;
					}

					++position;
					if( *(position) > 0 ) {
						if( *(position) == 1 ) {
							foundBlob.back()._tempPointX.push_back( xMap     );
							foundBlob.back()._tempPointY.push_back( yMap + 1 );
							*(position) = 2;
						}
						++neighbourCount;
					}

					++position;
					if( *(position) > 0 ) {
						if( *(position) == 1 ) {
							foundBlob.back()._tempPointX.push_back( xMap + 1 );
							foundBlob.back()._tempPointY.push_back( yMap + 1 );
							*(position) = 2;
						}
						++neighbourCount;
					}

					if( neighbourCount != 8 ) {
						foundBlob.back()._tempEdgeX.push_back( xMap );
						foundBlob.back()._tempEdgeY.push_back( yMap );
						*(position - 1 - mapWidth) = 3;
					}

					++xIter;
					++yIter;

				} while( xIter != foundBlob.back()._tempPointX.end() );

				// Now we can extract outer edge points or so called contour points
				foundBlob.back()._tempContourX.push_back( relativePosition % mapWidth );
				foundBlob.back()._tempContourY.push_back( relativePosition / mapWidth );

				xIter = foundBlob.back()._tempContourX.begin();
				yIter = foundBlob.back()._tempContourY.begin();

				*mapValueX = 4;

				do {
					uint32_t xMap = *xIter;
					uint32_t yMap = *yIter;

					std::vector < uint8_t >::iterator position = imageMap.begin() + yMap * mapWidth + xMap;

					position = position - mapWidth;
					if( *(position) == 3 ) {
						foundBlob.back()._tempContourX.push_back( xMap     );
						foundBlob.back()._tempContourY.push_back( yMap - 1 );
						*(position) = 4;
					}

					position = position - 1 + mapWidth;
					if( *(position) == 3 ) {
						foundBlob.back()._tempContourX.push_back( xMap - 1 );
						foundBlob.back()._tempContourY.push_back( yMap     );
						*(position) = 4;
					}

					position = position + 2;
					if( *(position) == 3 ) {
						foundBlob.back()._tempContourX.push_back( xMap + 1 );
						foundBlob.back()._tempContourY.push_back( yMap     );
						*(position) = 4;
					}

					position = position - 1 + mapWidth;
					if( *(position) == 3 ) {
						foundBlob.back()._tempContourX.push_back( xMap     );
						foundBlob.back()._tempContourY.push_back( yMap + 1 );
						*(position) = 4;
					}

					++xIter;
					++yIter;

				} while( xIter != foundBlob.back()._tempContourX.end() );

			}

		}

		std::for_each( foundBlob.begin(), foundBlob.end(), [&](BlobInfo & info) { info._preparePoints(x, y); } );

		// All blobs found. Now we need to sort them
		if( parameter.circularity.checkMaximum || parameter.circularity.checkMinimum ) {
			std::for_each( foundBlob.begin(), foundBlob.end(), [](BlobInfo & info) { info._getCircularity(); } );

			foundBlob.remove_if( [&](const BlobInfo & info) {
				return (parameter.circularity.checkMaximum && info.circularity() > parameter.circularity.maximum) ||
					   (parameter.circularity.checkMinimum && info.circularity() < parameter.circularity.minimum); } );
		}

		if( parameter.elongation.checkMaximum || parameter.elongation.checkMinimum ) {
			std::for_each( foundBlob.begin(), foundBlob.end(), [](BlobInfo & info) { info._getElongation(); } );

			foundBlob.remove_if( [&](const BlobInfo & info) {
				return (parameter.elongation.checkMaximum && info.elongation() > parameter.elongation.maximum) ||
					   (parameter.elongation.checkMinimum && info.elongation() < parameter.elongation.minimum); } );
		}

		if( parameter.height.checkMaximum || parameter.height.checkMinimum ) {
			std::for_each( foundBlob.begin(), foundBlob.end(), [](BlobInfo & info) { info._getHeight(); } );

			foundBlob.remove_if( [&](const BlobInfo & info) {
				return (parameter.height.checkMaximum && info.height() > parameter.height.maximum) ||
					   (parameter.height.checkMinimum && info.height() < parameter.height.minimum); } );
		}

		if( parameter.length.checkMaximum || parameter.length.checkMinimum ) {
			std::for_each( foundBlob.begin(), foundBlob.end(), [](BlobInfo & info) { info._getLength(); } );

			foundBlob.remove_if( [&](const BlobInfo & info) {
				return (parameter.length.checkMaximum && info.length() > parameter.length.maximum) ||
					   (parameter.length.checkMinimum && info.length() < parameter.length.minimum); } );
		}

		if( parameter.size.checkMaximum || parameter.size.checkMinimum ) {
			foundBlob.remove_if( [&](const BlobInfo & info) {
				return (parameter.size.checkMaximum && info.size() > parameter.size.maximum) ||
					   (parameter.size.checkMinimum && info.size() < parameter.size.minimum); } );
		}

		if( parameter.width.checkMaximum || parameter.width.checkMinimum ) {
			std::for_each( foundBlob.begin(), foundBlob.end(), [](BlobInfo & info) { info._getWidth(); } );

			foundBlob.remove_if( [&](const BlobInfo & info) {
				return (parameter.width.checkMaximum && info.width() > parameter.width.maximum) ||
					   (parameter.width.checkMinimum && info.width() < parameter.width.minimum); } );
		}

		// prepare data for output
		std::vector <BlobInfo> blobTemp ( foundBlob.begin(), foundBlob.end() );
		std::swap(_blob, blobTemp);

		return get();
	}

	const std::vector < BlobInfo > & BlobDetection::get() const
	{
		return _blob;
	}

	std::vector < BlobInfo > & BlobDetection::get()
	{
		return _blob;
	}

	const BlobInfo & BlobDetection::getBestBlob(SortCriterion criterion) const
	{
		switch(criterion) {
			case CIRCULARITY :
				return *( std::max_element( _blob.begin(), _blob.end(), [](const BlobInfo & blob1, const BlobInfo & blob2)
											{ return blob1.circularity() < blob2.circularity(); } ) );
			case ELONGATION :
				return *( std::max_element( _blob.begin(), _blob.end(), [](const BlobInfo & blob1, const BlobInfo & blob2)
											{ return blob1.elongation() < blob2.elongation(); } ) );
			case HEIGHT :
				return *( std::max_element( _blob.begin(), _blob.end(), [](const BlobInfo & blob1, const BlobInfo & blob2)
											{ return blob1.height() < blob2.height(); } ) );
			case LENGTH :
				return *( std::max_element( _blob.begin(), _blob.end(), [](const BlobInfo & blob1, const BlobInfo & blob2)
											{ return blob1.length() < blob2.length(); } ) );
			case SIZE :
				return *( std::max_element( _blob.begin(), _blob.end(), [](const BlobInfo & blob1, const BlobInfo & blob2)
											{ return blob1.size() < blob2.size(); } ) );
			case WIDTH :
				return *( std::max_element( _blob.begin(), _blob.end(), [](const BlobInfo & blob1, const BlobInfo & blob2)
											{ return blob1.width() < blob2.width(); } ) );
			default:
				throw imageException( "Bad criterion for blob finding" );
		}
	}

	void BlobDetection::sort(SortCriterion criterion)
	{
		switch(criterion) {
			case CIRCULARITY :
				std::sort( _blob.begin(), _blob.end(), [](BlobInfo & blob1, BlobInfo & blob2)
											{ return blob1.circularity() > blob2.circularity(); } );
				break;
			case ELONGATION :
				std::sort( _blob.begin(), _blob.end(), [](BlobInfo & blob1, BlobInfo & blob2)
											{ return blob1.elongation() > blob2.elongation(); } );
				break;
			case HEIGHT :
				std::sort( _blob.begin(), _blob.end(), [](BlobInfo & blob1, BlobInfo & blob2)
											{ return blob1.height() > blob2.height(); } );
				break;
			case LENGTH :
				std::sort( _blob.begin(), _blob.end(), [](BlobInfo & blob1, BlobInfo & blob2)
											{ return blob1.length() > blob2.length(); } );
				break;
			case SIZE :
				std::sort( _blob.begin(), _blob.end(), [](BlobInfo & blob1, BlobInfo & blob2)
											{ return blob1.size() > blob2.size(); } );
				break;
			case WIDTH :
				std::sort( _blob.begin(), _blob.end(), [](BlobInfo & blob1, BlobInfo & blob2)
											{ return blob1.width() > blob2.width(); } );
				break;
			default:
				throw imageException( "Bad criterion for blob sorting" );
		}
	}
}
