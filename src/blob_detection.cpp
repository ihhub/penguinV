#include <cmath>
#include <list>
#include <numeric>
#include <queue>
#include "blob_detection.h"
#include "parameter_validation.h"

namespace
{
    enum PixelState
    {
        EMPTY      = 0u,
        NOT_IN_USE = 1u,
        FOUND      = 2u,
        EDGE       = 3u,
        CONTOUR    = 4u
    };
}

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

    Point2d BlobInfo::center()
    {
        _getCenter();

        return _center.value;
    }

    Point2d BlobInfo::center() const
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
        return _contourX.size() == _edgeX.size();
    }

    void BlobInfo::_getArea()
    {
        if( !_contourX.empty() && !_contourY.empty() && !_area.found ) {
            _area.value.left   = *(std::min_element( _contourX.begin(), _contourX.end() ));
            _area.value.right  = *(std::max_element( _contourX.begin(), _contourX.end() )) + 1; // note that we add 1
            _area.value.top    = *(std::min_element( _contourY.begin(), _contourY.end() ));
            _area.value.bottom = *(std::max_element( _contourY.begin(), _contourY.end() )) + 1; // note that we add 1

            _area.found = true;
        }
    }

    void BlobInfo::_getCenter()
    {
        if( !_pointX.empty() && !_pointY.empty() && !_center.found ) {
            _center.value.x = static_cast <double>(std::accumulate( _pointX.begin(), _pointX.end(), 0 )) /
                static_cast <double>(size());
            _center.value.y = static_cast <double>(std::accumulate( _pointY.begin(), _pointY.end(), 0 )) /
                static_cast <double>(size());

            _center.found = true;
        }
    }

    void BlobInfo::_getCircularity()
    {
        if( !_contourX.empty() && !_circularity.found ) {
            const double radius = sqrt( static_cast<double>(size()) / pvmath::pi );
            _getCenter();

            double difference = 0;

            std::vector < uint32_t >::const_iterator x   = _contourX.begin();
            std::vector < uint32_t >::const_iterator y   = _contourY.begin();
            std::vector < uint32_t >::const_iterator end = _contourX.end();

            for( ; x != end; ++x, ++y ) {
                difference += fabs( sqrt( (*x - _center.value.x) * (*x - _center.value.x) +
                    (*y - _center.value.y) * (*y - _center.value.y) ) - radius );
            }

            _circularity.value = 1 - difference / (static_cast<double>(_contourX.size()) * radius);

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

                Point2d startPoint, endPoint;

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

                const double length = sqrt( static_cast<double>(maximumDistance) );
                const double angle  = -atan2( static_cast<double>(endPoint.y - startPoint.y),
                                              static_cast<double>(endPoint.x - startPoint.x) );

                const double _cos = cos( angle );
                const double _sin = sin( angle );

                std::vector < double > contourYTemp( _contourY.begin(), _contourY.end() );

                std::vector < uint32_t >::const_iterator xRotated   = _contourX.begin();
                std::vector < double >::iterator yRotated           = contourYTemp.begin();
                std::vector < uint32_t >::const_iterator endRotated = _contourX.end();

                for( ; xRotated != endRotated; ++xRotated, ++yRotated )
                    (*yRotated) = (*xRotated - startPoint.x) * _sin + (*yRotated - startPoint.y) * _cos;

                double height = *(std::max_element( contourYTemp.begin(), contourYTemp.end() )) -
                    *(std::min_element( contourYTemp.begin(), contourYTemp.end() ));

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
        if( !_contourY.empty() && !_height.found ) {
            _height.value = *(std::max_element( _contourY.begin(), _contourY.end() )) -
                *(std::min_element( _contourY.begin(), _contourY.end() )) + 1;

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

                int32_t maximumDistance = 0;

                for( ; x != (end - 1); ++x, ++y ) {
                    std::vector < uint32_t >::const_iterator xx = x + 1;
                    std::vector < uint32_t >::const_iterator yy = y + 1;

                    for( ; xx != end; ++xx, ++yy ) {
                        int32_t distance = static_cast<int32_t>(*x - *xx) * static_cast<int32_t>(*x - *xx) +
                            static_cast<int32_t>(*y - *yy) * static_cast<int32_t>(*y - *yy);

                        if( maximumDistance < distance )
                            maximumDistance = distance;
                    }
                }

                _length.value = sqrt( static_cast<double>(maximumDistance) );
            }
            else {
                _length.value = 0;
            }

            _length.found = true;
        }
    }

    void BlobInfo::_getWidth()
    {
        if( !_contourX.empty() && !_width.found ) {
            _width.value = *(std::max_element( _contourX.begin(), _contourX.end() )) -
                *(std::min_element( _contourX.begin(), _contourX.end() )) + 1;

            _width.found = true;
        }
    }


    const std::vector < BlobInfo > & BlobDetection::find( const PenguinV_Image::Image & image, BlobParameters parameter, uint8_t threshold )
    {
        return find( image, 0, 0, image.width(), image.height(), parameter, threshold );
    }

    const std::vector < BlobInfo > & BlobDetection::find( const PenguinV_Image::Image & image, uint32_t x, uint32_t y, uint32_t width,
                                                          uint32_t height, BlobParameters parameter, uint8_t threshold )
    {
        Image_Function::ParameterValidation( image, x, y, width, height );
        Image_Function::VerifyGrayScaleImage( image );

        parameter._verify();

        _blob.clear();

        // we make the area by 2 pixels bigger in each direction so we don't need to check borders of map
        std::vector < uint8_t > imageMap( (width + 2) * (height + 2), EMPTY );

        uint32_t rowSize = image.rowSize();

        const uint8_t * imageY    = image.data() + y * rowSize + x;
        const uint8_t * imageYEnd = imageY + height * rowSize;

        const uint32_t mapWidth = width + 2;

        std::vector < uint8_t >::iterator mapValueX = imageMap.begin() + mapWidth + 1;

        for( ; imageY != imageYEnd; imageY += rowSize, mapValueX += mapWidth ) {
            const uint8_t * imageX    = imageY;
            const uint8_t * imageXEnd = imageX + width;

            std::vector < uint8_t >::iterator mapValueY = mapValueX;

            for( ; imageX != imageXEnd; ++imageX, ++mapValueY ) {
                if( (*imageX) >= threshold )
                    *mapValueY = NOT_IN_USE;
            }
        }

        // find all blobs
        std::list < BlobInfo > foundBlob;

        mapValueX = imageMap.begin() + mapWidth;
        std::vector < uint8_t >::const_iterator endMap = imageMap.end() - mapWidth;

        for( ; mapValueX != endMap; ++mapValueX ) {
            if( *mapValueX == NOT_IN_USE ) { // blob found!
                foundBlob.push_back( BlobInfo() );

                BlobInfo & newBlob = foundBlob.back();

                uint32_t relativePosition = static_cast<uint32_t>(mapValueX - imageMap.begin());

                std::vector < uint32_t > & pointX = newBlob._pointX;
                std::vector < uint32_t > & pointY = newBlob._pointY;

                // we put extra shift [-1, -1] to point position because our map starts from [1, 1]
                // not from [0, 0]
                pointX.push_back( relativePosition % mapWidth + x - 1 );
                pointY.push_back( relativePosition / mapWidth + y - 1 );

                std::vector < uint32_t > & edgeX = newBlob._edgeX;
                std::vector < uint32_t > & edgeY = newBlob._edgeY;

                *mapValueX = FOUND;

                size_t pointId = 0;

                do {
                    uint32_t xMap = pointX[pointId];
                    uint32_t yMap = pointY[pointId++];

                    uint8_t neighbourCount = 0;

                    uint8_t * position = imageMap.data() + (yMap + 1 - y) * mapWidth + (xMap + 1 - x);

                    position = position - mapWidth - 1;
                    if( *(position) != EMPTY ) {
                        if( *(position) == NOT_IN_USE ) {
                            pointX.push_back( xMap - 1 );
                            pointY.push_back( yMap - 1 );
                            *(position) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    ++position;
                    if( *(position) != EMPTY ) {
                        if( *(position) == NOT_IN_USE ) {
                            pointX.push_back( xMap );
                            pointY.push_back( yMap - 1 );
                            *(position) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    ++position;
                    if( *(position) != EMPTY ) {
                        if( *(position) == NOT_IN_USE ) {
                            pointX.push_back( xMap + 1 );
                            pointY.push_back( yMap - 1 );
                            *(position) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    position = position + width; // (mapWidth - 2) is width
                    if( *(position) != EMPTY ) {
                        if( *(position) == NOT_IN_USE ) {
                            pointX.push_back( xMap - 1 );
                            pointY.push_back( yMap );
                            *(position) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    position = position + 2;
                    if( *(position) != EMPTY ) {
                        if( *(position) == NOT_IN_USE ) {
                            pointX.push_back( xMap + 1 );
                            pointY.push_back( yMap );
                            *(position) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    position = position + width; // (mapWidth - 2) is width
                    if( *(position) != EMPTY ) {
                        if( *(position) == NOT_IN_USE ) {
                            pointX.push_back( xMap - 1 );
                            pointY.push_back( yMap + 1 );
                            *(position) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    ++position;
                    if( *(position) != EMPTY ) {
                        if( *(position) == NOT_IN_USE ) {
                            pointX.push_back( xMap );
                            pointY.push_back( yMap + 1 );
                            *(position) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    ++position;
                    if( *(position) != EMPTY ) {
                        if( *(position) == NOT_IN_USE ) {
                            pointX.push_back( xMap + 1 );
                            pointY.push_back( yMap + 1 );
                            *(position) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    if( neighbourCount != 8 ) {
                        edgeX.push_back( xMap );
                        edgeY.push_back( yMap );
                        *(position - 1 - mapWidth) = EDGE;
                    }
                } while( pointId != pointX.size() );

                // Now we can extract outer edge points or so called contour points
                std::vector < uint32_t > & contourX = newBlob._contourX;
                std::vector < uint32_t > & contourY = newBlob._contourY;

                // we put extra shift [-1, -1] to point position because our map starts from [1, 1]
                // not from [0, 0]
                contourX.push_back( relativePosition % mapWidth + x - 1 );
                contourY.push_back( relativePosition / mapWidth + y - 1 );

                pointId = 0;

                *mapValueX = CONTOUR;

                do {
                    uint32_t xMap = contourX[pointId];
                    uint32_t yMap = contourY[pointId++];

                    uint8_t * position = imageMap.data() + (yMap + 1 - y) * mapWidth + (xMap + 1 - x);

                    position = position - mapWidth - 1;
                    if( *(position) == EDGE ) {
                        contourX.push_back( xMap - 1 );
                        contourY.push_back( yMap - 1 );
                        *(position) = CONTOUR;
                    }

                    ++position;
                    if( *(position) == EDGE ) {
                        contourX.push_back( xMap );
                        contourY.push_back( yMap - 1 );
                        *(position) = CONTOUR;
                    }

                    ++position;
                    if( *(position) == EDGE ) {
                        contourX.push_back( xMap + 1 );
                        contourY.push_back( yMap - 1 );
                        *(position) = CONTOUR;
                    }

                    position = position + width; // (mapWidth - 2) is width
                    if( *(position) == EDGE ) {
                        contourX.push_back( xMap - 1 );
                        contourY.push_back( yMap );
                        *(position) = CONTOUR;
                    }

                    position = position + 2;
                    if( *(position) == EDGE ) {
                        contourX.push_back( xMap + 1 );
                        contourY.push_back( yMap );
                        *(position) = CONTOUR;
                    }

                    position = position + width; // (mapWidth - 2) is width
                    if( *(position) == EDGE ) {
                        contourX.push_back( xMap - 1 );
                        contourY.push_back( yMap + 1 );
                        *(position) = CONTOUR;
                    }

                    ++position;
                    if( *(position) == EDGE ) {
                        contourX.push_back( xMap );
                        contourY.push_back( yMap + 1 );
                        *(position) = CONTOUR;
                    }

                    ++position;
                    if( *(position) == EDGE ) {
                        contourX.push_back( xMap + 1 );
                        contourY.push_back( yMap + 1 );
                        *(position) = CONTOUR;
                    }
                } while( pointId != contourX.size() );
            }
        }

        // All blobs found. Now we need to sort them
        if( parameter.circularity.checkMaximum || parameter.circularity.checkMinimum ) {
            std::for_each( foundBlob.begin(), foundBlob.end(), []( BlobInfo & info ) { info._getCircularity(); } );

            foundBlob.remove_if( [&parameter]( const BlobInfo & info ) {
                return (parameter.circularity.checkMaximum && info.circularity() > parameter.circularity.maximum) ||
                       (parameter.circularity.checkMinimum && info.circularity() < parameter.circularity.minimum); } );
        }

        if( parameter.elongation.checkMaximum || parameter.elongation.checkMinimum ) {
            std::for_each( foundBlob.begin(), foundBlob.end(), []( BlobInfo & info ) { info._getElongation(); } );

            foundBlob.remove_if( [&parameter]( const BlobInfo & info ) {
                return (parameter.elongation.checkMaximum && info.elongation() > parameter.elongation.maximum) ||
                       (parameter.elongation.checkMinimum && info.elongation() < parameter.elongation.minimum); } );
        }

        if( parameter.height.checkMaximum || parameter.height.checkMinimum ) {
            std::for_each( foundBlob.begin(), foundBlob.end(), []( BlobInfo & info ) { info._getHeight(); } );

            foundBlob.remove_if( [&parameter]( const BlobInfo & info ) {
                return (parameter.height.checkMaximum && info.height() > parameter.height.maximum) ||
                       (parameter.height.checkMinimum && info.height() < parameter.height.minimum); } );
        }

        if( parameter.length.checkMaximum || parameter.length.checkMinimum ) {
            std::for_each( foundBlob.begin(), foundBlob.end(), []( BlobInfo & info ) { info._getLength(); } );

            foundBlob.remove_if( [&parameter]( const BlobInfo & info ) {
                return (parameter.length.checkMaximum && info.length() > parameter.length.maximum) ||
                       (parameter.length.checkMinimum && info.length() < parameter.length.minimum); } );
        }

        if( parameter.size.checkMaximum || parameter.size.checkMinimum ) {
            foundBlob.remove_if( [&parameter]( const BlobInfo & info ) {
                return (parameter.size.checkMaximum && info.size() > parameter.size.maximum) ||
                       (parameter.size.checkMinimum && info.size() < parameter.size.minimum); } );
        }

        if( parameter.width.checkMaximum || parameter.width.checkMinimum ) {
            std::for_each( foundBlob.begin(), foundBlob.end(), []( BlobInfo & info ) { info._getWidth(); } );

            foundBlob.remove_if( [&parameter]( const BlobInfo & info ) {
                return (parameter.width.checkMaximum && info.width() > parameter.width.maximum) ||
                       (parameter.width.checkMinimum && info.width() < parameter.width.minimum); } );
        }

        // prepare data for output
        std::vector <BlobInfo> blobTemp ( foundBlob.begin(), foundBlob.end() );
        std::swap( _blob, blobTemp );

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

    const std::vector < BlobInfo > & BlobDetection::operator()() const
    {
        return _blob;
    }

    std::vector < BlobInfo > & BlobDetection::operator()()
    {
        return _blob;
    }

    const BlobInfo & BlobDetection::getBestBlob( BlobCriterion criterion ) const
    {
        switch( criterion ) {
            case CRITERION_CIRCULARITY:
                return *(std::max_element( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 )
                    { return blob1.circularity() < blob2.circularity(); } ));
            case CRITERION_ELONGATION:
                return *(std::max_element( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 )
                    { return blob1.elongation() < blob2.elongation(); } ));
            case CRITERION_HEIGHT:
                return *(std::max_element( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 )
                    { return blob1.height() < blob2.height(); } ));
            case CRITERION_LENGTH:
                return *(std::max_element( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 )
                    { return blob1.length() < blob2.length(); } ));
            case CRITERION_SIZE:
                return *(std::max_element( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 )
                    { return blob1.size() < blob2.size(); } ));
            case CRITERION_WIDTH:
                return *(std::max_element( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 )
                    { return blob1.width() < blob2.width(); } ));
            default:
                throw imageException( "Bad criterion for blob finding" );
        }
    }

    void BlobDetection::sort( BlobCriterion criterion )
    {
        switch( criterion ) {
            case CRITERION_CIRCULARITY:
                std::sort( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 )
                    { return blob1.circularity() > blob2.circularity(); } );
                break;
            case CRITERION_ELONGATION:
                std::sort( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 )
                    { return blob1.elongation() > blob2.elongation(); } );
                break;
            case CRITERION_HEIGHT:
                std::sort( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 )
                    { return blob1.height() > blob2.height(); } );
                break;
            case CRITERION_LENGTH:
                std::sort( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 )
                    { return blob1.length() > blob2.length(); } );
                break;
            case CRITERION_SIZE:
                std::sort( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 )
                    { return blob1.size() > blob2.size(); } );
                break;
            case CRITERION_WIDTH:
                std::sort( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 )
                    { return blob1.width() > blob2.width(); } );
                break;
            default:
                throw imageException( "Bad criterion for blob sorting" );
        }
    }
}
