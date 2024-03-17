/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2024                                             *
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

#include "blob_detection.h"
#include "parameter_validation.h"
#include <cmath>
#include <list>
#include <numeric>
#include <queue>

namespace
{
    enum PixelState : uint8_t
    {
        EMPTY = 0u,
        NOT_IN_USE = 1u,
        FOUND = 2u,
        EDGE = 3u,
        CONTOUR = 4u
    };

    double getLengthFromCountour( const std::vector<uint32_t> & contourX, const std::vector<uint32_t> & contourY, PointBase2D<uint32_t> & startPoint,
                                  PointBase2D<uint32_t> & endPoint )
    {
        if ( contourX.size() <= 1 ) {
            return 0;
        }

        auto x = contourX.cbegin();
        auto y = contourY.cbegin();
        auto end = contourX.cend();

        int32_t maximumDistance = 0;

        for ( ; x != ( end - 1 ); ++x, ++y ) {
            auto xx = x + 1;
            auto yy = y + 1;

            for ( ; xx != end; ++xx, ++yy ) {
                const int32_t distance
                    = static_cast<int32_t>( *x - *xx ) * static_cast<int32_t>( *x - *xx ) + static_cast<int32_t>( *y - *yy ) * static_cast<int32_t>( *y - *yy );

                if ( maximumDistance < distance ) {
                    maximumDistance = distance;

                    startPoint.x = *x;
                    startPoint.y = *y;

                    endPoint.x = *xx;
                    endPoint.y = *xx;
                }
            }
        }

        return sqrt( static_cast<double>( maximumDistance ) );
    }
}

namespace Blob_Detection
{
    void BlobInfo::_getArea()
    {
        if ( !_area.found && !_contourX.empty() && !_contourY.empty() ) {
            _area.value.left = *( std::min_element( _contourX.begin(), _contourX.end() ) );
            _area.value.right = *( std::max_element( _contourX.begin(), _contourX.end() ) ) + 1; // note that we add 1
            _area.value.top = *( std::min_element( _contourY.begin(), _contourY.end() ) );
            _area.value.bottom = *( std::max_element( _contourY.begin(), _contourY.end() ) ) + 1; // note that we add 1

            _area.found = true;
        }
    }

    void BlobInfo::_getCenter()
    {
        if ( !_center.found && !_pointX.empty() && !_pointY.empty() ) {
            _center.value.x = static_cast<double>( std::accumulate( _pointX.begin(), _pointX.end(), 0 ) ) / static_cast<double>( size() );
            _center.value.y = static_cast<double>( std::accumulate( _pointY.begin(), _pointY.end(), 0 ) ) / static_cast<double>( size() );

            _center.found = true;
        }
    }

    void BlobInfo::_getCircularity()
    {
        if ( !_circularity.found && !_contourX.empty() ) {
            const double radius = sqrt( static_cast<double>( size() ) / pvmath::pi );
            _getCenter();

            double difference = 0;

            auto x = _contourX.begin();
            auto y = _contourY.begin();
            auto end = _contourX.end();

            for ( ; x != end; ++x, ++y ) {
                difference += fabs( sqrt( ( *x - _center.value.x ) * ( *x - _center.value.x ) + ( *y - _center.value.y ) * ( *y - _center.value.y ) ) - radius );
            }

            _circularity.value = 1 - difference / ( static_cast<double>( _contourX.size() ) * radius );

            _circularity.found = true;
        }
    }

    void BlobInfo::_getElongation()
    {
        if ( !_elongation.found && !_contourX.empty() && !_contourY.empty() ) {
            if ( _contourX.size() > 1 ) {
                PointBase2D<uint32_t> startPoint;
                PointBase2D<uint32_t> endPoint;
                const double length = getLengthFromCountour( _contourX, _contourY, startPoint, endPoint );
                const double angle = -atan2( static_cast<double>( endPoint.y - startPoint.y ), static_cast<double>( endPoint.x - startPoint.x ) );

                const double _cos = cos( angle );
                const double _sin = sin( angle );

                std::vector<double> contourYTemp( _contourY.begin(), _contourY.end() );

                auto xRotated = _contourX.begin();
                auto yRotated = contourYTemp.begin();
                auto endRotated = _contourX.end();

                for ( ; xRotated != endRotated; ++xRotated, ++yRotated ) {
                    ( *yRotated ) = ( *xRotated - startPoint.x ) * _sin + ( *yRotated - startPoint.y ) * _cos;
                }

                double height = *( std::max_element( contourYTemp.begin(), contourYTemp.end() ) ) - *( std::min_element( contourYTemp.begin(), contourYTemp.end() ) );

                if ( height < 1 ) {
                    height = 1;
                }

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
        if ( !_height.found && !_contourY.empty() ) {
            _height.value = *( std::max_element( _contourY.begin(), _contourY.end() ) ) - *( std::min_element( _contourY.begin(), _contourY.end() ) ) + 1;

            _height.found = true;
        }
    }

    void BlobInfo::_getLength()
    {
        if ( !_length.found && !_contourX.empty() && !_contourY.empty() ) {
            PointBase2D<uint32_t> startPoint;
            PointBase2D<uint32_t> endPoint;
            _length.value = getLengthFromCountour( _contourX, _contourY, startPoint, endPoint );

            _length.found = true;
        }
    }

    void BlobInfo::_getWidth()
    {
        if ( !_width.found && !_contourX.empty() ) {
            _width.value = *( std::max_element( _contourX.begin(), _contourX.end() ) ) - *( std::min_element( _contourX.begin(), _contourX.end() ) ) + 1;

            _width.found = true;
        }
    }

    const std::vector<BlobInfo> & BlobDetection::find( const penguinV::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                                       const BlobParameters & parameter, uint8_t threshold )
    {
        Image_Function::ValidateImageParameters( image, x, y, width, height );
        Image_Function::VerifyGrayScaleImage( image );

        parameter._verify();

        _blob.clear();

        // we make the area by 2 pixels bigger in each direction so we don't need to check borders of map
        penguinV::Image imageMap( width + 2, height + 2 );
        imageMap.fill( EMPTY );

        uint32_t rowSize = image.rowSize();

        const uint8_t * imageY = image.data() + y * rowSize + x;
        const uint8_t * imageYEnd = imageY + height * rowSize;

        const uint32_t mapWidth = width + 2;

        uint8_t * mapValueX = imageMap.data() + mapWidth + 1;

        for ( ; imageY != imageYEnd; imageY += rowSize, mapValueX += mapWidth ) {
            const uint8_t * imageX = imageY;
            const uint8_t * imageXEnd = imageX + width;

            uint8_t * mapValueY = mapValueX;

            for ( ; imageX != imageXEnd; ++imageX, ++mapValueY ) {
                if ( ( *imageX ) >= threshold ) {
                    *mapValueY = NOT_IN_USE;
                }
            }
        }

        // find all blobs
        std::list<BlobInfo> foundBlob;

        mapValueX = imageMap.data() + mapWidth;
        uint8_t * endMap = imageMap.data() + ( width + 2 ) * ( height + 1 );

        for ( ; mapValueX != endMap; ++mapValueX ) {
            if ( *mapValueX == NOT_IN_USE ) { // blob found!
                foundBlob.push_back( BlobInfo() );

                BlobInfo & newBlob = foundBlob.back();

                uint32_t relativePosition = static_cast<uint32_t>( mapValueX - imageMap.data() );

                std::vector<uint32_t> & pointX = newBlob._pointX;
                std::vector<uint32_t> & pointY = newBlob._pointY;

                // we put extra shift [-1, -1] to point position because our map starts from [1, 1]
                // not from [0, 0]
                pointX.push_back( relativePosition % mapWidth + x - 1 );
                pointY.push_back( relativePosition / mapWidth + y - 1 );

                std::vector<uint32_t> & edgeX = newBlob._edgeX;
                std::vector<uint32_t> & edgeY = newBlob._edgeY;

                *mapValueX = FOUND;

                size_t pointId = 0;

                do {
                    uint32_t xMap = pointX[pointId];
                    uint32_t yMap = pointY[pointId++];

                    uint8_t neighbourCount = 0;

                    uint8_t * position = imageMap.data() + ( yMap + 1 - y ) * mapWidth + ( xMap + 1 - x );

                    position = position - mapWidth - 1;
                    if ( *( position ) != EMPTY ) {
                        if ( *( position ) == NOT_IN_USE ) {
                            pointX.push_back( xMap - 1 );
                            pointY.push_back( yMap - 1 );
                            *( position ) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    ++position;
                    if ( *( position ) != EMPTY ) {
                        if ( *( position ) == NOT_IN_USE ) {
                            pointX.push_back( xMap );
                            pointY.push_back( yMap - 1 );
                            *( position ) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    ++position;
                    if ( *( position ) != EMPTY ) {
                        if ( *( position ) == NOT_IN_USE ) {
                            pointX.push_back( xMap + 1 );
                            pointY.push_back( yMap - 1 );
                            *( position ) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    position = position + width; // (mapWidth - 2) is width
                    if ( *( position ) != EMPTY ) {
                        if ( *( position ) == NOT_IN_USE ) {
                            pointX.push_back( xMap - 1 );
                            pointY.push_back( yMap );
                            *( position ) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    position = position + 2;
                    if ( *( position ) != EMPTY ) {
                        if ( *( position ) == NOT_IN_USE ) {
                            pointX.push_back( xMap + 1 );
                            pointY.push_back( yMap );
                            *( position ) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    position = position + width; // (mapWidth - 2) is width
                    if ( *( position ) != EMPTY ) {
                        if ( *( position ) == NOT_IN_USE ) {
                            pointX.push_back( xMap - 1 );
                            pointY.push_back( yMap + 1 );
                            *( position ) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    ++position;
                    if ( *( position ) != EMPTY ) {
                        if ( *( position ) == NOT_IN_USE ) {
                            pointX.push_back( xMap );
                            pointY.push_back( yMap + 1 );
                            *( position ) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    ++position;
                    if ( *( position ) != EMPTY ) {
                        if ( *( position ) == NOT_IN_USE ) {
                            pointX.push_back( xMap + 1 );
                            pointY.push_back( yMap + 1 );
                            *( position ) = FOUND;
                        }
                        ++neighbourCount;
                    }

                    if ( neighbourCount != 8 ) {
                        edgeX.push_back( xMap );
                        edgeY.push_back( yMap );
                        *( position - 1 - mapWidth ) = EDGE;
                    }
                } while ( pointId != pointX.size() );

                // Now we can extract outer edge points or so called contour points
                std::vector<uint32_t> & contourX = newBlob._contourX;
                std::vector<uint32_t> & contourY = newBlob._contourY;

                // we put extra shift [-1, -1] to point position because our map starts from [1, 1]
                // not from [0, 0]
                contourX.push_back( relativePosition % mapWidth + x - 1 );
                contourY.push_back( relativePosition / mapWidth + y - 1 );

                pointId = 0;

                *mapValueX = CONTOUR;

                do {
                    uint32_t xMap = contourX[pointId];
                    uint32_t yMap = contourY[pointId++];

                    uint8_t * position = imageMap.data() + ( yMap + 1 - y ) * mapWidth + ( xMap + 1 - x );

                    position = position - mapWidth - 1;
                    if ( *( position ) == EDGE ) {
                        contourX.push_back( xMap - 1 );
                        contourY.push_back( yMap - 1 );
                        *( position ) = CONTOUR;
                    }

                    ++position;
                    if ( *( position ) == EDGE ) {
                        contourX.push_back( xMap );
                        contourY.push_back( yMap - 1 );
                        *( position ) = CONTOUR;
                    }

                    ++position;
                    if ( *( position ) == EDGE ) {
                        contourX.push_back( xMap + 1 );
                        contourY.push_back( yMap - 1 );
                        *( position ) = CONTOUR;
                    }

                    position = position + width; // (mapWidth - 2) is width
                    if ( *( position ) == EDGE ) {
                        contourX.push_back( xMap - 1 );
                        contourY.push_back( yMap );
                        *( position ) = CONTOUR;
                    }

                    position = position + 2;
                    if ( *( position ) == EDGE ) {
                        contourX.push_back( xMap + 1 );
                        contourY.push_back( yMap );
                        *( position ) = CONTOUR;
                    }

                    position = position + width; // (mapWidth - 2) is width
                    if ( *( position ) == EDGE ) {
                        contourX.push_back( xMap - 1 );
                        contourY.push_back( yMap + 1 );
                        *( position ) = CONTOUR;
                    }

                    ++position;
                    if ( *( position ) == EDGE ) {
                        contourX.push_back( xMap );
                        contourY.push_back( yMap + 1 );
                        *( position ) = CONTOUR;
                    }

                    ++position;
                    if ( *( position ) == EDGE ) {
                        contourX.push_back( xMap + 1 );
                        contourY.push_back( yMap + 1 );
                        *( position ) = CONTOUR;
                    }
                } while ( pointId != contourX.size() );
            }
        }

        // All blobs found. Now we need to sort them
        if ( parameter.circularity.checkMaximum || parameter.circularity.checkMinimum ) {
            std::for_each( foundBlob.begin(), foundBlob.end(), []( BlobInfo & info ) { info._getCircularity(); } );

            foundBlob.remove_if( [&parameter]( const BlobInfo & info ) {
                return ( parameter.circularity.checkMaximum && info.circularity() > parameter.circularity.maximum )
                       || ( parameter.circularity.checkMinimum && info.circularity() < parameter.circularity.minimum );
            } );
        }

        if ( parameter.elongation.checkMaximum || parameter.elongation.checkMinimum ) {
            std::for_each( foundBlob.begin(), foundBlob.end(), []( BlobInfo & info ) { info._getElongation(); } );

            foundBlob.remove_if( [&parameter]( const BlobInfo & info ) {
                return ( parameter.elongation.checkMaximum && info.elongation() > parameter.elongation.maximum )
                       || ( parameter.elongation.checkMinimum && info.elongation() < parameter.elongation.minimum );
            } );
        }

        if ( parameter.height.checkMaximum || parameter.height.checkMinimum ) {
            std::for_each( foundBlob.begin(), foundBlob.end(), []( BlobInfo & info ) { info._getHeight(); } );

            foundBlob.remove_if( [&parameter]( const BlobInfo & info ) {
                return ( parameter.height.checkMaximum && info.height() > parameter.height.maximum )
                       || ( parameter.height.checkMinimum && info.height() < parameter.height.minimum );
            } );
        }

        if ( parameter.length.checkMaximum || parameter.length.checkMinimum ) {
            std::for_each( foundBlob.begin(), foundBlob.end(), []( BlobInfo & info ) { info._getLength(); } );

            foundBlob.remove_if( [&parameter]( const BlobInfo & info ) {
                return ( parameter.length.checkMaximum && info.length() > parameter.length.maximum )
                       || ( parameter.length.checkMinimum && info.length() < parameter.length.minimum );
            } );
        }

        if ( parameter.size.checkMaximum || parameter.size.checkMinimum ) {
            foundBlob.remove_if( [&parameter]( const BlobInfo & info ) {
                return ( parameter.size.checkMaximum && info.size() > parameter.size.maximum ) || ( parameter.size.checkMinimum && info.size() < parameter.size.minimum );
            } );
        }

        if ( parameter.width.checkMaximum || parameter.width.checkMinimum ) {
            std::for_each( foundBlob.begin(), foundBlob.end(), []( BlobInfo & info ) { info._getWidth(); } );

            foundBlob.remove_if( [&parameter]( const BlobInfo & info ) {
                return ( parameter.width.checkMaximum && info.width() > parameter.width.maximum )
                       || ( parameter.width.checkMinimum && info.width() < parameter.width.minimum );
            } );
        }

        // prepare data for output
        std::vector<BlobInfo> blobTemp( foundBlob.begin(), foundBlob.end() );
        std::swap( _blob, blobTemp );

        return get();
    }

    const BlobInfo & BlobDetection::getBestBlob( BlobCriterion criterion ) const
    {
        switch ( criterion ) {
        case BlobCriterion::BY_CIRCULARITY:
            return *( std::max_element( _blob.begin(), _blob.end(),
                                        []( const BlobInfo & blob1, const BlobInfo & blob2 ) { return blob1.circularity() < blob2.circularity(); } ) );
        case BlobCriterion::BY_ELONGATION:
            return *( std::max_element( _blob.begin(), _blob.end(),
                                        []( const BlobInfo & blob1, const BlobInfo & blob2 ) { return blob1.elongation() < blob2.elongation(); } ) );
        case BlobCriterion::BY_HEIGHT:
            return *( std::max_element( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 ) { return blob1.height() < blob2.height(); } ) );
        case BlobCriterion::BY_LENGTH:
            return *( std::max_element( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 ) { return blob1.length() < blob2.length(); } ) );
        case BlobCriterion::BY_SIZE:
            return *( std::max_element( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 ) { return blob1.size() < blob2.size(); } ) );
        case BlobCriterion::BY_WIDTH:
            return *( std::max_element( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 ) { return blob1.width() < blob2.width(); } ) );
        default:
            throw penguinVException( "No criterion for blob sorting was set. Did you add a new criterion?" );
        }
    }

    void BlobDetection::sort( BlobCriterion criterion )
    {
        switch ( criterion ) {
        case BlobCriterion::BY_CIRCULARITY:
            std::sort( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 ) { return blob1.circularity() > blob2.circularity(); } );
            break;
        case BlobCriterion::BY_ELONGATION:
            std::sort( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 ) { return blob1.elongation() > blob2.elongation(); } );
            break;
        case BlobCriterion::BY_HEIGHT:
            std::sort( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 ) { return blob1.height() > blob2.height(); } );
            break;
        case BlobCriterion::BY_LENGTH:
            std::sort( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 ) { return blob1.length() > blob2.length(); } );
            break;
        case BlobCriterion::BY_SIZE:
            std::sort( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 ) { return blob1.size() > blob2.size(); } );
            break;
        case BlobCriterion::BY_WIDTH:
            std::sort( _blob.begin(), _blob.end(), []( const BlobInfo & blob1, const BlobInfo & blob2 ) { return blob1.width() > blob2.width(); } );
            break;
        default:
            throw penguinVException( "No criterion for blob sorting was set. Did you add a new criterion?" );
        }
    }
}
