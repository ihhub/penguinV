#include "edge_detection.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include "parameter_validation.h"

namespace
{
    void leaveFirstElement( std::vector< double > & data )
    {
        if ( data.size() > 1u )
            data.resize( 1u );
    }

    void leaveLastElement( std::vector< double > & data )
    {
        if ( data.size() > 1u ) {
            std::vector< double > temp;
            temp.push_back( data.back() );
            std::swap( data, temp );
        }
    }

    void createPositiveXEdge( const std::vector< double > & data, std::vector < Point2d > & point, double x, double y )
    {
        for ( std::vector < double >::const_iterator dataX = data.cbegin(); dataX != data.cend(); ++dataX )
            point.push_back( Point2d( (*dataX) + x, y ) );
    }

    void createNegativeXEdge( const std::vector< double > & data, std::vector < Point2d > & point, double x, double y )
    {
        for ( std::vector < double >::const_iterator dataX = data.cbegin(); dataX != data.cend(); ++dataX )
            point.push_back( Point2d( x - (*dataX), y ) );
    }

    void createPositiveYEdge( const std::vector< double > & data, std::vector < Point2d > & point, double x, double y )
    {
        for ( std::vector < double >::const_iterator dataY = data.cbegin(); dataY != data.cend(); ++dataY )
            point.push_back( Point2d( x, y + (*dataY) ) );
    }

    void createNegativeYEdge( const std::vector< double > & data, std::vector < Point2d > & point, double x, double y )
    {
        for ( std::vector < double >::const_iterator dataY = data.cbegin(); dataY != data.cend(); ++dataY )
            point.push_back( Point2d( x, y - (*dataY) ) );
    }

    bool findPoint( const std::vector< int > & data, const std::vector< int > & second, std::vector< double > & edge,
                    uint32_t leftSideOffset, uint32_t rightSideOffset, int minimumContrast, bool checkContrast,
                    uint32_t leftSideContrastCheck, uint32_t rightSideContrastCheck, uint32_t position, uint32_t size )
    {
        const int maxIntensity = *(std::max_element( data.begin() + leftSideOffset, data.begin() + rightSideOffset + 1 ));
        const int minIntensity = *(std::min_element( data.begin() + leftSideOffset, data.begin() + rightSideOffset + 1 ));

        if ( maxIntensity - minIntensity < minimumContrast )
            return false;

        if ( checkContrast && leftSideContrastCheck <= position && ( rightSideContrastCheck + position ) < size ) {
            const uint32_t blackContrastEnd   = position;
            const uint32_t whiteContrastStart = position + 1;

            const uint32_t blackContrastStart = position - leftSideContrastCheck;
            const uint32_t whiteContrastEnd   = position + rightSideContrastCheck;

            double sumBlack = std::accumulate( data.begin() + blackContrastStart, data.begin() + blackContrastEnd + 1, 0 );
            double sumWhite = std::accumulate( data.begin() + whiteContrastStart, data.begin() + whiteContrastEnd + 1, 0 );

            sumBlack /= (blackContrastEnd - blackContrastStart + 1);
            sumWhite /= (whiteContrastEnd - whiteContrastStart + 1);

            if ( sumWhite - sumBlack >= minimumContrast )
                checkContrast = false;
        }

        if ( !checkContrast ) {
            if ( second[position] != second[position + 1] )
                edge.push_back( position + static_cast<double>(second[position]) / (second[position] - second[position + 1u]) );
            else
                edge.push_back( position + 0.5 );
            return true;
        }

        return false;
    }
}

EdgeParameter::EdgeParameter( directionType _direction, gradientType _gradient, edgeType _edge, uint32_t _groupFactor, uint32_t _skipFactor,
                              uint32_t _contrastCheckLeftSideOffset, uint32_t _contrastCheckRightSideOffset, uint8_t _minimumContrast )
    : direction  ( _direction )
    , gradient   ( _gradient )
    , edge       ( _edge )
    , groupFactor( _groupFactor )
    , skipFactor ( _skipFactor )
    , contrastCheckLeftSideOffset( _contrastCheckLeftSideOffset )
    , contrastCheckRightSideOffset( _contrastCheckRightSideOffset )
    , minimumContrast( _minimumContrast )
{
}

void EdgeParameter::verify() const
{
    if ( groupFactor == 0u )
        throw imageException( "Grouping factor for edge detection cannot be 0" );
    if ( skipFactor == 0u )
        throw imageException( "Skip factor for edge detection cannot be 0" );
    if ( minimumContrast == 0u )
        throw imageException( "Minimum contrast for edge detection cannot be 0" );
}

void EdgeDetection::find( const PenguinV_Image::Image & image, const EdgeParameter & edgeParameter )
{
    find( image, 0, 0, image.width(), image.height(), edgeParameter );
}

void EdgeDetection::find( const PenguinV_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const EdgeParameter & edgeParameter )
{
    Image_Function::VerifyGrayScaleImage( image );
    Image_Function::ParameterValidation( image, x, y, width, height );
    edgeParameter.verify();

    const bool horizontalEdgeDetection = (edgeParameter.direction == EdgeParameter::LEFT_TO_RIGHT || edgeParameter.direction == EdgeParameter::RIGHT_TO_LEFT);

    if ( (horizontalEdgeDetection && (width  < 4u)) || (!horizontalEdgeDetection && (height < 4u)) )
        return;

    const uint32_t rowSize = image.rowSize();

    if ( horizontalEdgeDetection ) {
        std::vector < int > data            ( width );
        std::vector < int > firstDerivative ( width, 0 );
        std::vector < int > secondDerivative( width, 0 );

        const uint8_t * imageY = image.data() + y * rowSize + x;
        for ( uint32_t rowId = 0u; rowId < height - edgeParameter.groupFactor + 1u; rowId += edgeParameter.skipFactor, imageY += (rowSize * edgeParameter.skipFactor) ) {
            const uint8_t * imageX    = imageY;
            const uint8_t * imageXEnd = imageX + width;
            int * dataX = data.data();
            for ( ; imageX != imageXEnd; ++imageX, ++dataX )
                *dataX = *imageX;

            for ( uint32_t groupId = 1u; groupId < edgeParameter.groupFactor; ++groupId ) {
                imageX    = imageY + groupId * rowSize;
                imageXEnd = imageX + width;
                dataX = data.data();

                for ( ; imageX != imageXEnd; ++imageX, ++dataX )
                    *dataX += *imageX;
            }

            std::vector< double > edgePositive;
            std::vector< double > edgeNegative;
            findEdgePoints( edgePositive, edgeNegative, data, firstDerivative, secondDerivative, edgeParameter, (edgeParameter.direction == EdgeParameter::LEFT_TO_RIGHT) );

            if ( edgeParameter.direction == EdgeParameter::LEFT_TO_RIGHT ) {
                if ( edgeParameter.gradient == EdgeParameter::POSITIVE || edgeParameter.gradient == EdgeParameter::ANY )
                    createPositiveXEdge( edgePositive, positiveEdgePoint, x, y + rowId + (edgeParameter.groupFactor - 1) / 2.0 );

                if ( edgeParameter.gradient == EdgeParameter::NEGATIVE || edgeParameter.gradient == EdgeParameter::ANY )
                    createNegativeXEdge( edgeNegative, negativeEdgePoint, x + width - 1, y + rowId + (edgeParameter.groupFactor - 1) / 2.0 );
            }
            else {
                if ( edgeParameter.gradient == EdgeParameter::POSITIVE || edgeParameter.gradient == EdgeParameter::ANY )
                    createNegativeXEdge( edgeNegative, positiveEdgePoint, x + width - 1, y + rowId + (edgeParameter.groupFactor - 1) / 2.0 );

                if ( edgeParameter.gradient == EdgeParameter::NEGATIVE || edgeParameter.gradient == EdgeParameter::ANY )
                    createPositiveXEdge( edgePositive, negativeEdgePoint, x, y + rowId + (edgeParameter.groupFactor - 1) / 2.0 );
            }
        }
    }
    else {
        std::vector < int > data            ( height );
        std::vector < int > firstDerivative ( height, 0 );
        std::vector < int > secondDerivative( height, 0 );

        const uint8_t * imageX = image.data() + y * rowSize + x;
        for ( uint32_t rowId = 0u; rowId < width - edgeParameter.groupFactor + 1u; rowId += edgeParameter.skipFactor, imageX += edgeParameter.skipFactor ) {
            const uint8_t * imageY    = imageX;
            const uint8_t * imageYEnd = imageY + height * rowSize;
            int * dataX = data.data();
            for ( ; imageY != imageYEnd; imageY += rowSize, ++dataX )
                *dataX = *imageY;

            for ( uint32_t groupId = 1u; groupId < edgeParameter.groupFactor; ++groupId ) {
                imageY    = imageX + groupId;
                imageYEnd = imageY + height * rowSize;
                dataX = data.data();

                for ( ; imageY != imageYEnd; imageY += rowSize, ++dataX )
                    *dataX = *imageY;
            }

            std::vector< double > edgePositive;
            std::vector< double > edgeNegative;
            findEdgePoints( edgePositive, edgeNegative, data, firstDerivative, secondDerivative, edgeParameter, (edgeParameter.direction == EdgeParameter::TOP_TO_BOTTOM) );

            if ( edgeParameter.direction == EdgeParameter::TOP_TO_BOTTOM ) {
                if ( edgeParameter.gradient == EdgeParameter::POSITIVE || edgeParameter.gradient == EdgeParameter::ANY )
                    createPositiveYEdge( edgePositive, positiveEdgePoint, x + rowId + (edgeParameter.groupFactor - 1) / 2.0, y );

                if ( edgeParameter.gradient == EdgeParameter::NEGATIVE || edgeParameter.gradient == EdgeParameter::ANY )
                    createNegativeYEdge( edgeNegative, negativeEdgePoint, x + rowId + (edgeParameter.groupFactor - 1) / 2.0, y + height - 1 );
            }
            else {
                if ( edgeParameter.gradient == EdgeParameter::POSITIVE || edgeParameter.gradient == EdgeParameter::ANY )
                    createNegativeYEdge( edgeNegative, positiveEdgePoint, x + rowId + (edgeParameter.groupFactor - 1) / 2.0, y + height - 1 );

                if ( edgeParameter.gradient == EdgeParameter::NEGATIVE || edgeParameter.gradient == EdgeParameter::ANY )
                    createPositiveYEdge( edgePositive, negativeEdgePoint, x + rowId + (edgeParameter.groupFactor - 1) / 2.0, y );
            }
        }
    }
}

void EdgeDetection::findEdgePoints( std::vector < double > & positive, std::vector < double > & negative, std::vector < int > & data,
                                    std::vector < int > & first, std::vector < int > & second, const EdgeParameter & edgeParameter, bool forwardDirection )
{
    getDerivatives( data, first, second );
    getEdgePoints( positive, data, first, second, edgeParameter );
    removeSimilarPoints( positive );

    std::reverse( data.begin(), data.end() );
    getDerivatives( data, first, second );
    getEdgePoints( negative, data, first, second, edgeParameter );
    removeSimilarPoints( negative );

    if ( (forwardDirection && edgeParameter.edge == EdgeParameter::FIRST) || (!forwardDirection && edgeParameter.edge == EdgeParameter::LAST) ) {
        leaveFirstElement( positive );
        leaveLastElement( negative );
    }
    else if ( (forwardDirection && edgeParameter.edge == EdgeParameter::LAST) || (!forwardDirection && edgeParameter.edge == EdgeParameter::FIRST) ) {
        leaveLastElement( positive );
        leaveFirstElement( negative );
    }
}

const std::vector < Point2d > & EdgeDetection::positiveEdge() const
{
    return positiveEdgePoint;
}

const std::vector < Point2d > & EdgeDetection::negativeEdge() const
{
    return negativeEdgePoint;
}

void EdgeDetection::getDerivatives( const std::vector < int > & image, std::vector < int > & first, std::vector < int > & second ) const
{
    // input array range is [0; n)
    // first deriviative range is [0; n - 1)
    // second deriviative range is [1; n - 1)
    std::transform( image.begin() + 1u, image.end(), image.begin(), first.begin(), std::minus<int>() );
    std::transform( first.begin() + 1u, first.end(), first.begin(), second.begin() + 1u, std::minus<int>() );
}

void EdgeDetection::removeSimilarPoints( std::vector < double > & edge ) const
{
    for ( size_t i = 1u; i < edge.size(); ) {
        if ( (edge[i] - edge[i - 1]) < 1.0 )
            edge.erase( edge.begin() + i );
        else
            ++i;
    }
}

void EdgeDetection::getEdgePoints( std::vector < double > & edge, const std::vector < int > & data, const std::vector < int > & first, const std::vector < int > & second,
                                   const EdgeParameter & edgeParameter ) const
{
    const uint32_t dataSize = static_cast<uint32_t>( data.size() ); // we know that initial size is uint32_t
    const int minimumContrast = static_cast<int>(edgeParameter.minimumContrast * edgeParameter.groupFactor);

    const bool checkContrast = (edgeParameter.contrastCheckLeftSideOffset > 0u) && (edgeParameter.contrastCheckRightSideOffset > 0u);

    for ( uint32_t i = 1u; i < dataSize - 2u; i++ ) {
        if ( second[i] < 0 || second[i + 1] > 0 )
            continue;

        const int maxGradient = (first[i] > first[i + 1]) ? first[i] : first[i + 1];

        if ( maxGradient <= 0 )
            continue;

        const int minGradientValue = (maxGradient < 3) ? 1 : maxGradient / 3;
        const int halfGradient     = (maxGradient < 2) ? 1 : maxGradient / 2;

        // left side
        bool normalGradientleftEdgeFound  = false;
        bool minimumGradientLeftEdgeFound = false;

        uint32_t normalLeftSide  = i;
        uint32_t minimumLeftSide = i;

        for ( uint32_t j = i; ; --j ) {
            if ( !normalGradientleftEdgeFound && first[j] < halfGradient ) {
                normalLeftSide = j;
                normalGradientleftEdgeFound = true;
            }

            if ( first[j] < minGradientValue ) {
                minimumLeftSide = j;
                minimumGradientLeftEdgeFound = true;
                break;
            }

            if ( first[j] > maxGradient )
                break;
            
            if (j == 0u) // this is to avoid out of bounds situation
                break;
        }

        if ( !normalGradientleftEdgeFound && !minimumGradientLeftEdgeFound )
            continue;

        // right side
        bool normalGradientRightEdgeFound  = false;
        bool minimumGradientRightEdgeFound = false;

        uint32_t normalRightSide  = i + 1u;
        uint32_t minimumRightSide = i + 1u;

        for ( uint32_t j = i + 1u; j < dataSize - 1u; ++j ) {
            if ( !normalGradientRightEdgeFound && first[j] < halfGradient ) {
                normalRightSide = j;
                normalGradientRightEdgeFound = true;
            }

            if ( first[j] < minGradientValue ) {
                minimumRightSide = j;
                minimumGradientRightEdgeFound = true;
                break;
            }

            if ( first[j] > maxGradient )
                break;
        }

        bool edgeFound = false;

        if ( normalGradientleftEdgeFound && normalGradientRightEdgeFound ) {
            edgeFound = findPoint( data, second, edge, normalLeftSide, normalRightSide, minimumContrast, checkContrast,
                                   edgeParameter.contrastCheckLeftSideOffset, edgeParameter.contrastCheckRightSideOffset,
                                   i, dataSize );
        }

        if ( !edgeFound && minimumGradientLeftEdgeFound && minimumGradientRightEdgeFound ) {
            findPoint( data, second, edge, minimumLeftSide, minimumRightSide, minimumContrast, checkContrast,
                       edgeParameter.contrastCheckLeftSideOffset, edgeParameter.contrastCheckRightSideOffset,
                       i, dataSize );
        }
    }
}
