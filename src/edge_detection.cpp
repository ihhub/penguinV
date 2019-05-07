#include "edge_detection.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <cmath>
#include "parameter_validation.h"

namespace
{
    template<typename _Type>
    void leaveFirstElement( std::vector< _Type > & data )
    {
        if ( data.size() > 1u )
            data.resize( 1u );
    }

    template<typename _Type>
    void leaveLastElement( std::vector< _Type > & data )
    {
        if ( data.size() > 1u ) {
            std::vector< _Type > temp;
            temp.push_back( data.back() );
            std::swap( data, temp );
        }
    }

    template<typename _Type>
    void createPositiveXEdge( const std::vector< _Type > & data, std::vector < PointBase2D<_Type> > & point, _Type x, _Type y )
    {
        for ( typename std::vector < _Type >::const_iterator dataX = data.cbegin(); dataX != data.cend(); ++dataX )
            point.push_back( PointBase2D<_Type>( (*dataX) + x, y ) );
    }

    template<typename _Type>
    void createNegativeXEdge( const std::vector< _Type > & data, std::vector < PointBase2D<_Type> > & point, _Type x, _Type y )
    {
        for ( typename std::vector <_Type>::const_iterator dataX = data.cbegin(); dataX != data.cend(); ++dataX )
            point.push_back( PointBase2D<_Type>( x - (*dataX), y ) );
    }

    template<typename _Type>
    void createPositiveYEdge( const std::vector< _Type > & data, std::vector < PointBase2D<_Type> > & point, _Type x, _Type y )
    {
        for ( typename std::vector < _Type >::const_iterator dataY = data.cbegin(); dataY != data.cend(); ++dataY )
            point.push_back( PointBase2D<_Type>( x, y + (*dataY) ) );
    }

    template<typename _Type>
    void createNegativeYEdge( const std::vector< _Type > & data, std::vector < PointBase2D<_Type> > & point, _Type x, _Type y )
    {
        for ( typename std::vector < _Type >::const_iterator dataY = data.cbegin(); dataY != data.cend(); ++dataY )
            point.push_back( PointBase2D<_Type>( x, y - (*dataY) ) );
    }

    template<typename _Type>
    bool findPoint( const std::vector< int > & data, const std::vector< int > & second, std::vector< _Type > & edge,
                    uint32_t leftSideOffset, uint32_t rightSideOffset, int minimumContrast, bool checkContrast,
                    uint32_t leftSideContrastCheck, uint32_t rightSideContrastCheck, uint32_t position, uint32_t size )
    {
        const int maxIntensity = *(std::max_element( data.begin() + leftSideOffset, data.begin() + rightSideOffset + 1 ));
        const int minIntensity = *(std::min_element( data.begin() + leftSideOffset, data.begin() + rightSideOffset + 1 ));

        if ( maxIntensity - minIntensity < minimumContrast )
            return false;

        if ( checkContrast && leftSideContrastCheck <= position && (rightSideContrastCheck + position) < size ) {
            const uint32_t blackContrastEnd   = position;
            const uint32_t whiteContrastStart = position + 1;

            const uint32_t blackContrastStart = position - leftSideContrastCheck;
            const uint32_t whiteContrastEnd   = position + rightSideContrastCheck;

            _Type sumBlack = std::accumulate( data.begin() + blackContrastStart, data.begin() + blackContrastEnd + 1, 0.0f );
            _Type sumWhite = std::accumulate( data.begin() + whiteContrastStart, data.begin() + whiteContrastEnd + 1, 0.0f );

            sumBlack /= (blackContrastEnd - blackContrastStart + 1);
            sumWhite /= (whiteContrastEnd - whiteContrastStart + 1);

            if ( sumWhite - sumBlack >= minimumContrast )
                checkContrast = false;
        }

        if ( !checkContrast ) {
            if ( second[position] != second[position + 1] )
                edge.push_back( position + static_cast<_Type>(second[position]) / (second[position] - second[position + 1u]) );
            else
                edge.push_back( position + 0.5f );
            return true;
        }

        return false;
    }

    template<typename _Type>
    void getEdgePoints( std::vector < _Type > & edge, const std::vector < int > & data, const std::vector < int > & first, const std::vector < int > & second, const EdgeParameter & edgeParameter )
    {
        const uint32_t dataSize = static_cast<uint32_t>(data.size()); // we know that initial size is uint32_t
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

                if ( j == 0u ) // this is to avoid out of bounds situation
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

    template<typename _Type>
    void removeSimilarPoints( std::vector < _Type > & edge )
    {
        for ( size_t i = 1u; i < edge.size(); ) {
            if ( (edge[i] - edge[i - 1u]) < 1.0 )
                edge.erase( edge.begin() + static_cast<typename std::vector< _Type >::difference_type>(i) ); // it's safe to do
            else
                ++i;
        }
    }

    void getDerivatives( const std::vector < int > & image, std::vector < int > & first, std::vector < int > & second )
    {
        // input array range is [0; n)
        // first deriviative range is [0; n - 1)
        // second deriviative range is [1; n - 1)
        std::transform( image.begin() + 1u, image.end(), image.begin(), first.begin(), std::minus<int>() );
        std::transform( first.begin() + 1u, first.end(), first.begin(), second.begin() + 1u, std::minus<int>() );
    }

    template<typename _Type>
    void findEdgePoints( std::vector < _Type > & positive, std::vector < _Type > & negative, std::vector < int > & data,
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

    template <typename _Type>
    void findEdgePoints( const PenguinV_Image::Image & image, uint32_t x, uint32_t y, uint32_t width, uint32_t height, const EdgeParameter & edgeParameter,
                         std::vector < PointBase2D<_Type> > & positiveEdgePoint, std::vector < PointBase2D<_Type> > & negativeEdgePoint )
    {
        Image_Function::VerifyGrayScaleImage( image );
        Image_Function::ParameterValidation( image, x, y, width, height );
        edgeParameter.verify();

        const bool horizontalEdgeDetectionBase = (edgeParameter.direction == EdgeParameter::LEFT_TO_RIGHT || edgeParameter.direction == EdgeParameter::RIGHT_TO_LEFT);

        if ( (horizontalEdgeDetectionBase && (width  < 4u)) || (!horizontalEdgeDetectionBase && (height < 4u)) )
            return;

        const uint32_t rowSize = image.rowSize();

        if ( horizontalEdgeDetectionBase ) {
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

                std::vector< _Type > edgePositive;
                std::vector< _Type > edgeNegative;
                findEdgePoints( edgePositive, edgeNegative, data, firstDerivative, secondDerivative, edgeParameter, (edgeParameter.direction == EdgeParameter::LEFT_TO_RIGHT) );

                if ( edgeParameter.direction == EdgeParameter::LEFT_TO_RIGHT ) {
                    if ( edgeParameter.gradient == EdgeParameter::POSITIVE || edgeParameter.gradient == EdgeParameter::ANY )
                        createPositiveXEdge( edgePositive, positiveEdgePoint, static_cast<_Type>( x ),
                                             static_cast<_Type>(y + rowId + (edgeParameter.groupFactor - 1) / 2.0f ) );

                    if ( edgeParameter.gradient == EdgeParameter::NEGATIVE || edgeParameter.gradient == EdgeParameter::ANY )
                        createNegativeXEdge( edgeNegative, negativeEdgePoint, static_cast<_Type>( x + width - 1 ),
                                             static_cast<_Type>( y + rowId + (edgeParameter.groupFactor - 1) / 2.0f ) );
                }
                else {
                    if ( edgeParameter.gradient == EdgeParameter::POSITIVE || edgeParameter.gradient == EdgeParameter::ANY )
                        createNegativeXEdge( edgeNegative, positiveEdgePoint, static_cast<_Type>( x + width - 1 ),
                                             static_cast<_Type>( y + rowId + (edgeParameter.groupFactor - 1) / 2.0f ) );

                    if ( edgeParameter.gradient == EdgeParameter::NEGATIVE || edgeParameter.gradient == EdgeParameter::ANY )
                        createPositiveXEdge( edgePositive, negativeEdgePoint, static_cast<_Type>( x ),
                                             static_cast<_Type>( y + rowId + (edgeParameter.groupFactor - 1) / 2.0f ) );
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
                        *dataX += *imageY;
                }

                std::vector< _Type > edgePositive;
                std::vector< _Type > edgeNegative;
                findEdgePoints( edgePositive, edgeNegative, data, firstDerivative, secondDerivative, edgeParameter, (edgeParameter.direction == EdgeParameter::TOP_TO_BOTTOM) );

                if ( edgeParameter.direction == EdgeParameter::TOP_TO_BOTTOM ) {
                    if ( edgeParameter.gradient == EdgeParameter::POSITIVE || edgeParameter.gradient == EdgeParameter::ANY )
                        createPositiveYEdge( edgePositive, positiveEdgePoint, static_cast<_Type>( x + rowId + (edgeParameter.groupFactor - 1) / 2.0f ),
                                             static_cast<_Type>( y ) );

                    if ( edgeParameter.gradient == EdgeParameter::NEGATIVE || edgeParameter.gradient == EdgeParameter::ANY )
                        createNegativeYEdge( edgeNegative, negativeEdgePoint, static_cast<_Type>( x + rowId + (edgeParameter.groupFactor - 1) / 2.0f ),
                                             static_cast<_Type>( y + height - 1 ) );
                }
                else {
                    if ( edgeParameter.gradient == EdgeParameter::POSITIVE || edgeParameter.gradient == EdgeParameter::ANY )
                        createNegativeYEdge( edgeNegative, positiveEdgePoint, static_cast<_Type>( x + rowId + (edgeParameter.groupFactor - 1) / 2.0f ),
                                             static_cast<_Type>( y + height - 1 ) );

                    if ( edgeParameter.gradient == EdgeParameter::NEGATIVE || edgeParameter.gradient == EdgeParameter::ANY )
                        createPositiveYEdge( edgePositive, negativeEdgePoint, static_cast<_Type>( x + rowId + (edgeParameter.groupFactor - 1) / 2.0f ),
                                             static_cast<_Type>( y ) );
                }
            }
        }
    }
    
    inline uint32_t factorial(const uint32_t n)
    {
        if (n == 0u || n == 1u)
            return 1u;
        else
            return n * factorial(n - 1u);
    }

    // Gaussian function is actually the distribution function of the normal distribution
    // so we can use Pascal's triangle to get kernel values and normalize them
    template<typename _Type>
    void getPascalTriangleLine(std::vector< _Type > & line, const uint32_t & line_index)
    {
        if (!line.empty())
            line.clear();

        line.resize(line_index + 1);
        uint32_t sum = static_cast<uint32_t>(pow(2u, line_index));

        for (uint32_t i = 0u; i < line_index + 1; ++i)
        {
            line[i] = factorial(line_index) / static_cast<_Type>((factorial(i) * factorial(line_index - i))) / sum;
        }
    }

    const int32_t circle(const int32_t row_col, const int32_t value)
    {
        if (value < 0u)
            return value + row_col;
        if (value >= row_col)
            return value - row_col;
        return value;
    }

    template< typename _Type >
    void circularConvolution(const PenguinV_Image::Image & src, PenguinV_Image::Image & dst, const std::vector<_Type> & kernel)
    {
        int32_t height = static_cast<int32_t>(src.height());
        int32_t width  = static_cast<int32_t>(src.width());

        // Temp image
        PenguinV_Image::Image tmp;
        tmp.resize(width, height);

        int32_t deviation_start, deviation_end;

        // if 'x' or 'y' go beyond the border, we take an element on the oppisite side of the image 
        // by adding either width() or height(), thus we have 'x_altered' and 'y_altered' 
        int32_t x_altered, y_altered;

        if (kernel.size() % 2)
        {
            deviation_start = kernel.size() / 2 * (-1);
            deviation_end = kernel.size() / 2;
        }
        else
        {
            deviation_start = kernel.size() / 2 * (-1);
            deviation_end = kernel.size() / 2 - 1;
        }

        for (int32_t y = 0; y < height; ++y)
            for (int32_t x = 0; x < width; ++x)
            {
                _Type sum = 0.0f;

                for (int32_t k = deviation_start; k < deviation_end; ++k)
                {
                    x_altered = circle(width, x - k);
                    y_altered = circle(height, y - k);

                    sum += kernel[k - deviation_start] * src.data()[x + y_altered * width];
                    sum += kernel[k - deviation_start] * src.data()[x_altered + y * width];
                }

                tmp.data()[x + y * width] = static_cast<uint8_t>(sum);
            }

        dst = std::move(tmp);
    }
    
    template< typename _Type >
    std::vector<_Type> getGaussianKernel1D(uint32_t kernelSize, _Type sigma)
    {
        if (kernelSize == 0 || sigma < 0)
            throw imageException("Incorrect input parameters for 1D Gaussian filter kernel");

        std::vector<_Type> filter(kernelSize * 2 + 1, 0.0f);

        const _Type pi = 3.1415926536f;
        const _Type doubleSigma = sigma * 2;
        const _Type doubleSigmaPiInv = 1.0f / (doubleSigma * pi);

        _Type * x = filter.data();
        _Type sum = 0;

        const int32_t start = -static_cast<int32_t>(kernelSize);
        const int32_t end = static_cast<int32_t>(kernelSize) + 1;

        for (int32_t pos = start; pos < end; ++pos, ++x) {
            *x = doubleSigmaPiInv * exp(-static_cast<_Type>(pos * pos) / doubleSigma);
            sum += *x;
        }

        const _Type normalization = 1.0f / sum;
        x = filter.data();

        for (int32_t pos = start; pos < end; ++pos, ++x)
            (*x) *= normalization;

        return filter;
    }

    template< typename _Type >
    PenguinV_Image::Image applyFiltering(PenguinV_Image::Image & image, const EdgeParameter & edgeParameter)
    {
        switch (edgeParameter.filter)
        {
        case (edgeParameter.NONE):
            break;

        case (edgeParameter.MEDIAN):
            //return Image_Function::Median(image, edgeParameter.filterKernelSize);

        case (edgeParameter.GAUSSIAN):
            std::vector<_Type> kernel;
            
            kernel = getGaussianKernel1D< _Type >(edgeParameter.filterKernelSize, edgeParameter.sigma);
            circularConvolution< _Type >(image, image, kernel);

            return image;
        }

        return image;
    }
}

EdgeParameter::EdgeParameter( directionType _direction, gradientType _gradient, edgeType _edge, filterType _filter, 
                              uint32_t _filterKernelSize, float _sigma, uint32_t _groupFactor, uint32_t _skipFactor,
                              uint32_t _contrastCheckLeftSideOffset, uint32_t _contrastCheckRightSideOffset, uint8_t _minimumContrast )
    : direction        ( _direction )
    , gradient         ( _gradient )
    , edge             ( _edge )
    , filter           ( _filter )
    , filterKernelSize ( _filterKernelSize )
    , sigma            ( _sigma )
    , groupFactor      ( _groupFactor )
    , skipFactor       ( _skipFactor )
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

void EdgeDetectionHelper::find( EdgeDetectionBase<double> & edgeDetection, const PenguinV_Image::Image & image, uint32_t x, uint32_t y, 
                                uint32_t width, uint32_t height, const EdgeParameter & edgeParameter)
{
    if (edgeParameter.filter != edgeParameter.NONE)
    {
        PenguinV_Image::Image imageCopy(image);
        imageCopy = applyFiltering<double>(imageCopy, edgeParameter);

        findEdgePoints(imageCopy, x, y, width, height, edgeParameter, edgeDetection.positiveEdgePoint, edgeDetection.negativeEdgePoint);
    }
    else
        findEdgePoints(image, x, y, width, height, edgeParameter, edgeDetection.positiveEdgePoint, edgeDetection.negativeEdgePoint);
}

void EdgeDetectionHelper::find( EdgeDetectionBase<float> & edgeDetection, const PenguinV_Image::Image & image, uint32_t x, uint32_t y, 
                                uint32_t width, uint32_t height, const EdgeParameter & edgeParameter)
{
    if (edgeParameter.filter != edgeParameter.NONE)
    {
        PenguinV_Image::Image imageCopy(image);
        imageCopy = applyFiltering<float>(imageCopy, edgeParameter);

        findEdgePoints(imageCopy, x, y, width, height, edgeParameter, edgeDetection.positiveEdgePoint, edgeDetection.negativeEdgePoint);
    }
    else
        findEdgePoints(image, x, y, width, height, edgeParameter, edgeDetection.positiveEdgePoint, edgeDetection.negativeEdgePoint);
}
