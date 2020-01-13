#include "template_matching.h"
#include "image_exception.h"
#include "parameter_validation.h"

TemplateMatching::TemplateMatching()
    : _width( 0 )
    , _height( 0 )
{
}

TemplateMatching::TemplateMatching( const penguinV::Image & image )
    : _width( 0 )
    , _height( 0 )
{
    train( image, 0, 0, image.width(), image.height() );
}

TemplateMatching::TemplateMatching( const penguinV::Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height )
    : _width( 0 )
    , _height( 0 )
{
    train( image, startX, startY, width, height );
}

bool TemplateMatching::isTrained() const
{
    return !_trainedData.empty();
}

void TemplateMatching::train( const penguinV::Image & image )
{
    train( image, 0, 0, image.width(), image.height() );
}

void TemplateMatching::train( const penguinV::Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height )
{
    Image_Function::ParameterValidation( image, startX, startY, width, height );
    Image_Function::VerifyGrayScaleImage( image );

    for ( uint32_t y = 0; y < height; ++y ) {
        for ( uint32_t x = 0; x < width; ++x ) {

        }
    }
}

const std::vector < MatchedPatternInfo > & TemplateMatching::inspect( const penguinV::Image & image, double minAcceptanceScore )
{
    inspect( image, 0, 0, image.width(), image.height(), minAcceptanceScore );
}

const std::vector < MatchedPatternInfo > & TemplateMatching::inspect( const penguinV::Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height,
                                                                      double minAcceptanceScore )
{
    Image_Function::ParameterValidation( image, startX, startY, width, height );
    Image_Function::VerifyGrayScaleImage( image );

    if ( width < _width || height < _height )
        throw std::logic_error( "The size of search area is less than the size of trained data" );

    if ( minAcceptanceScore < 0 || minAcceptanceScore > 100 )
        throw imageException( "Acceptable score for pattern matching must be more than 0% and less than 100%" );
    
    _foundPattern.clear();

    const uint32_t endX = startX + width - _width + 1;
    const uint32_t endY = startY + height - _height + 1;

    for ( uint32_t y = startY; y < endY; ++y ) {
        for ( uint32_t x = startX; x < endX; ++x ) {
            const double score = _findPatternScore( image, x, y );
            if ( score >= minAcceptanceScore )
                _foundPattern.emplace_back( MatchedPatternInfo( score, Point2d( x, y ) ) );
        }
    }

    return _foundPattern;
}

const std::vector < MatchedPatternInfo > & TemplateMatching::get() const
{
    return _foundPattern;
}

double TemplateMatching::_findPatternScore( const penguinV::Image & image, uint32_t x, uint32_t y ) const
{
    return 0;
}
