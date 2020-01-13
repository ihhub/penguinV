#pragma once

#include "image_buffer.h"
#include "math/math_base.h"

struct MatchedPatternInfo
{
    MatchedPatternInfo( double score_ = 0, const Point2d & position_ = Point2d() )
        : score( score_ )
        , position( position_ )
    {}

    double score;
    Point2d position;
};

class TemplateMatching
{
public:
    TemplateMatching();
    TemplateMatching( const penguinV::Image & image );
    TemplateMatching( const penguinV::Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height );

    bool isTrained() const;

    void train( const penguinV::Image & image );
    void train( const penguinV::Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height );

    const std::vector < MatchedPatternInfo > & inspect( const penguinV::Image & image, double minAcceptanceScore = 75 );
    const std::vector < MatchedPatternInfo > & inspect( const penguinV::Image & image, uint32_t startX, uint32_t startY, uint32_t width, uint32_t height,
                                                        double minAcceptanceScore = 75 );

    const std::vector < MatchedPatternInfo > & get() const;
private:
    std::vector< uint8_t > _trainedData;
    uint32_t _width;
    uint32_t _height;
    std::vector< MatchedPatternInfo > _foundPattern;

    double _findPatternScore( const penguinV::Image & image, uint32_t x, uint32_t y ) const;
};
