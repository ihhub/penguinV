#pragma once
#include <string>
#include "../image_buffer.h"
#include "../math/math_base.h"

struct PaintColor
{
    PaintColor( uint8_t _red = 0u, uint8_t _green = 0u, uint8_t _blue = 0u, uint8_t _alpha = 255u )
        : red  ( _red   )
        , green( _green )
        , blue ( _blue  )
        , alpha( _alpha )
    {
    }

    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;
};

class UiWindow
{
public:
    explicit UiWindow( const penguinV::Image & image = penguinV::Image(), const std::string & title = std::string() );
    virtual ~UiWindow();
    void show(); // show window at the screen
    virtual void setImage( const penguinV::Image & image ); // replaces existing shown image by new image
    virtual void drawPoint( const Point2d & point, const PaintColor & color );
    virtual void drawLine( const Point2d & start, const Point2d & end, const PaintColor & color );
    virtual void drawEllipse( const Point2d & center, double xRadius, double yRadius, const PaintColor & color );
    virtual void drawRectangle( const Point2d & topLeftCorner, double width, double height, const PaintColor & color );
protected:
    virtual void _display();

    struct PointToDraw
    {
        PointToDraw( const Point2d & point_ = Point2d(), const PaintColor & color_ = PaintColor() )
            : point( point_ )
            , color( color_ )
        {
        }

        Point2d point;
        PaintColor color;
    };

    struct LineToDraw
    {
        LineToDraw( const Point2d & start_ = Point2d(), const Point2d & end_ = Point2d(), const PaintColor & color_ = PaintColor() )
            : start( start_ )
            , end  ( end_   )
            , color( color_ )
        {
        }

        Point2d start;
        Point2d end;
        PaintColor color;
    };

    struct EllipseToDraw
    {
        EllipseToDraw( const Point2d & topLeft_ = Point2d(), double width_ = 0.0, double height_ = 0.0, const PaintColor & color_ = PaintColor() )
            : topLeft( topLeft_ )
            , width( width_ )
            , height( height_ )
            , color( color_ )
        {}

        Point2d topLeft;
        double width;
        double height;
        PaintColor color;
    };

    std::vector<PointToDraw> _point;
    std::vector<LineToDraw> _lines;
    std::vector<EllipseToDraw> _ellipses;

    penguinV::Image _image; // we store a copy of image
    std::string _title;
    bool _shown;
};
