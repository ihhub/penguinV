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
    explicit UiWindow( const PenguinV_Image::Image & image = PenguinV_Image::Image(), const std::string & title = std::string() );
    virtual ~UiWindow();
    void show(); // show window at the screen
    virtual void setImage( const PenguinV_Image::Image & image ); // replaces existing shown image by new image
    virtual void drawPoint( const Point2d & point, const PaintColor & color );
    virtual void drawLine( const Point2d & start, const Point2d & end, const PaintColor & color );
    virtual void drawEllipse( const Point2d & center, double xRadius, double yRadius, const PaintColor & color );
    virtual void drawRectangle( const Point2d & topLeftCorner, double width, double height, const PaintColor & color );
protected:
    virtual void _display();

    PenguinV_Image::Image _image; // we store a copy of image
    std::string _title;
    bool _shown;
};
