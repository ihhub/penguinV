#pragma once

#ifndef _WIN32 // Not for Windows

#include <vector>
#include <utility>
#include <X11/Xlib.h>
#include "../ui.h"

class UiWindowX11 : public UiWindow
{
public:
    explicit UiWindowX11( const penguinV::Image & image = penguinV::Image(), const std::string & title = std::string() );
    virtual ~UiWindowX11();

    virtual void drawPoint( const Point2d & point, const PaintColor & color );
    virtual void drawLine( const Point2d & start, const Point2d & end, const PaintColor & color );
    virtual void drawEllipse( const Point2d & center, double xRadius, double yRadius, const PaintColor & color );
    virtual void drawRectangle( const Point2d & topLeftCorner, double width, double height, const PaintColor & color );

protected:
    virtual void _display();
private:
    std::vector<char> _data;
    Display * _uiDisplay;
    int _screen;
    Window _window;
    XImage * _image;
    Atom _deleteWindowEvent;
    uint32_t _width;
    uint32_t _height;

    struct RectangleToDraw
    {
        RectangleToDraw( const Point2d & topLeft_ = Point2d(), double width_ = 0.0, double height_ = 0.0, const PaintColor & color_ = PaintColor() )
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

    std::vector<RectangleToDraw> _rectangles;

    void _setupImage( const penguinV::Image & image );
    static uint32_t _convertColor( const PaintColor & color = PaintColor() );
};

#endif
