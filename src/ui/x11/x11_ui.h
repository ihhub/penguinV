#pragma once

#ifndef _WIN32 // Not for Windows

#include <vector>
#include <tuple>
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

    std::vector< std::pair<Point2d, uint32_t> > _point;
    std::vector< std::tuple<Point2d, Point2d, uint32_t> > _lines;
    std::vector< std::tuple<Point2d, double, double, uint32_t> > _ellipses;

    void _setupImage( const penguinV::Image & image );
};

#endif
