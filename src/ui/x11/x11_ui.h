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

    void _setupImage( const penguinV::Image & image );
};

#endif
