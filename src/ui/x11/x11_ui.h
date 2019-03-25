#pragma once
#include <vector>
#include <X11/Xlib.h>
#include "../ui.h"

class UiWindowX11 : public UiWindow
{
public:
    explicit UiWindowX11( const PenguinV_Image::Image & image = PenguinV_Image::Image(), const std::string & title = std::string() );
    virtual ~UiWindowX11();
protected:
    virtual void _display();
private:
    std::vector<char> _data;
    Display* _uiDisplay;
    int _screen;
    Window _window;
    XImage* _image;
    Atom _deleteWindowEvent;
    uint32_t _width;
    uint32_t _height;

    void _setupImage( const PenguinV_Image::Image & image );
};

