#pragma once

#ifdef _WIN32 // Windows

#include <windows.h>
#include <vector>
#include "../ui.h"

namespace WindowsUi
{
    class UiWindowWinInfo; // a helper class for drawing function
}

class UiWindowWin : public UiWindow
{
public:
    explicit UiWindowWin( const penguinV::Image & image = penguinV::Image(), const std::string & title = std::string() );
    virtual ~UiWindowWin();

    virtual void setImage( const penguinV::Image & image );
    virtual void drawPoint( const Point2d & point, const PaintColor & color );
    virtual void drawLine( const Point2d & start, const Point2d & end, const PaintColor & color );
    virtual void drawEllipse( const Point2d & center, double xRadius, double yRadius, const PaintColor & color );
    virtual void drawRectangle( const Point2d & topLeftCorner, double width, double height, const PaintColor & color );
protected:
    virtual void _display();
private:
    HWND _window; // Windows OS window handle (just an unique ID)
    BITMAPINFO * _bmpInfo; // bitmap structure needed for drawing

    friend class WindowsUi::UiWindowWinInfo;

    void _free();
};

#endif
