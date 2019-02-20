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
    explicit UiWindowWin( const PenguinV_Image::Image & image = PenguinV_Image::Image(), const std::string & title = std::string() );
    virtual ~UiWindowWin();

    virtual void setImage( const PenguinV_Image::Image & image );
    virtual void drawPoint( const Point2d & point, const PaintColor & color );
    virtual void drawLine( const Point2d & start, const Point2d & end, const PaintColor & color );
    virtual void drawEllipse( const Point2d & center, double xRadius, double yRadius, const PaintColor & color );
    virtual void drawRectangle( const Point2d & topLeftCorner, double width, double height, const PaintColor & color );
protected:
    virtual void _display();
private:
    HWND _window; // Windows OS window handle (just an unique ID)
    BITMAPINFO * _bmpInfo; // bitmap structure needed for drawing

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
        EllipseToDraw( double left_ = 0, double top_ = 0, double right_ = 0, double bottom_ = 0, const PaintColor & color_ = PaintColor() )
            : left  ( left_   )
            , top   ( top_    )
            , right ( right_  )
            , bottom( bottom_ )
            , color ( color_  )
        {
        }

        double left;
        double top;
        double right;
        double bottom;
        PaintColor color;
    };

    std::vector < PointToDraw   > _point;
    std::vector < LineToDraw    > _line;
    std::vector < EllipseToDraw > _ellipse;

    friend class WindowsUi::UiWindowWinInfo;

    void _free();
};

#endif
