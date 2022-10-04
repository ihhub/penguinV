/***************************************************************************
 *   penguinV: https://github.com/ihhub/penguinV                           *
 *   Copyright (C) 2017 - 2022                                             *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#pragma once

#ifdef _WIN32 // Windows

#include "../ui.h"
#include <vector>
#include <windows.h>

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

    struct PointToDraw
    {
        PointToDraw( const Point2d & point_ = Point2d(), const PaintColor & color_ = PaintColor() )
            : point( point_ )
            , color( color_ )
        {}

        Point2d point;
        PaintColor color;
    };

    struct LineToDraw
    {
        LineToDraw( const Point2d & start_ = Point2d(), const Point2d & end_ = Point2d(), const PaintColor & color_ = PaintColor() )
            : start( start_ )
            , end( end_ )
            , color( color_ )
        {}

        Point2d start;
        Point2d end;
        PaintColor color;
    };

    struct EllipseToDraw
    {
        EllipseToDraw( double left_ = 0, double top_ = 0, double right_ = 0, double bottom_ = 0, const PaintColor & color_ = PaintColor() )
            : left( left_ )
            , top( top_ )
            , right( right_ )
            , bottom( bottom_ )
            , color( color_ )
        {}

        double left;
        double top;
        double right;
        double bottom;
        PaintColor color;
    };

    std::vector<PointToDraw> _point;
    std::vector<LineToDraw> _line;
    std::vector<EllipseToDraw> _ellipse;

    double _scaleFactor;

    friend class WindowsUi::UiWindowWinInfo;

    void _free();
};

#endif
