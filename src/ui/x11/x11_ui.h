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

#ifndef _WIN32 // Not for Windows

#include "../ui.h"
#include <X11/Xlib.h>
#include <utility>
#include <vector>

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

    std::vector<std::pair<Point2d, uint32_t>> _point;

    void _setupImage( const penguinV::Image & image );
};

#endif
