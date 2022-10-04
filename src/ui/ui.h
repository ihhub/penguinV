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
#include "../image_buffer.h"
#include "../math/math_base.h"
#include <string>

struct PaintColor
{
    PaintColor( uint8_t _red = 0u, uint8_t _green = 0u, uint8_t _blue = 0u, uint8_t _alpha = 255u )
        : red( _red )
        , green( _green )
        , blue( _blue )
        , alpha( _alpha )
    {}

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

    penguinV::Image _image; // we store a copy of image
    std::string _title;
    bool _shown;
};
