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

#include "ui.h"

UiWindow::UiWindow( const penguinV::Image & image, const std::string & title )
    : _image( image )
    , _title( title )
    , _shown( false )
{}

UiWindow::~UiWindow() {}

void UiWindow::show()
{
    _shown = true;
    _display();
}

void UiWindow::setImage( const penguinV::Image & image )
{
    _image = image;
    _display();
}

void UiWindow::drawPoint( const Point2d &, const PaintColor & )
{
    _display();
}

void UiWindow::drawLine( const Point2d &, const Point2d &, const PaintColor & )
{
    _display();
}

void UiWindow::drawEllipse( const Point2d &, double, double, const PaintColor & )
{
    _display();
}

void UiWindow::drawRectangle( const Point2d &, double, double, const PaintColor & )
{
    _display();
}

void UiWindow::_display() {}
