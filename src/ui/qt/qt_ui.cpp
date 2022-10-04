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

#include "qt_ui.h"
#include <QImage>
#include <QPainter>
#include <QString>

UiWindowQt::UiWindowQt( const penguinV::Image & image, const std::string & title )
    : UiWindow( image, title )
{
    const QImage imageQt( _image.data(), static_cast<int>( _image.width() ), static_cast<int>( _image.height() ), static_cast<int>( _image.rowSize() ),
                          ( _image.colorCount() == 1u ) ? QImage::Format_Grayscale8 : QImage::Format_RGB888 );
    _pixmap = QPixmap::fromImage( imageQt );
    _window.setPixmap( _pixmap );
    _window.window()->setWindowTitle( QString::fromStdString( _title ) );
}

UiWindowQt::~UiWindowQt() {}

void UiWindowQt::_display()
{
    if ( !_shown ) // we don't need to display anything
        return;

    _window.show();
}

void UiWindowQt::drawPoint( const Point2d & point, const PaintColor & color )
{
    QPainter paint( &_pixmap );
    paint.setPen( QColor( color.red, color.green, color.blue, color.alpha ) );
    paint.drawPoint( static_cast<int>( point.x ), static_cast<int>( point.y ) );

    _window.setPixmap( _pixmap );
    _display();
}

void UiWindowQt::drawLine( const Point2d & start, const Point2d & end, const PaintColor & color )
{
    QPainter paint( &_pixmap );
    paint.setPen( QColor( color.red, color.green, color.blue, color.alpha ) );
    paint.drawLine( static_cast<int>( start.x ), static_cast<int>( start.y ), static_cast<int>( end.x ), static_cast<int>( end.y ) );

    _window.setPixmap( _pixmap );
    _display();
}

void UiWindowQt::drawEllipse( const Point2d & center, double xRadius, double yRadius, const PaintColor & color )
{
    QPainter paint( &_pixmap );
    paint.setPen( QColor( color.red, color.green, color.blue, color.alpha ) );
    paint.drawEllipse( QPointF( center.x, center.y ), xRadius, yRadius );
    _window.setPixmap( _pixmap );
    _display();
}

void UiWindowQt::drawRectangle( const Point2d & topLeftCorner, double width, double height, const PaintColor & color )
{
    QPainter paint( &_pixmap );
    paint.setPen( QColor( color.red, color.green, color.blue, color.alpha ) );
    paint.drawRect( static_cast<int>( topLeftCorner.x ), static_cast<int>( topLeftCorner.y ), static_cast<int>( width ), static_cast<int>( height ) );

    _window.setPixmap( _pixmap );
    _display();
}
