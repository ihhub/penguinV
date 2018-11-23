#include "qt_ui.h"
#include <QImage>
#include <QPainter>

UiWindowQt::UiWindowQt( const PenguinV_Image::Image & image, const std::string & title )
    : UiWindow( image, title )
{
    const QImage imageQt( _image.data(), static_cast<int>( _image.width() ), static_cast<int>( _image.height() ),
                          static_cast<int>( _image.rowSize() ), (_image.colorCount() == 1u) ? QImage::Format_Grayscale8 : QImage::Format_RGB888 );
    _pixmap = QPixmap::fromImage( imageQt );
    _window.setPixmap( _pixmap );
}

UiWindowQt::~UiWindowQt()
{
}

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
    paint.drawPoint( static_cast<int>(point.x), static_cast<int>(point.y) );

    _window.setPixmap( _pixmap );
    _display();
}

void UiWindowQt::drawLine( const Point2d & start, const Point2d & end, const PaintColor & color )
{
    QPainter paint( &_pixmap );
    paint.setPen( QColor( color.red, color.green, color.blue, color.alpha ) );
    paint.drawLine( static_cast<int>(start.x), static_cast<int>(start.y), static_cast<int>(end.x), static_cast<int>(end.y) );

    _window.setPixmap( _pixmap );
    _display();
}
