#include "qt_ui.h"
#include <QImage>
#include <QPainter>

UiWindowQt::UiWindowQt( const PenguinV_Image::Image & image )
    : UiWindow( image )
{
    const QImage imageQt( _image.data(), _image.width(), _image.height(), _image.rowSize(),
                          (_image.colorCount() == 1u) ? QImage::Format_Grayscale8 : QImage::Format_RGB888 );
    _pixmap = QPixmap::fromImage( imageQt );
    _window.setPixmap( _pixmap );
}

UiWindowQt::~UiWindowQt()
{
}

void UiWindowQt::draw()
{
    if( !_shown ) // we don't need to draw anything
        return;

    _window.show();
}

void UiWindowQt::drawPoint( const Point2d & point )
{
    QPainter paint(&_pixmap);
    paint.setPen( QColor(20, 255, 20, 255) );
    paint.drawPoint( point.x, point.y );
    
    _window.setPixmap( _pixmap );
	draw();
}

void UiWindowQt::drawLine( const Point2d & start, const Point2d & end )
{
    QPainter paint(&_pixmap);
    paint.setPen( QColor(20, 255, 20, 255) );
    paint.drawLine( start.x, start.y, end.x, end.y );
    
    _window.setPixmap( _pixmap );
	draw();
}
