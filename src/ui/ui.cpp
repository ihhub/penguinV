#include "ui.h"

UiWindow::UiWindow( const PenguinV_Image::Image & image)
    : _image( image )
    , _shown( false)
{
}

UiWindow::~UiWindow()
{
}

void UiWindow::show()
{
    _shown = true;
    draw();
}

void UiWindow::setImage( const PenguinV_Image::Image & image )
{
    _image = image;
    draw();
}

void UiWindow::drawPoint( const Point2d &, const PaintColor & )
{
    draw();
}

void UiWindow::drawLine( const Point2d &, const Point2d &, const PaintColor & )
{
    draw();
}

void UiWindow::draw()
{
}
