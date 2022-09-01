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
