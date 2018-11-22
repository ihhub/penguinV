#pragma once
#include <vector>
#include "../image_buffer.h"
#include "../math/math_base.h"

class UiWindow
{
public:
    UiWindow( const PenguinV_Image::Image & image = PenguinV_Image::Image() );
    virtual ~UiWindow();
    void show(); // show window at the screen
    void setImage( const PenguinV_Image::Image & image ); // replaces existing shown image by new image
    virtual void drawPoint( const Point2d & point );
    virtual void drawLine( const Point2d & start, const Point2d & end );
protected:
    virtual void draw();

    PenguinV_Image::Image _image; // we store a copy of image
    bool _shown;
};
