#include "x11_ui.h"
#include "../../image_exception.h"

UiWindowX11::UiWindowX11( const PenguinV_Image::Image & image, const std::string & title )
    : UiWindow( image, title )
    , _uiDisplay( nullptr )
    , _screen( 0 )
    , _window( 0 )
    , _image( nullptr )
    , _width( image.width() )
    , _height( image.height() )
{
    _uiDisplay = XOpenDisplay( NULL );
    if ( _uiDisplay == nullptr )
        throw imageException( "Cannot create window for display" );

    _window = XCreateSimpleWindow( _uiDisplay, RootWindow( _uiDisplay, _screen ), 10, 10, _width, _height, 1,
                                   BlackPixel( _uiDisplay, _screen ), WhitePixel( _uiDisplay, _screen ) );
    XSelectInput( _uiDisplay, _window, ExposureMask | KeyPressMask );
    XMapWindow( _uiDisplay, _window );

    XStoreName( _uiDisplay, _window, title.data() );

    _deleteWindowEvent = XInternAtom( _uiDisplay, "WM_DELETE_WINDOW", False );
    XSetWMProtocols( _uiDisplay, _window, &_deleteWindowEvent, 1 );

    _setupImage( image );
}

UiWindowX11::~UiWindowX11()
{
    XDestroyWindow( _uiDisplay, _window );
    XCloseDisplay( _uiDisplay );
}

void UiWindowX11::_display()
{
    if ( !_shown ) // we don't need to display anything
        return;

    XEvent e;
    while ( true ) {
        XNextEvent( _uiDisplay, &e );
        if ( e.type == Expose )
            XPutImage( _uiDisplay, _window, DefaultGC( _uiDisplay, _screen ), _image, 0, 0, 0, 0, _width, _height );
        else if ( (e.type == ClientMessage) && (static_cast<unsigned int>(e.xclient.data.l[0]) == _deleteWindowEvent) )
            break;
    }
}

void UiWindowX11::_setupImage( const PenguinV_Image::Image & image )
{
    if ( image.empty() )
        return;

    _data.resize( _width * _height * 4 );
    const uint32_t rowSize = image.rowSize();

    const uint8_t * imageY = image.data();
    const uint8_t * imageYEnd = imageY + _height * rowSize;
    char * imageData = _data.data();

    if ( image.colorCount() == PenguinV_Image::GRAY_SCALE ) {
        for ( ; imageY != imageYEnd; imageY += rowSize ) {
            const uint8_t * imageX    = imageY;
            const uint8_t * imageXEnd = imageX + _width;

            for ( ; imageX != imageXEnd; ++imageX ) {
                *imageData = *imageX;
                ++imageData;
                *imageData = *imageX;
                ++imageData;
                *imageData = *imageX;
                ++imageData;
                *imageData = 0;
                ++imageData;
            }
        }
    }
    else {
        for ( ; imageY != imageYEnd; imageY += rowSize ) {
            const uint8_t * imageX    = imageY;
            const uint8_t * imageXEnd = imageX + _width * image.colorCount();

            for ( ; imageX != imageXEnd; ) {
                *imageData = *imageX;
                ++imageData;
                ++imageX;
                *imageData = *imageX;
                ++imageData;
                ++imageX;
                *imageData = *imageX;
                ++imageData;
                ++imageX;
                *imageData = 0;
                ++imageData;
            }
        }
    }

    const int defaultScreen = DefaultScreen( _uiDisplay );
    _image = XCreateImage( _uiDisplay, DefaultVisual( _uiDisplay, defaultScreen ), DefaultDepth( _uiDisplay, defaultScreen ), ZPixmap,
                           0, _data.data(), _width, _height, 32, 0 );
}
