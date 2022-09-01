#include "x11_ui.h"

#ifndef _WIN32 // Not for Windows

#include "../../penguinv_exception.h"

UiWindowX11::UiWindowX11( const penguinV::Image & image, const std::string & title )
    : UiWindow( image, title )
    , _uiDisplay( nullptr )
    , _screen( 0 )
    , _window( 0 )
    , _image( nullptr )
    , _deleteWindowEvent( 0 )
    , _width( image.width() )
    , _height( image.height() )
{
    _uiDisplay = XOpenDisplay( NULL );
    if ( _uiDisplay == nullptr )
        throw penguinVException( "Cannot create window for display" );

    _window = XCreateSimpleWindow( _uiDisplay, RootWindow( _uiDisplay, _screen ), 10, 10, _width, _height, 1, BlackPixel( _uiDisplay, _screen ),
                                   WhitePixel( _uiDisplay, _screen ) );
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
        if ( e.type == Expose ) {
            GC defaultGC = DefaultGC( _uiDisplay, _screen );
            XPutImage( _uiDisplay, _window, defaultGC, _image, 0, 0, 0, 0, _width, _height );

            for ( size_t i = 0u; i < _point.size(); ++i ) {
                const Point2d & point = _point[i].first;
                XSetForeground( _uiDisplay, defaultGC, _point[i].second );
                XDrawLine( _uiDisplay, _window, defaultGC, static_cast<int>( point.x - 1 ), static_cast<int>( point.y - 1 ), static_cast<int>( point.x + 1 ),
                           static_cast<int>( point.y + 1 ) );
            }
        }
        else if ( ( e.type == ClientMessage ) && ( static_cast<unsigned int>( e.xclient.data.l[0] ) == _deleteWindowEvent ) )
            break;
    }
}

void UiWindowX11::_setupImage( const penguinV::Image & image )
{
    if ( image.empty() )
        return;

    _data.resize( _width * _height * 4 );
    const uint32_t rowSize = image.rowSize();

    const uint8_t * imageY = image.data();
    const uint8_t * imageYEnd = imageY + _height * rowSize;
    char * imageData = _data.data();

    if ( image.colorCount() == penguinV::GRAY_SCALE ) {
        for ( ; imageY != imageYEnd; imageY += rowSize ) {
            const uint8_t * imageX = imageY;
            const uint8_t * imageXEnd = imageX + _width;

            for ( ; imageX != imageXEnd; ++imageX ) {
                memset( imageData, *imageX, 3u );
                imageData += 3;
                *imageData = 0;
                ++imageData;
            }
        }
    }
    else {
        if ( image.colorCount() != penguinV::RGB )
            throw penguinVException( "Color image has different than 3 color channels." );
        for ( ; imageY != imageYEnd; imageY += rowSize ) {
            const uint8_t * imageX = imageY;
            const uint8_t * imageXEnd = imageX + _width * 3u;

            for ( ; imageX != imageXEnd; imageX += 3 ) {
                memcpy( imageData, imageX, 3u );
                imageData += 3;
                *imageData = 0;
                ++imageData;
            }
        }
    }

    const int defaultScreen = DefaultScreen( _uiDisplay );
    _image = XCreateImage( _uiDisplay, DefaultVisual( _uiDisplay, defaultScreen ), static_cast<uint32_t>( DefaultDepth( _uiDisplay, defaultScreen ) ), ZPixmap, 0,
                           _data.data(), _width, _height, 32, 0 );
}

void UiWindowX11::drawPoint( const Point2d & point, const PaintColor & color )
{
    _point.push_back( std::make_pair( point, ( color.red << 16 ) + ( color.green << 8 ) + color.blue ) );
}

#endif
